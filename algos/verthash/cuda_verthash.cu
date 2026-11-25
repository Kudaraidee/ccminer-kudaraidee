// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verthash (Vertcoin) CUDA device kernels + host launchers.
//
// Three-kernel pipeline, structurally ported from the VerthashMiner CUDA kernel
// (src/vhCuda/verthash.cu, CryptoGraphics GPLv2), reworked to use the shared
// FIPS-202 permutation in cuda/sha3_device.cuh and to take mdiv as a runtime
// __constant__ instead of a baked #define:
//
//   1. verthash_gpu_precompute (8 threads/job)   8x SHA3-512 prehash-72, job-scoped
//   2. verthash_gpu_sha3_256   (1 thread/nonce)  SHA3-256(header||nonce)
//   3. verthash_gpu_io         (4 threads/nonce) SHA3-512 final-8 into a shared
//      512-byte subset, then 4096 random 32-byte datafile reads, fnv1a-folded
//      across the 4 lanes with a width-4 shuffle.
//
// The IO kernel reports any nonce whose word 7 is <= target[7] -- a safe
// superset, so no real share is missed; the host re-verifies each candidate
// with the CPU oracle + fulltest before submit.

#include <cuda_runtime.h>
#include <stdint.h>
#include "cuda/sha3_device.cuh"

typedef unsigned int uint;

// header words 0..18 (bytes 0..75); word 18 (bytes 72..75) precedes the nonce.
__constant__ uint32_t c_vh_header[19];
// index modulus = ((datafile_size - 32) / 16) + 1.
__constant__ uint32_t c_vh_mdiv;
// Multiply-shift reciprocal of c_vh_mdiv (see vh_mod_mdiv / vh_magicu).
__constant__ uint32_t c_vh_magic;
__constant__ uint32_t c_vh_shift;

static __device__ __forceinline__ uint vh_fnv1a(const uint a, const uint b)
{
	return (a ^ b) * 0x1000193U;
}

// x % c_vh_mdiv without a division (Hacker's Delight 10-9 magic reciprocal).
// mdiv stays a RUNTIME value: the datafile has changed size before, and a baked
// divisor would silently mis-index a new one. ADD is a template parameter, not
// a branch -- a branch leaves the dead division in the kernel.
template <bool ADD>
static __device__ __forceinline__ uint vh_mod_mdiv(const uint x)
{
	const uint t = __umulhi(x, c_vh_magic);
	const uint q = ADD ? ((t + ((x - t) >> 1)) >> c_vh_shift)
	                   : (t >> c_vh_shift);
	return x - q * c_vh_mdiv;
}

static __device__ __forceinline__ uint vh_rotl32(const uint x, const uint n)
{
	return (x << n) | (x >> (32 - n));
}

// ---------------------------------------------------------------------------
// 1) 8x SHA3-512 prehash: absorb the 72-byte first block (header[0] += lane+1)
//    and run the first permutation. Store 8 states x 25 uint2 linearly.
__global__ void verthash_gpu_precompute(uint2 *kstates)
{
	const uint t = blockDim.x * blockIdx.x + threadIdx.x;   // 0..7
	if (t >= 8) return;

	uint2 st[25];
	#pragma unroll
	for (int i = 0; i < 25; i++) st[i] = make_uint2(0, 0);

	// The reference increments header BYTE 0 (wrapping within the byte); a word
	// add would carry into byte 1 when byte 0 >= 0xf8.
	const uint32_t h0 = c_vh_header[0];
	st[0].x = (h0 & 0xffffff00u) | ((h0 + (t & 7) + 1) & 0xffu); st[0].y = c_vh_header[1];
	st[1].x = c_vh_header[2];  st[1].y = c_vh_header[3];
	st[2].x = c_vh_header[4];  st[2].y = c_vh_header[5];
	st[3].x = c_vh_header[6];  st[3].y = c_vh_header[7];
	st[4].x = c_vh_header[8];  st[4].y = c_vh_header[9];
	st[5].x = c_vh_header[10]; st[5].y = c_vh_header[11];
	st[6].x = c_vh_header[12]; st[6].y = c_vh_header[13];
	st[7].x = c_vh_header[14]; st[7].y = c_vh_header[15];
	st[8].x = c_vh_header[16]; st[8].y = c_vh_header[17];

	sha3_keccakf_1600(st);

	uint2 *out = kstates + 25 * t;
	#pragma unroll
	for (int i = 0; i < 25; i++) out[i] = st[i];
}

// ---------------------------------------------------------------------------
// 2) SHA3-256(header || nonce) -> running 32-byte hash. One thread per nonce.
__global__ void verthash_gpu_sha3_256(uint2 *iohashes, const uint in18, const uint firstNonce)
{
	const uint gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint nonce = firstNonce + gid;

	uint2 st[25];
	#pragma unroll
	for (int i = 0; i < 25; i++) st[i] = make_uint2(0, 0);

	st[0].x = c_vh_header[0];  st[0].y = c_vh_header[1];
	st[1].x = c_vh_header[2];  st[1].y = c_vh_header[3];
	st[2].x = c_vh_header[4];  st[2].y = c_vh_header[5];
	st[3].x = c_vh_header[6];  st[3].y = c_vh_header[7];
	st[4].x = c_vh_header[8];  st[4].y = c_vh_header[9];
	st[5].x = c_vh_header[10]; st[5].y = c_vh_header[11];
	st[6].x = c_vh_header[12]; st[6].y = c_vh_header[13];
	st[7].x = c_vh_header[14]; st[7].y = c_vh_header[15];
	st[8].x = c_vh_header[16]; st[8].y = c_vh_header[17];

	st[9].x ^= in18; st[9].y ^= nonce;    // bytes 72..79
	st[10].x ^= 0x00000006U;              // byte 80 (0x06 pad)
	st[16].y ^= 0x80000000U;              // byte 135 (rate 136, final bit)

	sha3_keccakf_1600(st);

	iohashes[4 * gid + 0] = st[0];
	iohashes[4 * gid + 1] = st[1];
	iohashes[4 * gid + 2] = st[2];
	iohashes[4 * gid + 3] = st[3];
}

// ---------------------------------------------------------------------------
// 3) IO/mix. 4 lanes cooperate per nonce. WORK_SIZE threads/block.
#define VH_WORK_SIZE 64
#define VH_GROUPS    (VH_WORK_SIZE / 4)   // 4-lane groups per block

// Subset held TRANSPOSED: word w of group g at [w * VH_GROUPS + g]. The word
// index is the loop counter, uniform across the block, so the natural layout
// would put every group of a warp in bank w % 32 -- an 8-way conflict on all
// 4096 loads. Costs strided stores: 16 per nonce against 4096 loads.
//
// DUAL: the index is 16-byte granular (VH_BYTE_ALIGNMENT) but the item is 32
// bytes, so an odd index straddles two 32-byte sectors. A second copy of the
// datafile shifted left by 16 B lets an odd index read the same logical bytes
// from an aligned address:  even -> A[2*idx],  odd -> B[2*(idx-1)].
// Hash unchanged. Costs ~1.19 GiB; the host passes NULL when VRAM is short.
template <bool MAGIC_ADD, bool DUAL>
__global__ void
__launch_bounds__(VH_WORK_SIZE)
verthash_gpu_io(uint2 *iohashes, const uint2 *__restrict__ kstates,
                const uint2 *__restrict__ memory, const uint2 *__restrict__ memory_odd,
                const uint firstNonce, uint *results, const uint target)
{
	const uint globalThId = blockDim.x * blockIdx.x + threadIdx.x;
	const uint lgr4id = (globalThId & (VH_WORK_SIZE - 1)) >> 2;  // local 4-lane group
	const uint gr4id  = globalThId >> 2;                          // nonce index
	const uint gr4e   = globalThId & 3;                           // lane 0..3

	__shared__ uint sha3St[128 * VH_GROUPS];

	// --- SHA3-512 final-8: lane gr4e finishes states 2*gr4e and 2*gr4e+1 ---
	const uint nonce = firstNonce + gr4id;
	#pragma unroll
	for (int s3s = 0; s3s < 2; ++s3s) {
		const uint2 *ksrc = kstates + (2 * gr4e + s3s) * 25;
		uint2 st[25];
		#pragma unroll
		for (int i = 0; i < 25; ++i) st[i] = ksrc[i];

		st[0].x ^= c_vh_header[18]; st[0].y ^= nonce;   // final-8 block: bytes 72..79
		st[1].x ^= 0x00000006U;                          // byte 8 of block 2 (= byte 80)
		st[8].y ^= 0x80000000U;                          // byte 71 (rate 72, final bit)

		sha3_keccakf_1600(st);

		// word index of st[i].x in the untransposed subset; .y follows it
		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			const uint w0 = (gr4e * 32) + (s3s * 16) + 2 * i;
			sha3St[w0 * VH_GROUPS + lgr4id]       = st[i].x;
			sha3St[(w0 + 1) * VH_GROUPS + lgr4id] = st[i].y;
		}
	}
	// The subset lives in shared memory, written by the 4 lanes of this group and
	// read every IO iteration. Each 4-lane group is contained in one warp (4 | 32),
	// so a warp barrier suffices for visibility -- no block-wide __syncthreads.
	__syncwarp(0xffffffff);

	// --- IO/mix stage ---
	uint2 up1 = iohashes[globalThId];              // running hash words (2*gr4e, 2*gr4e+1)
	uint acc = 0x811c9dc5U;

	for (uint i = 0; i < 4096; ++i) {
		const uint s3idx  = i & 127;
		const uint rfac   = i >> 7;
		const uint seek = vh_rotl32(sha3St[s3idx * VH_GROUPS + lgr4id], rfac);
		const uint idx  = vh_mod_mdiv<MAGIC_ADD>(vh_fnv1a(seek, acc));

		uint2 v;
		if (DUAL) {
			// odd index -> the 16-byte-shifted copy, at a 32-byte-aligned offset
			const uint2 *__restrict__ base = (idx & 1u) ? memory_odd : memory;
			v = base[((idx & ~1u) << 1) + gr4e];
		} else {
			v = memory[(idx << 1) + gr4e];
		}

		up1.x = vh_fnv1a(up1.x, v.x);
		up1.y = vh_fnv1a(up1.y, v.y);

		// cross-lane accumulator sync via width-4 warp shuffle: every lane folds
		// blob_off[0..7] (= lanes 0..3, {x,y}) in order. No block barrier, no
		// extra shared memory -- the shuffle carries its own intra-warp sync.
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.x, 0, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.y, 0, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.x, 1, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.y, 1, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.x, 2, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.y, 2, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.x, 3, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.y, 3, 4));
	}

	iohashes[globalThId] = up1;

	// lane 3 holds words 6,7; up1.y is word 7 (MSW). Superset filter on '<='.
	if (gr4e == 3 && up1.y <= target) {
		uint slot = atomicAdd(results, 1u);
		results[slot + 1] = gr4id;
	}
}

// ===========================================================================
// Host launchers.

// Unsigned magic number for a constant divisor (Hacker's Delight fig. 10-3),
// exact for every d >= 2. The "add" indicator covers divisors whose magic does
// not fit in 32 bits.
static void vh_magicu(uint32_t d, uint32_t *magic, uint32_t *shift, int *add)
{
	uint32_t nc, delta, q1, r1, q2, r2;
	int p;

	*add = 0;
	nc = (uint32_t) (0xffffffffU - ((0u - d) % d));
	p  = 31;
	q1 = 0x80000000U / nc;          r1 = 0x80000000U - q1 * nc;
	q2 = 0x7fffffffU / d;           r2 = 0x7fffffffU - q2 * d;
	do {
		p++;
		if (r1 >= nc - r1) { q1 = 2 * q1 + 1; r1 = 2 * r1 - nc; }
		else               { q1 = 2 * q1;     r1 = 2 * r1;      }
		if (r2 + 1 >= d - r2) {
			if (q2 >= 0x7fffffffU) *add = 1;
			q2 = 2 * q2 + 1; r2 = 2 * r2 + 1 - d;
		} else {
			if (q2 >= 0x80000000U) *add = 1;
			q2 = 2 * q2;     r2 = 2 * r2 + 1;
		}
		delta = d - 1 - r2;
	} while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));

	*magic = q2 + 1;
	// the add variant shifts one less; fold it in so the device reads one value
	*shift = (uint32_t) (p - 32) - (uint32_t) *add;
}

// Which verthash_gpu_io instantiation the current divisor needs. mdiv is
// process-global (one datafile for every GPU thread), so every writer stores
// the same value.
static int s_magic_add = 0;

extern "C" {

void verthash_cuda_set_header(const uint32_t header19[19])
{
	cudaMemcpyToSymbol(c_vh_header, header19, sizeof(uint32_t) * 19);
}

void verthash_cuda_set_mdiv(uint32_t mdiv)
{
	uint32_t magic = 0, shift = 0;
	int add = 0;

	if (mdiv >= 2) vh_magicu(mdiv, &magic, &shift, &add);

	s_magic_add = add;
	cudaMemcpyToSymbol(c_vh_mdiv,  &mdiv,  sizeof(uint32_t));
	cudaMemcpyToSymbol(c_vh_magic, &magic, sizeof(uint32_t));
	cudaMemcpyToSymbol(c_vh_shift, &shift, sizeof(uint32_t));
}

void verthash_cuda_precompute(uint2 *d_kstates)
{
	verthash_gpu_precompute<<<1, 8>>>(d_kstates);
}

// nonces MUST be a multiple of 256 (exact grids; the 4-lane IO kernel launches
// nonces*4 threads with no bounds guard). The host rounds throughput down.
void verthash_cuda_hash(uint2 *d_iohashes, const uint2 *d_kstates, const uint2 *d_memory,
                        const uint2 *d_memory_odd,
                        uint32_t in18, uint32_t firstNonce, uint32_t nonces,
                        uint32_t *d_results, uint32_t target)
{
	const uint32_t blocks = (nonces * 4) / VH_WORK_SIZE;

	verthash_gpu_sha3_256<<<nonces / 256, 256>>>(d_iohashes, in18, firstNonce);

	if (d_memory_odd) {
		if (s_magic_add)
			verthash_gpu_io<true, true><<<blocks, VH_WORK_SIZE>>>(
				d_iohashes, d_kstates, d_memory, d_memory_odd, firstNonce, d_results, target);
		else
			verthash_gpu_io<false, true><<<blocks, VH_WORK_SIZE>>>(
				d_iohashes, d_kstates, d_memory, d_memory_odd, firstNonce, d_results, target);
	} else {
		if (s_magic_add)
			verthash_gpu_io<true, false><<<blocks, VH_WORK_SIZE>>>(
				d_iohashes, d_kstates, d_memory, NULL, firstNonce, d_results, target);
		else
			verthash_gpu_io<false, false><<<blocks, VH_WORK_SIZE>>>(
				d_iohashes, d_kstates, d_memory, NULL, firstNonce, d_results, target);
	}
}

} // extern "C"
