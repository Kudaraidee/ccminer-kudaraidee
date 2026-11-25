extern "C" {
#include "keccak_tiny.h"
#include "heavyhash-gate.h"
}

#include "cuda/helper.cuh"
#include "cuda/vectors.cuh"
#include "miner.h"
#include "cuda/selftest_gate.cuh"

#include <memory.h>

__constant__ static uint32_t c_data[20];
__constant__ static uint32_t c_matrix[64][16];	// 4 nibbles per word, one per byte (__dp4a)
__constant__ uint32_t pTarget[8];

static uint32_t *h_GNonces[MAX_GPUS];
static uint32_t *d_GNonces[MAX_GPUS];
static bool dirty_GNonces[MAX_GPUS];

typedef union {
    uint32_t h4[8];
    uint64_t h8[4];
    uint4 h16[2];
    ulong2 hl16[2];
    ulong4 h32;
} hash_t;

__constant__ uint2 keccak_round_constants35[24] = {
	{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
	{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
	{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
	{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
	{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
	{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
	{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
	{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
	{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
	{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
	{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
};

static void __forceinline__ __device__ keccak_block(uint2 *s)
{
	uint2 bc[5], tmpxor[5], u, v;
	//	uint2 s[25];

	#pragma unroll 1
	for (int i = 0; i < 24; i++)
	{
		#pragma unroll
		for (uint32_t x = 0; x < 5; x++)
			tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		u = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = ROL2(s[6] ^ bc[0], 44);
		s[6] = ROL2(s[9] ^ bc[3], 20);
		s[9] = ROL2(s[22] ^ bc[1], 61);
		s[22] = ROL2(s[14] ^ bc[3], 39);
		s[14] = ROL2(s[20] ^ bc[4], 18);
		s[20] = ROL2(s[2] ^ bc[1], 62);
		s[2] = ROL2(s[12] ^ bc[1], 43);
		s[12] = ROL2(s[13] ^ bc[2], 25);
		s[13] = ROL8(s[19] ^ bc[3]);
		s[19] = ROR8(s[23] ^ bc[2]);
		s[23] = ROL2(s[15] ^ bc[4], 41);
		s[15] = ROL2(s[4] ^ bc[3], 27);
		s[4] = ROL2(s[24] ^ bc[3], 14);
		s[24] = ROL2(s[21] ^ bc[0], 2);
		s[21] = ROL2(s[8] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[16] = ROL2(s[5] ^ bc[4], 36);
		s[5] = ROL2(s[3] ^ bc[2], 28);
		s[3] = ROL2(s[18] ^ bc[2], 21);
		s[18] = ROL2(s[17] ^ bc[1], 15);
		s[17] = ROL2(s[11] ^ bc[0], 10);
		s[11] = ROL2(s[7] ^ bc[1], 6);
		s[7] = ROL2(s[10] ^ bc[4], 3);
		s[10] = ROL2(u, 1);

		u = s[0]; v = s[1]; s[0] ^= (~v) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & u; s[4] ^= (~u) & v;
		u = s[5]; v = s[6]; s[5] ^= (~v) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & u; s[9] ^= (~u) & v;
		u = s[10]; v = s[11]; s[10] ^= (~v) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & u; s[14] ^= (~u) & v;
		u = s[15]; v = s[16]; s[15] ^= (~v) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & u; s[19] ^= (~u) & v;
		u = s[20]; v = s[21]; s[20] ^= (~v) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & u; s[24] ^= (~u) & v;
		s[0] ^= keccak_round_constants35[i];
	}
}

/* One nonce. Shared with the checksum kernel so the differential cannot
 * drift; keep __forceinline__ or the mining kernel's SASS changes. */
__device__ __forceinline__
hash_t heavyhash_one(const uint32_t *matrix, const uint32_t nonce)
{
        hash_t hash;
        uint32_t pdata[50] = {0};
        for (int i = 0; i < 19; i++) {
            pdata[i] = c_data[i];
        }
        pdata[19] = cuda_swab32(nonce);

        uint32_t vec[16];		// 64 nibbles, 4 per word
        uint32_t first_w[8];		// keccak #1 digest, 4 bytes per word
        uint32_t second_w[8] = { 0 };	// matrix product, packed nibbles

        ((uint8_t *) pdata)[80] = 0x06;
        ((uint8_t *) pdata)[135] = 0x80;

        keccak_block((uint2 *) pdata);

        #pragma unroll
        for (int i = 0; i < 8; ++i)
            first_w[i] = pdata[i];

        // nibbles 4w..4w+3 come from digest bytes 2w and 2w+1
        #pragma unroll
        for (int w = 0; w < 16; ++w) {
            const uint32_t hw = first_w[w >> 1], sh = (w & 1) ? 16 : 0;
            const uint32_t b0 = (hw >> sh) & 0xFF, b1 = (hw >> (sh + 8)) & 0xFF;
            vec[w] = (b0 >> 4) | ((b0 & 0xF) << 8) | ((b1 >> 4) << 16) | ((b1 & 0xF) << 24);
        }

        // Two rows at a time; the pair is one byte of the packed product. The
        // unroll keeps the byte index compile-time, so this stays in registers.
        #pragma unroll
        for (int m = 0; m < 32; ++m) {
            uint32_t s0 = 0, s1 = 0;
            const uint32_t *r0 = &matrix[(2*m)     * 16];
            const uint32_t *r1 = &matrix[(2*m + 1) * 16];
            #pragma unroll
            for (int w = 0; w < 16; ++w) {
                s0 = __dp4a(r0[w], vec[w], s0);
                s1 = __dp4a(r1[w], vec[w], s1);
            }
            second_w[m >> 2] |= (((s0 >> 10) << 4) | (s1 >> 10)) << (8 * (m & 3));
        }

        uint32_t tmp[50] = {0};
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            tmp[i] = first_w[i] ^ second_w[i];

        tmp[8]  = 0x06;			// byte 32
        tmp[33] = 0x80u << 24;		// byte 135

        keccak_block((uint2 *) tmp);

        for (int i = 0; i < 4; i++) {
            hash.h8[i] = ((uint64_t *) tmp)[i];
        }

        return hash;
}

/* Order-independent digest fold, one definition for both sides. Rotating by
 * word position keeps it sensitive to a permutation within a digest. */
__host__ __device__ __forceinline__
static uint64_t hh_fold(const uint64_t h[4])
{
	return h[0]
	     ^ ((h[1] << 16) | (h[1] >> 48))
	     ^ ((h[2] << 32) | (h[2] >> 32))
	     ^ ((h[3] << 48) | (h[3] >> 16));
}

/* Must be 2*nonce+1, not nonce|1: |1 gives 2k and 2k+1 the same weight, so
 * swapping that pair would be invisible. Odd => invertible mod 2^64. */
__host__ __device__ __forceinline__
static uint64_t hh_weight(const uint32_t nonce)
{
	return 2ull * (uint64_t)nonce + 1ull;
}

__global__
void heavyhash_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonces)
{
	__shared__ uint32_t matrix[64 * 16];	// 4 KB packed

	// Cooperative fill of the whole matrix, then one barrier. Both must sit OUTSIDE the
	// thread<threads guard: every thread reads every row below, and a ragged tail block
	// would otherwise leave part of the table unwritten and skip the barrier.
	{
		const uint32_t *cp = (const uint32_t *)(c_matrix);
		for (uint32_t i = threadIdx.x; i < 64 * 16; i += blockDim.x)
			matrix[i] = cp[i];
	}
	__syncthreads();

    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    uint32_t nonce = startNonce + thread;
    if (thread < threads)
	{
        const hash_t hash = heavyhash_one(matrix, nonce);

		if ( hash.h8[3] <= ((uint64_t *) pTarget)[3]) {
			// Keep the two lowest candidates. atomicMin returns the previous minimum, so the
			// displaced value goes to slot 1 in the same pass; reading resNonces[0] separately
			// lets two concurrent candidates both win slot 0 and drops one silently.
			const uint32_t prev = atomicMin(&resNonces[0], nonce);
			if (prev != UINT32_MAX)
				atomicMin(&resNonces[1], max(prev, nonce));
		}
    }
}

/* -D differential: the two checksums over a nonce range. Reads no target, so
 * it also covers digests the mining screen would discard. */
__global__
void heavyhash_gpu_checksum(const uint32_t threads, const uint32_t startNonce, uint64_t *acc)
{
	__shared__ uint32_t matrix[64 * 16];

	{
		const uint32_t *cp = (const uint32_t *)(c_matrix);
		for (uint32_t i = threadIdx.x; i < 64 * 16; i += blockDim.x)
			matrix[i] = cp[i];
	}
	__syncthreads();

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t nonce = startNonce + thread;
	if (thread < threads)
	{
		const hash_t hash = heavyhash_one(matrix, nonce);
		const uint64_t q = hh_fold(hash.h8);

		atomicXor((unsigned long long *)&acc[0], (unsigned long long)q);
		atomicXor((unsigned long long *)&acc[1], (unsigned long long)(q * hh_weight(nonce)));
	}
}

__host__
void heavyhash_cpu_setBlock_80(uint32_t *pdata)
{
	uint32_t data[20];
	for (int k = 0; k < 20; k++)
		be32enc(&data[k], pdata[k]);

	cudaMemcpyToSymbol(c_data, &data[0], sizeof(c_data), 0, cudaMemcpyHostToDevice);

    uint32_t seed[8];
    uint32_t matrix[64][64];
    struct xoshiro_state state;

    kt_sha3_256((uint8_t *)seed, 32, (const uint8_t *)(data+1), 32);

    for (int i = 0; i < 4; ++i) {
        state.s[i] = le64dec(seed + 2*i);
    }

    generate_matrix(matrix, &state);

    // pack 4 nibbles per word, one per byte, LSB-first to match __dp4a's lane order
    uint32_t packed[64][16];
    for (int i = 0; i < 64; i++)
        for (int w = 0; w < 16; w++)
            packed[i][w] =  matrix[i][4*w]
                         | (matrix[i][4*w+1] << 8)
                         | (matrix[i][4*w+2] << 16)
                         | (matrix[i][4*w+3] << 24);

    cudaMemcpyToSymbol(c_matrix, &packed[0][0], sizeof(c_matrix), 0, cudaMemcpyHostToDevice);
}

__host__
void heavyhash_cpu_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32, 0, cudaMemcpyHostToDevice);
}

extern uint32_t heavyhash_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, int order);

/* Init self-test, three legs, fail-closed.
 *   sha3 - host SHA3-256("") vs the published FIPS-202 vector, which pins the
 *          0x06 domain byte; no self-generated vector can.
 *   kat  - the shipping launcher must find the same nonce as the host.
 *   neg  - a flipped header word must change the device's answer.
 */
#define HH_ST_SPAN	256u
#define HH_ST_START	0x51000000u

/* Per-job host setup, mirroring heavyhash_cpu_setBlock_80. Call this once and
 * use heavyhash(): heavyhash_hash() regenerates the matrix on every call. */
static void hh_host_job(const uint32_t *pdata, uint32_t data[20], uint32_t matrix[64][64])
{
	uint32_t seed[8];
	struct xoshiro_state state;

	for (int k = 0; k < 20; k++)
		be32enc(&data[k], pdata[k]);

	kt_sha3_256((uint8_t *)seed, 32, (const uint8_t *)(data + 1), 32);
	for (int i = 0; i < 4; ++i)
		state.s[i] = le64dec(seed + 2*i);
	generate_matrix(matrix, &state);
}

static void hh_st_host_span(const uint32_t *pdata, uint32_t *best_nonce, uint32_t *best_dig)
{
	uint32_t data[20], matrix[64][64];

	hh_host_job(pdata, data, matrix);

	uint64_t best = 0xffffffffffffffffull;
	*best_nonce = UINT32_MAX;

	for (uint32_t n = 0; n < HH_ST_SPAN; ++n) {
		uint32_t dig[8];
		data[19] = swab32(HH_ST_START + n);
		heavyhash(matrix, (uint8_t *)data, 80, (uint8_t *)dig);
		const uint64_t top = ((uint64_t *)dig)[3];
		if (top < best) {
			best = top;
			*best_nonce = HH_ST_START + n;
			memcpy(best_dig, dig, 32);
		}
	}
}

__host__
bool heavyhash_device_selftest(int thr_id)
{
	/* SHA3-256("") -- FIPS-202 */
	static const uint8_t sha3_empty[32] = {
		0xa7,0xff,0xc6,0xf8,0xbf,0x1e,0xd7,0x66, 0x51,0xc1,0x47,0x56,0xa0,0x61,0xd6,0x62,
		0xf5,0x80,0xff,0x4d,0xe4,0x3b,0x49,0xfa, 0x82,0xd8,0x0a,0x4b,0x80,0xf8,0x43,0x4a
	};
	uint8_t d[32];
	kt_sha3_256(d, 32, (const uint8_t *)"", 0);
	const bool sha3_ok = memcmp(d, sha3_empty, 32) == 0;

	uint32_t pdata[20];
	for (int i = 0; i < 20; i++)
		pdata[i] = 0x01020304u * (uint32_t)(i + 1);

	/* The target is the lowest digest in the span, so exactly that nonce passes. */
	uint32_t want = UINT32_MAX, dig[8];
	hh_st_host_span(pdata, &want, dig);

	heavyhash_cpu_setBlock_80(pdata);
	heavyhash_cpu_setTarget(dig);
	const uint32_t got = heavyhash_cpu_hash(thr_id, HH_ST_SPAN, HH_ST_START, 0);
	const bool kat_ok = (want != UINT32_MAX) && (got == want);

	/* Same target, one header bit flipped: the answer must not survive. */
	pdata[0] ^= 1u;
	heavyhash_cpu_setBlock_80(pdata);
	const uint32_t flipped = heavyhash_cpu_hash(thr_id, HH_ST_SPAN, HH_ST_START, 0);
	const bool neg_ok = (flipped != want);

	// Don't leave the flipped header in the device symbols. scanhash uploads
	// its own before the first launch, but a stale one is not worth the risk.
	pdata[0] ^= 1u;
	heavyhash_cpu_setBlock_80(pdata);

	const bool passed = sha3_ok && kat_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "heavyhash self-test FAILED (sha3 %d kat %d neg %d)"
			" want=%08x got=%08x", (int)sha3_ok, (int)kat_ok, (int)neg_ok, want, got);

	return selftest_gate(thr_id, "heavyhash", passed);
}

/* ---- -D range differential -----------------------------------------------
 * The host re-verify only ever sees candidates, so a kernel that MISSES valid
 * nonces is silent. This compares every digest in a range. */
/* Not a multiple of the 256-thread block, so the ragged tail is covered. */
#define HH_DIFF_SPAN	4133u
#define HH_DIFF_SKEW	0x51u		// keep the start off zero and off a block boundary

static uint32_t hh_diff_sig[MAX_GPUS];
static bool hh_diff_seen[MAX_GPUS];

static bool hh_gpu_checksum(uint32_t threads, uint32_t startNonce, uint64_t out[2])
{
	uint64_t *d_acc = NULL;
	const uint32_t tpb = 256;
	bool ok = false;

	if (cudaMalloc(&d_acc, 2 * sizeof(uint64_t)) != cudaSuccess)
		return false;

	if (cudaMemset(d_acc, 0, 2 * sizeof(uint64_t)) == cudaSuccess) {
		heavyhash_gpu_checksum<<<(threads + tpb - 1) / tpb, tpb>>>(threads, startNonce, d_acc);
		ok = cudaMemcpy(out, d_acc, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess;
	}

	cudaFree(d_acc);
	return ok;
}

static void hh_host_checksum(const uint32_t *pdata, uint32_t threads, uint32_t startNonce,
	uint64_t out[2])
{
	uint32_t data[20], matrix[64][64];

	hh_host_job(pdata, data, matrix);
	out[0] = out[1] = 0;

	for (uint32_t i = 0; i < threads; ++i) {
		const uint32_t nonce = startNonce + i;
		uint64_t dig[4];

		data[19] = swab32(nonce);
		heavyhash(matrix, (uint8_t *)data, 80, (uint8_t *)dig);

		const uint64_t q = hh_fold(dig);
		out[0] ^= q;
		out[1] ^= q * hh_weight(nonce);
	}
}

// Once per job: the header words are what select the matrix and the digest.
void heavyhash_debug_differential(int thr_id, const uint32_t *pdata, uint32_t startNonce)
{
	uint32_t sig = 0x811c9dc5u;
	for (int i = 0; i < 19; i++)
		sig = (sig ^ pdata[i]) * 0x01000193u;

	if (hh_diff_seen[thr_id] && hh_diff_sig[thr_id] == sig)
		return;
	hh_diff_sig[thr_id] = sig;
	hh_diff_seen[thr_id] = true;

	/* Live jobs hand us pdata[19] == 0, which would leave startNonce untested. */
	startNonce += HH_DIFF_SKEW;

	uint64_t gpu[2], cpu[2];
	if (!hh_gpu_checksum(HH_DIFF_SPAN, startNonce, gpu)) {
		gpulog(LOG_WARNING, thr_id, "differential could not run (CUDA resource failure)");
		return;
	}
	hh_host_checksum(pdata, HH_DIFF_SPAN, startNonce, cpu);

	if (gpu[0] != cpu[0] || gpu[1] != cpu[1]) {
		gpulog(LOG_ERR, thr_id, "DIFFERENTIAL MISMATCH over %u nonces from %08x:"
			" gpu %016llx/%016llx cpu %016llx/%016llx", HH_DIFF_SPAN, startNonce,
			(unsigned long long)gpu[0], (unsigned long long)gpu[1],
			(unsigned long long)cpu[0], (unsigned long long)cpu[1]);
		return;
	}

	// Print the accumulator: a checksum that never varies is a passing test that
	// proves nothing, and this is the only place it can be seen to vary.
	gpulog(LOG_DEBUG, thr_id, "differential OK: %u nonces from %08x, acc %016llx/%016llx",
		HH_DIFF_SPAN, startNonce, (unsigned long long)cpu[0], (unsigned long long)cpu[1]);
}

__host__
void heavyhash_cpu_init(int thr_id)
{
    cudaMalloc(&d_GNonces[thr_id], 2*sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2*sizeof(uint32_t));
	dirty_GNonces[thr_id] = true;	// force the first arming

	heavyhash_device_selftest(thr_id);
}

__host__
uint32_t heavyhash_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, int order)
{
	uint32_t result = UINT32_MAX;
	const uint32_t threadsperblock = 256;

	// The kernel only ever writes this buffer when it finds a candidate, so re-arm it on the
	// launch after a hit rather than on every launch.
	if (dirty_GNonces[thr_id]) {
		cudaMemset(d_GNonces[thr_id], 0xff, 2*sizeof(uint32_t));
		dirty_GNonces[thr_id] = false;
	}

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// The matrix lives in a static __shared__ array; no dynamic shared memory is used.
	heavyhash_gpu_hash<<<grid, block>>>(threads, startNounce, d_GNonces[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);

	// get first found nonce
	cudaMemcpy(h_GNonces[thr_id], d_GNonces[thr_id], 1*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	result = *h_GNonces[thr_id];

	if (result != UINT32_MAX)
		dirty_GNonces[thr_id] = true;

	return result;
}

__host__
uint32_t heavyhash_getSecNonce(int thr_id, int num)
{
	uint32_t results[2];
	memset(results, 0xFF, sizeof(results));
	cudaMemcpy(results, d_GNonces[thr_id], sizeof(results), cudaMemcpyDeviceToHost);
	if (results[1] == results[0])
		return UINT32_MAX;
	return results[num];
}

__host__
void heavyhash_cpu_free(int thr_id)
{
	cudaFree(d_GNonces[thr_id]);
	cudaFreeHost(h_GNonces[thr_id]);
}
