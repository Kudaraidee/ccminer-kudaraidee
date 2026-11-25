/*
 * HoohashV110 (PEPEPOW) - CUDA device kernels + launchers.
 *
 * The 64x64 matrix and its matdiv*v table depend only on the nonce-zeroed header, so
 * they are generated once per scan call (hoohash_gen_matrix).
 *
 * MUST be compiled with strict FP (no --use_fast_math; --fmad=false --prec-div=true
 * --prec-sqrt=true --ftz=false) - see the per-file build settings.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "algos/hoohash/hoohash_device.cuh"  // matrix generation, transforms, hoo_sw_low
#include "cuda/selftest_gate.cuh"            // selftest_gate / selftest_cuda_fault

// 80-byte header (be32enc'd consensus serialization); mining kernel overwrites nonce @76..79.
__constant__ uint8_t c_hoohash_header[80];

// Per-job matrix, generated once from the (nonce-zeroed) header. Per-device global.
__device__ double d_hoo_matrix[64][64];
// mat[i][j] * divider, the reference's intermediate for the cheap step.
__device__ double d_hoo_matrix_div[64][64];
// matdiv * v for every nibble value v: the cheap step's addend.
__device__ double d_hoo_tab[16 * 4096];

// Generate the per-job matrix: matrixSeed = BLAKE3(header80 with nonce bytes zeroed),
// then xoshiro-fill d_hoo_matrix. xoshiro is a sequential stream -> single thread.
__global__ void hoohash_gen_matrix_kernel()
{
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		uint8_t masked[80];
		#pragma unroll
		for (int i = 0; i < 80; i++) masked[i] = c_hoohash_header[i];
		masked[76] = masked[77] = masked[78] = masked[79] = 0;

		uint8_t seed[32];
		blake3_256(masked, 80, seed);
		hoo_generateMatrix(seed, d_hoo_matrix, d_hoo_matrix_div);
	}
}

__global__ void hoohash_tab_kernel()
{
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	const double md = (&d_hoo_matrix_div[0][0])[k];
	for (int v = 0; v < 16; v++) d_hoo_tab[v * 4096 + k] = md * (double)v;
}

static void hoohash_gen_all(void)
{
	hoohash_gen_matrix_kernel<<<1, 1>>>();
	hoohash_tab_kernel<<<64, 64>>>();
}

// Run a transform phase once this many lanes wait (or none can step); tuned per arch.
// sm_6x also trades registers for occupancy; sm_75+ must stay without bounds.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define HOO_WAIT_K 20
#define HOOHASH_GPU_HASH_BOUNDS __launch_bounds__(128, 10)
#else
#define HOO_WAIT_K 24
#define HOOHASH_GPU_HASH_BOUNDS
#endif

// Nibble j of firstPass (high nibble first), from four big-endian words held in registers.
static __device__ __forceinline__ uint8_t hoo_nibble(const uint64_t w[4], int j)
{
	const uint64_t x = j < 16 ? w[0] : j < 32 ? w[1] : j < 48 ? w[2] : w[3];
	return (uint8_t)((x >> (60 - 4 * (j & 15))) & 0xF);
}

// Per-nonce: firstPass = BLAKE3(full header), then the 64x64 scan. Each lane walks its own
// (i, j) in reference order; lanes needing the transform wait and are evaluated together.
// Digest stored BYTE-REVERSED so cuda_check_hash / fulltest (word7 = MSB) compare the
// big-endian digest correctly. cuda_checkhash_64 strides 64-byte (16-word) slots.
__global__ void HOOHASH_GPU_HASH_BOUNDS hoohash_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t* outputHash)
{
	const unsigned lanes = __activemask();
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	const bool live = thread < threads;   // tail lanes still take part in the votes

	uint8_t header[80];
	#pragma unroll
	for (int i = 0; i < 80; i++) header[i] = c_hoohash_header[i];

	uint32_t nonce = startNonce + thread;
	// be32enc(&edata[19], nonce): nonce stored BIG-ENDIAN at bytes 76..79.
	header[76] = (uint8_t)(nonce >> 24);
	header[77] = (uint8_t)(nonce >> 16);
	header[78] = (uint8_t)(nonce >> 8);
	header[79] = (uint8_t)(nonce);

	uint8_t firstPass[32];
	blake3_256(header, 80, firstPass);

	uint32_t H[8];
	for (int k = 0; k < 8; k++) H[k] = hoo_read_u32be(firstPass + k * 4);
	const double hashXor = (double)(H[0] ^ H[1] ^ H[2] ^ H[3] ^ H[4] ^ H[5] ^ H[6] ^ H[7]);
	const double nonceMod = (double)(hoo_read_u32le(header + 76) & 0xFF);
	const double multiplier = 1234;

	uint64_t nw[4];
	for (int k = 0; k < 4; k++) {
		uint64_t w = 0;
		for (int b = 0; b < 8; b++) w = (w << 8) | firstPass[8 * k + b];
		nw[k] = w;
	}

	uint8_t scaled[32];
	int i = live ? 0 : 64, j = 0;
	double prod = 0.0;
	bool low = true;          // sw <= 0.02; sw starts at 0 and carries across rows
	uint64_t pend = 0;

	for (;;) {
		const bool done = (i >= 64);
		const uint8_t v = done ? 0 : hoo_nibble(nw, j);
		const bool needT = !done && v != 0 && low;   // v == 0 adds exactly +0.0 on either branch
		const unsigned mT = __ballot_sync(lanes, needT);
		const unsigned mC = __ballot_sync(lanes, !done && !needT);
		if (!(mT | mC)) break;
		bool step = false;
		if (mT && (__popc(mT) >= HOO_WAIT_K || !mC)) {
			if (needT) {
				const double input = (d_hoo_matrix[i][j] * hashXor * (double)v + nonceMod);
				prod += hoo_SafeComplexTransform(input) * (double)v * multiplier;
				step = true;
			}
		} else if (!done && !needT) {
			if (v) prod += d_hoo_tab[v * 4096 + i * 64 + j];
			step = true;
		}
		if (step) {
			low = hoo_sw_low(prod);
			if (++j == 64) {
				const uint64_t pv = (uint64_t)prod;
				if (i & 1) scaled[i >> 1] = (uint8_t)((pend + pv) & 0xFF);
				else pend = pv;
				i++; j = 0; prod = 0.0;
			}
		}
	}
	if (!live) return;

	uint8_t result[32], digest[32];
	for (int k = 0; k < 32; k++) result[k] = firstPass[k] ^ scaled[k];
	blake3_256(result, 32, digest);

	uint8_t* out = (uint8_t*)(outputHash + thread * 16);
	#pragma unroll
	for (int k = 0; k < 32; k++) out[k] = digest[31 - k];
}

extern "C" void hoohash_setBlock(const void* endiandata)
{
	cudaMemcpyToSymbol(c_hoohash_header, endiandata, 80, 0, cudaMemcpyHostToDevice);
}

extern "C" void hoohash_gen_matrix(void)
{
	hoohash_gen_all();
}

extern "C" void hoohash_cpu_hash(uint32_t threads, uint32_t startNonce, uint32_t* d_hash, uint32_t tpb)
{
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);
	hoohash_gpu_hash<<<grid, block>>>(threads, startNonce, d_hash);
}

// Real-block KAT (height 0x4734dd): this GPU's libdevice must match consensus libm (glibc).
// Header is the raw 80-byte on-chain serialization; expected is the big-endian block hash.
static const uint8_t hoohash_kat_header[80] = {
	0x00,0x40,0x00,0x20, 0xdf,0x13,0xf8,0xc7, 0x24,0x3b,0x6b,0x22, 0x6c,0x33,0x99,0xf4,
	0x85,0xe9,0x02,0x34, 0x7e,0x41,0xba,0x37, 0x0e,0x0c,0x8f,0xea, 0x75,0x67,0x2f,0x45,
	0x01,0x00,0x00,0x00, 0x22,0xda,0x19,0x46, 0xe7,0xdb,0xa6,0x53, 0x52,0xae,0x0f,0x65,
	0xce,0x77,0xdc,0xff, 0x51,0x05,0xe2,0xf4, 0x6a,0x44,0x3e,0x3a, 0x86,0xbb,0x35,0x9b,
	0xb6,0x78,0x8b,0xf9, 0x62,0xa2,0x41,0x6a, 0x33,0xce,0x01,0x1d, 0x4d,0x94,0xe7,0x55
};
static const uint8_t hoohash_kat_expected[32] = {
	0x00,0x00,0x00,0x01, 0x3e,0x74,0xaa,0xd7, 0x1e,0x79,0xfd,0x0e, 0x33,0x03,0xc5,0x14,
	0xaf,0x06,0xbc,0x1b, 0x9f,0x26,0xd6,0xa9, 0x94,0xb6,0x5e,0xb6, 0x6d,0x17,0x84,0x5d
};

// One header through the mining kernels. Returns false only on a CUDA failure.
static bool hoohash_kat_run(const uint8_t hdr[80], uint32_t nonce, uint8_t out_be[32])
{
	if (cudaMemcpyToSymbol(c_hoohash_header, hdr, 80, 0, cudaMemcpyHostToDevice) != cudaSuccess)
		return selftest_cuda_fault();
	hoohash_gen_all();

	uint32_t* d_out = NULL;
	if (cudaMalloc(&d_out, 16 * sizeof(uint32_t)) != cudaSuccess)
		return selftest_cuda_fault();
	hoohash_gpu_hash<<<1, 1>>>(1, nonce, d_out);
	if (cudaDeviceSynchronize() != cudaSuccess) {
		cudaFree(d_out);
		return selftest_cuda_fault();
	}

	// Only the first 32 bytes of the 64-byte slot are written.
	uint32_t got[8];
	cudaError_t err = cudaMemcpy(got, d_out, sizeof(got), cudaMemcpyDeviceToHost);
	cudaFree(d_out);
	if (err != cudaSuccess)
		return selftest_cuda_fault();

	// Stored byte-reversed; un-reverse to big-endian.
	const uint8_t* gr = (const uint8_t*)got;
	for (int i = 0; i < 32; i++) out_be[i] = gr[31 - i];
	return true;
}

// The CPU reference that re-verifies every candidate (hoohash.cu).
extern "C" void hoohash_ref_hash(const void* hdr80, uint8_t out_be[32]);

// Init self-test, fail-closed.
//   kat - the real block above must hash to its published digest on the GPU.
//   neg - a flipped header bit must change the digest.
//   cpu - the CPU reference must hash the same block to the same digest, or it
//         would reject every share.
extern "C" bool hoohash_device_selftest(int thr_id)
{
	uint8_t got[32], neg[32], cpu[32];
	bool kat_ok = false, neg_ok = false;

	hoohash_ref_hash(hoohash_kat_header, cpu);
	const bool cpu_ok = (memcmp(cpu, hoohash_kat_expected, 32) == 0);

	if (hoohash_kat_run(hoohash_kat_header, 0x4d94e755u, got))
		kat_ok = (memcmp(got, hoohash_kat_expected, 32) == 0);

	uint8_t bad[80];
	memcpy(bad, hoohash_kat_header, 80);
	bad[13] ^= 0x08;
	if (hoohash_kat_run(bad, 0x4d94e755u, neg))
		neg_ok = (memcmp(neg, got, 32) != 0);

	// Never leave the flipped header or its matrix in the device symbols.
	cudaMemcpyToSymbol(c_hoohash_header, hoohash_kat_header, 80, 0, cudaMemcpyHostToDevice);
	hoohash_gen_all();

	const bool passed = kat_ok && neg_ok && cpu_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "HoohashV110 self-test FAILED (kat %d neg %d cpu %d)"
			" -- libdevice or host libm != consensus libm?", (int)kat_ok, (int)neg_ok, (int)cpu_ok);

	return selftest_gate(thr_id, "HoohashV110", passed);
}
