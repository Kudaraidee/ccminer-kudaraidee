/*
 * HoohashV110 (PEPEPOW) — CUDA device kernels + launchers.
 *
 * Kept SEPARATE from the host scanhash glue (hoohash.cu): the device path includes
 * hoohash_device.cuh; this TU includes ONLY cuda_runtime + the device headers, while
 * the host glue includes miner.h (avoids macro clashes).
 *
 * PERF: matrixSeed = BLAKE3(header with nonce zeroed) is NONCE-INDEPENDENT, so the
 * 64x64 double matrix is identical for every nonce in a job. We generate it ONCE per
 * job (hoohash_gen_matrix -> d_hoo_matrix in device global) instead of per-thread; the
 * mining kernel then only does the nonce-dependent firstPass + matmul. The matmul's
 * mat[i][j] access is uniform across a warp (the i,j loop is nonce-independent), so the
 * single global matrix is a broadcast/L2-friendly read.
 *
 * MUST be compiled with strict FP (no --use_fast_math; --fmad=false --prec-div=true
 * --prec-sqrt=true --ftz=false) — see the per-file build settings.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "hoohash/hoohash_device.cuh"  // hoo_generateMatrix / hoo_matmul / bundled BLAKE3

// 80-byte header (be32enc'd consensus serialization); mining kernel overwrites nonce @76..79.
__constant__ uint8_t c_hoohash_header[80];

// Per-job matrix, generated once from the (nonce-zeroed) header. Per-device global.
__device__ double d_hoo_matrix[64][64];

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
		hoo_blake3_256(masked, 80, seed);
		hoo_generateMatrix(seed, d_hoo_matrix);
	}
}

// Per-nonce: firstPass = BLAKE3(full header), matmul against the precomputed matrix.
// Digest stored BYTE-REVERSED so cuda_check_hash / fulltest (word7 = MSB) compare the
// big-endian digest correctly. cuda_checkhash_64 strides 64-byte (16-word) slots.
__global__ void hoohash_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t* outputHash)
{
	uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads) return;

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
	hoo_blake3_256(header, 80, firstPass);

	uint64_t non = (uint64_t)hoo_read_u32le(header + 76);

	uint8_t digest[32];
	hoo_matmul(d_hoo_matrix, firstPass, digest, non);

	uint8_t* out = (uint8_t*)(outputHash + thread * 16);
	#pragma unroll
	for (int i = 0; i < 32; i++) out[i] = digest[31 - i];
}

extern "C" void hoohash_setBlock(const void* endiandata)
{
	cudaMemcpyToSymbol(c_hoohash_header, endiandata, 80, 0, cudaMemcpyHostToDevice);
}

extern "C" void hoohash_gen_matrix(void)
{
	hoohash_gen_matrix_kernel<<<1, 1>>>();
}

extern "C" void hoohash_cpu_hash(uint32_t threads, uint32_t startNonce, uint32_t* d_hash, uint32_t tpb)
{
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);
	hoohash_gpu_hash<<<grid, block>>>(threads, startNonce, d_hash);
}

// Real-block KAT (height 0x4734dd). Validates THIS GPU's libdevice == consensus libm
// (glibc) before mining; see hoohash-real-block-kat. Header is the raw 80-byte on-chain
// serialization; expected is the big-endian digest (= block hash, under target).
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

// Exercises the ACTUAL mining path (gen kernel + mining kernel + digest reversal) so it
// validates exactly what mines. KAT nonce 0x4d94e755 -> be32enc bytes 4d 94 e7 55.
// Returns 1 on consensus match, else 0.
extern "C" int hoohash_gpu_self_test(void)
{
	cudaMemcpyToSymbol(c_hoohash_header, hoohash_kat_header, 80, 0, cudaMemcpyHostToDevice);
	hoohash_gen_matrix_kernel<<<1, 1>>>();

	uint32_t* d_out = NULL;
	if (cudaMalloc(&d_out, 16 * sizeof(uint32_t)) != cudaSuccess) return 0;
	hoohash_gpu_hash<<<1, 1>>>(1, 0x4d94e755u, d_out);
	cudaDeviceSynchronize();

	uint32_t got16[16];
	cudaMemcpy(got16, d_out, 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(d_out);

	// First 32 bytes of the slot hold the REVERSED digest; un-reverse to big-endian.
	const uint8_t* gr = (const uint8_t*)got16;
	uint8_t got[32];
	for (int i = 0; i < 32; i++) got[i] = gr[31 - i];

	return memcmp(got, hoohash_kat_expected, 32) == 0 ? 1 : 0;
}
