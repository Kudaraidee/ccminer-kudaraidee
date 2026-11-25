/**
 * BLAKE2s finaliser for X25X (SUQA/SIN).
 *
 * One thread per hash: reads the 1536-byte accumulator (24 x 64-byte slots
 * hash[0..23]) and writes the 32-byte BLAKE2s digest (chain slot hash[24]).
 * Unkeyed, 32-byte output -- matches blake2s_simple() in sph/blake2s.h, i.e.
 * blake2s(out, in, NULL, 32, 1536, 0). 1536 = 24 * 64 = exactly 24 blocks.
 */

#include "cuda/helper.cuh"

__device__ __constant__ static const uint32_t blake2s_IV[8] = {
	0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
	0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

__device__ __constant__ static const uint8_t blake2s_sigma[10][16] = {
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 }
};

#define G2S(r, i, a, b, c, d) do { \
		a = a + b + m[blake2s_sigma[r][2*i + 0]]; \
		d = ROTR32(d ^ a, 16); \
		c = c + d; \
		b = ROTR32(b ^ c, 12); \
		a = a + b + m[blake2s_sigma[r][2*i + 1]]; \
		d = ROTR32(d ^ a, 8); \
		c = c + d; \
		b = ROTR32(b ^ c, 7); \
	} while (0)

__device__ __forceinline__
static void blake2s_compress(uint32_t h[8], const uint32_t m[16], uint32_t t0, uint32_t f0)
{
	uint32_t v[16];
	#pragma unroll
	for (int i = 0; i < 8; i++) v[i] = h[i];
	v[8]  = blake2s_IV[0]; v[9]  = blake2s_IV[1]; v[10] = blake2s_IV[2]; v[11] = blake2s_IV[3];
	v[12] = blake2s_IV[4] ^ t0;      // low 32 bits of byte counter
	v[13] = blake2s_IV[5];           // high 32 bits (always 0 here: 1536 < 2^32)
	v[14] = blake2s_IV[6] ^ f0;      // final-block flag
	v[15] = blake2s_IV[7];

	#pragma unroll
	for (int r = 0; r < 10; r++) {
		G2S(r, 0, v[0], v[4], v[ 8], v[12]);
		G2S(r, 1, v[1], v[5], v[ 9], v[13]);
		G2S(r, 2, v[2], v[6], v[10], v[14]);
		G2S(r, 3, v[3], v[7], v[11], v[15]);
		G2S(r, 4, v[0], v[5], v[10], v[15]);
		G2S(r, 5, v[1], v[6], v[11], v[12]);
		G2S(r, 6, v[2], v[7], v[ 8], v[13]);
		G2S(r, 7, v[3], v[4], v[ 9], v[14]);
	}

	#pragma unroll
	for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

// Hash 1536 bytes (24 blocks) -> 32 bytes, unkeyed (blake2s_simple semantics).
__device__ __forceinline__
static void blake2s_1536(const uint32_t *in /* 384 words */, uint32_t out[8])
{
	uint32_t h[8];
	#pragma unroll
	for (int i = 0; i < 8; i++) h[i] = blake2s_IV[i];
	h[0] ^= 0x01010020u;             // digest_length=32, fanout=1, depth=1

	uint32_t t0 = 0;
	#pragma unroll
	for (int b = 0; b < 24; b++) {
		uint32_t m[16];
		#pragma unroll
		for (int i = 0; i < 16; i++) m[i] = in[b * 16 + i];
		t0 += 64;
		const uint32_t f0 = (b == 23) ? 0xFFFFFFFFu : 0u;
		blake2s_compress(h, m, t0, f0);
	}

	#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = h[i];
}

__global__ void x25x_blake2s_gpu_hash(uint32_t threads, const uint64_t *g_acc, uint64_t *g_out)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// slot-major accumulator: gather this thread's 24 slots (block b = slot b,
		// 16 uint32) from their planes into a contiguous 1536-byte buffer.
		const uint32_t *ga = (const uint32_t*)g_acc;
		uint32_t acc[384];
		#pragma unroll
		for (int b = 0; b < 24; b++) {
			#pragma unroll
			for (int i = 0; i < 16; i++)
				acc[b * 16 + i] = ga[(size_t)b * threads * 16 + (size_t)thread * 16 + i];
		}
		uint32_t out[8];
		blake2s_1536(acc, out);
		uint32_t *o = (uint32_t*)&g_out[thread << 2];                  // 32B = 4 uint64
		#pragma unroll
		for (int i = 0; i < 8; i++) o[i] = out[i];
	}
}

// d_acc: threads x 1536 bytes (24 slots).  d_out: threads x 32 bytes.
__host__ void x25x_blake2s_cpu_hash(int thr_id, uint32_t threads, uint32_t *d_acc, uint32_t *d_out)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x25x_blake2s_gpu_hash<<<grid, block>>>(threads, (uint64_t*)d_acc, (uint64_t*)d_out);
}
