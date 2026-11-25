/**
 * PANAMA-256 stage for X25X (SUQA/SIN).
 *
 * One thread per hash. Consumes a 64-byte input slot (hash[21] in the chain),
 * runs the PANAMA sponge and writes the 256-bit (32-byte) result into the low
 * half of the slot, zeroing the high half (the accumulate-all buffer keeps
 * short digests zero-padded).
 *
 * Faithful transcription of algos/x25x/sph_panama.c: for a 64-byte message the
 * sponge takes three 32-byte push blocks (the two input halves + the 0x01
 * padding block) followed by 32 pull rounds, then emits state[9..16] LE.
 */

#include "cuda/helper.cuh"

// GAMMA -> PI -> THETA core: a[17] -> t[17]. Does not modify a[].
__device__ __forceinline__
static void panama_core(const uint32_t a[17], uint32_t t[17])
{
	uint32_t g[17], p[17];

	#pragma unroll
	for (int k = 0; k < 17; k++)
		g[k] = a[k] ^ (a[(k + 1) % 17] | ~a[(k + 2) % 17]);

	p[0]  = g[0];
	p[1]  = ROTL32(g[7],   1);
	p[2]  = ROTL32(g[14],  3);
	p[3]  = ROTL32(g[4],   6);
	p[4]  = ROTL32(g[11], 10);
	p[5]  = ROTL32(g[1],  15);
	p[6]  = ROTL32(g[8],  21);
	p[7]  = ROTL32(g[15], 28);
	p[8]  = ROTL32(g[5],   4);
	p[9]  = ROTL32(g[12], 13);
	p[10] = ROTL32(g[2],  23);
	p[11] = ROTL32(g[9],   2);
	p[12] = ROTL32(g[16], 14);
	p[13] = ROTL32(g[6],  27);
	p[14] = ROTL32(g[13],  9);
	p[15] = ROTL32(g[3],  24);
	p[16] = ROTL32(g[10],  8);

	#pragma unroll
	for (int k = 0; k < 17; k++)
		t[k] = p[k] ^ p[(k + 1) % 17] ^ p[(k + 4) % 17];
}

// PANAMA push: absorb one 32-byte block (8 words).
__device__ __forceinline__
static void panama_push(uint32_t a[17], uint32_t buf[32][8], uint32_t &ptr0, const uint32_t blk[8])
{
	const uint32_t ptr24 = (ptr0 - 8) & 31;
	const uint32_t ptr31 = (ptr0 - 1) & 31;

	#pragma unroll
	for (int n0 = 0; n0 < 8; n0++) {
		const int n2 = (n0 + 2) & 7;
		buf[ptr24][n0] ^= buf[ptr31][n2];
		buf[ptr31][n2] ^= blk[n2];
	}

	uint32_t t[17];
	panama_core(a, t);

	const uint32_t ptr16 = ptr0 ^ 16;
	a[0] = t[0] ^ 1u;
	#pragma unroll
	for (int i = 0; i < 8; i++) a[1 + i] = t[1 + i] ^ blk[i];
	#pragma unroll
	for (int i = 0; i < 8; i++) a[9 + i] = t[9 + i] ^ buf[ptr16][i];

	ptr0 = ptr31;
}

// PANAMA pull: one squeezing round (no external input).
__device__ __forceinline__
static void panama_pull(uint32_t a[17], uint32_t buf[32][8], uint32_t &ptr0)
{
	const uint32_t ptr4  = (ptr0 + 4) & 31;
	const uint32_t ptr24 = (ptr0 - 8) & 31;
	const uint32_t ptr31 = (ptr0 - 1) & 31;

	#pragma unroll
	for (int n0 = 0; n0 < 8; n0++) {
		const int n2 = (n0 + 2) & 7;
		buf[ptr24][n0] ^= buf[ptr31][n2];
		buf[ptr31][n2] ^= a[n2 + 1];      // INW1(n2) = state word (n2+1)
	}

	uint32_t t[17];
	panama_core(a, t);

	const uint32_t ptr16 = ptr0 ^ 16;
	a[0] = t[0] ^ 1u;
	#pragma unroll
	for (int i = 0; i < 8; i++) a[1 + i] = t[1 + i] ^ buf[ptr4][i];   // INW2 = buffer[ptr4]
	#pragma unroll
	for (int i = 0; i < 8; i++) a[9 + i] = t[9 + i] ^ buf[ptr16][i];

	ptr0 = ptr31;
}

__device__ __forceinline__
static void panama_hash_64(const uint32_t in[16], uint32_t out[8])
{
	uint32_t a[17];
	uint32_t buf[32][8];
	uint32_t ptr0 = 0;

	#pragma unroll
	for (int i = 0; i < 17; i++) a[i] = 0;
	#pragma unroll
	for (int r = 0; r < 32; r++)
		#pragma unroll
		for (int c = 0; c < 8; c++) buf[r][c] = 0;

	uint32_t blk[8];

	#pragma unroll
	for (int i = 0; i < 8; i++) blk[i] = in[i];          // first 32 bytes
	panama_push(a, buf, ptr0, blk);

	#pragma unroll
	for (int i = 0; i < 8; i++) blk[i] = in[8 + i];      // second 32 bytes
	panama_push(a, buf, ptr0, blk);

	blk[0] = 0x00000001u;                                 // 0x01 padding block
	#pragma unroll
	for (int i = 1; i < 8; i++) blk[i] = 0;
	panama_push(a, buf, ptr0, blk);

	#pragma unroll
	for (int i = 0; i < 32; i++) panama_pull(a, buf, ptr0);

	#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = a[9 + i];
}

__global__ void x25x_panama_gpu_hash_64(uint32_t threads, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *Hash = (uint32_t*)&g_hash[thread << 3];
		uint32_t in[16], out[8];
		#pragma unroll
		for (int i = 0; i < 16; i++) in[i] = Hash[i];

		panama_hash_64(in, out);

		#pragma unroll
		for (int i = 0; i < 8; i++) Hash[i] = out[i];
		#pragma unroll
		for (int i = 8; i < 16; i++) Hash[i] = 0;   // zero-pad the 256-bit digest
	}
}

__host__ void x25x_panama_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x25x_panama_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}
