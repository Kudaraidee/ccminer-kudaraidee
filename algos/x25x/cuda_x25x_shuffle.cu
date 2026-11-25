/**
 * X25X byte-shuffle stage (SUQA/SIN).
 *
 * One thread per hash. Operates in place on the 1536-byte accumulator
 * (24 x 64-byte slots hash[0..23]) viewed as 768 little-endian uint16 words:
 * 12 rounds of the SUQA "simple shuffle" mixing. Inherently serial per hash
 * (each word update may read an already-updated word), so it runs as a single
 * thread over a register/local copy of the 768 words.
 *
 * Transcribed from algos/x22/x25x.c (X25X_SHUFFLE_ROUNDS / _BLOCKS loop).
 */

#include "cuda/helper.cuh"

#define X25X_SHUF_WORDS  (24 * 64 / 2)   /* 768 */
#define X25X_SHUF_ROUNDS 12

__device__ __constant__ static const uint16_t x25x_round_const[X25X_SHUF_ROUNDS] = {
	0x142c, 0x5830, 0x678c, 0xe08c, 0x3c67, 0xd50d, 0xb1d8, 0xecb2,
	0xd7ee, 0x6783, 0xfa6c, 0x4b9c
};

// Serial dependent recurrence per hash: 9216 steps each doing 2 random reads +
// 1 write of the 768-word working set. In a LOCAL array those random accesses
// hit L2 (~1434 ms in-pipeline -- 82% of the whole hash). Held in SHARED memory
// instead, each access is a single fast transaction; even though 48 KB/block
// caps occupancy at ~64 threads/SM, the ~16x lower per-access latency wins
// decisively when the stage runs serialized in the real pipeline (the earlier
// "shared is worse" call was based on overlap-distorted isolated timings).
// thread-minor layout sm[idx*TPB + tid] keeps the load/store reasonably banked.
#define X25X_SHUF_TPB 32   // 32 * 768 * 2 = 48 KB shared / block (static max)

__global__ __launch_bounds__(X25X_SHUF_TPB, 2)
void x25x_shuffle_gpu(uint32_t threads, uint16_t *g_acc)
{
	__shared__ uint16_t sm[X25X_SHUF_TPB * X25X_SHUF_WORDS];
	const uint32_t tid    = threadIdx.x;
	const uint32_t thread = blockIdx.x * X25X_SHUF_TPB + tid;
	if (thread >= threads)
		return;

	#define SH(idx)   sm[(idx) * X25X_SHUF_TPB + tid]
	// slot-major accumulator: word i of this hash lives in plane (i/32) at
	// uint16 offset (thread*32 + i%32); coalesced across threads per plane.
	#define GACC(i)   g_acc[(size_t)((i) >> 5) * threads * 32 + (size_t)thread * 32 + ((i) & 31)]

	#pragma unroll 1
	for (int i = 0; i < X25X_SHUF_WORDS; i++) SH(i) = GACC(i);

	#pragma unroll 1
	for (int r = 0; r < X25X_SHUF_ROUNDS; r++) {
		const uint32_t rc = x25x_round_const[r];
		#pragma unroll 1
		for (int i = 0; i < X25X_SHUF_WORDS; i++) {
			const uint16_t bv = SH(X25X_SHUF_WORDS - i - 1);
			const uint32_t add = (uint32_t)SH(bv % X25X_SHUF_WORDS) + (rc << (i % 16));
			SH(i) ^= (uint16_t)add;
		}
	}

	#pragma unroll 1
	for (int i = 0; i < X25X_SHUF_WORDS; i++) GACC(i) = SH(i);
	#undef SH
	#undef GACC
}

// d_acc: threads x 1536 bytes (24 slots), shuffled in place.
__host__ void x25x_shuffle_cpu(int thr_id, uint32_t threads, uint32_t *d_acc)
{
	dim3 grid((threads + X25X_SHUF_TPB - 1) / X25X_SHUF_TPB);
	dim3 block(X25X_SHUF_TPB);

	x25x_shuffle_gpu<<<grid, block>>>(threads, (uint16_t*)d_acc);
}
