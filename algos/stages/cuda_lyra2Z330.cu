/**
 * Lyra2Z330 matrix kernel (-a lyra2z330): timeCost 2, 330 rows, 256 columns.
 *
 * The other lyra2 stages in this folder keep their whole matrix in shared memory
 * (8 rows x 8 cols x 3 uint2 = ~3 KB per hash). This one cannot: 330 x 256 x 96 B
 * is ~7.7 MB per hash, so the matrix lives in global memory and is *streamed* --
 * the row operations walk a row column by column and only the 16-word sponge
 * state stays live between columns.
 *
 * Layout follows the same 4-lanes-per-hash split as cuda_lyra2.cu: the 16-word
 * blake2b state is a 4x4 matrix of uint2 and lane L owns column L, i.e. sponge
 * words { L, 4+L, 8+L, 12+L } as state[0..3]. The 12-word bitrate is therefore
 * state[0..2] of each lane, and one 96-byte matrix column is 3 uint2 per lane.
 * The row ops (including the rotW cross-lane rotation) are transcribed from
 * cuda_lyra2.cu, which is self-tested bit-exact against the sph reference.
 *
 * Deliberately NOT using __ldg for matrix loads: the matrix is written by the
 * same kernel, and the read-only data cache is not coherent with those writes.
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda_lyra2_vectors.h"
#include "cuda/blake2b_device.cuh"

#define Nrow 330
#define Ncol 256
#define memshift 3

// 4 lanes per hash, so a 128-thread block carries 32 hashes
#define TPB 128

__constant__ uint2 c_pad[32];		// pad(pwd || salt || basil): 4 blake2-safe blocks
__constant__ uint2 c_blake2b_IV[8] = {
	{ 0xf3bcc908, 0x6a09e667 },
	{ 0x84caa73b, 0xbb67ae85 },
	{ 0xfe94f82b, 0x3c6ef372 },
	{ 0x5f1d36f1, 0xa54ff53a },
	{ 0xade682d1, 0x510e527f },
	{ 0x2b3e6c1f, 0x9b05688c },
	{ 0xfb41bd6b, 0x1f83d9ab },
	{ 0x137e2179, 0x5be0cd19 }
};

// Lane-interleaved so the 4 lanes of a quad hit 4 consecutive uint2 (64 B) and
// consecutive hashes are adjacent -- one warp covers 512 contiguous bytes.
__device__ __forceinline__
size_t mat_index(const uint32_t row, const uint32_t col, const int j, const uint32_t thread, const uint32_t threads)
{
	return ((size_t)(memshift * (Ncol * row + col) + j) * threads + thread) * 4 + threadIdx.x;
}

__device__ __forceinline__
void LD4G(uint2 res[3], const uint2 *M, const uint32_t row, const uint32_t col, const uint32_t thread, const uint32_t threads)
{
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = M[mat_index(row, col, j, thread, threads)];
}

__device__ __forceinline__
void ST4G(uint2 *M, const uint32_t row, const uint32_t col, const uint2 data[3], const uint32_t thread, const uint32_t threads)
{
	#pragma unroll
	for (int j = 0; j < 3; j++)
		M[mat_index(row, col, j, thread, threads)] = data[j];
}

// M[row*][col] ^= rotW(rand): sponge word w of row* takes state word (w+11) % 12,
// which is the preceding lane's state -- lane 0 additionally shifts by one index.
__device__ __forceinline__
void xor_rotW(uint2 inOut[3], const uint2 state[4], const uint32_t qmask)
{
	uint2 d0 = state[0], d1 = state[1], d2 = state[2];
	QuadShuffle3(d0, d1, d2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, qmask);

	if (threadIdx.x == 0) {
		inOut[0] ^= d2;
		inOut[1] ^= d0;
		inOut[2] ^= d1;
	} else {
		inOut[0] ^= d0;
		inOut[1] ^= d1;
		inOut[2] ^= d2;
	}
}

// reducedSqueezeRow0 + reducedDuplexRow1: fills rows 0 and 1
static __device__ __forceinline__
void reduceDuplex(uint2 *M, uint2 state[4], const uint32_t qmask, const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3];

	#pragma unroll 1
	for (uint32_t i = 0; i < Ncol; i++)
	{
		ST4G(M, 0, Ncol - i - 1, state, thread, threads);
		round_lyra_quad(state, qmask);
	}

	#pragma unroll 1
	for (uint32_t i = 0; i < Ncol; i++)
	{
		LD4G(state1, M, 0, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra_quad(state, qmask);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4G(M, 1, Ncol - i - 1, state1, thread, threads);
	}
}

// reducedDuplexRowSetup: M[rowOut] written back-to-front, M[rowInOut] rotW-updated
static __device__ __forceinline__
void reduceDuplexRowSetup(uint2 *M, const uint32_t rowIn, const uint32_t rowInOut, const uint32_t rowOut,
	uint2 state[4], const uint32_t qmask, const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];

	#pragma unroll 1
	for (uint32_t i = 0; i < Ncol; i++)
	{
		LD4G(state1, M, rowIn, i, thread, threads);
		LD4G(state2, M, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra_quad(state, qmask);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4G(M, rowOut, Ncol - i - 1, state1, thread, threads);

		xor_rotW(state2, state, qmask);
		ST4G(M, rowInOut, i, state2, thread, threads);
	}
}

// reducedDuplexRow (Wandering phase): M[rowOut] and M[rowInOut] both XOR-updated
static __device__ __forceinline__
void reduceDuplexRowt(uint2 *M, const uint32_t rowIn, const uint32_t rowInOut, const uint32_t rowOut,
	uint2 state[4], const uint32_t qmask, const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];

	#pragma unroll 1
	for (uint32_t i = 0; i < Ncol; i++)
	{
		LD4G(state1, M, rowIn, i, thread, threads);
		LD4G(state2, M, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra_quad(state, qmask);

		xor_rotW(state2, state, qmask);
		ST4G(M, rowInOut, i, state2, thread, threads);

		LD4G(state1, M, rowOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4G(M, rowOut, i, state1, thread, threads);
	}
}

__global__ __launch_bounds__(TPB, 1)
void lyra2z330_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint2 *M,
	uint2 *g_hash, uint32_t *resNonces, const uint32_t target)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;
	// Every matrix slot is owned by exactly one quad, so a tail quad must exit
	// rather than clamp onto a neighbour's matrix. Exiting leaves the warp partly
	// inactive, which is why the shuffles below are quad-scoped instead of
	// full-warp: a mask naming exited threads is undefined behaviour.
	if (thread >= threads)
		return;

	const uint32_t lane = threadIdx.x;
	const uint32_t qmask = quad_mask();
	const uint32_t nonce = startNonce + thread;
	// header word 19 is the high half of input words 9 (password) and 19 (salt);
	// the header reaching the sponge is byte-swapped, nonce included
	const uint32_t nonce_be = cuda_swab32(nonce);

	uint2 state[4];

	// initState: words 0..7 zero, words 8..15 the blake2b IV
	state[0] = make_uint2(0, 0);
	state[1] = make_uint2(0, 0);
	state[2] = c_blake2b_IV[lane];
	state[3] = c_blake2b_IV[4 + lane];

	// Setup phase, absorb: 4 x absorbBlockBlake2Safe over pad(pwd || salt || basil)
	#pragma unroll
	for (int b = 0; b < 4; b++)
	{
		uint2 w0 = c_pad[8 * b + lane];
		uint2 w1 = c_pad[8 * b + 4 + lane];

		if (b == 1 && lane == 1) w0.y = nonce_be;
		if (b == 2 && lane == 3) w0.y = nonce_be;

		state[0] ^= w0;
		state[1] ^= w1;

		#pragma unroll
		for (int r = 0; r < 12; r++)		// blake2bLyra
			round_lyra_quad(state, qmask);
	}

	// Setup phase, matrix fill
	reduceDuplex(M, state, qmask, thread, threads);

	uint32_t rowa = 0, prev = 1;
	int step = 1, window = 2, gap = 1;
	for (uint32_t row = 2; row < Nrow; row++)
	{
		reduceDuplexRowSetup(M, prev, rowa, row, state, qmask, thread, threads);

		rowa = (uint32_t)(((int)rowa + step) & (window - 1));
		prev = row;
		if (rowa == 0) {
			step = window + gap;		// bounded: window stops at 256 for 330 rows
			window *= 2;
			gap = -gap;
		}
	}

	// Wandering phase. Row indices are the generic (modulo) form and the modulo is
	// taken on the *unsigned* value, exactly as the reference miners do: for an
	// even tau the step is -1, so row 0 goes to (2^64 - 1) % 330 = 279, not to -1.
	prev = Nrow - 1;
	for (int tau = 1; tau <= 2; tau++)
	{
		const int64_t step64 = (tau & 1) ? (Nrow / 2 - 1) : -1;
		uint32_t row = 0;
		do {
			const uint2 w0 = QuadShuffle(state[0], 0, qmask);	// sponge word 0 lives in lane 0
			rowa = (uint32_t)(devectorize(w0) % (uint64_t)Nrow);

			reduceDuplexRowt(M, prev, rowa, row, state, qmask, thread, threads);

			prev = row;
			row = (uint32_t)((uint64_t)((int64_t)row + step64) % (uint64_t)Nrow);
		} while (row != 0);
	}

	// Wrap-up: absorbBlock over M[rowa]'s first column, then a full blake2bLyra
	uint2 last[3];
	LD4G(last, M, rowa, 0, thread, threads);

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];

	#pragma unroll
	for (int r = 0; r < 12; r++)
		round_lyra_quad(state, qmask);

	// squeeze 32 bytes: sponge words 0..3, i.e. state[0] of each lane
	g_hash[thread * 4 + lane] = state[0];

	// on-device screen on the top 32 bits (vhash[7] = word 3 high half, lane 3);
	// the host re-hashes every candidate before submitting
	if (lane == 3 && state[0].y <= target)
	{
		if (atomicCAS(&resNonces[0], UINT32_MAX, nonce) != UINT32_MAX)
			atomicCAS(&resNonces[1], UINT32_MAX, nonce);
	}
}

// pad(pwd || salt || basil) with the 10*1 padding, per Lyra2Z.c. The nonce word
// is patched per-thread in the kernel, so its value here is irrelevant.
__host__
void lyra2z330_setBlock_80(const uint32_t *endiandata)
{
	uint64_t pad[32];
	uint8_t *p = (uint8_t*) pad;
	const uint64_t basil[6] = { 32, 80, 80, 2, Nrow, Ncol };	// kLen pwdlen saltlen timeCost nRows nCols

	memset(pad, 0, sizeof(pad));
	memcpy(p, endiandata, 80);
	memcpy(p + 80, endiandata, 80);
	memcpy(p + 160, basil, sizeof(basil));
	p[208] = 0x80;
	p[255] ^= 0x01;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_pad, pad, sizeof(pad), 0, cudaMemcpyHostToDevice));
}

__host__
size_t lyra2z330_matrix_bytes()
{
	return (size_t)memshift * Ncol * Nrow * 4 * sizeof(uint2);
}

__host__
uint32_t lyra2z330_hashes_per_block()
{
	return TPB / 4;
}

__host__
void lyra2z330_cpu_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_matrix,
	uint64_t *d_hash, uint32_t *d_resNonces, uint32_t target)
{
	const uint32_t tpb = TPB;
	dim3 grid((threads * 4 + tpb - 1) / tpb);
	dim3 block(4, tpb >> 2);

	lyra2z330_gpu_hash <<< grid, block >>> (threads, startNonce, (uint2*)d_matrix,
		(uint2*)d_hash, d_resNonces, target);
}
