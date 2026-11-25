/**
 * Lyra2 (v1) cuda implementation based on djm34 work
 * tpruvot@github 2015, Nanashi 08/2016 (from 1.8-r2)
 * tpruvot@github 2018 for phi2 double lyra2-32 support
 */

#include <stdio.h>
#include <memory.h>

#define TPB52 32

#include "cuda_lyra2_vectors.h"
#include "cuda/blake2b_device.cuh"

#define Nrow 8
#define Ncol 8
#define memshift 3

__device__ uint2 *DMatrix;

/* The wander matrix: the first L2_SCOL columns in dynamic shared memory, the last
 * L2_RCOL in registers (half the shared memory, twice the hashes per SM). Register
 * cells take only a compile-time column; a data-dependent row is a select. */
#define L2_RCOL 4
#define L2_SCOL (Ncol - L2_RCOL)
#define L2_BDIMX 4                     // lanes per hash
#define L2_BDIMY (TPB52 / L2_BDIMX)    // hashes per block

typedef uint2 lyra2_regs_t[Nrow][L2_RCOL][3];

__device__ __forceinline__ void l2_lds(uint2 d[3], const int row, const int col)
{
	extern __shared__ uint2 shared_mem[];
	const int s0 = (L2_SCOL * row + col) * memshift;
	#pragma unroll
	for (int j = 0; j < 3; j++)
		d[j] = shared_mem[((s0 + j) * L2_BDIMY + threadIdx.y) * L2_BDIMX + threadIdx.x];
}

__device__ __forceinline__ void l2_sts(const int row, const int col, const uint2 d[3])
{
	extern __shared__ uint2 shared_mem[];
	const int s0 = (L2_SCOL * row + col) * memshift;
	#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s0 + j) * L2_BDIMY + threadIdx.y) * L2_BDIMX + threadIdx.x] = d[j];
}

// depth-3 tree, not a chain: the read sits on the sponge's critical path
__device__ __forceinline__ uint32_t l2_sel8(const bool b0, const bool b1, const bool b2,
	uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4, uint32_t v5, uint32_t v6, uint32_t v7)
{
	const uint32_t a = b0 ? v1 : v0, b = b0 ? v3 : v2, c = b0 ? v5 : v4, d = b0 ? v7 : v6;
	const uint32_t e = b1 ? b : a, f = b1 ? d : c;
	return b2 ? f : e;
}

__device__ __forceinline__ void l2_rld(uint2 d[3], lyra2_regs_t &R, const int row, const int c)
{
	const bool b0 = row & 1, b1 = row & 2, b2 = row & 4;
	#pragma unroll
	for (int j = 0; j < 3; j++) {
		d[j].x = l2_sel8(b0, b1, b2, R[0][c][j].x, R[1][c][j].x, R[2][c][j].x, R[3][c][j].x,
		                             R[4][c][j].x, R[5][c][j].x, R[6][c][j].x, R[7][c][j].x);
		d[j].y = l2_sel8(b0, b1, b2, R[0][c][j].y, R[1][c][j].y, R[2][c][j].y, R[3][c][j].y,
		                             R[4][c][j].y, R[5][c][j].y, R[6][c][j].y, R[7][c][j].y);
	}
}

__device__ __forceinline__ void l2_rst(lyra2_regs_t &R, const int row, const int c, const uint2 d[3])
{
	#pragma unroll
	for (int r = 0; r < Nrow; r++) {
		#pragma unroll
		for (int j = 0; j < 3; j++) {
			R[r][c][j].x = (row == r) ? d[j].x : R[r][c][j].x;
			R[r][c][j].y = (row == r) ? d[j].y : R[r][c][j].y;
		}
	}
}

// rotW of the sponge output into the rowInOut cell (lane 0 shifts by one word)
__device__ __forceinline__ void l2_rotw_xor(uint2 io[3], const uint2 state[4])
{
	uint2 D0 = state[0], D1 = state[1], D2 = state[2];
	WarpShuffle3(D0, D1, D2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
	if (threadIdx.x == 0) {
		io[0] ^= D2; io[1] ^= D0; io[2] ^= D1;
	} else {
		io[0] ^= D0; io[1] ^= D1; io[2] ^= D2;
	}
}

/* Setup writes rowOut back to front (column Ncol-1-i) while rowIn/rowInOut go
 * front to back, so each setup loop splits where either side turns resident. */

// rows 0 and 1
static __device__ __forceinline__ void l2_duplex(uint2 state[4], lyra2_regs_t &R)
{
	#pragma unroll
	for (int i = 0; i < Ncol; i++) {
		const int col = Ncol - i - 1;
		if (col >= L2_SCOL) l2_rst(R, 0, col - L2_SCOL, state);
		else                l2_sts(0, col, state);
		round_lyra(state);
	}

	uint2 a[3];
	#pragma unroll
	for (int i = 0; i < L2_RCOL; i++) {
		l2_lds(a, 0, i);
		for (int j = 0; j < 3; j++) state[j] ^= a[j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) a[j] ^= state[j];
		l2_rst(R, 1, L2_RCOL - 1 - i, a);
	}
	#pragma unroll 4
	for (int i = L2_RCOL; i < L2_SCOL; i++) {
		l2_lds(a, 0, i);
		for (int j = 0; j < 3; j++) state[j] ^= a[j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) a[j] ^= state[j];
		l2_sts(1, Ncol - i - 1, a);
	}
	#pragma unroll
	for (int i = L2_SCOL; i < Ncol; i++) {
		l2_rld(a, R, 0, i - L2_SCOL);
		for (int j = 0; j < 3; j++) state[j] ^= a[j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) a[j] ^= state[j];
		l2_sts(1, Ncol - i - 1, a);
	}
}

// rows 2..Nrow-1; rowOut is never rowIn or rowInOut here
static __device__ __forceinline__
void l2_setup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], lyra2_regs_t &R)
{
	uint2 a[3], b[3];

	#pragma unroll
	for (int i = 0; i < L2_RCOL; i++) {
		l2_lds(a, rowIn, i);
		l2_lds(b, rowInOut, i);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) a[j] ^= state[j];
		l2_rst(R, rowOut, L2_RCOL - 1 - i, a);
		l2_rotw_xor(b, state);
		l2_sts(rowInOut, i, b);
	}
	#pragma unroll 1
	for (int i = L2_RCOL; i < L2_SCOL; i++) {
		l2_lds(a, rowIn, i);
		l2_lds(b, rowInOut, i);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) a[j] ^= state[j];
		l2_sts(rowOut, Ncol - i - 1, a);
		l2_rotw_xor(b, state);
		l2_sts(rowInOut, i, b);
	}
	#pragma unroll
	for (int i = L2_SCOL; i < Ncol; i++) {
		l2_rld(a, R, rowIn, i - L2_SCOL);
		l2_rld(b, R, rowInOut, i - L2_SCOL);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) a[j] ^= state[j];
		l2_sts(rowOut, Ncol - i - 1, a);
		l2_rotw_xor(b, state);
		l2_rst(R, rowInOut, i - L2_SCOL, b);
	}
}

// wandering step; rowInOut is data-dependent and may equal rowIn or rowOut
static __device__ __forceinline__
void l2_wander(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], lyra2_regs_t &R)
{
	uint2 a[3], b[3], c[3];

	for (int i = 0; i < L2_SCOL; i++) {
		l2_lds(a, rowIn, i);
		l2_lds(b, rowInOut, i);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
		l2_rotw_xor(b, state);
		l2_sts(rowInOut, i, b);
		l2_lds(c, rowOut, i);                    // after the rowInOut store
		for (int j = 0; j < 3; j++) c[j] ^= state[j];
		l2_sts(rowOut, i, c);
	}
	#pragma unroll
	for (int i = L2_SCOL; i < Ncol; i++) {
		l2_rld(a, R, rowIn, i - L2_SCOL);
		l2_rld(b, R, rowInOut, i - L2_SCOL);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
		l2_rotw_xor(b, state);
		l2_rst(R, rowInOut, i - L2_SCOL, b);
		l2_rld(c, R, rowOut, i - L2_SCOL);       // after the rowInOut store
		for (int j = 0; j < 3; j++) c[j] ^= state[j];
		l2_rst(R, rowOut, i - L2_SCOL, c);
	}
}

// last wandering step: reads row 2 and rowInOut only
static __device__ __forceinline__
void l2_wander_last(const int rowInOut, uint2 state[4], lyra2_regs_t &R)
{
	uint2 a[3], b[3], last[3];

	l2_lds(a, 2, 0);
	l2_lds(last, rowInOut, 0);
	for (int j = 0; j < 3; j++) state[j] ^= a[j] + last[j];
	round_lyra(state);
	l2_rotw_xor(last, state);
	if (rowInOut == 5) {
		for (int j = 0; j < 3; j++) last[j] ^= state[j];
	}

	for (int i = 1; i < L2_SCOL; i++) {
		l2_lds(a, 2, i);
		l2_lds(b, rowInOut, i);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
	}
	#pragma unroll
	for (int i = L2_SCOL; i < Ncol; i++) {
		l2_rld(a, R, 2, i - L2_SCOL);
		l2_rld(b, R, rowInOut, i - L2_SCOL);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
	}

	for (int j = 0; j < 3; j++) state[j] ^= last[j];
}


__constant__ uint2x4 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
};

__global__ __launch_bounds__(64, 1)
void lyra2_gpu_hash_32_1(uint32_t threads, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2x4 state[4];
		state[0].x = state[1].x = __ldg(&g_hash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&g_hash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&g_hash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&g_hash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<24; i++)
			round_lyra(state); //because 12 is not enough

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__
__launch_bounds__(TPB52, 1)
void lyra2_gpu_hash_32_2(const uint32_t threads, uint64_t *g_hash)
{
	const uint32_t thread = L2_BDIMY * blockIdx.x + threadIdx.y;
	/* every lane runs the full-mask shuffles; a lane past the batch works on a
	 * copy of the last hash and stores nothing */
	const bool active = thread < threads;
	const uint32_t tl = active ? thread : threads - 1;
	{
		uint2 state[4];
		lyra2_regs_t R;
		state[0] = __ldg(&DMatrix[(0 * threads + tl) * L2_BDIMX + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + tl) * L2_BDIMX + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + tl) * L2_BDIMX + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + tl) * L2_BDIMX + threadIdx.x]);

		l2_duplex(state, R);

		// one body for all steps (rowIn, rowInOut, rowOut): (1,0,2) (2,1,3) (3,0,4) (4,3,5) (5,2,6) (6,1,7)
		#pragma unroll 1
		for (int s = 0; s < 6; s++)
			l2_setup(s + 1, (0x123010u >> (4 * s)) & 0xF, s + 2, state, R);

		// rowOut 0 3 6 1 4 7 2 = 3s mod 8; rowIn is 7, then the previous rowOut
		int rowIn = 7;
		#pragma unroll 1
		for (int s = 0; s < 7; s++) {
			const int rowOut = (3 * s) & 7;
			const uint32_t rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			l2_wander(rowIn, rowa, rowOut, state, R);
			rowIn = rowOut;
		}
		const uint32_t rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		l2_wander_last(rowa, state, R);

		if (active) {
			DMatrix[(0 * threads + thread) * L2_BDIMX + threadIdx.x] = state[0];
			DMatrix[(1 * threads + thread) * L2_BDIMX + threadIdx.x] = state[1];
			DMatrix[(2 * threads + thread) * L2_BDIMX + threadIdx.x] = state[2];
			DMatrix[(3 * threads + thread) * L2_BDIMX + threadIdx.x] = state[3];
		}
	}
}

__global__ __launch_bounds__(64, 1)
void lyra2_gpu_hash_32_3(uint32_t threads, uint2 *g_hash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{
		uint2x4 state[4];
		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		g_hash[thread + threads * 0] = state[0].x;
		g_hash[thread + threads * 1] = state[0].y;
		g_hash[thread + threads * 2] = state[0].z;
		g_hash[thread + threads * 3] = state[0].w;
	}
}

__global__ __launch_bounds__(64, 1)
void lyra2_gpu_hash_64_1(uint32_t threads, uint2* const d_hash_512, const uint32_t round)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2x4 state[4];
		const size_t offset = (size_t)8 * thread + (round * 4U);
		uint2 *psrc = (uint2*)(&d_hash_512[offset]);
		state[0].x = state[1].x = __ldg(&psrc[0]);
		state[0].y = state[1].y = __ldg(&psrc[1]);
		state[0].z = state[1].z = __ldg(&psrc[2]);
		state[0].w = state[1].w = __ldg(&psrc[3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<24; i++)
			round_lyra(state);

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__ __launch_bounds__(64, 1)
void lyra2_gpu_hash_64_3(uint32_t threads, uint2 *d_hash_512, const uint32_t round)
{
	// This kernel outputs 2x 256-bits hashes in 512-bits chain offsets in 2 rounds
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{
		uint2x4 state[4];
		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		const size_t offset = (size_t)8 * thread + (round * 4U);
		uint2 *pdst = (uint2*)(&d_hash_512[offset]);
		pdst[0] = state[0].x;
		pdst[1] = state[0].y;
		pdst[2] = state[0].z;
		pdst[3] = state[0].w;
	}
}
__host__
void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	// four blocks per SM need the largest shared carveout (Ampere)
	cudaFuncSetAttribute(lyra2_gpu_hash_32_2, cudaFuncAttributePreferredSharedMemoryCarveout,
		cudaSharedmemCarveoutMaxShared);
}

__host__
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint64_t *d_hash)
{
	const uint32_t tpb = TPB52;

	// the shared-memory columns of the wander matrix, for every thread of the block
	const size_t shared_mem = memshift * L2_SCOL * Nrow * sizeof(uint2) * tpb;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(L2_BDIMX, L2_BDIMY);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	lyra2_gpu_hash_32_1 <<< grid2, block2 >>> (threads, (uint2*)d_hash);
	lyra2_gpu_hash_32_2 <<< grid1, block1, shared_mem >>> (threads, d_hash);
	lyra2_gpu_hash_32_3 <<< grid2, block2 >>> (threads, (uint2*)d_hash);
}

__host__
void lyra2_cuda_hash_64(int thr_id, const uint32_t threads, uint64_t* d_hash_256, uint32_t* d_hash_512)
{
	const uint32_t tpb = TPB52;
	const size_t shared_mem = memshift * L2_SCOL * Nrow * sizeof(uint2) * tpb; // 24576

	dim3 grid1((size_t(threads) * 4 + tpb - 1) / tpb);
	dim3 block1(L2_BDIMX, L2_BDIMY);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	// two 256-bit lyra2 passes over the 512-bit chain offsets (phi2)
	lyra2_gpu_hash_64_1 <<< grid2, block2 >>> (threads, (uint2*)d_hash_512, 0);
	lyra2_gpu_hash_32_2 <<< grid1, block1, shared_mem >>> (threads, d_hash_256);
	lyra2_gpu_hash_64_3 <<< grid2, block2 >>> (threads, (uint2*)d_hash_512, 0);

	lyra2_gpu_hash_64_1 <<< grid2, block2 >>> (threads, (uint2*)d_hash_512, 1);
	lyra2_gpu_hash_32_2 <<< grid1, block1, shared_mem >>> (threads, d_hash_256);
	lyra2_gpu_hash_64_3 <<< grid2, block2 >>> (threads, (uint2*)d_hash_512, 1);
}
