#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#define TPB52 32

#include "cuda_lyra2_vectors.h"
#include "cuda/blake2b_device.cuh"

#define Nrow 4
#define Ncol 4
#define memshift 3

__device__ uint2x4 *DState;

/* The 4 x 4 wander matrix: the first V2_SCOL columns in dynamic shared memory, the
 * last V2_RCOL in registers (half the shared memory, more blocks per SM). Register
 * cells take only a compile-time column; the wander row is a select. Shared columns
 * stay in rolled loops: unrolled, the kernel is much slower on sm_61. */
#define V2_RCOL 2
#define V2_SCOL (Ncol - V2_RCOL)

typedef uint2 v2_regs_t[Nrow][V2_RCOL][3];

__device__ __forceinline__ void v2_lds(uint2 d[3], const int row, const int col)
{
	extern __shared__ uint2 shared_mem[];
	const int s = (row * V2_SCOL + col) * memshift;
	#pragma unroll
	for (int j = 0; j < 3; j++)
		d[j] = shared_mem[((s + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void v2_sts(const int row, const int col, const uint2 d[3])
{
	extern __shared__ uint2 shared_mem[];
	const int s = (row * V2_SCOL + col) * memshift;
	#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = d[j];
}

__device__ __forceinline__ void v2_rld(uint2 d[3], v2_regs_t &R, const int row, const int c)
{
	const bool b0 = row & 1, b1 = row & 2;
	#pragma unroll
	for (int j = 0; j < 3; j++) {
		uint2 a, b;
		a.x = b0 ? R[1][c][j].x : R[0][c][j].x;  a.y = b0 ? R[1][c][j].y : R[0][c][j].y;
		b.x = b0 ? R[3][c][j].x : R[2][c][j].x;  b.y = b0 ? R[3][c][j].y : R[2][c][j].y;
		d[j].x = b1 ? b.x : a.x;                 d[j].y = b1 ? b.y : a.y;
	}
}

__device__ __forceinline__ void v2_rst(v2_regs_t &R, const int row, const int c, const uint2 d[3])
{
	#pragma unroll
	for (int r = 0; r < Nrow; r++)
		#pragma unroll
		for (int j = 0; j < 3; j++) {
			R[r][c][j].x = (row == r) ? d[j].x : R[r][c][j].x;
			R[r][c][j].y = (row == r) ? d[j].y : R[r][c][j].y;
		}
}

// col must be a compile-time constant at every call
__device__ __forceinline__ void v2_ld(uint2 d[3], v2_regs_t &R, const int row, const int col)
{
	if (col >= V2_SCOL) v2_rld(d, R, row, col - V2_SCOL); else v2_lds(d, row, col);
}

__device__ __forceinline__ void v2_st(v2_regs_t &R, const int row, const int col, const uint2 d[3])
{
	if (col >= V2_SCOL) v2_rst(R, row, col - V2_SCOL, d); else v2_sts(row, col, d);
}

// rotW of the sponge output into the rowInOut cell (lane 0 shifts by one word)
__device__ __forceinline__ void v2_rotw_xor(uint2 io[3], const uint2 state[4])
{
	uint2 D0 = state[0], D1 = state[1], D2 = state[2];
	WarpShuffle3(D0, D1, D2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);
	if (threadIdx.x == 0) {
		io[0] ^= D2; io[1] ^= D0; io[2] ^= D1;
	} else {
		io[0] ^= D0; io[1] ^= D1; io[2] ^= D2;
	}
}

// rows 0 and 1 built in registers; rows 2 and 3 written back to front while 0 and 1 are rotW-updated
__device__ __forceinline__ void v2_setup(uint2 state[4], v2_regs_t &R)
{
	uint2 r0[Ncol][3], r1[Ncol][3], t[3];

	#pragma unroll
	for (int i = 0; i < Ncol; i++) {
		for (int j = 0; j < 3; j++) r0[Ncol - i - 1][j] = state[j];
		round_lyra(state);
	}
	#pragma unroll
	for (int i = 0; i < Ncol; i++) {
		for (int j = 0; j < 3; j++) state[j] ^= r0[i][j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) r1[Ncol - i - 1][j] = r0[i][j] ^ state[j];
	}
	#pragma unroll
	for (int i = 0; i < Ncol; i++) {
		for (int j = 0; j < 3; j++) state[j] ^= r1[i][j] + r0[i][j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) t[j] = r1[i][j] ^ state[j];
		v2_st(R, 2, Ncol - i - 1, t);
		v2_rotw_xor(r0[i], state);
		v2_st(R, 0, i, r0[i]);
		for (int j = 0; j < 3; j++) r0[i][j] = t[j];   // now row 2, column Ncol-1-i
	}
	#pragma unroll
	for (int i = 0; i < Ncol; i++) {
		for (int j = 0; j < 3; j++) state[j] ^= r1[i][j] + r0[Ncol - i - 1][j];
		round_lyra(state);
		for (int j = 0; j < 3; j++) r0[Ncol - i - 1][j] ^= state[j];
		v2_st(R, 3, Ncol - i - 1, r0[Ncol - i - 1]);
		v2_rotw_xor(r1[i], state);
		v2_st(R, 1, i, r1[i]);
	}
}

// one column of a wandering step; rowInOut may equal rowIn or rowOut
__device__ __forceinline__ void v2_wander_col(const int rowIn, const int rowInOut, const int rowOut,
	const int i, uint2 state[4], v2_regs_t &R, const bool in_shared)
{
	uint2 a[3], b[3], c[3];
	if (in_shared) { v2_lds(b, rowInOut, i); v2_lds(a, rowIn, i); }
	else           { v2_ld(b, R, rowInOut, i); v2_ld(a, R, rowIn, i); }
	for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
	round_lyra(state);
	v2_rotw_xor(b, state);
	if (in_shared) v2_sts(rowInOut, i, b); else v2_st(R, rowInOut, i, b);
	if (in_shared) v2_lds(c, rowOut, i);   else v2_ld(c, R, rowOut, i);   // after the rowInOut store
	for (int j = 0; j < 3; j++) c[j] ^= state[j];
	if (in_shared) v2_sts(rowOut, i, c);   else v2_st(R, rowOut, i, c);
}

__device__ __forceinline__ void v2_wander(const int rowIn, const int rowOut, uint2 state[4], v2_regs_t &R)
{
	const uint32_t rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;
	#pragma unroll 1
	for (int i = 0; i < V2_SCOL; i++)
		v2_wander_col(rowIn, rowInOut, rowOut, i, state, R, true);
	#pragma unroll
	for (int i = V2_SCOL; i < Ncol; i++)
		v2_wander_col(rowIn, rowInOut, rowOut, i, state, R, false);
}

// last wandering step: reads row 2 and rowInOut, stores nothing
__device__ __forceinline__ void v2_wander_last(uint2 state[4], v2_regs_t &R)
{
	const uint32_t rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;
	uint2 a[3], b[3], last[3];
	v2_lds(last, rowInOut, 0);
	v2_lds(a, 2, 0);
	for (int j = 0; j < 3; j++) state[j] ^= a[j] + last[j];
	round_lyra(state);
	v2_rotw_xor(last, state);
	if (rowInOut == 3)
		for (int j = 0; j < 3; j++) last[j] ^= state[j];

	#pragma unroll 1
	for (int i = 1; i < V2_SCOL; i++) {
		v2_lds(a, 2, i);
		v2_lds(b, rowInOut, i);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
	}
	#pragma unroll
	for (int i = V2_SCOL; i < Ncol; i++) {
		v2_ld(a, R, 2, i);
		v2_ld(b, R, rowInOut, i);
		for (int j = 0; j < 3; j++) state[j] ^= a[j] + b[j];
		round_lyra(state);
	}

	for (int j = 0; j < 3; j++) state[j] ^= last[j];
}


__constant__ uint28 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
};

__constant__ uint28 Mask[2] = {
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000001lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000080lu, 0x00000000lu,
	0x00000000lu, 0x01000000lu
};

// The 32-byte hash buffer has two layouts and the consumers disagree: SoA
// (thread + k*threads, 32 B/thread) for -a lyra2v2, AoS (thread*8 + k, 64 B/thread) for
// the x-family. Templated so each gets its own instantiation; one hardcoded layout
// silently breaks the others.
template <bool SOA>
__device__ __forceinline__ uint32_t h32_idx(const uint32_t thread, const uint32_t threads, const int k)
{
	return SOA ? thread + threads * k : thread * 8 + k;
}

template <bool SOA>
__global__ __launch_bounds__(64, 1)
void lyra2v2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0].x = state[1].x = __ldg(&outputHash[h32_idx<SOA>(thread, threads, 0)]);
		state[0].y = state[1].y = __ldg(&outputHash[h32_idx<SOA>(thread, threads, 1)]);
		state[0].z = state[1].z = __ldg(&outputHash[h32_idx<SOA>(thread, threads, 2)]);
		state[0].w = state[1].w = __ldg(&outputHash[h32_idx<SOA>(thread, threads, 3)]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

#pragma unroll 2
		for (int i = 0; i<12; i++)
			round_lyra(state);

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

#pragma unroll 2
		for (int i = 0; i<12; i++)
			round_lyra(state);

		DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x] = state[0];
		DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x] = state[1];
		DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x] = state[2];
		DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x] = state[3];

	} //thread
}

__global__ __launch_bounds__(TPB52, 1)
void lyra2v2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	/* every lane runs the full-mask shuffles; a lane past the batch works on a
	 * copy of the last hash and stores nothing */
	const bool active = thread < threads;
	const uint32_t tl = active ? thread : threads - 1;
	{
		uint2 state[4];
		v2_regs_t R;
		// the init/final kernels' 64-thread grid sets the DState stride
		const uint32_t stride = (threads + 63) & ~63u;
		state[0] = ((uint2*)DState)[(0 * stride + tl) * blockDim.x + threadIdx.x];
		state[1] = ((uint2*)DState)[(1 * stride + tl) * blockDim.x + threadIdx.x];
		state[2] = ((uint2*)DState)[(2 * stride + tl) * blockDim.x + threadIdx.x];
		state[3] = ((uint2*)DState)[(3 * stride + tl) * blockDim.x + threadIdx.x];

		v2_setup(state, R);
		v2_wander(3, 0, state, R);
		v2_wander(0, 1, state, R);
		v2_wander(1, 2, state, R);
		v2_wander_last(state, R);

		if (active) {
			((uint2*)DState)[(0 * stride + thread) * blockDim.x + threadIdx.x] = state[0];
			((uint2*)DState)[(1 * stride + thread) * blockDim.x + threadIdx.x] = state[1];
			((uint2*)DState)[(2 * stride + thread) * blockDim.x + threadIdx.x] = state[2];
			((uint2*)DState)[(3 * stride + thread) * blockDim.x + threadIdx.x] = state[3];
		}
	}
}

template <bool SOA>
__global__ __launch_bounds__(64, 1)
void lyra2v2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[1] = __ldg4(&DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[2] = __ldg4(&DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[3] = __ldg4(&DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x]);

#pragma unroll 2
		for (int i = 0; i < 12; i++)
			round_lyra(state);

		outputHash[h32_idx<SOA>(thread, threads, 0)] = state[0].x;
		outputHash[h32_idx<SOA>(thread, threads, 1)] = state[0].y;
		outputHash[h32_idx<SOA>(thread, threads, 2)] = state[0].z;
		outputHash[h32_idx<SOA>(thread, threads, 3)] = state[0].w;

	} //thread
}

__host__
void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DState, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	// the extra blocks per SM need the largest shared carveout (Ampere)
	cudaFuncSetAttribute(lyra2v2_gpu_hash_32_2, cudaFuncAttributePreferredSharedMemoryCarveout,
		cudaSharedmemCarveoutMaxShared);
}

template <bool SOA>
__host__ static void lyra2v2_cpu_hash_32_impl(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	const uint32_t tpb = TPB52;

	// the shared-memory columns of the matrix, for every thread of the block
	const size_t shared_mem = memshift * Nrow * V2_SCOL * sizeof(uint2) * tpb;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(4, tpb >> 2);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	lyra2v2_gpu_hash_32_1<SOA> << <grid2, block2 >> > (threads, startNounce, (uint2*)g_hash);

	lyra2v2_gpu_hash_32_2 << <grid1, block1, shared_mem >> > (threads, startNounce, g_hash);

	lyra2v2_gpu_hash_32_3<SOA> << <grid2, block2 >> > (threads, startNounce, (uint2*)g_hash);
	//MyStreamSynchronize(NULL, order, thr_id);
}

// AoS, 64 B/thread -- the x-family layout (x21s, x25x).
__host__
void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, int order)
{
	lyra2v2_cpu_hash_32_impl<false>(threads, startNounce, g_hash);
}

// SoA, 32 B/thread -- -a lyra2v2. The AoS entry point above would overrun its d_hash.
__host__
void lyra2v2_cpu_hash_32_soa(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, int order)
{
	lyra2v2_cpu_hash_32_impl<true>(threads, startNounce, g_hash);
}
