/**
 * Lyra2 (v1) cuda implementation based on djm34 work
 * tpruvot@github 2015, Nanashi 08/2016 (from 1.8-r2)
 * Lyra2Z implentation for Zcoin based on all the previous
 * djm34 2017
 **/

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#define TPB52 32



static uint32_t *h_GNonces[16]; // this need to get fixed as the rest of that routine
static uint32_t *d_GNonces[16];



#include "cuda_lyra2_vectors.h"
#include "cuda/blake2b_device.cuh"
//#include "cuda/vector_uint2x4.cuh"

#define Nrow 8
#define Ncol 8
#define memshift 3

#define BUF_COUNT 0

__constant__ uint32_t pTarget[8];

__device__ uint2 *DMatrix;

#define LYRA2_TPB TPB52   // launch shape shared with the header
#include "cuda/lyra2_device.cuh"



static __device__ __forceinline__
void reduceDuplexRowt_8_v2(const int rowIn, const int rowOut, const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3], last[3];

	LD4S(state1, rowIn, 0, thread, threads);
	LD4S(last, rowInOut, 0, thread, threads);

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= state1[j] + last[j];

	round_lyra(state);

	// simultaneously receive data from preceding thread and send data to following thread
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == rowOut) {
		#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < Nrow; i++)
	{
		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__global__
__launch_bounds__(64, 1)
void lyra2Z_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint2x4 Mask[2] = {
		0x00000020UL, 0x00000000UL, 0x00000020UL, 0x00000000UL,
		0x00000020UL, 0x00000000UL, 0x00000008UL, 0x00000000UL,
		0x00000008UL, 0x00000000UL, 0x00000008UL, 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};
	const uint2x4 blake2b_IV[2] = {
		0xf3bcc908lu, 0x6a09e667lu,
		0x84caa73blu, 0xbb67ae85lu,
		0xfe94f82blu, 0x3c6ef372lu,
		0x5f1d36f1lu, 0xa54ff53alu,
		0xade682d1lu, 0x510e527flu,
		0x2b3e6c1flu, 0x9b05688clu,
		0xfb41bd6blu, 0x1f83d9ablu,
		0x137e2179lu, 0x5be0cd19lu
	};
	if (thread < threads)
	{
		uint2x4 state[4];

		state[0].x = state[1].x = __ldg(&g_hash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&g_hash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&g_hash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&g_hash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<12; i++)
			round_lyra(state);

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

		for (int i = 0; i<12; i++)
			round_lyra(state); //because 12 is not enough

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__
__launch_bounds__(TPB52, 1)
void lyra2Z_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	const uint32_t thread = LYRA2_IDX_Y * blockIdx.x + threadIdx.y;

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = __ldg(&DMatrix[(0 * threads + thread) * LYRA2_IDX_X + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + thread) * LYRA2_IDX_X + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + thread) * LYRA2_IDX_X + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + thread) * LYRA2_IDX_X + threadIdx.x]);

		reduceDuplex(state, thread, threads);
		reduceDuplexRowSetup(1, 0, 2, state, thread, threads);
		reduceDuplexRowSetup(2, 1, 3, state, thread, threads);
		reduceDuplexRowSetup(3, 0, 4, state, thread, threads);
		reduceDuplexRowSetup(4, 3, 5, state, thread, threads);
		reduceDuplexRowSetup(5, 2, 6, state, thread, threads);
		reduceDuplexRowSetup(6, 1, 7, state, thread, threads);

		uint32_t rowa; // = WarpShuffle(state[0].x, 0, 4) & 7;
		uint32_t prev = 7;
		uint32_t iterator = 0;

	//for (uint32_t j=0;j<4;j++) {

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}

		for (uint32_t i = 0; i<7; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

	//}
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt_8_v2(prev,iterator,rowa, state, thread, threads);

		DMatrix[(0 * threads + thread) * LYRA2_IDX_X + threadIdx.x] = state[0];
		DMatrix[(1 * threads + thread) * LYRA2_IDX_X + threadIdx.x] = state[1];
		DMatrix[(2 * threads + thread) * LYRA2_IDX_X + threadIdx.x] = state[2];
		DMatrix[(3 * threads + thread) * LYRA2_IDX_X + threadIdx.x] = state[3];
	}
}

__global__
__launch_bounds__(64, 1)
void lyra2Z_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *resNonces)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		uint32_t nonce = startNounce + thread;
		if (((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3]) {
			// Keep the two lowest candidates in one pass. atomicMin returns the previous
			// minimum, so the displaced value moves to slot 1; reading resNonces[0]
			// outside the atomic loses one.
			const uint32_t prev = atomicMin(&resNonces[0], nonce);
			if (prev != UINT32_MAX)
				atomicMin(&resNonces[1], max(prev, nonce));
		}
/*
		g_hash[thread + threads * 0] = state[0].x;
		g_hash[thread + threads * 1] = state[0].y;
		g_hash[thread + threads * 2] = state[0].z;
		g_hash[thread + threads * 3] = state[0].w;
*/
	}
}

__host__
void lyra2Z_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&d_GNonces[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2 * sizeof(uint32_t));
}

__host__
void lyra2Z_cpu_free(int thr_id)
{
	cudaFree(d_GNonces[thr_id]);
	cudaFreeHost(h_GNonces[thr_id]);
}

__host__
uint32_t lyra2Z_getSecNonce(int thr_id, int num)
{
	uint32_t results[2];
	memset(results, 0xFF, sizeof(results));
	cudaMemcpy(results, d_GNonces[thr_id], sizeof(results), cudaMemcpyDeviceToHost);
	if (results[1] == results[0])
		return UINT32_MAX;
	return results[num];
}

__host__
void lyra2Z_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32, 0, cudaMemcpyHostToDevice);
}

__host__
uint32_t lyra2Z_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash)
{
	uint32_t result = UINT32_MAX;
	cudaMemset(d_GNonces[thr_id], 0xff, 2 * sizeof(uint32_t));

	const uint32_t tpb = TPB52;

	// the wander matrix lives in dynamic shared memory: memshift * Ncol per row
	// kept in shared, for every thread of the block
	const size_t shared_mem = memshift * Ncol * (Nrow - BUF_COUNT) * sizeof(uint2) * tpb;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(LYRA2_BDIMX, LYRA2_BDIMY);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	lyra2Z_gpu_hash_32_1 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);

	lyra2Z_gpu_hash_32_2 <<< grid1, block1, shared_mem >>> (threads, startNounce, d_hash);

	lyra2Z_gpu_hash_32_3 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash, d_GNonces[thr_id]);

	// get first found nonce
	cudaMemcpy(h_GNonces[thr_id], d_GNonces[thr_id], 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	result = *h_GNonces[thr_id];

	return result;
}
