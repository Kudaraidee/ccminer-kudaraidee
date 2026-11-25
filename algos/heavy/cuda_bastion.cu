#include <stdio.h>

#include "cuda/helper.cuh"

// Branch compaction: both branches share one buffer of `threads` records, branch 1
// filling it from the start and branch 2 from the end; d_slot remembers each record's slot.
static uint32_t *d_slot[MAX_GPUS] = { 0 };

static uint32_t *d_brcount1[MAX_GPUS] = { 0 };
static uint32_t *d_brcount2[MAX_GPUS] = { 0 };

// 4 lanes copy each 64-byte record, so a warp instruction moves 8 whole records
// (one record per lane wasted half of every fetched sector).
#define BASTION_REC_LANES 4

__global__ __launch_bounds__(128, 6)
void bastion_filter2_gpu(const uint32_t threads, const uint32_t* d_hash, uint32_t* d_branch, uint32_t* d_slot, uint32_t* d_count1, uint32_t* d_count2)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const bool valid = thread < threads;
	uint32_t slot = 0;
	if (valid)
	{
		if (((const uint8_t*) &d_hash[thread * 16U])[0] & 0x8)
			slot = atomicAdd(d_count1, 1U);
		else
			slot = threads - 1U - atomicAdd(d_count2, 1U);
		d_slot[thread] = slot;
	}

	// the whole warp copies its 32 records; the grid is a whole number of warps
	const uint32_t lane = threadIdx.x & 31U;
	const uint32_t warp0 = thread - lane;
	const uint32_t part = lane % BASTION_REC_LANES;
	#pragma unroll
	for (uint32_t r = lane / BASTION_REC_LANES; r < 32U; r += 32U / BASTION_REC_LANES) {
		const uint32_t s = __shfl_sync(0xffffffffu, slot, r);
		if (warp0 + r < threads)
			((uint4*) &d_branch[(size_t) s * 16U])[part] = ((const uint4*) &d_hash[(size_t) (warp0 + r) * 16U])[part];
	}
}

__global__ __launch_bounds__(128, 6)
void bastion_merge2_gpu(const uint32_t threads, uint32_t* d_hash, const uint32_t* d_branch, const uint32_t* d_slot)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t slot = thread < threads ? d_slot[thread] : 0;

	const uint32_t lane = threadIdx.x & 31U;
	const uint32_t warp0 = thread - lane;
	const uint32_t part = lane % BASTION_REC_LANES;
	#pragma unroll
	for (uint32_t r = lane / BASTION_REC_LANES; r < 32U; r += 32U / BASTION_REC_LANES) {
		const uint32_t s = __shfl_sync(0xffffffffu, slot, r);
		if (warp0 + r < threads)
			((uint4*) &d_hash[(size_t) (warp0 + r) * 16U])[part] = ((const uint4*) &d_branch[(size_t) s * 16U])[part];
	}
}


__host__
void bastion_cpu_init(const int thr_id, const uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_slot[thr_id], sizeof(uint32_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&d_brcount1[thr_id], sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc(&d_brcount2[thr_id], sizeof(uint32_t)));
}

__host__
void bastion_free(const int thr_id)
{
	cudaFree(d_slot[thr_id]);
	cudaFree(d_brcount1[thr_id]);
	cudaFree(d_brcount2[thr_id]);
}

// Splits the batch into d_branch: [0, n) branch 1, [n, threads) branch 2; returns n.
__host__
uint32_t bastion_filter2(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_branch)
{
	uint32_t num = 0;
	cudaMemset(d_brcount1[thr_id], 0, 4);
	cudaMemset(d_brcount2[thr_id], 0, 4);
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	bastion_filter2_gpu <<<grid, block>>> (threads, inpHashes, d_branch, d_slot[thr_id], d_brcount1[thr_id], d_brcount2[thr_id]);
	cudaMemcpy(&num, d_brcount1[thr_id], 4, cudaMemcpyDeviceToHost);
	return num;
}

__host__
void bastion_merge2(const int thr_id, const uint32_t threads, uint32_t *outpHashes, const uint32_t* d_branch)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// put back branch hashes to the common buffer d_hash
	bastion_merge2_gpu <<<grid, block>>> (threads, outpHashes, d_branch, d_slot[thr_id]);
}
