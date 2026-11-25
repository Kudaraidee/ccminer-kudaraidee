/*
 * fugue512 x13 kernel implementation — thin wrapper.
 *
 * The Fugue-512 device implementation (mixtab constants, TIX4/CMIX36/SMIX
 * macros, shared-table fill and fugue512_hash_64) lives in
 * cuda/fugue512_device.cuh (docs/coding-guideline.md §3). The donor's
 * texture apparatus (mixTab0Tex/d_textures) is gone: the shared fill reads
 * the same table from constant memory.
 */

#include "cuda/helper.cuh"

/* 384 x 2 blocks at 80 registers */
#define TPB 384

#include "cuda/fugue512_device.cuh"
#include "cuda/candidate_report.cuh"

/***************************************************/
// GPU Hash Function
__global__
__launch_bounds__(TPB, 2)
void fugue512_gpu_hash_64(uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint32_t mixtabs[256 * FUGUE512_R];

	fugue512_load_shared_r(mixtabs);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		const size_t hashPosition = thread;
		uint64_t *pHash = &g_hash[hashPosition<<3];
		uint32_t Hash[16];

		#pragma unroll 4
		for(int i = 0; i < 4; i++)
			AS_UINT4(&Hash[i*4]) = AS_UINT4(&pHash[i*2]);

		fugue512_hash_64_r(mixtabs, Hash);

		#pragma unroll 4
		for(int i = 0; i < 4; i++)
			AS_UINT4(&pHash[i*2]) = AS_UINT4(&Hash[i*4]);
	}
}

/***************************************************/
// Terminal variant: the full fugue (bit-identical to the CPU reference), the high 64
// bits compared against the target on-device, and the two lowest candidates
// reported (cuda/candidate_report.cuh) instead of storing d_hash.
__global__
__launch_bounds__(TPB, 2)
void fugue512_gpu_hash_64_final(uint32_t threads, uint64_t *g_hash, uint32_t *resNonce, const uint64_t target)
{
	__shared__ uint32_t mixtabs[256 * FUGUE512_R];

	fugue512_load_shared_r(mixtabs);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint64_t *pHash = &g_hash[thread<<3];
		uint32_t Hash[16];

		#pragma unroll 4
		for(int i = 0; i < 4; i++)
			AS_UINT4(&Hash[i*4]) = AS_UINT4(&pHash[i*2]);

		fugue512_hash_64_r(mixtabs, Hash);

		if (*(uint64_t*)&Hash[6] <= target) {
			report_candidate_2(resNonce, thread);
		}
	}
}

/* Unit self-test for cuda/fugue512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool fugue512_device_selftest(int thr_id);

__host__
void fugue512_cpu_init(int thr_id, uint32_t threads)
{
	fugue512_device_selftest(thr_id);
}

__host__
void fugue512_cpu_free(int thr_id)
{
}

__host__
void fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	fugue512_gpu_hash_64 <<<grid, block>>> (threads, (uint64_t*)d_hash);
}

__host__
void fugue512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	fugue512_gpu_hash_64_final <<<grid, block>>> (threads, (uint64_t*)d_hash, d_resNonce, target);
}
