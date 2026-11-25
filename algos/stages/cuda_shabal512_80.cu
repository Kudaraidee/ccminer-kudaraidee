/*
* Shabal-512 80-byte first-stage launcher.
* tpruvot 2018, based on alexis x14 and xevan kernlx code
*
* The generic Shabal transform, constants and the 80-byte hash body
* (shabal512_hash_80) live in cuda/shabal512_device.cuh
* (docs/coding-guideline.md §3). This launcher is generic (bare-named); the
* device header holds the math. `x16_shabal512_*` remain as forwarders until
* the other consumers (ghostrider, x21s) migrate to the bare names.
*/

#include "cuda/shabal512_device.cuh"

__constant__ static uint32_t c_PaddedMessage80[20];

__host__
void shabal512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

// tpb 128, not the sph-era 256: shabal512_hash_80 is ~66 reg / 0 spill and
// latency-bound on the serial permutation chain, so more resident warps help.
// tpb256 fits only 3 blocks/SM (50% occ, register-file quantization waste);
// tpb128 packs 7 blocks (58%) -> +~17% isolated on sm_86 (same profile as the
// 64-byte kernel measured in algos/stages/cuda_shabal512.cu).
#define TPB_SHABAL 128

__global__ __launch_bounds__(TPB_SHABAL, 2)
void shabal512_gpu_hash_80(uint32_t threads, const uint32_t startNonce, uint32_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		const uint64_t hashPosition = thread;
		uint32_t *Hash = &g_hash[hashPosition << 4];
		shabal512_hash_80(c_PaddedMessage80, startNonce + thread, Hash);
	}
}

__host__
void shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = TPB_SHABAL;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	shabal512_gpu_hash_80 <<<grid, block >>>(threads, startNonce, d_hash);
}

/* Legacy forwarders — ghostrider and x21s still call these names; remove once
 * they call the bare shabal512_* launchers directly. */
__host__
void x16_shabal512_setBlock_80(void *pdata)
{
	shabal512_setBlock_80(pdata);
}

__host__
void x16_shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	shabal512_cuda_hash_80(thr_id, threads, startNonce, d_hash);
}
