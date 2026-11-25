/**
 * echo512-80 — generic 80-byte first-stage echo launcher.
 * tpruvot 2018 - GPL code
 *
 * The generic ECHO device functions (AES_2ROUND, echo_round,
 * cuda_echo_round_80, echo_gpu_init) live in cuda/echo512_device.cuh
 * (docs/coding-guideline.md §3). This launcher is bare-named; only the 80-byte
 * block constant is local. `x16_echo512_*` remain as forwarders until the other
 * consumers (ghostrider, x21s) migrate to the bare names.
 */

#include <stdio.h>
#include <memory.h>

#include "cuda/echo512_device.cuh"

__host__
void echo512_cpu_init(int thr_id, const uint32_t threads)
{
	aes_cpu_init(thr_id);
}

__constant__ static uint32_t c_PaddedMessage80[20];

__host__
void echo512_setBlock_80(void *endiandata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, endiandata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__ __launch_bounds__(128, 7) /* will force 72 registers */
void echo512_gpu_hash_80(uint32_t threads, uint32_t startNonce, uint64_t *g_hash)
{
	__shared__ uint32_t sharedMemory[1024];

	echo_gpu_init(sharedMemory);
	__syncthreads(); // barrier: shared AES table filled cooperatively

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t hashPosition = thread;
		uint32_t *pHash = (uint32_t*)&g_hash[hashPosition<<3];

		cuda_echo_round_80(sharedMemory, c_PaddedMessage80, startNonce + thread, pHash);
	}
}

__host__
void echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	echo512_gpu_hash_80<<<grid, block>>>(threads, startNonce, (uint64_t*)d_hash);
}

/* Legacy forwarders — ghostrider and x21s still call these names; remove once
 * they call the bare echo512_* launchers directly. */
__host__
void x16_echo512_cpu_init(int thr_id, const uint32_t threads)
{
	echo512_cpu_init(thr_id, threads);
}

__host__
void x16_echo512_setBlock_80(void *endiandata)
{
	echo512_setBlock_80(endiandata);
}

__host__
void x16_echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	echo512_cuda_hash_80(thr_id, threads, startNonce, d_hash);
}
