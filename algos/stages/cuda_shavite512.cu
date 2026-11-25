/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
	optimized by sp - 2018/2019
*/
#include "cuda/helper_alexis.cuh"
#include "cuda/vectors_alexis.cuh"

#define INTENSIVE_GMF
#include "cuda/shavite512_sp_device.cuh"

// GPU Hash
__global__ __launch_bounds__(448, 2) /* 64 registers with 128,8 - 72 regs with 128,7 */
void x11_shavite512_gpu_hash_64_sp(const uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint32_t sharedMemory[256][32];

	if(threadIdx.x<256) aes_gpu_init256_s(sharedMemory);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
		shavite512_hash_64_s<true>(sharedMemory, NULL, &g_hash[thread<<3]);
}



__host__
/* Canonical optimised 64-byte SHAvite-512 launcher for the migrated x-family
 * (bare name). Self-contained sp kernel: in-kernel AES-table init, vectorised
 * __ldg4 I/O. The legacy 6-arg x11_shavite512_cpu_hash_64 (shared c512) stays
 * for the not-yet-migrated x11-family consumers (c11/sib/fresh/...). */
void shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + 256 - 1) / 256);
	dim3 block(256);

	x11_shavite512_gpu_hash_64_sp<<<grid, block>>>(threads, (uint64_t*)d_hash);
}