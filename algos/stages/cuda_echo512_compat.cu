/*
 * echo512 x11 kernel implementation — thin wrapper.
 *
 * The ECHO-512 device implementation (AES_2ROUND, cuda_echo_round,
 * echo_gpu_init and the static-init AES tables) lives in
 * cuda/echo512_device.cuh (docs/coding-guideline.md §3).
 */

#include "cuda/helper.cuh"

#include "cuda/echo512_device.cuh"

__global__ __launch_bounds__(128, 7) /* will force 72 registers */
void x11_echo512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	__shared__ uint32_t sharedMemory[1024];

	echo_gpu_init(sharedMemory);
	__syncthreads(); // barrier: shared AES table filled cooperatively (threads <128)

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];

		cuda_echo_round(sharedMemory, Hash);
	}
}

/* Unit self-test for cuda/echo512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool echo512_device_selftest(int thr_id);

__host__
void x11_echo512_cpu_init(int thr_id, uint32_t threads)
{
	aes_cpu_init(thr_id);

	echo512_device_selftest(thr_id);
}

__host__
void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x11_echo512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	//MyStreamSynchronize(NULL, order, thr_id);
}
