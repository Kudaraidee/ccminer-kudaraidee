/**
 * ECHO-512 64-byte launcher (optimised alexis formulation) — thin wrapper.
 *
 * The alexis ECHO-512 device implementation (c_echo_AES tables,
 * echo_round_alexis, echo512_hash_64_alexis) lives in
 * cuda/echo512_device.cuh (docs/coding-guideline.md §3).
 */

#include "cuda/helper.cuh"

#include "cuda/echo512_device.cuh"

/* 256x3 on Pascal (32 table copies), 128x5 above */
template <int TPB, int MINB>
__global__ __launch_bounds__(TPB, MINB)
static void x16_echo512_gpu_hash_64(uint32_t threads, uint32_t* g_hash)
{
	__shared__ uint32_t sharedMemory[1][ECHO_TAB];
	echo_fill_rep(sharedMemory);
	__syncthreads(); // barrier: shared AES table filled cooperatively

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *Hash = &g_hash[thread<<4];
		uint32_t hash[16];

		*(uint2x4*)&hash[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&hash[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		echo512_hash_64_alexis(sharedMemory, hash);

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&hash[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&hash[ 8];
	}
}

/* Unit self-test for the alexis section of cuda/echo512_device.cuh
 * (docs/coding-guideline.md §7 layer 1), defined in cuda/xfamily_selftest.cu.
 * No cpu_init exists for this stage, so the launcher runs it once (static
 * guard inside). */
extern bool echo512_alexis_device_selftest(int thr_id);

__host__
void echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	echo512_alexis_device_selftest(thr_id);

	if (device_sm[device_map[thr_id]] < 700)
		x16_echo512_gpu_hash_64<256, 3> <<<(threads + 255) / 256, 256>>> (threads, d_hash);
	else
		x16_echo512_gpu_hash_64<128, 5> <<<(threads + 127) / 128, 128>>> (threads, d_hash);
}

/* Legacy forwarder — consumers not yet migrated (x17, skydoge, x21s,
 * ghostrider) still call this name; remove once they call echo512_cpu_hash_64
 * directly. */
__host__
void x16_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	echo512_cpu_hash_64(thr_id, threads, d_hash);
}
