/*
 * shavite512 x11 kernel implementation — thin wrapper.
 *
 * The SHAvite-512 device implementation (AES tables, AES_ROUND_NOKEY,
 * KEY_EXPAND_ELT, c512, shavite_gpu_init) lives in
 * cuda/shavite512_device.cuh (docs/coding-guideline.md §3): this TU only holds
 * launchers over it (no device-code duplication).
 *
 * These are the LEGACY launchers kept for the not-yet-migrated x11-family
 * consumers (sib/timetravel/bitcore/x11evo + x13/x15/x17/x21s/qubit/...): the
 * 6-arg `x11_shavite512_cpu_hash_64` and the device self-test hook
 * `x11_shavite512_cpu_init` (which the migrated x-family still reaches through
 * the `shavite512_cpu_init` bridge in algos/common/cuda_x_stages.h). The
 * optimised 64-byte path used by the migrated chains is the sp launcher (bare
 * `shavite512_cpu_hash_64`). The legacy 80-byte first-stage launcher
 * (`x11_shavite512_cpu_hash_80`/`_setBlock_80`) was removed once its last
 * consumer (fresh) migrated to the de-branded `shavite512_*_80` in
 * algos/stages/cuda_shavite512_80.cu.
 */

#include <memory.h> // memcpy()

#include "cuda/helper.cuh"

#define TPB 128

#include "cuda/shavite512_device.cuh"

// GPU Hash
__global__ __launch_bounds__(TPB, 7) /* 64 registers with 128,8 - 72 regs with 128,7 */
void x11_shavite512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	__shared__ uint32_t sharedMemory[1024];

	shavite_gpu_init(sharedMemory);
	__syncthreads(); // barrier: shared AES table is filled cooperatively (threads <128)

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];

		// kopiere init-state
		uint32_t state[16] = {
			SPH_C32(0x72FCCDD8), SPH_C32(0x79CA4727), SPH_C32(0x128A077B), SPH_C32(0x40D55AEC),
			SPH_C32(0xD1901A06), SPH_C32(0x430AE307), SPH_C32(0xB29F5CD1), SPH_C32(0xDF07FBFC),
			SPH_C32(0x8E45D73D), SPH_C32(0x681AB538), SPH_C32(0xBDE86578), SPH_C32(0xDD577E47),
			SPH_C32(0xE275EADE), SPH_C32(0x502D9FCD), SPH_C32(0xB9357178), SPH_C32(0x022A4B9A)
		};

		// nachricht laden
		uint32_t msg[32];

		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		#pragma unroll 16
		for(int i=0;i<16;i++)
			msg[i] = Hash[i];

		// Nachrichtenende
		msg[16] = 0x80;
		#pragma unroll 10
		for(int i=17;i<27;i++)
			msg[i] = 0;

		msg[27] = 0x02000000;
		msg[28] = 0;
		msg[29] = 0;
		msg[30] = 0;
		msg[31] = 0x02000000;

		c512(sharedMemory, state, msg, 512);

		#pragma unroll 16
		for(int i=0;i<16;i++)
			Hash[i] = state[i];
	}
}

__host__
void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// note: 128 threads minimum are required to init the shared memory array
	x11_shavite512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	//MyStreamSynchronize(NULL, order, thr_id);
}

/* Unit self-test for cuda/shavite512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool shavite512_device_selftest(int thr_id);

__host__
void x11_shavite512_cpu_init(int thr_id, uint32_t threads)
{
	shavite512_device_selftest(thr_id);
}
