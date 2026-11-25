/*
 * SHAvite-3 (512) 80-byte first-stage launcher.
 *
 * The generic SHAvite device pieces (AES tables, aes rounds, AES_ROUND_NOKEY,
 * KEY_EXPAND_ELT, c512, shavite_gpu_init) live in cuda/shavite512_device.cuh
 * (docs/coding-guideline.md §3). This launcher is generic (bare-named); only
 * the 80-byte block constant is local. The kernel builds the padded 128-byte
 * block and calls the shared c512 with count=640 (the count==512 branch is
 * 64-byte only). `x16_shavite512_*` remain as forwarders until the other
 * consumers (ghostrider, x21s) migrate to the bare names.
 */

#include <memory.h> // memcpy()

#include "cuda/shavite512_device.cuh"

#define TPB 128

__constant__ uint32_t c_PaddedMessage80[20]; // padded message (80 bytes + padding)

__global__ __launch_bounds__(TPB, 5)
void shavite512_gpu_hash_80(uint32_t threads, uint32_t startNounce, void *outputHash)
{
	__shared__ uint32_t sharedMemory[1024];

	shavite_gpu_init(sharedMemory);
	__syncthreads(); // barrier: shared AES table filled cooperatively

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;

		// initial state
		uint32_t state[16] = {
			SPH_C32(0x72FCCDD8), SPH_C32(0x79CA4727), SPH_C32(0x128A077B), SPH_C32(0x40D55AEC),
			SPH_C32(0xD1901A06), SPH_C32(0x430AE307), SPH_C32(0xB29F5CD1), SPH_C32(0xDF07FBFC),
			SPH_C32(0x8E45D73D), SPH_C32(0x681AB538), SPH_C32(0xBDE86578), SPH_C32(0xDD577E47),
			SPH_C32(0xE275EADE), SPH_C32(0x502D9FCD), SPH_C32(0xB9357178), SPH_C32(0x022A4B9A)
		};

		uint32_t msg[32] = { 0 };

		#pragma unroll 32
		for(int i=0;i<20;i++)
		{
			msg[i] = c_PaddedMessage80[i];
		}

		#pragma unroll 16
		for (int i = 20; i<32; i++)
		{
			msg[i] = 0;
		}

		msg[19] = cuda_swab32(nounce);
		msg[20] = 0x80;
		msg[27] = 0x2800000;
		msg[31] = 0x2000000;

		c512(sharedMemory, state, msg, 640);

		uint32_t *outHash = (uint32_t *)outputHash + 16 * thread;

		#pragma unroll 16
		for(int i=0;i<16;i++)
			outHash[i] = state[i];

	} //thread < threads
}


__host__
void shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	shavite512_gpu_hash_80<<<grid, block>>>(threads, startNounce, d_outputHash);
}

__host__
void shavite512_setBlock_80(void *pdata)
{
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

/* Legacy forwarders — ghostrider and x21s still call these names; remove once
 * they call the bare shavite512_* launchers directly. */
__host__
void x16_shavite512_setBlock_80(void *pdata)
{
	shavite512_setBlock_80(pdata);
}

__host__
void x16_shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	shavite512_cpu_hash_80(thr_id, threads, startNounce, d_outputHash, order);
}
