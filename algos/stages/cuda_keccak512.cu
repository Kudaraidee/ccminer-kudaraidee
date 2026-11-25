/*
	Keccak-512 stage kernels for the quark / x-family 64-byte chaining.

	Based upon Tanguy Pruvot's and SP's work (2016, Provos Alexis lineage).
	The keccak-512 absorb/output-round specializations were extracted
	bit-identically into cuda/keccak_device.cuh (docs/coding-guideline.md §3);
	the kernels here are thin wrappers over that shared device library.
	The 80-byte first-stage path delegates to the JHA jackpot kernels.
*/

#include <stdio.h>
#include <memory.h>

#include "cuda/helper_alexis.cuh"
#include "cuda/vectors_alexis.cuh"
#include "miner.h"

#include "cuda/keccak_device.cuh"

#define TPB52 128

__global__
__launch_bounds__(TPB52, 7)
void keccak512_gpu_hash_64(uint32_t threads, const uint32_t startNounce, uint2 *g_hash, uint32_t *g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		/* branch nonce vectors (quark compaction) store absolute nonces, so map
		 * back to the hash-buffer slot: subtract startNounce (matches
		 * skein/jh/groestl). NULL vector => the slot is just the thread index. */
		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread] - startNounce;

		uint2x4 *d_hash = (uint2x4*)&g_hash[hashPosition << 3];

		uint2 hash[8];
		*(uint2x4*)&hash[0] = __ldg4(&d_hash[0]);
		*(uint2x4*)&hash[4] = __ldg4(&d_hash[1]);

		keccak512_hash_64(hash);

		d_hash[0] = *(uint2x4*)&hash[0];
		d_hash[1] = *(uint2x4*)&hash[4];
	}
}

__host__
void keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t startNounce)
{
	const dim3 grid((threads + TPB52-1)/TPB52);
	const dim3 block(TPB52);

	keccak512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint2*)d_hash, d_nonceVector);
}
void jackpot_keccak512_cpu_init(int thr_id, uint32_t threads);
void jackpot_keccak512_cpu_setBlock(void *pdata, size_t inlen);
void jackpot_keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

/* Unit self-test for the keccak512 blocks in cuda/keccak_device.cuh
 * (docs/coding-guideline.md §7 layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool keccak512_device_selftest(int thr_id);

__host__
void keccak512_cpu_init(int thr_id, uint32_t threads)
{
	keccak512_device_selftest(thr_id);

	jackpot_keccak512_cpu_init(thr_id, threads);
}

__host__
void keccak512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	jackpot_keccak512_cpu_setBlock((void*)endiandata, 80);
}

__host__
void keccak512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	jackpot_keccak512_cpu_hash(thr_id, threads, startNounce, d_hash, 0);
}

