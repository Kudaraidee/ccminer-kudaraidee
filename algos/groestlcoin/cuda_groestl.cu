// Auf Groestlcoin spezialisierte Version von Groestl inkl. Bitslice (two hashes per thread)

#include <stdio.h>
#include <memory.h>

#include "cuda/helper.cuh"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#define __byte_perm(x,y,n) x
#endif

#include "miner.h"
#include "cuda/selftest_gate.cuh"

__constant__ uint32_t pTarget[8]; // Single GPU
__constant__ uint32_t groestlcoin_gpu_msg[32];

static uint32_t *d_resultNonce[MAX_GPUS];

#include "cuda/groestl512_x2_device.cuh"

#define SWAB32(x) cuda_swab32(x)

/* two hashes per thread (~250 registers, 64 words of shared stash per thread) */
#define TPB 128
#define MINB 2

/* Groestl-512(Groestl-512(header)) of nonces n0, n1; shared by the mining kernel and the self-test */
__device__ __forceinline__
void groestlcoin_x2_hash(const uint32_t n0, const uint32_t n1, uint32_t *st, const uint32_t stride, uint32_t (&out)[2][16])
{
	uint32_t m[2][20];
	#pragma unroll
	for (int w = 0; w < 19; w++) m[0][w] = m[1][w] = groestlcoin_gpu_msg[w];
	m[0][19] = SWAB32(n0);
	m[1][19] = SWAB32(n1);

	uint32_t s[8][8];
	groestl512_x2_load80(m, s);
	groestl512_x2_compress(s, st, stride);
	groestl512_x2_digest_to_msg64(s);
	groestl512_x2_compress(s, st, stride);
	groestl512_x2_store(s, out);
}

__global__ __launch_bounds__(TPB, MINB)
void groestlcoin_gpu_hash_x2(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	extern __shared__ uint32_t stash[];                     /* [64][TPB] */
	const uint32_t i0 = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
	if (i0 >= threads) return;
	const bool two = i0 + 1 < threads;

	uint32_t out_state[2][16];
	groestlcoin_x2_hash(startNounce + i0, startNounce + i0 + 1, &stash[threadIdx.x], blockDim.x, out_state);

	#pragma unroll
	for (int h = 0; h < 2; h++) {
		if (h == 1 && !two) break;
		int i, position = -1;
		bool rc = true;

		#pragma unroll 8
		for (i = 7; i >= 0; i--) {
			if (out_state[h][i] > pTarget[i]) {
				if(position < i) {
					position = i;
					rc = false;
				}
			 }
			 if (out_state[h][i] < pTarget[i]) {
				if(position < i) {
					position = i;
					rc = true;
				}
			 }
		}

		// Atomic: several threads can report in one launch, and a
		// read-compare-write would drop the lower nonce.
		if (rc)
			atomicMin(resNounce, startNounce + i0 + h);
	}
}

/* Init self-test: the chained compressions and the nonce patch of both hashes vs groestlhash() */
__global__ __launch_bounds__(1, 1)
void groestlcoin_selftest_kernel(uint32_t nounce, uint32_t *out)
{
	extern __shared__ uint32_t stash[];
	uint32_t out_state[2][16];
	groestlcoin_x2_hash(nounce, nounce + 1, stash, 1, out_state);

	#pragma unroll
	for (int h = 0; h < 2; h++)
		#pragma unroll 8
		for (int i = 0; i < 8; i++) out[8 * h + i] = out_state[h][i];
}

/* One header through the device path, arranged as setBlock does: out16 = digests of nounce
 * and nounce + 1.
 * Returns false only when CUDA itself failed. */
static bool groestlcoin_selftest_hash(const uint32_t *pdata, uint32_t nounce, uint32_t *out16)
{
	uint32_t endiandata[20], msgBlock[32] = { 0 };

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	memcpy(&msgBlock[0], endiandata, 80);
	msgBlock[20] = 0x80;
	msgBlock[31] = 0x01000000;

	if (cudaMemcpyToSymbol(groestlcoin_gpu_msg, msgBlock, 128) != cudaSuccess)
		return selftest_cuda_fault();

	uint32_t *d_out = NULL;
	if (cudaMalloc(&d_out, 16 * sizeof(uint32_t)) != cudaSuccess)
		return selftest_cuda_fault();

	groestlcoin_selftest_kernel <<< 1, 1, 64 * sizeof(uint32_t) >>> (nounce, d_out);

	bool ok = cudaDeviceSynchronize() == cudaSuccess
	       && cudaMemcpy(out16, d_out, 16 * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d_out);

	return ok ? true : selftest_cuda_fault();
}

/* CPU reference, nonce placed as scanhash's candidate re-verify places it. */
static void groestlcoin_selftest_oracle(const uint32_t *pdata, uint32_t nounce, uint32_t *vhash)
{
	uint32_t endiandata[20];

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	endiandata[19] = swab32(nounce);

	groestlhash(vhash, endiandata);
}

__host__
bool groestlcoin_device_selftest(int thr_id)
{
	uint32_t pdata[20], gpu[16], cpu[8], cpu2[8], gpu_flipped[16];
	bool kat0 = false, kat1 = false, neg = false;

	for (int i = 0; i < 20; i++)
		pdata[i] = 0x04030201u * (uint32_t)(i + 1);

	/* two nonce pairs, both hashes of the thread checked */
	if (groestlcoin_selftest_hash(pdata, 0x00000000u, gpu)) {
		groestlcoin_selftest_oracle(pdata, 0x00000000u, cpu);
		groestlcoin_selftest_oracle(pdata, 0x00000001u, cpu2);
		kat0 = memcmp(gpu, cpu, 32) == 0 && memcmp(&gpu[8], cpu2, 32) == 0;
	}
	if (groestlcoin_selftest_hash(pdata, 0xdeadbeefu, gpu)) {
		groestlcoin_selftest_oracle(pdata, 0xdeadbeefu, cpu);
		groestlcoin_selftest_oracle(pdata, 0xdeadbef0u, cpu2);
		kat1 = memcmp(gpu, cpu, 32) == 0 && memcmp(&gpu[8], cpu2, 32) == 0;
	}

	/* Negative leg on the device: a flipped header bit must change the digest.
	 * Comparing a host-side constant would be vacuous. */
	pdata[0] ^= 1u;
	if (groestlcoin_selftest_hash(pdata, 0xdeadbeefu, gpu_flipped))
		neg = memcmp(gpu_flipped, gpu, 32) != 0;
	pdata[0] ^= 1u;

	const bool passed = kat0 && kat1 && neg;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "groestl self-test FAILED (kat %d%d neg %d)",
			(int)kat0, (int)kat1, (int)neg);
	else
		gpulog(LOG_DEBUG, thr_id, "groestl self-test passed");

	return selftest_gate(thr_id, "groestl", passed);
}

__host__
void groestlcoin_cpu_init(int thr_id, uint32_t threads)
{
	// populates cuda_arch[] for this device (global init state, kept)
	cuda_get_arch(thr_id);

	CUDA_SAFE_CALL(cudaMalloc(&d_resultNonce[thr_id], sizeof(uint32_t)));

	groestlcoin_device_selftest(thr_id);
}

__host__
void groestlcoin_cpu_free(int thr_id)
{
	cudaFree(d_resultNonce[thr_id]);
}

__host__
void groestlcoin_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
{
	uint32_t msgBlock[32] = { 0 };

	memcpy(&msgBlock[0], data, 80);

	// Erweitere die Nachricht auf den Nachrichtenblock (padding)
	// Unsere Nachricht hat 80 Byte
	msgBlock[20] = 0x80;
	msgBlock[31] = 0x01000000;

	// groestl512 braucht hierfür keinen CPU-Code (die einzige Runde wird
	// auf der GPU ausgeführt)

	// Blockheader setzen (korrekte Nonce und Hefty Hash fehlen da drin noch)
	cudaMemcpyToSymbol(groestlcoin_gpu_msg, msgBlock, 128);

	cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32);
}

__host__
void groestlcoin_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonce)
{
	const uint32_t nthr = (threads + 1) / 2;   /* two hashes per thread */

	dim3 grid((nthr + TPB - 1) / TPB);
	dim3 block(TPB);

	cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	groestlcoin_gpu_hash_x2 <<<grid, block, 64 * TPB * sizeof(uint32_t)>>> (threads, startNounce, d_resultNonce[thr_id]);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	// MyStreamSynchronize(NULL, 0, thr_id);

	cudaMemcpy(resNonce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
