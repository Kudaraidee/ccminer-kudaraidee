/*
 * sha256csm CUDA implementation.
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda/helper.cuh"
#include <miner.h>

#include "cuda/sha256_device.cuh"

__constant__ static uint32_t __align__(8) c_midstate76[8];
// block-2 round-3 prehash state (sha256_prehash_split_host, nonce = word 3)
__constant__ static uint32_t __align__(8) c_pre[8];
// block-2 schedule words 16..19 up to their nonce terms (sha256_preextend_w3_host)
__constant__ static uint32_t __align__(8) c_wx[4];

__constant__ static uint32_t __align__(8) c_target[2];

static uint32_t* d_resNonces[MAX_GPUS] = { 0 };

extern bool sha256_device_selftest(int thr_id);

// ------------------------------------------------------------------------------------------------

__device__ __forceinline__
uint64_t cuda_swab32ll(uint64_t x) {
	return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
}

#define SHA256CSM_TPB 256
#define SHA256CSM_NPT 8 // sequential nonces per thread (donor-style)

__global__ __launch_bounds__(SHA256CSM_TPB)
void sha256csm_gpu_hash_shared(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	for (uint32_t i = 0; i < SHA256CSM_NPT; i++)
	{
		const uint32_t idx = thread * SHA256CSM_NPT + i;
		if (idx >= threads) return;
		const uint32_t nonce = startNonce + idx;

		// block 2 of the first sha256: rounds 0..3 prehashed on the host,
		// dat[0..3] are ring slots only (assigned by the preextend words)
		uint32_t dat[16];
		#pragma unroll
		for (int j=4; j<15; j++) dat[j] = 0;
		dat[12] = 0x80000000;
		dat[15] = 0x380;

		uint32_t buf[8];
		sha256_transform_80_from_pre4(dat, c_pre, c_wx, nonce, c_midstate76, buf, c_sha256_K);

		// second sha256

		#pragma unroll
		for (int j=0; j<8; j++) dat[j] = buf[j];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int j=9; j<15; j++) dat[j] = 0;
		dat[15] = 0x100;

		#pragma unroll
		for (int j=0; j<8; j++) buf[j] = c_sha256_H[j];

		// truncated final rounds: only buf[6]/buf[7] are valid, compared
		// directly against the target; candidates are fully re-hashed on the CPU
		sha256_final_to_target(dat, buf, c_sha256_K);

		// valid nonces
		uint64_t high = cuda_swab32ll(((uint64_t*)buf)[3]);
		if (high <= c_target[0]) {
			resNonces[1] = atomicExch(resNonces, nonce);
		}
	}
}

__host__
void sha256csm_cpu_init(int thr_id)
{
	cuda_get_arch(thr_id);
	sha256_device_selftest(thr_id);
	CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], 2*sizeof(uint32_t)));
}

__host__
void sha256csm_free(int thr_id)
{
	if (d_resNonces[thr_id]) cudaFree(d_resNonces[thr_id]);
	d_resNonces[thr_id] = NULL;
}

__host__
void sha256csm_setBlock_80(uint32_t *pdata, uint32_t *ptarget)
{
	uint32_t _ALIGN(64) in[16], buf[8], pre[8], wx[4];
	for (int i=0;i<16;i++) in[i] = cuda_swab32(pdata[i]);
	for (int i=0;i<8;i++) buf[i] = h_sha256_H[i];
	sha256_transform_full(in, buf, h_sha256_K);

	// block-2 template: dataEnd words, nonce slot (unused), padding, length
	uint32_t _ALIGN(64) w[16] = { 0 };
	for (int i=0;i<3;i++) w[i] = cuda_swab32(pdata[16+i]);
	w[12] = 0x80000000;
	w[15] = 0x380;
	sha256_prehash_split_host(buf, w, 3, pre);
	sha256_preextend_w3_host(w, wx);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_midstate76, buf, 32, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_pre, pre, sizeof(pre), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_wx, wx, sizeof(wx), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_target, &ptarget[6], 8, 0, cudaMemcpyHostToDevice));
}

__host__
void sha256csm_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces)
{
	const uint32_t threadsperblock = SHA256CSM_TPB;

	dim3 grid((threads + threadsperblock*SHA256CSM_NPT - 1) / (threadsperblock*SHA256CSM_NPT));
	dim3 block(threadsperblock);

	CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xFF, 2 * sizeof(uint32_t)));
	cudaDeviceSynchronize();
	sha256csm_gpu_hash_shared <<<grid, block>>> (threads, startNonce, d_resNonces[thr_id]);
	cudaDeviceSynchronize();

	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if (resNonces[0] == resNonces[1]) {
		resNonces[1] = UINT32_MAX;
	}
}
