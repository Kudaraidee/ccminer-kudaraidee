// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * sha512256d CUDA implementation (double SHA-512/256 over an 80-byte header).
 *
 * An 80-byte message fits one 128-byte SHA-512 block, so each nonce costs
 * exactly two transforms. Rounds 0..8 of hash1 plus the constant halves of
 * round 9 are per-job host work (c_pre); the kernel resumes at round 9's
 * `+ w9`. The GPU only screens candidates against the target's high qword;
 * the host recomputes the full double hash and runs fulltest before submit.
 *
 * Ported from the cpuminer-opt scalar reference (algo/sha/sha512256d-4way.c).
 * Provenance, tuning log and rejected variants: see README.md.
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda/helper.cuh"
#include <miner.h>

#include "cuda/sha512_device.cuh"

#define TPB 256
/* min blocks/SM: 4 x 256 = 1024 threads, the occupancy this kernel already had
 * on every target. Naming it is the point - left unspecified ptxas stops short
 * of the register budget and schedules conservatively. Do not lower it: at 1 it
 * takes 145 registers on sm_75/sm_86 and occupancy collapses. */
#ifndef SHA512_MINB
#define SHA512_MINB 4
#endif
#define SHA512_BOUNDS __launch_bounds__(TPB, SHA512_MINB)

/* c_header[0..8] = 64-bit big-endian message words w0..w8 of the header
 * (built host-side from the 32-bit work->data words); c_header[9] = w9 with
 * the nonce half zeroed (nbits << 32) — the kernel ORs in its nonce.
 * c_pre = per-job prehash (registers after round 8 + round-9 t1c/t2c):
 * w0..w8 stay uploaded because the rounds 16+ schedule still reads them. */
__constant__ uint64_t c_header[10];
__constant__ uint64_t c_pre[10];

static uint32_t* d_resNonces[MAX_GPUS] = { 0 };
/* The kernel only clears the UINT32_MAX sentinel when it finds a candidate, so
 * a launch that found nothing leaves the buffer armed for the next one. */
static bool armed[MAX_GPUS] = { 0 };

extern bool sha512256d_device_selftest(int thr_id);

// ------------------------------------------------------------------------------------------------

__host__
void sha512256d_cpu_init(int thr_id)
{
	cuda_get_arch(thr_id);

	/* once per GPU; gates itself via cuda/selftest_gate.cuh */
	sha512256d_device_selftest(thr_id);

	CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], 2*sizeof(uint32_t)));
}

__host__
void sha512256d_free(int thr_id)
{
	if (d_resNonces[thr_id]) cudaFree(d_resNonces[thr_id]);
	d_resNonces[thr_id] = NULL;
	armed[thr_id] = false; // a re-init must memset the fresh allocation
}

__host__
void sha512256d_setBlock_80(const uint32_t *pdata)
{
	uint64_t hdr[10], pre[10];
	for (int i = 0; i < 9; i++)
		hdr[i] = ((uint64_t) pdata[2*i] << 32) | pdata[2*i + 1];
	hdr[9] = (uint64_t) pdata[18] << 32;
	sha512_prehash_split_host(hdr, pre);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_header, hdr, sizeof(hdr), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_pre, pre, sizeof(pre), 0, cudaMemcpyHostToDevice));
}

/* One per-nonce body, inlined by both kernels below so the diagnostic one cannot
 * drift. Returns the share's high qword in fulltest order = ((uint64_t*)vhash)[3]. */
__device__ __forceinline__
static uint64_t sha512256d_share_q3(const uint32_t nonce)
{
	uint64_t w[16], st[8];

	// hash1 = SHA512/256(header80): single block, nonce in the low half of w9
	#pragma unroll
	for (int i = 0; i < 9; i++) w[i] = c_header[i];
	w[9] = c_header[9] | nonce;
	w[10] = 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 11; i < 15; i++) w[i] = 0;
	w[15] = 640;
	sha512_256_init_state(st, c_sha512_256_H);
	sha512_transform_80_from_pre9(w, c_pre, st, c_sha512_K);

	// hash2 = SHA512/256(hash1 truncated to 32 bytes = words 0..3).
	// Full transform on purpose: eliding the last rounds measured slower
	// here (README.md). Don't re-add it.
	#pragma unroll
	for (int i = 0; i < 4; i++) w[i] = st[i];
	w[4] = 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 5; i < 15; i++) w[i] = 0;
	w[15] = 256;
	sha512_256_init_state(st, c_sha512_256_H);
	sha512_transform_full(w, st, c_sha512_K);

	return cuda_swab64(st[3]);
}

__global__ SHA512_BOUNDS
void sha512256d_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *result, const uint64_t targ_q3)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		if (sha512256d_share_q3(nonce) <= targ_q3)
		{
			/* Keep the two LOWEST nonces: the host resumes from max(slot0,slot1)+1,
			 * so anything higher steps over an unreported candidate. Both slots
			 * atomic; the loser comes from atomicMin's return, not a re-read. */
			const uint32_t prev = atomicMin(&result[0], nonce);
			if (prev != UINT32_MAX)
				atomicMin(&result[1], (prev > nonce) ? prev : nonce);
		}
	}
}

/* Diagnostic kernel (never launched while mining): accumulates the whole range
 * instead of screening it. acc[0] catches a wrong, missing or duplicated digest;
 * acc[1] weights by 2*nonce+1, so a permutation cannot cancel out (the weight
* must be injective and odd; nonce|1 is not injective). */
__global__ SHA512_BOUNDS
void sha512256d_gpu_checksum(const uint32_t threads, const uint32_t startNonce, uint64_t *acc)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		const uint64_t q3 = sha512256d_share_q3(nonce);

		// uint64_t is `unsigned long` on Linux: the casts pick the overload
		atomicXor((unsigned long long*)&acc[0], (unsigned long long) q3);
		atomicXor((unsigned long long*)&acc[1], (unsigned long long)(q3 * (2ull * (uint64_t)nonce + 1ull)));
	}
}

__host__
void sha512256d_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t targ_q3, uint32_t *resNonces)
{
	dim3 grid((threads + TPB - 1) / TPB);
	dim3 block(TPB);

	// No cudaDeviceSynchronize() is needed: the memset and the kernel are
	// ordered on the null stream and the blocking D2H copy waits for both.
	if (!armed[thr_id]) {
		CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xFF, 2 * sizeof(uint32_t)));
		armed[thr_id] = true;
	}

	sha512256d_gpu_hash <<<grid, block>>> (threads, startNonce, d_resNonces[thr_id], targ_q3);

	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if (resNonces[0] != UINT32_MAX)
		armed[thr_id] = false; // a candidate was written; the buffer is dirty
	if (resNonces[0] == resNonces[1]) {
		resNonces[1] = UINT32_MAX;
	}
}

/* Replays `threads` consecutive nonces and returns the checksums for the host to
 * reproduce. Own buffer, so it cannot disturb d_resNonces' armed state. */
__host__
void sha512256d_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc)
{
	uint64_t *d_acc = NULL;
	dim3 grid((threads + TPB - 1) / TPB);
	dim3 block(TPB);

	CUDA_SAFE_CALL(cudaMalloc(&d_acc, 2 * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMemset(d_acc, 0, 2 * sizeof(uint64_t)));

	sha512256d_gpu_checksum <<<grid, block>>> (threads, startNonce, d_acc);

	CUDA_SAFE_CALL(cudaMemcpy(acc, d_acc, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	cudaFree(d_acc);
}
