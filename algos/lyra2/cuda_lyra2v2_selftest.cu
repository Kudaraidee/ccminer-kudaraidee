/*
 * Init-time device self-test for the Lyra2 v2 matrix stage (-a lyra2v2).
 *
 * lyra2v2_gpu_hash_32_3<SOA> writes the digest back into g_hash, so a direct
 * GPU-vs-CPU comparison is possible here (unlike the Lyra2Z stage, whose only
 * output is a nonce).
 *
 * Drives the SoA entry point, the one lyra2REv2.cu calls. That matters here: this
 * stage also serves consumers that allocate 64 B/thread and index AoS, and using
 * the wrong entry point silently runs off the end of a 32 B/thread buffer.
 *
 * Fails closed via cuda/selftest_gate.cuh.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/selftest_gate.cuh"

extern "C" {
#include "algos/lyra2/Lyra2.h"
}

extern void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern void lyra2v2_cpu_hash_32_soa(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);

/* 64, not 8: _1/_3 index DState with stride blockDim.x * gridDim.x -- the padded
 * grid width, not `threads` -- so a smaller count is written past its allocation.
 * 64 makes threads == padded width. The miner is safe because throughput is always
 * a multiple of 64. */
#define V2ST_THREADS 64u
#define V2ST_MATRIX  (16 * sizeof(uint64_t) * 4 * 3)   /* as lyra2REv2.cu sizes it */

static bool v2st_run(int thr_id, const uint64_t *in, uint64_t *out)
{
	uint64_t *d_hash = NULL, *d_matrix = NULL;
	const size_t hash_sz = 32 * (size_t) V2ST_THREADS;   /* 32 B/thread: the SoA stride */

	if (cudaMalloc(&d_matrix, V2ST_MATRIX * V2ST_THREADS) != cudaSuccess)
		return selftest_cuda_fault();
	if (cudaMalloc(&d_hash, hash_sz) != cudaSuccess) {
		cudaFree(d_matrix); return selftest_cuda_fault();
	}
	if (cudaMemcpy(d_hash, in, hash_sz, cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(d_hash); cudaFree(d_matrix); return selftest_cuda_fault();
	}

	lyra2v2_cpu_init(thr_id, V2ST_THREADS, d_matrix);
	lyra2v2_cpu_hash_32_soa(thr_id, V2ST_THREADS, 0, d_hash, 0);

	bool ok = (cudaMemcpy(out, d_hash, hash_sz, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_hash);
	cudaFree(d_matrix);
	return ok ? true : selftest_cuda_fault();
}

bool lyra2v2_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	uint64_t gpu_in[4 * V2ST_THREADS], gpu_out[4 * V2ST_THREADS];
	uint8_t  cpu_in[V2ST_THREADS][32], cpu_out[32];

	for (uint32_t t = 0; t < V2ST_THREADS; t++)
		for (int b = 0; b < 32; b++)
			cpu_in[t][b] = (uint8_t) (0x2f + b * 13 + t * 41);
	for (uint32_t t = 0; t < V2ST_THREADS; t++)
		for (int k = 0; k < 4; k++)
			memcpy(&gpu_in[k * V2ST_THREADS + t], &cpu_in[t][k * 8], 8);

	bool pos_ok = v2st_run(thr_id, gpu_in, gpu_out);
	if (pos_ok) {
		for (uint32_t t = 0; t < V2ST_THREADS && pos_ok; t++) {
			uint8_t got[32];
			for (int k = 0; k < 4; k++)
				memcpy(&got[k * 8], &gpu_out[k * V2ST_THREADS + t], 8);
			LYRA2(cpu_out, 32, cpu_in[t], 32, cpu_in[t], 32, 1, 4, 4);
			if (memcmp(got, cpu_out, 32) != 0) {
				pos_ok = false;
				gpulog(LOG_ERR, thr_id, "Lyra2 v2 self-test: GPU/CPU mismatch on input %u", t);
			}
		}
	}

	bool neg_ok = false;
	if (pos_ok) {
		uint64_t alt_in[4 * V2ST_THREADS], alt_out[4 * V2ST_THREADS];
		memcpy(alt_in, gpu_in, sizeof(alt_in));
		alt_in[0] ^= 1ULL;
		if (v2st_run(thr_id, alt_in, alt_out)) {
			neg_ok = (memcmp(alt_out, gpu_out, sizeof(alt_out)) != 0);
			if (!neg_ok)
				gpulog(LOG_ERR, thr_id, "Lyra2 v2 self-test is vacuous"
					" (input change did not alter the digest)");
		}
	}

	passed = pos_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "Lyra2 v2 device self-test FAILED (positive %d negative %d)"
			" -- shares would be rejected", (int) pos_ok, (int) neg_ok);
	else if (!opt_quiet)
		gpulog(LOG_INFO, thr_id, "Lyra2 v2: self-test OK (GPU == CPU reference, SoA stride)");

	return selftest_gate(thr_id, "Lyra2 v2", passed);
}
