/*
 * Init-time device self-test for the Lyra2 v1 matrix stage.
 *
 * Covers blake2b_device.cuh and the v1 wander's split shared/register matrix through
 * the shipping launcher; the oracle is LYRA2() from Lyra2.c, which scanhash also uses.
 *
 * L2ST_THREADS is deliberately not a multiple of 8: the wander kernel runs 8 hashes
 * per warp with full-warp shuffles, and the last batch of a scan range can have any
 * size, so the partial warp (tail lanes kept in the shuffles) is tested here too.
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

extern void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint64_t *d_outputHash);

#define L2ST_THREADS 11u  /* one full warp of hashes plus a partial one: see above */
#define L2ST_MATRIX  (sizeof(uint64_t) * 4 * 4)

/* Runs the real launcher over L2ST_THREADS inputs and returns the digests.
 * in/out are plane-major exactly as the stage expects: word k of thread t at
 * [k * threads + t]. */
static bool lyra2_selftest_run(int thr_id, const uint64_t *in, uint64_t *out)
{
	uint64_t *d_hash = NULL, *d_matrix = NULL;
	const size_t hash_sz = 32 * (size_t) L2ST_THREADS;

	if (cudaMalloc(&d_matrix, L2ST_MATRIX * L2ST_THREADS) != cudaSuccess)
		return selftest_cuda_fault();
	if (cudaMalloc(&d_hash, hash_sz) != cudaSuccess) {
		cudaFree(d_matrix);
		return selftest_cuda_fault();
	}
	if (cudaMemcpy(d_hash, in, hash_sz, cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(d_hash); cudaFree(d_matrix);
		return selftest_cuda_fault();
	}

	/* sets the DMatrix device symbol; the caller must run this BEFORE the algo's
	 * own lyra2_cpu_init, which then re-points DMatrix at the real matrix. */
	lyra2_cpu_init(thr_id, L2ST_THREADS, d_matrix);
	lyra2_cpu_hash_32(thr_id, L2ST_THREADS, d_hash);

	bool ok = (cudaMemcpy(out, d_hash, hash_sz, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_hash);
	cudaFree(d_matrix);
	return ok ? true : selftest_cuda_fault();
}

/* CPU oracle for one 32-byte input, matching lyra2re_hash's call exactly. */
static void lyra2_selftest_cpu(const uint8_t in[32], uint8_t out[32])
{
	LYRA2(out, 32, in, 32, in, 32, 1, 8, 8);
}

bool lyra2_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested)
		return passed;
	tested = true;

	uint64_t gpu_in[4 * L2ST_THREADS], gpu_out[4 * L2ST_THREADS];
	uint8_t  cpu_in[L2ST_THREADS][32], cpu_out[32];

	/* distinct, non-degenerate inputs */
	for (uint32_t t = 0; t < L2ST_THREADS; t++)
		for (int b = 0; b < 32; b++)
			cpu_in[t][b] = (uint8_t) (0x5a + b * 7 + t * 31);

	for (uint32_t t = 0; t < L2ST_THREADS; t++)
		for (int k = 0; k < 4; k++)
			memcpy(&gpu_in[k * L2ST_THREADS + t], &cpu_in[t][k * 8], 8);

	bool pos_ok = lyra2_selftest_run(thr_id, gpu_in, gpu_out);
	if (pos_ok) {
		for (uint32_t t = 0; t < L2ST_THREADS && pos_ok; t++) {
			uint8_t got[32];
			for (int k = 0; k < 4; k++)
				memcpy(&got[k * 8], &gpu_out[k * L2ST_THREADS + t], 8);
			lyra2_selftest_cpu(cpu_in[t], cpu_out);
			if (memcmp(got, cpu_out, 32) != 0) {
				pos_ok = false;
				gpulog(LOG_ERR, thr_id, "Lyra2 v1 self-test: GPU/CPU mismatch on input %u", t);
			}
		}
	}

	/* Negative control: perturb one input bit and require the digest to move.
	 * Without this the test could "pass" against a kernel that ignores its input. */
	bool neg_ok = false;
	if (pos_ok) {
		uint64_t alt_in[4 * L2ST_THREADS], alt_out[4 * L2ST_THREADS];
		memcpy(alt_in, gpu_in, sizeof(alt_in));
		alt_in[0] ^= 1ULL;
		if (lyra2_selftest_run(thr_id, alt_in, alt_out)) {
			neg_ok = (memcmp(alt_out, gpu_out, sizeof(alt_out)) != 0);
			if (!neg_ok)
				gpulog(LOG_ERR, thr_id, "Lyra2 v1 self-test is vacuous"
					" (input change did not alter the digest)");
		}
	}

	passed = pos_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "Lyra2 v1 device self-test FAILED (positive %d negative %d)"
			" -- shares would be rejected", (int) pos_ok, (int) neg_ok);
	else if (!opt_quiet)
		gpulog(LOG_INFO, thr_id, "Lyra2 v1: self-test OK (GPU == CPU reference)");

	return selftest_gate(thr_id, "Lyra2 v1", passed);
}
