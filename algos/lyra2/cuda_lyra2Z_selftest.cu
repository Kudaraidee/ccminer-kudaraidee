/*
 * Init-time device self-test for the Lyra2Z stage.
 *
 * Needs a different shape from the v1 gate: lyra2Z_gpu_hash_32_3 writes no digest
 * (its g_hash stores are commented out), so its only output is the candidate nonce.
 * Reading d_hash back here would checksum this file's own input.
 *
 * So the observable is the nonce, and the test makes the set of passing nonces a
 * function of the digests: compute all 8 on the CPU with LYRA2Z, set the target to
 * the 3rd-lowest top-64 bits so exactly 3 pass, then require both reported nonces to
 * match. A wrong digest moves the passing set. This also checks the report keeps the
 * two LOWEST candidates, slot 0 below slot 1.
 *
 * Negative control: target 0, which no digest can meet, so the launcher must report
 * UINT32_MAX -- proving the screen is not stuck on.
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
#include "algos/lyra2/Lyra2Z.h"
}

extern void lyra2Z_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern uint32_t lyra2Z_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void lyra2Z_setTarget(const void *ptarget);
extern uint32_t lyra2Z_getSecNonce(int thr_id, int num);

#define ZST_THREADS 8u     /* whole quads: the wander kernel's full-warp shuffle mask
                            * requires it; a ragged count is undefined behaviour */
#define ZST_MATRIX  (sizeof(uint64_t) * 4 * 4)
#define ZST_START   0x1000u

/* the device screens ((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3], i.e. the top
 * 8 bytes of the digest as a little-endian u64. Mirror that exactly. */
static uint64_t zst_top64(const uint8_t d[32])
{
	uint64_t v = 0;
	for (int i = 7; i >= 0; i--) v = (v << 8) | d[24 + i];
	return v;
}

static bool zst_run(int thr_id, const uint64_t *in, uint64_t top64, uint32_t *first, uint32_t *second)
{
	uint64_t *d_hash = NULL, *d_matrix = NULL;
	const size_t hash_sz = 32 * (size_t) ZST_THREADS;
	uint32_t tgt[8];

	if (cudaMalloc(&d_matrix, ZST_MATRIX * ZST_THREADS) != cudaSuccess)
		return selftest_cuda_fault();
	if (cudaMalloc(&d_hash, hash_sz) != cudaSuccess) {
		cudaFree(d_matrix); return selftest_cuda_fault();
	}
	if (cudaMemcpy(d_hash, in, hash_sz, cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(d_hash); cudaFree(d_matrix); return selftest_cuda_fault();
	}

	for (int i = 0; i < 6; i++) tgt[i] = 0xffffffffu;
	tgt[6] = (uint32_t) (top64 & 0xffffffffu);
	tgt[7] = (uint32_t) (top64 >> 32);

	/* must run before the algo's own lyra2Z_cpu_init / lyra2Z_setTarget, which
	 * overwrite the two symbols this borrows */
	lyra2Z_cpu_init(thr_id, ZST_THREADS, d_matrix);
	lyra2Z_setTarget(tgt);

	*first  = lyra2Z_cpu_hash_32(thr_id, ZST_THREADS, ZST_START, d_hash);
	*second = lyra2Z_getSecNonce(thr_id, 1);

	cudaFree(d_hash);
	cudaFree(d_matrix);
	return true;
}

bool lyra2Z_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	uint64_t gpu_in[4 * ZST_THREADS];
	uint8_t  cpu_in[ZST_THREADS][32], dig[ZST_THREADS][32];
	uint64_t top[ZST_THREADS];

	for (uint32_t t = 0; t < ZST_THREADS; t++)
		for (int b = 0; b < 32; b++)
			cpu_in[t][b] = (uint8_t) (0x37 + b * 11 + t * 53);
	for (uint32_t t = 0; t < ZST_THREADS; t++)
		for (int k = 0; k < 4; k++)
			memcpy(&gpu_in[k * ZST_THREADS + t], &cpu_in[t][k * 8], 8);

	for (uint32_t t = 0; t < ZST_THREADS; t++) {
		LYRA2Z(dig[t], 32, cpu_in[t], 32, cpu_in[t], 32, 8, 8, 8);
		top[t] = zst_top64(dig[t]);
	}

	/* target = the median-ish digest, so several pass and several do not */
	uint64_t sorted[ZST_THREADS];
	memcpy(sorted, top, sizeof(sorted));
	for (uint32_t i = 0; i < ZST_THREADS; i++)
		for (uint32_t j = i + 1; j < ZST_THREADS; j++)
			if (sorted[j] < sorted[i]) { uint64_t s = sorted[i]; sorted[i] = sorted[j]; sorted[j] = s; }
	const uint64_t tgt64 = sorted[2];      /* exactly 3 of the 8 pass */

	uint32_t exp_first = UINT32_MAX, exp_second = UINT32_MAX;
	for (uint32_t t = 0; t < ZST_THREADS; t++) {
		if (top[t] > tgt64) continue;
		const uint32_t n = ZST_START + t;
		if (exp_first == UINT32_MAX) exp_first = n;
		else if (exp_second == UINT32_MAX) exp_second = n;
	}

	uint32_t got_first = 0, got_second = 0;
	bool pos_ok = zst_run(thr_id, gpu_in, tgt64, &got_first, &got_second);
	if (pos_ok && (got_first != exp_first || got_second != exp_second)) {
		pos_ok = false;
		gpulog(LOG_ERR, thr_id, "Lyra2Z self-test: nonce mismatch (first %08x want %08x,"
			" second %08x want %08x)", got_first, exp_first, got_second, exp_second);
	}

	/* negative control: nothing can hash to <= 0 */
	bool neg_ok = false;
	if (pos_ok) {
		uint32_t f = 0, s = 0;
		if (zst_run(thr_id, gpu_in, 0ULL, &f, &s)) {
			neg_ok = (f == UINT32_MAX);
			if (!neg_ok)
				gpulog(LOG_ERR, thr_id, "Lyra2Z self-test is vacuous"
					" (target 0 still reported nonce %08x)", f);
		}
	}

	passed = pos_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "Lyra2Z device self-test FAILED (positive %d negative %d)"
			" -- shares would be rejected", (int) pos_ok, (int) neg_ok);
	else if (!opt_quiet)
		gpulog(LOG_INFO, thr_id, "Lyra2Z: self-test OK (GPU == CPU reference, both nonce slots)");

	return selftest_gate(thr_id, "Lyra2Z", passed);
}
