// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time KAT driver for shared stage launchers without a device header of
 * their own (docs/coding-guideline.md section 7, layer 1). Each stage supplies two
 * callbacks, run the real launcher over n slots and hash one slot with the sph
 * reference, and stkat_run() does the rest:
 *
 *   - vectors: a byte pattern, all-zero, all-0xFF, trailing-0xFF runs of
 *     1..64 bytes, then LCG fill (the runs catch dropped carries, which random
 *     inputs rarely hit);
 *   - GPU == reference on every slot;
 *   - negative leg: slot 0 with one flipped bit, on a ragged count (n-57), must
 *     match the reference and differ from the unflipped digest;
 *   - anchor_ok: the caller's check of the sph reference against a published
 *     vector (true where none exists);
 *   - fail-closed through cuda/selftest_gate.cuh (a CUDA resource failure warns).
 */
#ifndef CUDA_STAGE_SELFTEST_CUH
#define CUDA_STAGE_SELFTEST_CUH

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "cuda/selftest_gate.cuh"

#define STKAT_MAX_LEN 64

/* false = a CUDA call failed (not a wrong answer); route it via selftest_cuda_fault() */
typedef bool (*stkat_gpu_fn)(const uint8_t *in, uint8_t *out, int n);
typedef void (*stkat_ref_fn)(const uint8_t *in, uint8_t *out);

static inline void stkat_fill(uint8_t *buf, int n, int len, uint32_t seed)
{
	static const int runs[] = { 1, 2, 4, 8, 9, 16, 17, 24, 32, 33, 48, 63, 64 };
	const int nruns = (int)(sizeof(runs) / sizeof(runs[0]));
	for (int i = 0; i < n * len; i++) {
		seed = seed * 1664525u + 1013904223u;
		buf[i] = (uint8_t)(seed >> 24);
	}
	if (n > 0) for (int b = 0; b < len; b++) buf[b] = (uint8_t)b;          /* pattern */
	if (n > 1) memset(buf + 1 * len, 0x00, len);                            /* all zero */
	if (n > 2) memset(buf + 2 * len, 0xFF, len);                            /* all 0xFF */
	for (int r = 0; r < nruns && 3 + r < n; r++) {                          /* trailing 0xFF runs */
		const int k = runs[r] < len ? runs[r] : len;
		memset(buf + (3 + r) * len + (len - k), 0xFF, k);
	}
}

/* AoS <-> the 256-bit stages' SoA layout: 4 x uint64 per slot, word k of slot i
 * at [k * n + i], each word the little-endian bytes 8k..8k+7 of the 32-byte value. */
static inline void stkat_to_soa32(const uint8_t *aos, uint64_t *soa, int n)
{
	for (int i = 0; i < n; i++)
		for (int k = 0; k < 4; k++)
			memcpy(&soa[(size_t)k * n + i], aos + i * 32 + k * 8, 8);
}
static inline void stkat_from_soa32(const uint64_t *soa, uint8_t *aos, int n)
{
	for (int i = 0; i < n; i++)
		for (int k = 0; k < 4; k++)
			memcpy(aos + i * 32 + k * 8, &soa[(size_t)k * n + i], 8);
}

static inline bool stkat_run(int thr_id, const char *name, int in_len, int out_len, int n,
	uint32_t seed, stkat_gpu_fn gpu, stkat_ref_fn ref, bool anchor_ok)
{
	uint8_t *in  = (uint8_t*)malloc((size_t)n * in_len);
	uint8_t *got = (uint8_t*)malloc((size_t)n * out_len);
	uint8_t *exp = (uint8_t*)malloc((size_t)n * out_len);
	uint8_t d0[STKAT_MAX_LEN];
	if (!in || !got || !exp) { free(in); free(got); free(exp); return selftest_gate(thr_id, name, selftest_cuda_fault()); }

	stkat_fill(in, n, in_len, seed);
	for (int i = 0; i < n; i++) ref(in + i * in_len, exp + i * out_len);
	memcpy(d0, exp, out_len);

	int bad = 0, first = -1;
	bool gpu_ok = gpu(in, got, n);
	if (gpu_ok)
		for (int i = 0; i < n; i++)
			if (memcmp(got + i * out_len, exp + i * out_len, out_len)) { if (first < 0) first = i; bad++; }
	gpu_ok = gpu_ok && bad == 0;

	/* negative leg: one flipped bit in slot 0, ragged count */
	const int n2 = n - 57;
	in[0] ^= 0x01;
	for (int i = 0; i < n2; i++) ref(in + i * in_len, exp + i * out_len);
	bool neg_ok = gpu(in, got, n2);
	int bad2 = 0;
	if (neg_ok)
		for (int i = 0; i < n2; i++)
			if (memcmp(got + i * out_len, exp + i * out_len, out_len)) bad2++;
	neg_ok = neg_ok && bad2 == 0 && memcmp(got, d0, out_len) != 0;

	free(in); free(got); free(exp);
	const bool passed = anchor_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "%s device self-test FAILED (ref-anchor %d; gpu %d: %d of %d slots wrong, first %d; neg %d: %d of %d)",
			name, (int)anchor_ok, (int)gpu_ok, bad, n, first, (int)neg_ok, bad2, n2);
	else
		gpulog(LOG_DEBUG, thr_id, "%s device self-test passed", name);
	return selftest_gate(thr_id, name, passed);
}

#endif // CUDA_STAGE_SELFTEST_CUH
