/**
 * Flex algorithm (Kylacoin / Lyncoin "flex") -- GPU driver.
 *
 * 15 core rounds + 3 CryptoNight-v1 rounds + a closing SHA3-256, where the
 * order is derived from SHA3-512 of the WHOLE 80-byte header and therefore
 * changes with EVERY NONCE.  The chain itself lives in
 * algos/flex/cuda_flex_pipeline.cuh; this file is the miner-facing driver:
 * throughput sizing, the startup guards, the candidate screen and the host
 * re-verify.
 *
 * Consensus notes that are easy to get wrong -- all four are gated, see
 * algos/flex/flex.h and the project notes
 * - every keccak is SHA-3 (0x06), never legacy Keccak (0x01);
 * - the CN finalization is `state[0] & 2` over {blake, groestl, skein-512};
 * - the high 32 bytes of a CN result are NOT zeroed and ARE hashed by the
 * closing SHA3-256;
 * - the core pool is 14 wide, drops JH, and is REINDEXED.
 *
 * Difficulty: flex leaves opt_target_factor at 1.0 -- it does NOT share
 * ghostrider's 2^16 factor.
 *
 * Reference: cpuminer-opt algo/flex (pool-confirmed on Kylacoin).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "algos/flex/cuda_flex_pipeline.cuh"
#include "algos/flex/cuda_flex_filter.cuh"
#include "algos/flex/flex_kat.h"
#include "cuda/selftest_gate.cuh"

static flex_ctx ctx[MAX_GPUS];
static bool init_done[MAX_GPUS] = { 0 };

/* FLEX_GATE_FAULT=<leg> corrupts one GPU digest on the host side, so each leg
 * of the startup gate and the -D differential can be shown to fire:
 * 1 = KAT, 2 = negative control, 3 = filtered chain, 4 = -D differential. */
static int flex_gate_fault(void)
{
	static int v = -1;
	if (v < 0) {
		const char *ev = getenv("FLEX_GATE_FAULT");
		v = ev ? atoi(ev) : 0;
	}
	return v;
}

// ----------------------------------------------------------------------------
// Startup guard 1 -- per-kernel self-consistency at full throughput.
//
// Each stage must produce the same answer on repeated runs. Variation means an
// intra-kernel race; that is how ghostrider's shared-AES-table bug in
// shavite/echo was found, and flex drives the same stages through a different
// launch pattern (many small launches instead of a few large ones), which is
// exactly the regime where such a race changes character.
// ----------------------------------------------------------------------------
static int flex_guard_races(int thr_id, uint32_t throughput)
{
	uint8_t tv[80];
	for (int i = 0; i < 80; i++) tv[i] = (uint8_t)(i * 13 + 5);

	int races = 0;
	for (int a = 0; a < FLEX_CORE_ALGO_COUNT; a++) {
		uint8_t first[64], g[64];
		int v64 = 0;
		for (int rep = 0; rep < 8; rep++) {
			cudaMemcpy(ctx[thr_id].d_hash, tv, 64, cudaMemcpyHostToDevice);
			flex_stage_hash_64(a, thr_id, throughput, ctx[thr_id].d_hash, 0);
			cudaMemcpy(g, ctx[thr_id].d_hash, 64, cudaMemcpyDeviceToHost);
			if (rep == 0) memcpy(first, g, 64);
			else if (memcmp(g, first, 64)) v64++;
		}
		if (v64) {
			races++;
			gpulog(LOG_ERR, thr_id, "flex: RACE in core algo %d (hash64 varied %d/7)", a, v64);
		}
	}
	return races;
}

// ----------------------------------------------------------------------------
// Startup guard 2 -- the whole chain, GPU vs CPU, with a bisect.
//
// Runs the real 19-step pipeline on a small batch and compares every lane's
// digest to flex_hash(). On a mismatch it re-runs with the pipeline's step
// limit to report the FIRST diverging step plus that lane's derived order --
// without it a failure says only "one of 19 stages, on one of 14 algorithms".
// ----------------------------------------------------------------------------
static void flex_cpu_chain_upto(const uint32_t *hdr, int steps, uint8_t out[64])
{
	uint8_t coreOrder[FLEX_CORE_CHAIN_LEN], cnOrder[FLEX_CN_ALGO_COUNT];
	uint8_t h1[64] = { 0 }, h2[64] = { 0 };
	int step = 0, cn = 0;

	flex_derive_order(hdr, coreOrder, cnOrder, NULL);
	uint8_t *in = h2, *o = h1;

	if (step++ >= steps) { memcpy(out, in, 64); return; }
	flex_core_algo_cpu(coreOrder[0], hdr, o, 80);
	{ uint8_t *t = in; in = o; o = t; }

	for (int r = 1; r < FLEX_CORE_CHAIN_LEN; r++) {
		if (step++ >= steps) { memcpy(out, in, 64); return; }
		flex_core_algo_cpu(coreOrder[r], in, o, 64);
		{ uint8_t *t = in; in = o; o = t; }
		if (r == 4 || r == 9 || r == 14) {
			if (step++ >= steps) { memcpy(out, in, 64); return; }
			flex_cn_algo_cpu(cnOrder[cn++], in, o, 64);
			{ uint8_t *t = in; in = o; o = t; }
		}
	}
	if (step++ >= steps) { memcpy(out, in, 64); return; }
	memset(out, 0, 64);
	flex_sha3_256(out, in, 64);
}

/* the nonce filter: the same chain check over the FILTERED path -- the one scanhash now
 * takes. flex_guard_chain() drives the CONTIGUOUS path, so on its own it
 * leaves the shipped code ungated: how -a lyra2v2 stayed dead behind a green
 * gate. This also proves what the filter gate cannot -- that the round-0
 * GATHER hands each scattered lane its OWN 80-byte state (a chunk off-by-one
 * would mis-pair lanes with nonces and is invisible to a filter-only check).
 */
/* The nonce pre-filter is ON by default; FLEX_NONCE_FILTER=0 takes the plain
 * path, which is the arm to run if the filtered path is ever suspected. */
static bool flex_filter_enabled(void)
{
	static int v = -1;
	if (v < 0) {
		const char *ev = getenv("FLEX_NONCE_FILTER");
		v = ev ? (atoi(ev) ? 1 : 0) : 1;
		if (!v)
			applog(LOG_INFO, "flex: nonce filter disabled by FLEX_NONCE_FILTER=0");
	}
	return v != 0;
}

static int flex_guard_filtered(int thr_id, uint32_t saved_threads)
{
	/* Startup latency matters: an API-driven algo switch pays it on every
	 * change. */
	if (!flex_filter_enabled())
		return 0;

	uint32_t guard_lanes = 256u;
	{
		const char *ev = getenv("FLEX_GUARD_LANES");
		if (ev) {
			const unsigned long v = strtoul(ev, NULL, 0);
			if (v >= FLEX_CN_TPB) guard_lanes = (uint32_t)v;
		}
	}
	const uint32_t T = min(saved_threads, guard_lanes);
	flex_ctx *c = &ctx[thr_id];
	const uint32_t real = c->threads;
	c->threads = T;

	uint8_t  *h_gpu = (uint8_t*)malloc((size_t)64 * T);
	uint32_t *h_n   = (uint32_t*)malloc(sizeof(uint32_t) * T);
	int diffs = 0, checked = 0;
	uint32_t lcg = 0x2468aceu;

	for (int it = 0; it < 2 && !diffs; it++) {
		uint32_t pd[20], ed[20];
		for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
		for (int k = 0; k < 20; k++) be32enc(&ed[k], pd[k]);

		/* Arm 1 uses a short span so the filter under-yields. At the normal span it
		 * over-yields and the count clamps to the capacity, which makes lanes equal
		 * stride and hides that class of fault. */
		const uint32_t span = (it == 1)
			? flex_filter_span(T, FLEX_FILTER_ACCEPT_SEED) / 4u
			: flex_filter_span(T, FLEX_FILTER_ACCEPT_SEED);
		uint32_t yield = 0;
		c->span = span;
		cudaMemset(c->d_count, 0, sizeof(uint32_t));
		flex_seed_setBlock_80(ed);
		flex_filter_cpu(span, pd[19], T, c->d_nonces, c->d_count);
		cudaMemcpy(&yield, c->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (yield > T) yield = T;
		yield = (yield / FLEX_CN_TPB) * FLEX_CN_TPB;
		if (yield == 0) continue;
		if (opt_debug)
			gpulog(LOG_DEBUG, thr_id, "flex: guard arm %d span %u -> yield %u of %u%s",
			       it, span, yield, T, (yield == T) ? " (CLAMPED)" : " (under-yield)");
		cudaMemcpy(h_n, c->d_nonces, sizeof(uint32_t) * yield, cudaMemcpyDeviceToHost);

		flex_pipeline_run(c, thr_id, ed, pd, pd[19], -1, c->d_nonces, yield);
		const uint32_t L = c->lanes;
		cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * L, cudaMemcpyDeviceToHost);

		for (uint32_t t = 0; t < L; t++) {
			uint32_t lane[20];
			uint32_t _ALIGN(64) ref[8];
			memcpy(lane, ed, 80);
			be32enc(&lane[19], h_n[t]);
			flex_hash(ref, lane);
			checked++;
			if (memcmp(h_gpu + (size_t)t * 64, ref, 32) != 0) {
				diffs++;
				if (diffs == 1)
					gpulog(LOG_ERR, thr_id,
					       "flex: FILTERED chain DIFF at lane %u, nonce %08x", t, h_n[t]);
			}
		}
	}

	c->threads = real;
	free(h_gpu); free(h_n);
	if (checked == 0) {
		gpulog(LOG_ERR, thr_id, "flex: filtered guard VACUOUS -- the filter yielded nothing");
		return -1;
	}
	if (!diffs)
		gpulog(LOG_INFO, thr_id, "flex: GPU==CPU on the filtered chain (%d lanes)", checked);
	return diffs;
}

static int flex_guard_chain(int thr_id, uint32_t saved_threads)
{
	/* A small batch keeps startup fast, because this guard is about chain
	 * logic rather than throughput.
	 *
	 * But a SMALL batch cannot see a SCALE-dependent fault, and an earlier phase shipped
	 * exactly one: with variable-stride packing every digest was wrong above
	 * ~4096 lanes while 256 lanes passed cleanly. FLEX_GUARD_LANES overrides
	 * the width so the guard can be run at the real batch size -- do that
	 * after ANY change to the scratchpad layout or the lane count. */
	/* 32 lanes x 2 headers = 64 chains. Each chain is a different per-nonce
	 * route, so 64 of them still sample all 14 core algos and all 6 CN variants
	 * many times over -- enough to catch a broken build or card. Raise it with
	 * FLEX_GUARD_LANES for real verification. */
	uint32_t guard_lanes = 32u;
	{
		const char *ev = getenv("FLEX_GUARD_LANES");
		if (ev) {
			const unsigned long v = strtoul(ev, NULL, 0);
			if (v >= FLEX_CN_TPB) guard_lanes = (uint32_t)v;
		}
	}
	const uint32_t T = min(saved_threads, guard_lanes);
	flex_ctx *c = &ctx[thr_id];
	/* Shrink the CAPACITY for the guard so the pipeline derives and runs only
	 * T lanes; flex_pipeline_run() resets c->lanes from it on every call. */
	const uint32_t real = c->threads;
	c->threads = T;

	uint8_t *h_gpu = (uint8_t*)malloc((size_t)64 * T);
	int diffs = 0;
	uint32_t lcg = 0x12345678u;

	for (int t = 0; t < 2 && !diffs; t++) {
		uint32_t pd[20], ed[20];
		for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
		for (int k = 0; k < 20; k++) be32enc(&ed[k], pd[k]);
		/* RAW frame, like pdata[19] -- see flex_sha3_512_seed_80. Lane l is the
		 * raw nonce startNonce + l, whose header word is be32enc of that. */
		const uint32_t startNonce = pd[19];

		flex_pipeline_run(c, thr_id, ed, pd, startNonce, -1, NULL, 0);
		cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * T, cudaMemcpyDeviceToHost);

		int firstbad = -1;
		for (uint32_t l = 0; l < T; l++) {
			uint32_t lane[20];
			uint8_t ref[32];
			memcpy(lane, ed, 80);
			be32enc(&lane[19], startNonce + l);
			flex_hash(ref, lane);
			if (memcmp(h_gpu + (size_t)l * 64, ref, 32) != 0) { firstbad = (int)l; break; }
		}
		if (firstbad < 0)
			continue;

		diffs++;
		uint32_t lane[20];
		memcpy(lane, ed, 80);
		be32enc(&lane[19], startNonce + (uint32_t)firstbad);
		for (int s = 1; s <= FLEX_PIPELINE_STEPS; s++) {
			uint8_t cref[64];
			flex_pipeline_run(c, thr_id, ed, pd, startNonce, s, NULL, 0);
			cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * T, cudaMemcpyDeviceToHost);
			flex_cpu_chain_upto(lane, s, cref);
			const int n = (s == FLEX_PIPELINE_STEPS) ? 32 : 64;
			if (memcmp(h_gpu + (size_t)firstbad * 64, cref, n) != 0) {
				uint8_t co[FLEX_CORE_CHAIN_LEN], cv[FLEX_CN_ALGO_COUNT];
				flex_derive_order(lane, co, cv, NULL);
				gpulog(LOG_ERR, thr_id,
					"flex: GPU!=CPU, first divergence at step %d/%d (lane %d)",
					s, FLEX_PIPELINE_STEPS, firstbad);
				gpulog(LOG_ERR, thr_id,
					"flex: that lane's core order %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d cn %d,%d,%d",
					co[0],co[1],co[2],co[3],co[4],co[5],co[6],co[7],co[8],co[9],
					co[10],co[11],co[12],co[13],co[14], cv[0],cv[1],cv[2]);
				break;
			}
		}
	}

	free(h_gpu);
	c->threads = real;
	return diffs;
}

// ----------------------------------------------------------------------------
// Startup guard 3 -- WIDTH INVARIANCE.
//
// Every lane's chain is a function of its own nonce alone, so lane L must
// produce the same digest whatever the batch width. Running one header at two
// widths and comparing the overlap therefore needs no CPU reference at all --
// and it catches the class guard 2 cannot: guard 2 runs a small batch, and a
// fault that only appears at scale passes it cleanly. an earlier phase shipped exactly such
// a fault (every digest wrong above ~6144 lanes, 256 lanes perfect).
//
// Reports the first step whose output stops being width-invariant, plus how
// many of the overlapping lanes moved -- the count distinguishes "one bucket
// is wrong" from "everything is wrong".
// ----------------------------------------------------------------------------
static int flex_guard_width(int thr_id, uint32_t wide)
{
	flex_ctx *c = &ctx[thr_id];
	const uint32_t narrow = 256;
	if (wide <= narrow)
		return 0;

	const uint32_t real = c->threads;
	uint32_t pd[20], ed[20];
	uint32_t lcg = 0x5bd1e995u;
	for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
	for (int k = 0; k < 20; k++) be32enc(&ed[k], pd[k]);
	const uint32_t startNonce = pd[19];

	uint8_t *a = (uint8_t*)malloc((size_t)64 * narrow);
	uint8_t *b = (uint8_t*)malloc((size_t)64 * narrow);
	int bad_step = 0;

	for (int s = 1; s <= FLEX_PIPELINE_STEPS && !bad_step; s++) {
		c->threads = narrow;
		flex_pipeline_run(c, thr_id, ed, pd, startNonce, s, NULL, 0);
		cudaMemcpy(a, c->d_hash, (size_t)64 * narrow, cudaMemcpyDeviceToHost);

		c->threads = wide;
		flex_pipeline_run(c, thr_id, ed, pd, startNonce, s, NULL, 0);
		cudaMemcpy(b, c->d_hash, (size_t)64 * narrow, cudaMemcpyDeviceToHost);

		const int n = (s == FLEX_PIPELINE_STEPS) ? 32 : 64;
		uint32_t moved = 0, first = 0;
		for (uint32_t l = 0; l < narrow; l++)
			if (memcmp(a + (size_t)l * 64, b + (size_t)l * 64, n) != 0) {
				if (!moved) first = l;
				moved++;
			}
		if (moved) {
			bad_step = s;
			gpulog(LOG_ERR, thr_id,
				"flex: WIDTH-DEPENDENT at step %d/%d -- %u of %u overlapping lanes differ between "
				"a %u-lane and a %u-lane batch (first lane %u)",
				s, FLEX_PIPELINE_STEPS, moved, narrow, narrow, wide, first);
			/* Which lanes SURVIVED? If the survivors all share one algorithm,
			 * the per-bucket base offset is not reaching the stage; if they
			 * are scattered across algos, the fault is in the gather/scatter
			 * or the data, not the dispatch. */
			uint32_t okhist[FLEX_CORE_ALGO_COUNT] = { 0 };
			uint32_t badhist[FLEX_CORE_ALGO_COUNT] = { 0 };
			const uint8_t *row = c->h_core + (size_t)(s - 1) * c->threads;
			for (uint32_t l = 0; l < narrow; l++) {
				const uint8_t alg = row[l];
				if (alg >= FLEX_CORE_ALGO_COUNT) continue;
				if (memcmp(a + (size_t)l * 64, b + (size_t)l * 64, n) == 0) okhist[alg]++;
				else badhist[alg]++;
			}
			char ok[160] = { 0 }, bd[160] = { 0 };
			for (uint32_t k = 0; k < FLEX_CORE_ALGO_COUNT; k++) {
				char t1[16];
				snprintf(t1, sizeof t1, "%u ", okhist[k]);  strncat(ok, t1, sizeof(ok) - strlen(ok) - 1);
				snprintf(t1, sizeof t1, "%u ", badhist[k]); strncat(bd, t1, sizeof(bd) - strlen(bd) - 1);
			}
			gpulog(LOG_ERR, thr_id, "flex: per-algo MATCH counts (algo 0..13): %s", ok);
			gpulog(LOG_ERR, thr_id, "flex: per-algo DIFFER counts (algo 0..13): %s", bd);
		}
	}

	c->threads = real;
	free(a); free(b);
	if (!bad_step)
		gpulog(LOG_INFO, thr_id, "flex: width-invariant (%u vs %u lanes)", narrow, wide);
	return bad_step;
}

// ----------------------------------------------------------------------------
// Startup gate, fail-closed, one CN block per leg: kat (pool-accepted digest),
// neg (nonce + 1 must differ), filtered (scanhash's path vs flex_hash()).
// A CUDA failure goes through selftest_cuda_fault() and only warns.
// ----------------------------------------------------------------------------
static bool flex_gate_kat(int thr_id, uint8_t *h_gpu, bool *neg_ok)
{
	flex_ctx *c = &ctx[thr_id];
	const uint32_t T = FLEX_CN_TPB;
	uint32_t ed[20], pd[20];

	memcpy(ed, flex_kat_header, 80);
	for (int k = 0; k < 20; k++) pd[k] = swab32(ed[k]);

	c->threads = T;
	flex_pipeline_run(c, thr_id, ed, pd, pd[19], -1, NULL, 0);
	if (cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * T, cudaMemcpyDeviceToHost) != cudaSuccess
	    || c->lanes < 2)
		return selftest_cuda_fault();

	if (flex_gate_fault() == 1) h_gpu[0] ^= 1;
	if (flex_gate_fault() == 2) memcpy(h_gpu + 64, flex_kat_expected, 32);

	*neg_ok = memcmp(h_gpu + 64, flex_kat_expected, 32) != 0;
	return memcmp(h_gpu, flex_kat_expected, 32) == 0;
}

static bool flex_gate_filtered(int thr_id, uint8_t *h_gpu, int *checked)
{
	flex_ctx *c = &ctx[thr_id];
	const uint32_t T = FLEX_CN_TPB;
	uint32_t h_n[FLEX_CN_TPB];
	uint32_t lcg = 0x6a09e667u;

	*checked = 0;
	c->threads = T;
	/* a few headers, in case one does not fill a block */
	for (int it = 0; it < 4; it++) {
		uint32_t pd[20], ed[20];
		for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
		for (int k = 0; k < 20; k++) be32enc(&ed[k], pd[k]);

		const uint32_t span = flex_filter_span(T, FLEX_FILTER_ACCEPT_SEED);
		uint32_t yield = 0;
		c->span = span;
		cudaMemset(c->d_count, 0, sizeof(uint32_t));
		flex_seed_setBlock_80(ed);
		flex_filter_cpu(span, pd[19], T, c->d_nonces, c->d_count);
		if (cudaMemcpy(&yield, c->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
			return selftest_cuda_fault();
		if (yield > T) yield = T;
		if (yield < T) continue;
		if (cudaMemcpy(h_n, c->d_nonces, sizeof(uint32_t) * T, cudaMemcpyDeviceToHost) != cudaSuccess)
			return selftest_cuda_fault();

		flex_pipeline_run(c, thr_id, ed, pd, pd[19], -1, c->d_nonces, T);
		const uint32_t L = c->lanes;
		if (L == 0 || cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * L, cudaMemcpyDeviceToHost) != cudaSuccess)
			return selftest_cuda_fault();
		if (flex_gate_fault() == 3) h_gpu[(size_t)(L - 1) * 64] ^= 1;

		/* first, last and evenly between */
		const uint32_t probes = (L < 8) ? L : 8;
		for (uint32_t i = 0; i < probes; i++) {
			const uint32_t t = (probes == 1) ? 0 : (i * (L - 1)) / (probes - 1);
			uint32_t lane[20];
			uint32_t _ALIGN(64) ref[8];
			memcpy(lane, ed, 80);
			be32enc(&lane[19], h_n[t]);
			flex_hash(ref, lane);
			(*checked)++;
			if (memcmp(h_gpu + (size_t)t * 64, ref, 32) != 0) {
				gpulog(LOG_ERR, thr_id, "flex: gate FILTERED chain DIFF at lane %u of %u, nonce %08x",
				       t, L, h_n[t]);
				return false;
			}
		}
		return true;
	}
	gpulog(LOG_ERR, thr_id, "flex: gate filtered leg VACUOUS -- the filter never filled a block");
	return false;
}

static bool flex_gate(int thr_id)
{
	flex_ctx *c = &ctx[thr_id];
	const uint32_t real = c->threads;
	uint8_t *h_gpu = (uint8_t*)malloc((size_t)64 * FLEX_CN_TPB);
	bool neg_ok = false;
	int checked = 0;

	const bool kat_ok = flex_gate_kat(thr_id, h_gpu, &neg_ok);
	const bool fil_ok = kat_ok && flex_gate_filtered(thr_id, h_gpu, &checked);

	c->threads = real;
	free(h_gpu);

	if (!kat_ok)
		gpulog(LOG_ERR, thr_id, "flex: GPU KAT FAILED -- the pool-accepted header does not hash to its digest");
	else if (!neg_ok)
		gpulog(LOG_ERR, thr_id, "flex: gate negative FAILED -- nonce + 1 gave the KAT digest");
	const bool ok = kat_ok && neg_ok && fil_ok;
	if (ok)
		gpulog(LOG_INFO, thr_id, "flex: GPU self-test OK (KAT, negative, filtered %d lanes)", checked);
	return ok;
}

// ----------------------------------------------------------------------------
// -D differential: once per job, re-hash FLEX_DIFF_LANES (default 16) of the
// batch's lanes on the CPU, spread first to last. Catches wrong digests that
// never reach the screen.
// ----------------------------------------------------------------------------
static void flex_debug_differential(int thr_id, const uint32_t *pdata, const uint32_t *endiandata,
                                    uint32_t hashed, int use_filter, uint32_t scan_base)
{
	static uint32_t last_sig[MAX_GPUS];
	static int nlanes = -1;
	if (nlanes < 0) {
		const char *ev = getenv("FLEX_DIFF_LANES");
		nlanes = ev ? atoi(ev) : 16;
		if (nlanes < 1) nlanes = 1;
	}

	/* FNV-1a over the header minus the nonce: once per job, not per batch. */
	uint32_t sig = 2166136261u;
	for (int k = 0; k < 19; k++) { sig ^= pdata[k]; sig *= 16777619u; }
	if (sig == last_sig[thr_id] || hashed == 0)
		return;
	last_sig[thr_id] = sig;

	flex_ctx *c = &ctx[thr_id];
	uint8_t *h_gpu = (uint8_t*)malloc((size_t)64 * hashed);
	if (cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * hashed, cudaMemcpyDeviceToHost) != cudaSuccess) {
		gpulog(LOG_WARNING, thr_id, "flex: -D differential skipped, digest copy failed");
		free(h_gpu);
		return;
	}

	const uint32_t probes = ((uint32_t)nlanes < hashed) ? (uint32_t)nlanes : hashed;
	if (flex_gate_fault() == 4) h_gpu[(size_t)(hashed - 1) * 64] ^= 1;

	uint32_t bad = 0;
	for (uint32_t i = 0; i < probes; i++) {
		const uint32_t t = (probes == 1) ? 0 : (uint32_t)(((uint64_t)i * (hashed - 1)) / (probes - 1));
		const uint32_t n = use_filter ? c->h_nonces[t] : scan_base + t;
		uint32_t lane[20];
		uint32_t _ALIGN(64) ref[8];
		memcpy(lane, endiandata, 76);
		be32enc(&lane[19], n);
		flex_hash(ref, lane);
		if (memcmp(h_gpu + (size_t)t * 64, ref, 32) != 0) {
			if (!bad) {
				uint8_t co[FLEX_CORE_CHAIN_LEN], cv[FLEX_CN_ALGO_COUNT];
				flex_derive_order(lane, co, cv, NULL);
				gpulog(LOG_ERR, thr_id,
				       "flex: -D differential MISMATCH lane %u of %u, nonce %08x, cn %d,%d,%d",
				       t, hashed, n, cv[0], cv[1], cv[2]);
			}
			bad++;
		}
	}
	free(h_gpu);
	if (bad)
		gpulog(LOG_ERR, thr_id, "flex: -D differential job %08x: %u of %u lanes WRONG",
		       sig, bad, probes);
	else
		gpulog(LOG_DEBUG, thr_id, "flex: -D differential job %08x: %u/%u lanes GPU==CPU",
		       sig, probes, probes);
}

// ----------------------------------------------------------------------------
extern "C" int scanhash_flex(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	/* --benchmark resets the nonce every call; carry it so batches stay fresh */
	static uint32_t bench_cursor[MAX_GPUS];
	if (opt_benchmark) {
		pdata[19] = bench_cursor[thr_id];
		max_nonce = UINT32_MAX;
	}

	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	uint32_t _ALIGN(64) endiandata[20];

	/* Sized so the CPU re-verify actually fires at this algo's rates; the
	 * usual 0x00ff screen yields roughly one candidate every few hours. */
	if (opt_benchmark) {
		const char *env = getenv("FLEX_BENCH_TARGET");
		ptarget[7] = env ? (uint32_t) strtoul(env, NULL, 0) : 0x0003ffff;
	}

	if (!init_done[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}

		/* feeds the fail-closed gate below */
		bool cpu_ok;
		{
			char det[FLEX_SELFTEST_DETAIL];
			cpu_ok = flex_self_test(det, sizeof det);
			if (cpu_ok)
				gpulog(LOG_INFO, thr_id, "flex: CPU self-test OK (%s)", det);
			else
				gpulog(LOG_ERR, thr_id, "flex: CPU SELF-TEST FAILED -- %s", det);
		}

		/* Lane count is sized from the MEAN per-lane CryptoNight cost, not the
		 * maximum. The variant is chosen per nonce, so a batch is a mix:
		 * reserving the largest variant's 2 MiB for every lane leaves ~5 of 6
		 * lanes sitting on memory they never touch, and costs ~2.7x the
		 * achievable batch. flex_pipeline_run() trims the tail on the rare
		 * batch whose mix exceeds the budget.
		 *
		 * The margin covers the sampling spread: the total is a sum of `lanes`
		 * independent draws with sd ~627 KiB, so at these lane counts the
		 * deviation is well under 1%. */
		size_t freeMem = 0, totalMem = 0;
		cudaMemGetInfo(&freeMem, &totalMem);
		const size_t reserve = 900ULL << 20;   /* stage buffers + check + driver */
		size_t budget = (freeMem > reserve) ? (freeMem - reserve) : (freeMem / 2);

		/* ~700 B/lane of ctx + chain state on top of the scratchpad. */
		const size_t per_lane = (size_t)(FLEX_CN_MEM_MEAN * 1.06) + 1024;
		uint32_t throughput = (uint32_t)(budget / per_lane);

		/* Cap the live batch below the VRAM maximum: a launch cannot be preempted,
		 * so a batch running at a job change is wasted. Per arch because the CN
		 * groups run concurrently and the best size differs. -i overrides it. */
		const uint32_t lane_cap = (device_sm[device_map[thr_id]] < 700) ? 4096 : 8192;
		if (gpus_intensity[thr_id] > 0) {
			if (gpus_intensity[thr_id] < throughput)
				throughput = gpus_intensity[thr_id];
		} else if (throughput > lane_cap) {
			throughput = lane_cap;
		}
		throughput = (throughput / 128) * 128;
		if (throughput < 128) throughput = 128;

		/* The scratchpad arena the packer lays groups into. */
		const size_t scratch = (size_t)throughput * (size_t)(FLEX_CN_MEM_MEAN * 1.06);

		gpulog(LOG_INFO, thr_id, "flex: %u lanes, %.0f MiB CryptoNight arena (variable stride, mean %u KiB/lane)",
			throughput, (double)scratch / (1024.0 * 1024.0), FLEX_CN_MEM_MEAN >> 10);

		flex_stage_init_all(thr_id, throughput);
		if (!flex_pipeline_init(&ctx[thr_id], thr_id, throughput, scratch)) {
			gpulog(LOG_ERR, thr_id, "flex: out of memory sizing %u lanes", throughput);
			return -1;
		}
		cuda_check_cpu_init(thr_id, throughput);

		/* Cheap checks by default (an API algo switch pays startup each time);
		 * FLEX_GUARD=1 adds the full set. All fail closed. */
		const int races = flex_guard_races(thr_id, throughput);
		if (!races)
			gpulog(LOG_INFO, thr_id, "flex: core kernels race-free");
		const bool gate_ok = flex_gate(thr_id);
		bool wrong = !cpu_ok || races;

		if (getenv("FLEX_GUARD")) {
			gpulog(LOG_INFO, thr_id, "flex: running full startup guards...");
			const int chain = flex_guard_chain(thr_id, throughput);
			if (!chain)
				gpulog(LOG_INFO, thr_id, "flex: GPU==CPU on the full chain");
			const int filtered = flex_guard_filtered(thr_id, throughput);
			const int width = getenv("FLEX_GUARD_WIDTH") ? flex_guard_width(thr_id, throughput) : 0;
			wrong = wrong || chain || filtered || width;
		}
		/* A CUDA fault in the gate must not excuse a wrong answer found elsewhere. */
		if (wrong)
			s_selftest_cuda_fault = false;
		selftest_gate(thr_id, "flex", gate_ok && !wrong);
		flex_prof_reset();

		init_done[thr_id] = true;
	}

	flex_ctx *c = &ctx[thr_id];

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);
	be32enc(&endiandata[19], pdata[19]);

	if (work_restart[thr_id].restart) {
		*hashes_done = 0;
		return 0;
	}

	cuda_check_cpu_setTarget(ptarget);

	/* ---- the nonce filter: scan a span, chain only the cheap-CN-triple nonces ----
	 *
	 * The filter keeps ~3.7% of nonces, so the span is ~31x the lane count.
	 * Every accepted nonce is ordinary work; the saving is that its three
	 * CryptoNight rounds total 4 iteration units against an unfiltered mean
	 * of 7.388. Consensus-legal, because any nonce is a valid nonce.
	 *
	 * Enabled by default; FLEX_NONCE_FILTER=0 takes the plain path.
	 *
	 * The header MUST be uploaded before the filter runs. Round 0 of the
	 * filtered path runs only the four core[0] stages the filter admits, and
	 * the filter is what establishes that, so a selection made under a stale
	 * header leaves most lanes without a round-0 stage at all. */
	const int flex_use_filter = flex_filter_enabled() ? 1 : 0;
	flex_prof_begin();

	const uint32_t scan_base = pdata[19];
	const uint32_t span = flex_use_filter
		? flex_filter_span(c->threads, c->accept_est)
		: c->threads;               /* disabled: one nonce per lane, no scan */
	uint32_t yield = 0;
	c->span = span;
	if (flex_use_filter) {
		/* Upload this job's header BEFORE filtering. flex_pipeline_run() also
		 * uploads it, but that runs after the filter, and a selection made
		 * under the previous job's seeds breaks round 0's precondition. */
		flex_seed_setBlock_80(endiandata);
		cudaMemset(c->d_count, 0, sizeof(uint32_t));
		flex_filter_cpu(span, pdata[19], c->threads, c->d_nonces, c->d_count);
		cudaMemcpy(&yield, c->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	} else {
		yield = c->threads;
	}

	/* Re-fit the acceptance from the TRUE yield, BEFORE the clamp below throws
	 * the information away -- a clamped yield divided by the span is not a
	 * measurement of acceptance, it is the clamp. An EWMA because the estimate
	 * only needs to track a change in the predicate, not per-batch noise. */
	if (flex_use_filter && span) {
		const double obs = (double)yield / (double)span;
		if (obs > 1e-4 && obs <= 1.0)
			c->accept_est += 0.10 * (obs - c->accept_est);
	}

	/* The kernel keeps counting past the buffer so this can legitimately
	 * exceed the capacity; clamp, and say so rather than silently believing
	 * the span was exactly right. */
	if (yield > c->threads) {
		if (opt_debug)
			applog(LOG_DEBUG, "flex: filter saturated, %u accepted for %u lanes "
			       "(span %u, accept %.5f)", yield, c->threads, span, c->accept_est);
		yield = c->threads;
	} else if (opt_debug && yield < c->threads) {
		applog(LOG_DEBUG, "flex: filter under-yield, %u accepted for %u lanes "
		       "(span %u, accept %.5f)", yield, c->threads, span, c->accept_est);
	}
	if (yield < FLEX_CN_TPB) {
		/* Nothing worth launching for. Still consume the span. */
		pdata[19] += span;
		if (pdata[19] > max_nonce || pdata[19] < first_nonce) pdata[19] = max_nonce;
		bench_cursor[thr_id] = pdata[19];
		*hashes_done = 0;
		return 0;
	}
	yield = (yield / FLEX_CN_TPB) * FLEX_CN_TPB;
	if (flex_use_filter)
		cudaMemcpy(c->h_nonces, c->d_nonces, sizeof(uint32_t) * yield, cudaMemcpyDeviceToHost);

	flex_prof_mark(FLEX_PH_FILTER);
	{
		static int ab = -1;
		if (ab < 0) {
			const char *ev = getenv("CN_MIDBATCH_ABORT");
			ab = ev ? (atoi(ev) != 0) : 1;
		}
		c->abort_on_restart = (ab == 1);
	}
	flex_pipeline_run(c, thr_id, endiandata, pdata, pdata[19], -1,
	                  flex_use_filter ? c->d_nonces : NULL, flex_use_filter ? yield : 0);
	c->abort_on_restart = false;

	/* Abandoned at a job change: nothing was hashed against a live header. */
	if (c->lanes == 0 && work_restart[thr_id].restart) {
		*hashes_done = 0;
		return 0;
	}

	/* The pipeline may have trimmed the batch to fit this mix of CN
	 * variants, so the screen and the nonce cursor must both use the lanes
	 * ACTUALLY hashed. Screening  here would read past the end of
	 * the batch and report nonces that were never computed. */
	const uint32_t hashed = c->lanes;

	if (opt_debug)
		flex_debug_differential(thr_id, pdata, endiandata, hashed, flex_use_filter, scan_base);

	/* The lanes are SCATTERED in nonce space now, so the shared screen --
	 * which reports startNounce + thread -- must be given a base of 0. What it
	 * returns is then a LANE INDEX, and the nonce is h_nonces[lane]. Passing
	 * pdata[19] here would report nonces that were never hashed. */
	const uint32_t lane0 = cuda_check_hash(thr_id, hashed,
	                                       flex_use_filter ? 0 : pdata[19], c->d_hash);
	work->nonces[0] = (!flex_use_filter || lane0 == UINT32_MAX) ? lane0
	                : (lane0 < hashed ? c->h_nonces[lane0] : UINT32_MAX);
	flex_prof_mark(FLEX_PH_SCREEN);
	flex_prof_end(thr_id, hashed);

	static int flex_verify = -1;
	if (flex_verify < 0) {
		const char *ev = getenv("FLEX_VERIFY");
		flex_verify = ev ? atoi(ev) : 0;
	}
	/* The audit rebuilds each lane's nonce as start + lane, which is only
	 * true on the unfiltered path. Under the nonce filter the lanes are scattered, so it is
	 * reported as UNAVAILABLE rather than run against the wrong nonces --
	 * a silently-wrong instrument is worse than an absent one, and its silence
	 * would read as a pass. */
	if (flex_verify) {
		static bool warned = false;
		if (!warned) {
			gpulog(LOG_WARNING, thr_id,
			       "flex: FLEX_VERIFY screen audit is unavailable under the nonce filter "
			       "(scattered nonces); use the chain guard instead");
			warned = true;
		}
	}

	if (work->nonces[0] != UINT32_MAX) {
		uint32_t _ALIGN(64) vhash[8];
		be32enc(&endiandata[19], work->nonces[0]);
		flex_hash(vhash, endiandata);

		if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
			work->valid_nonces = 1;
			work_set_target_ratio(work, vhash);
			/* Every candidate the screen kept, in ONE launch. Under the filter
			 * the cursor below advances past the whole span regardless, so a
			 * candidate not taken here is lost for good. */
			uint32_t cand[MAX_NONCES];
			uint32_t found = 0;
			const uint32_t keep = cuda_check_hash_suppl_all(thr_id, hashed,
						flex_use_filter ? 0 : pdata[19], c->d_hash,
						cand, MAX_NONCES, &found);
			for (uint32_t k = 1; k < keep; k++) {
				/* Lane index again, for the same reason as nonces[0]. */
				const uint32_t n = (!flex_use_filter) ? cand[k]
						 : (cand[k] < hashed ? c->h_nonces[cand[k]] : UINT32_MAX);
				if (n == UINT32_MAX)
					continue;
				be32enc(&endiandata[19], n);
				flex_hash(vhash, endiandata);
				/* Guard every further nonce like the first; a failure is a GPU/CPU mismatch. */
				if (vhash[7] > ptarget[7] || !fulltest(vhash, ptarget)) {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", n);
					continue;
				}
				bn_set_target_ratio(work, vhash, work->valid_nonces);
				work->nonces[work->valid_nonces] = n;
				work->valid_nonces++;
			}

			/* Past the device screen's retention the candidates are gone and the
			 * cursor advances anyway, so they cannot be recovered. Counted so the
			 * loss is observable. */
			if (found > keep && opt_debug)
				gpulog(LOG_DEBUG, thr_id, "flex: %u candidates, %u retrievable",
				       found, keep);

			/* the nonce cursor under a filter: the cursor advances by the SCANNED SPAN,
			 * never by a found nonce. The old max(nonce)+1 form is wrong here in
			 * BOTH directions -- the accepted nonces are unordered, so the max is
			 * not the high-water mark of the scan, and resuming from it would
			 * re-scan most of a span we have already rejected while also skipping
			 * any accepted nonce that happened to sort below it. */
			uint32_t resume = flex_use_filter ? (scan_base + span) : (first_nonce + hashed);
			if (resume > max_nonce || resume < first_nonce)
				resume = max_nonce;
			pdata[19] = resume;
			bench_cursor[thr_id] = resume;

			/* the miner loop does not log benchmark finds */
			if (opt_benchmark && opt_debug)
				for (int k = 0; k < work->valid_nonces; k++)
					gpulog(LOG_DEBUG, thr_id, "flex: benchmark candidate %08x CPU-verified",
					       work->nonces[k]);

			/* CHAINED lanes, not the span. Counting the ~31x scanned nonces
			 * would inflate the displayed rate by that factor -- the rinhash
			 * defect. A scanned-and-rejected nonce produces no digest and so has
			 * no chance of being a share; it is not hashing. */
			*hashes_done = hashed;
			return work->valid_nonces;
		} else {
			gpu_increment_reject(thr_id);
			if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
		}
	}

	pdata[19] = flex_use_filter ? (scan_base + span) : (scan_base + hashed);
	if (pdata[19] > max_nonce || pdata[19] < first_nonce)
		pdata[19] = max_nonce;
	bench_cursor[thr_id] = pdata[19];
	*hashes_done = hashed;      /* chained, not scanned -- see above */
	return 0;
}

extern "C" void free_flex(int thr_id)
{
	if (!init_done[thr_id])
		return;

	cudaDeviceSynchronize();

	flex_pipeline_free(&ctx[thr_id]);
	cryptonight_core_cuda_flex_free(thr_id);
	flex_stage_free_all(thr_id);
	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init_done[thr_id] = false;
}
