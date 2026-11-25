/**
 * Mike algorithm (VKAX / FortuneBlock "mike") -- GPU driver.
 *
 * GhostRider with the core pool cut from 15 algorithms to 11, which shortens
 * the third core group from five rounds to one:
 *
 * ghostrider   5 core, CN, 5 core, CN, 5 core, CN   (18 steps)
 * mike         5 core, CN, 5 core, CN, 1 core, CN   (14 steps)
 *
 * Both the core order (11 distinct algos) and the CN triple (first 3 of 6
 * distinct variants) are derived from header bytes [4,36) and are therefore
 * CONSTANT for an entire job -- the nonce at byte 76 does not affect them --
 * which is what makes a whole-batch GPU pipeline legal.
 *
 * Derived mechanically from algos/ghostrider/ghostrider.cu; the CryptoNight
 * kernels, the core stages, the candidate screen and both startup guards are
 * the same code.  The CPU reference and the known-answer vectors live in
 * algos/mike/mike_hash.cpp + mike_kat.h so they can be gated without building
 * the miner (the project notes
 *
 * WARNING: the core order is a permutation of 0..10 from `nibble % 11`, NOT
 * ghostrider's 15-wide permutation truncated.  See algos/mike/mike.h.
 *
 * Reference: cpuminer-opt algo/mike (pool-confirmed on VKAX + FortuneBlock).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"
#include "algos/mike/mike.h"

// Mike CryptoNight-v1 GPU path (crypto/cryptonight-core.cu, -extra.cu).
extern "C" void cryptonight_core_cuda_gr(int thr_id, int blocks, int threads, int variant, uint32_t stride64,
	uint64_t *d_long_state, uint64_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b,
	uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, uint64_t *d_ctx_tweak);
extern "C" void cryptonight_extra_cpu_prepare_gr(int thr_id, uint32_t threads, uint64_t *d_hash,
	uint64_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b,
	uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, uint64_t *d_ctx_tweak);
extern "C" void cryptonight_extra_cpu_final_gr(int thr_id, uint32_t threads, uint64_t *d_ctx_state,
	uint64_t *d_hash, int zero_high);

// Per-thread device buffers.
static uint32_t *d_hash[MAX_GPUS]      = { 0 };
static uint64_t *d_long_state[MAX_GPUS] = { 0 };
static uint64_t *d_ctx_state[MAX_GPUS] = { 0 };
static uint32_t *d_ctx_a[MAX_GPUS]     = { 0 };
static uint32_t *d_ctx_b[MAX_GPUS]     = { 0 };
static uint32_t *d_ctx_key1[MAX_GPUS]  = { 0 };
static uint32_t *d_ctx_key2[MAX_GPUS]  = { 0 };
static uint64_t *d_ctx_tweak[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { 0 };

// --------------------------------------------------------------------------
// GPU core-round dispatch. The 15 core algos reuse the existing x16 CUDA
// kernels (Mike's algo indices 0..14 match x16r's exactly). The first
// round consumes the 80-byte header; the rest hash the 64-byte d_hash in place.
// ----------------------------------------------------------------------------
static void mike_core_setBlock_80(int algo, int thr_id, uint32_t *endiandata, uint32_t *pdata)
{
	switch (algo) {
	case MIKE_BLAKE:     blake512_cpu_setBlock_80(thr_id, endiandata); break;
	case MIKE_BMW:       bmw512_cpu_setBlock_80(endiandata); break;
	case MIKE_GROESTL:   groestl512_setBlock_80(thr_id, endiandata); break;
	case MIKE_JH:        jh512_setBlock_80(thr_id, endiandata); break;
	case MIKE_KECCAK:    keccak512_setBlock_80(thr_id, endiandata); break;
	case MIKE_SKEIN:     skein512_cpu_setBlock_80((void*)endiandata); break;
	case MIKE_LUFFA:     qubit_luffa512_cpu_setBlock_80((void*)endiandata); break;
	case MIKE_CUBEHASH:  cubehash512_setBlock_80(thr_id, endiandata); break;
	case MIKE_SHAVITE:   x16_shavite512_setBlock_80((void*)endiandata); break;
	case MIKE_SIMD:      x16_simd512_setBlock_80((void*)endiandata); break;
	case MIKE_ECHO:      x16_echo512_setBlock_80((void*)endiandata); break;
	}
}

static void mike_core_hash_80(int algo, int thr_id, uint32_t throughput, uint32_t nonce, uint32_t *d_h)
{
	switch (algo) {
	case MIKE_BLAKE:     blake512_cpu_hash_80(thr_id, throughput, nonce, d_h); break;
	case MIKE_BMW:       bmw512_cpu_hash_80(thr_id, throughput, nonce, d_h, 0); break;
	case MIKE_GROESTL:   groestl512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case MIKE_JH:        jh512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case MIKE_KECCAK:    keccak512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case MIKE_SKEIN:     skein512_cpu_hash_80(thr_id, throughput, nonce, d_h, 1); break;
	case MIKE_LUFFA:     qubit_luffa512_cpu_hash_80(thr_id, throughput, nonce, d_h, 0); break;
	case MIKE_CUBEHASH:  cubehash512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case MIKE_SHAVITE:   x16_shavite512_cpu_hash_80(thr_id, throughput, nonce, d_h, 0); break;
	case MIKE_SIMD:      x16_simd512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case MIKE_ECHO:      x16_echo512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	}
}

static void mike_core_hash_64(int algo, int thr_id, uint32_t throughput, uint32_t nonce, uint32_t *d_h, int order)
{
	switch (algo) {
	case MIKE_BLAKE:     blake512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case MIKE_BMW:       bmw512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case MIKE_GROESTL:   groestl512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case MIKE_JH:        jh512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case MIKE_KECCAK:    keccak512_cpu_hash_64(thr_id, throughput, NULL, d_h); break;
	case MIKE_SKEIN:     skein512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case MIKE_LUFFA:     luffa512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case MIKE_CUBEHASH:  cubehash512_cpu_hash_64(thr_id, throughput, d_h); break;
	case MIKE_SHAVITE:   x11_shavite512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case MIKE_SIMD:      simd512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case MIKE_ECHO:      echo512_cpu_hash_64(thr_id, throughput, d_h); break;
	}
}

// Per-variant scratchpad sizes (bytes): dark, darklite, fast, lite, turtle, turtlelite.
static const uint32_t mike_cn_mem[MIKE_CN_ALGO_COUNT] = { 524288u, 524288u, 2097152u, 1048576u, 262144u, 262144u };

/* CryptoNight iteration units per variant (gr_iters[] / 65536 in
 * algos/cryptonight/cryptonight-core.cu). A batch's cost is
 * threads x sum(units), not threads: a cheap triple has smaller scratchpad
 * slots, so more lanes fit and it does MORE total work. Capping lanes
 * therefore cannot equalise batch duration. */
static const uint32_t mike_cn_units[6] = { 2u, 2u, 4u, 4u, 1u, 1u };

/* Batch work budget in lane-units. A launch cannot be preempted, so a batch
 * still running when a new job arrives was hashed against a retired header
 * and is discarded whole. The budget bounds that exposure by holding batch
 * duration roughly constant across triples. MIKE_WORK_BUDGET overrides it;
 * -i overrides everything. */
#define MIKE_WORK_BUDGET_DEFAULT 24576u
/* Human-readable chain order:
 *   ShbSknFugSmdHamCNDarkLShvLufCubEchBmwCNTrtlLGroBlkWrlJh5KckCNDark
 * Core groups of 5, 5, 1, each closed by its CryptoNight round. Printed once per
 * ORDER CHANGE under -D, not per scanhash call -- the order is derived from
 * header bytes [4,36) and so is constant for a whole job. */
static const char *const mike_core_abbr[11] = { "Blk","Bmw","Gro","Jh5","Kck","Skn","Luf","Cub","Shv","Smd","Ech" };
static const char *const mike_cn_abbr[6]  = { "CNDark","CNDarkL","CNFast","CNLite","CNTrtl","CNTrtlL" };

static void mike_format_order(const uint8_t *core, const uint8_t *cn, char *out, size_t cap)
{
	static const int grp[] = { 5, 5, 1 };
	size_t o = 0;
	int ci = 0;
	out[0] = 0;
	for (size_t g2 = 0; g2 < sizeof(grp)/sizeof(grp[0]); g2++) {
		for (int i = 0; i < grp[g2] && o + 8 < cap; i++, ci++)
			o += (size_t)snprintf(out + o, cap - o, "%s", mike_core_abbr[core[ci]]);
		if (o + 10 < cap)
			o += (size_t)snprintf(out + o, cap - o, "%s", mike_cn_abbr[cn[g2]]);
	}
}

/* Returns 1 when the order differs from the last one logged on this thread. */
static int mike_order_changed(int thr_id, const char *s)
{
	static char last[MAX_GPUS][256];
	if (thr_id < 0 || thr_id >= MAX_GPUS) return 1;
	if (strcmp(last[thr_id], s) == 0) return 0;
	snprintf(last[thr_id], sizeof(last[thr_id]), "%s", s);
	return 1;
}


// Scratchpad budget (bytes) and the thread count the per-thread buffers were
// sized for, established once at init.
static size_t   mike_scratch_bytes[MAX_GPUS]  = { 0 };
static uint32_t mike_max_throughput[MAX_GPUS] = { 0 };

/* CN_MIDBATCH_ABORT=0 disables the job-change checks between CN groups (for
 * A/B); default on. Read once. */
static int mike_midbatch_abort = -1;
static inline bool mike_midbatch_abort_enabled()
{
	if (mike_midbatch_abort < 0) {
		const char *ev = getenv("CN_MIDBATCH_ABORT");
		mike_midbatch_abort = ev ? (atoi(ev) != 0) : 1;
	}
	return mike_midbatch_abort == 1;
}

// One CryptoNight-v1 round: 64-byte d_hash -> 32-byte d_hash (+ zero high 32).
// stride64 = the job's per-thread slot (largest CN variant) in uint64 words.
static void mike_cn_round(int thr_id, int blocks, int threads, int variant, uint32_t stride64, uint32_t *d_h, int zero_high)
{
	const uint32_t throughput = (uint32_t)(blocks * threads);
	cryptonight_extra_cpu_prepare_gr(thr_id, throughput, (uint64_t*)d_h,
		d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id],
		d_ctx_key1[thr_id], d_ctx_key2[thr_id], d_ctx_tweak[thr_id]);
	cryptonight_core_cuda_gr(thr_id, blocks, threads, variant, stride64,
		d_long_state[thr_id], d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id],
		d_ctx_key1[thr_id], d_ctx_key2[thr_id], d_ctx_tweak[thr_id]);
	cryptonight_extra_cpu_final_gr(thr_id, throughput, d_ctx_state[thr_id], (uint64_t*)d_h, zero_high);
}

// ----------------------------------------------------------------------------
// Benchmark rotation sweep.
//
// The CN triple comes from header bytes [4..36), and --benchmark hands every
// algo the same synthetic header (0x55 everywhere), which pins one cheap chain
// for the whole run. Sweep all C(6,3)=20 rotations instead and report the
// arithmetic mean: jobs arrive on a timer, so each rotation gets an equal share
// of wall clock and the mean is the expected live rate.
// ----------------------------------------------------------------------------
#define MIKE_BENCH_ROTATIONS 20
#define MIKE_BENCH_DWELL_SEC 4.0  // measured seconds per rotation (~80s per pass)
#define MIKE_BENCH_MID_ROT   7    // dark+lite+turtle: 7 work units, the mean cost

static const uint8_t mike_bench_combo[MIKE_BENCH_ROTATIONS][3] = {
	{0,1,2},{0,1,3},{0,1,4},{0,1,5},{0,2,3},{0,2,4},{0,2,5},{0,3,4},{0,3,5},{0,4,5},
	{1,2,3},{1,2,4},{1,2,5},{1,3,4},{1,3,5},{1,4,5},{2,3,4},{2,3,5},{2,4,5},{3,4,5}
};

static const char* mike_cn_names[MIKE_CN_ALGO_COUNT] = {
	"dark", "darklite", "fast", "lite", "turtle", "turtlelite"
};

static int    mike_bench_rot[MAX_GPUS]   = { 0 };
static int    mike_bench_pass[MAX_GPUS]  = { 0 };
static bool   mike_bench_warm[MAX_GPUS]  = { 0 };
static double mike_bench_dwell[MAX_GPUS] = { 0 };
static double mike_bench_secs[MAX_GPUS][MIKE_BENCH_ROTATIONS]   = { 0 };
static double mike_bench_hashes[MAX_GPUS][MIKE_BENCH_ROTATIONS] = { 0 };

static double mike_bench_now(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

// Rewrite header bytes [4..36) so the derivation yields exactly `combo` as the
// CN triple, followed by a full 11-algo core order (nothing back-filled). pdata
// is byteswapped word by word into endiandata, so the pattern is stored BE, and
// selectAlgo reads the low nibble of each byte before the high one.
static void mike_bench_set_rotation(uint32_t* pdata, const uint8_t* combo)
{
	uint8_t nib[64], pat[32];
	nib[0] = combo[0]; nib[1] = combo[1]; nib[2] = combo[2];
	for (int i = 3; i < 64; i++)
		nib[i] = (uint8_t)((i - 3) % MIKE_CORE_ALGO_COUNT);
	for (int i = 0; i < 32; i++)
		pat[i] = (uint8_t)(nib[2*i] | (nib[2*i + 1] << 4));
	for (int i = 0; i < 8; i++)
		pdata[1 + i] = be32dec(pat + 4 * i);
}

static void mike_bench_report(int thr_id)
{
	char s[32];
	double sum = 0., lo = 0., hi = 0.;

	gpulog(LOG_BLUE, thr_id, "mike benchmark pass %d - %d chain rotations, %.0fs each",
		mike_bench_pass[thr_id], MIKE_BENCH_ROTATIONS, (double)MIKE_BENCH_DWELL_SEC);
	for (int r = 0; r < MIKE_BENCH_ROTATIONS; r++) {
		const uint8_t* c = mike_bench_combo[r];
		double rate = (mike_bench_secs[thr_id][r] > 0.)
			? mike_bench_hashes[thr_id][r] / mike_bench_secs[thr_id][r] : 0.;
		sum += rate;
		if (r == 0 || rate < lo) lo = rate;
		if (rate > hi) hi = rate;
		gpulog(LOG_INFO, thr_id, "  %-10s %-10s %-10s : %9.2f kH/s",
			mike_cn_names[c[0]], mike_cn_names[c[1]], mike_cn_names[c[2]], rate / 1024.);
	}
	format_hashrate(sum / MIKE_BENCH_ROTATIONS, s);
	gpulog(LOG_NOTICE, thr_id, "mike average = %s (worst %.2f, best %.2f kH/s)",
		s, lo / 1024., hi / 1024.);
}

// Charge one batch to the current rotation and advance the sweep. Totals are
// cumulative across passes so the average keeps converging.
static void mike_bench_account(int thr_id, uint32_t hashes, double secs)
{
	const int r = mike_bench_rot[thr_id];

	if (!mike_bench_warm[thr_id]) { // first batch of a rotation carries the switch cost
		mike_bench_warm[thr_id] = true;
		return;
	}
	mike_bench_secs[thr_id][r]   += secs;
	mike_bench_hashes[thr_id][r] += (double)hashes;
	mike_bench_dwell[thr_id]     += secs;
	if (mike_bench_dwell[thr_id] < MIKE_BENCH_DWELL_SEC)
		return;

	mike_bench_dwell[thr_id] = 0.;
	mike_bench_warm[thr_id] = false;
	if (++mike_bench_rot[thr_id] >= MIKE_BENCH_ROTATIONS) {
		mike_bench_rot[thr_id] = 0;
		mike_bench_pass[thr_id]++;
		mike_bench_report(thr_id);
	}
}

// ----------------------------------------------------------------------------
// Audit the candidate screen for missed nonces.
//
// The host re-verify only sees nonces the screen reported, so a screen that
// misses one produces no reject and no failed verify. No re-hashing is needed to
// check: the GPU has already written every digest, so copy them back and apply
// the same compare the kernel uses. Indexes hash[thread << 4] exactly as
// cuda_checkhash_64 does, so this audits the screen's logic, not the layout.
//
// MIKE_VERIFY=1 audits every batch; MIKE_VERIFY=2 also corrupts one host-side digest
// to force a miss, so the gate itself can be shown to fire. 3/4: see below.
// ----------------------------------------------------------------------------
static uint32_t *mike_audit_buf[MAX_GPUS] = { 0 };
static uint32_t  mike_audit_cap[MAX_GPUS] = { 0 };

static bool mike_host_below(const uint32_t *h, const uint32_t *t)
{
	for (int i = 7; i >= 0; i--) {
		if (h[i] > t[i]) return false;
		if (h[i] < t[i]) return true;
	}
	return true;
}

static void mike_screen_audit(int thr_id, uint32_t throughput, uint32_t start_nonce,
	uint32_t *dh, const uint32_t *ptarget, uint32_t reported, int mode)
{
	if (mike_audit_cap[thr_id] < throughput) {
		free(mike_audit_buf[thr_id]);
		mike_audit_buf[thr_id] = (uint32_t*) malloc((size_t)throughput * 64);
		if (!mike_audit_buf[thr_id]) { mike_audit_cap[thr_id] = 0; return; }
		mike_audit_cap[thr_id] = throughput;
	}
	uint32_t *hb = mike_audit_buf[thr_id];
	cudaMemcpy(hb, dh, (size_t)throughput * 64, cudaMemcpyDeviceToHost);

	if (mode == 2) hb[7] = 0; // negative control: lane 0 becomes unmissable

	uint32_t hostcnt = 0, hostfirst = UINT32_MAX;
	bool sawreported = (reported == UINT32_MAX);
	for (uint32_t t = 0; t < throughput; t++) {
		if (mike_host_below(&hb[t * 16], ptarget)) {
			if (!hostcnt) hostfirst = start_nonce + t;
			if (start_nonce + t == reported) sawreported = true;
			hostcnt++;
		}
	}

	// Make the screen state its own count over the same range.
	cuda_check_hash_suppl(thr_id, throughput, start_nonce, dh, 1);
	const uint32_t gpucnt = cuda_check_hash_count(thr_id);
	const char *ctl = (mode == 2) ? "  [negative control armed - a MISS here is EXPECTED]" : "";

	if (gpucnt != hostcnt)
		gpulog(LOG_ERR, thr_id, "mike screen/host MISS: screen counted %u, host counted %u over [%08x,+%u) first=%08x%s",
			gpucnt, hostcnt, start_nonce, throughput, hostfirst, ctl);
	else if (!sawreported)
		gpulog(LOG_ERR, thr_id, "mike screen/host: screen reported %08x which the host does not find%s", reported, ctl);
	else if (opt_debug)
		gpulog(LOG_DEBUG, thr_id, "mike screen/host ok: %u candidates over [%08x,+%u)", hostcnt, start_nonce, throughput);
}

// MIKE_VERIFY=3: once per header, re-hash MIKE_VERIFY_LANES (default 64) of the
// copied-back digests on the CPU: lanes 0, 1, 127, 128, the last, then random.
// MIKE_VERIFY=4 flips one sampled digest first, which must be reported.
static void mike_hash_audit(int thr_id, uint32_t throughput, uint32_t start_nonce,
	const uint32_t *endiandata, int mode)
{
	static uint32_t last_sig[MAX_GPUS];
	uint32_t sig = 2166136261u;
	for (int i = 0; i < 19; i++) sig = (sig ^ endiandata[i]) * 16777619u;
	if (sig == last_sig[thr_id] || !mike_audit_buf[thr_id]) return;
	last_sig[thr_id] = sig;

	uint32_t want = 64;
	const char *ev = getenv("MIKE_VERIFY_LANES");
	if (ev && atoi(ev) > 0) want = (uint32_t) atoi(ev);
	if (want > throughput) want = throughput;

	uint32_t *hb = mike_audit_buf[thr_id];
	uint32_t lanes[1024];
	uint32_t n = 0;
	const uint32_t fixed[5] = { 0, 1, 127, 128, throughput - 1 };
	for (int i = 0; i < 5 && n < want; i++)
		if (fixed[i] < throughput) lanes[n++] = fixed[i];
	uint32_t x = start_nonce ^ sig;
	while (n < want && n < 1024) {
		x = x * 1664525u + 1013904223u;
		lanes[n++] = x % throughput;
	}

	const uint32_t flip = (mode >= 4 && n > 2) ? lanes[2] : UINT32_MAX;
	if (flip != UINT32_MAX) hb[flip * 16 + 3] ^= 0x10u;

	uint32_t _ALIGN(64) ed[20], chash[8];
	memcpy(ed, endiandata, 76);
	uint32_t bad = 0, firstbad = UINT32_MAX;
	for (uint32_t i = 0; i < n; i++) {
		const uint32_t t = lanes[i];
		be32enc(&ed[19], start_nonce + t);
		mike_hash(chash, ed);
		if (memcmp(&hb[t * 16], chash, 32)) {
			if (!bad) firstbad = t;
			bad++;
		}
	}
	const char *ctl = (flip != UINT32_MAX) ? "  [negative control armed - lane flipped]" : "";
	if (bad)
		gpulog(LOG_ERR, thr_id, "mike GPU != CPU on %u of %u lanes over [%08x,+%u), first lane %u (nonce %08x)%s",
			bad, n, start_nonce, throughput, firstbad, start_nonce + firstbad, ctl);
	else
		gpulog(LOG_INFO, thr_id, "mike digest differential ok: %u of %u lanes match the CPU over [%08x,+%u), tail lane %u",
			n, n, start_nonce, throughput, throughput - 1);
}

extern "C" int scanhash_mike(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	uint32_t* pdata = work->data;
	uint32_t* ptarget = work->target;

	/* --benchmark resets the nonce every call; carry it so batches stay fresh */
	static uint32_t bench_cursor[MAX_GPUS];
	if (opt_benchmark) {
		pdata[19] = bench_cursor[thr_id];
		max_nonce = UINT32_MAX;
	}

	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	uint32_t _ALIGN(64) endiandata[20];

	// Sized so the CPU re-verify actually fires at this algo's kH/s rates; the
	// usual 0x00ff screen yields roughly one candidate every few hours, i.e. no
	// gate at all. MIKE_BENCH_TARGET overrides it for benchmark experiments.
	if (opt_benchmark) {
		const char* env = getenv("MIKE_BENCH_TARGET");
		ptarget[7] = env ? (uint32_t) strtoul(env, NULL, 0) : 0x0003ffff;
	}

	// A benchmark header pins one CN rotation, so drive the rotation ourselves.
	// -a all has no time for a sweep (3 loops per algo), so it gets the single
	// mean-cost rotation instead of the near-best one the 0x55 header selects.
	const bool mike_bench = opt_benchmark;
	const bool mike_sweep = (opt_benchmark && bench_algo < 0);
	if (mike_bench)
		mike_bench_set_rotation(pdata, mike_bench_combo[mike_sweep ? mike_bench_rot[thr_id] : MIKE_BENCH_MID_ROT]);

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		{
			char det[MIKE_SELFTEST_DETAIL];
			if (mike_self_test(det, sizeof det))
				gpulog(LOG_INFO, thr_id, "mike: CPU self-test OK (%s)", det);
			else
				gpulog(LOG_ERR, thr_id, "mike: CPU SELF-TEST FAILED -- %s", det);
		}

		// Fixed throughput sized from free VRAM at the worst-case 2 MiB/thread, so
		// the kernels are init and run at the SAME thread count (init==run, like
		// the smallest CN variant (256 KiB). Per job we pick stride = the job's
		// largest CN variant and threads = budget/stride, so light jobs (turtle/
		// dark) run far more threads than heavy (fast/lite) ones.
		size_t freeMem = 0, totalMem = 0;
		cudaMemGetInfo(&freeMem, &totalMem);
		size_t reserve = 768ULL << 20; // headroom for ctx/core/check buffers + driver
		size_t budget = (freeMem > reserve) ? (freeMem - reserve) : (freeMem / 2);
		budget = (budget / 2097152) * 2097152; // whole 2 MiB slots
		mike_scratch_bytes[thr_id] = budget;

		uint32_t max_throughput = (uint32_t) min((size_t)(budget / 262144u), (size_t)(1U << 18));

		/* Upper bound on lanes, kept as a backstop. The binding limit is the
		 * per-job WORK budget applied in scanhash; see mike_cn_units[].
		 * MIKE_LANE_CAP overrides this; -i overrides everything. */
		uint32_t lane_cap = 16384;
		{
			const char *ev = getenv("MIKE_LANE_CAP");
			if (ev) {
				const unsigned long v = strtoul(ev, NULL, 0);
				if (v >= 128) lane_cap = (uint32_t)v;
			}
		}
		if (gpus_intensity[thr_id] > 0) {
			if (gpus_intensity[thr_id] < max_throughput)
				max_throughput = gpus_intensity[thr_id]; // -i N caps thread count
		} else if (max_throughput > lane_cap) {
			max_throughput = lane_cap;
		}
		max_throughput = (max_throughput / 128) * 128;
		if (max_throughput < 128) max_throughput = 128;
		mike_max_throughput[thr_id] = max_throughput;

		gpulog(LOG_INFO, thr_id, "mike: %.0f MiB scratchpad, up to %u threads",
			(double)budget / (1024*1024), max_throughput);

		blake512_cpu_init(thr_id, max_throughput);
		bmw512_cpu_init(thr_id, max_throughput);
		groestl512_cpu_init(thr_id, max_throughput);
		skein512_cpu_init(thr_id, max_throughput);
		jh512_cpu_init(thr_id, max_throughput);
		keccak512_cpu_init(thr_id, max_throughput);
		qubit_luffa512_cpu_init(thr_id, max_throughput);
		luffa512_cpu_init(thr_id, max_throughput);
		x11_shavite512_cpu_init(thr_id, max_throughput);
		simd512_cpu_init(thr_id, max_throughput);
		x11_echo512_cpu_init(thr_id, max_throughput);
		x16_echo512_cpu_init(thr_id, max_throughput);     // needed by x16_echo512_cuda_hash_80

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t)64 * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_long_state[thr_id], budget), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_state[thr_id], (size_t)26 * sizeof(uint64_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_a[thr_id], (size_t)4 * sizeof(uint32_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_b[thr_id], (size_t)4 * sizeof(uint32_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_key1[thr_id], (size_t)40 * sizeof(uint32_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_key2[thr_id], (size_t)40 * sizeof(uint32_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_tweak[thr_id], (size_t)sizeof(uint64_t) * max_throughput), -1);

		cuda_check_cpu_init(thr_id, max_throughput);

		gpulog(LOG_INFO, thr_id, "mike: startup self-test (races%s)",
			getenv("MIKE_GUARD") ? " + full chain, ~10s" : "; MIKE_GUARD=1 adds the chain check");

		// Guard 1 (races): every core kernel must be self-consistent at full
		// throughput. Variation across repeats => an intra-kernel race (e.g. a
		// shared AES table read before it is cooperatively filled).
		{
			uint8_t tv[80];
			for (int i = 0; i < 80; i++) tv[i] = (uint8_t)(i * 13 + 5);
			const uint32_t tp = max_throughput;
			int races = 0;
			for (int a = 0; a < MIKE_CORE_ALGO_COUNT; a++) {
				uint8_t first[64], g[64]; int v64 = 0, v80 = 0;
				for (int rep = 0; rep < 8; rep++) {
					cudaMemcpy(d_hash[thr_id], tv, 64, cudaMemcpyHostToDevice);
					mike_core_hash_64(a, thr_id, tp, 0, d_hash[thr_id], 0);
					cudaMemcpy(g, d_hash[thr_id], 64, cudaMemcpyDeviceToHost);
					if (rep == 0) memcpy(first, g, 64); else if (memcmp(g, first, 64)) v64++;
				}
				mike_core_setBlock_80(a, thr_id, (uint32_t*)tv, (uint32_t*)tv);
				for (int rep = 0; rep < 8; rep++) {
					mike_core_hash_80(a, thr_id, tp, 0, d_hash[thr_id]);
					cudaMemcpy(g, d_hash[thr_id], 64, cudaMemcpyDeviceToHost);
					if (rep == 0) memcpy(first, g, 64); else if (memcmp(g, first, 64)) v80++;
				}
				if (v64 || v80) { races++; gpulog(LOG_ERR, thr_id, "mike: RACE in algo %d (hash64=%d hash80=%d)", a, v64, v80); }
			}
			if (!races) gpulog(LOG_INFO, thr_id, "mike: kernels race-free");
		}

		// Guard 2 (chain logic): verify the full GPU pipeline against the CPU
		// reference across several header orders (order derivation, CN interleaving,
		// first-round init).
		//
		// Opt-in via MIKE_GUARD=1: it is most of startup, and an API-driven
		// algo switch pays it on every change. Consensus is unaffected --
		// every candidate is host re-verified before submit.
		if (getenv("MIKE_GUARD")) {
			uint32_t lcg = 0x12345678u;
			int diffs = 0;
			const int NTEST = 6;
			for (int t = 0; t < NTEST; t++) {
				uint32_t pd[20], edata[20];
				for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
				for (int k = 0; k < 20; k++) be32enc(&edata[k], pd[k]); // mirror scanhash

				uint8_t cOrd[MIKE_CORE_ALGO_COUNT], nOrd[MIKE_CN_ALGO_COUNT];
				mike_get_algo_string(&edata[1], 64, cOrd, MIKE_CORE_ALGO_COUNT);
				mike_get_algo_string(&edata[1], 64, nOrd, MIKE_CN_ALGO_COUNT);
				uint32_t vmx = 0;
				for (int g = 0; g < 3; g++) vmx = max(vmx, mike_cn_mem[nOrd[g]]);
				uint32_t s64 = vmx >> 3;
				// This guard validates chain logic (order derivation, CN interleaving),
				// not kernel races (covered separately), so run at a small fixed thread
				// count to keep startup fast regardless of the job's variant mix.
				uint32_t tp = 512;
				uint32_t bl = tp / 128;

				uint32_t *dh = d_hash[thr_id];
				const int seq[14]    = { 1,0,0,0,0, 2, 0,0,0,0,0, 2, 0, 3 };
				const int cnstep[14] = { 0,0,0,0,0, 0, 0,0,0,0,0, 1, 0, 2 };
				uint8_t cbuf[64] = {0}, tmp[64], gbuf[64];
				be32enc(&edata[19], pd[19]);
				mike_core_setBlock_80(cOrd[0], thr_id, edata, pd);
				int od = 0, ci = 0;
				bool orderDiff = false;
				for (int s = 0; s < 14; s++) {
					int algo;
					if (seq[s] == 1) { algo = cOrd[ci++]; mike_core_hash_80(algo, thr_id, tp, pd[19], dh);
						mike_core_algo_cpu(algo, edata, cbuf, 80); }
					else if (seq[s] == 0) { algo = cOrd[ci++]; mike_core_hash_64(algo, thr_id, tp, pd[19], dh, od++);
						mike_core_algo_cpu(algo, cbuf, tmp, 64); memcpy(cbuf, tmp, 64); }
					else { algo = nOrd[cnstep[s]]; mike_cn_round(thr_id, bl, 128, algo, s64, dh, seq[s] == 2 ? 1 : 0);
						mike_cn_algo_cpu(algo, cbuf, tmp, 64); memcpy(cbuf, tmp, 32); if (seq[s] == 2) memset(cbuf + 32, 0, 32); }
					int cl = (seq[s] == 3) ? 32 : 64;
					cudaMemcpy(gbuf, dh, cl, cudaMemcpyDeviceToHost);
					if (memcmp(gbuf, cbuf, cl) != 0) {
						gpulog(LOG_ERR, thr_id, "mike DIFF t=%d tp=%u stage=%d algo=%d core=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d cn=%d,%d,%d",
							t, tp, s, algo, cOrd[0],cOrd[1],cOrd[2],cOrd[3],cOrd[4],cOrd[5],
							cOrd[6],cOrd[7],cOrd[8],cOrd[9],cOrd[10], nOrd[0],nOrd[1],nOrd[2]);
						orderDiff = true; break;
					}
				}
				if (orderDiff) { diffs++; if (diffs >= 8) break; }
			}
			if (diffs == 0) gpulog(LOG_INFO, thr_id, "mike: GPU==CPU on %d random orders", NTEST);
			else gpulog(LOG_ERR, thr_id, "mike: %d/%d orders DIFF", diffs, NTEST);
		}

		gpulog(LOG_INFO, thr_id, "mike: self-test complete, starting mining");

		if (mike_sweep)
			gpulog(LOG_INFO, thr_id, "mike: benchmark sweeps %d chain rotations, %.0fs each",
				MIKE_BENCH_ROTATIONS, (double)MIKE_BENCH_DWELL_SEC);
		else if (mike_bench) {
			const uint8_t* c = mike_bench_combo[MIKE_BENCH_MID_ROT];
			gpulog(LOG_WARNING, thr_id, "mike: -a all times ONE rotation (%s+%s+%s); use --benchmark -a mike for the %d-rotation average",
				mike_cn_names[c[0]], mike_cn_names[c[1]], mike_cn_names[c[2]], MIKE_BENCH_ROTATIONS);
		}

		init[thr_id] = true;
	}

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	// Order is constant across the nonce batch (depends only on header [4..36)).
	uint8_t coreOrder[MIKE_CORE_ALGO_COUNT];
	uint8_t cnOrder[MIKE_CN_ALGO_COUNT];
	mike_get_algo_string(&endiandata[1], 64, coreOrder, MIKE_CORE_ALGO_COUNT);
	mike_get_algo_string(&endiandata[1], 64, cnOrder, MIKE_CN_ALGO_COUNT);

	if (opt_debug) {
		char ord[256];
		mike_format_order(coreOrder, cnOrder, ord, sizeof(ord));
		if (mike_order_changed(thr_id, ord))
			gpulog(LOG_DEBUG, thr_id, "mike: hash order: %s", ord);
	}

	// Pack threads for this job: per-thread slot = the job's largest CN variant.
	uint32_t vmax = 0;
	for (int g = 0; g < 3; g++)
		vmax = max(vmax, mike_cn_mem[cnOrder[g]]);
	const uint32_t stride64 = vmax >> 3;
	const uint32_t threads = 128;
	uint32_t throughput = (uint32_t) min((size_t)(mike_scratch_bytes[thr_id] / vmax), (size_t)mike_max_throughput[thr_id]);
	throughput = (throughput / threads) * threads;
	if (throughput < threads) throughput = threads;

	/* Clamp by WORK, not by lanes: sum the job's CN units and give the batch a
	 * fixed unit budget, so a cheap triple gets proportionally more lanes and
	 * every job lands at roughly the same wall-clock duration. */
	uint32_t cn_units = 0;
	for (int g = 0; g < 3; g++) cn_units += mike_cn_units[cnOrder[g]];
	if (!cn_units) cn_units = 1;
	uint32_t work_budget = MIKE_WORK_BUDGET_DEFAULT;
	{
		const char *ev = getenv("MIKE_WORK_BUDGET");
		if (ev) {
			const unsigned long v = strtoul(ev, NULL, 0);
			if (v >= 1024) work_budget = (uint32_t)v;
		}
	}
	{
		uint32_t cap = work_budget / cn_units;
		cap = (cap / threads) * threads;
		if (cap < threads) cap = threads;
		if (gpus_intensity[thr_id] == 0 && throughput > cap)
			throughput = cap;
	}
	const uint32_t blocks = throughput / threads;

	/* The cap acts on THROUGHPUT, and throughput is min(scratch/vmax, cap):
	 * a triple containing fast or lite is already below the cap, so the cap
	 * binds only on all-cheap triples. Logged because the CN triple is fixed
	 * per job and spans ~2.5x in work. */
	if (opt_debug)
		gpulog(LOG_DEBUG, thr_id, "mike: job cn %d,%d,%d = %u units, vmax %u KiB -> %u threads = %u of %u units",
			cnOrder[0], cnOrder[1], cnOrder[2], cn_units, vmax >> 10,
			throughput, throughput * cn_units, work_budget);

	// Bail out before launching the 18-kernel pipeline if the job changed or the
	// miner is shutting down (avoids wasted work and a teardown-race CUDA error).
	if (work_restart[thr_id].restart) {
		*hashes_done = 0;
		return 0;
	}

	const double mike_t0 = mike_sweep ? mike_bench_now() : 0.;

	mike_core_setBlock_80(coreOrder[0], thr_id, endiandata, pdata);
	cuda_check_cpu_setTarget(ptarget);

	// One batch per call: the GPU pipeline dominates, and per-call setup (order
	// derive + setBlock + setTarget) is microseconds, so returning each batch lets
	// the hashrate meter update without measurable overhead.
	uint32_t *dh = d_hash[thr_id];
	int order = 0;

	// Group 1
	mike_core_hash_80(coreOrder[0], thr_id, throughput, pdata[19], dh);
	mike_core_hash_64(coreOrder[1], thr_id, throughput, pdata[19], dh, order++);
	mike_core_hash_64(coreOrder[2], thr_id, throughput, pdata[19], dh, order++);
	mike_core_hash_64(coreOrder[3], thr_id, throughput, pdata[19], dh, order++);
	mike_core_hash_64(coreOrder[4], thr_id, throughput, pdata[19], dh, order++);
	mike_cn_round(thr_id, blocks, threads, cnOrder[0], stride64, dh, 1);

	/* A job change mid-batch retires the header, so stop enqueuing. The sync
	 * makes the check timely; the batch is otherwise queued within milliseconds. */
	if (mike_midbatch_abort_enabled()) {
		cudaDeviceSynchronize();
		if (work_restart[thr_id].restart) {
			*hashes_done = 0;
			return 0;
		}
	}

	// Group 2
	mike_core_hash_64(coreOrder[5], thr_id, throughput, pdata[19], dh, order++);
	mike_core_hash_64(coreOrder[6], thr_id, throughput, pdata[19], dh, order++);
	mike_core_hash_64(coreOrder[7], thr_id, throughput, pdata[19], dh, order++);
	mike_core_hash_64(coreOrder[8], thr_id, throughput, pdata[19], dh, order++);
	mike_core_hash_64(coreOrder[9], thr_id, throughput, pdata[19], dh, order++);
	mike_cn_round(thr_id, blocks, threads, cnOrder[1], stride64, dh, 1);

	/* Same mid-batch job-change check as after group 1. */
	if (mike_midbatch_abort_enabled()) {
		cudaDeviceSynchronize();
		if (work_restart[thr_id].restart) {
			*hashes_done = 0;
			return 0;
		}
	}

	// Group 3 -- ONE core round; the 11-wide pool ends at index 10.
	mike_core_hash_64(coreOrder[10], thr_id, throughput, pdata[19], dh, order++);
	mike_cn_round(thr_id, blocks, threads, cnOrder[2], stride64, dh, 0);

	// One-time GPU vs CPU correctness check on the batch's first nonce.
	static bool mike_gpu_checked = false;
	if (!mike_gpu_checked) {
		mike_gpu_checked = true;
		uint32_t _ALIGN(64) ghash[8], chash[8], ed[20];
		cudaMemcpy(ghash, dh, 32, cudaMemcpyDeviceToHost);
		memcpy(ed, endiandata, 80);
		be32enc(&ed[19], pdata[19]);
		mike_hash(chash, ed);
		if (memcmp(ghash, chash, 32) == 0)
			gpulog(LOG_INFO, thr_id, "mike: GPU matches CPU reference");
		else
			gpulog(LOG_ERR, thr_id, "mike: GPU != CPU! gpu[0]=%08x cpu[0]=%08x", ghash[0], chash[0]);
	}

	work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], dh);

	// Timed here: cuda_check_hash has synced, and the CPU re-verify below (3 CN
	// rounds) stays outside the window so a benchmark candidate cannot skew it.
	if (mike_sweep)
		mike_bench_account(thr_id, throughput, mike_bench_now() - mike_t0);

	static int mike_verify = -1;
	if (mike_verify < 0) {
		const char* ev = getenv("MIKE_VERIFY");
		mike_verify = ev ? atoi(ev) : 0;
	}
	if (mike_verify)
		mike_screen_audit(thr_id, throughput, pdata[19], dh, ptarget, work->nonces[0], mike_verify);
	if (mike_verify >= 3)
		mike_hash_audit(thr_id, throughput, pdata[19], endiandata, mike_verify);

	if (work->nonces[0] != UINT32_MAX) {
		uint32_t _ALIGN(64) vhash[8];
		be32enc(&endiandata[19], work->nonces[0]);
		mike_hash(vhash, endiandata);

		if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
			work->valid_nonces = 1;
			work_set_target_ratio(work, vhash);
			/* Every candidate of the launch, each host re-verified. The screen is exact,
			 * so a failure is a GPU/CPU mismatch and is counted like the first. */
			uint32_t cand[MAX_NONCES];
			uint32_t found = 0;
			const uint32_t keep = cuda_check_hash_suppl_all(thr_id, throughput,
						pdata[19], dh, cand, MAX_NONCES, &found);
			uint32_t hi = work->nonces[0];
			for (uint32_t k = 1; k < keep; k++) {
				const uint32_t n = cand[k];
				if (n > hi) hi = n;          /* highest RETURNED, accepted or not */
				be32enc(&endiandata[19], n);
				mike_hash(vhash, endiandata);
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

			/* The screen covered [first_nonce, first_nonce + throughput), so resume
			 * past the whole batch once every kept candidate has been returned. If
			 * the screen kept more than work can carry, resume just above the
			 * highest returned one so the rest are re-found next pass. */
			uint32_t resume = (keep > 0 && found <= keep)
				? first_nonce + throughput
				: hi + 1;
			if (resume > max_nonce || resume < first_nonce)
				resume = max_nonce;
			pdata[19] = resume;
			bench_cursor[thr_id] = resume;

			/* the miner loop does not log benchmark finds */
			if (opt_benchmark && opt_debug)
				for (int k = 0; k < work->valid_nonces; k++)
					gpulog(LOG_DEBUG, thr_id, "benchmark candidate %08x CPU-verified", work->nonces[k]);

			*hashes_done = pdata[19] - first_nonce;
			return work->valid_nonces;
		} else {
			gpu_increment_reject(thr_id);
			if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
		}
	}

	pdata[19] += throughput;
	if (pdata[19] > max_nonce || pdata[19] < first_nonce)
		pdata[19] = max_nonce;
	bench_cursor[thr_id] = pdata[19];
	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_mike(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_long_state[thr_id]);
	cudaFree(d_ctx_state[thr_id]);
	cudaFree(d_ctx_a[thr_id]);
	cudaFree(d_ctx_b[thr_id]);
	cudaFree(d_ctx_key1[thr_id]);
	cudaFree(d_ctx_key2[thr_id]);
	cudaFree(d_ctx_tweak[thr_id]);

	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);
	simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
