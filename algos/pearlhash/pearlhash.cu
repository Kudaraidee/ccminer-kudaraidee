// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * pearlhash: Pearl (PRL) noisy int8 GEMM proof of work, pool mode. See README.md.
 *
 * A = 0; B^T = 0 except its first 1024-byte chunk, which carries the trial (the
 * nonce). Every reported tile is re-verified on the CPU before its PlainProof is
 * built. The hashrate is in int8 MAC/s.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>	/* usleep (compat/unistd.h on Windows) */
#include <vector>

#include "miner.h"
#include "cuda/helper.cuh"
#include "cuda/selftest_gate.cuh"
#include "algos/pearlhash/pearl_device.cuh"
#include "algos/pearlhash/pearl_proof.h"

#define PEARL_M		8192
#define PEARL_N		8192
#define PEARL_K		2048
#define PEARL_R		128
/* fused geometry (sm_80+): k at the consensus maximum, A = B = 64 in every mined entry, and A
 * committed with one extra row (row PEARL_M) whose first chunk carries the nonce; that row is
 * never mined, so the mined operands are pure 64 + noise */
#define PEARL_FK		65536
#define PEARL_FILL		64
#define PEARL_CALL_SECONDS	1.0

/* ------------------------------------------------------------ proof stash */
/* struct work is copied by value to the submit thread, so it carries a stash id
 * in nonces[] instead of the ~45 KB proof. */
#define PEARL_STASH	64
static pthread_mutex_t stash_lock = PTHREAD_MUTEX_INITIALIZER;
static char *stash_proof[PEARL_STASH];
static uint32_t stash_id[PEARL_STASH];
static uint32_t stash_next = 1;

extern "C" uint32_t pearl_proof_put(char *b64)
{
	pthread_mutex_lock(&stash_lock);
	uint32_t id = stash_next++;
	if (!stash_next) stash_next = 1;
	const uint32_t s = id % PEARL_STASH;
	free(stash_proof[s]);
	stash_proof[s] = b64;
	stash_id[s] = id;
	pthread_mutex_unlock(&stash_lock);
	return id;
}

extern "C" char *pearl_proof_take(uint32_t id)
{
	char *p = NULL;
	pthread_mutex_lock(&stash_lock);
	const uint32_t s = id % PEARL_STASH;
	if (stash_id[s] == id && stash_proof[s]) {
		p = stash_proof[s];
		stash_proof[s] = NULL;
		stash_id[s] = 0;
	}
	pthread_mutex_unlock(&stash_lock);
	return p;
}

/* ------------------------------------------------------------ per-GPU state */
#define PEARL_HITS_BYTES	((1 + 2 * PEARL_MAX_HITS) * 4)

/* one queued trial (scanhash keeps one trial ahead of the one it waits for) */
struct pearl_slot {
	uint32_t trial;
	uint8_t as[32], bs[32];
	uint8_t chunk[BLAKE3_CHUNK_LEN];
	uint32_t *hits;				/* mapped pinned: the GEMM writes it directly */
	uint32_t *dhits;			/* its device alias */
	cudaEvent_t done;
};

struct pearl_state {
	bool imma;
	const uint32_t *rows, *cols;
	uint32_t h, w;				/* pattern sizes: 8 x 8 dp4a, 8 x 16 IMMA */
	bool dp4a_local;			/* dp4a jackpot window in local memory (sm_6x) */
	pearl_config cfg;
	bool fused;				/* sm_80+: operands built inside the GEMM */
	uint32_t m, mc, n;			/* mined rows, committed A rows, B^T rows */
	uint32_t units;				/* hashes_done units (2^20 MAC) per trial */
	size_t nonce_leaf;			/* fused: leaf of A's tree holding the nonce chunk */
	uint32_t *dU, *dP, *dApan, *dBpan, *dHits;	/* dp4a path; dHits: self-test only */
	int8_t *dUA, *dUB;			/* fused: E of A and of B^T */
	uint32_t *dPA, *dPB;			/* fused: (p0, p1) per k */
	uint2 *dO2;				/* fused: packed smem offsets */
	uint8_t *dFill;				/* fused: one 1024-byte chunk of PEARL_FILL */
	uint32_t *dTree0, *dTree1;		/* fused: per-job tree layers, ping-pong */
	pearl_slot slot[2];
	double t_host, t_gpu, t_hit;		/* -D timing: seconds since the last log */
	uint32_t t_trials;
	time_t t_last;
	uint8_t *A, *Bt;			/* dp4a host matrices: A = 0, B^T = 0 but chunk 0 (NULL when fused) */
	pearl_merkle ta, tb;
	bool have_job;
	uint8_t header[PEARL_HEADER_LEN];
	uint8_t job_key[32];
	uint64_t salt;
	uint32_t bench_trial;			/* --benchmark's own cursor, see scanhash */
};
static pearl_state st[MAX_GPUS];
static bool init_done[MAX_GPUS];
static bool sched_set[MAX_GPUS];

/* (trial, salt, gpu) as 6-bit digits, each byte a value in [-32, 31] */
static void pearl_nonce_chunk(uint8_t *chunk, uint32_t trial, uint64_t salt, int thr_id)
{
	memset(chunk, 0, BLAKE3_CHUNK_LEN);
	int p = 0;
	for (int i = 0; i < 6; i++) chunk[p++] = (uint8_t)(int8_t)((int)((trial >> (6 * i)) & 63) - 32);
	for (int i = 0; i < 11; i++) chunk[p++] = (uint8_t)(int8_t)((int)((salt >> (6 * i)) & 63) - 32);
	for (int i = 0; i < 2; i++) chunk[p++] = (uint8_t)(int8_t)((int)(((uint32_t)thr_id >> (6 * i)) & 63) - 32);
}

/* One trial on the GPU from given seeds. hits must be device-visible (mapped host memory in the
 * miner); the trial clears it. dbg may be NULL.
 * Fused: noise and offset tables, then the fused GEMM (A = B = 64; chunk_b is not used).
 * dp4a: noise, prepass, kernel; B^T is zero except the 1024-byte chunk_b (host bytes, passed as
 * kernel parameters) at the start of row 0. */
static void pearl_gpu_trial(pearl_state &s, uint32_t m, uint32_t n, uint32_t k, const uint8_t a_seed[32],
	const uint8_t b_seed[32], const uint32_t bound[8], const uint8_t *chunk_b, uint32_t *hits, uint32_t *dbg)
{
	const uint32_t r = s.cfg.r;
	pearl_key8 ka, kb;
	pearl_key_words(a_seed, ka.w);
	pearl_key_words(b_seed, kb.w);
	if (s.fused) {
		pearl_noise_uniform_kernel <<<(m * r / 32 + 255) / 256, 256>>> (ka, 'A', m * r / 32, (uint32_t *)s.dUA);
		pearl_noise_perm_kernel <<<(k / 8 + 127) / 128, 128>>> (ka, 'A', k, r, s.dPA);
		pearl_noise_uniform_kernel <<<(n * r / 32 + 255) / 256, 256>>> (kb, 'B', n * r / 32, (uint32_t *)s.dUB);
		pearl_noise_perm_kernel <<<(k / 8 + 127) / 128, 128>>> (kb, 'B', k, r, s.dPB);
		pearl_fused_offsets <<<(k + PEARL_FU_OPAD + 255) / 256, 256>>> (s.dPA, s.dPB, k, s.dO2, hits);
		pearl_gemm_params gp;
		gp.m = m; gp.n = n; gp.k = k; gp.r = r; gp.a_seed = ka;
		memcpy(gp.bound, bound, 32);
		pearl_tiles_fused_t<PEARL_FU_GM> <<<(m / PEARL_FU_BT) * (n / PEARL_BT), 256, PEARL_FU_SMEM>>>
			(gp, s.dUA, s.dUB, s.dO2, dbg, hits);
		return;
	}
	pearl_chunk1k cb;
	memcpy(cb.w, chunk_b, BLAKE3_CHUNK_LEN);
	const size_t smem = PEARL_PREP_KW * 16 + (size_t)PEARL_BT * (r + 4);
	const dim3 ga(m / PEARL_BT, (k / 4 + PEARL_PREP_KW - 1) / PEARL_PREP_KW);
	const dim3 gb(n / PEARL_BT, (k / 4 + PEARL_PREP_KW - 1) / PEARL_PREP_KW);
	pearl_noise_uniform_kernel <<<(m * r / 32 + 255) / 256, 256>>> (ka, 'A', m * r / 32, s.dU);
	pearl_noise_perm_kernel <<<(k / 8 + 127) / 128, 128>>> (ka, 'A', k, r, s.dP);
	if (s.imma)	/* IMMA reads its own panel layout (pearl_panel_slot) */
		pearl_operand_panel_fast_t<PEARL_PERM_IMMA_A> <<<ga, 256, smem>>> (NULL, (int8_t *)s.dU, s.dP, k, r, s.dApan);
	else
		pearl_operand_panel_fast_kernel <<<ga, 256, smem>>> (NULL, (int8_t *)s.dU, s.dP, k, r, s.dApan);
	pearl_noise_uniform_kernel <<<(n * r / 32 + 255) / 256, 256>>> (kb, 'B', n * r / 32, s.dU);
	pearl_noise_perm_kernel <<<(k / 8 + 127) / 128, 128>>> (kb, 'B', k, r, s.dP);
	if (s.imma)
		pearl_operand_panel_fast_t<PEARL_PERM_IMMA_B64> <<<gb, 256, smem>>> (NULL, (int8_t *)s.dU, s.dP, k, r, s.dBpan);
	else
		pearl_operand_panel_fast_kernel <<<gb, 256, smem>>> (NULL, (int8_t *)s.dU, s.dP, k, r, s.dBpan);
	if (s.imma)
		pearl_add_row0_v_t<PEARL_PERM_IMMA_B64> <<<1, BLAKE3_CHUNK_LEN / 4>>> (cb, s.dBpan, hits);
	else
		pearl_add_row0_v_kernel <<<1, BLAKE3_CHUNK_LEN / 4>>> (cb, s.dBpan, hits);

	pearl_gemm_params gp;
	gp.m = m; gp.n = n; gp.k = k; gp.r = r; gp.a_seed = ka;
	memcpy(gp.bound, bound, 32);
	const uint32_t blocks = (m / PEARL_BT) * (n / PEARL_BT);
	if (s.imma)
		pearl_tiles_imma_w_t<PEARL_IMMA_WN, PEARL_IMMA_W_STAGES, PEARL_IMMA_W_GM> <<<blocks, 64 * PEARL_IMMA_WN, PEARL_IMMA_W_SMEM>>>
			(gp, (const uint4 *)s.dApan, (const uint4 *)s.dBpan, dbg, hits);
	else
		if (s.dp4a_local)
			pearl_tiles_dp4a_t<PEARL_WIN_LOCAL> <<<blocks, 256>>> (gp, (const uint4 *)s.dApan, (const uint4 *)s.dBpan, dbg, hits);
		else
			pearl_tiles_dp4a_kernel <<<blocks, 256>>> (gp, (const uint4 *)s.dApan, (const uint4 *)s.dBpan, dbg, hits);
}

/* CPU re-verify of one tile; true if it meets the bound */
static bool pearl_cpu_tile(const pearl_state &s, const uint8_t *A, const uint8_t *Bt,
	const uint8_t a_seed[32], const uint8_t b_seed[32], uint32_t ro, uint32_t co,
	const uint32_t bound[8], uint32_t jp_out[16], uint32_t hash_out[8])
{
	const uint32_t k = s.cfg.k, r = s.cfg.r;
	uint32_t tr[16], tc[16], loc[16];
	for (uint32_t i = 0; i < s.h; i++) tr[i] = ro + s.rows[i];
	for (uint32_t i = 0; i < s.w; i++) tc[i] = co + s.cols[i];
	for (int i = 0; i < 16; i++) loc[i] = i;
	int8_t *na = (int8_t *)malloc(s.h * (size_t)k), *nb = (int8_t *)malloc(s.w * (size_t)k);
	pearl_noise_rows(a_seed, 'A', tr, s.h, k, r, na);
	pearl_noise_rows(b_seed, 'B', tc, s.w, k, r, nb);
	/* A, Bt NULL: the fused geometry's constant PEARL_FILL */
	for (uint32_t i = 0; i < s.h; i++)
		for (uint32_t l = 0; l < k; l++)
			na[(size_t)i * k + l] = (int8_t)((A ? (int8_t)A[(size_t)tr[i] * k + l] : PEARL_FILL) + na[(size_t)i * k + l]);
	for (uint32_t i = 0; i < s.w; i++)
		for (uint32_t l = 0; l < k; l++)
			nb[(size_t)i * k + l] = (int8_t)((Bt ? (int8_t)Bt[(size_t)tc[i] * k + l] : PEARL_FILL) + nb[(size_t)i * k + l]);
	uint32_t jp[16];
	uint8_t hb[32];
	pearl_tile_jackpot(na, nb, k, r, loc, s.h, loc, s.w, jp);
	pearl_jackpot_hash(jp, a_seed, hb);
	free(na); free(nb);
	if (jp_out) memcpy(jp_out, jp, 64);
	if (hash_out) pearl_key_words(hb, hash_out);
	return pearl_meets(hb, bound) != 0;
}

/* ------------------------------------------------------------ self-test */
/* Known answers from the Pearl Rust reference: seed.rs commitment_hash_pinned_vectors
 * (Salted), and a generated case (k 2048, rank 128) for noise and the ticket hash. */
static const char *KAT_SEED_B = "60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5";
static const char *KAT_SEED_A = "301784168005ec833ab0aa60006f7fe7faaa95307d8c1fc6819b2ffdd717eccf";
static const char *KAT_C2_A_SEED = "1f3718ce3fb826e0e1a2588f7fa34c801475ee0cd70b160f719ac1b10ca5b2e7";
static const char *KAT_C2_B_SEED = "00ec501cf14fb4468b690f460c3ad90b94c53db61236e7427d4ce8009cd958fe";
static const char *KAT_C2_NOISE_A0 = "c7ff1edd13e50be217daf1ebfcd30a2201f7eb0cfe1df00b1a0710011b09f013f51af9cb2af428011d04090ed614d4e3";
static const char *KAT_C2_NOISE_A1 = "eafc21fdd0d0f11a28d105cefc0aea1a";
static const char *KAT_C2_NOISE_B0 = "0df9e0fa06f5091cecf51111f8ddcb0a0b081a15f41a08f10312f902f3f9dc15";
static const uint32_t KAT_C2_JP[16] = {
	0x0001954c, 0xfffed045, 0xfffff348, 0xfffcd5c1, 0xfffc114b, 0xfffd87a5, 0x0002578e, 0x0003fcbb,
	0xfffc98da, 0x0002bbe5, 0xfffe7f2f, 0x0003190e, 0x0001fdd6, 0xfffcc8ee, 0x0002c5d0, 0xfffe2e48 };
static const char *KAT_C2_JP_HASH = "02793b88ea575adbc98cbe26890ef31e9504445263afa3f6dc2c2fb6136e4247";

static bool hex_eq(const uint8_t *b, size_t n, const char *hex)
{
	char t[3];
	for (size_t i = 0; i < n; i++) {
		snprintf(t, 3, "%02x", b[i]);
		if (t[0] != hex[2 * i] || t[1] != hex[2 * i + 1]) return false;
	}
	return true;
}

static void hex_bytes(const char *hex, uint8_t *out, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		char t[3] = { hex[2 * i], hex[2 * i + 1], 0 };
		out[i] = (uint8_t)strtoul(t, NULL, 16);
	}
}

static bool pearl_host_kats(int fault)
{
	bool ok = true;
	uint8_t jk[32], ha[32], hb[32], bs[32], as[32];
	memset(jk, 0x11, 32); memset(ha, 0xAA, 32); memset(hb, 0xBB, 32);
	pearl_seeds(jk, ha, hb, 192, 320, bs, as);
	if (fault == 1) as[0] ^= 1;
	ok &= hex_eq(bs, 32, KAT_SEED_B) && hex_eq(as, 32, KAT_SEED_A);

	uint8_t sa[32], sb[32];
	hex_bytes(KAT_C2_A_SEED, sa, 32);
	hex_bytes(KAT_C2_B_SEED, sb, 32);
	const uint32_t r01[2] = { 0, 1 };
	int8_t na[2 * PEARL_K], nb[PEARL_K];
	pearl_noise_rows(sa, 'A', r01, 2, PEARL_K, PEARL_R, na);
	pearl_noise_rows(sb, 'B', r01, 1, PEARL_K, PEARL_R, nb);
	ok &= hex_eq((uint8_t *)na, 48, KAT_C2_NOISE_A0) && hex_eq((uint8_t *)na + PEARL_K, 16, KAT_C2_NOISE_A1)
		&& hex_eq((uint8_t *)nb, 32, KAT_C2_NOISE_B0);

	uint8_t h[32];
	pearl_jackpot_hash(KAT_C2_JP, sa, h);
	ok &= hex_eq(h, 32, KAT_C2_JP_HASH);
	return ok;
}

/* GPU vs host on every tile of a 256 x 256 trial, then a changed seed must change the tiles */
static bool pearl_gpu_differential(int thr_id, pearl_state &s, int fault, bool *ran)
{
	const uint32_t m = 256, n = 256, k = PEARL_K, r = s.cfg.r;	/* fused: same kernel at a small k */
	const uint32_t ntile = (m / s.h) * (n / s.w);
	*ran = false;
	uint8_t *A = (uint8_t *)calloc((size_t)m * k, 1), *Bt = (uint8_t *)malloc((size_t)n * k);
	/* the miner's shape: fused A = B = 64; dp4a A = 0, B^T = 0 except a random first chunk of row 0 */
	memset(Bt, s.fused ? PEARL_FILL : 0, (size_t)n * k);
	if (s.fused) memset(A, PEARL_FILL, (size_t)m * k);
	uint32_t lcg = 0x9e3779b9u;
	for (size_t i = 0; i < BLAKE3_CHUNK_LEN && !s.fused; i++) {
		lcg = lcg * 1664525u + 1013904223u;
		Bt[i] = (uint8_t)(int8_t)((int)(lcg >> 25) - 64);
	}
	uint8_t header[PEARL_HEADER_LEN];
	for (int i = 0; i < PEARL_HEADER_LEN; i++) header[i] = (uint8_t)(i * 37 + 11);
	uint8_t jk[32], ha[32], hb[32], bs[32], as[32];
	pearl_config tcfg = s.cfg;
	tcfg.k = k;
	pearl_job_key(header, &tcfg, jk);
	pearl_matrix_root(A, (size_t)m * k, jk, ha);
	pearl_matrix_root(Bt, (size_t)n * k, jk, hb);
	pearl_seeds(jk, ha, hb, m, n, bs, as);
	uint32_t bound[8] = { 0, 0, 0, 0, 0, 0, 0, 0x08000000u };	/* ~1/32 tickets */

	uint32_t *d_dbg = NULL;
	uint32_t *dbg = (uint32_t *)malloc((size_t)ntile * 96), *dbg2 = (uint32_t *)malloc((size_t)ntile * 96);
	bool ok = false;
	if (cudaMalloc(&d_dbg, (size_t)ntile * 96) != cudaSuccess) {
		cudaGetLastError();
		goto done;
	}
	cudaMemset(s.dHits, 0xff, (1 + 2 * PEARL_MAX_HITS) * 4);	/* the trial must clear it */
	{
		uint8_t as_gpu[32];
		memcpy(as_gpu, as, 32);
		if (fault == 2) as_gpu[3] ^= 0x40;
		pearl_gpu_trial(s, m, n, k, as_gpu, bs, bound, Bt, s.dHits, d_dbg);
	}
	if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); goto done; }
	uint32_t hits[1 + 2 * PEARL_MAX_HITS];
	cudaMemcpy(dbg, d_dbg, (size_t)ntile * 96, cudaMemcpyDeviceToHost);
	cudaMemcpy(hits, s.dHits, sizeof(hits), cudaMemcpyDeviceToHost);
	*ran = true;
	{
		int8_t *na = (int8_t *)malloc((size_t)m * k), *nb = (int8_t *)malloc((size_t)n * k);
		pearl_noise_full(as, 'A', m, k, r, na);
		pearl_noise_full(bs, 'B', n, k, r, nb);
		for (size_t i = 0; i < (size_t)m * k; i++) na[i] = (int8_t)((int8_t)A[i] + na[i]);
		for (size_t i = 0; i < (size_t)n * k; i++) nb[i] = (int8_t)((int8_t)Bt[i] + nb[i]);
		std::vector<uint32_t> roff, coff;
		for (uint32_t o = 0; o < m; o++) if (pearl_offset_is_valid(&s.cfg.rows, o)) roff.push_back(o);	/* m multiple of the period */
		for (uint32_t o = 0; o < n; o++) if (pearl_offset_is_valid(&s.cfg.cols, o)) coff.push_back(o);
		uint32_t bad = 0, ref_hits = 0;
		for (uint32_t t = 0; t < ntile; t++) {
			uint32_t tr[16], tc[16], jp[16], hw[8];
			uint8_t hh[32];
			for (uint32_t i = 0; i < s.h; i++) tr[i] = roff[t / (n / s.w)] + s.rows[i];
			for (uint32_t i = 0; i < s.w; i++) tc[i] = coff[t % (n / s.w)] + s.cols[i];
			pearl_tile_jackpot(na, nb, k, r, tr, s.h, tc, s.w, jp);
			pearl_jackpot_hash(jp, as, hh);
			pearl_key_words(hh, hw);
			ref_hits += pearl_meets(hh, bound);
			if (memcmp(jp, dbg + (size_t)t * 24, 64) || memcmp(hw, dbg + (size_t)t * 24 + 16, 32)) bad++;
		}
		free(na); free(nb);
		ok = bad == 0 && hits[0] == ref_hits && ref_hits > 0;
		if (!ok)
			gpulog(LOG_ERR, thr_id, "pearlhash differential: %u of %u tiles differ, hits %u vs %u",
				bad, ntile, hits[0], ref_hits);
	}
	if (ok) {
		uint8_t as_bad[32];
		memcpy(as_bad, as, 32);
		as_bad[17] ^= 1;
		pearl_gpu_trial(s, m, n, k, as_bad, bs, bound, Bt, s.dHits, d_dbg);
		if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); ok = false; *ran = false; goto done; }
		cudaMemcpy(dbg2, d_dbg, (size_t)ntile * 96, cudaMemcpyDeviceToHost);
		if (!memcmp(dbg, dbg2, (size_t)ntile * 96)) {
			gpulog(LOG_ERR, thr_id, "pearlhash differential: negative control did not change the tiles");
			ok = false;
		}
	}
done:
	cudaFree(d_dbg);
	free(A); free(Bt); free(dbg); free(dbg2);
	return ok;
}

/* Merkle tree (layers on the host, as pearl_merkle_build) of a matrix whose every chunk is
 * PEARL_FILL, built on the GPU: leaves x 16 BLAKE3 compressions is ~1 GB of hashing per job at the
 * fused geometry, seconds on the CPU. */
static void pearl_tree_const_gpu(pearl_state &s, pearl_merkle *t, size_t leaves, const uint8_t key[32])
{
	memset(t, 0, sizeof(*t));
	pearl_key_words(key, t->key);
	t->leaves = leaves;
	pearl_key8 k8;
	memcpy(k8.w, t->key, 32);
	pearl_merkle_const_leaves <<<(uint32_t)((leaves + 255) / 256), 256>>> (k8, s.dFill, (uint32_t)leaves, s.dTree0);
	uint32_t *cur = s.dTree0, *nxt = s.dTree1;
	size_t n = leaves;
	int L = 0;
	t->layer[0] = (uint32_t *)malloc(n * 32);
	t->len[0] = n;
	CUDA_SAFE_CALL(cudaMemcpy(t->layer[0], cur, n * 32, cudaMemcpyDeviceToHost));
	while (n > 2) {
		const size_t nn = (n + 1) / 2;
		pearl_merkle_parents <<<(uint32_t)((nn + 255) / 256), 256>>> (k8, cur, (uint32_t)n, nxt);
		L++;
		t->layer[L] = (uint32_t *)malloc(nn * 32);
		t->len[L] = nn;
		CUDA_SAFE_CALL(cudaMemcpy(t->layer[L], nxt, nn * 32, cudaMemcpyDeviceToHost));
		uint32_t *tmp = cur; cur = nxt; nxt = tmp;
		n = nn;
	}
	t->nlayers = L + 1;
	uint32_t root[8];
	pearl_parent_cv(t->key, t->layer[L], t->layer[L] + 8, BLAKE3_ROOT, root);
	blake3_cv_to_bytes(root, t->root);
}

/* self-test leg (fused): the GPU tree of a constant matrix == the host tree, every layer, for an
 * odd leaf count (carried nodes) */
static bool pearl_tree_selftest(pearl_state &s, int fault)
{
	const size_t leaves = 37;
	uint8_t *data = (uint8_t *)malloc(leaves * BLAKE3_CHUNK_LEN), key[32];
	memset(data, PEARL_FILL, leaves * BLAKE3_CHUNK_LEN);
	for (int i = 0; i < 32; i++) key[i] = (uint8_t)(i * 29 + 3);
	pearl_merkle h, g;
	pearl_merkle_build(&h, data, leaves * BLAKE3_CHUNK_LEN, key);
	pearl_tree_const_gpu(s, &g, leaves, key);
	if (fault == 3) g.layer[1][5] ^= 1;
	bool ok = h.nlayers == g.nlayers && !memcmp(h.root, g.root, 32);
	for (int L = 0; L < h.nlayers && ok; L++) ok = h.len[L] == g.len[L] && !memcmp(h.layer[L], g.layer[L], h.len[L] * 32);
	pearl_merkle_free(&h); pearl_merkle_free(&g);
	free(data);
	return ok;
}

static bool pearl_selftest(int thr_id, pearl_state &s)
{
	const char *fenv = getenv("PEARLHASH_SELFTEST_FAULT");
	const int fault = fenv ? atoi(fenv) : 0;
	if (fault)
		gpulog(LOG_WARNING, thr_id, "pearlhash: SELF-TEST FAULT INJECTION ACTIVE (%d) -- for testing only", fault);
	const bool kat = pearl_host_kats(fault);
	bool ran = false;
	const bool diff = pearl_gpu_differential(thr_id, s, fault, &ran);
	if (!ran) selftest_cuda_fault();
	const bool tree = !s.fused || (ran && pearl_tree_selftest(s, fault));
	const bool passed = kat && diff && tree;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "pearlhash self-test FAILED (host KATs %s, GPU differential %s, GPU tree %s)",
			kat ? "ok" : "FAIL", diff ? "ok" : (ran ? "FAIL" : "not run"), tree ? "ok" : "FAIL");
	else
		gpulog(LOG_INFO, thr_id, "pearlhash self-test passed (seed/noise/ticket KATs, %s kernel == oracle on %u tiles, negative control%s)",
			s.fused ? "fused IMMA" : "dp4a", (256 / s.h) * (256 / s.w), s.fused ? ", GPU tree" : "");
	return selftest_gate(thr_id, "pearlhash", passed);
}

/* ------------------------------------------------------------ init / job */
static void pearl_blocking_sync(int thr_id)
{
	unsigned flags = 0;
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaGetLastError();
	if (cudaGetDeviceFlags(&flags) != cudaSuccess ||
	    (flags & cudaDeviceScheduleMask) != cudaDeviceScheduleBlockingSync)
		cuda_reset_device(thr_id, NULL);
	sched_set[thr_id] = true;
}

static void pearl_init(int thr_id)
{
	pearl_state &s = st[thr_id];
	const int dev = device_map[thr_id];
	cudaSetDevice(dev);
	if (opt_cudaschedule < 0) pearl_blocking_sync(thr_id);

	s.imma = device_sm[dev] >= 800;
	s.fused = s.imma;		/* the fused kernel replaces the panel GEMM on sm_80+ */
	/* sm_6x spills the register window inside the loop; sm_75 spills the local one */
	s.dp4a_local = device_sm[dev] < 700;
	s.rows = s.imma ? PEARL_IMMA_ROWS : PEARL_DP4A_PATTERN;
	s.cols = s.imma ? PEARL_IMMA_W_COLS : PEARL_DP4A_PATTERN;
	s.h = 8;
	s.w = s.imma ? 16 : 8;
	s.cfg.k = s.fused ? PEARL_FK : PEARL_K;
	s.cfg.r = PEARL_R;
	s.m = PEARL_M; s.n = PEARL_N;
	s.mc = s.fused ? PEARL_M + 1 : PEARL_M;
	s.units = (uint32_t)(((uint64_t)s.m * s.n * s.cfg.k) >> 20);
	s.nonce_leaf = (size_t)PEARL_M * s.cfg.k / BLAKE3_CHUNK_LEN;
	pearl_pattern_from_list(s.rows, s.h, &s.cfg.rows);
	pearl_pattern_from_list(s.cols, s.w, &s.cfg.cols);

	if (s.fused) {
		const size_t leaves = (size_t)s.mc * s.cfg.k / BLAKE3_CHUNK_LEN;
		CUDA_SAFE_CALL(cudaMalloc(&s.dUA, (size_t)s.m * PEARL_R));
		CUDA_SAFE_CALL(cudaMalloc(&s.dUB, (size_t)s.n * PEARL_R));
		CUDA_SAFE_CALL(cudaMalloc(&s.dPA, (size_t)s.cfg.k * 4));
		CUDA_SAFE_CALL(cudaMalloc(&s.dPB, (size_t)s.cfg.k * 4));
		CUDA_SAFE_CALL(cudaMalloc(&s.dO2, (size_t)(s.cfg.k + PEARL_FU_OPAD) * 8));
		CUDA_SAFE_CALL(cudaMalloc(&s.dFill, BLAKE3_CHUNK_LEN));
		CUDA_SAFE_CALL(cudaMemset(s.dFill, PEARL_FILL, BLAKE3_CHUNK_LEN));
		CUDA_SAFE_CALL(cudaMalloc(&s.dTree0, leaves * 32));
		CUDA_SAFE_CALL(cudaMalloc(&s.dTree1, (leaves + 1) / 2 * 32));
		CUDA_SAFE_CALL(cudaFuncSetAttribute(pearl_tiles_fused_t<PEARL_FU_GM>, cudaFuncAttributeMaxDynamicSharedMemorySize, PEARL_FU_SMEM));
	} else {
		CUDA_SAFE_CALL(cudaMalloc(&s.dU, (size_t)PEARL_M * PEARL_R));
		CUDA_SAFE_CALL(cudaMalloc(&s.dP, (size_t)PEARL_K * 4));
		CUDA_SAFE_CALL(cudaMalloc(&s.dApan, (size_t)PEARL_M * PEARL_K));
		CUDA_SAFE_CALL(cudaMalloc(&s.dBpan, (size_t)PEARL_N * PEARL_K));
	}
	CUDA_SAFE_CALL(cudaMalloc(&s.dHits, (1 + 2 * PEARL_MAX_HITS) * 4));
	for (int b = 0; b < 2; b++) {
		CUDA_SAFE_CALL(cudaHostAlloc(&s.slot[b].hits, PEARL_HITS_BYTES, cudaHostAllocMapped));
		CUDA_SAFE_CALL(cudaHostGetDevicePointer(&s.slot[b].dhits, s.slot[b].hits, 0));
		CUDA_SAFE_CALL(cudaEventCreateWithFlags(&s.slot[b].done, cudaEventDisableTiming | cudaEventBlockingSync));
	}
	s.A = s.fused ? NULL : (uint8_t *)calloc((size_t)PEARL_M * PEARL_K, 1);
	s.Bt = s.fused ? NULL : (uint8_t *)calloc((size_t)PEARL_N * PEARL_K, 1);
	s.have_job = false;

	pearl_selftest(thr_id, s);
	gpulog(LOG_INFO, thr_id, "pearlhash: %s kernel, m=n=%u k=%u rank %u, %u tickets of %u MAC per trial%s",
		s.fused ? "fused IMMA (sm_80+)" : "dp4a", s.m, s.cfg.k, PEARL_R,
		(s.m / s.h) * (s.n / s.w), s.h * s.w * s.cfg.k, s.fused ? ", A = B = 64, nonce in an unmined row" : "");
	init_done[thr_id] = true;
}

/* new header: job_key, both trees (B^T chunk 0 is patched per trial), fresh salt */
static void pearl_new_job(int thr_id, const uint8_t *header)
{
	pearl_state &s = st[thr_id];
	memcpy(s.header, header, PEARL_HEADER_LEN);
	pearl_job_key(header, &s.cfg, s.job_key);
	if (s.have_job) { pearl_merkle_free(&s.ta); pearl_merkle_free(&s.tb); }
	if (s.fused) {	/* constant matrices; the nonce leaf of A is set per trial */
		pearl_tree_const_gpu(s, &s.ta, (size_t)s.mc * s.cfg.k / BLAKE3_CHUNK_LEN, s.job_key);
		pearl_tree_const_gpu(s, &s.tb, (size_t)s.n * s.cfg.k / BLAKE3_CHUNK_LEN, s.job_key);
	} else {
		memset(s.Bt, 0, BLAKE3_CHUNK_LEN);
		pearl_merkle_build(&s.ta, s.A, (size_t)PEARL_M * PEARL_K, s.job_key);
		pearl_merkle_build(&s.tb, s.Bt, (size_t)PEARL_N * PEARL_K, s.job_key);
	}
	s.salt = ((uint64_t)rand() << 40) ^ ((uint64_t)rand() << 20) ^ (uint64_t)rand() ^ ((uint64_t)time(NULL) << 24)
		^ ((uint64_t)thr_id << 58);
	s.have_job = true;
}

/* ------------------------------------------------------------ scanhash */
extern "C" int scanhash_pearlhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	pearl_state &s = st[thr_id];
	if (!init_done[thr_id]) pearl_init(thr_id);

	/* job refused by the stratum layer (unsupported cert_version): idle */
	if (!work->is_pearl && !opt_benchmark) {
		usleep(200 * 1000);
		*hashes_done = 0;
		return 0;
	}

	uint8_t header[PEARL_HEADER_LEN];
	memcpy(header, pdata, PEARL_HEADER_LEN);
	uint32_t bound[8];
	if (work->is_pearl) {
		if (!pearl_share_bound(work->pearl_target, &s.cfg, bound)) {
			static char logged[MAX_GPUS][65];
			if (strcmp(logged[thr_id], work->pearl_target)) {
				strcpy(logged[thr_id], work->pearl_target);
				gpulog(LOG_ERR, thr_id, "pearlhash: unusable share target %s", work->pearl_target);
			}
			*hashes_done = 0;
			return 0;
		}
	} else {
		/* --benchmark: about one share per 16 trials of 2^37 MAC, written into the header
		 * so a dumped share is valid against its own nbits (fused: target / 32 against the
		 * 32x ticket work, so still ~1 share per 16 trials) */
		const uint32_t nbits = s.fused ? 0x1c000400 : 0x1c008000;
		memcpy(header + 72, &nbits, 4);
		pearl_bound(nbits, &s.cfg, bound);
	}
	/* --benchmark changes data[18] and resets the cursor on every call: ignore that
	 * word and keep a private cursor, or every call re-mines trial 0 */
	const bool bench = opt_benchmark;
	const size_t cmp_len = bench ? 72 : PEARL_HEADER_LEN;
	if (!s.have_job || memcmp(header, s.header, cmp_len)) {
		pearl_new_job(thr_id, header);
		s.bench_trial = 0;
	}
	uint32_t *cursor = bench ? &s.bench_trial : &pdata[19];

	work->valid_nonces = 0;
	uint32_t trials = 0;
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);

	/* One trial is queued ahead: trial t+1's host work and kernels are issued before
	 * waiting for trial t. s.Bt / s.tb hold the newest trial's chunk; a hit of the older
	 * trial swaps its chunk back for the re-verify and proof. */
	pearl_slot *q = s.slot;
	int newest = -1;
	auto set_chunk = [&](int b) {
		if (s.fused) {	/* the first chunk of A's unmined row PEARL_M */
			pearl_merkle_update_leaf(&s.ta, s.nonce_leaf, q[b].chunk);
			return;
		}
		memcpy(s.Bt, q[b].chunk, BLAKE3_CHUNK_LEN);
		pearl_merkle_update_leaf(&s.tb, 0, q[b].chunk);
	};
	auto enqueue = [&](int b, uint32_t trial) {
		struct timeval ta, tb2;
		gettimeofday(&ta, NULL);
		q[b].trial = trial;
		pearl_nonce_chunk(q[b].chunk, trial, s.salt, thr_id);
		set_chunk(b);
		newest = b;
		pearl_seeds(s.job_key, s.ta.root, s.tb.root, s.mc, s.n, q[b].bs, q[b].as);
		/* slot b's hit list is only rewritten by this trial, after its previous reader is done */
		pearl_gpu_trial(s, s.m, s.n, s.cfg.k, q[b].as, q[b].bs, bound, q[b].chunk, q[b].dhits, NULL);
		cudaEventRecord(q[b].done);
		gettimeofday(&tb2, NULL);
		s.t_host += (tb2.tv_sec - ta.tv_sec) + 1e-6 * (tb2.tv_usec - ta.tv_usec);
	};

	int cur = 0;
	bool more = true;
	enqueue(cur, *cursor);
	for (;;) {
		const uint32_t next = q[cur].trial + 1;
		const bool ahead = more && (bench || next < max_nonce);
		if (ahead) enqueue(cur ^ 1, next);

		struct timeval tb2, tc;
		gettimeofday(&tb2, NULL);
		CUDA_SAFE_CALL(cudaEventSynchronize(q[cur].done));
		gettimeofday(&tc, NULL);
		const uint32_t trial = q[cur].trial;
		const uint32_t *hits = q[cur].hits;
		*cursor = trial + 1;
		trials++;
		s.t_gpu += (tc.tv_sec - tb2.tv_sec) + 1e-6 * (tc.tv_usec - tb2.tv_usec);
		s.t_trials++;

		const uint32_t nh = min(hits[0], (uint32_t)PEARL_MAX_HITS);
		if (nh && newest != cur) set_chunk(cur);
		for (uint32_t h = 0; h < nh && work->valid_nonces < MAX_NONCES; h++) {
			const uint32_t ro = hits[1 + 2 * h], co = hits[2 + 2 * h];
			if (!pearl_cpu_tile(s, s.A, s.Bt, q[cur].as, q[cur].bs, ro, co, bound, NULL, NULL)) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "pearlhash: tile %u,%u of trial %08x does not validate on CPU!", ro, co, trial);
				continue;
			}
			uint32_t tr[16], tc[16];
			for (uint32_t i = 0; i < s.h; i++) tr[i] = ro + s.rows[i];
			for (uint32_t i = 0; i < s.w; i++) tc[i] = co + s.cols[i];
			pearl_buf b = { 0, 0, 0 };
			if (s.fused) {	/* mined rows hold only PEARL_FILL chunks */
				static uint8_t fill[BLAKE3_CHUNK_LEN];
				if (!fill[0]) memset(fill, PEARL_FILL, sizeof(fill));
				pearl_plain_proof_const(&b, s.mc, s.n, s.cfg.k, PEARL_R, &s.ta, fill, tr, s.h, &s.tb, fill, tc, s.w);
			} else
				pearl_plain_proof(&b, PEARL_M, PEARL_N, PEARL_K, PEARL_R, &s.ta, s.A, tr, s.h, &s.tb, s.Bt, tc, s.w);
			/* PEARLHASH_DUMP=<dir>: write the first share as header76.hex + proof.bin */
			static bool dumped[MAX_GPUS];
			const char *dump = getenv("PEARLHASH_DUMP");
			if (dump && !dumped[thr_id]) {
				char path[512];
				snprintf(path, sizeof(path), "%s/proof.bin", dump);
				FILE *f = fopen(path, "wb");
				if (f) { fwrite(b.p, 1, b.n, f); fclose(f); }
				snprintf(path, sizeof(path), "%s/header76.hex", dump);
				f = fopen(path, "w");
				if (f) { for (int i = 0; i < PEARL_HEADER_LEN; i++) fprintf(f, "%02x", header[i]); fclose(f); }
				dumped[thr_id] = true;
				gpulog(LOG_NOTICE, thr_id, "pearlhash: share of trial %08x dumped to %s (%zu B)", trial, dump, b.n);
			}
			char *b64 = pearl_base64(b.p, b.n);
			free(b.p);
			if (opt_benchmark) {
				free(b64);
				work->nonces[work->valid_nonces] = trial;
			} else {
				work->nonces[work->valid_nonces] = pearl_proof_put(b64);
			}
			work->sharediff[work->valid_nonces] = work->targetdiff;
			work->valid_nonces++;
		}
		if (nh && newest != cur) set_chunk(newest);
		if (nh) {
			struct timeval td;
			gettimeofday(&td, NULL);
			s.t_hit += (td.tv_sec - tc.tv_sec) + 1e-6 * (td.tv_usec - tc.tv_usec);
		}
		if (opt_debug && time(NULL) - s.t_last >= 10 && s.t_trials) {
			gpulog(LOG_DEBUG, thr_id, "pearlhash: %u trials, per trial: host+launch %.3f ms, gpu wait %.3f ms, share path %.3f ms",
				s.t_trials, 1e3 * s.t_host / s.t_trials, 1e3 * s.t_gpu / s.t_trials, 1e3 * s.t_hit / s.t_trials);
			s.t_host = s.t_gpu = s.t_hit = 0; s.t_trials = 0; s.t_last = time(NULL);
		}
		/* stop queueing on a share, the time budget or a restart; the queued trial is
		 * still drained, so the cursor stays exact */
		gettimeofday(&t1, NULL);
		if (work->valid_nonces || work_restart[thr_id].restart ||
		    (t1.tv_sec - t0.tv_sec) + 1e-6 * (t1.tv_usec - t0.tv_usec) >= PEARL_CALL_SECONDS)
			more = false;
		if (!ahead) break;
		cur ^= 1;
	}

	*hashes_done = (unsigned long)trials * s.units;
	return work->valid_nonces;
}

extern "C" void free_pearlhash(int thr_id)
{
	if (!init_done[thr_id]) return;
	pearl_state &s = st[thr_id];
	cudaSetDevice(device_map[thr_id]);
	cudaFree(s.dU); cudaFree(s.dP); cudaFree(s.dApan); cudaFree(s.dBpan); cudaFree(s.dHits);
	cudaFree(s.dUA); cudaFree(s.dUB); cudaFree(s.dPA); cudaFree(s.dPB); cudaFree(s.dO2);
	cudaFree(s.dFill); cudaFree(s.dTree0); cudaFree(s.dTree1);
	for (int b = 0; b < 2; b++) { cudaFreeHost(s.slot[b].hits); cudaEventDestroy(s.slot[b].done); }
	free(s.A); free(s.Bt);
	if (s.have_job) { pearl_merkle_free(&s.ta); pearl_merkle_free(&s.tb); }
	s.have_job = false;
	if (sched_set[thr_id]) {
		cudaSetDeviceFlags(cudaDeviceScheduleAuto);
		cudaGetLastError();
		sched_set[thr_id] = false;
	}
	init_done[thr_id] = false;
}
