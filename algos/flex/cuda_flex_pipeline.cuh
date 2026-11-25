/**
 * The Flex chain on the GPU.
 *
 * 15 core rounds + 3 CryptoNight rounds + a final SHA3-256, where EVERY LANE
 * RUNS A DIFFERENT ORDER.  The x16r/ghostrider model (derive once per job,
 * then run whole-batch kernels) does not apply; instead each round sorts the
 * lanes by the algorithm they need, compacts them, and calls the existing
 * shared stage kernel exactly the way x16r calls it -- base pointer at offset
 * 0, `count` threads, no index vector.  Total lane-rounds are unchanged; only
 * the launch granularity shrinks.
 *
 * Header-only so the whole pipeline can be driven from a harness and compared
 * to the CPU reference step by step, without building the miner.
 *
 * v1 deliberately keeps the CN scratchpad stride UNIFORM at the largest
 * variant (2 MiB/lane).  Variable-stride packing (~2.7x more lanes) is a
 * separate, measurable step -- see the the project notes.
 */

#ifndef CUDA_FLEX_PIPELINE_CUH
#define CUDA_FLEX_PIPELINE_CUH

#include <stdint.h>
#include <string.h>

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"
#include "algos/flex/flex.h"
#include "algos/flex/cuda_flex_sha3.cuh"
#include "algos/flex/cuda_flex_order.cuh"
/* for FLEX_FILTER_CORE_MASK: round 0 of the filtered path depends on it */
#include "algos/flex/cuda_flex_filter.cuh"
#include "algos/flex/flex_schedule.h"

#ifdef __CUDACC__

/* ---------------------------------------------------------------------------
 * Per-phase batch profile (FLEX_PROFILE=1). Each mark syncs the device and
 * books the host time since the previous mark to one phase; the syncs remove
 * phase overlap, so read the split, not the rate. Off: one branch per mark.
 * --------------------------------------------------------------------------- */
enum {
	FLEX_PH_FILTER = 0, FLEX_PH_ORDER, FLEX_PH_ROUND0, FLEX_PH_CORE,
	FLEX_PH_CN_PREP, FLEX_PH_CN_CORE, FLEX_PH_CN_FINAL, FLEX_PH_SHA3,
	FLEX_PH_SCREEN, FLEX_PH_COUNT
};
static const char *flex_ph_name[FLEX_PH_COUNT] = {
	"filter", "order", "round0", "core1-14", "cn-prep", "cn-core", "cn-final", "sha3", "screen"
};
static int flex_prof_on = -1;
static double flex_prof_ms[FLEX_PH_COUNT];
static double flex_prof_t;
static uint32_t flex_prof_batches;
/* per CN variant: summed group time, group count and lanes, across the 3 rounds */
static double flex_prof_v_ms[6];
static uint32_t flex_prof_v_groups[6];
static double flex_prof_v_lanes[6];


static inline bool flex_prof_enabled()
{
	if (flex_prof_on < 0)
		flex_prof_on = getenv("FLEX_PROFILE") ? 1 : 0;
	return flex_prof_on == 1;
}

static inline double flex_prof_now_ms()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Start a batch: the next mark books from here. */
static inline void flex_prof_begin()
{
	if (!flex_prof_enabled()) return;
	cudaDeviceSynchronize();
	flex_prof_t = flex_prof_now_ms();
}

static inline void flex_prof_mark(int phase)
{
	if (!flex_prof_enabled()) return;
	cudaDeviceSynchronize();
	const double now = flex_prof_now_ms();
	flex_prof_ms[phase] += now - flex_prof_t;
	flex_prof_t = now;
}

/* Book one CN variant group (the device work since the previous mark). */
static inline void flex_prof_group(int v, uint32_t lanes)
{
	if (!flex_prof_enabled()) return;
	cudaDeviceSynchronize();
	const double now = flex_prof_now_ms();
	flex_prof_v_ms[v] += now - flex_prof_t;
	flex_prof_ms[FLEX_PH_CN_CORE] += now - flex_prof_t;
	flex_prof_v_groups[v]++;
	flex_prof_v_lanes[v] += lanes;
	flex_prof_t = now;
}

/* also called after the startup checks, which book phases too */
static inline void flex_prof_reset()
{
	memset(flex_prof_ms, 0, sizeof(flex_prof_ms));
	memset(flex_prof_v_ms, 0, sizeof(flex_prof_v_ms));
	memset(flex_prof_v_groups, 0, sizeof(flex_prof_v_groups));
	memset(flex_prof_v_lanes, 0, sizeof(flex_prof_v_lanes));
	flex_prof_batches = 0;
}

/* End a batch; every 20 batches log the mean per phase and reset. */
static inline void flex_prof_end(int thr_id, uint32_t lanes)
{
	if (!flex_prof_enabled()) return;
	if (++flex_prof_batches < 20) return;
	char buf[512];
	int n = 0;
	double total = 0;
	for (int i = 0; i < FLEX_PH_COUNT; i++) total += flex_prof_ms[i];
	for (int i = 0; i < FLEX_PH_COUNT; i++)
		n += snprintf(buf + n, sizeof(buf) - n, " %s %.0f", flex_ph_name[i],
		              flex_prof_ms[i] / flex_prof_batches);
	gpulog(LOG_INFO, thr_id, "flex profile, ms/batch over %u batches (%u lanes): total %.0f |%s",
	       flex_prof_batches, lanes, total / flex_prof_batches, buf);
	/* per variant: ms per GROUP and mean lanes per group */
	static const char *vn[6] = { "dark", "darklite", "fast", "lite", "turtle", "turtlelite" };
	n = 0;
	for (int v = 0; v < 6; v++) {
		if (!flex_prof_v_groups[v]) continue;
		n += snprintf(buf + n, sizeof(buf) - n, " %s %.1fms/%.0fl x%.1f", vn[v],
		              flex_prof_v_ms[v] / flex_prof_v_groups[v],
		              flex_prof_v_lanes[v] / flex_prof_v_groups[v],
		              (double)flex_prof_v_groups[v] / flex_prof_batches);
	}
	gpulog(LOG_INFO, thr_id, "flex profile, per CN group (ms/group, lanes/group, groups/batch):%s", buf);
	flex_prof_reset();
}

/* CryptoNight-v1 GPU path. prepare and the core are shared with ghostrider
 * UNCHANGED (demonstrated in the an earlier device gate); only the finalization is flex's. */
extern "C" void cryptonight_extra_cpu_prepare_gr(int thr_id, uint32_t threads, uint64_t *d_hash,
	uint64_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b,
	uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, uint64_t *d_ctx_tweak);
extern "C" void cryptonight_core_cuda_flex(int thr_id, int blocks, int threads, uint32_t nlanes,
	int variant, uint32_t stride64,
	uint64_t *d_long_state, uint64_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b,
	uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, uint64_t *d_ctx_tweak);
extern "C" void cryptonight_core_cuda_flex_multi(int thr_id, int threads, int ngroups,
	const uint32_t *nlanes, const int *variant, const uint32_t *stride64,
	uint64_t * const *d_long_state, uint64_t * const *d_ctx_state,
	uint32_t * const *d_ctx_a, uint32_t * const *d_ctx_b,
	uint32_t * const *d_ctx_key1, uint32_t * const *d_ctx_key2, uint64_t * const *d_ctx_tweak);
extern "C" void cryptonight_core_cuda_flex_free(int thr_id);

/* FLEX_CN_SERIAL=1 runs the variant groups one after another (the old path,
 * kept for A/B and for the per-group profile); default is concurrent. */
static int flex_cn_serial = -1;
static inline bool flex_cn_serial_enabled()
{
	if (flex_cn_serial < 0)
		flex_cn_serial = getenv("FLEX_CN_SERIAL") ? 1 : 0;
	return flex_cn_serial == 1;
}
extern "C" void cryptonight_extra_cpu_final_flex(int thr_id, uint32_t threads,
	uint64_t *d_ctx_state, uint64_t *d_hash);

#define FLEX_CN_TPB          32                /* lanes per CN block */

/* Per-variant scratchpad size, indexed by FlexCNAlgo. Byte-identical to
 * ghostrider's gr_mem[] and to flex's own cryptonight_*.c. */
static const uint32_t flex_cn_mem[FLEX_CN_ALGO_COUNT] = {
	524288u,   /* dark       */
	524288u,   /* darklite   */
	2097152u,  /* fast       */
	1048576u,  /* lite       */
	262144u,   /* turtle     */
	262144u,   /* turtlelite */
};
#define FLEX_CN_MEM_MAX  2097152u              /* fast, the largest */

/* Mean bytes per lane over a uniform variant draw: (512+512+2048+1024+256+256)
 * KiB / 6 = 768 KiB. Sizing the batch by the MEAN instead of the MAX is the
 * whole of variable-stride packing -- the CN variant is chosen per nonce, so a batch holds a mix and
 * reserving 2 MiB for every lane wastes ~2.7x the scratchpad.
 *
 * MEASURED, not assumed. The variant is picked by reducing a 4-bit nibble
 * %% 6, and 16 %% 6 leaves values 0..3 arriving 3/16 of the time against
 * 2/16 for 4 and 5 -- i.e. the FOUR EXPENSIVE variants are over-weighted,
 * not uniform. Simulating the real walk over 200k lanes gives 837 677 B
 * per lane per CN round; the uniform-draw figure (786 432) is 6.5%% low,
 * which was enough to make the batch trim fire on essentially every
 * batch. */
#define FLEX_CN_MEM_MEAN 837677u

/* ===========================================================================
 * Flex-local kernels: SHA-3 rounds, and the round-0 select-copy.
 * =========================================================================== */

/* SHA3-512 of the 80-byte header, for the lanes whose FIRST core round is
 * KECCAK. Writes the 64-byte chain state. */
static __global__
void flex_sha3_512_80_gpu(uint32_t threads, uint32_t startNonce, uint2 * __restrict__ g_hash)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= threads) return;
	uint2 out[8];
	flex_sha3_512_seed_80(startNonce + t, out);
	#pragma unroll 8
	for (int i = 0; i < 8; i++) g_hash[(size_t)t * 8 + i] = out[i];
}

/* SHA3-512 of the 64-byte chain state, in place -- the KECCAK core round. */
static __global__
void flex_sha3_512_64_gpu(uint32_t threads, uint2 * __restrict__ g_hash)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= threads) return;
	uint2 h[8];
	uint2 *p = &g_hash[(size_t)t * 8];
	#pragma unroll 8
	for (int i = 0; i < 8; i++) h[i] = p[i];
	flex_sha3_512_hash_64(h);
	#pragma unroll 8
	for (int i = 0; i < 8; i++) p[i] = h[i];
}

/* The closing SHA3-256 over all 64 bytes, into the low half of the slot.
 * All 64 bytes are hashed, which is why the high half of a CN result is
 * consensus-load-bearing in flex where ghostrider discards it. */
static __global__
void flex_sha3_256_final_gpu(uint32_t threads, uint2 * __restrict__ g_hash)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= threads) return;
	uint2 *p = &g_hash[(size_t)t * 8];
	uint2 in[8], out[4];
	#pragma unroll 8
	for (int i = 0; i < 8; i++) in[i] = p[i];
	flex_sha3_256_hash_64(in, out);
	#pragma unroll 4
	for (int i = 0; i < 4; i++) p[i] = out[i];
}

/* Round 0 select-copy: dst[t] = src[t] for the lanes whose core[0] == algo.
 *
 * The 80-byte stage kernels derive their nonce as startNounce + thread and so
 * cannot be run over a scattered subset. v1 therefore runs each of the 14 over
 * the WHOLE batch into a scratch buffer and keeps only the lanes that wanted
 * it -- 14 extra 80-byte core rounds, about one extra core chain, which is
 * ~+0.6% of wall clock at ghostrider's core/CN split. The cheaper fix is
 * per-lane-nonce entry points for the 14 stages; see the the project notes. */
static __global__
void flex_select_round0_gpu(uint32_t threads, uint4 * __restrict__ dst,
                            const uint4 * __restrict__ src,
                            const uint8_t * __restrict__ core0, uint32_t algo)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= threads || core0[t] != algo) return;
	const uint4 *s = src + (size_t)t * 4;
	uint4 *d = dst + (size_t)t * 4;
	d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
}

/* Filtered round-0 gather: dst[lane] = src[nonce(lane) - chunkStart] for the accepted
 * lanes whose core[0] == algo and whose nonce falls in this chunk.
 *
 * Why chunks at all: the shared stage kernels are initialised by
 * flex_stage_init_all() for T lanes and allocate per-thread scratch for exactly
 * that many, so they CANNOT be launched over the whole 31T scan span. The span
 * is therefore walked in T-sized pieces, one stage launch per (chunk, algo).
 *
 * And why a gather at all: the shared algos/stages/ *_cuda_hash_80 entry
 * points derive nonce = startNounce + thread, so they cannot be run over a
 * scattered nonce set. Pinning core[0] to KECCAK -- flex's own 80-byte kernel,
 * which IS per-lane -- would have avoided this entirely and is PROVABLY
 * IMPOSSIBLE; see the header of cuda_flex_filter.cuh. */
static __global__
void flex_gather_round0_gpu(uint32_t lanes, uint32_t chunkStart, uint32_t chunkLen,
                            uint4 * __restrict__ dst, const uint4 * __restrict__ src,
                            const uint32_t * __restrict__ d_nonces,
                            const uint8_t * __restrict__ core0, uint32_t algo)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= lanes || core0[t] != algo)
		return;
	const uint32_t off = d_nonces[t] - chunkStart;
	if (off >= chunkLen)          /* unsigned: also rejects nonce < chunkStart */
		return;
	const uint4 *s = src + (size_t)off * 4;
	uint4 *dd = dst + (size_t)t * 4;
	dd[0] = s[0]; dd[1] = s[1]; dd[2] = s[2]; dd[3] = s[3];
}

/* ===========================================================================
 * Context
 * =========================================================================== */

struct flex_ctx {
	uint32_t  threads;        /* CAPACITY: lanes this context is sized for */
	uint32_t  lanes;          /* ACTIVE: lanes in the current batch (<= threads).
	                           * They differ when a batch's variant mix needs
	                           * more scratchpad than the budget holds, which is
	                           * resolved by trimming the tail -- legal because
	                           * each lane's chain depends only on its own
	                           * nonce, so dropping lanes changes nothing about
	                           * the ones kept. */
	size_t    scratch_bytes;  /* d_long_state budget */
	uint32_t *d_hash;         /* 64 B/lane, canonical lane order */
	uint32_t *d_scratch;      /* 64 B/lane, compacted working buffer */
	uint8_t  *d_core;         /* [FLEX_CORE_CHAIN_LEN][threads] */
	uint8_t  *d_cn;           /* [FLEX_CN_ALGO_COUNT][threads] */
	uint32_t *d_idx;          /* lane permutation for the current round */
	uint64_t *d_long_state;
	uint64_t *d_ctx_state;
	uint32_t *d_ctx_a, *d_ctx_b, *d_ctx_key1, *d_ctx_key2;
	uint64_t *d_ctx_tweak;
	/* host mirrors */
	uint8_t  *h_core, *h_cn;
	uint32_t *h_idx, *h_off;

	/* Nonce-filter state. `span` is the nonce range the filter scanned to produce
	 * the current batch; round 0 needs it to chunk, and scanhash advances the
	 * cursor by it -- NOT by the lane count, which is ~31x smaller. */
	uint32_t  span;
	uint32_t *d_nonces;       /* [threads] accepted nonces, unordered */
	uint32_t *d_count;        /* filter yield; may EXCEED threads (saturated) */
	uint32_t *h_nonces;       /* host copy: maps a screen's lane index -> nonce */
	/* Running estimate of the filter's acceptance, re-fitted from the TRUE
	 * yield each batch. Seeded from a measurement but not trusting it: this is
	 * what keeps the span correct if the admitted set or the unit threshold
	 * ever changes, instead of leaving a stale constant nobody re-measures. */
	double    accept_est;

	/* Stop at a job change (flex_pipeline_abandoned). Set only around the
	 * mining call, so the startup guards run to the end. */
	bool      abort_on_restart;
};

static bool flex_pipeline_init(flex_ctx *c, int thr_id, uint32_t threads, size_t scratch_bytes)
{
	memset(c, 0, sizeof(*c));
	c->threads = threads;
	c->lanes   = threads;
	c->scratch_bytes = scratch_bytes;

	const size_t st = scratch_bytes;
	if (cudaMalloc(&c->d_hash,    (size_t)64 * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_scratch, (size_t)64 * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_core, (size_t)FLEX_CORE_CHAIN_LEN * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_cn,   (size_t)FLEX_CN_ALGO_COUNT  * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_idx,  sizeof(uint32_t) * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_nonces, sizeof(uint32_t) * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_count,  sizeof(uint32_t)) != cudaSuccess) return false;
	c->h_nonces = (uint32_t*)malloc(sizeof(uint32_t) * threads);
	if (!c->h_nonces) return false;
	c->span = 0;
	c->accept_est = FLEX_FILTER_ACCEPT_SEED;
	if (cudaMalloc(&c->d_long_state, st) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_ctx_state, (size_t)26 * sizeof(uint64_t) * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_ctx_a,    (size_t)4 * sizeof(uint32_t) * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_ctx_b,    (size_t)4 * sizeof(uint32_t) * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_ctx_key1, (size_t)40 * sizeof(uint32_t) * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_ctx_key2, (size_t)40 * sizeof(uint32_t) * threads) != cudaSuccess) return false;
	if (cudaMalloc(&c->d_ctx_tweak, sizeof(uint64_t) * threads) != cudaSuccess) return false;

	c->h_core = (uint8_t*)  malloc((size_t)FLEX_CORE_CHAIN_LEN * threads);
	c->h_cn   = (uint8_t*)  malloc((size_t)FLEX_CN_ALGO_COUNT  * threads);
	c->h_idx  = (uint32_t*) malloc(sizeof(uint32_t) * threads);
	c->h_off  = (uint32_t*) malloc(sizeof(uint32_t) * (FLEX_MAX_BUCKETS + 1));
	return c->h_core && c->h_cn && c->h_idx && c->h_off;
}

static void flex_pipeline_free(flex_ctx *c)
{
	cudaFree(c->d_hash); cudaFree(c->d_scratch);
	cudaFree(c->d_core); cudaFree(c->d_cn); cudaFree(c->d_idx);
	cudaFree(c->d_nonces); cudaFree(c->d_count);
	cudaFree(c->d_long_state); cudaFree(c->d_ctx_state);
	cudaFree(c->d_ctx_a); cudaFree(c->d_ctx_b);
	cudaFree(c->d_ctx_key1); cudaFree(c->d_ctx_key2); cudaFree(c->d_ctx_tweak);
	free(c->h_core); free(c->h_cn); free(c->h_idx); free(c->h_off);
	free(c->h_nonces);
	memset(c, 0, sizeof(*c));
}

/* ===========================================================================
 * Stage dispatch.
 *
 * These indices are FLEX's, not ghostrider's: the pool drops JH and is
 * REINDEXED, so everything from 3 upward means a different algorithm in the
 * two enums. See algos/flex/flex.h.
 * =========================================================================== */

static void flex_stage_setBlock_80(int algo, int thr_id, uint32_t *endiandata, uint32_t *pdata)
{
	switch (algo) {
	case FLEX_BLAKE:     blake512_cpu_setBlock_80(thr_id, endiandata); break;
	case FLEX_BMW:       bmw512_cpu_setBlock_80(endiandata); break;
	case FLEX_GROESTL:   groestl512_setBlock_80(thr_id, endiandata); break;
	case FLEX_KECCAK:    break;                       /* flex-local, uses c_flex_head */
	case FLEX_SKEIN:     skein512_cpu_setBlock_80((void*)endiandata); break;
	case FLEX_LUFFA:     qubit_luffa512_cpu_setBlock_80((void*)endiandata); break;
	case FLEX_CUBEHASH:  cubehash512_setBlock_80(thr_id, endiandata); break;
	case FLEX_SHAVITE:   x16_shavite512_setBlock_80((void*)endiandata); break;
	case FLEX_SIMD:      x16_simd512_setBlock_80((void*)endiandata); break;
	case FLEX_ECHO:      x16_echo512_setBlock_80((void*)endiandata); break;
	case FLEX_HAMSI:     x16_hamsi512_setBlock_80((void*)endiandata); break;
	case FLEX_FUGUE:     x16_fugue512_setBlock_80((void*)pdata); break;   /* byteswaps internally */
	case FLEX_SHABAL:    x16_shabal512_setBlock_80((void*)endiandata); break;
	case FLEX_WHIRLPOOL: x16_whirlpool512_setBlock_80((void*)endiandata); break;
	}
}

static void flex_stage_hash_80(int algo, int thr_id, uint32_t threads, uint32_t nonce, uint32_t *d_h)
{
	switch (algo) {
	case FLEX_BLAKE:     blake512_cpu_hash_80(thr_id, threads, nonce, d_h); break;
	case FLEX_BMW:       bmw512_cpu_hash_80(thr_id, threads, nonce, d_h, 0); break;
	case FLEX_GROESTL:   groestl512_cuda_hash_80(thr_id, threads, nonce, d_h); break;
	case FLEX_KECCAK: {
		const uint32_t tpb = 128;
		flex_sha3_512_80_gpu <<<dim3((threads + tpb - 1) / tpb), dim3(tpb)>>>
			(threads, nonce, (uint2*)d_h);
		break;
	}
	case FLEX_SKEIN:     skein512_cpu_hash_80(thr_id, threads, nonce, d_h, 1); break;
	case FLEX_LUFFA:     qubit_luffa512_cpu_hash_80(thr_id, threads, nonce, d_h, 0); break;
	case FLEX_CUBEHASH:  cubehash512_cuda_hash_80(thr_id, threads, nonce, d_h); break;
	case FLEX_SHAVITE:   x16_shavite512_cpu_hash_80(thr_id, threads, nonce, d_h, 0); break;
	case FLEX_SIMD:      x16_simd512_cuda_hash_80(thr_id, threads, nonce, d_h); break;
	case FLEX_ECHO:      x16_echo512_cuda_hash_80(thr_id, threads, nonce, d_h); break;
	case FLEX_HAMSI:     x16_hamsi512_cuda_hash_80(thr_id, threads, nonce, d_h); break;
	case FLEX_FUGUE:     x16_fugue512_cuda_hash_80(thr_id, threads, nonce, d_h); break;
	case FLEX_SHABAL:    x16_shabal512_cuda_hash_80(thr_id, threads, nonce, d_h); break;
	case FLEX_WHIRLPOOL: x16_whirlpool512_hash_80(thr_id, threads, nonce, d_h); break;
	}
}

static void flex_stage_hash_64(int algo, int thr_id, uint32_t threads, uint32_t *d_h, int order)
{
	switch (algo) {
	case FLEX_BLAKE:     blake512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_BMW:       bmw512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_GROESTL:   groestl512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_KECCAK: {
		const uint32_t tpb = 128;
		flex_sha3_512_64_gpu <<<dim3((threads + tpb - 1) / tpb), dim3(tpb)>>>
			(threads, (uint2*)d_h);
		break;
	}
	case FLEX_SKEIN:     skein512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_LUFFA:     luffa512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_CUBEHASH:  cubehash512_cpu_hash_64(thr_id, threads, d_h); break;
	case FLEX_SHAVITE:   x11_shavite512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_SIMD:      simd512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_ECHO:      echo512_cpu_hash_64(thr_id, threads, d_h); break;
	case FLEX_HAMSI:     hamsi512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_FUGUE:     fugue512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_SHABAL:    shabal512_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	case FLEX_WHIRLPOOL: x15_whirlpool_cpu_hash_64(thr_id, threads, 0, NULL, d_h, order); break;
	}
}

static void flex_stage_init_all(int thr_id, uint32_t threads)
{
	blake512_cpu_init(thr_id, threads);
	bmw512_cpu_init(thr_id, threads);
	groestl512_cpu_init(thr_id, threads);
	skein512_cpu_init(thr_id, threads);
	qubit_luffa512_cpu_init(thr_id, threads);
	luffa512_cpu_init(thr_id, threads);
	x11_shavite512_cpu_init(thr_id, threads);
	simd512_cpu_init(thr_id, threads);
	x11_echo512_cpu_init(thr_id, threads);
	hamsi512_cpu_init(thr_id, threads);
	fugue512_cpu_init(thr_id, threads);
	shabal512_cpu_init(thr_id, threads);
	whirlpool512_cpu_init(thr_id, threads, 0);
	x16_echo512_cpu_init(thr_id, threads);        /* for x16_echo512_cuda_hash_80 */
	x16_fugue512_cpu_init(thr_id, threads);        /* for x16_fugue512_cuda_hash_80 */
	x16_whirlpool512_cpu_init(thr_id, threads);        /* for x16_whirlpool512_hash_80
	 *
	 * No JH: flex's pool does not contain it. */
}

static void flex_stage_free_all(int thr_id)
{
	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);
	simd512_cpu_free(thr_id);
	fugue512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id);
	x15_whirlpool_cpu_free(thr_id);
}

/* ===========================================================================
 * One CryptoNight round over the whole batch, lanes sorted by variant.
 * ===========================================================================
 *
 * Scratchpad bytes one CN round needs, given the per-lane variants. */
static size_t flex_cn_round_bytes(const flex_ctx *c, uint32_t lanes, uint32_t cn_round)
{
	const uint8_t *row = c->h_cn + (size_t)cn_round * c->threads;
	size_t total = 0;
	for (uint32_t t = 0; t < lanes; t++)
		total += flex_cn_mem[row[t]];
	return total;
}

static void flex_cn_round(flex_ctx *c, int thr_id, uint32_t cn_round)
{
	const uint32_t T = c->lanes;

	/* Sort lanes by the variant THIS round needs, and compact. */
	flex_bucket_round(T, c->threads, c->h_cn, cn_round, FLEX_CN_ALGO_COUNT, c->h_idx, c->h_off);
	cudaMemcpy(c->d_idx, c->h_idx, sizeof(uint32_t) * T, cudaMemcpyHostToDevice);
	flex_gather64(T, c->d_scratch, c->d_hash, c->d_idx);

	/* prepare and final are variant-independent, so they run over the whole
	 * compacted batch; ctx slot p then belongs to compact slot p. */
	cryptonight_extra_cpu_prepare_gr(thr_id, T, (uint64_t*)c->d_scratch,
		c->d_ctx_state, c->d_ctx_a, c->d_ctx_b, c->d_ctx_key1, c->d_ctx_key2, c->d_ctx_tweak);
	flex_prof_mark(FLEX_PH_CN_PREP);

	/* ---- variable-stride packing ----
	 * Each variant group gets its OWN stride, and the groups are laid end to
	 * end by a running byte offset. A uniform 2 MiB stride would reserve the
	 * largest variant's scratchpad for every lane; since the variant is chosen
	 * per nonce, ~5 of every 6 lanes then sit on memory they never touch.
	 *
	 * Legal because `stride64` is already a per-call argument and every phase
	 * kernel indexes purely relative to the pointers it is handed -- the group
	 * base is applied here, by the caller, exactly as for the ctx arrays. */
	size_t byteoff = 0;
	const bool serial = flex_cn_serial_enabled();
	int ng = 0;
	uint32_t g_lanes[FLEX_CN_ALGO_COUNT], g_stride[FLEX_CN_ALGO_COUNT];
	int g_var[FLEX_CN_ALGO_COUNT];
	uint64_t *g_ls[FLEX_CN_ALGO_COUNT], *g_st[FLEX_CN_ALGO_COUNT], *g_tw[FLEX_CN_ALGO_COUNT];
	uint32_t *g_a[FLEX_CN_ALGO_COUNT], *g_b[FLEX_CN_ALGO_COUNT];
	uint32_t *g_k1[FLEX_CN_ALGO_COUNT], *g_k2[FLEX_CN_ALGO_COUNT];
	for (uint32_t v = 0; v < FLEX_CN_ALGO_COUNT; v++) {
		const uint32_t off = c->h_off[v];
		const uint32_t cnt = c->h_off[v + 1] - off;
		if (!cnt) continue;

		const uint32_t stride64 = flex_cn_mem[v] >> 3;

		if (serial) {
			const int blocks = (int)((cnt + FLEX_CN_TPB - 1) / FLEX_CN_TPB);
			cryptonight_core_cuda_flex(thr_id, blocks, FLEX_CN_TPB, cnt, (int)v, stride64,
				c->d_long_state + (byteoff >> 3),
				c->d_ctx_state  + (size_t)off * 26,
				c->d_ctx_a      + (size_t)off * 4,
				c->d_ctx_b      + (size_t)off * 4,
				c->d_ctx_key1   + (size_t)off * 40,
				c->d_ctx_key2   + (size_t)off * 40,
				c->d_ctx_tweak  + off);
			flex_prof_group((int)v, cnt);
		} else {
			g_lanes[ng] = cnt; g_var[ng] = (int)v; g_stride[ng] = stride64;
			g_ls[ng] = c->d_long_state + (byteoff >> 3);
			g_st[ng] = c->d_ctx_state  + (size_t)off * 26;
			g_a[ng]  = c->d_ctx_a      + (size_t)off * 4;
			g_b[ng]  = c->d_ctx_b      + (size_t)off * 4;
			g_k1[ng] = c->d_ctx_key1   + (size_t)off * 40;
			g_k2[ng] = c->d_ctx_key2   + (size_t)off * 40;
			g_tw[ng] = c->d_ctx_tweak  + off;
			ng++;
		}

		byteoff += (size_t)cnt * flex_cn_mem[v];
	}
	if (ng)
		cryptonight_core_cuda_flex_multi(thr_id, FLEX_CN_TPB, ng, g_lanes, g_var, g_stride,
			g_ls, g_st, g_a, g_b, g_k1, g_k2, g_tw);
	flex_prof_mark(FLEX_PH_CN_CORE);

	cryptonight_extra_cpu_final_flex(thr_id, T, c->d_ctx_state, (uint64_t*)c->d_scratch);

	/* 64 bytes back, not 32: on the blake branch the high half is this
	 * round's own input and the closing SHA3-256 hashes all of it. */
	flex_scatter64(T, c->d_hash, c->d_scratch, c->d_idx);
	flex_prof_mark(FLEX_PH_CN_FINAL);
}

/* ===========================================================================
 * The whole chain for one batch.
 *
 * On return c->d_hash holds, per lane, the 32-byte digest in its low half.
 *
 * `steps` bounds how much of the chain to execute, counting
 * 0..14  -> core round 0..14
 * the CN round that follows core rounds 4, 9 and 14
 * finally the closing SHA3-256
 * for FLEX_PIPELINE_STEPS steps in total.  Pass FLEX_PIPELINE_STEPS (or -1)
 * to run everything; a smaller value stops early, which is what lets a
 * failing batch be bisected to a single round instead of just reported.
 * =========================================================================== */
#define FLEX_PIPELINE_STEPS (FLEX_CORE_CHAIN_LEN + FLEX_CN_ROUNDS + 1)   /* 19 */

/* A job change mid-batch retires the header, so the rest of the batch cannot
 * produce a share. Checked after round 0 and each CN round; the sync makes
 * the check see the change when it happens. An abandoned batch reports zero
 * lanes. */
static inline bool flex_pipeline_abandoned(flex_ctx *c, int thr_id)
{
	if (!c->abort_on_restart) return false;
	cudaDeviceSynchronize();
	if (!work_restart[thr_id].restart) return false;
	c->lanes = 0;
	return true;
}

/* `d_nonces` is the nonce filter's compacted list, or NULL for the contiguous
 * path. When it is set, `lanes` is the filter's YIELD and `startNonce` is the
 * base of the SCAN SPAN (needed to chunk round 0), not the first lane's nonce. */
static void flex_pipeline_run(flex_ctx *c, int thr_id, uint32_t *endiandata,
                              uint32_t *pdata, uint32_t startNonce, int steps,
                              const uint32_t *d_nonces, uint32_t filtered_lanes)
{
	const uint32_t T = c->threads;          /* capacity: derive for all of it */
	const uint32_t tpb = 128;
	int order = 0;
	int step = 0;
	if (steps < 0) steps = FLEX_PIPELINE_STEPS;
	c->lanes = d_nonces ? filtered_lanes : T;

	/* 1. per-lane chain derivation. The filtered path derives over the
	 * YIELD only, and reuses the same kernel so the walk and the SoA layout
	 * stay singly-sourced. */
	flex_seed_setBlock_80(endiandata);
	if (d_nonces)
		flex_seed_order_nonces_cpu(c->lanes, T, d_nonces, c->d_core, c->d_cn, NULL);
	else
		flex_seed_order_cpu(T, startNonce, c->d_core, c->d_cn, NULL);
	cudaMemcpy(c->h_core, c->d_core, (size_t)FLEX_CORE_CHAIN_LEN * T, cudaMemcpyDeviceToHost);
	cudaMemcpy(c->h_cn,   c->d_cn,   (size_t)FLEX_CN_ALGO_COUNT  * T, cudaMemcpyDeviceToHost);
	/* The row pitch stays the CAPACITY T even when only `lanes` rows are
	 * populated -- flex_bucket_round() takes lanes and stride separately for
	 * exactly this reason (an earlier phase fatal bug was conflating them). */

	/* 1b. Fit the batch to the scratchpad.
	 *
	 * The lane count is sized from the MEAN variant cost, so an unlucky batch
	 * -- too many `fast` lanes -- can need more than the budget. Trim the tail
	 * until the worst of the three CN rounds fits. This is done HERE, before
	 * any hashing, and it is legal because each lane's chain is a function of
	 * its own nonce alone: dropping the tail changes nothing about the lanes
	 * that remain, and scanhash simply advances the cursor by fewer nonces.
	 *
	 * With the lane counts this runs at the deviation is tiny (sd of the total
	 * is ~sqrt(T) * 627 KiB, i.e. well under 1% at T ~ 14000), so the loop
	 * should essentially never fire -- it exists so that "essentially" is not
	 * load-bearing. */
	{
		/* Start from the ACTIVE lane count, not the capacity. On the the nonce filter
		 * filtered path c->lanes is the filter's yield and is usually far below
		 * T; seeding this loop with T silently promoted the batch back to the
		 * full capacity and ran lanes for which no nonce exists. */
		uint32_t lanes = c->lanes;
		while (lanes > FLEX_CN_TPB) {
			size_t worst = 0;
			for (uint32_t k = 0; k < FLEX_CN_ROUNDS; k++) {
				const size_t need = flex_cn_round_bytes(c, lanes, k);
				if (need > worst) worst = need;
			}
			if (worst <= c->scratch_bytes)
				break;
			/* Scale down by the overshoot, with a little slack, rather than
			 * stepping one lane at a time. */
			const uint32_t next = (uint32_t)((double)lanes * 0.98 * (double)c->scratch_bytes / (double)worst);
			lanes = (next < lanes) ? next : lanes - 1;
			lanes = (lanes / FLEX_CN_TPB) * FLEX_CN_TPB;
		}
		if (lanes != c->lanes && opt_debug)
			applog(LOG_DEBUG, "flex: batch trimmed %u -> %u lanes for the variant mix",
			       c->lanes, lanes);
		c->lanes = lanes;
	}

	/* From here on L is the ACTIVE lane count -- everything downstream of the
	 * trim must use it, never the capacity. */
	const uint32_t L = c->lanes;
	const dim3 grid((L + tpb - 1) / tpb), block(tpb);
	flex_prof_mark(FLEX_PH_ORDER);

	/* 2. round 0: each of the 14 over the whole batch, keep the lanes that
	 * wanted it (see flex_select_round0_gpu for why). */
	if (step++ >= steps) return;
	if (!d_nonces) {
		for (uint32_t a = 0; a < FLEX_CORE_ALGO_COUNT; a++) {
			flex_stage_setBlock_80((int)a, thr_id, endiandata, pdata);
			flex_stage_hash_80((int)a, thr_id, L, startNonce, c->d_scratch);
			flex_select_round0_gpu <<<grid, block>>>
				(L, (uint4*)c->d_hash, (const uint4*)c->d_scratch, c->d_core, a);
		}
	} else {
		/* the nonce filter: the accepted lanes are scattered across the scan span, so walk
		 * the span in T-sized chunks and gather. Only the FOUR core[0] values
		 * the filter admits can occur, so this is 4 stages per chunk, not 14 --
		 * which is what keeps the overhead near 5% instead of 18%. */
		static const uint32_t f_algos[4] =
			{ FLEX_SKEIN, FLEX_LUFFA, FLEX_HAMSI, FLEX_FUGUE };

		/* Running four stages instead of fourteen is an OPTIMISATION resting on
		 * a precondition, so check it rather than assume it: a lane whose
		 * core[0] is not admitted never gets a round-0 stage and is hashed
		 * wrong -- silently, because a wrong lane just fails the target screen.
		 * It cost a live regression once (the filter ran before the header was
		 * uploaded, so it selected on the previous job: 375 of 512 lanes). */
		uint32_t unadmitted = 0;
		for (uint32_t t = 0; t < L; t++)
			if (!((FLEX_FILTER_CORE_MASK >> c->h_core[t]) & 1u)) unadmitted++;
		if (unadmitted) {
			applog(LOG_ERR, "flex: %u of %u filtered lanes have an unadmitted core[0] -- "
			       "the filter and the chain disagree on the header", unadmitted, L);
			c->lanes = 0;
			return;
		}

		const uint32_t span = c->span;
		for (uint32_t base = 0; base < span; base += T) {
			const uint32_t chunk = (span - base < T) ? (span - base) : T;
			for (uint32_t k = 0; k < 4; k++) {
				const uint32_t a = f_algos[k];
				flex_stage_setBlock_80((int)a, thr_id, endiandata, pdata);
				flex_stage_hash_80((int)a, thr_id, chunk, startNonce + base, c->d_scratch);
				flex_gather_round0_gpu <<<grid, block>>>
					(L, startNonce + base, chunk, (uint4*)c->d_hash,
					 (const uint4*)c->d_scratch, d_nonces, c->d_core, a);
			}
		}
	}

	flex_prof_mark(FLEX_PH_ROUND0);
	if (flex_pipeline_abandoned(c, thr_id)) return;

	/* 3. rounds 1..14, with a CN round closing each group of five. */
	for (uint32_t r = 1; r < FLEX_CORE_CHAIN_LEN; r++) {
		if (step++ >= steps) return;
		if (r == FLEX_CORE_ALGO_COUNT) {
			/* Slot 14 is always blake512 -- no bucketing needed. */
			flex_stage_hash_64(FLEX_BLAKE, thr_id, L, c->d_hash, order++);
		} else {
			flex_bucket_round(L, c->threads, c->h_core, r, FLEX_CORE_ALGO_COUNT, c->h_idx, c->h_off);
			cudaMemcpy(c->d_idx, c->h_idx, sizeof(uint32_t) * L, cudaMemcpyHostToDevice);
			flex_gather64(L, c->d_scratch, c->d_hash, c->d_idx);
			for (uint32_t a = 0; a < FLEX_CORE_ALGO_COUNT; a++) {
				const uint32_t off = c->h_off[a];
				const uint32_t cnt = c->h_off[a + 1] - off;
				if (!cnt) continue;
				/* Offset 0 of a compacted run: the stage sees exactly the
				 * shape x16r gives it. */
				flex_stage_hash_64((int)a, thr_id, cnt, c->d_scratch + (size_t)off * 16, order++);
			}
			flex_scatter64(L, c->d_hash, c->d_scratch, c->d_idx);
		}

		flex_prof_mark(FLEX_PH_CORE);
		if (r == 4 || r == 9 || r == 14) {
			if (step++ >= steps) return;
			flex_cn_round(c, thr_id, (r == 4) ? 0 : (r == 9) ? 1 : 2);
			if (flex_pipeline_abandoned(c, thr_id)) return;
		}
	}

	/* 4. the closing SHA3-256 over all 64 bytes */
	if (step++ >= steps) return;
	flex_sha3_256_final_gpu <<<grid, block>>> (L, (uint2*)c->d_hash);
	flex_prof_mark(FLEX_PH_SHA3);
}

#endif /* __CUDACC__ */

#endif /* CUDA_FLEX_PIPELINE_CUH */
