/**
 * Optional nonce pre-filter for flex.
 *
 * flex derives its CryptoNight triple from the nonce, so the miner can scan
 * cheap seeds and chain only the nonces whose three CN rounds are the cheapest
 * available. Any nonce is a valid nonce, so this is consensus-legal.
 *
 * Iteration units (gr_iters[] / 65536 in cryptonight-core.cu):
 *     dark 2  darklite 2  fast 4  lite 4  turtle 1  turtlelite 1
 * Three distinct variants, so the cheapest triple totals 4 units against an
 * unfiltered mean of 7.39.
 *
 * Constraints that are easy to get wrong:
 *  - TWO triples reach 4 units, not one: {dark,turtle,turtlelite} and
 *    {darklite,turtle,turtlelite}. Accepting only one halves the yield.
 *  - core[0] and cn[0] both reduce the FIRST seed nibble, so they are not
 *    independent. core[0] = keccak forces cn[0] = lite and a 4-unit triple
 *    becomes impossible; the admitted core[0] set below is exactly those whose
 *    nibble maps to a 1-unit CN variant.
 *  - Round 0 must run shared 80-byte stages over the whole scan span, because
 *    those derive nonce = startNounce + thread and cannot take a scattered set.
 *
 * Enabled by default; FLEX_NONCE_FILTER=0 takes the plain path.
 *
 * The caller must upload the header (flex_seed_setBlock_80) BEFORE running
 * this kernel. The admitted core[0] set below is a precondition the pipeline's
 * round 0 relies on to run four stages instead of fourteen, so a selection
 * made under a different header is not merely suboptimal -- those lanes never
 * get a round-0 stage. flex_pipeline_run() checks this and refuses the batch.
 */

#ifndef CUDA_FLEX_FILTER_CUH
#define CUDA_FLEX_FILTER_CUH

#include "flex.h"

/* CN iteration units packed one nibble per variant: dark 2, darklite 2,
 * fast 4, lite 4, turtle 1, turtlelite 1 -- i.e. gr_iters[] / 65536 from
 * algos/cryptonight/cryptonight-core.cu. Packed into ONE immediate rather than
 * a __constant__ table so the lookup costs a shift and no memory at all.
 * Must track gr_iters[]; flex_filter_units_selftest() checks it. */
#define FLEX_CN_UNITS_PACKED 0x114422u
#define FLEX_CN_UNITS(d)     (((FLEX_CN_UNITS_PACKED) >> (4u * (d))) & 0xFu)

/* The accepted core[0] set as a bitmask over FlexAlgo. */
#define FLEX_FILTER_CORE_MASK ( (1u << FLEX_SKEIN) | (1u << FLEX_LUFFA) | \
                                (1u << FLEX_HAMSI) | (1u << FLEX_FUGUE) )

/* Accept a chain whose three CN rounds total at most this many units. 4 is the
 * arithmetic minimum; raising it trades speedup for a shorter scan span. */
#define FLEX_FILTER_MAX_UNITS 4u

/* Acceptance: P(core[0] admitted AND the CN triple totals <= MAX_UNITS). A
 * property of the nibble reductions over a uniform hash, so it is the same on
 * every card and every toolkit -- NOT arch- or memory-dependent.
 *
 * Only a seed: scanhash re-estimates it from the observed yield each batch, so
 * changing FLEX_FILTER_CORE_MASK or FLEX_FILTER_MAX_UNITS cannot leave a stale
 * constant behind. Re-measure it from the TRUE yield, never from the clamped
 * one -- yield/span after the host clamp is just the clamp over the span. */
#define FLEX_FILTER_ACCEPT_SEED 0.03732

/* Sigma of binomial margin to carry on the yield. Under-yield costs lanes
 * directly; over-scan costs round-0 work linearly (the span is walked in
 * T-sized chunks at 4 stages each, so it is the dominant filtered-path
 * overhead, not the filter kernel). The asymmetry favours a generous k. */
#define FLEX_FILTER_SIGMA 5.0

/* Scan span needed to yield `lanes` accepted nonces at acceptance `accept`.
 *
 * The yield is Binomial(span, accept); require it >= lanes with k sigma of
 * headroom. With y = span*accept, solving y - k*sqrt(y) = lanes gives
 * sqrt(y) = (k + sqrt(k^2 + 4*lanes)) / 2.
 *
 * A fixed nonces-per-lane constant cannot serve both ends of the lane range,
 * because the margin is statistical and shrinks as sqrt(lanes): it over-scans
 * on a big card and under-yields on a small one. Lane count is sized from free
 * VRAM, so that is the indirect memory dependence. */
static inline uint32_t flex_filter_span(uint32_t lanes, double accept)
{
	if (!(accept > 1e-4)) accept = 1e-4;     /* also catches NaN */
	if (accept > 1.0) accept = 1.0;
	const double k = FLEX_FILTER_SIGMA;
	const double r = (k + sqrt(k * k + 4.0 * (double)lanes)) * 0.5;
	double span = (r * r) / accept;
	if (span > 268435456.0) span = 268435456.0;   /* 1<<28, keeps the cursor sane */
	if (span < (double)lanes) span = (double)lanes;
	return (uint32_t)(span + 0.5);
}

/* Fold one 32-bit seed word's eight nibbles into the running "first three
 * distinct CN variants" state.
 *
 * Written as unrolled shifts on REGISTERS, not as a byte array. The obvious
 * form (materialise uint8_t sb[64], then walk it with a runtime index) puts a
 * per-thread LOCAL FRAME on every lane -- and this kernel runs over ~31x more
 * lanes than any other in the algo, so it is the worst possible place for one.
 * cuda_flex_order.cuh carries the same warning for its `seen` bitmask. */
#define FLEX_F_NIB(v, sh) do { \
	if (n < 3u) { \
		const uint32_t d = (((v) >> (sh)) & 0x0Fu) % FLEX_CN_ALGO_COUNT; \
		if (!((seen >> d) & 1u)) { seen |= 1u << d; units += FLEX_CN_UNITS(d); n++; } \
	} } while (0)
#define FLEX_F_WORD(v) do { \
	FLEX_F_NIB(v, 0);  FLEX_F_NIB(v, 4);  FLEX_F_NIB(v, 8);  FLEX_F_NIB(v, 12); \
	FLEX_F_NIB(v, 16); FLEX_F_NIB(v, 20); FLEX_F_NIB(v, 24); FLEX_F_NIB(v, 28); \
	} while (0)

/**
 * One thread per candidate nonce: derive the seed, take core[0] and the CN
 * triple, and append the nonce to `d_nonces` if it passes.
 *
 * The append is the the candidate report shape -- many threads writing into shared slots --
 * so it is an atomicAdd with a hard bound, not a compare-and-store. Order is
 * NOT preserved and need not be: each accepted nonce is independent work, and
 * the host advances its cursor by the SPAN, never by a found nonce.
 */
static __global__
void flex_filter_gpu(uint32_t span, uint32_t startNonce, uint32_t maxOut,
                     uint32_t * __restrict__ d_nonces, uint32_t * __restrict__ d_count)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= span)
		return;

	const uint32_t nonce = startNonce + t;

	uint2 seed[8];
	flex_sha3_512_seed_80(nonce, seed);

	/* core[0] needs no walk: the walk takes the low nibble of byte 0 first
	 * with an empty `seen` set, so it is ALWAYS accepted and core[0] is just
	 * that nibble reduced. This is also why core[0] and cn[0] are correlated --
	 * they are two reductions of one nibble. */
	const uint32_t d0 = seed[0].x & 0x0Fu;
	if (!((FLEX_FILTER_CORE_MASK >> (d0 % FLEX_CORE_ALGO_COUNT)) & 1u))
		return;   /* rejects ~93% for one AND, before the CN walk */

	uint32_t seen = 0u, n = 0u, units = 0u;
	FLEX_F_WORD(seed[0].x); FLEX_F_WORD(seed[0].y);
	FLEX_F_WORD(seed[1].x); FLEX_F_WORD(seed[1].y);
	FLEX_F_WORD(seed[2].x); FLEX_F_WORD(seed[2].y);
	FLEX_F_WORD(seed[3].x); FLEX_F_WORD(seed[3].y);
	/* Back-fill, exactly as flex_walk_nibbles does. 32 bytes of nibbles failing
	 * to yield 3 of 6 values is astronomically unlikely and NOT impossible, and
	 * the CPU reference would take this branch, so the filter must too. */
	for (uint32_t d = 0; d < FLEX_CN_ALGO_COUNT && n < 3u; d++)
		if (!((seen >> d) & 1u)) { seen |= 1u << d; units += FLEX_CN_UNITS(d); n++; }

	if (units > FLEX_FILTER_MAX_UNITS)
		return;

	const uint32_t slot = atomicAdd(d_count, 1u);
	if (slot < maxOut)
		d_nonces[slot] = nonce;
	/* Past maxOut the count keeps rising, so the host reads the TRUE yield and
	 * can report a saturated batch instead of believing the span was exact. */
}

/**
 * Check the packed unit immediate against the AUTHORITATIVE per-variant
 * iteration counts, and re-derive the acceptance the span constant assumes.
 *
 * FLEX_CN_UNITS_PACKED is a hand-derived constant standing in for a table
 * that lives in ANOTHER FILE (gr_iters[] in algos/cryptonight/cryptonight-core.cu).
 * Nothing links them, so a future edit there silently makes this filter select
 * the wrong triples -- and the failure is invisible: the miner keeps hashing
 * correctly and merely loses the speedup, which no correctness gate can see.
 * Returns 0 on success, or the first offending variant index + 1.
 */
static inline int flex_filter_units_selftest(char *detail, size_t detail_len)
{
	/* Mirror of gr_iters[]; turtle (65536) is one unit. */
	static const uint32_t iters[FLEX_CN_ALGO_COUNT] =
		{ 131072u, 131072u, 262144u, 262144u, 65536u, 65536u };

	for (uint32_t d = 0; d < FLEX_CN_ALGO_COUNT; d++) {
		const uint32_t want = iters[d] / 65536u;
		const uint32_t got  = FLEX_CN_UNITS(d);
		if (want != got) {
			if (detail) snprintf(detail, detail_len,
				"CN variant %u: packed units %u but gr_iters implies %u",
				d, got, want);
			return (int)d + 1;
		}
	}

	/* The cheapest achievable triple must be exactly FLEX_FILTER_MAX_UNITS --
	 * otherwise the threshold accepts nothing (too low) or more than the
	 * cheapest class (too high), and the span constant is wrong either way. */
	uint32_t u[FLEX_CN_ALGO_COUNT];
	for (uint32_t d = 0; d < FLEX_CN_ALGO_COUNT; d++) u[d] = FLEX_CN_UNITS(d);
	for (uint32_t i = 0; i < FLEX_CN_ALGO_COUNT; i++)
		for (uint32_t j = i + 1; j < FLEX_CN_ALGO_COUNT; j++)
			if (u[j] < u[i]) { const uint32_t t = u[i]; u[i] = u[j]; u[j] = t; }
	const uint32_t cheapest = u[0] + u[1] + u[2];
	if (cheapest != FLEX_FILTER_MAX_UNITS) {
		if (detail) snprintf(detail, detail_len,
			"cheapest triple is %u units but the filter accepts <= %u",
			cheapest, (uint32_t)FLEX_FILTER_MAX_UNITS);
		return -1;
	}
	return 0;
}

static void flex_filter_cpu(uint32_t span, uint32_t startNonce, uint32_t maxOut,
                            uint32_t *d_nonces, uint32_t *d_count)
{
	const uint32_t tpb = 128;
	dim3 grid((span + tpb - 1) / tpb);
	dim3 block(tpb);
	flex_filter_gpu <<<grid, block>>> (span, startNonce, maxOut, d_nonces, d_count);
}

#endif /* CUDA_FLEX_FILTER_CUH */
