/**
 * Host-side lane scheduling for Flex.
 *
 * Because flex's chain is per-nonce, a batch of T lanes needs a different
 * algorithm at round r for (almost) every lane.  The GPU cannot be asked to
 * run "14 different stage kernels at once", so instead the host sorts the
 * lanes by the algorithm they need and the driver runs each bucket as its own
 * launch over a compacted buffer.  Total lane-rounds are unchanged; only the
 * launch granularity shrinks.
 *
 * This is a counting sort, and the same function serves both axes:
 * - core rounds, 14 buckets, one call per round r in 0..14;
 * - CN rounds, 6 buckets, one call per CN round.
 *
 * Header-only and dependency-free (stdint + string.h) so it can be unit-tested
 * without CUDA and included by the driver without a build-file entry.
 *
 * Gated in the project notes the bucketing is
 * checked for being a true permutation, for grouping (every lane in bucket a
 * really does need algo a), and against a gather/scatter round-trip on device.
 */

#ifndef FLEX_SCHEDULE_H
#define FLEX_SCHEDULE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Largest bucket count this header is used with (14 core algos). */
#define FLEX_MAX_BUCKETS 16

/**
 * Counting-sort the lanes of one round into buckets.
 *
 * @param lanes      how many lanes this batch actually hashes
 * @param stride     the row pitch of `order`, i.e. the CAPACITY it was
 * allocated with. Equals `lanes` only when nothing trimmed
 * the batch -- never assume it.
 * @param order      the derived orders, read back from the device in the SoA
 * layout flex_seed_order_gpu writes: order[round * stride
 * + lane].
 * @param round      which chain position to bucket by
 * @param nbuckets   14 for core rounds, 6 for CN rounds
 * @param idx    [out] T lane indices, grouped by bucket, ascending within a
 * bucket (a stable sort, so a run is reproducible)
 * @param off    [out] nbuckets+1 offsets; bucket a occupies
 * idx[off[a] .. off[a+1])
 *
 * Deterministic and in-place-free: no allocation, no RNG, so a failing batch
 * can be replayed exactly.
 */
static inline void flex_bucket_round(uint32_t lanes, uint32_t stride,
                                     const uint8_t *order,
                                     uint32_t round, uint32_t nbuckets,
                                     uint32_t *idx, uint32_t *off)
{
	uint32_t count[FLEX_MAX_BUCKETS + 1];
	uint32_t cursor[FLEX_MAX_BUCKETS];
	uint32_t a;

	memset(count, 0, sizeof count);

	/* `lanes` and `stride` are DIFFERENT NUMBERS and must stay separate.
	 * `order` is written by the device as [round][capacity], so the row pitch
	 * is the CAPACITY the array was allocated with -- not however many lanes
	 * this batch happens to use. They coincide only while nothing trims the
	 * batch, and an earlier phase made trimming routine: one parameter serving both roles
	 * silently read the wrong ROW, handing every lane another round's
	 * algorithm. Every digest was wrong, and the 256-lane startup guard could
	 * not see it because a small batch never trims. */
	const uint8_t *row = order + (size_t)round * stride;

	for (uint32_t t = 0; t < lanes; t++)
		count[row[t]]++;

	off[0] = 0;
	for (a = 0; a < nbuckets; a++) {
		off[a + 1] = off[a] + count[a];
		cursor[a] = off[a];
	}

	for (uint32_t t = 0; t < lanes; t++)
		idx[cursor[row[t]]++] = t;
}

/**
 * Check that a bucketing is well-formed.  Returns 0 on success, or a negative
 * code; `detail` (>= 128 bytes) receives a one-line reason.
 *
 * Exists because the two obvious ways to be wrong both survive a
 * gather/scatter round-trip:
 * - an IDENTITY idx round-trips perfectly and groups nothing;
 * - a duplicated lane index round-trips for the surviving copy and silently
 * drops the other lane's work.
 * So the round-trip is necessary and not sufficient; this is the other half.
 */
static inline int flex_check_bucketing(uint32_t lanes, uint32_t stride,
                                       const uint8_t *order,
                                       uint32_t round, uint32_t nbuckets,
                                       const uint32_t *idx, const uint32_t *off,
                                       char *detail, size_t detail_len)
{
	const uint32_t threads = lanes;       /* body below counts LANES */
	const uint8_t *row = order + (size_t)round * stride;

	if (off[0] != 0 || off[nbuckets] != threads) {
		if (detail) snprintf(detail, detail_len,
			"round %u: offsets span %u..%u, expected 0..%u",
			round, off[0], off[nbuckets], threads);
		return -1;
	}

	/* A permutation: every lane exactly once. */
	for (uint32_t t = 0; t < threads; t++) {
		if (idx[t] >= threads) {
			if (detail) snprintf(detail, detail_len,
				"round %u: idx[%u] = %u is out of range", round, t, idx[t]);
			return -2;
		}
	}
	{
		/* Bit-set rather than a byte array (T can be ~10^5), heap-allocated
		 * rather than a function-static: a static here would be 128 KB of
		 * per-TU BSS, would cap T, and would not be re-entrant. */
		const size_t words = ((size_t)threads + 31) / 32;
		uint32_t *seen = (uint32_t*)calloc(words, sizeof(uint32_t));
		if (!seen) {
			if (detail) snprintf(detail, detail_len,
				"round %u: out of memory for the permutation check", round);
			return -5;
		}
		for (uint32_t t = 0; t < threads; t++) {
			const uint32_t v = idx[t];
			if (seen[v >> 5] & (1u << (v & 31))) {
				if (detail) snprintf(detail, detail_len,
					"round %u: lane %u appears twice in idx", round, v);
				free(seen);
				return -3;
			}
			seen[v >> 5] |= 1u << (v & 31);
		}
		free(seen);
	}

	/* Grouping: every lane placed in bucket a must actually need algo a. */
	for (uint32_t a = 0; a < nbuckets; a++)
		for (uint32_t p = off[a]; p < off[a + 1]; p++)
			if (row[idx[p]] != a) {
				if (detail) snprintf(detail, detail_len,
					"round %u: lane %u sits in bucket %u but needs algo %u",
					round, idx[p], a, (uint32_t)row[idx[p]]);
				return -4;
			}

	return 0;
}

#endif /* FLEX_SCHEDULE_H */
