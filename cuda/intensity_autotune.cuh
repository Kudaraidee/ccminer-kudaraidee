// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/*
 * Runtime launch-size (intensity) auto-tuner.
 *
 * The best launch size depends on the card, which is only known at run time, so
 * this measures the candidates on the device that is running and keeps the best.
 * It scores hashes/second over a whole batch (launch + sync + host work). Large
 * batches also mean more discarded work on a job change, which a throughput
 * measurement cannot see -- hence max_batch_ms rather than timing alone.
 *
 * Method: candidates are visited forward then reversed so each is sampled at two
 * points on the thermal ramp; the first batch of a visit is dropped as a resize
 * transient; the median is used; a batch that found a share is dropped, since it
 * paid a host re-verify unrelated to launch size.
 *
 * Guarantees: does nothing if the user passed -i; never returns more than
 * intensity_tuner_init() reported (allocate for that); every candidate is a
 * multiple of `granularity`; bounded in both batches and seconds.
 *
 * Usage:
 *   static intensity_tuner_t s_tuner[MAX_GPUS];
 *   uint32_t tp_max = intensity_tuner_init(&s_tuner[thr_id], thr_id,
 *                        cuda_default_throughput(thr_id, 1U<<15), 256, 250.0);
 *   ... allocate for tp_max ...
 *   uint32_t tp = intensity_tuner_size(&s_tuner[thr_id]);
 *   double t0 = tuner_now();
 *   ... launch tp nonces, sync, read results ...
 *   intensity_tuner_sample(&s_tuner[thr_id], tuner_now() - t0, found_a_candidate);
 *
 * Include miner.h first (gpus_intensity, gpulog, throughput2intensity).
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifndef INTENSITY_TUNER_MAX_CAND
#define INTENSITY_TUNER_MAX_CAND 6
#endif
#ifndef INTENSITY_TUNER_KEEP
#define INTENSITY_TUNER_KEEP 2          /* retained samples per visit */
#endif

struct intensity_tuner_t {
	int      active;                 /* 0 = disabled (explicit -i, or done) */
	int      thr_id;
	int      ncand;
	int      visit;                  /* index into the forward+reverse schedule */
	int      nvisits;
	int      got;                    /* samples taken at the current visit */
	uint32_t cand[INTENSITY_TUNER_MAX_CAND];
	double   samp[INTENSITY_TUNER_MAX_CAND][2 * INTENSITY_TUNER_KEEP];
	int      nsamp[INTENSITY_TUNER_MAX_CAND];
	double   max_batch_ms;
	double   spent;                  /* seconds burned tuning */
	double   budget;                 /* seconds allowed */
	uint32_t chosen;
};

static inline double tuner_now(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double) tv.tv_sec + 1e-6 * (double) tv.tv_usec;
}

static inline int tuner_dcmp(const void *a, const void *b)
{
	const double x = *(const double *) a, y = *(const double *) b;
	return (x > y) - (x < y);
}

/* schedule position -> candidate index: forward then reversed */
static inline int tuner_slot(const intensity_tuner_t *t, int visit)
{
	return (visit < t->ncand) ? visit : (2 * t->ncand - 1 - visit);
}

/* Returns the LARGEST size the tuner may ever ask for; allocate for this. */
static inline uint32_t intensity_tuner_init(intensity_tuner_t *t, int thr_id,
                                            uint32_t base, uint32_t granularity,
                                            double max_batch_ms)
{
	static const int shift[INTENSITY_TUNER_MAX_CAND] = { -2, -1, 0, 1, 2, 3 };
	int i, n = 0;

	memset(t, 0, sizeof(*t));
	t->thr_id = thr_id;
	t->chosen = base;
	t->max_batch_ms = max_batch_ms;
	t->budget = 60.0;

	/* An explicit -i is a decision, not a default: never override it. */
	if (gpus_intensity[thr_id] != 0) {
		t->active = 0;
		return base;
	}

	for (i = 0; i < INTENSITY_TUNER_MAX_CAND; i++) {
		int64_t v = (shift[i] < 0) ? (int64_t) (base >> (-shift[i]))
		                           : (int64_t) base << shift[i];
		v -= v % (int64_t) granularity;
		if (v < (int64_t) granularity || v > 0x40000000LL) continue;
		if (n && t->cand[n - 1] == (uint32_t) v) continue;   /* granularity collapse */
		t->cand[n++] = (uint32_t) v;
	}
	t->ncand = n;
	t->nvisits = 2 * n;
	t->active = (n > 1);
	return t->cand[n - 1];
}

/* Size to launch for the next batch. */
static inline uint32_t intensity_tuner_size(const intensity_tuner_t *t)
{
	if (!t->active) return t->chosen;
	return t->cand[tuner_slot(t, t->visit)];
}

static inline uint32_t intensity_tuner_chosen(const intensity_tuner_t *t)
{
	return t->chosen;
}

static inline int intensity_tuner_running(const intensity_tuner_t *t)
{
	return t->active;
}

/* Feed one batch back. `tainted` = the batch did something a normal batch would
 * not (found a candidate, hit a job restart). */
static inline void intensity_tuner_sample(intensity_tuner_t *t, double secs, int tainted)
{
	int slot, i, best = 0;
	double med[INTENSITY_TUNER_MAX_CAND], tmp[2 * INTENSITY_TUNER_KEEP];

	if (!t->active) return;

	t->spent += secs;
	slot = tuner_slot(t, t->visit);

	/* first sample of each visit is the post-switch transient */
	if (!tainted && secs > 0.0) {
		if (t->got > 0 && t->nsamp[slot] < 2 * INTENSITY_TUNER_KEEP)
			t->samp[slot][t->nsamp[slot]++] = (double) t->cand[slot] / secs;
		t->got++;
	}

	if (t->got <= INTENSITY_TUNER_KEEP) {
		if (t->spent > t->budget) goto finish;   /* out of budget mid-visit */
		return;
	}

	t->got = 0;
	if (++t->visit < t->nvisits && t->spent <= t->budget)
		return;

finish:
	/* median per candidate, then the fastest whose batch is not overlong */
	for (i = 0; i < t->ncand; i++) {
		int k = t->nsamp[i];
		if (k <= 0) { med[i] = 0.0; continue; }
		memcpy(tmp, t->samp[i], sizeof(double) * (size_t) k);
		qsort(tmp, (size_t) k, sizeof(double), tuner_dcmp);
		med[i] = (k & 1) ? tmp[k / 2] : 0.5 * (tmp[k / 2 - 1] + tmp[k / 2]);
	}
	for (i = 0; i < t->ncand; i++) {
		double batch_ms;
		if (med[i] <= 0.0) continue;
		batch_ms = 1000.0 * (double) t->cand[i] / med[i];
		if (batch_ms > t->max_batch_ms) continue;     /* discarded-work guard */
		if (med[best] <= 0.0 || med[i] > med[best]) best = i;
	}
	if (med[best] > 0.0) t->chosen = t->cand[best];

	gpulog(LOG_INFO, t->thr_id, "intensity auto-tune: %.0f kH/s at %u nonces (i%.2f)",
	       med[best] / 1000.0, t->chosen, throughput2intensity(t->chosen));
	for (i = 0; i < t->ncand; i++) {
		if (med[i] <= 0.0) continue;
		gpulog(LOG_DEBUG, t->thr_id, "  i%.2f %8u nonces: %7.1f kH/s, batch %6.1f ms%s",
		       throughput2intensity(t->cand[i]), t->cand[i], med[i] / 1000.0,
		       1000.0 * (double) t->cand[i] / med[i],
		       (1000.0 * (double) t->cand[i] / med[i] > t->max_batch_ms) ? "  (too long)" : "");
	}
	t->active = 0;
}
