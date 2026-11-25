/*
 * balloon - init-time consensus gate. Fails closed via cuda/selftest_gate.cuh.
 *
 * Four legs, all required:
 *   0. vector    - the CPU reference still reproduces a pinned digest. The other
 *                  legs only prove GPU == CPU; this one catches the reference
 *                  itself drifting. See BALLOON_KAT_DIGEST for its provenance.
 *   1. agreement - the real launcher returns the lowest nonce in a span that
 *                  passes the screen, as enumerated on the host. Driving the
 *                  launcher rather than a copy also covers the grid math, the
 *                  screen, the per-work uploads and the "no candidate" sentinel,
 *                  and unlike a digest check it can see a MISSED nonce.
 *   2. range     - re-run with max_nonce one below that winner; the range guard
 *                  must exclude it. The span is not a whole number of blocks, so
 *                  the final block is partial in every leg.
 *   3. negative  - perturb the DEVICE input by one bit and require a failure.
 *                  A self-test that cannot fail is not a gate.
 */

#include <stdint.h>
#include <string.h>

#include "miner.h"
#include "balloon.h"
#include "cuda/selftest_gate.cuh"

extern void balloon_setBlock_80(int thr_id, void *pdata, const void *pTargetIn);
extern "C" void reset_host_prebuf(int thr_id);
uint32_t balloon_cpu_hash(int thr_id, unsigned char *input, uint32_t threads,
	uint32_t max_nonce);

/* 128 threads (2 blocks of 64) with 100 nonces in range => the last block is
 * 36/64 live. Small enough that the host enumeration costs ~0.1 s. */
#define BALLOON_ST_THREADS 128u
#define BALLOON_ST_LIVE    100u
#define BALLOON_ST_START   0x0000d1c7u   /* non-zero, not block-aligned */

/* Deterministic stand-in for a block header; only has to be fixed. */
static void balloon_st_header(uint32_t hdr[20])
{
	for (int i = 0; i < 20; i++)
		hdr[i] = 0x5a5a5a5au ^ (0x01010101u * (uint32_t)i);
}

static inline void balloon_st_set_nonce(uint32_t hdr[20], uint32_t nonce)
{
	/* The launcher reads the batch's first nonce out of bytes 76..79, big-endian. */
	be32enc(&hdr[19], nonce);
}

/* Known-answer vector for the CPU reference, over the header 0x00..0x4f.
 *
 * Provenance: the digest was NOT captured from this code. It was produced by a
 * second implementation written from the algorithm description -- another
 * language, its own AES and SHA-256 -- and the two agree byte for byte. So an
 * edit to balloon.cpp or sha256-ref.c that changes the digest fails here at
 * startup, rather than being discovered by a pool rejecting shares.
 */
static const uint8_t BALLOON_KAT_DIGEST[32] = {
	0x85,0xa5,0x6b,0x45, 0xdd,0xee,0xe7,0x0c, 0xbf,0xd1,0xc6,0x1e, 0x66,0x93,0x44,0xb7,
	0xb6,0x82,0xf8,0xe8, 0x17,0xa5,0xa1,0xc8, 0x43,0x00,0x07,0x72, 0xe4,0x10,0x8b,0xcc
};

/* Returns false if the CPU reference no longer reproduces the pinned vector. */
static bool balloon_kat_ok(int thr_id)
{
	uint8_t hdr[80];
	for (int i = 0; i < 80; i++) hdr[i] = (uint8_t)i;
	uint32_t _ALIGN(64) out[8];
	balloon_reset();
	balloon_128_orig(hdr, (unsigned char *)out);
	if (!memcmp(out, BALLOON_KAT_DIGEST, 32))
		return true;
	const uint8_t *g = (const uint8_t *)out;
	gpulog(LOG_ERR, thr_id, "balloon KAT FAILED: the CPU reference no longer matches"
		" the pinned vector -- got %02x%02x%02x%02x..., expected %02x%02x%02x%02x...",
		g[0], g[1], g[2], g[3],
		BALLOON_KAT_DIGEST[0], BALLOON_KAT_DIGEST[1],
		BALLOON_KAT_DIGEST[2], BALLOON_KAT_DIGEST[3]);
	return false;
}

extern "C" bool balloon_selftest(int thr_id)
{
	uint32_t _ALIGN(128) hdr[20];
	uint32_t _ALIGN(64)  vhash[8];
	uint32_t _ALIGN(64)  target[8];

	/* Not static: one miner thread per GPU runs this concurrently. 400 B of stack. */
	uint32_t w7[BALLOON_ST_LIVE];

	/* ---- leg 0: does the CPU reference still match the pinned vector? ---- */
	const bool leg_kat = balloon_kat_ok(thr_id);

	balloon_st_header(hdr);

	/* ---- host enumeration of the whole span ----------------------------- */
	uint32_t w7min = 0xffffffffu;
	for (uint32_t i = 0; i < BALLOON_ST_LIVE; i++) {
		balloon_st_set_nonce(hdr, BALLOON_ST_START + i);
		balloon_reset();
		balloon_128_orig((unsigned char *)hdr, (unsigned char *)vhash);
		/* vhash[7] is the same word the kernel screens: hash_state_extract()
		 * copies the last buffer block, which is what the kernel indexes. */
		w7[i] = vhash[7];
		if (w7[i] < w7min) w7min = w7[i];
	}

	if (w7min == 0xffffffffu) {
		/* Would make the screen threshold below unrepresentable. Not a wrong
		 * answer, so do not fail closed on it. */
		gpulog(LOG_WARNING, thr_id, "balloon self-test: degenerate span, skipped");
		return true;
	}

	/* Threshold taken from the data, so the span provably contains a candidate;
	 * a fixed constant risks a leg that silently never fires. */
	const uint32_t targ7 = w7min + 1u;

	uint32_t expect_full = UINT32_MAX;
	for (uint32_t i = 0; i < BALLOON_ST_LIVE; i++) {
		if (w7[i] < targ7) { expect_full = BALLOON_ST_START + i; break; }
	}

	memset(target, 0, sizeof(target));
	target[7] = targ7;

	/* ---- leg 1: agreement over the full span ---------------------------- */
	balloon_st_header(hdr);
	balloon_st_set_nonce(hdr, BALLOON_ST_START);
	reset_host_prebuf(thr_id);
	balloon_reset();
	balloon_setBlock_80(thr_id, hdr, target);
	const uint32_t got_full = balloon_cpu_hash(thr_id, (unsigned char *)hdr,
		BALLOON_ST_THREADS, BALLOON_ST_START + BALLOON_ST_LIVE - 1u);
	const bool leg_agree = (got_full == expect_full);

	/* ---- leg 2: the range guard must exclude that winner ----------------- */
	bool leg_tail = true;
	if (expect_full != UINT32_MAX && expect_full > BALLOON_ST_START) {
		const uint32_t cap = expect_full - 1u;
		uint32_t expect_tail = UINT32_MAX;
		for (uint32_t i = 0; i < BALLOON_ST_LIVE; i++) {
			const uint32_t n = BALLOON_ST_START + i;
			if (n > cap) break;
			if (w7[i] < targ7) { expect_tail = n; break; }
		}
		balloon_st_header(hdr);
		balloon_st_set_nonce(hdr, BALLOON_ST_START);
		reset_host_prebuf(thr_id);
		balloon_reset();
		balloon_setBlock_80(thr_id, hdr, target);
		const uint32_t got_tail = balloon_cpu_hash(thr_id, (unsigned char *)hdr,
			BALLOON_ST_THREADS, cap);
		leg_tail = (got_tail == expect_tail);
		if (!leg_tail)
			gpulog(LOG_ERR, thr_id, "balloon self-test: range guard leaked or dropped"
				" (max_nonce %08x: expected %08x, got %08x)", cap, expect_tail, got_tail);
	}

	/* ---- leg 3: the gate must be able to fail ---------------------------- */
	balloon_st_header(hdr);
	hdr[3] ^= 1u;                      /* perturbs what the DEVICE hashes */
	balloon_st_set_nonce(hdr, BALLOON_ST_START);
	reset_host_prebuf(thr_id);
	balloon_reset();
	balloon_setBlock_80(thr_id, hdr, target);
	const uint32_t got_neg = balloon_cpu_hash(thr_id, (unsigned char *)hdr,
		BALLOON_ST_THREADS, BALLOON_ST_START + BALLOON_ST_LIVE - 1u);
	const bool leg_neg = (got_neg != expect_full);

	/* Leave device state as found: a real launch must not inherit the test header. */
	reset_host_prebuf(thr_id);
	balloon_reset();

	const bool passed = leg_kat && leg_agree && leg_tail && leg_neg;

	if (!passed) {
		gpulog(LOG_ERR, thr_id, "balloon self-test FAILED (kat=%d agree=%d tail=%d neg=%d;"
			" target7=%08x expected=%08x got=%08x)",
			(int)leg_kat, (int)leg_agree, (int)leg_tail, (int)leg_neg,
			targ7, expect_full, got_full);
	} else if (opt_debug) {
		gpulog(LOG_DEBUG, thr_id, "balloon self-test OK (KAT + %u nonces enumerated,"
			" lowest passing %08x at target7 %08x)",
			BALLOON_ST_LIVE, expect_full, targ7);
	}

	return selftest_gate(thr_id, "balloon", passed);
}
