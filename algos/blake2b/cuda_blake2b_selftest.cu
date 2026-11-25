// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test for -a blake2b (docs/coding-guideline.md section 7,
 * layer 1). Runs once per device and FAILS CLOSED via
 * cuda/selftest_gate.cuh.
 *
 * Five legs, in the order they fail:
 *   1. KAT  - host reference against a digest produced outside this codebase.
 *      The construction is plain BLAKE2b-256 over the 80-byte header, so
 *      Python hashlib is the oracle.
 *   2. NEG  - one flipped header bit must change the digest.
 *   3. DIFF - GPU-vs-CPU over a nonce RANGE, over the shipping arithmetic.
 *   4. NEGD - the differential re-run with one contribution perturbed must
 *      fail. A gate whose trigger never occurs in normal operation is
 *      untested by construction.
 *   5. REPORT - the shipping launcher must return the two LOWEST candidates:
 *      once with every nonce a candidate, once with five in one block. The
 *      host resumes after the higher one, so a report that kept the first
 *      and last arrivals instead skipped everything between them.
 *
 * Leg 3 compares output word h[3], which is *all* the device computes: the
 * other seven words are never formed, since only one feeds the target screen.
 *
 * NOTE: on Windows the host reference comes from blake2b.cu's own #ifdef
 * _WIN32 block; on Linux it comes from sph/blake2b.c. This test therefore
 * exercises a different implementation on each platform - which is the reason
 * to run it on both, not a defect in the test.
 */

#include <string.h>
#include <algorithm>

#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/selftest_gate.cuh"

/* Host reference and the device-side pieces, all defined in blake2b.cu. */
extern "C" void blake2b_hash(void *output, const void *input);
extern void blake2b_setBlock(uint32_t *data);
extern void blake2b_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc);
extern uint32_t blake2b_hash_cuda(const int thr_id, const uint32_t threads, const uint32_t startNonce,
	const uint64_t target, uint32_t &secNonce);

/* BLAKE2b-256 of the 80-byte header built below, from Python hashlib
 * (hashlib.blake2b(msg, digest_size=32)) - an oracle outside this codebase. */
static const uint64_t kat_digest[4] = {
	0xB2C82E45D2BC722FULL, 0x3135B5AFFAE47644ULL,
	0x85DA021839B9C0FBULL, 0xF9D42D51AE249E58ULL,
};
#define KAT_NONCE 0x1234abcdu

/* endiandata[] exactly as scanhash_blake2b builds it: be32enc over all 20 words,
 * then word 19 overwritten with the raw nonce. */
static void blake2b_selftest_header(uint32_t *ed, uint32_t nonce)
{
	for (int i = 0; i < 20; i++)
		be32enc(&ed[i], (uint32_t)(0x01020304u + i));
	ed[19] = nonce;
}

/* The host half of the differential, kept here rather than shared with
 * scanhash's copy so a fault injected for leg 4 cannot reach the mining path. */
static void blake2b_selftest_cpu_acc(const uint32_t *ed, uint32_t startNonce,
	uint32_t count, uint64_t *cpu, uint32_t perturb_at)
{
	uint32_t _ALIGN(64) vhash[8], td[20];

	cpu[0] = cpu[1] = 0;
	memcpy(td, ed, sizeof(td));
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		td[19] = nonce;
		blake2b_hash(vhash, td);
		uint64_t w = ((uint64_t*)vhash)[3];
		if (i == perturb_at)
			w ^= 1ull; // leg 4 only: must break the comparison
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t)nonce + 1ull);
	}
}

/* Leg 5: does the launcher report (lo, hi) for this target over this range? */
static bool blake2b_selftest_report(int thr_id, uint32_t start, uint32_t count,
	uint64_t target, uint32_t lo, uint32_t hi)
{
	uint32_t sec = UINT32_MAX;
	const uint32_t first = blake2b_hash_cuda(thr_id, count, start, target, sec);
	return first == lo && sec == hi;
}

/* The 5th-lowest h[3] over one TPB-sized block, so exactly five candidates share
 * a block, and the two lowest NONCES among them - computed on the host. */
static uint64_t blake2b_selftest_block_target(const uint32_t *ed, uint32_t start,
	uint32_t *lo, uint32_t *hi)
{
	uint32_t _ALIGN(64) vhash[8], td[20];
	uint64_t w[512], sorted[512];

	memcpy(td, ed, sizeof(td));
	for (uint32_t i = 0; i < 512; i++) {
		td[19] = start + i;
		blake2b_hash(vhash, td);
		sorted[i] = w[i] = ((uint64_t*)vhash)[3];
	}
	std::sort(sorted, sorted + 512);
	const uint64_t target = sorted[4];
	*lo = *hi = UINT32_MAX;
	for (uint32_t i = 0; i < 512; i++) {
		if (w[i] > target) continue;
		if (*lo == UINT32_MAX) *lo = start + i;
		else if (*hi == UINT32_MAX) *hi = start + i;
	}
	return target;
}

__host__
bool blake2b_device_selftest(int thr_id)
{
	/* Per-device, not per-process: the GPU leg only proves the device that ran
	 * it, so a second card must be tested on its own. */
	static bool tested[MAX_GPUS] = { 0 }, passed[MAX_GPUS] = { 0 };
	if (tested[thr_id]) return passed[thr_id];
	tested[thr_id] = true;

	uint32_t _ALIGN(64) ed[20], dig[8];

	// --- leg 1: host reference vs the external oracle ---
	blake2b_selftest_header(ed, KAT_NONCE);
	blake2b_hash(dig, ed);
	const bool kat_ok = (memcmp(dig, kat_digest, sizeof(kat_digest)) == 0);

	// --- leg 2: negative test, one flipped header bit must move the digest ---
	uint32_t _ALIGN(64) ed_bad[20], dig_bad[8];
	memcpy(ed_bad, ed, sizeof(ed_bad));
	ed_bad[0] ^= 0x00000001u;
	blake2b_hash(dig_bad, ed_bad);
	const bool neg_ok = (memcmp(dig_bad, kat_digest, sizeof(kat_digest)) != 0);

	/* Legs 3 and 4 need the job words the kernel reads, and they overwrite them,
	 * so scanhash must upload the real ones afterwards - it does, once per job. */
	const uint32_t start = 0x00010000u, count = 2048u;
	uint64_t gpu[2] = { 0, 0 }, cpu[2] = { 0, 0 }, cpu_bad[2] = { 0, 0 };

	blake2b_setBlock(ed);
	blake2b_differential(thr_id, count, start, gpu);
	if (gpu[0] == 0 && gpu[1] == 0)
		selftest_cuda_fault(); // could not run != produced a wrong word

	// --- leg 3: GPU == CPU over the range ---
	blake2b_selftest_cpu_acc(ed, start, count, cpu, UINT32_MAX);
	const bool diff_ok = (gpu[0] == cpu[0] && gpu[1] == cpu[1]);

	// --- leg 4: the same comparison with one host word perturbed must fail ---
	blake2b_selftest_cpu_acc(ed, start, count, cpu_bad, count / 2);
	const bool negd_ok = !(gpu[0] == cpu_bad[0] && gpu[1] == cpu_bad[1]);

	// --- leg 5: the candidate report keeps the two lowest ---
	uint32_t lo, hi;
	const uint64_t blk_target = blake2b_selftest_block_target(ed, start, &lo, &hi);
	const bool rep_ok = blake2b_selftest_report(thr_id, start, count, UINT64_MAX, start, start + 1)
		&& blake2b_selftest_report(thr_id, start, 512, blk_target, lo, hi);

	passed[thr_id] = kat_ok && neg_ok && diff_ok && negd_ok && rep_ok;
	if (!passed[thr_id]) {
		gpulog(LOG_ERR, thr_id, "blake2b self-test FAILED (kat %d neg %d diff %d negd %d report %d)",
			(int) kat_ok, (int) neg_ok, (int) diff_ok, (int) negd_ok, (int) rep_ok);
		if (!diff_ok) {
			gpulog(LOG_ERR, thr_id, "  gpu %016llx / %016llx",
				(unsigned long long) gpu[0], (unsigned long long) gpu[1]);
			gpulog(LOG_ERR, thr_id, "  cpu %016llx / %016llx",
				(unsigned long long) cpu[0], (unsigned long long) cpu[1]);
			if (gpu[0] == cpu[0])
				gpulog(LOG_ERR, thr_id, "  words agree but their nonces do not:"
					" an index/permutation bug");
		}
	}
	return selftest_gate(thr_id, "blake2b", passed[thr_id]);
}
