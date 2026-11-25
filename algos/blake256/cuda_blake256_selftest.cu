// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test for the blake256 family (docs/coding-guideline.md
 * section 7, layer 1). Runs once per device and FAILS CLOSED via
 * cuda/selftest_gate.cuh.
 *
 * Five legs, in the order they fail:
 *   1. KAT14 - host reference against a 14-round digest produced outside this
 *      codebase. This is BLAKE-256, the SHA-3 finalist, not BLAKE2s, so there
 *      is no hashlib oracle: the constants come from a from-spec generator
 *      that validates itself against two published BLAKE-256 vectors first.
 *   2. KAT8  - the same header through the 8-round reduction blakecoin and
 *      vanilla mine. No published vector exists for it (it is not standard
 *      BLAKE-256); the oracle is the same from-spec code at fewer rounds,
 *      independent of sph/blake.c but not a standard.
 *   3. NEG   - one flipped header bit must change the digest, which proves
 *      legs 1-2 are not comparing something against itself.
 *   4. DIFF  - GPU-vs-CPU over a nonce RANGE at 14 AND 8 rounds, using the
 *      two accumulators the -D path uses. The only leg that can see a nonce
 *      the mining kernel never reported, and it exercises the shipping
 *      arithmetic: the checksum kernels inline the same helpers the mining
 *      kernels do (blake256_compress_14(), blake256_8_head()/_tail()), and the
 *      mining kernels' SASS was byte-identical before and after the extraction.
 *   5. NEGD  - the differential re-run with one nonce's contribution
 *      perturbed on the host must fail. A gate whose trigger never occurs in
 *      normal operation is untested by construction.
 *   6. REPORT - the shipping launcher must return the two LOWEST candidates,
 *      at 14 and at 8 rounds: once with every nonce a candidate, once with
 *      five in one block. The host resumes after the higher one, so a report
 *      that kept the first arrivals instead skipped every lower candidate.
 *
 * vanilla.cu has its own kernel and its own gate, vanilla_device_selftest()
 * below: KAT8, the differential through vanilla_8_head()/_tail(), and its
 * negative. Not leg 6: its launcher lives inside scanhash_vanilla.
 */

#include <string.h>
#include <algorithm>

#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/selftest_gate.cuh"

/* Host reference and the device-side pieces, all defined in blake256.cu. */
extern "C" void blake256hash(void *output, const void *input, int8_t rounds);
extern void blake256_setBlock_selftest(const uint32_t *pdata_words, int8_t rounds);
extern void blake256_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc, int8_t rounds);
extern "C" void vanillahash(void *output, const void *input, int8_t blakerounds);
extern void vanilla_differential(int thr_id, const uint32_t *pdata, uint32_t count, uint32_t startNonce, uint64_t *acc);
extern uint32_t blake256_report_selftest(const int thr_id, const uint32_t threads, const uint32_t startNonce,
	const uint64_t highTarget, const int8_t rounds, uint32_t *second);

/* BLAKE-256 of the 80-byte header built below, from a from-spec generator
 * validated against published BLAKE-256 vectors.
 *
 * These are in the word order blake256hash() RETURNS, i.e. byte-swapped state
 * words: sph_blake256_close() writes the digest big-endian, so a
 * little-endian host reads back swab32(H[k]). The DEVICE keeps the raw H[k]
 * and swaps only for the target compare. Both orders live in shipping code,
 * so the KAT is written in the order the host reference returns. */
static const uint32_t kat_digest_14[8] = {
	0xEE127EE2u, 0x4332943Cu, 0x444438B4u, 0x365C66D8u,
	0xC4542EA5u, 0xFFC7821Cu, 0x79B4ED34u, 0x95589BE4u,
};
static const uint32_t kat_digest_8[8] = {
	0x74748716u, 0x588E028Du, 0x45D64B66u, 0xB1F9056Bu,
	0x3F7F52FAu, 0xE315ADDEu, 0x8152423Fu, 0x97AA63BCu,
};
#define KAT_NONCE 0x1234abcdu

/* The words scanhash_blake256 would hold in pdata[]. The endianness subtlety is
 * that sph_blake256 reads the message as BIG-ENDIAN words, so the values BLAKE
 * actually compresses are these raw words - which is also why the device can be
 * handed &pdata[16] un-swapped and still agree with the host. */
static void blake256_selftest_pdata(uint32_t *pd, uint32_t nonce)
{
	for (int i = 0; i < 19; i++)
		pd[i] = 0x01020304u + i;
	pd[19] = nonce;
}

/* endiandata[] as the host re-verify builds it: be32enc over all 20 words. */
static void blake256_selftest_header(uint32_t *ed, const uint32_t *pd)
{
	for (int i = 0; i < 20; i++)
		be32enc(&ed[i], pd[i]);
}

/* The host half of the differential, kept here rather than shared with
 * scanhash's copy so a fault injected for leg 5 cannot reach the mining path. */
static void blake256_selftest_cpu_acc(const uint32_t *pd, uint32_t startNonce,
	uint32_t count, uint64_t *cpu, uint32_t perturb_at, int8_t rounds)
{
	uint32_t _ALIGN(64) ed[20], vhash[8], td[20];

	cpu[0] = cpu[1] = 0;
	memcpy(td, pd, sizeof(td));
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		td[19] = nonce;
		blake256_selftest_header(ed, td);
		blake256hash(vhash, ed, rounds);
		/* blake256hash() returns the digest byte-swapped relative to the state
		 * words (sph writes it big-endian; a little-endian host reads back
		 * swab32(H[k])). The device accumulates raw h[6]/h[7], so un-swap here
		 * or every nonce mismatches. */
		uint64_t w = ((uint64_t) swab32(vhash[7]) << 32) | swab32(vhash[6]);
		if (i == perturb_at)
			w ^= 1ull; // leg 5 only: must break the comparison
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t) nonce + 1ull);
	}
}

static void blake256_selftest_log_diff(int thr_id, const char *tag, const uint64_t *gpu, const uint64_t *cpu)
{
	gpulog(LOG_ERR, thr_id, "  %s gpu %016llx / %016llx", tag,
		(unsigned long long) gpu[0], (unsigned long long) gpu[1]);
	gpulog(LOG_ERR, thr_id, "  %s cpu %016llx / %016llx", tag,
		(unsigned long long) cpu[0], (unsigned long long) cpu[1]);
	if (gpu[0] == cpu[0])
		gpulog(LOG_ERR, thr_id, "  pairs agree but their nonces do not:"
			" an index/permutation bug");
}

/* Leg 6: does the launcher report (lo, hi) for this target over this range? */
static bool blake256_selftest_report(int thr_id, uint32_t start, uint32_t count,
	uint64_t target, int8_t rounds, uint32_t lo, uint32_t hi)
{
	uint32_t sec = UINT32_MAX;
	const uint32_t first = blake256_report_selftest(thr_id, count, start, target, rounds, &sec);
	return first == lo && sec == hi;
}

/* The 5th-lowest screened word over one 512-nonce block (one TPB), so exactly
 * five candidates share a block, and the two lowest NONCES among them. The
 * device screens (swab32(H7) << 32 | swab32(H6)), which is (vhash[7], vhash[6])
 * of what blake256hash() returns. */
static uint64_t blake256_selftest_block_target(const uint32_t *pd, uint32_t start,
	int8_t rounds, uint32_t *lo, uint32_t *hi)
{
	uint32_t _ALIGN(64) ed[20], vhash[8], td[20];
	uint64_t w[512], sorted[512];

	memcpy(td, pd, sizeof(td));
	for (uint32_t i = 0; i < 512; i++) {
		td[19] = start + i;
		blake256_selftest_header(ed, td);
		blake256hash(vhash, ed, rounds);
		sorted[i] = w[i] = ((uint64_t) vhash[7] << 32) | vhash[6];
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

static bool blake256_selftest_report_rounds(int thr_id, const uint32_t *pd, uint32_t start,
	uint32_t count, int8_t rounds)
{
	uint32_t lo, hi;
	blake256_setBlock_selftest(pd, rounds); // midstate depends on the round count
	const uint64_t blk_target = blake256_selftest_block_target(pd, start, rounds, &lo, &hi);
	return blake256_selftest_report(thr_id, start, count, UINT64_MAX, rounds, start, start + 1)
		&& blake256_selftest_report(thr_id, start, 512, blk_target, rounds, lo, hi);
}

__host__
bool blake256_device_selftest(int thr_id)
{
	/* Per-device, not per-process: the GPU leg only proves the device that ran
	 * it, so a second card must be tested on its own. */
	static bool tested[MAX_GPUS] = { 0 }, passed[MAX_GPUS] = { 0 };
	if (tested[thr_id]) return passed[thr_id];
	tested[thr_id] = true;

	uint32_t _ALIGN(64) pd[20], ed[20], dig[8];

	blake256_selftest_pdata(pd, KAT_NONCE);
	blake256_selftest_header(ed, pd);

	// --- legs 1 and 2: host reference vs the external oracle, both round counts ---
	blake256hash(dig, ed, 14);
	const bool kat14_ok = (memcmp(dig, kat_digest_14, sizeof(kat_digest_14)) == 0);

	blake256hash(dig, ed, 8);
	const bool kat8_ok = (memcmp(dig, kat_digest_8, sizeof(kat_digest_8)) == 0);

	// --- leg 3: negative test, one flipped header bit must move the digest ---
	uint32_t _ALIGN(64) ed_bad[20], dig_bad[8];
	memcpy(ed_bad, ed, sizeof(ed_bad));
	ed_bad[0] ^= 0x00000001u;
	blake256hash(dig_bad, ed_bad, 14);
	const bool neg_ok = (memcmp(dig_bad, kat_digest_14, sizeof(kat_digest_14)) != 0);

	/* Legs 4 and 5 need the job words the kernel reads, and they overwrite them,
	 * so scanhash must upload the real ones afterwards - it does, once per job. */
	const uint32_t start = 0x00010000u, count = 2048u;
	uint64_t gpu[2] = { 0, 0 }, cpu[2] = { 0, 0 }, cpu_bad[2] = { 0, 0 };
	uint64_t gpu8[2] = { 0, 0 }, cpu8[2] = { 0, 0 }, cpu8_bad[2] = { 0, 0 };

	blake256_setBlock_selftest(pd, 14);
	blake256_differential(thr_id, count, start, gpu, 14);
	blake256_setBlock_selftest(pd, 8); // the midstate depends on the round count
	blake256_differential(thr_id, count, start, gpu8, 8);
	if ((gpu[0] == 0 && gpu[1] == 0) || (gpu8[0] == 0 && gpu8[1] == 0))
		selftest_cuda_fault(); // could not run != produced a wrong word

	// --- leg 4: GPU == CPU over the range, both round counts ---
	blake256_selftest_cpu_acc(pd, start, count, cpu, UINT32_MAX, 14);
	blake256_selftest_cpu_acc(pd, start, count, cpu8, UINT32_MAX, 8);
	const bool diff14_ok = (gpu[0] == cpu[0] && gpu[1] == cpu[1]);
	const bool diff8_ok = (gpu8[0] == cpu8[0] && gpu8[1] == cpu8[1]);
	const bool diff_ok = diff14_ok && diff8_ok;

	// --- leg 5: the same comparisons with one host word perturbed must fail ---
	blake256_selftest_cpu_acc(pd, start, count, cpu_bad, count / 2, 14);
	blake256_selftest_cpu_acc(pd, start, count, cpu8_bad, count / 2, 8);
	const bool negd_ok = !(gpu[0] == cpu_bad[0] && gpu[1] == cpu_bad[1])
		&& !(gpu8[0] == cpu8_bad[0] && gpu8[1] == cpu8_bad[1]);

	// --- leg 6: the candidate report keeps the two lowest, both kernels ---
	const bool rep_ok = blake256_selftest_report_rounds(thr_id, pd, start, count, 14)
		&& blake256_selftest_report_rounds(thr_id, pd, start, count, 8);

	passed[thr_id] = kat14_ok && kat8_ok && neg_ok && diff_ok && negd_ok && rep_ok;
	if (!passed[thr_id]) {
		gpulog(LOG_ERR, thr_id, "blake256 self-test FAILED"
			" (kat14 %d kat8 %d neg %d diff14 %d diff8 %d negd %d report %d)",
			(int) kat14_ok, (int) kat8_ok, (int) neg_ok, (int) diff14_ok, (int) diff8_ok,
			(int) negd_ok, (int) rep_ok);
		/* A failing KAT must show what it got, or the next person re-derives the
		 * oracle from scratch to find out. */
		if (!kat14_ok || !kat8_ok) {
			uint32_t _ALIGN(64) g14[8], g8[8];
			blake256hash(g14, ed, 14);
			blake256hash(g8, ed, 8);
			gpulog(LOG_ERR, thr_id, "  host 14r %08x %08x %08x %08x %08x %08x %08x %08x",
				g14[0], g14[1], g14[2], g14[3], g14[4], g14[5], g14[6], g14[7]);
			gpulog(LOG_ERR, thr_id, "  want 14r %08x %08x %08x %08x %08x %08x %08x %08x",
				kat_digest_14[0], kat_digest_14[1], kat_digest_14[2], kat_digest_14[3],
				kat_digest_14[4], kat_digest_14[5], kat_digest_14[6], kat_digest_14[7]);
			gpulog(LOG_ERR, thr_id, "  host  8r %08x %08x %08x %08x %08x %08x %08x %08x",
				g8[0], g8[1], g8[2], g8[3], g8[4], g8[5], g8[6], g8[7]);
			gpulog(LOG_ERR, thr_id, "  want  8r %08x %08x %08x %08x %08x %08x %08x %08x",
				kat_digest_8[0], kat_digest_8[1], kat_digest_8[2], kat_digest_8[3],
				kat_digest_8[4], kat_digest_8[5], kat_digest_8[6], kat_digest_8[7]);
		}
		if (!diff14_ok)
			blake256_selftest_log_diff(thr_id, "14r", gpu, cpu);
		if (!diff8_ok)
			blake256_selftest_log_diff(thr_id, " 8r", gpu8, cpu8);
	}
	return selftest_gate(thr_id, "blake256", passed[thr_id]);
}

/* vanilla.cu's gate, separate because it reaches a different kernel and the
 * blake256 report leg needs scanhash_blake256's buffers. Same header, nonce
 * range and accumulators as legs 2, 4 and 5 above, so its expected values are
 * the 8-round ones by construction. */
__host__
bool vanilla_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 }, passed[MAX_GPUS] = { 0 };
	if (tested[thr_id]) return passed[thr_id];
	tested[thr_id] = true;

	uint32_t _ALIGN(64) pd[20], ed[20], dig[8];

	blake256_selftest_pdata(pd, KAT_NONCE);
	blake256_selftest_header(ed, pd);

	// --- KAT8: vanilla's own host reference against the external oracle ---
	vanillahash(dig, ed, 8);
	const bool kat8_ok = (memcmp(dig, kat_digest_8, sizeof(kat_digest_8)) == 0);

	/* The differential uploads the job words itself; scanhash_vanilla uploads
	 * the real ones on every call afterwards. */
	const uint32_t start = 0x00010000u, count = 2048u;
	uint64_t gpu[2] = { 0, 0 }, cpu[2] = { 0, 0 }, cpu_bad[2] = { 0, 0 };

	vanilla_differential(thr_id, pd, count, start, gpu);
	if (gpu[0] == 0 && gpu[1] == 0)
		selftest_cuda_fault();

	// --- DIFF: GPU == CPU over the range ---
	blake256_selftest_cpu_acc(pd, start, count, cpu, UINT32_MAX, 8);
	const bool diff_ok = (gpu[0] == cpu[0] && gpu[1] == cpu[1]);

	// --- NEGD: one perturbed host word must break it ---
	blake256_selftest_cpu_acc(pd, start, count, cpu_bad, count / 2, 8);
	const bool negd_ok = !(gpu[0] == cpu_bad[0] && gpu[1] == cpu_bad[1]);

	passed[thr_id] = kat8_ok && diff_ok && negd_ok;
	if (!passed[thr_id]) {
		gpulog(LOG_ERR, thr_id, "vanilla self-test FAILED (kat8 %d diff %d negd %d)",
			(int) kat8_ok, (int) diff_ok, (int) negd_ok);
		if (!diff_ok)
			blake256_selftest_log_diff(thr_id, " 8r", gpu, cpu);
	}
	return selftest_gate(thr_id, "vanilla", passed[thr_id]);
}
