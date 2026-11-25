// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test for -a blake2s (docs/coding-guideline.md section 7,
 * layer 1). Runs once per device and FAILS CLOSED via
 * cuda/selftest_gate.cuh.
 *
 * Four legs, in the order they fail:
 *   1. KAT  - host reference against a BLAKE2s-256 digest of the 80-byte
 *      header produced outside this codebase (Python hashlib).
 *   2. NEG  - one flipped header bit must change the digest.
 *   3. DIFF - GPU-vs-CPU over a nonce RANGE, using the two accumulators the
 *      -D path uses. The only leg that can see a nonce the mining kernel
 *      never reported, and it exercises the shipping arithmetic:
 *      blake2s_word7() is shared verbatim with the mining kernel.
 *   4. NEGD - the differential re-run with one contribution perturbed must
 *      fail. A gate whose trigger never occurs in normal operation is
 *      untested by construction.
 *
 * Leg 3 compares output word 7 rather than the whole digest, because word 7
 * is all the device computes - the final round prunes the ops that do not
 * feed the target screen. Word 7 is exactly the value the screen acts on.
 */

#include <string.h>

#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/selftest_gate.cuh"

/* Host reference and the device-side pieces, all defined in blake2s.cu. */
extern "C" void blake2s_hash(void *output, const void *input);
extern void blake2s_setBlock_selftest(const uint32_t *input);
extern void blake2s_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc);

/* BLAKE2s-256 of the 80-byte header built below, from Python hashlib
 * (hashlib.blake2s(msg, digest_size=32)) - an oracle outside this codebase. */
static const uint32_t kat_digest[8] = {
	0x7DF496BF, 0x051787F6, 0x6A267CE3, 0xE07F65F1,
	0xCA2201F1, 0x6569EB39, 0xFB3271A2, 0x3ECE729B,
};
#define KAT_NONCE 0x1234abcdu

/* endiandata[] exactly as scanhash builds it: be32enc over words 0..18, and the
 * nonce be32enc'd into word 19. */
static void blake2s_selftest_header(uint32_t *ed, uint32_t nonce)
{
	for (int i = 0; i < 19; i++)
		be32enc(&ed[i], (uint32_t)(0x01020304u + i));
	be32enc(&ed[19], nonce);
}

/* The host half of the differential, kept here rather than shared with
 * scanhash's copy so a fault injected for leg 4 cannot reach the mining path. */
static void blake2s_selftest_cpu_acc(const uint32_t *ed, uint32_t startNonce,
	uint32_t count, uint64_t *cpu, uint32_t perturb_at)
{
	uint32_t _ALIGN(64) vhash[8], td[20];

	cpu[0] = cpu[1] = 0;
	memcpy(td, ed, sizeof(td));
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		be32enc(&td[19], nonce);
		blake2s_hash(vhash, td);
		uint64_t w = vhash[7];
		if (i == perturb_at)
			w ^= 1ull; // leg 4 only: must break the comparison
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t)nonce + 1ull);
	}
}

__host__
bool blake2s_device_selftest(int thr_id)
{
	/* Per-device, not per-process: the GPU leg only proves the device that ran
	 * it, so a second card must be tested on its own. */
	static bool tested[MAX_GPUS] = { 0 }, passed[MAX_GPUS] = { 0 };
	if (tested[thr_id]) return passed[thr_id];
	tested[thr_id] = true;

	uint32_t _ALIGN(64) ed[20], dig[8];

	// --- leg 1: host reference vs the external oracle ---
	blake2s_selftest_header(ed, KAT_NONCE);
	blake2s_hash(dig, ed);
	const bool kat_ok = (memcmp(dig, kat_digest, sizeof(kat_digest)) == 0);

	// --- leg 2: negative test, one flipped header bit must move the digest ---
	uint32_t _ALIGN(64) ed_bad[20], dig_bad[8];
	memcpy(ed_bad, ed, sizeof(ed_bad));
	ed_bad[0] ^= 0x01000000u; // flips one bit of header byte 0
	blake2s_hash(dig_bad, ed_bad);
	const bool neg_ok = (memcmp(dig_bad, kat_digest, sizeof(kat_digest)) != 0);

	/* Legs 3 and 4 need the job midstate the kernel reads, and they overwrite it,
	 * so scanhash must upload the real one afterwards - it does, once per job. */
	const uint32_t start = 0x00010000u, count = 2048u;
	uint64_t gpu[2] = { 0, 0 }, cpu[2] = { 0, 0 }, cpu_bad[2] = { 0, 0 };

	blake2s_setBlock_selftest(ed);
	blake2s_differential(thr_id, count, start, gpu);
	if (gpu[0] == 0 && gpu[1] == 0)
		selftest_cuda_fault(); // could not run != produced a wrong word

	// --- leg 3: GPU == CPU over the range ---
	blake2s_selftest_cpu_acc(ed, start, count, cpu, UINT32_MAX);
	const bool diff_ok = (gpu[0] == cpu[0] && gpu[1] == cpu[1]);

	// --- leg 4: the same comparison with one host word perturbed must fail ---
	blake2s_selftest_cpu_acc(ed, start, count, cpu_bad, count / 2);
	const bool negd_ok = !(gpu[0] == cpu_bad[0] && gpu[1] == cpu_bad[1]);

	passed[thr_id] = kat_ok && neg_ok && diff_ok && negd_ok;
	if (!passed[thr_id]) {
		gpulog(LOG_ERR, thr_id, "blake2s self-test FAILED (kat %d neg %d diff %d negd %d)",
			(int) kat_ok, (int) neg_ok, (int) diff_ok, (int) negd_ok);
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
	return selftest_gate(thr_id, "blake2s", passed[thr_id]);
}
