/**
 * Mike (VKAX / FortuneBlock) -- CPU reference implementation.
 *
 * GhostRider with an 11-wide core pool instead of 15; see algos/mike/mike.h
 * for the delta and the `% 11` warning.  Transcribed from the pool-confirmed
 * CPU implementation in cpuminer-opt (algo/mike/mike-gate.c), which is the
 * specification for this file.
 *
 * This TU pulls in sph/ and the CryptoNight-v1 CPU cores and NOTHING else --
 * no miner.h, no CUDA -- so it links into a standalone known-answer harness.
 * The GPU driver (mike.cu) calls into it for the startup pipeline check and
 * for the host-side re-verify of every candidate nonce.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "mike.h"
#include "mike_kat.h"

extern "C" {
#include <sph/sph_blake.h>
#include <sph/sph_bmw.h>
#include <sph/sph_groestl.h>
#include <sph/sph_jh.h>
#include <sph/sph_keccak.h>
#include <sph/sph_skein.h>
#include <sph/sph_luffa.h>
#include <sph/sph_cubehash.h>
#include <sph/sph_shavite.h>
#include <sph/sph_simd.h>
#include <sph/sph_echo.h>
}

/* CryptoNight-v1 variant cores, shared with ghostrider and byte-identical for
 * mike: same scratchpad sizes, same iteration counts, same lite
 * half-addressing, same `extra_hashes[state[0] & 3]` finalization over the
 * 4-entry {blake, groestl, jh, skein} set, same HASH_SIZE of 32.
 * (algos/cryptonight/cryptonight-cpu.cpp) */
extern "C" void cryptonight_gr_dark      (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_darklite  (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_fast      (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_lite      (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_turtle    (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_turtlelite(void* output, const void* input, size_t len);

// ----------------------------------------------------------------------------
// Order derivation
// ----------------------------------------------------------------------------

static void mike_select_algo(unsigned char nibble, bool* selectedAlgos,
                             uint8_t* selectedIndex, int algoCount, int* currentCount)
{
	uint8_t algoDigit = (nibble & 0x0F) % algoCount;
	if (!selectedAlgos[algoDigit]) {
		selectedAlgos[algoDigit] = true;
		selectedIndex[currentCount[0]] = algoDigit;
		currentCount[0] += 1;
	}
	algoDigit = (nibble >> 4) % algoCount;
	if (!selectedAlgos[algoDigit]) {
		selectedAlgos[algoDigit] = true;
		selectedIndex[currentCount[0]] = algoDigit;
		currentCount[0] += 1;
	}
}

extern "C" void mike_get_algo_string(const void* mem, unsigned int size,
                                     uint8_t* selectedAlgoOutput, int algoCount)
{
	unsigned char* p = (unsigned char*)mem;
	unsigned int len = size / 2;
	bool selectedAlgo[16] = { false };   // >= every algoCount used here (11, 6)
	int selectedCount = 0;

	for (unsigned int i = 0; i < len; i++) {
		mike_select_algo(p[i], selectedAlgo, selectedAlgoOutput, algoCount, &selectedCount);
		if (selectedCount == algoCount) break;
	}
	if (selectedCount < algoCount)
		for (uint8_t i = 0; i < algoCount; i++)
			if (!selectedAlgo[i])
				selectedAlgoOutput[selectedCount++] = i;
}

// ----------------------------------------------------------------------------
// Core / CN dispatch.  A core round writes 64 bytes (512-bit); a CN round
// writes 32 and leaves the high 32 alone.
// ----------------------------------------------------------------------------

extern "C" void mike_core_algo_cpu(int algo, const void* in, void* out, size_t size)
{
	switch (algo) {
	case MIKE_BLAKE: {
		sph_blake512_context ctx; sph_blake512_init(&ctx);
		sph_blake512(&ctx, in, size); sph_blake512_close(&ctx, out); break;
	}
	case MIKE_BMW: {
		sph_bmw512_context ctx; sph_bmw512_init(&ctx);
		sph_bmw512(&ctx, in, size); sph_bmw512_close(&ctx, out); break;
	}
	case MIKE_GROESTL: {
		sph_groestl512_context ctx; sph_groestl512_init(&ctx);
		sph_groestl512(&ctx, in, size); sph_groestl512_close(&ctx, out); break;
	}
	case MIKE_JH: {
		sph_jh512_context ctx; sph_jh512_init(&ctx);
		sph_jh512(&ctx, in, size); sph_jh512_close(&ctx, out); break;
	}
	case MIKE_KECCAK: {
		sph_keccak512_context ctx; sph_keccak512_init(&ctx);
		sph_keccak512(&ctx, in, size); sph_keccak512_close(&ctx, out); break;
	}
	case MIKE_SKEIN: {
		sph_skein512_context ctx; sph_skein512_init(&ctx);
		sph_skein512(&ctx, in, size); sph_skein512_close(&ctx, out); break;
	}
	case MIKE_LUFFA: {
		sph_luffa512_context ctx; sph_luffa512_init(&ctx);
		sph_luffa512(&ctx, in, size); sph_luffa512_close(&ctx, out); break;
	}
	case MIKE_CUBEHASH: {
		sph_cubehash512_context ctx; sph_cubehash512_init(&ctx);
		sph_cubehash512(&ctx, in, size); sph_cubehash512_close(&ctx, out); break;
	}
	case MIKE_SHAVITE: {
		sph_shavite512_context ctx; sph_shavite512_init(&ctx);
		sph_shavite512(&ctx, in, size); sph_shavite512_close(&ctx, out); break;
	}
	case MIKE_SIMD: {
		sph_simd512_context ctx; sph_simd512_init(&ctx);
		sph_simd512(&ctx, in, size); sph_simd512_close(&ctx, out); break;
	}
	case MIKE_ECHO: {
		sph_echo512_context ctx; sph_echo512_init(&ctx);
		sph_echo512(&ctx, in, size); sph_echo512_close(&ctx, out); break;
	}
	/* No default: an out-of-range index must leave `out` untouched so the
	 * startup pipeline check reports a DIFF, rather than quietly hashing. */
	}
}

extern "C" void mike_cn_algo_cpu(int cnAlgo, const void* in, void* out, size_t size)
{
	switch (cnAlgo) {
	case MIKE_CN_DARK:       cryptonight_gr_dark(out, in, size); break;
	case MIKE_CN_DARKLITE:   cryptonight_gr_darklite(out, in, size); break;
	case MIKE_CN_FAST:       cryptonight_gr_fast(out, in, size); break;
	case MIKE_CN_LITE:       cryptonight_gr_lite(out, in, size); break;
	case MIKE_CN_TURTLE:     cryptonight_gr_turtle(out, in, size); break;
	case MIKE_CN_TURTLELITE: cryptonight_gr_turtlelite(out, in, size); break;
	}
}

// ----------------------------------------------------------------------------
// The chain.  Eleven core rounds consumed in groups of five, so the third
// group is one round, not five.  The first core round hashes the full 80-byte
// header; every later round hashes 64 bytes.
// ----------------------------------------------------------------------------

extern "C" void mike_hash(void* output, const void* input)
{
	uint8_t coreOrder[MIKE_CORE_ALGO_COUNT];
	uint8_t cnOrder[MIKE_CN_ALGO_COUNT];
	uint8_t hash_1[64] = { 0 };
	uint8_t hash_2[64] = { 0 };

	/* Both orders come from the prevblock region, bytes [4,36) = 64 nibbles.
	 * Nonce-independent, which is what makes a GPU batch viable. */
	mike_get_algo_string((const uint8_t*)input + 4, 64, coreOrder, MIKE_CORE_ALGO_COUNT);
	mike_get_algo_string((const uint8_t*)input + 4, 64, cnOrder,   MIKE_CN_ALGO_COUNT);

	// Group 1: first core round consumes the full 80-byte header.
	mike_core_algo_cpu(coreOrder[0], input,  hash_1, 80);
	mike_core_algo_cpu(coreOrder[1], hash_1, hash_2, 64);
	mike_core_algo_cpu(coreOrder[2], hash_2, hash_1, 64);
	mike_core_algo_cpu(coreOrder[3], hash_1, hash_2, 64);
	mike_core_algo_cpu(coreOrder[4], hash_2, hash_1, 64);
	mike_cn_algo_cpu  (cnOrder[0],   hash_1, hash_2, 64);
	memset(hash_2 + 32, 0, 32);

	// Group 2.
	mike_core_algo_cpu(coreOrder[5], hash_2, hash_1, 64);
	mike_core_algo_cpu(coreOrder[6], hash_1, hash_2, 64);
	mike_core_algo_cpu(coreOrder[7], hash_2, hash_1, 64);
	mike_core_algo_cpu(coreOrder[8], hash_1, hash_2, 64);
	mike_core_algo_cpu(coreOrder[9], hash_2, hash_1, 64);
	mike_cn_algo_cpu  (cnOrder[1],   hash_1, hash_2, 64);
	memset(hash_2 + 32, 0, 32);

	// Group 3: one core round -- the pool is exhausted at index 10.
	mike_core_algo_cpu(coreOrder[10], hash_2, hash_1, 64);
	mike_cn_algo_cpu  (cnOrder[2],    hash_1, hash_2, 64);

	memcpy(output, hash_2, 32);
}

// ----------------------------------------------------------------------------
// Known-answer self-test.  Both sets are needed: the xmrig vector covers the
// chain shape and all six CN variants but is blind to the `% 11` selection;
// the dense vectors are what pin it.  See mike_kat.h.
// ----------------------------------------------------------------------------

#define MIKE_KAT_LANES 8

static void mike_hex(char* dst, const uint8_t* src, int n)
{
	for (int i = 0; i < n; i++)
		sprintf(dst + i * 2, "%02x", src[i]);
}

extern "C" bool mike_self_test(char* detail, size_t detail_len)
{
	uint8_t blob[MIKE_KAT_LANES][80];
	uint8_t h1[MIKE_KAT_LANES][32];
	uint8_t h2[MIKE_KAT_LANES][32];
	uint8_t got[256];
	char ghex[65], ehex[65];
	int i, nonzero = 0;

	if (detail && detail_len) detail[0] = '\0';

	// ---- set 1: the two zero-blob passes, XORed ----
	memset(blob, 0, sizeof blob);
	for (i = 0; i < MIKE_KAT_LANES; i++) {
		blob[i][0] = (uint8_t)i;  blob[i][4] = 0x10;  blob[i][5] = 0x02;
	}
	for (i = 0; i < MIKE_KAT_LANES; i++) mike_hash(h1[i], blob[i]);

	for (i = 0; i < MIKE_KAT_LANES; i++) {
		blob[i][0] = (uint8_t)i;  blob[i][4] = 0x43;  blob[i][5] = 0x05;
	}
	for (i = 0; i < MIKE_KAT_LANES; i++) mike_hash(h2[i], blob[i]);

	for (i = 0; i < 256; i++) {
		got[i] = ((const uint8_t*)h1)[i] ^ ((const uint8_t*)h2)[i];
		nonzero |= got[i];
	}

	/* Guard against a vacuous pass: two identical rotations XOR to zero, and
	 * an all-zero reference would then "match" whatever we computed. */
	if (!nonzero) {
		if (detail) snprintf(detail, detail_len,
			"xmrig vector vacuous: the two rotations agree");
		return false;
	}

	if (memcmp(got, mike_kat_xmrig, 256) != 0) {
		mike_hex(ghex, got, 32);
		mike_hex(ehex, mike_kat_xmrig, 32);
		if (detail) snprintf(detail, detail_len,
			"xmrig vector mismatch: first 32 got %s expected %s", ghex, ehex);
		return false;
	}

	// ---- set 2: dense prevhashes, which see the `% 11` selection ----
	for (int v = 0; v < MIKE_KAT_DENSE_COUNT; v++) {
		uint8_t hash[32];
		mike_hash(hash, mike_kat_dense[v].hdr);
		if (memcmp(hash, mike_kat_dense[v].expected, 32) != 0) {
			mike_hex(ghex, hash, 32);
			mike_hex(ehex, mike_kat_dense[v].expected, 32);
			if (detail) snprintf(detail, detail_len,
				"dense vector %d mismatch: got %s expected %s (core count wrong?)",
				v, ghex, ehex);
			return false;
		}
	}

	if (detail) snprintf(detail, detail_len,
		"xmrig vector + %d dense vectors", MIKE_KAT_DENSE_COUNT);
	return true;
}
