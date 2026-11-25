/**
 * Flex (Kylacoin / Lyncoin) -- CPU reference implementation.
 *
 * Transcribed from the pool-confirmed CPU implementation in cpuminer-opt
 * (algo/flex/flex-gate.c + algo/flex/cryptonote/), which is the specification
 * for this file.  See algos/flex/flex.h for the four consensus differences
 * from GhostRider; every one of them is load-bearing.
 *
 * This TU pulls in sph/ and the CryptoNight-v1 CPU cores and NOTHING else --
 * no miner.h, no CUDA -- so it links into a standalone known-answer harness.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "flex.h"
#include "flex_kat.h"

extern "C" {
#include <sph/sph_blake.h>
#include <sph/sph_bmw.h>
#include <sph/sph_groestl.h>
#include <sph/sph_skein.h>
#include <sph/sph_luffa.h>
#include <sph/sph_cubehash.h>
#include <sph/sph_shavite.h>
#include <sph/sph_simd.h>
#include <sph/sph_echo.h>
#include <sph/sph_hamsi.h>
#include <sph/sph_fugue.h>
#include <sph/sph_shabal.h>
#include <sph/sph_whirlpool.h>

/* sph_sha3d.h (0x06 padding) and sph_keccak.h (0x01 padding) SHARE the
 * include guard SPH_KECCAK_H__, so whichever is included first wins and the
 * other silently becomes a no-op.  flex needs SHA-3 everywhere and legacy
 * Keccak nowhere, so this TU must include sha3d and must never include
 * sph_keccak.h -- directly or transitively. */
#include <sph/sph_sha3d.h>
}

/* There is no preprocessor check worth writing here: if sph_keccak.h had won
 * the guard, sph_sha3d512_init would simply be undeclared and this TU would
 * not compile.  The runtime discriminator is what actually protects the
 * chain -- flex_self_test() compares against BOTH paddings and reports "you
 * produced the legacy Keccak digest" by name.  See flex_kat.h. */

/* Flex's own CryptoNight-v1 variants: identical parameters to ghostrider's,
 * different finalization (`state[0] & 2` over {blake, groestl, skein-512}).
 * They require `output` to be a 64-byte buffer pre-seeded with the input.
 * (algos/cryptonight/cryptonight-cpu.cpp) */
extern "C" void cryptonight_flex_dark      (void* output, const void* input, size_t len);
extern "C" void cryptonight_flex_darklite  (void* output, const void* input, size_t len);
extern "C" void cryptonight_flex_fast      (void* output, const void* input, size_t len);
extern "C" void cryptonight_flex_lite      (void* output, const void* input, size_t len);
extern "C" void cryptonight_flex_turtle    (void* output, const void* input, size_t len);
extern "C" void cryptonight_flex_turtlelite(void* output, const void* input, size_t len);

// ----------------------------------------------------------------------------
// SHA-3 helpers.  flex applies SHA3-style padding to every keccak it uses.
// ----------------------------------------------------------------------------

extern "C" void flex_sha3_512(void* out, const void* in, size_t len)
{
	sph_sha3d512_context ctx;
	sph_sha3d512_init(&ctx);
	sph_sha3d512(&ctx, in, len);
	sph_sha3d512_close(&ctx, out);
}

extern "C" void flex_sha3_256(void* out, const void* in, size_t len)
{
	sph_sha3d256_context ctx;
	sph_sha3d256_init(&ctx);
	sph_sha3d256(&ctx, in, len);
	sph_sha3d256_close(&ctx, out);
}

extern "C" void flex_sha3d(void* out, const void* in, size_t len)
{
	uint8_t buf[32];
	flex_sha3_256(buf, in, len);
	flex_sha3_256(out, buf, 32);
}

// ----------------------------------------------------------------------------
// Order derivation
// ----------------------------------------------------------------------------

static void flex_select_algo(unsigned char nibble, bool* selectedAlgos,
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

extern "C" void flex_get_algo_string(const void* mem, unsigned int size,
                                     uint8_t* selectedAlgoOutput, int algoCount)
{
	unsigned char* p = (unsigned char*)mem;
	unsigned int len = size / 2;
	bool selectedAlgo[16] = { false };   // >= every algoCount used here (14, 6)
	int selectedCount = 0;

	for (unsigned int i = 0; i < len; i++) {
		flex_select_algo(p[i], selectedAlgo, selectedAlgoOutput, algoCount, &selectedCount);
		if (selectedCount == algoCount) break;
	}
	if (selectedCount < algoCount)
		for (uint8_t i = 0; i < algoCount; i++)
			if (!selectedAlgo[i])
				selectedAlgoOutput[selectedCount++] = i;
}

extern "C" void flex_derive_order(const void* input, uint8_t* coreOrder,
                                  uint8_t* cnOrder, uint8_t* seed)
{
	uint8_t local[64];
	uint8_t* s = seed ? seed : local;

	/* The seed is SHA3-512 of the WHOLE header, nonce included, so this is a
	 * per-nonce derivation. */
	flex_sha3_512(s, input, 80);

	/* coreOrder[14] is left 0 (= blake): selection fills only [0..13] for a
	 * 14-entry pool, but the chain reads slot 14.  This reproduces the
	 * reference's uninitialized-slot behaviour, and it is consensus -- the
	 * pool-accepted vector in flex_kat.h fails without it. */
	memset(coreOrder, 0, FLEX_CORE_CHAIN_LEN);
	flex_get_algo_string(s, 64, coreOrder, FLEX_CORE_ALGO_COUNT);
	flex_get_algo_string(s, 64, cnOrder,   FLEX_CN_ALGO_COUNT);
}

// ----------------------------------------------------------------------------
// Core / CN dispatch
// ----------------------------------------------------------------------------

extern "C" void flex_core_algo_cpu(int algo, const void* in, void* out, size_t size)
{
	switch (algo) {
	case FLEX_BLAKE: {
		sph_blake512_context ctx; sph_blake512_init(&ctx);
		sph_blake512(&ctx, in, size); sph_blake512_close(&ctx, out); break;
	}
	case FLEX_BMW: {
		sph_bmw512_context ctx; sph_bmw512_init(&ctx);
		sph_bmw512(&ctx, in, size); sph_bmw512_close(&ctx, out); break;
	}
	case FLEX_GROESTL: {
		sph_groestl512_context ctx; sph_groestl512_init(&ctx);
		sph_groestl512(&ctx, in, size); sph_groestl512_close(&ctx, out); break;
	}
	case FLEX_KECCAK: {
		/* SHA3-512, NOT Keccak-512.  One byte of padding apart, and that byte
		 * changes every digest in the chain. */
		flex_sha3_512(out, in, size); break;
	}
	case FLEX_SKEIN: {
		sph_skein512_context ctx; sph_skein512_init(&ctx);
		sph_skein512(&ctx, in, size); sph_skein512_close(&ctx, out); break;
	}
	case FLEX_LUFFA: {
		sph_luffa512_context ctx; sph_luffa512_init(&ctx);
		sph_luffa512(&ctx, in, size); sph_luffa512_close(&ctx, out); break;
	}
	case FLEX_CUBEHASH: {
		sph_cubehash512_context ctx; sph_cubehash512_init(&ctx);
		sph_cubehash512(&ctx, in, size); sph_cubehash512_close(&ctx, out); break;
	}
	case FLEX_SHAVITE: {
		sph_shavite512_context ctx; sph_shavite512_init(&ctx);
		sph_shavite512(&ctx, in, size); sph_shavite512_close(&ctx, out); break;
	}
	case FLEX_SIMD: {
		sph_simd512_context ctx; sph_simd512_init(&ctx);
		sph_simd512(&ctx, in, size); sph_simd512_close(&ctx, out); break;
	}
	case FLEX_ECHO: {
		sph_echo512_context ctx; sph_echo512_init(&ctx);
		sph_echo512(&ctx, in, size); sph_echo512_close(&ctx, out); break;
	}
	case FLEX_HAMSI: {
		sph_hamsi512_context ctx; sph_hamsi512_init(&ctx);
		sph_hamsi512(&ctx, in, size); sph_hamsi512_close(&ctx, out); break;
	}
	case FLEX_FUGUE: {
		sph_fugue512_context ctx; sph_fugue512_init(&ctx);
		sph_fugue512(&ctx, in, size); sph_fugue512_close(&ctx, out); break;
	}
	case FLEX_SHABAL: {
		sph_shabal512_context ctx; sph_shabal512_init(&ctx);
		sph_shabal512(&ctx, in, size); sph_shabal512_close(&ctx, out); break;
	}
	case FLEX_WHIRLPOOL: {
		sph_whirlpool_context ctx; sph_whirlpool_init(&ctx);
		sph_whirlpool(&ctx, in, size); sph_whirlpool_close(&ctx, out); break;
	}
	/* No default: an out-of-range index must leave `out` untouched so a
	 * step-by-step check reports a DIFF rather than quietly hashing. */
	}
}

extern "C" void flex_cn_algo_cpu(int cnAlgo, const void* in, void* out, size_t size)
{
	const char* i = (const char*)in;
	char* o = (char*)out;

	/* The reference runs each CN round IN PLACE (input buffer == output
	 * buffer).  On the blake branch the finalization writes only the low 32 of
	 * the 64 output bytes, so the high 32 are the round's own input bytes.  We
	 * use separate ping-pong buffers, so pre-seed the output with the input to
	 * reproduce them.  (On the skein-512 branch all 64 are overwritten, making
	 * this a no-op there.)
	 *
	 * This is not cosmetic: flex's final SHA3-256 hashes all 64 bytes, so
	 * dropping the pre-seed changes every digest whose last CN round took the
	 * blake branch -- about half of them. */
	memcpy(o, i, size);

	switch (cnAlgo) {
	case FLEX_CN_DARK:       cryptonight_flex_dark      (o, i, size); break;
	case FLEX_CN_DARKLITE:   cryptonight_flex_darklite  (o, i, size); break;
	case FLEX_CN_FAST:       cryptonight_flex_fast      (o, i, size); break;
	case FLEX_CN_LITE:       cryptonight_flex_lite      (o, i, size); break;
	case FLEX_CN_TURTLE:     cryptonight_flex_turtle    (o, i, size); break;
	case FLEX_CN_TURTLELITE: cryptonight_flex_turtlelite(o, i, size); break;
	}
}

// ----------------------------------------------------------------------------
// The chain: three groups of (5 core + 1 CN), then a final SHA3-256.
//
// Fifteen core rounds, not fourteen -- the fifteenth is coreOrder[14], which
// selection never fills, so it is always blake512.  No zeroing after a CN
// round, unlike ghostrider.
// ----------------------------------------------------------------------------

extern "C" void flex_hash(void* output, const void* input)
{
	uint8_t coreOrder[FLEX_CORE_CHAIN_LEN];
	uint8_t cnOrder[FLEX_CN_ALGO_COUNT];
	uint8_t hash_1[64] = { 0 };
	uint8_t hash_2[64] = { 0 };

	flex_derive_order(input, coreOrder, cnOrder, NULL);

	// Group 1: first core round consumes the full 80-byte header.
	flex_core_algo_cpu(coreOrder[0], input,  hash_1, 80);
	flex_core_algo_cpu(coreOrder[1], hash_1, hash_2, 64);
	flex_core_algo_cpu(coreOrder[2], hash_2, hash_1, 64);
	flex_core_algo_cpu(coreOrder[3], hash_1, hash_2, 64);
	flex_core_algo_cpu(coreOrder[4], hash_2, hash_1, 64);
	flex_cn_algo_cpu  (cnOrder[0],   hash_1, hash_2, 64);
	// No memset of hash_2[32..64) -- flex does not zero.

	// Group 2.
	flex_core_algo_cpu(coreOrder[5], hash_2, hash_1, 64);
	flex_core_algo_cpu(coreOrder[6], hash_1, hash_2, 64);
	flex_core_algo_cpu(coreOrder[7], hash_2, hash_1, 64);
	flex_core_algo_cpu(coreOrder[8], hash_1, hash_2, 64);
	flex_core_algo_cpu(coreOrder[9], hash_2, hash_1, 64);
	flex_cn_algo_cpu  (cnOrder[1],   hash_1, hash_2, 64);

	// Group 3.  coreOrder[14] is the always-blake slot.
	flex_core_algo_cpu(coreOrder[10], hash_2, hash_1, 64);
	flex_core_algo_cpu(coreOrder[11], hash_1, hash_2, 64);
	flex_core_algo_cpu(coreOrder[12], hash_2, hash_1, 64);
	flex_core_algo_cpu(coreOrder[13], hash_1, hash_2, 64);
	flex_core_algo_cpu(coreOrder[14], hash_2, hash_1, 64);
	flex_cn_algo_cpu  (cnOrder[2],    hash_1, hash_2, 64);

	// Final SHA3-256 over all 64 bytes of the last CN output.
	flex_sha3_256(output, hash_2, 64);
}

// ----------------------------------------------------------------------------
// Known-answer self-test.
// ----------------------------------------------------------------------------

static void flex_hex(char* dst, const uint8_t* src, int n)
{
	for (int i = 0; i < n; i++)
		sprintf(dst + i * 2, "%02x", src[i]);
}

extern "C" bool flex_self_test(char* detail, size_t detail_len)
{
	uint8_t got[64];
	char ghex[129], ehex[129];

	if (detail && detail_len) detail[0] = '\0';

	/* Gate 1: the SHA-3 padding, against vectors from an independent
	 * implementation (Python hashlib).  Checked FIRST because a padding error
	 * makes every later failure unreadable, and because this is the one trap
	 * that no amount of GhostRider experience warns you about. */
	struct { const char* what; const uint8_t* in; size_t len;
	         void (*fn)(void*, const void*, size_t);
	         const uint8_t* want; const uint8_t* wrong; int n; } sha3[] = {
		{ "SHA3-512(header80)", flex_kat_header,    80, flex_sha3_512,
		  flex_sha3_512_of_header, NULL, 64 },
		{ "SHA3-512(msg64)",    flex_sha3_msg64,    64, flex_sha3_512,
		  flex_sha3_512_of_msg64, flex_keccak_512_of_msg64, 64 },
		{ "SHA3-256(msg64)",    flex_sha3_msg64,    64, flex_sha3_256,
		  flex_sha3_256_of_msg64, flex_keccak_256_of_msg64, 32 },
	};

	for (int k = 0; k < 3; k++) {
		sha3[k].fn(got, sha3[k].in, sha3[k].len);
		if (memcmp(got, sha3[k].want, sha3[k].n) == 0)
			continue;
		/* Self-identifying failure: say so if we produced the 0x01 answer. */
		if (sha3[k].wrong && memcmp(got, sha3[k].wrong, sha3[k].n) == 0) {
			if (detail) snprintf(detail, detail_len,
				"%s produced the LEGACY KECCAK (0x01) digest -- flex needs SHA-3 (0x06)",
				sha3[k].what);
			return false;
		}
		flex_hex(ghex, got, 16);
		flex_hex(ehex, sha3[k].want, 16);
		if (detail) snprintf(detail, detail_len,
			"%s mismatch: got %s... expected %s...", sha3[k].what, ghex, ehex);
		return false;
	}

	/* Gate 2: the end-to-end consensus vector, from a pool-accepted share. */
	flex_hash(got, flex_kat_header);
	if (memcmp(got, flex_kat_expected, 32) != 0) {
		flex_hex(ghex, got, 32);
		flex_hex(ehex, flex_kat_expected, 32);
		if (detail) snprintf(detail, detail_len,
			"consensus vector mismatch: got %s expected %s", ghex, ehex);
		return false;
	}

	if (detail) snprintf(detail, detail_len,
		"3 SHA-3 vectors + the pool-accepted consensus vector");
	return true;
}
