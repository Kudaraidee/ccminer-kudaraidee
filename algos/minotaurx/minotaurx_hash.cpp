/**
 * minotaurx -- CPU reference, transcribed from cpuminer-opt algo/x16/minotaur.c.
 * Uses only sph/ and the yespower reference (no miner.h, no CUDA).
 *
 * MINOTAURX_FAULT (test builds only) injects a known-wrong transcription:
 * 1 pers byte, 2 r=16, 3 blake/bmw swapped, 4 branch on byte 60,
 * 5 algo from the running hash instead of the seed.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "minotaurx.h"
#include "minotaurx_kat.h"

extern "C" {
#include <sph/sph_blake.h>
#include <sph/sph_bmw.h>
#include <sph/sph_groestl.h>
#include <sph/sph_skein.h>
#include <sph/sph_jh.h>
#include <sph/sph_keccak.h>
#include <sph/sph_luffa.h>
#include <sph/sph_cubehash.h>
#include <sph/sph_shavite.h>
#include <sph/sph_simd.h>
#include <sph/sph_echo.h>
#include <sph/sph_hamsi.h>
#include <sph/sph_fugue.h>
#include <sph/sph_shabal.h>
#include <sph/sph_whirlpool.h>
#include <sph/sph_sha2.h>
}
#include <sph/yespower.h>

#ifndef MINOTAURX_FAULT
#define MINOTAURX_FAULT 0
#endif

/* the length is checked beside the literal: one wrong byte is a silent wrong hash */
static const char MINOTAURX_PERS[] = "et in arcadia ego";
static_assert(sizeof(MINOTAURX_PERS) - 1 == 17, "minotaurx pers length changed");

#if MINOTAURX_FAULT == 1
static const char MINOTAURX_PERS_USED[] = "et in arcadia egp";
#else
#define MINOTAURX_PERS_USED MINOTAURX_PERS
#endif

static const yespower_params_t minotaurx_yp_params = {
	YESPOWER_1_0, 2048, (MINOTAURX_FAULT == 2) ? 16u : 8u,
	(const uint8_t *) MINOTAURX_PERS_USED, 17
};

/* minotaur's own index order (get_hash's switch), not x16r's */
enum MinotaurAlgo {
	MX_BLAKE = 0, MX_BMW, MX_CUBEHASH, MX_ECHO, MX_FUGUE, MX_GROESTL, MX_HAMSI,
	MX_SHA512, MX_JH, MX_KECCAK, MX_LUFFA, MX_SHABAL, MX_SHAVITE, MX_SIMD,
	MX_SKEIN, MX_WHIRLPOOL,
	MX_YESPOWER                        /* == MINOTAURX_ALGO_COUNT, node 21 only */
};
static_assert(MX_YESPOWER == MINOTAURX_ALGO_COUNT, "minotaurx algo table size");

/* initialize_torture_garden(): children per node; every walk ends on node 21 */
static const int8_t minotaurx_child[MINOTAURX_NODES][2] = {
	{  1,  2 }, {  3,  4 }, {  5,  6 }, {  7,  8 }, {  9, 10 }, { 11, 12 },
	{ 13, 14 }, { 15, 16 }, { 15, 16 }, { 15, 16 }, { 15, 16 }, { 17, 18 },
	{ 17, 18 }, { 17, 18 }, { 17, 18 }, { 19, 20 }, { 19, 20 }, { 19, 20 },
	{ 19, 20 }, { 21, 21 }, { 21, 21 }, { -1, -1 },
};

/* One cheap 64-byte -> 64-byte node.  in and out may alias. */
static void minotaurx_node(int algo, const void *in, void *out)
{
	uint8_t hash[64];

#if MINOTAURX_FAULT == 3
	if (algo == MX_BLAKE) algo = MX_BMW; else if (algo == MX_BMW) algo = MX_BLAKE;
#endif

	switch (algo) {
	case MX_BLAKE: {
		sph_blake512_context c;
		sph_blake512_init(&c); sph_blake512(&c, in, 64); sph_blake512_close(&c, hash);
	} break;
	case MX_BMW: {
		sph_bmw512_context c;
		sph_bmw512_init(&c); sph_bmw512(&c, in, 64); sph_bmw512_close(&c, hash);
	} break;
	case MX_CUBEHASH: {
		sph_cubehash512_context c;
		sph_cubehash512_init(&c); sph_cubehash512(&c, in, 64); sph_cubehash512_close(&c, hash);
	} break;
	case MX_ECHO: {
		sph_echo512_context c;
		sph_echo512_init(&c); sph_echo512(&c, in, 64); sph_echo512_close(&c, hash);
	} break;
	case MX_FUGUE: {
		sph_fugue512_context c;
		sph_fugue512_init(&c); sph_fugue512(&c, in, 64); sph_fugue512_close(&c, hash);
	} break;
	case MX_GROESTL: {
		sph_groestl512_context c;
		sph_groestl512_init(&c); sph_groestl512(&c, in, 64); sph_groestl512_close(&c, hash);
	} break;
	case MX_HAMSI: {
		sph_hamsi512_context c;
		sph_hamsi512_init(&c); sph_hamsi512(&c, in, 64); sph_hamsi512_close(&c, hash);
	} break;
	case MX_SHA512: {
		sph_sha512_context c;
		sph_sha512_init(&c); sph_sha512(&c, in, 64); sph_sha512_close(&c, hash);
	} break;
	case MX_JH: {
		sph_jh512_context c;
		sph_jh512_init(&c); sph_jh512(&c, in, 64); sph_jh512_close(&c, hash);
	} break;
	case MX_KECCAK: {
		sph_keccak512_context c;
		sph_keccak512_init(&c); sph_keccak512(&c, in, 64); sph_keccak512_close(&c, hash);
	} break;
	case MX_LUFFA: {
		sph_luffa512_context c;
		sph_luffa512_init(&c); sph_luffa512(&c, in, 64); sph_luffa512_close(&c, hash);
	} break;
	case MX_SHABAL: {
		sph_shabal512_context c;
		sph_shabal512_init(&c); sph_shabal512(&c, in, 64); sph_shabal512_close(&c, hash);
	} break;
	case MX_SHAVITE: {
		sph_shavite512_context c;
		sph_shavite512_init(&c); sph_shavite512(&c, in, 64); sph_shavite512_close(&c, hash);
	} break;
	case MX_SIMD: {
		sph_simd512_context c;
		sph_simd512_init(&c); sph_simd512(&c, in, 64); sph_simd512_close(&c, hash);
	} break;
	case MX_SKEIN: {
		sph_skein512_context c;
		sph_skein512_init(&c); sph_skein512(&c, in, 64); sph_skein512_close(&c, hash);
	} break;
	case MX_WHIRLPOOL: {
		sph_whirlpool_context c;
		sph_whirlpool_init(&c); sph_whirlpool(&c, in, 64); sph_whirlpool_close(&c, hash);
	} break;
	}
	memcpy(out, hash, 64);
}

extern "C" void minotaurx_chain(void *out64, const void *input, uint8_t *path)
{
	uint8_t seed[64], h[64];
	int node = 0, step = 0;

	sph_sha512_context c;
	sph_sha512_init(&c);
	sph_sha512(&c, input, 80);
	sph_sha512_close(&c, seed);
	memcpy(h, seed, 64);

	/* stop at node 21 (yespower): the caller runs it */
	while (node != MINOTAURX_NODES - 1) {
#if MINOTAURX_FAULT == 5
		const int algo = h[node] % MINOTAURX_ALGO_COUNT;
#else
		const int algo = seed[node] % MINOTAURX_ALGO_COUNT;
#endif
		if (path) path[step] = (uint8_t) algo;
		step++;
		minotaurx_node(algo, h, h);
#if MINOTAURX_FAULT == 4
		node = minotaurx_child[node][h[60] & 1];
#else
		node = minotaurx_child[node][h[63] & 1];
#endif
	}
	memcpy(out64, h, 64);
}

/* yespower_tls_ref returns 1 on success.  Fail closed: all-ones never clears a target. */
extern "C" bool minotaurx_yespower64(void *out32, const void *in64)
{
	yespower_binary_t out;
	if (yespower_tls_ref((const uint8_t *) in64, 64, &minotaurx_yp_params, &out) != 1) {
		memset(out32, 0xff, 32);
		return false;
	}
	memcpy(out32, out.uc, 32);
	return true;
}

extern "C" bool minotaurx_hash(void *output, const void *input)
{
	uint8_t h[64];
	minotaurx_chain(h, input, NULL);
	return minotaurx_yespower64(output, h);
}

// ----------------------------------------------------------------------------
// Known-answer self-test.
// ----------------------------------------------------------------------------

extern "C" const uint8_t minotaurx_kat_node21_in[64] = {
	0x96, 0x76, 0x7d, 0xec, 0xb8, 0x25, 0x3e, 0xae, 0xed, 0xa2, 0xb9, 0x8a,
	0x57, 0xe7, 0xa0, 0x9c, 0xe8, 0x81, 0x02, 0x09, 0x8f, 0x03, 0x23, 0x91,
	0xb8, 0xbb, 0x52, 0x0e, 0x4f, 0xa6, 0xf9, 0x74, 0xee, 0x11, 0x6a, 0xd3,
	0x38, 0x8c, 0xc4, 0x6b, 0x18, 0x12, 0x0d, 0x6d, 0xbf, 0x09, 0x76, 0x16,
	0x63, 0x04, 0xf9, 0x25, 0x15, 0x36, 0xf9, 0x73, 0x95, 0x3f, 0xc8, 0x6b,
	0x20, 0xd8, 0x23, 0x78
};

static void minotaurx_hex(char *dst, const uint8_t *src, int n)
{
	for (int i = 0; i < n; i++)
		sprintf(dst + i * 2, "%02x", src[i]);
}

/* digest <= target, both little-endian 256-bit (fulltest's comparison). */
static bool minotaurx_le256_le(const uint8_t *a, const uint8_t *b)
{
	for (int i = 31; i >= 0; i--)
		if (a[i] != b[i]) return a[i] < b[i];
	return true;
}

extern "C" bool minotaurx_self_test(char *detail, size_t detail_len)
{
	uint8_t got[32], hdr[80], state[64];
	char ghex[65], ehex[65];

	if (detail && detail_len) detail[0] = '\0';

	/* each half on its own first, so a failure names the half that broke */
	minotaurx_chain(state, minotaurx_kats[0].header, NULL);
	if (memcmp(state, minotaurx_kat_node21_in, 64) != 0) {
		minotaurx_hex(ghex, state, 16);
		minotaurx_hex(ehex, minotaurx_kat_node21_in, 16);
		if (detail) snprintf(detail, detail_len,
			"chain (six cheap nodes) wrong for %s: got %s... expected %s...",
			minotaurx_kats[0].name, ghex, ehex);
		return false;
	}
	if (!minotaurx_yespower64(got, minotaurx_kat_node21_in) ||
	    memcmp(got, minotaurx_kats[0].digest, 32) != 0) {
		minotaurx_hex(ghex, got, 16);
		minotaurx_hex(ehex, minotaurx_kats[0].digest, 16);
		if (detail) snprintf(detail, detail_len,
			"yespower node (N=2048 r=8 pers) wrong on a known chain state: got %s... expected %s...",
			ghex, ehex);
		return false;
	}

	for (unsigned k = 0; k < MINOTAURX_NUM_KATS; k++) {
		const minotaurx_kat_t *v = &minotaurx_kats[k];
		if (!minotaurx_hash(got, v->header)) {
			if (detail) snprintf(detail, detail_len,
				"%s: the yespower reference refused its parameters", v->name);
			return false;
		}
		if (memcmp(got, v->digest, 32) != 0) {
			minotaurx_hex(ghex, got, 32);
			minotaurx_hex(ehex, v->digest, 32);
			if (detail) snprintf(detail, detail_len,
				"%s mismatch: got %s expected %s", v->name, ghex, ehex);
			return false;
		}
		/* a mined block's digest must clear its own target */
		if (!minotaurx_le256_le(got, v->target)) {
			if (detail) snprintf(detail, detail_len,
				"%s: digest is over the block's own target", v->name);
			return false;
		}
	}

	/* one flipped nonce bit must change the digest */
	memcpy(hdr, minotaurx_kats[0].header, 80);
	hdr[79] ^= 1;
	if (!minotaurx_hash(got, hdr) || memcmp(got, minotaurx_kats[0].digest, 32) == 0) {
		if (detail) snprintf(detail, detail_len,
			"vacuous: a flipped nonce bit did not change the digest");
		return false;
	}

	if (detail) snprintf(detail, detail_len,
		"chain state, yespower node, %u Pulsar mainnet headers %u..%u (digest and target)",
		(unsigned) MINOTAURX_NUM_KATS, minotaurx_kats[0].height,
		minotaurx_kats[MINOTAURX_NUM_KATS - 1].height);
	return true;
}
