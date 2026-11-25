/*
 * curvehash (CurvehashCoin) proof-of-work - host reference / CPU miner.
 *
 * PoW (from CurvehashCoin src/curvehash.cpp, authoritative):
 *     phash = SHA256( header[0..75] || nonce_le[4] )        // 80 bytes
 *     for round in 0..7:                                    // exactly 8 rounds
 *         pubkey  = secp256k1_ec_pubkey_create(phash)       // phash IS the privkey
 *         pub[65] = serialize_uncompressed(pubkey)          // 0x04 || X_be || Y_be
 *         phash   = SHA256( pub[0..64] )                     // 65 bytes
 *     compare phash (MSW-first) to target
 *
 * The eight fixed-base scalar-multiplications dominate; this CPU path is the
 * bring-up miner and the authoritative candidate re-verify oracle for the GPU
 * port. secp256k1 is vendored (see secp256k1_unity.c).
 *
 * SHA-256 transcribed from the upstream reference miner (termux-miner
 * algo/curvehash.c) so byte/endianness handling matches bit-for-bit.
 */

#include "miner.h"

#include <string.h>
#include <stdlib.h>

#include "secp256k1/include/secp256k1.h"

/* ------------------------------------------------------------------ SHA-256 */

static const uint32_t sha256_k[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static const uint32_t sha256_iv[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

#define CH_ROTR(x, n)   (((x) >> (n)) | ((x) << (32 - (n))))
#define CH_Ch(x, y, z)  ((x & (y ^ z)) ^ z)
#define CH_Maj(x, y, z) ((x & (y | z)) | (y & z))
#define CH_S0(x)        (CH_ROTR(x, 2) ^ CH_ROTR(x, 13) ^ CH_ROTR(x, 22))
#define CH_S1(x)        (CH_ROTR(x, 6) ^ CH_ROTR(x, 11) ^ CH_ROTR(x, 25))
#define CH_s0(x)        (CH_ROTR(x, 7) ^ CH_ROTR(x, 18) ^ ((x) >> 3))
#define CH_s1(x)        (CH_ROTR(x, 17) ^ CH_ROTR(x, 19) ^ ((x) >> 10))

#define CH_RND(a, b, c, d, e, f, g, h, k) do { \
		t0 = h + CH_S1(e) + CH_Ch(e, f, g) + k; \
		t1 = CH_S0(a) + CH_Maj(a, b, c); \
		d += t0; \
		h  = t0 + t1; \
	} while (0)

#define CH_RNDr(S, W, i) \
	CH_RND(S[(64 - i) % 8], S[(65 - i) % 8], S[(66 - i) % 8], S[(67 - i) % 8], \
	       S[(68 - i) % 8], S[(69 - i) % 8], S[(70 - i) % 8], S[(71 - i) % 8], \
	       W[i] + sha256_k[i])

static void curve_sha256_transform(uint32_t *state, uint32_t *W)
{
	uint32_t S[8];
	uint32_t t0, t1;
	int i;

	for (i = 16; i < 64; i += 2) {
		W[i]     = CH_s1(W[i - 2]) + W[i - 7] + CH_s0(W[i - 15]) + W[i - 16];
		W[i + 1] = CH_s1(W[i - 1]) + W[i - 6] + CH_s0(W[i - 14]) + W[i - 15];
	}

	memcpy(S, state, 32);

	CH_RNDr(S, W,  0); CH_RNDr(S, W,  1); CH_RNDr(S, W,  2); CH_RNDr(S, W,  3);
	CH_RNDr(S, W,  4); CH_RNDr(S, W,  5); CH_RNDr(S, W,  6); CH_RNDr(S, W,  7);
	CH_RNDr(S, W,  8); CH_RNDr(S, W,  9); CH_RNDr(S, W, 10); CH_RNDr(S, W, 11);
	CH_RNDr(S, W, 12); CH_RNDr(S, W, 13); CH_RNDr(S, W, 14); CH_RNDr(S, W, 15);
	CH_RNDr(S, W, 16); CH_RNDr(S, W, 17); CH_RNDr(S, W, 18); CH_RNDr(S, W, 19);
	CH_RNDr(S, W, 20); CH_RNDr(S, W, 21); CH_RNDr(S, W, 22); CH_RNDr(S, W, 23);
	CH_RNDr(S, W, 24); CH_RNDr(S, W, 25); CH_RNDr(S, W, 26); CH_RNDr(S, W, 27);
	CH_RNDr(S, W, 28); CH_RNDr(S, W, 29); CH_RNDr(S, W, 30); CH_RNDr(S, W, 31);
	CH_RNDr(S, W, 32); CH_RNDr(S, W, 33); CH_RNDr(S, W, 34); CH_RNDr(S, W, 35);
	CH_RNDr(S, W, 36); CH_RNDr(S, W, 37); CH_RNDr(S, W, 38); CH_RNDr(S, W, 39);
	CH_RNDr(S, W, 40); CH_RNDr(S, W, 41); CH_RNDr(S, W, 42); CH_RNDr(S, W, 43);
	CH_RNDr(S, W, 44); CH_RNDr(S, W, 45); CH_RNDr(S, W, 46); CH_RNDr(S, W, 47);
	CH_RNDr(S, W, 48); CH_RNDr(S, W, 49); CH_RNDr(S, W, 50); CH_RNDr(S, W, 51);
	CH_RNDr(S, W, 52); CH_RNDr(S, W, 53); CH_RNDr(S, W, 54); CH_RNDr(S, W, 55);
	CH_RNDr(S, W, 56); CH_RNDr(S, W, 57); CH_RNDr(S, W, 58); CH_RNDr(S, W, 59);
	CH_RNDr(S, W, 60); CH_RNDr(S, W, 61); CH_RNDr(S, W, 62); CH_RNDr(S, W, 63);

	for (i = 0; i < 8; i++)
		state[i] += S[i];
}

/* SHA-256 of an arbitrary-length buffer, big-endian digest out. */
static void curve_sha256(unsigned char *hash, const unsigned char *data, int len)
{
	uint32_t S[8];
	uint32_t T[64];
	int i, r;

	memcpy(S, sha256_iv, 32);
	for (r = len; r > -9; r -= 64) {
		if (r < 64)
			memset(T, 0, 64);
		memcpy(T, data + len - r, r > 64 ? 64 : (r < 0 ? 0 : r));
		if (r >= 0 && r < 64)
			((unsigned char *)T)[r] = 0x80;
		for (i = 0; i < 16; i++)
			T[i] = be32dec(T + i);
		if (r < 56)
			T[15] = 8 * len;
		curve_sha256_transform(S, T);
	}
	for (i = 0; i < 8; i++)
		be32enc((uint32_t *)hash + i, S[i]);
}

/* ---------------------------------------------------------------- curvehash */

/*
 * Full curvehash over an 80-byte header (already in final hashed byte order:
 * whole header big-endian, nonce byte-swapped). Digest is 32 big-endian bytes.
 * Authoritative oracle; ctx must be a SECP256K1_CONTEXT_SIGN context.
 */
static void curvehash_80(secp256k1_context *ctx, unsigned char *phash /*32B*/,
                         const unsigned char *header80)
{
	secp256k1_pubkey pubkey;
	unsigned char pub[65];
	size_t publen = 65;

	curve_sha256(phash, header80, 80);
	for (int round = 0; round < 8; round++) {
		/* Consensus asserts pubkey_create == 1; an invalid seckey
		 * (phash == 0 || phash >= n) can never be a valid block. */
		if (!secp256k1_ec_pubkey_create(ctx, &pubkey, phash))
			return; /* leave phash as-is; caller fulltest will reject */
		secp256k1_ec_pubkey_serialize(ctx, pub, &publen, &pubkey,
		                              SECP256K1_EC_UNCOMPRESSED);
		curve_sha256(phash, pub, 65);
	}
}

/* Oracle entry point (byte-oriented). input = 80-byte hashed header. */
extern "C" void curvehash_hash(void *output, const void *input)
{
	secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
	curvehash_80(ctx, (unsigned char *)output, (const unsigned char *)input);
	secp256k1_context_destroy(ctx);
}

/*
 * One-time self-test: full curvehash over a fixed 80-byte header (bytes
 * 0x00..0x4f) must match the reference, and a one-bit header flip must change
 * the digest (proves the test isn't vacuous). Reference digest computed by an
 * independent textbook-secp256k1 + hashlib oracle. Validates the vendored EC
 * stack (as configured/compiled here) plus the SHA-256 byte handling.
 */
static bool curvehash_selftest(secp256k1_context *ctx)
{
	static bool tested = false, passed = false;
	if (tested)
		return passed;
	tested = true;

	static const unsigned char kat_digest[32] = {
		0xb2,0x64,0x54,0x16,0xce,0x97,0xcf,0x39,0x35,0x59,0x2d,0x82,0xea,0xeb,0xf2,0x52,
		0x12,0x00,0x8e,0xbf,0x04,0xf6,0x23,0x73,0x20,0x3a,0x71,0x53,0xfa,0x1e,0x14,0x66
	};
	unsigned char hdr[80], out[32];
	for (int i = 0; i < 80; i++)
		hdr[i] = (unsigned char)i;

	curvehash_80(ctx, out, hdr);
	const bool kat_ok = (memcmp(out, kat_digest, 32) == 0);

	hdr[40] ^= 0x01; /* flip one bit */
	curvehash_80(ctx, out, hdr);
	const bool neg_ok = (memcmp(out, kat_digest, 32) != 0);

	passed = kat_ok && neg_ok;
	if (!passed)
		applog(LOG_ERR, "curvehash self-test FAILED (kat %d neg %d)",
		       (int)kat_ok, (int)neg_ok);
	return passed;
}

/* Per-thread signing context (one-per-thread keeps bring-up simple and avoids
 * first-use races on the runtime-built ecmult_gen table). */
static secp256k1_context *g_ctx[MAX_GPUS] = { 0 };

/*
 * Build the GPU fixed-base window table: out[(j*16+i)*64 .. +64] =
 * (i * 16^j) * G as X[32]||Y[32] big-endian (i==0 left zero = infinity).
 * Uses the vendored libsecp256k1 — the authoritative table the device k*G sums.
 * Also runs the host self-test once (validates the re-verify oracle).
 */
extern "C" void curvehash_build_gtable(unsigned char *out)
{
	secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
	memset(out, 0, (size_t)32 * 256 * 64);
	for (int j = 0; j < 32; j++) {
		for (int i = 1; i < 256; i++) {
			unsigned char s[32] = { 0 };
			s[31 - j] = (unsigned char)i;   /* i * 256^j */
			secp256k1_pubkey pk;
			unsigned char pub[65];
			size_t publen = 65;
			if (!secp256k1_ec_pubkey_create(ctx, &pk, s))
				continue;
			secp256k1_ec_pubkey_serialize(ctx, pub, &publen, &pk, SECP256K1_EC_UNCOMPRESSED);
			memcpy(out + ((size_t)(j * 256 + i) * 64), pub + 1, 64); /* X || Y */
		}
	}
	curvehash_selftest(ctx);
	secp256k1_context_destroy(ctx);
}

/*
 * Authoritative pre-submit re-verify of a single GPU candidate nonce: recompute
 * curvehash(header, nonce) on the host and test it against the target. Returns 1
 * (and fills hash[8]) iff it validates. A kernel bug can then only ever cause a
 * local reject, never a bad share.
 */
extern "C" int curvehash_host_reverify(int thr_id, const uint32_t *pdata, uint32_t nonce,
                                       const uint32_t *ptarget, uint32_t *hash)
{
	uint32_t _ALIGN(64) endiandata[20];
	if (!g_ctx[thr_id])
		g_ctx[thr_id] = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
	for (int i = 0; i < 19; i++)
		be32enc(&endiandata[i], pdata[i]);
	endiandata[19] = swab32(nonce);
	curvehash_80(g_ctx[thr_id], (unsigned char *)hash, (const unsigned char *)endiandata);
	return (hash[7] <= ptarget[7] && fulltest(hash, ptarget));
}

extern "C" void curvehash_host_free(int thr_id)
{
	if (g_ctx[thr_id]) {
		secp256k1_context_destroy(g_ctx[thr_id]);
		g_ctx[thr_id] = NULL;
	}
}
