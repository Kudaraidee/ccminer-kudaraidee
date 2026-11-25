/* BLAKE2b for device code -- shared primitive header (coding guideline section 3).
 *
 * Transcribed from RFC 7693 via the host reference already in this tree
 * (algos/blake2b/blake2b.cu), which is what the blake2b/sia consensus path uses.
 *
 * WHY THIS EXISTS: yespower-b2b (mined as `-a power2b` and `-a yespower-b2b`)
 * is yespower 1.0 with BLAKE2b replacing SHA-256 in the head and the tail only
 * -- smix, pwxform and salsa are byte-identical. So this header supplies the
 * three things that head/tail needs and nothing more:
 *
 * b2b_hash256    unkeyed BLAKE2b, 32-byte digest
 * b2b_hmac256    HMAC-BLAKE2b with 64-byte pads, 32-byte digest
 * b2b_pbkdf2_c1  PBKDF2-HMAC-BLAKE2b with c == 1
 *
 * Sigma is fully unrolled on purpose: an indexed sigma table pins m[16] into
 * local memory. With literal indices this compiles to STACK 0 / LOCAL 0 --
 * check with cuobjdump -res-usage after any edit.
 */

#ifndef CUDA_BLAKE2B_HASH_DEVICE_CUH
#define CUDA_BLAKE2B_HASH_DEVICE_CUH

#include <stdint.h>

#define B2B_BLOCKBYTES 128

/* The HMAC pad is 64 bytes, not the 128-byte BLAKE2b block. Non-standard
 * (RFC 2104 pads to the block size), but it is what the consensus code does
 * (cpuminer-opt algo/yespower/crypto/hmac-blake2b.c); 128 gives digests every
 * pool rejects. The tail's 64-byte key is not pre-hashed (`keylen > 64`). */
#define B2B_HMAC_PAD   64

__device__ __constant__ static const uint64_t c_b2b_iv[8] = {
	0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
	0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
	0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
	0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

__device__ __forceinline__ uint64_t b2b_rotr64(uint64_t x, int n)
{
	return (x >> n) | (x << (64 - n));
}

#define B2B_G(a, b, c, d, x, y) {            \
	v[a] = v[a] + v[b] + (x);                \
	v[d] = b2b_rotr64(v[d] ^ v[a], 32);      \
	v[c] = v[c] + v[d];                      \
	v[b] = b2b_rotr64(v[b] ^ v[c], 24);      \
	v[a] = v[a] + v[b] + (y);                \
	v[d] = b2b_rotr64(v[d] ^ v[a], 16);      \
	v[c] = v[c] + v[d];                      \
	v[b] = b2b_rotr64(v[b] ^ v[c], 63); }

/* One round with its sigma permutation spelled out as literals. */
#define B2B_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb,sc,sd,se,sf) { \
	B2B_G(0, 4,  8, 12, m[s0], m[s1]);  \
	B2B_G(1, 5,  9, 13, m[s2], m[s3]);  \
	B2B_G(2, 6, 10, 14, m[s4], m[s5]);  \
	B2B_G(3, 7, 11, 15, m[s6], m[s7]);  \
	B2B_G(0, 5, 10, 15, m[s8], m[s9]);  \
	B2B_G(1, 6, 11, 12, m[sa], m[sb]);  \
	B2B_G(2, 7,  8, 13, m[sc], m[sd]);  \
	B2B_G(3, 4,  9, 14, m[se], m[sf]); }

typedef struct {
	uint8_t  b[B2B_BLOCKBYTES];
	uint64_t h[8];
	uint64_t t[2];
	uint32_t c;
	uint32_t outlen;
} b2b_ctx;

__device__ __forceinline__ void b2b_compress(b2b_ctx *ctx, int last)
{
	uint64_t v[16], m[16];

#pragma unroll
	for (int i = 0; i < 8; i++) { v[i] = ctx->h[i]; v[i + 8] = c_b2b_iv[i]; }

	v[12] ^= ctx->t[0];
	v[13] ^= ctx->t[1];
	if (last) v[14] = ~v[14];

	/* little-endian load of the 128-byte block */
#pragma unroll
	for (int i = 0; i < 16; i++) {
		const uint8_t *p = ctx->b + 8 * i;
		m[i] =  (uint64_t)p[0]        ^ ((uint64_t)p[1] << 8)  ^
		       ((uint64_t)p[2] << 16) ^ ((uint64_t)p[3] << 24) ^
		       ((uint64_t)p[4] << 32) ^ ((uint64_t)p[5] << 40) ^
		       ((uint64_t)p[6] << 48) ^ ((uint64_t)p[7] << 56);
	}

	B2B_ROUND( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15);
	B2B_ROUND(14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3);
	B2B_ROUND(11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4);
	B2B_ROUND( 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8);
	B2B_ROUND( 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13);
	B2B_ROUND( 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9);
	B2B_ROUND(12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11);
	B2B_ROUND(13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10);
	B2B_ROUND( 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5);
	B2B_ROUND(10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0);
	B2B_ROUND( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15);
	B2B_ROUND(14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3);

#pragma unroll
	for (int i = 0; i < 8; i++) ctx->h[i] ^= v[i] ^ v[i + 8];
}

/* keylen == 0 for the unkeyed hash. A keyed init pads the key to one full
 * block and counts it as the first block, exactly as RFC 7693 specifies. */
__device__ __forceinline__ void b2b_init(b2b_ctx *ctx, uint32_t outlen,
                                         const uint8_t *key, uint32_t keylen)
{
#pragma unroll
	for (int i = 0; i < 8; i++) ctx->h[i] = c_b2b_iv[i];
	ctx->h[0] ^= 0x01010000ULL ^ ((uint64_t)keylen << 8) ^ (uint64_t)outlen;
	ctx->t[0] = 0; ctx->t[1] = 0; ctx->c = 0; ctx->outlen = outlen;

	for (uint32_t i = 0; i < B2B_BLOCKBYTES; i++) ctx->b[i] = 0;
	if (keylen) {
		for (uint32_t i = 0; i < keylen; i++) ctx->b[i] = key[i];
		ctx->c = B2B_BLOCKBYTES;
	}
}

__device__ __forceinline__ void b2b_update(b2b_ctx *ctx, const uint8_t *in, uint32_t inlen)
{
	for (uint32_t i = 0; i < inlen; i++) {
		if (ctx->c == B2B_BLOCKBYTES) {
			ctx->t[0] += ctx->c;
			if (ctx->t[0] < ctx->c) ctx->t[1]++;
			b2b_compress(ctx, 0);
			ctx->c = 0;
		}
		ctx->b[ctx->c++] = in[i];
	}
}

__device__ __forceinline__ void b2b_final(b2b_ctx *ctx, uint8_t *out)
{
	ctx->t[0] += ctx->c;
	if (ctx->t[0] < ctx->c) ctx->t[1]++;
	while (ctx->c < B2B_BLOCKBYTES) ctx->b[ctx->c++] = 0;
	b2b_compress(ctx, 1);

	for (uint32_t i = 0; i < ctx->outlen; i++)
		out[i] = (uint8_t)(ctx->h[i >> 3] >> (8 * (i & 7)));
}

/* ---- the three entry points yespower-b2b needs ------------------------- */

__device__ __forceinline__ void b2b_hash256(uint8_t out[32],
                                            const uint8_t *in, uint32_t inlen)
{
	b2b_ctx c;
	b2b_init(&c, 32, NULL, 0);
	b2b_update(&c, in, inlen);
	b2b_final(&c, out);
}

/* HMAC-BLAKE2b, 32-byte digest, B2B_HMAC_PAD-byte pads. A longer key is hashed
 * first (not reached by the call sites here: 32 and 64 bytes). */
__device__ __forceinline__ void b2b_hmac256(uint8_t out[32],
                                            const uint8_t *key, uint32_t keylen,
                                            const uint8_t *in, uint32_t inlen)
{
	uint8_t k[B2B_HMAC_PAD], pad[B2B_HMAC_PAD], inner[32];
	b2b_ctx c;

#pragma unroll
	for (int i = 0; i < B2B_HMAC_PAD; i++) k[i] = 0;
	if (keylen > B2B_HMAC_PAD) {
 b2b_hash256(k, key, keylen); /* then keylen becomes 32 */
	} else {
		for (uint32_t i = 0; i < keylen; i++) k[i] = key[i];
	}

#pragma unroll
	for (int i = 0; i < B2B_HMAC_PAD; i++) pad[i] = k[i] ^ 0x36;
	b2b_init(&c, 32, NULL, 0);
	b2b_update(&c, pad, B2B_HMAC_PAD);
	b2b_update(&c, in, inlen);
	b2b_final(&c, inner);

#pragma unroll
	for (int i = 0; i < B2B_HMAC_PAD; i++) pad[i] = k[i] ^ 0x5c;
	b2b_init(&c, 32, NULL, 0);
	b2b_update(&c, pad, B2B_HMAC_PAD);
	b2b_update(&c, inner, 32);
	b2b_final(&c, out);
}

/* PBKDF2-HMAC-BLAKE2b with c == 1, which is the only count yespower uses.
 * T_i = HMAC(passwd, salt || BE32(i)), i counted from 1; dkLen need not be a
 * multiple of 32 (it is 128*r here, so it always is). */
__device__ __forceinline__ void b2b_pbkdf2_c1(uint8_t *dk, uint32_t dklen,
                                              const uint8_t *passwd, uint32_t passwdlen,
                                              const uint8_t *salt, uint32_t saltlen)
{
	/* The ipad/opad blocks do not depend on the block index: absorb them once
	 * and clone the states per block. */
	uint8_t k[B2B_HMAC_PAD];
	b2b_ctx ictx, octx;

	for (int q = 0; q < B2B_HMAC_PAD; q++) k[q] = 0;
	if (passwdlen > B2B_HMAC_PAD) b2b_hash256(k, passwd, passwdlen);
	else for (uint32_t q = 0; q < passwdlen; q++) k[q] = passwd[q];

	/* Absorb the 64-byte pad into the block buffer and leave c = 64, so a later
 * update continues in the SAME block -- which is what
 * sph_blake2b_update(ctx, pad, 64) does in the reference. */
	b2b_init(&ictx, 32, NULL, 0);
	for (int q = 0; q < B2B_HMAC_PAD; q++) ictx.b[q] = k[q] ^ 0x36;
	ictx.c = B2B_HMAC_PAD;

	b2b_init(&octx, 32, NULL, 0);
	for (int q = 0; q < B2B_HMAC_PAD; q++) octx.b[q] = k[q] ^ 0x5c;
	octx.c = B2B_HMAC_PAD;

	uint32_t done = 0, i = 1;
	while (done < dklen) {
		const uint8_t be[4] = { (uint8_t)(i >> 24), (uint8_t)(i >> 16),
		                        (uint8_t)(i >> 8),  (uint8_t)i };
		uint8_t inner[32], block[32];
		b2b_ctx c;

 c = ictx; /* ipad already absorbed */
		b2b_update(&c, salt, saltlen);
		b2b_update(&c, be, 4);
		b2b_final(&c, inner);

 c = octx; /* opad already absorbed */
		b2b_update(&c, inner, 32);
		b2b_final(&c, block);

		const uint32_t n = (dklen - done < 32) ? (dklen - done) : 32;
		for (uint32_t q = 0; q < n; q++) dk[done + q] = block[q];
		done += n;
		i++;
	}
}

#endif /* CUDA_BLAKE2B_HASH_DEVICE_CUH */
