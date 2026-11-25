// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Shared SHA-512 primitive library (host + device).
 *
 * FIPS 180-4 SHA-512 core (80 rounds, 64-bit words), structured after
 * cuda/sha256_device.cuh: every cryptographic primitive is a separately
 * callable building block (docs/coding-guideline.md §3); fused kernels call
 * these instead of re-implementing them. The SHA-512/256 IV lives here too
 * (sha512256d is the first consumer).
 */

#ifndef CUDA_SHA512_DEVICE_CUH
#define CUDA_SHA512_DEVICE_CUH

#include <stdint.h>
#include "cuda/helper.cuh" // ROTR64 (funnel shift on device, shift/or on host)

/* Shared initializers so the host copies and the per-TU device copies can
 * never drift apart. */
#define SHA512_K_INIT { \
	0x428A2F98D728AE22ULL, 0x7137449123EF65CDULL, 0xB5C0FBCFEC4D3B2FULL, 0xE9B5DBA58189DBBCULL, \
	0x3956C25BF348B538ULL, 0x59F111F1B605D019ULL, 0x923F82A4AF194F9BULL, 0xAB1C5ED5DA6D8118ULL, \
	0xD807AA98A3030242ULL, 0x12835B0145706FBEULL, 0x243185BE4EE4B28CULL, 0x550C7DC3D5FFB4E2ULL, \
	0x72BE5D74F27B896FULL, 0x80DEB1FE3B1696B1ULL, 0x9BDC06A725C71235ULL, 0xC19BF174CF692694ULL, \
	0xE49B69C19EF14AD2ULL, 0xEFBE4786384F25E3ULL, 0x0FC19DC68B8CD5B5ULL, 0x240CA1CC77AC9C65ULL, \
	0x2DE92C6F592B0275ULL, 0x4A7484AA6EA6E483ULL, 0x5CB0A9DCBD41FBD4ULL, 0x76F988DA831153B5ULL, \
	0x983E5152EE66DFABULL, 0xA831C66D2DB43210ULL, 0xB00327C898FB213FULL, 0xBF597FC7BEEF0EE4ULL, \
	0xC6E00BF33DA88FC2ULL, 0xD5A79147930AA725ULL, 0x06CA6351E003826FULL, 0x142929670A0E6E70ULL, \
	0x27B70A8546D22FFCULL, 0x2E1B21385C26C926ULL, 0x4D2C6DFC5AC42AEDULL, 0x53380D139D95B3DFULL, \
	0x650A73548BAF63DEULL, 0x766A0ABB3C77B2A8ULL, 0x81C2C92E47EDAEE6ULL, 0x92722C851482353BULL, \
	0xA2BFE8A14CF10364ULL, 0xA81A664BBC423001ULL, 0xC24B8B70D0F89791ULL, 0xC76C51A30654BE30ULL, \
	0xD192E819D6EF5218ULL, 0xD69906245565A910ULL, 0xF40E35855771202AULL, 0x106AA07032BBD1B8ULL, \
	0x19A4C116B8D2D0C8ULL, 0x1E376C085141AB53ULL, 0x2748774CDF8EEB99ULL, 0x34B0BCB5E19B48A8ULL, \
	0x391C0CB3C5C95A63ULL, 0x4ED8AA4AE3418ACBULL, 0x5B9CCA4F7763E373ULL, 0x682E6FF3D6B2B8A3ULL, \
	0x748F82EE5DEFB2FCULL, 0x78A5636F43172F60ULL, 0x84C87814A1F0AB72ULL, 0x8CC702081A6439ECULL, \
	0x90BEFFFA23631E28ULL, 0xA4506CEBDE82BDE9ULL, 0xBEF9A3F7B2C67915ULL, 0xC67178F2E372532BULL, \
	0xCA273ECEEA26619CULL, 0xD186B8C721C0C207ULL, 0xEADA7DD6CDE0EB1EULL, 0xF57D4F7FEE6ED178ULL, \
	0x06F067AA72176FBAULL, 0x0A637DC5A2C898A6ULL, 0x113F9804BEF90DAEULL, 0x1B710B35131C471BULL, \
	0x28DB77F523047D84ULL, 0x32CAAB7B40C72493ULL, 0x3C9EBE0A15C9BEBCULL, 0x431D67C49C100D4CULL, \
	0x4CC5D4BECB3E42B6ULL, 0x597F299CFC657E2AULL, 0x5FCB6FAB3AD6FAECULL, 0x6C44198C4A475817ULL }

/* SHA-512/256 IV (FIPS 180-4 §5.3.6.2) — NOT the plain SHA-512 IV. */
#define SHA512_256_H_INIT { \
	0x22312194FC2BF72CULL, 0x9F555FA3C84C64C2ULL, 0x2393B86B6F53B151ULL, 0x963877195940EABDULL, \
	0x96283EE2A88EFFE3ULL, 0xBE5E1E2553863992ULL, 0x2B0199FC2C85B8AAULL, 0x0EB72DDC81C52CA2ULL }

static const uint64_t h_sha512_K[80]     = SHA512_K_INIT;
static const uint64_t h_sha512_256_H[8]  = SHA512_256_H_INIT;

#ifdef __CUDACC__
/* Statically initialized __constant__ copies: no cudaMemcpyToSymbol needed,
 * one 704-byte copy per translation unit. */
static __constant__ uint64_t c_sha512_K[80]    = SHA512_K_INIT;
static __constant__ uint64_t c_sha512_256_H[8] = SHA512_256_H_INIT;
#endif

/* --------------------------------------------------------------------------
 * Round primitives — same LOP3-friendly Ch/Maj forms as the SHA-256 library.
 * ------------------------------------------------------------------------ */

__host__ __device__ __forceinline__
uint64_t sha512_ch(const uint64_t e, const uint64_t f, const uint64_t g)
{
	return ((f ^ g) & e) ^ g; // xandx form
}

__host__ __device__ __forceinline__
uint64_t sha512_maj(const uint64_t a, const uint64_t b, const uint64_t c)
{
	return (b & c) | ((b | c) & a); // andor form
}

__host__ __device__ __forceinline__
uint64_t sha512_bsg0(const uint64_t x) { return ROTR64(x, 28) ^ ROTR64(x, 34) ^ ROTR64(x, 39); }

__host__ __device__ __forceinline__
uint64_t sha512_bsg1(const uint64_t x) { return ROTR64(x, 14) ^ ROTR64(x, 18) ^ ROTR64(x, 41); }

__host__ __device__ __forceinline__
uint64_t sha512_ssg0(const uint64_t x) { return ROTR64(x, 1) ^ ROTR64(x, 8) ^ (x >> 7); }

__host__ __device__ __forceinline__
uint64_t sha512_ssg1(const uint64_t x) { return ROTR64(x, 19) ^ ROTR64(x, 61) ^ (x >> 6); }

/* One compression round, message word supplied by the caller (rounds 0..15,
 * or any round whose schedule word is precomputed). */
__host__ __device__ __forceinline__
void sha512_round(const uint64_t a, const uint64_t b, const uint64_t c, uint64_t &d,
	const uint64_t e, const uint64_t f, const uint64_t g, uint64_t &h,
	const uint64_t in, const uint64_t k)
{
	const uint64_t t1 = h + sha512_bsg1(e) + sha512_ch(e, f, g) + k + in;
	const uint64_t t2 = sha512_bsg0(a) + sha512_maj(a, b, c);
	d += t1;
	h = t1 + t2;
}

/* One compression round for rounds 16..79 over a 16-word rolling schedule:
 * extends in[pc] in place, then compresses. */
__host__ __device__ __forceinline__
void sha512_round_sched(const uint64_t a, const uint64_t b, const uint64_t c, uint64_t &d,
	const uint64_t e, const uint64_t f, const uint64_t g, uint64_t &h,
	uint64_t *in, const uint32_t pc, const uint64_t k)
{
	const uint64_t inx1 = in[(pc - 2) & 0xF];
	const uint64_t inx2 = in[(pc - 7) & 0xF];
	const uint64_t inx3 = in[(pc - 15) & 0xF];

	in[pc] += sha512_ssg1(inx1) + inx2 + sha512_ssg0(inx3);

	sha512_round(a, b, c, d, e, f, g, h, in[pc], k);
}

/* --------------------------------------------------------------------------
 * Full 80-round transform. `in` is the 16-word (128-byte) message block in
 * host word order and is consumed (overwritten by the rolling schedule);
 * `state` is updated in place with the feed-forward. `k` is the round
 * constant table: pass h_sha512_K on the host, c_sha512_K on the device.
 *
 * SHA-512/256 = this transform seeded from SHA512_256_H_INIT, output
 * truncated to state[0..3] (serialized big-endian). The truncation is a
 * pure output rule — both hashes of sha512256d must run all 80 rounds and
 * the full 8-word feed-forward.
 * ------------------------------------------------------------------------ */

__host__ __device__ static inline
void sha512_transform_full(uint64_t *in, uint64_t *state, const uint64_t *k)
{
	uint64_t a = state[0];
	uint64_t b = state[1];
	uint64_t c = state[2];
	uint64_t d = state[3];
	uint64_t e = state[4];
	uint64_t f = state[5];
	uint64_t g = state[6];
	uint64_t h = state[7];

	sha512_round(a, b, c, d, e, f, g, h, in[ 0], k[ 0]);
	sha512_round(h, a, b, c, d, e, f, g, in[ 1], k[ 1]);
	sha512_round(g, h, a, b, c, d, e, f, in[ 2], k[ 2]);
	sha512_round(f, g, h, a, b, c, d, e, in[ 3], k[ 3]);
	sha512_round(e, f, g, h, a, b, c, d, in[ 4], k[ 4]);
	sha512_round(d, e, f, g, h, a, b, c, in[ 5], k[ 5]);
	sha512_round(c, d, e, f, g, h, a, b, in[ 6], k[ 6]);
	sha512_round(b, c, d, e, f, g, h, a, in[ 7], k[ 7]);
	sha512_round(a, b, c, d, e, f, g, h, in[ 8], k[ 8]);
	sha512_round(h, a, b, c, d, e, f, g, in[ 9], k[ 9]);
	sha512_round(g, h, a, b, c, d, e, f, in[10], k[10]);
	sha512_round(f, g, h, a, b, c, d, e, in[11], k[11]);
	sha512_round(e, f, g, h, a, b, c, d, in[12], k[12]);
	sha512_round(d, e, f, g, h, a, b, c, in[13], k[13]);
	sha512_round(c, d, e, f, g, h, a, b, in[14], k[14]);
	sha512_round(b, c, d, e, f, g, h, a, in[15], k[15]);

#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
	for (uint32_t i = 16; i < 80; i += 16) {
		sha512_round_sched(a, b, c, d, e, f, g, h, in,  0, k[i     ]);
		sha512_round_sched(h, a, b, c, d, e, f, g, in,  1, k[i +  1]);
		sha512_round_sched(g, h, a, b, c, d, e, f, in,  2, k[i +  2]);
		sha512_round_sched(f, g, h, a, b, c, d, e, in,  3, k[i +  3]);
		sha512_round_sched(e, f, g, h, a, b, c, d, in,  4, k[i +  4]);
		sha512_round_sched(d, e, f, g, h, a, b, c, in,  5, k[i +  5]);
		sha512_round_sched(c, d, e, f, g, h, a, b, in,  6, k[i +  6]);
		sha512_round_sched(b, c, d, e, f, g, h, a, in,  7, k[i +  7]);
		sha512_round_sched(a, b, c, d, e, f, g, h, in,  8, k[i +  8]);
		sha512_round_sched(h, a, b, c, d, e, f, g, in,  9, k[i +  9]);
		sha512_round_sched(g, h, a, b, c, d, e, f, in, 10, k[i + 10]);
		sha512_round_sched(f, g, h, a, b, c, d, e, in, 11, k[i + 11]);
		sha512_round_sched(e, f, g, h, a, b, c, d, in, 12, k[i + 12]);
		sha512_round_sched(d, e, f, g, h, a, b, c, in, 13, k[i + 13]);
		sha512_round_sched(c, d, e, f, g, h, a, b, in, 14, k[i + 14]);
		sha512_round_sched(b, c, d, e, f, g, h, a, in, 15, k[i + 15]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

/* Seed a state with the SHA-512/256 IV. */
__host__ __device__ __forceinline__
void sha512_256_init_state(uint64_t *state, const uint64_t *hinit)
{
	for (int i = 0; i < 8; i++) state[i] = hinit[i];
}

/* --------------------------------------------------------------------------
 * Per-job prehash for an 80-byte header hashed in one block with the nonce
 * in the low half of w9 (the sha512256d shape). Rounds 0..8 consume only
 * the per-job constants w0..w8, and round 9's t1/t2 are nonce-independent
 * except for the `+ w9` term — so the host runs them once per job and the
 * kernel resumes at round 10.
 * ------------------------------------------------------------------------ */

/* Host side: from header words w0..w8 (w9 not read), produce
 * pre[0..7] = registers a..h after round 8, pre[8] = round-9 t1 minus w9,
 * pre[9] = round-9 t2. Seeded with the SHA-512/256 IV. */
__host__ static inline
void sha512_prehash_split_host(const uint64_t *w, uint64_t *pre)
{
	const uint64_t *k = h_sha512_K;
	uint64_t a = h_sha512_256_H[0];
	uint64_t b = h_sha512_256_H[1];
	uint64_t c = h_sha512_256_H[2];
	uint64_t d = h_sha512_256_H[3];
	uint64_t e = h_sha512_256_H[4];
	uint64_t f = h_sha512_256_H[5];
	uint64_t g = h_sha512_256_H[6];
	uint64_t h = h_sha512_256_H[7];

	sha512_round(a, b, c, d, e, f, g, h, w[0], k[0]);
	sha512_round(h, a, b, c, d, e, f, g, w[1], k[1]);
	sha512_round(g, h, a, b, c, d, e, f, w[2], k[2]);
	sha512_round(f, g, h, a, b, c, d, e, w[3], k[3]);
	sha512_round(e, f, g, h, a, b, c, d, w[4], k[4]);
	sha512_round(d, e, f, g, h, a, b, c, w[5], k[5]);
	sha512_round(c, d, e, f, g, h, a, b, w[6], k[6]);
	sha512_round(b, c, d, e, f, g, h, a, w[7], k[7]);
	sha512_round(a, b, c, d, e, f, g, h, w[8], k[8]);

	pre[0] = a; pre[1] = b; pre[2] = c; pre[3] = d;
	pre[4] = e; pre[5] = f; pre[6] = g; pre[7] = h;
	// round 9 signature (h,a,b,c,d,e,f,g): t1 = g + BSG1(d) + Ch(d,e,f) + K9 (+ w9)
	pre[8] = g + sha512_bsg1(d) + sha512_ch(d, e, f) + k[9];
	pre[9] = sha512_bsg0(h) + sha512_maj(h, a, b);
}

/* Resume the header transform at round 9 from a host prehash. `in` is the
 * full 16-word block with in[9] = (nbits<<32)|nonce set by the caller (in[0..8]
 * are still read by the rounds 16+ schedule); `state` must hold the
 * SHA-512/256 IV for the feed-forward. Bit-exact vs sha512_transform_full. */
__host__ __device__ static inline
void sha512_transform_80_from_pre9(uint64_t *in, const uint64_t *pre, uint64_t *state, const uint64_t *k)
{
	uint64_t a = pre[0];
	uint64_t b = pre[1];
	uint64_t c = pre[2];
	uint64_t d = pre[3];
	uint64_t e = pre[4];
	uint64_t f = pre[5];
	uint64_t g = pre[6];
	uint64_t h = pre[7];

	// round 9, resumed: only the nonce word remains
	const uint64_t t1 = pre[8] + in[9];
	c += t1;
	g = t1 + pre[9];

	sha512_round(g, h, a, b, c, d, e, f, in[10], k[10]);
	sha512_round(f, g, h, a, b, c, d, e, in[11], k[11]);
	sha512_round(e, f, g, h, a, b, c, d, in[12], k[12]);
	sha512_round(d, e, f, g, h, a, b, c, in[13], k[13]);
	sha512_round(c, d, e, f, g, h, a, b, in[14], k[14]);
	sha512_round(b, c, d, e, f, g, h, a, in[15], k[15]);

#ifdef __CUDA_ARCH__
	#pragma unroll
#endif
	for (uint32_t i = 16; i < 80; i += 16) {
		sha512_round_sched(a, b, c, d, e, f, g, h, in,  0, k[i     ]);
		sha512_round_sched(h, a, b, c, d, e, f, g, in,  1, k[i +  1]);
		sha512_round_sched(g, h, a, b, c, d, e, f, in,  2, k[i +  2]);
		sha512_round_sched(f, g, h, a, b, c, d, e, in,  3, k[i +  3]);
		sha512_round_sched(e, f, g, h, a, b, c, d, in,  4, k[i +  4]);
		sha512_round_sched(d, e, f, g, h, a, b, c, in,  5, k[i +  5]);
		sha512_round_sched(c, d, e, f, g, h, a, b, in,  6, k[i +  6]);
		sha512_round_sched(b, c, d, e, f, g, h, a, in,  7, k[i +  7]);
		sha512_round_sched(a, b, c, d, e, f, g, h, in,  8, k[i +  8]);
		sha512_round_sched(h, a, b, c, d, e, f, g, in,  9, k[i +  9]);
		sha512_round_sched(g, h, a, b, c, d, e, f, in, 10, k[i + 10]);
		sha512_round_sched(f, g, h, a, b, c, d, e, in, 11, k[i + 11]);
		sha512_round_sched(e, f, g, h, a, b, c, d, in, 12, k[i + 12]);
		sha512_round_sched(d, e, f, g, h, a, b, c, in, 13, k[i + 13]);
		sha512_round_sched(c, d, e, f, g, h, a, b, in, 14, k[i + 14]);
		sha512_round_sched(b, c, d, e, f, g, h, a, in, 15, k[i + 15]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}


/* ------------------------------------------------------------------------
 * x17-lineage unrolled SHA-512 for a 64-byte input (x-family stage
 * function), extracted bit-identically from algos/stages/cuda_sha512.cu
 * (djm34 2014 / tpruvot 2017). Round constants come from c_sha512_K above
 * (the same FIPS table the donor uploaded into its c_WB symbol).
 * ------------------------------------------------------------------------ */

#define SHA512_SWAB64(u64) cuda_swab64(u64)

#define SHA512_BSG5_0(x) xor3(ROTR64(x,28), ROTR64(x,34), ROTR64(x,39))
#define SHA512_SSG5_0(x) xor3(ROTR64(x, 1), ROTR64(x ,8), ((x) >> 7))
#define SHA512_SSG5_1(x) xor3(ROTR64(x,19), ROTR64(x,61), ((x) >> 6))

//#define SHA512_MAJ(X, Y, Z)   (((X) & (Y)) | (((X) | (Y)) & (Z)))
#define SHA512_MAJ(x, y, z)   andor(x,y,z)

__device__ __forceinline__
uint64_t SHA512_TONE(uint64_t* K, uint64_t* r, uint64_t* W, const int a, const int i)
{
	//asm("// TONE \n");
	const uint64_t e = r[(a+4) & 7];
	uint64_t BSG51 = xor3(ROTR64(e, 14), ROTR64(e, 18), ROTR64(e, 41));
	const uint64_t f = r[(a+5) & 7];
	const uint64_t g = r[(a+6) & 7];
	uint64_t CHl = ((f ^ g) & e) ^ g; // xandx(e, f, g);
	return (r[(a+7) & 7] + BSG51 + CHl + K[i] + W[i]);
}

#define SHA512_STEP(K, r, W, ord, i) { \
	const int a = (8 - ord) & 7; \
	uint64_t T1 = SHA512_TONE(K, r, W, a, i); \
	r[(a+3) & 7] += T1; \
	uint64_t T2 = (SHA512_BSG5_0(r[a]) + SHA512_MAJ(r[a], r[(a+1) & 7], r[(a+2) & 7])); \
	r[(a+7) & 7] = T1 + T2; \
}

/* SHA-512 of a 64-byte input, in place, d_hash word order in and out
 * (big-endian message view handled internally). */
__device__ __forceinline__
void sha512_hash_64(uint64_t *pHash)
{
	// Rolling 16-word schedule (via the shared sha512_transform_full primitive)
	// rather than a flat W[80]: bit-identical output, but the 16-word live
	// window instead of 80 materialized words cuts register/spill pressure and
	// lifts occupancy — event-timed +~4% on RTX 3060 vs the old flat expansion
	// (the win is the register footprint, not the ALU: transform_full's plain
	// `^` already lowers to the same LOP3s as the xor3 macros).
	uint64_t in[16];
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		in[i] = SHA512_SWAB64(pHash[i]);
	}
	in[8] = 0x8000000000000000;
	#pragma unroll
	for (int i = 9; i < 15; i++) {
		in[i] = 0U;
	}
	in[15] = 0x0000000000000200;

	uint64_t st[8] = {
		0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
		0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
		0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
		0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
	};

	sha512_transform_full(in, st, c_sha512_K); // 80 rounds + feed-forward

	#pragma unroll
	for (int u = 0; u < 8; u++) {
		pHash[u] = SHA512_SWAB64(st[u]);
	}
}

#endif // CUDA_SHA512_DEVICE_CUH
