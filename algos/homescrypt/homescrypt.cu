/* homescrypt (Lumenite): HomeScrypt v1.2 kernels and host driver.
 *
 * One 16 MiB scratchpad per hash.  Stages: hs_seed, hs_fill, hs_mix (in
 * chunks), hs_final (fold, SHA-256, target); hs_fused runs the fold of one
 * batch with the fill of the next.  FP is __fmaf_rn only.  Mix reads are
 * plain loads: a round can read a word the previous round wrote.
 * HS_FAULT_* macros are for test builds. */

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos.h"

#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "cuda/sha256_device.cuh"
#include "cuda/selftest_gate.cuh"

#include "homescrypt.h"
#include "homescrypt_kat.h"

#define HS_WORDS     HOMESCRYPT_WORDS
#define HS_MASK      (HS_WORDS - 1)
#define HS_TPB       8u
#define HS_MIX_CHUNK 16384u  /* restart is polled between mix chunks */
#define HS_MAX_RES   8u
#define HS_V_WORDS   16384u  /* scrypt V, 1024 x 128 B */

struct HsState {
	uint64_t s[4];
	uint64_t x;
	uint32_t seed[8];   /* seed digest as big-endian SHA-256 state words */
};

__constant__ static uint32_t c_hs_hdr[20];    /* the be32enc'd header; word 19 is replaced per thread */
__constant__ static uint32_t c_hs_target[8];

__device__ __forceinline__ uint64_t hs_rotl(uint64_t x, unsigned r) { return (x << r) | (x >> (64 - r)); }
/* rotate left by a runtime r in 1..32 as two clamped funnel shifts (sm_7x+) */
__device__ __forceinline__ uint64_t hs_rotv(uint64_t x, uint32_t r)
{
	const uint32_t lo = (uint32_t) x, hi = (uint32_t) (x >> 32);
	return ((uint64_t) __funnelshift_lc(lo, hi, r) << 32) | __funnelshift_lc(hi, lo, r);
}


__device__ __forceinline__ uint32_t hs_bswap(uint32_t v) { return __byte_perm(v, 0, 0x0123); }

__device__ __forceinline__ uint64_t hs_av(uint64_t x)
{
	x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
	x ^= x >> 27; x *= 0x94d049bb133111ebull;
	x ^= x >> 31;
	return x;
}

/* any word -> a positive normal in [2^-9, 2^7): never NaN, inf or denormal */
__device__ __forceinline__ float hs_safe(uint32_t w)
{
	return __uint_as_float(((118u + ((w >> 23) & 15u)) << 23) | (w & 0x7FFFFFu));
}

__device__ __forceinline__ void hs_header(uint32_t hdr[20], uint32_t nonce)
{
	#pragma unroll
	for (int i = 0; i < 19; i++) hdr[i] = c_hs_hdr[i];
	hdr[19] = hs_bswap(nonce);   /* as be32enc(&endiandata[19], nonce) */
}

/* ---- byte-wise SHA-256, used by the seed and final stages ---- */

struct HsSha { uint32_t h[8]; uint32_t w[16]; uint32_t n; };

__device__ __forceinline__ void hs_sha_init(HsSha &c)
{
	#pragma unroll
	for (int i = 0; i < 8; i++) c.h[i] = c_sha256_H[i];
	#pragma unroll
	for (int i = 0; i < 16; i++) c.w[i] = 0;
	c.n = 0;
}

__device__ __noinline__ void hs_sha_byte(HsSha &c, uint32_t b)
{
	const uint32_t p = c.n & 63;
	c.w[p >> 2] |= (b & 0xff) << (24 - 8 * (p & 3));
	c.n++;
	if ((c.n & 63) == 0) {
		sha256_transform_full(c.w, c.h, c_sha256_K);
		#pragma unroll
		for (int i = 0; i < 16; i++) c.w[i] = 0;
	}
}

/* len bytes of a little-endian word array */
__device__ void hs_sha_le(HsSha &c, const uint32_t *v, uint32_t len)
{
	for (uint32_t k = 0; k < len; k++) hs_sha_byte(c, v[k >> 2] >> (8 * (k & 3)));
}

/* big-endian words, e.g. a SHA-256 output */
__device__ void hs_sha_be(HsSha &c, const uint32_t *v, uint32_t nwords)
{
	for (uint32_t k = 0; k < nwords * 4; k++) hs_sha_byte(c, v[k >> 2] >> (24 - 8 * (k & 3)));
}

__device__ void hs_sha_final(HsSha &c, uint32_t out[8])
{
	const uint64_t bits = (uint64_t) c.n * 8;
	hs_sha_byte(c, 0x80);
	while ((c.n & 63) != 56) hs_sha_byte(c, 0);
	for (int i = 7; i >= 0; i--) hs_sha_byte(c, (uint32_t) (bits >> (8 * i)));
	#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = c.h[i];
}

/* ---- scrypt(N=1024, r=1, p=1), Litecoin's scrypt_1024_1_1_256 ---- */

__device__ __forceinline__ void hs_salsa8(uint32_t B[16], const uint32_t Bx[16])
{
	uint32_t x[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) x[i] = (B[i] ^= Bx[i]);
	#pragma unroll
	for (int r = 0; r < 4; r++) {
#define R(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
		x[ 4] ^= R(x[ 0] + x[12],  7); x[ 8] ^= R(x[ 4] + x[ 0],  9);
		x[12] ^= R(x[ 8] + x[ 4], 13); x[ 0] ^= R(x[12] + x[ 8], 18);
		x[ 9] ^= R(x[ 5] + x[ 1],  7); x[13] ^= R(x[ 9] + x[ 5],  9);
		x[ 1] ^= R(x[13] + x[ 9], 13); x[ 5] ^= R(x[ 1] + x[13], 18);
		x[14] ^= R(x[10] + x[ 6],  7); x[ 2] ^= R(x[14] + x[10],  9);
		x[ 6] ^= R(x[ 2] + x[14], 13); x[10] ^= R(x[ 6] + x[ 2], 18);
		x[ 3] ^= R(x[15] + x[11],  7); x[ 7] ^= R(x[ 3] + x[15],  9);
		x[11] ^= R(x[ 7] + x[ 3], 13); x[15] ^= R(x[11] + x[ 7], 18);
		x[ 1] ^= R(x[ 0] + x[ 3],  7); x[ 2] ^= R(x[ 1] + x[ 0],  9);
		x[ 3] ^= R(x[ 2] + x[ 1], 13); x[ 0] ^= R(x[ 3] + x[ 2], 18);
		x[ 6] ^= R(x[ 5] + x[ 4],  7); x[ 7] ^= R(x[ 6] + x[ 5],  9);
		x[ 4] ^= R(x[ 7] + x[ 6], 13); x[ 5] ^= R(x[ 4] + x[ 7], 18);
		x[11] ^= R(x[10] + x[ 9],  7); x[ 8] ^= R(x[11] + x[10],  9);
		x[ 9] ^= R(x[ 8] + x[11], 13); x[10] ^= R(x[ 9] + x[ 8], 18);
		x[12] ^= R(x[15] + x[14],  7); x[13] ^= R(x[12] + x[15],  9);
		x[14] ^= R(x[13] + x[12], 13); x[15] ^= R(x[14] + x[13], 18);
#undef R
	}
	#pragma unroll
	for (int i = 0; i < 16; i++) B[i] += x[i];
}

/* HMAC-SHA256 keyed with the header: longer than a block, so K0 = SHA256(header) */
__device__ void hs_hmac(const uint32_t k0[8], const uint32_t *msg_le, uint32_t len, uint32_t be_index, uint32_t out[8])
{
	HsSha c;
	uint32_t pad[16], inner[8];
	#pragma unroll
	for (int i = 0; i < 16; i++) pad[i] = (i < 8 ? k0[i] : 0) ^ 0x36363636u;
	hs_sha_init(c);
	hs_sha_be(c, pad, 16);
	hs_sha_le(c, msg_le, len);
	hs_sha_be(c, &be_index, 1);
	hs_sha_final(c, inner);
	#pragma unroll
	for (int i = 0; i < 16; i++) pad[i] = (i < 8 ? k0[i] : 0) ^ 0x5c5c5c5cu;
	hs_sha_init(c);
	hs_sha_be(c, pad, 16);
	hs_sha_be(c, inner, 8);
	hs_sha_final(c, out);
}

__global__ void __launch_bounds__(HS_TPB)
hs_seed(uint64_t *V, HsState *st, uint32_t startNonce, uint32_t n)
{
	const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
	if (h >= n) return;
	uint64_t *Sh = V + (size_t) h * HS_V_WORDS;
	uint32_t hdr[20];
	hs_header(hdr, startNonce + h);

	uint32_t k0[8];
	{ HsSha c; hs_sha_init(c); hs_sha_le(c, hdr, 80); hs_sha_final(c, k0); }

	/* B = PBKDF2(hdr, hdr, 1, 128), read as 32 little-endian words */
	uint32_t X[32];
	for (uint32_t blk = 0; blk < 4; blk++) {
		uint32_t u[8];
		hs_hmac(k0, hdr, 80, blk + 1, u);
		#pragma unroll
		for (int i = 0; i < 8; i++) X[blk * 8 + i] = hs_bswap(u[i]);
	}
	/* V in its own buffer: the slot may still be folded */
	for (uint32_t i = 0; i < 1024; i++) {
		#pragma unroll
		for (int k = 0; k < 16; k++)
			Sh[i * 16 + k] = (uint64_t) X[2 * k] | ((uint64_t) X[2 * k + 1] << 32);
		hs_salsa8(&X[0], &X[16]);
		hs_salsa8(&X[16], &X[0]);
	}
	for (uint32_t i = 0; i < 1024; i++) {
		const uint32_t j = X[16] & 1023;
		#pragma unroll
		for (int k = 0; k < 16; k++) {
			const uint64_t v = Sh[j * 16 + k];
			X[2 * k] ^= (uint32_t) v; X[2 * k + 1] ^= (uint32_t) (v >> 32);
		}
		hs_salsa8(&X[0], &X[16]);
		hs_salsa8(&X[16], &X[0]);
	}
	uint32_t sseed[8];
	hs_hmac(k0, X, 128, 1, sseed);

	const uint32_t tag[5] = { 0x454d4f48u, 0x59524353u, 0x562d5450u, 0x2d322e31u, 0x44454553u }; /* "HOMESCRYPT-V1.2-SEED" */
	HsSha c;
	hs_sha_init(c);
	hs_sha_le(c, tag, 20);
	hs_sha_le(c, hdr, 80);
	hs_sha_be(c, sseed, 8);
	uint32_t seed[8];
	hs_sha_final(c, seed);

	HsState o;
	#pragma unroll
	for (int i = 0; i < 4; i++)
		o.s[i] = (uint64_t) hs_bswap(seed[2 * i]) | ((uint64_t) hs_bswap(seed[2 * i + 1]) << 32);
	o.x = o.s[0] ^ hs_rotl(o.s[1], 17) ^ hs_rotl(o.s[2], 31) ^ hs_rotl(o.s[3], 47);
	#pragma unroll
	for (int i = 0; i < 8; i++) o.seed[i] = seed[i];
	st[h] = o;
}

/* fill words [i, i + 8) on sm_7x+; the state lanes rotate with i & 3 */
#define HS_FILL_STEP(q, sa, sb) { const uint32_t ii = i + q; \
	x += 0x9e3779b97f4a7c15ull + ii; x ^= sa + hs_rotv(x, (ii & 31) + 1); x = hs_av(x); \
	w[q & 3] = x; sa ^= x + hs_rotl(sb, 23); }
#define HS_FILL_STORE(o) { ulonglong2 *p = (ulonglong2 *) &Sh[i + o]; \
	p[0] = make_ulonglong2(w[0], w[1]); p[1] = make_ulonglong2(w[2], w[3]); }

__device__ __forceinline__ void hs_fill8(uint64_t *Sh, uint32_t i, uint64_t &s0, uint64_t &s1, uint64_t &s2,
                                         uint64_t &s3, uint64_t &x)
{
	uint64_t w[4];
	HS_FILL_STEP(0, s0, s1) HS_FILL_STEP(1, s1, s2) HS_FILL_STEP(2, s2, s3) HS_FILL_STEP(3, s3, s0)
	HS_FILL_STORE(0)
	HS_FILL_STEP(4, s0, s1) HS_FILL_STEP(5, s1, s2) HS_FILL_STEP(6, s2, s3) HS_FILL_STEP(7, s3, s0)
	HS_FILL_STORE(4)
}
#undef HS_FILL_STEP
#undef HS_FILL_STORE

__global__ void __launch_bounds__(HS_TPB)
hs_fill(uint64_t *S, HsState *st, uint32_t n)
{
	const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
	if (h >= n) return;
	uint64_t *Sh = S + (size_t) h * HS_WORDS;
	uint64_t s0 = st[h].s[0], s1 = st[h].s[1], s2 = st[h].s[2], s3 = st[h].s[3], x = st[h].x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
	for (uint32_t i = 0; i < HS_WORDS; i += 4) {
		uint64_t w[4];
		/* unrolled by 4 so the state lane i & 3 is static */
#define FILL_STEP(k, sa, sb) { \
		const uint32_t ii = i + k; \
		x += 0x9e3779b97f4a7c15ull + ii; \
		x ^= sa + hs_rotl(x, (ii & 31) + 1); \
		x = hs_av(x); \
		w[k] = x; \
		sa ^= x + hs_rotl(sb, 23); }
		FILL_STEP(0, s0, s1)
		FILL_STEP(1, s1, s2)
		FILL_STEP(2, s2, s3)
		FILL_STEP(3, s3, s0)
#undef FILL_STEP
		ulonglong2 *p = (ulonglong2 *) &Sh[i];
		p[0] = make_ulonglong2(w[0], w[1]);
		p[1] = make_ulonglong2(w[2], w[3]);
	}
#else
	for (uint32_t i = 0; i < HS_WORDS; i += 8)
		hs_fill8(Sh, i, s0, s1, s2, s3, x);
#endif
	st[h].s[0] = s0; st[h].s[1] = s1; st[h].s[2] = s2; st[h].s[3] = s3; st[h].x = x;
}

/* rounds [r0, r0 + cnt), r0 and cnt multiples of 4 */
__global__ void __launch_bounds__(HS_TPB)
hs_mix(uint64_t *S, HsState *st, uint32_t n, uint32_t r0, uint32_t cnt)
{
	const uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
	if (h >= n) return;
	uint64_t *Sh = S + (size_t) h * HS_WORDS;
	uint64_t s[4] = { st[h].s[0], st[h].s[1], st[h].s[2], st[h].s[3] };
	uint64_t x = st[h].x;
	for (uint32_t rb = r0; rb < r0 + cnt; rb += 4) {
		#pragma unroll
		for (int q = 0; q < 4; q++) {
			const uint32_t r = rb + q;
			uint64_t &L0 = s[q], &L1 = s[(q + 1) & 3], &L2 = s[(q + 2) & 3], &L3 = s[(q + 3) & 3];
			const uint32_t i1 = (uint32_t) (L0 ^ x ^ r) & HS_MASK;
			const uint64_t a = Sh[i1];
			const uint32_t i2 = (uint32_t) (a ^ hs_rotl(L1, 17)) & HS_MASK;
			const uint64_t b = Sh[i2];
			const uint32_t i3 = (uint32_t) (b ^ hs_rotl(a, 31) ^ L2) & HS_MASK;
			const uint64_t c = Sh[i3];
			const uint64_t mixed = a + hs_rotl(b, 23) + hs_rotl(c, 41) + L3 + r;

			float m0 = hs_safe((uint32_t) (c >> 32)), m1 = hs_safe((uint32_t) c);
			float m2 = hs_safe((uint32_t) (mixed >> 32)), m3 = hs_safe((uint32_t) mixed);
			float g0 = hs_safe((uint32_t) (a >> 32)), g1 = hs_safe((uint32_t) a);
			float g2 = hs_safe((uint32_t) (b >> 32)), g3 = hs_safe((uint32_t) b);
			#pragma unroll
			for (int p = 0; p < (int) HOMESCRYPT_FP_PASSES; p++) {
				#pragma unroll
				for (int j = 0; j < 8; j++) {
					g0 = __fmaf_rn(g0, m0, g3);
					g1 = __fmaf_rn(g1, m1, g0);
					g2 = __fmaf_rn(g2, m2, g1);
					g3 = __fmaf_rn(g3, m3, g2);
				}
				const uint32_t w0 = __float_as_uint(g0), w1 = __float_as_uint(g1);
				const uint32_t w2 = __float_as_uint(g2), w3 = __float_as_uint(g3);
				g0 = hs_safe(w0); g1 = hs_safe(w1); g2 = hs_safe(w2); g3 = hs_safe(w3);
				const uint32_t t = __float_as_uint(m0);
				m0 = hs_safe(__float_as_uint(m1) ^ w0);
				m1 = hs_safe(__float_as_uint(m2) ^ w1);
				m2 = hs_safe(__float_as_uint(m3) ^ w2);
				m3 = hs_safe(t ^ w3);
			}
			const uint64_t fp = (((uint64_t) __float_as_uint(g0) << 32) | __float_as_uint(g1))
				^ hs_rotl(((uint64_t) __float_as_uint(g2) << 32) | __float_as_uint(g3), 29);

			L0 ^= mixed ^ fp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
			L0 = hs_rotl(L0, (unsigned) (mixed & 31) + 1);
#else
			L0 = hs_rotv(L0, (uint32_t) (mixed & 31) + 1);
#endif
			L1 += a ^ hs_rotl(c, 13);

			const uint64_t v1 = a ^ L0 ^ hs_rotl(c, 7);
			const uint64_t v2 = b + L1 + hs_rotl(a, 19);
			const uint64_t v3 = c ^ mixed ^ hs_rotl(b, 37);
			/* the indices can alias: this store order is consensus */
#ifdef HS_FAULT_ALIAS12
			Sh[i2] = v2;
			Sh[i1] = v1;
#else
			Sh[i1] = v1;
			Sh[i2] = v2;
#endif
			Sh[i3] = v3;
			x = hs_rotl(x ^ mixed ^ fp ^ v3, 29);
		}
	}
	st[h].s[0] = s[0]; st[h].s[1] = s[1]; st[h].s[2] = s[2]; st[h].s[3] = s[3]; st[h].x = x;
}

/* Exact 256-bit compare, MSW first, as fulltest() does. */
__device__ __forceinline__ bool hs_below_target(const uint32_t h[8])
{
	#pragma unroll
	for (int i = 7; i >= 0; i--) {
		if (h[i] > c_hs_target[i]) return false;
		if (h[i] < c_hs_target[i]) return true;
	}
	return true;
}

/* ---- fold + final ----
 * A quad of threads per hash, lane k running fold chain k over words 4g + k
 * with loads HS_FOLD_RING groups ahead; lane 0 does the final SHA-256. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define HS_FOLD_RING 8
#else
#define HS_FOLD_RING 16
#endif
#define HS_GROUPS    (HS_WORDS / 4)
#define HS_PUB       64u     /* fold progress is published every HS_PUB groups */

/* Fold progress flag: atomics on sm_7x+; a volatile word and fences on sm_6x
 * (HS_ATOMIC_FLAG forces the atomics there, for test builds). */
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700) || defined(HS_ATOMIC_FLAG)
#define HS_FLAG_T              uint32_t
#define HS_FLAG_SET(p, v)      atomicExch((p), (v))
#define HS_FLAG_GET(p)         atomicAdd((p), 0u)
#define HS_FLAG_ACQUIRE()      __threadfence_block()
#else
#define HS_FLAG_T              volatile uint32_t
#define HS_FLAG_SET(p, v)      (*(p) = (v))
#define HS_FLAG_GET(p)         (*(p))
#define HS_FLAG_ACQUIRE()
#endif
#define HS_FINAL_TPB 32u

template <bool PUBLISH>
__device__ __forceinline__ uint64_t hs_fold_chain(const uint64_t *S, const HsState *st, uint32_t h, uint32_t k,
                                                  unsigned quad, HS_FLAG_T *prog)
{
	static_assert(HS_GROUPS % HS_FOLD_RING == 0 && HS_PUB % HS_FOLD_RING == 0, "ring must divide the groups");
	const uint64_t *Sk = S + (size_t) h * HS_WORDS + k;
	uint64_t f = st[h].s[k];
	uint64_t buf[HS_FOLD_RING];
	#pragma unroll
	for (int j = 0; j < HS_FOLD_RING; j++) buf[j] = Sk[4 * j];
	for (uint32_t g = 0; g < HS_GROUPS; g += HS_FOLD_RING) {
		#pragma unroll
		for (int j = 0; j < HS_FOLD_RING; j++) {
			const uint64_t w = buf[j];
			const uint32_t nx = g + j + HS_FOLD_RING;
			if (nx < HS_GROUPS) buf[j] = Sk[4 * (size_t) nx];
			/* lane k: f0 ^ w, f1 + w, f2 ^ rotl(w,17), f3 + rotl(w,41) */
			const uint64_t w2 = (k == 2) ? hs_rotl(w, 17) : (k == 3) ? hs_rotl(w, 41) : w;
			f = hs_av((k & 1) ? f + w2 : f ^ w2);
		}
		if (PUBLISH && ((g + HS_FOLD_RING) % HS_PUB) == 0) {
			/* the whole quad has consumed these groups: release them to the fill */
			__syncwarp(quad);
			__threadfence_block();
			if (k == 0) HS_FLAG_SET(prog, g + HS_FOLD_RING);
		}
	}
	return f;
}

/* `digests` is NULL when mining; the self-test and -D pass a buffer. */
__device__ __forceinline__ void hs_fold_final(uint64_t f, const HsState *st, uint32_t h, uint32_t k, unsigned quad,
                                              uint32_t startNonce, uint32_t *resNonces, uint32_t *digests)
{
	const uint64_t f1 = __shfl_down_sync(quad, f, 1, 4);
	const uint64_t f2 = __shfl_down_sync(quad, f, 2, 4);
	const uint64_t f3 = __shfl_down_sync(quad, f, 3, 4);
	if (k) return;
	const uint64_t f0 = f;
	uint32_t hdr[20];
	hs_header(hdr, startNonce + h);
	const uint32_t folded[8] = { (uint32_t) f0, (uint32_t) (f0 >> 32), (uint32_t) f1, (uint32_t) (f1 >> 32),
	                             (uint32_t) f2, (uint32_t) (f2 >> 32), (uint32_t) f3, (uint32_t) (f3 >> 32) };
	const uint32_t tag[5] = { 0x454d4f48u, 0x59524353u, 0x562d5450u, 0x2d322e31u, 0x414e4946u }; /* "HOMESCRYPT-V1.2-FINA" */
	uint32_t seed[8];
	#pragma unroll
	for (int i = 0; i < 8; i++) seed[i] = st[h].seed[i];
	HsSha c;
	hs_sha_init(c);
	hs_sha_le(c, tag, 20);
	hs_sha_byte(c, 'L');
	hs_sha_be(c, seed, 8);
	hs_sha_le(c, folded, 32);
	hs_sha_le(c, hdr, 80);
	uint32_t out[8];
	hs_sha_final(c, out);
	#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = hs_bswap(out[i]);   /* uint256 words, [7] most significant */

	if (digests) {
		#pragma unroll
		for (int i = 0; i < 8; i++) digests[h * 8u + i] = out[i];
	}
	if (hs_below_target(out)) {
		const uint32_t pos = atomicAdd(&resNonces[0], 1u);
		if (pos < HS_MAX_RES) resNonces[1u + pos] = startNonce + h;
	}
}

__global__ void __launch_bounds__(HS_FINAL_TPB)
hs_final(const uint64_t *S, const HsState *st, uint32_t startNonce, uint32_t n,
         uint32_t *resNonces, uint32_t *digests)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t h = t >> 2, k = t & 3;
	if (h >= n) return;   /* whole quads: 4n threads in blocks of a multiple of 4 */
	const unsigned quad = 0xFu << (threadIdx.x & 28);
	const uint64_t f = hs_fold_chain<false>(S, st, h, k, quad, NULL);
	hs_fold_final(f, st, h, k, quad, startNonce, resNonces, digests);
}

/* Fold of batch A (st_a) and fill of batch B (st_b) in the same slots: hpb
 * fold quads, then one warp with the hpb fill chains.  The fill of a slot
 * waits, per HS_PUB groups, until the fold has consumed them. */
#define HS_FUSED_HPB_MAX 32u
#define HS_FILL_STEP4(q, sa, sb) { const uint32_t ii = i + q; \
	x += 0x9e3779b97f4a7c15ull + ii; x ^= sa + hs_rotl(x, (ii & 31) + 1); x = hs_av(x); \
	w[q] = x; sa ^= x + hs_rotl(sb, 23); }
__global__ void __launch_bounds__(160)
hs_fused(uint64_t *S, const HsState *st_a, uint32_t start_a, uint32_t n_a, uint32_t *resNonces, uint32_t *digests,
         HsState *st_b, uint32_t n_b, uint32_t hpb)
{
	__shared__ HS_FLAG_T prog[HS_FUSED_HPB_MAX];
	const uint32_t fold_threads = ((4 * hpb + 31) / 32) * 32;
	if (threadIdx.x < hpb)   /* a slot without a fold may be filled at once */
		prog[threadIdx.x] = (blockIdx.x * hpb + threadIdx.x < n_a) ? 0 : HS_GROUPS;
	__syncthreads();

	if (threadIdx.x < fold_threads) {
		const uint32_t j = threadIdx.x >> 2, k = threadIdx.x & 3, h = blockIdx.x * hpb + j;
		if (j >= hpb || h >= n_a) return;
		const unsigned quad = 0xFu << (threadIdx.x & 28);
		const uint64_t f = hs_fold_chain<true>(S, st_a, h, k, quad, &prog[j]);
		hs_fold_final(f, st_a, h, k, quad, start_a, resNonces, digests);
		return;
	}
	const uint32_t j = threadIdx.x - fold_threads, h = blockIdx.x * hpb + j;
	if (j >= hpb || h >= n_b) return;
	uint64_t *Sh = S + (size_t) h * HS_WORDS;
	uint64_t s0 = st_b[h].s[0], s1 = st_b[h].s[1], s2 = st_b[h].s[2], s3 = st_b[h].s[3], x = st_b[h].x;
	for (uint32_t gc = 0; gc < HS_GROUPS; gc += HS_PUB) {
		while (HS_FLAG_GET(&prog[j]) < gc + HS_PUB) { }
		HS_FLAG_ACQUIRE();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
		for (uint32_t i = 4 * gc; i < 4 * (gc + HS_PUB); i += 4) {
			uint64_t w[4];
			HS_FILL_STEP4(0, s0, s1)
			HS_FILL_STEP4(1, s1, s2)
			HS_FILL_STEP4(2, s2, s3)
			HS_FILL_STEP4(3, s3, s0)
			ulonglong2 *p = (ulonglong2 *) &Sh[i];
			p[0] = make_ulonglong2(w[0], w[1]);
			p[1] = make_ulonglong2(w[2], w[3]);
		}
#else
		for (uint32_t i = 4 * gc; i < 4 * (gc + HS_PUB); i += 8)
			hs_fill8(Sh, i, s0, s1, s2, s3, x);
#endif
	}
	st_b[h].s[0] = s0; st_b[h].s[1] = s1; st_b[h].s[2] = s2; st_b[h].s[3] = s3; st_b[h].x = x;
}

static THREAD uint64_t *d_S[MAX_GPUS] = { 0 };
static THREAD uint64_t *d_V[MAX_GPUS] = { 0 };    /* scrypt V, HS_V_WORDS per instance */
static THREAD HsState  *d_st[MAX_GPUS][2] = { { 0 } };
static THREAD uint32_t *d_res[MAX_GPUS] = { 0 };
static THREAD uint32_t hs_instances[MAX_GPUS] = { 0 };
static THREAD uint32_t hs_hpb[MAX_GPUS] = { 0 };
static THREAD uint32_t hs_mix_tpb[MAX_GPUS] = { 0 };   /* 4 on sm_7x+, 8 on sm_6x */
/* The batch whose fill and mix are done and whose fold is still owed. */
static THREAD bool     hs_pend[MAX_GPUS] = { false };
static THREAD int      hs_pend_cur[MAX_GPUS];
static THREAD uint32_t hs_pend_start[MAX_GPUS], hs_pend_n[MAX_GPUS], hs_pend_next[MAX_GPUS];
static THREAD uint32_t hs_pend_hdr[MAX_GPUS][19];
static THREAD uint32_t hs_diff_prev[MAX_GPUS][8];   /* -D: prevhash last checked */
static THREAD time_t   hs_diff_time[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { false };

static inline dim3 hs_grid8(uint32_t n) { return dim3((n + HS_TPB - 1) / HS_TPB); }

/* Seed and fill of a batch into state cur, without a fold (the first batch). */
static bool hs_seed_fill(int dev, int cur, uint32_t start, uint32_t n)
{
	hs_seed<<<hs_grid8(n), HS_TPB>>>(d_V[dev], d_st[dev][cur], start, n);
	hs_fill<<<hs_grid8(n), HS_TPB>>>(d_S[dev], d_st[dev][cur], n);
	return cudaGetLastError() == cudaSuccess;
}

/* Seed of batch cur, then fold of the pending batch prev fused with the fill of cur. */
static bool hs_seed_fused(int dev, int prev, uint32_t start_a, uint32_t n_a, uint32_t *d_dig,
                          int cur, uint32_t start_b, uint32_t n_b)
{
	const uint32_t hpb = hs_hpb[dev], nmax = n_a > n_b ? n_a : n_b;
	if (n_b) hs_seed<<<hs_grid8(n_b), HS_TPB>>>(d_V[dev], d_st[dev][cur], start_b, n_b);
	hs_fused<<<(nmax + hpb - 1) / hpb, ((4 * hpb + 31) / 32) * 32 + 32>>>(d_S[dev], d_st[dev][prev], start_a, n_a,
		d_res[dev], d_dig, d_st[dev][cur], n_b, hpb);
	return cudaGetLastError() == cudaSuccess;
}

/* Mix of batch cur.  1 = a new job arrived between chunks (thr_id >= 0 only),
 * -1 = CUDA error. */
static int hs_mix_all(int thr_id, int dev, int cur, uint32_t n)
{
	for (uint32_t r = 0; r < HOMESCRYPT_MIX_ROUNDS; r += HS_MIX_CHUNK) {
		hs_mix<<<(n + hs_mix_tpb[dev] - 1) / hs_mix_tpb[dev], hs_mix_tpb[dev]>>>(d_S[dev], d_st[dev][cur], n, r, HS_MIX_CHUNK);
		if (cudaDeviceSynchronize() != cudaSuccess) return -1;
		if (thr_id >= 0 && work_restart[thr_id].restart) return 1;
	}
	return 0;
}

static bool hs_fold(int dev, int cur, uint32_t start, uint32_t n, uint32_t *d_dig)
{
	hs_final<<<(4 * n + HS_FINAL_TPB - 1) / HS_FINAL_TPB, HS_FINAL_TPB>>>(d_S[dev], d_st[dev][cur], start, n,
		d_res[dev], d_dig);
	return cudaDeviceSynchronize() == cudaSuccess;
}

static bool hs_arm(int dev)
{
	const uint32_t zero[1u + HS_MAX_RES] = { 0 };
	return cudaMemcpy(d_res[dev], zero, sizeof(zero), cudaMemcpyHostToDevice) == cudaSuccess;
}

/* One non-pipelined batch; digests of n instances into dig (host).  It uses
 * the slots, so a pending batch is dropped. */
static bool hs_run(int dev, uint32_t n, uint32_t startNonce, uint32_t *dig, uint32_t *res)
{
	uint32_t *d_dig = NULL;
	hs_pend[dev] = false;
	bool ok = cudaMalloc(&d_dig, (size_t) n * 32) == cudaSuccess && hs_arm(dev)
	       && hs_seed_fill(dev, 0, startNonce, n) && hs_mix_all(-1, dev, 0, n) == 0
	       && hs_fold(dev, 0, startNonce, n, d_dig)
	       && cudaMemcpy(dig, d_dig, (size_t) n * 32, cudaMemcpyDeviceToHost) == cudaSuccess
	       && (!res || cudaMemcpy(res, d_res[dev], (1u + HS_MAX_RES) * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
	if (d_dig) cudaFree(d_dig);
	return ok;
}

/* Two batches through the pipelined path, digests of both. */
static bool hs_run_pipelined(int dev, uint32_t start, uint32_t n, uint32_t *dig_a, uint32_t *dig_b)
{
	uint32_t *d_dig = NULL;
	hs_pend[dev] = false;
	bool ok = cudaMalloc(&d_dig, (size_t) n * 32) == cudaSuccess && hs_arm(dev)
	       && hs_seed_fill(dev, 0, start, n) && hs_mix_all(-1, dev, 0, n) == 0
	       && hs_seed_fused(dev, 0, start, n, d_dig, 1, start + n, n) && cudaDeviceSynchronize() == cudaSuccess
	       && cudaMemcpy(dig_a, d_dig, (size_t) n * 32, cudaMemcpyDeviceToHost) == cudaSuccess
	       && hs_mix_all(-1, dev, 1, n) == 0 && hs_fold(dev, 1, start + n, n, d_dig)
	       && cudaMemcpy(dig_b, d_dig, (size_t) n * 32, cudaMemcpyDeviceToHost) == cudaSuccess;
	if (d_dig) cudaFree(d_dig);
	return ok;
}

/* 256-bit compare of two uint256 word arrays, [7] most significant */
static int hs_cmp256(const uint32_t *a, const uint32_t *b)
{
	for (int i = 7; i >= 0; i--)
		if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
	return 0;
}

/* Fail-closed init gate: CPU KAT, the kernels on the full-shape vectors, a
 * ragged batch vs the CPU with an exact target report, and the pipelined path. */
#define HS_ST_BATCH 13u
#define HS_ST_START 0x51u

static bool homescrypt_device_selftest(int thr_id, int dev)
{
	char det[HOMESCRYPT_SELFTEST_DETAIL];
	if (!homescrypt_self_test(det, sizeof det)) {
		gpulog(LOG_ERR, thr_id, "homescrypt CPU self-test FAILED: %s", det);
		return selftest_gate(thr_id, "homescrypt", false);
	}
	gpulog(LOG_INFO, thr_id, "homescrypt CPU self-test OK (%s)", det);

	/* single-instance legs: every full-shape KAT vector through the mining kernels */
	bool kat_ok = true;
	int legs = 0;
	for (const homescrypt_kat_vec &v : homescrypt_kat) {
		if (v.mix_rounds != HOMESCRYPT_MIX_ROUNDS || v.fp_passes != HOMESCRYPT_FP_PASSES)
			continue;
		uint32_t hdr[20], dig[8];
		memcpy(hdr, v.header, 80);
		if (v.override_nonce) memcpy(&hdr[19], &v.nonce, 4);
		if (cudaMemcpyToSymbol(c_hs_hdr, hdr, 80) != cudaSuccess ||
		    !hs_run(dev, 1, cuda_swab32(hdr[19]), dig, NULL))
			return selftest_gate(thr_id, "homescrypt", selftest_cuda_fault());
		legs++;
		if (memcmp(dig, v.digest, 32)) {
			gpulog(LOG_ERR, thr_id, "homescrypt GPU self-test: %s differs", v.name);
			kat_ok = false;
		}
	}

	/* batch leg: consecutive nonces vs the CPU, and the target report */
	uint32_t hdr[20], dig[HS_ST_BATCH * 8], ref[2 * HS_ST_BATCH * 8], res[1u + HS_MAX_RES], target[8];
	memcpy(hdr, homescrypt_kat[0].header, 80);
	for (uint32_t i = 0; i < 2 * HS_ST_BATCH; i++) {
		uint32_t h[20];
		memcpy(h, hdr, 80);
		be32enc(&h[19], HS_ST_START + i);
		homescrypt_hash(ref + i * 8, h);
	}
	/* target = the 4th-lowest digest, so exactly four report */
	int order[HS_ST_BATCH];
	for (uint32_t i = 0; i < HS_ST_BATCH; i++) order[i] = i;
	for (uint32_t a = 1; a < HS_ST_BATCH; a++)
		for (uint32_t b = a; b > 0 && hs_cmp256(ref + order[b] * 8, ref + order[b - 1] * 8) < 0; b--) {
			const int t = order[b]; order[b] = order[b - 1]; order[b - 1] = t;
		}
	memcpy(target, ref + order[3] * 8, 32);
	if (cudaMemcpyToSymbol(c_hs_hdr, hdr, 80) != cudaSuccess ||
	    cudaMemcpyToSymbol(c_hs_target, target, 32) != cudaSuccess ||
	    !hs_run(dev, HS_ST_BATCH, HS_ST_START, dig, res))
		return selftest_gate(thr_id, "homescrypt", selftest_cuda_fault());
	const bool batch_ok = memcmp(dig, ref, sizeof(dig)) == 0;
	uint32_t want = 0, seen = 0;
	for (int k = 0; k < 4; k++) want |= 1u << order[k];
	bool report_ok = res[0] == 4u;
	for (uint32_t s = 0; report_ok && s < res[0]; s++) {
		const uint32_t i = res[1u + s] - HS_ST_START;
		report_ok = i < HS_ST_BATCH && !(seen & (1u << i));
		seen |= 1u << i;
	}
	report_ok = report_ok && seen == want;

	/* pipelined leg: the fused fold + fill on two consecutive batches */
	uint32_t dig_b[HS_ST_BATCH * 8];
	if (!hs_run_pipelined(dev, HS_ST_START, HS_ST_BATCH, dig, dig_b))
		return selftest_gate(thr_id, "homescrypt", selftest_cuda_fault());
	const bool pipe_ok = memcmp(dig, ref, sizeof(dig)) == 0 && memcmp(dig_b, ref + HS_ST_BATCH * 8, sizeof(dig_b)) == 0;

	const bool passed = kat_ok && batch_ok && report_ok && pipe_ok && legs == 7;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "homescrypt GPU self-test FAILED: kat=%d (%d legs) batch=%d report=%d pipeline=%d (count %u)",
		       (int) kat_ok, legs, (int) batch_ok, (int) report_ok, (int) pipe_ok, res[0]);
	else
		gpulog(LOG_INFO, thr_id, "homescrypt GPU self-test OK (3 mainnet, zero-header, 3 alias vectors, "
		       "%u-nonce batch, target report, pipelined batches)", HS_ST_BATCH);
	return selftest_gate(thr_id, "homescrypt", passed);
}

/* -D: the kernels' digests for HS_DIFF_SPAN nonces vs the CPU reference, as a
 * plain and a (2*nonce+1)-weighted XOR, on a new prevhash or every
 * HS_DIFF_EVERY s. */
#define HS_DIFF_SPAN  61u
#define HS_DIFF_OFF   0x51u
#define HS_DIFF_EVERY 60

static void hs_debug_differential(int thr_id, int dev, const uint32_t pdata[20],
                                  const uint32_t *endiandata, uint32_t base)
{
	const time_t now = time(NULL);
	const bool new_block = memcmp(hs_diff_prev[dev], &pdata[1], 32) != 0;
	if (!new_block && hs_diff_time[dev] && now - hs_diff_time[dev] < HS_DIFF_EVERY)
		return;
	memcpy(hs_diff_prev[dev], &pdata[1], 32);
	hs_diff_time[dev] = now;

	const uint32_t start = base + HS_DIFF_OFF;
	const uint32_t span = HS_DIFF_SPAN < hs_instances[dev] ? HS_DIFF_SPAN : hs_instances[dev];
	uint32_t dig[HS_DIFF_SPAN * 8];
	unsigned long long g[2] = { 0ull, 0ull }, c[2] = { 0ull, 0ull };
	if (!hs_run(dev, span, start, dig, NULL)) {
		gpulog(LOG_WARNING, thr_id, "homescrypt differential: could not run (CUDA resource "
		       "failure) -- not evidence of a wrong hash");
		return;
	}
	uint32_t _ALIGN(64) endian[20];
	memcpy(endian, endiandata, sizeof(endian));
	for (uint32_t i = 0; i < span; i++) {
		const uint32_t nonce = start + i;
		uint32_t h[8];
		unsigned long long q = ((unsigned long long) dig[i * 8u + 7u] << 32) | dig[i * 8u + 6u];
		unsigned long long w = 2ull * (unsigned long long) nonce + 1ull;
#ifdef HS_FAULT_DIFF_CONST
		q ^= 1ull;
#endif
#ifdef HS_FAULT_DIFF_PERM
		w = 2ull * (unsigned long long) (nonce ^ 1u) + 1ull;
#endif
		g[0] ^= q;
		g[1] ^= q * w;
		be32enc(&endian[19], nonce);
		homescrypt_hash(h, endian);
		const unsigned long long p = ((unsigned long long) h[7] << 32) | h[6];
		c[0] ^= p;
		c[1] ^= p * (2ull * (unsigned long long) nonce + 1ull);
	}
	if (g[0] == c[0] && g[1] == c[1])
		gpulog(LOG_DEBUG, thr_id, "homescrypt differential ok: %u nonces from %08x, "
		       "acc %016llx/%016llx", span, start, g[0], g[1]);
	else
		gpulog(LOG_ERR, thr_id, "homescrypt DIFFERENTIAL MISMATCH over %u nonces from %08x "
		       "(acc0 %s, acc1 %s): gpu %016llx/%016llx != cpu %016llx/%016llx",
		       span, start, g[0] == c[0] ? "same" : "MOVED",
		       g[1] == c[1] ? "same" : "MOVED", g[0], g[1], c[0], c[1]);
}

/* Host re-verify of the folded batch's candidates into work->nonces, all of
 * them (up to MAX_NONCES): the scan does not come back.  Returns their count,
 * or -1 on a CUDA error. */
static int hs_check(int thr_id, int dev, struct work *work, uint32_t *endiandata)
{
	uint32_t res[1u + HS_MAX_RES], vhash[8];
	if (cudaMemcpy(res, d_res[dev], sizeof(res), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
	if (res[0] == 0u) return 0;
	uint32_t cand[HS_MAX_RES], ncand = res[0];
	int found = 0;
	if (ncand > HS_MAX_RES) {
		applog(LOG_WARNING, "GPU #%d: homescrypt candidates flood: %u (keeping %u)",
		       device_map[thr_id], ncand, (uint32_t) HS_MAX_RES);
		ncand = HS_MAX_RES;
	}
	for (uint32_t s = 0; s < ncand; s++) cand[s] = res[1u + s];
	for (uint32_t a = 1; a < ncand; a++) {   /* ascending */
		const uint32_t v = cand[a];
		uint32_t b = a;
		while (b > 0 && cand[b - 1] > v) { cand[b] = cand[b - 1]; b--; }
		cand[b] = v;
	}
	for (uint32_t s = 0; s < ncand && found < MAX_NONCES; s++) {
		be32enc(&endiandata[19], cand[s]);
		homescrypt_hash(vhash, endiandata);
		if (vhash[7] <= work->target[7] && fulltest(vhash, work->target)) {
			work->nonces[found] = cand[s];
			if (found == 0) work_set_target_ratio(work, vhash);
			else            bn_set_target_ratio(work, vhash, found);
			found++;
		} else {
			gpu_increment_reject(thr_id);
			applog(LOG_WARNING, "GPU #%d: homescrypt result %08x does not validate",
			       device_map[thr_id], cand[s]);
		}
	}
	return found;
}

extern "C" int scanhash_homescrypt(int thr_id, struct work *work, uint32_t max_nonce,
                                   unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t _ALIGN(64) endiandata[20];
	const int dev = device_map[thr_id];
	unsigned long done = 0;
	time_t t_start;

	/* the benchmark rescans one small window, so make a hit in it likely */
	if (opt_benchmark)
		ptarget[7] = 0x003fffff;

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	if (!init[dev]) {
		CUDA_CALL_OR_RET_X(cudaSetDevice(device_map[thr_id]), -1);
		const size_t per_inst = (size_t) (HS_WORDS + HS_V_WORDS) * 8 + 2 * sizeof(HsState);
		size_t avail = (size_t) cuda_available_memory(thr_id) * 1024u * 1024u;
		avail = (avail > (256u << 20)) ? (avail - (256u << 20)) : 0u;
		const uint32_t fit = (uint32_t) (avail / per_inst);
		/* -i caps the instance count; the default is whatever fits */
		const uint32_t want = cuda_default_throughput(thr_id, fit);
		hs_instances[dev] = (want < fit) ? want : fit;
		if (hs_instances[dev] < HS_ST_BATCH) {
			applog(LOG_ERR, "homescrypt: GPU #%d needs 16 MB per instance and fits only %u "
			       "(need %u)", device_map[thr_id], fit, HS_ST_BATCH);
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}
		/* one fused block per SM */
		hs_hpb[dev] = (hs_instances[dev] + device_mpcount[dev] - 1) / device_mpcount[dev];
		if (hs_hpb[dev] > HS_FUSED_HPB_MAX) hs_hpb[dev] = HS_FUSED_HPB_MAX;
		hs_mix_tpb[dev] = device_sm[dev] >= 700 ? 4u : HS_TPB;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_S[dev], (size_t) HS_WORDS * 8 * hs_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_V[dev], (size_t) HS_V_WORDS * 8 * hs_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_st[dev][0], sizeof(HsState) * hs_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_st[dev][1], sizeof(HsState) * hs_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_res[dev], (1u + HS_MAX_RES) * sizeof(uint32_t)), -1);

		applog(LOG_INFO, "GPU #%d: homescrypt %u instances, %.0f MB of scratchpad",
		       device_map[thr_id], hs_instances[dev], (double) hs_instances[dev] * 16.125);
		homescrypt_device_selftest(thr_id, dev);
		init[dev] = true;
	}

	/* before the header upload: the differential borrows c_hs_hdr */
	if (opt_debug) {
		cudaMemcpyToSymbol(c_hs_hdr, endiandata, 80);
		hs_debug_differential(thr_id, dev, pdata, endiandata, first_nonce);
	}
	cudaMemcpyToSymbol(c_hs_hdr, endiandata, 80);
	cudaMemcpyToSymbol(c_hs_target, ptarget, 32);

	/* a pending batch carries over only within the same header */
	uint32_t n = first_nonce;
	if (hs_pend[dev]) {
		if (memcmp(hs_pend_hdr[dev], endiandata, sizeof(hs_pend_hdr[dev])))
			hs_pend[dev] = false;
		else if (n < hs_pend_next[dev] && n + hs_instances[dev] > hs_pend_start[dev])
			n = hs_pend_next[dev];
	}

	const uint32_t batch = hs_instances[dev];
	t_start = time(NULL);
	*hashes_done = 0;

	for (;;) {
		if (work_restart[thr_id].restart)
			break;                       /* a pending batch of the old job is dropped next call */
		const bool launch = (uint64_t) n + batch < (uint64_t) max_nonce;
		const bool fold = hs_pend[dev];
		if (!launch && !fold)
			break;
		const int cur = fold ? 1 - hs_pend_cur[dev] : 0;
		const uint32_t nb = launch ? batch : 0;

		if (!hs_arm(dev)) {
			applog(LOG_ERR, "homescrypt: GPU #%d upload failed", device_map[thr_id]);
			return -1;
		}
		bool ok = fold ? hs_seed_fused(dev, hs_pend_cur[dev], hs_pend_start[dev], hs_pend_n[dev], NULL, cur, n, nb)
		               : hs_seed_fill(dev, cur, n, nb);
		const uint32_t a_n = fold ? hs_pend_n[dev] : 0;
		hs_pend[dev] = false;
		const int mix = (ok && nb) ? hs_mix_all(thr_id, dev, cur, nb) : (ok ? 1 : -1);
		if (mix < 0 || cudaDeviceSynchronize() != cudaSuccess) {
			applog(LOG_ERR, "homescrypt: GPU #%d launch failed", device_map[thr_id]);
			return -1;
		}
		if (mix == 0) {   /* cur is filled and mixed: its fold is owed */
			hs_pend[dev] = true;
			hs_pend_cur[dev] = cur;
			hs_pend_start[dev] = n;
			hs_pend_n[dev] = nb;
			memcpy(hs_pend_hdr[dev], endiandata, sizeof(hs_pend_hdr[dev]));
			n += nb;
			hs_pend_next[dev] = n;
		}
		if (fold) {
			done += a_n;
			const int found = hs_check(thr_id, dev, work, endiandata);
			if (found < 0) {
				applog(LOG_ERR, "homescrypt: GPU #%d readback failed", device_map[thr_id]);
				return -1;
			}
			if (found) {
				/* nonces[0] is what is submitted; the pending batch continues at n */
				work->valid_nonces = found;
				*hashes_done = done;
				pdata[19] = n;
				return found;
			}
		}
		if (mix == 1 && nb)
			break;                       /* new job: the unmixed batch is dropped */
		if (time(NULL) - t_start >= 1)
			break;                       /* yield: report the rate, poll for work */
		if (!launch)
			break;
	}

	*hashes_done = done;
	pdata[19] = n;
	return 0;
}

/* Must stay registered in algo_free_all(): an algo switch re-arms init here. */
extern "C" void free_homescrypt(int thr_id)
{
	const int dev = device_map[thr_id];

	if (!init[dev])
		return;

	cudaSetDevice(dev);
	cudaDeviceSynchronize();

	if (d_S[dev])     cudaFree(d_S[dev]);
	if (d_V[dev])     cudaFree(d_V[dev]);
	if (d_st[dev][0]) cudaFree(d_st[dev][0]);
	if (d_st[dev][1]) cudaFree(d_st[dev][1]);
	if (d_res[dev])   cudaFree(d_res[dev]);
	d_S[dev] = NULL; d_V[dev] = NULL; d_st[dev][0] = NULL; d_st[dev][1] = NULL; d_res[dev] = NULL;

	hs_instances[dev] = 0;
	hs_pend[dev] = false;
	hs_diff_time[dev] = 0;
	init[dev] = false;
}
