/*
 * secp256k1 on 8 x 32-bit limbs: field (PTX carry-chain products, lazy values),
 * Jacobian + affine addition and the 64-byte table entry load. Used by bc3hash
 * and curvehash.
 */
#pragma once
#include "bc3_uint_device.cuh"
#include "bc3_modinv_device.cuh"

/* ---- secp256k1 field on 8 x 32-bit limbs ------------------------------------
 * p = 2^256 - C, C = 2^32 + 977, on the mix's products (bc3_mul_full /
 * bc3_sqr_full). Values are lazy (x or x + p); results leaving this file are
 * made canonical. */
#define BC3_SECP_C0 977u

/* + C to r where m is all ones (a carry out of 2^256 folded back); returns the
 * carry out of that addition as a mask */
__device__ __forceinline__ uint32_t bc3_fp_addc(bc3u256& r, uint32_t m)
{
	uint64_t c = (uint64_t)r.w[0] + (BC3_SECP_C0 & m);
	r.w[0] = (uint32_t)c;
	c = (c >> 32) + r.w[1] + (1u & m);
	r.w[1] = (uint32_t)c;
#pragma unroll
	for (int i = 2; i < 8; i++) {
		c = (c >> 32) + r.w[i];
		r.w[i] = (uint32_t)c;
	}
	return 0u - (uint32_t)(c >> 32);
}

/* - C from r where m is all ones (a borrow below 0 folded back, i.e. + p);
 * returns the borrow out of that subtraction as a mask */
__device__ __forceinline__ uint32_t bc3_fp_subc(bc3u256& r, uint32_t m)
{
	int64_t c = (int64_t)r.w[0] - (int64_t)(BC3_SECP_C0 & m);
	r.w[0] = (uint32_t)c;
	c = (c >> 32) + (int64_t)r.w[1] - (int64_t)(1u & m);
	r.w[1] = (uint32_t)c;
#pragma unroll
	for (int i = 2; i < 8; i++) {
		c = (c >> 32) + (int64_t)r.w[i];
		r.w[i] = (uint32_t)c;
	}
	return 0u - (uint32_t)(c < 0);
}

/* t (16 words) mod p, to 8 words, not canonical. t = lo + hi*977 + hi*2^32. */
#ifndef BC3_MUL_EO
/* First fold with A0 seeded as lo and A1 as hi (= hi*2^32), plus the hi*977
 * row: q < 2^290, w = q >> 256 < 2^34. Second fold + w*(2^32 + 977) < 2^67; a
 * wrap leaves r < 2^67, so + C stays within r0..r2. */
__device__ __forceinline__ bc3u256 bc3_fp_reduce(const uint32_t (&t)[16])
{
	uint32_t A0[16], A1[16], q[10];
	bool D0[16], D1[16];
#pragma unroll
	for (int k = 0; k < 16; k++) {
		D0[k] = D1[k] = k < 8;
		A0[k] = k < 8 ? t[k] : 0u;
		A1[k] = k < 8 ? t[8 + k] : 0u;
	}
	bc3_mul_chain<8, 0, 0, 0, 9>(A0, D0, BC3_SECP_C0, t + 8);
	bc3_mul_chain<8, 0, 1, 1, 9>(A1, D1, BC3_SECP_C0, t + 8);
	bc3_mul_join<8, 10>(q, A0, D0, A1, D1);
	/* column 1: hi(w0*977) + w1*977 + w0 (33 bits: v1, carry into v2) */
	const uint32_t w0 = q[8], w1 = q[9];
	uint32_t s1, v1, v2;
	BC3_MAC0("mad.lo.u32", s1, w1, BC3_SECP_C0);
	BC3_ACC("mad.hi.u32", s1, w0, BC3_SECP_C0);
	BC3_A3("add.cc.u32", v1, s1, w0);
	BC3_A2I("addc.u32", v2, w1);
	bc3u256 r;
#pragma unroll
	for (int k = 0; k < 8; k++) r.w[k] = q[k];
	BC3_ACC("mad.lo.cc.u32", r.w[0], w0, BC3_SECP_C0);
	BC3_ACCI("addc.cc.u32", r.w[1], v1);
	BC3_ACCI("addc.cc.u32", r.w[2], v2);
#pragma unroll
	for (int k = 3; k < 8; k++) BC3_A2I("addc.cc.u32", r.w[k], r.w[k]);
	uint32_t c;
	BC3_A0I("addc.u32", c);
	const uint32_t m = 0u - c;
	BC3_ACCI("add.cc.u32", r.w[0], m & BC3_SECP_C0);
	BC3_ACCI("addc.cc.u32", r.w[1], m & 1u);
	BC3_A2I("addc.u32", r.w[2], r.w[2]);
	return r;
}
#else
__device__ __forceinline__ bc3u256 bc3_fp_reduce(const uint32_t (&t)[16])
{
	bc3u256 r;
	uint64_t c = 0;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		c += (uint64_t)t[i] + (uint64_t)t[8 + i] * BC3_SECP_C0 + (i ? t[7 + i] : 0u);
		r.w[i] = (uint32_t)c;
		c >>= 32;
	}
	const uint64_t w = c + t[15]; /* < 2^44 */
	const uint64_t f = w * BC3_SECP_C0;
	c = (uint64_t)r.w[0] + (uint32_t)f;
	r.w[0] = (uint32_t)c;
	c = (c >> 32) + r.w[1] + (f >> 32) + (uint32_t)w;
	r.w[1] = (uint32_t)c;
	c = (c >> 32) + r.w[2] + (w >> 32);
	r.w[2] = (uint32_t)c;
#pragma unroll
	for (int i = 3; i < 8; i++) {
		c = (c >> 32) + r.w[i];
		r.w[i] = (uint32_t)c;
	}
	/* wrapped past 2^256: r is now below 2^77, so + C cannot wrap again */
	bc3_fp_addc(r, 0u - (uint32_t)(c >> 32));
	return r;
}
#endif

__device__ __forceinline__ bc3u256 bc3_fp_mul(const bc3u256& a, const bc3u256& b)
{
	uint32_t t[16];
	bc3_mul_full<8>(t, a.w, b.w);
	return bc3_fp_reduce(t);
}
__device__ __forceinline__ bc3u256 bc3_fp_sqr(const bc3u256& a)
{
	uint32_t t[16];
	bc3_sqr_full<8>(t, a.w);
	return bc3_fp_reduce(t);
}

/* a + b; a carry out of 2^256 is folded back as + C, which itself carries out
 * (once at most) when a + b > 2^257 - C */
__device__ __forceinline__ bc3u256 bc3_fp_add(const bc3u256& a, const bc3u256& b)
{
	bc3u256 r;
	uint64_t c = 0;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		c = (c >> 32) + a.w[i] + b.w[i];
		r.w[i] = (uint32_t)c;
	}
	const uint32_t m = bc3_fp_addc(r, 0u - (uint32_t)(c >> 32));
	bc3_fp_addc(r, m);
	return r;
}

/* a - b; a borrow is folded back as + p (- C), which itself borrows (once at
 * most) when b - a > p */
__device__ __forceinline__ bc3u256 bc3_fp_sub(const bc3u256& a, const bc3u256& b)
{
	bc3u256 r;
	int64_t c = 0;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		c = (c >> 32) + (int64_t)a.w[i] - (int64_t)b.w[i];
		r.w[i] = (uint32_t)c;
	}
	const uint32_t m = bc3_fp_subc(r, 0u - (uint32_t)(c < 0));
	bc3_fp_subc(r, m);
	return r;
}

/* canonical: x >= p exactly when x + C carries out of 2^256 */
__device__ __forceinline__ bc3u256 bc3_fp_canon(const bc3u256& a)
{
	bc3u256 s = a;
	return bc3_fp_addc(s, 0xffffffffu) ? s : a;
}

/* lazy x == 0 (mod p): x is 0 or p */
__device__ __forceinline__ bool bc3_fp_is0(const bc3u256& a)
{
	uint32_t z = 0, q = (a.w[0] ^ 0xfffffc2fu) | (a.w[1] ^ 0xfffffffeu);
#pragma unroll
	for (int i = 0; i < 8; i++) z |= a.w[i];
#pragma unroll
	for (int i = 2; i < 8; i++) q |= ~a.w[i];
	return z == 0 || q == 0;
}

struct bc3_jac {
	bc3u256 x, y, z; /* affine (x / z^2, y / z^3) */
	bool inf;
};

/* r = 2r (a = 0); reached only when an addend equals the running sum */
#ifdef BC3_RT_DIGITS
__device__ __noinline__ void bc3_jac_double(bc3_jac& r)
#else
__device__ __forceinline__ void bc3_jac_double(bc3_jac& r)
#endif
{
	const bc3u256 a = bc3_fp_sqr(r.x), b = bc3_fp_sqr(r.y), c = bc3_fp_sqr(b);
	bc3u256 d = bc3_fp_sub(bc3_fp_sub(bc3_fp_sqr(bc3_fp_add(r.x, b)), a), c);
	d = bc3_fp_add(d, d);
	const bc3u256 e = bc3_fp_add(bc3_fp_add(a, a), a);
	const bc3u256 x3 = bc3_fp_sub(bc3_fp_sqr(e), bc3_fp_add(d, d));
	bc3u256 c8 = bc3_fp_add(c, c);
	c8 = bc3_fp_add(c8, c8);
	c8 = bc3_fp_add(c8, c8);
	const bc3u256 y3 = bc3_fp_sub(bc3_fp_mul(e, bc3_fp_sub(d, x3)), c8);
	const bc3u256 yz = bc3_fp_mul(r.y, r.z);
	r.z = bc3_fp_add(yz, yz);
	r.x = x3;
	r.y = y3;
}

/* r += (x2, y2) affine, with the degenerate cases of libsecp's gej_add_ge_var */
__device__ __forceinline__ void bc3_jac_add_affine(bc3_jac& r, const bc3u256& x2, const bc3u256& y2)
{
	if (r.inf) {
		r.x = x2;
		r.y = y2;
		r.z = bc3_small<8>(1);
		r.inf = false;
		return;
	}
	const bc3u256 z12 = bc3_fp_sqr(r.z);
	const bc3u256 u2 = bc3_fp_mul(x2, z12);
	const bc3u256 s2 = bc3_fp_mul(bc3_fp_mul(y2, r.z), z12);
	const bc3u256 h = bc3_fp_sub(u2, r.x);
	const bc3u256 i = bc3_fp_sub(s2, r.y);
	if (bc3_fp_is0(h)) {
		if (bc3_fp_is0(i)) bc3_jac_double(r);
		else r.inf = true;
		return;
	}
	const bc3u256 h2 = bc3_fp_sqr(h);
	const bc3u256 h3 = bc3_fp_mul(h, h2);
	const bc3u256 t = bc3_fp_mul(r.x, h2);
	const bc3u256 x3 = bc3_fp_sub(bc3_fp_sub(bc3_fp_sqr(i), h3), bc3_fp_add(t, t));
	r.y = bc3_fp_sub(bc3_fp_mul(i, bc3_fp_sub(t, x3)), bc3_fp_mul(r.y, h3));
	r.z = bc3_fp_mul(r.z, h);
	r.x = x3;
}

/* table entry (big-endian X || Y, 64 B aligned) as two limb vectors */
__device__ __forceinline__ void bc3_ec_load(bc3u256& x, bc3u256& y, const uint8_t* __restrict__ p)
{
	const uint4* q = (const uint4*)p;
	const uint4 a = __ldg(q), b = __ldg(q + 1), c = __ldg(q + 2), d = __ldg(q + 3);
	const uint32_t xs[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
	const uint32_t ys[8] = { c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w };
#pragma unroll
	for (int k = 0; k < 8; k++) {
		x.w[7 - k] = __byte_perm(xs[k], 0, 0x0123);
		y.w[7 - k] = __byte_perm(ys[k], 0, 0x0123);
	}
}

