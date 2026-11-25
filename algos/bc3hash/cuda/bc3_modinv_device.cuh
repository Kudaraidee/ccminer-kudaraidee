/*
 * Modular inverse by safegcd on bc3u256. Used by bc3hash and curvehash.
 */
#pragma once
#include "bc3_uint_device.cuh"

/* ---- inverses by safegcd (Bernstein-Yang divsteps) --------------------------
 * libsecp256k1's constant-time modinv32 (MIT, Peter Dettman). Both moduli have
 * the signed 30-bit limb shape {M0, M1, 0 x6, 65536}: F256 = {-36113, 0, ..} and
 * secp256k1's p = {-0x3d1, -4, ..}. 20 batches of 30 divsteps bound the worst
 * case; once g is 0 a batch moves d only by a multiple of the modulus, so the
 * loop stops there with the same result. */
struct bc3_s30 { int32_t v[9]; };

#define BC3_M30 0x3fffffff
#define BC3_MOD_F256 -36113, 0, 0x0f6bfc0fu    /* M0, M1, modulus^-1 mod 2^30 */
#define BC3_MOD_SECP -0x3d1, -4, 0x2ddacacfu

template <int M0, int M1>
__device__ __forceinline__ int32_t bc3_mod_limb(int i) { return i == 0 ? M0 : i == 1 ? M1 : i == 8 ? 65536 : 0; }

/* 30 divsteps on the bottom limbs: transition matrix [u v; q r] (scaled by 2^30) */
__device__ __forceinline__ int32_t bc3_divsteps_30(int32_t zeta, uint32_t f, uint32_t g,
                                                   int32_t& tu, int32_t& tv, int32_t& tq, int32_t& tr)
{
	uint32_t u = 1, v = 0, q = 0, r = 1;
#if defined(BC3_DIVSTEPS_MASKS) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700)
	/* masks below sm_70 (no full-rate 32-bit multiply there) */
	#pragma unroll 1
	for (int i = 0; i < 30; i++) {
		uint32_t c1 = (uint32_t)(zeta >> 31);
		const uint32_t c2 = 0u - (g & 1u);
		const uint32_t x = (f ^ c1) - c1, y = (u ^ c1) - c1, z = (v ^ c1) - c1;
		g += x & c2;
		q += y & c2;
		r += z & c2;
		c1 &= c2;
		zeta = (int32_t)(((uint32_t)zeta ^ c1) - 1u);
		f += g & c1;
		u += q & c1;
		v += r & c1;
		g >>= 1;
		u <<= 1;
		v <<= 1;
	}
#else
	/* sm_70+: the same conditional (negated) adds as multiply-adds by m in
	 * {-1, 0, 1} and by the swap's 0/1 */
	#pragma unroll 1
	for (int i = 0; i < 30; i++) {
		const uint32_t c1 = (uint32_t)(zeta >> 31); /* -1: zeta < 0 */
		const uint32_t c2 = 0u - (g & 1u);          /* -1: g odd */
		const uint32_t m = c2 & (c1 | 1u);          /* g odd: -1 if zeta < 0 else 1 */
		g += f * m;
		q += u * m;
		r += v * m;
		const uint32_t sw = c1 & c2;                /* -1: swap */
		zeta = (int32_t)(((uint32_t)zeta ^ sw) - 1u);
		const uint32_t s1 = sw & 1u;
		f += g * s1;
		u += q * s1;
		v += r * s1;
		g >>= 1;
		u <<= 1;
		v <<= 1;
	}
#endif
	tu = (int32_t)u; tv = (int32_t)v; tq = (int32_t)q; tr = (int32_t)r;
	return zeta;
}

/* d, e <- (t / 2^30) [d, e] mod M; d, e stay in (-2M, M) */
template <int M0, int M1, uint32_t INV30>
__device__ __forceinline__ void bc3_update_de_30(bc3_s30& d, bc3_s30& e, int32_t u, int32_t v, int32_t q, int32_t r)
{
	const int32_t sd = d.v[8] >> 31, se = e.v[8] >> 31;
	int32_t md = (u & sd) + (v & se), me = (q & sd) + (r & se);
	int64_t cd = (int64_t)u * d.v[0] + (int64_t)v * e.v[0];
	int64_t ce = (int64_t)q * d.v[0] + (int64_t)r * e.v[0];
	md -= (INV30 * (uint32_t)cd + md) & BC3_M30;
	me -= (INV30 * (uint32_t)ce + me) & BC3_M30;
	cd += (int64_t)bc3_mod_limb<M0, M1>(0) * md;
	ce += (int64_t)bc3_mod_limb<M0, M1>(0) * me;
	cd >>= 30;
	ce >>= 30;
	#pragma unroll
	for (int i = 1; i < 9; i++) {
		cd += (int64_t)u * d.v[i] + (int64_t)v * e.v[i] + (int64_t)bc3_mod_limb<M0, M1>(i) * md;
		ce += (int64_t)q * d.v[i] + (int64_t)r * e.v[i] + (int64_t)bc3_mod_limb<M0, M1>(i) * me;
		d.v[i - 1] = (int32_t)cd & BC3_M30; cd >>= 30;
		e.v[i - 1] = (int32_t)ce & BC3_M30; ce >>= 30;
	}
	d.v[8] = (int32_t)cd;
	e.v[8] = (int32_t)ce;
}

/* f, g <- (t / 2^30) [f, g] */
__device__ __forceinline__ void bc3_update_fg_30(bc3_s30& f, bc3_s30& g, int32_t u, int32_t v, int32_t q, int32_t r)
{
	int64_t cf = (int64_t)u * f.v[0] + (int64_t)v * g.v[0];
	int64_t cg = (int64_t)q * f.v[0] + (int64_t)r * g.v[0];
	cf >>= 30;
	cg >>= 30;
	#pragma unroll
	for (int i = 1; i < 9; i++) {
		cf += (int64_t)u * f.v[i] + (int64_t)v * g.v[i];
		cg += (int64_t)q * f.v[i] + (int64_t)r * g.v[i];
		f.v[i - 1] = (int32_t)cf & BC3_M30; cf >>= 30;
		g.v[i - 1] = (int32_t)cg & BC3_M30; cg >>= 30;
	}
	f.v[8] = (int32_t)cf;
	g.v[8] = (int32_t)cg;
}

/* r from (-2M, M), negated if sign < 0, to [0, M) with limbs in [0, 2^30) */
template <int M0, int M1>
__device__ __forceinline__ void bc3_normalize_30(bc3_s30& r, int32_t sign)
{
	int32_t cond = r.v[8] >> 31;
	#pragma unroll
	for (int i = 0; i < 9; i++) r.v[i] += bc3_mod_limb<M0, M1>(i) & cond;
	cond = sign >> 31;
	#pragma unroll
	for (int i = 0; i < 9; i++) r.v[i] = (r.v[i] ^ cond) - cond;
	#pragma unroll
	for (int i = 0; i < 8; i++) { r.v[i + 1] += r.v[i] >> 30; r.v[i] &= BC3_M30; }
	cond = r.v[8] >> 31;
	#pragma unroll
	for (int i = 0; i < 9; i++) r.v[i] += bc3_mod_limb<M0, M1>(i) & cond;
	#pragma unroll
	for (int i = 0; i < 8; i++) { r.v[i + 1] += r.v[i] >> 30; r.v[i] &= BC3_M30; }
}

/* x^-1 mod M for x in [0, M); 0 gives 0 */
template <int M0, int M1, uint32_t INV30>
__device__ __forceinline__ bc3u256 bc3_modinv(const bc3u256& x)
{
	bc3_s30 d, e, f, g;
	#pragma unroll
	for (int i = 0; i < 9; i++) {
		const int o = 30 * i, w = o >> 5, s = o & 31;
		const uint64_t two = (uint64_t)x.w[w] | (w + 1 < 8 ? (uint64_t)x.w[w + 1] << 32 : 0);
		g.v[i] = (int32_t)((two >> s) & BC3_M30);
		d.v[i] = 0;
		e.v[i] = 0;
		f.v[i] = bc3_mod_limb<M0, M1>(i);
	}
	e.v[0] = 1;
	int32_t zeta = -1;
	#pragma unroll 1
	for (int i = 0; i < 20; i++) {
		int32_t u, v, q, r;
		zeta = bc3_divsteps_30(zeta, (uint32_t)f.v[0], (uint32_t)g.v[0], u, v, q, r);
		bc3_update_de_30<M0, M1, INV30>(d, e, u, v, q, r);
		bc3_update_fg_30(f, g, u, v, q, r);
#ifndef BC3_MODINV_NO_EARLY_EXIT /* define to always run all 20 batches */
		if (i >= 15) {
			int32_t any = 0;
			#pragma unroll
			for (int j = 0; j < 9; j++) any |= g.v[j];
			if (!any) break;
		}
#endif
	}
	bc3_normalize_30<M0, M1>(d, f.v[8]);
	bc3u256 res;
	#pragma unroll
	for (int i = 0; i < 8; i++) res.w[i] = 0;
	#pragma unroll
	for (int i = 0; i < 9; i++) {
		const int o = 30 * i, w = o >> 5, s = o & 31;
		const uint64_t sh = (uint64_t)(uint32_t)d.v[i] << s;
		res.w[w] |= (uint32_t)sh;
		if (w + 1 < 8) res.w[w + 1] |= (uint32_t)(sh >> 32);
	}
	return res;
}

