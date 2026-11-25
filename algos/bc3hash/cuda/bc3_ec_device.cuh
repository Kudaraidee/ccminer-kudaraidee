/*
 * bc3hash fixed-base secp256k1 multiplication: k*P for the four mix512 bases
 * H, J, K, L (mix.cpp CurveTable).
 *
 * Table per base: W-bit windows (W = 16 or 20), entry (j, i) = (i * 2^(Wj)) * P
 * as big-endian X||Y (64 bytes), entry 0 unused. Built on the device by
 * bc3_ec_table_kernel from the window bases 2^(Wj)*P (derived on the host with
 * libsecp). The table builder and the single-point bc3_curve_mul use
 * curvehash's libsecp256k1 transcription; the scan path (bc3_curve_mul4) uses
 * the 8 x 32-bit field below.
 *
 * Semantics (mix.cpp CurveTable::operator()): k >= n is reduced, k == 0 mod n
 * gives (INVALID_POINT_x, INVALID_POINT_y). Raw 16-bit windows of any k < 2^256
 * already sum to (k mod n)*P, and k mod n == 0 only for k = 0 and k = n, where
 * the sum is infinity, so "infinity => sentinel" covers both.
 */
#pragma once
#include "bc3_uint_device.cuh"
#include "bc3_constants.cuh"
#include "bc3_exp_device.cuh"
#include "../../curvehash/cuda/secp256k1_group_device.cuh"
#include "bc3_secp_device.cuh"

#define BC3_EC_WBITS 16
#define BC3_EC_WINDOWS (256 / BC3_EC_WBITS)
#define BC3_EC_ENTRIES (1u << BC3_EC_WBITS)
#define BC3_EC_ENTRY 64
#define BC3_EC_TABLE_BYTES ((size_t)BC3_EC_WINDOWS * BC3_EC_ENTRIES * BC3_EC_ENTRY)

/* W-bit window layout: ceil(256/W) windows, window j covers bits [jW, jW+W) and
 * starts at entry j * 2^W; the last one may be shorter. */
template <int W>
struct bc3_ecw {
	static const int windows = (256 + W - 1) / W;
	static const int last_bits = 256 - W * (windows - 1);
	__host__ __device__ static size_t entries() { return ((size_t)(windows - 1) << W) + ((size_t)1 << last_bits); }
	__host__ __device__ static size_t bytes() { return entries() * BC3_EC_ENTRY; }
};

/* Normalized field element -> bc3u256 (the integer value, limbs LS first). */
__device__ __forceinline__ bc3u256 bc3_fe_to_u256(const secp256k1_fe* a)
{
	const uint32_t* n = a->n;
	bc3u256 r;
	r.w[0] = n[0] | n[1] << 26;
	r.w[1] = n[1] >> 6 | n[2] << 20;
	r.w[2] = n[2] >> 12 | n[3] << 14;
	r.w[3] = n[3] >> 18 | n[4] << 8;
	r.w[4] = n[4] >> 24 | n[5] << 2 | n[6] << 28;
	r.w[5] = n[6] >> 4 | n[7] << 22;
	r.w[6] = n[7] >> 10 | n[8] << 16;
	r.w[7] = n[8] >> 16 | n[9] << 10;
	return r;
}

/* Jacobian sum of the 16 window entries, before the affine conversion. */
__device__ __forceinline__ void bc3_ec_accumulate(secp256k1_gej* r, const uint8_t* __restrict__ table,
                                                  const bc3u256& k)
{
	secp256k1_ge add;
	add.infinity = 0;
	secp256k1_gej_set_infinity(r);
#pragma unroll 1
	for (int j = 0; j < BC3_EC_WINDOWS; j++) {
		const uint32_t bits = (k.w[j >> 1] >> (16 * (j & 1))) & 0xffff;
		if (bits == 0) continue; /* entry 0 is infinity: additive no-op */
		const uint8_t* p = table + ((size_t)j * BC3_EC_ENTRIES + bits) * BC3_EC_ENTRY;
		secp256k1_fe_set_b32(&add.x, p);
		secp256k1_fe_set_b32(&add.y, p + 32);
		secp256k1_gej_add_ge_var(r, r, &add, NULL);
	}
}

/* CurveTable::operator(): affine (x, y) of k*P, or the sentinel. */
__device__ void bc3_curve_mul(bc3u256& x, bc3u256& y, const uint8_t* __restrict__ table, const bc3u256& k)
{
	secp256k1_gej rj;
	secp256k1_ge ra;
	bc3_ec_accumulate(&rj, table, k);
	secp256k1_ge_set_gej_var(&ra, &rj);
	if (ra.infinity) {
#pragma unroll
		for (int i = 0; i < 8; i++) {
			x.w[i] = BC3_INVALID_X[i];
			y.w[i] = BC3_INVALID_Y[i];
		}
		return;
	}
	secp256k1_fe_normalize_var(&ra.x);
	secp256k1_fe_normalize_var(&ra.y);
	x = bc3_fe_to_u256(&ra.x);
	y = bc3_fe_to_u256(&ra.y);
}

/* Jacobian sum of the W-bit window entries of k, before the affine step */
template <int W>
__device__ __forceinline__ void bc3_jac_accumulate(bc3_jac& r, const uint8_t* __restrict__ table, const bc3u256& k)
{
	r.inf = true;
	r.x = r.y = r.z = bc3_small<8>(0);
	bc3u256 kk = k;
#pragma unroll 1
	for (int j = 0; j < bc3_ecw<W>::windows; j++) {
#ifdef BC3_RT_DIGITS
		const int pos = j * W, wd = pos >> 5;
		uint64_t v = k.w[wd];
		if (wd + 1 < 8) v |= (uint64_t)k.w[wd + 1] << 32;
		const uint32_t bits = (uint32_t)(v >> (pos & 31)) & ((1u << W) - 1);
#else
		/* digits from a running shifted copy (see bc3_fixed_pow256_pair) */
		const uint32_t bits = kk.w[0] & ((1u << W) - 1);
		kk = bc3_shr<W>(kk);
#endif
		if (bits == 0) continue; /* entry 0 is infinity: additive no-op */
		bc3u256 x2, y2;
		bc3_ec_load(x2, y2, table + (((size_t)j << W) + bits) * BC3_EC_ENTRY);
		bc3_jac_add_affine(r, x2, y2);
	}
}

/* canonical affine (x, y) of r given 1/z; infinity gives the sentinel */
__device__ __forceinline__ void bc3_jac_affine(bc3u256& x, bc3u256& y, const bc3_jac& r, const bc3u256& zi)
{
	if (r.inf) {
#pragma unroll
		for (int i = 0; i < 8; i++) {
			x.w[i] = BC3_INVALID_X[i];
			y.w[i] = BC3_INVALID_Y[i];
		}
		return;
	}
	const bc3u256 z2 = bc3_fp_sqr(zi), z3 = bc3_fp_mul(z2, zi);
	x = bc3_fp_canon(bc3_fp_mul(r.x, z2));
	y = bc3_fp_canon(bc3_fp_mul(r.y, z3));
}

/* Four bc3_curve_mul at once, one per base, sharing one inversion (Montgomery's
 * trick). An infinite sum takes z = 1 in the product so it does not zero the
 * others. out = x0 y0 .. x3 y3. */
template <int W = BC3_EC_WBITS>
__device__ void bc3_curve_mul4(bc3u256 (&out)[8], const uint8_t* const (&tab)[4], const bc3u256& k0,
                               const bc3u256& k1, const bc3u256& k2, const bc3u256& k3)
{
	bc3_jac r[4];
	bc3_jac_accumulate<W>(r[0], tab[0], k0);
	bc3_jac_accumulate<W>(r[1], tab[1], k1);
	bc3_jac_accumulate<W>(r[2], tab[2], k2);
	bc3_jac_accumulate<W>(r[3], tab[3], k3);
#pragma unroll
	for (int i = 0; i < 4; i++)
		if (r[i].inf) r[i].z = bc3_small<8>(1);
	const bc3u256 p01 = bc3_fp_mul(r[0].z, r[1].z);
	const bc3u256 p012 = bc3_fp_mul(p01, r[2].z);
	bc3u256 inv = bc3_modinv<BC3_MOD_SECP>(bc3_fp_canon(bc3_fp_mul(p012, r[3].z))); /* 1/(z0 z1 z2 z3) */
	bc3_jac_affine(out[6], out[7], r[3], bc3_fp_mul(inv, p012));
	inv = bc3_fp_mul(inv, r[3].z); /* 1/(z0 z1 z2) */
	bc3_jac_affine(out[4], out[5], r[2], bc3_fp_mul(inv, p01));
	inv = bc3_fp_mul(inv, r[2].z); /* 1/(z0 z1) */
	bc3_jac_affine(out[2], out[3], r[1], bc3_fp_mul(inv, r[0].z));
	bc3_jac_affine(out[0], out[1], r[0], bc3_fp_mul(inv, r[1].z));
}

/* Table builder: one thread per (window j, entry i >= 1) of one base; entry =
 * i * B_j by MSB-first double-and-add, then affine. wbase: the window bases
 * B_j = 2^(Wj)*P as big-endian X||Y. Builds entries [first, end). */
template <int W = BC3_EC_WBITS>
__global__ void bc3_ec_table_kernel(uint8_t* __restrict__ table, const uint8_t* __restrict__ wbase,
                                    uint32_t first = 0, uint32_t end = 0xffffffffu)
{
	const uint32_t t = first + blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= end || t >= bc3_ecw<W>::entries()) return;
	const uint32_t j = t >> W, i = t & ((1u << W) - 1);
	uint8_t* out = table + (size_t)t * BC3_EC_ENTRY;
	if (i == 0) { /* never read; zeroed */
		for (int b = 0; b < BC3_EC_ENTRY; b++) out[b] = 0;
		return;
	}
	secp256k1_ge base;
	secp256k1_fe_set_b32(&base.x, wbase + 64 * j);
	secp256k1_fe_set_b32(&base.y, wbase + 64 * j + 32);
	base.infinity = 0;
	secp256k1_gej r;
	secp256k1_gej_set_infinity(&r);
	for (int b = W - 1; b >= 0; b--) {
		secp256k1_gej_double_var(&r, &r, NULL);
		if ((i >> b) & 1) secp256k1_gej_add_ge_var(&r, &r, &base, NULL);
	}
	secp256k1_ge a;
	secp256k1_ge_set_gej_var(&a, &r);
	secp256k1_fe_normalize_var(&a.x);
	secp256k1_fe_normalize_var(&a.y);
	secp256k1_fe_get_b32(out, &a.x);
	secp256k1_fe_get_b32(out + 32, &a.y);
}
