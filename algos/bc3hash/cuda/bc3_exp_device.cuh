/*
 * bc3hash fixed exponents, fixed-base powers and inverse256 on device.
 *
 * FixedExponent (uint.hpp): sliding-window chain whose schedule is a constant of
 * the algorithm (generated into bc3_constants.cuh). The schedule is read from
 * __constant__ memory (warp-uniform); the odd-power table is indexed at run time.
 * .pair evaluates two independent chains, like the coin. Products inside a chain
 * are lazy; only its result is made canonical.
 *
 * FixedPower (uint.hpp): g^e = prod_i table[i][byte_i(e)], tables built on the
 * host by the oracle and uploaded (global memory, read through __ldg).
 *
 * inverse256 by safegcd; gives 0 for a == 0 mod p, as the coin requires.
 */
#pragma once
#include "bc3_uint_device.cuh"
#include "bc3_constants.cuh"
#include "bc3_modinv_device.cuh"

#define BC3_SCHED(NAME) BC3_##NAME##_WINDOW, BC3_##NAME##_COUNT, BC3_##NAME##_TAIL, BC3_##NAME##_SQ, BC3_##NAME##_IX

template <int N, uint32_t C, int W>
__device__ __forceinline__ void bc3_odd_table(bc3u<N> (&odd)[1 << (W - 1)], const bc3u<N>& a)
{
	odd[0] = bc3_mod_normalize<N, C>(a);
	const bc3u<N> a2 = bc3_mod_sqr_lazy<N, C>(odd[0]);
	for (int i = 1; i < (1 << (W - 1)); i++)
		odd[i] = bc3_mod_mul_lazy<N, C>(odd[i - 1], a2);
}

template <int N, uint32_t C, int W>
__device__ void bc3_fixed_exp(bc3u<N>& r, const bc3u<N>& a, int count, int tail,
                              const uint8_t* sq, const uint8_t* ix)
{
	bc3u<N> odd[1 << (W - 1)];
	bc3_odd_table<N, C, W>(odd, a);
	r = odd[ix[0]];
	for (int i = 1; i < count; i++) {
		for (int j = 0; j < sq[i]; j++)
			r = bc3_mod_sqr_lazy<N, C>(r);
		r = bc3_mod_mul_lazy<N, C>(r, odd[ix[i]]);
	}
	for (int i = 0; i < tail; i++)
		r = bc3_mod_sqr_lazy<N, C>(r);
	r = bc3_mod_normalize<N, C>(r);
}

template <int N, uint32_t C, int W>
__device__ void bc3_fixed_exp_pair(bc3u<N>& r, bc3u<N>& s, const bc3u<N>& a, const bc3u<N>& b,
                                   int count, int tail, const uint8_t* sq, const uint8_t* ix)
{
	bc3u<N> odda[1 << (W - 1)], oddb[1 << (W - 1)];
	bc3_odd_table<N, C, W>(odda, a);
	bc3_odd_table<N, C, W>(oddb, b);
	r = odda[ix[0]];
	s = oddb[ix[0]];
	for (int i = 1; i < count; i++) {
		for (int j = 0; j < sq[i]; j++) {
			r = bc3_mod_sqr_lazy<N, C>(r);
			s = bc3_mod_sqr_lazy<N, C>(s);
		}
		r = bc3_mod_mul_lazy<N, C>(r, odda[ix[i]]);
		s = bc3_mod_mul_lazy<N, C>(s, oddb[ix[i]]);
	}
	for (int i = 0; i < tail; i++) {
		r = bc3_mod_sqr_lazy<N, C>(r);
		s = bc3_mod_sqr_lazy<N, C>(s);
	}
	r = bc3_mod_normalize<N, C>(r);
	s = bc3_mod_normalize<N, C>(s);
}

/* The three exponents mix512 uses (with the coin's windows) and the inverse. */
__device__ __forceinline__ void bc3_exp64_pair(bc3u64& r, bc3u64& s, const bc3u64& a, const bc3u64& b)
{
	bc3_fixed_exp_pair<BC3_M64, BC3_EXP64_WINDOW>(r, s, a, b, BC3_EXP64_COUNT, BC3_EXP64_TAIL, BC3_EXP64_SQ, BC3_EXP64_IX);
}
__device__ __forceinline__ void bc3_exp128_pair(bc3u128& r, bc3u128& s, const bc3u128& a, const bc3u128& b)
{
	bc3_fixed_exp_pair<BC3_M128, BC3_EXP128_WINDOW>(r, s, a, b, BC3_EXP128_COUNT, BC3_EXP128_TAIL, BC3_EXP128_SQ, BC3_EXP128_IX);
}
__device__ __forceinline__ void bc3_exp256_pair(bc3u256& r, bc3u256& s, const bc3u256& a, const bc3u256& b)
{
	bc3_fixed_exp_pair<BC3_M256, BC3_EXP256_WINDOW>(r, s, a, b, BC3_EXP256_COUNT, BC3_EXP256_TAIL, BC3_EXP256_SQ, BC3_EXP256_IX);
}

/* a^-1 mod F256 for any 256-bit a; 0 (mod F256) gives 0, as the coin requires */
__device__ __forceinline__ bc3u256 bc3_inverse256(const bc3u256& a)
{
	return bc3_modinv<BC3_MOD_F256>(bc3_mod_normalize<BC3_F256>(a));
}

/* ---- FixedPower: table[i][256] of bc3u<N>, i over the 4N exponent bytes ---- */

template <int N>
__device__ __forceinline__ bc3u<N> bc3_ldg(const bc3u<N>* p)
{
	bc3u<N> r;
	const uint4* q = (const uint4*)p;
#pragma unroll
	for (int k = 0; k < N / 4; k++) {
		const uint4 v = __ldg(q + k);
		r.w[4 * k] = v.x;
		r.w[4 * k + 1] = v.y;
		r.w[4 * k + 2] = v.z;
		r.w[4 * k + 3] = v.w;
	}
	if (N == 2) {
		const uint2 v = __ldg((const uint2*)p);
		r.w[0] = v.x;
		r.w[1] = v.y;
	}
	return r;
}

template <int N>
__device__ __forceinline__ uint32_t bc3_byte(const bc3u<N>& e, int i)
{
	return (e.w[i >> 2] >> (8 * (i & 3))) & 255;
}

template <int N, uint32_t C>
__device__ void bc3_fixed_pow_pair(bc3u<N>& r, bc3u<N>& s, const bc3u<N>* __restrict__ table,
                                   const bc3u<N>& a, const bc3u<N>& b)
{
	r = bc3_ldg(table + bc3_byte(a, 0));
	s = bc3_ldg(table + bc3_byte(b, 0));
	for (int i = 1; i < 4 * N; i++) {
		r = bc3_mod_mul_lazy<N, C>(r, bc3_ldg(table + 256 * i + bc3_byte(a, i)));
		s = bc3_mod_mul_lazy<N, C>(s, bc3_ldg(table + 256 * i + bc3_byte(b, i)));
	}
	r = bc3_mod_normalize<N, C>(r);
	s = bc3_mod_normalize<N, C>(s);
}

/* ---- 256-bit FixedPower with RB-bit rows (RB = 8 is the coin's table) -------
 * Row i holds g^(d * 2^(RB*i)) for every RB-bit digit d; ceil(256/RB) rows, the
 * last one may be shorter. Wider rows: fewer products per power, bigger table. */
template <int RB>
struct bc3_powrows {
	static const int rows = (256 + RB - 1) / RB;
	__host__ __device__ static size_t entries() { return (size_t)rows << RB; }
};

/* digit i (bits [RB*i, RB*i + RB)) of a 256-bit exponent */
template <int RB>
__device__ __forceinline__ uint32_t bc3_digit(const bc3u256& e, int i)
{
	const int pos = RB * i, wd = pos >> 5;
	uint64_t v = e.w[wd];
	if (wd + 1 < 8) v |= (uint64_t)e.w[wd + 1] << 32;
	return (uint32_t)(v >> (pos & 31)) & ((1u << RB) - 1);
}

template <uint32_t C, int RB>
__device__ void bc3_fixed_pow256_pair(bc3u256& r, bc3u256& s, const bc3u256* __restrict__ table,
                                      const bc3u256& a, const bc3u256& b)
{
#ifdef BC3_RT_DIGITS
	r = bc3_ldg(table + bc3_digit<RB>(a, 0));
	s = bc3_ldg(table + bc3_digit<RB>(b, 0));
	for (int i = 1; i < bc3_powrows<RB>::rows; i++) {
		r = bc3_mod_mul_lazy<8, C>(r, bc3_ldg(table + ((size_t)i << RB) + bc3_digit<RB>(a, i)));
		s = bc3_mod_mul_lazy<8, C>(s, bc3_ldg(table + ((size_t)i << RB) + bc3_digit<RB>(b, i)));
	}
#else
	/* digits from running shifted copies (keeps the exponents in registers) */
	const uint32_t mask = (1u << RB) - 1;
	bc3u256 ea = bc3_shr<RB>(a), eb = bc3_shr<RB>(b);
	r = bc3_ldg(table + (a.w[0] & mask));
	s = bc3_ldg(table + (b.w[0] & mask));
	for (int i = 1; i < bc3_powrows<RB>::rows; i++) {
		const bc3u256* row = table + ((size_t)i << RB);
		const uint32_t da = ea.w[0] & mask, db = eb.w[0] & mask;
		ea = bc3_shr<RB>(ea);
		eb = bc3_shr<RB>(eb);
		r = bc3_mod_mul_lazy<8, C>(r, bc3_ldg(row + da));
		s = bc3_mod_mul_lazy<8, C>(s, bc3_ldg(row + db));
	}
#endif
	r = bc3_mod_normalize<8, C>(r);
	s = bc3_mod_normalize<8, C>(s);
}

/* RB-bit-row table from the coin's 8-bit one: entry (i, d) = g^(d * 2^(RB*i)) is
 * the product of the 8-bit entries for the bytes of d * 2^(RB*i). One thread
 * per entry; builds entries [first, end). */
template <uint32_t C, int RB>
__global__ void bc3_pow_table_kernel(bc3u256* __restrict__ out, const bc3u256* __restrict__ t8,
                                     uint32_t first, uint32_t end)
{
	const uint32_t t = first + blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= end || t >= bc3_powrows<RB>::entries()) return;
	const uint32_t i = t >> RB, d = t & ((1u << RB) - 1);
	bc3u256 r = bc3_small<8>(1);
	for (int k = (RB * i) >> 3; k <= (int)((RB * i + RB - 1) >> 3) && k < 32; k++) {
		/* byte k of d << (RB*i) */
		const int sh = RB * (int)i - 8 * k;
		const uint32_t byte = (sh >= 0 ? d << sh : d >> -sh) & 255;
		r = bc3_mod_mul<8, C>(r, t8[256 * k + byte]);
	}
	out[t] = r;
}

template <int N, uint32_t C>
__device__ bc3u<N> bc3_fixed_pow(const bc3u<N>* __restrict__ table, const bc3u<N>& a)
{
	bc3u<N> r = bc3_ldg(table + bc3_byte(a, 0));
	for (int i = 1; i < 4 * N; i++)
		r = bc3_mod_mul_lazy<N, C>(r, bc3_ldg(table + 256 * i + bc3_byte(a, i)));
	return bc3_mod_normalize<N, C>(r);
}
