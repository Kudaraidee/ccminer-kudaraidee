/*
 * bc3mix512 on device (mix.cpp mix512 with folded = 0, the only PoW path).
 * Split at the phase boundaries:
 *   AB: schedule + phase A (32 rounds, 64-bit) + phase B (16 rounds, 128-bit)
 *       -> h1, h2 and the 16 words m256[48..63] that phase C reads
 *   C : 16 rounds, 256-bit: 2 exp256.pair + M256 mul + 4 EC mults per round
 *   D : 64 rounds: pow256.pair + inverse256, then the digest
 * Every line mirrors mix.cpp; names are the coin's.
 */
#pragma once
#include "bc3_exp_device.cuh"
#include "bc3_ec_device.cuh"

struct bc3_tables {
	const bc3u64* pow64;   /* [8][256]  F64  */
	const bc3u128* pow128; /* [16][256] F128 */
	const bc3u256* pow256; /* F256 powers: [32][256], or wider rows (bc3_powrows) */
	const uint8_t* curve[4]; /* H, J, K, L: 64 MB each */
};

template <int N>
__device__ __forceinline__ bc3u<N> bc3_const(const uint32_t* c)
{
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) r.w[i] = c[i];
	return r;
}

__device__ void bc3_mix_ab(bc3u256& h1, bc3u256& h2, bc3u256 m256_out[16],
                           const uint8_t* __restrict__ in64, const bc3_tables& t)
{
	bc3u64 m64[32];
	bc3u128 m128[16]; /* m128[32..47] */
	bc3u256 m256[23]; /* m256[41..63] */
	for (int i = 0; i < 8; i++)
		m64[i] = bc3_load_be<2>(in64 + 8 * i);
	bc3u64 c1 = m64[0] ^ bc3_const<2>(BC3_BASE[0]), c2 = m64[1] ^ bc3_const<2>(BC3_BASE[1]);
	bc3u64 c3 = m64[2] ^ bc3_const<2>(BC3_BASE[2]), c4 = m64[3] ^ bc3_const<2>(BC3_BASE[3]);
	bc3u64 c5 = m64[4] ^ bc3_const<2>(BC3_BASE[4]), c6 = m64[5] ^ bc3_const<2>(BC3_BASE[5]);
	bc3u64 c7 = m64[6] ^ bc3_const<2>(BC3_BASE[6]), c8 = m64[7] ^ bc3_const<2>(BC3_BASE[7]);
	for (int i = 8; i < 32; i++)
		m64[i] = bc3_gamma(m64[i - 8], m64[i - 4]) + bc3_gamma(m64[i - 7], m64[i - 3]) +
		         bc3_gamma(m64[i - 6], m64[i - 2]) + bc3_gamma(m64[i - 5], m64[i - 1]);
	/* m128[i] for i in 24..31 is U128(m64[i]); index m128 by i - 32 for 32..47 */
#define BC3_SCHED128(i) ((i) < 32 ? bc3_resize<4>(m64[i]) : m128[(i) - 32])
	for (int i = 32; i < 48; i++)
		m128[i - 32] = bc3_beta(BC3_SCHED128(i - 8)) + bc3_beta(BC3_SCHED128(i - 7)) +
		               bc3_beta(BC3_SCHED128(i - 4)) + bc3_beta(BC3_SCHED128(i - 3));
#undef BC3_SCHED128
	for (int i = 41; i < 48; i++)
		m256[i - 41] = bc3_resize<8>(m128[i - 32]);
	for (int i = 48; i < 64; i++)
		m256[i - 41] = bc3_beta(m256[i - 7 - 41]) + bc3_beta(m256[i - 6 - 41]) +
		               bc3_beta(m256[i - 2 - 41]) + bc3_beta(m256[i - 1 - 41]);
	for (int i = 0; i < 16; i++)
		m256_out[i] = m256[48 + i - 41];

	for (int i = 0; i < 32; i++) {
		bc3u64 p1, p2;
		bc3_exp64_pair(p1, p2, c1 ^ bc3_sig0(c2), c3 ^ bc3_sig1(c4));
		const bc3u64 d1 = p1 + bc3_ror<7>(c6) + bc3_rol<13>(c5);
		const bc3u64 d2 = p2 + bc3_ch(c5, c6, c7) + bc3_maj(c1, c4, c8);
		const bc3u64 d3 = c8 + bc3_Sig0(c7) + bc3_const<2>(BC3_ROUND64[i]);
		const bc3u64 d4 = c5 + bc3_Sig1(c1) + d1 + m64[i] + bc3_fixed_pow<BC3_F64>(t.pow64, d3);
		c8 = c7; /* + U64(folded), folded = 0 */
		c7 = c6 + d1 + bc3_sig0(bc3_mod_mul<BC3_M64>(d1, d3));
		c6 = c5 + bc3_sig1(d2 ^ d3);
		c5 = c4;
		c4 = c3 + d4;
		c3 = c2;
		c2 = c1 ^ d3;
		c1 = bc3_gamma(d1, d2) + (d3 ^ d4) + bc3_beta(bc3_mod_mul<BC3_M64>(d2, d4));
	}

	/* f.v = {lo, hi} in 64-bit limbs */
	bc3u128 f1 = {{c5.w[0], c5.w[1], c1.w[0], c1.w[1]}};
	bc3u128 f2 = {{c6.w[0], c6.w[1], c2.w[0], c2.w[1]}};
	bc3u128 f3 = {{c7.w[0], c7.w[1], c3.w[0], c3.w[1]}};
	bc3u128 f4 = {{c8.w[0], c8.w[1], c4.w[0], c4.w[1]}};
	for (int i = 0; i < 16; i++) {
		bc3u128 p1, p2;
		bc3_exp128_pair(p1, p2, f1 ^ bc3_beta(f3), f2 ^ bc3_beta(f4));
		const bc3u128 a = p1 + bc3_sig0(f2 ^ f4) + bc3_ch(f2, f3, f4);
		const bc3u128 b = p2 + bc3_sig1(f1 ^ f3) + bc3_maj(f1, f2, f3);
		const bc3u128 c = bc3_rol<29>(~f1) + bc3_ror<7>(~f2) + f3 + (~f4) + m128[i] +
		                  bc3_fixed_pow<BC3_F128>(t.pow128, f3);
		const bc3u128 d = bc3_rol<29>(~f3) + bc3_ror<7>(f4) + f2 + (~f1) + bc3_const<4>(BC3_ROUND128[i]);
		f4 = f3 + a;
		f3 = f2;
		f2 = f1 + c + bc3_beta(d);
		f1 = b + bc3_Sig0(a ^ c ^ d);
	}
	/* h1.v = {f3.v[0], f3.v[1], f1.v[0], f1.v[1]} */
	h1 = {{f3.w[0], f3.w[1], f3.w[2], f3.w[3], f1.w[0], f1.w[1], f1.w[2], f1.w[3]}};
	h2 = {{f4.w[0], f4.w[1], f4.w[2], f4.w[3], f2.w[0], f2.w[1], f2.w[2], f2.w[3]}};
}

/* One round of phase C in three steps (the scan runs each as its own kernel):
 *   exp1: (h1, h2)          -> j1, j2
 *   exp2: (h1, h2, j1, j2)  -> j3, j4
 *   ec  : (h1, h2, j1..j4)  -> next h1, h2 (four curve mults and the combine) */
__device__ __forceinline__ void bc3_mix_c_exp1(bc3u256& j1, bc3u256& j2, const bc3u256& h1, const bc3u256& h2)
{
	bc3u256 p1, p2;
	bc3_exp256_pair(p1, p2, bc3_rol<11>(h1) ^ h2, h1 ^ bc3_beta(~h2));
	j1 = p1 + bc3_beta(h1 ^ h2);
	j2 = p2 + bc3_gamma(h1, h2);
}
__device__ __forceinline__ void bc3_mix_c_exp2(bc3u256& j3, bc3u256& j4, const bc3u256& h2, const bc3u256& j1,
                                               const bc3u256& j2, const bc3u256& m, int i)
{
	bc3u256 p1, p2;
	bc3_exp256_pair(p1, p2, j1, j2);
	j3 = bc3_gamma(j1, j2) + p1 + bc3_const<8>(BC3_ROUND256[i]);
	j4 = bc3_gamma(j3, bc3_beta(h2)) + p2 + m + bc3_ch(j1, j2, j3);
}
template <int W = BC3_EC_WBITS>
__device__ __forceinline__ void bc3_mix_c_ec(bc3u256& h1, bc3u256& h2, const bc3u256& j1, const bc3u256& j2,
                                             const bc3u256& j3, const bc3u256& j4, const bc3_tables& t)
{
	bc3u256 pt[8];
	bc3_curve_mul4<W>(pt, t.curve, h1, h2, j1, j3);
	const bc3u256 &j5 = pt[0], &j6 = pt[1], &j7 = pt[2], &j8 = pt[3];
	const bc3u256 &j9 = pt[4], &j10 = pt[5], &j11 = pt[6], &j12 = pt[7];
	const bc3u256 n2 = h1 + bc3_gamma(j1, j3) + bc3_maj(j2, j3, j4) + j5 + j8 + bc3_ch(j1, j6, j7) + bc3_beta(j10) + j11;
	h1 = bc3_beta(j2) + bc3_gamma(j3, j4) + bc3_Sig1(bc3_mod_mul<BC3_M256>(j1, j2)) +
	     bc3_maj(j7, j6, j5) + bc3_Sig0(j9) + j12;
	h2 = n2;
}

__device__ void bc3_mix_c(bc3u256& h1, bc3u256& h2, const bc3u256 m256[16], const bc3_tables& t)
{
	for (int i = 0; i < 16; i++) {
		bc3u256 j1, j2, j3, j4;
		bc3_mix_c_exp1(j1, j2, h1, h2);
		bc3_mix_c_exp2(j3, j4, h2, j1, j2, m256[i], i);
		bc3_mix_c_ec(h1, h2, j1, j2, j3, j4, t);
	}
}

/* One round of phase D in two steps:
 *   pow: (u1, u2)          -> k1, k2 (the two table powers)
 *   inv: (u1, u2, k1, k2)  -> next u1, u2 (inverse256 and the combine) */
template <int RB = 8>
__device__ __forceinline__ void bc3_mix_d_pow(bc3u256& k1, bc3u256& k2, const bc3u256& u1, const bc3u256& u2,
                                              const bc3_tables& t)
{
	bc3u256 p1, p2;
	bc3_fixed_pow256_pair<36113u, RB>(p1, p2, t.pow256, u1, u2);
	k1 = p1 + bc3_beta(u2);
	k2 = p2 + bc3_gamma(u1, u2) + bc3_ch(k1, u2, u1);
}
__device__ __forceinline__ void bc3_mix_d_inv(bc3u256& u1, bc3u256& u2, const bc3u256& k1, const bc3u256& k2, int i)
{
	const bc3u256 k3 = bc3_inverse256(k2) + bc3_beta(u1) + bc3_beta(u2 ^ k1 ^ bc3_const<8>(BC3_FINAL[i]));
	u2 = (k1 ^ u1) + bc3_maj(k1, k2, u2) + bc3_beta(k3);
	u1 = bc3_beta(k1) + k2;
}
/* the mix output from the last u1, u2 */
__device__ __forceinline__ void bc3_mix_d_out(uint8_t* __restrict__ out32, const bc3u256& u1, const bc3u256& u2)
{
	bc3_store_be(out32, u1 + bc3_beta(u2));
}

template <int RB = 8>
__device__ void bc3_mix_d(uint8_t* __restrict__ out32, const bc3u256& h1, const bc3u256& h2, const bc3_tables& t)
{
	bc3u256 u1 = h1, u2 = h2; /* + U256(folded), folded = 0 */
	for (int i = 0; i < 64; i++) {
		bc3u256 k1, k2;
		bc3_mix_d_pow<RB>(k1, k2, u1, u2, t);
		bc3_mix_d_inv(u1, u2, k1, k2, i);
	}
	bc3_mix_d_out(out32, u1, u2);
}

__device__ __forceinline__ void bc3_mix512(uint8_t* __restrict__ out32, const uint8_t* __restrict__ in64, const bc3_tables& t)
{
	bc3u256 h1, h2, m256[16];
	bc3_mix_ab(h1, h2, m256, in64, t);
	bc3_mix_c(h1, h2, m256, t);
	bc3_mix_d(out32, h1, h2, t);
}
