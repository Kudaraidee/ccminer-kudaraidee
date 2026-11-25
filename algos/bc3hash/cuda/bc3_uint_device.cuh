/*
 * bc3hash device big integers: bc3u<N> = N 32-bit limbs, least significant first
 * (so its memory image equals the coin's UInt<N/2> on a little-endian host).
 * Transcribes ref/uint.hpp: wrapping +, -, bitwise ops, ror/rol/shr, the
 * per-width SHA-2-style mixers, and Mod<L,C> (pseudo-Mersenne 2^(32N) - C).
 *
 * Note: + WRAPS mod 2^(32N), as in the coin's UInt; only bc3_mod_* are modular.
 */
#pragma once
#include <stdint.h>

template <int N>
struct bc3u {
	uint32_t w[N];
};
typedef bc3u<2> bc3u64;
typedef bc3u<4> bc3u128;
typedef bc3u<8> bc3u256;

template <int N>
__device__ __forceinline__ bc3u<N> bc3_small(uint32_t x)
{
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) r.w[i] = 0;
	r.w[0] = x;
	return r;
}

/* Zero-extend / truncate between widths (UInt<L>(const UInt<M>&)). */
template <int N, int M>
__device__ __forceinline__ bc3u<N> bc3_resize(const bc3u<M>& a)
{
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) r.w[i] = i < M ? a.w[i] : 0;
	return r;
}

template <int N>
__device__ __forceinline__ bc3u<N> operator+(const bc3u<N>& a, const bc3u<N>& b)
{
	bc3u<N> r;
	uint64_t c = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		c = (uint64_t)a.w[i] + b.w[i] + (c >> 32);
		r.w[i] = (uint32_t)c;
	}
	return r;
}

template <int N>
__device__ __forceinline__ bc3u<N> operator-(const bc3u<N>& a, const bc3u<N>& b)
{
	bc3u<N> r;
	uint32_t borrow = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		const uint64_t d = (uint64_t)a.w[i] - b.w[i] - borrow;
		r.w[i] = (uint32_t)d;
		borrow = (uint32_t)(d >> 63);
	}
	return r;
}

#define BC3_BITOP(op)                                                               \
	template <int N>                                                                \
	__device__ __forceinline__ bc3u<N> operator op(const bc3u<N>& a, const bc3u<N>& b) \
	{                                                                               \
		bc3u<N> r;                                                                  \
		_Pragma("unroll") for (int i = 0; i < N; i++) r.w[i] = a.w[i] op b.w[i];    \
		return r;                                                                   \
	}
BC3_BITOP(^)
BC3_BITOP(&)
BC3_BITOP(|)
#undef BC3_BITOP

template <int N>
__device__ __forceinline__ bc3u<N> operator~(const bc3u<N>& a)
{
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) r.w[i] = ~a.w[i];
	return r;
}

template <int N>
__device__ __forceinline__ bool bc3_is_zero(const bc3u<N>& a)
{
	uint32_t r = 0;
#pragma unroll
	for (int i = 0; i < N; i++) r |= a.w[i];
	return r == 0;
}

/* Rotate right by R over the full 32N-bit value (uint.hpp ror<R>). */
template <int R, int N>
__device__ __forceinline__ bc3u<N> bc3_ror(const bc3u<N>& a)
{
	constexpr int q = (R / 32) % N, s = R % 32;
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) {
		const uint32_t lo = a.w[(i + q) % N], hi = a.w[(i + q + 1) % N];
		r.w[i] = s ? __funnelshift_r(lo, hi, s) : lo;
	}
	return r;
}

template <int R, int N>
__device__ __forceinline__ bc3u<N> bc3_rol(const bc3u<N>& a)
{
	return bc3_ror<32 * N - R % (32 * N)>(a);
}

/* Logical shift right by R (uint.hpp shr<R>). */
template <int R, int N>
__device__ __forceinline__ bc3u<N> bc3_shr(const bc3u<N>& a)
{
	constexpr int q = R / 32, s = R % 32;
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) {
		const uint32_t lo = i + q < N ? a.w[i + q] : 0, hi = i + q + 1 < N ? a.w[i + q + 1] : 0;
		r.w[i] = s ? __funnelshift_r(lo, hi, s) : lo;
	}
	return r;
}

template <int N>
__device__ __forceinline__ bc3u<N> bc3_ch(const bc3u<N>& x, const bc3u<N>& y, const bc3u<N>& z)
{
	return z ^ (x & (y ^ z));
}

template <int N>
__device__ __forceinline__ bc3u<N> bc3_maj(const bc3u<N>& x, const bc3u<N>& y, const bc3u<N>& z)
{
	return (x & y) | (z & (x | y));
}

/* Per-width rotation constants (uint.hpp, L = N/2 64-bit limbs). */
template <int N> struct bc3_rot;
template <> struct bc3_rot<2> {
	enum { S0a = 28, S0b = 34, S0c = 39, S1a = 14, S1b = 18, S1c = 41,
	       s0a = 1, s0b = 8, s0s = 7, s1a = 19, s1b = 61, s1s = 6,
	       beta = 51, gr = 30, gl = 38, gs = 12 };
};
template <> struct bc3_rot<4> {
	enum { S0a = 57, S0b = 63, S0c = 92, S1a = 35, S1b = 44, S1c = 51,
	       s0a = 42, s0b = 83, s0s = 20, s1a = 24, s1b = 119, s1s = 18,
	       beta = 54, gr = 48, gl = 99, gs = 22 };
};
template <> struct bc3_rot<8> {
	enum { S0a = 113, S0b = 146, S0c = 171, S1a = 88, S1b = 97, S1c = 201,
	       s0a = 131, s0b = 57, s0s = 56, s1a = 53, s1b = 247, s1s = 26,
	       beta = 81, gr = 58, gl = 183, gs = 32 };
};

template <int N>
__device__ __forceinline__ bc3u<N> bc3_Sig0(const bc3u<N>& x)
{
	typedef bc3_rot<N> K;
	return bc3_ror<K::S0a>(x) ^ bc3_ror<K::S0b>(x) ^ bc3_ror<K::S0c>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_Sig1(const bc3u<N>& x)
{
	typedef bc3_rot<N> K;
	return bc3_ror<K::S1a>(x) ^ bc3_ror<K::S1b>(x) ^ bc3_ror<K::S1c>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_sig0(const bc3u<N>& x)
{
	typedef bc3_rot<N> K;
	return bc3_ror<K::s0a>(x) ^ bc3_ror<K::s0b>(x) ^ bc3_shr<K::s0s>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_sig1(const bc3u<N>& x)
{
	typedef bc3_rot<N> K;
	return bc3_ror<K::s1a>(x) ^ bc3_ror<K::s1b>(x) ^ bc3_shr<K::s1s>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_beta(const bc3u<N>& x)
{
	return x ^ bc3_Sig0(x) ^ bc3_Sig1(x) ^ bc3_sig0(x) ^ bc3_sig1(x) ^ bc3_rol<bc3_rot<N>::beta>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_gamma(const bc3u<N>& x, const bc3u<N>& y)
{
	typedef bc3_rot<N> K;
	return x ^ bc3_beta(y) ^ bc3_ror<K::gr>(y) ^ bc3_rol<K::gl>(bc3_sig1(x)) ^ bc3_shr<K::gs>(y);
}

/* UInt::load / store: BIG-endian bytes, byte 0 is the top of the value. */
template <int N>
__device__ __forceinline__ bc3u<N> bc3_load_be(const uint8_t* p)
{
	bc3u<N> r;
#pragma unroll
	for (int k = 0; k < N; k++)
		r.w[N - 1 - k] = ((uint32_t)p[4 * k] << 24) | ((uint32_t)p[4 * k + 1] << 16) |
		                 ((uint32_t)p[4 * k + 2] << 8) | p[4 * k + 3];
	return r;
}
template <int N>
__device__ __forceinline__ void bc3_store_be(uint8_t* p, const bc3u<N>& a)
{
#pragma unroll
	for (int k = 0; k < N; k++) {
		const uint32_t v = a.w[N - 1 - k];
		p[4 * k] = (uint8_t)(v >> 24);
		p[4 * k + 1] = (uint8_t)(v >> 16);
		p[4 * k + 2] = (uint8_t)(v >> 8);
		p[4 * k + 3] = (uint8_t)v;
	}
}

/* ---- Mod<L, C>: modulus 2^(32N) - C, C < 2^16 ------------------------------ */

/* a >= m  =>  a - m. Since m = 2^(32N) - C, a >= m iff a + C carries out. */
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_normalize(const bc3u<N>& a)
{
	bc3u<N> s;
	uint64_t c = C;
#pragma unroll
	for (int i = 0; i < N; i++) {
		c += a.w[i];
		s.w[i] = (uint32_t)c;
		c >>= 32;
	}
	return c ? s : a;
}

#ifndef BC3_C_MODMUL
/* PTX carry chains. One asm statement per instruction, all volatile so they keep
 * their order: the carry flag lives between them. */
#define BC3_MADLO_CC(d, a, b)  asm volatile("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define BC3_MADCLO_CC(d, a, b) asm volatile("madc.lo.cc.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define BC3_MADCHI_CC(d, a, b) asm volatile("madc.hi.cc.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define BC3_MADCHI(d, a, b)    asm volatile("madc.hi.u32 %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define BC3_ADD_CC(d, a)       asm volatile("add.cc.u32 %0, %0, %1;" : "+r"(d) : "r"(a))
#define BC3_ADDC_CC(d, a)      asm volatile("addc.cc.u32 %0, %0, %1;" : "+r"(d) : "r"(a))
#define BC3_ADDC(d, a)         asm volatile("addc.u32 %0, %0, %1;" : "+r"(d) : "r"(a))

#ifndef BC3_MUL_EO
/* Products with every lo/hi pair on an even word (k, k+1), so each pair can be
 * one wide multiply-add: a_i * b_j goes to A0 when i+j is even (lo at word i+j),
 * to A1 when odd (lo at word i+j-1, i.e. A1[k] is column k+1). Indices are
 * compile-time; the bool arrays (folded away) mark written words, and a first
 * write adds an immediate 0. */
#define BC3_ACC(op, d, a, b)  asm volatile(op " %0, %1, %2, %0;" : "+r"(d) : "r"(a), "r"(b))
#define BC3_MAC0(op, d, a, b) asm volatile(op " %0, %1, %2, 0;" : "=r"(d) : "r"(a), "r"(b))
#define BC3_ACCI(op, d, x)    asm volatile(op " %0, %0, %1;" : "+r"(d) : "r"(x))
#define BC3_A3(op, d, x, y)   asm volatile(op " %0, %1, %2;" : "=r"(d) : "r"(x), "r"(y))
#define BC3_A2I(op, d, x)     asm volatile(op " %0, %1, 0;" : "=r"(d) : "r"(x))
#define BC3_A0I(op, d)        asm volatile(op " %0, 0, 0;" : "=r"(d))

/* d (+)= lo or hi of a*b, with carry in CI and carry out CO */
template <bool HI, bool CI, bool CO>
__device__ __forceinline__ void bc3_mac(uint32_t& d, bool& def, uint32_t a, uint32_t b)
{
	if (def) {
		if (!HI && !CI && CO) BC3_ACC("mad.lo.cc.u32", d, a, b);
		if (!HI && CI && CO)  BC3_ACC("madc.lo.cc.u32", d, a, b);
		if (!HI && CI && !CO) BC3_ACC("madc.lo.u32", d, a, b);
		if (HI && CI && CO)   BC3_ACC("madc.hi.cc.u32", d, a, b);
		if (HI && CI && !CO)  BC3_ACC("madc.hi.u32", d, a, b);
	} else {
		if (!HI && !CI && CO) BC3_MAC0("mad.lo.cc.u32", d, a, b);
		if (!HI && CI && CO)  BC3_MAC0("madc.lo.cc.u32", d, a, b);
		if (!HI && CI && !CO) BC3_MAC0("madc.lo.u32", d, a, b);
		if (HI && CI && CO)   BC3_MAC0("madc.hi.cc.u32", d, a, b);
		if (HI && CI && !CO)  BC3_MAC0("madc.hi.u32", d, a, b);
	}
	def = true;
}

/* a_i * b_j for j = J0, J0+2, .. < N, lo at word I+j-S, hi at the next; the
 * chain's carry goes to the word after the last hi, if that is below LIM */
template <int N, int I, int J0, int S, int LIM>
__device__ __forceinline__ void bc3_mul_chain(uint32_t (&A)[2 * N], bool (&D)[2 * N], uint32_t ai, const uint32_t* b)
{
	if (J0 >= N) return;
	constexpr int JL = J0 + ((N - 1 - J0) & ~1); /* last j */
	constexpr int TOP = I + JL - S + 2;
	constexpr bool top = TOP < LIM;
#pragma unroll
	for (int j = J0; j < N; j += 2) {
		const int k = I + j - S;
		if (j == J0) bc3_mac<false, false, true>(A[k], D[k], ai, b[j]);
		else bc3_mac<false, true, true>(A[k], D[k], ai, b[j]);
		if (j == JL && !top) bc3_mac<true, true, false>(A[k + 1], D[k + 1], ai, b[j]);
		else bc3_mac<true, true, true>(A[k + 1], D[k + 1], ai, b[j]);
	}
	if (top) {
		constexpr int T = TOP < LIM ? TOP : 0;
		if (D[T]) BC3_ACCI("addc.u32", A[T], 0u);
		else BC3_A0I("addc.u32", A[T]);
		D[T] = true;
	}
}

/* rows I..IE-1 of a*b; SQ: the cross terms only (j > i) */
template <int N, int I, int IE, bool SQ>
struct bc3_mul_rows {
	static __device__ __forceinline__ void run(uint32_t (&A0)[2 * N], bool (&D0)[2 * N], uint32_t (&A1)[2 * N],
	                                           bool (&D1)[2 * N], const uint32_t* a, const uint32_t* b)
	{
		constexpr int jmin = SQ ? I + 1 : 0;
		constexpr int je = jmin + ((I + jmin) & 1), jo = jmin + (((I + jmin) & 1) ^ 1);
		bc3_mul_chain<N, I, je, 0, 2 * N>(A0, D0, a[I], b);
		bc3_mul_chain<N, I, jo, 1, 2 * N - 1>(A1, D1, a[I], b);
		bc3_mul_rows<N, I + 1, IE, SQ>::run(A0, D0, A1, D1, a, b);
	}
};
template <int N, int IE, bool SQ>
struct bc3_mul_rows<N, IE, IE, SQ> {
	static __device__ __forceinline__ void run(uint32_t (&)[2 * N], bool (&)[2 * N], uint32_t (&)[2 * N],
	                                           bool (&)[2 * N], const uint32_t*, const uint32_t*) {}
};

/* t[0..K) = A0 + A1 * 2^32, words not yet written counting as zero */
template <int N, int K>
__device__ __forceinline__ void bc3_mul_join(uint32_t* t, const uint32_t (&E)[2 * N], const bool (&DE)[2 * N],
                                             const uint32_t (&O)[2 * N], const bool (&DO)[2 * N])
{
	t[0] = DE[0] ? E[0] : 0u;
	bool cy = false; /* a carry is pending */
#pragma unroll
	for (int k = 1; k < K; k++) {
		const bool e = DE[k], o = DO[k - 1], last = k == K - 1;
		if (!cy) {
			if (e && o) {
				if (last) t[k] = E[k] + O[k - 1];
				else BC3_A3("add.cc.u32", t[k], E[k], O[k - 1]);
				cy = !last;
			} else t[k] = e ? E[k] : o ? O[k - 1] : 0u;
		} else if (e && o) {
			if (last) BC3_A3("addc.u32", t[k], E[k], O[k - 1]);
			else BC3_A3("addc.cc.u32", t[k], E[k], O[k - 1]);
		} else if (e || o) {
			const uint32_t z = e ? E[k] : O[k - 1];
			if (last) BC3_A2I("addc.u32", t[k], z);
			else BC3_A2I("addc.cc.u32", t[k], z);
		} else {
			if (last) BC3_A0I("addc.u32", t[k]);
			else BC3_A0I("addc.cc.u32", t[k]);
		}
	}
}

template <int N>
__device__ __forceinline__ void bc3_mul_full(uint32_t (&t)[2 * N], const uint32_t* a, const uint32_t* b)
{
	uint32_t A0[2 * N], A1[2 * N];
	bool D0[2 * N], D1[2 * N];
#pragma unroll
	for (int k = 0; k < 2 * N; k++) D0[k] = D1[k] = false;
	bc3_mul_rows<N, 0, N, false>::run(A0, D0, A1, D1, a, b);
	bc3_mul_join<N, 2 * N>(t, A0, D0, A1, D1);
}

/* a^2: the cross terms a_i a_j (j > i) once, doubled, plus the squares a_i^2. */
template <int N>
__device__ __forceinline__ void bc3_sqr_full(uint32_t (&t)[2 * N], const uint32_t* a)
{
	uint32_t A0[2 * N], A1[2 * N], x[2 * N];
	bool D0[2 * N], D1[2 * N];
#pragma unroll
	for (int k = 0; k < 2 * N; k++) D0[k] = D1[k] = false;
	bc3_mul_rows<N, 0, N - 1, true>::run(A0, D0, A1, D1, a, a);
	bc3_mul_join<N, 2 * N>(x, A0, D0, A1, D1);
	/* cross sum < 2^(64N-1), so doubling fits */
	t[0] = x[0] << 1;
#pragma unroll
	for (int k = 1; k < 2 * N; k++) t[k] = __funnelshift_l(x[k - 1], x[k], 1);
	bool d = true;
	bc3_mac<false, false, true>(t[0], d, a[0], a[0]);
	bc3_mac<true, true, true>(t[1], d, a[0], a[0]);
#pragma unroll
	for (int i = 1; i < N - 1; i++) {
		bc3_mac<false, true, true>(t[2 * i], d, a[i], a[i]);
		bc3_mac<true, true, true>(t[2 * i + 1], d, a[i], a[i]);
	}
	bc3_mac<false, true, true>(t[2 * N - 2], d, a[N - 1], a[N - 1]);
	bc3_mac<true, true, false>(t[2 * N - 1], d, a[N - 1], a[N - 1]);
}

#else /* BC3_MUL_EO: the previous even/odd-column products */

/* Row i of a product into the two column accumulators: E takes a_i * b_j for
 * even j (lo at column i+j, hi at i+j+1, so the row is one contiguous chain),
 * O the odd j one column down. j starts at j0 (> i for the squaring's cross
 * terms). Each row's chain ends one column higher than the last row's, on a
 * word no row has written yet, so its final carry is never lost. */
template <int N>
__device__ __forceinline__ void bc3_mul_row(uint32_t (&E)[2 * N], uint32_t (&O)[2 * N], uint32_t ai,
                                            const uint32_t* b, int i, int j0)
{
	const int je = j0 + (j0 & 1), jo = j0 | 1;
	if (je < N) {
		BC3_MADLO_CC(E[i + je], ai, b[je]);
		BC3_MADCHI_CC(E[i + je + 1], ai, b[je]);
#pragma unroll
		for (int j = je + 2; j < N; j += 2) {
			BC3_MADCLO_CC(E[i + j], ai, b[j]);
			BC3_MADCHI_CC(E[i + j + 1], ai, b[j]);
		}
		BC3_ADDC(E[i + N], 0u); /* N even: the last even j is N-2 */
	}
	if (jo < N) {
		BC3_MADLO_CC(O[i + jo - 1], ai, b[jo]);
		if (jo + 2 < N || i + N < 2 * N - 1)
			BC3_MADCHI_CC(O[i + jo], ai, b[jo]);
		else
			BC3_MADCHI(O[i + jo], ai, b[jo]);
#pragma unroll
		for (int j = jo + 2; j < N; j += 2) {
			BC3_MADCLO_CC(O[i + j - 1], ai, b[j]);
			if (j + 2 < N || i + N < 2 * N - 1)
				BC3_MADCHI_CC(O[i + j], ai, b[j]);
			else
				BC3_MADCHI(O[i + j], ai, b[j]); /* top of the product: no carry out */
		}
		if (i + N < 2 * N - 1) BC3_ADDC(O[i + N], 0u);
	}
}

/* t = E + O * 2^32 */
template <int N>
__device__ __forceinline__ void bc3_mul_join(uint32_t (&t)[2 * N], uint32_t (&E)[2 * N], const uint32_t (&O)[2 * N])
{
	t[0] = E[0];
#pragma unroll
	for (int k = 1; k < 2 * N; k++) t[k] = E[k];
	BC3_ADD_CC(t[1], O[0]);
#pragma unroll
	for (int k = 2; k < 2 * N - 1; k++) BC3_ADDC_CC(t[k], O[k - 1]);
	BC3_ADDC(t[2 * N - 1], O[2 * N - 2]);
}

template <int N>
__device__ __forceinline__ void bc3_mul_full(uint32_t (&t)[2 * N], const uint32_t* a, const uint32_t* b)
{
	uint32_t E[2 * N], O[2 * N];
#pragma unroll
	for (int k = 0; k < 2 * N; k++) E[k] = O[k] = 0;
#pragma unroll
	for (int i = 0; i < N; i++) bc3_mul_row<N>(E, O, a[i], b, i, 0);
	bc3_mul_join<N>(t, E, O);
}

/* a^2: the cross terms a_i a_j (j > i) once, doubled, plus the squares a_i^2. */
template <int N>
__device__ __forceinline__ void bc3_sqr_full(uint32_t (&t)[2 * N], const uint32_t* a)
{
	uint32_t E[2 * N], O[2 * N], x[2 * N];
#pragma unroll
	for (int k = 0; k < 2 * N; k++) E[k] = O[k] = 0;
#pragma unroll
	for (int i = 0; i < N - 1; i++) bc3_mul_row<N>(E, O, a[i], a, i, i + 1);
	bc3_mul_join<N>(x, E, O);
	/* cross sum < 2^(64N-1), so doubling fits */
	t[0] = x[0] << 1;
#pragma unroll
	for (int k = 1; k < 2 * N; k++) t[k] = __funnelshift_l(x[k - 1], x[k], 1);
	BC3_MADLO_CC(t[0], a[0], a[0]);
	BC3_MADCHI_CC(t[1], a[0], a[0]);
#pragma unroll
	for (int i = 1; i < N - 1; i++) {
		BC3_MADCLO_CC(t[2 * i], a[i], a[i]);
		BC3_MADCHI_CC(t[2 * i + 1], a[i], a[i]);
	}
	BC3_MADCLO_CC(t[2 * N - 2], a[N - 1], a[N - 1]);
	BC3_MADCHI(t[2 * N - 1], a[N - 1], a[N - 1]);
}
#endif /* BC3_MUL_EO */

/* Mod::reduce of t mod 2^(32N) - C, C < 2^16, to N words but NOT canonical:
 * r < 2^(32N), r == t (mod m). r = lo + hi*C (folded twice). */
#if defined(BC3_MUL_EO) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700)
/* one lo chain and one hi chain (smaller on sm_6x) */
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_reduce_lazy(const uint32_t (&t)[2 * N])
{
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) r.w[i] = t[i];
	uint32_t w = 0;
	BC3_MADLO_CC(r.w[0], t[N], C);
#pragma unroll
	for (int i = 1; i < N; i++) BC3_MADCLO_CC(r.w[i], t[N + i], C);
	BC3_ADDC(w, 0u);
	{
		/* hi(t[N+i] * C) into column i+1; the top one into w */
		uint32_t h[N];
#pragma unroll
		for (int i = 0; i < N; i++) h[i] = __umulhi(t[N + i], C);
		BC3_ADD_CC(r.w[1], h[0]);
#pragma unroll
		for (int i = 2; i < N; i++) BC3_ADDC_CC(r.w[i], h[i - 1]);
		BC3_ADDC(w, h[N - 1]);
	}
	/* second fold: w < 2^17, w*C < 2^33 */
	const uint64_t f = (uint64_t)w * C;
	uint32_t c = 0;
	BC3_ADD_CC(r.w[0], (uint32_t)f);
	BC3_ADDC_CC(r.w[1], (uint32_t)(f >> 32));
#pragma unroll
	for (int i = 2; i < N; i++) BC3_ADDC_CC(r.w[i], 0u);
	BC3_ADDC(c, 0u);
	if (c) { /* wrapped past 2^(32N): r is now tiny, adding C cannot wrap again */
		BC3_ADD_CC(r.w[0], C);
#pragma unroll
		for (int i = 1; i < N - 1; i++) BC3_ADDC_CC(r.w[i], 0u);
		BC3_ADDC(r.w[N - 1], 0u);
	}
	return r;
}
#else
/* q = lo + hi*C < 2^(32N+17) with A0 seeded as lo (A1's part < 2^(32N+16), so
 * it never carries past column N); r = q mod 2^(32N) + q[N]*C. A wrap leaves
 * r < 2^33, so + C stays within r0, r1. */
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_reduce_lazy(const uint32_t (&t)[2 * N])
{
	uint32_t A0[2 * N], A1[2 * N], q[N + 1];
	bool D0[2 * N], D1[2 * N];
#pragma unroll
	for (int k = 0; k < 2 * N; k++) {
		D0[k] = k < N;
		D1[k] = false;
		A0[k] = k < N ? t[k] : 0u;
		A1[k] = 0u;
	}
	bc3_mul_chain<N, 0, 0, 0, N + 1>(A0, D0, C, t + N);
	bc3_mul_chain<N, 0, 1, 1, N>(A1, D1, C, t + N);
	bc3_mul_join<N, N + 1>(q, A0, D0, A1, D1);
	bc3u<N> r;
#pragma unroll
	for (int k = 0; k < N; k++) r.w[k] = q[k];
	BC3_ACC("mad.lo.cc.u32", r.w[0], q[N], C);
	BC3_ACC("madc.hi.cc.u32", r.w[1], q[N], C);
#pragma unroll
	for (int k = 2; k < N; k++) BC3_A2I("addc.cc.u32", r.w[k], r.w[k]);
	uint32_t c;
	BC3_A0I("addc.u32", c);
	const uint32_t m = 0u - c;
	BC3_ACCI("add.cc.u32", r.w[0], m & C);
	BC3_A2I("addc.u32", r.w[1], r.w[1]);
	return r;
}
#endif

/* Lazy products: any inputs < 2^(32N), any output < 2^(32N) congruent mod m.
 * A chain of them is made canonical once, by bc3_mod_normalize, at its end. */
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_mul_lazy(const bc3u<N>& a, const bc3u<N>& b)
{
	uint32_t t[2 * N];
	bc3_mul_full<N>(t, a.w, b.w);
	return bc3_reduce_lazy<N, C>(t);
}
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_sqr_lazy(const bc3u<N>& a)
{
	uint32_t t[2 * N];
	bc3_sqr_full<N>(t, a.w);
	return bc3_reduce_lazy<N, C>(t);
}

/* Canonical (< m) results, bit for bit the coin's. */
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_mul(const bc3u<N>& a, const bc3u<N>& b)
{
	return bc3_mod_normalize<N, C>(bc3_mod_mul_lazy<N, C>(a, b));
}
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_sqr(const bc3u<N>& a)
{
	return bc3_mod_normalize<N, C>(bc3_mod_sqr_lazy<N, C>(a));
}

#else /* BC3_C_MODMUL: the plain-C transcription, kept as a reference */

/* Mod::reduce of a 2N-limb product: two carry folds by C, then normalize.
 * Accepts any a, b < 2^(32N); the result is canonical (< m), so it equals the
 * coin's bit for bit. */
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_mul(const bc3u<N>& a, const bc3u<N>& b)
{
	uint32_t t[2 * N];
#pragma unroll
	for (int i = 0; i < 2 * N; i++) t[i] = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		uint64_t c = 0;
#pragma unroll
		for (int j = 0; j < N; j++) {
			c = (uint64_t)a.w[i] * b.w[j] + t[i + j] + (c >> 32);
			t[i + j] = (uint32_t)c;
		}
		t[i + N] = (uint32_t)(c >> 32);
	}
	bc3u<N> r;
	uint64_t c = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		c = (uint64_t)t[N + i] * C + t[i] + (c >> 32);
		r.w[i] = (uint32_t)c;
	}
	c = (c >> 32) * C; /* first carry <= C, so this is < 2^32 */
#pragma unroll
	for (int i = 0; i < N; i++) {
		c += r.w[i];
		r.w[i] = (uint32_t)c;
		c >>= 32;
	}
	if (c) { /* the second fold overflowed: adding C cannot overflow again */
		c = C;
#pragma unroll
		for (int i = 0; i < N; i++) {
			c += r.w[i];
			r.w[i] = (uint32_t)c;
			c >>= 32;
		}
	}
	return bc3_mod_normalize<N, C>(r);
}

template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_sqr(const bc3u<N>& a)
{
	return bc3_mod_mul<N, C>(a, a);
}
template <int N>
__device__ __forceinline__ void bc3_mul_full(uint32_t (&t)[2 * N], const uint32_t* a, const uint32_t* b)
{
#pragma unroll
	for (int i = 0; i < 2 * N; i++) t[i] = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		uint64_t c = 0;
#pragma unroll
		for (int j = 0; j < N; j++) {
			c = (uint64_t)a[i] * b[j] + t[i + j] + (c >> 32);
			t[i + j] = (uint32_t)c;
		}
		t[i + N] = (uint32_t)(c >> 32);
	}
}
template <int N>
__device__ __forceinline__ void bc3_sqr_full(uint32_t (&t)[2 * N], const uint32_t* a)
{
	bc3_mul_full<N>(t, a, a);
}
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_mul_lazy(const bc3u<N>& a, const bc3u<N>& b)
{
	return bc3_mod_mul<N, C>(a, b);
}
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_sqr_lazy(const bc3u<N>& a)
{
	return bc3_mod_mul<N, C>(a, a);
}
#endif /* BC3_C_MODMUL */

/* The six moduli of mix512 (uint.hpp: M64/M128/M256 for the fixed exponents and
 * M*::mul, F64/F128/F256 for the fixed-base tables and inverse256). */
#define BC3_M64 2, 59u
#define BC3_M128 4, 159u
#define BC3_M256 8, 189u
#define BC3_F64 2, 1469u
#define BC3_F128 4, 15449u
#define BC3_F256 8, 36113u
