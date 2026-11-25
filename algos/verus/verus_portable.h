/* VerusHash 2.2 per-nonce path in portable integer code, __host__ __device__.
 *
 * Same function as verus_host_hash() (AES-NI + PCLMUL + FixKey), without x86
 * intrinsics, so one source runs on the host and on the GPU.
 *
 * The one structural difference from the reference: the key is NOT mutated in
 * place and restored by FixKey. A hash writes at most 64 slots (2 per
 * iteration x 32) and every hash of a job starts from the same pristine key, so
 * the private state is a copy-on-write overlay over one shared read-only key:
 *
 *   read(i)    i >= 512 || map[i] == 0xff ? K[i] : ov[map[i]]
 *   write(i,v) first write to i takes the next ov slot; ov[map[i]] = v
 *   reset      map[i] = 0xff for every written i
 *
 * This holds exactly what a private copy would, for every write order, and
 * needs no FixKey journal.
 *
 * vp_clhash transcribes cpuminer-opt algo/verus/verus_clhash_iter.h.
 * VP_FAULT_* plant deliberate errors for test builds; never define them in the
 * miner.
 */
#ifndef VERUS_PORTABLE_H
#define VERUS_PORTABLE_H

#include <stdint.h>

#if defined(__CUDACC__)
#define VP_HD __host__ __device__ __forceinline__
#else
#define VP_HD static inline
#endif

/* the same for member functions, which cannot be `static` */
#if defined(__CUDACC__)
#define VP_HDM __host__ __device__ __forceinline__
#else
#define VP_HDM inline
#endif

#define VP_KEY_SLOTS   552   /* 8832 B: 512 mutable + 40 read-only tail */
#define VP_KEY_MASK    511
#define VP_OVL_SLOTS    64   /* 2 writes x 32 iterations */

typedef struct { uint32_t w[4]; } vp128;   /* little-endian 32-bit words */

/* coverage counters for test builds (NULL in the miner) */
typedef struct {
	uint64_t cases[8];
	uint64_t monk_rounds[2][8];   /* [0x14, 0x18][rounds-1] */
	uint64_t monk_branch[2][2];   /* [case][0 = bit set, 1 = clear] */
	uint64_t collide;             /* prand_idx == prandex_idx */
	uint64_t rc_cross;            /* an rc run read a slot >= 512 */
	uint64_t haraka_tail;         /* (intermediate & 511) > 472 */
	uint64_t negdiv;              /* signed modulo with a negative dividend */
	uint64_t rewrite;             /* write to an already-dirty slot */
	uint64_t dirty_reads;         /* reads served by the overlay */
	int      ovl_max;
} vp_stats;

#if defined(VP_STATS)
#define VP_STAT(st, x) do { if (st) { x; } } while (0)
#else
#define VP_STAT(st, x) do { } while (0)
#endif

/* ------------------------------------------------------------------------- */
/* 128-bit helpers                                                            */
/* ------------------------------------------------------------------------- */
VP_HD vp128 vp_xor(vp128 a, vp128 b)
{
	vp128 r;
	r.w[0] = a.w[0] ^ b.w[0]; r.w[1] = a.w[1] ^ b.w[1];
	r.w[2] = a.w[2] ^ b.w[2]; r.w[3] = a.w[3] ^ b.w[3];
	return r;
}
VP_HD uint64_t vp_lo(vp128 a) { return (uint64_t)a.w[1] << 32 | a.w[0]; }
VP_HD uint64_t vp_hi(vp128 a) { return (uint64_t)a.w[3] << 32 | a.w[2]; }
VP_HD vp128 vp_make(uint64_t lo, uint64_t hi)
{
	vp128 r;
	r.w[0] = (uint32_t)lo; r.w[1] = (uint32_t)(lo >> 32);
	r.w[2] = (uint32_t)hi; r.w[3] = (uint32_t)(hi >> 32);
	return r;
}
VP_HD vp128 vp_load(const uint8_t *p)
{
	vp128 r;
	for (int k = 0; k < 4; k++)
		r.w[k] = (uint32_t)p[4*k] | (uint32_t)p[4*k+1] << 8 |
		         (uint32_t)p[4*k+2] << 16 | (uint32_t)p[4*k+3] << 24;
	return r;
}
VP_HD void vp_store(uint8_t *p, vp128 a)
{
	for (int k = 0; k < 4; k++) {
		p[4*k]   = (uint8_t)a.w[k];         p[4*k+1] = (uint8_t)(a.w[k] >> 8);
		p[4*k+2] = (uint8_t)(a.w[k] >> 16); p[4*k+3] = (uint8_t)(a.w[k] >> 24);
	}
}
/* _mm_unpacklo_epi32 / _mm_unpackhi_epi32 */
VP_HD vp128 vp_unpacklo(vp128 a, vp128 b)
{ vp128 r; r.w[0] = a.w[0]; r.w[1] = b.w[0]; r.w[2] = a.w[1]; r.w[3] = b.w[1]; return r; }
VP_HD vp128 vp_unpackhi(vp128 a, vp128 b)
{ vp128 r; r.w[0] = a.w[2]; r.w[1] = b.w[2]; r.w[2] = a.w[3]; r.w[3] = b.w[3]; return r; }

/* ------------------------------------------------------------------------- */
/* Primitives                                                                 */
/* ------------------------------------------------------------------------- */
VP_HD uint32_t vp_rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

/* ShiftRows: input column offsets of rows 1 and 3 */
#if defined(VP_FAULT_AES_SHIFTROWS)
#define VP_SR1 3
#define VP_SR3 1
#else
#define VP_SR1 1
#define VP_SR3 3
#endif

/* One output column of an AES round from the words holding its rows 0..3.
 * T0[x] is the column contribution of a row-0 byte, little-endian
 * [2S, S, S, 3S]; rows 1..3 use T0 rotated by 8, 16 and 24. */
VP_HD uint32_t vp_aes_col(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
                          uint32_t rk, const uint32_t *T0)
{
	return T0[w0 & 0xff] ^ vp_rotl32(T0[(w1 >> 8) & 0xff], 8) ^
	       vp_rotl32(T0[(w2 >> 16) & 0xff], 16) ^ vp_rotl32(T0[w3 >> 24], 24) ^ rk;
}

/* _mm_aesenc_si128: row r of output column j comes from input column j + r */
VP_HD vp128 vp_aesenc(vp128 s, vp128 rk, const uint32_t *T0)
{
	vp128 r;
	for (int j = 0; j < 4; j++)
		r.w[j] = vp_aes_col(s.w[j], s.w[(j + VP_SR1) & 3], s.w[(j + 2) & 3],
		                    s.w[(j + VP_SR3) & 3], rk.w[j], T0);
	return r;
}

/* 64 x 64 -> 128 carry-less multiply (reference form: one bit per step) */
VP_HD void vp_clmul64(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi)
{
	uint64_t l = 0, h = 0;
	for (int i = 0; i < 64; i++)
		if ((b >> i) & 1) {
			l ^= a << i;
			if (i) h ^= a >> (64 - i);
		}
	*lo = l; *hi = h;
}

/* first factor of vp_clmul_x10 */
#if defined(VP_FAULT_CLMUL_IMM)
#define VP_CLMUL_A(lo, hi) (hi)
#else
#define VP_CLMUL_A(lo, hi) (lo)
#endif

/* _mm_clmulepi64_si128(x, x, 0x10) = x.lo (x) x.hi: imm bit 0 selects the
 * first operand's qword, bit 4 the second's */
VP_HD vp128 vp_clmul_x10(vp128 x)
{
	uint64_t l, h;
	vp_clmul64(VP_CLMUL_A(vp_lo(x), vp_hi(x)), vp_hi(x), &l, &h);
	return vp_make(l, h);
}

/* rounding term of vp_mulhrs_w */
#if defined(VP_FAULT_MULHRS_NOROUND)
#define VP_MULHRS_RND 0
#else
#define VP_MULHRS_RND 0x4000
#endif

/* One 32-bit word of _mm_mulhrs_epi16 (two signed 16-bit lanes). Its rounding,
 * ((p >> 14) + 1) >> 1, equals (p + 0x4000) >> 15 for arithmetic shifts, and
 * |p| <= 2^30 cannot overflow. */
VP_HD uint32_t vp_mulhrs_w(uint32_t a, uint32_t b)
{
	const int32_t xl = (int32_t)(a << 16) >> 16, yl = (int32_t)(b << 16) >> 16;
	const int32_t xh = (int32_t)a >> 16, yh = (int32_t)b >> 16;
	const int32_t tl = (xl * yl + VP_MULHRS_RND) >> 15;
	const int32_t th = (xh * yh + VP_MULHRS_RND) >> 15;
	return ((uint32_t)tl & 0xffff) | (uint32_t)th << 16;
}
VP_HD vp128 vp_mulhrs(vp128 a, vp128 b)
{
	vp128 r;
	for (int k = 0; k < 4; k++) r.w[k] = vp_mulhrs_w(a.w[k], b.w[k]);
	return r;
}

/* 1/x within 1 ulp: one MUFU on the GPU, IEEE division on the host */
VP_HD float vp_rcpf(float x)
{
#if defined(__CUDA_ARCH__)
	float r;
	asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
	return r;
#else
	return 1.0f / x;
#endif
}

/* u % v, exact, for u < 2^64 and 1 <= v <= 2^31, without a 64-bit division.
 * Each step r -= trunc(fl(r) * fl(1/v)) * v leaves |r| <= |r| * 2^-21 + v
 * (2^43 + v, then 2^22 + v, then < 2v), so two corrections each way finish it.
 * q * v may wrap mod 2^64, which is exact for the remainder. */
VP_HD uint32_t vp_umod64_32(uint64_t u, uint32_t v)
{
	const float fv = vp_rcpf((float)v);
	const uint64_t q = (uint64_t)((float)u * fv);
	int64_t r = (int64_t)(u - q * (uint64_t)v);
	r -= (int64_t)((float)r * fv) * (int64_t)v;
	r -= (int64_t)((float)r * fv) * (int64_t)v;
	if (r < 0) r += v;
	if (r < 0) r += v;
	if (r >= (int64_t)v) r -= v;
	if (r >= (int64_t)v) r -= v;
	return (uint32_t)r;
}

/* _mm_cvtsi32_si128((int)(dividend % divisor)): C truncation toward zero, so
 * the remainder takes the dividend's sign. The divisor is never 0 or -1 (its
 * & 0x1c is 0xc or 0x18). */
VP_HD vp128 vp_modulo(int64_t dividend, int32_t divisor)
{
#if defined(VP_FAULT_MOD_UNSIGNED)
	const uint32_t m = (uint32_t)((uint64_t)dividend % (uint32_t)divisor);
#else
	const uint64_t u = dividend < 0 ? 0 - (uint64_t)dividend : (uint64_t)dividend;
	const uint32_t v = divisor < 0 ? 0u - (uint32_t)divisor : (uint32_t)divisor;
	const uint32_t um = vp_umod64_32(u, v);
	const uint32_t m = dividend < 0 ? 0u - um : um;
#endif
	vp128 r; r.w[0] = m; r.w[1] = 0; r.w[2] = 0; r.w[3] = 0;
	return r;
}

/* precompReduction64(acc ^ lazyLengthHash(1024, 64)) -> 64-bit intermediate */
VP_HD uint64_t vp_finalize(vp128 acc)
{
	/* lazyLengthHash: clmul(set_epi64x(1024, 64), same, 0x10) = 64 (x) 1024 */
	uint64_t l, h;
	vp_clmul64(64, 1024, &l, &h);
	acc = vp_xor(acc, vp_make(l, h));

	/* Q2 = clmul(A, 0x1b, 0x01) = A.hi (x) 0x1b */
	uint64_t q2l, q2h;
	vp_clmul64(vp_hi(acc), 0x1b, &q2l, &q2h);
	/* Q3 = shuffle_epi8(table, Q2 >> 64): per index byte 0 if bit 7 is set,
	 * else table[idx & 15]. Only Q2.hi's bytes contribute (the upper bytes of
	 * the shifted vector are zero and table[0] is 0). The table,
	 * { 0,27,54,45,108,119,90,65, 216,195,238,245,180,175,130,153 }, is packed
	 * into two words: an indexed byte array would go to local memory. */
	const uint64_t tab_lo = 0x415a776c2d361b00ULL, tab_hi = 0x9982afb4f5eec3d8ULL;
	uint64_t q3 = 0;
	for (int k = 0; k < 8; k++) {
		const uint32_t ix = (uint32_t)(q2h >> (8 * k)) & 0xff;
		const uint64_t t = (ix & 8) ? tab_hi : tab_lo;
		const uint64_t v = (ix & 0x80) ? 0 : (t >> (8 * (ix & 7))) & 0xff;
		q3 |= v << (8 * k);
	}
	/* final = Q3 ^ Q2 ^ A, low 64 bits (the high half is garbage upstream) */
	return q3 ^ q2l ^ vp_lo(acc);
}

/* value-type adaptors: the templates below work on KEY::V, which is vp128 for
 * the scalar paths and one 32-bit word per lane for the warp path
 * (verus_warp.cuh). The pointer argument only selects the overload. */
VP_HD vp128 vp_loadw(const uint32_t *p)
{
	vp128 r; r.w[0] = p[0]; r.w[1] = p[1]; r.w[2] = p[2]; r.w[3] = p[3];
	return r;
}
VP_HD vp128 vp_loadv(const uint32_t *p, const vp128 *) { return vp_loadw(p); }
VP_HD vp128 vp_mod_v(int64_t d, int32_t dv, const vp128 *) { return vp_modulo(d, dv); }
VP_HD vp128 vp_gather(vp128 v) { return v; }

/* lets a __host__ __device__ template call the warp path's __device__-only
 * functions; nvcc never makes a host instantiation of those */
#if defined(__CUDACC__)
#define VP_TPL _Pragma("nv_exec_check_disable")
#else
#define VP_TPL
#endif

/* ------------------------------------------------------------------------- */
/* Key storage. vp_clhash / vp_hash are templates over it; a KEY provides     */
/*   rd(i, st)            the current value of slot i (0..551)                 */
/*   wr(i, v, st)         a store to slot i (0..511); vp_wr2 does a pair       */
/*   begin(it, ip, ix, kprand, kprandex)  start of iteration `it`, after the   */
/*                        two selected slots were read (the FixKey journal)    */
/*   end_hash()           restore the pristine key for the next nonce          */
/*   pristine(i)          the job key, for the VP_FAULT_*_PRISTINE controls    */
/* ------------------------------------------------------------------------- */

/* The copy-on-write overlay. */
typedef struct vp_ovl {
	typedef vp128 V;
	const vp128 *K;                /* pristine job key, VP_KEY_SLOTS */
	uint8_t      map[512];         /* 0xff = clean, else index into ov */
	uint16_t     idx[VP_OVL_SLOTS];/* written key indices, for reset */
	vp128        ov[VP_OVL_SLOTS];
	int          n;

	VP_HDM vp128 rd(uint32_t i, vp_stats *st) const
	{
		if (i < 512 && map[i] != 0xff) {
			VP_STAT(st, st->dirty_reads++);
			return ov[map[i]];
		}
		(void)st;
		return K[i];
	}
	VP_HDM void wr(uint32_t i, vp128 v, vp_stats *st)
	{
		if (map[i] == 0xff) {
			map[i] = (uint8_t)n;
			idx[n] = (uint16_t)i;
			n++;
		} else {
			VP_STAT(st, st->rewrite++);
		}
		(void)st;
		ov[map[i]] = v;
	}
	VP_HDM void begin(int, uint32_t, uint32_t, vp128, vp128) { }
	VP_HDM void end_hash()
	{
#if !defined(VP_FAULT_NO_RESET)
		for (int k = 0; k < n; k++) map[idx[k]] = 0xff;
#endif
		n = 0;
	}
	VP_HDM vp128 pristine(uint32_t i) const { return K[i]; }
	VP_HDM int dirty() const { return n; }
} vp_ovl;

VP_HD void vp_ovl_init(vp_ovl *o, const vp128 *K)
{
	o->K = K;
	for (int i = 0; i < 512; i++) o->map[i] = 0xff;
	o->n = 0;
}
VP_HD int vp_ovl_clean(const vp_ovl *o)
{
	for (int i = 0; i < 512; i++) if (o->map[i] != 0xff) return 0;
	return 1;
}

/* The reference scheme: a private copy of the key, restored after each hash
 * by FixKey from the per-iteration journal. Used by test builds. */
typedef struct vp_keycopy {
	typedef vp128 V;
	const vp128 *K;                /* pristine job key */
	vp128       *k;                /* private copy, VP_KEY_SLOTS */
	vp128        sp[32], spx[32];  /* g_prand / g_prandex */
	uint32_t     fr[32], frx[32];  /* fixrand / fixrandex */

	VP_HDM vp128 rd(uint32_t i, vp_stats *st) const { (void)st; return k[i]; }
	VP_HDM void wr(uint32_t i, vp128 v, vp_stats *st) { (void)st; k[i] = v; }
	VP_HDM void begin(int it, uint32_t ip, uint32_t ix, vp128 kp, vp128 kx)
	{
		sp[it] = kp; spx[it] = kx; fr[it] = ip; frx[it] = ix;
	}
	/* FixKey: reverse order (indices repeat, the oldest save lands last),
	 * prandex before prand within an iteration */
	VP_HDM void end_hash()
	{
		for (int it = 31; it > -1; it--) {
			k[frx[it]] = spx[it];
			k[fr[it]]  = sp[it];
		}
	}
	VP_HDM vp128 pristine(uint32_t i) const { return K[i]; }
	VP_HDM int dirty() const { return 0; }
} vp_keycopy;

VP_HD void vp_keycopy_init(vp_keycopy *c, const vp128 *K, vp128 *k)
{
	c->K = K; c->k = k;
	for (int i = 0; i < VP_KEY_SLOTS; i++) k[i] = K[i];
}

/* rc[] reads in cases 0x10/0x14/0x18 (consensus reads the MUTATED key) */
#if defined(VP_FAULT_RC_PRISTINE)
#define VP_RC(i) (o->pristine(i))
#else
#define VP_RC(i) o->rd((i), st)
#endif

/* The iteration's two key writes, in order (the second wins if both hit the
 * same slot). No key read happens between them in any case, so a KEY may
 * overload this to apply both at once. */
template <class KEY, class V>
VP_HD void vp_wr2(KEY *o, uint32_t i0, V v0, uint32_t i1, V v1, vp_stats *st)
{
	o->wr(i0, v0, st);
	o->wr(i1, v1, st);
}

/* 4-way select on the two low bits of k */
#define VP_SEL4(k, a, b, c, d) (((k) & 2) ? (((k) & 1) ? (d) : (c)) : (((k) & 1) ? (b) : (a)))

/* AES2 with round keys key[base + off .. + 3], then MIX2. A function so the
 * warp path can overload it. */
template <class KEY>
VP_HD void vp_aes2_mix2(vp128 &s0, vp128 &s1, const KEY *o, uint32_t base,
                        uint32_t off, const uint32_t *T0, vp_stats *st)
{
	s0 = vp_aesenc(s0, VP_RC(base + off + 0), T0);
	s1 = vp_aesenc(s1, VP_RC(base + off + 1), T0);
	s0 = vp_aesenc(s0, VP_RC(base + off + 2), T0);
	s1 = vp_aesenc(s1, VP_RC(base + off + 3), T0);
	VP_STAT(st, if (base + off + 3 >= 512) st->rc_cross++);
	const vp128 t = vp_unpacklo(s0, s1);
	s1 = vp_unpackhi(s0, s1);
	s0 = t;
}
#define VP_AES2_MIX2(s0, s1, base, off) vp_aes2_mix2(s0, s1, o, (base), (off), T0, st)

/* ------------------------------------------------------------------------- */
/* verusclhashv2_2                                                            */
/* ------------------------------------------------------------------------- */
VP_TPL
template <class KEY>
VP_HD uint64_t vp_clhash(KEY *o, const uint32_t cb[16], const uint32_t *T0,
                         vp_stats *st)
{
	typedef typename KEY::V V;
	const V *tag = 0;
	const V b0 = vp_loadv(cb, tag), b1 = vp_loadv(cb + 4, tag),
	            b2 = vp_loadv(cb + 8, tag), b3 = vp_loadv(cb + 12, tag);
	/* pbuf_copy[4] = { b0^b2, b1^b3, b2, b3 } */
	const V pc0 = vp_xor(b0, b2), pc1 = vp_xor(b1, b3);

	/* cases 4, 8 and 0xc (odd) multiply an unchanged pbuf entry by itself:
	 * those 4 products are computed once per hash */
	const V cq0 = vp_clmul_x10(pc0), cq1 = vp_clmul_x10(pc1),
	        cq2 = vp_clmul_x10(b2),  cq3 = vp_clmul_x10(b3);
#define VP_CQ(k) VP_SEL4(k, cq0, cq1, cq2, cq3)

	V acc = o->rd(VP_KEY_MASK + 2, st);    /* slot 513 */

	for (int i = 0; i < 32; i++) {
		const uint64_t selector = vp_lo(acc);
		const uint32_t prand_idx   = (uint32_t)(selector >> 5) & VP_KEY_MASK;
		const uint32_t prandex_idx = (uint32_t)(selector >> 32) & VP_KEY_MASK;

		/* pbuf = pbuf_copy + (selector & 3); pbufx = pbuf - 1 if bit 0, else
		 * + 1 (stays in 0..3). Selects, not an indexed array (local memory). */
		const uint32_t ip = (uint32_t)selector & 3;
		const uint32_t ix = (selector & 1) ? ip - 1 : ip + 1;
		const V pb0 = VP_SEL4(ip, pc0, pc1, b2, b3);
		const V pbx = VP_SEL4(ix, pc0, pc1, b2, b3);

		const V kprand   = o->rd(prand_idx, st);
		const V kprandex = o->rd(prandex_idx, st);
		o->begin(i, prand_idx, prandex_idx, kprand, kprandex);

		VP_STAT(st, st->cases[(selector & 0x1c) >> 2]++);
		VP_STAT(st, if (prand_idx == prandex_idx) st->collide++);

		/* the two key writes of this iteration, applied together after the case */
		uint32_t wi0 = 0, wi1 = 0;
		V wv0 = acc, wv1 = acc;
		switch (selector & 0x1c) {
		case 0: {
			const V temp1 = kprandex;
			const V add1 = vp_xor(temp1, pbx);
			acc = vp_xor(vp_clmul_x10(add1), acc);
			const V tempa2 = vp_xor(vp_mulhrs(acc, temp1), temp1);
			const V temp12 = kprand;
			wi0 = prand_idx; wv0 = tempa2;
			const V add12 = vp_xor(temp12, pb0);
			acc = vp_xor(vp_clmul_x10(add12), acc);
			const V tempb2 = vp_xor(vp_mulhrs(acc, temp12), temp12);
			wi1 = prandex_idx; wv1 = tempb2;
			break;
		}
		case 4: {
			const V temp1 = kprand;
			const V temp2 = pb0;
			const V add1 = vp_xor(temp1, temp2);
			acc = vp_xor(vp_clmul_x10(add1), acc);
			acc = vp_xor(VP_CQ(ip), acc);                    /* clmul(temp2 = pb0) */
			const V tempa2 = vp_xor(vp_mulhrs(acc, temp1), temp1);
			const V temp12 = kprandex;
			wi0 = prandex_idx; wv0 = tempa2;
			const V add12 = vp_xor(temp12, pbx);
			acc = vp_xor(add12, acc);
			wi1 = prand_idx; wv1 = vp_xor(vp_mulhrs(acc, temp12), temp12);
			break;
		}
		case 8: {
			const V temp1 = kprandex;
			const V add1 = vp_xor(temp1, pb0);
			acc = vp_xor(add1, acc);
			const V tempa2 = vp_xor(vp_mulhrs(acc, temp1), temp1);
			const V temp12 = kprand;
			wi0 = prand_idx; wv0 = tempa2;
			const V temp22 = pbx;
			const V add12 = vp_xor(temp12, temp22);
			acc = vp_xor(vp_clmul_x10(add12), acc);
			acc = vp_xor(VP_CQ(ix), acc);                    /* clmul(temp22 = pbx) */
			const V tempb2 = vp_xor(vp_mulhrs(acc, temp12), temp12);
			wi1 = prandex_idx; wv1 = tempb2;
			break;
		}
		case 0xc: {
			const V temp1 = kprand;
			const V add1 = vp_xor(temp1, pbx);
			const int32_t divisor = (int32_t)(uint32_t)selector;   /* never 0 */
			acc = vp_xor(add1, acc);
			const int64_t dividend = (int64_t)vp_lo(acc);
			VP_STAT(st, if (dividend < 0) st->negdiv++);
			acc = vp_xor(vp_mod_v(dividend, divisor, tag), acc);
			const V tempa2 = vp_xor(vp_mulhrs(acc, temp1), temp1);
			if (dividend & 1) {
				const V temp12 = kprandex;
				wi0 = prandex_idx; wv0 = tempa2;
				const V temp22 = pb0;
				const V add12 = vp_xor(temp12, temp22);
				acc = vp_xor(vp_clmul_x10(add12), acc);
				acc = vp_xor(VP_CQ(ip), acc);                /* clmul(temp22 = pb0) */
				const V tempb2 = vp_xor(vp_mulhrs(acc, temp12), temp12);
				wi1 = prand_idx; wv1 = tempb2;
			} else {
#if defined(VP_FAULT_STORE_ORDER)
				wi0 = prandex_idx; wv0 = tempa2;
				wi1 = prand_idx; wv1 = kprandex;
#else
				wi0 = prand_idx; wv0 = kprandex;
				wi1 = prandex_idx; wv1 = tempa2;
#endif
				acc = vp_xor(pb0, acc);
			}
			break;
		}
		case 0x10: {
			/* rc = prand: three AES2+MIX2 with keys rc[0..11] */
			V temp1 = pbx, temp2 = pb0;
			VP_AES2_MIX2(temp1, temp2, prand_idx, 0);
			VP_AES2_MIX2(temp1, temp2, prand_idx, 4);
			VP_AES2_MIX2(temp1, temp2, prand_idx, 8);
			acc = vp_xor(temp2, vp_xor(temp1, acc));
			const V tempa1 = kprand;
			const V tempa2 = vp_mulhrs(acc, tempa1);
			wi0 = prand_idx; wv0 = kprandex;
			wi1 = prandex_idx; wv1 = vp_xor(tempa1, tempa2);
			break;
		}
		case 0x14: {
			/* monkins loop: rounds+1 = 1..8 passes, rc advancing */
			uint64_t rounds = selector >> 61;
			uint32_t rc = prand_idx;
			uint32_t aesroundoffset = 0;
			VP_STAT(st, st->monk_rounds[0][rounds]++);
			do {
				if (selector & (((uint64_t)0x10000000) << rounds)) {
					VP_STAT(st, st->monk_branch[0][0]++);
					const V temp2 = (rounds & 1) ? pb0 : pbx;
					const V add1 = vp_xor(VP_RC(rc), temp2); rc++;
					acc = vp_xor(vp_clmul_x10(add1), acc);
				} else {
					VP_STAT(st, st->monk_branch[0][1]++);
					V onekey = VP_RC(rc); rc++;
					V temp2 = (rounds & 1) ? pbx : pb0;
					VP_AES2_MIX2(onekey, temp2, rc, aesroundoffset);
					aesroundoffset += 4;
					acc = vp_xor(onekey, acc);
					acc = vp_xor(temp2, acc);
				}
			} while (rounds--);
			const V tempa1 = kprand;
			const V tempa3 = vp_xor(tempa1, vp_mulhrs(acc, tempa1));
			const V tempa4 = kprandex;
			wi0 = prandex_idx; wv0 = tempa3;
			wi1 = prand_idx; wv1 = tempa4;
			break;
		}
		case 0x18: {
			uint64_t rounds = selector >> 61;
			uint32_t rc = prand_idx;
			V onekey;
			VP_STAT(st, st->monk_rounds[1][rounds]++);
			do {
				if (selector & (((uint64_t)0x10000000) << rounds)) {
					VP_STAT(st, st->monk_branch[1][0]++);
					const V temp2 = (rounds & 1) ? pb0 : pbx;
					onekey = vp_xor(VP_RC(rc), temp2); rc++;
					const int32_t divisor = (int32_t)(uint32_t)selector;
					const int64_t dividend = (int64_t)vp_lo(onekey);
					VP_STAT(st, if (dividend < 0) st->negdiv++);
					acc = vp_xor(vp_mod_v(dividend, divisor, tag), acc);
				} else {
					VP_STAT(st, st->monk_branch[1][1]++);
					const V temp2 = (rounds & 1) ? pbx : pb0;
					const V add1 = vp_xor(VP_RC(rc), temp2); rc++;
					onekey = vp_clmul_x10(add1);
					acc = vp_xor(vp_mulhrs(acc, onekey), acc);
				}
			} while (rounds--);
			VP_STAT(st, if (rc - 1 >= 512) st->rc_cross++);
			const V tempa3 = kprandex;
			wi0 = prandex_idx; wv0 = onekey;
			wi1 = prand_idx; wv1 = vp_xor(tempa3, acc);
			break;
		}
		case 0x1c: {
			const V temp2 = kprandex;
			const V add1 = vp_xor(pb0, temp2);
			acc = vp_xor(vp_clmul_x10(add1), acc);
			const V tempa2 = vp_xor(vp_mulhrs(acc, temp2), temp2);
			const V tempa3 = kprand;
			wi0 = prand_idx; wv0 = tempa2;
			acc = vp_xor(tempa3, acc);
			acc = vp_xor(pbx, acc);
			wi1 = prandex_idx; wv1 = vp_xor(vp_mulhrs(acc, tempa3), tempa3);
			break;
		}
		}
		vp_wr2(o, wi0, wv0, wi1, wv1, st);
	}
	VP_STAT(st, if (o->dirty() > st->ovl_max) st->ovl_max = o->dirty());
	return vp_finalize(acc);
}

/* ------------------------------------------------------------------------- */
/* Keyed Haraka512 with round constants key[base .. base+39] (mutated key)    */
/* ------------------------------------------------------------------------- */

/* round-constant reads */
#if defined(VP_FAULT_HARAKA_PRISTINE)
#define VP_HK(i) (o->pristine(i))
#else
#define VP_HK(i) o->rd((i), st)
#endif

/* VP_AES4: two AES rounds on each of the 4 states; VP_MIX4: the word mix that
 * ends a Haraka512 round */
#define VP_AES4(s, base, rci) do { \
	s[0] = vp_aesenc(s[0], VP_HK((base) + (rci) + 0), T0); \
	s[1] = vp_aesenc(s[1], VP_HK((base) + (rci) + 1), T0); \
	s[2] = vp_aesenc(s[2], VP_HK((base) + (rci) + 2), T0); \
	s[3] = vp_aesenc(s[3], VP_HK((base) + (rci) + 3), T0); \
	s[0] = vp_aesenc(s[0], VP_HK((base) + (rci) + 4), T0); \
	s[1] = vp_aesenc(s[1], VP_HK((base) + (rci) + 5), T0); \
	s[2] = vp_aesenc(s[2], VP_HK((base) + (rci) + 6), T0); \
	s[3] = vp_aesenc(s[3], VP_HK((base) + (rci) + 7), T0); \
} while (0)

#define VP_MIX4(s) do { \
	const auto _tmp = vp_unpacklo(s[0], s[1]); \
	s[0] = vp_unpackhi(s[0], s[1]); \
	s[1] = vp_unpacklo(s[2], s[3]); \
	s[2] = vp_unpackhi(s[2], s[3]); \
	s[3] = vp_unpacklo(s[0], s[2]); \
	s[0] = vp_unpackhi(s[0], s[2]); \
	s[2] = vp_unpackhi(s[1], _tmp); \
	s[1] = vp_unpacklo(s[1], _tmp); \
} while (0)

/* Full digest (haraka512_port_keyed): 5 rounds, feed-forward, truncation to
 * bytes 8..15, 24..31, 32..39, 48..55 of the 64-byte state. */
VP_TPL
template <class KEY>
VP_HD void vp_haraka512_keyed_full(uint8_t out[32], const uint32_t in[16],
                                   const KEY *o, uint32_t base,
                                   const uint32_t *T0, vp_stats *st)
{
	typedef typename KEY::V V;
	const V *tag = 0;
	V s[4];
	for (int k = 0; k < 4; k++) s[k] = vp_loadv(in + 4 * k, tag);
	for (int r = 0; r < 5; r++) {
		VP_AES4(s, base, 8 * r);
		VP_MIX4(s);
	}
	for (int k = 0; k < 4; k++) s[k] = vp_xor(s[k], vp_loadv(in + 4 * k, tag));
	uint8_t buf[64];
	for (int k = 0; k < 4; k++) vp_store(buf + 16 * k, vp_gather(s[k]));
	for (int k = 0; k < 8; k++) {
		out[k]      = buf[8 + k];  out[8 + k]  = buf[24 + k];
		out[16 + k] = buf[32 + k]; out[24 + k] = buf[48 + k];
	}
}

/* Truncated digest word 7 (haraka512_keyed): 4 full rounds with MIX4 after
 * the first 3, MIX4_LAST, then the two AES of round 5 that reach s[2]. */
VP_TPL
template <class KEY>
VP_HD uint32_t vp_haraka512_keyed_w7(const uint32_t in[16], const KEY *o,
                                     uint32_t base, const uint32_t *T0,
                                     vp_stats *st)
{
	typedef typename KEY::V V;
	const V *tag = 0;
	V s[4];
	for (int k = 0; k < 4; k++) s[k] = vp_loadv(in + 4 * k, tag);
	for (int r = 0; r < 3; r++) {
		VP_AES4(s, base, 8 * r);
		VP_MIX4(s);
	}
	VP_AES4(s, base, 24);
	/* MIX4_LAST */
	const V tmp = vp_unpacklo(s[0], s[1]);
	s[1] = vp_unpacklo(s[2], s[3]);
	s[2] = vp_unpackhi(s[1], tmp);
	/* AES4_LAST at 32: s2 with rc[34], rc[38] */
	s[2] = vp_aesenc(s[2], VP_HK(base + 34), T0);
	s[2] = vp_aesenc(s[2], VP_HK(base + 38), T0);
	/* ((u32*)&s[0])[10] ^ ((u32*)&in[52])[0] */
	return vp_gather(s[2]).w[2] ^ in[13];
}

/* ------------------------------------------------------------------------- */
/* One nonce from curBuf as 16 little-endian words: half[0..63] with the nonce
 * over bytes 32..46. cb is modified. full: out[0..31], else only out[28..31].
 * The key is pristine again on return. Returns the clhash intermediate.
 * vp_hash_tail is the part after clhash, for callers that rebuild cb first.  */
/* ------------------------------------------------------------------------- */
VP_TPL
template <class KEY>
VP_HD void vp_hash_tail(uint8_t out[32], int full, uint32_t cb[16], uint64_t inter,
                        KEY *o, const uint32_t *T0, vp_stats *st)
{

	/* fill2: byte 47 = byte 0 of inter, bytes 48..63 = bytes 1..7,0,1..7,0 */
	cb[11] = (cb[11] & 0x00ffffffu) | (uint32_t)(inter & 0xff) << 24;
	cb[12] = (uint32_t)(inter >> 8);
	cb[13] = (uint32_t)(inter >> 40) | (uint32_t)(inter & 0xff) << 24;
	cb[14] = cb[12];
	cb[15] = cb[13];

	const uint32_t base = (uint32_t)(inter & VP_KEY_MASK);
	VP_STAT(st, if (base > 472) st->haraka_tail++);
	if (full) {
		vp_haraka512_keyed_full(out, cb, o, base, T0, st);
	} else {
		const uint32_t w7 = vp_haraka512_keyed_w7(cb, o, base, T0, st);
		out[28] = (uint8_t)w7;         out[29] = (uint8_t)(w7 >> 8);
		out[30] = (uint8_t)(w7 >> 16); out[31] = (uint8_t)(w7 >> 24);
	}
	o->end_hash();
}

VP_TPL
template <class KEY>
VP_HD uint64_t vp_hash_cb(uint8_t out[32], int full, uint32_t cb[16], KEY *o,
                          const uint32_t *T0, vp_stats *st)
{
	const uint64_t inter = vp_clhash(o, cb, T0, st);
	vp_hash_tail(out, full, cb, inter, o, T0, st);
	return inter;
}

/* ------------------------------------------------------------------------- */
/* One nonce from bytes. half = VerusHashHalf output (job-constant), nonce15 =
 * the last 15 preimage bytes.                                                 */
/* ------------------------------------------------------------------------- */
VP_TPL
template <class KEY>
VP_HD uint64_t vp_hash(uint8_t out[32], int full, const uint8_t half[64],
                       const uint8_t nonce15[15], KEY *o, const uint32_t *T0,
                       vp_stats *st)
{
	/* curBuf as 16 little-endian words: half[0..63], the nonce over bytes 32..46 */
	uint32_t cb[16];
	for (int k = 0; k < 16; k++) {
		uint32_t w = 0;
		for (int b = 0; b < 4; b++) {
			const int p = 4 * k + b;
			const uint8_t v = (p >= 32 && p < 47) ? nonce15[p - 32] : half[p];
			w |= (uint32_t)v << (8 * b);
		}
		cb[k] = w;
	}
	return vp_hash_cb(out, full, cb, o, T0, st);
}

/* Host only: the AES T0 table, derived from the field definition. */
static uint8_t vp_gmul(uint8_t a, uint8_t b)
{
	uint8_t p = 0;
	while (b) {
		if (b & 1) p ^= a;
		a = (uint8_t)((a << 1) ^ ((a & 0x80) ? 0x1b : 0));
		b >>= 1;
	}
	return p;
}
static void vp_build_t0(uint32_t T0[256])
{
	for (int x = 0; x < 256; x++) {
		uint8_t inv = 0;
		if (x) {                            /* x^254 = x^-1 in GF(2^8) */
			uint8_t r = 1, b = (uint8_t)x;
			for (int e = 254; e; e >>= 1) {
				if (e & 1) r = vp_gmul(r, b);
				b = vp_gmul(b, b);
			}
			inv = r;
		}
		uint8_t s = inv;
		for (int k = 1; k <= 4; k++)
			s ^= (uint8_t)((inv << k) | (inv >> (8 - k)));
		s ^= 0x63;
		const uint8_t s2 = vp_gmul(s, 2), s3 = vp_gmul(s, 3);
		T0[x] = (uint32_t)s2 | (uint32_t)s << 8 | (uint32_t)s << 16 |
		        (uint32_t)s3 << 24;
	}
}

#endif /* VERUS_PORTABLE_H */
