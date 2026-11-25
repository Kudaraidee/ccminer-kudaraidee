/* VerusHash 2.2: one hash per WARP. Device only.
 *
 * Why a warp: with 32 independent hashes a warp would run every one of the 8
 * data-dependent cases each iteration. With one hash per warp all branches are
 * uniform and the lanes split the primitives instead.
 *
 * Value layout (vl): a 128-bit value lives as one 32-bit word per lane, word
 * q = lane & 3, replicated in all 8 groups of 4 lanes. Every op keeps the
 * replicas identical, so any lane can serve as the source of a broadcast.
 *
 * The hash body lives once, in verus_portable.h (templates over KEY::V). This
 * file only provides the vl overloads and the warp key (vw_key, the
 * copy-on-write overlay in shared memory).
 *
 * Requires blockDim.x to be a multiple of 32 and all 32 lanes of a warp to
 * call every function together (no lane may exit early).
 */
#ifndef VERUS_WARP_CUH
#define VERUS_WARP_CUH

#include "verus_portable.h"

#define VW_FULL 0xffffffffu

/* AES tables in shared memory:
 *   1 table   T0 only, rows 1..3 rotated at run time
 *   4 tables  T0..T3 = T0 rotated by 0/8/16/24 (4 KB), no rotates
 * 4 from sm_70 up; on sm_61 the extra 3 KB per block would cost a resident
 * block. */
#ifndef VW_AES_TABLES
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define VW_AES_TABLES 1
#else
#define VW_AES_TABLES 4
#endif
#define VW_AES_TABLES_BY_ARCH 1
#endif
#define VW_T_WORDS (256 * VW_AES_TABLES)

/* the host side of the same choice, for sizing shared memory */
static inline int vw_aes_tables(int cc_major)
{
#if defined(VW_AES_TABLES_BY_ARCH)
	return cc_major >= 7 ? 4 : 1;
#else
	(void)cc_major;
	return VW_AES_TABLES;   /* forced with -DVW_AES_TABLES=n */
#endif
}

__device__ __forceinline__ uint32_t vw_lane() { return threadIdx.x & 31; }
__device__ __forceinline__ uint32_t vw_q()    { return threadIdx.x & 3; }
__device__ __forceinline__ uint32_t vw_base() { return threadIdx.x & 28; }

typedef struct { uint32_t w; } vl;

__device__ __forceinline__ vl vl_make(uint32_t w) { vl r; r.w = w; return r; }

/* word k (warp-uniform k) of v, in every lane */
__device__ __forceinline__ uint32_t vl_word(vl v, uint32_t k)
{
	return __shfl_sync(VW_FULL, v.w, vw_base() | k);
}

/* ---- overloads the templates in verus_portable.h resolve to ---- */

__device__ __forceinline__ vl vp_xor(vl a, vl b) { return vl_make(a.w ^ b.w); }

__device__ __forceinline__ uint64_t vp_lo(vl v)
{
	return (uint64_t)vl_word(v, 1) << 32 | vl_word(v, 0);
}

__device__ __forceinline__ vp128 vp_gather(vl v)
{
	vp128 r;
	r.w[0] = vl_word(v, 0); r.w[1] = vl_word(v, 1);
	r.w[2] = vl_word(v, 2); r.w[3] = vl_word(v, 3);
	return r;
}

/* curBuf is uniform; pick this lane's word without a runtime-indexed array */
__device__ __forceinline__ vl vp_loadv(const uint32_t *p, const vl *)
{
	const uint32_t q = vw_q();
	return vl_make(q == 0 ? p[0] : q == 1 ? p[1] : q == 2 ? p[2] : p[3]);
}

__device__ __forceinline__ vl vp_mod_v(int64_t d, int32_t dv, const vl *)
{
	const uint32_t m = vp_modulo(d, dv).w[0];
	return vl_make(vw_q() == 0 ? m : 0);
}

__device__ __forceinline__ uint64_t vp_finalize(vl acc) { return vp_finalize(vp_gather(acc)); }

__device__ __forceinline__ vl vp_mulhrs(vl a, vl b) { return vl_make(vp_mulhrs_w(a.w, b.w)); }

/* _mm_unpacklo_epi32: r = { a0, b0, a1, b1 }; word q comes from word q>>1 of
 * a (q even) or b (q odd). A source lane can send one register per shuffle, so
 * two shuffles. */
__device__ __forceinline__ vl vp_unpacklo(vl a, vl b)
{
	const uint32_t src = vw_base() | (vw_q() >> 1);
	const uint32_t x = __shfl_sync(VW_FULL, a.w, src);
	const uint32_t y = __shfl_sync(VW_FULL, b.w, src);
	return vl_make((vw_q() & 1) ? y : x);
}
/* _mm_unpackhi_epi32: r = { a2, b2, a3, b3 } */
__device__ __forceinline__ vl vp_unpackhi(vl a, vl b)
{
	const uint32_t src = vw_base() | (2 + (vw_q() >> 1));
	const uint32_t x = __shfl_sync(VW_FULL, a.w, src);
	const uint32_t y = __shfl_sync(VW_FULL, b.w, src);
	return vl_make((vw_q() & 1) ? y : x);
}

/* aesenc: lane q computes output column q from words q, q+1, q+2, q+3 (rows
 * 0..3 after ShiftRows): 3 shuffles and 4 table lookups. The lookups use byte
 * offsets, (w >> s) & 0x3fc. */
__device__ __forceinline__ vl vp_aesenc(vl s, vl rk, const uint32_t *T0)
{
	const uint32_t q = vw_q(), b = vw_base();
	const uint32_t w1 = __shfl_sync(VW_FULL, s.w, b | ((q + VP_SR1) & 3));
	const uint32_t w2 = __shfl_sync(VW_FULL, s.w, b | ((q + 2) & 3));
	const uint32_t w3 = __shfl_sync(VW_FULL, s.w, b | ((q + VP_SR3) & 3));
	const char *tb = (const char *)T0;
#define VW_TB(off) (*(const uint32_t *)(tb + (off)))
#if VW_AES_TABLES == 4
	return vl_make(VW_TB((s.w << 2) & 0x3fc) ^ VW_TB(1024 + ((w1 >> 6) & 0x3fc)) ^
	               VW_TB(2048 + ((w2 >> 14) & 0x3fc)) ^ VW_TB(3072 + ((w3 >> 22) & 0x3fc)) ^ rk.w);
#else
	return vl_make(VW_TB((s.w << 2) & 0x3fc) ^ vp_rotl32(VW_TB((w1 >> 6) & 0x3fc), 8) ^
	               vp_rotl32(VW_TB((w2 >> 14) & 0x3fc), 16) ^ vp_rotl32(VW_TB((w3 >> 22) & 0x3fc), 24) ^
	               rk.w);
#endif
#undef VW_TB
}

/* clmul(x, x, 0x10) = a (x) m with a = x.lo, m = x.hi. Group g = lane >> 2
 * takes multiplier bits i = 8g..8g+7 and lane q keeps word q of its partial
 * product; three xor-shuffles across the groups sum the partials.
 * Word q of a << i is bits [32q - i, 32q - i + 32) of a (zero-extended), so for
 * the group's 8 consecutive i all 8 words are windows of ONE 64-bit value
 * X = bits [32q - 8g - 7, +64) of a: each bit costs a constant shift of X, a
 * mask and an xor. */
__device__ __forceinline__ vl vp_clmul_x10(vl x)
{
	/* a = x.lo (x.hi under the fault), and only byte g of m = x.hi */
	const uint32_t aw = VP_CLMUL_A(0u, 2u);
	const uint64_t a = (uint64_t)vl_word(x, aw + 1) << 32 | vl_word(x, aw);
	const uint32_t g = vw_lane() >> 2, q = vw_q();
	const uint32_t mw = __shfl_sync(VW_FULL, x.w, vw_base() | (2 + (g >> 2)));
	const int s0 = 32 * (int)q - 8 * (int)g - 7;   /* -63 .. 89 */
	const uint64_t X = s0 >= 64 ? 0 : s0 >= 0 ? a >> s0 : s0 > -64 ? a << -s0 : 0;
	const uint32_t mb = (mw >> (8 * (g & 3))) & 0xff;
	uint32_t w = 0;
#pragma unroll
	for (int t = 0; t < 8; t++) {
#if defined(VW_CLMUL_MASK)
		w ^= (uint32_t)(X >> (7 - t)) & (0u - ((mb >> t) & 1));
#else
		/* predicated xor, cheaper than building a mask */
		if (mb & (1u << t)) w ^= (uint32_t)(X >> (7 - t));
#endif
	}
	w ^= __shfl_xor_sync(VW_FULL, w, 4);
	w ^= __shfl_xor_sync(VW_FULL, w, 8);
	w ^= __shfl_xor_sync(VW_FULL, w, 16);
	return vl_make(w);
}

/* ---- the warp key: copy-on-write overlay in shared memory ----
 * Per warp:
 *   map[512]   0xff = clean, else the ov slot of the latest write
 *   idx[64]    the key slot each write went to, for the reset
 *   ov[64][4]  the written values
 * The overlay is an append-only log: write k of a hash takes ov slot k and
 * points map[i] at it, so the newest write wins and a write never reads the
 * map. A hash makes exactly 64 writes (VP_OVL_SLOTS); n is warp-uniform.
 *
 * Only lane 0 writes the map and only lanes 0..3 write ov; __syncwarp orders
 * each write against the reads around it. */
typedef struct vw_key {
	typedef vl V;
	const uint32_t *K;      /* pristine key as words, VP_KEY_SLOTS * 4 */
	uint8_t        *map;
	uint16_t       *idx;
	uint32_t       *ov;
	int             n;

	__device__ __forceinline__ vl rd(uint32_t i, vp_stats *) const
	{
		/* the pristine load does not wait for the map: most reads are clean */
		const uint32_t kv = K[i * 4 + vw_q()];
		const uint32_t m = i < 512 ? map[i] : 0xffu;
		return vl_make(m != 0xffu ? ov[m * 4 + vw_q()] : kv);
	}
	__device__ __forceinline__ void wr(uint32_t i, vl v, vp_stats *)
	{
		const uint32_t m = (uint32_t)n++;
		__syncwarp();      /* earlier reads of map[i] / ov complete first */
		if (vw_lane() == 0) { map[i] = (uint8_t)m; idx[m] = (uint16_t)i; }
		if (vw_lane() < 4) ov[m * 4 + vw_q()] = v.w;
		__syncwarp();
	}
	__device__ __forceinline__ void begin(int, uint32_t, uint32_t, vl, vl) { }
	__device__ __forceinline__ void end_hash()
	{
		/* other lanes' last map/ov reads of this hash complete before the reset */
		__syncwarp();
#if !defined(VP_FAULT_NO_RESET)
		for (int k = vw_lane(); k < n; k += 32) map[idx[k]] = 0xff;
#endif
		n = 0;
		__syncwarp();
	}
	__device__ __forceinline__ vl pristine(uint32_t i) const { return vl_make(K[i * 4 + vw_q()]); }
	__device__ __forceinline__ int dirty() const { return n; }
} vw_key;

/* both writes of an iteration under one pair of __syncwarp; map[i1] is
 * written after map[i0], so i0 == i1 keeps the second value */
__device__ __forceinline__ void vp_wr2(vw_key *o, uint32_t i0, vl v0, uint32_t i1, vl v1, vp_stats *)
{
	const uint32_t m = (uint32_t)o->n;
	o->n += 2;
	__syncwarp();      /* earlier reads of map / ov complete first */
	if (vw_lane() == 0) {
		o->map[i0] = (uint8_t)m;       o->idx[m] = (uint16_t)i0;
		o->map[i1] = (uint8_t)(m + 1); o->idx[m + 1] = (uint16_t)i1;
	}
	if (vw_lane() < 4) {
		o->ov[m * 4 + vw_q()] = v0.w;
		o->ov[(m + 1) * 4 + vw_q()] = v1.w;
	}
	__syncwarp();
}

__device__ __forceinline__ void vw_key_init(vw_key *o, const uint32_t *K, uint8_t *map,
                                            uint16_t *idx, uint32_t *ov)
{
	o->K = K; o->map = map; o->idx = idx; o->ov = ov; o->n = 0;
	for (int i = vw_lane(); i < 512; i += 32) map[i] = 0xff;
	__syncwarp();
}

__device__ __forceinline__ int vw_key_clean(const vw_key *o)
{
	int dirty = 0;
	for (int i = vw_lane(); i < 512; i += 32) dirty |= o->map[i] != 0xff;
	return !__any_sync(VW_FULL, dirty);
}

/* ---- parallel-state overloads: independent AES states on different lane
 * groups instead of one after another on every lane ---- */

/* round-key reads from the mutated key, as in verus_portable.h */
#if defined(VP_FAULT_RC_PRISTINE)
#define VW_RC(i) (o->pristine(i))
#else
#define VW_RC(i) o->rd((i), st)
#endif
#if defined(VP_FAULT_HARAKA_PRISTINE)
#define VW_HK(i) (o->pristine(i))
#else
#define VW_HK(i) o->rd((i), st)
#endif

/* AES2 + MIX2: s0 on even groups, s1 on odd groups (round keys base+off+h and
 * base+off+2+h, per lane), MIX2 is one shuffle per lane, and two shuffles
 * merge the pair back into the replicated layout. */
__device__ __forceinline__ void vp_aes2_mix2(vl &s0, vl &s1, const vw_key *o, uint32_t base,
                                             uint32_t off, const uint32_t *T0, vp_stats *st)
{
	const uint32_t g = vw_lane() >> 2, h = g & 1, q = vw_q(), pair = (g & ~1u) * 4;
	vl v = h ? s1 : s0;
	v = vp_aesenc(v, VW_RC(base + off + h), T0);
	v = vp_aesenc(v, VW_RC(base + off + 2 + h), T0);
	/* unpacklo -> s0' = {a0,b0,a1,b1}, unpackhi -> s1' = {a2,b2,a3,b3}:
	 * word q from state (q & 1), word (h ? 2 : 0) + (q >> 1) */
	v.w = __shfl_sync(VW_FULL, v.w, pair + (q & 1) * 4 + h * 2 + (q >> 1));
	s0.w = __shfl_sync(VW_FULL, v.w, pair + q);
	s1.w = __shfl_sync(VW_FULL, v.w, pair + 4 + q);
}

/* MIX4 as one permutation: new state k word q comes from lane nibble[4k+q]
 * (4 * source state + source word). MIX4_LAST's s2 equals MIX4's s2. */
#define VW_MIX4_SRC 0xe6a25d194c08f7b3ULL

/* Truncated keyed Haraka512, word 7: state k on group k (groups 4..7 repeat
 * 0..3), so each AES4 is two aesenc per lane instead of eight. */
__device__ __forceinline__ uint32_t vp_haraka512_keyed_w7(const uint32_t in[16], const vw_key *o,
                                                         uint32_t base, const uint32_t *T0,
                                                         vp_stats *st)
{
	const uint32_t lane = vw_lane(), k = (lane >> 2) & 3, j = lane & 15;
	uint32_t w = 0;
#pragma unroll
	for (int t = 0; t < 16; t++) w = j == (uint32_t)t ? in[t] : w;
	vl s = vl_make(w);
	const uint32_t src = (lane & 16) | (uint32_t)((VW_MIX4_SRC >> (4 * j)) & 15);
#pragma unroll
	for (int r = 0; r < 4; r++) {
		s = vp_aesenc(s, VW_HK(base + 8 * r + k), T0);
		s = vp_aesenc(s, VW_HK(base + 8 * r + 4 + k), T0);
		s.w = __shfl_sync(VW_FULL, s.w, src);    /* MIX4 (MIX4_LAST after r = 3) */
	}
	/* AES4_LAST: only s2 (group 2) */
	s = vp_aesenc(s, VW_HK(base + 34), T0);
	s = vp_aesenc(s, VW_HK(base + 38), T0);
	return __shfl_sync(VW_FULL, s.w, 2 * 4 + 2) ^ in[13];
}

/* shared bytes per warp: map + idx + ov */
#define VW_WARP_SMEM (512 + VP_OVL_SLOTS * 2 + VP_OVL_SLOTS * 16)

#endif /* VERUS_WARP_CUH */
