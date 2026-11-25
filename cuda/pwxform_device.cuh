/* pwxform for yespower 1.0 -- shared device header (coding guideline section 3).
 *
 * Transcribed from the normative reference, sph/yespower_ref.c:pwxform(), NOT
 * from the optimized path.  Constants for 1.0: PWXrounds 3, PWXgather 4,
 * PWXsimple 2, Swidth 11 -> Smask 0x7FF0, three 32 KiB boxes.
 *
 * THREAD MAPPING: one PWXgather lane per thread, so a thread owns the whole
 * 16-byte X[j] and the gather -> multiply -> xor chain needs no shuffles.
 *
 * WRITE-BACK.  A call writes 8 x 16 B -- all four lanes in round 0, then only
 * lanes 0 and 1 in rounds 1 and 2 (`i == 0 || j < PWXgather/2`) -- into S0 and S1,
 * the two boxes being gathered from, not the resting third box.  `w` advances
 * 32/16/16 bytes across the three rounds and only on odd lanes; the boxes rotate
 * once, at the end.
 *
 * Because it writes where it gathers, THE FOUR LANES ARE A SEQUENCE: lane j must
 * see the writes of lanes < j and not those of lanes > j.  A lane only notices
 * when its gather index hits one of the <= 4 slots the round writes (~0.2%), so
 * a short test passes with it broken.  Handled by testing for the collision
 * before writing and falling back to an ordered round. */

#ifndef CUDA_PWXFORM_DEVICE_CUH
#define CUDA_PWXFORM_DEVICE_CUH

#include <stdint.h>
#include "salsa_device.cuh"

/* S-box placement.  yespower 1.0 fixes Swidth = 11, so S is 98304 B per
 * instance for every variant and every (N, r).
 *
 *   PWX_S_SHARED  S in shared memory, which pins the SM to one instance.
 *   PWX_S_GLOBAL  S in a global arena read with `.cg`.  Slower per access, but
 *                 it lifts the one-block-per-SM cap and the occupancy more than
 *                 pays for it; this is the path that ships.
 *
 * `.cg` not `.ca` on purpose: the resident instances hold far more table than
 * L1 can keep, so caching it only thrashes. */
enum { PWX_S_SHARED = 0, PWX_S_GLOBAL = 1 };

template<int PLACE, typename V> struct PwxS;
template<typename V> struct PwxS<PWX_S_SHARED, V> {
	static __device__ __forceinline__ V    ld(const V *p)      { return *p; }
	static __device__ __forceinline__ void st(V *p, const V v) { *p = v;    }
};
template<typename V> struct PwxS<PWX_S_GLOBAL, V> {
	static __device__ __forceinline__ V    ld(const V *p)      { return __ldcg(p); }
	static __device__ __forceinline__ void st(V *p, const V v) { __stcg(p, v);     }
};

/* yespower 1.0 shape, in uint4 (16-byte) units unless noted. */
#define PWX_SMASK        0x7FF0u   /* ((1 << Swidth) - 1) * PWXsimple * 8       */
#define PWX_BOX_ENTRIES  2048u     /* 32 KiB / 16 B                             */
#define PWX_W_MASK       2047u     /* w wraps within one box                    */
#define PWX_ROUNDS       3
#define PWX_GATHER       4

/* Test only: always take the parallel round (the lane-forwarding defect), to
 * show the KAT catches it. No build defines it. */
#ifdef YP_PWX_FAULT_NO_LANE_FORWARDING
#define YP_PWX_NEED_SEQ(mask, hz) (false)
#else
#define YP_PWX_NEED_SEQ(mask, hz) __any_sync((mask), (hz))
#endif

/* Test only: drop the intra-round barriers -- a real race, so the digest varies
 * run to run (unlike the stable wrong digest above). No build defines it. */
#ifdef YP_PWX_FAULT_NO_SYNCWARP
#define YP_PWX_SYNC(mask) ((void)0)
#else
#define YP_PWX_SYNC(mask) __syncwarp(mask)
#endif

/* One lane's round: gather from S0/S1, multiply-add-xor, then the conditional
 * write-back.  Factored out so the ordered and the parallel round below are the
 * SAME code -- a divergence between them would be invisible until a hazard. */
template<int PLACE = PWX_S_SHARED>
__device__ __forceinline__ void pwx_lane_round(uint4 &X, uint4 *S,
                                               const uint32_t b0, const uint32_t b1,
                                               const uint32_t w4,
                                               const int i, const int j)
{
	/* The gather address comes from the LOW sub-lane only, and is captured
	 * before the state is overwritten (reference: xl/xh read from X[j][0]
	 * outside the k loop). */
	const uint32_t e0 = (X.x & PWX_SMASK) >> 4;
	const uint32_t e1 = (X.y & PWX_SMASK) >> 4;

	const uint4 s0 = PwxS<PLACE, uint4>::ld(&S[b0 + e0]);
	const uint4 s1 = PwxS<PLACE, uint4>::ld(&S[b1 + e1]);

	/* PWXsimple = 2 independent 64-bit sub-lanes:
	 *   x = hi32 * lo32 + S0_k, then xor S1_k                            */
	uint64_t x0 = (uint64_t)X.y * (uint64_t)X.x;
	uint64_t x1 = (uint64_t)X.w * (uint64_t)X.z;
	x0 += ((uint64_t)s0.y << 32) | s0.x;
	x1 += ((uint64_t)s0.w << 32) | s0.z;
	x0 ^= ((uint64_t)s1.y << 32) | s1.x;
	x1 ^= ((uint64_t)s1.w << 32) | s1.z;

	X = make_uint4((uint32_t)x0, (uint32_t)(x0 >> 32),
	               (uint32_t)x1, (uint32_t)(x1 >> 32));

	/* Write-back: every lane in round 0, lanes 0 and 1 afterwards. */
	if ((i == 0) || (j < PWX_GATHER / 2)) {
		const uint32_t off = w4 + ((i == 0) ? (uint32_t)(j >> 1) : 0u);
		PwxS<PLACE, uint4>::st(&S[((j & 1) ? b1 : b0) + (off & PWX_W_MASK)], X);
	}
}

/* One pwxform call on this thread's lane.
 *
 *   X      this lane's 16 bytes of state, updated in place
 *   S      the 96 KiB box array as uint4[], 3 boxes of PWX_BOX_ENTRIES
 *   b0,b1,b2  box bases in uint4 units; rotated on return
 *   w4     write cursor in uint4 units; advanced and wrapped on return
 *   j      this thread's PWXgather lane, 0..3
 *   mask   __syncwarp participation mask for the 4 cooperating threads
 *
 * b0/b1/b2/w4 evolve identically in every thread without communication -- the
 * reference's j loop is sequential, but its effect on `w` is a closed form:
 * in round 0 lane j writes at w4 + (j >> 1) and w4 then advances 2; in rounds
 * 1 and 2 lanes 0 and 1 both write at w4 and it advances 1.
 */
template<int PLACE = PWX_S_SHARED>
__device__ __forceinline__ void pwxform_1_0(uint4 &X, uint4 *S,
                                            uint32_t &b0, uint32_t &b1,
                                            uint32_t &b2, uint32_t &w4,
                                            const int j, const unsigned mask)
{
#pragma unroll
	for (int i = 0; i < PWX_ROUNDS; i++) {
		/* The lanes are sequential, not parallel: the reference writes into
		 * S0/S1, the boxes it gathers from, so lane j must see the writes of
		 * lanes < j in this round and not those of lanes > j.  Rare and
		 * data-dependent (only when a gather index lands on one of the <= 4 slots
		 * the round writes), so it hides from short tests.
		 *
		 * Decidable before anything is written -- gather indices from X, written
		 * slots from w4 -- so test first and order only when it matters (~2%). */
		const uint32_t Wa = w4;
		const uint32_t Wb = (w4 + 1u) & PWX_W_MASK;   /* round 0 writes 2 slots */
		const uint32_t e0 = (X.x & PWX_SMASK) >> 4;
		const uint32_t e1 = (X.y & PWX_SMASK) >> 4;
		/* Deliberately conservative: this also flags a lane colliding with its
		 * own write, and lanes that write nothing.  Over-detection only costs
		 * speed -- the ordered path is correct for every case. */
		bool hz = (e0 == Wa) || (e1 == Wa);
		if (i == 0) hz |= (e0 == Wb) || (e1 == Wb);

		if (YP_PWX_NEED_SEQ(mask, hz)) {
			for (int t = 0; t < PWX_GATHER; t++) {
				if (j == t) pwx_lane_round<PLACE>(X, S, b0, b1, w4, i, j);
				YP_PWX_SYNC(mask);
			}
		} else {
			pwx_lane_round<PLACE>(X, S, b0, b1, w4, i, j);
			/* The next round gathers from boxes this round just wrote. */
			YP_PWX_SYNC(mask);
		}

		w4 = (w4 + ((i == 0) ? 2u : 1u)) & PWX_W_MASK;
	}

	/* (S0, S1, S2) <- (S2, S0, S1), once per call. */
	const uint32_t t = b2; b2 = b1; b1 = b0; b0 = t;
}

/* blockmix_pwxform<R>(B, ...) -- reference blockmix_pwxform(), r = R.
 *
 *   B  128*R bytes, i.e. 2R blocks of 64 bytes, in the SIMD-shuffled word order
 *   j  this thread's PWXgather lane, 0..3
 *
 * With PWXbytes = 64 the reference's r1 is exactly 2R, so this is 2R chained
 * pwxform calls followed by ONE Salsa20 on the final 64-byte block.  The
 * reference's trailing `for (i++; i < 2r; i++)` loop is dead at these settings
 * (i starts at 2r) -- it is kept there defensively and is intentionally absent
 * here; if PWXbytes ever changed, this would need it back.
 *
 * Two lane layouts meet here, and mixing them silently corrupts the last block:
 * pwxform's thread j owns the contiguous words 4j..4j+3 of a 64-byte block, while
 * salsa_core's lane L owns the strided words L, L+4, L+8, L+12.  Instead of
 * transposing registers across lanes, the final block is re-read from B with the
 * strided pattern -- one 64-byte round trip per blockmix instead of four
 * shuffles, and salsa_core stays exactly as verified.
 */
template<uint32_t R, int PLACE = PWX_S_SHARED>
__device__ __forceinline__ void blockmix_pwxform(uint32_t *B, uint4 *S,
                                                 uint32_t &b0, uint32_t &b1,
                                                 uint32_t &b2, uint32_t &w4,
                                                 const int j, const unsigned mask)
{
	const uint32_t r1 = 2u * R;              /* 128*R / PWXbytes */
	uint4 *B4 = (uint4 *)B;

	/* X <- B'_{r1-1} */
	uint4 X = B4[(r1 - 1) * 4 + j];

/* Unroll factor for the 2R pwxform calls; unrolling measured no gain (the
 * loop is dependency-bound). */
#ifndef YP_BMX_UNROLL
#define YP_BMX_UNROLL 1
#endif
/* _Pragma + stringify: `#pragma unroll MACRO` is NOT reliably macro-expanded. */
#define YP_STR2(x) #x
#define YP_STR(x)  YP_STR2(x)
#define YP_UNROLL_N(n) _Pragma(YP_STR(unroll n))
	YP_UNROLL_N(YP_BMX_UNROLL)
	for (uint32_t i = 0; i < r1; i++) {
		if (r1 > 1) {                        /* X <- X xor B'_i */
			const uint4 b = B4[i * 4 + j];
			X.x ^= b.x; X.y ^= b.y; X.z ^= b.z; X.w ^= b.w;
		}
		pwxform_1_0<PLACE>(X, S, b0, b1, b2, w4, j, mask);
		B4[i * 4 + j] = X;                   /* B'_i <- X */
	}

	/* B_{2R-1} <- H(B_{2R-1}), Salsa20/2 for yespower 1.0.
	 * The __syncwarp matters: below, each lane reads words written by a
	 * DIFFERENT lane above (word j+4m came from lane m). */
	__syncwarp(mask);
	uint32_t *L = B + (r1 - 1) * 16;
	uint32_t x0 = L[j], x1 = L[j + 4], x2 = L[j + 8], x3 = L[j + 12];
	salsa_core<2>(x0, x1, x2, x3, mask);
	L[j] = x0; L[j + 4] = x1; L[j + 8] = x2; L[j + 12] = x3;
	__syncwarp(mask);
}

#endif /* CUDA_PWXFORM_DEVICE_CUH */
