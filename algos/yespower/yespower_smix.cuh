/* yespower 1.0 SMix (smix1 / smix2), transcribed from sph/yespower_ref.c.
 *
 * pwxform lane j (0..3) owns uint4 index j (mod 4) of each 64-byte block, the
 * blockmix_pwxform layout.  With WIDTH > 4 the extra threads only move V; the
 * YP_WSYNC barriers order their accesses against blockmix.
 */

#ifndef YESPOWER_SMIX_CUH
#define YESPOWER_SMIX_CUH

#include "cuda/pwxform_device.cuh"   /* also pulls in salsa_device.cuh */

/* V access. YP_V_STREAM selects evict-first (.cs); off, it measured no gain. */
#ifdef YP_V_STREAM
__device__ __forceinline__ uint4 yp_vld(const uint4 *p) { return __ldcs(p); }
__device__ __forceinline__ void  yp_vst(uint4 *p, const uint4 v) { __stcs(p, v); }
#else
__device__ __forceinline__ uint4 yp_vld(const uint4 *p) { return *p; }
__device__ __forceinline__ void  yp_vst(uint4 *p, const uint4 v) { *p = v; }
#endif

/* Test only: YP_FAULT_NO_WIDTH_SYNC drops the WIDTH barriers.  Redundant at
 * WIDTH == 4; above it the digest must break.  No build defines it. */
#ifdef YP_FAULT_NO_WIDTH_SYNC
#define YP_WSYNC(m) ((void)0)
#else
#define YP_WSYNC(m) __syncwarp(m)
#endif

/* YP_VBATCH: V loads staged per thread before use. Off: no gain at width 32,
 * and each step costs 4 registers. */
#ifndef YP_VBATCH
#define YP_VBATCH 1
#endif

__device__ __forceinline__ uint32_t yp_p2floor(uint32_t x)
{
	uint32_t y;
	while ((y = x & (x - 1))) x = y;
	return x;
}

/* wrap(x, i): fold x into [0, i). */
__device__ __forceinline__ uint32_t yp_wrap(uint32_t x, uint32_t i)
{
	const uint32_t n = yp_p2floor(i);
	return (x & (n - 1)) + (i - n);
}

/* integerify(X, r) = X[(2r-1) * 16], first word of the last 64-byte block. */
template<uint32_t R>
__device__ __forceinline__ uint32_t yp_integerify(const uint32_t *X)
{
	return X[(2u * R - 1u) * 16u];
}

/* B <-> X SIMD shuffle per 64-byte block: X[k*16 + i] = B[k*16 + (i*5 % 16)].
 * Lane j handles i = j, j+4, j+8, j+12. */
__device__ __forceinline__ void yp_shuffle_in(uint32_t *X, const uint32_t *B,
                                              uint32_t blocks, int j)
{
	for (uint32_t k = 0; k < blocks; k++)
#pragma unroll
		for (int m = 0; m < 4; m++) {
			const int i = j + 4 * m;
			X[k * 16 + i] = B[k * 16 + (i * 5 % 16)];
		}
}

__device__ __forceinline__ void yp_shuffle_out(uint32_t *B, const uint32_t *X,
                                               uint32_t blocks, int j)
{
	for (uint32_t k = 0; k < blocks; k++)
#pragma unroll
		for (int m = 0; m < 4; m++) {
			const int i = j + 4 * m;
			B[k * 16 + (i * 5 % 16)] = X[k * 16 + i];
		}
}

/* smix1.  SFILL = the S-box fill pass (reference: V == ctx->S, blockmix_salsa,
 * r = 1), a template parameter so the branch folds. */
template<uint32_t R, bool SFILL, int PLACE = PWX_S_SHARED, int WIDTH = 4>
__device__ __forceinline__ void yp_smix1(uint32_t *B, uint32_t N, uint32_t *V,
                                         uint32_t *X, uint4 *S,
                                         uint32_t &b0, uint32_t &b1,
                                         uint32_t &b2, uint32_t &w4,
                                         const int j, const unsigned mask,
                                         const int tid, const unsigned wmask)
{
	const uint32_t s4 = 8u * R;            /* words-per-V-slot, in uint4 units */
	uint4 *X4 = (uint4 *)X;
	uint4 *V4 = (uint4 *)V;

	yp_shuffle_in(X, B, 2u * R, j);
	__syncwarp(wmask);	/* all WIDTH threads; `mask` = the 4 pwxform lanes */

	/* r > 1 (yespower 1.0 only): seed each 128-byte chunk from the previous one. */
	if (!SFILL) {
#pragma unroll 1
		for (uint32_t k = 1; k < R; k++) {
			for (uint32_t c = (uint32_t)tid; c < 8u; c += (uint32_t)WIDTH)
				X4[k * 8u + c] = X4[(k - 1u) * 8u + c];
			YP_WSYNC(wmask);
			if (tid < PWX_GATHER)
				blockmix_pwxform<1, PLACE>(X + k * 32u, S, b0, b1, b2, w4, j, mask);
			YP_WSYNC(wmask);
		}
	}

	uint32_t n2 = 1u;	/* p2floor(i), updated incrementally */

#pragma unroll 1
	for (uint32_t i = 0; i < N; i++) {
		/* V_i <- X, strided by WIDTH so the extra threads share the copy. */
		for (uint32_t c = (uint32_t)tid; c < s4; c += (uint32_t)WIDTH)
			yp_vst(&V4[i * s4 + c], X4[c]);

		if (i > 1) {
			if (i == 2u * n2) n2 = i;
			/* jj = wrap(integerify(X), i) */
			const uint32_t jj = (yp_integerify<R>(X) & (n2 - 1u)) + (i - n2);
			const uint32_t step = (uint32_t)WIDTH * (uint32_t)YP_VBATCH;
			for (uint32_t c0 = (uint32_t)tid; c0 < s4; c0 += step) {
				uint4 vb[YP_VBATCH];
#pragma unroll
				for (int k = 0; k < YP_VBATCH; k++) {
					const uint32_t c = c0 + (uint32_t)k * (uint32_t)WIDTH;
					if (c < s4) vb[k] = yp_vld(&V4[jj * s4 + c]);
				}
#pragma unroll
				for (int k = 0; k < YP_VBATCH; k++) {
					const uint32_t c = c0 + (uint32_t)k * (uint32_t)WIDTH;
					if (c < s4) {
						X4[c].x ^= vb[k].x; X4[c].y ^= vb[k].y;
						X4[c].z ^= vb[k].z; X4[c].w ^= vb[k].w;
			}
		}
			}
		}
		YP_WSYNC(wmask);	/* XOR by WIDTH threads, blockmix reads all of X */

		if (tid < PWX_GATHER) {
		if (SFILL) blockmix_salsa<2>(X, mask);
			else       blockmix_pwxform<R, PLACE>(X, S, b0, b1, b2, w4, j, mask);
		}
		YP_WSYNC(wmask);	/* WIDTH > 4: next V loop reads words blockmix wrote */
	}

	yp_shuffle_out(B, X, 2u * R, j);
	__syncwarp(wmask);	/* block scope: all WIDTH threads reach this */
}

/* smix2.  Nloop == 2 is the read-only pass (the reference skips the V write). */
template<uint32_t R, int PLACE = PWX_S_SHARED, int WIDTH = 4>
__device__ __forceinline__ void yp_smix2(uint32_t *B, uint32_t N, uint32_t Nloop,
                                         uint32_t *V, uint32_t *X, uint4 *S,
                                         uint32_t &b0, uint32_t &b1,
                                         uint32_t &b2, uint32_t &w4,
                                         const int j, const unsigned mask,
                                         const int tid, const unsigned wmask)
{
	const uint32_t s4 = 8u * R;
	uint4 *X4 = (uint4 *)X;
	uint4 *V4 = (uint4 *)V;

	yp_shuffle_in(X, B, 2u * R, j);
	__syncwarp(wmask);	/* block scope: all WIDTH threads reach this */

#pragma unroll 1
	for (uint32_t i = 0; i < Nloop; i++) {
		const uint32_t jj = yp_integerify<R>(X) & (N - 1u);

		const uint32_t step2 = (uint32_t)WIDTH * (uint32_t)YP_VBATCH;
		for (uint32_t c0 = (uint32_t)tid; c0 < s4; c0 += step2) {
			uint4 vb[YP_VBATCH];
#pragma unroll
			for (int k = 0; k < YP_VBATCH; k++) {
				const uint32_t c = c0 + (uint32_t)k * (uint32_t)WIDTH;
				if (c < s4) vb[k] = yp_vld(&V4[jj * s4 + c]);
		}
#pragma unroll
			for (int k = 0; k < YP_VBATCH; k++) {
				const uint32_t c = c0 + (uint32_t)k * (uint32_t)WIDTH;
				if (c < s4) {
					X4[c].x ^= vb[k].x; X4[c].y ^= vb[k].y;
					X4[c].z ^= vb[k].z; X4[c].w ^= vb[k].w;
					if (Nloop != 2u) yp_vst(&V4[jj * s4 + c], X4[c]);
				}
			}
		}
		YP_WSYNC(wmask);

		if (tid < PWX_GATHER)
			blockmix_pwxform<R, PLACE>(X, S, b0, b1, b2, w4, j, mask);
		YP_WSYNC(wmask);
	}

	yp_shuffle_out(B, X, 2u * R, j);
	__syncwarp(wmask);	/* block scope: all WIDTH threads reach this */
}

#endif /* YESPOWER_SMIX_CUH */
