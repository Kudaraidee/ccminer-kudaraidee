/* Salsa20/ROUNDS core, 4 threads cooperating on one 64-byte block.
 *
 * Shared device header (coding guideline section 3).  yespower 1.0 needs
 * Salsa20/2, yescrypt 0.5 needs /8.  The round count is the only
 * difference, so it is a template parameter rather than two copies.
 *
 * This does NOT replace algos/yescrypt's private SALSA_CORE.  That is live
 * consensus code on six algos and stays as it is; migrating it here would be a
 * separate opt-in refactor, gated on generated-code identity.
 *
 * DATA LAYOUT -- the part that is easy to get wrong:
 *   The block is held in the SIMD-SHUFFLED order, i.e. exactly the order the
 *   upstream reference's `salsa20(uint32_t B[16], rounds)` receives.  The
 *   reference unshuffles internally (`x[i * 5 % 16] = B[i]`); we do not, because
 *   in shuffled space the double round is a quarter-round plus a lane rotation,
 *   which is what makes the 4-thread form work at all.
 *
 *   Lane L (0..3) holds words   B[L], B[L + 4], B[L + 8], B[L + 12]
 *                          as   x0,   x1,       x2,       x3
 */

#ifndef CUDA_SALSA_DEVICE_CUH
#define CUDA_SALSA_DEVICE_CUH

#include <stdint.h>

#ifndef SALSA_ROTL32
#define SALSA_ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#endif

/* One Salsa20 quarter-round group, operating on this lane's four words. */
#define SALSA_QR(a, b, c, d) { \
	(b) ^= SALSA_ROTL32((a) + (d),  7); \
	(c) ^= SALSA_ROTL32((b) + (a),  9); \
	(d) ^= SALSA_ROTL32((c) + (b), 13); \
	(a) ^= SALSA_ROTL32((d) + (c), 18); \
}

/* salsa_core<ROUNDS>(x0..x3, mask)
 *
 * ROUNDS must be even (Salsa20 is defined in double rounds): 2 for yespower 1.0,
 * 8 for yescrypt/yespower 0.5.  `mask` is the __shfl_sync participation mask of
 * the calling group -- pass the full warp mask when whole warps call together,
 * or the 4-lane mask when only one instance is active in the warp.
 *
 * The width-4 shuffles rotate words between the four lanes, which is the row
 * step of the double round; the column step is SALSA_QR on each lane's own
 * words.  Lane index comes from threadIdx.x & 3, so the four cooperating
 * threads must be contiguous and 4-aligned within the warp.
 */
template<int ROUNDS>
__device__ __forceinline__ void salsa_core(uint32_t &x0, uint32_t &x1,
                                           uint32_t &x2, uint32_t &x3,
                                           const unsigned mask)
{
	static_assert(ROUNDS > 0 && (ROUNDS & 1) == 0, "Salsa20 needs an even round count");

	const uint32_t t0 = x0, t1 = x1, t2 = x2, t3 = x3;
	const int lane = threadIdx.x & 3;

#pragma unroll
	for (int i = 0; i < ROUNDS / 2; i++) {
		SALSA_QR(x0, x1, x2, x3);
		x1 = __shfl_sync(mask, x1, lane + 3, 4);
		x2 = __shfl_sync(mask, x2, lane + 2, 4);
		x3 = __shfl_sync(mask, x3, lane + 1, 4);

		SALSA_QR(x0, x3, x2, x1);
		x1 = __shfl_sync(mask, x1, lane + 1, 4);
		x2 = __shfl_sync(mask, x2, lane + 2, 4);
		x3 = __shfl_sync(mask, x3, lane + 3, 4);
	}

	/* Salsa20 adds the input back to the permuted state. */
	x0 += t0; x1 += t1; x2 += t2; x3 += t3;
}

/* blockmix_salsa<ROUNDS>(B, mask) -- reference blockmix_salsa(), r fixed at 1.
 *
 * B is 128 bytes (two 64-byte blocks) in the SIMD-shuffled order.  Used only for
 * the S-box fill, where smix1 runs with r = 1 and V aliased onto S.
 *
 * Everything stays in salsa_core's strided layout, so no transpose is needed:
 * lane L touches only words congruent to L mod 4, which are disjoint between
 * lanes.  That is why this one needs no __syncwarp of its own -- the only
 * cross-lane traffic is inside salsa_core's shuffles.
 */
template<int ROUNDS>
__device__ __forceinline__ void blockmix_salsa(uint32_t *B, const unsigned mask)
{
	const int L = threadIdx.x & 3;

	/* X <- B_1 (the last of the two blocks) */
	uint32_t x0 = B[16 + L], x1 = B[16 + L + 4],
	         x2 = B[16 + L + 8], x3 = B[16 + L + 12];

#pragma unroll
	for (int i = 0; i < 2; i++) {
		uint32_t *Bi = B + i * 16;
		x0 ^= Bi[L]; x1 ^= Bi[L + 4]; x2 ^= Bi[L + 8]; x3 ^= Bi[L + 12];

		salsa_core<ROUNDS>(x0, x1, x2, x3, mask);

		Bi[L] = x0; Bi[L + 4] = x1; Bi[L + 8] = x2; Bi[L + 12] = x3;
	}
}

#endif /* CUDA_SALSA_DEVICE_CUH */
