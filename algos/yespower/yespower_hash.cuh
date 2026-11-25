/* yespower 1.0 -- the whole hash as one inlinable device function, transcribed
 * from sph/yespower_ref.c (yespower_ref() + smix()). The mining kernel and the
 * self-test/differential kernel inline this same body.
 *
 * Four threads (the PWXgather lanes) compute the hash; with WIDTH > 4 the others
 * only help move V. S is the 96 KiB box array, in shared or global memory (PLACE).
 *
 * The S-box fill is smix1 with r = 1, N = 768, V aliased onto S and Salsa20 in
 * place of pwxform; it writes all of S before the first gather. */

#ifndef YESPOWER_HASH_CUH
#define YESPOWER_HASH_CUH

#include "yespower_smix.cuh"
#include "yespower_head_tail.cuh"

/* Head/tail primitive. yespower 1.0 uses SHA-256; yespower-b2b (mined as
 * `-a power2b` and `-a yespower-b2b`) uses BLAKE2b for the SAME structure --
 * smix/pwxform/salsa are byte-identical between them.
 *
 * SHA256_181 is yespower 1.0 over EqPay's 181-byte extended header. It differs
 * from SHA256 only in the length of the message the head hashes: 1.0 replaces
 * the PBKDF2 salt with `pers`, so the header never reaches the salt and the
 * rest of the pipeline is untouched.
 *
 * SHA256_64 hashes a 64-byte per-instance message (minotaurx): `hdr` points at
 * 16 words and w19 is unused. */
enum { YP_HEAD_SHA256 = 0, YP_HEAD_B2B = 1, YP_HEAD_SHA256_181 = 2, YP_HEAD_SHA256_64 = 3 };

#define YP_SBOX_UINT4   6144u    /* 96 KiB / 16 B, three boxes of 2048 */
#define YP_SFILL_N       768u    /* Sbytes / 128 */

/* hdr : 20 words in SHA-256 input order (big-endian valued); hdr[19] ignored
 * w19 : the last header word, passed separately so a mining kernel can hold the
 *       other 19 in __constant__ and vary only the nonce
 * out : 8 words, the digest as LITTLE-endian words of the byte stream, which is
 *       what fulltest()/the target compare expect.
 */

/* WIDTH is capped at one warp (32) and the cap is load-bearing: every barrier
 * in here is a `__syncwarp`, which orders one warp and nothing more.  Above 32
 * the code would need `__syncthreads()`.  The static_assert enforces it. */
template<uint32_t R, int PLACE = PWX_S_SHARED, int HEAD = YP_HEAD_SHA256, int WIDTH = 4>
__device__ __forceinline__ void yespower_hash_1_0(const uint32_t *hdr, const uint32_t w19,
                                                  const uint32_t N,
                                                  uint4 *S, uint32_t *B, uint32_t *X,
                                                  uint32_t *V, uint32_t out[8],
                                                  const int j, const unsigned mask,
 /* no default: every caller passes its own thread index */
                                                  const int tid, const unsigned wmask)
{
	static_assert(WIDTH >= 4 && WIDTH <= 32 && (WIDTH & (WIDTH - 1)) == 0,
	              "yespower: WIDTH must be a power of two in [4,32] -- one warp; "
	              "above 32 __syncwarp cannot order the block and __syncthreads is required");
	uint32_t key[8], saved[8];
	uint8_t  b2b_init[32], b2b_saved[32];

	/* The head and tail are 4-lane code; the other threads must not run them
	 * (j = tid & 3 would alias their lanes onto 0..3). */
	if (tid < PWX_GATHER) {
	if (HEAD == YP_HEAD_B2B) {
		/* -- head: init = BLAKE2b(header); B = PBKDF2-BLAKE2b(init, pers, 1, 128r) --
		* No swab anywhere: (uint8_t *)B already IS the byte stream. */
		yp_b2b_init_hash(hdr, w19, b2b_init);
		yp_b2b_fill_B<R>(b2b_init, B, j, mask);
#pragma unroll
		for (int i = 0; i < 32; i++) b2b_saved[i] = ((const uint8_t *)B)[i];
	} else {
	/* -- head: sha256 = SHA256(header); B = PBKDF2(sha256, pers, 1, 128r) --
	 * HEAD is a template constant, so only one of these is compiled. */
	if (HEAD == YP_HEAD_SHA256_181)     yp_sha256_181(hdr, w19, key);
	else if (HEAD == YP_HEAD_SHA256_64) yp_sha256_64(hdr, key);
	else                                yp_sha256_80(hdr, w19, key);
	yp_pbkdf2_fill_B<R>(key, B, j, mask);

	/* sha256 <- B[0..7].  The reference copies WORDS out of B and later feeds
	 * them to HMAC as BYTES, so this is where the LE word convention of B meets
	 * the BE word convention of SHA-256. */
#pragma unroll
	for (int i = 0; i < 8; i++) saved[i] = cuda_swab32(B[i]);
	}
	} /* tid < PWX_GATHER */
	YP_WSYNC(wmask); /* B is now built; every thread may read it in smix */

	/* -- smix -- */
	uint32_t b0 = 0, b1 = 2048u, b2 = 4096u, w4 = 0;

	uint32_t Nloop_all = (N + 2u) / 3u;               /* 1/3, round up   */
	uint32_t Nloop_rw  = Nloop_all;
	Nloop_all++; Nloop_all &= ~(uint32_t)1;           /* round up to even */
	Nloop_rw++;  Nloop_rw  &= ~(uint32_t)1;           /* 1.0: also up     */

	yp_smix1<1, true , PLACE, WIDTH>(B, YP_SFILL_N, (uint32_t *)S, X, S, b0, b1, b2, w4, j, mask, tid, wmask);
	yp_smix1<R, false, PLACE, WIDTH>(B, N, V, X, S, b0, b1, b2, w4, j, mask, tid, wmask);
	yp_smix2<R, PLACE, WIDTH>(B, N, Nloop_rw,              V, X, S, b0, b1, b2, w4, j, mask, tid, wmask);
	yp_smix2<R, PLACE, WIDTH>(B, N, Nloop_all - Nloop_rw,  V, X, S, b0, b1, b2, w4, j, mask, tid, wmask);

	/* -- tail: HMAC(key = last 64 B of B, msg = the saved 32 B) --
	 * Test only: -DYP_ABLATE_TAIL replaces the tail with a copy out of B (a wrong
	 * digest by construction); no build defines it. */
#ifdef YP_ABLATE_TAIL
	YP_WSYNC(wmask);
	if (tid >= PWX_GATHER) return;
#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = B[i];
	return;
#endif
	YP_WSYNC(wmask); /* smix's last write to B must be visible to lane 0 */
	if (tid >= PWX_GATHER) return; /* spare threads are done */
	if (HEAD == YP_HEAD_B2B) {
		/* The 32 output bytes written into (uint8_t *)out ARE the little-endian
		* words fulltest wants, so unlike the SHA-256 twin there is no swab. */
		yp_b2b_tail<R>(B, b2b_saved, out);
	} else {
	yp_hmac_tail<R>(B, saved, out);
#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = cuda_swab32(out[i]);
}
}

#endif /* YESPOWER_HASH_CUH */
