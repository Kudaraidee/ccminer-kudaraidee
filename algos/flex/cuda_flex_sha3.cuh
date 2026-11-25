/**
 * SHA-3 (0x06 padding) device primitives for Flex.
 *
 * Flex applies SHA3-style padding to EVERY keccak in its chain: the selection
 * seed, the KECCAK core round (at 80 bytes when it is round 0, at 64 bytes
 * otherwise) and the final SHA3-256.  The tree's shared keccak stage
 * (cuda/keccak_device.cuh, algos/stages/cuda_keccak512.cu) is legacy Keccak,
 * 0x01, and is consumed by quark, x11, x13, x16, x17, x21s, x25x, timetravel,
 * tribus, soterg, evohash, nist5, hmq1725 and ghostrider -- changing it is a
 * tree-wide consensus change.
 *
 * But no rounds need copying.  keccak_device.cuh does NOT bake the pad byte
 * into its absorb specialization: keccak512_hash_64() sets it in one wrapper
 * line, `s[8] = make_uint2(1, 0x80000000)` -- lane 8's low byte is the domain
 * byte and its high byte is the end-of-rate bit.  So the only difference
 * between Keccak-512 and SHA3-512 here is `1` -> `6`, and these wrappers reuse
 * keccak512_absorb_round_64 / keccak_round / keccak512_output_round /
 * keccakf1600_full verbatim.
 *
 * Lane layout reminder (uint2 = { low 32 bits, high 32 bits }):
 * rate 72  (SHA3-512) =  9 lanes; a 64-byte message fills lanes 0..7, the
 * domain byte lands at lane 8 byte 0 and the
 * end-of-rate bit at lane 8 byte 7.
 * rate 136 (SHA3-256) = 17 lanes; a 64-byte message fills lanes 0..7, the
 * domain byte lands at lane 8 byte 0 and the
 * end-of-rate bit at lane 16 byte 7.
 */

#ifndef CUDA_FLEX_SHA3_CUH
#define CUDA_FLEX_SHA3_CUH

#include <stdint.h>
#include "cuda/keccak_device.cuh"

#ifdef __CUDACC__

/* The SHA-3 domain separator, in the position the absorb expects it. */
#define FLEX_SHA3_PAD_LANE  make_uint2(6, 0x80000000)   /* rate 72:  lane 8 */

/* ---------------------------------------------------------------------------
 * SHA3-512 of a 64-byte input, in place.  Identical to keccak512_hash_64()
 * except for the domain byte.
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void flex_sha3_512_hash_64(uint2 hash[8])
{
	uint2 s[25];

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		s[i] = hash[i];
	s[8] = FLEX_SHA3_PAD_LANE;          /* 0x06 here; keccak512 uses 0x01 */

	keccak512_absorb_round_64(s);
	#pragma unroll 4
	for (int i = 1; i < 23; i++)
		keccak_round(s, c_keccak_rc[i]);
	keccak512_output_round(s);

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		hash[i] = s[i];
}

/* ---------------------------------------------------------------------------
 * SHA3-512 of an 80-byte input -> 64 bytes.
 *
 * 80 bytes does not fit one 72-byte block, so this absorbs twice: bytes 0..71
 * then bytes 72..79 plus the padding.  Written straightforwardly rather than
 * with the c_mid/c_msg midstate trick the legacy 80-byte kernels use: bytes
 * 0..71 of the header are constant across a nonce batch, so the first
 * permutation is a per-job midstate and can be hoisted later.  Correctness
 * first; see the the project notes/a separate items.
 *
 * `in` is the 80-byte header as 20 little-endian-loaded uint32 words, i.e.
 * exactly the `endiandata` layout scanhash prepares.  Keccak is
 * little-endian native, so the words are absorbed without byte swabbing.
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void flex_sha3_512_hash_80(const uint32_t * __restrict__ in, uint2 out[8])
{
	uint2 s[25];

	/* ---- block 1: header bytes 0..71 = words 0..17 = lanes 0..8 ---- */
	#pragma unroll 9
	for (int i = 0; i < 9; i++)
		s[i] = make_uint2(in[2 * i], in[2 * i + 1]);
	#pragma unroll 16
	for (int i = 9; i < 25; i++)
		s[i] = make_uint2(0, 0);

	keccakf1600_full(s);

	/* ---- block 2: header bytes 72..79 = words 18,19 -> lane 0, then pad.
	 * Lanes 9..24 carry over from the first permutation and are NOT cleared;
	 * only the rate lanes (0..8) are XORed. */
	s[0] = make_uint2(s[0].x ^ in[18], s[0].y ^ in[19]);
	s[1] = make_uint2(s[1].x ^ 6u, s[1].y);          /* domain byte 0x06 */
	s[8] = make_uint2(s[8].x, s[8].y ^ 0x80000000u); /* end-of-rate bit */

	keccakf1600_full(s);

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		out[i] = s[i];
}

/* ---------------------------------------------------------------------------
 * SHA3-256 of a 64-byte input -> 32 bytes.  Rate 136, so one block.
 * This is flex's final hash: it consumes ALL 64 bytes of the last CN output,
 * which is why the high 32 bytes of a CN result are consensus-load-bearing in
 * flex where ghostrider discards them.
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void flex_sha3_256_hash_64(const uint2 * __restrict__ in, uint2 out[4])
{
	uint2 s[25];

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		s[i] = in[i];
	s[8] = make_uint2(6, 0);              /* domain byte, block offset 64 */
	#pragma unroll 7
	for (int i = 9; i < 16; i++)
		s[i] = make_uint2(0, 0);
	s[16] = make_uint2(0, 0x80000000);    /* end-of-rate, block offset 135 */
	#pragma unroll 8
	for (int i = 17; i < 25; i++)
		s[i] = make_uint2(0, 0);

	keccakf1600_full(s);

	#pragma unroll 4
	for (int i = 0; i < 4; i++)
		out[i] = s[i];
}

#endif /* __CUDACC__ */

#endif /* CUDA_FLEX_SHA3_CUH */
