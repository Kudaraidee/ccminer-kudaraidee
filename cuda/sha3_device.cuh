// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Shared FIPS-202 SHA-3 primitive library (device side).
 *
 * Consumers: verthash (sha3_256_80 / sha3_512_80) and rinhash (sha3_256_32).
 * Changing anything here needs every consumer re-gated on DIGESTS.
 *
 * Keccak-f[1600] permutation + SHA3-256 / SHA3-512 with 0x06 domain padding
 * (FIPS-202). This is DISTINCT from cuda/keccak_device.cuh (Keccak-256 with the
 * bare 0x01 pad used by the keccak family) and from the ProgPoW keccak_f800 --
 * do not conflate the padding byte or the rate.
 *
 * Provenance: permutation transcribed from the VerthashMiner CUDA kernel
 * (src/vhCuda/verthash.cu, CryptoGraphics GPLv2); bit-exact against tiny_sha3
 * (algos/verthash/tiny_sha3.c). Every primitive is a separately callable device
 * function (docs/coding-guideline.md §3); consuming kernels call these.
 *
 * Rates: SHA3-256 rsiz = 136 bytes (17 lanes), SHA3-512 rsiz = 72 bytes
 * (9 lanes). Both absorb an 80-byte message in a single block (80 < rate), so
 * finalization is a single permutation.
 */

#ifndef CUDA_SHA3_DEVICE_CUH
#define CUDA_SHA3_DEVICE_CUH

#include <stdint.h>

#ifdef __CUDACC__

// Keccak round constants, uint2 (lo,hi) form. One 192-byte __constant__ copy per
// translation unit; statically initialized so no cudaMemcpyToSymbol is needed.
#define SHA3_RC2_INIT { \
	{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 }, { 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 }, \
	{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 }, { 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 }, \
	{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 }, { 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 }, \
	{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 }, { 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 }, \
	{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 }, { 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 }, \
	{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 } }

static __constant__ uint2 c_sha3_rc[24] = SHA3_RC2_INIT;

__device__ __forceinline__
uint2 sha3_rotl64(const uint2 w, const int offset)
{
	uint2 result;
	if (offset < 32) {
		result.y = (w.y << offset) | (w.x >> (32 - offset));
		result.x = (w.x << offset) | (w.y >> (32 - offset));
	} else {
		result.y = (w.x << (offset - 32)) | (w.y >> (64 - offset));
		result.x = (w.y << (offset - 32)) | (w.x >> (64 - offset));
	}
	return result;
}

__device__ __forceinline__
uint2 sha3_xor(const uint2 a, const uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }

__device__ __forceinline__
uint2 sha3_andnot(const uint2 a, const uint2 b) { return make_uint2((~a.x) & b.x, (~a.y) & b.y); }

// Keccak-f[1600] in-place on a 25-lane uint2 state (verbatim round body from the
// VerthashMiner kernel; bit-identical to tiny_sha3 sha3_keccakf).
__device__ __forceinline__
void sha3_keccakf_1600(uint2 st[25])
{
	uint2 t[5], u[5], v, w;

	#pragma unroll 1
	for (int r = 0; r < 24; r++) {
		// Theta
		t[0] = sha3_xor(st[0], sha3_xor(st[5], sha3_xor(st[10], sha3_xor(st[15], st[20]))));
		t[1] = sha3_xor(st[1], sha3_xor(st[6], sha3_xor(st[11], sha3_xor(st[16], st[21]))));
		t[2] = sha3_xor(st[2], sha3_xor(st[7], sha3_xor(st[12], sha3_xor(st[17], st[22]))));
		t[3] = sha3_xor(st[3], sha3_xor(st[8], sha3_xor(st[13], sha3_xor(st[18], st[23]))));
		t[4] = sha3_xor(st[4], sha3_xor(st[9], sha3_xor(st[14], sha3_xor(st[19], st[24]))));

		u[0] = sha3_xor(sha3_rotl64(t[0], 1), t[3]);
		u[1] = sha3_xor(sha3_rotl64(t[1], 1), t[4]);
		u[2] = sha3_xor(sha3_rotl64(t[2], 1), t[0]);
		u[3] = sha3_xor(sha3_rotl64(t[3], 1), t[1]);
		u[4] = sha3_xor(sha3_rotl64(t[4], 1), t[2]);

		st[4] = sha3_xor(st[4], u[0]); st[9] = sha3_xor(st[9], u[0]); st[14] = sha3_xor(st[14], u[0]); st[19] = sha3_xor(st[19], u[0]); st[24] = sha3_xor(st[24], u[0]);
		st[0] = sha3_xor(st[0], u[1]); st[5] = sha3_xor(st[5], u[1]); st[10] = sha3_xor(st[10], u[1]); st[15] = sha3_xor(st[15], u[1]); st[20] = sha3_xor(st[20], u[1]);
		st[1] = sha3_xor(st[1], u[2]); st[6] = sha3_xor(st[6], u[2]); st[11] = sha3_xor(st[11], u[2]); st[16] = sha3_xor(st[16], u[2]); st[21] = sha3_xor(st[21], u[2]);
		st[2] = sha3_xor(st[2], u[3]); st[7] = sha3_xor(st[7], u[3]); st[12] = sha3_xor(st[12], u[3]); st[17] = sha3_xor(st[17], u[3]); st[22] = sha3_xor(st[22], u[3]);
		st[3] = sha3_xor(st[3], u[4]); st[8] = sha3_xor(st[8], u[4]); st[13] = sha3_xor(st[13], u[4]); st[18] = sha3_xor(st[18], u[4]); st[23] = sha3_xor(st[23], u[4]);

		// Rho Pi
		v = st[1];
		st[1]  = sha3_rotl64(st[6], 44);
		st[6]  = sha3_rotl64(st[9], 20);
		st[9]  = sha3_rotl64(st[22], 61);
		st[22] = sha3_rotl64(st[14], 39);
		st[14] = sha3_rotl64(st[20], 18);
		st[20] = sha3_rotl64(st[2], 62);
		st[2]  = sha3_rotl64(st[12], 43);
		st[12] = sha3_rotl64(st[13], 25);
		st[13] = sha3_rotl64(st[19], 8);
		st[19] = sha3_rotl64(st[23], 56);
		st[23] = sha3_rotl64(st[15], 41);
		st[15] = sha3_rotl64(st[4], 27);
		st[4]  = sha3_rotl64(st[24], 14);
		st[24] = sha3_rotl64(st[21], 2);
		st[21] = sha3_rotl64(st[8], 55);
		st[8]  = sha3_rotl64(st[16], 45);
		st[16] = sha3_rotl64(st[5], 36);
		st[5]  = sha3_rotl64(st[3], 28);
		st[3]  = sha3_rotl64(st[18], 21);
		st[18] = sha3_rotl64(st[17], 15);
		st[17] = sha3_rotl64(st[11], 10);
		st[11] = sha3_rotl64(st[7], 6);
		st[7]  = sha3_rotl64(st[10], 3);
		st[10] = sha3_rotl64(v, 1);

		// Chi
		v = st[0]; w = st[1]; st[0] = sha3_xor(v, sha3_andnot(w, st[2])); st[1] = sha3_xor(w, sha3_andnot(st[2], st[3])); st[2] = sha3_xor(st[2], sha3_andnot(st[3], st[4])); st[3] = sha3_xor(st[3], sha3_andnot(st[4], v)); st[4] = sha3_xor(st[4], sha3_andnot(v, w));
		v = st[5]; w = st[6]; st[5] = sha3_xor(v, sha3_andnot(w, st[7])); st[6] = sha3_xor(w, sha3_andnot(st[7], st[8])); st[7] = sha3_xor(st[7], sha3_andnot(st[8], st[9])); st[8] = sha3_xor(st[8], sha3_andnot(st[9], v)); st[9] = sha3_xor(st[9], sha3_andnot(v, w));
		v = st[10]; w = st[11]; st[10] = sha3_xor(v, sha3_andnot(w, st[12])); st[11] = sha3_xor(w, sha3_andnot(st[12], st[13])); st[12] = sha3_xor(st[12], sha3_andnot(st[13], st[14])); st[13] = sha3_xor(st[13], sha3_andnot(st[14], v)); st[14] = sha3_xor(st[14], sha3_andnot(v, w));
		v = st[15]; w = st[16]; st[15] = sha3_xor(v, sha3_andnot(w, st[17])); st[16] = sha3_xor(w, sha3_andnot(st[17], st[18])); st[17] = sha3_xor(st[17], sha3_andnot(st[18], st[19])); st[18] = sha3_xor(st[18], sha3_andnot(st[19], v)); st[19] = sha3_xor(st[19], sha3_andnot(v, w));
		v = st[20]; w = st[21]; st[20] = sha3_xor(v, sha3_andnot(w, st[22])); st[21] = sha3_xor(w, sha3_andnot(st[22], st[23])); st[22] = sha3_xor(st[22], sha3_andnot(st[23], st[24])); st[23] = sha3_xor(st[23], sha3_andnot(st[24], v)); st[24] = sha3_xor(st[24], sha3_andnot(v, w));

		// Iota
		st[0] = sha3_xor(st[0], c_sha3_rc[r]);
	}
}

// ---------------------------------------------------------------------------
// SHA3-256 of an 80-byte message given as 20 little-endian uint32 words.
// Absorbs into a fresh state (rate 136 bytes > 80, single block), applies the
// 0x06 pad and 0x80 final bit, permutes, and writes 8 LE words to out[8].
__device__ __forceinline__
void sha3_256_80(const uint32_t in[20], uint32_t out[8])
{
	uint2 st[25];
	#pragma unroll
	for (int i = 0; i < 25; i++) st[i] = make_uint2(0, 0);

	// absorb 80 bytes = 10 lanes
	#pragma unroll
	for (int i = 0; i < 10; i++) st[i] = make_uint2(in[2 * i], in[2 * i + 1]);

	// pad: byte[80] ^= 0x06 -> lane 10 low word; byte[135] ^= 0x80 -> lane 16 hi word
	st[10].x ^= 0x00000006U;
	st[16].y ^= 0x80000000U;

	sha3_keccakf_1600(st);

	#pragma unroll
	for (int i = 0; i < 4; i++) { out[2 * i] = st[i].x; out[2 * i + 1] = st[i].y; }
}

// ---------------------------------------------------------------------------
// SHA3-256 of a 32-byte message given as bytes. Rate 136 > 32, so this is a
// single block: absorb 4 lanes, pad, one permutation. Writes 8 LE words.
//
// Byte input rather than uint32 (unlike sha3_256_80) because the caller's buffer
// is a uint8_t[32] carrying another primitive's digest and is not guaranteed to
// be 4-byte aligned; the loads below make no alignment assumption.
__device__ __forceinline__
void sha3_256_32(const uint8_t *in, uint32_t out[8])
{
	uint2 st[25];
	#pragma unroll
	for (int i = 0; i < 25; i++) st[i] = make_uint2(0, 0);

	// absorb 32 bytes = 4 lanes, little-endian
	#pragma unroll
	for (int i = 0; i < 4; i++) {
		uint32_t lo = 0, hi = 0;
		#pragma unroll
		for (int b = 0; b < 4; b++) {
			lo |= (uint32_t)in[8 * i + b] << (8 * b);
			hi |= (uint32_t)in[8 * i + 4 + b] << (8 * b);
		}
		st[i] = make_uint2(lo, hi);
	}

	// pad: byte[32] ^= 0x06 -> lane 4 low word; byte[135] ^= 0x80 -> lane 16 hi word
	st[4].x ^= 0x00000006U;
	st[16].y ^= 0x80000000U;

	sha3_keccakf_1600(st);

	#pragma unroll
	for (int i = 0; i < 4; i++) { out[2 * i] = st[i].x; out[2 * i + 1] = st[i].y; }
}

// SHA3-512 of an 80-byte message (rate 72 bytes < 80, so the message spans two
// blocks: 72 bytes absorbed + permute, then remaining 8 bytes + pad). Writes 16
// LE words to out[16].
//
// The 80-byte header is split as words in[0..17] (72 bytes) + tail two words
// in[18],in[19]. The first block is a per-job constant except header[0]; the
// consuming kernel exploits this via the precompute path. This standalone form
// is the reference used by the KAT harness.
__device__ __forceinline__
void sha3_512_80(const uint32_t in[20], uint32_t out[16])
{
	uint2 st[25];
	#pragma unroll
	for (int i = 0; i < 25; i++) st[i] = make_uint2(0, 0);

	// block 1: absorb first 72 bytes = 9 lanes, permute
	#pragma unroll
	for (int i = 0; i < 9; i++) st[i] = make_uint2(in[2 * i], in[2 * i + 1]);
	sha3_keccakf_1600(st);

	// block 2: absorb remaining 8 bytes (lane 0), pad
	st[0].x ^= in[18];
	st[0].y ^= in[19];
	st[1].x ^= 0x00000006U; // byte[8] of block 2 = byte[80] of message
	st[8].y ^= 0x80000000U; // byte[71] of block 2 = last byte of rate
	sha3_keccakf_1600(st);

	#pragma unroll
	for (int i = 0; i < 8; i++) { out[2 * i] = st[i].x; out[2 * i + 1] = st[i].y; }
}

#endif // __CUDACC__

#endif // CUDA_SHA3_DEVICE_CUH
