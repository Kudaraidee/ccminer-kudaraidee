// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Shared BLAKE-512 primitive library (device side) — x-family stage function.
 *
 * Compression extracted bit-identically from the alexis-lineage
 * blake512_gpu_hash_64 kernel (algos/stages/cuda_blake512.cu, Provos
 * Alexis 2016 / SP 2018). Every cryptographic primitive is a separately
 * callable building block (docs/coding-guideline.md §3); fused kernels call
 * these instead of re-implementing them.
 *
 * The 80-byte first-stage kernel and its host midstate precompute
 * (c_m/c_v/c_x) stay in the consuming file: they read per-algo __constant__
 * symbols (same rule as cuda/keccak_device.cuh's absorb midstate).
 *
 * Interface convention: hash[8] holds the 64-byte value exactly as it sits
 * in the inter-stage d_hash buffer (little-endian uint2 words); byte
 * swabbing to BLAKE's big-endian message view happens inside.
 */

#ifndef CUDA_BLAKE512_DEVICE_CUH
#define CUDA_BLAKE512_DEVICE_CUH

#include <stdint.h>
#include "cuda/helper_alexis.cuh"  // ROR2/ROR16/SWAPDWORDS2/xor3x/cuda_swab64_U2/devectorize

#ifdef __CUDACC__

/* BLAKE-512 pi-digit constants (the spec's u[16]) and IV. */
static __constant__ uint2 c_blake512_z[16] = {
	{ 0x85a308d3, 0x243f6a88 }, { 0x03707344, 0x13198a2e }, { 0x299f31d0, 0xa4093822 }, { 0xec4e6c89, 0x082efa98 },
	{ 0x38d01377, 0x452821e6 }, { 0x34e90c6c, 0xbe5466cf }, { 0xc97c50dd, 0xc0ac29b7 }, { 0xb5470917, 0x3f84d5b5 },
	{ 0x8979fb1b, 0x9216d5d9 }, { 0x98dfb5ac, 0xd1310ba6 }, { 0xd01adfb7, 0x2ffd72db }, { 0x6a267e96, 0xb8e1afed },
	{ 0xf12c7f99, 0xba7c9045 }, { 0xb3916cf7, 0x24a19947 }, { 0x858efc16, 0x0801f2e2 }, { 0x71574e69, 0x636920d8 }
};

static __constant__ uint2 c_blake512_h[8] = {
	{ 0xf3bcc908, 0x6a09e667 }, { 0x84caa73b, 0xbb67ae85 },
	{ 0xfe94f82b, 0x3c6ef372 }, { 0x5f1d36f1, 0xa54ff53a },
	{ 0xade682d1, 0x510e527f }, { 0x2b3e6c1f, 0x9b05688c },
	{ 0xfb41bd6b, 0x1f83d9ab }, { 0x137e2179, 0x5be0cd19 }
};

/* Four column/diagonal G steps of one round, operating on function-local
 * m[16]/v[16] (donor GS4 formulation, kept bit-identical). */
#define BLAKE512_GS4(a,b,c,d,e,f,a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3){\
	v[ a]+= (m[ e] ^ c_blake512_z[ f]) + v[ b];	v[a1]+= (m[e1] ^ c_blake512_z[f1]) + v[b1];	v[a2]+= (m[e2] ^ c_blake512_z[f2]) + v[b2];	v[a3]+= (m[e3] ^ c_blake512_z[f3]) + v[b3];\
	v[ d] = SWAPDWORDS2(v[ d] ^ v[ a]);	v[d1] = SWAPDWORDS2(v[d1] ^ v[a1]);	v[d2] = SWAPDWORDS2(v[d2] ^ v[a2]);	v[d3] = SWAPDWORDS2(v[d3] ^ v[a3]);\
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];				v[c3]+= v[d3];\
	v[ b] = ROR2(v[b] ^ v[c], 25);		v[b1] = ROR2(v[b1] ^ v[c1], 25);	v[b2] = ROR2(v[b2] ^ v[c2], 25);	v[b3] = ROR2(v[b3] ^ v[c3], 25); \
	v[ a]+= (m[ f] ^ c_blake512_z[ e]) + v[ b];	v[a1]+= (m[f1] ^ c_blake512_z[e1]) + v[b1];	v[a2]+= (m[f2] ^ c_blake512_z[e2]) + v[b2];	v[a3]+= (m[f3] ^ c_blake512_z[e3]) + v[b3];\
	v[ d] = ROR16(v[d] ^ v[a]);		v[d1] = ROR16(v[d1] ^ v[a1]);		v[d2] = ROR16(v[d2] ^ v[a2]);		v[d3] = ROR16(v[d3] ^ v[a3]);\
	v[ c]+= v[ d];				v[c1]+= v[d1];				v[c2]+= v[d2];				v[c3]+= v[d3];\
	v[ b] = ROR2(v[b] ^ v[c], 11);		v[b1] = ROR2(v[b1] ^ v[c1], 11);	v[b2] = ROR2(v[b2] ^ v[c2], 11);	v[b3] = ROR2(v[b3] ^ v[c3], 11);\
}

/* Full 16-round BLAKE-512 compression for a single 64-byte message block
 * (t = 512). Caller prepares m (blake512_message_64) and v
 * (blake512_init_state_64). */
__device__ __forceinline__
void blake512_compress_64(uint2 v[16], const uint2 m[16])
{
	BLAKE512_GS4(0, 4, 8, 12, 0, 1, 1, 5, 9, 13, 2, 3, 2, 6, 10, 14, 4, 5, 3, 7, 11, 15, 6, 7);
	BLAKE512_GS4(0, 5, 10, 15, 8, 9, 1, 6, 11, 12, 10, 11, 2, 7, 8, 13, 12, 13, 3, 4, 9, 14, 14, 15);

	BLAKE512_GS4(0, 4, 8, 12, 14, 10, 1, 5, 9, 13, 4, 8, 2, 6, 10, 14, 9, 15, 3, 7, 11, 15, 13, 6);
	BLAKE512_GS4(0, 5, 10, 15, 1, 12, 1, 6, 11, 12, 0, 2, 2, 7, 8, 13, 11, 7, 3, 4, 9, 14, 5, 3);

	BLAKE512_GS4(0, 4, 8, 12, 11, 8, 1, 5, 9, 13, 12, 0, 2, 6, 10, 14, 5, 2, 3, 7, 11, 15, 15, 13);
	BLAKE512_GS4(0, 5, 10, 15, 10, 14, 1, 6, 11, 12, 3, 6, 2, 7, 8, 13, 7, 1, 3, 4, 9, 14, 9, 4);

	BLAKE512_GS4(0, 4, 8, 12, 7, 9, 1, 5, 9, 13, 3, 1, 2, 6, 10, 14, 13, 12, 3, 7, 11, 15, 11, 14);
	BLAKE512_GS4(0, 5, 10, 15, 2, 6, 1, 6, 11, 12, 5, 10, 2, 7, 8, 13, 4, 0, 3, 4, 9, 14, 15, 8);

	BLAKE512_GS4(0, 4, 8, 12, 9, 0, 1, 5, 9, 13, 5, 7, 2, 6, 10, 14, 2, 4, 3, 7, 11, 15, 10, 15);
	BLAKE512_GS4(0, 5, 10, 15, 14, 1, 1, 6, 11, 12, 11, 12, 2, 7, 8, 13, 6, 8, 3, 4, 9, 14, 3, 13);

	BLAKE512_GS4(0, 4, 8, 12, 2, 12, 1, 5, 9, 13, 6, 10, 2, 6, 10, 14, 0, 11, 3, 7, 11, 15, 8, 3);
	BLAKE512_GS4(0, 5, 10, 15, 4, 13, 1, 6, 11, 12, 7, 5, 2, 7, 8, 13, 15, 14, 3, 4, 9, 14, 1, 9);

	BLAKE512_GS4(0, 4, 8, 12, 12, 5, 1, 5, 9, 13, 1, 15, 2, 6, 10, 14, 14, 13, 3, 7, 11, 15, 4, 10);
	BLAKE512_GS4(0, 5, 10, 15, 0, 7, 1, 6, 11, 12, 6, 3, 2, 7, 8, 13, 9, 2, 3, 4, 9, 14, 8, 11);

	BLAKE512_GS4(0, 4, 8, 12, 13, 11, 1, 5, 9, 13, 7, 14, 2, 6, 10, 14, 12, 1, 3, 7, 11, 15, 3, 9);
	BLAKE512_GS4(0, 5, 10, 15, 5, 0, 1, 6, 11, 12, 15, 4, 2, 7, 8, 13, 8, 6, 3, 4, 9, 14, 2, 10);

	BLAKE512_GS4(0, 4, 8, 12, 6, 15, 1, 5, 9, 13, 14, 9, 2, 6, 10, 14, 11, 3, 3, 7, 11, 15, 0, 8);
	BLAKE512_GS4(0, 5, 10, 15, 12, 2, 1, 6, 11, 12, 13, 7, 2, 7, 8, 13, 1, 4, 3, 4, 9, 14, 10, 5);

	BLAKE512_GS4(0, 4, 8, 12, 10, 2, 1, 5, 9, 13, 8, 4, 2, 6, 10, 14, 7, 6, 3, 7, 11, 15, 1, 5);
	BLAKE512_GS4(0, 5, 10, 15, 15, 11, 1, 6, 11, 12, 9, 14, 2, 7, 8, 13, 3, 12, 3, 4, 9, 14, 13, 0);

	/* rounds 10..15 repeat the schedule of rounds 0..5 */
	BLAKE512_GS4(0, 4, 8, 12, 0, 1, 1, 5, 9, 13, 2, 3, 2, 6, 10, 14, 4, 5, 3, 7, 11, 15, 6, 7);
	BLAKE512_GS4(0, 5, 10, 15, 8, 9, 1, 6, 11, 12, 10, 11, 2, 7, 8, 13, 12, 13, 3, 4, 9, 14, 14, 15);

	BLAKE512_GS4(0, 4, 8, 12, 14, 10, 1, 5, 9, 13, 4, 8, 2, 6, 10, 14, 9, 15, 3, 7, 11, 15, 13, 6);
	BLAKE512_GS4(0, 5, 10, 15, 1, 12, 1, 6, 11, 12, 0, 2, 2, 7, 8, 13, 11, 7, 3, 4, 9, 14, 5, 3);

	BLAKE512_GS4(0, 4, 8, 12, 11, 8, 1, 5, 9, 13, 12, 0, 2, 6, 10, 14, 5, 2, 3, 7, 11, 15, 15, 13);
	BLAKE512_GS4(0, 5, 10, 15, 10, 14, 1, 6, 11, 12, 3, 6, 2, 7, 8, 13, 7, 1, 3, 4, 9, 14, 9, 4);

	BLAKE512_GS4(0, 4, 8, 12, 7, 9, 1, 5, 9, 13, 3, 1, 2, 6, 10, 14, 13, 12, 3, 7, 11, 15, 11, 14);
	BLAKE512_GS4(0, 5, 10, 15, 2, 6, 1, 6, 11, 12, 5, 10, 2, 7, 8, 13, 4, 0, 3, 4, 9, 14, 15, 8);

	BLAKE512_GS4(0, 4, 8, 12, 9, 0, 1, 5, 9, 13, 5, 7, 2, 6, 10, 14, 2, 4, 3, 7, 11, 15, 10, 15);
	BLAKE512_GS4(0, 5, 10, 15, 14, 1, 1, 6, 11, 12, 11, 12, 2, 7, 8, 13, 6, 8, 3, 4, 9, 14, 3, 13);

	BLAKE512_GS4(0, 4, 8, 12, 2, 12, 1, 5, 9, 13, 6, 10, 2, 6, 10, 14, 0, 11, 3, 7, 11, 15, 8, 3);
	BLAKE512_GS4(0, 5, 10, 15, 4, 13, 1, 6, 11, 12, 7, 5, 2, 7, 8, 13, 15, 14, 3, 4, 9, 14, 1, 9);
}

#undef BLAKE512_GS4

/* Message block for a 64-byte input: swab to big-endian view, then the
 * fixed BLAKE-512 padding for a 512-bit message. */
__device__ __forceinline__
void blake512_message_64(uint2 m[16], const uint2 hash[8])
{
	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		m[i] = cuda_swab64_U2(hash[i]);
	m[ 8] = make_uint2(0, 0x80000000);
	m[ 9] = make_uint2(0, 0);
	m[10] = make_uint2(0, 0);
	m[11] = make_uint2(0, 0);
	m[12] = make_uint2(0, 0);
	m[13] = make_uint2(1, 0);
	m[14] = make_uint2(0, 0);
	m[15] = make_uint2(0x200, 0);
}

/* Initial working state for a single 64-byte block: IV, pi constants, and
 * the t = 512 counter XORed into v[12]/v[13]. */
__device__ __forceinline__
void blake512_init_state_64(uint2 v[16])
{
	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		v[i] = c_blake512_h[i];
	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		v[i + 8] = c_blake512_z[i];
	v[12].x ^= 512U;
	v[13].x ^= 512U;
}

/* BLAKE-512 of a 64-byte input, in place, d_hash word order in and out. */
__device__ __forceinline__
void blake512_hash_64(uint2 hash[8])
{
	uint2 m[16], v[16];
	blake512_message_64(m, hash);
	blake512_init_state_64(v);
	blake512_compress_64(v, m);
	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		hash[i] = cuda_swab64_U2(xor3x(v[i], c_blake512_h[i], v[i + 8]));
}

/* Word-3 projection for target-compare-only final stages: same full
 * compression, but finalizes only digest word 3 (bytes 24..31, the word
 * the 64-bit target compare reads). Never feed this into another hash
 * stage or the submit path — the scanhash CPU re-verify stays
 * authoritative. */
__device__ __forceinline__
uint64_t blake512_hash_64_word3(const uint2 hash[8])
{
	uint2 m[16], v[16];
	blake512_message_64(m, hash);
	blake512_init_state_64(v);
	blake512_compress_64(v, m);
	return devectorize(cuda_swab64_U2(xor3x(v[3], c_blake512_h[3], v[11])));
}

#endif /* __CUDACC__ */

#endif /* CUDA_BLAKE512_DEVICE_CUH */
