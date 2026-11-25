// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Shared Keccak-f[1600] primitive library (device side).
 *
 * Round body and truncated final round extracted bit-identically from the
 * alexis-lineage keccak256 kernel (Algo256/cuda_keccak256.cu) and the Pkules
 * sha3t kernel. Every cryptographic primitive is a separately callable
 * building block (docs/coding-guideline.md §3); fused kernels call these
 * instead of re-implementing them.
 *
 * The 80-byte absorb midstate trick (c_mid[17]/c_msg[6] first-round
 * precompute) stays in the consuming kernels: it reads per-algo __constant__
 * symbols, and routing those through a pointer parameter would demote the
 * ld.const accesses. Keep those blocks textually in sync across consumers.
 *
 * Reach goes well past the keccak algos: three of the direct includers are
 * shared stages pulled in across the x-family. Re-benchmark and re-KAT an
 * x-family algo after any change here, not just keccak256/sha3t/sha3d.
 */

#ifndef CUDA_KECCAK_DEVICE_CUH
#define CUDA_KECCAK_DEVICE_CUH

#include <stdint.h>
#include "cuda/helper.cuh"  // ROL2/ROR2/ROL8/ROR8, xor3x, vectorize/devectorize

/* Shared initializers so the host copy and the per-TU device copies can
 * never drift apart. */
#define KECCAK_RC64_INIT { \
	0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull, \
	0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull, \
	0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull, \
	0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull, \
	0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull, \
	0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull }

#define KECCAK_RC2_INIT { \
	{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 }, { 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 }, \
	{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 }, { 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 }, \
	{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 }, { 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 }, \
	{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 }, { 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 }, \
	{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 }, { 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 }, \
	{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 } }

static const uint64_t h_keccak_rc[24] = KECCAK_RC64_INIT;

#ifdef __CUDACC__

/* Statically initialized __constant__ copy: no cudaMemcpyToSymbol needed,
 * one 192-byte copy per translation unit. */
static __constant__ uint2 c_keccak_rc[24] = KECCAK_RC2_INIT;

/* chi: a ^ (~b & c) — single LOP3.LUT on sm_50+. */
__device__ __forceinline__
uint2 keccak_chi(const uint2 a, const uint2 b, const uint2 c)
{
	uint2 r;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(r.x) : "r"(a.x), "r"(b.x), "r"(c.x)); // 0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
	asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(r.y) : "r"(a.y), "r"(b.y), "r"(c.y));
#else
	r = a ^ (~b) & c;
#endif
	return r;
}

/* theta column parity: 64-bit xor chain keeps the operands paired for the
 * scheduler (donor formulation). */
__device__ __forceinline__
uint64_t keccak_xor5(const uint64_t a, const uint64_t b, const uint64_t c,
	const uint64_t d, const uint64_t e)
{
	uint64_t r;
	asm("xor.b64 %0, %1, %2;" : "=l"(r) : "l"(d), "l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(r) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(r) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(r) : "l"(a));
	return r;
}

/* rho-pi, chi and iota — the round tail shared by the generic round and
 * the keccak512 absorb specialization below. */
__device__ __forceinline__
void keccak_rhopi_chi_iota(uint2 s[25], const uint2 rc)
{
	uint2 v, w;

	/* rho-pi: b[..] = rotl(a[..], ..) */
	v = s[1];
	s[ 1]=ROL2(s[ 6],44); s[ 6]=ROL2(s[ 9],20); s[ 9]=ROL2(s[22],61); s[22]=ROL2(s[14],39);
	s[14]=ROL2(s[20],18); s[20]=ROL2(s[ 2],62); s[ 2]=ROL2(s[12],43); s[12]=ROL2(s[13],25);
	s[13]=ROL8(s[19]);    s[19]=ROR8(s[23]);    s[23]=ROL2(s[15],41); s[15]=ROL2(s[ 4],27);
	s[ 4]=ROL2(s[24],14); s[24]=ROL2(s[21], 2); s[21]=ROL2(s[ 8],55); s[ 8]=ROL2(s[16],45);
	s[16]=ROL2(s[ 5],36); s[ 5]=ROL2(s[ 3],28); s[ 3]=ROL2(s[18],21); s[18]=ROL2(s[17],15);
	s[17]=ROL2(s[11],10); s[11]=ROL2(s[ 7], 6); s[ 7]=ROL2(s[10], 3); s[10]=ROL2(v, 1);

	/* chi */
	#pragma unroll 5
	for (int j = 0; j < 25; j += 5) {
		v = s[j]; w = s[j+1];
		s[j]   = keccak_chi(v, w, s[j+2]);
		s[j+1] = keccak_chi(w, s[j+2], s[j+3]);
		s[j+2] = keccak_chi(s[j+2], s[j+3], s[j+4]);
		s[j+3] = keccak_chi(s[j+3], s[j+4], v);
		s[j+4] = keccak_chi(s[j+4], v, w);
	}

	/* iota */
	s[0] ^= rc;
}

/* One full Keccak-f[1600] round: theta, rho-pi, chi, iota. */
__device__ __forceinline__
void keccak_round(uint2 s[25], const uint2 rc)
{
	uint2 t[5], u[5];

	/* theta: column parities and d[i] = c[i+4] ^ rotl(c[i+1],1) */
	#pragma unroll 5
	for (int j = 0; j < 5; j++)
		t[j] = vectorize(keccak_xor5(devectorize(s[j]), devectorize(s[j+5]),
		                             devectorize(s[j+10]), devectorize(s[j+15]),
		                             devectorize(s[j+20])));
	#pragma unroll 5
	for (int j = 0; j < 5; j++)
		u[j] = ROL2(t[j], 1);

	s[ 4]=xor3x(s[ 4],t[3],u[0]); s[ 9]=xor3x(s[ 9],t[3],u[0]); s[14]=xor3x(s[14],t[3],u[0]); s[19]=xor3x(s[19],t[3],u[0]); s[24]=xor3x(s[24],t[3],u[0]);
	s[ 0]=xor3x(s[ 0],t[4],u[1]); s[ 5]=xor3x(s[ 5],t[4],u[1]); s[10]=xor3x(s[10],t[4],u[1]); s[15]=xor3x(s[15],t[4],u[1]); s[20]=xor3x(s[20],t[4],u[1]);
	s[ 1]=xor3x(s[ 1],t[0],u[2]); s[ 6]=xor3x(s[ 6],t[0],u[2]); s[11]=xor3x(s[11],t[0],u[2]); s[16]=xor3x(s[16],t[0],u[2]); s[21]=xor3x(s[21],t[0],u[2]);
	s[ 2]=xor3x(s[ 2],t[1],u[3]); s[ 7]=xor3x(s[ 7],t[1],u[3]); s[12]=xor3x(s[12],t[1],u[3]); s[17]=xor3x(s[17],t[1],u[3]); s[22]=xor3x(s[22],t[1],u[3]);
	s[ 3]=xor3x(s[ 3],t[2],u[4]); s[ 8]=xor3x(s[ 8],t[2],u[4]); s[13]=xor3x(s[13],t[2],u[4]); s[18]=xor3x(s[18],t[2],u[4]); s[23]=xor3x(s[23],t[2],u[4]);

	keccak_rhopi_chi_iota(s, rc);
}

/* Full 24-round Keccak-f[1600] permutation. Caller initializes s[0..24]
 * (absorbed message, padding and end-of-rate bit included). */
__device__ __forceinline__
void keccakf1600_full(uint2 s[25])
{
	#pragma unroll 24
	for (int i = 0; i < 24; i++)
		keccak_round(s, c_keccak_rc[i]);
}

/* Truncated final round: state has been advanced through round 22 (23 of 24
 * rounds); returns lane 3 after round 23 without computing the other 24
 * lanes (iota does not touch lane 3). Only for kernels whose 64-bit target
 * compare reads lane 3 — never feed this into another hash stage; the
 * scanhash CPU re-verify stays authoritative. */
__device__ __forceinline__
uint2 keccak_final_lane3(const uint2 s[25])
{
	uint2 t[5];
	#pragma unroll 5
	for (int j = 0; j < 5; j++)
		t[j] = xor3x(xor3x(s[j], s[j+5], s[j+10]), s[j+15], s[j+20]);
	/* round-23 output lane 3 = chi(b3, b4, b0) with
	 * b3 = rotl(a[18] ^ d3, 21), b4 = rotl(a[24] ^ d4, 14), b0 = a[0] ^ d0 */
	const uint2 b4 = ROL2(xor3x(s[24], t[3], ROL2(t[0], 1)), 14);
	const uint2 b3 = ROL2(xor3x(s[18], t[2], ROL2(t[4], 1)), 21);
	const uint2 b0 = xor3x(s[ 0], t[4], ROL2(t[1], 1));
	return keccak_chi(b3, b4, b0);
}

/* ------------------------------------------------------------------------
 * Keccak-512 (rate 72) building blocks for 64-byte chained inputs — the
 * x-family stage function. Extracted bit-identically from the alexis
 * keccak512 kernels (algos/stages/cuda_keccak512.cu).
 * ------------------------------------------------------------------------ */

/* Round 0 with the absorb structure folded into theta: lanes 0..7 hold the
 * 64-byte message, lane 8 the padding (0x01 start, 0x80 end-of-rate),
 * lanes 9..24 are implicitly zero and must not be read before this call. */
__device__ __forceinline__
void keccak512_absorb_round_64(uint2 s[25])
{
	uint2 t[5], u[5];

	t[0] = vectorize(devectorize(s[0]) ^ devectorize(s[5]));
	t[1] = vectorize(devectorize(s[1]) ^ devectorize(s[6]));
	t[2] = vectorize(devectorize(s[2]) ^ devectorize(s[7]));
	t[3] = vectorize(devectorize(s[3]) ^ devectorize(s[8]));
	t[4] = s[4];

	#pragma unroll 5
	for (int j = 0; j < 5; j++)
		u[j] = ROL2(t[j], 1);

	s[ 4] = xor3x(s[ 4], t[3], u[0]);
	s[24] = s[19] = s[14] = s[ 9] = t[3] ^ u[0];
	s[ 0] = xor3x(s[ 0], t[4], u[1]);
	s[ 5] = xor3x(s[ 5], t[4], u[1]);
	s[20] = s[15] = s[10] = t[4] ^ u[1];
	s[ 1] = xor3x(s[ 1], t[0], u[2]);
	s[ 6] = xor3x(s[ 6], t[0], u[2]);
	s[21] = s[16] = s[11] = t[0] ^ u[2];
	s[ 2] = xor3x(s[ 2], t[1], u[3]);
	s[ 7] = xor3x(s[ 7], t[1], u[3]);
	s[22] = s[17] = s[12] = t[1] ^ u[3];
	s[ 3] = xor3x(s[ 3], t[2], u[4]);
	s[ 8] = xor3x(s[ 8], t[2], u[4]);
	s[23] = s[18] = s[13] = t[2] ^ u[4];

	keccak_rhopi_chi_iota(s, c_keccak_rc[0]);
}

/* Truncated round 23: computes only the 8 digest lanes s[0..7] (theta,
 * rho-pi and chi restricted to the lanes they need; iota on s[0]). The
 * other 17 lanes are left stale — only for the last permutation of a
 * keccak512 whose output is the 64-byte digest. */
__device__ __forceinline__
void keccak512_output_round(uint2 s[25])
{
	uint2 t[5], u[5], v, w;

	#pragma unroll 5
	for (int j = 0; j < 5; j++)
		t[j] = xor3x(xor3x(s[j], s[j+5], s[j+10]), s[j+15], s[j+20]);
	#pragma unroll 5
	for (int j = 0; j < 5; j++)
		u[j] = ROL2(t[j], 1);

	s[ 9] = xor3x(s[ 9], t[3], u[0]);
	s[24] = xor3x(s[24], t[3], u[0]);
	s[ 0] = xor3x(s[ 0], t[4], u[1]);
	s[10] = xor3x(s[10], t[4], u[1]);
	s[ 6] = xor3x(s[ 6], t[0], u[2]);
	s[16] = xor3x(s[16], t[0], u[2]);
	s[12] = xor3x(s[12], t[1], u[3]);
	s[22] = xor3x(s[22], t[1], u[3]);
	s[ 3] = xor3x(s[ 3], t[2], u[4]);
	s[18] = xor3x(s[18], t[2], u[4]);

	/* rho-pi, only b0..b9 (b0 = a0, unrotated) */
	s[ 1] = ROL2(s[ 6], 44);
	s[ 2] = ROL2(s[12], 43);
	s[ 5] = ROL2(s[ 3], 28);
	s[ 7] = ROL2(s[10],  3);
	s[ 3] = ROL2(s[18], 21);
	s[ 4] = ROL2(s[24], 14);
	s[ 6] = ROL2(s[ 9], 20);
	s[ 8] = ROL2(s[16], 45);
	s[ 9] = ROL2(s[22], 61);

	/* chi rows 0 and 1 (partial) */
	v = s[0]; w = s[1];
	s[0] = keccak_chi(v, w, s[2]);
	s[1] = keccak_chi(w, s[2], s[3]);
	s[2] = keccak_chi(s[2], s[3], s[4]);
	s[3] = keccak_chi(s[3], s[4], v);
	s[4] = keccak_chi(s[4], v, w);
	v = s[5]; w = s[6];
	s[5] = keccak_chi(v, w, s[7]);
	s[6] = keccak_chi(w, s[7], s[8]);
	s[7] = keccak_chi(s[7], s[8], s[9]);

	/* iota */
	s[0] ^= c_keccak_rc[23];
}

/* Keccak-512 of a 64-byte input, in place, d_hash word order in and out
 * (keccak is little-endian native — no byte swabbing). */
__device__ __forceinline__
void keccak512_hash_64(uint2 hash[8])
{
	uint2 s[25];

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		s[i] = hash[i];
	s[8] = make_uint2(1, 0x80000000);

	keccak512_absorb_round_64(s);
	#pragma unroll 4
	for (int i = 1; i < 23; i++)
		keccak_round(s, c_keccak_rc[i]);
	keccak512_output_round(s);

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		hash[i] = s[i];
}

/* Lane-3 projection for target-compare-only final stages: full 23 rounds,
 * then keccak_final_lane3 (digest bytes 24..31). Never feed this into
 * another hash stage or the submit path — the scanhash CPU re-verify
 * stays authoritative. */
__device__ __forceinline__
uint2 keccak512_hash_64_lane3(const uint2 hash[8])
{
	uint2 s[25];

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		s[i] = hash[i];
	s[8] = make_uint2(1, 0x80000000);

	keccak512_absorb_round_64(s);
	#pragma unroll 4
	for (int i = 1; i < 23; i++)
		keccak_round(s, c_keccak_rc[i]);
	return keccak_final_lane3(s);
}

#endif /* __CUDACC__ */

#endif /* CUDA_KECCAK_DEVICE_CUH */
