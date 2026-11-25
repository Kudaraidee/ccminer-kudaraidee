// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Unit self-test for cuda/keccak_device.cuh (docs/coding-guideline.md §7,
 * layer 1): exercises the shared permutation building blocks against the sph
 * CPU references and two known-answer vectors anchored outside this codebase
 * (the NIST SHA3-256 and Keccak-256 empty-string digests). Includes a
 * negative test (single flipped input bit must change the digest) so the
 * check can never pass vacuously. Runs once per process at algo init and is
 * FAIL-CLOSED via cuda/selftest_gate.cuh.
 */

#include <string.h>

#include <miner.h>
#include "cuda/helper.cuh"

extern "C" {
#include "sph/sph_keccak.h"  /* sph_keccak256* — 0x01 padding */
/* sph/sph_sha3d.h (0x06 padding) shares sph_keccak.h's include guard —
 * the header is a verbatim copy — so declare its functions directly;
 * the context layout is identical. */
void sph_sha3d256_init(void *cc);
void sph_sha3d256(void *cc, const void *data, size_t len);
void sph_sha3d256_close(void *cc, void *dst);
}

#include "cuda/selftest_gate.cuh"
#include "cuda/keccak_device.cuh"

/* NIST FIPS 202: SHA3-256("") */
static const uint8_t kat_sha3_256_empty[32] = {
	0xa7, 0xff, 0xc6, 0xf8, 0xbf, 0x1e, 0xd7, 0x66, 0x51, 0xc1, 0x47, 0x56, 0xa0, 0x61, 0xd6, 0x62,
	0xf5, 0x80, 0xff, 0x4d, 0xe4, 0x3b, 0x49, 0xfa, 0x82, 0xd8, 0x0a, 0x4b, 0x80, 0xf8, 0x43, 0x4a
};
/* Keccak-256("") (pre-NIST padding; the well-known Ethereum empty hash) */
static const uint8_t kat_keccak_256_empty[32] = {
	0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
	0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b, 0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70
};

/* Device output layout (uint64 lanes, little-endian == digest bytes):
 * out[0..3]  keccak256(header80)             — 0x01 padding, one permutation
 * out[4..7]  sha3d(header80) = SHA3-256^2    — 0x06 padding, two permutations
 * out[8]     lane 3 of the second sha3d permutation via 23 shared rounds +
 *            keccak_final_lane3 (must equal out[7])
 * out[9..12] SHA3-256("")                    — pinned NIST KAT
 * out[13..16] Keccak-256("")                 — pinned KAT
 * out[17..20] sha3t(header80) = SHA3-256^3   — triple chain (sha3t structure)
 */
#define SELFTEST_OUT_LANES 21

__device__ static void keccak_selftest_absorb80(uint2 s[25], const uint2 *m, const uint32_t pad)
{
	#pragma unroll
	for (int i = 0; i < 10; i++) s[i] = m[i];
	s[10] = make_uint2(pad, 0);
	#pragma unroll
	for (int i = 11; i < 25; i++) s[i] = make_uint2(0, 0);
	s[16] = make_uint2(0, 0x80000000);
}

__global__ __launch_bounds__(32, 1)
void keccak_selftest_gpu(const uint2 *m80, uint2 *out)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	uint2 s[25];

	/* keccak256(header80), 0x01 padding */
	keccak_selftest_absorb80(s, m80, 0x01);
	keccakf1600_full(s);
	#pragma unroll
	for (int i = 0; i < 4; i++) out[i] = s[i];

	/* sha3d(header80): first SHA3-256, 0x06 padding */
	keccak_selftest_absorb80(s, m80, 0x06);
	keccakf1600_full(s);
	uint2 h[4];
	#pragma unroll
	for (int i = 0; i < 4; i++) h[i] = s[i];

	/* second SHA3-256 over the 32-byte digest */
	#pragma unroll
	for (int i = 0; i < 4; i++) s[i] = h[i];
	s[ 4] = make_uint2(0x06, 0);
	#pragma unroll
	for (int i = 5; i < 25; i++) s[i] = make_uint2(0, 0);
	s[16] = make_uint2(0, 0x80000000);
	keccakf1600_full(s);
	#pragma unroll
	for (int i = 0; i < 4; i++) out[4 + i] = s[i];

	/* third SHA3-256 (sha3t chain structure) over the second digest */
	s[ 4] = make_uint2(0x06, 0);
	#pragma unroll
	for (int i = 5; i < 25; i++) s[i] = make_uint2(0, 0);
	s[16] = make_uint2(0, 0x80000000);
	keccakf1600_full(s);
	#pragma unroll
	for (int i = 0; i < 4; i++) out[17 + i] = s[i];

	/* same second permutation, truncated: 23 rounds + final lane 3 */
	#pragma unroll
	for (int i = 0; i < 4; i++) s[i] = h[i];
	s[ 4] = make_uint2(0x06, 0);
	#pragma unroll
	for (int i = 5; i < 25; i++) s[i] = make_uint2(0, 0);
	s[16] = make_uint2(0, 0x80000000);
	#pragma unroll
	for (int i = 0; i < 23; i++)
		keccak_round(s, c_keccak_rc[i]);
	out[8] = keccak_final_lane3(s);

	/* SHA3-256(""): empty rate block, pad byte at position 0 */
	#pragma unroll
	for (int i = 0; i < 25; i++) s[i] = make_uint2(0, 0);
	s[ 0] = make_uint2(0x06, 0);
	s[16] = make_uint2(0, 0x80000000);
	keccakf1600_full(s);
	#pragma unroll
	for (int i = 0; i < 4; i++) out[9 + i] = s[i];

	/* Keccak-256("") */
	#pragma unroll
	for (int i = 0; i < 25; i++) s[i] = make_uint2(0, 0);
	s[ 0] = make_uint2(0x01, 0);
	s[16] = make_uint2(0, 0x80000000);
	keccakf1600_full(s);
	#pragma unroll
	for (int i = 0; i < 4; i++) out[13 + i] = s[i];
}

/* Run one 80-byte vector through the GPU building blocks and the sph CPU
 * references. digests_out (optional, 8 bytes header) receives the GPU sha3d
 * digest for the caller's negative test. */
static bool keccak_selftest_vector(const uint8_t m[80], bool check_empty_kats, uint8_t *sha3d_digest_out)
{
	/* CPU references */
	uint8_t ref_keccak[32], ref_sha3d[32], ref_sha3t[32], buf[32];
	sph_keccak_context ctx;

	sph_keccak256_init(&ctx);
	sph_keccak256(&ctx, m, 80);
	sph_keccak256_close(&ctx, ref_keccak);

	sph_sha3d256_init(&ctx);
	sph_sha3d256(&ctx, m, 80);
	sph_sha3d256_close(&ctx, buf);
	sph_sha3d256_init(&ctx);
	sph_sha3d256(&ctx, buf, 32);
	sph_sha3d256_close(&ctx, ref_sha3d);
	sph_sha3d256_init(&ctx);
	sph_sha3d256(&ctx, ref_sha3d, 32);
	sph_sha3d256_close(&ctx, ref_sha3t);

	/* GPU */
	uint64_t out[SELFTEST_OUT_LANES] = { 0 };
	uint2 *d_buf = NULL;
	bool gpu_ok = false;
	if (cudaMalloc(&d_buf, 10 * sizeof(uint2) + sizeof(out)) != cudaSuccess) {
		selftest_cuda_fault(); // could not run != produced a wrong digest
	} else {
		uint2 *d_out = d_buf + 10;
		gpu_ok = (cudaMemcpy(d_buf, m, 80, cudaMemcpyHostToDevice) == cudaSuccess);
		keccak_selftest_gpu <<<1, 32>>> (d_buf, d_out);
		gpu_ok = gpu_ok && (cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost) == cudaSuccess);
		cudaFree(d_buf);
	}

	if (!gpu_ok)
		selftest_cuda_fault(); // a failed copy is not a wrong digest either

	bool ok = gpu_ok;
	ok = ok && (memcmp(out + 0, ref_keccak, 32) == 0);   /* keccak256 vs sph */
	ok = ok && (memcmp(out + 4, ref_sha3d, 32) == 0);    /* sha3d vs sph */
	ok = ok && (memcmp(out + 17, ref_sha3t, 32) == 0);   /* sha3t chain vs sph */
	ok = ok && (out[8] == out[7]);                       /* truncated == full lane 3 */
	if (check_empty_kats) {
		ok = ok && (memcmp(out + 9,  kat_sha3_256_empty,   32) == 0);
		ok = ok && (memcmp(out + 13, kat_keccak_256_empty, 32) == 0);
	}
	if (sha3d_digest_out)
		memcpy(sha3d_digest_out, out + 4, 32);
	return ok;
}

__host__
bool keccak_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	uint8_t m[80], digest_a[32], digest_b[32];
	for (int i = 0; i < 80; i++) m[i] = (uint8_t) i;

	const bool pos_ok = keccak_selftest_vector(m, true, digest_a);

	/* negative test: one flipped bit must change the sha3d digest (and the
	 * flipped vector must still agree with the CPU reference) */
	m[0] ^= 0x01;
	const bool flip_ok = keccak_selftest_vector(m, false, digest_b);
	const bool neg_ok = flip_ok && (memcmp(digest_a, digest_b, 32) != 0);

	passed = pos_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "Keccak device-library self-test FAILED (kat %d negative %d)", (int) pos_ok, (int) neg_ok);
	return selftest_gate(thr_id, "Keccak", passed);
}
