// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Unit self-test for cuda/sha256_device.cuh (docs/coding-guideline.md §7, layer 1):
 * exercises each header building block against the OpenSSL CPU reference and
 * a known-answer vector (a real Bitcoin mainnet block header, so the expected
 * digest is anchored outside this codebase). Runs once per process at algo
 * init and is FAIL-CLOSED via cuda/selftest_gate.cuh.
 */

#include <string.h>

#include <miner.h>
#include "cuda/helper.cuh"
#include <openssl/sha.h>

#include "cuda/selftest_gate.cuh"
#include "cuda/sha256_device.cuh"

/* Bitcoin block 957533 (2026-07, empty block): 80-byte header and its
 * sha256d digest (block hash 00000000000000000000e9b8431c31db...). */
static const uint8_t kat_header[80] = {
	0x00, 0xe0, 0xff, 0x3f, 0x43, 0xd4, 0x81, 0xb0, 0x07, 0x92, 0x8a, 0x01, 0xbd, 0x44, 0xd1, 0x65,
	0x57, 0xc3, 0xb7, 0xe1, 0xd5, 0xdb, 0x3a, 0x5f, 0x6f, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x10, 0x9b, 0xa8, 0x64, 0xa7, 0x69, 0x50, 0x10, 0xb1, 0x92, 0x6e, 0x6d,
	0xc4, 0xaf, 0xf1, 0x17, 0xa0, 0xe7, 0x7c, 0xc8, 0xbe, 0x78, 0x09, 0x92, 0x3e, 0x5e, 0xff, 0x80,
	0x4c, 0x36, 0x8e, 0xd2, 0x64, 0xe9, 0x51, 0x6a, 0x42, 0x1a, 0x02, 0x17, 0xad, 0x4e, 0xf3, 0x29
};
static const uint32_t kat_digest_be[8] = {
	0xb44c277b, 0xc4de100b, 0xd02f2276, 0xaef4bcc0,
	0xdb311c43, 0xb8e90000, 0x00000000, 0x00000000
};

/* Device buffer layout: in[0..19] header words, in[20..27] round-3 prehash
 * state, in[28..31] preextend words; out[0..7] full second-hash state,
 * out[8..9] truncated words 6/7, out[10..17] first hash via the prehash
 * path (must equal the full-transform first hash). */
#define SELFTEST_IN_WORDS  32
#define SELFTEST_OUT_WORDS 18

/* GPU side: double SHA-256 of an 80-byte header (20 big-endian words) via
 * sha256_transform_full, sha256_final_to_target over the second hash, and
 * the block-2 transform again via sha256_transform_80_from_pre4. */
__global__ __launch_bounds__(1, 1)
void sha256_selftest_gpu(const uint32_t *in32, uint32_t *out)
{
	uint32_t in[16], ms[8], st[8];

	// first block -> midstate
	#pragma unroll
	for (int i = 0; i < 16; i++) in[i] = in32[i];
	#pragma unroll
	for (int i = 0; i < 8; i++) ms[i] = c_sha256_H[i];
	sha256_transform_full(in, ms, c_sha256_K);

	// second block: header bytes 64..79, padding, length 0x280 bits
	#pragma unroll
	for (int i = 0; i < 4; i++) in[i] = in32[16 + i];
	in[4] = 0x80000000U;
	#pragma unroll
	for (int i = 5; i < 15; i++) in[i] = 0;
	in[15] = 0x280;
	#pragma unroll
	for (int i = 0; i < 8; i++) st[i] = ms[i];
	sha256_transform_full(in, st, c_sha256_K);

	// same block via the round-4 resume path (word 3 = "nonce")
	{
		uint32_t dat[16], st1[8];
		dat[4] = 0x80000000U;
		#pragma unroll
		for (int i = 5; i < 15; i++) dat[i] = 0;
		dat[15] = 0x280;
		sha256_transform_80_from_pre4(dat, in32 + 20, in32 + 28, in32[19], ms, st1, c_sha256_K);
		#pragma unroll
		for (int i = 0; i < 8; i++) out[10 + i] = st1[i];
	}

	// second SHA-256 over the 32-byte digest: full transform...
	uint32_t in2[16], st2[8];
	#pragma unroll
	for (int i = 0; i < 8; i++) in2[i] = st[i];
	in2[8] = 0x80000000U;
	#pragma unroll
	for (int i = 9; i < 15; i++) in2[i] = 0;
	in2[15] = 0x100;
	#pragma unroll
	for (int i = 0; i < 8; i++) st2[i] = c_sha256_H[i];
	sha256_transform_full(in2, st2, c_sha256_K);
	#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = st2[i];

	// ...and truncated: only words 6/7 are defined afterwards
	#pragma unroll
	for (int i = 0; i < 8; i++) in2[i] = st[i];
	in2[8] = 0x80000000U;
	#pragma unroll
	for (int i = 9; i < 15; i++) in2[i] = 0;
	in2[15] = 0x100;
	#pragma unroll
	for (int i = 0; i < 8; i++) st2[i] = c_sha256_H[i];
	sha256_final_to_target(in2, st2, c_sha256_K);
	out[8] = st2[6];
	out[9] = st2[7];
}

/* Run one 80-byte vector through OpenSSL, the host header path and the GPU
 * kernel; expected_be (optional) additionally pins the digest to a known
 * answer. Returns true when all agree. */
static bool sha256_selftest_vector(const uint8_t m[80], const uint32_t *expected_be)
{
	// CPU reference: OpenSSL double SHA-256
	uint8_t ref[32];
	{
		uint8_t h1[32];
		SHA256_CTX ctx;
		SHA256_Init(&ctx); SHA256_Update(&ctx, m, 80); SHA256_Final(h1, &ctx);
		SHA256_Init(&ctx); SHA256_Update(&ctx, h1, 32); SHA256_Final(ref, &ctx);
	}

	// host header path: sha256_midstate_host + sha256_transform_full
	uint32_t data20[20], in[16], ms[8], st[8];
	for (int i = 0; i < 20; i++) data20[i] = be32dec(m + 4*i);
	sha256_midstate_host(data20, ms);
	for (int i = 0; i < 4; i++) in[i] = data20[16 + i];
	in[4] = 0x80000000U;
	for (int i = 5; i < 15; i++) in[i] = 0;
	in[15] = 0x280;
	for (int i = 0; i < 8; i++) st[i] = ms[i];
	sha256_transform_full(in, st, h_sha256_K);

	uint32_t in2[16], st2[8];
	for (int i = 0; i < 8; i++) in2[i] = st[i];
	in2[8] = 0x80000000U;
	for (int i = 9; i < 15; i++) in2[i] = 0;
	in2[15] = 0x100;
	for (int i = 0; i < 8; i++) st2[i] = h_sha256_H[i];
	sha256_transform_full(in2, st2, h_sha256_K);

	uint8_t hostdig[32];
	for (int i = 0; i < 8; i++) be32enc(hostdig + 4*i, st2[i]);
	bool ok = (memcmp(hostdig, ref, 32) == 0);
	if (expected_be)
		ok = ok && (memcmp(st2, expected_be, 32) == 0);

	// prehash inputs for the device round-4 resume path: block-2 template
	// with word 3 treated as the nonce
	uint32_t inbuf[SELFTEST_IN_WORDS];
	memcpy(inbuf, data20, sizeof(data20));
	{
		uint32_t w[16] = { 0 };
		for (int i = 0; i < 3; i++) w[i] = data20[16 + i];
		w[4] = 0x80000000U;
		w[15] = 0x280;
		sha256_prehash_split_host(ms, w, 3, inbuf + 20);
		sha256_preextend_w3_host(w, inbuf + 28);
	}

	// device path: full transform must match the host, truncated words 6/7
	// must match the full transform, prehash-path first hash must match the
	// full-transform first hash
	uint32_t out[SELFTEST_OUT_WORDS] = { 0 };
	uint32_t *d_buf = NULL;
	bool gpu_ok = false;
	if (cudaMalloc(&d_buf, sizeof(inbuf) + sizeof(out)) != cudaSuccess) {
		selftest_cuda_fault(); // could not run != produced a wrong digest
	} else {
		uint32_t *d_out = d_buf + SELFTEST_IN_WORDS;
		gpu_ok = (cudaMemcpy(d_buf, inbuf, sizeof(inbuf), cudaMemcpyHostToDevice) == cudaSuccess);
		sha256_selftest_gpu <<<1, 1>>> (d_buf, d_out);
		gpu_ok = gpu_ok && (cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost) == cudaSuccess);
		cudaFree(d_buf);
	}
	if (!gpu_ok)
		selftest_cuda_fault(); // a failed copy is not a wrong digest either
	gpu_ok = gpu_ok && (memcmp(out, st2, 32) == 0)
	                && out[8] == st2[6] && out[9] == st2[7]
	                && (memcmp(out + 10, st, 32) == 0);

	return ok && gpu_ok;
}

__host__
bool sha256_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	// synthetic vector (m[i] = i) plus a real mainnet header with pinned digest
	uint8_t m[80];
	for (int i = 0; i < 80; i++) m[i] = (uint8_t) i;
	const bool synth_ok = sha256_selftest_vector(m, NULL);
	const bool kat_ok = sha256_selftest_vector(kat_header, kat_digest_be);

	passed = synth_ok && kat_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "SHA256 device-library self-test FAILED (synthetic %d kat %d)", (int) synth_ok, (int) kat_ok);
	return selftest_gate(thr_id, "SHA256", passed);
}
