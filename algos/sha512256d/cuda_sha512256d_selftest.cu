// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Unit self-test for cuda/sha512_device.cuh (docs/coding-guideline.md §7,
 * layer 1): exercises the SHA-512/256 transform against the FIPS 180-4
 * "abc" known-answer vector (anchored outside this codebase), cross-checks
 * the header double-hash against the independent sph_sha512-based CPU
 * reference, and runs a one-time negative test (bit flip must change the
 * digest — proves the test isn't vacuous). Runs once per device at algo init
 * and is FAIL-CLOSED via cuda/selftest_gate.cuh.
 */

#include <string.h>

#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/selftest_gate.cuh"
#include "cuda/sha512_device.cuh"

/* FIPS 180-4 SHA-512/256("abc") = 53048E26...07E7AF23, as state words. */
static const uint64_t kat_abc_digest[4] = {
	0x53048E2681941EF9ULL, 0x9B2E29B76B4C7DABULL,
	0xE4C2D0C634FC6D46ULL, 0xE0E2F13107E7AF23ULL
};

/* sph-based double SHA-512/256 (defined in sha512256d.cu — the scanhash
 * revalidation path, an implementation independent of the .cuh). */
extern "C" void sha512256d_hash(void *output, const void *input);

/* Device buffer layout: in[0..15] "abc" message block, in[16..25] header
 * words w0..w9 (nonce already merged into w9), in[26..35] host prehash;
 * out[0..3] abc digest, out[4..11] full first-hash state, out[12..15]
 * second-hash digest, out[16..23] first hash via the round-9 resume path
 * (must equal the full-transform first hash). */
#define SELFTEST_IN_WORDS  36
#define SELFTEST_OUT_WORDS 24

/* GPU side: one SHA-512/256 block ("abc") plus the header double hash,
 * mirroring the production kernel's message construction exactly. */
__global__ __launch_bounds__(1, 1)
void sha512256d_selftest_gpu(const uint64_t *in, uint64_t *out)
{
	uint64_t w[16], st[8];

	// "abc": single padded block
	#pragma unroll
	for (int i = 0; i < 16; i++) w[i] = in[i];
	sha512_256_init_state(st, c_sha512_256_H);
	sha512_transform_full(w, st, c_sha512_K);
	#pragma unroll
	for (int i = 0; i < 4; i++) out[i] = st[i];

	// hash1 over the 80-byte header (w10..w15 = padding + 640-bit length)
	#pragma unroll
	for (int i = 0; i < 10; i++) w[i] = in[16 + i];
	w[10] = 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 11; i < 15; i++) w[i] = 0;
	w[15] = 640;
	sha512_256_init_state(st, c_sha512_256_H);
	sha512_transform_full(w, st, c_sha512_K);
	#pragma unroll
	for (int i = 0; i < 8; i++) out[4 + i] = st[i];

	// hash2 over the 32-byte truncation
	#pragma unroll
	for (int i = 0; i < 4; i++) w[i] = st[i];
	w[4] = 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 5; i < 15; i++) w[i] = 0;
	w[15] = 256;
	sha512_256_init_state(st, c_sha512_256_H);
	sha512_transform_full(w, st, c_sha512_K);
	#pragma unroll
	for (int i = 0; i < 4; i++) out[12 + i] = st[i];

	// hash1 again via the round-9 resume path (the production kernel's path)
	#pragma unroll
	for (int i = 0; i < 10; i++) w[i] = in[16 + i];
	w[10] = 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 11; i < 15; i++) w[i] = 0;
	w[15] = 640;
	sha512_256_init_state(st, c_sha512_256_H);
	sha512_transform_80_from_pre9(w, in + 26, st, c_sha512_K);
	#pragma unroll
	for (int i = 0; i < 8; i++) out[16 + i] = st[i];
}

/* Host-side double SHA-512/256 of an 80-byte header via the .cuh host path;
 * q1out (optional) receives the full first-hash state. */
static void sha512256d_host_hash80(const uint8_t m[80], uint64_t q[4], uint64_t q1out[8])
{
	uint64_t w[16], st[8];

	for (int i = 0; i < 10; i++) {
		w[i] = 0;
		for (int j = 0; j < 8; j++)
			w[i] = (w[i] << 8) | m[8*i + j];
	}
	w[10] = 0x8000000000000000ULL;
	for (int i = 11; i < 15; i++) w[i] = 0;
	w[15] = 640;
	sha512_256_init_state(st, h_sha512_256_H);
	sha512_transform_full(w, st, h_sha512_K);
	if (q1out)
		memcpy(q1out, st, sizeof(uint64_t[8]));

	for (int i = 0; i < 4; i++) w[i] = st[i];
	w[4] = 0x8000000000000000ULL;
	for (int i = 5; i < 15; i++) w[i] = 0;
	w[15] = 256;
	sha512_256_init_state(st, h_sha512_256_H);
	sha512_transform_full(w, st, h_sha512_K);
	memcpy(q, st, sizeof(uint64_t[4]));
}

__host__
bool sha512256d_device_selftest(int thr_id)
{
	/* Per-device, not per-process: the GPU legs below only prove the device
	 * that ran them, so a second card must be tested on its own. */
	static bool tested[MAX_GPUS] = { 0 }, passed[MAX_GPUS] = { 0 };
	if (tested[thr_id]) return passed[thr_id];
	tested[thr_id] = true;

	// --- "abc" KAT, host path ---
	uint64_t w[16], st[8];
	w[0] = 0x6162638000000000ULL; // "abc" + 0x80 pad
	for (int i = 1; i < 15; i++) w[i] = 0;
	w[15] = 24; // bit length
	sha512_256_init_state(st, h_sha512_256_H);
	sha512_transform_full(w, st, h_sha512_K);
	const bool abc_ok = (memcmp(st, kat_abc_digest, sizeof(kat_abc_digest)) == 0);

	// --- negative test: one flipped message bit must change the digest ---
	w[0] = 0x6162638000000000ULL ^ (1ULL << 63);
	for (int i = 1; i < 15; i++) w[i] = 0;
	w[15] = 24;
	sha512_256_init_state(st, h_sha512_256_H);
	sha512_transform_full(w, st, h_sha512_K);
	const bool neg_ok = (memcmp(st, kat_abc_digest, sizeof(kat_abc_digest)) != 0);

	// --- synthetic 80-byte header: host path vs the independent sph path ---
	uint8_t m[80];
	for (int i = 0; i < 80; i++) m[i] = (uint8_t) i;
	uint64_t q[4], q1[8];
	sha512256d_host_hash80(m, q, q1);

	uint8_t hostdig[32], refdig[32];
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 8; j++)
			hostdig[8*i + j] = (uint8_t)(q[i] >> (56 - 8*j));
	sha512256d_hash(refdig, m);
	const bool sph_ok = (memcmp(hostdig, refdig, 32) == 0);

	// --- GPU: abc digest, first-hash state and final digest must match ---
	uint64_t inbuf[SELFTEST_IN_WORDS], out[SELFTEST_OUT_WORDS] = { 0 };
	inbuf[0] = 0x6162638000000000ULL;
	for (int i = 1; i < 15; i++) inbuf[i] = 0;
	inbuf[15] = 24;
	for (int i = 0; i < 10; i++) {
		uint64_t v = 0;
		for (int j = 0; j < 8; j++)
			v = (v << 8) | m[8*i + j];
		inbuf[16 + i] = v;
	}
	sha512_prehash_split_host(inbuf + 16, inbuf + 26);

	// host-side resume path must also reproduce the full-transform hash1
	bool pre_ok;
	{
		uint64_t w2[16], st2[8];
		for (int i = 0; i < 10; i++) w2[i] = inbuf[16 + i];
		w2[10] = 0x8000000000000000ULL;
		for (int i = 11; i < 15; i++) w2[i] = 0;
		w2[15] = 640;
		sha512_256_init_state(st2, h_sha512_256_H);
		sha512_transform_80_from_pre9(w2, inbuf + 26, st2, h_sha512_K);
		pre_ok = (memcmp(st2, q1, sizeof(q1)) == 0);
	}

	uint64_t *d_buf = NULL;
	bool gpu_ok = false;
	if (cudaMalloc(&d_buf, sizeof(inbuf) + sizeof(out)) != cudaSuccess) {
		selftest_cuda_fault(); // could not run != produced a wrong digest
	} else {
		uint64_t *d_out = d_buf + SELFTEST_IN_WORDS;
		gpu_ok = (cudaMemcpy(d_buf, inbuf, sizeof(inbuf), cudaMemcpyHostToDevice) == cudaSuccess);
		sha512256d_selftest_gpu <<<1, 1>>> (d_buf, d_out);
		gpu_ok = gpu_ok && (cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost) == cudaSuccess);
		cudaFree(d_buf);
	}
	if (!gpu_ok)
		selftest_cuda_fault(); // a failed copy is not a wrong digest either
	gpu_ok = gpu_ok && (memcmp(out, kat_abc_digest, sizeof(kat_abc_digest)) == 0)
	                && (memcmp(out + 4, q1, sizeof(q1)) == 0)
	                && (memcmp(out + 12, q, sizeof(q)) == 0)
	                && (memcmp(out + 16, q1, sizeof(q1)) == 0);

	passed[thr_id] = abc_ok && neg_ok && sph_ok && pre_ok && gpu_ok;
	if (!passed[thr_id])
		gpulog(LOG_ERR, thr_id, "SHA512/256 device-library self-test FAILED (abc %d neg %d sph %d pre %d gpu %d)",
			(int) abc_ok, (int) neg_ok, (int) sph_ok, (int) pre_ok, (int) gpu_ok);
	return selftest_gate(thr_id, "SHA512/256", passed[thr_id]);
}
