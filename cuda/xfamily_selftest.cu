// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Unit self-tests for the x-family shared device library in cuda/
 * (docs/coding-guideline.md §7, layer 1). One function per extracted
 * primitive, each run once per process from the first consumer's
 * *_cpu_init: the __device__ building blocks are compared against the
 * vendored sph_* CPU references (independent implementations), the sph
 * reference itself is anchored against an official spec vector, and a
 * one-time negative test (flipped input bit must change the digest)
 * proves the test isn't vacuous. FAIL-CLOSED via cuda/selftest_gate.cuh.
 */

#include <string.h>

#include <miner.h>

#include "cuda/selftest_gate.cuh"

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_keccak.h"
#include "sph/sph_jh.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_sha2.h"
#include "sph/sph_luffa.h"
#include "sph/sph_shabal.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_groestl.h"
#include "sph/sph_echo.h"
#include "sph/sph_shavite.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_tiger.h"
}

#include "cuda/helper_alexis.cuh"
#include "cuda/vectors_alexis.cuh"

#include "cuda/blake512_device.cuh"
#include "cuda/keccak_device.cuh"
#include "cuda/jh512_device.cuh"
#include "cuda/bmw512_device.cuh"
#include "cuda/skein512_device.cuh"
#include "cuda/sha512_device.cuh"
#include "cuda/luffa512_device.cuh"
#include "cuda/shabal512_device.cuh"
#include "cuda/cubehash512_device.cuh"
#include "cuda/fugue512_device.cuh"
#include "cuda/groestl512_x2_device.cuh"
#include "cuda/echo512_device.cuh"
#include "cuda/shavite512_device.cuh"
#include "cuda/whirlpool512_x4_device.cuh"
#include "cuda/tiger192_device.cuh"
#include "cuda/hamsi512_device.cuh"  /* keep LAST: exports SBOX/ROUND_BIG macros */

/* ---------------------------------------------------------------- blake512 */

#define BLAKE512_ST_VEC 4

/* BLAKE submission appendix: BLAKE-512 of a single 0x00 byte — anchors the
 * sph reference outside this codebase. */
static const uint8_t kat_blake512_zero[64] = {
	0x97, 0x96, 0x15, 0x87, 0xF6, 0xD9, 0x70, 0xFA, 0xBA, 0x6D, 0x24, 0x78, 0x04, 0x5D, 0xE6, 0xD1,
	0xFA, 0xBD, 0x09, 0xB6, 0x1A, 0xE5, 0x09, 0x32, 0x05, 0x4D, 0x52, 0xBC, 0x29, 0xD3, 0x1B, 0xE4,
	0xFF, 0x91, 0x02, 0xB9, 0xF6, 0x9E, 0x2B, 0xBD, 0xB8, 0x3B, 0xE1, 0x3D, 0x4B, 0x9C, 0x06, 0x09,
	0x1E, 0x5F, 0xA0, 0xB4, 0x8B, 0xD0, 0x81, 0xB6, 0x34, 0x05, 0x8B, 0xE0, 0xEC, 0x49, 0xBE, 0xB3
};

/* BLAKE-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_blake512_pat64[64] = {
	0x4D, 0x47, 0x29, 0x1B, 0x80, 0x77, 0x50, 0xD2, 0xCE, 0x6C, 0xED, 0x17, 0xAE, 0x71, 0xDC, 0x24,
	0xF5, 0xA3, 0x20, 0x5F, 0x4F, 0xE3, 0x09, 0x53, 0x74, 0x88, 0x24, 0x2C, 0x44, 0x20, 0xCD, 0x32,
	0xD9, 0x97, 0xBE, 0xDA, 0x4D, 0x56, 0x02, 0x00, 0xCB, 0xCF, 0x3E, 0x9D, 0x68, 0x14, 0x3E, 0x69,
	0xF0, 0x8C, 0x54, 0xB8, 0x2C, 0xE7, 0x7D, 0xB7, 0xC2, 0x2D, 0x0E, 0x17, 0xB5, 0xA1, 0x36, 0x3E
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order),
 * also recording the word-3 projection taken from the same input. */
__global__ __launch_bounds__(1, 1)
void blake512_selftest_gpu(uint2 *io, uint64_t *w3, int count)
{
	for (int v = 0; v < count; v++) {
		uint2 hash[8];
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			hash[i] = io[(v << 3) + i];
		w3[v] = blake512_hash_64_word3(hash);
		blake512_hash_64(hash);
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			io[(v << 3) + i] = hash[i];
	}
}

static bool blake512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64],
	uint64_t *w3, int count)
{
	uint8_t *d_base = NULL;
	if (cudaMalloc(&d_base, (size_t) count * (64 + 8)) != cudaSuccess)
		return selftest_cuda_fault();
	uint2 *d_io = (uint2*) d_base;
	uint64_t *d_w3 = (uint64_t*)(d_base + (size_t) count * 64);

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	blake512_selftest_gpu <<<1, 1>>> (d_io, d_w3, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	ok = ok && (cudaMemcpy(w3, d_w3, (size_t) count * 8, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_base);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool blake512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_blake512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	const uint8_t zero = 0x00;
	sph_blake512_init(&ctx);
	sph_blake512(&ctx, &zero, 1);
	sph_blake512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_blake512_zero, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[BLAKE512_ST_VEC][64], ref[BLAKE512_ST_VEC][64];
	uint32_t seed = 0x58313652; /* 'X16R' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < BLAKE512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < BLAKE512_ST_VEC; v++) {
		sph_blake512_init(&ctx);
		sph_blake512(&ctx, msg[v], 64);
		sph_blake512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_blake512_pat64, 64) == 0);

	// --- GPU hash + word-3 projection vs the sph digests ---
	uint8_t gpu[BLAKE512_ST_VEC][64];
	uint64_t w3[BLAKE512_ST_VEC];
	bool gpu_ok = blake512_selftest_run(msg, gpu, w3, BLAKE512_ST_VEC);
	bool w3_ok = gpu_ok;
	if (gpu_ok) {
		gpu_ok = (memcmp(gpu, ref, sizeof(ref)) == 0);
		for (int v = 0; v < BLAKE512_ST_VEC; v++) {
			uint64_t r;
			memcpy(&r, ref[v] + 24, 8);
			w3_ok = w3_ok && (w3[v] == r);
		}
	}

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	uint64_t negw3[1];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = blake512_selftest_run(negmsg, negdig, negw3, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && w3_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "blake512 device-library self-test FAILED (sph %d kat %d gpu %d w3 %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) w3_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "blake512 device-library self-test passed");
	return selftest_gate(thr_id, "blake512", passed);
}

/* --------------------------------------------------------------- keccak512 */

#define KECCAK512_ST_VEC 4

/* Keccak team vector: Keccak-512 of the empty message — anchors the sph
 * reference outside this codebase. */
static const uint8_t kat_keccak512_empty[64] = {
	0x0E, 0xAB, 0x42, 0xDE, 0x4C, 0x3C, 0xEB, 0x92, 0x35, 0xFC, 0x91, 0xAC, 0xFF, 0xE7, 0x46, 0xB2,
	0x9C, 0x29, 0xA8, 0xC3, 0x66, 0xB7, 0xC6, 0x0E, 0x4E, 0x67, 0xC4, 0x66, 0xF3, 0x6A, 0x43, 0x04,
	0xC0, 0x0F, 0xA9, 0xCA, 0xF9, 0xD8, 0x79, 0x76, 0xBA, 0x46, 0x9B, 0xCB, 0xE0, 0x67, 0x13, 0xB4,
	0x35, 0xF0, 0x91, 0xEF, 0x27, 0x69, 0xFB, 0x16, 0x0C, 0xDA, 0xB3, 0x3D, 0x36, 0x70, 0x68, 0x0E
};

/* Keccak-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_keccak512_pat64[64] = {
	0x59, 0xBF, 0xF1, 0xED, 0xB3, 0x7C, 0x40, 0x3B, 0xEA, 0x63, 0x87, 0xE2, 0x83, 0xC5, 0xD4, 0xD8,
	0x87, 0x82, 0x46, 0x59, 0x28, 0x07, 0xD2, 0x23, 0x28, 0xFB, 0xC1, 0x1E, 0xC1, 0xE0, 0x29, 0xCD,
	0xB6, 0x65, 0x93, 0x00, 0x52, 0x98, 0x49, 0x18, 0x9A, 0xD6, 0x47, 0xFD, 0xE9, 0xAD, 0x4A, 0x89,
	0x18, 0x20, 0x2B, 0xA3, 0x10, 0xB9, 0x36, 0xAC, 0x6A, 0x1D, 0x47, 0x7E, 0x42, 0x84, 0xAC, 0x4B
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order),
 * also recording the lane-3 projection taken from the same input. */
__global__ __launch_bounds__(1, 1)
void keccak512_selftest_gpu(uint2 *io, uint64_t *w3, int count)
{
	for (int v = 0; v < count; v++) {
		uint2 hash[8];
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			hash[i] = io[(v << 3) + i];
		w3[v] = devectorize(keccak512_hash_64_lane3(hash));
		keccak512_hash_64(hash);
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			io[(v << 3) + i] = hash[i];
	}
}

static bool keccak512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64],
	uint64_t *w3, int count)
{
	uint8_t *d_base = NULL;
	if (cudaMalloc(&d_base, (size_t) count * (64 + 8)) != cudaSuccess)
		return selftest_cuda_fault();
	uint2 *d_io = (uint2*) d_base;
	uint64_t *d_w3 = (uint64_t*)(d_base + (size_t) count * 64);

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	keccak512_selftest_gpu <<<1, 1>>> (d_io, d_w3, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	ok = ok && (cudaMemcpy(w3, d_w3, (size_t) count * 8, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_base);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool keccak512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_keccak512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_keccak512_init(&ctx);
	sph_keccak512(&ctx, NULL, 0);
	sph_keccak512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_keccak512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[KECCAK512_ST_VEC][64], ref[KECCAK512_ST_VEC][64];
	uint32_t seed = 0x4B363443; /* 'K64C' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < KECCAK512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < KECCAK512_ST_VEC; v++) {
		sph_keccak512_init(&ctx);
		sph_keccak512(&ctx, msg[v], 64);
		sph_keccak512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_keccak512_pat64, 64) == 0);

	// --- GPU hash + lane-3 projection vs the sph digests ---
	uint8_t gpu[KECCAK512_ST_VEC][64];
	uint64_t w3[KECCAK512_ST_VEC];
	bool gpu_ok = keccak512_selftest_run(msg, gpu, w3, KECCAK512_ST_VEC);
	bool w3_ok = gpu_ok;
	if (gpu_ok) {
		gpu_ok = (memcmp(gpu, ref, sizeof(ref)) == 0);
		for (int v = 0; v < KECCAK512_ST_VEC; v++) {
			uint64_t r;
			memcpy(&r, ref[v] + 24, 8);
			w3_ok = w3_ok && (w3[v] == r);
		}
	}

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	uint64_t negw3[1];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = keccak512_selftest_run(negmsg, negdig, negw3, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && w3_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "keccak512 device-library self-test FAILED (sph %d kat %d gpu %d w3 %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) w3_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "keccak512 device-library self-test passed");
	return selftest_gate(thr_id, "keccak512", passed);
}

/* ------------------------------------------------------------------- jh512 */

#define JH512_ST_VEC 4

/* JH submission appendix: JH-512 of the empty message — anchors the sph
 * reference outside this codebase. */
static const uint8_t kat_jh512_empty[64] = {
	0x90, 0xEC, 0xF2, 0xF7, 0x6F, 0x9D, 0x2C, 0x80, 0x17, 0xD9, 0x79, 0xAD, 0x5A, 0xB9, 0x6B, 0x87,
	0xD5, 0x8F, 0xC8, 0xFC, 0x4B, 0x83, 0x06, 0x0F, 0x3F, 0x90, 0x07, 0x74, 0xFA, 0xA2, 0xC8, 0xFA,
	0xBE, 0x69, 0xC5, 0xF4, 0xFF, 0x1E, 0xC2, 0xB6, 0x1D, 0x6B, 0x31, 0x69, 0x41, 0xCE, 0xDE, 0xE1,
	0x17, 0xFB, 0x04, 0xB1, 0xF4, 0xC5, 0xBC, 0x1B, 0x91, 0x9A, 0xE8, 0x41, 0xC5, 0x0E, 0xEC, 0x4F
};

/* JH-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_jh512_pat64[64] = {
	0x48, 0x35, 0x60, 0xD1, 0x0C, 0xAD, 0xEC, 0x86, 0xDB, 0x6F, 0x39, 0x0F, 0x62, 0x67, 0xE1, 0x2F,
	0x99, 0x59, 0x45, 0x87, 0xD4, 0x4C, 0x20, 0x29, 0x02, 0xE8, 0xE4, 0xBB, 0x6C, 0x70, 0xC6, 0xC7,
	0xFD, 0xFF, 0x6B, 0x19, 0x96, 0x56, 0x50, 0xE1, 0x5E, 0x24, 0x0B, 0xCF, 0xCE, 0xFE, 0x4E, 0x50,
	0x51, 0x56, 0x7E, 0xF9, 0x6C, 0x75, 0x8B, 0x80, 0x0E, 0xFD, 0xCA, 0xF5, 0x0A, 0x5D, 0x5B, 0xBD
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order). */
__global__ __launch_bounds__(1, 1)
void jh512_selftest_gpu(uint32_t *io, int count)
{
	for (int v = 0; v < count; v++) {
		uint32_t h[16];
		#pragma unroll 16
		for (int i = 0; i < 16; i++)
			h[i] = io[(v << 4) + i];
		jh512_hash_64(h);
		#pragma unroll 16
		for (int i = 0; i < 16; i++)
			io[(v << 4) + i] = h[i];
	}
}

static bool jh512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	jh512_selftest_gpu <<<1, 1>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool jh512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_jh512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_jh512_init(&ctx);
	sph_jh512(&ctx, NULL, 0);
	sph_jh512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_jh512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[JH512_ST_VEC][64], ref[JH512_ST_VEC][64];
	uint32_t seed = 0x4A483531; /* 'JH51' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < JH512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < JH512_ST_VEC; v++) {
		sph_jh512_init(&ctx);
		sph_jh512(&ctx, msg[v], 64);
		sph_jh512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_jh512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[JH512_ST_VEC][64];
	bool gpu_ok = jh512_selftest_run(msg, gpu, JH512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = jh512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "jh512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "jh512 device-library self-test passed");
	return selftest_gate(thr_id, "jh512", passed);
}

/* ------------------------------------------------------------------ bmw512 */

#define BMW512_ST_VEC 4

/* BMW submission appendix: BMW-512 of the empty message — anchors the sph
 * reference outside this codebase. */
static const uint8_t kat_bmw512_empty[64] = {
	0x6A, 0x72, 0x56, 0x55, 0xC4, 0x2B, 0xC8, 0xA2, 0xA2, 0x05, 0x49, 0xDD, 0x5A, 0x23, 0x3A, 0x6A,
	0x2B, 0xEB, 0x01, 0x61, 0x69, 0x75, 0x85, 0x1F, 0xD1, 0x22, 0x50, 0x4E, 0x60, 0x4B, 0x46, 0xAF,
	0x7D, 0x96, 0x69, 0x7D, 0x0B, 0x63, 0x33, 0xDB, 0x1D, 0x17, 0x09, 0xD6, 0xDF, 0x32, 0x8D, 0x2A,
	0x6C, 0x78, 0x65, 0x51, 0xB0, 0xCC, 0xE2, 0x25, 0x5E, 0x8C, 0x73, 0x32, 0xB4, 0x81, 0x9C, 0x0E
};

/* BMW-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_bmw512_pat64[64] = {
	0x82, 0x41, 0x68, 0x67, 0x1C, 0x2E, 0x3F, 0x35, 0xEB, 0xBA, 0x82, 0xB6, 0x3B, 0x9E, 0x6C, 0x42,
	0xB8, 0x41, 0x1C, 0xDC, 0xDA, 0x10, 0x41, 0x26, 0x4B, 0xB5, 0xF5, 0x0A, 0xBD, 0x50, 0x7D, 0x18,
	0x27, 0xED, 0xCF, 0xFF, 0x05, 0x0F, 0x6C, 0x86, 0x75, 0xCB, 0x8C, 0xCB, 0xA8, 0x69, 0x9C, 0x84,
	0x3D, 0xCF, 0x5F, 0xB8, 0x1C, 0xCA, 0xDA, 0xB1, 0xDE, 0xEF, 0x0D, 0x9C, 0xF4, 0x77, 0x02, 0x57
};

/* One thread hashes `count` 64-byte vectors (d_hash word order), writing
 * the digest back over the input slot; also records the word-3 projection
 * taken from the same input. */
__global__ __launch_bounds__(1, 1)
void bmw512_selftest_gpu(uint64_t *io, uint64_t *w3, int count)
{
	for (int v = 0; v < count; v++) {
		uint64_t __align__(16) msg[16];
		uint64_t __align__(16) msg2[16];
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			msg2[i] = msg[i] = io[(v << 3) + i];
		w3[v] = bmw512_hash_64_word3(msg2);
		bmw512_hash_64(msg);
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			io[(v << 3) + i] = msg[8 + i];
	}
}

static bool bmw512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64],
	uint64_t *w3, int count)
{
	uint8_t *d_base = NULL;
	if (cudaMalloc(&d_base, (size_t) count * (64 + 8)) != cudaSuccess)
		return selftest_cuda_fault();
	uint64_t *d_io = (uint64_t*) d_base;
	uint64_t *d_w3 = (uint64_t*)(d_base + (size_t) count * 64);

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	bmw512_selftest_gpu <<<1, 1>>> (d_io, d_w3, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	ok = ok && (cudaMemcpy(w3, d_w3, (size_t) count * 8, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_base);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool bmw512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_bmw512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_bmw512_init(&ctx);
	sph_bmw512(&ctx, NULL, 0);
	sph_bmw512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_bmw512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[BMW512_ST_VEC][64], ref[BMW512_ST_VEC][64];
	uint32_t seed = 0x424D5735; /* 'BMW5' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < BMW512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < BMW512_ST_VEC; v++) {
		sph_bmw512_init(&ctx);
		sph_bmw512(&ctx, msg[v], 64);
		sph_bmw512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_bmw512_pat64, 64) == 0);

	// --- GPU hash + word-3 projection vs the sph digests ---
	uint8_t gpu[BMW512_ST_VEC][64];
	uint64_t w3[BMW512_ST_VEC];
	bool gpu_ok = bmw512_selftest_run(msg, gpu, w3, BMW512_ST_VEC);
	bool w3_ok = gpu_ok;
	if (gpu_ok) {
		gpu_ok = (memcmp(gpu, ref, sizeof(ref)) == 0);
		for (int v = 0; v < BMW512_ST_VEC; v++) {
			uint64_t r;
			memcpy(&r, ref[v] + 24, 8);
			w3_ok = w3_ok && (w3[v] == r);
		}
	}

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	uint64_t negw3[1];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = bmw512_selftest_run(negmsg, negdig, negw3, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && w3_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "bmw512 device-library self-test FAILED (sph %d kat %d gpu %d w3 %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) w3_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "bmw512 device-library self-test passed");
	return selftest_gate(thr_id, "bmw512", passed);
}

/* ---------------------------------------------------------------- skein512 */

#define SKEIN512_ST_VEC 4

/* Skein 1.3 reference: Skein-512-512 of the empty message — anchors the
 * sph reference outside this codebase. */
static const uint8_t kat_skein512_empty[64] = {
	0xBC, 0x5B, 0x4C, 0x50, 0x92, 0x55, 0x19, 0xC2, 0x90, 0xCC, 0x63, 0x42, 0x77, 0xAE, 0x3D, 0x62,
	0x57, 0x21, 0x23, 0x95, 0xCB, 0xA7, 0x33, 0xBB, 0xAD, 0x37, 0xA4, 0xAF, 0x0F, 0xA0, 0x6A, 0xF4,
	0x1F, 0xCA, 0x79, 0x03, 0xD0, 0x65, 0x64, 0xFE, 0xA7, 0xA2, 0xD3, 0x73, 0x0D, 0xBD, 0xB8, 0x0C,
	0x1F, 0x85, 0x56, 0x2D, 0xFC, 0xC0, 0x70, 0x33, 0x4E, 0xA4, 0xD1, 0xD9, 0xE7, 0x2C, 0xBA, 0x7A
};

/* Skein-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_skein512_pat64[64] = {
	0x78, 0xCF, 0xDB, 0xDB, 0x2B, 0xD1, 0x25, 0xF4, 0x9D, 0x26, 0x14, 0x6E, 0x20, 0x8E, 0xBC, 0x7C,
	0xEA, 0xE5, 0x76, 0x19, 0xBD, 0x68, 0xA2, 0xE4, 0xE9, 0xCD, 0xB1, 0xDB, 0x19, 0x8C, 0x99, 0x5E,
	0x37, 0x95, 0xFA, 0xDB, 0xCC, 0xAA, 0xBB, 0x00, 0x04, 0x63, 0x52, 0x5E, 0xEE, 0x2E, 0x1E, 0x7F,
	0x6E, 0x83, 0x09, 0xC7, 0x65, 0xA6, 0x1E, 0x19, 0xFC, 0xCD, 0xB1, 0x8F, 0x52, 0x84, 0xC0, 0x70
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order). */
__global__ __launch_bounds__(1, 1)
void skein512_selftest_gpu(uint2 *io, int count)
{
	for (int v = 0; v < count; v++) {
		uint2 p[8];
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			p[i] = io[(v << 3) + i];
		skein512_hash_64(p);
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			io[(v << 3) + i] = p[i];
	}
}

static bool skein512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint2 *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	skein512_selftest_gpu <<<1, 1>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool skein512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_skein512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_skein512_init(&ctx);
	sph_skein512(&ctx, NULL, 0);
	sph_skein512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_skein512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[SKEIN512_ST_VEC][64], ref[SKEIN512_ST_VEC][64];
	uint32_t seed = 0x534B4549; /* 'SKEI' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < SKEIN512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < SKEIN512_ST_VEC; v++) {
		sph_skein512_init(&ctx);
		sph_skein512(&ctx, msg[v], 64);
		sph_skein512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_skein512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[SKEIN512_ST_VEC][64];
	bool gpu_ok = skein512_selftest_run(msg, gpu, SKEIN512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = skein512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "skein512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "skein512 device-library self-test passed");
	return selftest_gate(thr_id, "skein512", passed);
}

/* -------------------------------------------------------- sha512 (x17) ---- */

#define SHA512X_ST_VEC 4

/* FIPS 180-4: SHA-512("abc") — anchors the sph reference outside this
 * codebase. */
static const uint8_t kat_sha512_abc[64] = {
	0xDD, 0xAF, 0x35, 0xA1, 0x93, 0x61, 0x7A, 0xBA, 0xCC, 0x41, 0x73, 0x49, 0xAE, 0x20, 0x41, 0x31,
	0x12, 0xE6, 0xFA, 0x4E, 0x89, 0xA9, 0x7E, 0xA2, 0x0A, 0x9E, 0xEE, 0xE6, 0x4B, 0x55, 0xD3, 0x9A,
	0x21, 0x92, 0x99, 0x2A, 0x27, 0x4F, 0xC1, 0xA8, 0x36, 0xBA, 0x3C, 0x23, 0xA3, 0xFE, 0xEB, 0xBD,
	0x45, 0x4D, 0x44, 0x23, 0x64, 0x3C, 0xE8, 0x0E, 0x2A, 0x9A, 0xC9, 0x4F, 0xA5, 0x4C, 0xA4, 0x9F
};

/* SHA-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_sha512_pat64[64] = {
	0xEE, 0x43, 0x20, 0xEB, 0xAF, 0x3F, 0xDB, 0x4F, 0x2C, 0x83, 0x2B, 0x13, 0x72, 0x00, 0xC0, 0x8E,
	0x23, 0x5E, 0x0F, 0xA7, 0xBB, 0xD0, 0xEB, 0x17, 0x40, 0xC7, 0x06, 0x3B, 0xA8, 0xA0, 0xD1, 0x51,
	0xDA, 0x77, 0xE0, 0x03, 0x39, 0x8E, 0x17, 0x14, 0xA9, 0x55, 0xD4, 0x75, 0xB0, 0x5E, 0x3E, 0x95,
	0x0B, 0x63, 0x95, 0x03, 0xB4, 0x52, 0xEC, 0x18, 0x5D, 0xE4, 0x22, 0x9B, 0xC4, 0x87, 0x39, 0x49
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order). */
__global__ __launch_bounds__(1, 1)
void sha512x_selftest_gpu(uint64_t *io, int count)
{
	for (int v = 0; v < count; v++)
		sha512_hash_64(&io[v << 3]);
}

static bool sha512x_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint64_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	sha512x_selftest_gpu <<<1, 1>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool sha512x_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_sha512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_sha512_init(&ctx);
	sph_sha512(&ctx, "abc", 3);
	sph_sha512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_sha512_abc, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[SHA512X_ST_VEC][64], ref[SHA512X_ST_VEC][64];
	uint32_t seed = 0x53483531; /* 'SH51' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < SHA512X_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < SHA512X_ST_VEC; v++) {
		sph_sha512_init(&ctx);
		sph_sha512(&ctx, msg[v], 64);
		sph_sha512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_sha512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[SHA512X_ST_VEC][64];
	bool gpu_ok = sha512x_selftest_run(msg, gpu, SHA512X_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = sha512x_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "sha512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "sha512 device-library self-test passed");
	return selftest_gate(thr_id, "sha512", passed);
}

/* ---------------------------------------------------------------- luffa512 */

#define LUFFA512_ST_VEC 4

/* Luffa submission appendix: Luffa-512 of the empty message — anchors the
 * sph reference outside this codebase. */
static const uint8_t kat_luffa512_empty[64] = {
	0x6E, 0x7D, 0xE4, 0x50, 0x11, 0x89, 0xB3, 0xCA, 0x58, 0xF3, 0xAC, 0x11, 0x49, 0x16, 0x65, 0x4B,
	0xBC, 0xD4, 0x92, 0x20, 0x24, 0xB4, 0xCC, 0x1C, 0xD7, 0x64, 0xAC, 0xFE, 0x8A, 0xB4, 0xB7, 0x80,
	0x5D, 0xF1, 0x33, 0xEA, 0xB3, 0x45, 0xFF, 0xDB, 0x1C, 0x41, 0x45, 0x64, 0xC9, 0x24, 0xF4, 0x8E,
	0x0A, 0x30, 0x18, 0x24, 0xE2, 0xAC, 0x4C, 0x34, 0xBD, 0x4E, 0xFD, 0xE2, 0xE4, 0x3D, 0xA9, 0x0E
};

/* Luffa-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_luffa512_pat64[64] = {
	0xA7, 0xFA, 0x7B, 0x1F, 0x6E, 0xFD, 0xEE, 0xBB, 0x4E, 0xB2, 0xD5, 0x3B, 0x28, 0x41, 0x2A, 0x49,
	0x62, 0xFA, 0x72, 0x36, 0xC7, 0x75, 0xA0, 0x47, 0x2E, 0x97, 0xAF, 0x30, 0x35, 0xD0, 0xCE, 0xBB,
	0x79, 0x4C, 0x97, 0x46, 0x5B, 0xC8, 0x9E, 0x9D, 0x26, 0x98, 0x3A, 0x7B, 0x0A, 0x92, 0x83, 0xCC,
	0x71, 0x6D, 0xEE, 0x3A, 0x60, 0x83, 0x81, 0x70, 0x30, 0xDD, 0xA4, 0xA4, 0x62, 0x5F, 0x5D, 0x00
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order). */
__global__ __launch_bounds__(1, 1)
void luffa512_selftest_gpu(uint32_t *io, int count)
{
	for (int v = 0; v < count; v++)
		luffa512_hash_64(&io[v << 4]);
}

static bool luffa512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	luffa512_selftest_gpu <<<1, 1>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool luffa512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_luffa512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_luffa512_init(&ctx);
	sph_luffa512(&ctx, NULL, 0);
	sph_luffa512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_luffa512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[LUFFA512_ST_VEC][64], ref[LUFFA512_ST_VEC][64];
	uint32_t seed = 0x4C554646; /* 'LUFF' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < LUFFA512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < LUFFA512_ST_VEC; v++) {
		sph_luffa512_init(&ctx);
		sph_luffa512(&ctx, msg[v], 64);
		sph_luffa512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_luffa512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[LUFFA512_ST_VEC][64];
	bool gpu_ok = luffa512_selftest_run(msg, gpu, LUFFA512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = luffa512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "luffa512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "luffa512 device-library self-test passed");
	return selftest_gate(thr_id, "luffa512", passed);
}

/* --------------------------------------------------------------- shabal512 */

#define SHABAL512_ST_VEC 4

/* Shabal submission appendix: Shabal-512 of the empty message — anchors the
 * sph reference outside this codebase. */
static const uint8_t kat_shabal512_empty[64] = {
	0xFC, 0x2D, 0x5D, 0xFF, 0x5D, 0x70, 0xB7, 0xF6, 0xB1, 0xF8, 0xC2, 0xFC, 0xC8, 0xC1, 0xF9, 0xFE,
	0x99, 0x34, 0xE5, 0x42, 0x57, 0xED, 0xED, 0x0C, 0xF2, 0xB5, 0x39, 0xA2, 0xEF, 0x0A, 0x19, 0xCC,
	0xFF, 0xA8, 0x4F, 0x8D, 0x9F, 0xA1, 0x35, 0xE4, 0xBD, 0x3C, 0x09, 0xF5, 0x90, 0xF3, 0xA9, 0x27,
	0xEB, 0xD6, 0x03, 0xAC, 0x29, 0xEB, 0x72, 0x9E, 0x6F, 0x2A, 0x9A, 0xF0, 0x31, 0xAD, 0x8D, 0xC6
};

/* Shabal-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_shabal512_pat64[64] = {
	0x2E, 0xE4, 0xA7, 0x46, 0x33, 0x34, 0x8E, 0x70, 0xCD, 0x57, 0x2F, 0xE2, 0x37, 0x05, 0x1E, 0x66,
	0x28, 0x69, 0xFE, 0x7E, 0xE5, 0xF1, 0x60, 0xE2, 0xA2, 0x00, 0x7F, 0x15, 0x13, 0x99, 0x36, 0xFC,
	0xF4, 0xC6, 0x31, 0x5E, 0x4C, 0x93, 0x0F, 0xC7, 0xBD, 0x6D, 0x86, 0x79, 0xD9, 0x91, 0x5D, 0x9F,
	0xDB, 0x0F, 0x24, 0x88, 0x27, 0x45, 0x5E, 0x05, 0x3F, 0x70, 0x02, 0xCE, 0x5F, 0xDB, 0xC8, 0xDA
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order). */
__global__ __launch_bounds__(1, 1)
void shabal512_selftest_gpu(uint32_t *io, int count)
{
	for (int v = 0; v < count; v++)
		shabal512_hash_64(&io[v << 4]);
}

static bool shabal512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	shabal512_selftest_gpu <<<1, 1>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool shabal512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_shabal512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_shabal512_init(&ctx);
	sph_shabal512(&ctx, NULL, 0);
	sph_shabal512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_shabal512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[SHABAL512_ST_VEC][64], ref[SHABAL512_ST_VEC][64];
	uint32_t seed = 0x53484142; /* 'SHAB' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < SHABAL512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < SHABAL512_ST_VEC; v++) {
		sph_shabal512_init(&ctx);
		sph_shabal512(&ctx, msg[v], 64);
		sph_shabal512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_shabal512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[SHABAL512_ST_VEC][64];
	bool gpu_ok = shabal512_selftest_run(msg, gpu, SHABAL512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = shabal512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "shabal512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "shabal512 device-library self-test passed");
	return selftest_gate(thr_id, "shabal512", passed);
}

/* ------------------------------------------------------------- cubehash512 */

#define CUBEHASH512_ST_VEC 4

/* CubeHash16/32-512 of the empty message — anchors the sph reference
 * outside this codebase. */
static const uint8_t kat_cubehash512_empty[64] = {
	0x4A, 0x1D, 0x00, 0xBB, 0xCF, 0xCB, 0x5A, 0x95, 0x62, 0xFB, 0x98, 0x1E, 0x7F, 0x7D, 0xB3, 0x35,
	0x0F, 0xE2, 0x65, 0x86, 0x39, 0xD9, 0x48, 0xB9, 0xD5, 0x74, 0x52, 0xC2, 0x23, 0x28, 0xBB, 0x32,
	0xF4, 0x68, 0xB0, 0x72, 0x20, 0x84, 0x50, 0xBA, 0xD5, 0xEE, 0x17, 0x82, 0x71, 0x40, 0x8B, 0xE0,
	0xB1, 0x6E, 0x56, 0x33, 0xAC, 0x8A, 0x1E, 0x3C, 0xF9, 0x86, 0x4C, 0xFB, 0xFC, 0x8E, 0x04, 0x3A
};

/* CubeHash-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_cubehash512_pat64[64] = {
	0x50, 0x13, 0x36, 0x30, 0x4D, 0x23, 0x5C, 0xF9, 0x82, 0x5D, 0xA6, 0xF2, 0x66, 0x23, 0xF2, 0x0C,
	0xF3, 0x38, 0xC2, 0xEF, 0xA1, 0xEA, 0xC0, 0xF6, 0x68, 0xDC, 0x1D, 0x99, 0xA1, 0xF6, 0x7B, 0xF6,
	0x17, 0x45, 0x3A, 0x1C, 0x86, 0x1C, 0x96, 0x28, 0xF7, 0xB9, 0x8D, 0xD5, 0xCA, 0x50, 0xBA, 0xF6,
	0xD7, 0x52, 0x99, 0xF1, 0x3F, 0xA4, 0x3D, 0x28, 0xD3, 0x9B, 0x40, 0x31, 0x41, 0x61, 0x13, 0x5F
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order). */
__global__ __launch_bounds__(1, 1)
void cubehash512_selftest_gpu(uint32_t *io, int count)
{
	for (int v = 0; v < count; v++)
		cubehash512_hash_64(&io[v << 4]);
}

static bool cubehash512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	cubehash512_selftest_gpu <<<1, 1>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool cubehash512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_cubehash512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_cubehash512_init(&ctx);
	sph_cubehash512(&ctx, NULL, 0);
	sph_cubehash512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_cubehash512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[CUBEHASH512_ST_VEC][64], ref[CUBEHASH512_ST_VEC][64];
	uint32_t seed = 0x43554245; /* 'CUBE' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < CUBEHASH512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < CUBEHASH512_ST_VEC; v++) {
		sph_cubehash512_init(&ctx);
		sph_cubehash512(&ctx, msg[v], 64);
		sph_cubehash512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_cubehash512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[CUBEHASH512_ST_VEC][64];
	bool gpu_ok = cubehash512_selftest_run(msg, gpu, CUBEHASH512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = cubehash512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "cubehash512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "cubehash512 device-library self-test passed");
	return selftest_gate(thr_id, "cubehash512", passed);
}

/* ---------------------------------------------------------------- hamsi512 */

#define HAMSI512_ST_VEC 4

/* Hamsi submission appendix: Hamsi-512 of the empty message — anchors the
 * sph reference outside this codebase. */
static const uint8_t kat_hamsi512_empty[64] = {
	0x5C, 0xD7, 0x43, 0x6A, 0x91, 0xE2, 0x7F, 0xC8, 0x09, 0xD7, 0x01, 0x5C, 0x34, 0x07, 0x54, 0x06,
	0x33, 0xDA, 0xB3, 0x91, 0x12, 0x71, 0x13, 0xCE, 0x6B, 0xA3, 0x60, 0xF0, 0xC1, 0xE3, 0x5F, 0x40,
	0x45, 0x10, 0x83, 0x4A, 0x55, 0x16, 0x10, 0xD6, 0xE8, 0x71, 0xE7, 0x56, 0x51, 0xEA, 0x38, 0x1A,
	0x8B, 0xA6, 0x28, 0xAF, 0x1D, 0xCF, 0x2B, 0x2B, 0xE1, 0x3A, 0xF2, 0xEB, 0x62, 0x47, 0x29, 0x0F
};

/* Hamsi-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_hamsi512_pat64[64] = {
	0xF8, 0xC6, 0xD6, 0xAB, 0x54, 0x2C, 0xE3, 0x20, 0x43, 0xE0, 0x6A, 0x04, 0xA3, 0x7E, 0xE4, 0x11,
	0x66, 0x52, 0xAD, 0xC8, 0x77, 0xB3, 0x60, 0xDC, 0x12, 0x32, 0xE3, 0xF0, 0x95, 0xB2, 0x94, 0x95,
	0x60, 0x53, 0x6B, 0x79, 0x5B, 0x18, 0x9B, 0x39, 0x3B, 0x3C, 0x44, 0x59, 0xDE, 0xC7, 0xCF, 0xB0,
	0xEA, 0xB0, 0x03, 0x0D, 0x61, 0x90, 0x77, 0x0D, 0xE8, 0x49, 0x38, 0x12, 0x32, 0xE8, 0x16, 0xB4
};

/* One thread hashes `count` 64-byte vectors in place (d_hash word order). */
__global__ __launch_bounds__(1, 1)
void hamsi512_selftest_gpu(uint32_t *io, int count)
{
	for (int v = 0; v < count; v++)
		hamsi512_hash_64(&io[v << 4]);
}

static bool hamsi512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	hamsi512_selftest_gpu <<<1, 1>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool hamsi512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_hamsi512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_hamsi512_init(&ctx);
	sph_hamsi512(&ctx, NULL, 0);
	sph_hamsi512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_hamsi512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[HAMSI512_ST_VEC][64], ref[HAMSI512_ST_VEC][64];
	uint32_t seed = 0x48414D53; /* 'HAMS' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < HAMSI512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < HAMSI512_ST_VEC; v++) {
		sph_hamsi512_init(&ctx);
		sph_hamsi512(&ctx, msg[v], 64);
		sph_hamsi512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_hamsi512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[HAMSI512_ST_VEC][64];
	bool gpu_ok = hamsi512_selftest_run(msg, gpu, HAMSI512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = hamsi512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "hamsi512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "hamsi512 device-library self-test passed");
	return selftest_gate(thr_id, "hamsi512", passed);
}

/* ---------------------------------------------------------------- fugue512 */

#define FUGUE512_ST_VEC 4

/* Fugue submission appendix: Fugue-512 of the empty message — anchors the
 * sph reference outside this codebase. */
static const uint8_t kat_fugue512_empty[64] = {
	0x31, 0x24, 0xF0, 0xCB, 0xB5, 0xA1, 0xC2, 0xFB, 0x3C, 0xE7, 0x47, 0xAD, 0xA6, 0x3E, 0xD2, 0xAB,
	0x3B, 0xCD, 0x74, 0x79, 0x5C, 0xEF, 0x2B, 0x0E, 0x80, 0x5D, 0x53, 0x19, 0xFC, 0xC3, 0x60, 0xB4,
	0x61, 0x7B, 0x6A, 0x7E, 0xB6, 0x31, 0xD6, 0x6F, 0x6D, 0x10, 0x6E, 0xD0, 0x72, 0x4B, 0x56, 0xFA,
	0x8C, 0x11, 0x10, 0xF9, 0xB8, 0xDF, 0x1C, 0x68, 0x98, 0xE7, 0xCA, 0x3C, 0x2D, 0xFC, 0xCF, 0x79
};

/* Fugue-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_fugue512_pat64[64] = {
	0x8D, 0xAF, 0x6F, 0xDF, 0x35, 0x8C, 0x3C, 0x83, 0x17, 0x9A, 0xFC, 0x8D, 0x07, 0x2D, 0x5F, 0x8B,
	0x64, 0x82, 0x37, 0x17, 0x5E, 0x6C, 0x82, 0xAA, 0x7C, 0xA4, 0xC3, 0x76, 0xCE, 0x7E, 0xF6, 0xF0,
	0xFB, 0x85, 0xE4, 0xD7, 0xB8, 0xEE, 0xC8, 0x6B, 0x5B, 0x1D, 0xD0, 0x6B, 0x2B, 0xF2, 0xC9, 0xBC,
	0x0E, 0xC6, 0x1C, 0xEE, 0x1E, 0x32, 0x02, 0xA0, 0x04, 0xE5, 0xED, 0x28, 0xAE, 0x90, 0xC9, 0x8B
};

/* One 256-thread block (for the cooperative shared fill). Every thread t hashes vector
 * t % count into out[t], so every copy of the replicated table is read (d_hash word order). */
#define FUGUE512_ST_THREADS 256
__global__ __launch_bounds__(FUGUE512_ST_THREADS, 1)
void fugue512_selftest_gpu(const uint32_t *in, uint32_t *out, int count)
{
	__shared__ uint32_t mixtabs[256 * FUGUE512_R];
	fugue512_load_shared_r(mixtabs);

	uint32_t h[16];
	for (int k = 0; k < 16; k++) h[k] = in[((threadIdx.x % count) << 4) + k];
	fugue512_hash_64_r(mixtabs, h);
	for (int k = 0; k < 16; k++) out[(threadIdx.x << 4) + k] = h[k];
}

/* dig[v] = the digest of thread v; false if any other thread computing vector v disagrees */
static bool fugue512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_in = NULL, *d_out = NULL;
	if (cudaMalloc(&d_in, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();
	if (cudaMalloc(&d_out, (size_t) FUGUE512_ST_THREADS * 64) != cudaSuccess) {
		cudaFree(d_in);
		return selftest_cuda_fault();
	}

	static uint8_t all[FUGUE512_ST_THREADS][64];
	bool ok = (cudaMemcpy(d_in, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	fugue512_selftest_gpu <<<1, FUGUE512_ST_THREADS>>> (d_in, d_out, count);
	ok = ok && (cudaMemcpy(all, d_out, sizeof(all), cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_in);
	cudaFree(d_out);
	if (!ok)
		return selftest_cuda_fault();
	memcpy(dig, all, (size_t) count * 64);
	for (int t = count; t < FUGUE512_ST_THREADS; t++)
		if (memcmp(all[t], all[t % count], 64) != 0)
			return false;
	return true;
}

__host__
bool fugue512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_fugue512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_fugue512_init(&ctx);
	sph_fugue512(&ctx, NULL, 0);
	sph_fugue512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_fugue512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[FUGUE512_ST_VEC][64], ref[FUGUE512_ST_VEC][64];
	uint32_t seed = 0x46554755; /* 'FUGU' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < FUGUE512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < FUGUE512_ST_VEC; v++) {
		sph_fugue512_init(&ctx);
		sph_fugue512(&ctx, msg[v], 64);
		sph_fugue512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_fugue512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[FUGUE512_ST_VEC][64];
	bool gpu_ok = fugue512_selftest_run(msg, gpu, FUGUE512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = fugue512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "fugue512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "fugue512 device-library self-test passed");
	return selftest_gate(thr_id, "fugue512", passed);
}

/* -------------------------------------------------------------- groestl512 */

#define GROESTL512_ST_VEC 4

/* Groestl submission KAT: Groestl-512 of the empty message — anchors the
 * sph reference outside this codebase. */
static const uint8_t kat_groestl512_empty[64] = {
	0x6D, 0x3A, 0xD2, 0x9D, 0x27, 0x91, 0x10, 0xEE, 0xF3, 0xAD, 0xBD, 0x66, 0xDE, 0x2A, 0x03, 0x45,
	0xA7, 0x7B, 0xAE, 0xDE, 0x15, 0x57, 0xF5, 0xD0, 0x99, 0xFC, 0xE0, 0xC0, 0x3D, 0x6D, 0xC2, 0xBA,
	0x8E, 0x6D, 0x4A, 0x66, 0x33, 0xDF, 0xBD, 0x66, 0x05, 0x3C, 0x20, 0xFA, 0xA8, 0x7D, 0x1A, 0x11,
	0xF3, 0x9A, 0x7F, 0xBE, 0x4A, 0x6C, 0x2F, 0x00, 0x98, 0x01, 0x37, 0x03, 0x08, 0xFC, 0x4A, 0xD8
};

/* Groestl-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_groestl512_pat64[64] = {
	0x6E, 0x8C, 0x9B, 0x90, 0xE3, 0x6C, 0xEA, 0x68, 0xC0, 0x29, 0xA7, 0xD8, 0xB9, 0x5B, 0x71, 0x8C,
	0x84, 0x20, 0x5D, 0x81, 0xBE, 0x22, 0x7B, 0xA6, 0x15, 0x10, 0xF5, 0x67, 0xD4, 0x6B, 0x83, 0xED,
	0xD1, 0x1F, 0x30, 0x1B, 0xF1, 0xE7, 0x04, 0x1B, 0xE9, 0x91, 0xB2, 0x2F, 0xDB, 0xEE, 0x82, 0xDB,
	0xDC, 0xE7, 0xAB, 0x0E, 0x0E, 0xE4, 0x2A, 0x79, 0x5C, 0xA9, 0x65, 0xA4, 0x39, 0x53, 0x2A, 0x39
};

/* two hashes per thread, as groestl512_gpu_hash_64_x2: an odd count leaves a lone last hash */
__global__ __launch_bounds__(32, 1)
void groestl512_x2_selftest_gpu(uint32_t *io, int count)
{
	extern __shared__ uint32_t stash[];
	const int v0 = threadIdx.x * 2;
	if (v0 >= count) return;
	const bool two = v0 + 1 < count;
	uint32_t in[2][16];
	#pragma unroll
	for (int h = 0; h < 2; h++)
		#pragma unroll
		for (int w = 0; w < 16; w++) in[h][w] = io[((two ? v0 + h : v0) << 4) + w];
	groestl512_x2_hash_64(in, &stash[threadIdx.x], blockDim.x);
	#pragma unroll
	for (int h = 0; h < 2; h++)
		if (h == 0 || two)
			#pragma unroll
			for (int w = 0; w < 16; w++) io[((v0 + h) << 4) + w] = in[h][w];
}

static bool groestl512_x2_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) GROESTL512_ST_VEC * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) GROESTL512_ST_VEC * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	groestl512_x2_selftest_gpu <<<1, 32, 32 * 64 * sizeof(uint32_t)>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) GROESTL512_ST_VEC * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

/* two hashes per lane PAIR, as groestl512_gpu_hash_64_pair (the Pascal stage); lanes past the end compute
 * on vector 0 and skip the store */
__global__ __launch_bounds__(32, 1)
void groestl512_pair_selftest_gpu(uint32_t *io, int count)
{
	extern __shared__ uint32_t stash[];
	const uint32_t lane = threadIdx.x & 1;
	const int v0 = (threadIdx.x >> 1) * 2;
	const bool active = v0 < count, two = v0 + 1 < count;
	uint32_t inq[2][8];
	#pragma unroll
	for (int h = 0; h < 2; h++) {
		const int v = !active ? 0 : (two ? v0 + h : v0);
		#pragma unroll
		for (int n = 0; n < 8; n++) inq[h][n] = io[(v << 4) + 2 * n + lane];
	}
	groestl512_pair_hash_64(inq, &stash[threadIdx.x], blockDim.x);
	if (!active) return;
	#pragma unroll
	for (int h = 0; h < 2; h++)
		if (h == 0 || two)
			#pragma unroll
			for (int n = 0; n < 8; n++) io[((v0 + h) << 4) + 2 * n + lane] = inq[h][n];
}

static bool groestl512_pair_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) GROESTL512_ST_VEC * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) GROESTL512_ST_VEC * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	groestl512_pair_selftest_gpu <<<1, 32, 32 * 32 * sizeof(uint32_t)>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) GROESTL512_ST_VEC * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

/* 80-byte headers as groestl512_gpu_hash_80_x2 builds them: word 19 = nonce[2t], nonce[2t+1] */
__global__ __launch_bounds__(32, 1)
void groestl512_x2_80_selftest_gpu(const uint32_t *hdr, const uint32_t *nonce, uint32_t *out, int count)
{
	extern __shared__ uint32_t stash[];
	const int v0 = threadIdx.x * 2;
	if (v0 >= count) return;
	const bool two = v0 + 1 < count;
	uint32_t m[2][20];
	#pragma unroll
	for (int w = 0; w < 19; w++) m[0][w] = m[1][w] = hdr[w];
	m[0][19] = nonce[v0];
	m[1][19] = nonce[two ? v0 + 1 : v0];
	uint32_t s[8][8], d[2][16];
	groestl512_x2_load80(m, s);
	groestl512_x2_compress(s, &stash[threadIdx.x], blockDim.x);
	groestl512_x2_store(s, d);
	#pragma unroll
	for (int h = 0; h < 2; h++)
		if (h == 0 || two)
			#pragma unroll
			for (int w = 0; w < 16; w++) out[((v0 + h) << 4) + w] = d[h][w];
}

static bool groestl512_x2_80_selftest_run(const uint32_t *hdr20, const uint32_t *nonce, uint8_t (*dig)[64], int count)
{
	uint32_t *d_hdr = NULL, *d_nonce = NULL, *d_out = NULL;
	if (cudaMalloc(&d_hdr, 20 * 4) != cudaSuccess || cudaMalloc(&d_nonce, (size_t) count * 4) != cudaSuccess
	 || cudaMalloc(&d_out, (size_t) count * 64) != cudaSuccess) {
		cudaFree(d_hdr); cudaFree(d_nonce); cudaFree(d_out);
		return selftest_cuda_fault();
	}
	bool ok = cudaMemcpy(d_hdr, hdr20, 20 * 4, cudaMemcpyHostToDevice) == cudaSuccess
	       && cudaMemcpy(d_nonce, nonce, (size_t) count * 4, cudaMemcpyHostToDevice) == cudaSuccess;
	groestl512_x2_80_selftest_gpu <<<1, 32, 32 * 64 * sizeof(uint32_t)>>> (d_hdr, d_nonce, d_out, count);
	ok = ok && (cudaMemcpy(dig, d_out, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_hdr); cudaFree(d_nonce); cudaFree(d_out);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool groestl512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_groestl512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_groestl512_init(&ctx);
	sph_groestl512(&ctx, NULL, 0);
	sph_groestl512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_groestl512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[GROESTL512_ST_VEC][64], ref[GROESTL512_ST_VEC][64];
	uint32_t seed = 0x47524F45; /* 'GROE' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < GROESTL512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < GROESTL512_ST_VEC; v++) {
		sph_groestl512_init(&ctx);
		sph_groestl512(&ctx, msg[v], 64);
		sph_groestl512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_groestl512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[GROESTL512_ST_VEC][64];
	bool gpu_ok = groestl512_x2_selftest_run(msg, gpu, GROESTL512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- an odd count: the last thread hashes one message, the slot after it stays untouched ---
	uint8_t odd[GROESTL512_ST_VEC][64];
	const bool odd_ok = groestl512_x2_selftest_run(msg, odd, GROESTL512_ST_VEC - 1)
	      && (memcmp(odd, ref, (GROESTL512_ST_VEC - 1) * 64) == 0)
	      && (memcmp(odd[GROESTL512_ST_VEC - 1], msg[GROESTL512_ST_VEC - 1], 64) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[GROESTL512_ST_VEC][64], negdig[GROESTL512_ST_VEC][64];
	memcpy(negmsg, msg, sizeof(negmsg));
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = groestl512_x2_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	// --- 80-byte loader: one header, 3 nonces (2 + a lone one) vs sph ---
	uint32_t hdr[20], nonce[3] = { 0x00000000u, 0x01000000u, 0xefbeaddeu };
	for (int i = 0; i < 20; i++) hdr[i] = 0x04030201u * (uint32_t)(i + 1);
	uint8_t d80[3][64];
	bool h80_ok = groestl512_x2_80_selftest_run(hdr, nonce, d80, 3);
	for (int v = 0; v < 3 && h80_ok; v++) {
		uint32_t m80[20]; uint8_t r80[64];
		memcpy(m80, hdr, 76); m80[19] = nonce[v];
		sph_groestl512_init(&ctx);
		sph_groestl512(&ctx, m80, 80);
		sph_groestl512_close(&ctx, r80);
		h80_ok = memcmp(d80[v], r80, 64) == 0;
	}

	// --- the lane-pair path (Pascal stage), odd count: lone last hash, the slot after it untouched ---
	uint8_t pr[GROESTL512_ST_VEC][64];
	const bool pair_ok = groestl512_pair_selftest_run(msg, pr, GROESTL512_ST_VEC - 1)
	      && (memcmp(pr, ref, (GROESTL512_ST_VEC - 1) * 64) == 0)
	      && (memcmp(pr[GROESTL512_ST_VEC - 1], msg[GROESTL512_ST_VEC - 1], 64) == 0);

	passed = sph_ok && kat_ok && gpu_ok && odd_ok && neg_ok && h80_ok && pair_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "groestl512 device-library self-test FAILED (sph %d kat %d gpu %d odd %d neg %d 80 %d pair %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) odd_ok, (int) neg_ok, (int) h80_ok, (int) pair_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "groestl512 device-library self-test passed");
	return selftest_gate(thr_id, "groestl512", passed);
}

/* ----------------------------------------------------------------- echo512 */

#define ECHO512_ST_VEC 4

/* ECHO submission KAT: ECHO-512 of the empty message — anchors the sph
 * reference outside this codebase. */
static const uint8_t kat_echo512_empty[64] = {
	0x15, 0x8F, 0x58, 0xCC, 0x79, 0xD3, 0x00, 0xA9, 0xAA, 0x29, 0x25, 0x15, 0x04, 0x92, 0x75, 0xD0,
	0x51, 0xA2, 0x8A, 0xB9, 0x31, 0x72, 0x6D, 0x0E, 0xC4, 0x4B, 0xDD, 0x9F, 0xAE, 0xF4, 0xA7, 0x02,
	0xC3, 0x6D, 0xB9, 0xE7, 0x92, 0x2F, 0xFF, 0x07, 0x74, 0x02, 0x23, 0x64, 0x65, 0x83, 0x3C, 0x5C,
	0xC7, 0x6A, 0xF4, 0xEF, 0xC3, 0x52, 0xB4, 0xB4, 0x4C, 0x7F, 0xA1, 0x5A, 0xA0, 0xEF, 0x23, 0x4E
};

/* ECHO-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_echo512_pat64[64] = {
	0x2F, 0x7A, 0x64, 0xCE, 0xC7, 0xE0, 0x7C, 0x9D, 0x79, 0x1F, 0x90, 0x2B, 0x83, 0x8E, 0x9A, 0x77,
	0x6C, 0x03, 0xDA, 0x43, 0xEF, 0x88, 0x58, 0xE8, 0x9C, 0x16, 0xBB, 0xFA, 0x7E, 0xFF, 0x64, 0x1D,
	0x5E, 0x30, 0x9D, 0x9A, 0x51, 0xE1, 0x31, 0x77, 0xCB, 0xB8, 0x6F, 0xB1, 0x02, 0x10, 0x70, 0xC6,
	0x47, 0x63, 0xFA, 0x93, 0xB3, 0x98, 0x24, 0xDA, 0xFD, 0x77, 0x31, 0x54, 0xCF, 0x2E, 0xC0, 0x58
};

/* One 128-thread block (echo_gpu_init fills the shared AES table with
 * threads < 128); threads 0..count-1 then hash one vector each in place. */
__global__ __launch_bounds__(128, 1)
void echo512_selftest_gpu(uint32_t *io, int count)
{
	__shared__ uint32_t sharedMemory[1024];

	echo_gpu_init(sharedMemory);
	__syncthreads(); // barrier: shared AES table filled cooperatively

	if (threadIdx.x < count)
		cuda_echo_round(sharedMemory, &io[threadIdx.x << 4]);
}

static bool echo512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	echo512_selftest_gpu <<<1, 128>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool echo512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_echo512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_echo512_init(&ctx);
	sph_echo512(&ctx, NULL, 0);
	sph_echo512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_echo512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[ECHO512_ST_VEC][64], ref[ECHO512_ST_VEC][64];
	uint32_t seed = 0x4543484F; /* 'ECHO' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < ECHO512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < ECHO512_ST_VEC; v++) {
		sph_echo512_init(&ctx);
		sph_echo512(&ctx, msg[v], 64);
		sph_echo512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_echo512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[ECHO512_ST_VEC][64];
	bool gpu_ok = echo512_selftest_run(msg, gpu, ECHO512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = echo512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "echo512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "echo512 device-library self-test passed");
	return selftest_gate(thr_id, "echo512", passed);
}

/* ---------------------------------------------------- echo512 (alexis x16) */

/* Same ECHO-512 function as above, alexis formulation — validated against
 * the same sph reference and baked vectors (kat_echo512_*). */
__global__ __launch_bounds__(128, 1)
void echo512_alexis_selftest_gpu(uint32_t *io, int count)
{
	__shared__ uint32_t sharedMemory[1][ECHO_TAB];
	echo_fill_rep(sharedMemory);
	__syncthreads(); // barrier: shared AES table filled cooperatively

	if (threadIdx.x < count) {
		uint32_t hash[16];
		uint32_t *h = &io[threadIdx.x << 4];

		#pragma unroll 16
		for (int i = 0; i < 16; i++) hash[i] = h[i];

		echo512_hash_64_alexis(sharedMemory, hash);

		#pragma unroll 16
		for (int i = 0; i < 16; i++) h[i] = hash[i];
	}
}

static bool echo512_alexis_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	echo512_alexis_selftest_gpu <<<1, 128>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool echo512_alexis_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_echo512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_echo512_init(&ctx);
	sph_echo512(&ctx, NULL, 0);
	sph_echo512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_echo512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[ECHO512_ST_VEC][64], ref[ECHO512_ST_VEC][64];
	uint32_t seed = 0x4543484F; /* 'ECHO' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < ECHO512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < ECHO512_ST_VEC; v++) {
		sph_echo512_init(&ctx);
		sph_echo512(&ctx, msg[v], 64);
		sph_echo512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_echo512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[ECHO512_ST_VEC][64];
	bool gpu_ok = echo512_alexis_selftest_run(msg, gpu, ECHO512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = echo512_alexis_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "echo512-alexis device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "echo512-alexis device-library self-test passed");
	return selftest_gate(thr_id, "echo512-alexis", passed);
}

/* -------------------------------------------------------------- shavite512 */

#define SHAVITE512_ST_VEC 4

/* SHAvite-3 submission KAT: SHAvite-512 of the empty message — anchors the
 * sph reference outside this codebase. */
static const uint8_t kat_shavite512_empty[64] = {
	0xA4, 0x85, 0xC1, 0xB2, 0x57, 0x84, 0x59, 0xD1, 0xEF, 0xC5, 0xDD, 0xDD, 0x84, 0x0B, 0xB0, 0xB4,
	0xA6, 0x50, 0xAC, 0x82, 0xFE, 0x68, 0xF5, 0x8C, 0x44, 0x42, 0xCC, 0xDA, 0x74, 0x7D, 0xA0, 0x06,
	0xB2, 0xD1, 0xDC, 0x6B, 0x4A, 0x4E, 0xB7, 0xD8, 0x4F, 0xF9, 0x1E, 0x1F, 0x46, 0x6F, 0xEF, 0x42,
	0x9D, 0x25, 0x9A, 0xCD, 0x99, 0x5D, 0xDD, 0xCA, 0xD1, 0x6F, 0xA5, 0x45, 0xC7, 0xA6, 0xE5, 0xBA
};

/* SHAvite-512 of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_shavite512_pat64[64] = {
	0x4B, 0x53, 0x73, 0x45, 0x38, 0xB1, 0x13, 0xC1, 0x63, 0x71, 0x04, 0x88, 0x7E, 0x9F, 0x21, 0x50,
	0xFA, 0x4A, 0xD9, 0xEC, 0x70, 0x55, 0x2D, 0x8E, 0xD6, 0x2F, 0x01, 0x34, 0xA4, 0x7A, 0x2F, 0x4E,
	0x81, 0x34, 0xB2, 0x36, 0x69, 0x32, 0x98, 0x3B, 0x41, 0x27, 0xCB, 0xCB, 0xA5, 0x9C, 0xDA, 0x04,
	0xBF, 0x6D, 0x00, 0x05, 0xB5, 0xBA, 0x04, 0xDE, 0xA9, 0x28, 0x79, 0xF1, 0x5E, 0x80, 0xA2, 0x8A
};

/* One 128-thread block (shavite_gpu_init fills the shared AES table with
 * threads < 128); threads 0..count-1 then hash one vector each in place.
 * State init and 64-byte padding mirror x11_shavite512_gpu_hash_64. */
__global__ __launch_bounds__(128, 1)
void shavite512_selftest_gpu(uint32_t *io, int count)
{
	__shared__ uint32_t sharedMemory[1024];

	shavite_gpu_init(sharedMemory);
	__syncthreads(); // barrier: shared AES table filled cooperatively

	if (threadIdx.x < count) {
		uint32_t *Hash = &io[threadIdx.x << 4];

		uint32_t state[16] = {
			SPH_C32(0x72FCCDD8), SPH_C32(0x79CA4727), SPH_C32(0x128A077B), SPH_C32(0x40D55AEC),
			SPH_C32(0xD1901A06), SPH_C32(0x430AE307), SPH_C32(0xB29F5CD1), SPH_C32(0xDF07FBFC),
			SPH_C32(0x8E45D73D), SPH_C32(0x681AB538), SPH_C32(0xBDE86578), SPH_C32(0xDD577E47),
			SPH_C32(0xE275EADE), SPH_C32(0x502D9FCD), SPH_C32(0xB9357178), SPH_C32(0x022A4B9A)
		};

		uint32_t msg[32];

		#pragma unroll 16
		for (int i = 0; i < 16; i++)
			msg[i] = Hash[i];

		msg[16] = 0x80;
		#pragma unroll 10
		for (int i = 17; i < 27; i++)
			msg[i] = 0;

		msg[27] = 0x02000000;
		msg[28] = 0;
		msg[29] = 0;
		msg[30] = 0;
		msg[31] = 0x02000000;

		c512(sharedMemory, state, msg, 512);

		#pragma unroll 16
		for (int i = 0; i < 16; i++)
			Hash[i] = state[i];
	}
}

static bool shavite512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	shavite512_selftest_gpu <<<1, 128>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool shavite512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_shavite512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_shavite512_init(&ctx);
	sph_shavite512(&ctx, NULL, 0);
	sph_shavite512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_shavite512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[SHAVITE512_ST_VEC][64], ref[SHAVITE512_ST_VEC][64];
	uint32_t seed = 0x53484156; /* 'SHAV' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < SHAVITE512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < SHAVITE512_ST_VEC; v++) {
		sph_shavite512_init(&ctx);
		sph_shavite512(&ctx, msg[v], 64);
		sph_shavite512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_shavite512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[SHAVITE512_ST_VEC][64];
	bool gpu_ok = shavite512_selftest_run(msg, gpu, SHAVITE512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = shavite512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "shavite512 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "shavite512 device-library self-test passed");
	return selftest_gate(thr_id, "shavite512", passed);
}

/* ------------------------------------------------------------ whirlpool512 */

#define WHIRLPOOL512_ST_VEC 4

/* ISO/NESSIE KAT: Whirlpool of the empty message — anchors the sph
 * reference outside this codebase. */
static const uint8_t kat_whirlpool512_empty[64] = {
	0x19, 0xFA, 0x61, 0xD7, 0x55, 0x22, 0xA4, 0x66, 0x9B, 0x44, 0xE3, 0x9C, 0x1D, 0x2E, 0x17, 0x26,
	0xC5, 0x30, 0x23, 0x21, 0x30, 0xD4, 0x07, 0xF8, 0x9A, 0xFE, 0xE0, 0x96, 0x49, 0x97, 0xF7, 0xA7,
	0x3E, 0x83, 0xBE, 0x69, 0x8B, 0x28, 0x8F, 0xEB, 0xCF, 0x88, 0xE3, 0xE0, 0x3C, 0x4F, 0x07, 0x57,
	0xEA, 0x89, 0x64, 0xE5, 0x9B, 0x63, 0xD9, 0x37, 0x08, 0xB1, 0x38, 0xCC, 0x42, 0xA6, 0x6E, 0xB3
};

/* Whirlpool of the 64-byte pattern 00 01 .. 3F, computed once from the
 * anchored sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_whirlpool512_pat64[64] = {
	0x5C, 0x3C, 0x6F, 0x52, 0x4C, 0x8A, 0xE1, 0xE7, 0xA4, 0xF7, 0x6B, 0x84, 0x97, 0x7B, 0x15, 0x60,
	0xE7, 0x8E, 0xB5, 0x68, 0xE2, 0xFD, 0x8D, 0x72, 0x69, 0x9A, 0xD7, 0x91, 0x86, 0x48, 0x1B, 0xD4,
	0x2B, 0x53, 0xAB, 0x39, 0xA0, 0xB7, 0x41, 0xD9, 0xC0, 0x98, 0xA4, 0xEC, 0xB0, 0x1F, 0x3E, 0xCC,
	0xF3, 0x84, 0x4C, 0xF1, 0xB7, 0x3A, 0x93, 0x55, 0xEE, 0x5D, 0x49, 0x6A, 0x2A, 0x1F, 0xB5, 0xB3
};

/* four hashes per thread, as whirlpool512_gpu_hash_64_x4; count 1..4 per thread covers the
 * partial last thread; slots past count are not written */
__global__ __launch_bounds__(32, 1)
void whirlpool512_selftest_gpu(uint32_t *io, int count)
{
	extern __shared__ uint32_t stash[];
	const int v0 = threadIdx.x * 4;
	if (v0 >= count) return;
	uint32_t in[4][16];
	#pragma unroll
	for (int h = 0; h < 4; h++) {
		const int v = v0 + h < count ? v0 + h : v0;
		#pragma unroll
		for (int w = 0; w < 16; w++) in[h][w] = io[(v << 4) + w];
	}
	whirlpool512_x4_hash_64(in, &stash[threadIdx.x], blockDim.x, io, v0, min(4, count - v0));
}

static bool whirlpool512_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) WHIRLPOOL512_ST_VEC * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) WHIRLPOOL512_ST_VEC * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	whirlpool512_selftest_gpu <<<1, 32, 32 * 64 * sizeof(uint32_t)>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) WHIRLPOOL512_ST_VEC * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool whirlpool512_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_whirlpool_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the official spec vector ---
	sph_whirlpool_init(&ctx);
	sph_whirlpool(&ctx, NULL, 0);
	sph_whirlpool_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_whirlpool512_empty, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[WHIRLPOOL512_ST_VEC][64], ref[WHIRLPOOL512_ST_VEC][64];
	uint32_t seed = 0x57484952; /* 'WHIR' */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < WHIRLPOOL512_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < WHIRLPOOL512_ST_VEC; v++) {
		sph_whirlpool_init(&ctx);
		sph_whirlpool(&ctx, msg[v], 64);
		sph_whirlpool_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_whirlpool512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests ---
	uint8_t gpu[WHIRLPOOL512_ST_VEC][64];
	bool gpu_ok = whirlpool512_selftest_run(msg, gpu, WHIRLPOOL512_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- a partial thread: 3 and 1 hashes, the slots after them stay untouched ---
	bool odd_ok = true;
	for (int n = 3; n >= 1 && odd_ok; n -= 2) {
		uint8_t odd[WHIRLPOOL512_ST_VEC][64];
		odd_ok = whirlpool512_selftest_run(msg, odd, n)
		      && (memcmp(odd, ref, (size_t) n * 64) == 0)
		      && (memcmp(odd[n], msg[n], (size_t) (WHIRLPOOL512_ST_VEC - n) * 64) == 0);
	}

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[WHIRLPOOL512_ST_VEC][64], negdig[WHIRLPOOL512_ST_VEC][64];
	memcpy(negmsg, msg, sizeof(negmsg));
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = whirlpool512_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 64) != 0);

	passed = sph_ok && kat_ok && gpu_ok && odd_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "whirlpool512 device-library self-test FAILED (sph %d kat %d gpu %d odd %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) odd_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "whirlpool512 device-library self-test passed");
	return selftest_gate(thr_id, "whirlpool512", passed);
}

/* -------------------------------------------------------- x-fused chain */

/* Validates the fused multi-stage kernel (algos/common/cuda_x_fused.cu)
 * against the sph chain: per-stage single launches first (pinpoints a broken
 * glue path), then one chained launch (validates register-resident state
 * carry). Uses the caller-visible launcher, so the constant order array is
 * clobbered — callers must re-upload their order afterwards. */
extern void x_fused_setOrder(const uint8_t *ids, int count);
extern void x_fused_cpu_hash_64(int thr_id, uint32_t threads, int start, int len, int has_tiger, uint32_t *d_hash);
#include "algos/common/cuda_x_fused.h"

static void x_fused_sph_stage(int id, uint8_t *h /* 64 bytes in/out */)
{
	switch (id) {
	case 0:  { sph_blake512_context c;   sph_blake512_init(&c);   sph_blake512(&c, h, 64);   sph_blake512_close(&c, h);   break; }
	case 1:  { sph_bmw512_context c;     sph_bmw512_init(&c);     sph_bmw512(&c, h, 64);     sph_bmw512_close(&c, h);     break; }
	case 3:  { sph_jh512_context c;      sph_jh512_init(&c);      sph_jh512(&c, h, 64);      sph_jh512_close(&c, h);      break; }
	case 4:  { sph_keccak512_context c;  sph_keccak512_init(&c);  sph_keccak512(&c, h, 64);  sph_keccak512_close(&c, h);  break; }
	case 5:  { sph_skein512_context c;   sph_skein512_init(&c);   sph_skein512(&c, h, 64);   sph_skein512_close(&c, h);   break; }
	case 6:  { sph_luffa512_context c;   sph_luffa512_init(&c);   sph_luffa512(&c, h, 64);   sph_luffa512_close(&c, h);   break; }
	case 7:  { sph_cubehash512_context c; sph_cubehash512_init(&c); sph_cubehash512(&c, h, 64); sph_cubehash512_close(&c, h); break; }
	case 11: { sph_hamsi512_context c;   sph_hamsi512_init(&c);   sph_hamsi512(&c, h, 64);   sph_hamsi512_close(&c, h);   break; }
	case 13: { sph_shabal512_context c;  sph_shabal512_init(&c);  sph_shabal512(&c, h, 64);  sph_shabal512_close(&c, h);  break; }
	case 15: { sph_sha512_context c;     sph_sha512_init(&c);     sph_sha512(&c, h, 64);     sph_sha512_close(&c, h);     break; }
	case X_FUSED_SHAVITE: { sph_shavite512_context c; sph_shavite512_init(&c); sph_shavite512(&c, h, 64); sph_shavite512_close(&c, h); break; }
	case 16: { /* tiger192, zero-padded */
		sph_tiger_context c; uint8_t d[24];
		sph_tiger_init(&c); sph_tiger(&c, h, 64); sph_tiger_close(&c, d);
		memcpy(h, d, 24); memset(h + 24, 0, 40); break; }
	}
}

static bool x_fused_selftest_run(const uint8_t *ids, int nids, int has_tiger,
	const uint8_t msg[64], uint8_t dig[64])
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, 64) != cudaSuccess)
		return selftest_cuda_fault();

	x_fused_setOrder(ids, nids);
	bool ok = (cudaMemcpy(d_io, msg, 64, cudaMemcpyHostToDevice) == cudaSuccess);
	x_fused_cpu_hash_64(0, 1, 0, nids, has_tiger, d_io);
	cudaDeviceSynchronize();
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		applog(LOG_WARNING, "x-fused run(nids=%d id0=%u tiger=%d) cuda error %d: %s",
			nids, ids[0], has_tiger, (int)e, cudaGetErrorString(e));
	ok = ok && (cudaMemcpy(dig, d_io, 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool x_fused_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	static const uint8_t all_ids[11] = { 0, 1, 3, 4, 5, 6, 7, 11, 13, 15, 16 };

	uint8_t msg[64];
	uint32_t seed = 0x46555345; /* 'FUSE' */
	for (int i = 0; i < 64; i++) {
		seed = seed * 1664525u + 1013904223u;
		msg[i] = (uint8_t)(seed >> 24);
	}

	// --- each fusible stage alone vs its sph reference ---
	bool single_ok = true;
	for (int k = 0; k < 11; k++) {
		const uint8_t id = all_ids[k];
		uint8_t ref[64], gpu[64];
		memcpy(ref, msg, 64);
		x_fused_sph_stage(id, ref);
		if (!x_fused_selftest_run(&id, 1, (id == 16), msg, gpu)
		    || memcmp(gpu, ref, 64) != 0) {
			gpulog(LOG_WARNING, thr_id, "x-fused stage id %u FAILED single-stage check", id);
			single_ok = false;
		}
	}

	// --- the full fusible set as one chained launch ---
	uint8_t ref[64], gpu[64];
	memcpy(ref, msg, 64);
	for (int k = 0; k < 11; k++)
		x_fused_sph_stage(all_ids[k], ref);
	const bool chain_ok = x_fused_selftest_run(all_ids, 11, 1, msg, gpu)
	                   && (memcmp(gpu, ref, 64) == 0);

	// --- specific fused-run adjacencies seen in live x16r orders that the
	//     degenerate benchmark order (0123456789ABCDEF) never exercises:
	//     it leaves bmw/hamsi/shabal/sha512 standalone, so a fused sha512 or a
	//     bmw.bmw pair is otherwise untested. Regression coverage. ---
	static const uint8_t adj_runs[3][4] = {
		{ 15, 5, 4, 0 },   /* sha512,skein,keccak         (len 3) */
		{ 3, 4, 1, 1 },    /* jh,keccak,bmw,bmw           (len 4) */
		{ 1, 1, 15, 0 },   /* bmw,bmw,sha512              (len 3) */
	};
	static const int adj_len[3] = { 3, 4, 3 };
	bool adj_ok = true;
	for (int r = 0; r < 3; r++) {
		uint8_t rref[64], rgpu[64];
		memcpy(rref, msg, 64);
		for (int k = 0; k < adj_len[r]; k++)
			x_fused_sph_stage(adj_runs[r][k], rref);
		if (!x_fused_selftest_run(adj_runs[r], adj_len[r], 0, msg, rgpu)
		    || memcmp(rgpu, rref, 64) != 0) {
			gpulog(LOG_WARNING, thr_id, "x-fused adjacency run %d FAILED", r);
			adj_ok = false;
		}
	}

	// --- the compile-time-specialised kernels of the fixed-order chains ---
	bool fixed_ok = true;
	for (int seq = 0; seq < XF_FIXED_COUNT; seq++) {
		uint8_t ids[8], rref[64], rgpu[64];
		const int n = x_fused_fixed_ids(seq, ids);
		memcpy(rref, msg, 64);
		for (int k = 0; k < n; k++)
			x_fused_sph_stage(ids[k], rref);
		uint32_t *d_io = NULL;
		bool ok = n > 0 && cudaMalloc(&d_io, 64) == cudaSuccess
		       && cudaMemcpy(d_io, msg, 64, cudaMemcpyHostToDevice) == cudaSuccess;
		if (ok) {
			x_fused_fixed_cpu_hash_64(1, seq, d_io);
			ok = cudaDeviceSynchronize() == cudaSuccess
			  && cudaMemcpy(rgpu, d_io, 64, cudaMemcpyDeviceToHost) == cudaSuccess;
		}
		if (d_io) cudaFree(d_io);
		if (!ok) return selftest_cuda_fault();
		if (memcmp(rgpu, rref, 64) != 0) {
			gpulog(LOG_WARNING, thr_id, "x-fused fixed sequence %d FAILED", seq);
			fixed_ok = false;
		}
	}

	passed = single_ok && chain_ok && adj_ok && fixed_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "x-fused device self-test FAILED (single %d chain %d adj %d fixed %d)",
			(int) single_ok, (int) chain_ok, (int) adj_ok, (int) fixed_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "x-fused device self-test passed");
	return selftest_gate(thr_id, "x-fused", passed);
}

/* ---------------------------------------------------------------- tiger192 */

#define TIGER192_ST_VEC 4

/* Official Tiger KAT: Tiger of the empty message (24-byte digest) — anchors
 * the sph reference outside this codebase. */
static const uint8_t kat_tiger192_empty[24] = {
	0x32, 0x93, 0xAC, 0x63, 0x0C, 0x13, 0xF0, 0x24, 0x5F, 0x92, 0xBB, 0xB1,
	0x76, 0x6E, 0x16, 0x16, 0x7A, 0x4E, 0x58, 0x49, 0x2D, 0xDE, 0x73, 0xF3
};

/* Tiger of the 64-byte pattern 00 01 .. 3F, computed once from the anchored
 * sph reference and fixed here so drift on either side is caught. */
static const uint8_t kat_tiger192_pat64[24] = {
	0x21, 0x2D, 0xF8, 0x9C, 0x57, 0x15, 0x52, 0x70, 0x34, 0x4A, 0xCC, 0xB1,
	0x90, 0x27, 0xB0, 0xB2, 0x6B, 0x10, 0x4F, 0xA0, 0xFB, 0xBE, 0x0F, 0xE4
};

/* One 256-thread block (the donor's shared fill is UNGUARDED — exactly one
 * uint64 per thread); threads 0..count-1 hash one 64-byte vector each in
 * place, mirroring tiger192_gpu_hash_64 with zero_pad_64 = 1. */
__global__ __launch_bounds__(256, 1)
void tiger192_selftest_gpu(uint32_t *io, int count)
{
	__shared__ uint64_t sharedMem[768];

	tiger192_load_shared(sharedMem);
	__syncthreads();

	if (threadIdx.x < count) {
		uint64_t *inout = (uint64_t*)&io[threadIdx.x << 4];
		uint64_t buf[3], in[8];

		#pragma unroll
		for (int i = 0; i < 8; i++) in[i] = inout[i];

		tiger192_hash_64(sharedMem, in, buf);

		#pragma unroll
		for (int i = 0; i < 3; i++) inout[i] = buf[i];
		#pragma unroll
		for (int i = 3; i < 8; i++) inout[i] = 0;
	}
}

static bool tiger192_selftest_run(const uint8_t (*msg)[64], uint8_t (*dig)[64], int count)
{
	uint32_t *d_io = NULL;
	if (cudaMalloc(&d_io, (size_t) count * 64) != cudaSuccess)
		return selftest_cuda_fault();

	bool ok = (cudaMemcpy(d_io, msg, (size_t) count * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	tiger192_selftest_gpu <<<1, 256>>> (d_io, count);
	ok = ok && (cudaMemcpy(dig, d_io, (size_t) count * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_io);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool tiger192_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_tiger_context ctx;
	uint8_t dig[24];

	// --- anchor the sph reference against the official spec vector ---
	sph_tiger_init(&ctx);
	sph_tiger(&ctx, NULL, 0);
	sph_tiger_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_tiger192_empty, 24) == 0);

	// --- test vectors: fixed pattern + LCG-filled ---
	uint8_t msg[TIGER192_ST_VEC][64], ref[TIGER192_ST_VEC][64];
	uint32_t seed = 0x54494752; /* 'TIGR' */
	memset(ref, 0, sizeof(ref)); /* GPU zero-pads words 3..7: ref must too */
	for (int i = 0; i < 64; i++)
		msg[0][i] = (uint8_t) i;
	for (int v = 1; v < TIGER192_ST_VEC; v++) {
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			msg[v][i] = (uint8_t)(seed >> 24);
		}
	}
	for (int v = 0; v < TIGER192_ST_VEC; v++) {
		sph_tiger_init(&ctx);
		sph_tiger(&ctx, msg[v], 64);
		sph_tiger_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_tiger192_pat64, 24) == 0);

	// --- GPU hash vs the sph digests (24-byte digest + 40 zero bytes) ---
	uint8_t gpu[TIGER192_ST_VEC][64];
	bool gpu_ok = tiger192_selftest_run(msg, gpu, TIGER192_ST_VEC)
	           && (memcmp(gpu, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	uint8_t negmsg[1][64], negdig[1][64];
	memcpy(negmsg[0], msg[0], 64);
	negmsg[0][0] ^= 0x01;
	const bool neg_ok = tiger192_selftest_run(negmsg, negdig, 1)
	                 && (memcmp(negdig[0], ref[0], 24) != 0);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "tiger192 device-library self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "tiger192 device-library self-test passed");
	return selftest_gate(thr_id, "tiger192", passed);
}
