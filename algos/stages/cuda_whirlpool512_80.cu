/**
 * Whirlpool-512 CUDA implementation. (better for SM 3.0)
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014-2016 djm34, tpruvot, SP
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 * @author djm34 (initial draft)
 * @author tpruvot (dual old/whirlpool modes, midstate)
 * @author SP ("final" function opt and tuning)
 */

/* 80-byte header stage, four hashes per thread (cuda/whirlpool512_x4_device.cuh). Block 1 and the
 * block-2 round keys depend only on the job and are computed on the host. */

#include <stdio.h>
#include <memory.h>

extern "C" {
#include "sph/sph_whirlpool.h"
}
#include "cuda/helper.cuh"
#include <miner.h>
#include "cuda_whirlpool_tables.cuh"
#include "cuda/whirlpool512_x4_device.cuh"
#include "cuda/selftest_gate.cuh"

#define TPB80_X4 128

/* planes of H1 ^ block 2 (nonce bytes zero) and of the 10 block-2 round keys */
static __constant__ uint32_t c_wp80_st0[64];
static __constant__ uint32_t c_wp80_key[10][64];

__global__ __launch_bounds__(TPB80_X4, 2)
void whirlpool512_gpu_hash_80_x4(const uint32_t threads, const uint32_t startNounce, uint32_t *g_out)
{
	const uint32_t i0 = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
	if (i0 >= threads) return;

	uint32_t s[8][8];
	whirlpool512_x4_hash80<false>(c_wp80_st0, c_wp80_key, startNounce + i0, s);
	#pragma unroll
	for (int c = 0; c < 8; c++) whirlpool512_x4_transpose(s[c]);
	whirlpool512_x4_store(s, g_out, i0, (int) min(4u, threads - i0));
}

extern void whirlpool_midstate(void *state, const void *input);
void x16_whirlpool512_setBlock_80(void *pdata);
void x16_whirlpool512_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_outputHash);

/* Init self-test: 3 nonces across the 2^32 wrap vs sph, the next slot untouched, a flipped header
 * bit must change the digest. Clobbers the job constants. Fail-closed. */
static bool wp80_selftest_run(const uint8_t *hdr, uint8_t (*dig)[64])
{
	uint32_t *d = NULL;
	if (cudaMalloc(&d, 4 * 64) != cudaSuccess)
		return selftest_cuda_fault();
	bool ok = cudaMemset(d, 0x5a, 4 * 64) == cudaSuccess;
	x16_whirlpool512_setBlock_80((void*) hdr);
	x16_whirlpool512_hash_80(0, 3, 0xfffffffeu, d);
	ok = ok && cudaDeviceSynchronize() == cudaSuccess
	        && cudaMemcpy(dig, d, 4 * 64, cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d);
	return ok ? true : selftest_cuda_fault();
}

static bool wp80_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	uint8_t hdr[80], dig[4][64], neg[4][64];
	for (int i = 0; i < 80; i++) hdr[i] = (uint8_t)(i * 37 + 5);
	bool kat = wp80_selftest_run(hdr, dig), tail = kat;
	for (int v = 0; v < 3 && kat; v++) {
		const uint32_t n = 0xfffffffeu + (uint32_t) v;
		uint8_t m[80], ref[64];
		memcpy(m, hdr, 80);
		m[76] = n >> 24; m[77] = n >> 16; m[78] = n >> 8; m[79] = n;
		sph_whirlpool_context c;
		sph_whirlpool_init(&c); sph_whirlpool(&c, m, 80); sph_whirlpool_close(&c, ref);
		kat = memcmp(dig[v], ref, 64) == 0;
	}
	for (int i = 0; i < 64 && tail; i++) tail = dig[3][i] == 0x5a;
	hdr[0] ^= 1;
	const bool neg_ok = wp80_selftest_run(hdr, neg) && memcmp(neg[0], dig[0], 64) != 0;

	passed = kat && tail && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "whirlpool512-80 self-test FAILED (kat %d tail %d neg %d)", (int) kat, (int) tail, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "whirlpool512-80 self-test passed");
	return selftest_gate(thr_id, "whirlpool512-80", passed);
}

__host__
void x16_whirlpool512_cpu_init(int thr_id, uint32_t threads)
{
	wp80_device_selftest(thr_id);
}

__host__
void x16_whirlpool512_setBlock_80(void *pdata)
{
	uint64_t h1[8];
	uint32_t planes[11][64];
	whirlpool_midstate(h1, pdata);                           /* block 1, key = zero IV */
	whirlpool512_x4_prepare80(plain_T0, plain_RC, h1, pdata, planes);
	cudaMemcpyToSymbol(c_wp80_st0, planes[0], sizeof(c_wp80_st0), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_wp80_key, planes[1], sizeof(c_wp80_key), 0, cudaMemcpyHostToDevice);
}

__host__
void x16_whirlpool512_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_outputHash)
{
	const uint32_t nthr = (threads + 3) / 4;   /* four hashes per thread */
	dim3 grid((nthr + TPB80_X4 - 1) / TPB80_X4);
	dim3 block(TPB80_X4);
	whirlpool512_gpu_hash_80_x4 <<<grid, block>>> (threads, startNonce, d_outputHash);
}
