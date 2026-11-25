/**
 * Whirlpool-512 CUDA implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014-2016 djm34, tpruvot, SP, Provos Alexis
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
 * @author Provos Alexis (Applied partial shared memory utilization, precomputations, merging & tuning for 970/750ti under CUDA7.5 -> +93% increased throughput of whirlpool)
 */

extern "C" {
#include <sph/sph_whirlpool.h>
#include <miner.h>
}

#include "cuda/helper.cuh"

#include "cuda_whirlpool_tables.cuh"
#include "cuda/whirlpool512_x4_device.cuh"
#include "cuda/candidate_report.cuh"
#include "cuda/selftest_gate.cuh"

/* four hashes per thread, ~255 registers + 64 words of shared stash per thread */
#define TPB80 128
#define TPB64_X4 128

static uint32_t *d_resNonce[MAX_GPUS] = { 0 };

/* whirlpool1 80-byte header (-a whirlpool): planes of H1 ^ block 2 and the 10 block-2 round keys */
static __constant__ uint32_t c_wc80_st0[64];
static __constant__ uint32_t c_wc80_key[10][64];

/* Unit self-test for cuda/whirlpool512_x4_device.cuh (docs/coding-guideline.md
 * section 7 layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool whirlpool512_device_selftest(int thr_id);
static bool whirlcoin_selftest(int thr_id);

__host__
void whirlpool512_cpu_init(int thr_id, uint32_t threads, int mode)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

	cuda_get_arch(thr_id);

	whirlpool512_device_selftest(thr_id);
	if (mode == 1)
		whirlcoin_selftest(thr_id);
}

/* whirlpool1 (legacy Whirlcoin) chaining value after the first 64 bytes */
__host__
static void whirl_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, input, 64);

	memcpy(state, ctx.state, 64);
}

__host__
void whirlpool512_setBlock_80(void *pdata, const void *ptarget)
{
	uint64_t h1[8];
	uint32_t planes[11][64];
	whirl_midstate(h1, pdata);
	whirlpool512_x4_prepare80(old1_T0, old1_RC, h1, pdata, planes);
	cudaMemcpyToSymbol(c_wc80_st0, planes[0], sizeof(c_wc80_st0), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_wc80_key, planes[1], sizeof(c_wc80_key), 0, cudaMemcpyHostToDevice);
}

__host__
extern void whirlpool512_cpu_free(int thr_id)
{
	if (d_resNonce[thr_id])
		cudaFree(d_resNonce[thr_id]);
}

/* 4 x whirlpool1 (80, 64, 64, 64 bytes) chained in planes, then words 6..7 vs the target */
__global__
__launch_bounds__(TPB80, 2)
void whirlcoin_gpu_hash_80_x4(uint32_t threads, uint32_t startNounce, uint32_t *resNonce, const uint64_t target)
{
	extern __shared__ uint32_t stash[];                     /* [64][TPB80] */
	const uint32_t i0 = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
	if (i0 >= threads) return;

	uint32_t s[8][8];
	whirlpool512_x4_hash80<true>(c_wc80_st0, c_wc80_key, startNounce + i0, s);
	#pragma unroll 1
	for (int k = 0; k < 3; k++)
		whirlpool512_x4_compress64<true>(s, c_wpx4t_k1, &stash[threadIdx.x], blockDim.x);
	#pragma unroll
	for (int c = 0; c < 8; c++) whirlpool512_x4_transpose(s[c]);

	/* digest words 6, 7 = row 3, cols 0..3 / 4..7 -> W_c[4 + h] byte 1 */
	#pragma unroll
	for (int h = 0; h < 4; h++) {
		if (i0 + h >= threads) break;
		const uint32_t w6 = __byte_perm(__byte_perm(s[0][4 + h], s[1][4 + h], 0x0051), __byte_perm(s[2][4 + h], s[3][4 + h], 0x0051), 0x5410);
		const uint32_t w7 = __byte_perm(__byte_perm(s[4][4 + h], s[5][4 + h], 0x0051), __byte_perm(s[6][4 + h], s[7][4 + h], 0x0051), 0x5410);
		if ((((uint64_t) w7 << 32) | w6) <= target)
			report_candidate_2(resNonce, i0 + h);
	}
}

__host__
void whirlpool512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *h_resNonces, const uint64_t target)
{
	const uint32_t nthr = (threads + 3) / 4;   /* four hashes per thread */
	dim3 grid((nthr + TPB80 - 1) / TPB80);
	dim3 block(TPB80);

	cudaMemset(d_resNonce[thr_id], 0xff, 2*sizeof(uint32_t));

	whirlcoin_gpu_hash_80_x4<<<grid, block, 64 * TPB80 * sizeof(uint32_t)>>>(threads, startNounce, d_resNonce[thr_id], target);

	cudaMemcpy(h_resNonces, d_resNonce[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (h_resNonces[0] != UINT32_MAX) h_resNonces[0] += startNounce;
	if (h_resNonces[1] != UINT32_MAX) h_resNonces[1] += startNounce;
}

/* Init self-test of the -a whirlpool chain vs 4 x sph_whirlpool1, at each hash's value and value - 1
 * (the range avoids nonce 0xffffffff, the report's "none"). Clobbers the job constants. Fail-closed. */
static bool whirlcoin_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	uint8_t hdr[80];
	for (int i = 0; i < 80; i++) hdr[i] = (uint8_t)(i * 29 + 3);
	const uint32_t start = 0x7ffffffeu;
	uint64_t v[3];
	for (int j = 0; j < 3; j++) {
		uint8_t m[80], h[64];
		memcpy(m, hdr, 80);
		const uint32_t n = start + (uint32_t) j;
		m[76] = n >> 24; m[77] = n >> 16; m[78] = n >> 8; m[79] = n;
		sph_whirlpool_context c;
		sph_whirlpool1_init(&c); sph_whirlpool1(&c, m, 80); sph_whirlpool1_close(&c, h);
		for (int k = 0; k < 3; k++) {
			sph_whirlpool1_init(&c); sph_whirlpool1(&c, h, 64); sph_whirlpool1_close(&c, h);
		}
		memcpy(&v[j], h + 24, 8);
	}
	whirlpool512_setBlock_80(hdr, NULL);
	int bad = 0;
	for (int t = 0; t < 6; t++) {
		const uint64_t tg = (t & 1) ? v[t >> 1] - 1 : v[t >> 1];
		uint32_t e[2] = { UINT32_MAX, UINT32_MAX }, res[2];
		for (int j = 0; j < 3; j++)
			if (v[j] <= tg) { if (e[0] == UINT32_MAX) e[0] = start + j; else if (e[1] == UINT32_MAX) e[1] = start + j; }
		whirlpool512_cpu_hash_80(thr_id, 3, start, res, tg);
		if (cudaPeekAtLastError() != cudaSuccess)
			return selftest_gate(thr_id, "whirlpool", selftest_cuda_fault());
		bad += res[0] != e[0] || res[1] != e[1];
	}
	passed = bad == 0;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "whirlpool self-test FAILED (%d of 6 targets wrong)", bad);
	else
		gpulog(LOG_DEBUG, thr_id, "whirlpool self-test passed");
	return selftest_gate(thr_id, "whirlpool", passed);
}

/* 64-byte stage: four hashes per thread (cuda/whirlpool512_x4_device.cuh), plain Whirlpool */
__global__
__launch_bounds__(TPB64_X4, 2)
void whirlpool512_gpu_hash_64_x4(uint32_t threads, uint32_t *g_hash)
{
	extern __shared__ uint32_t stash[];                     /* [64][TPB64_X4] */
	const uint32_t i0 = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
	if (i0 >= threads) return;

	/* slots past the batch reload the first hash and are not written */
	uint32_t in[4][16];
	#pragma unroll
	for (int h = 0; h < 4; h++) {
		const uint32_t i = (i0 + h < threads) ? i0 + h : i0;
		const uint4 *p = (const uint4*) &g_hash[(size_t)i << 4];
		#pragma unroll
		for (int k = 0; k < 4; k++) {
			const uint4 v = p[k];
			in[h][4*k] = v.x; in[h][4*k+1] = v.y; in[h][4*k+2] = v.z; in[h][4*k+3] = v.w;
		}
	}

	whirlpool512_x4_hash_64(in, &stash[threadIdx.x], blockDim.x, g_hash, i0, (int) min(4u, threads - i0));
}

__host__
static void whirlpool512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t nthr = (threads + 3) / 4;   /* four hashes per thread */
	dim3 grid((nthr + TPB64_X4 - 1) / TPB64_X4);
	dim3 block(TPB64_X4);

	whirlpool512_gpu_hash_64_x4 <<<grid, block, 64 * TPB64_X4 * sizeof(uint32_t)>>> (threads, d_hash);
}

__host__
void whirlpool512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	whirlpool512_cpu_hash_64(thr_id, threads, d_hash);
}

/* Legacy-name forwarders (x15 whirlpool) for the not-yet-migrated consumers
 * (x17/skydoge/hmq17, x21s, ghostrider, evohash, bastion); each drops out as
 * its family switches to the bare name. */
__host__ void x15_whirlpool_cpu_free(int thr_id)
{
	whirlpool512_cpu_free(thr_id);
}

__host__ void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	whirlpool512_cpu_hash_64(thr_id, threads, startNounce, d_nonceVector, d_hash, order);
}

