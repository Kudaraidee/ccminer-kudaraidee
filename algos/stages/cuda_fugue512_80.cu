
#include <stdio.h>
#include "cuda/fugue512_device.cuh"
#include "cuda/selftest_gate.cuh"
#include "sph/sph_fugue.h"

#define TPB 256

/*
 * fugue512-80 x16r kernel implementation.
 *
 * The generic Fugue device pieces (mixtab access macros, TIX4/CMIX36/SMIX,
 * SUB_ROR*, FUGUE512_3/FUGUE512_F, FUGUE_ROL/ROR helpers) live in
 * cuda/fugue512_device.cuh (docs/coding-guideline.md section 3). What remains here is
 * x16-specific: the 80-byte block constant and the texture-backed mixtab load
 * (a fragment kept distinct from the header's __constant__ path).
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2018 tpruvot
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
 */

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, m) (x|y)
#define __CUDACC__
#endif

__constant__ static uint64_t c_PaddedMessage80[10];

/* Header words 0..18 are job constants: fugue512_80_midstate absorbs them once per setBlock
 * (same macros as the per-nonce kernel) and fugue512_gpu_hash_80 starts at word 19, the nonce. */
static __device__ uint32_t d_fugue80_mid[36];
__constant__ static uint32_t c_fugue80_mid[36];

/* FUGUE512_F(w, x, y, z) split after its first word: HEAD absorbs w, TAIL the other three */
#define FUGUE512_F_HEAD(w) { \
	TIX4(w, S00, S01, S04, S07, S08, S22, S24, S27, S30); \
	CMIX36(S33, S34, S35, S01, S02, S03, S15, S16, S17); \
	SMIX(S33, S34, S35, S00); \
	CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14); \
	SMIX(S30, S31, S32, S33); \
	CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11); \
	SMIX(S27, S28, S29, S30); \
	CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08); \
	SMIX(S24, S25, S26, S27); \
}
#define FUGUE512_F_TAIL(x, y, z) { \
	TIX4(x, S24, S25, S28, S31, S32, S10, S12, S15, S18); \
	CMIX36(S21, S22, S23, S25, S26, S27, S03, S04, S05); \
	SMIX(S21, S22, S23, S24); \
	CMIX36(S18, S19, S20, S22, S23, S24, S00, S01, S02); \
	SMIX(S18, S19, S20, S21); \
	CMIX36(S15, S16, S17, S19, S20, S21, S33, S34, S35); \
	SMIX(S15, S16, S17, S18); \
	CMIX36(S12, S13, S14, S16, S17, S18, S30, S31, S32); \
	SMIX(S12, S13, S14, S15); \
	\
	TIX4(y, S12, S13, S16, S19, S20, S34, S00, S03, S06); \
	CMIX36(S09, S10, S11, S13, S14, S15, S27, S28, S29); \
	SMIX(S09, S10, S11, S12); \
	CMIX36(S06, S07, S08, S10, S11, S12, S24, S25, S26); \
	SMIX(S06, S07, S08, S09); \
	CMIX36(S03, S04, S05, S07, S08, S09, S21, S22, S23); \
	SMIX(S03, S04, S05, S06); \
	CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20); \
	SMIX(S00, S01, S02, S03); \
	\
	FUGUE512_F_HEAD(z); \
}

/* one T0 in FUGUE512_R lane-group copies, rotations after the load, byte-permute SMIX
 * (the 64-byte stage's form, cuda/fugue512_device.cuh) */
#undef SMIX
#undef mixtab0
#undef mixtab1
#undef mixtab2
#undef mixtab3
#define FUGUE512_LANE (threadIdx.x & (FUGUE512_R - 1))
#define mixtab0(i) mixtabs[(i) * FUGUE512_R + FUGUE512_LANE]
#define mixtab1(i) FUGUE_ROR8(mixtabs[(i) * FUGUE512_R + FUGUE512_LANE])
#define mixtab2(i) FUGUE_ROL16(mixtabs[(i) * FUGUE512_R + FUGUE512_LANE])
#define mixtab3(i) FUGUE_ROL8(mixtabs[(i) * FUGUE512_R + FUGUE512_LANE])
#define SMIX SMIX_PRMT

__global__ __launch_bounds__(256)
void fugue512_80_midstate(void)
{
	__shared__ uint32_t mixtabs[256 * FUGUE512_R];
	fugue512_load_shared_r(mixtabs);
	if (threadIdx.x)
		return;

	uint32_t Data[20];
	#pragma unroll
	for (int i = 0; i < 10; i++)
		AS_UINT2(&Data[i * 2]) = AS_UINT2(&c_PaddedMessage80[i]);

	uint32_t S00, S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11;
	uint32_t S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23;
	uint32_t S24, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35;

	S00 = S01 = S02 = S03 = S04 = S05 = S06 = S07 = S08 = S09 = 0;
	S10 = S11 = S12 = S13 = S14 = S15 = S16 = S17 = S18 = S19 = 0;
	S20 = 0x8807a57e; S21 = 0xe616af75; S22 = 0xc5d3e4db; S23 = 0xac9ab027;
	S24 = 0xd915f117; S25 = 0xb6eecc54; S26 = 0x06e8020b; S27 = 0x4a92efd1;
	S28 = 0xaac6e2c9; S29 = 0xddb21398; S30 = 0xcae65838; S31 = 0x437f203f;
	S32 = 0x25ea78e7; S33 = 0x951fddd6; S34 = 0xda6ed11d; S35 = 0xe13e3567;

	FUGUE512_3((Data[ 0]), (Data[ 1]), (Data[ 2]));
	FUGUE512_3((Data[ 3]), (Data[ 4]), (Data[ 5]));
	FUGUE512_3((Data[ 6]), (Data[ 7]), (Data[ 8]));
	FUGUE512_3((Data[ 9]), (Data[10]), (Data[11]));
	FUGUE512_3((Data[12]), (Data[13]), (Data[14]));
	FUGUE512_3((Data[15]), (Data[16]), (Data[17]));
	FUGUE512_F_HEAD((Data[18]));

	uint32_t *m = d_fugue80_mid;
	m[ 0] = S00; m[ 1] = S01; m[ 2] = S02; m[ 3] = S03; m[ 4] = S04; m[ 5] = S05;
	m[ 6] = S06; m[ 7] = S07; m[ 8] = S08; m[ 9] = S09; m[10] = S10; m[11] = S11;
	m[12] = S12; m[13] = S13; m[14] = S14; m[15] = S15; m[16] = S16; m[17] = S17;
	m[18] = S18; m[19] = S19; m[20] = S20; m[21] = S21; m[22] = S22; m[23] = S23;
	m[24] = S24; m[25] = S25; m[26] = S26; m[27] = S27; m[28] = S28; m[29] = S29;
	m[30] = S30; m[31] = S31; m[32] = S32; m[33] = S33; m[34] = S34; m[35] = S35;
}

__host__
void fugue512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
	fugue512_80_midstate <<<1, 256>>> ();
	void *mid = NULL;
	cudaGetSymbolAddress(&mid, d_fugue80_mid);
	cudaMemcpyToSymbol(c_fugue80_mid, mid, sizeof(c_fugue80_mid), 0, cudaMemcpyDeviceToDevice);
}

/***************************************************/

__global__
__launch_bounds__(TPB)
void fugue512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	__shared__ uint32_t mixtabs[256 * FUGUE512_R];
	fugue512_load_shared_r(mixtabs);

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t Data[16];

		uint32_t S00, S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11;
		uint32_t S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23;
		uint32_t S24, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35;
		//uint32_t B24, B25, B26,
		uint32_t B27, B28, B29, B30, B31, B32, B33, B34, B35;
		//const uint64_t bc = 640 bits to hash
		//const uint32_t bclo = (uint32_t)(bc);
		//const uint32_t bchi = (uint32_t)(bc >> 32);

		/* the job-constant state after words 0..18 (fugue512_80_midstate) */
		S00 = c_fugue80_mid[ 0]; S01 = c_fugue80_mid[ 1]; S02 = c_fugue80_mid[ 2]; S03 = c_fugue80_mid[ 3];
		S04 = c_fugue80_mid[ 4]; S05 = c_fugue80_mid[ 5]; S06 = c_fugue80_mid[ 6]; S07 = c_fugue80_mid[ 7];
		S08 = c_fugue80_mid[ 8]; S09 = c_fugue80_mid[ 9]; S10 = c_fugue80_mid[10]; S11 = c_fugue80_mid[11];
		S12 = c_fugue80_mid[12]; S13 = c_fugue80_mid[13]; S14 = c_fugue80_mid[14]; S15 = c_fugue80_mid[15];
		S16 = c_fugue80_mid[16]; S17 = c_fugue80_mid[17]; S18 = c_fugue80_mid[18]; S19 = c_fugue80_mid[19];
		S20 = c_fugue80_mid[20]; S21 = c_fugue80_mid[21]; S22 = c_fugue80_mid[22]; S23 = c_fugue80_mid[23];
		S24 = c_fugue80_mid[24]; S25 = c_fugue80_mid[25]; S26 = c_fugue80_mid[26]; S27 = c_fugue80_mid[27];
		S28 = c_fugue80_mid[28]; S29 = c_fugue80_mid[29]; S30 = c_fugue80_mid[30]; S31 = c_fugue80_mid[31];
		S32 = c_fugue80_mid[32]; S33 = c_fugue80_mid[33]; S34 = c_fugue80_mid[34]; S35 = c_fugue80_mid[35];

		FUGUE512_F_TAIL((startNonce + thread), 0/*bchi*/, (80*8)/*bclo*/);

		// rotate right state by 3 dwords (S00 = S33, S03 = S00)
		SUB_ROR3;
		SUB_ROR9;

		#pragma unroll 32
		for (int i = 0; i < 32; i++) {
			SUB_ROR3;
			CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20);
			SMIX(S00, S01, S02, S03);
		}
		#pragma unroll 13
		for (int i = 0; i < 13; i++) {
			S04 ^= S00;
			S09 ^= S00;
			S18 ^= S00;
			S27 ^= S00;
			SUB_ROR9;
			SMIX(S00, S01, S02, S03);
			S04 ^= S00;
			S10 ^= S00;
			S18 ^= S00;
			S27 ^= S00;
			SUB_ROR9;
			SMIX(S00, S01, S02, S03);
			S04 ^= S00;
			S10 ^= S00;
			S19 ^= S00;
			S27 ^= S00;
			SUB_ROR9;
			SMIX(S00, S01, S02, S03);
			S04 ^= S00;
			S10 ^= S00;
			S19 ^= S00;
			S28 ^= S00;
			SUB_ROR8;
			SMIX(S00, S01, S02, S03);
		}
		S04 ^= S00;
		S09 ^= S00;
		S18 ^= S00;
		S27 ^= S00;

		Data[ 0] = cuda_swab32(S01);
		Data[ 1] = cuda_swab32(S02);
		Data[ 2] = cuda_swab32(S03);
		Data[ 3] = cuda_swab32(S04);
		Data[ 4] = cuda_swab32(S09);
		Data[ 5] = cuda_swab32(S10);
		Data[ 6] = cuda_swab32(S11);
		Data[ 7] = cuda_swab32(S12);
		Data[ 8] = cuda_swab32(S18);
		Data[ 9] = cuda_swab32(S19);
		Data[10] = cuda_swab32(S20);
		Data[11] = cuda_swab32(S21);
		Data[12] = cuda_swab32(S27);
		Data[13] = cuda_swab32(S28);
		Data[14] = cuda_swab32(S29);
		Data[15] = cuda_swab32(S30);

		const size_t hashPosition = thread;
		uint64_t* pHash = &g_hash[hashPosition << 3];
		#pragma unroll 4
		for(int i = 0; i < 4; i++)
			AS_UINT4(&pHash[i * 2]) = AS_UINT4(&Data[i * 4]);
	}
}

#undef SMIX
#define SMIX SMIX_TAB
#undef FUGUE512_LANE
#undef mixtab0
#undef mixtab1
#undef mixtab2
#undef mixtab3
#define mixtab0(x) mixtabs[(x)]
#define mixtab1(x) mixtabs[(x)+256]
#define mixtab2(x) mixtabs[(x)+512]
#define mixtab3(x) mixtabs[(x)+768]

/* Init-time self-test of the 80-byte stage: the real setBlock + launcher against sph_fugue512 on
 * 80-byte headers (word 19 = the nonce), a ragged count over two blocks, plus a negative leg
 * (one header bit flipped on the device side only). Fail-closed via cuda/selftest_gate.cuh. */
#define FUGUE80_ST_N     299
#define FUGUE80_ST_START 0x7fffff80u

void fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

static bool fugue512_80_selftest_run(const uint32_t *hdr, uint8_t *out)
{
	uint32_t *d = NULL;
	if (cudaMalloc(&d, (size_t) FUGUE80_ST_N * 64) != cudaSuccess)
		return selftest_cuda_fault();
	fugue512_setBlock_80((void *) hdr);
	fugue512_cuda_hash_80(0, FUGUE80_ST_N, FUGUE80_ST_START, d);
	const bool ok = cudaMemcpy(out, d, (size_t) FUGUE80_ST_N * 64, cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d);
	return ok ? true : selftest_cuda_fault();
}

static void fugue512_80_ref(const uint32_t *hdr, uint32_t nonce, uint8_t *dig)
{
	uint8_t buf[80];
	for (int i = 0; i < 20; i++) {
		const uint32_t w = (i == 19) ? nonce : hdr[i]; /* the kernel absorbs the words big-endian */
		buf[4 * i] = (uint8_t)(w >> 24); buf[4 * i + 1] = (uint8_t)(w >> 16);
		buf[4 * i + 2] = (uint8_t)(w >> 8); buf[4 * i + 3] = (uint8_t) w;
	}
	sph_fugue512_context ctx;
	sph_fugue512_init(&ctx);
	sph_fugue512(&ctx, buf, 80);
	sph_fugue512_close(&ctx, dig);
}

__host__
bool fugue512_80_device_selftest(int thr_id)
{
	uint32_t hdr[20];
	for (int i = 0; i < 20; i++)
		hdr[i] = 0x04030201u * (uint32_t)(i + 1);

	uint8_t *gpu = (uint8_t *) malloc((size_t) FUGUE80_ST_N * 64);
	if (!gpu) return selftest_gate(thr_id, "fugue512-80", selftest_cuda_fault());

	bool kat_ok = fugue512_80_selftest_run(hdr, gpu);
	for (int t = 0; t < FUGUE80_ST_N && kat_ok; t++) {
		uint8_t ref[64];
		fugue512_80_ref(hdr, FUGUE80_ST_START + t, ref);
		kat_ok = memcmp(gpu + (size_t) t * 64, ref, 64) == 0;
	}

	uint32_t neg[20];
	memcpy(neg, hdr, sizeof(neg));
	neg[0] ^= 1u;
	uint8_t ref0[64];
	fugue512_80_ref(hdr, FUGUE80_ST_START, ref0);
	const bool neg_ok = fugue512_80_selftest_run(neg, gpu) && memcmp(gpu, ref0, 64) != 0;
	free(gpu);

	const bool passed = kat_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "fugue512-80 stage self-test FAILED (kat %d neg %d)", (int) kat_ok, (int) neg_ok);
	return selftest_gate(thr_id, "fugue512-80", passed);
}

__host__
void x16_fugue512_cpu_init(int thr_id, uint32_t threads)
{
	/* Refuses to start on a device that gets the stage wrong. Leaves its own header in the
	 * device symbols; every caller uploads the real one with setBlock before hashing. */
	fugue512_80_device_selftest(thr_id);
}

__host__
void x16_fugue512_cpu_free(int thr_id)
{
}

__host__
void fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	fugue512_gpu_hash_80 <<<grid, block>>> (threads, startNonce, (uint64_t*)d_hash);
}

/* Legacy forwarders - ghostrider and x21s still call these names; remove once
 * they call the bare fugue512_* launchers directly. cpu_init/cpu_free stay
 * x16_-named: the bare fugue512_cpu_init/free are the 64-byte x13 fugue (bridge
 * in cuda_x_stages.h) and this 80-byte texture lifecycle is distinct (pending the
 * "merge with x13_fugue512" TODO). */
__host__
void x16_fugue512_setBlock_80(void *pdata)
{
	fugue512_setBlock_80(pdata);
}

__host__
void x16_fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	fugue512_cuda_hash_80(thr_id, threads, startNonce, d_hash);
}
