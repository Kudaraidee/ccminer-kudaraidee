/*
 * sha-512 cuda kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014 djm34
 *               2016 tpruvot
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
#include <stdio.h>

#include "cuda/sha512_device.cuh"

// The unrolled SHA-512 device code and its round constants live in
// cuda/sha512_device.cuh (sha512_hash_64 + SHA512_* macros + c_sha512_K);
// the 64-byte kernel below is a thin wrapper, the 80-byte first-stage
// kernel expands the shared macros directly.

// (256,4) beats the file's register cap on Pascal and Ampere; Turing is untested.
__global__
#if __CUDA_ARCH__ >= 800 || __CUDA_ARCH__ == 610
__launch_bounds__(256, 4)
#endif
void sha512_gpu_hash_64(const uint32_t threads, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint64_t hashPosition = thread;
		uint64_t *pHash = &g_hash[hashPosition*8U];

		sha512_hash_64(pHash);
	}
}

/* Unit self-test for the sha512_hash_64 block in cuda/sha512_device.cuh
 * (docs/coding-guideline.md §7 layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool sha512x_device_selftest(int thr_id);

__host__
void sha512_cpu_init(int thr_id, uint32_t threads)
{
	sha512x_device_selftest(thr_id);
}

__host__
void sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	sha512_gpu_hash_64 <<<grid, block>>> (threads, (uint64_t*)d_hash);
}

__constant__
static uint64_t c_PaddedMessage80[10];

__global__
/*__launch_bounds__(256, 4)*/
void x16_sha512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t W[80];
		#pragma unroll
		for (int i = 0; i < 9; i ++) {
			W[i] = SHA512_SWAB64(c_PaddedMessage80[i]);
		}
		const uint32_t nonce = startNonce + thread;
		//((uint32_t*)W)[19] = cuda_swab32(nonce);
		W[9] = REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(nonce));
		W[9] = cuda_swab64(W[9]);
		W[10] = 0x8000000000000000;

		#pragma unroll
		for (int i = 11; i<15; i++) {
			W[i] = 0U;
		}
		W[15] = 0x0000000000000280;

		#pragma unroll 64
		for (int i = 16; i < 80; i ++) {
			W[i] = SHA512_SSG5_1(W[i-2]) + W[i-7];
			W[i] += SHA512_SSG5_0(W[i-15]) + W[i-16];
		}

		const uint64_t IV512[8] = {
			0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
			0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
			0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
			0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
		};

		uint64_t r[8];
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			r[i] = IV512[i];
		}

		#pragma unroll
		for (int i = 0; i < 80; i++) {
			SHA512_STEP(c_sha512_K, r, W, i&7, i);
		}

		const uint64_t hashPosition = thread;
		uint64_t *pHash = &g_hash[hashPosition << 3];
		#pragma unroll
		for (int u = 0; u < 8; u ++) {
			pHash[u] = SHA512_SWAB64(r[u] + IV512[u]);
		}
	}
}

__host__
void x16_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x16_sha512_gpu_hash_80 <<<grid, block >>> (threads, startNounce, (uint64_t*)d_hash);
}

__host__
void x16_sha512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}
