/*
 * tiger-192 djm34
 *
 */

/*
 * tiger-192 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  djm34
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
 *
 * @author   phm <phm@inbox.com>
 */
/*
 * tiger192 kernel implementation — thin wrapper.
 *
 * The Tiger-192 device implementation (tables, TIGER_* round macros,
 * tiger192_load_shared, tiger192_hash_64) lives in cuda/tiger192_device.cuh
 * (docs/coding-guideline.md §3). The 80-byte kernel below builds its own
 * message blocks with the header's exported TIGER_ROUND_BODY.
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda/helper.cuh"

#include "cuda/tiger192_device.cuh"

__global__ void __launch_bounds__(256,5) tiger192_gpu_hash_64(int threads, int zero_pad_64, uint32_t *d_hash)
{
	__shared__ uint64_t sharedMem[768];

	tiger192_load_shared(sharedMem);
	__syncthreads();

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		uint64_t* inout = (uint64_t*)&d_hash[thread<<4];
		uint64_t buf[3], in[8];

		#pragma unroll
		for (int i = 0; i < 8; i++) in[i] = inout[i];

		tiger192_hash_64(sharedMem, in, buf);

		#pragma unroll
		for (int i = 0; i < 3; i++) inout[i] = buf[i];
		if (zero_pad_64)
        {
            #pragma unroll
            for (int i = 3; i < 8; i++) inout[i] = 0;
        }
	}
}

__constant__ uint64_t c_PaddedMessage80[10];

__global__ void __launch_bounds__(256,5) tiger192_gpu_hash_80(int threads, uint32_t startNonce, uint32_t *d_hash)
{
	__shared__ uint64_t sharedMem[768];
//	if(threadIdx.x < 256)
	{
		sharedMem[threadIdx.x]      = c_tiger_T1[threadIdx.x];
		sharedMem[threadIdx.x+256]  = c_tiger_T2[threadIdx.x];
		sharedMem[threadIdx.x+512]  = c_tiger_T3[threadIdx.x];
		//sharedMem[threadIdx.x+768]  = T4[threadIdx.x];
	}
	__syncthreads();

  int thread = (blockDim.x * blockIdx.x + threadIdx.x);
  if (thread < threads) {
		uint64_t* out = (uint64_t*)&d_hash[thread<<4];
		uint64_t buf[3], in[8], in2[8];

        const uint32_t nonce = cuda_swab32(startNonce + thread);

		#pragma unroll
		for (int i = 0; i < 8; i++) in[i] = c_PaddedMessage80[i];

		#pragma unroll
		for (int i = 0; i < 3; i++) buf[i] = c_tiger_III[i];

        TIGER_ROUND_BODY(in, buf);

		in2[0] = c_PaddedMessage80[8];
		in2[1] = (((uint64_t) nonce) << 32) | (c_PaddedMessage80[9] & 0xffffffff);
        in2[2] = 1;
        #pragma unroll
        for (int i = 3; i < 7; i++) in2[i] = 0;
		in2[7] = 0x280;

		TIGER_ROUND_BODY(in2, buf);

		#pragma unroll
		for (int i = 0; i < 3; i++) out[i] = buf[i];
		#pragma unroll
		for (int i = 3; i < 8; i++) out[i] = 0;
  }
}

/* Unit self-test for cuda/tiger192_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. No cpu_init exists for
 * tiger192, so the launcher runs it once (static guard inside). */
extern bool tiger192_device_selftest(int thr_id);

__host__ void tiger192_cpu_hash_64(int thr_id, int threads, int zero_pad_64, uint32_t *d_hash)
{
	tiger192_device_selftest(thr_id);

	const int threadsperblock = 256;
	// Ceiling division: a truncating `threads/threadsperblock` drops the final
	// partial block (up to 255 tail nonces silently never hashed) — the kernel's
	// `if (thread < threads)` guard exists precisely to make this safe. Matches
	// every sibling stage (jh512/luffa512/…).
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	tiger192_gpu_hash_64<<<grid, block>>>(threads, zero_pad_64, d_hash);
}

__host__
void tiger192_setBlock_80(void *pdata)
{
    cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__host__ void tiger192_cpu_hash_80(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash)
{
	const int threadsperblock = 256;
	// Ceiling division (see tiger192_cpu_hash_64) — truncating grid drops tail
	// nonces; the `if (thread < threads)` guard makes the round-up safe.
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	tiger192_gpu_hash_80<<<grid, block>>>(threads, startNonce, d_hash);
}
