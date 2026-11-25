/*
 * Shabal-512 for X14/X15
 */
#include "cuda/helper.cuh"

/* $Id: shabal.c 175 2010-05-07 16:03:20Z tp $ */
/*
 * Shabal implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2007-2010 Projet RNRT SAPHIR
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
 * @author Thomas Pornin <thomas.pornin@cryptolog.com>
 */

/*
 * Part of this code was automatically generated (the part between
 * the "BEGIN" and "END" markers).
 */
#include "cuda/shabal512_device.cuh"
#include "cuda/candidate_report.cuh"

// The Shabal-512 device implementation (permutation macros, constants and
// shabal512_hash_64) lives in cuda/shabal512_device.cuh; the kernel below
// is a thin wrapper.

/***************************************************/
// GPU Hash Function
__global__ void shabal512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);
		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition << 3]; // [8 * hashPosition];

		// hash a register copy: 128-bit record access instead of 16 x 32-bit each way
		{ uint4 *p_ = (uint4*)(Hash); uint4 l_[4];
		#pragma unroll
		for (int i_ = 0; i_ < 4; i_++) l_[i_] = __ldg(&p_[i_]);
		shabal512_hash_64((uint32_t*)l_);
		#pragma unroll
		for (int i_ = 0; i_ < 4; i_++) p_[i_] = l_[i_]; }
	}
}

/***************************************************/
// Terminal variant: the full shabal (bit-identical to the CPU reference), the high 64
// bits compared against the target on-device, and the two lowest candidates
// reported (cuda/candidate_report.cuh) instead of storing d_hash.
__global__ void shabal512_gpu_hash_64_final(uint32_t threads, uint64_t *g_hash, uint32_t *resNonce, const uint64_t target)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint32_t *pHash = (uint32_t*)&g_hash[thread << 3];
		uint32_t Hash[16];

		#pragma unroll 16
		for (int i = 0; i < 16; i++)
			Hash[i] = pHash[i];

		shabal512_hash_64(Hash);

		if (*(uint64_t*)&Hash[6] <= target) {
			report_candidate_2(resNonce, thread);
		}
	}
}

/* Unit self-test for cuda/shabal512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool shabal512_device_selftest(int thr_id);

__host__ void shabal512_cpu_init(int thr_id, uint32_t threads)
{
	shabal512_device_selftest(thr_id);
}

extern short device_map[];
extern long  device_sm[];

/* Block size per arch: 256 on Pascal, 128 on newer cards, where the smaller block
 * packs more resident warps to hide the serial permutation's latency. */
static inline uint32_t shabal512_tpb(int thr_id)
{
	return (device_sm[device_map[thr_id]] < 700) ? 256 : 128;
}

__host__ void shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = shabal512_tpb(thr_id);

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	shabal512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
}

__host__ void shabal512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target)
{
	const uint32_t threadsperblock = shabal512_tpb(thr_id);

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	shabal512_gpu_hash_64_final<<<grid, block>>>(threads, (uint64_t*)d_hash, d_resNonce, target);
}
