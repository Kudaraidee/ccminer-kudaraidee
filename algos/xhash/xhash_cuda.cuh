/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CUDA-side types and helpers for the XHash kernels. Adapted from
 * ParallaxProtocol/hashwarp libxhash-cuda/cuda_helper.h and
 * xhash_cuda_miner_kernel_globals.h (GPL-3.0).
 */
#pragma once

#include "../../cuda/helper.cuh"
#include "xhash_cuda.h"

#define DEV_INLINE __device__ __forceinline__

typedef struct
{
	uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;

typedef union
{
	uint32_t words[128 / sizeof(uint32_t)];
	uint2 uint2s[128 / sizeof(uint2)];
	uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

typedef union
{
	uint32_t words[64 / sizeof(uint32_t)];
	uint2 uint2s[64 / sizeof(uint2)];
	uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

/* Device constants, set from the host before each DAG build/job. */
__constant__ uint32_t d_dag_size;	/* number of 128-byte DAG items */
__constant__ hash128_t *d_dag;
__constant__ uint32_t d_light_size;	/* number of 64-byte cache items */
__constant__ hash64_t *d_light;
__constant__ hash32_t d_header;
__constant__ uint64_t d_target;

#if __CUDACC_VER_MAJOR__ > 8
#define XHASH_SHFL(x, y, z) __shfl_sync(0xFFFFFFFFu, (x), (y), (z))
#else
#define XHASH_SHFL(x, y, z) __shfl((x), (y), (z))
#endif

#define LDG(x) __ldg(&(x))

/* uint2 <-> uint4 helpers from hashwarp's cuda_helper.h, not present in
 * ccminer's cuda/helper.cuh. */
DEV_INLINE void devectorize2(uint4 in, uint2 &x, uint2 &y)
{
	x = make_uint2(in.x, in.y);
	y = make_uint2(in.z, in.w);
}

DEV_INLINE uint4 vectorize2(uint2 x, uint2 y)
{
	return make_uint4(x.x, x.y, y.x, y.y);
}

#define ACCESSES 64
#define THREADS_PER_HASH (128 / 16)
