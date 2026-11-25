/**
 * JH512 64 and 80 kernels
 *
 * JH80 by tpruvot - 2017 - under GPLv3
 **/
#include "cuda/helper.cuh"

// #include <stdio.h>  // printf
// #include <unistd.h> // sleep

#include "cuda/jh512_device.cuh"

// The extracted JH-512 device library (constants, bitsliced E8 and the
// 64-byte compression) lives in cuda/jh512_device.cuh; the kernels below
// are thin wrappers / the 80-byte first-stage paths.

__global__
//__launch_bounds__(256,2)
void jh512_gpu_hash_64(const uint32_t threads, const uint32_t startNounce, uint32_t* g_hash, uint32_t * g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);
		const uint32_t hashPosition = nounce - startNounce;
		uint32_t *Hash = &g_hash[(size_t)16 * hashPosition];

		uint32_t h[16];
		AS_UINT4(&h[ 0]) = AS_UINT4(&Hash[ 0]);
		AS_UINT4(&h[ 4]) = AS_UINT4(&Hash[ 4]);
		AS_UINT4(&h[ 8]) = AS_UINT4(&Hash[ 8]);
		AS_UINT4(&h[12]) = AS_UINT4(&Hash[12]);

		jh512_hash_64(h);

		AS_UINT4(&Hash[ 0]) = AS_UINT4(&h[ 0]);
		AS_UINT4(&Hash[ 4]) = AS_UINT4(&h[ 4]);
		AS_UINT4(&Hash[ 8]) = AS_UINT4(&h[ 8]);
		AS_UINT4(&Hash[12]) = AS_UINT4(&h[12]);
	}
}
__host__
void jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	jh512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
}

// Setup function
/* Unit self-test for cuda/jh512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool jh512_device_selftest(int thr_id);

__host__ void  jh512_cpu_init(int thr_id, uint32_t threads)
{
	jh512_device_selftest(thr_id);
}

#define WANT_JH80_MIDSTATE
#ifdef WANT_JH80

__constant__
static uint32_t c_PaddedMessage80[20]; // padded message (80 bytes)

__host__
void jh512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, endiandata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__
void jh512_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t h[20];
		AS_UINT4(&h[ 0]) = AS_UINT4(&c_PaddedMessage80[ 0]);
		AS_UINT4(&h[ 4]) = AS_UINT4(&c_PaddedMessage80[ 4]);
		AS_UINT4(&h[ 8]) = AS_UINT4(&c_PaddedMessage80[ 8]);
		AS_UINT4(&h[12]) = AS_UINT4(&c_PaddedMessage80[12]);
		AS_UINT2(&h[16]) = AS_UINT2(&c_PaddedMessage80[16]);
		h[18] = c_PaddedMessage80[18];
		h[19] = cuda_swab32(startNounce + thread);

		uint32_t x[8][4] = { /* init */
			{ 0x964bd16f, 0x17aa003e, 0x052e6a63, 0x43d5157a },
			{ 0x8d5e228a, 0x0bef970c, 0x591234e9, 0x61c3b3f2 },
			{ 0xc1a01d89, 0x1e806f53, 0x6b05a92a, 0x806d2bea },
			{ 0xdbcc8e58, 0xa6ba7520, 0x763a0fa9, 0xf73bf8ba },
			{ 0x05e66901, 0x694ae341, 0x8e8ab546, 0x5ae66f2e },
			{ 0xd0a74710, 0x243c84c1, 0xb1716e3b, 0x99c15a2d },
			{ 0xecf657cf, 0x56f8b19d, 0x7c8806a7, 0x56b11657 },
			{ 0xdffcc2e3, 0xfb1785e6, 0x78465a54, 0x4bdd8ccc }
		};

		// 1 (could be precomputed)
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[i/4][i & 3] ^= h[i];
		jh512_E8(x);
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[(i+16)/4][(i+16) & 3] ^= h[i];

		// 2 (16 bytes with nonce)
		#pragma unroll
		for (int i = 0; i < 4; i++)
			x[0][i] ^= h[16+i];
		x[1][0] ^= 0x80U;
		jh512_E8(x);
		#pragma unroll
		for (int i = 0; i < 4; i++)
			x[4][i] ^= h[16+i];
		x[5][0] ^= 0x80U;

		// 3 close
		x[3][3] ^= 0x80020000U; // 80 bytes = 640bits (0x280)
		jh512_E8(x);
		x[7][3] ^= 0x80020000U;

		uint32_t *Hash = &g_outhash[(size_t)16 * thread];
		AS_UINT4(&Hash[ 0]) = AS_UINT4(&x[4][0]);
		AS_UINT4(&Hash[ 4]) = AS_UINT4(&x[5][0]);
		AS_UINT4(&Hash[ 8]) = AS_UINT4(&x[6][0]);
		AS_UINT4(&Hash[12]) = AS_UINT4(&x[7][0]);
	}
}

__host__
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	jh512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, d_hash);
}

#endif

#ifdef WANT_JH80_MIDSTATE

__constant__ static uint32_t c_JHState[32];
__constant__ static uint32_t c_Message[4];

__global__
void jh512_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// 1 (precomputed state)
		uint32_t x[8][4];
		AS_UINT4(&x[0][0]) = AS_UINT4(&c_JHState[ 0]);
		AS_UINT4(&x[1][0]) = AS_UINT4(&c_JHState[ 4]);
		AS_UINT4(&x[2][0]) = AS_UINT4(&c_JHState[ 8]);
		AS_UINT4(&x[3][0]) = AS_UINT4(&c_JHState[12]);

		AS_UINT4(&x[4][0]) = AS_UINT4(&c_JHState[16]);
		AS_UINT4(&x[5][0]) = AS_UINT4(&c_JHState[20]);
		AS_UINT4(&x[6][0]) = AS_UINT4(&c_JHState[24]);
		AS_UINT4(&x[7][0]) = AS_UINT4(&c_JHState[28]);

		// 2 (16 bytes with nonce)
		uint32_t h[4];
		AS_UINT2(&h[0]) = AS_UINT2(&c_Message[0]);
		h[2] = c_Message[2];
		h[3] = cuda_swab32(startNounce + thread);

		#pragma unroll
		for (int i = 0; i < 4; i++)
			x[0][i] ^= h[i];
		x[1][0] ^= 0x80U;
		jh512_E8(x);
		#pragma unroll
		for (int i = 0; i < 4; i++)
			x[4][i] ^= h[i];
		x[5][0] ^= 0x80U;

		// 3 close
		x[3][3] ^= 0x80020000U; // 80 bytes = 640bits (0x280)
		jh512_E8(x);
		x[7][3] ^= 0x80020000U;

		uint32_t *Hash = &g_outhash[(size_t)16 * thread];
		AS_UINT4(&Hash[ 0]) = AS_UINT4(&x[4][0]);
		AS_UINT4(&Hash[ 4]) = AS_UINT4(&x[5][0]);
		AS_UINT4(&Hash[ 8]) = AS_UINT4(&x[6][0]);
		AS_UINT4(&Hash[12]) = AS_UINT4(&x[7][0]);
	}
}

__host__
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	jh512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, d_hash);
}

extern "C" {
#undef SPH_C32
#undef SPH_T32
#undef SPH_C64
#undef SPH_T64
#include <sph/sph_jh.h>
}

__host__
void jh512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	sph_jh512_context ctx_jh;

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, endiandata, 64);

	cudaMemcpyToSymbol(c_JHState, ctx_jh.H.narrow, 128, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_Message, &endiandata[16], sizeof(c_Message), 0, cudaMemcpyHostToDevice);
}

#endif

