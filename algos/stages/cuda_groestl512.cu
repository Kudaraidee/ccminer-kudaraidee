// Groestl-512 x-family stage: 64-byte hash and 80-byte header, two hashes per thread,
// bitsliced in registers (cuda/groestl512_x2_device.cuh)

#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

#include "cuda/helper.cuh"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#endif

#include "cuda/groestl512_x2_device.cuh"

/* ~250 registers per thread and 64 words of shared stash per thread */
#define TPB 128
#define MINB 2

#define WANT_GROESTL80
#ifdef WANT_GROESTL80
__constant__ static uint32_t c_Message80[20];
#endif

__global__ __launch_bounds__(TPB, MINB)
void groestl512_gpu_hash_64_x2(const uint32_t threads, const uint32_t startNounce, uint32_t * g_hash, uint32_t * __restrict g_nonceVector)
{
	extern __shared__ uint32_t stash[];                     /* [64][TPB] */
	const uint32_t i0 = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
	if (i0 >= threads) return;
	const bool two = i0 + 1 < threads;

	uint4 *pHash[2];
	uint32_t in[2][16];
	#pragma unroll
	for (int h = 0; h < 2; h++) {
		const uint32_t i = two ? i0 + h : i0;
		const uint32_t nounce = g_nonceVector ? g_nonceVector[i] : (startNounce + i);
		const off_t hashPosition = nounce - startNounce;
		pHash[h] = (uint4*) &g_hash[hashPosition << 4];
		#pragma unroll
		for (int k = 0; k < 4; k++) {
			const uint4 v = pHash[h][k];
			in[h][4*k] = v.x; in[h][4*k+1] = v.y; in[h][4*k+2] = v.z; in[h][4*k+3] = v.w;
		}
	}

	groestl512_x2_hash_64(in, &stash[threadIdx.x], blockDim.x);

	/* a lone last hash was loaded twice; only its first copy is written */
	#pragma unroll
	for (int h = 0; h < 2; h++) {
		if (h == 0 || two) {
			#pragma unroll
			for (int k = 0; k < 4; k++)
				pHash[h][k] = make_uint4(in[h][4*k], in[h][4*k+1], in[h][4*k+2], in[h][4*k+3]);
		}
	}
}

/* sm < 75: two hashes per thread pair. Lanes past the end compute on hash 0 so every lane
 * reaches the shuffles; they only skip the store. */
#define TPB_PAIR 256

__global__ __launch_bounds__(TPB_PAIR, 2)
void groestl512_gpu_hash_64_pair(const uint32_t threads, const uint32_t startNounce, uint32_t * g_hash, uint32_t * __restrict g_nonceVector)
{
	extern __shared__ uint32_t stash[];                     /* [32][TPB_PAIR] */
	const uint32_t lane = threadIdx.x & 1;
	const uint32_t i0 = ((blockDim.x * blockIdx.x + threadIdx.x) >> 1) * 2;
	const bool active = i0 < threads;
	const bool two = i0 + 1 < threads;

	uint32_t *pHash[2];
	uint32_t inq[2][8];
	#pragma unroll
	for (int h = 0; h < 2; h++) {
		const uint32_t i = !active ? 0 : (two ? i0 + h : i0);
		const uint32_t nounce = g_nonceVector ? g_nonceVector[i] : (startNounce + i);
		pHash[h] = &g_hash[(size_t)(nounce - startNounce) << 4];
		#pragma unroll
		for (int n = 0; n < 8; n++) inq[h][n] = pHash[h][2 * n + lane];
	}

	groestl512_pair_hash_64(inq, &stash[threadIdx.x], blockDim.x);

	if (!active) return;
	#pragma unroll
	for (int h = 0; h < 2; h++)
		if (h == 0 || two) {
			#pragma unroll
			for (int n = 0; n < 8; n++) pHash[h][2 * n + lane] = inq[h][n];
		}
}

extern short device_map[];
extern long  device_sm[];

/* Unit self-test for cuda/groestl512_x2_device.cuh (docs/coding-guideline.md section 7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool groestl512_device_selftest(int thr_id);

__host__
void groestl512_cpu_init(int thr_id, uint32_t threads)
{
	groestl512_device_selftest(thr_id);
}

__host__
void groestl512_cpu_free(int thr_id)
{
}

__host__
void groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	if (device_sm[device_map[thr_id]] < 750) {
		const uint32_t lanes = (threads + 1) / 2 * 2;      /* two lanes per two hashes */
		groestl512_gpu_hash_64_pair<<<(lanes + TPB_PAIR - 1) / TPB_PAIR, TPB_PAIR, 32 * TPB_PAIR * sizeof(uint32_t)>>>(
			threads, startNounce, d_hash, d_nonceVector);
		return;
	}

	const uint32_t nthr = (threads + 1) / 2;   /* two hashes per thread */

	dim3 grid((nthr + TPB - 1) / TPB);
	dim3 block(TPB);

	groestl512_gpu_hash_64_x2<<<grid, block, 64 * TPB * sizeof(uint32_t)>>>(threads, startNounce, d_hash, d_nonceVector);
}

// --------------------------------------------------------------------------------------------------------------------------------------------

#ifdef WANT_GROESTL80

__host__
void groestl512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	cudaMemcpyToSymbol(c_Message80, endiandata, sizeof(c_Message80), 0, cudaMemcpyHostToDevice);
}

__global__ __launch_bounds__(TPB, MINB)
void groestl512_gpu_hash_80_x2(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
	extern __shared__ uint32_t stash[];                     /* [64][TPB] */
	const uint32_t i0 = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
	if (i0 >= threads) return;
	const bool two = i0 + 1 < threads;

	uint32_t m[2][20];
	#pragma unroll
	for (int w = 0; w < 19; w++) m[0][w] = m[1][w] = c_Message80[w];
	m[0][19] = cuda_swab32(startNounce + i0);
	m[1][19] = cuda_swab32(startNounce + i0 + 1);

	uint32_t s[8][8], out[2][16];
	groestl512_x2_load80(m, s);
	groestl512_x2_compress(s, &stash[threadIdx.x], blockDim.x);
	groestl512_x2_store(s, out);

	#pragma unroll
	for (int h = 0; h < 2; h++) {
		if (h == 0 || two) {
			uint4 *outpt = (uint4*) &g_outhash[(size_t)(i0 + h) << 4];
			#pragma unroll
			for (int k = 0; k < 4; k++)
				outpt[k] = make_uint4(out[h][4*k], out[h][4*k+1], out[h][4*k+2], out[h][4*k+3]);
		}
	}
}

__host__
void groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t nthr = (threads + 1) / 2;   /* two hashes per thread */

	dim3 grid((nthr + TPB - 1) / TPB);
	dim3 block(TPB);

	groestl512_gpu_hash_80_x2<<<grid, block, 64 * TPB * sizeof(uint32_t)>>>(threads, startNounce, d_hash);
}

#endif
