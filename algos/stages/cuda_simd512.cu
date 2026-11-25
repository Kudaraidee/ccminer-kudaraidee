/*
 * SIMD-512 64-byte stage (shared x-family stage, see docs/coding-guideline.md).
 *
 * One kernel, eight lanes per hash: the lanes run the NTT message expansion
 * (cuda_simd512_func.cuh), keep the 1 KB expanded message in shared memory and
 * then compress together, lane j holding state words j, 8+j, 16+j and 24+j; a
 * step is lane-local except one xor-shuffle. Still a stage boundary for the
 * x-family fused kernel.
 *
 * Based on the 2 Christians, klaus_t, Tanguy Pruvot, tsiv and SP (2013-2016);
 * Provos Alexis 2016; optimised by sp 2018.
 */

#include "miner.h"
#include "cuda/helper_alexis.cuh"
#include "cuda/vectors_alexis.cuh"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#include "cuda_simd512_func.cuh"

#define SC_STEP(w, p, r1, r2, ga, gb, gc, gd, F) { \
	const uint32_t R = ROTL32(ga, r1); \
	gd += (w) + F(ga, gb, gc); \
	gd = __shfl_xor_sync(gmask, R, p, 8) + ROTL32(gd, r2); \
	ga = R; }

#define SIMDF_TPB 128
#define SIMDF_ROW 264 /* +8 words: the four hashes of a warp start on different banks */

__global__ __launch_bounds__(SIMDF_TPB, 4)
static void simd512_gpu_fused_64(uint32_t threads, uint32_t *g_hash)
{
	__shared__ uint32_t s_w[SIMDF_TPB / 8][SIMDF_ROW];

	const uint32_t hash = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	const uint32_t lane = threadIdx.x & 7;
	uint32_t *const w = s_w[threadIdx.x >> 3];
	const uint32_t gmask = 0xFFu << (threadIdx.x & 24);

	if (hash < threads)
	{
		uint32_t *Hash = &g_hash[hash << 4];
		const uint32_t data0 = __ldg(&Hash[lane]);
		const uint32_t data1 = __ldg(&Hash[lane + 8]);

		uint4 v[8];
		simd512_expand_lane(data0, data1, v);
		#pragma unroll
		for (int k = 0; k < 8; k++)
			*(uint4*)&w[(k * 8 + lane) * 4] = v[k];
		__syncwarp(gmask);

		const uint32_t iv0 = c_IV_512[lane], iv1 = c_IV_512[8 + lane], iv2 = c_IV_512[16 + lane], iv3 = c_IV_512[24 + lane];
		uint32_t g0 = data0 ^ iv0, g1 = data1 ^ iv1, g2 = iv2, g3 = iv3;

		SC_STEP(w[0 + lane], 1, 3, 23, g0, g1, g2, g3, IF);
		SC_STEP(w[8 + lane], 6, 23, 17, g3, g0, g1, g2, IF);
		SC_STEP(w[16 + lane], 2, 17, 27, g2, g3, g0, g1, IF);
		SC_STEP(w[24 + lane], 3, 27, 3, g1, g2, g3, g0, IF);
		SC_STEP(w[32 + lane], 5, 3, 23, g0, g1, g2, g3, MAJ);
		SC_STEP(w[40 + lane], 7, 23, 17, g3, g0, g1, g2, MAJ);
		SC_STEP(w[48 + lane], 4, 17, 27, g2, g3, g0, g1, MAJ);
		SC_STEP(w[56 + lane], 1, 27, 3, g1, g2, g3, g0, MAJ);
		SC_STEP(w[64 + lane], 6, 28, 19, g0, g1, g2, g3, IF);
		SC_STEP(w[72 + lane], 2, 19, 22, g3, g0, g1, g2, IF);
		SC_STEP(w[80 + lane], 3, 22, 7, g2, g3, g0, g1, IF);
		SC_STEP(w[88 + lane], 5, 7, 28, g1, g2, g3, g0, IF);
		SC_STEP(w[96 + lane], 7, 28, 19, g0, g1, g2, g3, MAJ);
		SC_STEP(w[104 + lane], 4, 19, 22, g3, g0, g1, g2, MAJ);
		SC_STEP(w[112 + lane], 1, 22, 7, g2, g3, g0, g1, MAJ);
		SC_STEP(w[120 + lane], 6, 7, 28, g1, g2, g3, g0, MAJ);
		SC_STEP(w[128 + lane], 2, 29, 9, g0, g1, g2, g3, IF);
		SC_STEP(w[136 + lane], 3, 9, 15, g3, g0, g1, g2, IF);
		SC_STEP(w[144 + lane], 5, 15, 5, g2, g3, g0, g1, IF);
		SC_STEP(w[152 + lane], 7, 5, 29, g1, g2, g3, g0, IF);
		SC_STEP(w[160 + lane], 4, 29, 9, g0, g1, g2, g3, MAJ);
		SC_STEP(w[168 + lane], 1, 9, 15, g3, g0, g1, g2, MAJ);
		SC_STEP(w[176 + lane], 6, 15, 5, g2, g3, g0, g1, MAJ);
		SC_STEP(w[184 + lane], 2, 5, 29, g1, g2, g3, g0, MAJ);
		SC_STEP(w[192 + lane], 3, 4, 13, g0, g1, g2, g3, IF);
		SC_STEP(w[200 + lane], 5, 13, 10, g3, g0, g1, g2, IF);
		SC_STEP(w[208 + lane], 7, 10, 25, g2, g3, g0, g1, IF);
		SC_STEP(w[216 + lane], 4, 25, 4, g1, g2, g3, g0, IF);
		SC_STEP(w[224 + lane], 1, 4, 13, g0, g1, g2, g3, MAJ);
		SC_STEP(w[232 + lane], 6, 13, 10, g3, g0, g1, g2, MAJ);
		SC_STEP(w[240 + lane], 2, 10, 25, g2, g3, g0, g1, MAJ);
		SC_STEP(w[248 + lane], 3, 25, 4, g1, g2, g3, g0, MAJ);

		SC_STEP(iv0, 5, 4, 13, g0, g1, g2, g3, IF);
		SC_STEP(iv1, 7, 13, 10, g3, g0, g1, g2, IF);
		SC_STEP(iv2, 4, 10, 25, g2, g3, g0, g1, IF);
		SC_STEP(iv3, 1, 25, 4, g1, g2, g3, g0, IF);

		const uint32_t f0 = g0, f1 = g1, f2 = g2, f3 = g3;
		if (lane == 0) g0 ^= 512;

		SC_STEP(__ldg(&d_cw[0][0][0 + lane]), 1, 3, 23, g0, g1, g2, g3, IF);
		SC_STEP(__ldg(&d_cw[0][0][8 + lane]), 6, 23, 17, g3, g0, g1, g2, IF);
		SC_STEP(__ldg(&d_cw[0][0][16 + lane]), 2, 17, 27, g2, g3, g0, g1, IF);
		SC_STEP(__ldg(&d_cw[0][0][24 + lane]), 3, 27, 3, g1, g2, g3, g0, IF);
		SC_STEP(__ldg(&d_cw[0][0][32 + lane]), 5, 3, 23, g0, g1, g2, g3, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][40 + lane]), 7, 23, 17, g3, g0, g1, g2, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][48 + lane]), 4, 17, 27, g2, g3, g0, g1, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][56 + lane]), 1, 27, 3, g1, g2, g3, g0, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][64 + lane]), 6, 28, 19, g0, g1, g2, g3, IF);
		SC_STEP(__ldg(&d_cw[0][0][72 + lane]), 2, 19, 22, g3, g0, g1, g2, IF);
		SC_STEP(__ldg(&d_cw[0][0][80 + lane]), 3, 22, 7, g2, g3, g0, g1, IF);
		SC_STEP(__ldg(&d_cw[0][0][88 + lane]), 5, 7, 28, g1, g2, g3, g0, IF);
		SC_STEP(__ldg(&d_cw[0][0][96 + lane]), 7, 28, 19, g0, g1, g2, g3, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][104 + lane]), 4, 19, 22, g3, g0, g1, g2, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][112 + lane]), 1, 22, 7, g2, g3, g0, g1, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][120 + lane]), 6, 7, 28, g1, g2, g3, g0, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][128 + lane]), 2, 29, 9, g0, g1, g2, g3, IF);
		SC_STEP(__ldg(&d_cw[0][0][136 + lane]), 3, 9, 15, g3, g0, g1, g2, IF);
		SC_STEP(__ldg(&d_cw[0][0][144 + lane]), 5, 15, 5, g2, g3, g0, g1, IF);
		SC_STEP(__ldg(&d_cw[0][0][152 + lane]), 7, 5, 29, g1, g2, g3, g0, IF);
		SC_STEP(__ldg(&d_cw[0][0][160 + lane]), 4, 29, 9, g0, g1, g2, g3, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][168 + lane]), 1, 9, 15, g3, g0, g1, g2, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][176 + lane]), 6, 15, 5, g2, g3, g0, g1, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][184 + lane]), 2, 5, 29, g1, g2, g3, g0, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][192 + lane]), 3, 4, 13, g0, g1, g2, g3, IF);
		SC_STEP(__ldg(&d_cw[0][0][200 + lane]), 5, 13, 10, g3, g0, g1, g2, IF);
		SC_STEP(__ldg(&d_cw[0][0][208 + lane]), 7, 10, 25, g2, g3, g0, g1, IF);
		SC_STEP(__ldg(&d_cw[0][0][216 + lane]), 4, 25, 4, g1, g2, g3, g0, IF);
		SC_STEP(__ldg(&d_cw[0][0][224 + lane]), 1, 4, 13, g0, g1, g2, g3, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][232 + lane]), 6, 13, 10, g3, g0, g1, g2, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][240 + lane]), 2, 10, 25, g2, g3, g0, g1, MAJ);
		SC_STEP(__ldg(&d_cw[0][0][248 + lane]), 3, 25, 4, g1, g2, g3, g0, MAJ);

		SC_STEP(f0, 5, 4, 13, g0, g1, g2, g3, IF);
		SC_STEP(f1, 7, 13, 10, g3, g0, g1, g2, IF);
		SC_STEP(f2, 4, 10, 25, g2, g3, g0, g1, IF);
		SC_STEP(f3, 1, 25, 4, g1, g2, g3, g0, IF);

		Hash[lane] = g0;
		Hash[lane + 8] = g1;
	}
}

bool simd512_device_selftest(int thr_id, uint32_t threads);

__host__
int simd512_cpu_init(int thr_id, uint32_t threads)
{
	simd512_device_selftest(thr_id, threads);
	return 0;
}

__host__
void simd512_cpu_free(int thr_id)
{
}

__host__
void simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const dim3 grid((8 * threads + SIMDF_TPB - 1) / SIMDF_TPB);
	simd512_gpu_fused_64 <<<grid, SIMDF_TPB>>> (threads, d_hash);
}

/* ------------------------------------------------------------------ self-test
 * Init-time KAT (cuda/stage_selftest.cuh) of the launcher vs sph_simd512.
 * Fail-closed. */
extern "C" {
#include "sph/sph_simd.h"
}
#include "cuda/stage_selftest.cuh"

static int s_simd_st_thr;

static bool simd512_st_gpu(const uint8_t *in, uint8_t *out, int n)
{
	uint32_t *d = NULL;
	if (cudaMalloc(&d, (size_t)n * 64) != cudaSuccess) return selftest_cuda_fault();
	bool ok = (cudaMemcpy(d, in, (size_t)n * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	simd512_cpu_hash_64(s_simd_st_thr, n, 0, NULL, d, 0);
	ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
	ok = ok && (cudaMemcpy(out, d, (size_t)n * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d);
	return ok ? true : selftest_cuda_fault();
}

static void simd512_st_ref(const uint8_t *in, uint8_t *out)
{
	sph_simd512_context c;
	sph_simd512_init(&c);
	sph_simd512(&c, in, 64);
	sph_simd512_close(&c, out);
}

__host__
bool simd512_device_selftest(int thr_id, uint32_t threads)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;
	s_simd_st_thr = thr_id;
	const int n = threads < 256 ? (int)threads : 256;
	if (n < 64) { // not at a mining throughput; report it rather than pass silently
		gpulog(LOG_WARNING, thr_id, "simd512 device self-test NOT RUN (throughput %u < 64)", threads);
		return passed = true;
	}
	passed = stkat_run(thr_id, "simd512", 64, 64, n, 0x53494D44u, simd512_st_gpu, simd512_st_ref, true);
	return passed;
}
