/*
 * Quick Hamsi-512 for X13 by tsiv - 2014
 * + Hamsi-512 80 by tpruvot - 2018
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "miner.h"
#include "cuda/helper.cuh"

typedef unsigned char BitSequence;

#include "cuda/hamsi512_device.cuh"

// Tables, round macros and hamsi512_hash_64 live in cuda/hamsi512_device.cuh
// (statically initialized — the init-time uploads are gone); the 64-byte
// kernel below is a thin wrapper, the 80-byte first-stage kernel expands
// the shared macros/tables directly.

/* byte table for hamsi512_hash_64_tb, one per CUDA device (shared with the x-fused kernel) */
static uint4 *d_hamsi_TB[MAX_GPUS];
static pthread_mutex_t hamsi_tb_lock = PTHREAD_MUTEX_INITIALIZER;

__global__ void hamsi512_fill_tb(uint32_t *t)
{
	const int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= 8 * 256 * 16) return;
	const int p = e >> 12, v = (e >> 4) & 255, w = e & 15;
	uint32_t x = 0;
	for (int b = 0; b < 8; b++) if (v & (1 << b)) x ^= d_T512[8 * p + b][w];
	t[e] = x;
}

/* the current device's table, built on first use */
__host__
const uint4 *hamsi512_table(void)
{
	int dev = 0;
	cudaGetDevice(&dev);
	pthread_mutex_lock(&hamsi_tb_lock);
	if (!d_hamsi_TB[dev]) {
		CUDA_SAFE_CALL(cudaMalloc(&d_hamsi_TB[dev], 8 * 256 * 16 * sizeof(uint32_t)));
		hamsi512_fill_tb<<<(8 * 256 * 16) / 256, 256>>>((uint32_t*)d_hamsi_TB[dev]);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}
	pthread_mutex_unlock(&hamsi_tb_lock);
	return d_hamsi_TB[dev];
}

/* nibble table for hamsi512_hash_64_nt, [8][256] uint2, per device; copied to shared per block */
static uint2 *d_hamsi_NT[MAX_GPUS];

__global__ void hamsi512_fill_nt(uint2 *nt)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 8 * 256) return;
	const int k = i >> 8, e = i & 255, p = e >> 4, v = e & 15;
	uint32_t x = 0, y = 0;
	for (int b = 0; b < 4; b++) if (v & (1 << b)) { x ^= d_T512[4 * p + b][2 * k]; y ^= d_T512[4 * p + b][2 * k + 1]; }
	nt[i] = make_uint2(x, y);
}

static const uint2 *hamsi512_ntable(void)
{
	int dev = 0;
	cudaGetDevice(&dev);
	pthread_mutex_lock(&hamsi_tb_lock);
	if (!d_hamsi_NT[dev]) {
		CUDA_SAFE_CALL(cudaMalloc(&d_hamsi_NT[dev], 8 * 256 * sizeof(uint2)));
		hamsi512_fill_nt<<<8, 256>>>(d_hamsi_NT[dev]);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
	}
	pthread_mutex_unlock(&hamsi_tb_lock);
	return d_hamsi_NT[dev];
}

/* 512x2 on Pascal, 256x3 above */
template <int TPB, int MINB>
__global__ __launch_bounds__(TPB, MINB)
void hamsi512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector, const uint2 *__restrict__ gnt)
{
	__shared__ uint2 nt[8][256];
	for (int i = threadIdx.x; i < 8 * 256; i += TPB) nt[i >> 8][i & 255] = __ldg(&gnt[i]);
	__syncthreads();

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];

		hamsi512_hash_64_nt(Hash, nt);
	}
}

/* Unit self-test for cuda/hamsi512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool hamsi512_device_selftest(int thr_id);
static bool hamsi512_stage_selftest(int thr_id, uint32_t threads);

__host__
void hamsi512_cpu_init(int thr_id, uint32_t threads)
{
	hamsi512_device_selftest(thr_id);
	hamsi512_table();
	hamsi512_ntable();
	hamsi512_stage_selftest(thr_id, threads);
}

__host__
void hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint2 *nt = hamsi512_ntable();
	if (device_sm[device_map[thr_id]] < 700) {
		hamsi512_gpu_hash_64<512, 2><<<(threads + 511) / 512, 512>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector, nt);
	} else {
		hamsi512_gpu_hash_64<256, 3><<<(threads + 255) / 256, 256>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector, nt);
	}
	//MyStreamSynchronize(NULL, order, thr_id);
}

__constant__ static uint64_t c_PaddedMessage80[10];

__host__
void x16_hamsi512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__
void x16_hamsi512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		unsigned char h1[80];
		#pragma unroll
		for (int i = 0; i < 10; i++)
			((uint2*)h1)[i] = ((uint2*)c_PaddedMessage80)[i];
		//((uint64_t*)h1)[9] = REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(startNonce + thread));
		((uint32_t*)h1)[19] = cuda_swab32(startNonce + thread);

		uint32_t c0 = 0x73746565, c1 = 0x6c706172, c2 = 0x6b204172, c3 = 0x656e6265;
		uint32_t c4 = 0x72672031, c5 = 0x302c2062, c6 = 0x75732032, c7 = 0x3434362c;
		uint32_t c8 = 0x20422d33, c9 = 0x30303120, cA = 0x4c657576, cB = 0x656e2d48;
		uint32_t cC = 0x65766572, cD = 0x6c65652c, cE = 0x2042656c, cF = 0x6769756d;
		uint32_t h[16] = { c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF };
		uint32_t m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, mA, mB, mC, mD, mE, mF;
		uint32_t *tp, db, dm;

		for(int i = 0; i < 80; i += 8)
		{
			m0 = 0; m1 = 0; m2 = 0; m3 = 0; m4 = 0; m5 = 0; m6 = 0; m7 = 0;
			m8 = 0; m9 = 0; mA = 0; mB = 0; mC = 0; mD = 0; mE = 0; mF = 0;
			tp = &d_T512[0][0];

			#pragma unroll
			for (int u = 0; u < 8; u++) {
				db = h1[i + u];
				#pragma unroll 2
				for (int v = 0; v < 8; v++, db >>= 1) {
					dm = -(uint32_t)(db & 1);
					m0 ^= dm & tp[ 0]; m1 ^= dm & tp[ 1];
					m2 ^= dm & tp[ 2]; m3 ^= dm & tp[ 3];
					m4 ^= dm & tp[ 4]; m5 ^= dm & tp[ 5];
					m6 ^= dm & tp[ 6]; m7 ^= dm & tp[ 7];
					m8 ^= dm & tp[ 8]; m9 ^= dm & tp[ 9];
					mA ^= dm & tp[10]; mB ^= dm & tp[11];
					mC ^= dm & tp[12]; mD ^= dm & tp[13];
					mE ^= dm & tp[14]; mF ^= dm & tp[15];
					tp += 16;
				}
			}

			#pragma unroll
			for (int r = 0; r < 6; r++) {
				ROUND_BIG(r, d_alpha_n);
			}
			T_BIG;
		}

		#define INPUT_BIG { \
			m0 = 0; m1 = 0; m2 = 0; m3 = 0; m4 = 0; m5 = 0; m6 = 0; m7 = 0; \
			m8 = 0; m9 = 0; mA = 0; mB = 0; mC = 0; mD = 0; mE = 0; mF = 0; \
			tp = &d_T512[0][0]; \
			for (int u = 0; u < 8; u++) { \
				db = endtag[u]; \
				for (int v = 0; v < 8; v++, db >>= 1) { \
					dm = -(uint32_t)(db & 1); \
					m0 ^= dm & tp[ 0]; m1 ^= dm & tp[ 1]; \
					m2 ^= dm & tp[ 2]; m3 ^= dm & tp[ 3]; \
					m4 ^= dm & tp[ 4]; m5 ^= dm & tp[ 5]; \
					m6 ^= dm & tp[ 6]; m7 ^= dm & tp[ 7]; \
					m8 ^= dm & tp[ 8]; m9 ^= dm & tp[ 9]; \
					mA ^= dm & tp[10]; mB ^= dm & tp[11]; \
					mC ^= dm & tp[12]; mD ^= dm & tp[13]; \
					mE ^= dm & tp[14]; mF ^= dm & tp[15]; \
					tp += 16; \
				} \
			} \
		}

		// close
		uint8_t endtag[8] = { 0x80, 0x00, 0x00, 0x00,  0x00, 0x00, 0x00, 0x00 };
		INPUT_BIG;

		#pragma unroll
		for (int r = 0; r < 6; r++) {
			ROUND_BIG(r, d_alpha_n);
		}
		T_BIG;

		endtag[0] = endtag[1] = 0x00;
		endtag[6] = 0x02;
		endtag[7] = 0x80;
		INPUT_BIG;

		// PF_BIG
		#pragma unroll
		for(int r = 0; r < 12; r++) {
			ROUND_BIG(r, d_alpha_f);
		}
		T_BIG;

		uint64_t hashPosition = thread;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition << 3];
		#pragma unroll 16
		for(int i = 0; i < 16; i++)
			Hash[i] = cuda_swab32(h[i]);

		#undef INPUT_BIG
	}
}

__host__
void x16_hamsi512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x16_hamsi512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, (uint64_t*)d_hash);
}

/* ------------------------------------------------------------------ self-test
 * Init-time KAT (cuda/stage_selftest.cuh) of the table launcher vs sph_hamsi512. Fail-closed. */
extern "C" {
#include "sph/sph_hamsi.h"
}
#include "cuda/stage_selftest.cuh"

static int s_hamsi_st_thr;

static bool hamsi512_st_gpu(const uint8_t *in, uint8_t *out, int n)
{
	uint32_t *d = NULL;
	if (cudaMalloc(&d, (size_t)n * 64) != cudaSuccess) return selftest_cuda_fault();
	bool ok = (cudaMemcpy(d, in, (size_t)n * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	hamsi512_cpu_hash_64(s_hamsi_st_thr, n, 0, NULL, d, 0);
	ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
	ok = ok && (cudaMemcpy(out, d, (size_t)n * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d);
	return ok ? true : selftest_cuda_fault();
}

static void hamsi512_st_ref(const uint8_t *in, uint8_t *out)
{
	sph_hamsi512_context c;
	sph_hamsi512_init(&c);
	sph_hamsi512(&c, in, 64);
	sph_hamsi512_close(&c, out);
}

static bool hamsi512_stage_selftest(int thr_id, uint32_t threads)
{
	static bool tested[MAX_GPUS], passed[MAX_GPUS];
	if (tested[thr_id]) return passed[thr_id];
	tested[thr_id] = true;
	s_hamsi_st_thr = thr_id;
	const int n = threads < 256 ? (int)threads : 256;
	if (n < 64) { // not at a mining throughput; report it rather than pass silently
		gpulog(LOG_WARNING, thr_id, "hamsi512 stage self-test NOT RUN (throughput %u < 64)", threads);
		return passed[thr_id] = true;
	}
	passed[thr_id] = stkat_run(thr_id, "hamsi512 stage", 64, 64, n, 0x48414D53u, hamsi512_st_gpu, hamsi512_st_ref, true);
	return passed[thr_id];
}
