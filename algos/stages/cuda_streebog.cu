/*
 * Streebog GOST R 34.10-2012 CUDA implementation — consolidated stage.
 *
 * The single canonical Streebog stage: a plain 64-byte hash
 * (streebog_cpu_hash_64) and a final variant with an on-device target compare
 * (streebog_cpu_hash_64_final / streebog_set_target). Consolidated from the
 * former x11/cuda_streebog_maxwell.cu (phi/sib) and skunk/cuda_skunk_streebog.cu
 * (x21s + veltor/polytimos/skunk) — bit-identical kernels that differed only in
 * shared-memory footprint; the leaner 7-table variant is kept. The dead
 * sub-sm_61 compat kernels (former x11/cuda_streebog.cu) were dropped.
 *
 * https://tools.ietf.org/html/rfc6986
 * https://en.wikipedia.org/wiki/Streebog
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * @author   Tanguy Pruvot - 2017
 * @author   Alexis Provos - 2016
 */

#include <string.h>

#include <miner.h>
#include "cuda/helper.cuh"
#include "cuda/vectors.cuh"
#include "cuda/vector_uint2x4.cuh"

#include "cuda/selftest_gate.cuh"

extern "C" {
#include "sph/sph_streebog.h"
}

#include "streebog_arrays.cuh"
#include "cuda/candidate_report.cuh"

// Sigma += pad for a 64-byte message: +1 at byte 63 of a big-endian 512-bit
// integer, with the carry rippling toward byte 0 as in sph AddModulo512.
__device__ __forceinline__
static void streebog_sigma_add_pad(uint2 *h)
{
	#pragma unroll
	for (int j = 7; j >= 0; j--) {
		const uint64_t w = cuda_swab64(devectorize(h[j])) + 1;
		h[j] = vectorize(cuda_swab64(w));
		if (w != 0) break;
	}
}

//#define FULL_UNROLL
__device__ __forceinline__
static void GOST_FS(const uint2 shared[8][256],const uint2 *const __restrict__ state,uint2* return_state)
{
	return_state[0] = __ldg(&T02[__byte_perm(state[7].x,0,0x44440)])
			^ shared[1][__byte_perm(state[6].x,0,0x44440)]
			^ shared[2][__byte_perm(state[5].x,0,0x44440)]
			^ shared[3][__byte_perm(state[4].x,0,0x44440)]
			^ shared[4][__byte_perm(state[3].x,0,0x44440)]
			^ shared[5][__byte_perm(state[2].x,0,0x44440)]
			^ shared[6][__byte_perm(state[1].x,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44440)]);

	return_state[1] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44441)])
			^ shared[2][__byte_perm(state[5].x,0,0x44441)]
			^ shared[3][__byte_perm(state[4].x,0,0x44441)]
			^ shared[4][__byte_perm(state[3].x,0,0x44441)]
			^ shared[5][__byte_perm(state[2].x,0,0x44441)]
			^ shared[6][__byte_perm(state[1].x,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44441)]);

	return_state[2] = __ldg(&T02[__byte_perm(state[7].x,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44442)])
			^ shared[2][__byte_perm(state[5].x,0,0x44442)]
			^ shared[3][__byte_perm(state[4].x,0,0x44442)]
			^ shared[4][__byte_perm(state[3].x,0,0x44442)]
			^ shared[5][__byte_perm(state[2].x,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44442)])
			^ shared[6][__byte_perm(state[1].x,0,0x44442)];

	return_state[3] = __ldg(&T02[__byte_perm(state[7].x,0,0x44443)])
			^ shared[1][__byte_perm(state[6].x,0,0x44443)]
			^ shared[2][__byte_perm(state[5].x,0,0x44443)]
			^ shared[3][__byte_perm(state[4].x,0,0x44443)]
			^ __ldg(&T42[__byte_perm(state[3].x,0,0x44443)])
			^ shared[5][__byte_perm(state[2].x,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44443)])
			^ shared[6][__byte_perm(state[1].x,0,0x44443)];

	return_state[4] = __ldg(&T02[__byte_perm(state[7].y,0,0x44440)])
			^ shared[1][__byte_perm(state[6].y,0,0x44440)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44440)])
			^ shared[3][__byte_perm(state[4].y,0,0x44440)]
			^ shared[4][__byte_perm(state[3].y,0,0x44440)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44440)])
			^ shared[5][__byte_perm(state[2].y,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44440)]);

	return_state[5] = __ldg(&T02[__byte_perm(state[7].y,0,0x44441)])
			^ shared[2][__byte_perm(state[5].y,0,0x44441)]
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44441)])
			^ shared[3][__byte_perm(state[4].y,0,0x44441)]
			^ shared[4][__byte_perm(state[3].y,0,0x44441)]
			^ shared[5][__byte_perm(state[2].y,0,0x44441)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44441)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44441)]);

	return_state[6] = __ldg(&T02[__byte_perm(state[7].y,0,0x44442)])
			^ shared[1][__byte_perm(state[6].y,0,0x44442)]
			^ shared[2][__byte_perm(state[5].y,0,0x44442)]
			^ shared[3][__byte_perm(state[4].y,0,0x44442)]
			^ shared[4][__byte_perm(state[3].y,0,0x44442)]
			^ shared[5][__byte_perm(state[2].y,0,0x44442)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44442)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44442)]);

	return_state[7] = __ldg(&T02[__byte_perm(state[7].y,0,0x44443)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44443)])
			^ shared[2][__byte_perm(state[5].y,0,0x44443)]
			^ shared[3][__byte_perm(state[4].y,0,0x44443)]
			^ shared[4][__byte_perm(state[3].y,0,0x44443)]
			^ shared[5][__byte_perm(state[2].y,0,0x44443)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44443)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44443)]);
}

__device__ __forceinline__
static void GOST_FS_LDG(const uint2 shared[8][256],const uint2 *const __restrict__ state,uint2* return_state)
{
	return_state[0] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44440)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44440)])
			^ shared[2][__byte_perm(state[5].x,0,0x44440)]
			^ shared[3][__byte_perm(state[4].x,0,0x44440)]
			^ shared[4][__byte_perm(state[3].x,0,0x44440)]
			^ shared[5][__byte_perm(state[2].x,0,0x44440)]
			^ shared[6][__byte_perm(state[1].x,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44440)]);

	return_state[1] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44441)])
			^ shared[2][__byte_perm(state[5].x,0,0x44441)]
			^ shared[3][__byte_perm(state[4].x,0,0x44441)]
			^ shared[4][__byte_perm(state[3].x,0,0x44441)]
			^ shared[5][__byte_perm(state[2].x,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44441)])
			^ shared[6][__byte_perm(state[1].x,0,0x44441)];

	return_state[2] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44442)])
			^ shared[2][__byte_perm(state[5].x,0,0x44442)]
			^ shared[3][__byte_perm(state[4].x,0,0x44442)]
			^ shared[4][__byte_perm(state[3].x,0,0x44442)]
			^ shared[5][__byte_perm(state[2].x,0,0x44442)]
			^ shared[6][__byte_perm(state[1].x,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44442)]);

	return_state[3] = __ldg(&T02[__byte_perm(state[7].x,0,0x44443)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44443)])
			^ shared[2][__byte_perm(state[5].x,0,0x44443)]
			^ shared[3][__byte_perm(state[4].x,0,0x44443)]
			^ shared[4][__byte_perm(state[3].x,0,0x44443)]
			^ shared[5][__byte_perm(state[2].x,0,0x44443)]
			^ shared[6][__byte_perm(state[1].x,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44443)]);

	return_state[4] = __ldg(&T02[__byte_perm(state[7].y,0,0x44440)])
			^ shared[1][__byte_perm(state[6].y,0,0x44440)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44440)])
			^ shared[3][__byte_perm(state[4].y,0,0x44440)]
			^ shared[4][__byte_perm(state[3].y,0,0x44440)]
			^ shared[5][__byte_perm(state[2].y,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44440)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44440)]);

	return_state[5] =  __ldg(&T02[__byte_perm(state[7].y,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44441)])
			^ shared[2][__byte_perm(state[5].y,0,0x44441)]
			^ shared[3][__byte_perm(state[4].y,0,0x44441)]
			^ shared[4][__byte_perm(state[3].y,0,0x44441)]
			^ shared[5][__byte_perm(state[2].y,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44441)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44441)]);

	return_state[6] = __ldg(&T02[__byte_perm(state[7].y,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44442)])
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44442)])
			^ shared[3][__byte_perm(state[4].y,0,0x44442)]
			^ shared[4][__byte_perm(state[3].y,0,0x44442)]
			^ shared[5][__byte_perm(state[2].y,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44442)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44442)]);

	return_state[7] = __ldg(&T02[__byte_perm(state[7].y,0,0x44443)])
			^ shared[1][__byte_perm(state[6].y,0,0x44443)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44443)])
			^ shared[3][__byte_perm(state[4].y,0,0x44443)]
			^ shared[4][__byte_perm(state[3].y,0,0x44443)]
			^ shared[5][__byte_perm(state[2].y,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44443)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44443)]);
}

__device__ __forceinline__
static void GOST_E12(const uint2 shared[8][256],uint2 *const __restrict__ K, uint2 *const __restrict__ state)
{
	uint2 t[ 8];
	//#pragma unroll 12
	for(int i=0; i<12; i++){
		GOST_FS(shared,state, t);

		#pragma unroll 8
		for(int j=0;j<8;j++)
			K[ j] ^= *(uint2*)&CC[i][j];

		#pragma unroll 8
		for(int j=0;j<8;j++)
			state[ j] = t[ j];

		GOST_FS_LDG(shared,K, t);

		#pragma unroll 8
		for(int j=0;j<8;j++)
			state[ j]^= t[ j];

		#pragma unroll 8
		for(int j=0;j<8;j++)
			K[ j] = t[ j];
	}
}


#define TPB 256
__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB, 2)
#else
__launch_bounds__(TPB, 3)
#endif
void streebog_gpu_hash_64(const uint32_t threads, uint64_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint2 buf[8], t[8], temp[8], K0[8], hash[8];

	__shared__ uint2 shared[7][256];
	// shared[0] (T02) is never read here: GOST_FS/GOST_FS_LDG reach the T0 slot
	// via __ldg and only gather shared[1..6]; T72 stays in __ldg too. Staging
	// row 0 was a dead cooperative load (only the _final kernel's tail reads it).
	// the fill needs all TPB (== row length) threads, so tail threads leave after the barrier
	//shared[0][threadIdx.x] = __ldg(&T02[threadIdx.x]);
	shared[1][threadIdx.x] = __ldg(&T12[threadIdx.x]);
	shared[2][threadIdx.x] = __ldg(&T22[threadIdx.x]);
	shared[3][threadIdx.x] = __ldg(&T32[threadIdx.x]);
	shared[4][threadIdx.x] = __ldg(&T42[threadIdx.x]);
	shared[5][threadIdx.x] = __ldg(&T52[threadIdx.x]);
	shared[6][threadIdx.x] = __ldg(&T62[threadIdx.x]);
	//shared[7][threadIdx.x] = __ldg(&T72[threadIdx.x]);

	// input load before the barrier so its latency overlaps the fill
	const bool active = (thread < threads);
	uint64_t* inout = &g_hash[thread<<3];
	if (active) {
		*(uint2x4*)&hash[0] = __ldg4((uint2x4*)&inout[0]);
		*(uint2x4*)&hash[4] = __ldg4((uint2x4*)&inout[4]);
	}

	__syncthreads();

	if (!active)
		return;

	#pragma unroll
	for(int i = 0; i < 8; i++) buf[i] = vectorize(0x74a5d4ce2efc83b3) ^ hash[i];

	#pragma nounroll
	for(int i = 0; i < 12; i++) {
		GOST_FS(shared, buf, temp);
		#pragma unroll
		for(uint32_t j = 0; j < 8; j++) buf[j] = temp[j] ^ *(uint2*)&precomputed_values[i][j];
	}

	#pragma unroll
	for(int j = 0; j < 8; j++) buf[j] ^= hash[j];

	#pragma unroll
	for(int j = 0; j < 8; j++) K0[j] = buf[j];
	K0[7].y ^= 0x00020000;

	GOST_FS(shared, K0, t);

	#pragma unroll
	for(int i = 0; i < 8; i++) K0[i] = t[i];

	t[7].y ^= 0x01000000;

	GOST_E12(shared, K0, t);

	#pragma unroll
	for(int j = 0; j < 8; j++) buf[j] ^= t[j];

	buf[7].y ^= 0x01000000;

	GOST_FS(shared, buf,K0);

	buf[7].y ^= 0x00020000;

	#pragma unroll
	for(int j = 0; j < 8; j++) t[j] = K0[j];

	t[7].y ^= 0x00020000;

	GOST_E12(shared, K0, t);

	#pragma unroll
	for(int j = 0; j < 8; j++) buf[j] ^= t[j];

	GOST_FS(shared, buf,K0); // K = F(h)

	streebog_sigma_add_pad(hash); // Sigma = M + pad, with the carry

	#pragma unroll
	for(int j = 0; j < 8; j++) t[j] = K0[j] ^ hash[j];

	GOST_E12(shared, K0, t);

	*(uint2x4*)&inout[ 0] = *(uint2x4*)&t[ 0] ^ *(uint2x4*)&hash[0] ^ *(uint2x4*)&buf[0];
	*(uint2x4*)&inout[ 4] = *(uint2x4*)&t[ 4] ^ *(uint2x4*)&hash[4] ^ *(uint2x4*)&buf[4];
}

__host__
void streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	streebog_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

__constant__ uint64_t target64[4];

__host__
void streebog_set_target(uint32_t* ptarget)
{
	cudaMemcpyToSymbol(target64, ptarget, 4*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

#define TPB 256
__global__
__launch_bounds__(TPB, 2)
void streebog_gpu_hash_64_final(const uint32_t threads, uint64_t *g_hash, uint32_t* resNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint2 buf[8], t[8], temp[8], K0[8], hash[8];

	__shared__ uint2 shared[8][256];
	shared[0][threadIdx.x] = __ldg(&T02[threadIdx.x]);
	shared[1][threadIdx.x] = __ldg(&T12[threadIdx.x]);
	shared[2][threadIdx.x] = __ldg(&T22[threadIdx.x]);
	shared[3][threadIdx.x] = __ldg(&T32[threadIdx.x]);
	shared[4][threadIdx.x] = __ldg(&T42[threadIdx.x]);
	shared[5][threadIdx.x] = __ldg(&T52[threadIdx.x]);
	shared[6][threadIdx.x] = __ldg(&T62[threadIdx.x]);
	shared[7][threadIdx.x] = __ldg(&T72[threadIdx.x]);

	// a barrier, not a fence: the gathers below read rows other warps wrote
	const bool active = (thread < threads);
	uint64_t* inout = &g_hash[thread<<3];
	if (active) {
		*(uint2x4*)&hash[0] = __ldg4((uint2x4*)&inout[0]);
		*(uint2x4*)&hash[4] = __ldg4((uint2x4*)&inout[4]);
	}

	__syncthreads();

	if (!active)
		return; // tail of a ragged batch; no barrier follows

	K0[0] = vectorize(0x74a5d4ce2efc83b3);

	#pragma unroll 8
	for(uint32_t i=0;i<8;i++){
		buf[ i] = hash[ i] ^ K0[ 0];
	}
	//#pragma unroll 12
	for(int i=0; i<12; i++){
		GOST_FS(shared, buf, temp);
		#pragma unroll 8
		for(uint32_t j=0;j<8;j++){
			buf[ j] = temp[ j] ^ *(uint2*)&precomputed_values[i][j];
		}
	}
	#pragma unroll 8
	for(int j=0;j<8;j++){
		buf[ j]^= hash[ j];
	}
	#pragma unroll 8
	for(int j=0;j<8;j++){
		K0[ j] = buf[ j];
	}

	K0[7].y ^= 0x00020000;

	GOST_FS(shared, K0, t);

	#pragma unroll 8
	for(uint32_t i=0;i<8;i++)
		K0[ i] = t[ i];

	t[7].y ^= 0x01000000;
	GOST_E12(shared, K0, t);

	#pragma unroll 8
	for(int j=0;j<8;j++)
		buf[ j] ^= t[ j];

	buf[7].y ^= 0x01000000;

	GOST_FS(shared, buf,K0);

	buf[7].y ^= 0x00020000;

	#pragma unroll 8
	for(uint32_t j=0;j<8;j++)
		t[ j] = K0[ j];

	t[7].y ^= 0x00020000;
	GOST_E12(shared, K0, t);

	#pragma unroll 8
	for(uint32_t j=0;j<8;j++)
		buf[ j] ^= t[ j];

	GOST_FS(shared, buf,K0); // K = F(h)

	streebog_sigma_add_pad(hash); // Sigma = M + pad, with the carry

	#pragma unroll 8
	for(uint32_t j=0;j<8;j++)
		t[ j] = K0[ j] ^ hash[ j];

//	#pragma unroll
	for(uint32_t i=0; i<10; i++){
		GOST_FS(shared, t, temp);

		#pragma unroll 8
		for(uint32_t j=0;j<8;j++){
			t[ j] = temp[ j];
			K0[ j] = K0[ j] ^ *(uint2*)&CC[ i][ j];
		}

		GOST_FS(shared, K0, temp);

		#pragma unroll 8
		for(uint32_t j=0;j<8;j++){
			K0[ j] = temp[ j];
			t[ j]^= temp[ j];
		}
	}

	GOST_FS(shared, t, temp);

	#pragma unroll 8
	for(uint32_t j=0;j<8;j++){
		t[ j] = temp[ j];
		K0[ j] = K0[ j] ^ *(uint2*)&CC[10][ j];
	}

	GOST_FS(shared, K0, temp);

	#pragma unroll 8
	for(int i=7;i>=0;i--){
		t[i].x = t[i].x ^ temp[i].x;
		temp[i].x = temp[i].x ^ ((uint32_t*)&CC[11])[i<<1];
	}

	uint2 last[2];

#define T0(x) shared[0][x]
#define T1(x) shared[1][x]
#define T2(x) shared[2][x]
#define T3(x) shared[3][x]
#define T4(x) shared[4][x]
#define T5(x) shared[5][x]
#define T6(x) shared[6][x]
#define T7(x) shared[7][x]

	last[ 0] = T0(__byte_perm(t[7].x,0,0x44443)) ^ T1(__byte_perm(t[6].x,0,0x44443))
		 ^ T2(__byte_perm(t[5].x,0,0x44443)) ^ T3(__byte_perm(t[4].x,0,0x44443))
		 ^ T4(__byte_perm(t[3].x,0,0x44443)) ^ T5(__byte_perm(t[2].x,0,0x44443))
		 ^ T6(__byte_perm(t[1].x,0,0x44443)) ^ T7(__byte_perm(t[0].x,0,0x44443));

	last[ 1] = T0(__byte_perm(temp[7].x,0,0x44443)) ^ T1(__byte_perm(temp[6].x,0,0x44443))
		 ^ T2(__byte_perm(temp[5].x,0,0x44443)) ^ T3(__byte_perm(temp[4].x,0,0x44443))
		 ^ T4(__byte_perm(temp[3].x,0,0x44443)) ^ T5(__byte_perm(temp[2].x,0,0x44443))
		 ^ T6(__byte_perm(temp[1].x,0,0x44443)) ^ T7(__byte_perm(temp[0].x,0,0x44443));

	if(devectorize(buf[3] ^ hash[3] ^ last[ 0] ^ last[ 1]) <= target64[3]){
		report_candidate_2(resNonce, thread);
	}
}

__host__
void streebog_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	streebog_gpu_hash_64_final <<< grid, block >>> (threads, (uint64_t*)d_hash, d_resNonce);
}

/* ------------------------------------------------------------------ self-test
 * Init-time KAT for the streebog stage (docs/coding-guideline.md §7): drives the
 * real production launcher (streebog_cpu_hash_64) over a full 256-thread block
 * and compares GPU output against the vendored sph_gost512 CPU
 * reference, itself anchored to the GOST R 34.11-2012 / RFC 6986 H_512(M1)
 * vector. A flipped-input-bit negative test proves the test isn't vacuous.
 * FAIL-CLOSED via cuda/selftest_gate.cuh.
 */
#define STREEBOG_ST_ALLOC 256   /* one full block */
#define STREEBOG_ST_VEC   4     /* leading vectors actually verified */

/* GOST R 34.11-2012 / RFC 6986 §10.1.1 message M1 (63 bytes, in message order):
 * the ASCII digit run the standard hashes to its published H_512(M1). */
static const uint8_t streebog_m1[63] = {
	0x32,0x31,0x30,0x39,0x38,0x37,0x36,0x35,0x34,0x33,0x32,0x31,0x30,0x39,0x38,0x37,
	0x36,0x35,0x34,0x33,0x32,0x31,0x30,0x39,0x38,0x37,0x36,0x35,0x34,0x33,0x32,0x31,
	0x30,0x39,0x38,0x37,0x36,0x35,0x34,0x33,0x32,0x31,0x30,0x39,0x38,0x37,0x36,0x35,
	0x34,0x33,0x32,0x31,0x30,0x39,0x38,0x37,0x36,0x35,0x34,0x33,0x32,0x31,0x30
};

/* sph_gost512(M1) — verified byte-for-byte against RFC 6986 §10.1.1 H_512(M1). */
static const uint8_t kat_streebog512_m1[64] = {
	0x48,0x6f,0x64,0xc1,0x91,0x78,0x79,0x41,0x7f,0xef,0x08,0x2b,0x33,0x81,0xa4,0xe2,
	0x11,0xc3,0x24,0xf0,0x74,0x65,0x4c,0x38,0x82,0x3a,0x7b,0x76,0xf8,0x30,0xad,0x00,
	0xfa,0x1f,0xba,0xe4,0x2b,0x12,0x85,0xc0,0x35,0x2f,0x22,0x75,0x24,0xbc,0x9a,0xb1,
	0x62,0x54,0x28,0x8d,0xd6,0x86,0x3d,0xcc,0xd5,0xb9,0xf5,0x4a,0x1a,0xd0,0x54,0x1b
};
/* sph_gost512 of the 64-byte pattern 00 01 .. 3F — anchored drift check
 * (computed once from the RFC-anchored sph reference). */
static const uint8_t kat_streebog512_pat64[64] = {
	0xfb,0x41,0x80,0x21,0x9f,0x50,0x7b,0x8f,0x4b,0xf5,0x5e,0xbf,0x96,0x4e,0x3c,0xfd,
	0x80,0x62,0xaa,0x87,0x23,0xc1,0xb8,0x78,0x3c,0x54,0x69,0xb6,0x1e,0xe9,0xd4,0xe6,
	0xc5,0xc2,0xfb,0x9d,0x78,0x4c,0x66,0xf3,0xaa,0xaa,0x07,0xb1,0x26,0x6b,0x70,0x74,
	0x8e,0x6d,0x90,0x77,0x56,0x28,0x1c,0x7d,0x28,0x39,0x1b,0x83,0x7b,0x7c,0xe0,0xb9
};

static bool streebog_selftest_run(uint8_t (*io)[64] /* [STREEBOG_ST_ALLOC] */)
{
	uint32_t *d_hash = NULL;
	if (cudaMalloc(&d_hash, (size_t) STREEBOG_ST_ALLOC * 64) != cudaSuccess)
		return selftest_cuda_fault();
	bool ok = (cudaMemcpy(d_hash, io, (size_t) STREEBOG_ST_ALLOC * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	streebog_cpu_hash_64(0, STREEBOG_ST_ALLOC, d_hash);
	ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
	ok = ok && (cudaMemcpy(io, d_hash, (size_t) STREEBOG_ST_ALLOC * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_hash);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool streebog_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_gost512_context ctx;
	uint8_t dig[64];

	// --- anchor the sph reference against the RFC 6986 M1 spec vector ---
	sph_gost512_init(&ctx);
	sph_gost512(&ctx, streebog_m1, sizeof(streebog_m1));
	sph_gost512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_streebog512_m1, 64) == 0);

	// --- test vectors: fixed pattern + LCG-filled (256 to fill the block) ---
	uint8_t (*io)[64] = (uint8_t(*)[64]) malloc((size_t) STREEBOG_ST_ALLOC * 64);
	if (!io) return false;
	uint8_t ref[STREEBOG_ST_VEC][64];
	uint32_t seed = 0x47535431; /* 'GST1' */
	for (int i = 0; i < 64; i++) io[0][i] = (uint8_t) i;
	for (int v = 1; v < STREEBOG_ST_ALLOC; v++)
		for (int i = 0; i < 64; i++) {
			seed = seed * 1664525u + 1013904223u;
			io[v][i] = (uint8_t)(seed >> 24);
		}
	// vectors 1..3 force the Sigma pad carry: within a word, into the next, across two
	io[1][63] = 0xFF;
	memset(&io[2][56], 0xFF, 8);
	memset(&io[3][48], 0xFF, 16);
	for (int v = 0; v < STREEBOG_ST_VEC; v++) {
		sph_gost512_init(&ctx);
		sph_gost512(&ctx, io[v], 64);
		sph_gost512_close(&ctx, ref[v]);
	}
	const bool kat_ok = (memcmp(ref[0], kat_streebog512_pat64, 64) == 0);

	// --- GPU hash vs the sph digests (first STREEBOG_ST_VEC vectors) ---
	bool gpu_ok = streebog_selftest_run(io);
	if (gpu_ok)
		gpu_ok = (memcmp(io, ref, sizeof(ref)) == 0);

	// --- negative test: one flipped input bit must change the digest ---
	for (int i = 0; i < 64; i++) io[0][i] = (uint8_t) i;
	io[0][0] ^= 0x01;
	bool neg_ok = streebog_selftest_run(io)
	           && (memcmp(io[0], ref[0], 64) != 0);

	free(io);

	passed = sph_ok && kat_ok && gpu_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "streebog device self-test FAILED (sph %d kat %d gpu %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "streebog device self-test passed");
	return selftest_gate(thr_id, "streebog", passed);
}