/*
 * Fused multi-stage kernel for the x-family: executes a run of
 * consecutive register-resident stages in one launch, keeping the 64-byte
 * state in registers instead of bouncing it through d_hash between stages.
 * Shared by every migrated x-family chain (x11/x16r/x16rv2/... ) via the
 * cuda_x_stages.h aggregate.
 *
 * Fusible stages are the ones whose device-library primitive needs neither
 * a shared table fill (shavite/echo/fugue/whirlpool), nor a quad-lane
 * interface (groestl), nor a multi-kernel pipeline (simd): blake, bmw, jh,
 * keccak, skein, luffa, cubehash, hamsi, shabal, sha512 — plus tiger192
 * (tiger'd chains), which gets a kernel variant with its 6KB shared table.
 *
 * The stage switch is uniform (every thread takes the same branch), so
 * there is no divergence; register footprint is the max over the fused
 * primitives, not the sum.
 *
 * Stage ids match the x-family enum Algo (see the cuda_x_stages.h
 * consumers); 16 = tiger192.
 */

#include "cuda/helper_alexis.cuh"
#include "cuda/vectors_alexis.cuh"
#include <miner.h>
#include <string.h>

#include "cuda_x_fused.h"

#include "cuda/blake512_device.cuh"
#include "cuda/keccak_device.cuh"
#include "cuda/jh512_device.cuh"
#include "cuda/bmw512_device.cuh"
#include "cuda/skein512_device.cuh"
#include "cuda/sha512_device.cuh"
#include "cuda/luffa512_device.cuh"
#include "cuda/shabal512_device.cuh"
#include "cuda/cubehash512_device.cuh"
#include "cuda/tiger192_device.cuh"
#include "cuda/shavite512_sp_device.cuh"
#include "cuda/hamsi512_device.cuh"  /* keep LAST: exports SBOX/ROUND_BIG macros */

#define X_FUSED_TIGER 16
#define TPB_FUSED 256 /* tiger192_load_shared needs exactly 256 threads */

/* the full 64-byte-stage id sequence of the current hash order (uploaded
 * once per order change); kernels take (start, len) into it. 30 = x16rv2:
 * 15 stages, each may be preceded by tiger */
__constant__ uint8_t c_fused_order[30];

__device__ __forceinline__
void x_fused_stage(const int id, uint64_t *const s, const uint64_t *sharedMem, const uint4 *__restrict__ TB)
{
	switch (id) {
	case 0: /* BLAKE */
		blake512_hash_64((uint2*)s);
		break;
	case 1: { /* BMW: msg[0..7] in, digest comes back in msg[8..15] */
		uint64_t __align__(16) msg[16];
		#pragma unroll
		for (int k = 0; k < 8; k++) msg[k] = s[k];
		bmw512_hash_64(msg);
		#pragma unroll
		for (int k = 0; k < 8; k++) s[k] = msg[8 + k];
		break;
	}
	case 3: /* JH */
		jh512_hash_64((uint32_t*)s);
		break;
	case 4: /* KECCAK */
		keccak512_hash_64((uint2*)s);
		break;
	case 5: /* SKEIN */
		skein512_hash_64((uint2*)s);
		break;
	case 6: /* LUFFA */
		luffa512_hash_64((uint32_t*)s);
		break;
	case 7: /* CUBEHASH */
		cubehash512_hash_64((uint32_t*)s);
		break;
	case 11: /* HAMSI */
		hamsi512_hash_64_tb((uint32_t*)s, TB);
		break;
	case 13: /* SHABAL */
		shabal512_hash_64((uint32_t*)s);
		break;
	case 15: /* SHA512 */
		sha512_hash_64(s);
		break;
	case X_FUSED_TIGER: { /* 24-byte digest, zero-padded to 64 */
		uint64_t buf[3];
		tiger192_hash_64(sharedMem, s, buf);
		s[0] = buf[0]; s[1] = buf[1]; s[2] = buf[2];
		s[3] = 0; s[4] = 0; s[5] = 0; s[6] = 0; s[7] = 0;
		break;
	}
	}
}

__global__ __launch_bounds__(TPB_FUSED, 2)
void x_fused_gpu_hash_64(const uint32_t threads, uint64_t *g_hash, const int start, const int len, const uint4 *__restrict__ TB)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t *pHash = &g_hash[thread << 3];
		uint64_t __align__(16) s[8];

		*(uint2x4*)&s[0] = __ldg4((uint2x4*)&pHash[0]);
		*(uint2x4*)&s[4] = __ldg4((uint2x4*)&pHash[4]);

		for (int i = 0; i < len; i++)
			x_fused_stage(c_fused_order[start + i], s, NULL, TB);

		*(uint2x4*)&pHash[0] = *(uint2x4*)&s[0];
		*(uint2x4*)&pHash[4] = *(uint2x4*)&s[4];
	}
}

__global__ __launch_bounds__(TPB_FUSED, 2)
void x_fused_gpu_hash_64_tiger(const uint32_t threads, uint64_t *g_hash, const int start, const int len, const uint4 *__restrict__ TB)
{
	__shared__ uint64_t sharedMem[768];

	tiger192_load_shared(sharedMem);
	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t *pHash = &g_hash[thread << 3];
		uint64_t __align__(16) s[8];

		*(uint2x4*)&s[0] = __ldg4((uint2x4*)&pHash[0]);
		*(uint2x4*)&s[4] = __ldg4((uint2x4*)&pHash[4]);

		for (int i = 0; i < len; i++)
			x_fused_stage(c_fused_order[start + i], s, sharedMem, TB);

		*(uint2x4*)&pHash[0] = *(uint2x4*)&s[0];
		*(uint2x4*)&pHash[4] = *(uint2x4*)&s[4];
	}
}

/* fixed-order chains: the stage ids are template constants, so each kernel
 * holds only its own stages (no switch over all cases, no local stack) */
template <int... IDS>
__global__ __launch_bounds__(TPB_FUSED, 2)
void x_fused_fixed_gpu_hash_64(const uint32_t threads, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t *pHash = &g_hash[thread << 3];
		uint64_t __align__(16) s[8];

		*(uint2x4*)&s[0] = __ldg4((uint2x4*)&pHash[0]);
		*(uint2x4*)&s[4] = __ldg4((uint2x4*)&pHash[4]);

		/* braced-list expansion runs the stages left to right */
		const int seq[] = { (x_fused_stage(IDS, s, NULL, NULL), 0)... };
		(void) seq;

		*(uint2x4*)&pHash[0] = *(uint2x4*)&s[0];
		*(uint2x4*)&pHash[4] = *(uint2x4*)&s[4];
	}
}

/* the same, followed by shavite (AES table in shared memory) in the same kernel */
template <int... IDS>
__global__ __launch_bounds__(TPB_FUSED, 3)
void x_fused_fixed_shavite_gpu_hash_64(const uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint32_t sharedMemory[256][32];

	aes_gpu_init256_s(sharedMemory); /* TPB_FUSED == 256 threads fill the 256 rows */

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint64_t *pHash = &g_hash[thread << 3];
	uint64_t __align__(16) s[8];

	if (thread < threads)
	{
		*(uint2x4*)&s[0] = __ldg4((uint2x4*)&pHash[0]);
		*(uint2x4*)&s[4] = __ldg4((uint2x4*)&pHash[4]);

		const int seq[] = { (x_fused_stage(IDS, s, NULL, NULL), 0)... };
		(void) seq;
	}
	__syncthreads(); /* outside the guard: the whole block reaches it */

	if (thread < threads)
	{
		shavite512_hash_64_s<false>(sharedMemory, (uint32_t*)s, NULL);

		*(uint2x4*)&pHash[0] = *(uint2x4*)&s[0];
		*(uint2x4*)&pHash[4] = *(uint2x4*)&s[4];
	}
}

/* sequence, stage ids (no hamsi/tiger: they need a table argument) */
#define X_FUSED_FIXED_LIST(X) \
	X(XF_JH_LUFFA_KECCAK_CUBE,       3, 6, 4, 7) \
	X(XF_JH_LUFFA_KECCAK,            3, 6, 4) \
	X(XF_SKEIN_JH_KECCAK,            5, 3, 4) \
	X(XF_SKEIN_BMW,                  5, 1) \
	X(XF_LUFFA_CUBE,                 6, 7) \
	X(XF_JH_KECCAK,                  3, 4) \
	X(XF_JH_CUBE,                    3, 7)

/* sequences ending in shavite: the ids before it */
#define X_FUSED_FIXED_SHAVITE_LIST(X) \
	X(XF_SKEIN_JH_KECCAK_LUFFA_CUBE_SHAVITE, 5, 3, 4, 6, 7) \
	X(XF_JH_KECCAK_SKEIN_LUFFA_CUBE_SHAVITE, 3, 4, 5, 6, 7) \
	X(XF_LUFFA_CUBE_SHAVITE,                 6, 7) \
	X(XF_CUBE_SHAVITE,                       7)

__host__
void x_fused_fixed_cpu_hash_64(uint32_t threads, int seq, uint32_t *d_hash)
{
	dim3 grid((threads + TPB_FUSED - 1) / TPB_FUSED);
	dim3 block(TPB_FUSED);

	#define XF_LAUNCH(name, ...) case name: \
		x_fused_fixed_gpu_hash_64<__VA_ARGS__> <<<grid, block>>> (threads, (uint64_t*)d_hash); break;
	#define XF_LAUNCH_SHAVITE(name, ...) case name: \
		x_fused_fixed_shavite_gpu_hash_64<__VA_ARGS__> <<<grid, block>>> (threads, (uint64_t*)d_hash); break;
	switch (seq) {
	X_FUSED_FIXED_LIST(XF_LAUNCH)
	X_FUSED_FIXED_SHAVITE_LIST(XF_LAUNCH_SHAVITE)
	}
	#undef XF_LAUNCH
	#undef XF_LAUNCH_SHAVITE
}

/* the stage ids of a fixed sequence (for the self-test), shavite as X_FUSED_SHAVITE; returns the count */
__host__
int x_fused_fixed_ids(int seq, uint8_t ids[8])
{
	#define XF_IDS(name, ...) case name: { \
		static const uint8_t v[] = { __VA_ARGS__ }; memcpy(ids, v, sizeof(v)); return (int) sizeof(v); }
	#define XF_IDS_SHAVITE(name, ...) case name: { \
		static const uint8_t v[] = { __VA_ARGS__, X_FUSED_SHAVITE }; memcpy(ids, v, sizeof(v)); return (int) sizeof(v); }
	switch (seq) {
	X_FUSED_FIXED_LIST(XF_IDS)
	X_FUSED_FIXED_SHAVITE_LIST(XF_IDS_SHAVITE)
	}
	#undef XF_IDS
	#undef XF_IDS_SHAVITE
	return 0;
}

/* hamsi byte table of the current device (algos/stages/cuda_hamsi512.cu) */
extern const uint4 *hamsi512_table(void);

__host__
void x_fused_setOrder(const uint8_t *ids, int count)
{
	cudaMemcpyToSymbol(c_fused_order, ids, count, 0, cudaMemcpyHostToDevice);
}

__host__
void x_fused_cpu_hash_64(int thr_id, uint32_t threads, int start, int len, int has_tiger, uint32_t *d_hash)
{
	dim3 grid((threads + TPB_FUSED - 1) / TPB_FUSED);
	dim3 block(TPB_FUSED);

	if (has_tiger)
		x_fused_gpu_hash_64_tiger <<<grid, block>>> (threads, (uint64_t*)d_hash, start, len, hamsi512_table());
	else
		x_fused_gpu_hash_64 <<<grid, block>>> (threads, (uint64_t*)d_hash, start, len, hamsi512_table());
}
