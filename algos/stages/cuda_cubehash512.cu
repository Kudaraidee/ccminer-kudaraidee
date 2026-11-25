/*
 * cubehash512 x11 kernel — thin wrapper.
 *
 * The CubeHash-512 device implementation (cubehash512_rrounds +
 * cubehash512_hash_64) lives in cuda/cubehash512_device.cuh
 * (docs/coding-guideline.md §3): this TU only holds the standalone 64-byte
 * launcher over it (no device-code duplication).
 *
 * Provenance: Tanguy Pruvot / Provos Alexis 2016, sp 2018/2019.
 *
 * NOTE: the former fused `x11_cubehashShavite512` kernel (and its
 * `x11_cubehash_shavite512_cpu_hash_64` launcher) was removed here — it had no
 * callers anywhere in the tree, and it carried a full private copy of the
 * SHAvite AES machinery (d_AES0 table, aes_round, AES_ROUND_NOKEY,
 * KEY_EXPAND_ELT, aes_gpu_init, round_3_7_11/4_8_12) duplicating
 * cuda/shavite512_device.cuh and cuda/aes_sp_device.cuh. (A cubehash+shavite
 * fusion for the x-family, if ever wanted, belongs in the shared fused
 * pipeline, not a branded private copy.)
 */

#include "cuda/helper_alexis.cuh"
#include "cuda/vectors_alexis.cuh"

#include "cuda/cubehash512_device.cuh"

/* Block size per arch: 128 on newer cards (more resident blocks than the donor's
 * 1024), 512 on Pascal, where 128 is clearly slower. */
#define TPB 128
#define TPB_PASCAL 512

extern short device_map[];
extern long  device_sm[];

__global__
void cubehash512_gpu_hash_64(uint32_t threads, uint64_t *g_hash){

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		uint32_t *Hash = (uint32_t*)&g_hash[8 * thread];
		cubehash512_hash_64(Hash);
	}
}

__host__
extern bool cubehash512_device_selftest(int thr_id);

void cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

	/* guideline §7 layer-1 self-test (defined in cuda/xfamily_selftest.cu);
	 * no init fn exists for this stage, so it runs once from the launcher */
	cubehash512_device_selftest(thr_id);

	const uint32_t tpb = (device_sm[device_map[thr_id]] < 700) ? TPB_PASCAL : TPB;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + tpb-1)/tpb);
	dim3 block(tpb);

	cubehash512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}
