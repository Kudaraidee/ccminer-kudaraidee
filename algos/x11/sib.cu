/**
 * SibCoin (sib) algorithm — fixed 11-stage chain (x11 with streebog/gost)
 *
 * blake - bmw - groestl - skein - jh - keccak - streebog - luffa - cubehash -
 * shavite - simd - echo
 *
 * Migrated to the shared x-family machinery (docs/coding-guideline.md §2/§3):
 * the 64-byte stages call the bare <prim>512 device-launcher names through the
 * cuda_x_stages.h bridge. The two consecutive fusible runs are executed by the
 * shared register-resident fused kernel (cuda_x_fused.cu), one kernel compiled per run:
 *   run A = skein->jh->keccak
 *   run B = luffa->cubehash
 * groestl and streebog are the quad/table boundaries between/around them;
 * shavite/simd/echo are the trailing boundary stages.
 */

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_streebog.h"
}

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

/* consolidated streebog stage (algos/stages/cuda_streebog.cu) */
extern void streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern bool streebog_device_selftest(int thr_id);

#include <stdio.h>
#include <memory.h>

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

// Sibcoin CPU Hash
extern "C" void sibhash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_skein512_context ctx_skein;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_gost512_context ctx_gost;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;

	sph_blake512_init(&ctx_blake);
	sph_blake512 (&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, (void*) hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*) hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512 (&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512 (&ctx_jh, (const void*) hash, 64);
	sph_jh512_close(&ctx_jh, (void*) hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*) hash, 64);
	sph_gost512_close(&ctx_gost, (void*) hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, (const void*) hash, 64);
	sph_luffa512_close (&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512 (&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512 (&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "sib"
#include "cuda/debug.cuh"

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_sib(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 19 : 18; // 2^18 = 262144 cuda threads
	if (device_sm[dev_id] >= 600) intensity = 20;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0xf;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id])
			echo512_cpu_init_compat(thr_id, throughput);

		blake512_cpu_init(thr_id, throughput);
		bmw512_cpu_init(thr_id, throughput);
		groestl512_cpu_init(thr_id, throughput);
		skein512_cpu_init(thr_id, throughput);
		keccak512_cpu_init(thr_id, throughput);
		jh512_cpu_init(thr_id, throughput);
		luffa512_cpu_init(thr_id, throughput); // 64
		shavite512_cpu_init(thr_id, throughput);
		if (simd512_cpu_init(thr_id, throughput) != 0) {
			return -1;
		}
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

		cuda_check_cpu_init(thr_id, throughput);

		streebog_device_selftest(thr_id);
		x_fused_device_selftest(thr_id);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake512_cpu_setBlock_80(thr_id, endiandata);
	if (use_compat_kernels[thr_id])
		cuda_check_cpu_setTarget(ptarget);
	else
		cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));

	do {
		int order = 0;

		// Hash with CUDA
		blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		TRACE("blake  :");
		bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("bmw    :");
		groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("groestl:");
		/* fused run A: skein - jh - keccak (register-resident) */
		x_fused_fixed_cpu_hash_64(throughput, XF_SKEIN_JH_KECCAK, d_hash[thr_id]); order += 3;
		TRACE("fusedA :");
		streebog_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		TRACE("gost   :");
		/* fused run B: luffa - cubehash - shavite (one kernel) */
		x_fused_fixed_cpu_hash_64(throughput, XF_LUFFA_CUBE_SHAVITE, d_hash[thr_id]); order += 3;
		TRACE("shavite:");
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("simd   :");
		if (use_compat_kernels[thr_id]) {
			echo512_cpu_hash_64_compat(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
			work->nonces[1] = UINT32_MAX;
		} else {
			/* echo + on-device target compare, 2 nonces via atomicExch chain */
			echo512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], AS_U64(&ptarget[6]));
			cudaMemcpy(&work->nonces[0], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		}
		TRACE("echo => ");

		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t _ALIGN(64) vhash[8];
			if (!use_compat_kernels[thr_id]) work->nonces[0] += startNounce;
			be32enc(&endiandata[19], work->nonces[0]);
			sibhash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (work->nonces[1] != UINT32_MAX) {
					work->nonces[1] += startNounce;
					be32enc(&endiandata[19], work->nonces[1]);
					sibhash(vhash, endiandata);
					work_check_second_nonce(thr_id, work, vhash);
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_sib(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);
	simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
