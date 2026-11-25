/**
 * X15 algorithm (fixed 15-stage chain) — x14 + Whirlpool
 *
 * blake - bmw - groestl - skein - jh - keccak - luffa - cubehash -
 * shavite - simd - echo - hamsi - fugue - shabal - whirlpool
 *
 * Migrated to the shared x-family machinery (docs/coding-guideline.md §2/§3):
 * the 64-byte stages call the bare <prim>512 device-launcher names through the
 * cuda_x_stages.h bridge, and the consecutive fusible run skein->jh->keccak->
 * luffa->cubehash (identical to x11/x13's) is executed by the shared register-
 * resident fused kernel (cuda_x_fused.cu). Order is fixed, so the fused
 * kernel is compiled for this sequence. Echo/hamsi/fugue/shabal are mid-chain here
 * (whirlpool follows), so echo uses the plain 64-byte launcher; whirlpool is
 * the terminal and the best nonce is found by the shared cuda_check_hash pass.
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
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
}

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

#include <stdio.h>
#include <memory.h>

// Memory for the hash functions
static uint32_t *d_hash[MAX_GPUS] = { 0 };

// X15 CPU Hash function
extern "C" void x15hash(void *output, const void *input)
{
	sph_blake512_context     ctx_blake;
	sph_bmw512_context       ctx_bmw;
	sph_groestl512_context   ctx_groestl;
	sph_jh512_context        ctx_jh;
	sph_keccak512_context    ctx_keccak;
	sph_skein512_context     ctx_skein;
	sph_luffa512_context     ctx_luffa;
	sph_cubehash512_context  ctx_cubehash;
	sph_shavite512_context   ctx_shavite;
	sph_simd512_context      ctx_simd;
	sph_echo512_context      ctx_echo;
	sph_hamsi512_context     ctx_hamsi;
	sph_fugue512_context     ctx_fugue;
	sph_shabal512_context    ctx_shabal;
	sph_whirlpool_context    ctx_whirlpool;

	unsigned char hash[128]; // uint32_t hashA[16], hashB[16];
	#define hashB hash+64

	memset(hash, 0, sizeof hash);

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, hash, 64);
	sph_bmw512_close(&ctx_bmw, hashB);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, hashB, 64);
	sph_groestl512_close(&ctx_groestl, hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, 64);
	sph_skein512_close(&ctx_skein, hashB);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, hashB, 64);
	sph_jh512_close(&ctx_jh, hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, hash, 64);
	sph_keccak512_close(&ctx_keccak, hashB);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, hashB, 64);
	sph_luffa512_close(&ctx_luffa, hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, hash, 64);
	sph_cubehash512_close(&ctx_cubehash, hashB);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, hashB, 64);
	sph_shavite512_close(&ctx_shavite, hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, hash, 64);
	sph_simd512_close(&ctx_simd, hashB);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, hashB, 64);
	sph_echo512_close(&ctx_echo, hash);

	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, hash, 64);
	sph_hamsi512_close(&ctx_hamsi, hashB);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, hashB, 64);
	sph_fugue512_close(&ctx_fugue, hash);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash, 64);
	sph_shabal512_close(&ctx_shabal, hashB);

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, hashB, 64);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	memcpy(output, hash, 32);
	#undef hashB
}

//#define _DEBUG
#define _DEBUG_PREFIX "x15"
#include "cuda/debug.cuh"

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_x15(int thr_id,  struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	uint32_t endiandata[20];

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 19); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x00FF;

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
		jh512_cpu_init(thr_id, throughput);
		keccak512_cpu_init(thr_id, throughput);
		luffa512_cpu_init(thr_id, throughput); // 64
		shavite512_cpu_init(thr_id, throughput);
		if (simd512_cpu_init(thr_id, throughput) != 0) {
			return 0;
		}
		hamsi512_cpu_init(thr_id, throughput);
		fugue512_cpu_init(thr_id, throughput);
		shabal512_cpu_init(thr_id, throughput);
		whirlpool512_cpu_init(thr_id, throughput, 0);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		x_fused_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;
		blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		TRACE("blake  :");
		bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("bmw    :");
		groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("groestl:");
		/* fused: skein - jh - keccak - luffa - cubehash - shavite (one kernel) */
		x_fused_fixed_cpu_hash_64(throughput, XF_SKEIN_JH_KECCAK_LUFFA_CUBE_SHAVITE, d_hash[thr_id]); order += 6;
		TRACE("shavite:");
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("simd   :");
		if (use_compat_kernels[thr_id])
			echo512_cpu_hash_64_compat(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		else {
			echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		}
		TRACE("echo   :");
		hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("hamsi  :");
		fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("fugue  :");
		shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("shabal :");
		whirlpool512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("whirl  :");

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			/* check now with the CPU to confirm */
			be32enc(&endiandata[19], work->nonces[0]);
			x15hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					x15hash(vhash, endiandata);
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
extern "C" void free_x15(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);

	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);
	simd512_cpu_free(thr_id);
	fugue512_cpu_free(thr_id);
	whirlpool512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
