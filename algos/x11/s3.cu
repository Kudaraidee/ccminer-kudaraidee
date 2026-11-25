/**
 * S3 Hash (Also called Triple S - Used by 1Coin)
 *
 * shavite(80) - simd - skein
 *
 * Migrated to the shared x-family machinery (docs/coding-guideline.md §2/§3):
 * the stages call the bare device-launcher names through the cuda_x_stages.h
 * bridge (shavite512_* is the optimised sp 80-byte launcher, same as the x16
 * family). No fusible run of >= 2 consecutive register-resident stages exists
 * (shavite/simd are boundaries, skein is isolated), so there is no fused kernel.
 */

extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
}

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

#include <stdint.h>

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

/* CPU HASH */
extern "C" void s3hash(void *output, const void *input)
{
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_skein512_context ctx_skein;

	unsigned char hash[64];

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, input, 80);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	memcpy(output, hash, 32);
}

#ifdef _DEBUG
#define TRACE(algo) { \
	if (max_nonce == 1 && pdata[19] <= 1) { \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 32); \
		cudaMemcpy(debugbuf, d_hash[thr_id], 32, cudaMemcpyDeviceToHost); \
		printf("S3 %s %08x %08x %08x %08x...%08x\n", algo, swab32(debugbuf[0]), swab32(debugbuf[1]), \
			swab32(debugbuf[2]), swab32(debugbuf[3]), swab32(debugbuf[7])); \
		cudaFreeHost(debugbuf); \
	} \
}
#else
#define TRACE(algo) {}
#endif

static bool init[MAX_GPUS] = { 0 };

/* Main S3 entry point */
extern "C" int scanhash_s3(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = 20; // 256*256*8*2;
#ifdef WIN32
	// reduce by one the intensity on windows
	intensity--;
#endif
	uint32_t throughput =  cuda_default_throughput(thr_id, 1 << intensity);
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0xF;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

		shavite512_cpu_init(thr_id, throughput);
		simd512_cpu_init(thr_id, throughput);
		skein512_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	shavite512_setBlock_80((void*)endiandata);
	cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));

	do {
		int order = 0;

		shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		TRACE("shavite:");
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("simd   :");
		/* skein terminal + on-device target compare, 2 nonces via atomicExch chain */
		skein512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], AS_U64(&ptarget[6]), d_resNonce[thr_id]);
		cudaMemcpy(&work->nonces[0], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		TRACE("skein  :");

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			work->nonces[0] += startNounce;
			be32enc(&endiandata[19], work->nonces[0]);
			s3hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					work->nonces[1] += startNounce;
					be32enc(&endiandata[19], work->nonces[1]);
					s3hash(vhash, endiandata);
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
extern "C" void free_s3(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	simd512_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
