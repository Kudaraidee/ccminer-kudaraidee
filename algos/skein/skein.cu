/**
 * SKEIN512 80 + SHA256 64
 * by tpruvot@github - 2015
 */

#include "sph/sph_skein.h"

#include "miner.h"
#include "cuda/helper.cuh"

#include <openssl/sha.h>

extern void skeincoin_cpu_init(int thr_id);
extern void skeincoin_free(int thr_id);
extern void skeincoin_setBlock_80(int thr_id, void *pdata);
extern uint32_t skeincoin_hash_sm5(int thr_id, uint32_t threads, uint32_t startNounce, int swap, uint64_t target64, uint32_t *secNonce);

extern "C" void skeincoinhash(void *output, const void *input)
{
	sph_skein512_context ctx_skein;
	SHA256_CTX sha256;

	uint32_t hash[16];

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, (unsigned char *)hash, 64);
	SHA256_Final((unsigned char *)hash, &sha256);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_skeincoin(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 20);
	if (init[thr_id]) throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x03;

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

		skeincoin_cpu_init(thr_id);

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	skeincoin_setBlock_80(thr_id, (void*)endiandata);
	const uint64_t target64 = ((uint64_t*)ptarget)[3];

	do {
		// Hash with CUDA
		*hashes_done = pdata[19] - first_nonce + throughput;

		/* cuda_skein.cu — the fused kernel reports up to two candidates per
		 * batch, so nonces[1] must be armed before every launch (it is only
		 * written when a second one is found). */
		work->nonces[1] = UINT32_MAX;
		work->nonces[0] = skeincoin_hash_sm5(thr_id, throughput, pdata[19], 1, target64, &work->nonces[1]);

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];

			endiandata[19] = swab32(work->nonces[0]);
			skeincoinhash(vhash, endiandata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				pdata[19] = work->nonces[0] + 1; // cursor for next scan

				if (work->nonces[1] != UINT32_MAX) {
					endiandata[19] = swab32(work->nonces[1]);
					skeincoinhash(vhash, endiandata);
					if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
						pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
					}
				}
				return work->valid_nonces;
			}
			 else if (vhash[7] > ptarget[7]) {
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
extern "C" void free_skeincoin(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	skeincoin_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
