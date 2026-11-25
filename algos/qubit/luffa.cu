/*
 * luffa 80 algo (Introduced by Doomcoin)
 *
 * The single luffa-80 stage is also the terminal: luffa512_cpu_hash_80_final
 * screens the top 64 bits on-device and reports the two lowest passing nonces
 * into d_resNonce, skipping the third round (digest words 8..15), the 64-byte
 * hash store and the cuda_check_hash passes.
 */
extern "C" {
#include "sph/sph_luffa.h"
}

#include "miner.h"

#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

static uint32_t *d_resNonce[MAX_GPUS];

extern "C" void luffa_hash(void *state, const void *input)
{
	uint8_t _ALIGN(64) hash[64];

	sph_luffa512_context ctx_luffa;

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, input, 80);
	sph_luffa512_close(&ctx_luffa, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_luffa(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	/* the terminal kernel is short; smaller batches leave the GPU idle between them */
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 24);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x0000f;

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

		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

		luffa512_80_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	luffa512_setBlock_80((void*)endiandata);
	cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));

	do {
		/* luffa + on-device target compare, 2 nonces via the two-lowest report */
		luffa512_cpu_hash_80_final(thr_id, throughput, pdata[19], d_resNonce[thr_id], AS_U64(&ptarget[6]));
		cudaMemcpy(&work->nonces[0], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t _ALIGN(64) vhash[8];
			work->nonces[0] += startNounce;
			be32enc(&endiandata[19], work->nonces[0]);
			luffa_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					work->nonces[1] += startNounce;
					be32enc(&endiandata[19], work->nonces[1]);
					luffa_hash(vhash, endiandata);
					if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else {
				/* a screen pass failing fulltest is no GPU error; resume past it */
				if (vhash[7] > Htarg) {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				}
				cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_luffa(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;
	cudaDeviceSynchronize();
}
