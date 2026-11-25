/*
 * Keccak 256
 *
 * (sm_30 "compat" kernel selector removed 2026-07  - sm_61 arch floor)
 */

extern "C"
{
#include "sph/sph_keccak.h"

#include "miner.h"
}

#include "cuda/helper.cuh"

extern void keccak256_cpu_init(int thr_id);
extern void keccak256_cpu_free(int thr_id);
extern void keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t* resNonces, const uint2 highTarget);
extern void keccak256_setBlock_80(uint64_t *endiandata);
extern void keccak256_setOutput(int thr_id);

// CPU Hash
extern "C" void keccak256_hash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_keccak_context ctx_keccak;

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256 (&ctx_keccak, input, 80);
	sph_keccak256_close(&ctx_keccak, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_keccak256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	/* Default launch size. Bigger launches amortise the fixed per-launch cost,
	 * but a batch outliving the job is discarded on every job change, so this
	 * targets the start of the plateau. Reference is 2^25 threads on a 28-SM
	 * card (both cards measured here are 28-SM), scaled by SM count so the
	 * batch stays roughly constant on other sizes. Named cards override it. */
	uint32_t defthreads = 1U << 25;
	if (device_mpcount[dev_id] > 0)
		defthreads = (uint32_t) (((uint64_t) defthreads * device_mpcount[dev_id]) / 28);
	/* per-card values, measured or inherited */
	if (strstr(device_name[dev_id], "GTX 1070")) defthreads = 1U << 25;
	if (strstr(device_name[dev_id], "GTX 1080")) defthreads = 1U << 26;
	if (strstr(device_name[dev_id], "RTX 30"))   defthreads = 1U << 25;
	uint32_t throughput = cuda_default_throughput(thr_id, defthreads);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000f;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		cuda_get_arch(thr_id);
		keccak256_cpu_init(thr_id);

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

	const uint2 highTarget = make_uint2(ptarget[6], ptarget[7]);
	keccak256_setBlock_80((uint64_t*)endiandata);
	keccak256_setOutput(thr_id);

	do {
		*hashes_done = pdata[19] - first_nonce + throughput;

		keccak256_cpu_hash_80(thr_id, throughput, pdata[19], work->nonces, highTarget);

		if (work->nonces[0] != UINT32_MAX && bench_algo < 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
			keccak256_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					keccak256_hash(vhash, endiandata);
					if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						work->valid_nonces++;
						bn_set_target_ratio(work, vhash, 1);
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1;
				}
				return work->valid_nonces;
			}
			else {
				/* Covers vhash[7] <= Htarg failing fulltest: without the bare else
				 * that outcome matched no branch, leaving a stale nonce in an
				 * un-armed buffer to be re-read every iteration. */
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				keccak256_setOutput(thr_id);
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
extern "C" void free_keccak256(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	keccak256_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
