#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "sph/sph_fugue.h"

#include "miner.h"

#include "cuda_fugue256.h"

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

void fugue256_hash(unsigned char* output, const unsigned char* input, int len)
{
	sph_fugue256_context ctx;

	sph_fugue256_init(&ctx);
	sph_fugue256(&ctx, input, len);
	sph_fugue256_close(&ctx, (void *)output);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_fugue256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	/* 2^25 (~35 ms launches), tuned on a GTX 1080 Ti (sm_61) and an RTX 3060 (sm_86) */
	int intensity = (device_sm[device_map[thr_id]] > 500) ? 25 : 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0xf;

	// init
	if(!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		fugue256_cpu_init(thr_id);
		/* fail-closed; leaves its own header in the device symbols, setBlock below replaces it */
		fugue256_device_selftest(thr_id);
		init[thr_id] = true;
	}

	// Endian
	for (int kk=0; kk < 20; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	fugue256_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

	do {
		// GPU: the two lowest candidates of the launch
		work->nonces[0] = fugue256_cpu_hash(thr_id, throughput, pdata[19], &work->nonces[1]);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX && bench_algo < 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			/* everything up to the higher of the two has been screened */
			const uint32_t scanned = (work->nonces[1] != UINT32_MAX) ? work->nonces[1] : work->nonces[0];
			work->valid_nonces = 0;

			endiandata[19] = SWAP32(work->nonces[0]);
			fugue256_hash((unsigned char*)vhash, (const unsigned char*)endiandata, 80);
			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work_set_target_ratio(work, vhash);
				work->valid_nonces++;
			} else {
				gpu_increment_reject(thr_id);
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
			}

			if (work->nonces[1] != UINT32_MAX) {
				endiandata[19] = SWAP32(work->nonces[1]);
				fugue256_hash((unsigned char*)vhash, (const unsigned char*)endiandata, 80);
				if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
					if (!work->valid_nonces) {
						/* the first failed: this one is the share (slot 0) */
						work->nonces[0] = work->nonces[1];
						work_set_target_ratio(work, vhash);
					} else if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]) {
						work->sharediff[1] = work->sharediff[0];
						work->shareratio[1] = work->shareratio[0];
						const uint32_t n0 = work->nonces[0];
						work->nonces[0] = work->nonces[1];
						work->nonces[1] = n0;
						work_set_target_ratio(work, vhash);
					} else {
						bn_set_target_ratio(work, vhash, 1);
					}
					work->valid_nonces++;
				} else {
					gpu_increment_reject(thr_id);
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[1]);
				}
			}

			if (work->valid_nonces) {
				pdata[19] = scanned + 1; // next scan start
				return work->valid_nonces;
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
void free_fugue256(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	fugue256_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
