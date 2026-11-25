/*
* balloon algorithm - CUDA scan driver
*
*/
#include "miner.h"
#include <string.h>
#include <stdint.h>

#include <openssl/sha.h>

#include "balloon.h"
#include "cuda/helper.cuh"

// GPU entry points (cuda_balloon.cu)
extern void balloon_cpu_init(int thr_id);
extern void balloon_setBlock_80(int thr_id, void *pdata, const void *ptarget);
uint32_t balloon_cpu_hash(int thr_id, unsigned char *input, uint32_t threads,
	uint32_t max_nonce);
uint32_t balloon_wave_threads(int thr_id);
uint32_t balloon_get_candidates(int thr_id, uint32_t *out, uint32_t max_out);
extern "C" bool balloon_selftest(int thr_id);

int scanhash_balloon(int thr_id, struct work *work, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	uint32_t _ALIGN(128) endiandata[20];
	uint32_t _ALIGN(64) vhash[8];

	// benchmark only: loose enough to find candidates at a kH/s rate
	if (opt_benchmark)
		ptarget[7] = 0x00ffff;

	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce;

	// Nonces per launch (one thread each), a multiple of the 64-thread block.
	// Default: one wave = the threads the device keeps resident; a partial wave
	// leaves SMs idle, extra waves only lengthen the launch. -i overrides it.
	// A launch cannot be interrupted and lasts batch/hashrate seconds: that is the
	// job-switch delay, and past ~2 s Windows resets a display-attached GPU. The
	// first launch is timed below.
	static THREAD volatile bool init = false;
	static THREAD uint32_t batch = 0;
	if (!init)
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		if (opt_cudaschedule == -1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		balloon_cpu_init(thr_id);
		// Consensus gate; fails closed inside balloon_selftest().
		balloon_selftest(thr_id);
		uint32_t wave = balloon_wave_threads(thr_id);
		if (!wave) wave = 1U << 14;
		batch = cuda_default_throughput(thr_id, wave) & ~0x3fU;
		if (batch < 64) batch = 64;
		gpulog(LOG_INFO, thr_id, "intensity %.2f, %u nonces/launch",
			throughput2intensity(batch), batch);
		init = true;
	}

	for (int i = 0; i < 19; i++) {
		be32enc(&endiandata[i], pdata[i]);
	}

	// The pre-buffer depends only on the first SALT_LEN header bytes and is
	// nonce-independent, so it is refreshed once per work unit and then cached
	// on the GPU across the nonce batches below.
	reset_host_prebuf(thr_id);
	balloon_reset();

	// Upload the share target.
	balloon_setBlock_80(thr_id, endiandata, ptarget);

	// See the batch-size note above.
	static THREAD bool launch_timed = false;

	do {
		be32enc(&endiandata[19], n);

		struct timeval tv_a, tv_b, tv_d;
		const bool time_it = (!launch_timed || opt_debug);
		if (time_it) gettimeofday(&tv_a, NULL);

		uint32_t winning_nonce = balloon_cpu_hash(thr_id, (unsigned char *)endiandata,
			batch, max_nonce);

		if (time_it) {
			gettimeofday(&tv_b, NULL);
			timeval_subtract(&tv_d, &tv_b, &tv_a);
			const double secs = (double)tv_d.tv_sec + (double)tv_d.tv_usec / 1e6;
			const bool first = !launch_timed;
			launch_timed = true;
			if (first && secs > 1.5) {
				gpulog(LOG_WARNING, thr_id,
					"a single launch takes %.2fs at intensity %.2f; it cannot be interrupted, "
					"so that is both the driver-reset window and the job-switch delay. "
					"Lower -i -- throughput does not improve above the default.",
					secs, throughput2intensity(batch));
			} else if (opt_debug) {
				gpulog(LOG_DEBUG, thr_id, "launch %.3fs at intensity %.2f (%u nonces)",
					secs, throughput2intensity(batch), batch);
			}
		}

		if (work_restart[thr_id].restart)
			break;

		// CPU re-hash is authoritative: a kernel fault costs a local reject, not a bad share
		if (winning_nonce != UINT32_MAX) {
			// All candidates of the launch, lowest first; resume past the batch.
			// If the kernel found more than it keeps, only the lowest is certain:
			// submit it alone and resume just past it.
			uint32_t cand[MAX_NONCES];
			const uint32_t found = balloon_get_candidates(thr_id, cand, MAX_NONCES);
			const bool complete = (found <= MAX_NONCES);
			const uint32_t ncand = complete ? found : 1;
			if (!complete) cand[0] = winning_nonce;
			uint32_t resume = complete ? n + batch : winning_nonce + 1;
			if (resume > max_nonce || resume < first_nonce)
				resume = max_nonce;

			work->valid_nonces = 0;
			for (uint32_t k = 0; k < ncand; k++) {
				be32enc(&endiandata[19], cand[k]);
				balloon_128_orig((unsigned char *)endiandata, (unsigned char *)vhash);
				if (vhash[7] > Htarg || !fulltest(vhash, ptarget)) {
					gpu_increment_reject(thr_id);
					if (!opt_quiet) gpulog(LOG_WARNING, thr_id,
						"result for %08x does not validate on CPU!", cand[k]);
					continue;
				}
				bn_set_target_ratio(work, vhash, work->valid_nonces);
				work->nonces[work->valid_nonces++] = cand[k];
			}
			if (work->valid_nonces) {
				*hashes_done = resume - first_nonce;
				pdata[19] = resume;
				return work->valid_nonces;
			}
			if (!complete) {
				n = resume;
				continue;
			}
		}

		n += batch;
	} while (n < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce;
	pdata[19] = n;

	return 0;
}
