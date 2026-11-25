/*
 * deepcoin algorithm  (luffa80 - cubehash - echo)
 *
 * The fixed echo terminal is folded with the on-device target compare
 * (echo512_cpu_hash_64_final: 2 nonces via an atomicExch chain into
 * d_resNonce, eliding the echo d_hash store + the cuda_check_hash/suppl
 * passes). The compat path (arch < 500, below the sm_61 build floor) keeps
 * the tpruvot echo + cuda_check_hash.
 */
extern "C" {
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}

#include "miner.h"

#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

extern "C" void deephash(void *state, const void *input)
{
	uint8_t _ALIGN(64) hash[64];

	// luffa-80 cubehash-64 echo-64
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_echo512_context ctx_echo;

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, input, 80);
	sph_luffa512_close(&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_deep(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const int dev_id = device_map[thr_id];
	const uint32_t first_nonce = pdata[19];
	/* smaller batches leave the GPU idle between them (Pascal and Ampere) */
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 22);
	if (init[thr_id]) throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000f;

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

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id])
			echo512_cpu_init_compat(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);
		luffa512_80_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	luffa512_setBlock_80((void*)endiandata);
	if (use_compat_kernels[thr_id])
		cuda_check_cpu_setTarget(ptarget);
	else
		cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));

	do {
		int order = 0;

		luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		if (use_compat_kernels[thr_id]) {
			echo512_cpu_hash_64_compat(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
			work->nonces[1] = UINT32_MAX;
		} else {
			/* echo + on-device target compare, 2 nonces via atomicExch chain */
			echo512_cpu_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id], AS_U64(&ptarget[6]));
			cudaMemcpy(&work->nonces[0], d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t _ALIGN(64) vhash[8];
			if (!use_compat_kernels[thr_id]) work->nonces[0] += startNounce;
			be32enc(&endiandata[19], work->nonces[0]);
			deephash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					work->nonces[1] += startNounce;
					be32enc(&endiandata[19], work->nonces[1]);
					deephash(vhash, endiandata);
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

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}

// cleanup
extern "C" void free_deep(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
