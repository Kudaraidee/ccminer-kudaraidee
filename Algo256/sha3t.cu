/*
 * SHA3-256t — triple NIST SHA3-256 GPU miner
 *
 * Author: Pkules (https://github.com/Pkules)
 *
 * CPU reference hash and scanhash wrapper.
 * GPU kernel: cuda_sha3t.cu
 */

extern "C" {
#include "sph/sha3.h"
#include "miner.h"
}

#undef ROTL64
#include "cuda_helper.h"

extern void sha3t_cpu_init(int thr_id);
extern void sha3t_cpu_free(int thr_id);
extern void sha3t_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce,
                               uint32_t *resNonces, const uint2 highTarget);
extern void sha3t_setBlock_80(uint64_t *endiandata);
extern void sha3t_setOutput(int thr_id);

/* CPU reference: triple SHA3-256 over 80-byte input */
extern "C" void sha3t256_hash(void *state, const void *input)
{
	sha3_ctx_t ctx;
	uint8_t buf1[32], buf2[32];

	sha3_init(&ctx, 32);
	sha3_update(&ctx, input, 80);
	sha3_final(buf1, &ctx);

	sha3_init(&ctx, 32);
	sha3_update(&ctx, buf1, 32);
	sha3_final(buf2, &ctx);

	sha3_init(&ctx, 32);
	sha3_update(&ctx, buf2, 32);
	sha3_final(state, &ctx);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_sha3t(int thr_id, struct work *work,
                               uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata    = work->data;
	uint32_t *ptarget  = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id   = device_map[thr_id];
	uint32_t intensity = 23;

	if (strstr(device_name[dev_id], "RTX 30")) intensity = 25;
	if (strstr(device_name[dev_id], "RTX 40")) intensity = 26;
	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 25;
	if (strstr(device_name[dev_id], "GTX 1070")) intensity = 24;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x0000ffff;

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		cuda_get_arch(thr_id);
		sha3t_cpu_init(thr_id);
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
		       throughput2intensity(throughput), throughput);
		init[thr_id] = true;
	}

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	const uint2 highTarget = make_uint2(ptarget[6], ptarget[7]);
	sha3t_setBlock_80((uint64_t*) endiandata);
	sha3t_setOutput(thr_id);

	do {
		*hashes_done = pdata[19] - first_nonce + throughput;

		sha3t_cpu_hash_80(thr_id, throughput, pdata[19], work->nonces, highTarget);

		if (work->nonces[0] != UINT32_MAX && bench_algo < 0) {
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
			sha3t256_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					sha3t256_hash(vhash, endiandata);
					if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						work->valid_nonces++;
						bn_set_target_ratio(work, vhash, 1);
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1;
				}
				return work->valid_nonces;
			} else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id,
					       "result for %08x does not validate on CPU!",
					       work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				sha3t_setOutput(thr_id);
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

extern "C" void free_sha3t(int thr_id)
{
	if (!init[thr_id]) return;
	cudaThreadSynchronize();
	sha3t_cpu_free(thr_id);
	cudaDeviceSynchronize();
	init[thr_id] = false;
}
