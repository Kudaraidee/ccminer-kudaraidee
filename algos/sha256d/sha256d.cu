/**
 * SHA256d
 * by tpruvot@github - 2017
 */

#include <miner.h>
#include "cuda/helper.cuh"
#include <openssl/sha.h>

#include "cuda/sha256_device.cuh"

static bool init[MAX_GPUS] = { 0 };
extern void sha256d_cpu_init(int thr_id);
extern void sha256d_free(int thr_id);
extern void sha256d_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, const uint32_t* const ms, uint32_t merkle, uint32_t time, uint32_t compacttarget, uint32_t* resNonces);

extern void sha256d_midstate(const uint32_t* data, uint32_t* midstate);

// CPU Check
extern "C" void sha256d_hash(void* output, const void* input)
{
	unsigned char _ALIGN(64) hash[64];
	SHA256_CTX sha256;

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, (unsigned char*)input, 80);
	SHA256_Final(hash, &sha256);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, hash, 32);
	SHA256_Final((unsigned char*)output, &sha256);
}

void sha256d_opt_hash(uint32_t* output, const uint32_t* data, uint32_t nonce, const uint32_t* midstate)
{
	uint32_t in[16], st[8];

	for (int i = 0; i < 16; i++) in[i] = data[i + 16];
	in[3] = nonce;
	for (int i = 0; i < 8; i++) st[i] = midstate[i];
	sha256_transform_full(in, st, h_sha256_K);

	for (int i = 0; i < 8; i++) in[i] = st[i];
	in[8] = 0x80000000U;
	for (int i = 9; i < 15; i++) in[i] = 0U;
	in[15] = 0x100U;
	for (int i = 0; i < 8; i++) st[i] = h_sha256_H[i];
	sha256_transform_full(in, st, h_sha256_K);

	for (int i = 0; i < 8; i++) be32enc(&output[i], st[i]);
}

extern "C" int scanhash_sha256d(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 25);
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

		sha256d_cpu_init(thr_id);

		init[thr_id] = true;
	}

	uint32_t ms[8];
	sha256d_midstate(pdata, ms);

	do {
		// Hash with CUDA
		*hashes_done = pdata[19] - first_nonce + throughput;

		sha256d_hash_80(thr_id, throughput, pdata[19], ms, pdata[16], pdata[17], pdata[18], work->nonces);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];

			sha256d_opt_hash(vhash, pdata, work->nonces[0], ms);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					sha256d_opt_hash(vhash, pdata, work->nonces[1], ms);
					if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
						work->valid_nonces++;
						bn_set_target_ratio(work, vhash, 1);
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1;
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet) {
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				}
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
extern "C" void free_sha256d(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	sha256d_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
