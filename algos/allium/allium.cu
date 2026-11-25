extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_skein.h"
#include "sph/sph_groestl.h"
#include "algos/lyra2/Lyra2.h"
}

#include <miner.h>
#include "cuda/helper.cuh"

static uint64_t* d_hash[MAX_GPUS];
static uint64_t* d_matrix[MAX_GPUS];

extern void blake256_cpu_init(int thr_id, uint32_t threads);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);

extern void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);

extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);

extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void skein256_cpu_init(int thr_id, uint32_t threads);

extern void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint64_t *d_outputHash);
extern bool lyra2_device_selftest(int thr_id);

extern void groestl256_cpu_init(int thr_id, uint32_t threads);
extern void groestl256_cpu_free(int thr_id);
extern void groestl256_setTarget(const void *ptarget);
extern uint32_t groestl256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order);
extern uint32_t groestl256_getSecNonce(int thr_id, int num);


extern "C" void allium_hash(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context     ctx_blake;
	sph_keccak256_context    ctx_keccak;
	sph_cubehash256_context  ctx_cube;
	sph_skein256_context     ctx_skein;
	sph_groestl256_context   ctx_groestl;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashA, 32);
	sph_skein256_close(&ctx_skein, hashB);

	sph_groestl256_init(&ctx_groestl);
	sph_groestl256(&ctx_groestl, hashB, 32);
	sph_groestl256_close(&ctx_groestl, hashA);

	memcpy(state, hashA, 32);
}

static bool init[MAX_GPUS] = { 0 };
static __thread uint32_t throughput = 0;

extern "C" int scanhash_allium(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!init[thr_id])
	{
		int dev_id = device_map[thr_id];
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}

		// Bottom of the throughput plateau; -i overrides.
		throughput = cuda_default_throughput(thr_id, 1U << 20);

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		blake256_cpu_init(thr_id, throughput);
		skein256_cpu_init(thr_id, throughput);
		groestl256_cpu_init(thr_id, throughput);

		// before lyra2_cpu_init: the self-test borrows the DMatrix symbol
		lyra2_device_selftest(thr_id);

		const size_t matrix_sz = sizeof(uint64_t) * 4 * 4;
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
		lyra2_cpu_init(thr_id, throughput, d_matrix[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput));

		init[thr_id] = true;
	}

	uint32_t _ALIGN(128) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_cpu_setBlock_80(pdata);
	groestl256_setTarget(ptarget);

	do {
		int order = 0;

		// clamp the launch to the range left, max_nonce included
		const uint32_t n = (uint32_t)min((uint64_t)throughput, (uint64_t)max_nonce - pdata[19] + 1);
		if (n == 0) {
			pdata[19] = max_nonce;
			break;
		}

		blakeKeccak256_cpu_hash_80(thr_id, n, pdata[19], d_hash[thr_id], order++);
		lyra2_cpu_hash_32(thr_id, n, d_hash[thr_id]);
		cubehash256_cpu_hash_32(thr_id, n, pdata[19], d_hash[thr_id], order++);
		lyra2_cpu_hash_32(thr_id, n, d_hash[thr_id]);
		skein256_cpu_hash_32(thr_id, n, pdata[19], d_hash[thr_id], order++);

		*hashes_done = pdata[19] - first_nonce + n;

		work->nonces[0] = groestl256_cpu_hash_32(thr_id, n, pdata[19], d_hash[thr_id], order++);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
			allium_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = groestl256_getSecNonce(thr_id, 1);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					allium_hash(vhash, endiandata);
					if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
					} else if (vhash[7] > Htarg) {
						gpu_increment_reject(thr_id);
						if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[1]);
					}
					// only the two lowest are reported: rescan above them
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					// sole candidate: skip the rest of the batch (the caller adds 1)
					const uint64_t next = (uint64_t)pdata[19] + n;
					pdata[19] = (next > max_nonce) ? max_nonce : (uint32_t)next - 1;
				}
				return work->valid_nonces;
			}
			// a screen near-miss is no GPU error
			if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
			}
			// rescan above it only if slot 1 holds another candidate; else the batch is done
			if (groestl256_getSecNonce(thr_id, 1) != UINT32_MAX) {
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)pdata[19] + n > max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += n;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_allium(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);

	groestl256_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
