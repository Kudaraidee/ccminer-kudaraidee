/*
 * HoohashV110 (PEPEPOW) - ccminer host glue (scanhash + free).
 *
 *   - Input : be32enc the 20 work words -> raw 80-byte header.
 *   - Output: the digest is big-endian; the kernel stores it byte-reversed so
 *             cuda_check_hash / fulltest (word 7 = MSB) work.
 *   - Difficulty: 2^16 share-target factor, set in ccminer.cpp.
 *
 * Every candidate is re-hashed on the CPU with the node's reference code
 * (ref/hoohash.c, MIT, vendored unchanged) and submitted on THAT digest. The old
 * reason for skipping it, "MSVC libm != consensus glibc", did not survive a
 * measurement: the reference built with MSVC 19.44 (UCRT, /O2 /MT, the miner's
 * flags) and with glibc 2.35 gave byte-identical digests on 11,004,136 nonces
 * (2026-11-23), and PePe nodes now build it with openlibm, which agrees too.
 * The init self-test checks this CPU path against a real block as well.
 */
#include "miner.h"
#include "cuda/helper.cuh"

// The reference, compiled in this TU: its "../blake3.h" is algos/hoohash/blake3.h,
// a shim onto cuda/blake3_device.cuh. Every math call in it takes a double, so C++
// overload resolution picks the same functions as C. MSBuild does not track this
// include: touch hoohash.cu after editing anything under ref/.
#include "ref/hoohash.c"

// Big-endian digest of a raw 80-byte header; the init self-test calls it too.
extern "C" void hoohash_ref_hash(const void* hdr80, uint8_t out_be[32])
{
	hoohashv110(hdr80, 80, out_be);
}

// CPU digest of one nonce in the kernel's layout (byte-reversed, word 7 = MSB).
static void hoohash_cpu_vhash(const uint32_t* endiandata, uint32_t nonce, uint32_t* vhash)
{
	uint32_t _ALIGN(64) hdr[20];
	uint8_t be[32];
	memcpy(hdr, endiandata, 80);
	be32enc(&hdr[19], nonce);
	hoohashv110(hdr, 80, be);
	uint8_t* v = (uint8_t*)vhash;
	for (int i = 0; i < 32; i++) v[i] = be[31 - i];
}

// CPU check of a candidate; a GPU/CPU split is a kernel fault worth its own line.
static bool hoohash_check(int thr_id, const uint32_t* endiandata, const uint32_t* d_slot,
	uint32_t nonce, const uint32_t* ptarget, uint32_t* vhash)
{
	uint32_t _ALIGN(64) gpu[8];
	cudaMemcpy(gpu, d_slot, 32, cudaMemcpyDeviceToHost);
	hoohash_cpu_vhash(endiandata, nonce, vhash);
	if (memcmp(gpu, vhash, 32) != 0)
		gpulog(LOG_WARNING, thr_id, "GPU/CPU digest mismatch for nonce %08x", nonce);
	return vhash[7] <= ptarget[7] && fulltest(vhash, ptarget);
}

static uint32_t* d_hash[MAX_GPUS];
static bool init[MAX_GPUS] = { 0 };

// launchers from cuda_hoohashv110.cu
extern "C" void hoohash_setBlock(const void* endiandata);
extern "C" void hoohash_gen_matrix(void);
extern "C" void hoohash_cpu_hash(uint32_t threads, uint32_t startNonce, uint32_t* d_hash, uint32_t tpb);
extern "C" bool hoohash_device_selftest(int thr_id);

#define HOOHASH_TPB 64u  // FP64-heavy: 64 threads/block

extern "C" int scanhash_hoohash(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	uint32_t* pdata = work->data;
	uint32_t* ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 18); // 1<<18 default (FP64)
	throughput &= 0xffffffc0u; // multiple of 64
	if (init[thr_id])
		throughput = min(throughput, max_nonce - first_nonce);

	// loose enough to produce candidates within the fixed benchmark window
	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ffff;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);

		// fail-closed: exits on a KAT mismatch
		if (hoohash_device_selftest(thr_id))
			gpulog(LOG_INFO, thr_id, "HoohashV110 self-test OK (real-block KAT on GPU and CPU + negative)");

		// 16 words/entry: cuda_check_hash (cuda_checkhash_64) strides 64-byte slots.
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput), 0);
		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	hoohash_setBlock(endiandata);
	hoohash_gen_matrix();           // nonce-independent: once per scan call
	cuda_check_cpu_setTarget(ptarget);

	do {
		hoohash_cpu_hash(throughput, pdata[19], d_hash[thr_id], HOOHASH_TPB);
		cudaDeviceSynchronize();

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t idx0 = work->nonces[0] - pdata[19];

			if (hoohash_check(thr_id, endiandata, d_hash[thr_id] + idx0 * 16,
			                  work->nonces[0], ptarget, vhash))
			{
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (work->nonces[1] != UINT32_MAX)
				{
					const uint32_t idx1 = work->nonces[1] - pdata[19];
					// Guard the second nonce like the first.
					if (hoohash_check(thr_id, endiandata, d_hash[thr_id] + idx1 * 16,
					                  work->nonces[1], ptarget, vhash))
					{
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				}
				else
				{
					pdata[19] = work->nonces[0] + 1;
				}
				return work->valid_nonces;
			}
			else
			{
				// also the screen-pass/fulltest-fail case; resume re-scans the tail
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_hoohash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_hash[thr_id]);
	cuda_check_cpu_free(thr_id);
	cudaDeviceSynchronize();
	init[thr_id] = false;
}
