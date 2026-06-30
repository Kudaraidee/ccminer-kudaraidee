/*
 * HoohashV110 (PEPEPOW) — ccminer host glue (scanhash + free).
 *
 * Device kernels/launchers live in cuda_hoohashv110.cu (kept separate to avoid the
 * thrust/miner.h macro clash). Byte order verified against a real PEPEPOW block
 * (height 0x4734dd) by both the CPU and CUDA ports:
 *   - Input : be32enc the 20 work words -> raw 80-byte header (ver word 0x00400020
 *             -> bytes 00 40 00 20).
 *   - Output: HoohashV110 digest is BIG-ENDIAN (byte 0 = MSB); the kernel stores it
 *             byte-reversed so cuda_check_hash / fulltest (word 7 = MSB) work.
 *   - Difficulty: plain 0xffff base (no 256x factor) -> set in ccminer.cpp.
 *
 * CONSENSUS-CRITICAL DIVERGENCE from the usual ccminer pattern: candidates are NOT
 * re-hashed on the CPU. On Windows the CPU would link MSVC libm, which differs from
 * the consensus libm (glibc) at large-arg sin/cos, so a CPU recheck would REJECT the
 * GPU's correct shares. Instead we trust the GPU digest (read it back from d_hash and
 * fulltest that). A GPU startup self-test against the real-block KAT verifies this
 * GPU's libdevice matches consensus. (A 1.2M-header GPU-vs-glibc sweep found zero
 * divergence; see memory hoohash-cuda-port.)
 */
#include "miner.h"
#include "cuda_helper.h"

static uint32_t* d_hash[MAX_GPUS];
static bool init[MAX_GPUS] = { 0 };

// launchers from cuda_hoohashv110.cu
extern "C" void hoohash_setBlock(const void* endiandata);
extern "C" void hoohash_gen_matrix(void);
extern "C" void hoohash_cpu_hash(uint32_t threads, uint32_t startNonce, uint32_t* d_hash, uint32_t tpb);
extern "C" int  hoohash_gpu_self_test(void);

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

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);

		if (!hoohash_gpu_self_test())
			gpulog(LOG_WARNING, thr_id, "HoohashV110 GPU self-test FAILED "
				"(libdevice != consensus libm?) — shares may be rejected");
		else
			gpulog(LOG_INFO, thr_id, "HoohashV110 GPU self-test PASSED (real-block KAT)");

		// 16 words/entry: cuda_check_hash (cuda_checkhash_64) strides 64-byte slots.
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput), 0);
		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	hoohash_setBlock(endiandata);
	hoohash_gen_matrix();           // matrix is nonce-independent -> generate once per job
	cuda_check_cpu_setTarget(ptarget);

	do {
		hoohash_cpu_hash(throughput, pdata[19], d_hash[thr_id], HOOHASH_TPB);
		cudaDeviceSynchronize();

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			// Read the GPU digest for the found nonce (reversed -> fulltest layout).
			// NO CPU recompute: MSVC libm != consensus glibc would reject valid shares.
			uint32_t idx0 = work->nonces[0] - pdata[19];
			cudaMemcpy(vhash, d_hash[thr_id] + idx0 * 16, 32, cudaMemcpyDeviceToHost);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
			{
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (work->nonces[1] != 0)
				{
					uint32_t idx1 = work->nonces[1] - pdata[19];
					cudaMemcpy(vhash, d_hash[thr_id] + idx1 * 16, 32, cudaMemcpyDeviceToHost);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				}
				else
				{
					pdata[19] = work->nonces[0] + 1;
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg)
			{
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
