/**
 * lyra2z330 (-a lyra2z330)
 *
 * Lyra2 used directly as the block hash: the 80-byte header is both password
 * and salt, with timeCost 2, 330 rows and 256 columns. There is no prehash
 * stage (unlike lyra2z, which blake256's the header first) and no power-of-2
 * row count, so the row indices use Lyra2's generic (modulo) form -- see
 * Lyra2Z.c. The memory matrix is 330 * 256 * 96 bytes = ~7.7 MB per hash, which
 * is why the throughput is derived from free VRAM rather than from an intensity
 * default: the matrix, not the kernel, is the limit.
 *
 * Reference miners: cpuminer-opt (algo/lyra2/lyra2z330.c) and KlausT ccminer
 * (lyra2/lyra2Z330.cu). Both agree bit-for-bit with the KAT below.
 *
 * The kernel lives in algos/stages/cuda_lyra2Z330.cu; lyra2z330_hash is the
 * host-side verifier that re-hashes every GPU candidate before it is submitted.
 */

extern "C" {
#include "Lyra2Z.h"
}

#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/selftest_gate.cuh"

#define LYRA2Z330_TIMECOST 2
#define LYRA2Z330_ROWS     330
#define LYRA2Z330_COLS     256
#define LYRA2Z330_MATRIX_BYTES ((size_t)BLOCK_LEN_INT64 * LYRA2Z330_COLS * 8 * LYRA2Z330_ROWS)

extern void lyra2z330_setBlock_80(const uint32_t *endiandata);
extern void lyra2z330_cpu_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_matrix,
	uint64_t *d_hash, uint32_t *d_resNonces, uint32_t target);
extern size_t lyra2z330_matrix_bytes();
extern uint32_t lyra2z330_hashes_per_block();

static uint64_t *d_matrix[MAX_GPUS] = { 0 };
static uint64_t *d_hash[MAX_GPUS]   = { 0 };
static uint32_t *d_resNonces[MAX_GPUS] = { 0 };
static uint64_t *h_matrix[MAX_GPUS] = { 0 };	// host verifier scratch
static uint32_t s_throughput[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { 0 };
static bool selftested[MAX_GPUS] = { 0 };

extern "C" void lyra2z330_hash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[8];

	LYRA2Z(hash, 32, input, 80, input, 80,
		LYRA2Z330_TIMECOST, LYRA2Z330_ROWS, LYRA2Z330_COLS);

	memcpy(state, hash, 32);
}

// Known-answer test on a fixed 80-byte input, cross-checked against both
// reference miners, plus a negative test (a one-bit input change must alter the
// digest -- proves the check is not vacuous).
static bool lyra2z330_cpu_selftest(int thr_id)
{
	static const uint8_t kat_digest[32] = {
		0xbb, 0xc0, 0x73, 0x08, 0x85, 0x6e, 0xef, 0x23,
		0x05, 0x23, 0x7f, 0xd2, 0xaa, 0x66, 0x2c, 0x65,
		0x73, 0xd2, 0xe1, 0x73, 0xfd, 0xde, 0xe5, 0x68,
		0x78, 0x8b, 0xc3, 0x00, 0x48, 0xd5, 0x4a, 0xb0
	};
	uint8_t header[80], digest[32], flipped[32];

	for (int i = 0; i < 80; i++) header[i] = (uint8_t)i;
	lyra2z330_hash(digest, header);

	if (memcmp(digest, kat_digest, 32) != 0) {
		gpulog(LOG_ERR, thr_id, "lyra2z330: CPU self-test FAILED -- shares would be rejected");
		return false;
	}

	header[79] ^= 0x01;
	lyra2z330_hash(flipped, header);
	if (memcmp(flipped, kat_digest, 32) == 0) {
		gpulog(LOG_ERR, thr_id, "lyra2z330: self-test is vacuous (input change did not alter the digest)");
		return false;
	}
	return true;
}

// GPU vs the CPU reference over consecutive nonces on a fixed header: covers the
// kernel, the padded-input upload and the per-thread nonce insertion. The digests
// must also differ from each other, so a kernel that wrote nothing cannot pass.
static bool lyra2z330_gpu_selftest(int thr_id)
{
	const uint32_t startNonce = 0x0f0f0f00u;
	const uint32_t n = min(4u, s_throughput[thr_id]);
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t _ALIGN(64) gpu[4 * 8], vhash[8];
	bool ok = true;

	for (int i = 0; i < 20; i++)
		endiandata[i] = 0x11111111u * (uint32_t)(i + 1);

	lyra2z330_setBlock_80(endiandata);
	CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xff, 2 * sizeof(uint32_t)));
	// target 0 keeps the on-device screen from reporting anything
	lyra2z330_cpu_hash(thr_id, n, startNonce, d_matrix[thr_id], d_hash[thr_id], d_resNonces[thr_id], 0);
	CUDA_SAFE_CALL(cudaMemcpy(gpu, d_hash[thr_id], (size_t)n * 32, cudaMemcpyDeviceToHost));

	for (uint32_t i = 0; i < n; i++) {
		be32enc(&endiandata[19], startNonce + i);
		lyra2z330_hash(vhash, endiandata);
		if (memcmp(&gpu[i * 8], vhash, 32) != 0) {
			gpulog(LOG_ERR, thr_id, "lyra2z330: GPU/CPU mismatch on nonce %08x", startNonce + i);
			ok = false;
		}
	}
	if (ok && n > 1 && memcmp(&gpu[0], &gpu[8], 32) == 0) {
		gpulog(LOG_ERR, thr_id, "lyra2z330: self-test is vacuous (two nonces gave the same digest)");
		ok = false;
	}
	return ok;
}

extern "C" int scanhash_lyra2z330(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t _ALIGN(64) vhash[8];
	uint32_t h_resNonces[2];
	const uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];

	if (opt_benchmark)
		ptarget[7] = 0x0000ff;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}

		// The matrix dominates everything else, so size the launch from free VRAM
		// and only then let -i / --throughput lower it.
		const size_t matrix_sz = lyra2z330_matrix_bytes();
		const uint32_t granularity = lyra2z330_hashes_per_block();
		size_t free_mem = 0, total_mem = 0;
		CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));

		const size_t reserve = 192ULL << 20;	// context + our small buffers + headroom
		uint32_t cap = (uint32_t)((free_mem > reserve ? free_mem - reserve : 0) / matrix_sz);
		cap -= cap % granularity;
		if (cap == 0) {
			gpulog(LOG_ERR, thr_id, "lyra2z330: needs %.0f MB of free VRAM per hash, only %.0f MB free",
				(double)matrix_sz / (1024.0 * 1024.0), (double)free_mem / (1024.0 * 1024.0));
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}

		uint32_t throughput = cuda_default_throughput(thr_id, cap);
		if (throughput > cap) {
			gpulog(LOG_WARNING, thr_id, "lyra2z330: intensity clamped to %u threads by the %.0f MB matrix",
				cap, (double)matrix_sz / (1024.0 * 1024.0));
			throughput = cap;
		}
		throughput -= throughput % granularity;
		if (throughput == 0) throughput = granularity;
		s_throughput[thr_id] = throughput;

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads (%.1f GB matrix)",
			throughput2intensity(throughput), throughput,
			(double)((size_t)throughput * matrix_sz) / (1024.0 * 1024.0 * 1024.0));

		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], (size_t)throughput * matrix_sz));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)throughput * 32));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], 2 * sizeof(uint32_t)));

		h_matrix[thr_id] = (uint64_t*) malloc(LYRA2Z330_MATRIX_BYTES);
		if (h_matrix[thr_id] == NULL) {
			gpulog(LOG_ERR, thr_id, "lyra2z330: cannot allocate the host verifier matrix");
			proper_exit(EXIT_CODE_SW_INIT_ERROR);
		}

		init[thr_id] = true;
	}

	if (!selftested[thr_id]) {
		selftested[thr_id] = true;
		// Both legs return false ONLY for a wrong answer: the CPU leg touches no CUDA and
		// the GPU leg uses CUDA_SAFE_CALL, which exits on a resource failure. So the verdict
		// is safe to gate on -- a KAT mismatch must refuse to start, not just warn.
		const bool st_ok = lyra2z330_cpu_selftest(thr_id) && lyra2z330_gpu_selftest(thr_id);
		if (st_ok && !opt_quiet)
			gpulog(LOG_INFO, thr_id, "lyra2z330: self-test OK (GPU == CPU reference)");
		selftest_gate(thr_id, "lyra2z330", st_ok);
	}

	const uint32_t throughput = s_throughput[thr_id];

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	lyra2z330_setBlock_80(endiandata);

	do {
		// clamp the launch to the range left, max_nonce included
		const uint32_t n = (uint32_t)min((uint64_t)throughput, (uint64_t)max_nonce - pdata[19] + 1);
		if (n == 0) {
			pdata[19] = max_nonce;
			break;
		}

		CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xff, 2 * sizeof(uint32_t)));

		lyra2z330_cpu_hash(thr_id, n, pdata[19], d_matrix[thr_id], d_hash[thr_id],
			d_resNonces[thr_id], ptarget[7]);

		CUDA_SAFE_CALL(cudaMemcpy(h_resNonces, d_resNonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

		*hashes_done = pdata[19] - first_nonce + n;

		if (h_resNonces[0] != UINT32_MAX)
		{
			be32enc(&endiandata[19], h_resNonces[0]);
			LYRA2Z_reuse(h_matrix[thr_id], vhash, 32, endiandata, 80, endiandata, 80,
				LYRA2Z330_TIMECOST, LYRA2Z330_ROWS, LYRA2Z330_COLS);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->nonces[0] = h_resNonces[0];
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);

				if (h_resNonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], h_resNonces[1]);
					LYRA2Z_reuse(h_matrix[thr_id], vhash, 32, endiandata, 80, endiandata, 80,
						LYRA2Z330_TIMECOST, LYRA2Z330_ROWS, LYRA2Z330_COLS);
					if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
						work->nonces[1] = h_resNonces[1];
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
					} else if (vhash[7] > ptarget[7]) {
						gpu_increment_reject(thr_id);
						if (!opt_quiet) gpulog(LOG_WARNING, thr_id,
							"result for %08x does not validate on CPU!", h_resNonces[1]);
					}
					// the device reports at most two: rescan above them
					pdata[19] = max(work->nonces[0], h_resNonces[1]) + 1;
				} else {
					// sole candidate: skip the rest of the batch (the caller adds 1)
					const uint64_t next = (uint64_t)pdata[19] + n;
					pdata[19] = (next > max_nonce) ? max_nonce : (uint32_t)next - 1;
				}
				return work->valid_nonces;
			}
			// a screen near-miss is no GPU error
			if (vhash[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet) gpulog(LOG_WARNING, thr_id,
					"result for %08x does not validate on CPU!", h_resNonces[0]);
			}
			// rescan above it only if slot 1 holds another candidate; else the batch is done
			if (h_resNonces[1] != UINT32_MAX) {
				pdata[19] = h_resNonces[0] + 1;
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
extern "C" void free_lyra2z330(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_matrix[thr_id]);
	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonces[thr_id]);
	free(h_matrix[thr_id]);

	d_matrix[thr_id] = NULL;
	d_hash[thr_id] = NULL;
	d_resNonces[thr_id] = NULL;
	h_matrix[thr_id] = NULL;

	selftested[thr_id] = false;
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
