// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Decred (DCR) proof-of-work: BLAKE3-256 over the 180-byte header, per DCP-0011.
 *
 * The header is 180 bytes, so BLAKE3 sees one chunk of 64 + 64 + 52, and the
 * nonce at byte 140 lands 12 bytes into block 3 -- word 3, verbatim, because
 * BLAKE3 words are little-endian. Blocks 1-2 are therefore per-job constants:
 * the host compresses them once into a chaining value and the device only ever
 * compresses the final block.
 *
 * The stratum and header assembly live in ccminer.cpp and are unchanged; only
 * the hash moved. Nothing here byte-swaps the nonce: BLAKE3 hashes the header
 * bytes as they are, unlike the pre-fork BLAKE-256 kernel.
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda/helper.cuh"
#include "cuda/blake3_device.cuh"
#include "cuda/selftest_gate.cuh"

#define TPB		256
#define MAX_RESULTS	16

/* nonce word index inside work->data (byte 140) */
#define DCR_NONCE_OFT32	35

__constant__ static uint32_t c_cv[8];	/* midstate after header bytes 0..127 */
__constant__ static uint32_t c_m[16];	/* final block words, nonce slot patched per thread */

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

__global__ __launch_bounds__(TPB, 1)
void decred_blake3_gpu_hash(const uint32_t threads, const uint32_t startNonce,
	uint32_t *resNonce, const uint64_t highTarget)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	const uint32_t nonce = startNonce + thread;

	uint32_t cv[8], m[16];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		cv[i] = c_cv[i];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		m[i] = c_m[i];

	uint8_t dig[32];
	decred_blake3_nonce(cv, m, nonce, dig);

	/* Real 64-bit compare on the top two words. BLAKE3 emits little-endian
	 * words, so word 7 is the most significant and no swap is needed -- and a
	 * "top word == 0" screen would drop every share below difficulty 1. */
	const uint64_t high = ((const uint64_t *)dig)[3];
	if (high <= highTarget) {
		/* The counter is deliberately unbounded so the host can see a flood;
		 * the slot write is what must be bounded. */
		const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
		if (pos < MAX_RESULTS)
			resNonce[pos] = nonce;
	}
}

/* ---- host ---------------------------------------------------------------- */

/* work->data[0..44] already holds the header in wire byte order (ccminer.cpp
 * builds it), so the bytes are the message. */
extern "C" void decred_hash(void *output, const void *input)
{
	blake3_256((const uint8_t *)input, DECRED_HDR_LEN, (uint8_t *)output);
}

__host__
void decred_blake3_cpu_setBlock(const uint32_t *pdata)
{
	uint32_t cv[8], m[16];
	decred_blake3_prepare((const uint8_t *)pdata, cv, m);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_cv, cv, sizeof(cv), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_m, m, sizeof(m), 0, cudaMemcpyHostToDevice));
}

/* ---- init self-test, fail-closed ----------------------------------------- */

__global__ void decred_blake3_selftest_kernel(const uint32_t nonce, uint8_t *out)
{
	uint32_t cv[8], m[16];
	for (int i = 0; i < 8; i++)  cv[i] = c_cv[i];
	for (int i = 0; i < 16; i++) m[i] = c_m[i];
	decred_blake3_nonce(cv, m, nonce, out);
}

static bool decred_selftest_device(const uint32_t *pdata, uint32_t nonce, uint8_t *out32)
{
	decred_blake3_cpu_setBlock(pdata);

	uint8_t *d_out = NULL;
	if (cudaMalloc(&d_out, 32) != cudaSuccess)
		return selftest_cuda_fault();

	decred_blake3_selftest_kernel <<< 1, 1 >>> (nonce, d_out);

	const bool ok = cudaDeviceSynchronize() == cudaSuccess
	             && cudaMemcpy(out32, d_out, 32, cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d_out);

	return ok ? true : selftest_cuda_fault();
}

__host__
bool decred_blake3_device_selftest(int thr_id)
{
	/* BLAKE3("") from the published test vectors: the only leg that ties this
	 * kernel to the standard rather than to another copy of our own code. */
	static const char *kat_empty =
		"af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";
	uint8_t dig[32];
	char hex[65];
	blake3_256((const uint8_t *)"", 0, dig);
	for (int i = 0; i < 32; i++)
		sprintf(hex + 2 * i, "%02x", dig[i]);
	hex[64] = 0;
	const bool kat_ok = strcmp(hex, kat_empty) == 0;

	/* A deterministic header, hashed three ways. */
	uint32_t pdata[45];
	for (int i = 0; i < 45; i++)
		pdata[i] = 0x01020304u * (uint32_t)(i + 1);
	const uint32_t nonce = 0x7a3b1c5du;

	uint8_t full[32], dev[32];
	pdata[DCR_NONCE_OFT32] = nonce;
	decred_hash(full, pdata);				/* full 180-byte path */

	const bool ran = decred_selftest_device(pdata, nonce, dev);	/* midstate on device */
	const bool dev_ok = ran && memcmp(full, dev, 32) == 0;

	/* Midstate on the host must agree too, so a mismatch above localises to the
	 * device rather than to the split. */
	uint32_t cv[8], m[16];
	uint8_t mid[32];
	decred_blake3_prepare((const uint8_t *)pdata, cv, m);
	decred_blake3_nonce(cv, m, nonce, mid);
	const bool mid_ok = memcmp(full, mid, 32) == 0;

	/* Negative: a flipped header bit must change the DEVICE's answer. Comparing
	 * two host hashes would hold however broken the kernel is. */
	uint8_t dev2[32];
	pdata[0] ^= 1u;
	const bool ran2 = decred_selftest_device(pdata, nonce, dev2);
	const bool neg_ok = ran2 && memcmp(dev, dev2, 32) != 0;
	pdata[0] ^= 1u;

	const bool passed = kat_ok && dev_ok && mid_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "decred BLAKE3 self-test FAILED"
			" (kat %d dev %d mid %d neg %d)",
			(int)kat_ok, (int)dev_ok, (int)mid_ok, (int)neg_ok);

	return selftest_gate(thr_id, "decred", passed);
}

/* ---- scanhash ------------------------------------------------------------ */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_decred(int thr_id, struct work *work, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) hdr[48];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *pnonce = &pdata[DCR_NONCE_OFT32];

	const uint32_t first_nonce = *pnonce;
	uint64_t targetHigh = ((uint64_t *)ptarget)[3];

	if (opt_benchmark) {
		/* Loosen both target words and re-derive the device bound from them:
		 * loosening only ptarget[7] leaves the lower word narrow, which makes
		 * the accept region a strict subset of the screen's. */
		ptarget[7] = 0x00000003u;	/* 2^-30: ~0.5 candidates per 2^29 launch, no flood */
		ptarget[6] = 0xffffffffu;
		targetHigh = ((uint64_t *)ptarget)[3];
	}

	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 29 : 25;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id])
		throughput = min(throughput, max_nonce - first_nonce);

	const dim3 grid((throughput + TPB - 1) / TPB);
	const dim3 block(TPB);

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id],
			MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id],
			MAX_RESULTS * sizeof(uint32_t)), -1);

		decred_blake3_device_selftest(thr_id);
		init[thr_id] = true;
	}

	memcpy(hdr, pdata, DECRED_HDR_LEN);
	decred_blake3_cpu_setBlock(hdr);
	cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));

	do {
		uint32_t *resNonces = h_resNonce[thr_id];

		if (resNonces[0])
			cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));

		decred_blake3_gpu_hash <<<grid, block>>> (throughput, *pnonce,
			d_resNonce[thr_id], targetHigh);

		*hashes_done = (*pnonce) - first_nonce + throughput;

		cudaMemcpy(resNonces, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (resNonces[0]) {
			uint32_t _ALIGN(64) vhash[8];

			/* Clamp into a local and restore it after the copy: the copy
			 * overwrites slot 0 with the unbounded device counter. */
			uint32_t count = resNonces[0];
			if (count >= MAX_RESULTS) {
				gpulog(LOG_WARNING, thr_id, "candidates flood: %u", count);
				count = MAX_RESULTS - 1;
			}
			cudaMemcpy(resNonces, d_resNonce[thr_id],
				(count + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			resNonces[0] = count;

			/* CPU-check every slot and keep the two lowest valid nonces. */
			uint32_t valid[2], vh[2][8];
			int nvalid = 0;
			for (uint32_t n = 1; n <= resNonces[0]; n++) {
				/* No byte swap: the nonce is a little-endian header word. */
				const uint32_t nonce = resNonces[n];
				hdr[DCR_NONCE_OFT32] = nonce;
				decred_hash(vhash, hdr);
				if (fulltest(vhash, ptarget)) {
					if (nvalid == 0 || nonce < valid[0]) {
						valid[1] = valid[0]; memcpy(vh[1], vh[0], 32);
						valid[0] = nonce; memcpy(vh[0], vhash, 32);
					} else if (nvalid == 1 || nonce < valid[1]) {
						valid[1] = nonce; memcpy(vh[1], vhash, 32);
					}
					nvalid++;
				} else if (((uint64_t *)vhash)[3] > targetHigh) {
					/* above the device's 64-bit screen: a GPU error */
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id,
							"result %u for %08x does not validate on CPU!", n, nonce);
				}
			}

			if (nvalid) {
				work->nonces[0] = valid[0];
				work_set_target_ratio(work, vh[0]);
				work->valid_nonces = 1;
				if (nvalid > 1) {
					work->nonces[1] = valid[1];
					bn_set_target_ratio(work, vh[1], 1);
					work->valid_nonces = 2;
				}
				/* resume after the higher reported one; the rest lie above it */
				*pnonce = valid[work->valid_nonces - 1] + 1;
				return work->valid_nonces;
			}
		}
		if ((uint64_t) throughput + (*pnonce) >= max_nonce) {
			*pnonce = max_nonce;
			break;
		}
		*pnonce += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = (*pnonce) - first_nonce;
	MyStreamSynchronize(NULL, 0, device_map[thr_id]);
	return 0;
}

extern "C" void free_decred(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
