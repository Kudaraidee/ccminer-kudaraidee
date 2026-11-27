// SPDX-License-Identifier: GPL-3.0-or-later
//
// XHash (Parallax) ccminer bridge: scanhash_xhash + init/free.
//
// Work arrives through the shared KawPoW/ethproxy stratum path (see util.cpp
// and ccminer.cpp): header hash, seed hash, 256-bit target and extranonce
// prefix. The seed hash selects the epoch; every GPU candidate is re-verified
// against the CPU XHash reference before it can become a submitted share.

#include <cuda_runtime.h>
#include <cuda.h>

#include "xhash_core.h"
#include "miner.h"
#include "cuda/helper.cuh"

#include <string.h>

static void   *s_core[MAX_GPUS]   = { 0 };
static bool    s_init[MAX_GPUS]   = { 0 };
static bool    s_selftested[MAX_GPUS] = { 0 };
static uint64_t s_cursor[MAX_GPUS] = { 0 };
static uint32_t s_sig[MAX_GPUS]    = { 0 };
static bool    s_sig_set[MAX_GPUS] = { 0 };
static int     s_epoch[MAX_GPUS]   = { -1 };
static bool    s_nojob_logged[MAX_GPUS] = { 0 };

// Parallax seeds every 720 blocks. If the seed hash is not recognised, fall
// back to the height supplied by the pool.
static int xhash_epoch_for_work(struct work *work)
{
	int seed_epoch = -1;
	bool have_seed = false;
	for (int i = 0; i < 32; i++)
		if (work->kawpow_seed[i]) { have_seed = true; break; }
	if (have_seed)
		seed_epoch = xhash_core_epoch_from_seed(work->kawpow_seed);
	if (seed_epoch >= 0)
		return seed_epoch;
	return (int)(work->height / 720u);
}

extern "C" int scanhash_xhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	const int dev_id = device_map[thr_id];

	// --benchmark: deterministic epoch-0 job with an easy share target.
	if (opt_benchmark && !work->is_kawpow) {
		work->is_kawpow = true;
		work->height = 0;
		for (int i = 0; i < 32; i++) {
			work->kawpow_header[i] = (unsigned char)(i + 1);
			work->kawpow_seed[i] = 0;
		}
		memset(work->kawpow_target, 0, 32);
		work->kawpow_target[0] = 0xff; // easy target for --benchmark
		work->xhash_prefix = 0;
		work->xhash_prefix_len = 0;
	}

	if (!work->is_kawpow) {
		if (!s_nojob_logged[thr_id]) {
			s_nojob_logged[thr_id] = true;
			gpulog(LOG_WARNING, thr_id, "XHash: job has no kawpow header/target");
		}
		*hashes_done = 0;
		return 0;
	}

	const uint32_t throughput = cuda_default_throughput(thr_id, 1U << 20);
	const int epoch = xhash_epoch_for_work(work);
	if (epoch < 0) {
		gpulog(LOG_ERR, thr_id, "XHash: could not determine epoch from seed/height");
		*hashes_done = 0;
		return 0;
	}

	if (!s_init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		s_core[thr_id] = xhash_core_create();
		if (!s_core[thr_id]) {
			gpulog(LOG_ERR, thr_id, "XHash: core init failed");
			return 0;
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);
		s_init[thr_id] = true;
	}

	int regenerated = 0;
	if (!xhash_core_ensure(s_core[thr_id], epoch, &regenerated)) {
		gpulog(LOG_ERR, thr_id, "XHash: DAG build failed for epoch %d", epoch);
		*hashes_done = 0;
		return 0;
	}
	if (regenerated && !opt_quiet)
		gpulog(LOG_INFO, thr_id, "XHash DAG ready (epoch %d)", epoch);
	s_epoch[thr_id] = epoch;

	if (!s_selftested[thr_id]) {
		if (xhash_core_selftest(s_core[thr_id], epoch))
			s_selftested[thr_id] = true;
		else
			gpulog(LOG_WARNING, thr_id, "XHash self-test FAILED (GPU/CPU mismatch)");
	}

	// Reset the cursor on a genuinely new job.
	uint32_t sig = 2166136261u;
	for (int i = 0; i < 32; i++) { sig ^= work->kawpow_header[i]; sig *= 16777619u; }
	for (int i = 0; i < 32; i++) { sig ^= work->kawpow_target[i]; sig *= 16777619u; }
	for (int i = 0; i < 32; i++) { sig ^= work->kawpow_seed[i]; sig *= 16777619u; }
	sig ^= (uint32_t)epoch; sig *= 16777619u;
	sig ^= (uint32_t)work->xhash_prefix; sig *= 16777619u;
	if (!s_sig_set[thr_id] || sig != s_sig[thr_id]) {
		s_sig[thr_id] = sig;
		s_sig_set[thr_id] = true;
		s_cursor[thr_id] = (uint64_t)thr_id << 40; // disjoint sub-ranges
	}

	const unsigned plen = work->xhash_prefix_len;
	if (plen >= 8) {
		gpulog(LOG_ERR, thr_id, "XHash: extranonce prefix consumes the whole nonce");
		*hashes_done = 0;
		return 0;
	}
	const uint64_t mask = (~0ULL) >> (8 * plen);
	uint64_t cursor = s_cursor[thr_id] & mask;
	uint64_t start_nonce = work->xhash_prefix | cursor;

	uint64_t found_nonce = 0;
	unsigned char mix[32], final[32];
	int rc = xhash_core_search(s_core[thr_id], work->kawpow_header, start_nonce,
		work->kawpow_target, throughput, &found_nonce, mix, final);

	*hashes_done = throughput;
	s_cursor[thr_id] = (cursor + throughput) & mask;

	if (rc == 1) {
		work->kawpow_nonce = found_nonce;
		memcpy(work->kawpow_mix, mix, 32);
		work->nonces[0] = (uint32_t)found_nonce;
		work->valid_nonces = 1;
		// Final hash (MSB-first) -> LE words for --show-diff, as in kawpow.
		uint32_t hash_le[8];
		for (int i = 0; i < 8; i++) hash_le[7 - i] = be32dec(final + 4 * i);
		bn_set_target_ratio(work, hash_le, 0);
		return 1;
	}
	if (rc == -2) {
		gpu_increment_reject(thr_id);
		if (!opt_quiet)
			gpulog(LOG_WARNING, thr_id, "XHash result does not validate on CPU!");
	} else if (rc < 0) {
		gpulog(LOG_ERR, thr_id, "XHash: kernel launch failed");
	}
	return 0;
}

extern "C" void free_xhash(int thr_id)
{
	if (!s_init[thr_id]) return;
	if (s_core[thr_id]) { xhash_core_destroy(s_core[thr_id]); s_core[thr_id] = NULL; }
	s_sig_set[thr_id] = false;
	s_selftested[thr_id] = false;
	s_epoch[thr_id] = -1;
	s_init[thr_id] = false;
}
