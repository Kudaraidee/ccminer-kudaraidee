// SPDX-License-Identifier: GPL-3.0-or-later
//
// KawPoW (Ravencoin) ccminer bridge: scanhash_kawpow + init/free.
//
// This translation unit is the ccminer-facing side (miner.h world). The GPU
// orchestration, DAG state machine, NVRTC JIT and host ProgPoW re-verification
// live in kawpow_core.cpp behind a plain-C interface (kawpow_core.h), because
// miner.h macroizes `bool` (compat/stdbool.h) in a way that is incompatible with
// the C++ standard-library headers ethash requires -- so the two never mix in
// one TU. Every submitted share is host-reverified in the core (progpow::verify)
// before this bridge reports it, so a kernel bug can only ever cause a local
// reject, never a bad share.
//
// Provenance and design: algos/kawpow/README.md; stratum wiring in util.cpp
// (kawpow_stratum_notify) and ccminer.cpp (submit / gen_work).

// Include the CUDA and core headers (which pull C++ standard headers) BEFORE
// miner.h: miner.h macroizes `bool` via compat/stdbool.h, which the STL rejects
// if it is parsed afterwards (codebase convention, cf. rinhash_scanhash.cpp).
#include <cuda_runtime.h>
#include <cuda.h>
#include "kawpow_core.h"

#include "miner.h"
#include "cuda/helper.cuh"

#include <string.h>

static void*    s_core[MAX_GPUS]   = { 0 };
static bool     s_init[MAX_GPUS]   = { 0 };
static bool     s_selftested[MAX_GPUS] = { 0 };
static uint64_t s_cursor[MAX_GPUS] = { 0 };
static uint32_t s_sig[MAX_GPUS]    = { 0 };
static bool     s_sig_set[MAX_GPUS] = { 0 };

extern "C" int scanhash_kawpow(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	const int dev_id = device_map[thr_id];

	// --benchmark: synthesize a deterministic epoch-4 job so the kernel can run.
	if (opt_benchmark && !work->is_kawpow) {
		work->is_kawpow = true;
		work->height = 30000;
		for (int i = 0; i < 32; i++) work->kawpow_header[i] = (unsigned char)(i + 1);
		memset(work->kawpow_target, 0, 32);
		work->kawpow_target[3] = 0xff;  // easy target
		work->kawpow_prefix = 0;
	}

	if (!work->is_kawpow) { *hashes_done = 0; return 0; }

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 20);

	if (!s_init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		// Bind the runtime primary context for the driver-API (NVRTC) launches.
		if (cuInit(0) != CUDA_SUCCESS) { gpulog(LOG_ERR, thr_id, "KawPoW: cuInit failed"); return 0; }
		CUdevice cudev; CUcontext cuctx = NULL;
		cuDeviceGet(&cudev, dev_id);
		cuDevicePrimaryCtxRetain(&cuctx, cudev);
		cuCtxSetCurrent(cuctx);

		// Derive this device's SM arch for NVRTC (--gpu-architecture=compute_XY).
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, dev_id);
		const int sm_arch = prop.major * 10 + prop.minor;

		s_core[thr_id] = kawpow_core_create(sm_arch);
		if (!s_core[thr_id]) { gpulog(LOG_ERR, thr_id, "KawPoW: core init failed"); return 0; }

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);
		s_init[thr_id] = true;
	}

	// Ensure the DAG for this job's epoch is resident (may block on epoch change).
	int regenerated = 0;
	if (!kawpow_core_ensure(s_core[thr_id], (int)work->height, &regenerated)) {
		gpulog(LOG_ERR, thr_id, "KawPoW: DAG build failed for height %u", work->height);
		*hashes_done = 0;
		return 0;
	}
	if (regenerated && !opt_quiet)
		gpulog(LOG_INFO, thr_id, "KawPoW DAG ready (height %u)", work->height);

	if (!s_selftested[thr_id]) {
		if (kawpow_core_selftest(s_core[thr_id], (int)work->height))
			s_selftested[thr_id] = true;
		else
			gpulog(LOG_WARNING, thr_id, "KawPoW self-test FAILED (GPU/CPU mismatch)");
	}

	// Job signature: reset the nonce cursor on a genuinely new job (see sha256dv).
	uint32_t sig = 2166136261u;
	for (int i = 0; i < 32; i++) { sig ^= work->kawpow_header[i]; sig *= 16777619u; }
	for (int i = 0; i < 32; i++) { sig ^= work->kawpow_target[i]; sig *= 16777619u; }
	sig ^= work->height; sig *= 16777619u;
	if (!s_sig_set[thr_id] || sig != s_sig[thr_id]) {
		s_sig[thr_id] = sig; s_sig_set[thr_id] = true;
		s_cursor[thr_id] = (uint64_t)thr_id << 44;  // disjoint 48-bit sub-ranges
	}

	const uint64_t lo = s_cursor[thr_id] & 0xffffffffffffULL;
	const uint64_t start_nonce = ((uint64_t)work->kawpow_prefix << 48) | lo;

	uint64_t found_nonce = 0;
	unsigned char mix[32], final[32];
	int rc = kawpow_core_search(s_core[thr_id], work->kawpow_header, start_nonce,
		work->kawpow_target, (int)work->height, throughput, &found_nonce, mix, final);

	*hashes_done = throughput;
	s_cursor[thr_id] += throughput;

	if (rc == 1) {
		work->kawpow_nonce = found_nonce;
		memcpy(work->kawpow_mix, mix, 32);
		work->nonces[0] = (uint32_t)found_nonce;  // low word for hashlog dedup
		work->valid_nonces = 1;
		// Share diff for --show-diff: final hash (MSB-first) -> LE words (word 7 = MSW).
		uint32_t hash_le[8];
		for (int i = 0; i < 8; i++) hash_le[7 - i] = be32dec(final + 4 * i);
		bn_set_target_ratio(work, hash_le, 0);
		return 1;
	} else if (rc == -2) {
		gpu_increment_reject(thr_id);
		if (!opt_quiet)
			gpulog(LOG_WARNING, thr_id, "KawPoW result does not validate on CPU!");
	} else if (rc < 0) {
		gpulog(LOG_ERR, thr_id, "KawPoW: kernel launch/JIT failed");
	}

	return 0;
}

extern "C" void free_kawpow(int thr_id)
{
	if (!s_init[thr_id]) return;
	if (s_core[thr_id]) { kawpow_core_destroy(s_core[thr_id]); s_core[thr_id] = NULL; }
	s_sig_set[thr_id] = false;
	s_selftested[thr_id] = false;
	s_init[thr_id] = false;
}
