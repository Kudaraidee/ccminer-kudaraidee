// SPDX-License-Identifier: GPL-3.0-or-later
//
// ccminer bridge for the ProgPoW variants MeowPow / EvrProgPow / FiroPoW. One
// shared scan routine (ppmulti_scan) drives the core with a per-variant
// pp_params; the three scanhash_* / free_* entry points ccminer dispatches to
// are thin wrappers. All three share the KawPoW stratum wire format, so they
// reuse work->kawpow_* (see util.cpp kawpow_stratum_notify + ccminer.cpp). Only
// one algo is active per process, so a single per-GPU state set is sufficient.
//
// Core/JIT headers (which pull C++ STL) precede miner.h, which macroizes `bool`
// via compat/stdbool.h (codebase convention; see algos/kawpow/kawpow.cpp).

#include <cuda_runtime.h>
#include <cuda.h>
#include "ppmulti_core.h"

#include "miner.h"
#include "cuda/helper.cuh"

#include <string.h>

// ---- Per-variant parameters -------------------------------------------------

static const pp_params PP_MEOWPOW = {
    /*epoch_length */ 7500,
    /*period_length*/ 6,
    /*num_regs     */ 16,
    /*cnt_cache    */ 6,
    /*cnt_math     */ 9,
    /*cnt_dag      */ 64,
    /*seal_mode    */ PP_SEAL_SEEDWORDS,
    /*seed_words   */ { 0x4D, 0x45, 0x4F, 0x57, 0x43, 0x4F, 0x49, 0x4E,   // MEOWCOIN
                        0x4D, 0x45, 0x4F, 0x57, 0x50, 0x4F, 0x57 },       // MEOWPOW
    /*name         */ "meowpow",
    /*dagchange    */ 110,   // Meowcoin: epoch >= 110 sizes the DAG for epoch*4
    /*dag_epoch_mul*/ 4,
    /*dag_full_off */ 0,     // (MeowPow scales via epoch*4, standard full init)
};

static const pp_params PP_EVRPROGPOW = {
    /*epoch_length */ 12000,
    /*period_length*/ 3,
    /*num_regs     */ 32,
    /*cnt_cache    */ 11,
    /*cnt_math     */ 18,
    /*cnt_dag      */ 64,
    /*seal_mode    */ PP_SEAL_SEEDWORDS,
    /*seed_words   */ { 0x45, 0x56, 0x52, 0x4D, 0x4F, 0x52, 0x45, 0x2D,   // EVRMORE-
                        0x50, 0x52, 0x4F, 0x47, 0x50, 0x4F, 0x57 },       // PROGPOW
    /*name         */ "evrprogpow",
    /*dagchange    */ 0,
    /*dag_epoch_mul*/ 1,
    /*dag_full_off */ 256,   // Evrmore: full_dataset_init_size 3x (1<<30 -> 3<<30)
};

static const pp_params PP_FIROPOW = {
    /*epoch_length */ 1300,
    /*period_length*/ 1,
    /*num_regs     */ 32,
    /*cnt_cache    */ 11,
    /*cnt_math     */ 18,
    /*cnt_dag      */ 64,
    /*seal_mode    */ PP_SEAL_VANILLA,
    /*seed_words   */ { 0 },
    /*name         */ "firopow",
    /*dagchange    */ 0,
    /*dag_epoch_mul*/ 1,
    /*dag_full_off */ 64,    // Firo: full_dataset_init_size 1.5x (1<<30 + 1<<29)
};

// Telestai (TLS). Reuses KawPoW's exact keccak seal (RAVENCOINKAWPOW seed words)
// but halves the ProgPoW main loop (cnt_dag 64 -> 32) and reshapes the per-loop
// op counts (cnt_cache 11 -> 12, cnt_math 18 -> 5). Standard ethash DAG sizing.
static const pp_params PP_MERAKI = {
    /*epoch_length */ 27500,
    /*period_length*/ 3,
    /*num_regs     */ 32,
    /*cnt_cache    */ 12,
    /*cnt_math     */ 5,
    /*cnt_dag      */ 32,
    /*seal_mode    */ PP_SEAL_SEEDWORDS,
    /*seed_words   */ { 0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49,   // rAVENCOI  (kawpow's
                        0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57 },       // NKAWPOW   1st word is 0x72)
    /*name         */ "meraki",
    /*dagchange    */ 0,
    /*dag_epoch_mul*/ 1,
    /*dag_full_off */ 0,     // Telestai: standard full_dataset_init_size (1<<30)
};

// ---- Shared per-GPU state (one algo active per process) ---------------------

static void*    s_core[MAX_GPUS]       = { 0 };
static bool     s_init[MAX_GPUS]       = { 0 };
static bool     s_selftested[MAX_GPUS] = { 0 };
static uint64_t s_cursor[MAX_GPUS]     = { 0 };
static uint32_t s_sig[MAX_GPUS]        = { 0 };
static bool     s_sig_set[MAX_GPUS]    = { 0 };

static int ppmulti_scan(const pp_params* pp, int thr_id, struct work* work,
    uint32_t max_nonce, unsigned long* hashes_done)
{
    (void)max_nonce;
    const int dev_id = device_map[thr_id];

    // --benchmark: synthesize a deterministic job so the kernel can run.
    if (opt_benchmark && !work->is_kawpow) {
        work->is_kawpow = true;
        work->height = pp->epoch_length * 4;  // a stable mid-range epoch
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
        if (cuInit(0) != CUDA_SUCCESS) { gpulog(LOG_ERR, thr_id, "%s: cuInit failed", pp->name); return 0; }
        CUdevice cudev; CUcontext cuctx = NULL;
        cuDeviceGet(&cudev, dev_id);
        cuDevicePrimaryCtxRetain(&cuctx, cudev);
        cuCtxSetCurrent(cuctx);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev_id);
        const int sm_arch = prop.major * 10 + prop.minor;

        s_core[thr_id] = ppmulti_core_create(sm_arch, pp);
        if (!s_core[thr_id]) { gpulog(LOG_ERR, thr_id, "%s: core init failed", pp->name); return 0; }

        gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
            throughput2intensity(throughput), throughput);
        s_init[thr_id] = true;
    }

    int regenerated = 0;
    if (!ppmulti_core_ensure(s_core[thr_id], (int)work->height, &regenerated)) {
        gpulog(LOG_ERR, thr_id, "%s: DAG build failed for height %u", pp->name, work->height);
        *hashes_done = 0;
        return 0;
    }
    if (regenerated && !opt_quiet)
        gpulog(LOG_INFO, thr_id, "%s DAG ready (height %u)", pp->name, work->height);

    if (!s_selftested[thr_id]) {
        if (ppmulti_core_selftest(s_core[thr_id], (int)work->height))
            s_selftested[thr_id] = true;
        else
            gpulog(LOG_WARNING, thr_id, "%s self-test FAILED (GPU/CPU mismatch)", pp->name);
    }

    // Job signature: reset the nonce cursor on a genuinely new job.
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
    int rc = ppmulti_core_search(s_core[thr_id], work->kawpow_header, start_nonce,
        work->kawpow_target, (int)work->height, throughput, &found_nonce, mix, final);

    *hashes_done = throughput;
    s_cursor[thr_id] += throughput;

    if (rc == 1) {
        work->kawpow_nonce = found_nonce;
        memcpy(work->kawpow_mix, mix, 32);
        work->nonces[0] = (uint32_t)found_nonce;
        work->valid_nonces = 1;
        uint32_t hash_le[8];
        for (int i = 0; i < 8; i++) hash_le[7 - i] = be32dec(final + 4 * i);
        bn_set_target_ratio(work, hash_le, 0);
        return 1;
    } else if (rc == -2) {
        gpu_increment_reject(thr_id);
        if (!opt_quiet)
            gpulog(LOG_WARNING, thr_id, "%s result does not validate on CPU!", pp->name);
    } else if (rc < 0) {
        gpulog(LOG_ERR, thr_id, "%s: kernel launch/JIT failed", pp->name);
    }

    return 0;
}

static void ppmulti_free(int thr_id)
{
    if (!s_init[thr_id]) return;
    if (s_core[thr_id]) { ppmulti_core_destroy(s_core[thr_id]); s_core[thr_id] = NULL; }
    s_sig_set[thr_id] = false;
    s_selftested[thr_id] = false;
    s_init[thr_id] = false;
}

// ---- Per-variant ccminer entry points ---------------------------------------

extern "C" int scanhash_meowpow(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{ return ppmulti_scan(&PP_MEOWPOW, thr_id, work, max_nonce, hashes_done); }

extern "C" int scanhash_evrprogpow(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{ return ppmulti_scan(&PP_EVRPROGPOW, thr_id, work, max_nonce, hashes_done); }

extern "C" int scanhash_firopow(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{ return ppmulti_scan(&PP_FIROPOW, thr_id, work, max_nonce, hashes_done); }

extern "C" int scanhash_meraki(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{ return ppmulti_scan(&PP_MERAKI, thr_id, work, max_nonce, hashes_done); }

extern "C" void free_meowpow(int thr_id)    { ppmulti_free(thr_id); }
extern "C" void free_evrprogpow(int thr_id) { ppmulti_free(thr_id); }
extern "C" void free_firopow(int thr_id)    { ppmulti_free(thr_id); }
extern "C" void free_meraki(int thr_id)     { ppmulti_free(thr_id); }
