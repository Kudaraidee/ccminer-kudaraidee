// SPDX-License-Identifier: GPL-3.0-or-later
//
// Parameterized ProgPoW per-period JIT: generates a period-specialized CUDA
// search kernel, compiles it with NVRTC and caches the module keyed on period.
// The register count, cache/math op counts and the keccak seal are taken from a
// pp_params, so one code path serves kawpow, meowpow, evrprogpow, firopow and
// meraki. Each core owns its own ppmulti_jit instance, so the period-keyed cache
// never collides across variants.

#pragma once

#include <string>
#include <cuda.h>
#include "pp_params.h"

// GPU search result (layout mirrored verbatim in the generated device source).
struct ppmulti_result
{
    uint32_t found;     // 0/1, reset before each launch
    uint32_t nonce_lo;  // winning thread index (gid); nonce = start_nonce + gid
    uint32_t mix[8];    // ProgPoW mix_hash
    uint32_t final[8];  // final hash (little-endian words, state[0..7])
};

// Complete, self-contained CUDA source for the ProgPoW search kernel specialized
// to variant `p`, `period` and `num_items` (256-byte DAG elements). Entry point:
// progpow_search.
std::string ppmulti_progpow_source(const pp_params& p, uint64_t period, uint32_t num_items);

// Per-period compiled-module cache. Not thread-safe; driven from one scan thread
// per GPU.
class ppmulti_jit
{
public:
    ppmulti_jit(int sm_arch, const pp_params& p) : sm_arch_(sm_arch), params_(p) {}
    ~ppmulti_jit();

    bool get(uint64_t period, uint32_t num_items, CUfunction* fn);
    int compiles() const { return compiles_; }

private:
    int sm_arch_;
    pp_params params_;
    int compiles_ = 0;
    uint64_t cached_period_ = UINT64_MAX;
    CUmodule cached_module_ = nullptr;
    CUfunction cached_fn_ = nullptr;
};
