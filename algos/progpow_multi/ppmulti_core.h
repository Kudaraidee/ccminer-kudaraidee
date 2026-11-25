// SPDX-License-Identifier: GPL-3.0-or-later
//
// ProgPoW-family core C interface. Isolates the ethash/C++-STL world (DAG state
// machine, NVRTC JIT, host reverify) from the ccminer bridge (miner.h macroizes
// `bool`), exactly as algos/kawpow/kawpow_core.h does. The per-variant behavior
// is selected by the pp_params passed at create time.

#pragma once

#include <cstdint>
#include "pp_params.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque per-GPU core handle. `p` is copied; the caller may pass a temporary.
void* ppmulti_core_create(int sm_arch, const pp_params* p);
void  ppmulti_core_destroy(void* h);

// Ensure the DAG for `height`'s epoch (height / p.epoch_length) is resident.
int ppmulti_core_ensure(void* h, int height, int* regenerated);

// One-time differential self-test on the resident DAG (GPU == host + negative).
int ppmulti_core_selftest(void* h, int height);

// Launch the search kernel over `throughput` nonces from start_nonce for the
// period of `height`. Returns 1 if a candidate meets the target AND passes host
// re-verification, filling *nonce_out, mix_out[32] and final_out[32].
int ppmulti_core_search(void* h, const unsigned char* header32, uint64_t start_nonce,
    const unsigned char* target32, int height, uint32_t throughput,
    uint64_t* nonce_out, unsigned char* mix_out, unsigned char* final_out);

#ifdef __cplusplus
}
#endif
