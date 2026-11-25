// SPDX-License-Identifier: GPL-3.0-or-later
//
// KawPoW core C interface. Isolates the ethash/C++-STL world (DAG state machine,
// NVRTC JIT, host ProgPoW reverify) from ccminer's bridge translation unit,
// which includes miner.h -- and miner.h pulls compat/stdbool.h, which macroizes
// `bool` and is incompatible with the C++ standard library headers ethash needs.
// The two never share a TU; they talk only through these plain-C signatures.

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque per-GPU core handle.
void* kawpow_core_create(int sm_arch);
void  kawpow_core_destroy(void* h);

// Ensure the DAG for `height`'s epoch is resident (may block on epoch change).
// Returns 1 on success, 0 on failure; *regenerated (if non-null) is set to 1 if
// a (re)build happened this call.
int kawpow_core_ensure(void* h, int height, int* regenerated);

// One-time differential self-test on the resident DAG (GPU == host + a negative
// check). Returns 1 on pass. Requires a prior successful ensure().
int kawpow_core_selftest(void* h, int height);

// Launch the search kernel over `throughput` nonces starting at start_nonce for
// the period of `height`. Returns 1 if a candidate meets the target AND passes
// host re-verification, filling *nonce_out, mix_out[32] and final_out[32] (the
// final hash, MSB-first, for share-diff display); else 0.
int kawpow_core_search(void* h, const unsigned char* header32, uint64_t start_nonce,
    const unsigned char* target32, int height, uint32_t throughput,
    uint64_t* nonce_out, unsigned char* mix_out, unsigned char* final_out);

#ifdef __cplusplus
}
#endif
