// Host-side consensus verifiers for the DIGITBITS=24 variants.
//
// The re-verify keeps a solver defect a local reject rather than a bad share,
// and it is the only independent check on the solver: no second implementation
// remains in the tree to compare against.
//
// A verifier must match its variant. The 144/5 one handed a 192/7 proof reads
// 32 of the 128 indices under the wrong personalization and rejects everything,
// so pick by variant at the call site.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Verify an unpacked proof (1<<WK indices) against a 140-byte header+nonce
// under `personal`. Returns 0 (POW_OK) when valid.
int eq_verify_144(const char *headernonce, const char *personal,
                  const uint32_t *indices);
int eq_verify_192(const char *headernonce, const char *personal,
                  const uint32_t *indices);

#ifdef __cplusplus
}
#endif
