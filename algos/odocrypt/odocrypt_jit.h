#ifndef ODOCRYPT_JIT_H__
#define ODOCRYPT_JIT_H__ 1

#include <stdint.h>
#include "odocrypt.h"

// Per-epoch NVRTC recompile of the Odocrypt kernel with the cipher's rotation
// amounts as compile-time literals.
//
// The amounts live in __constant__ tables and are identical in all 84 rounds,
// but they are runtime values, so a general rotate must be emitted for each of
// the 110 rotations per round; as literals each folds to a funnel shift. The
// tables are regenerated once per ODO_SHAPECHANGE_INTERVAL, so one compile per
// epoch costs nothing. Anything that varies per job (header, target) or is
// indexed by the round counter (round keys) stays runtime data.
//
// Every entry point degrades to "not available" rather than failing: the caller
// keeps the statically compiled kernel as its fallback and only has to check
// odocrypt_jit_ready().

#ifdef __cplusplus
extern "C" {
#endif

// Compile (or reuse the cached module) for this epoch key. Returns false if the
// kernel is unavailable for any reason, having logged why.
bool odocrypt_jit_prepare( int thr_id, const OdoCrypt *c, uint32_t key, int sm_arch );

// True when prepare() succeeded and the gate in cuda_odocrypt.cu has not
// retired the JIT path for this thread.
bool odocrypt_jit_ready( int thr_id );

// Give up on the JIT path for the rest of the session (the gate calls this).
void odocrypt_jit_disable( int thr_id, const char *why );

// Push the per-job header + target high word into the module's own constant
// bank. Call once per job, before launching.
bool odocrypt_jit_set_job( int thr_id, const uint32_t *header19, uint32_t target_hi );

// Launch. Both return false if the launch could not be issued.
bool odocrypt_jit_launch_hash( int thr_id, uint32_t threads, uint32_t startNonce,
                               const void *sbox1, const void *sbox2,
                               void *resNonce, uint32_t tpb );
bool odocrypt_jit_launch_digest( int thr_id, uint32_t threads, uint32_t startNonce,
                                 const void *sbox1, const void *sbox2,
                                 void *out, uint32_t tpb );

// Registers the JIT kernel actually compiled to (0 if unavailable).
int odocrypt_jit_regs( int thr_id );

// Compiles performed so far.
uint32_t odocrypt_jit_compiles( int thr_id );

#ifdef __cplusplus
}
#endif

#endif /* ODOCRYPT_JIT_H__ */
