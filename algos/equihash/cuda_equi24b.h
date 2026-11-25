// equi24b -- Equihash solver for the DIGITBITS=24 family (144/5 and 192/7) on a
// heap of few large buckets. C entry points mirror the shape the miner expects;
// one solver, two instantiations keyed on WN.
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// NULL when the device cannot provide the arena (see *_arena_needed).
void *equi24b_144_init(void);
void  equi24b_144_free(void *ctx);
void *equi24b_192_init(void);
void  equi24b_192_free(void *ctx);

// Device bytes the context will try to allocate. Every layer is retained, so
// this grows with the round count and differs between the variants.
unsigned long long equi24b_144_arena_needed(void);
unsigned long long equi24b_192_arena_needed(void);

// Solve one 140-byte header+nonce under `personal`. Emits each distinct,
// non-trivial proof and returns the count emitted, negative on error.
// Proofs are NOT verified here; the caller re-verifies before submit.
int equi24b_144_solve(void *ctx, const char *headernonce, const char *personal,
                      void (*emit)(void *, const uint32_t *, uint32_t), void *ud);
int equi24b_192_solve(void *ctx, const char *headernonce, const char *personal,
                      void (*emit)(void *, const uint32_t *, uint32_t), void *ud);

// Loss counters from the last solve() (collect() returns its own), silent
// without this call:
//   [0] scatter overflow (destination bucket full)
//   [1] round staging overflow (chunk full)
//   [2] unused, always 0 (digitK stages every slot)
//   [3] candidates counted but not stored (capacity clamp)
void equi24b_144_drops(void *ctx, unsigned *out4);
void equi24b_192_drops(void *ctx, unsigned *out4);

// Pipelined solve on one context: launch() queues a whole solve and returns at
// once (at most two queued), collect() waits for the oldest, emits its proofs
// as solve() does and fills its loss counters (as drops()) when out4 is given.
// pipe_reset() drops the queued solves without emitting.
void *equi24b_144_pipe_new(void);
void  equi24b_144_pipe_free(void *pipe);
void  equi24b_144_pipe_reset(void *pipe);
int   equi24b_144_launch(void *ctx, void *pipe, const char *headernonce, const char *personal);
int   equi24b_144_collect(void *ctx, void *pipe, void (*emit)(void *, const uint32_t *, uint32_t),
                          void *ud, unsigned *out4);
void *equi24b_192_pipe_new(void);
void  equi24b_192_pipe_free(void *pipe);
void  equi24b_192_pipe_reset(void *pipe);
int   equi24b_192_launch(void *ctx, void *pipe, const char *headernonce, const char *personal);
int   equi24b_192_collect(void *ctx, void *pipe, void (*emit)(void *, const uint32_t *, uint32_t),
                          void *ud, unsigned *out4);

#ifdef __cplusplus
}
#endif
