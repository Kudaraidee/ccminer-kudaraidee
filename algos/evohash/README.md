# evohash (`-a evohash`)

EvoHash, relocated from the repo root (`evohash/`). A memory-hard hybrid that
interleaves a long chain of 512-bit hash primitives with repeated Lyra2 passes.

```
CubeHash(80-byte header) -> BMW -> Lyra2
  -> Hamsi -> Fugue -> Lyra2
  -> SIMD -> Echo -> Lyra2
  -> CubeHash -> Shavite -> Lyra2
  -> Luffa -> Lyra2 -> ... (further hash/Lyra2 rounds)
```

Each Lyra2 pass is applied as two 32-byte halves (`LYRA2(..., 1, 8, 8)`), so the
GPU path reuses the shared Lyra2 matrix kernel.

## Layout

Relocation only — every symbol (`scanhash_evohash`, `free_evohash`, the CPU
reference `evohash()`) is unchanged, so the dispatch wiring (`algos.h`,
`miner.h`, `ccminer.cpp`) is untouched.

- `evohash.cu` — dispatcher, CPU reference, and GPU orchestration.

It consumes the shared x-family stage library and the migrated Lyra2:
- `#include "algos/common/cuda_x_stages.h"` — the shared bridge (bare stage
  launchers).
- `#include "algos/lyra2/Lyra2.h"` + the extern `lyra2_cpu_init` /
  `lyra2_cuda_hash_64` — co-owned with the lyra2 family.

All includes are project-include-dir (`.`) relative (`sph/*`, `miner.h`,
`cuda/helper.cuh`) or already repo-root-relative (`algos/...`), and the folder has
no local headers, so the move is source-transparent — no include edits were
needed.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host `evohash()` re-verifies every GPU candidate
before submit.
