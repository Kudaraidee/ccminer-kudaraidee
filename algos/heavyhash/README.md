# heavyhash (`-a heavyhash`)

HeavyHash proof-of-work, relocated from the repo root (`heavyhash/`).

HeavyHash sandwiches a matrix step between two Keccak passes: Keccak → a
64×64 matrix-vector multiply over the hash (the "heavy" core) → Keccak.

## Layout

Relocation only — every symbol (`scanhash_heavyhash`, `free_heavyhash`, the CPU
reference `heavyhash_hash`) is unchanged, so the dispatch wiring (`algos.h`
`ALGO_HEAVYHASH` / `"heavyhash"`, `miner.h`, `ccminer.cpp`) is untouched.

- `heavyhash.cu` — dispatcher / GPU orchestration.
- `cuda_heavyhash.cu` — the GPU kernel (Keccak + matrix).
- `heavyhash-gate.c` / `heavyhash-gate.h` — the algo gate and host reference.
- `keccak_tiny.c` / `keccak_tiny.h` — the compact Keccak used by both sides.

All includes are own-folder-relative (`keccak_tiny.h`, `heavyhash-gate.h`) or
from the project include dirs (`miner.h`, `cuda/helper.cuh`, `cuda/vectors.cuh`);
there were no parent-relative (`../`) includes and no external references to the
`heavyhash/` path, so the move is source-transparent.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host `heavyhash_hash` re-verifies every GPU
candidate before submit.
