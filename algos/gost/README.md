# gost (`-a gostcoin`)

GOSTcoin proof-of-work (`gostd`), relocated from the repo root (`gost/`).

```
gost512(80-byte header) -> gost256(64-byte result) -> 256-bit hash
```

i.e. a Streebog-512 pass over the header followed by a Streebog-256 pass over
its output (GOST R 34.11-2012).

## Layout

Relocation only — every symbol (`scanhash_gostd`, `free_gostd`, the CPU
references `gostd` / `gostd_hash`, and the device entry points `gostd_cpu_init` /
`gostd_setBlock_80` / `gostd_hash_80`) is unchanged, so the dispatch wiring
(`algos.h` `ALGO_GOSTCOIN` / `"gostcoin"`, `miner.h`, `ccminer.cpp`) is
untouched.

- `gost.cu` — dispatcher, CPU reference, GPU orchestration.
- `cuda_gosthash.cu` — the GOST/Streebog GPU kernels.

All includes are from the project include dirs (`sph/sph_streebog.h`,
`miner.h`, `cuda/helper.cuh`, `cuda/debug.cuh`) — no parent-relative (`../`)
includes and no external references to the `gost/` path — so the move is
source-transparent.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host `gostd_hash` re-verifies every GPU candidate
before submit.
