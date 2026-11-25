# sia (`-a sia`)

Siacoin proof-of-work (BLAKE2b), relocated from the repo root (`sia/`).

## Layout

Relocation only — every symbol is unchanged, so the dispatch wiring (`algos.h`
`ALGO_SIA` / `"sia"`, `miner.h`, `ccminer.cpp`) is untouched.

- `sia.cu` — the Sia BLAKE2b GPU driver.
- `sia-rpc.cpp` / `sia-rpc.h` — Sia's non-standard getwork/submit RPC (Sia uses
  a distinct header layout and submission format, handled here).

All includes are project-include-dir (`miner.h`, `sph/blake2b.h`,
`cuda/helper.cuh`, `cuda/vector_uint2x4.cuh`, `curl/curl.h`, `ccminer-config.h`) or
own-folder (`sia-rpc.h`); there were no parent-relative (`../`) includes.

## Move fix

`ccminer.cpp` included the RPC header as `"sia/sia-rpc.h"`; that external
reference was repointed to `"algos/sia/sia-rpc.h"`. No other file referenced the
`sia/` path.

> Note: `algos/blake2b/blake2b.cu` (`-a blake2b`) is a separate BLAKE2b algo and
> is not part of this folder.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename plus the one include
fix; correctness follows from the diff).
