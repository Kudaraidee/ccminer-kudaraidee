# blake2b (`-a blake2b`)

BLAKE2b proof-of-work (Siacoin, via pool/getwork), relocated from the repo root
(`blake2b.cu`) as part of clearing algo sources out of the root directory.

## Layout

Relocation only — every symbol (`scanhash_blake2b`, `blake2b_hash`,
`free_blake2b`, `blake2b_setBlock`, the device `blake2b_gpu_hash`) is unchanged,
so the dispatch wiring (`algos.h` `ALGO_BLAKE2B` / `"blake2b"`, `miner.h`,
`ccminer.cpp` — live at `case ALGO_BLAKE2B`) is untouched.

- `blake2b.cu` — the full BLAKE2b GPU driver + CPU reference.

All includes are project-include-dir (`miner.h`, `sph/blake2b.h`,
`cuda/helper.cuh`, `cuda/vector_uint2x4.cuh`); no parent-relative (`../`) includes
and no `#include` of this TU anywhere (its symbols are linked, not included), so
the move is source-transparent.

> Distinct from `-a sia` (`algos/sia/`), which mines Siacoin via Sia's own RPC
> and dispatches to `scanhash_sia`; and from the shared `sph/blake2b.c` and the
> Argon2/Equihash BLAKE2b variants. This folder is only the `-a blake2b` algo.

## Build

Registered in `ccminer-kudaraidee.vcxproj` + `.filters`. Historically Windows-only (it was
not in `Makefile.am`); left at that state by the relocation.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff).
