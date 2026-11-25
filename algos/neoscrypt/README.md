# neoscrypt (`-a neoscrypt`, `-a neoscrypt-xaya`)

NeoScrypt proof-of-work, relocated from the repo root (`neoscrypt/`). NeoScrypt
is a memory-hard scrypt variant (Salsa20/ChaCha20 core + BLAKE2s), used by
Feathercoin and others; `neoscrypt-xaya` selects the XAYA parameterization.

## Layout

Relocation only — every symbol (`scanhash_neoscrypt`, `free_neoscrypt`, the CPU
reference `neoscrypt()`) is unchanged, so the dispatch wiring (`algos.h`
`ALGO_NEOSCRYPT` / `"neoscrypt"`, `miner.h`, `ccminer.cpp`) is untouched.

- `neoscrypt.cpp` — scan driver.
- `neoscrypt-cpu.c` / `neoscrypt.h` — CPU reference.
- `cuda_neoscrypt.cu` — GPU kernel (built with `MaxRegCount=160`, preserved).
- `cuda_vectors.h` — NeoScrypt's own expanded vector-types header.

### cuda_vectors.h note

`cuda_neoscrypt.cu` includes `"cuda_vectors.h"`, which resolves **own-folder**
to this folder's `cuda_vectors.h` (the expanded NeoScrypt fork of the header),
not the small shared `cuda/vectors.cuh`. That resolution is preserved because the
header moves in the same folder as its only consumer. NeoScrypt is the sole
remaining user of this header (streebog was decoupled from it earlier, with the
one type it needed — `ulonglong2to8` — backported into the root header).

All other includes are project-include-dir (`miner.h`, `cuda/helper.cuh`,
`cuda/vector_uint2x4.cuh`) or own-folder (`neoscrypt.h`); there were no
parent-relative (`../`) includes and no external references to the `neoscrypt/`
path, so the move is source-transparent.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host `neoscrypt()` re-verifies GPU candidates.
