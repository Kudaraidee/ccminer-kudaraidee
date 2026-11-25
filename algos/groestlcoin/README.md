# groestlcoin (`-a groestl`)

Groestlcoin proof-of-work (Groestl-512), relocated from the repo root — the last
algo family to leave the root directory.

## Layout

Relocation only — every symbol (`scanhash_groestlcoin`, `free_groestlcoin`, the
device `groestlcoin_cpu_*`) is unchanged, so the dispatch wiring (`algos.h`
`ALGO_GROESTL`, `miner.h`, `ccminer.cpp` — live at `case ALGO_GROESTL`) is
untouched.

- `groestlcoin.cpp` — host driver / CPU reference (`sph_groestl` + `SHA256`).
- `cuda_groestl.cu` — the GPU kernel (`groestlcoin_gpu_hash_quad`) and
  launchers (`groestlcoin_cpu_init/setBlock/hash/free`).
- `cuda_groestl.h` — the driver↔kernel interface header (moved with the
  folder; included own-folder by `groestlcoin.cpp`).

It is self-contained and independent of `algos/myriadgroestl/`; the GPU Groestl
core is the shared `cuda/groestl512_x2_device.cuh` (two hashes per thread, bitsliced),
included via the project include path. Includes are project-include-dir
(`miner.h`, `cuda/helper.cuh`, `sph/sph_groestl.h`, `openssl/sha.h`) or own-folder
(`cuda_groestl.h`), with no parent-relative (`../`) includes, so the move is
source-transparent.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host reference re-verifies GPU candidates.
