# skein (`-a skein`, `-a skein2`)

The Skein-family standalone algos, relocated from the repo root as part of
clearing algo sources out of the root directory. Two live algos share this
folder (both dispatched in `ccminer.cpp`):

- `skein` (`ALGO_SKEIN`) — Skeincoin: Skein-512 over the 80-byte header, then
  SHA-256. Driver `skein.cu` (`scanhash_skeincoin`) + kernel `cuda_skein.cu`
  (`skeincoin_setBlock_80` + the SHA-256 tail); the pair must stay together.
- `skein2` (`ALGO_SKEIN2`) — double-Skein (Woodcoin): two Skein-512 passes.
  Host driver `skein2.cpp` (`scanhash_skein2`), no own kernel.

## Layout

Relocation only — symbols unchanged, so the dispatch wiring (`algos.h`,
`miner.h`, `ccminer.cpp`) is untouched.

- `skein.cu` / `cuda_skein.cu` — the Skeincoin driver + GPU kernel.
- `skein2.cpp` — the double-Skein driver.

Both drivers use the shared Skein-512 stage launchers already in
`algos/stages/` (`skein512_cpu_setBlock_80` / `skein512_cpu_hash_80`, and
`quark_skein512_cpu_hash_64` for skein2) plus the shared `cuda_check_hash`.
All includes are project-include-dir (`miner.h`, `sph/sph_skein.h`,
`cuda/helper.cuh`, `openssl/sha.h`); no parent-relative (`../`) includes and no
`#include` of these TUs (their symbols are linked), so the move is
source-transparent.

## Build

Per-file CUDA settings preserved: `skein.cu` `MaxRegCount=64`,
`cuda_skein.cu` `MaxRegCount=48`, and the autotools `skein.o`
`--maxrregcount=64` rule was repointed to `algos/skein/`. All three files were
already in both build systems.

## Dead pre-sm_50 path removed

`skein.cu` carried a second, self-contained SHA-256 implementation plus a `d_hash`
buffer and `cuda_check_cpu_*` wiring, selected by `sm5 = (device_sm >= 500)`. The
build floor is sm_61, so that branch was unreachable and was deleted along with
the `sm5`/`checkSecnonce` plumbing.

That also fixed a dropped share: `skeincoin_hash_sm5` reports a *second* candidate
per batch, but its only consumer sat behind
`checkSecnonce = (have_stratum || have_longpoll) && !sm5`, never true on a
supported card. The driver now CPU-re-verifies that nonce and submits it when it
passes; `work->nonces[1]` is armed before each launch, since the kernel only
writes a slot on a find.

## Constants

SHA-256's round constants and IV come from `cuda/sha256_device.cuh`. Kept local:
`sha256_endingTable`, which folds K+W for the fully constant second block and so
is strictly less work than a generic transform.

`TPB` was swept: the kernel needs 40 registers with no spills, so every size up to
768 already runs at full occupancy and they all measure alike. Only 1024 is worse,
being the one shape that both drops occupancy and raises the register count.

## Validation

Full rebuild, then benchmark and live pool: accepted, 0 rejects, 0 "does not
validate", and the benchmark CPU-re-verify fires with 0 failures. Both algos
CPU-re-verify candidates before submit.
