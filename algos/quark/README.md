# quark family

Guideline: `docs/coding-guideline.md`

## Provenance

- `quark.cu` (`-a quark`) — tpruvot/ccminer Quark: a 9-effective-stage chain
  with **per-nonce branching** (blake(80) → bmw → {groestl|skein} → groestl → jh
  → {blake|bmw} → keccak → skein → {keccak|jh}, each fork chosen by `hash[0] & 0x8`).
- `anime.cu` (`-a anime`) — Animecoin: the same branching family, different
  fork layout.
- `nist5.cu` (`-a nist5`) — Talkcoin/NIST5: the fixed 5-stage chain
  blake(80) → groestl → jh → keccak → skein.

## Shared core primitives

The six core hashes this family hosts — blake512, bmw512, jh512, keccak512,
skein512, groestl512 — plus the branch machinery `cuda_quark_compactionTest.cu`
were relocated from `quark/` to the shared `algos/stages/` tree (layout B). Each
already `#include`s its register-resident `cuda/<prim>512_device.cuh` device
primitive. They keep their `quark_<prim>512_cpu_*` symbol names and are reached
by migrated chains through the bare-name macros in `algos/common/cuda_x_stages.h`;
the ~40 legacy callers across the family (x13/x15/x17/tribus/JHA) and standalone
algos (zr5, skein, pentablake, bastion, Algo256/bmw512) keep linking to the
`quark_*` names unchanged. Renaming these shared symbols to bare form is the
family-wide naming pass; every family has since migrated, so it is now purely a
churn decision over those ~40 call sites rather than a migration dependency —
still deferred.
`algos/stages/cuda_quark.h` was folded into `algos/common/cuda_x_stages.h`
2026-07-16 (the bare launcher prototypes + `quark_compactTest_*` +
`cuda_check_hash_branch` now live there); quark/anime/jackpot/JHA include the
bridge header directly.

## Branching vs. fusion

quark and anime derive a **per-nonce** function order via `quark_compactTest`
(it splits the live nonces into branch buffers by `hash[0] & 0x8` and each branch
runs a different stage), finishing with `cuda_check_hash_branch`. The
register-resident fused-run kernel needs a single order uniform across all
threads in a launch, so these two chains are **not** fusible. nist5 is a fixed
chain and is fusible (the `jh`→`keccak`→`skein` tail is a run over three fusible
stages, `groestl` being the quad-lane boundary) with a skein-final on-device
compare terminal — that optimisation is owed as a follow-up increment.

## Correctness note

Every GPU candidate is re-hashed on the host (`quarkhash` / `animehash` /
`nist5hash`) before submit, so a kernel/thermal glitch can only ever cause a
local reject, never a bad share. The relocation changed no hashing source; the
init-time GPU-vs-`sph` self-tests for all six primitives (in
`cuda/xfamily_selftest.cu`, each with a negative bit-flip check) pass fresh on
every start.

## Measured rates

| algo  | card     | driver | CUDA | intensity | rate |
|-------|----------|--------|------|-----------|------|
| nist5 | RTX 3060 | 595.95 | 11.8 | 20        | ~48 MH/s (benchmark; 0 CPU-validation failures, all six-primitive self-tests green). Live owed |
| quark | RTX 3060 | 595.95 | 11.8 | 20        | ~28.5 MH/s (benchmark; 0 CPU-validation failures). Live owed |
| anime | RTX 3060 | 595.95 | 11.8 | 20        | ~27 MH/s benchmark; **live-validated** (zpool, 24/24 accepted, 0 rejects, 0 does-not-validate) after both fixes below |

## Branch-nonce fix (2026-07-14)

The compaction (`quark_compactTest`) writes **absolute** nonces into its branch
vectors, so a stage consuming a branch vector must recover the hash-buffer slot
by subtracting the launch's start nonce. skein/jh/groestl did; the refactored
blake/bmw/keccak 64-byte kernels did not, indexing `g_hash` by the raw absolute
nonce — an out-of-bounds read (illegal memory access) for any nonzero start
nonce, i.e. both `--benchmark` and live. The three kernels now subtract the
start nonce (matching skein/jh/groestl); the NULL-vector callers across the rest
of the family are unaffected (they index by thread). The same fix repairs
`jackpotcoin`, which mixed the corrected and broken kernels on the same vectors.

## anime final-stage fix (2026-07-14)

anime's last stage is conditional (`animehash` step 9: `if hash[0]&8: keccak
else jh`). The GPU split branch3 into branch1 (`&8`) / branch2 (`!&8`), but then
ran the keccak on **all** threads before jh on branch2 — so every branch2 slot
became `jh(keccak(skein))` instead of `jh(skein)`. Those ~half-of-branch3 nonces
hashed wrong, surfacing as a flood of `result … does not validate on CPU!`
(GPU false-positives caught by the host re-hash — never submitted, but wasteful
and it hid half the real search space). Fixed by running the final keccak on
**branch1 only** (`nrm1`, `d_branch1Nonces`), exactly as `quark.cu` does;
branch1 is unchanged (still keccak), branch2 now correctly gets `jh(skein)`.
Confirmed with a temporarily-loosened benchmark target: does-not-validate went
740→0 in 30 s, and quark (already branch1-only) measured 0. quark was not
affected.
