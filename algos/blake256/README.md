# blake256 family (`-a blakecoin`, `-a blake`, `-a vanilla`)

Blake-256 based coins (tpruvot lineage, GPLv3), relocated from `Algo256/`.

- `blake256.cu` — `scanhash_blake256`: Blake-256 with the round count selected
  per algo (blakecoin = 8 rounds, blake = 14 rounds).
- `vanilla.cu` — `scanhash_vanilla`: BlakeVanilla (VNL), Blake-256 8-round.

## Layout

Relocation only (layout B). The Blake-256 device kernels live in the shared
`algos/stages/cuda_blake256.cu` (`blake256_cpu_*`, plus the fused
`blakeKeccak256_cpu_hash_80` head used by the lyra2 family). `blake256.cu` keeps
its `--maxrregcount=64` + `-dlcm=cg` + FastMath build overrides. Dispatchers
reach the primitive through their own `extern` declarations (unchanged by the
move). Symbol names are **kept** because `cuda_blake256.cu` is co-owned with the
lyra2 family (lyra2RE/REv2/Z) and allium, all of which are now migrated too
(`algos/lyra2/`, `algos/allium/`). A de-brand is therefore a family-wide 256-bit
naming pass across every owner rather than a blake256-local change, so it stays a
separate follow-up — it is no longer waiting on a migration.

## Validation

Relocation is a pure `git mv` + build-system repoint (correctness follows from
the unchanged symbols). Rebuild + benchmark re-validation owed.
