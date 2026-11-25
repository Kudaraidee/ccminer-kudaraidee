# x21s (`-a x21s`)

X21S (penfold 2018) — the x16s variable-order chain plus a fixed 5-stage tail
(GPLv3, from the SUQA x22i lineage).

```
[ x16s: 16 hashes in a per-block permuted order, first stage on the 80-byte
  header (blake/bmw/groestl/jh/keccak/skein/luffa/cube/shavite/simd/echo/
  hamsi/fugue/shabal/whirlpool/sha512), the rest 64-byte ]
      →  Haval-256  →  Tiger-192  →  Lyra2 (v2)  →  Streebog (GOST)  →  SHA-256
```

## Layout

- `x21s.cu` — dispatcher (`scanhash_x21s`, `x21s_hash` CPU reference). Includes
  `algos/common/cuda_x_stages.h` and drives the x16s stages through that bridge,
  then the fixed tail. The permutation is derived from the previous-block hash
  (`getAlgoString`), same as x16r/x16s.
- The two x21-specific stage primitives moved to `algos/stages/` during this
  migration (their device implementations already live in `cuda/*_device.cuh`):
  - `cuda_tiger192.cu` — Tiger-192 (`tiger192_cpu_hash_64`), thin wrapper over
    `cuda/tiger192_device.cuh`.
  - `cuda_sha256_2.cu` — the 64-byte SHA-256 terminal (`sha256_cpu_hash_64`,
    plus the `…_64z` zero-pad variant).
- All other stages are the shared launchers: the x16 80-byte heads + 64-byte
  bodies, `haval256_cpu_*` and `sha512_cpu_*` (shared with x17 — this migration
  switched x21s onto their **bare** names, retiring the last `x17_` forwarders),
  `streebog_cpu_hash_64`, and `lyra2v2_cpu_*` (the lyra2 family, not yet
  migrated, kept as local externs).

## Optimization

Relocation + de-brand only — **no fusion**. Like x16r/x16s the chain is
**variable-order** (the terminal stage differs per block), so the fused-compare
terminal is declined for the same reason as x16 (order-dependent terminal, only
a ~1-stage win, forks the consensus nonce path). The best nonce is found by the
shared `cuda_check_hash` / `cuda_check_hash_suppl` pass. Lyra2v2 is a matrix
boundary in the tail. Fusing the x16s permuted body (x16r-style per-permutation
run map) is a possible future step, shared with the x16 family.

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060,
`ptarget[7]=0x003f` so the CPU re-verify fires non-vacuously): **0
does-not-validate / 0 CUDA errors**. Live pool run owed.
