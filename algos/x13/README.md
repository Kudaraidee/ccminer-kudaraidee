# x13 (`-a x13`)

Classic fixed 13-stage chain (tpruvot lineage, GPLv3).

```
Blake-512 (80-byte header)  →  BMW-512  →  Groestl-512  →  Skein-512  →
JH-512  →  Keccak-512  →  Luffa-512  →  CubeHash-512  →  Shavite-512  →
SIMD-512  →  Echo-512  →  Hamsi-512  →  Fugue-512 (terminal)
```

## Layout

- `x13.cu` — dispatcher (`scanhash_x13`, `x13hash` CPU reference). Includes
  `algos/common/cuda_x_stages.h` and calls every stage by its bare
  `<prim>512` name through that header's bridge.
- The first 11 stages are identical to x11, so they use the shared launchers in
  `algos/stages/` and the shared **register-resident fused kernel**
  (`algos/common/cuda_x_fused.cu`): the consecutive fusible run
  skein→jh→keccak→luffa→cubehash runs in a single launch. The order is fixed,
  so the fused kernel is compiled for this sequence (unlike x16r's per-hash order).
- The two x13-specific stages moved to `algos/stages/` during this migration:
  - `cuda_x13_hamsi512.cu` — the 64-byte Hamsi stage (plus the 80-byte
    `x16_hamsi512` head variant used by the x16 family). Built with
    `--maxrregcount=72`.
  - `cuda_x13_fugue512.cu` — the 64-byte Fugue stage.

  Both keep their `x13_` symbol names (aliased to the bare
  `hamsi512_*`/`fugue512_*` names by the `cuda_x_stages.h` bridge) because the
  `x13_`-prefixed spellings are still linked by the x16 dispatchers (`x16r`,
  `x16rv2`, `x16s`); every consuming family has since migrated, so dropping the
  aliases is now just a churn decision, not a migration dependency.

The `hsr` algo (HShare — x13 chain with an SM3 stage inserted) and its `sm3.*`
support still live in `x13/`; it is a separate algo and was not part of this
migration.

## Optimization

Echo is **mid-chain** here (Hamsi and Fugue follow), so it uses the plain
64-byte echo launcher rather than the fused-compare terminal that x11 uses.
Fugue is the terminal and the best nonce is found by the shared
`cuda_check_hash` / `cuda_check_hash_suppl` pass. No further fusion: the only
≥2 fusible run is the skein→cubehash block, which is already fused.

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060, target
loosened to `0x00ff` so the CPU re-verify fires non-vacuously): **0
does-not-validate / 0 CUDA errors**. Live pool run owed.
