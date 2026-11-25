# tribus (`-a tribus`)

3-stage chain for **Denarius (DNR)** — tpruvot 2017, GPLv3.

```
JH-512 (80-byte header)  →  Keccak-512 (64)  →  Echo-512 (64, terminal)
```

## Layout

- `tribus.cu` — dispatcher (`scanhash_tribus`, `tribus_hash` CPU reference).
  Includes `algos/common/cuda_x_stages.h` and calls the stages by their bare
  names (`jh512_setBlock_80`/`jh512_cuda_hash_80`, `keccak512_cpu_hash_64`,
  `echo512_cpu_hash_64_final`).
- The stage launchers live in `algos/stages/`:
  - JH / Keccak — the shared core primitives (`cuda_jh512.cu`,
    `cuda_keccak512.cu`).
  - Echo terminal — `cuda_echo512_final.cu` (+ its AES helper
    `cuda_echo512_aes.cuh`), moved here from `tribus/` during the migration
    because it is a **family-shared** terminal: the fixed echo last-stage folded
    with the on-device target compare (2 nonces via an atomicExch chain into
    `d_resNonce`, eliding the echo `d_hash` store + the `cuda_check_hash`/suppl
    passes). Exposed under the bare name `echo512_cpu_hash_64_final` and reused
    by 0x10/c11/deep/qubit/fresh/phi/sib/x16 terminals.

The compat path (`echo512_cpu_hash_64_compat` + `cuda_check_hash`) is retained
for arch < 500, which is below the sm_61 build floor — dead on every shipped
target but kept for structural parity with the other terminal-echo dispatchers.

## Optimization

No fusion: Keccak is a run-of-1 between the 80-byte JH boundary and the echo
terminal, so there is no ≥2 run of fusible 64-byte stages to merge. The
fused-compare echo terminal is already in place (it is the shared
`echo512_cpu_hash_64_final`).

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060, target
loosened to `0x00FF` so the on-device compare fires non-vacuously): ~67 MH/s,
**0 does-not-validate / 0 CUDA errors** over 45 s. Live pool run owed.
