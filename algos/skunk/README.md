# skunk (`-a skunk`)

4-stage chain for **Signatum (SIGT)** — tpruvot 2017, GPLv3.

```
Skein-512 (80-byte header)  →  CubeHash-512 (64)  →  Fugue-512 (64)  →  Streebog/GOST-512 (64, terminal)
```

## Layout

- `skunk.cu` — dispatcher (`scanhash_skunk`, `skunk_hash` CPU reference).
- `cuda_skunk.cu` + `skein_header.h` — the **merged** skein+cubehash+fugue
  kernel (krnlx / alexis lineage, tpruvot final touch): `skunk_gpu_hash_80`
  computes all three of skein-80, cubehash-64 and fugue-64 in a single launch
  (`skunk_cuda_hash_80`), with the skein midstate precomputed on the host in
  `skunk_setBlock_80`. This is bespoke to skunk — not a shared stage — so it
  keeps the `skunk_` name.
- Terminal Streebog is the shared `streebog_cpu_hash_64_final`
  (`algos/stages/cuda_streebog.cu`): the last stage folded with the on-device
  target compare (2 nonces via an atomicExch chain into `d_resNonce`, eliding
  the `d_hash` store + the `cuda_check_hash` pass).

## Migration notes (2026-07-15)

- Moved from `skunk/` to `algos/skunk/`.
- Dropped the vestigial compat path: `WANT_COMPAT_KERNEL` was commented out and
  the scan loop always ran the merged kernel, so the `use_compat_kernels` flag
  plus the unused `x13_fugue512` / `skein512` / `x11_cubehash512` externs and
  their init/free calls were dead — removed.
- Normalized `cuda_skunk.cu`'s codegen from a Maxwell-only
  `compute_50,sm_50;compute_52,sm_52` override (below the sm_61 build floor, so
  it PTX-JIT'd compute_52 on every shipped card) to the project default
  (sm_61/75/86) → native SASS. `MaxRegCount=64` kept (the merged kernel is
  register-heavy).

No fusion work: the merged kernel already fuses skein/cube/fugue, and Streebog
is the terminal.

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060, target
`0xf` so the on-device compare fires): ~33 MH/s, **0 does-not-validate / 0 CUDA
errors** over 40 s. Live pool run owed.
