# JHA v8 family (`-a jha`, `-a jackpot`)

Two **JackpotCoin (JHA v8)** algos, tpruvot lineage, GPLv3. Both share one CPU
hash and one per-nonce **branching** chain — they differ only in the GPU
branch-walking strategy.

```
Keccak-512 (80-byte header)
then x3 rounds of:
    hash[0]&1 ? Groestl-512 : Skein-512   (64)
    hash[0]&1 ? Blake-512   : JH-512      (64)
```

Each nonce takes one of two branches per round, chosen by the low bit of the
running hash. This is the same per-nonce branch structure as `-a anime`, so —
like anime — neither algo is **fusible** into the shared register-resident
chain kernel.

## Layout

- `jha.cu` — `-a jha` dispatcher (`scanhash_jha`, `jha_hash` CPU reference).
  Walks the branches with its own inline `jha_filter`/`jha_merge` compaction
  kernels into a second buffer `d_hash_br2`.
- `jackpot.cu` — `-a jackpot` dispatcher (`scanhash_jackpot`,
  `jackpothash` CPU reference). Walks the branches with the prefix-sum
  compaction in `cuda_jha_compactionTest.cu`.
- `cuda_jha_compactionTest.cu` — `jackpot_compactTest_*`, the scan/scatter
  branch-compaction used **only** by `-a jackpot`. Built with
  `--maxrregcount=80`.

Both dispatchers moved from `JHA/` and were de-branded to call the core stages
by their bare `<prim>512_cpu_*` names (blake/groestl/jh/skein). They include
`algos/common/cuda_x_stages.h` — which declares those bare launchers plus
`cuda_check_hash_branch` (the old `quark/cuda_quark.h` was folded into the bridge
header 2026-07-16) — and no longer use any `quark_`-prefixed forwarders.
Neither algo adopts the fused-chain machinery the fixed x11/x13 chains use: the
branching structure isn't fusible.

### Shared keccak head — moved to `algos/stages/`

The variable-length **jackpot keccak** (`jackpot_keccak512_*`) is *not*
JHA-only: it also backs the 80-byte path of the shared core keccak stage
(`algos/stages/cuda_keccak512.cu`) and is used by `-a zr5`. So its TU moved to
`algos/stages/cuda_jha_keccak512.cu` (shared-stage home), keeping the
`jackpot_keccak512_` name for its four consumers (core keccak stage, zr5, jha,
jackpot).

The best nonce is found by the shared `cuda_check_hash` (jha) /
`cuda_check_hash_branch` (jackpot) pass.

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060, target
`0x000f` so the CPU re-verify fires): **0 does-not-validate / 0 CUDA errors**
for both `jha` and `jackpot`. Live pool run owed.
