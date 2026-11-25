# zr5 (`-a zr5`)

ZR5 (Ziftrcoin, tpruvot lineage, GPLv3), relocated from the repo root.

```
Keccak-512 (80-byte, with PoK) → 4 rounds of {Blake,Groestl,JH,Skein}-512 in a
per-nonce permuted order (phash[0] % 24) → repeat once for Proof-of-Knowledge
```

- `zr5.cu` — dispatcher (`scanhash_zr5`, `zr5hash`/`zr5hash_pok` CPU reference)
  plus its bespoke permutation-buffer kernels (`zr5_init_vars`,
  `zr5_move_data_to_hash`, `zr5_get_poks[_xor]`, `zr5_final_round`) — these keep
  their `zr5_` names (algo-specific, not shared stages).

## Layout

Relocation + de-brand of the shared core stages:
- The four permuted rounds now call the bare `blake512_cpu_*` / `groestl512_cpu_*`
  / `jh512_cpu_*` / `skein512_cpu_*` launchers (real symbols in `algos/stages/`)
  instead of the `quark_*` forwarders — same kernels, pure rename (matching the
  pentablake migration).
- The PoK Keccak (`jackpot_keccak512_cpu_*`, `zr5_keccak512_cpu_hash[_pok]`) keeps
  its name: it is the bespoke 80-byte-with-PoK Keccak defined in the JHA folder
  (`algos/jha/cuda_jha_keccak512.cu`, shared with `-a jackpot`), not the shared
  quark Keccak — left branded until the JHA keccak is addressed.
- Includes are `miner.h` / `cuda/helper.cuh` / `sph/*` (root-relative, unaffected
  by the move).

## Validation

Benchmark sets `ptarget[7]=0x0000ff`. Rebuild + benchmark re-validation owed
(0 does-not-validate expected — the core de-brand is a pure rename).
