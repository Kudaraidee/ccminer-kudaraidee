# pentablake (`-a pentablake`)

Penta Blake — five sequential Blake-512 rounds (tpruvot lineage, GPLv3).

```
Blake-512 (80-byte header)  →  Blake-512 ×4 (64-byte) (terminal)
```

## Layout

- `pentablake.cu` — dispatcher (`scanhash_pentablake`, `pentablakehash` CPU
  reference). A standalone algo: its only GPU stage is Blake-512, which it now
  calls by the bare `blake512_cpu_*` names (real symbols in
  `algos/stages/cuda_blake512.cu`) — the `quark_blake512_*` forwarders it used
  before the migration are gone from this file.
- Built with `--maxrregcount=80`, `--ptxas-options="-dlcm=cg"` and FastMath
  (preserved from the original build entry).

## Optimization

Relocation + de-brand only — nothing to fuse (a single repeated stage). The
80-byte first round uses `blake512_cpu_hash_80`; the four 64-byte rounds use
`blake512_cpu_hash_64`; the best nonce is found by the shared `cuda_check_hash`
/ `cuda_check_hash_suppl` pass.

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060,
`ptarget[7]=0x000F`): 0 does-not-validate / 0 CUDA errors. Live pool run owed.
