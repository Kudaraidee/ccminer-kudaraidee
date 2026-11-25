# x17 family (`-a x17`, `-a hmq1725`, `-a skydoge`)

Three algos that share the x17 stage set (tpruvot lineage, GPLv3).

```
x17 (fixed 17-stage chain):
Blake-512 (80-byte header)  →  BMW-512  →  Groestl-512  →  Skein-512  →
JH-512  →  Keccak-512  →  Luffa-512  →  CubeHash-512  →  Shavite-512  →
SIMD-512  →  Echo-512  →  Hamsi-512  →  Fugue-512  →  Shabal-512  →
Whirlpool-512  →  SHA-512  →  Haval-256 (terminal)
```

- `x17.cu` — the fixed 17-stage chain (`scanhash_x17`, `x17hash` CPU reference).
- `hmq1725.cu` — HMQ1725, a **per-nonce branching** variant (each stage picks
  one of two hashes by a data bit, walked through branch/merge compaction
  buffers). Not a fixed chain, so the 5-run fusion does not apply.
- `skydoge.cu` — a fixed-chain x17 variant (yiimp/cpuminer-opt lineage).

## Layout

Each dispatcher includes `algos/common/cuda_x_stages.h` and calls **every stage
by its bare `<prim>512` name** through that header's bridge (no `quark_`/`x11_`/
`x13_`/`x14_`/`x15_`/`x16_` prefixes remain). All stage launchers are the shared
ones in `algos/stages/`. Where the bare name maps to the optimised launcher this
also adopts it: 64-byte Shavite is the sp kernel (3-arg), the 64-byte Echo is the
alexis kernel, and the old combined `luffaCubehash` launch is split into the bare
`luffa512` + `cubehash512` stages (all bit-identical output, proven family-wide).

The two x17-specific stages moved to `algos/stages/` during this migration and
were **de-branded** to their bare `<prim>` names (real symbols):

- `cuda_sha512.cu` — 64-byte SHA-512 stage (`sha512_cpu_*`) plus the 80-byte
  `x16_sha512` head variant used by the x16 family. Built with
  `--maxrregcount=80`.
- `cuda_haval256.cu` — the Haval-256 terminal stage (`haval256_cpu_*`;
  `outlen` selects the 256- or 512-bit tail write).

Every caller (x17/hmq1725/skydoge here, and x21s) now calls these by their bare
names, so the transitional `x17_sha512_*` / `x17_haval256_*` forwarders were
removed — the de-brand is complete.

## Optimization

Both fixed chains adopt the shared register-resident fused kernel
(`algos/common/cuda_x_fused.*`), which keeps the 64-byte state in registers
across a run of consecutive stages instead of bouncing it through `d_hash`:

- **x17** fuses its skein→jh→keccak→luffa→cubehash run (the same run x13/x11
  use) into one launch, replacing 5 separate stage launches.
- **skydoge** has two consecutive fusible runs, split by the non-fusible
  groestl/simd/echo stages: skein→bmw and jh→luffa→keccak. Both are packed
  into a single uploaded stage-id list and launched as two fused calls
  (indexing each sub-run by `(start,len)`), replacing 5 launches with 2.
- **hmq1725** is per-nonce branching, so most of it can't fuse — but two
  consecutive all-nonce fusible pairs sit *between* its branch merge/filter
  points and touch only the common `d_hash`: jh→keccak and luffa→cubehash.
  Both are packed into one uploaded stage-id list and run as two fused calls
  (`(0,2)` + `(2,2)`), replacing 4 launches with 2. The eight per-nonce
  filter/merge branches and their branch-A/branch-B single stages are left on
  the per-stage launcher path (branch-A runs on `d_hash`, branch-B on a second
  buffer — they can't be register-fused).

The CPU reference hash and consensus self-test of each algo are unchanged, so
GPU output stays bit-identical. Haval is the terminal and the best nonce is
found by the shared `cuda_check_hash` / `cuda_check_hash_suppl` pass.

## Validation

The initial relocation (branded calls) was benchmark- and live-validated:
x17 ~13.2 MH/s (debug run: `found =>` fires ~24×/25s, every candidate passes the
`x17hash` CPU re-verify), hmq1725 ~6.36 MH/s, skydoge ~10.4 MH/s, all **0
does-not-validate / 0 CUDA errors** at `ptarget[7]=0x00ff`; **skydoge
live-confirmed on zpool (skydoge.na.mine.zpool.ca:7091): 4/4 accepted, 0
rejects**.

The subsequent call-site de-brand (sp shavite / alexis echo / split
luffa+cubehash) is byte-identical by construction but changes which kernels run.

The stage-fusion pass was rebuilt clean and validated:
- **skydoge — live-confirmed** on zpool (skydoge.na.mine.zpool.ca:7091):
  accepted with 0 rejects, steady ~11.0-11.2 MH/s (up from ~10.4 unfused).
  Benchmark is non-vacuous 0-does-not-validate: at `ptarget[7]=0x00ff` the
  `found =>` candidate path fires repeatedly and every found nonce passes the
  `skydoge_hash` CPU re-verify (0 mismatches, 0 CUDA errors).
- **x17 — benchmark-validated** (non-vacuous 0-does-not-validate): at
  `ptarget[7]=0x00ff` the `found =>` candidate path fires repeatedly and every
  found nonce passes the `x17hash` CPU re-verify (0 mismatches, 0 CUDA errors),
  ~13.3-13.4 MH/s. Live re-validation still owed.
- **hmq1725 — live-testing exposed a pre-existing bug, now fixed; live
  re-validation owed.** `--benchmark` is a *weak* correctness check for this
  chain: it re-scans one nonce window, so its `found =>` line re-finds a single
  nonce, and because hmq1725 is per-nonce **branching** that one nonce only
  walks one set of branch paths (for a fixed chain like x17/skydoge a single
  nonce still exercises every stage, so there the check is meaningful). A live
  run surfaced `does not validate on CPU!` on multiple nonces — a **pre-existing**
  GPU/CPU mismatch, unrelated to the fusion: the shared haval stage's 512-bit
  path passed the pre-haval high 32 bytes through, but hmq1725's CPU reference
  `memset`s them to zero before the next stage. Fixed with a zero-high haval
  variant (`haval256_cpu_hash_64z`); the pass-through path is retained for x21s,
  which needs it. **Fix live-confirmed on zpool (hmq1725.na.mine.zpool.ca:3747):
  3/3 accepted, 0 rejects, no `does not validate`, ~6.7 MH/s.**
