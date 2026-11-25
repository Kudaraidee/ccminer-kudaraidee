# yescrypt (`-a yescrypt`, `yescryptr8`, `yescryptr16`, `yescryptr16v2`, `yescryptr24`, `yescryptr32`)

Yescrypt memory-hard proof-of-work, relocated from the repo root (`yescrypt/`).

Six parameterizations share one GPU implementation (each has its own
`ALGO_YESCRYPT*` enum, `scanhash_yescrypt*` and `yescrypt*_hash` entry point):

- `yescrypt`, `yescryptr8`, `yescryptr16`, `yescryptr16v2`, `yescryptr24`,
  `yescryptr32` -- differing in the yescrypt N/r/ROM parameters and client key.

## Layout

Relocation only -- every symbol is unchanged, so the dispatch wiring (`algos.h`,
`miner.h`, `ccminer.cpp`) is untouched.

- `yescrypt.cu` -- the dispatcher and CPU-reference driver for all six variants.
- `cuda_yescrypt.cu` -- the GPU kernel.

The yescrypt **CPU reference** lives in the shared `sph/` tree
(`sph/yescrypt-common.c`, `sph/yescrypt-opt.c`, `sph/yescrypt-platform.h`,
`sph/yescrypt.h`) and is used only by this algo. It was left in `sph/` -- the
`.cu` reaches it via the include path (`#include "sph/yescrypt.h"`), so the move
is source-transparent (no `../` includes, no external references to the old
`yescrypt/` path).

## Build

Both `.cu` are repointed in `ccminer-kudaraidee.vcxproj` + `.filters` (keeping the
`Source Files\CUDA\yescrypt` filter). The algo was previously Windows-only; for
parity with the Windows build it was added to `Makefile.am` (the two `.cu` plus
its `sph/yescrypt-*` sources/headers).


## Variants and measured throughput

RTX 3060 (28 SM, 12 GB), CUDA 11.8, `--benchmark`. Scratchpad is
`(520 + 2r(N + 16p)) x 4` bytes per nonce, so VRAM -- not the core count -- is what
limits the batch size on the larger parameterizations.

| `-a` | N | r | p | client key | H/s | VRAM at default |
|---|---|---|---|---|---|---|
| `yescrypt` | 2048 | 8 | 1 | *(none)* | 13 180 | 1.3 GB |
| `yescryptr8` | 2048 | 8 | 1 | `Client Key` | 13 260 | 1.3 GB |
| `yescryptr16` | 4096 | 16 | 1 | `Client Key` | 3 450 | 5.3 GB |
| `yescryptr16v2` | 4096 | 16 | 4 | `PPTPPubKey` | 840 | 5.4 GB |
| `yescryptr24` | 4096 | 24 | 1 | `Jagaricoin` | 2 463 | 7.9 GB |
| `yescryptr32` | 4096 | 32 | 1 | `WaviBanana` | 1 707 | 10.6 GB |

`-a yescrypt` takes its parameters from `--yescrypt-param` / `--yescrypt-key`; the
row above is its default (identical work to `yescryptr8` minus the client key).

### Intensity

The default batch size is `3 x (SMs x 128)` nonces, clamped to what VRAM allows, so
it adapts down automatically on smaller cards -- `yescryptr32` needs ~10.6 GB at the
default and will step down to `2 x` on an 8 GB card.

**`yescryptr8` is the one variant where the default leaves throughput on the
table.** Its scratchpad is only 131 KB per nonce, so VRAM permits far more
parallelism than the default asks for. On a 12 GB card:

| `-i` | nonces | H/s |
|---|---|---|
| default (~13.3) | 10 752 | 13 400 - 13 800 |
| **14.4** | **21 504** | **15 400 - 16 000** |
| 15.1 | 35 840 | 15 800 |
| 15.6 | 46 592 | 15 100 |

**`-i 14.4` is worth ~+15 % on `yescryptr8`** and costs 2.8 GB. It peaks there and
falls off above ~36 000 nonces. The other five variants are already at their
plateau -- raising `-i` on `yescryptr16` measured ~3 % *slower*, because the extra
scratchpad crowds VRAM without buying parallelism.

## Validation

All six variants are verified GPU-vs-CPU against the bundled yescrypt reference
(`sph/yescrypt-opt.c`): for each one, the nonce with the lowest digest in a scanned
range is hashed on the host and must match the nonce the GPU reports, and a target
one below that minimum must report nothing.

At runtime the host `yescrypt*_hash` re-hashes **every** GPU candidate and
`fulltest`s it before submit, printing `result does not validate on CPU!` on a
mismatch. That check is unconditional -- there is no force-accept path.

### Live pool

A KAT proves the hash; only a pool proves the submit path (header byte order, nonce
byte order in `mining.submit`, and the difficulty scale). Recorded per variant:

| Date | `-a` | Pool | Shares | Card / driver | `-i` | H/s | H/W |
|---|---|---|---|---|---|---|---|
| 2026-08-20 | `yescryptr32` | `yescryptR32.na.mine.zpool.ca:6343` | **10/10 accepted, 0 rejects** | RTX 3060 / 595.95 | 13.312 (10 752 nonces) | 1 808 | 15.5 at 75 W |
| 2026-08-19 | `yescryptr16` | `yescryptR16.na.mine.zpool.ca:6333` | **10/10 accepted, 0 rejects** | RTX 3060 / 595.95 | 13.312 (10 752 nonces) | 3 520 | 35.6 at 98 W |
| 2026-08-17 | `yescryptr16` | zpool | 6/6 accepted, 0 rejects | RTX 3060 | default | -- | -- |
| 2026-08-17 | `yescryptr32` | zpool | 10/10 accepted, 0 rejects | RTX 3060 | default | -- | -- |

Both 2026-08-19/20 runs: stratum difficulty 0.01 (job diff 0.005), ~20-45 s -- enough
to settle the submit path, **not** a stability soak. Read the steady-state rate, not
the first sample: r32's first report was 1 161 H/s and settled at ~1 808.

Live comes out at or above `--benchmark` for both variants -- 3 520 vs 3 450 (~2 %) and
1 808 vs 1 707 (~6 %) -- so for yescrypt the two measurement modes agree.

**`yescryptr32` reproduces its 2026-08-17 live rate of ~1 808 H/s exactly**, which is
the useful part: it is the variant with the largest recent codegen change and the only
one whose kernel still spills, so it is where a regression would have shown first.

The remaining four (`yescrypt`, `yescryptr8`, `yescryptr16v2`, `yescryptr24`) have no
live-pool record. `r24` and `r16v2` have no pool carrying them, so for those the KAT
and the unconditional host re-verify are the only available gates.

Efficiency scales with `r`, not with the hashrate: r16 gives 35.6 H/W at 98 W, r32
15.5 H/W at 75 W -- roughly the 2x work per hash, at lower draw.
