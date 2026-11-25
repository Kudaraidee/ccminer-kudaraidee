# ghostrider (`-a ghostrider`)

GhostRider (Raptoreum) proof-of-work.

The chain runs a header-determined order of **15 core 512-bit hash algos**
interleaved with **3 CryptoNight-v1 rounds**, arranged in three groups of
(5 core + 1 CN). Both the core order (15 distinct algos) and the CN triple
(the first 3 of 6 CN-v1 variants) come from header bytes `[4,36)`, so they are
constant for a whole job and a batch shares one pipeline.

## Layout

- `ghostrider.cu`: the driver: core/CN order derivation, the 512-bit core-stage
  pipeline, GPU orchestration, startup guards and CPU re-verify.

Dependencies are resolved by include path and by link-time externs:
- the shared x-family stage library via `#include "algos/common/cuda_x_stages.h"`
  and the `sph/*` headers;
- the CryptoNight-v1 GPU/CPU paths via `extern "C"` to the `*_gr` functions in
  `algos/cryptonight/cryptonight-core.cu`, `.../cryptonight-extra.cu` and
  `.../cryptonight-cpu.cpp` (shared with `mike` and `flex`).

Registered in `ccminer-kudaraidee.vcxproj` and `Makefile.am`.

## Diagnostics

| Env var | Effect |
|---|---|
| `GR_GUARD=1` | add the full GPU-vs-CPU chain check to the startup self-test |
| `GR_VERIFY=1` | audit the candidate screen against a host-side compare every batch |
| `GR_VERIFY=2` | same, plus corrupt one host digest to prove the audit can fire |
| `GR_VERIFY=3` | same as 1, plus re-hash a sample of each header's lanes on the CPU (`GR_VERIFY_LANES`, default 64) |
| `GR_VERIFY=4` | same as 3, plus flip one sampled digest to prove the check can fire |
| `GR_WORK_BUDGET` | batch size in lane x CN-work units (default 16384); `-i` overrides |
| `GR_LANE_CAP` | upper bound on lanes per batch (default 16384) |
| `GR_BENCH_TARGET` | override the relaxed `--benchmark` screen target |
| `CN_MIDBATCH_ABORT=0` | do not stop a batch between CN groups on a job change |

`--benchmark` sweeps all 20 CN rotations (4 s each) and reports the mean, because
a synthetic header otherwise pins one cheap chain; `-a all` times the single
mean-cost rotation instead.
