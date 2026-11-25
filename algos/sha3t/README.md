# sha3t  - triple NIST SHA3-256

Guideline: `docs/coding-guideline.md`

`hash = SHA3-256^3( header80 )`  - NIST FIPS 202 padding (0x06). CPU
reference / share re-verify: `sph/sha3.c` (tiny-sha3, `sha3_init/final`).

Bitcoin III mined sha3t only up to block 64 360; from there it is BC3Hash,
`-a bc3hash` (see `algos/bc3hash/README.md`).

## Provenance

- `cuda_sha3t.cu`  - Pkules donor kernel, already sp-style optimized when it
  arrived: first-round absorb precompute (`c_sha3t_mid[17]`/`c_sha3t_msg[6]`
  cover the constant 72 header bytes; only the nonce lane is folded
  per-thread). Launch shape retuned 2026-07-12 to 128 threads/block, 1
  nonce/thread (donor's 512/1/2 loses ~1-2%; parameterized via TPB/BPM/NPT
  defines). The blocks-per-SM bound is per-arch since 2026-09-16: 6 on sm_61,
  5 elsewhere. Pascal compiles this kernel to 96 registers where Ampere needs
  80 and so runs at a lower occupancy; 6 blocks/SM caps it to 80, spill-free.
  A tighter 8 blocks/SM reaches 50% occupancy but only via 64 registers plus a
  spill, and measures 1.6% slower on sm_61 and ~3% slower on Ampere.
- Since 2026-07 the round body / full permutation come from the shared
  `cuda/keccak_device.cuh` (bit-identical extraction); the sub-sm_61
  launch-shape branches were deleted per the arch floor.
- Init-time self-test: `algos/keccak/cuda_keccak_selftest.cu`, called from
  `sha3t_cpu_init`.
- The candidate report keeps the two lowest nonces via paired `atomicMin`
  (2026-09-16). The host resumes from `max(slot0, slot1) + 1`, so reporting
  the lowest is what keeps the cursor from stepping over an unreported
  candidate; the previous form left slot 1 as a non-atomic store.

## Benchmarks

| Date | Card | Driver | CUDA | Intensity | Hashrate | Notes |
|---|---|---|---|---|---|---|
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (25) | 247.9 MH/s | baseline before shared-header migration (98% of keccak256-alone per-permutation rate) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (25) | 267.5 MH/s | after shared-header migration (+7.9%) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (25) | 256.4-261.5 MH/s | + launch shape 128/5/1 (warm card; beat donor 512/1/2 by ~1-2% in three back-to-back orderings) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | 25 | ~250-263 MH/s | **live, zpool sha3-256t: accepted 4/4, 0 rejects** (post-migration kernel, 128/5/1) |
| 2026-09-16 | GTX 1080 Ti | 580.173.02 | 11.8 | default (25) | 323.5 MH/s | first Pascal measurement, 5 blocks/SM; cool card, 1898 MHz sustained |
| 2026-09-16 | GTX 1080 Ti | 580.173.02 | 11.8 | default (25) | 379.6 MH/s | 8 blocks/SM on sm_61: +17.3%, +13% perf/W (interleaved, cooldown-gated, medians); superseded below |
| 2026-09-16 | GTX 1080 Ti | 580.173.02 | 11.8 | default (25) | 385.6 MH/s | 6 blocks/SM on sm_61 (shipped): +19.2% over stock, +1.6% over 8 blocks/SM, spill-free. Two cooldown-gated A-B-B-A runs (+1.59%, +1.65%) |
| 2026-09-16 | RTX 3060 | 595.95 | 11.8 | default (25) | 271.5 MH/s | 5 blocks/SM; 8 blocks/SM measures -3.0% here, hence the per-arch bound |
| 2026-09-16 | GTX 1080 Ti | 580.173.02 | 11.8 | 25 | ~310 MH/s | live, zpool sha3-256t: accepted 60/60, 0 rejects, share diffs 0.52-24.2 (measured at 8 blocks/SM) |

Launch-shape sweep 2026-07-12 (warm card, within-sweep comparable):
**128/5/1 wins** 261.5; 256/2/2 258.5; 128/5/2 257.5; 512/1/1 257.1;
512/1/2 (donor) 256.9  - confirmed by a direct A/B pair 256.4 vs 254.1.
Same winner as sha3d. **Rejected after measuring:** `keccak_final_lane3`
truncation of pass 3 (253.6 vs 256.9  - the full permutation schedules
better here; sha3d keeps its truncation, where it measured neutral); and
partial unrolling of the round loops, which costs 4.6-9.9% at every setting
tried (8/4/2/1)  - full unroll is the optimum for this kernel.

Blocks-per-SM is the one launch parameter that differs by architecture, and
the two cards disagree in sign, so it must not be collapsed to one constant:
8 blocks/SM is +17.3% on sm_61 and -3.0% on sm_86. sm_75 is untested and
keeps the default.
