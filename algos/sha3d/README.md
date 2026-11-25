# sha3d — double NIST SHA3-256 (BSHA3 / Yilacoin)

Guideline: `docs/coding-guideline.md`

`hash = SHA3-256( SHA3-256( header80 ) )` — NIST FIPS 202 padding (0x06
domain separator), *not* 0x01 Keccak. CPU reference / share re-verify:
`sph/sha3d.c` (`sph_sha3d256*`).

## Provenance

- `cuda_sha3d.cu` — from `Algo256/cuda_keccak256_sha3d.cu`, the tpruvot-era
  "compat" kernel (full 25-lane absorb, two full permutations, lane-3 target
  compare). The scanhash wrapper used to force `use_compat_kernels = true`
  unconditionally: the "modern" path it never took wired the plain
  keccak256 kernel (0x01 padding, single permutation) and would have been
  wrong — that selector and the equally wrong pre-sm_35 single-permutation
  branch were deleted in the 2026-07 migration.
- Since 2026-07 the permutation comes from the shared
  `cuda/keccak_device.cuh` (alexis-lineage round body: `xor.b64` theta
  chains, LOP3 chi, PRMT byte-rotates); dual-nonce atomicExch result buffer
  like the sibling kernels; per-launch `cudaMemset`+`cudaDeviceSynchronize`
  replaced by armed-once output (re-armed per job / on reject).
- Init-time self-test: `algos/keccak/cuda_keccak_selftest.cu` (shared-header
  KAT + negative test), called from `sha3d_cpu_init`.

## Benchmarks

| Date | Card | Driver | CUDA | Intensity | Hashrate | Notes |
|---|---|---|---|---|---|---|
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (23) | 330.9 MH/s | baseline before migration (87% of keccak256-alone per-permutation rate) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (23) | 400.2 MH/s | shared-header migration: alexis round body, armed-once output buffer (per-launch cudaMemset+deviceSync removed), dual-nonce (+21%; cool card) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (23) | 395.9 MH/s | + absorb midstate precompute + truncated final round (both perf-neutral at this shape — kernel not ALU-bound — kept for structure) + launch sweep (warm card) |

Launch-shape sweep 2026-07-12 (TPB/blocks-per-SM/nonces-per-thread, matched
45 s runs, warm card): **128/5/1 wins** at 389.9; 128/5/2 385.2; 512/1/2
(sha3t donor shape) 382.3; 256/2/2 376.5; 512/1/4 373.8; 256/4/1 371.2;
1024/1/2 (keccak donor shape) 365.2. Donor configs don't transfer.
Per-permutation rate now ~800 Mperm/s ≈ sha3t parity (saturated).
