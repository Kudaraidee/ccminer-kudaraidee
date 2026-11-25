# keccak256 — single Keccak-256 (Maxcoin et al.)

Guideline: `docs/coding-guideline.md`

`hash = Keccak-256( header80 )` — original 0x01 Keccak padding (*not* NIST
SHA3 0x06). CPU reference / share re-verify: `sph/keccak.c`
(`sph_keccak256*`).

## Provenance

- `cuda_keccak256.cu` — alexis-lineage optimized kernel: first-round absorb
  precompute (`c_mid[17]`/`c_message48[6]`), grid-stride 2-nonces-per-thread
  at 1024 threads/block, and a truncated final round (only lane 3 of round
  23 is computed, feeding the 64-bit target compare). This kernel is the
  **donor** for the shared `cuda/keccak_device.cuh` round body,
  `keccakf1600_full` and `keccak_final_lane3`; since 2026-07 it consumes the
  header instead of private copies.
- `Algo256/cuda_keccak256_sm3.cu` (sm_30-era compat kernel and its
  `use_compat_kernels` selector in the scanhash wrapper) deleted 2026-07 —
  below the sm_61 arch floor.
- `cuda_keccak_selftest.cu` — init-time unit test of the shared header
  against the sph references plus pinned external KATs (NIST SHA3-256("")
  and Keccak-256("") digests) and a flipped-bit negative test; called from
  `keccak256_cpu_init` / `sha3d_cpu_init` / `sha3t_cpu_init`.
- The absorb-precompute block is intentionally duplicated in the consuming
  kernels (it reads per-algo `__constant__` symbols); keep the copies in
  `cuda_keccak256.cu`, `algos/sha3t/cuda_sha3t.cu` textually in sync.

## Benchmarks

| Date | Card | Driver | CUDA | Intensity | Hashrate | Notes |
|---|---|---|---|---|---|---|
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (23) | 757.0 MH/s | baseline before shared-header migration (family per-permutation reference) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (23) | 773.7 MH/s | after shared-header migration (+2.2%) |
