# sha256d family (sha256d / sha256t / sha256csm)

Guideline: `docs/coding-guideline.md`

## Provenance

- `cuda_sha256d.cu` — KlausT-lineage optimized double-SHA256 kernel (backported in
  `bff177c`, not tpruvot's). Contains 4-round host prehash, preextend, constant
  padding folded into K immediates, 32-nonce-per-thread ILP and last-round
  elision (`h == 0xa41f32e7` ⇔ h7 + H7 == 0, i.e. exactly 32 leading zero bits;
  GPU-side share target is fixed at diff 1, the CPU re-hash does the real
  `fulltest`). This kernel is the **donor** for the shared header's optimized
  building blocks and is intentionally not rewritten onto the generic header
  primitives.
- `cuda_sha256t.cu`, `cuda_sha256csm.cu` — tpruvot 2017 lineage; since the
  2026-07 migration they consume `cuda/sha256_device.cuh`
  (`sha256_transform_full` / `sha256_final_to_target`) instead of private
  round-body copies.
- `cuda_sha256_selftest.cu` — init-time unit test of the shared header against
  OpenSSL (guideline §7 layer 1), called from `sha256d_cpu_init`/`sha256t_cpu_init`/
  `sha256csm_cpu_init`.
- SHA256Dv (Veil) is a distinct algo and lives in `algos/sha256dv/`.

## Optimization state (2026-07)

- `sha256t`/`sha256csm` hash 1 now resumes at round 4 via
  `sha256_transform_80_from_pre4` (host prehash `sha256_prehash_split_host` +
  preextend `sha256_preextend_w3_host`, KlausT-style, all in the shared
  header); K comes straight from `c_sha256_K` (constant-memory operands fold
  into the ALU instructions). The former `__shared__ s_K` staging — which also
  lacked a `__syncthreads()` after the fill — is gone.
- `sha256t`/`sha256csm` process 8 sequential nonces per thread at 256
  threads/block (donor-style loop; swept on RTX 3060: NPT 8 beats 1/4/16/32,
  TPB 256 beats 128/512 — the donor's own 512/32 shape loses on these
  multi-transform kernels). K-immediate folding is already achieved by
  constant propagation through the inlined header blocks.
- Still open: the remaining sha256-chained consumers (skeincoin, lbry,
  myr-gr) onto the header building blocks.

## Benchmarks

| Date | Card | Driver | CUDA | Algo | Intensity | Hashrate | Notes |
|---|---|---|---|---|---|---|---|
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256d | default (25) | 1380.7 MH/s | baseline before shared-header migration |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256t | default (23) | 829.7 MH/s | baseline before shared-header migration |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256d | default (25) | 1397.9 MH/s | after shared-header migration (02017c8) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256t | default (23) | 846.3 MH/s | after shared-header migration (02017c8) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256csm | default | 1295.2 MH/s | after shared-header migration (02017c8) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256t | default (23) | 873.1 MH/s | round-4 prehash + preextend, s_K removed |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256csm | default | 1395.8 MH/s | round-4 prehash + preextend, s_K removed |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256t | default (23) | 876.6-895.7 MH/s | + TPB 256, 8 nonces/thread (spread = warm vs cool card) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256csm | default | 1408.2 MH/s | + TPB 256, 8 nonces/thread |
