# x11 family

Guideline: `docs/coding-guideline.md`

## Provenance

- `x11.cu` — tpruvot/ccminer x11 (Dash lineage): the fixed 11-stage chain
  blake → bmw → groestl → skein → jh → keccak → luffa → cubehash → shavite →
  simd → echo. Migrated onto the shared x-family machinery: the 64-byte stages
  call the bare `<prim>512_cpu_*` device-launcher names through the
  `algos/common/cuda_x_stages.h` bridge (the launcher TUs still live in their legacy
  `quark/` / `x11/` folders; the bridge forwards until they de-brand), and the
  consecutive fusible run is executed by the shared fused kernel.
- The other x11-folder consumers (`c11`, `sib`, `fresh`, `timetravel`,
  `bitcore`, `x11evo`, …) and the per-stage launcher TUs (`cuda_x11_*.cu`) are
  untouched by this migration; they keep their existing prefixed names.

## Fused stage run

x11's order is fixed, so unlike x16r there is no per-hash-order run search: the
one maximal fusible run — **skein → jh → keccak → luffa → cubehash** (five
consecutive register-resident stages) — is executed by
`x_fused_fixed_cpu_hash_64`, a kernel compiled for this sequence (one launch,
64-byte state kept in registers instead of bouncing through `d_hash`). This
replaces the four separate launches of the old path (standalone skein/jh/keccak
plus the combined `x11_luffaCubehash512` kernel). blake is the 80-byte first
stage; bmw and groestl stay standalone (bmw is a length-1 run bounded by the
groestl boundary); shavite, simd and echo are boundary stages with their own
launchers. shavite uses the sp-optimised 64-byte launcher (bare
`shavite512_cpu_hash_64` → `cuda_x11_shavite512_sp.cu`: self-contained,
in-kernel AES-table init, vectorised `__ldg4` I/O — ~+2.5% over the shared
`c512` path, measured A/B; the legacy 6-arg `x11_shavite512_cpu_hash_64` stays
for its remaining consumers, evohash and ghostrider). simd is the sp-optimised
`+20%` kernel (`cuda_x11_simd512.cu`). echo uses the optimised alexis 64-byte launcher
(`echo512_cpu_hash_64`) with the tpruvot `*_compat` variant only on arch < 500
(below the sm_61 build floor — effectively dead).

The fused kernel's init-time device self-test (`x_fused_device_selftest`,
`cuda/xfamily_selftest.cu`) validates the fusible primitives against their
`sph_*` references on the GPU before the real order upload.

## Correctness note

Every GPU candidate is re-hashed on the host (`x11hash`) before submit, so a
kernel/thermal glitch can only ever cause a local reject
(`result … does not validate on CPU!`), never a bad share.

## Measured rates

| algo | card     | driver | CUDA | intensity | rate |
|------|----------|--------|------|-----------|------|
| x11  | RTX 3060 | 595.95 | 11.8 | 19        | ~19.3 MH/s (benchmark, fused + sp-shavite; ~+2.5% from sp shavite over the ~18.5 shared-c512 build, interleaved A/B, 0 CPU-validation failures); fusion **live-validated** (zpool, 23/23 accepted, 0 rejects, held rate even at 88 °C vs pre-fusion drift to ~17.6) |
