# argon2d (`-a argon2d500`, `-a argon2d1000`, `-a argon2d4096`, `-a argon2d16000`)

Memory-hard Argon2d proof-of-work. Four coins share this folder — a single
Argon2d over the 80-byte header (password = salt = header, outlen = 32),
differing only in the Argon2 cost parameters and version:

| algo         | coin               | m_cost (KiB) | lanes | t_cost | version |
|--------------|--------------------|--------------|-------|--------|---------|
| argon2d500   | Dynamic (DYN)      | 500          | 8     | 2      | 0x10    |
| argon2d1000  | Zero Dynamics Cash | 1000         | 8     | 2      | 0x10    |
| argon2d4096  | Argentum / Myriad  | 4096         | 1     | 1      | 0x13    |
| argon2d16000 | Alterdot           | 16000        | 1     | 1      | 0x10    |

## Layout

- `argon2d.cu` — the per-coin variant table (`argon2d_variant`), CPU reference
  hashes (`argon2d*_hash`, used to re-verify every GPU candidate before
  submit), the shared `scanhash_argon2d` driver and the init-time self-test.
- `argon2d_fill.cu`, `blake2b_kernels.cu` — the GPU fill and BLAKE2b kernels.
- `argon2d_kernel.h`, `cudaexception.h` — device-side headers.
- `argon2ref/` — the vendored Argon2 reference implementation. `argon2.c`,
  `core.c`, `encoding.c`, `opt.c`, `thread.c` and `blake2/blake2b.c` are built;
  `run.c` / `test.c` / `bench.c` / `genkat.c` / `ref.c` ship with the upstream
  reference but are not compiled here.

> Note: `rinhash/argon2d_device.cuh` is a separate, rinhash-owned Argon2d device
> header and is unrelated to this folder.

## Variant coexistence (runtime geometry)

The kernels used to be compiled for exactly one `(m_cost, lanes, t_cost,
version)` tuple via `ALGO_*` enum constants, so only one coin per build could
be GPU-correct. The full parameter set (m_cost, lanes, passes, version,
total/segment blocks) is now passed to `argon2_initialize` / `argon2_finalize`
as runtime kernel arguments — the heavy `argon2_fill` kernel always took it
that way — so all variants are GPU-correct in one binary at no measurable cost
(the parameterized kernels are a tiny fraction of the pipeline).

Two device-code subtleties this required:

- **lanes < 8**: the fill kernel's final-block epilogue (XOR of each lane's
  last-column block, staged in the shared cache) hardcoded 8 lanes and 256
  threads. It now XORs `lanes` rows with a stride loop over the block's
  `lanes*32` threads — bit-identical for lanes=8, and it is what made the
  lanes=1 coins (argon2d4096, argon2d16000) work on GPU at all. argon2d16000
  had never produced a valid GPU share in this fork before this fix (its
  false candidates were silently discarded by the host re-verify).
- **Last-column/cache aliasing race (donor-era wrong-hash bug, fixed)**: the
  donor kernel staged the final pass's last-column block into shared slot
  `c[lane][0]` for the epilogue XOR — the same slot that caches column 2.
  Cross-lane refs to column 2 are legal within the final slice, and there is
  no barrier until the slice ends, so a lane that finished early could serve
  another lane its final block instead of column 2 → a wrong (but
  deterministic-per-race-outcome) hash for that nonce. Symptoms: sporadic
  "does not validate on CPU!" on live pools (~0.3% of nonces affected;
  harmless for share validity thanks to the host re-verify, but lost work)
  present since the donor and invisible until the warning was added. Fix:
  the last column is stored to global like every other ≥8 column and the
  epilogue XOR reads it from global (after the block-wide `__syncthreads`);
  the column-2..7 cache is untouched. Proven by a 100-iteration same-batch
  stress harness: 86/100 iterations produced the same wrong candidate before
  the fix, 0/300 after, with legit candidates perfectly deterministic.
- **Dynamic shared-cache sizing** (the lanes=1 throughput fix, ~2.8×): the
  fill kernel's shared staging used to be a static 48 KB `c[8][6]`, which
  capped occupancy at 2 blocks/SM regardless of lanes — for 32-thread lanes=1
  blocks that is 64 threads/SM (~4%), the reason argon2d4096 ran at a third
  of what the card can do. The cache is now `extern __shared__`, sized
  `lanes * 6` blocks at launch: lanes=8 keeps its 48 KB (and its cache hits —
  columns 2..7 are 10% of argon2d500's refs; removing the cache outright
  measured −10%/−6% on 500/1000), lanes=1 takes 6 KB and reaches the
  16-blocks/SM hardware cap. Measured (RTX 3060): argon2d4096 13.6 → 38.0
  kH/s (2.8×, parity with CryptoDredge 0.26.0), argon2d16000 3.2 → 7.9 kH/s
  (2.5×), argon2d500/1000 unchanged. The fill kernel's per-kernel splits are
  loggable with `-D` (init/fill/final percentages every 20 batches).
- **Argon2 version 0x13** (argon2d4096): the version number is hashed into H0
  by `argon2_initialize`, so it must be per-variant. The v1.0-vs-v1.3
  difference inside the fill core (XOR-overwrite on passes after the first)
  never triggers at t_cost=1, so the fill kernel serves both versions; a
  future t_cost>1 v0x13 variant would need the version-conditional XOR there
  (see `argon2ref/core.c` `fill_block` `with_xor`).

## Provenance

Donor: `duality-solutions/Dynamic-GPU-Miner-Nvidia` (this fork's `argon2d/`
descends from it; `argon2d_fill.cu` was byte-identical before the
parameterization). Dynamic's chain params verified against the donor;
argon2d4096's (4096/1/1/v0x13) against cpuminer-opt's
`algo/argon2d/argon2d-gate.c` (`register_argon2d4096_algo`).

## Validation

- Init-time self-test: `argon2d500_hash`, `argon2d1000_0dync_hash` and
  `argon2d4096_hash` over a fixed 80-byte header must match digests from the
  independent official argon2 library (argon2-cffi, type=D, per-variant
  version), plus a one-bit-flip negative test proving the comparison isn't
  vacuous.
- The host `argon2d*_hash` re-verifies every GPU candidate before submit; a
  candidate that fails re-verify logs "does not validate on CPU!" plus an
  `argon2d replay:` line with the exact 80-byte input for offline analysis.
- Loosened-target benchmark battery (RTX 3060, 2026-07-18, re-run after the
  race fix): every GPU candidate CPU-validated, zero mismatches —
  argon2d4096 1210/1210, argon2d500 750/750, argon2d1000 705/705,
  argon2d16000 334/334; plus the 100-iteration same-batch stress on the three
  live-captured race inputs (0/200/100 candidates, all deterministic, zero
  wrong). Steady-state benchmark rates (warm card): argon2d500 ~189 kH/s,
  argon2d1000 ~85-88 kH/s, argon2d4096 ~36.4 kH/s, argon2d16000 ~7.6-7.9
  kH/s (the race fix costs <1% by traffic accounting).
- Live-validated on zpool (2026-07-18, RTX 3060, after the optimization and
  the race fix, all four variants in one session): argon2d500 22/22
  (~193 kH/s), argon2d1000 44/44 (~87 kH/s), argon2d4096 35/35 (~37.6 kH/s),
  argon2d16000 32/32 (~7.3 kH/s, the first live shares this variant ever
  produced on GPU in this fork) — 133 accepted, 0 rejects, 0 re-verify
  failures (the pre-fix kernel threw ~2 "does not validate" per minute on
  argon2d500). An earlier same-day session also validated argon2d4096 across
  a mid-run pool coin switch (two distinct chains).
