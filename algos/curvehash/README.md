# curvehash — CurvehashCoin elliptic-curve PoW

## Provenance

New port (2026-07). A full CUDA implementation with a host CPU oracle used for
self-test and per-candidate re-verification.

- **Consensus (authoritative):** CurvehashCoin `src/curvehash.cpp` — `CurveHash`.
- **Miner-side reference (endianness / scanhash):** termux-miner
  `algo/curvehash.c` (`scanhash_curvehash`, self-contained SHA-256).
- **secp256k1:** vendored under `secp256k1/` (the same libsecp256k1 the
  reference miner bundled), compiled as a single translation unit by
  `secp256k1_unity.c`.

## Consensus definition

```
phash = SHA256( header[0..75] || nonce_le[4] )        // single SHA-256 over 80 bytes
for round in 0..7:                                    // exactly 8 rounds
    pubkey  = secp256k1_ec_pubkey_create(phash)        // phash (32B) IS the private key
    pub[65] = serialize_uncompressed(pubkey)           // 0x04 || X_be[32] || Y_be[32]
    phash   = SHA256( pub[0..64] )                      // single SHA-256 over 65 bytes
output = phash                                          // compare phash (MSW-first) to target
```

- Per nonce: 1×SHA-256(80B) + 8×( secp256k1 fixed-base scalar-mult + affine +
  SHA-256(65B) ). The eight scalar-mults dominate.
- Consensus asserts `pubkey_create == 1`; an invalid seckey (`phash == 0` or
  `phash >= n`) can never be a valid block (≈2⁻¹²⁸). The host path treats such
  a nonce as non-winning (never fabricates a hash).
- Nonce: 32-bit at `pdata[19]`, byte-swapped into the big-endian header exactly
  like the rest of the header words — same shape as sha256d. Generic stratum
  path, default (LE) submit byte order.
- Compare is MSW-first (`hash[7]` vs `ptarget[7]`), `fulltest`.
- **scanhash MUST set `work->nonces[0]` and `work->valid_nonces`** (not just
  `pdata[19]`): ccminer's submit path overwrites `pdata[19]` with
  `work.nonces[0]` right before `submit_work` (ccminer.cpp `nonceptr[0] =
  work.nonces[0]`). Setting only `pdata[19]` makes every share submit nonce 0 →
  the pool recomputes curvehash with nonce 0, which misses target → "Invalid
  share".

## Vendored secp256k1 config (`secp256k1_unity.c`)

Portable / MSVC-safe unity build — **no `__int128`, no build-time
`gen_context`**:

- `USE_FIELD_10X26` + `USE_SCALAR_8X32` — 32-bit limbs, pure C.
- `USE_NUM_NONE` — no GMP dependency.
- `USE_FIELD_INV_BUILTIN` / `USE_SCALAR_INV_BUILTIN` — self-contained modinv.
- `USE_ECMULT_STATIC_PRECOMPUTATION` intentionally **undefined** → the
  `ecmult_gen` table is built at runtime in `secp256k1_context_create`, so no
  precomputed static table / codegen step is needed. A `SECP256K1_CONTEXT_SIGN`
  context (needed only for `pubkey_create`) is created once per mining thread.

Build wiring: `curvehash.cpp` + `secp256k1_unity.c` are registered in
`ccminer-kudaraidee.vcxproj`/`.filters` (the unity TU carries a per-file
`AdditionalIncludeDirectories=algos\curvehash\secp256k1` so the internal
`#include "include/secp256k1.h"` resolves) and in `Makefile.am` (with
`-I$(top_srcdir)/algos/curvehash/secp256k1` in `ccminer_CPPFLAGS`).

## Implementation notes

- `scanhash_curvehash` re-hashes every candidate with the same authoritative
  `curvehash_80` and runs `fulltest` before returning — a hashing bug can only
  cause a local reject, never a bad share.
- Init-time self-test, **fail-closed**: a GPU that cannot reproduce the consensus
  hash can only produce local rejects, so a failure logs each leg and exits
  instead of mining. Four legs:
  - `tblbuild` — sha256d of the host-built 512 KB G-table against a known
    constant, so every one of its 8192 entries is checked. A single corrupt entry
    would cause only rare wrong hashes, which a digest KAT cannot detect.
  - `tblup` — the table read back from device memory against the host buffer.
  - `kat` — three header/digest vectors: the fixed 80-byte header `00 01 .. 4f`
    must give `b2645416ce97cf3935592d82eaebf25212008ebf04f62373203a7153fa1e1466`,
    plus the all-`00` and all-`ff` edges. Digests come from an independent
    textbook-secp256k1 + hashlib oracle.
  - `neg` — a one-bit header flip must change the digest, computed on the device,
    so the comparison cannot be vacuous.

  Each leg was confirmed to fire by deliberately breaking the build it guards
  (a corrupted table byte, a corrupted SHA-256 round constant): the miner logs
  e.g. `self-test FAILED (tblbuild 0 tblup 1 kat 111 neg 1)` and refuses to
  start.

## GPU implementation

`-a curvehash` (alias `curve`) runs the full hash on the GPU, one thread per
nonce. The device secp256k1 stack is a straight port of the vendored
libsecp256k1, split into header-only pieces under `cuda/`:

- `secp256k1_field_device.cuh` — the 10×26 field mod `p = 2²⁵⁶−2³²−977` (uint32
  limbs; NVCC on Windows has no `__int128`, so the 5×52 representation is out).
- `secp256k1_group_device.cuh` — Jacobian point arithmetic (double, mixed-add,
  to-affine) in the variable-time variants (mining needs no constant-time).
- `secp256k1_ecmult_gen_device.cuh` — fixed-base `k·G` by an **8-bit window** (32
  windows, one 512 KB precomputed table, no blinding). `k·G = Σⱼ table[j][kⱼ]`,
  so a scalar-mult is 32 point-adds and one affine conversion.
- `curvehash_device.cuh` — device SHA-256 (80- and 65-byte shapes) and the
  8-round loop.

`curvehash.cu` holds `scanhash_curvehash`: it uploads the 76-byte base header,
each thread appends its nonce, computes the hash, and screens `hash[7]` against
the target high word with an `atomicMin` on the winning nonce. `curvehash.cpp`
builds and uploads the G-table at init (via libsecp256k1) and provides the host
oracle and per-candidate re-verify.

The kernel is compute-bound (elliptic-curve, tiny per-thread state) and
register/occupancy-limited on sm_86. It launches at `tpb 512` under
`__launch_bounds__(512, 1)` (a 128-register cap) to raise occupancy, with a
default throughput of `1<<16` to fill the GPU at that block size (`-i` to tune).
The per-round field inversion uses **safegcd** (Bernstein-Yang divsteps, a device
port of modern libsecp256k1's `modinv32`) rather than the `p-2` addition chain the
vendored 2015 tree ships; that is worth ~1.4x on the full hash. Only the
constant-time variant is ported — the `_var` variants are faster on a CPU but
their data-dependent trip counts become warp divergence on a GPU.
`-DCURVEHASH_INV_ADDCHAIN` restores the addition chain. Cross-thread batching of
the inversion was evaluated and found slower at this occupancy, so each lane
inverts on its own.

## Testing

The device stack is validated bottom-up against an independent oracle before
being trusted: the field ops, the fixed-base `k·G` affine (X, Y) (against
`secp256k1_ec_pubkey_create`), and the full hash are each checked bit-for-bit on
the GPU, and each check is confirmed non-vacuous (a one-byte flip must fail it).
The init-time self-test above repeats the key checks on every start.

## Benchmarks

RTX 3060 (sm_86), Windows / MSVC Release x64, unless the row says GTX 1080 Ti
(sm_61, Linux build). Pool rows were run on zpool `curve.na.mine.zpool.ca:4633`
(accepted, 0 rejects).

| date | rate | notes |
|------|------|-------|
| 2026-07-17 | ~2.8 kH/s (1 CPU thread) | CPU oracle path; accepted 1/1. |
| 2026-07-17 | ~1.09 MH/s (GPU) | first GPU kernel (4-bit window); accepted 1/1. |
| 2026-07-18 | ~1.65 MH/s (GPU) | 8-bit window (512 KB table), half the point-adds; accepted 11/11. Warm 69 °C / 144 W / 11.4 kH/W. |
| 2026-07-18 | ~1.85 MH/s (GPU) | occupancy tuning (`__launch_bounds__(512,1)`, tpb 512, throughput 1<<16); accepted 6/6, ~1846–1857 kH/s. Warm 80 °C / 158 W / 11.6 kH/W. |
| 2026-08-11 | ~2.70 MH/s (GPU) | safegcd inversion replaces the `p-2` addition chain (~1.4x, measured interleaved in both orders); accepted 26/26, ~2662–2719 kH/s. 69 °C / 155 W / 17.3 kH/W. Launch shape re-checked against tpb 384/448/640 and kept. |
| 2026-08-16 | ~4.0 MH/s (GPU) | 16-bit window (64 MB table), 1.6x; accepted 45/45, 25.0 kH/W. |
| 2026-10-23 | ~5.0-5.3 MH/s live, 5.3-5.6 bench | 8 x 32-bit field on PTX carry chains (1.31x in an interleaved A/B); accepted 29/29. |
| 2026-10-23 | GTX 1080 Ti: ~2.05 MH/s bench | same field change (1.60x in an interleaved A/B); accepted 28/28. |
| 2026-10-24 | GTX 1080 Ti: ~2.91 MH/s bench, ~2.6 live | word-domain hash chain, 320 x 2 launch shape, 2^18 batch; accepted 66/66. |
| 2026-10-24 | ~6.1 MH/s bench, ~6.0 live | word-domain hash chain, 128 x 4 launch shape (1.09x); accepted 43/43. |
| 2026-11-09 | ~7.0 MH/s bench, ~7.0 live | field products with every multiply pair register-aligned (1.25x in an interleaved A/B); accepted 99/99. |
| 2026-11-09 | GTX 1080 Ti: ~3.08 MH/s bench | same change (1.055x in an interleaved A/B); pool not run. |
