# sha512256d — double SHA-512/256

## Provenance

New CUDA port (2026-07).
CPU reference (donor): cpuminer-opt `algo/sha/sha512256d-4way.c` (scalar
`scanhash_sha512256d`, `H512_256` IV). Host skeleton copied from
`algos/sha256d/sha256d.cu`; device primitives are the new shared library
`cuda/sha512_device.cuh`.

## Consensus definition

- `hash = SHA512/256( SHA512/256(header80) truncated to 32 bytes )`,
  truncated to 32 bytes (state words 0..3, big-endian serialized).
- SHA-512/256 = FIPS 180-4 SHA-512 core (80 rounds, K512) seeded with the
  SHA-512/256 IV (`22312194FC2BF72C ...`) — **not** SHA-512 truncated with
  the plain IV.
- Nonce: 32-bit at `pdata[19]`, big-endian in the header, exactly like
  sha256d. Generic stratum path, default (LE) submit byte order.
- 80 bytes fit a single 128-byte SHA-512 block: each nonce = exactly
  2 transforms.

## Implementation notes / quirks

- The kernel screens with `cuda_swab64(q3) <= target_q3` (the share value's
  high qword in fulltest word order); the host recomputes the full double
  hash via `sph_sha512` + the IV-override trick and runs `fulltest` before
  submit — a kernel bug can only cause local rejects, never bad shares.
- Truncation to 256 bits is an output rule only: both hashes run all 80
  rounds + full feed-forward. A truncated final round on hash2 would be legal
  but measured slower — see the optimization log.
- Init-time self-test (`cuda_sha512256d_selftest.cu`), once per device: FIPS
  "abc" KAT, host-vs-sph cross-check, host resume-path vs full transform,
  GPU-vs-host header double hash, plus a negative test (flipped bit must
  change the digest). Silent on success; on failure it logs the failing legs
  and **the miner refuses to start** — a GPU that cannot reproduce the
  consensus hash can only ever produce local rejects.
- Launch: TPB 256, 1 nonce/thread, default intensity 24 (`1U<<24`). All three
  were swept and none can be improved on — see the optimization log below.
  Per-job constants (`c_header`, `c_pre`) uploaded once per scanhash call.
- `-D` runs a **nonce-range differential** once per job: 4133 consecutive
  nonces (deliberately not a whole number of 256-thread blocks) are replayed
  on GPU and CPU and compared through two order-independent checksums,
  `xor(q3)` and `xor(q3 * (2*nonce+1))`. It covers what the candidate re-verify
  cannot — a nonce the kernel never hashed, or hashed under the wrong index.
  The second checksum is not redundant: a plain xor sum is permutation-blind.
  The weight must be injective and odd -- `nonce|1` is neither, because clearing
  bit 0 gives an aligned pair (2k, 2k+1) the same weight.
  The checksum kernel inlines the same hash body the mining kernel does.
- `--benchmark -D` logs every CPU-validated candidate (blue) — the only
  positive proof channel in benchmark mode (no submit; the API ACC counter
  never ticks there for any algo). Use it for every kernel A/B.

## Optimization log (RTX 3060 sm_86)

In-miner A/Bs are limited to ±5% here (clocks are not lockable without admin);
the 2026-08 entries were measured outside the miner, all arms interleaved in
one process, which resolves ~1–3%.

- **KEPT — per-job prehash (`sha512_prehash_split_host` +
  `sha512_transform_80_from_pre9`):** hash1 rounds 0..8 and the constant
  halves of round 9 hoisted to the host per job; kernel resumes at
  round 9's `+ w9`. Bit-exact (self-tested host+device vs full transform).
  Measured +1.7% at thermal steady state (508.9 vs 500.3 MH/s, consistent
  sign in all interleaved pairs); theoretically ~5% fewer rounds.
- **REJECTED — hash2 truncated final (q3-only, rounds 77..79 elided):**
  measured NEGATIVE (-1..-4%) in adjacent A/B pairs despite fewer rounds —
  the irregular 12-round tail hurts codegen more than 3 rounds save (same
  finding as the sha3 family). Code deleted; do not re-add without a
  measured win.
- **NO CHANGE — launch shape, re-measured 2026-08-13 at ~1–3% resolution**
  (all arms interleaved in one process, each gated bit-identical to the
  shipping launcher before timing):
  - *Default intensity, i20–i26:* i23–i26 are one plateau and the fastest arm
    swaps between runs, so they are inseparable; the shipped i24 is on it.
    Below the knee the loss is real: i22 −2.6%, i21 −4.7%, i20 −8.7%.
  - *TPB 64/128/256/512/1024:* nothing beats 256; 512 ties (same 58 regs);
    the rest are ~0.7% below in 3/3 runs. Note TPB 128 compiles to 56 regs and
    therefore reaches *more* occupancy (36 warps/SM vs 32) yet is slower —
    this kernel is issue-bound, so occupancy is not a lever.
  - *Nonces per thread 1/2/4:* ±0.5%, i.e. nothing. NPT costs 6 registers and
    spills nothing, so it was never register-blocked — it simply does not help
    a kernel already at the integer-issue limit.
- **REJECTED — donor kernel transcription (2026-07-13):** radifier's fully
  hand-unrolled Radiant kernel (`ccminer-radiator/cuda_rad.cu`, d40c089)
  was script-transcribed verbatim, proven bit-correct on GPU at every layer
  (host+GPU test programs vs `hashlib.sha512_256`; in-miner `r` matched; known
  candidate `255e95a6` found and CPU-validated), then A/B-measured:
  **490.4 vs 489.4 MH/s — +0.2%, a wash.** nvcc 11.8/sm_86 already extracts
  the donor's wins from the clean library kernel. Kept the 8× smaller one.
  Fold constant if ever revisited: `lo32(q3) = r + 0x247f2d73`; donor magic
  `0xdb80d28d` ⟺ `vhash[7]==0`.
- **Benchmark-window trap (cost hours — remember):** `--benchmark` rescans
  the same 2^30-nonce window forever (work regen resets the nonce). A
  diff-1 screen has ~0.25 expected hits/window → 78% chance of permanent
  zero hits, indistinguishable from a broken kernel. Zero benchmark hits ≠
  broken; validate by widening the screen to a population with a known
  member (target 0x03 has candidate `255e95a6` for the 0x55555555 header).

## Benchmarks

| date | card | driver/CUDA | intensity | hashrate | notes |
|------|------|-------------|-----------|----------|-------|
| 2026-07-12 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | ~530 MH/s | naive baseline (TPB 256, 1 nonce/thread), benchmark; 43/43 GPU candidates CPU-validated at target 0x03, 0 mismatches. 58–64 regs, 0 spills. |
| 2026-07-12 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | ~510 MH/s | **live-validated** (naive kernel): zpool `sha512256d.na.mine.zpool.ca:3342`, 4/4 accepted, 0 rejects, share diffs 5.90/4.71/2.51/2.17 — confirms header build, target compare and default (LE) submit byte order. Card warm: 76 °C, 147 W. |
| 2026-07-13 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | ~535 cool / ~505 hot MH/s | prehash kernel (lever A): +1.7% vs naive at matched thermal state; 44/44 benchmark candidates CPU-validated, 0 mismatches. |
| 2026-07-13 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | ~537 MH/s | **live-validated** (prehash kernel): zpool `sha512256d.na.mine.zpool.ca:3342`, 2/2 accepted, 0 rejects, share diffs 12.13/7.46. Cool card (short run). |
| 2026-08-13 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | 535 cool → 502 hot MH/s | **live-validated** (armed-once result buffer + shared hash body + fail-closed self-test): zpool `sha512256d.na.mine.zpool.ca:3342`, **8/8 accepted, 0 rejects**, stratum diff 2, share diffs 47.08/2.44/2.13/3.91/3.23/2.67/2.50/6.77. 1911 MHz, 144 W, 67 °C. No regression vs the 2026-07-13 kernel. |
| 2026-08-13 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | 557–563 MH/s | isolated launcher, no stratum thread or scan loop — not comparable to the in-miner figures above, only to itself. Kernel unchanged at 58 regs / 0 spills / 0 B stack. Nonce-range differential bit-exact over 16 777 253 nonces from a non-zero start. |
