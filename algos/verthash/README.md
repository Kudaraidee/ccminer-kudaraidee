# Verthash (Vertcoin, VTC) — `-a verthash`

Memory-hard proof-of-work used by Vertcoin since 2021 (replaced Lyra2REv3).
Standard Bitcoin-fork mining path: 80-byte header, 32-bit nonce at `data[19]`,
generic stratum/getwork and submit. The one new capability versus other algos is
a fixed **~1.19 GiB mining data file (`verthash.dat`)** resident in VRAM.

## The data file

Verthash reads from a single deterministic data file, identical for every miner,
generated once and reused forever (no epochs, no per-block regeneration). The
current Vertcoin file is **1,283,457,024 bytes**, so `mdiv = ((size-32)/16)+1 =
80,216,063`, and its SHA-256 is
`a55531e843cd56b010114aaf6325b0d529ecf88f8ad47639b6ededafd721aa48`.

Point the miner at it with `--verthash-data <path>` (default: `verthash.dat` in
the working directory). If the file is present it is loaded, digest-verified,
uploaded once into VRAM, and reused. The data file is a user asset and is **never
committed** to this repository.

Generating it (if you do not already have one) takes several minutes of CPU +
disk. Use the one-shot flag:

```
ccminer --generate-verthash-dat --verthash-data verthash.dat
```

This (re)creates the file at the `--verthash-data` path (default `verthash.dat`),
verifies its SHA-256 against the canonical digest, and exits — no pool needed.
The generator is `verthash_generate_data_file()` in `verthash-data.cpp` (ported
graph construction). Prefer copying an existing verified file if you have one;
generation is slow.

## Algorithm

Per nonce:
1. `hash[32] = SHA3-256(header[80])` — the running 32-byte output.
2. `subset[512] = 8 × SHA3-512(header)` with `header[0]` incremented by `i+1`
   for `i = 0..7` (the prehash-72 / final-8 split: the 72-byte first block is a
   per-job constant, only the 8-byte nonce block differs per nonce).
3. `acc = 0x811c9dc5`; then 32 rotations × 128 iterations = **4096 random 32-byte
   reads** into the data file: `idx = fnv1a(rol32(subset[i], r), acc) % mdiv`,
   fold the 8 read words into `acc` (fnv1a chain) and into `hash[j]`.

Memory-latency/bandwidth bound, like KawPoW.

## Implementation

| File | Role |
| :-- | :-- |
| `verthash.cu` | ccminer bridge: `scanhash_verthash`, per-device init/free, host re-verify |
| `cuda_verthash.cu` | device kernels: 8×SHA3-512 precompute, SHA3-256, 4-lane IO/mix |
| `cuda/sha3_device.cuh` | shared FIPS-202 Keccak-f[1600] + SHA3-256/512 (0x06 pad) |
| `verthash-cpu.c` | scalar CPU oracle `verthash_hash_oracle` (authoritative re-verify) |
| `verthash-data.cpp` | data-file load / SHA-256 verify / mdiv / generator |
| `tiny_sha3.c` | FIPS-202 host reference (Saarinen tiny_sha3) |

The GPU IO kernel runs **4 threads per nonce**: each lane reads a `uint2` (8 of
the 32 bytes) per iteration, the seek index comes from a shared 512-byte subset,
and the fnv1a accumulator is synchronised across the 4 lanes through shared
memory each iteration. `mdiv` is a runtime `__constant__`. The kernel flags any
nonce whose most-significant hash word is `<= target[7]` (a safe superset filter —
never misses a real share); **every candidate is re-hashed on the host with the
CPU oracle + `fulltest` before submit**, so a kernel bug can only ever cost a
local reject, never a bad share. An init-time self-test hashes 256 nonces on both
GPU and CPU (fail-closed) with a negative control.

Submit byte-order: Verthash hashes the raw nonce counter little-endian, so the
stratum submit nonce is big-endian (same as odocrypt); ntime stays little-endian.

## Provenance

- CUDA kernels ported from **VerthashMiner** (CryptoGraphics, GPLv2,
  `src/vhCuda/verthash.cu`).
- CPU oracle + data-file generator from **cpuminer-opt** (`algo/verthash`, GPLv2).
- `tiny_sha3` — Markku-Juhani O. Saarinen, public domain (FIPS-202).

GPL-3.0-or-later (compatible with the GPLv2-or-later upstreams).

## Benchmark log

- **RTX 3060 (sm_86), CUDA 11.8, driver via NVML** — live on zpool
  (`verthash.na.mine.zpool.ca:6144`, `-p c=BTC`): **accepted 9/9, 0 rejects**
  (diff 0.2), **~755 kH/s** at default intensity (2026-07-21). Init self-test
  (GPU==CPU on 256 nonces + negative control) passes.
- **Reference parity:** the upstream **VerthashMiner** on the *same* card / pool /
  datafile does **~745–760 kH/s** (2026-07-21) — i.e. this port matches the
  reference miner. Confirms ~755 kH/s is the RTX 3060 ceiling for Verthash (both
  hit the same DRAM-random-access wall); no performance deficit.
### Optimization (Phase 6) — measured the wall

The IO loop is **4096 dependent random 32-byte reads** into the 1.28 GiB datafile
with zero locality → **DRAM-random-access bound** (~94 GB/s of scattered reads,
well under the card's ~360 GB/s sequential peak). Two levers were measured and
neither moved the needle on the RTX 3060:

- **Warp-shuffle accumulator sync** (replacing the per-iteration shared-memory +
  `__syncthreads` cross-lane fold): perf-neutral. **Kept** — it is simply cleaner
  (no block barriers, `verthash_gpu_io` drops from 79→lower registers path aside).
- **Splitting the SHA3 subset into its own kernel** to cut the IO kernel to 28
  registers / 0 shared and lift occupancy 50%→67%: measured **slightly slower**
  (the extra kernel launch + global subset round-trip cost more than the
  occupancy bought). **Reverted.**

Conclusion: throughput does not scale with `-i` (i15 == i20) or with occupancy —
the dependent random-read latency is the wall, matching the KawPoW memory-hard
finding. ~755 kH/s is at the card's random-access ceiling for this access
pattern; further gains would need a fundamentally different memory layout.
