# x25x (SUQA / SIN)

`-a x25x` — the 25-stage SUQA/SIN hash (successor to x22i). Unlike the x16
family it is a **fixed-order accumulate-all** chain: each of 24 hashing stages
writes its own 64-byte slot (nothing is overwritten), then a 12-round byte
shuffle mixes all 24 slots and a final BLAKE2s over the whole 1536 bytes yields
the PoW output. Standard 80-byte header, default difficulty.

Chain: BLAKE-512 → BMW → Groestl → Skein → JH → Keccak → Luffa → CubeHash →
Shavite → SIMD → Echo → Hamsi → Fugue → Shabal → Whirlpool → SHA-512 →
**SWIFFTX** (reads the 256-byte window of slots 12–15) → Haval-256 → Tiger-192 →
LYRA2 (1,4,4) → Streebog → SHA-256 → **Panama** → **LANE-512** → shuffle →
**BLAKE2s** over all 24 slots.

## Provenance
CUDA port written for this fork; the CPU reference is cpuminer-opt
`algo/x22/x25x.c` (scalar path) by JayDDee. The 20 shared stages are this tree's
existing x-family launchers; five primitives are new here:
- `cuda_x25x_swifftx.cu` — SWIFFTX (Z₂₅₇ FFT compression), the compute tentpole;
  `FFT()` copied verbatim from the vendored scalar `swifftx.c`, the surrounding
  `SWIFFTFFT`/`SWIFFTSum`/`ComputeSingleSWIFFTX` rewritten for the device.
- `cuda_x25x_lane.cu` — LANE-512; tables + `lane512_compress` generated verbatim
  from the vendored `lane.c` (Indesteege).
- `cuda_x25x_panama.cu` — PANAMA-256 sponge (from `sph/sph_panama.c`).
- `cuda_x25x_blake2s.cu` — BLAKE2s over the 1536-byte accumulator.
- `cuda_x25x_shuffle.cu` — the 12-round `uint16` byte shuffle.

Vendored CPU references: `swifftx.{c,h}` and `lane.{c,h}` live here (scalar-only
compile — SWIFFTX's AVX2/SSE paths are `#undef`'d out); `sph_panama` is a genuine
SPH file and lives in `sph/`.

## GPU pipeline
A flat thread-major working buffer carries the linear sub-chains through the
shared stage kernels; each stage's output is snapshotted into a per-thread
24-slot accumulator (`d_acc`, **slot-major**: plane `s` at `s*threads*64`, so the
snapshots and the shuffle/SWIFFTX/BLAKE2s gathers are coalesced). SWIFFTX reads
accumulator slots 12–15; the shuffle runs in place; BLAKE2s reduces all 24 slots.
Short digests (Haval/Tiger/SHA-256/LYRA2) are zero-padded to 64 bytes so the
whole-buffer shuffle + BLAKE2s match the reference. Every GPU candidate is
re-hashed on the host (`x25x_hash`) before submit, so a kernel bug can only
local-reject, never emit a bad share.

## Validation
- Init device self-test: the five new primitives are KAT-checked against their
  CPU references at startup (plus a negative bit-flip check); a mismatch logs a
  loud error. The 20 shared stages self-test as usual.
- Benchmark: `ccminer -a x25x --benchmark` (target `ptarget[7]=0x08ff`).
- Live: an x25x pool (zpool lists x25x); bar is "accepted N/N, 0 rejects".
  Live-validated on zpool (accepted share, block reproduced bit-for-bit).

## Notes
- SWIFFTX builds a large lookup table on the host (`InitializeSWIFFTX`) uploaded
  once to device symbols; `multipliers` is in constant memory (uniform access),
  the larger `fftTable`/`As`/`SBox` in global.
- The tiger/sha256 stage launchers use a truncating grid, so any standalone test
  must use ≥512 threads (real throughput is unaffected).
