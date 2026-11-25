# lyra2 family (`-a lyra2`, `-a lyra2v2`, `-a lyra2z`, `-a lyra2z330`)

Lyra2-based coins, tpruvot lineage, GPLv3.

- `lyra2RE.cu` — `scanhash_lyra2` (Lyra2RE): blake256 → keccak256 → lyra2 →
  skein256 → groestl256.
- `lyra2REv2.cu` — `scanhash_lyra2v2` (Lyra2REv2): blake256/keccak (fused
  `blakeKeccak256`) → cubehash256 → lyra2v2 → skein256 → cubehash256 → bmw256.
- `lyra2Z.cu` — `scanhash_lyra2Z` (Lyra2Z): blake256 → lyra2Z.
- `lyra2Z330.cu` — `scanhash_lyra2z330` (Lyra2Z330, Gravity/GXX): Lyra2 alone over
  the 80-byte header, `timeCost 2 / 330 rows / 256 cols`. See below.

CPU references: `Lyra2.c`/`.h`, `Lyra2Z.c`/`.h`, `Sponge.c`/`.h`, also included by
allium, x21s and evohash via `algos/lyra2/Lyra2.h`.

Lyra2REv3 is not present (it was never wired up); see git history if it is wanted.

## Layout

The Lyra2 **matrix primitives** live in `algos/stages/` (`cuda_lyra2.cu`,
`cuda_lyra2v2.cu`, `cuda_lyra2Z.cu`, `cuda_lyra2Z330.cu` + `cuda_lyra2_vectors.h`),
the dispatchers and CPU references here. The stage launchers keep their
`lyra2*_cpu_*` names because they are co-owned with allium/x21s/evohash, which reach
them via `extern`.

Two shared device headers:

- `cuda/blake2b_device.cuh` — the reduced-round BLAKE2b permutation all four stages
  use: `Gfunc`, the warp shuffles, both `round_lyra` layouts (4-lanes-per-hash
  `uint2[4]`, single-thread `uint2x4[4]`) and the quad-scoped variants z330 needs.
- `cuda/lyra2_device.cuh` — the shared-memory row ops (`LD4S`/`ST4S`,
  `reduceDuplex`, `reduceDuplexRowSetup`, `reduceDuplexRowt`), macro-driven off the
  includer's `Nrow`/`Ncol`/`memshift`/`BUF_COUNT`. Used by lyra2z (8 x 8, memshift 3,
  BUF_COUNT 0).

The v1 stage (`cuda_lyra2.cu`, used by lyra2, allium and evohash) has its own row
ops: it keeps the first four columns of each matrix row in shared memory and the
last four in registers, which halves the shared memory per hash and doubles the
hashes resident per SM. Register cells are only indexed by a compile-time column;
the data-dependent wander row becomes a select.

The v2 stage (`cuda_lyra2v2.cu`, used by lyra2v2, x21s and x25x) does the same on
its 4 x 4 matrix: two columns in shared memory, two in registers. Its shared-memory
columns stay in rolled loops, because fully unrolled the kernel is much slower on
sm_61. **lyra2z330** streams a global matrix, so its accessors take the matrix
pointer and interleave by thread. The `blake2b_IV` tables likewise stay per-TU —
v1/v2 `__constant__`, Z kernel-local `const`, z330 a lane-indexed `uint2` view.

The build floor is sm_61, so the family carries no arch guards, no pre-`__shfl`
shuffle emulation and no `cuda_arch`/`device_sm` dispatch — the hosts, the stages
and `cuda_lyra2_vectors.h` alike. The vestigial `gtx750ti` parameter is gone from
`lyra2_cpu_hash_32` / `lyra2_cuda_hash_64` / `lyra2Z_cpu_hash_32` and from their
callers. The one remaining arch read is `lyra2REv2.cu`'s `cuda_get_arch()`, which
is there for `cubehash256`, not for lyra2.

The 256-bit primitives this family shares with Algo256
(`blake256`/`bmw256`/`cubehash256`/`groestl256`/`skein256`) are still family-branded;
a family-wide de-brand is possible now that both trees are migrated.

## lyra2z330

Two differences from lyra2z:

- **No prehash.** lyra2z blake256's the header and feeds Lyra2 32 bytes; lyra2z330
  feeds it the raw 80-byte header as *both* password and salt.
- **330 rows is not a power of 2.** Lyra2's Wandering phase picks rows with
  `state[0] & (nRows-1)` in the classic code, which only holds for a power-of-2
  `nRows`; `Lyra2Z.c` uses the generic modulo form, as both reference miners do.
  Taken on the *unsigned* value it is bit-identical to the mask for power-of-2
  `nRows` (including the negative-step wrap), so lyra2z is unaffected.

`LYRA2Z_reuse` takes a caller-owned matrix so the hashing loop does not allocate
~7.7 MB (330 × 256 × 96 B) per hash; `LYRA2Z` is the one-shot wrapper. Neither
pre-zeroes the matrix — Setup writes every row before it can be read.

### GPU kernel (`algos/stages/cuda_lyra2Z330.cu`)

The other matrix stages hold the whole matrix on chip (a few KB per hash);
7.7 MB cannot fit, so this one keeps it in global memory and **streams** it — the row
operations walk a row column by column with only the 16-word sponge state live.
Everything is one launch: absorb → rows 0/1 → Setup → Wandering → wrap-up → squeeze.

Lane split as in `cuda_lyra2.cu`: the blake2b state is a 4 × 4 `uint2` matrix and
lane L owns column L (sponge words {L, 4+L, 8+L, 12+L}), so the 12-word bitrate is
`state[0..2]` and a 96-byte matrix column is 3 `uint2` per lane. Row operations and
the `rotW` cross-lane rotation are transcribed from `cuda_lyra2.cu`.

Specific to this kernel:

- Matrix layout is lane-interleaved, `((3·(256·row + col) + j)·threads + thread)·4 +
  lane`: a quad hits 4 consecutive `uint2` (64 B) and consecutive hashes are
  adjacent, so a warp covers 512 contiguous bytes.
- **No `__ldg` on matrix loads** — the same kernel writes the matrix, and the
  read-only cache is not coherent with those writes.
- **Shuffles are quad-scoped** (`0xf << (laneid & ~3)`), not full-warp: a tail quad
  must exit on the thread-count guard (matrix slots are per-quad, so it cannot clamp
  onto a neighbour's), which leaves the warp partly inactive, and a `__shfl_sync`
  mask naming exited threads is undefined behaviour.

Throughput is bounded by VRAM, not by the kernel: `scanhash_lyra2z330` derives it
from `cudaMemGetInfo` (192 MB reserved, rounded to whole blocks) and `-i` may only
lower it. 72 registers, 0 spill.

### Measured rates

| | rate |
|---|---|
| CPU reference, 1 thread (`LYRA2Z_reuse`, gcc -O2) | ~437 H/s (2.29 ms/hash) |
| GPU, RTX 3060, 1376 concurrent hashes | ~4.48 kH/s |
| GPU, live on zpool (auto-sized 1408 threads, 10.6 GB) | ~4.37–4.56 kH/s, 108–124 W |

Flat past ~1024 concurrent hashes, i.e. **at the DRAM roofline**: ~87 MB of traffic
per hash (the Wandering phase read-modify-writes both `M[row]` and `M[row*]`) ×
4.5 kH/s ≈ 390 GB/s on a ~360 GB/s card. Occupancy and launch-shape knobs are dead
levers; only less traffic would help, and the algorithm fixes that. One launch is
~307 ms — inside the Windows TDR window, but it also sets the job-switch latency.

### Validation

Ground truth: cpuminer-opt (`algo/lyra2/lyra2z330.c`) and KlausT ccminer
(`lyra2/lyra2Z330.cu`), which agree bit-for-bit. Header `00..4f` ⇒
`bbc07308856eef2305237fd2aa662c6573d2e173fddee568788bc30048d54ab0`, embedded as the
init-time self-test with a bit-flip negative check; lyra2z's own shape
(`8/8/8`, input `00..1f`) ⇒ `7bf350a4…`, unchanged by the modulo switch.

`lyra2z330_gpu_selftest` compares the kernel against the CPU reference over 4 nonces
at init, and every candidate the on-device screen reports is re-hashed on the host
before submit.

**Live** on zpool: 50/51 accepted, 0 does-not-validate, 14 blocks in ~6.5 min. The
non-accepted share was `Invalid job id` — stale at a job rollover (the ~307 ms launch
granularity), not a bad hash.
