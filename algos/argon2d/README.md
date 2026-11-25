# Argon2 family (`-a argon2d500`, `argon2d1000`, `argon2d4096`, `argon2d16000`, `argon2id1024`)

Memory-hard Argon2 proof-of-work. Five coins share this folder: a single
Argon2 hash over the 80-byte header (password = salt = header, outlen = 32),
differing only in the Argon2 cost parameters, version and type.

| algo         | coin               | m_cost (KiB) | lanes | t_cost | version | type |
|--------------|--------------------|--------------|-------|--------|---------|------|
| argon2d500   | Dynamic (DYN)      | 500          | 8     | 2      | 0x10    | d    |
| argon2d1000  | Zero Dynamics Cash | 1000         | 8     | 2      | 0x10    | d    |
| argon2d4096  | Argentum / Myriad  | 4096         | 1     | 1      | 0x13    | d    |
| argon2d16000 | Alterdot           | 16000        | 1     | 1      | 0x10    | d    |
| argon2id1024 | Bitweb (BTW)       | 1024         | 1     | 3      | 0x13    | **id** |

## Argon2id vs Argon2d

Argon2**d** picks each reference block from the previous block's first word
(data-dependent addressing). Argon2**id** uses Argon2i-style data-independent
addresses for pass 0, slices 0 and 1, and Argon2d after that. The type is also
hashed into H0, so the two differ from the first block on.

## Layout

- `argon2d.cu`: per-coin variant table (`argon2d_variant`), CPU reference
  hashes (used to re-verify every GPU candidate before submit), the shared
  `scanhash_argon2d` driver and the init-time self-tests.
- `argon2d_fill.cu`, `blake2b_kernels.cu`: the GPU fill and BLAKE2b kernels.
- `argon2d_kernel.h`, `argon2d_blake2b_device.cuh`, `cudaexception.h`:
  device-side headers. The fill kernel and the device header are also used by
  rinhash.
- `argon2ref/`: the vendored Argon2 reference. `argon2.c`, `core.c`,
  `encoding.c`, `opt.c`, `thread.c` and `blake2/blake2b.c` are built; the
  upstream `run.c` / `test.c` / `bench.c` / `genkat.c` / `ref.c` are not.

## Implementation notes

- **Runtime geometry.** m_cost, lanes, passes, version, type and the block
  counts are kernel arguments, so all five variants are GPU-correct in one
  binary. The fill epilogue XORs `lanes` rows with a stride loop, which is what
  makes the lanes=1 coins work.
- **Shared cache.** Columns 2..7 of each lane live only in shared memory
  (never stored to global), sized `lanes * 6` blocks at launch: lanes=8 keeps a
  48 KB cache that serves about 10% of refs, lanes=1 takes 6 KB and keeps
  occupancy high. The last column goes to global, because its natural slot
  would alias the column-2 entry that other lanes may still read in the final
  slice (an old donor race that hashed ~0.3% of nonces wrong).
- **Version 0x13** is hashed into H0, and from the second pass on it XORs the
  new block into the old one (`argon2ref/opt.c` `fill_block` `with_xor`). Only
  argon2id1024 (t_cost=3) takes that branch; its read of the old block follows
  the same shared/global split as the store.
- **Argon2id addressing** comes from `next_addresses()`; one address block
  covers 128 offsets, so argon2id1024's 256-block segments regenerate once.
- **Candidate report.** `argon2_finalize` keeps the two lowest candidates with
  `cuda/candidate_report.cuh`; the host re-verifies each slot on its own.

## Provenance

Donor: `duality-solutions/Dynamic-GPU-Miner-Nvidia` (`argon2d_fill.cu` was
byte-identical before the parameterization). argon2d4096's parameters are
from cpuminer-opt's `algo/argon2d/argon2d-gate.c`. The Argon2id addressing and
the v1.3 overwrite logic follow Ondrej Mosnáček's `argon2-gpu`
(`lib/argon2-cuda/kernels.cu`, MIT), the same upstream as the donor, via
`cpuminer-bitweb`; argon2id1024's parameters are from its
`register_argon2id1024_algo`, confirmed against Bitweb mainnet block 76523.

## Validation

- **Host self-test** at init: each CPU reference hash over a fixed header must
  match the independent argon2-cffi library (all five variants), plus a
  bit-flip negative. argon2id1024 adds Bitweb block 76523 (its header is
  checked by sha256d first, and the hash satisfies the block's nBits) and the
  RFC 9106 section 5.3 Argon2id vector.
- **GPU self-test** at init, per card: the three kernels against the CPU
  reference over 32 nonces, plus a header bit flipped on the device that must
  change the digest. Every variant **refuses to start** on a GPU or host KAT
  mismatch; a CUDA resource failure only warns.
  `ARGON2D_SELFTEST_FAULT=kat|neg|host|cuda` forces each failure path.
- **Host re-verify** of every candidate before submit. A failure logs
  "does not validate on CPU!" plus an `argon2d replay:` line with the 80-byte
  input.
- **`-D` differential**: 64 consecutive GPU digests compared with the CPU on
  the first job, from an unaligned start. Unlike the re-verify, this can catch
  a kernel that misses valid nonces.
- `--benchmark` rescans one fixed window, so use it for hashrate; the self-tests
  above are the correctness check.
- Pool-validated on zpool (argon2d500/1000/4096/16000) and a Bitweb pool
  (argon2id1024), on sm_61 and sm_86.
