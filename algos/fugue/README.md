# fugue (`-a fugue256`)

Fuguecoin proof-of-work: Fugue-256 of the 80-byte header.

## Files

- `fuguecoin.cpp` - host driver (`scanhash_fugue256`) and CPU reference (`fugue256_hash`, `sph_fugue256`).
- `cuda_fugue256.h` - interface to the kernel (`fugue256_cpu_init` / `_setBlock` / `_hash` / `_free`,
  `fugue256_device_selftest`).
- `algos/stages/cuda_fugue256.cu` - the kernel. Single consumer: this algo.

## Kernel

- The host absorbs the 19 nonce-free header words (plus the nonce-free part of the next sub-round and
  of its first SMIX); each thread hashes one nonce from there.
- One mixing table (T0) in shared memory, replicated per lane group; the other three tables are byte
  rotations of it. On sm_80+ the table index is computed on the IMAD pipe.
- Digest word 7 is compared against the target before the last SMIX, which does not change it.
- Launch: TPB 512 on sm_6x, 256 elsewhere; default intensity 25.
- Candidates: the two lowest nonces of a launch (`report_candidate_2`); the host re-verifies both
  with `sph_fugue256` + `fulltest` before submitting.

## Validation

- Init-time self-test, fail-closed: the published Fugue-256 empty-message digest for the CPU
  reference, a two-candidate KAT through the real launcher over a ragged span, and a flipped-header
  negative.
- `--benchmark` finds and CPU re-verifies candidates (benchmark target `ptarget[7] = 0xf`).
- No Fuguecoin pool is known to be running, so there is no live-share record.
