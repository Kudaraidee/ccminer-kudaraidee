# bmw family (`-a bmw`/`bmw256`, `-a bmw512`)

Standalone Blue Midnight Wish coins (tpruvot lineage, GPLv3).

- `bmw.cu` - `scanhash_bmw`: BMW-256 of the 80-byte header. Kernel in
  `algos/stages/cuda_bmw.cu`: the host hashes block 1 and precomputes the
  nonce-free part of block 2 once per job (`bmw256_setBlock_80`); the kernel
  `bmw256_gpu_hash_80_final` finishes the hash and screens it against the
  target. On sm_80+ part of the additions run on the IMAD pipe. Default
  intensity 27. `cuda_bmw256.cu` (the 32-byte stage used by `lyra2v2`) is a
  separate body.
- `bmw512.cu` - `scanhash_bmw512`: BMW-512 of the 80-byte header. Kernel
  `bmw512_gpu_hash_80_final` in `algos/stages/cuda_bmw512.cu`, with the
  nonce-free part of the first compression precomputed once per job
  (`bmw512_final_setBlock_80`). The 64-byte stage and `bmw512_gpu_hash_80`
  in the same file serve the x-family and are separate.

Both kernels report the two lowest candidates; the host re-verifies every one
with `fulltest` before submitting.

## Validation

- Start-up self-test per algo, fails closed: known-answer test against an
  independent implementation of the spec, a negative test, a GPU-vs-CPU check
  over a nonce range through the mining arithmetic, and a check of the
  candidate report.
- `-D` repeats the GPU-vs-CPU check over 2048 nonces of every new job.
- `BMW_SELFTEST_FAULT=<check>` (kat, neg, diff, negd, report, cuda, jobdiff)
  breaks one check on purpose; for testing the self-test only.
