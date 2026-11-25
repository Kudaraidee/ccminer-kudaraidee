# allium (`-a allium`)

Lyra2RE-family chain for **Garlicoin (GRLC)** -- tpruvot 2018, GPLv3.

```
Blake-256 (80-byte header, 14 rounds)  ->  Keccak-256 (32)  ->  LYRA2  ->
CubeHash-256 (32)  ->  LYRA2  ->  Skein-256 (32)  ->  Groestl-256 (32, terminal)
```

## Layout

- `allium.cu` -- dispatcher (`scanhash_allium`, `allium_hash` CPU reference).

The stage launchers are **not** allium-owned -- they are the shared Lyra2RE
primitives and stay where the lyra2 family keeps them:

- `algos/stages/cuda_blake256.cu` -- Blake-256 setBlock plus the fused
  `blakeKeccak256_cpu_hash_80` (Blake + Keccak over the 80-byte header in one
  launch); the Skein-256, CubeHash-256 and Groestl-256 stages sit beside it in
  `algos/stages/`.
- `algos/stages/cuda_lyra2.cu` -- the `lyra2_cpu_hash_32` sponge stage and
  `lyra2_cpu_init`, checked at init by the fail-closed self-test in
  `algos/lyra2/cuda_lyra2_selftest.cu`.

The primitives are co-owned with `-a lyra2` and `-a lyra2v2`, so they keep their
family names and locations.

## Performance notes

- Almost all of the time is the two Lyra2 passes; see `algos/lyra2/README.md`
  for the v1 stage's shared/register matrix split.
- The Blake+Keccak head is fused, and the Groestl-256 terminal keeps the two
  lowest candidates on device (`groestl256_cpu_hash_32` + `groestl256_getSecNonce`).
- The default intensity is 20; `-i` overrides it.
- A batch with a single candidate resumes at the end of the batch rather than
  rescanning the rest of it.

## Validation

The Lyra2 v1 self-test runs at init and refuses to mine on a mismatch. Every
candidate is re-hashed on the CPU before submit; the second nonce gets the same
check as the first.
