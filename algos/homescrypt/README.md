# homescrypt (`-a homescrypt`)

HomeScrypt v1.2 ("FP6") proof-of-work: **Lumenite (LMT)**.

Reference: `homescrypt_v12` in lumenite-core `src/crypto/homescrypt.cpp`, called from
`CBlockHeader::GetPoWHash()`.

## The algorithm

```
sseed    = scrypt_1024_1_1_256(header80)
seed     = SHA-256("HOMESCRYPT-V1.2-SEED" || header80 || sseed)
fill     S[0 .. 2^21) (16 MiB of u64) with a serial splitmix64 chain from the seed
mix      2^16 rounds: three dependent random reads a, b, c;
         a 192-FMA binary32 chain seeded from a, b, c; three ordered writes
fold     all of S into four words
digest   = SHA-256("HOMESCRYPT-V1.2-FINAL" || seed || fold || header80)
```

Details that each give a well-formed wrong hash if missed:

| | Correct |
|---|---|
| FP | `fmaf` only, correctly rounded; no `a*b+c`, fast-math or FTZ |
| mix writes | in the order i1, i2, i3; the indices can collide and the last write wins |
| mix reads | plain loads; a round can read a word the previous round wrote |
| FP rotate-in | `m3` takes the old `m0`; the XOR uses the raw bits before renormalising |
| rotate amounts | `(i & 31) + 1` and `(mixed & 31) + 1` reach 32 |

Difficulty is on the 2^32 scale (Bitcoin diff-1 target). The nonce is hashed big-endian and
submitted on the default little-endian path.

MeshPool sends its own `mining.notify`, `[job_id, header, nbits, clean, extranonce2, ntime]`,
with the finished 80-byte header in place of a coinbase and merkle branch. The miner hashes that
header as is and echoes the job's extranonce2 in the standard `mining.submit`. A standard
9-field notify takes the normal coinbase path.

## How it runs

Each hash holds its 16 MiB scratchpad for the whole hash, so the instance count is free VRAM /
16 MiB (`-i` lowers it). Seed, fill and mix run one thread per hash; the fold runs its four
independent chains on four threads, in the same kernel as the fill of the next batch. The mix
runs in four launches, and a new job abandons the batch between them.

| Card | Pool rate |
|---|---|
| RTX 3060 (sm_86), 662 instances | ~2500 H/s |
| GTX 1080 Ti (sm_61), 666 instances | ~1900 H/s |

## Layout

| File | Role |
|---|---|
| `homescrypt.h` | CPU-reference API |
| `homescrypt_hash.cpp` | CPU reference and self-test |
| `homescrypt_kat.h` | Lumenite blocks 250/255/261, the zero header, one vector per mix-write collision, 3 short-shape vectors |
| `homescrypt.cu` | kernels, host driver, device self-test, `-D` differential |

`homescrypt.cu` is built with `--fmad=false --ftz=false` in both build systems.

## Correctness checks

- **CPU self-test at startup**: all 10 vectors, plus a flipped header that must change the digest.
- **GPU self-test at startup**, through the mining kernels: the 7 full-shape vectors, then a
  13-nonce batch compared with the CPU and a target that must report exactly the four lowest
  digests. A failure stops the miner (exit code 3).
- **Host re-verify** of every candidate before submit.
- **`-D`**: the kernels' digests for 61 nonces are compared with the CPU reference on each new
  block and at most once a minute otherwise.
