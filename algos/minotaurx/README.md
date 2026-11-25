# minotaurx (`-a minotaurx`)

MinotaurX proof-of-work: **Avian (AVN)**, **Pulsar (PLSR)** and other minotaurx chains.

Tested on an RTX 3060 (sm_86) and a GTX 1080 Ti (sm_61), pool-proven on both
(zpool `minotaurx.na.mine.zpool.ca:7019`).

## The algorithm

yespower 1.0 (N = 2048, r = 8, pers `"et in arcadia ego"`, 17 bytes) behind a short walk
through a tree of the sixteen x16 hashes:

```
seed     = SHA-512(header80)
alg[i]   = seed[i] % 16            i = 0..21, all fixed before the walk
alg[21]  = yespower
h = seed; node = 0
repeat:  h = H_alg[node](h)        always 64 bytes in, 64 out
         node = child[node][h[63] & 1]
digest   = yespower(h, 64 bytes)
```

Each nonce costs the seed, six cheap 64-byte hashes and one yespower; yespower dominates.

Details that each give a well-formed wrong hash if missed:

| | Correct |
|---|---|
| node algorithm | from the **seed**, not the running hash |
| branch bit | bit 0 of **byte 63** of the node's output, `(word15 >> 24) & 1` |
| algorithm order | blake, bmw, cubehash, echo, fugue, groestl, hamsi, sha512, jh, keccak, luffa, shabal, shavite, simd, skein, whirlpool (not x16r's) |
| yespower input | the 64-byte running hash |
| Hamsi | always hashed (the Hamsi skip is plain `minotaur` only) |

**Difficulty is on the 2^32 scale**, not yespower's /65536. The nonce is hashed
big-endian and submitted on the default little-endian path.

## How it runs

The walk runs on the host, overlapped with the GPU; the GPU runs only the yespower node,
using the shared yespower kernel body (`YP_HEAD_SHA256_64`).

| Card | minotaurx | yespowerTIDE (same r=8 kernel) |
|---|---|---|
| RTX 3060 (sm_86) | 2243 H/s | 2309 H/s |
| GTX 1080 Ti (sm_61) | 1739 H/s | 1763 H/s |

`--benchmark`, 90 s each. The benchmark understates pool mode (~2440 H/s on the 3060).

## Layout

| File | Role |
|---|---|
| `minotaurx.h` | CPU-reference API |
| `minotaurx_hash.cpp` | CPU reference and self-test; only `sph/` and the yespower reference |
| `minotaurx_kat.h` | four Pulsar mainnet blocks (7920000/3/4/11), from cpuminer-opt `algo/x16/minotaurx-kat.h` |
| `minotaurx.cu` | mining kernel, host driver, device self-test, `-D` differential |

## Correctness checks

- **CPU self-test at startup**: the four mainnet headers, each digest exact and under its
  own block's target, plus a flipped nonce that must change the digest. On a Pulsar
  minotaurx block the block id is the PoW hash, so the vectors come from the chain.
- **GPU self-test at startup**, through the mining kernel: the same headers, a published
  yespower-node vector, and the kernel's own target report. A failure stops the miner
  (exit code 3).
- **Host re-verify** of every candidate, the second nonce included, before submit.
- **`-D`**: the kernel's digests for 457 nonces are compared against the CPU reference on
  each new block and at most once a minute otherwise.

`algos/yespower/yespower_head_tail.cuh` is shared with `-a yespower`; a change there must
be checked on both algorithms.
