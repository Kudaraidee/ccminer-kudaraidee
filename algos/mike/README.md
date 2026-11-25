# mike (`-a mike`)

Mike proof-of-work, used by **VKAX** and **FortuneBlock**.

GhostRider with the core pool reduced from 15 algorithms to 11, which shortens
the third core group from five rounds to one:

```
ghostrider  5 core, CN, 5 core, CN, 5 core, CN  (15 core + 3 CN, 18 steps)
mike        5 core, CN, 5 core, CN, 1 core, CN  (11 core + 3 CN, 14 steps)
```

Nothing else differs from GhostRider. The core table is ghostrider's entries
0..10 (`blake512, bmw512, groestl512, jh512, keccak512, skein512, luffa512,
cubehash512, shavite512, simd512, echo512`; JH is **kept**, hamsi, fugue,
shabal and whirlpool are the four dropped). All six CryptoNight-v1 variants and
their parameters, the CN finalization (`extra[state[0] & 3]` over the 4-entry
set), `HASH_SIZE = 32`, the post-CN zeroing, the sha256d merkle root and the
2^16 difficulty factor are shared verbatim.

Both the core order and the CN triple come from header bytes `[4,36)`, so they
are constant for a whole job (the nonce does not affect them), which is what
makes a whole-batch GPU pipeline legal.

## The one trap

The core order is a permutation of `0..10` produced by `nibble % 11`. It is
**not** GhostRider's 15-wide permutation truncated to 11 entries. Using 15
gives a wrong hash on nearly every input, **with no symptom until a pool
rejects the share**, and a sparse test vector passes either way. The count
lives in exactly one place, `MIKE_CORE_ALGO_COUNT` in `mike.h`, defined as the
core enum's terminator so it cannot drift from the table.

Do not "fix" the nibble walk to match VKAX Core's descending loop:
`uint256::GetNibble()` starts with `index = 63 - index`, so their descending
loop already yields ascending raw nibble order, the same as ours.

## Layout

| File | Role |
|---|---|
| `mike.h` | core/CN enums, the pool sizes, and the CPU-reference API |
| `mike_hash.cpp` | CPU reference: order derivation, core/CN dispatch, the chain, the self-test. Pulls in `sph/` and the CryptoNight CPU cores and **nothing else** (no `miner.h`, no CUDA), so it links into a standalone gate |
| `mike_kat.h` | **generated**; the two known-answer sets, extracted verbatim from the CPU reference |
| `mike.cu` | GPU driver: order derivation, the 14-step pipeline, the startup guards, the candidate screen, the benchmark rotation sweep |

`mike.cu` mirrors `algos/ghostrider/ghostrider.cu`; a driver change there has to
be carried over by hand.

Dependencies are resolved by include path and by link-time externs: the shared
x-family stage library via `algos/common/cuda_x_stages.h`, and the
CryptoNight-v1 GPU/CPU paths via `extern "C"` to the `*_gr` functions in
`algos/cryptonight/`. Those are shared with `ghostrider` unchanged; mike adds
no CryptoNight code.

## Correctness gates

Two known-answer sets, both hard-fail, both needed:

1. **xmrig PR #3131 vector** (256 bytes): eight 80-byte zero blobs XORed across
   two rotations. Covers the chain shape, all six CN variants, the CN
   finalization and the post-CN zeroing. Carries a vacuity guard, since two
   identical rotations would XOR to zero. It is **blind to the `% 11`
   selection** and passes with the wrong count.
2. **Four dense-prevhash vectors**: these pin the selection. All four reject
   count 15. Vector 0 reuses ghostrider's own test header so one input
   exercises both algos.

Plus, at GPU startup: a per-kernel race check at full throughput (8 repeats per
stage), and with `MIKE_GUARD=1` a step-by-step GPU-vs-CPU comparison of the
whole 14-step pipeline over several random header orders.

## Diagnostics

| Env var | Effect |
|---|---|
| `MIKE_GUARD=1` | add the full GPU-vs-CPU chain check to the startup self-test |
| `MIKE_VERIFY=1` | audit the candidate screen against a host-side compare every batch |
| `MIKE_VERIFY=2` | same, plus corrupt one host digest to prove the audit can fire |
| `MIKE_VERIFY=3` | same as 1, plus re-hash a sample of each header's lanes on the CPU (`MIKE_VERIFY_LANES`, default 64) |
| `MIKE_VERIFY=4` | same as 3, plus flip one sampled digest to prove the check can fire |
| `MIKE_WORK_BUDGET` | batch size in lane x CN-work units (default 24576); `-i` overrides |
| `MIKE_LANE_CAP` | upper bound on lanes per batch (default 16384) |
| `MIKE_BENCH_TARGET` | override the relaxed `--benchmark` screen target |
| `CN_MIDBATCH_ABORT=0` | do not stop a batch between CN groups on a job change |

`--benchmark` sweeps all 20 CN rotations (4 s each) and reports the mean, because
a synthetic header otherwise pins one cheap chain; `-a all` times the single
mean-cost rotation instead.
