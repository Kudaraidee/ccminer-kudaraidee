# XHash (Parallax Protocol)

This directory adds `-a xhash` support for Parallax's XHash proof-of-work to
ccminer. XHash is an Ethash derivative with a roughly 1 GiB-plus DAG, so this
is a full GPU miner rather than a small chained-hash kernel.

## Consensus parameters

| Parameter | Value |
| --- | --- |
| Epoch length | 720 blocks |
| Cache item / initial size | 64 B / 16 MiB (+128 KiB per epoch) |
| DAG item / initial size | 128 B / 1 GiB (+8 MiB per epoch) |
| Cache rounds / DAG parents | 3 / 256 |
| Mix bytes / accesses | 128 B / 64 |
| Cache & DAG item hash | SHA3-512 (NIST, domain 0x06) |
| Final hash | SHA3-256(seed \|\| digest) |
| Seed hash | iterated SHA3-256(`"PARALLAX\x01" \|\| seed`) |
| FNV | byte-wise FNV-1 (multiply by `0x01000193`, XOR each little-endian byte) |

The header hash delivered by the pool is `SealHash(header)`: Keccak-256 over
the RLP block header including `EpochStartTime`. Pools/getwork supply this
finished hash; ccminer does not re-derive it.

## Layout

* `vendor/xhash/` — the official ParallaxProtocol/xhash C/C++ reference library
  (Apache-2.0, `vendor/xhash/LICENSE`). Used for the host light cache, epoch
  recovery from the seed hash, and CPU re-verification of every GPU candidate.
* `xhash_cuda.cu`, `xhash_dagger.cuh`, `xhash_keccak.cuh`, `xhash_fnv.cuh` —
  CUDA search/DAG kernels adapted from ParallaxProtocol/hashwarp
  (`libxhash-cuda`, GPL-3.0, ethminer-derived). Consensus-critical details
  (SHA3 padding, byte-wise FNV-1, DAG item layout) are unchanged.
* `xhash_core.cpp` — epoch/DAG lifecycle, GPU batch orchestration and host
  re-verification.
* `xhash.cpp` — ccminer bridge (`scanhash_xhash` / `free_xhash`), nonce
  allocation and share bookkeeping.

Every GPU candidate is recomputed with the canonical CPU verifier before
submission: a kernel bug can only cause a local reject, never an invalid
share. Once per GPU, at the first job, a synthetic header/nonce is checked
against the CPU reference on the live DAG (the epoch-0 known-answer vector
from the reference test suite was also used to validate the port). Both search
variants are self-tested, because the faster one is selected automatically.

## Kernel selection

`xhash_dagger.cuh` has two search variants: `standard` issues one DAG address
per lane, while `dedup` shares address computation across lanes (fewer
instructions, but fewer memory requests in flight). Which one is faster
depends on the GPU: memory-bound cards generally prefer `standard`, compute-
bound cards prefer `dedup`. At the first job of each GPU, ccminer benchmarks
both with an impossible target for a few batches and keeps the faster one,
logging e.g.:

```
XHash: kernel benchmark standard 7.82 MH/s, dedup 9.85 MH/s -> dedup
```

On a GTX 1050 Ti this improved sustained hashrate from ~8.6 MH/s (`standard`
forced) to ~9.4 MH/s (`dedup` forced). The choice is re-made whenever a GPU
thread is reinitialized.

## Pool protocol

XHash supports two common XHash stratum dialects, selected by the shape of
`mining.notify`:

* **KawPoW/ethproxy style** (7 params): `[job_id, header_hash, seed_hash,
  target, clean, height?, nbits?]`. `target` is the 256-bit boundary in
  big-endian hex; `mining.set_target` may also push it. Submit is
  `mining.submit [worker, job_id, 0x<nonce16>, 0x<header_hash>, 0x<mix>]`.
* **EthereumStratum/1.0.0 style** (4–5 params): `[job_id, seed_hash,
  header_hash, clean, height?]`. `rplant` adds an optional hex `height`;
  other pools omit it. The share boundary is supplied by
  `mining.set_difficulty` as `floor((2^256-1) / (diff * 2^32))`. Submit is
  `mining.submit [worker, job_id, <nonce_low_hex>]` — the pool prepends the
  session extranonce. Subscribe uses `[agent, "EthereumStratum/1.0.0"]` and
  may return either the array form `[subscriptions, extranonce]` or an object
  with `extranonce` and an embedded placeholder `mining_notify`.

The epoch is always recovered from `seed_hash` (falling back to `height/720`),
so the miner works across servers that swap notify order or omit height.
All-zero placeholder hashes are ignored so they never trigger a needless DAG
rebuild.

End-to-end validation on a GTX 1050 Ti:

* Local mock ethproxy pool: synthetic job → DAG → CPU-verified share → submit.
* `pool.readyidc-seedbox.duckdns.org:3647`: both dialects observed; shares
  accepted with zero rejects.
* `parallax.cedric-crispin.com:4554`: EthereumStratum/1.0.0; shares accepted.
* `asia.rplant.xyz:7159`: EthereumStratum with optional `height` param;
  difficulty changes handled, 4/4 shares accepted with zero rejects.

For solo mining against a Parallax node, the equivalent JSON-RPC methods are
`xhash_getWork` (returns `[headerHash, seedHash, boundary, blockNumber]`) and
`xhash_submitWork(nonce, hash, digest)`; a compatible stratum proxy can be
placed in front of ccminer.
