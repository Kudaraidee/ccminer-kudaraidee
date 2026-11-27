# Nebhash (NEB, `neb-pool/1`) — `-a nebhash` / `-a neb`

[ฉบับภาษาไทย](README.th.md)

Integer-only proof of work used by NEB. One attempt derives a ChaCha8 key from
an 80-byte header with BLAKE3, picks 16 samples from a fixed dataset, evaluates
the training gradient of a convolutional model on those samples, clips and
sums it into a 256-bit little-endian ticket, and compares that ticket with the
pool's share target. No floating point, no DAG and no epoch regeneration.

`-a neb` is an alias for `-a nebhash`. Both plaintext `stratum+tcp://` and TLS
`stratum+ssl://` / `stratum+tcps://` are supported.

## Dataset

The dataset is consensus-pinned. Download `neb-datasets.bin` from the NEB
release page and pass it with `--neb-dataset=PATH` (default: `neb-datasets.bin`
in the working directory). The official file is 16,087,037 bytes:

```text
SHA256  57e22a4b8aacecb2ebae62f76307ddccb04198ed5c4ba15c1dee90ca7a7c2174
BLAKE3  790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72
```

The miner computes the BLAKE3 digest of the whole file at startup and compares
it with the `dataset_hash` sent in `mining.subscribe`. A mismatch aborts mining
with `Nebhash: pool dataset hash does not match the local file`. The file is a
user asset and is never committed to this repository. It contains two model
slots; slot 0 is architecture 1 (KEP1, 784 inputs, 26,000 weights).

## Layout

| File | Role |
| :-- | :-- |
| `nebhash.cpp` / `nebhash.h` | ccminer bridge: dataset load, header build, scan glue, host re-verify and submit. |
| `nebhash2.c` / `nebhash2.h` | Authoritative portable C reference for nebhash v2 (KEP1/RET2): model init, sample gradient, share evaluation, ticket, model step and state hashes. |
| `nebhash2_table_cuda.cu` / `.h` | CUDA kernel actually used by the miner: one block per nonce, table of per-sample gradients, `dp4a` for the linear layers. |
| `nebhash2_cuda.cu` / `.h` | Earlier direct forward/backward kernel, kept for bring-up and reference; the miner does not call it. |
| `shares.c` / `shares.h` | Portable reference for the original (v1) training-share PoW and the shared constants (`NEB_NB=16`, `NEB_CLIP=256`, ...). |
| `cuda/blake3_device.cuh` | Shared BLAKE3 for both CUDA kernels and the host reference. The host uses the full chunk tree for dataset and model hashes. |
| `chacha8.c` | Portable ChaCha8 used by the C reference. |
| `LICENSE.NEB` | Apache-2.0 licence for the vendored reference code. |

`NEB_SCAN_CAP` (8192 nonces per `scan_run`) keeps a scan window inside a job's
lifetime while amortising host/device synchronisation; the hit list is
`MAX_NONCES + 1` entries.

For a linear layer the kernel stages only the `4 × cout` error words in shared
memory and reads the input factors directly from the selected table rows. The 16 row indices are converted to 32-bit row offsets once
per block (the builder rejects a table larger than 4 GiB), which removes a
64-bit multiply from every row byte load. Table rows (linear factors and
convolution weights) are read with `__ldg` through the read-only cache, and
`sm_61` uses one warp per block. Other architectures use the configured
block size, which defaults to 128 threads. The eight lane accumulators are reduced per warp and then summed
once by thread 0, instead of 256 shared-memory atomics per lane. All sums are
modulo `2^32`, so changing the order does not change the ticket.

The scan handle caches the BLAKE3 chaining value for the first 64 bytes
of the header and recomputes it when those bytes change. Each nonce still
uses the final 16 bytes of the header to finish the key.

For each linear layer, the kernel skips outputs whose error factors are zero
across all 16 selected samples. The digest starts with the contributions of
an all-zero linear gradient and adds the difference for each active output.
These additions use modulo `2^32` and preserve the ticket.

For KEP1 and RET2, the input width of each linear layer is divisible by
eight. For `f = o * cin + q`, the digest lane `f & 7` is therefore set by
`q & 7`. Each thread keeps separate sums for the one, two, or four digest
lanes it contributes to and maps them into the eight lanes at warp
reduction. The same mapping applies to grouped convolution factors.

## Provenance

`nebhash2.{c,h}`, `shares.{c,h}`, `chacha8.{c,h}` and the
reference-derived table kernel are vendored from the NEB implementation under
Apache-2.0 (`LICENSE.NEB`). The ccminer bridge and stratum glue follow the
project's GPLv2.

## Stratum (`neb-pool/1`)

| Message | Shape |
| :-- | :-- |
| `mining.subscribe` | `["<user agent>", "neb-pool/1"]` → object with `session`, `extranonce1`, `extranonce2_size`, `chain_id`, `dataset_hash`. |
| `mining.set_target` | `["<32-byte LE hex>"]` — the share threshold. |
| `mining.set_model` | `[slot, "<weights_hash hex>", "<packed weights hex>"]`. |
| `mining.notify` | One job object: `job_id`, `version`, `parent`, `time_start`, `bits`, `target`, `share_target`, `coinbase_prefix/suffix`, `branch`, `tx_count`, `shares_root`, `height`, `slot`, `weights_hash`, `expiry`. |
| `mining.submit` | `[login, job_id, extranonce2_hex, ntime(number), nonce(number)]`. |
| `mining.block` | Optional pool extension, sent only after the node accepted a candidate. Params are either `["<job_id>", "<nonce>"]` or an object with `job_id`, `extranonce2`, `ntime`, `nonce`, `blockhash`, `height`. ccminer then logs `yay!!! ... solved: N`. |

The notify target fields describe chain/block state; the value that gates shares
is `mining.set_target`. Target-to-difficulty uses yiimp's
`2^256 / (target + 1)`. The `sd=` password option is interpreted by the pool
(yiimp scale `sd * 1e10`); the miner never parses it.

## Correctness

- The GPU table kernel is checked against the C reference with an exact-ticket
  differential harness on `sm_61`: 384 nonces, five rank boundaries and eight
  exact tickets per case, for both model architectures and initial / trained /
  all-7 / all-(-8) / alternating weights. All cases pass.
- Every GPU hit is recomputed on the host with `neb2_share_eval()` before a
  `mining.submit`; a kernel bug can only cause a local miss, never a bad share.
- `NEB_NEGATIVE_CONTROL_TABLE_CLIP` deliberately clips one below `NEB_CLIP` so
  the differential test demonstrates that it detects a wrong gradient.

The miner passes `max_nonce` as the exclusive end of the scan range. Each CUDA
batch scans at most 8,192 nonces, with a shorter final batch when fewer remain.
An empty range skips the CUDA launch. After a successful scan, the cursor holds
the last scanned nonce so the miner loop starts the next batch at the following
nonce when the job is unchanged.

## Usage

```bash
ccminer -a nebhash \
  -o stratum+ssl://pool.example.org:3645 \
  -u <neb-address>.<worker> -p c=NEB \
  --neb-dataset=/path/to/neb-datasets.bin
```

`stratum+tcps://` is accepted as an alias for `stratum+ssl://`. TLS certificate
verification stays enabled; use `--cert` only to supply a custom CA bundle.

## Benchmark log

| GPU | Arch | Rate | Notes |
| :-- | :-- | :-- | :-- |
| GTX 1050 Ti | sm_61 | ~0.43 MH/s avg (~0.25–0.55) | 2026-11-26, live pool, official dataset, `sm_61` kernel. 5-minute run: 432.33 kH/s, 1/1 share accepted, no reject. A 150-second A/B run averaged 493.47 kH/s. Earlier runs also accepted every share, with and without `sd=`. |
| GTX 1050 Ti | sm_61 | ~0.50–0.53 MH/s avg | 2026-11-26, packed int8 convolution rows with automatic int16 fallback. Two 170-second local `neb-pool/1` runs with the official dataset averaged 529.02 and 506.61 kH/s; the adjacent 16-bit baseline runs averaged 517.30 and 486.20 kH/s. All shares accepted, no rejects. |

Lowering the linear-layer shared-memory footprint from `4 × (cout + cin)` to
`4 × cout` words kept every ticket identical in the differential test but did
not change the rate materially. A/B runs of the warp reduction versus the old
shared-atomic reduction showed about 3% higher throughput. Precomputing 32-bit
row offsets, using `__ldg` for the table rows, and dropping `THREADS` from 256
to 128 raised the 5-minute average from ~0.34 to ~0.41–0.43 MH/s. Convolution
factors are now stored as signed 8-bit values when every sample gradient fits;
the builder uses int16 or int32 for a layer when its gradients need a wider
representation, preserving the ticket. Four rows are transposed into one word and
summed with `__dp4a`, which gave a further ~2–4% on the GTX 1050 Ti. Variants
that passed the differential test but were discarded as slower or less stable:
aligned 32-bit transposed staging, two convolution weights per load, branchless
`dp4a`, precomputed row offsets in shared memory, 512 threads, and 64 threads.
A `__dp2a`-based packed int16 variant was also discarded because it failed the
exact-ticket differential.

### Combined kernel on live Yiimp

The GTX 1050 Ti (`sm_61`) ran each variant for 90 seconds with the official
dataset, the same 32-byte share target (`d=0.0002`), and 8,192 nonces per
batch. Scan rate uses time spent inside the CUDA scan call. Median batch
time measures how long that call took to return.

| Variant | Scan rate | Median batch time | Shares accepted |
| :-- | --: | --: | --: |
| BLAKE3 cache only | 577.97 kH/s | 14.32 ms | 8/8 |
| Skip zero linear gradients only | 647.91 kH/s | 12.42 ms | 22/22 |
| Both optimizations | 651.25 kH/s | 12.16 ms | 7/8 |

The combined run spent 18.5 seconds scanning, while the zero-gradient-only
run spent 57.4 seconds scanning. Yiimp returned code 21 (`Job not found or
past its expiry`) for one combined-run share during a job change. Skipping
zero gradients accounted for most of the observed speedup. The 0.5%
difference between the combined and zero-gradient-only runs needs more
repetitions to establish a consistent gain.

### Per-thread linear accumulation on live Yiimp

Compared with `15294b5`, the new kernel reached 741.42 kH/s during CUDA
scans on the GTX 1050 Ti (`sm_61`), up 17.6%. Both versions used the
official dataset, 8,192 nonces per batch, and the same 32-byte share target
(`d=0.0002`). The order was baseline, new kernel, new kernel, baseline.
Each round stopped at the third distinct height announced by Yiimp.
Repeated notifications at the same height were counted once.

| Kernel | Scan rate | Median batch time | Sent | Accepted | No reply |
| :-- | --: | --: | --: | --: | --: |
| `15294b5` baseline | 630.68 kH/s | 12.78 ms | 59 | 31 | 28 |
| Per-thread linear accumulation | 741.42 kH/s | 10.68 ms | 33 | 33 | 0 |

The two new-kernel rounds measured 737.67 and 743.24 kH/s. Connection
resets occurred in the baseline rounds, and 28 baseline submissions
have no recorded reply. Accepted-share counts therefore have different
coverage across the two versions. The new kernel passed the 384-nonce
differential against the C reference for both KEP1 and RET2, using five
weight patterns per architecture, five rank boundaries and eight exact
tickets per case.

### Batched convolution-width fallback

Commit `e918186` recorded every convolution layer that overflowed int8 or
int16 during the first pass and widened them together in one rebuild.
Commit `6a9d8d5` removed the repeat evaluation by retaining full-width
convolution gradients for one pass, then compacting the table in place
before uploading it to CUDA. The host buffer retains its raw capacity.

Live runs on the GTX 1050 Ti with the official dataset accepted 41/41
shares on Yiimp and 8/8 on MeshPool. The table below measures subsequent
local A/B builds of one model captured from each pool, with two
repetitions per builder. Time includes table creation and upload.

| Captured model | Dataset rows | Passes, old → new | Mean time, old → new | Time saved |
| :-- | --: | --: | --: | --: |
| Yiimp, slot 0 | 13,404 | 4 → 2 | 18.42 s → 10.11 s | 45.1% |
| MeshPool, slot 1 | 7,089 | 5 → 2 | 11.18 s → 5.66 s | 49.4% |

Each replay produced the same table and layout hash, and the same set of
hits in a 384-nonce scan. The measurements quantify the wait for a new model
table. CUDA scan throughput is measured separately.

### Reuse the table created when opening a scan handle

`neb2t_scan_open()` builds and uploads the supplied model. After a
successful open, the host records its weights hash in the scan handle,
so the first scan uses that table immediately. A different weights hash
for the same handle still triggers one rebuild and upload.

A GTX 1050 Ti replay using the one-pass builder and the same captured
Yiimp model in both variants reduced the mean time from `mining.notify`
to the first completed CUDA batch from 7.98 s to 2.89 s. Each variant ran
twice. The number of table builds before the first scan dropped from two
to one. Both runs of the new variant also built a replacement table and
submitted a share after the replay changed the model in the same slot.

The optimized miner ran for three minutes per pool with the same
32-byte miner-side share target, about 8.59 million nonces per expected
share. It received 11/11 accepted shares on Yiimp and 18/18 on MeshPool.
The first scan handle built one table on both pools. MeshPool remained
at height 12048 during its run. The same-slot model transition was
covered by the local replay.

### Paired int16 convolution loads

On an int16 convolution layer with an even number of factors, each
aligned 32-bit read returns two adjacent signed gradients from one
sample. The kernel adds the two factors into separate 32-bit sums
across all 16 samples, clips each sum, and adds it to its original
digest lane. This path comes from [tpfuemp's paired-load change](https://github.com/tpfuemp/ccminer-tpfuemp/commit/a56374932fa8674d503cbce2df84eb4d787f1973).

A GTX 1050 Ti replay used the official dataset and one model captured
from each pool. Each variant ran twice with 8,192 nonces per batch.
The GPU was idle before each run. All runs were local replays.

| Captured model | Scalar int16 | Paired int16 | Four factors per load |
| :-- | --: | --: | --: |
| Yiimp | 771.45 kH/s | 905.58 kH/s (+17.4%) | 928.09 kH/s (+20.3%) |
| MeshPool | 779.91 kH/s | 902.16 kH/s (+15.7%) | 868.77 kH/s (+11.4%) |

The miner uses the paired path, which improved throughput on both models.
The integrated kernel matched the C reference hit sets for KEP1 and RET2
over 384 nonces at five target boundaries.

### Grouped digest sums and linear input loads

The scan kernel accumulates scalar, paired and four-factor results
in separate thread-local sums and maps them to the original digest
lanes at warp reduction. Linear inputs use aligned 16-bit loads and
byte permutation before `__dp4a`. On `sm_61`, one warp handles each
nonce, uses a ballot to compact the active linear outputs, and reads
four inputs at a time when the linear input segment is 4-byte aligned.
Other architectures use the configured block size and paired linear
loads. The paired linear load follows [tpfuemp's implementation](https://github.com/tpfuemp/ccminer-tpfuemp/commit/5227ec136be86f5c96d1c311cbb2360d92324a26).

Local A/B on GTX 1050 Ti used the same captured model for each pair
of kernels, 8,192 nonces per batch, and two runs per kernel. The
integrated kernel matched the CPU hit sets across five target
boundaries in 384-nonce checks on both KEP1 models and a RET2 fixture.

| Captured model | `5df4efd` baseline | New kernel | Increase |
| :-- | --: | --: | --: |
| Yiimp | 938.59 kH/s | 1,456.32 kH/s | 55.2% |
| MeshPool | 913.63 kH/s | 1,476.53 kH/s | 61.6% |

Live runs used the integrated kernel for three minutes per pool and
the same 32-byte target on the miner, equivalent to about 12 million
nonces per expected share. Each pool validated its shares against its
own target. All 31 submissions were accepted.

| Pool | Accepted | Median scan rate | Peak sample |
| :-- | --: | --: | --: |
| Yiimp | 13/13 | 1,428.55 kH/s | 1,457.65 kH/s |
| MeshPool | 18/18 | 1,146.70 kH/s | 1,440.48 kH/s |

The GPU reached 98°C and its SM clock declined during sustained load.
These live measurements capture that thermal behavior. The local and
live measurements remained below 1.50 MH/s.
