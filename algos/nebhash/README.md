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
memory and reloads the `4 × cin` input words directly from the table rows once
per input element. The 16 row indices are converted to 32-bit row offsets once
per block (the builder rejects a table larger than 4 GiB), which removes a
64-bit multiply from every row byte load. Table rows (linear factors and
convolution weights) are read with `__ldg` through the read-only cache, and
`THREADS` is 128 so twice as many blocks fit on an SM at the same thread
occupancy. The eight lane accumulators are reduced per warp and then summed
once by thread 0, instead of 256 shared-memory atomics per lane. All sums are
modulo `2^32`, so changing the order does not change the ticket.

The scan handle caches the BLAKE3 chaining value for the first 64 bytes
of the header and recomputes it when those bytes change. Each nonce still
uses the final 16 bytes of the header to finish the key.

For each linear layer, the kernel skips outputs whose error factors are zero
across all 16 selected samples. The digest starts with the contributions of
an all-zero linear gradient and adds the difference for each active output.
These additions use modulo `2^32` and preserve the ticket.

For KEP1 and RET2, both the linear layer's input width (`cin`) and the
CUDA block size are multiples of eight. With flattened weight index
`f = o * cin + q` and input index `q` advancing from `threadIdx.x` in
block-sized steps, `f & 7` always equals `threadIdx.x & 7` for that thread.
The thread accumulates its linear contributions in one unsigned 32-bit
sum and adds it to that digest lane just before the warp reduction.
Convolution contributions keep their per-lane accumulation. The scan
checks the input-width requirement per architecture and the block-size
requirement at compile time.

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
the builder retries with the original signed 16-bit layout if any row would
overflow, so the ticket is unchanged. Four rows are transposed into one word and
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
