# Nebhash (NEB, `neb-pool/1`) — `-a nebhash` / `-a neb`

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
| `blake3_min.c`, `chacha8.c` | Small shared BLAKE3 and ChaCha8 implementations used by both references. |
| `LICENSE.NEB` | Apache-2.0 licence for the vendored reference code. |

`NEB_SCAN_CAP` (8192 nonces per `scan_run`) keeps a scan window inside a job's
lifetime while amortising host/device synchronisation; the hit list is
`MAX_NONCES + 1` entries.

For a linear layer the kernel stages only the `4 × cout` error words in shared
memory and reloads the `4 × cin` input words directly from the table rows once
per input element. The order of the inner `cout × cin` accumulation changes, but
the lane sums are modulo `2^32`, so the resulting ticket is unchanged.

## Provenance

`nebhash2.{c,h}`, `shares.{c,h}`, `blake3_min.{c,h}`, `chacha8.{c,h}` and the
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
| GTX 1050 Ti | sm_61 | ~0.33 MH/s avg (~0.20–0.38) | 2026-11-26, live pool, official dataset, table kernel. 5-minute run: 1/1 share accepted, no reject; earlier 15-minute run also accepted every share, with and without `sd=`. |

Lowering the linear-layer shared-memory footprint from `4 × (cout + cin)` to
`4 × cout` words kept every ticket identical in the differential test, but did
not change the measured rate materially on this card; it remains as a smaller
per-block resource footprint.

