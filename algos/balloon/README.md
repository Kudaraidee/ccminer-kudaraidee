# balloon (`-a balloon`)

Balloon memory-hard hashing (Boneh, Corrigan-Gibbs, Schechter) over SHA-256, with
`s_cost = 128` (a 128 KB buffer per hash) and 4 mix rounds.

## Files

- `balloon.cpp` / `balloon.h` - the CPU reference, used to re-verify every GPU result.
- `sha256-ref.c` / `sha256.h` - the SHA-256 used by the CPU reference.
- `balloon_scan.cpp` - the scan driver (`scanhash_balloon`).
- `cuda_balloon.cu` - the GPU kernel.
- `balloon_selftest.cpp` - the start-up self-test.

## Kernel

One thread per nonce. Each thread keeps its 4096-block buffer in local memory and
stores the blocks as SHA-256 state words, so every message is assembled in
registers and the previous block never round-trips through memory. The block
index stream depends only on the first 32 header bytes; it is computed on the host
once per job and shared by all threads, so a warp always reads the same block.

On sm_80 and newer the SHA-256 additions are issued as multiply-adds (`x * 1 + y`),
which moves them from the busy integer ALU onto the otherwise idle IMAD pipe. Older
architectures compile the plain additions.

Every candidate a launch finds is reported (up to 7 per launch) and submitted, and
the scan then continues after the whole batch.

## Intensity

The default batch is one "wave": as many threads as the card can keep resident,
queried from the CUDA occupancy API at start-up and printed as
`intensity N, M nonces/launch`. A batch that is not a whole number of waves leaves
part of the GPU idle; more waves add little throughput but lengthen every launch.

| card | default batch | live rate | launch |
|---|--:|--:|--:|
| GTX 1080 Ti | 21,504 | ~50-55 kH/s | ~0.37 s |
| RTX 3060 | 17,920 | ~46 kH/s | ~0.39 s |

Live rates are 10-minute pool runs on a hot card (85-90 C); a cold card and
`--benchmark` read higher.

A launch cannot be interrupted and lasts `batch / hashrate` seconds (~0.4 s at the
default). That is how long the miner keeps working on a job the pool has already
replaced, and past about 2 s Windows resets a display-attached GPU. The first
launch is timed and a warning is printed above 1.5 s; raise `-i` with care, and
remember a hot card runs slower than a cold one.

## Validation

- Every candidate is re-hashed on the CPU before it is submitted.
- `balloon_selftest.cpp` runs at start-up and **refuses to start on failure**: a
  pinned known-answer vector for the CPU reference, the GPU's lowest passing nonce
  over a span compared with a CPU enumeration, a range-limit check, and a negative
  test that must fail.
- `--benchmark` loosens the target so candidates are found at a kH/s rate; add
  `--debug` to see them (`found =>`).
- Pool-tested on both cards above: every share accepted (63/63 on the RTX 3060,
  62/62 on the GTX 1080 Ti), including batches that found two shares, at share
  difficulties down to 0.0001.
