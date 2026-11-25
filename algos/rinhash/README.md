# rinhash (`-a rinhash`)

RinHash proof-of-work. Chains three primitives: **BLAKE3 -> Argon2d -> SHA3-256**.

## Layout

Two translation units are compiled, and this folder holds nothing else:

- `rinhash_scanhash.cpp` — the host scan driver (`scanhash_rinhash`), the CPU
  re-verify (`rinhash_hash`) and the fail-closed init self-test.
- `rinhash_coop.cu` — the GPU driver plus the two kernels this algorithm needs
  of its own.

**No primitive is implemented here.** All three come from code the tree already
had:

| stage | comes from |
|---|---|
| BLAKE3-256 | `cuda/blake3_device.cuh` (shared with decred, hoohash) |
| Argon2d fill | `algos/argon2d/argon2d_fill.cu`, launched unmodified |
| SHA3-256 | `cuda/sha3_device.cuh` (shared with verthash) |

That is deliberate. Nine unreferenced duplicates of BLAKE3, Argon2d, BLAKE2b and
SHA3-256 used to sit in this folder; a second copy of a primitive attracts fixes
that never reach the code that runs, and one of those copies really had gone on
carrying a defect that had been fixed in the live path.

## Kernels

Argon2d dominates: with the other two stages stubbed out, a launch of the same
span runs in 0.04% of the time. So the shape is chosen entirely for the fill.

`argon2_fill` is the argon2d family's, reused **verbatim**: it takes only
geometry (`passes`, `lanes`, `segment_blocks`, `version`, `type`), references no
coin constant, and was already launched across a translation-unit boundary by
`algos/argon2d/argon2d.cu`. RinHash's parameters — t=2, m=64 KiB, lanes=1,
v1.3 — become `segment_blocks = 16`, i.e. the 64 one-kilobyte blocks per nonce
that its `m_cost` asks for. **32 threads cooperate on each block, holding it in
registers**, where a one-thread-per-nonce fill has to stream it through
local memory; that difference is worth about 7x on this card.

Only the two ends are local to RinHash, because the argon2d coins hardwire their
own convention (password and salt both the 80-byte header, so `pwdlen = saltlen
= 80`):

- `rin_coop_initialize` — BLAKE3 of the header, then the Argon2 pre-hash H0, then
  blocks 0 and 1. RinHash's H0 input is 83 bytes, so it is a single Blake2b
  block, where the coins' needs two.
- `rin_coop_finalize` — `argon2_fill` leaves the final XOR in block 0; this takes
  it through Blake2b-long to 32 bytes, then SHA3-256, then the target compare and
  the candidate report.

Nonces per call are sized from the SM count. The count barely matters here —
sweeping it 8x moves the rate under 4% — because with the block in registers
there is no per-thread stream left to evict anything from cache.

## Validation

`rinhash_device_selftest()` runs once per miner thread and is fail-closed: two
published digests (SHA3-256 and BLAKE3 of the empty string), a device-vs-CPU
comparison through the shipping launcher, and a negative control. Every GPU
candidate is additionally re-hashed on the host and `fulltest`ed before submit.

Note for anyone extending the self-test or a harness: the launcher reports **one**
candidate per launch, and with a wide-open target many nonces qualify at once, so
**which** one is reported is a race between concurrently finalizing blocks. A
check that demands a particular nonce will either fail on healthy hardware or
pass vacuously. Verify the reported nonce lies in the launched span, then hold
the device to the reference for *that* nonce.
