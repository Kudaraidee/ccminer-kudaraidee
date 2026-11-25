# decred (`-a decred`)

Decred (DCR) proof-of-work: BLAKE3-256 over the 180-byte block header, as
specified by DCP-0011. This replaces the Blake-256 implementation that used to
carry the `-a decred` name; the network stopped accepting it when DCP-0011
activated, so the old kernel could not produce a valid share. The previous
implementation is in the git history.

- `decred_blake3.cu` — `scanhash_decred`: the mining kernel, the per-job host
  midstate, the init self-test, and the CPU reference `decred_hash`.
- `cuda/blake3_device.cuh` — the BLAKE3-256 single-chunk primitive, one body
  compiled for both host and device.

## Layout

The header is 180 bytes, so BLAKE3 sees a single chunk of 64 + 64 + 52 and the
nonce at byte 140 falls in block 3, word 3. Blocks 1-2 are therefore per-job
constants: `decred_blake3_prepare` compresses them on the host once per job and
the device kernel only ever compresses the final block.

Stratum and header assembly are unchanged and stay in `ccminer.cpp`
(`ALGO_DECRED`); only the hash function moved. Nothing here byte-swaps the
nonce or the ntime — BLAKE3 consumes the little-endian header words as they
are, and the stratum submit sends raw header bytes, hex-encoded verbatim.

## Validation

- Init self-test, fail-closed, four legs: `BLAKE3("")` against the published
  test vector; a device-vs-host digest on a real header; the host midstate
  against the full-header path; and a flipped header bit that must change the
  device's answer.
- The BLAKE3 primitive is checked against the published test vectors at the
  lengths that decide the flag and block-length logic, including the 180-byte
  three-block case with a flagless middle block.
- Live: accepted shares on a public DCR pool, 2/2 with no rejects, ~5.45 GH/s
  on an RTX 3060 (warm card).
- `compute-sanitizer` memcheck, initcheck, synccheck and racecheck: no
  findings. The racecheck result is weak evidence here, the kernel declaring no
  shared memory.
