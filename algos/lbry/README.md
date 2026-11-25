# lbry — LBRY Credits (LBC)

Migrated from the top-level `lbry/` into `algos/lbry/` (layout B). LBRY is a
self-contained standalone algo — no shared-stage family, and nothing outside this
folder consumes its symbols (only the usual `scanhash_lbry` / `free_lbry`
dispatcher wiring). The chain is
`SHA256d(112B header) → SHA512(32B) → RIPEMD160 ×2 → SHA256d`.

## Bespoke: dead sub-sm_61 split path dropped

The original carried two code paths, selected at runtime by `device_sm`:

- **merged** (`device_sm > 500`) — a single fused kernel, `cuda_lbry_merged.cu`.
- **split** (`device_sm <= 500`) — three separate launches across
  `cuda_sha256_lbry.cu` + `cuda_sha512_lbry.cu` (with a `d_hash` scratch buffer).

Our build floor is **sm_61**, so `device_sm` is always `> 500` at runtime and the
split path was unreachable. The migration therefore **removed** the split path:

- deleted `cuda_sha256_lbry.cu` and `cuda_sha512_lbry.cu` (the only home of the
  `lbry_sha256*` / `lbry_sha512*` / `lbry_ripemd` split kernels and the sole
  RIPEMD-160 in the tree — no external consumers);
- dropped the `merged_kernel` branch, the split-path externs, and the `d_hash`
  buffer from `lbry.cu` (now unconditionally runs `lbry_merged`).

The kept `cuda_lbry_merged.cu` is fully self-contained (its own SHA-256 / SHA-512 /
RIPEMD-160 constants) — the SHA cores are the standard algorithms, but every kernel
is an LBRY-specialized fixed-length / midstate / fused harness, so there is no
shared-stage de-dup to do against `cuda/sha256_device.cuh` or the SHA-512 lib.

## Files

- `lbry.cu` — scan driver (`scanhash_lbry`, `free_lbry`, CPU `lbry_hash` oracle).
- `cuda_lbry_merged.cu` — the fused single-kernel device TU.

## Bug fixed during migration: `c_dataEnd112` symbol overflow

`cuda_lbry_merged.cu` uploaded the block tail with `sizeof(end)` (the 16-word
local, 64 B) into `c_dataEnd112`, a `uint32_t[12]` (48 B) `__constant__`. CUDA 11.8
rejects the oversized copy with `cudaErrorInvalidValue` ("invalid argument") and
copies nothing, so the fused kernel ran with an unset tail and produced wrong
hashes (no valid shares; the error surfaced one loop later via ccminer's
top-of-loop `cudaGetLastError`). Fixed to copy `sizeof(c_dataEnd112)` (48 B) — the
kernel only reads the first 11 words. Latent bug exposed by the CUDA 11.8 migration
(older runtimes tolerated the overflow).
