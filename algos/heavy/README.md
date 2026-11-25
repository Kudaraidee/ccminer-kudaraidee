# heavy family

Guideline: `docs/coding-guideline.md`

## Provenance

- `heavy.cu` - Heavycoin (`-a heavy`) and Mjollnir (`-a mjollnir`, via the same
  `scanhash_heavy`): the HEFTY1-anchored multi-hash (hefty -> {sha256, keccak512,
  blake512, groestl512} -> combine).
- `bastion.cu` - Bastion (`-a bastion`): a HEFTY1 prelude feeding a per-nonce
  branching x-family chain (luffa/skein/fugue/whirlpool/shabal/hamsi/echo).
- Support kernels: `cuda_hefty1.cu` (HEFTY1), `cuda_sha256.cu`,
  `cuda_heavy_{blake512,keccak512,groestl512}.cu`, `cuda_combine.cu`,
  `cuda_bastion.cu`.

## Heavy's own blake512/keccak512/groestl512 (namespaced)

heavy's blake512/keccak512/groestl512 are **not** the x-family chain primitives:
they mix the per-nonce HEFTY1 hash into the message, use a variable 80/84-byte
block size, and carry heavycoin's own padding. They previously exported the bare
`blake512_cpu_*` / `keccak512_cpu_*` / `groestl512_cpu_*` names, which collide
**by exact signature** with the names the shared quark primitives need to
de-brand to (`blake512_cpu_init`, ...). They cannot reuse `cuda/*_device.cuh` (that
library exposes only a fixed 64-byte hash, not the hefty-mixed message compress),
so they were renamed to a **`heavy_` namespace** (`heavy_blake512_*`, etc.) -
`heavy.cu` is their sole consumer, and the files are `cuda_heavy_*` - freeing the
bare names (and filenames) for the family-wide de-brand. `hefty_*` was already
namespaced; `sha256_*`/`combine_*` stay bare (heavy-only, no collision).

## heavy/mjollnir status

Both build (`WITH_HEAVY_ALGO`, Windows and Linux) and run `--benchmark`; every GPU
candidate is re-hashed with `heavycoin_hash` and compared byte for byte. Neither
coin has a live pool, so submission is untested end to end. A verified find sets
`work->nonces[0]`/`valid_nonces` and resumes at `nonce + 1`. heavy still uses its
Thrust-based per-stage nonce prune.

## bastion

- **Self-test:** at init the full chain hashes 256 nonces against `bastionhash()`,
  plus a flipped-header negative; a mismatch refuses to start. With `-D`, each new
  job also checks the first and last 515 digests of a full-size batch, which catches
  wrong digests the CPU re-check of candidates cannot see.
- **Cursor:** the kernels scan nonce `n` and hash it byte-swapped: `n + 1` is the
  resume point, `swab32(n)` the submitted nonce. Both candidates get the full CPU check.
- **Stage syncs:** each launch has its own `MyStreamSynchronize` slot from 1 up;
  slot 0 is `hefty_cpu_hash_64`'s.
- **Default intensity:** 22.
- **Branches:** both sides share one buffer (branch 1 from the start, branch 2 from
  the end) with one slot index per nonce; filter and merge copy each 64-byte record
  with 4 lanes. HEFTY1 writes 64-byte records straight into the chain buffer
  (`hefty_cpu_hash_64`).

## HEFTY1 device code

`Mangle`/`Br` have a device-only form (`#ifdef __CUDA_ARCH__`) computing the same
function without branches; the host form (`hefty_cpu_setBlock`, CPU reference) is
unchanged. On sm_80+ the additions are issued as `mad.lo` with a constant 1, which
moves them to the IMAD pipe; older arches compile as before. Shared by heavy,
mjollnir and bastion.

## Correctness note

The `heavy_` rename is a consistent symbolic change (compiles + links;
byte-identical machine code).
