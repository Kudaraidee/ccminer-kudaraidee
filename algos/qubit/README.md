# qubit family

Guideline: `docs/coding-guideline.md`

## Provenance

- `qubit.cu` — tpruvot/ccminer qubit (Myriadcoin-Qubit / Geocoin lineage): the
  fixed 5-stage chain luffa(80) → cubehash → shavite → simd → echo.
- `deep.cu` — deepcoin: luffa(80) → cubehash → echo.
- `luffa.cu` — luffa (Doomcoin): the single luffa(80) stage.

All three call the bare `<prim>512_cpu_*` device-launcher names through the
`algos/common/cuda_x_stages.h` bridge (the 64-byte cubehash/shavite/simd/echo
launcher TUs still live in their legacy `x11/` folders; the bridge forwards
until they de-brand). shavite uses the sp-optimised 64-byte launcher (bare
`shavite512_cpu_hash_64`), simd the single-kernel stage (expansion kept in shared
memory, compression on the same eight lanes), and echo the optimised
alexis 64-byte launcher (`echo512_cpu_hash_64`) with the tpruvot `*_compat`
variant only on arch < 500 (below the sm_61 build floor — effectively dead).

## The luffa-80 first stage

The 80-byte Luffa-512 first stage (klausT midstate precalc) was extracted from
the old branded `qubit/qubit_luffa512.cu` to the shared stage
`algos/stages/cuda_luffa512_80.cu`, de-branded to the bare
`luffa512_setBlock_80` / `luffa512_cpu_hash_80` names. The message injection
of the nonce block is linear and only the nonce word varies, so the host also
precomputes it; the kernels XOR the nonce into one word of each of the five
sub-states and run only the rest of that round. The full kernel is compiled
with `__launch_bounds__(128)` (56 registers instead of 64, faster on Pascal,
neutral on Ampere); the terminal kernel keeps the default. The job-invariant
round-constant upload is folded into `setBlock_80` (no separate `cpu_init`,
which would otherwise collide with the 64-byte `luffa512_cpu_init`). The
branded `qubit_luffa512_*` names remain as thin forwarders for the not-yet
migrated consumers (x16/x21s/ghostrider/timetravel).

qubit runs cubehash and shavite as one kernel (`x_fused_fixed_cpu_hash_64`,
sequence `XF_CUBE_SHAVITE`): the 64-byte state stays in registers between the
two, with shavite's AES table in shared memory. deep has no stage after
cubehash that can fuse (echo is its terminal), so it is unchanged.

## Fused terminals

qubit and deep end in a **fixed** echo stage, so the terminal is folded with the
on-device target compare: `echo512_cpu_hash_64_final` writes the two lowest
passing nonces into `d_resNonce`, eliding the echo `d_hash` store and both
`cuda_check_hash` / `cuda_check_hash_suppl` passes. The compat path
(arch < 500) keeps the tpruvot echo + `cuda_check_hash`.

luffa hashes to 32 bytes and is screened on digest words 6..7, so it runs
`luffa512_cpu_hash_80_final`: the 80-byte kernel without the third Luffa round
(digest words 8..15), with an on-device 64-bit target screen and the same
two-lowest report. No `d_hash` buffer and no `cuda_check_hash` pass remain.

## Correctness note

Every GPU candidate, including the second nonce of a launch, is re-hashed on
the host (`qubithash` / `deephash` / `luffa_hash`) and checked with `fulltest`
before submit, so a kernel/thermal glitch can only ever cause a local reject
(`result ... does not validate on CPU!`), never a bad share.

All three algos run `luffa512_80_device_selftest()` at init and refuse to start
on a mismatch: sph Luffa-512 against the published empty-message vector and a
fixed 80-byte vector, 1000 GPU digests (nonces wrapping past 2^32) against sph,
untouched slots past the batch, the terminal kernel's two-lowest report, and a
flipped-header negative.

## Measured rates

Benchmark rates, CUDA 11.8, default intensity, 0 CPU-validation failures.

| algo  | card        | intensity | rate          |
|-------|-------------|-----------|---------------|
| qubit | GTX 1080 Ti | 22        | ~57 MH/s      |
| qubit | RTX 3060    | 22        | ~45.5 MH/s    |
| deep  | GTX 1080 Ti | 22        | ~90 MH/s      |
| deep  | RTX 3060    | 22        | ~63 MH/s      |
| luffa | GTX 1080 Ti | 24        | ~1350 MH/s    |
| luffa | RTX 3060    | 24        | ~760 MH/s     |

qubit, x11 and x13 shares on these kernels were accepted by a public pool.
