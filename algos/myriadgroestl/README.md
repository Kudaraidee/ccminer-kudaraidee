# myriadgroestl (`-a myr-gr`)

Myriad-Groestl proof-of-work, relocated from the repo root as part of clearing
algo sources out of the root directory. Groestl-512 over the 80-byte header
followed by SHA-256.

## Layout

Relocation only — every symbol (`scanhash_myriad`, `free_myriad`, and the device
`myriadgroestl_cpu_*`) is unchanged, so the dispatch wiring (`algos.h`
`ALGO_MYR_GR`, `miner.h`, `ccminer.cpp` — live at `case ALGO_MYR_GR`) is
untouched.

- `myriadgroestl.cpp` — host driver / CPU reference (`sph_groestl` + `SHA256`).
- `cuda_myriadgroestl.cu` — the GPU kernels (`myriadgroestl_gpu_hash_quad` +
  `_gpu_hash_sha`) and launchers (`myriadgroestl_cpu_init/setBlock/hash/free`).

It is self-contained and **not** coupled to `cuda_groestl.cu` — its symbols
are its own (`myriadgroestl_*`). The GPU Groestl core comes from the shared
`cuda/groestl512_x2_device.cuh` (two hashes per thread, bitsliced), included via the
project include path. Includes are project-include-dir (`miner.h`,
`cuda/helper.cuh`, `sph/sph_groestl.h`, `openssl/sha.h`) with no parent-relative
(`../`) includes, so the move is source-transparent.

## Bug fixed: the candidate screen was not a 64-bit compare

`myriadgroestl_gpu_hash_sha` screened with
`out_state[7] <= pTarget[1] && out_state[6] <= pTarget[0]`, which also demands the
low word be under target when the high word is already *strictly* below it. That
discards a `t1*(2^32-t0)/(t1*2^32+t0)` share of valid nonces: nil at difficulty
>= 1, where the target's top word is zero and the two forms coincide, but
substantial at the sub-1 difficulties pools hand out. Under `--benchmark` it
additionally demanded `out_state[6] == 0`, leaving the host re-verify unreachable —
so this algo's "0 failures" benchmark result had never meant anything. Now a proper
64-bit compare; the host re-verifies every candidate, so a loose screen is safe by
construction.

## Constants

SHA-256's round constants come from `cuda/sha256_device.cuh`.
`myr_sha256_gpu_constantTable2` stays local: it is the algo-specific K+W fold for
the fully constant second block, expanded at init.

## Profile

Groestl dominates; the SHA-256 kernel is a small remainder, so optimizing that side
cannot pay off. Groestl-512 has no host midstate to exploit either — its block is
1024 bits, so an 80-byte header pads to exactly one block with the nonce inside it.

## Validation

Full rebuild, then benchmark and live pool across a stratum difficulty change:
accepted, 0 rejects, 0 "does not validate", and the benchmark CPU-re-verify now
fires with 0 failures. The host reference re-verifies GPU candidates.
