# KawPoW (Ravencoin) — `-a kawpow`

ProgPoW-family proof of work: an Ethash DAG in VRAM plus a per-period,
JIT-compiled random program. This is the PoW of Ravencoin (RVN) and other
KawPoW chains.

## Layout

| File | Role |
| :-- | :-- |
| `ethash/` | Vendored CPU reference (cache/DAG/keccak + ProgPoW). Authoritative host hash + share re-verification. |
| `cuda_kawpow.cu` | Device kernels: Ethash DAG generation (`kawpow_gpu_calculate_dag_item` + `kawpow_generate_dag_cpu` launcher) and a self-contained ProgPoW hash used by the standalone bring-up tests. |
| `kawpow_params.h` | kawpow as a `pp_params` instance: epoch/period lengths, the register/cache/math/DAG counts and the RAVENCOINKAWPOW seal words. |
| `progpow_gen_shared.h` | The variant-independent half of the ProgPoW kernel generator (host RNG walk, `merge()`/`math()` emitters, device primitives), used by `../progpow_multi/ppmulti_jit.cpp`. |
| `kawpow_core.{h,cpp}` | Core orchestration (ethash/C++-STL world): ties the DAG, JIT and host reference together behind a plain-C interface, and host-reverifies every candidate. |
| `kawpow.cpp` | ccminer bridge (`scanhash_kawpow` / `free_kawpow`, miner.h world). |

The per-period NVRTC JIT and the epoch/DAG state machine are **not** here: kawpow
drives the shared `../progpow_multi/ppmulti_jit.*` and `ppmulti_dag.*`, the same
pair meowpow, evrprogpow, firopow and meraki use, passing `kPpKawpow` from
`kawpow_params.h`. The kernel is generated per period and the `CUmodule` cached
on it; the DAG is rebuilt on epoch change (every 7500 blocks for kawpow).

`kawpow_core` and `kawpow.cpp` live in separate translation units on purpose:
`miner.h` macroizes `bool` (via `compat/stdbool.h`), which is incompatible with
the C++ standard-library headers ethash requires, so the two never share a TU.

## Provenance

- **`ethash/`** — RavenCommunity/**cpp-kawpow** (Apache-2.0), the KawPoW-tuned
  fork of chfast/ethash: `EPOCH_LENGTH=7500`, `period_length=3`, ProgPoW 0.9.4,
  16 lanes / 32 regs / 11 cache / 18 math / 64 DAG accesses. Vendored flat with
  the `<ethash/…>` includes rewritten to same-directory quoted form.
- **DAG kernel** — translated from xmrig's `kawpow_dag.cl` (one thread per
  64-byte node), GPLv3.
- **JIT program generator** — mirrors kawpowminer/ethminer `libprogpow`
  `ProgPow::getKern` (KISS99 + Fisher-Yates + `math()`/`merge()`), GPLv3;
  verified to produce the identical program to cpp-kawpow's `mix_rng_state`.
- **Stratum** — ethproxy/KawPoW dialect, implemented from the yiimp pool server
  reference (`mining.subscribe` nonce prefix; 7-param `mining.notify`
  header_hash/seed_hash/target/height/nbits; 5-param `mining.submit`
  worker/job/nonce/header_hash/mixhash). Wiring: `core/util.cpp`
  (`kawpow_stratum_notify`, subscribe) + `ccminer.cpp` (submit / gen_work).

## Correctness

Validated bit-for-bit against the official cpp-kawpow ProgPoW test vectors at
every layer (host reference, GPU DAG generation, GPU ProgPoW hash, NVRTC JIT,
and across an epoch boundary). In the miner, an init-time differential self-test
(GPU vs host on the live DAG, plus a bit-flip negative check) gates start, and
**every GPU candidate is recomputed on the host with `progpow::verify` before
submit** — a kernel bug can only ever cause a local reject, never a bad share.

## Nonce / target

- 64-bit nonce. The pool fixes the top 16 bits via the subscribe extranonce
  prefix; the miner searches the low 48 bits. `period = height/3`,
  `epoch = height/7500`.
- The pool sends the full 256-bit share target directly in `mining.notify`
  (not diff-derived); the GPU compares the final hash byte-wise big-endian
  (matching ethash `is_less_or_equal`).

## Benchmark log

| GPU | Arch | Rate | Notes |
| :-- | :-- | :-- | :-- |
| RTX 3060 | sm_86 | ~19.5 MH/s | 2026-07-19, `--benchmark`, warp-cooperative 16-lane JIT search kernel (16 threads/nonce, `__shfl_sync`, coalesced `LDG.E.128` DAG reads, shared `c_dag`, `hack_false` load-scheduling barrier, compile-time-constant DAG modulo, 256-thread blocks). ~20.7× over the initial single-thread-per-nonce kernel (~0.94 MH/s); ~3% off CryptoDredge (20.16 MH/s, same card). DRAM-bandwidth-bound (~89% of the card's ~360 GB/s peak). |

NVRTC is linked statically on both platforms (`nvrtc_static.lib` and
`nvrtc-builtins_static.lib` in the vcxproj; `-lnvrtc_static
-lnvrtc-builtins_static -lnvptxcompiler_static` in `Makefile.am`), so the
binary needs no NVRTC DLL or shared library at run time — only the driver.
