# Coding Guideline — ccminer-kudaraidee

Status: 2026-07-03 · Scope: all new development and the step-by-step repo cleanup.
This is a **guideline**, not a migration plan. Whenever an algo is touched, it is brought into compliance with this guideline.

---

## 1. Toolchain & Build Targets

- **CUDA Toolkit: 11.8 (fixed).** All code must compile warning-clean with nvcc 11.8. Features that require CUDA 12+ are not used; features removed in CUDA 12 (see §4) are not used either, so a later toolkit upgrade stays cheap.
- **Host language level: C++17** (`-std=c++17` for nvcc and the host compiler). No compiler-specific extensions unless guarded.
- **GPU architectures (default build): sm_61, sm_75, sm_86.** sm_89 may be added as `compute_86`-compatible PTX or its own gencode entry once tested. No gencode entries below sm_61. PTX fallback: embed `compute_86` PTX so newer cards can JIT.
- **Register budget:** keep the global `--maxrregcount=128` out of new per-algo builds; new kernels declare their needs locally via `__launch_bounds__` instead of relying on a global cap.
- **One build system truth:** `Makefile.am`/`configure.ac` (Linux) and one maintained VS solution (Windows, CUDA 11.8). Stale solution files (e.g., CUDA 10 era) are deleted, not kept "just in case" — the git history preserves them.
- **MSBuild does not track `.cuh` dependencies.** Editing a device header does **not** invalidate the `.cu.obj` files that include it, so an incremental build silently keeps stale object code. After changing any `.cuh` (or moving device code), **force-recompile** the dependent `.cu` files — otherwise builds, and especially perf A/Bs, are testing the old kernel.

## 2. Repository Layout

Target structure (applied per algo as it is migrated; new algos start here directly):

```
algos/                     # ALL algo-related files live here
  <algo>/                  # one folder per algo family, lowercase
    <algo>.cu              # host side: scanhash_*, setBlock_*, dispatch
    cuda_<algo>.cu         # device side: kernels
    <algo>-cpu.c|.cpp      # CPU reference used by the verify path (if algo-specific)
    README.md              # optional: algo notes, tuning table, benchmark log
sph/                       # vendored SPH reference hashes (sph_*) — LEAVE AS-IS
cuda/                      # shared DEVICE headers (the device library)
  <primitive>_device.cuh   # one per primitive: sha256_device.cuh, sha512_device.cuh,
                           # blake512_device.cuh, keccak_device.cuh, groestl512_x2_device.cuh,
                           # aes_device.cuh, lyra2_device.cuh, … (23 headers, 2026-08-13)
  selftest_gate.cuh        # shared fail-closed gate for init-time device self-tests
  helper.cuh, vectors.cuh  # math/type helpers (+ helper_alexis, vectors_alexis, vector_uint2x4,
                           # debug); moved from the repo root, include as "cuda/helper.cuh"
core/                      # miner runtime: util, pools, stats, hashlog, bench, sysinfos, cuda.cpp
  gpu/                     # telemetry: nvml, nvapi, nvsettings
  api/                     # HTTP/REST + JSON API (api_*.c/.cpp/.h)
compat/, api/ ...          # unchanged: vendored libs; api/ holds API samples + tests
```

**The `cuda/` listing is descriptive; the directory is the authoritative list.** An earlier
revision named five headers of which only `sha256_device.cuh` existed, and the other four were
planned against as if they did. **Do not name a file here until it exists.**

Rules:
- **Everything algo-specific moves under `algos/`.** No new `.cu` files in the repo root. Root-level algo files (`cuda_skeincoin.cu`, `heavy/`, `x21/`, `sha256/`, `sha256dv/`, …) are relocated the moment their algo is migrated.
- **`sph/` is vendored — leave it as-is** (like `compat/`). It holds the SPH reference hashes (`sph_*`) with ~200 include sites; it is *not* renamed or folded into `crypto/`. The existing top-level `crypto/` is **cryptonight**, not shared crypto — it migrates to `algos/cryptonight/` and then ceases to exist as a top-level dir. There is no repurposed generic `crypto/` folder.
- Variant folders are merged only when the contents are true kernel variants of one algo, selected by the dispatcher (§5) — not by parallel directories. Distinct algos get their own folder even when they share primitives (those live in `cuda/`). Concrete outcome (2026-07-12): `sha256/` → `algos/sha256d/` (sha256d/sha256t/sha256csm family), but `sha256dv/` → `algos/sha256dv/`, because SHA256Dv is Veil's own PoW (64-bit nonce, own stage2 header and stratum path), not a sha256d kernel variant as the plan originally assumed.
- Shared device code lives in `cuda/` only. An algo folder must not contain a private copy of a primitive that exists in `cuda/` (see §3).
- Include paths: algo files include shared headers as `#include "cuda/sha256_device.cuh"` — no relative `../../` chains.

## 3. The Device Library & the Interface Rule

The interface rule from the plan family is binding for all new code:

- Every cryptographic primitive is a **separately callable device function** in a shared header under `cuda/`: e.g., `sha256_transform_full()`, `sha256_final_to_target()`, `blake2s_g()`, `blake2s_compress()`, `blake2s_final_to_target()`, `salsa_core<ROUNDS>()`, `chacha_core<ROUNDS>()`, `scrypt_smix_write()/read()`.
- Fused/merged kernels **call or template-instantiate** these building blocks; they never re-implement them. If a fusion requires a primitive variant, the variant goes into the shared header, not into the algo folder.
- `*_final_to_target()` variants (truncated final rounds) are **only** legal where the hash output is compared directly against a target. They must never feed another hash stage, the CPU-verify path, or share submission. Document the call site with a one-line comment stating why truncation is safe there.
- Round counts, input widths, and similar variations are **template parameters**, not copies.
- Every header function has a unit test against a CPU reference (§7). A header change without passing unit tests does not merge.

## 4. CUDA Coding Standards

Allowed / required:
- `__launch_bounds__` on every kernel; block sizes chosen per arch via the dispatcher, not hard-coded magic numbers.
- Funnel shift (`__funnelshift_r/l`) or verified single-SHF rotations; `__byte_perm` for byte-granular rotations/permutes.
- Vectorized memory access (`uint4`/`uint2x4`), `__restrict__`, `__ldg` or `const __restrict__` read-only paths for scratchpad reads.
- `cp.async` only behind `#if __CUDA_ARCH__ >= 800` with a functional fallback.
- Constants as compile-time literals via full unrolling where the plans call for it; otherwise `__constant__` memory.
- `cudaMemcpyToSymbol` only on job/block changes, never per kernel launch in a loop.

Forbidden in new code (deprecated or removed after 11.8, or legacy patterns):
- **Classic texture/surface references** (`texture<>` globals, `tex1Dfetch` on references). Use texture objects or `__ldg`.
- Warp intrinsics without sync suffix (`__shfl`, `__any`, `__ballot`) — always the `_sync` variants with explicit masks.
- `cudaThreadSynchronize` and other pre-runtime-5 APIs; use `cudaDeviceSynchronize`.
- `%` on thread indices for lane math where `& 31` / shifts are meant; implicit warp-size assumptions without `warpSize`/static_assert.
- New code paths for sm < 61, including `__CUDA_ARCH__` branches for Fermi/Kepler/Maxwell.
- Global `-maxrregcount` reliance (see §1).

Error handling:
- Every CUDA API call goes through `CUDA_SAFE_CALL`/`CUDA_LOG_ERROR`-style checking; kernels are followed by error checks in debug builds. No silent `cudaGetLastError()` swallowing.

## 5. Dispatch & the Selector Gate (post-cleanup form)

The three-level gate from the plan family applies unchanged. It selects between kernel variants (including still-working legacy variants on sm_61+); only code failing the CUDA 11.8 removal criterion (§6) drops out of the gate entirely.

1. **Compile time:** `__CUDA_ARCH__` guards for genuinely arch-specific instructions (e.g., `cp.async` ≥ sm_80) and, where the plans specify it, a build option (e.g., `--enable-legacy-sm`) for retained legacy variants. No branches for arches below the sm_61 build floor.
2. **Runtime:** the selector reads the globals declared in `cuda/helper.cuh` — `device_sm[]` (compute capability × 100), `device_mpcount[]`, and `cuda_arch[]` via `cuda_get_arch(thr_id)` — directly in the algo's host code, to pick a kernel variant, a default intensity or an allocation size. Per-card tuning tables (e.g. the NeoScrypt card detection) live in the algo folder. Throughput is a runtime knob (`-i/--intensity`, through `cuda_default_throughput()`); block size is not — it is a compile-time `TPB` constant with a matching `__launch_bounds__`.

   **There is no `cuda/dispatch.cuh`, and there never has been** — an earlier revision of this section required routing launch configuration through one, and it was planned against for a year. The mechanism is real (`device_sm[]` in ~50 files, `cuda_arch[]` in ~34); only that header was fictional. The paragraph above is the rule. Consolidating these globals behind one header is possible future work, not a requirement in force.
3. **API level:** `scanhash_<algo>`, `<algo>_setBlock_*`, and init/free entry points remain **signature-stable**. Renames or signature changes require touching `ccminer.cpp`/`algos.h` in the same commit and a changelog entry.

**`algos.h` stays hand-edited — never auto-generated.** The `enum` ↔ `algo_names[]` lockstep is **consensus-critical**: the enum index is used on the wire and aliases can't be derived from filenames. File moves and renames do **not** touch it, the enum is **never renumbered**, and there is no generated registry. Removing an algo removes its `scanhash_*` and stubs the dispatch case, but leaves the enum slot intact.

## 6. Cleanup Rules (applied whenever an algo is migrated)

**Removal criterion:** code is removed only if it is **incompatible with the CUDA 11.8 baseline** — it does not build, is removed/non-functional in the 11.x toolchain, or exists solely to serve architectures below the sm_61 build floor. Everything else (working legacy kernel variants for supported arches) is kept and routed through the selector gate (§5), as specified in the plan documents.

Removal list under this criterion:
- **Classic texture/surface reference usage** (deprecated in CUDA 11.x, removed in 12): ported to texture objects or `__ldg` during the algo's migration; the old reference-based code path is deleted once the port is validated.
- **Arch-specific kernels and `__CUDA_ARCH__` branches that only serve sm < 61** (Fermi/Kepler/early-Maxwell special kernels, e.g., the sm_35-only titan kernel path): deleted, including their gencode remnants and build entries.
- Pre-CUDA-11 API relics (`cudaThreadSynchronize`, unsynced warp intrinsics, etc., see §4): replaced in place.
- Build-system relics for unsupported toolchains (e.g., the CUDA 10 VS solution): deleted; one maintained VS solution on CUDA 11.8 remains.

General rules:
- **Delete, don't disable** — for code meeting the removal criterion. Git history is the archive; no commented-out blocks or dead `#ifdef 0` sections in the working tree.
- Kernel variants that still work on sm_61+ under CUDA 11.8 are **not** deleted for being old; they stay selectable behind the runtime gate until a plan-driven migration replaces them (and even then per the plans' deprecation path).
- Removing a code path also removes: its Makefile entries, VS project entries, orphaned `algos.h` aliases, and orphaned helper files (check with the linker, not by guessing).
- A migration commit is only complete when: files live under `algos/<algo>/`, primitives come from `cuda/`, the build has no references to removed paths, and the validation suite (§7) passes.
- Repo-wide sweeps (texture-reference inventory, sm_35 remnants, CUDA 10 solution) are their own small commits, separate from feature work.

## 7. Validation & Benchmarks

- **CPU verify stays on.** `--no-cpu-verify` is a diagnostic flag, never a documented workaround. Any change that only works with verify off is a bug.
- Three test layers, all mandatory for merges that touch hashing code:
  1. **Unit:** each device-library primitive against its CPU reference (RFC/paper test vectors where they exist, e.g., RFC 7693 for BLAKE2s).
  2. **Algo:** `scanhash` end-to-end against the CPU implementation for a fixed header set, including edge nonces.
  3. **Live:** `--benchmark` per target card (at least one Pascal, one Turing, one Ampere) plus a short pool run watching the reject rate.
- Benchmark results are recorded in the algo's `README.md` (card, driver, toolkit, intensity, hashrate) so regressions are attributable.
- Truncated-final-round paths get an explicit test: candidate found by GPU → full hash recomputed on CPU → share accepted.
- **Per-arch validation for FP-transcendental algos.** CUDA libdevice `sin`/`cos`/etc. are compiled **per architecture**, so the SASS for one `sm_*` can differ from another in the last ULP. Any algo whose consensus rests on bit-exact transcendentals (currently **hoohash**) gets its KAT re-run **on each target arch** (at minimum a Pascal sm_61 run in addition to the sm_86 validation), not just once. Integer/bit-exact algos are arch-independent and need this only once.

## 8. Style & Housekeeping

- Language: code, comments, commit messages, and docs in **English** (matches the translated plan documents; avoids mixed-language drift).
- Naming: `snake_case` for functions/files, `UPPER_CASE` for macros, `<algo>_` prefix for exported symbols; kernel names describe the phase (`neoscrypt_fastkdf_in`, `scrypt_smix_read`), not the author or fork of origin. Lowercase filenames; **drop the `coin` suffix** (`quarkcoin.cu`→`quark.cu`, `cuda_skeincoin.cu`→`skein/cuda_skein.cu`); `.cuh` for device headers, `.cu` only for compiled CUDA, `.cpp` for host-only code. No version suffix in a filename when the directory already disambiguates.

  **The `_cpu_` / `_gpu_` tag (document this — it is the convention's biggest legibility gap):**
  > **`_cpu_` = host side. `_gpu_` (and `__global__`) = device side.** A `*_cpu_*` function runs on the **CPU/host**; its job is to *drive* the GPU — allocate device memory, upload constants (`cudaMemcpyToSymbol`), configure grid/block, launch the matching `*_gpu_*` kernel. It is **not** a CPU implementation of the hash. Model pairing: `x11_luffa512_cpu_hash_64` (host launcher) calls `x11_luffa512_gpu_hash_64` (the `__global__` kernel).

  We do **not** rename `_cpu_`→`_host_`: the tag is a *pair*, so it would force renaming every `*_cpu_hash`/`*_gpu_hash` (hundreds of sites). Documenting the meaning is the fix.

  **Symbol patterns (codify current usage):**

  | Kind | Pattern | Example |
  |---|---|---|
  | Scan driver | `scanhash_<algo>` | `scanhash_hoohash` |
  | Cleanup | `free_<algo>` | `free_balloon` |
  | Host-side kernel init | `<family>_<primitive>_cpu_init` | `quark_blake512_cpu_init` |
  | Host-side kernel launch | `<family>_<primitive>_cpu_hash_64` / `_80` | `x11_luffa512_cpu_hash_64` |
  | Device kernel | `<family>_<primitive>_gpu_hash_64` (`__global__`) | `x11_luffa512_gpu_hash_64` |
  | Set block / target | `<algo>_setBlock_80` / `_setTarget` | `balloon_setBlock_80` |
  | Device global | `d_<name>` | `d_hash`, `d_long_state` |
  | `__constant__` symbol | `c_<name>` | `c_data` |

  **Init is canonically `*_cpu_init`.** The three deviant spellings all mean the same thing (host-side init of a GPU module) and are converted onto `*_cpu_init` as each owning file is touched: `*_cuda_init` (e.g. `balloon_cuda_init`), `*_gpu_init` (e.g. `cn_aes_gpu_init`), and bare `*_init` (e.g. `heavyhash_init`). Avoid bare `init()`/`hash()` and `h_`/`d_` mismatches.
- No author-tag comment blocks for new code beyond a standard SPDX/GPL-3.0 header; provenance notes ("based on KlausT …") go in the algo README, once.
- One logical change per commit; migration commits reference the corresponding plan document section.
- Warnings are errors for new files (`-Werror` on the algo being migrated is acceptable staging; repo-wide once cleanup completes).

## 9. Relationship to the Plan Documents

This guideline and the plan documents are complementary, with no conflict: the plans' selector gate and deprecation path remain fully authoritative for working legacy kernel variants; this guideline adds the **removal criterion** (§6) that defines which code leaves the tree entirely — namely only what is incompatible with the CUDA 11.8 / sm_61+ baseline (texture references, sub-sm_61 arch paths, pre-CUDA-11 API relics). The plans remain authoritative for *what to optimize and in which order* per algo; this guideline governs *how* new and migrated code is written and where it lives.
