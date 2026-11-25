# x16 family (x16r / x16rt / x16rv2 / x16s)

Guideline: `docs/coding-guideline.md`

## Provenance

- `x16r.cu` / `x16s.cu` — tpruvot 2018 lineage; the 16 stage kernels are
  dispatched per-block from the hash-order nibbles of the previous block hash
  (x16r) or a sorted permutation of them (x16s).
- `x16rt.cu` — x16r with the order derived from the **block timestamp** rather than
  the previous block hash: `sha256d` of `ntime & 0xffffff80`, a 128-second window, so
  the order changes on a clock boundary. Does not use the fused-run kernel.
  `whirlpool_midstate` lives in `x16r.cu` — its only definition in the project.
- `x16rv2.cu` — penfold 2019: x16r with Tiger-192 inserted before the keccak,
  luffa and sha512 stages (`tiger192_cpu_hash_64` with `zero_pad_64=1`;
  `tiger192_cpu_hash_80` when the tiger'd stage comes first).
- **Stage launchers moved out.** `x16/` is now dispatcher-only. The 80-byte
  first-stage launchers (echo, fugue, shabal, shavite, simd) plus the alexis
  64-byte echo were de-branded and relocated to the shared `algos/stages/` home
  (layout B): `cuda_echo512_80.cu`, `cuda_fugue512_80.cu`, `cuda_shabal512_80.cu`,
  `cuda_shavite512_80.cu`, `cuda_simd512_80.cu`, `cuda_echo512_64.cu`. Each
  (except simd) sources its generic device code from the matching
  `cuda/*_device.cuh`; the launchers carry the bare `<prim>512_setBlock_80` /
  `<prim>512_cuda_hash_80` symbols (matching the `groestl512_*`/`jh512_*`
  precedent), with `x16_<prim>512_*` forwarders kept until the remaining
  consumers (ghostrider, x21s, x17, skydoge) migrate to the bare names. Only
  `c_PaddedMessage80` stays local to each. Notable exceptions retained across the
  move:
  - **simd** (`cuda_simd512_80.cu`) — no device library (simd is deliberately
    multi-kernel, a fusion boundary — see below), so its FFT machinery stays
    self-contained.
  - **fugue** (`cuda_fugue512_80.cu`) — keeps `x16_fugue512_cpu_init`/`cpu_free`
    `x16_`-named (the bare `fugue512_cpu_init`/`free` are the 64-byte x13 fugue
    bridge; the 80-byte texture lifecycle is distinct, pending the x13/x16 fugue
    merge).
  - **shavite** (`cuda_shavite512_80.cu`) — builds the padded 128-byte block and
    calls the shared `c512` with `count=640` (dead `c512_80`/`x16_shavite512_cpu_init`
    dropped).
  - **echo-64** (`cuda_echo512_64.cu`) — bare `echo512_cpu_hash_64` is canonical
    (optimised alexis 64-byte echo); the tpruvot compat variant is
    `echo512_cpu_hash_64_compat` (arch < 500, below the sm_61 floor — dead). Its
    vcxproj CodeGeneration is the project default `compute_61/75/86` (native
    sm_86 cubin, not the legacy `compute_50/52`-only PTX-JIT).

  The hamsi/whirlpool/sha512 80-byte launchers still live in their legacy family
  folders (`x13/`, `x15/`, `x17/`) and de-brand + relocate when those families
  migrate.

## Device library

The 64-byte stage implementations live in `cuda/*_device.cuh` (§3), each with
an init-time GPU self-test against its `sph_*` reference
(`cuda/xfamily_selftest.cu`, §7 layer 1): blake512, bmw512, groestl512 (quad),
skein512, jh512, keccak512, luffa512, cubehash512, shavite512, simd (kept
multi-kernel, boundary stage), echo512 (tpruvot + alexis), hamsi512, fugue512,
shabal512, whirlpool512, sha512, tiger192. The per-stage launcher TUs still
live in their legacy folders (quark/, x11/, ...) until their own families
migrate.

## Stage-launcher naming

The migrated sources call each 64-byte stage by its bare `<prim>512_cpu_*`
name (`blake512_cpu_hash_64`, `keccak512_cpu_hash_64`, ...). The launcher TUs
that define them still live in their originating family folders and keep the
old family prefix (`quark_`/`x11_`/`x13_`/`x14_`/`x15_`/`x17_`), so a name
bridge in `algos/common/cuda_x_stages.h` forwards the bare names to the current real symbols.
Each family drops its bridge line and renames its launcher to the bare form
when it migrates; the prefixed name then has no users left and is removed. The
`_cpu_` infix keeps these host launchers distinct from the register-resident
device primitives (`blake512_hash_64`) and the bare 80-byte launchers
(`keccak512_cuda_hash_80`).

Exception (fast-path promotion, like the alexis echo): the bare 64-byte
`shavite512_cpu_hash_64` is now the **sp-optimised** launcher (3-arg,
`cuda_x11_shavite512_sp.cu`; self-contained, in-kernel AES init, `__ldg4` I/O —
~+2.5% on x11) rather than a bridge alias. The legacy 6-arg
`x11_shavite512_cpu_hash_64` (shared `c512`) stays for its remaining consumers,
evohash and ghostrider. All x-family sources (x11/x16r/x16rv2/x16s) get the
faster shavite on their mid-chain 64-byte stage.

## Fused stage runs

`algos/common/cuda_x_fused.cu` executes maximal runs of >= 2 consecutive
register-resident stages in one launch (uniform in-kernel switch over a
constant order array), keeping the 64-byte state in registers instead of
bouncing it through `d_hash` between stages. Fusible set: blake, bmw, jh,
keccak, skein, luffa, cubehash, hamsi, shabal, sha512 + tiger192 (kernel
variant with its 6KB shared table). Boundary stages (groestl quad, shavite,
simd, echo, fugue, whirlpool) keep their standalone launchers. The scanhash
computes the run structure once per hash order; x16rv2 builds an effective
sequence with tiger pseudo-stages (id 16).

The fused kernel is pinned to `__launch_bounds__(256, 2)` (128 regs, some
spill): without the min-blocks clause ptxas allocates 235 regs and occupancy
halves, eating the entire win. Fused-path unit test
(`x_fused_device_selftest`, runs once at first order change): per-stage
single launches, an 11-stage chained launch vs the sph chain, and specific
adjacency runs (`sha512·skein·keccak`, `jh·keccak·bmw·bmw`, `bmw·bmw·sha512`).
Those last cover a real blind spot: the deterministic benchmark order leaves
bmw / hamsi / shabal / sha512 **standalone**, so a *fused* sha512 or a bmw·bmw
pair is otherwise never exercised and only appears for live hash orders.

Device-library note learned here: primitive device functions must be
**address-space agnostic** — no `__ldg` on caller pointers (fused kernels
pass register-local state; `__ldg` then faults with error 717).

## Measured rates

| algo   | card     | driver | CUDA | intensity | rate |
|--------|----------|--------|------|-----------|------|
| x16r   | RTX 3060 | 595.95 | 11.8 | default   | ~12.0 MH/s (benchmark, fused; +0.8% vs unfused ~11.9); **live-validated** (19/19 accepted, 6 blocks solved; ~15-16 MH/s warm; live orders exercised fused runs up to length 7; no invalid shares) |
| x16rt  | RTX 3060 | 595.95 | 11.8 | 19        | **live-validated** after the split out of `x16r.cu` (zpool, 45/45 accepted, 0 does-not-validate, 6 distinct timestamp orders): 10.1-18.1 MH/s, avg ~13.3, order-dependent |
| x16rv2 | RTX 3060 | 595.95 | 11.8 | default   | ~12.0 MH/s (benchmark, fused; +3% vs unfused ~11.6); **~14-15.5 MH/s live** (zpool 2026-07-13, 32/32 accepted, vs ~11.6 live unfused) |
| x16s   | RTX 3060 | 595.95 | 11.8 | default   | ~12.4 MH/s (benchmark; fused delta within thermal drift) |

**Correctness note:** every GPU candidate is re-hashed on the host before
submit, so a kernel/thermal glitch can only ever cause a local reject
(`result ... does not validate on CPU!`), never a bad share. A live run at
86 °C produced a few such rejects for one order; the same build at 62 °C ran
19/19 clean — i.e. GPU compute instability under heat, caught by the guard,
not a logic error.

**Benchmark note:** `--benchmark` used to run the degenerate hash order
`AA55555555555555` (echo, echo, skein×14). Since 2026-07-13 the benchmark
header for x16r/x16rv2 encodes the order `0123456789ABCDEF` — every stage
exactly once, deterministic — so benchmark rates are honest full-chain
numbers, comparable across runs. The CPU re-hash validation also covers all
16 stages. **For fusion the alternating bench order is the worst case** (it
minimizes run lengths); random live orders cluster fusible stages into longer
runs, which is why the live gain (~+20-30%) far exceeds the +3% bench delta.
