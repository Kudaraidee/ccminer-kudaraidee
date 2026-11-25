# soterg (X12R / Soteria)

`-a soterg` — X12R, an x16r-family **core-rotation** hash with **12** functions:
`BLAKE, SHABAL, GROESTL, JH512, KECCAK, SKEIN, LUFFA, CUBEHASH, SIMD, ECHO,
HAMSI, SHA512`. The per-hash order is derived x16rt-style from a time-masked seed
(`TIME_MASK 0xFFFFFFA0` → `sha256d` of the masked ntime → nibble selection),
constant for all nonces in a job. Standard 80-byte header, default difficulty
(no special target/diff handling). Pure integer/sph + SHA-512, so bit-exact and
arch-independent (sm_61/75/86).

## Provenance
Backported from upstream `Kudaraidee/ccminer-kudaraidee` commit
`322456eaef4576851bffcc5773b85ad85c23170d` ("Add soterg", GitHub @xiaolin1579).
The kernels are the shared x-family stages already in this tree; only the host
driver (`soterg_hash`, `scanhash_soterg`, order derivation) is new.

## Adaptation to this fork
- **Include** `algos/common/cuda_x_stages.h` (the upstream file included the
  retired `cuda_x16.h`, which was folded into the bridge here).
- **De-branded launcher calls** to the current bare `<prim>512_cpu_*` names
  (blake/bmw/groestl/skein/jh/keccak/luffa/cubehash/simd/hamsi/shabal, and the
  64-byte `sha512_cpu_*`) — upstream still called the removed `quark_*` /
  `x11_*` / `x13_*` / `x14_*` / `x17_sha512_*` forwarders. The 80-byte head
  apparatus keeps its `x16_*` / `qubit_luffa512_*` names (no bare equivalent),
  as does the alexis 64-byte `x16_echo512_cpu_hash_64` and the sub-sm_50
  `x11_echo512_*` compat path (unreachable here — `use_compat_kernels` stays 0).
- **Safety-net fix:** upstream's candidate check read
  `if (1 || vhash[7] <= Htarg && fulltest(...))` — the `1 ||` force-accepted
  every GPU candidate and bypassed the host `fulltest` re-verify. Restored to the
  plain guard so every candidate is CPU-re-hashed before submit (a kernel bug can
  then only local-reject, never emit a bad share).

## Validation
- Self-test / benchmark: `ccminer -a soterg --benchmark --no-color -q`
  (loosen the benchmark target if you want the CPU re-verify to fire on more
  candidates). Every GPU candidate is re-hashed on the host and must match.
- Live: a Soteria / X12R pool; bar is "accepted N/N, 0 rejects".

## Optimization: register-resident fused runs
The 64-byte stage chain fuses maximal runs of ≥2 register-resident stages into a
single kernel launch (the shared `cuda_x_fused.cu` machinery used by x16r/x16rv2/
x21s), eliding the 64-byte global round-trip between fused stages. Because
soterg's `enum Algo` is permuted relative to the fused kernel's switch (which
uses the x16r ids), a `soterg_to_x[]` map translates each order nibble to the
shared id before the run detection / upload; the standalone switch still uses
soterg's own enum. Non-fusible boundaries (groestl, simd, echo) stay standalone.
The order (hence the fused-run layout) is derived once per job.

## Benchmark log
- **RTX 3060 (sm_86), CUDA 11.8, intensity 19** — ~15–38 MH/s (varies with the
  per-job function order). **Live-validated on a Soteria pool: naive baseline
  2026-07-16 accepted 3/3 (2 solved); fused build 2026-07-17 accepted 4/4, 0
  rejects, 1 solved** across distinct real orders (one all-fusible 11-run, one
  with a SIMD boundary splitting two runs) — fusion bit-correct on live varied
  orders.
- **Fusion A/B** (interleaved warm, same card): the `--benchmark` fixed order is
  SIMD-heavy (2 standalone SIMD boundaries) — the worst case for fusion — and
  still measures **~+2.5%** (fused ~22.6 vs naive ~22.1 MH/s; fused recovered
  above naive after naive ran, ruling out thermal drift). Favorable live orders
  (longer light-stage runs) gain more; a fused run is never slower than the
  standalone launches it replaces. Bit-correct (GPU==CPU, 0 "does not validate").
