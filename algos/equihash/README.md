# equihash

Equihash proof-of-work. Three parameter sets share `ALGO_EQUIHASH`; the `(n,k)`
pair and the personalization are selected at runtime (see `algos.h`):

| algo | (n,k) | solver |
|---|---|---|
| `equihash` | 200/9 | `cuda_equi.cu` |
| `equihash144` / `equihash144_5` | 144/5 | `cuda_equi24b.cu` |
| `equihash192` / `equihash192_7` | 192/7 | `cuda_equi24b_192.cu` |

`(n,k)` is fixed by the `-a` algo name because it selects the CUDA kernel. A
pool may override only the personalization, through `mining.notify`.

## Layout

- `equi.cpp` -- miner driver (`scanhash_equihash` / `free_equihash`).
- `equihash.cpp` / `equihash.h` -- variant dispatch, solver selection, submit.
- `equi-stratum.cpp` -- stratum protocol handling.
- `equi_pack.h` -- index/minimal-representation packing for submit.
- `cuda_equi.cu` / `eqcuda.hpp` -- the 200/9 GPU solver.
- `cuda_equi24b.cu` / `cuda_equi24b_192.cu` / `cuda_equi24b.h` /
  `equi24b_params.h` -- the 144/5 and 192/7 GPU solver. The `_192` file is a
  wrapper that includes the first with different constants.
- `equi_verify.h` / `equi_verify_144.cpp` / `equi_verify_192.cpp` /
  `equi_tromp.h` -- host consensus verifiers, one TU per variant.
- `blake2b_tromp.cuh`, `blake2/` -- BLAKE2b for the device and the host.

## Solver notes

The 144/5 and 192/7 solver uses few large buckets (4096), a per-rest linked
list in shared memory, and independent per-layer arrays. Its arena is large and
grows with the round count, so it is larger for 192/7 than for 144/5; a device
that cannot allocate it cannot mine that variant, and there is no fallback
solver. `equi24b_params.h` documents the geometry.

Every solution is re-verified on the host before submit, so a solver defect
costs a local reject rather than a bad share. The verifier must match the
variant -- see `equi_verify.h`.

## Build

Per-file CUDA options: `cuda_equi.cu` keeps its code-generation and
`-Xptxas -dlcm=ca -dscm=cs` settings; the 144/5 and 192/7 objects are built
from one source with `-DEQ_WN`/`-DEQ_WK` and `-Xptxas -dscm=cs`.

A build rule naming a wrapper `.cu` must also name the source it includes and
that source's headers as prerequisites. An explicit rule gets no automatic
dependency scanning, so without them the wrapper's object goes stale and the
two variants in one binary run different code.
