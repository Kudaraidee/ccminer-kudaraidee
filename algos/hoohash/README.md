# hoohash - HoohashV110 (PEPEPOW)

## Files

- `hoohash.cu` - scan driver (`scanhash_hoohash`, `free_hoohash`) and the CPU check.
- `ref/` - the node's reference `hoohash.c` / `hoohash.h` (MIT, `ref/COPYING`); `blake3.h` maps its BLAKE3 API.
- `cuda_hoohashv110.cu` - device kernels, launchers and the init self-test.
- `hoohash_device.cuh` - matrix generation, the transforms and the branch predicate;
  BLAKE3 comes from the shared `cuda/blake3_device.cuh`.

## Consensus-critical strict floating point

`cuda_hoohashv110.cu` **must** be compiled with strict FP so the on-device
transcendentals match the consensus reference (glibc libm) bit-for-bit. Both build
systems carry this per file; do not "clean it up":

- MSBuild: `FastMath=false`, `FmadOptimization=false`,
  `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`.
- Autotools: dedicated rule `algos/hoohash/cuda_hoohashv110.o` with
  `--maxrregcount=128 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false`.

Every candidate is re-hashed on the CPU with the node's reference code
(`ref/hoohash.c`, MIT, vendored unchanged; its BLAKE3 calls go through the
`blake3.h` shim onto `cuda/blake3_device.cuh`) and submitted on that digest; a
GPU/CPU disagreement is logged. The reference built with MSVC (UCRT, the miner's
`/O2 /MT`) and with glibc gives byte-identical digests (11,004,136 nonces checked),
and the nodes build it with openlibm. The fail-closed init self-test checks a real
block's published digest on the GPU and on the CPU path. Per
`docs/coding-guideline.md` it runs on every target arch, since libdevice code is
compiled per `sm_*`. `ref/hoohash.c` is `#include`d by `hoohash.cu`, which the build
systems do not track: touch `hoohash.cu` after editing anything under `ref/`.

The mining kernel lets each lane walk its own position in the 64x64 scan and evaluates
the expensive transform for all waiting lanes together (`HOO_WAIT_K`, tuned per arch);
results are bit-identical to the reference. On sm_6x it also carries `__launch_bounds__`
(fewer registers, more resident warps); sm_75 and newer are compiled without bounds.
