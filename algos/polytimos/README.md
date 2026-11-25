# polytimos

Guideline: `docs/coding-guideline.md`

## Provenance

- `polytimos.cu` — tpruvot/ccminer polytimos (Polytimos coin): the fixed 6-stage
  chain skein(80) → shabal → echo → luffa → fugue → streebog.

Migrated onto the shared x-family machinery: the 64-byte stages call the bare
`<prim>512_cpu_*` device-launcher names through the
`algos/common/cuda_x_stages.h` bridge (previously this file `#include`d
`x11/cuda_x11.h` directly, bypassing the bridge — it was the last root-level
direct consumer). skein/shabal/luffa/fugue resolve to the bare launchers; echo
uses the optimised alexis 64-byte launcher (`echo512_cpu_hash_64`) with the
tpruvot `*_compat` variant only on arch < 500 (below the sm_61 build floor).

## No fused run

No two consecutive stages are register-resident (shabal, the echo boundary,
luffa, the fugue boundary, then streebog), so there is no fusible run to hand to
the shared fused kernel. The streebog terminal already folds the on-device
target compare (`streebog_cpu_hash_64_final`: two nonces via an atomicExch chain
into `d_resNonce`), so the final stage needs no `cuda_check_hash` pass.

## Correctness note

Every GPU candidate is re-hashed on the host (`polytimos_hash`) before submit,
so a kernel/thermal glitch can only ever cause a local reject
(`result … does not validate on CPU!`), never a bad share.

## Measured rates

| algo      | card     | driver | CUDA | intensity | rate |
|-----------|----------|--------|------|-----------|------|
| polytimos | RTX 3060 | 595.95 | 11.8 | 19        | ~25.8 MH/s (benchmark; 0 CPU-validation failures). Live owed |
