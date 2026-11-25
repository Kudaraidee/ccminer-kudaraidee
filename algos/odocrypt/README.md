# odocrypt (`-a odo` / `-a odocrypt`)

OdoCrypt, DigiByte's Odo algorithm. A periodically-regenerated SPN cipher: the
S-boxes, P-boxes, rotation amounts and round keys are all derived from an epoch
key, so the cipher itself changes on a fixed schedule to resist fixed-function
ASICs.

    hash = first 32 bytes of KeccakP800_12( OdoCrypt(key).Encrypt( header || 0x01 ) )
    key  = nTime - (nTime % ODO_SHAPECHANGE_INTERVAL)          // 864000 s

Ported from cpuminer-opt (not from upstream ccminer); the CUDA side is original
to this fork. `odo_hash_host()` is validated against DigiByte Core.

## Files

- `cuda_odocrypt.cu` - GPU kernel and scan driver.
- `odocrypt_host.cpp` - host reference: epoch table generator plus the CPU hash
  used to re-verify every candidate before submit.
- `odocrypt_jit.cpp` / `.h` - optional per-epoch NVRTC recompile (below).
- `odocrypt.h` - shared definitions.

Includes use the include-path forms (`"miner.h"`, `"cuda/helper.cuh"`); the
parent-relative forms that predate the move into `algos/` would not resolve.

## Kernel notes

- The S-boxes are epoch data, uploaded once per epoch, then staged into shared
  memory per block. They are read as arrays rather than through a pointer
  parameter so the accesses stay `LDS`. `__constant__` is not an option: it
  serialises divergent reads, and these lookups are divergent by construction.
- Rotation amounts are always in `1..63`, which `odocrypt_upload_tables()`
  asserts, so `dev_rot64()` can use a funnel shift with no zero-distance guard.
- One nonce per thread, block size 256. The GPU screens on the top hash word
  against the runtime target; the host re-hashes every candidate before submit,
  so a kernel fault can only ever cost a local reject.

## Per-epoch JIT

The rotation amounts are constant for a whole epoch but are runtime values to
the compiler, so the statically compiled kernel emits a general rotate for each
of the 110 rotations per round. `odocrypt_jit.cpp` recompiles the kernel through
NVRTC once per epoch with those amounts as literals, which lets each one fold to
a funnel shift.

It is optional in the strict sense: if compilation, module load or launch fails
for any reason, the statically compiled kernel continues to be used and mining
is unaffected. Before the JIT'd kernel screens anything it is checked against
`odo_hash_host()` over a range of nonces on every epoch change, and any
disagreement retires it for the rest of the session.

`ODO_JIT_DUMP=<path>` writes the generated source out, which is the only way to
inspect an NVRTC kernel's register usage or SASS (it never appears in a build
log).

## Consensus and stratum

- Header words 0..18 are `be32enc`'d; the nonce occupies the high half of state
  word 9 and is hashed **little-endian**.
- Submit therefore needs `be32enc(nonce)` with `le32enc(ntime)`: a dedicated
  `case ALGO_ODO` in `submit_upstream_work`, different from both the LE/LE
  default and the BE/BE ZR5/rinhash case. This is the algorithm's one historical
  bring-up bug; if shares are rejected as invalid, check here first.
- The epoch boundary changes the entire cipher, so it is a real test axis.

## Validation

`odo_self_test()` runs at init and fails closed: it checks the host reference
against a known-answer vector, then compares 256 GPU digests against that
reference. A card that cannot reproduce the consensus hash refuses to start
rather than mining a session into local rejects. The JIT'd kernel is gated the
same way on every epoch change.

Benchmark (RTX 3060, CUDA 11.8, `-a odo --benchmark`): **~45 MH/s** warm,
about 293-315 kH/W. Live-validated on a DigiByte pool with shares accepted and
no rejects. `--benchmark` loosens the target so candidates are actually found;
without that the screen effectively never fires and the run would only prove
that the kernel launches.
