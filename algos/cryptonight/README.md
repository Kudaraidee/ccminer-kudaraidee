# cryptonight (`-a cryptonight`, `-a cryptolight`, `-a wildkeccak`)

The CryptoNight family, relocated from the repo-root `crypto/` folder (renamed
to `algos/cryptonight/`). It hosts three dispatched algos plus the shared
CryptoNight device library and the CryptoNight-v1 path used by GhostRider.

- `cryptonight` (`ALGO_CRYPTONIGHT`) — CryptoNight (Monero-style).
- `cryptolight` (`ALGO_CRYPTOLIGHT`) — CryptoLight (AEON-style).
- `wildkeccak` (`ALGO_WILDKECCAK`) — Wild Keccak (Boolberry), with its scratchpad
  RPC.

## Layout

Relocation + folder rename only — every symbol is unchanged, so the dispatch
wiring (`algos.h`, `miner.h`, `ccminer.cpp`) is untouched.

- `cryptonight.cu` / `cryptonight-core.cu` / `cryptonight-extra.cu` /
  `cryptonight-cpu.cpp` / `cryptonight.h` — CryptoNight, and the `*_gr`
  CryptoNight-v1 entry points consumed by `algos/ghostrider/`.
- `cryptolight.cu` / `cryptolight-core.cu` / `cryptolight-cpu.cpp` /
  `cryptolight.h` — CryptoLight.
- `wildkeccak.cu` / `wildkeccak-cpu.cpp` / `wildkeccak.h` — Wild Keccak.
- `cn_{aes,blake,groestl,jh,keccak,skein}.cuh` — shared CN device helpers.
- `xmr-rpc.{cpp,h}` — Monero/stratum RPC; `oaes_lib`/`aesb`/`oaes_config` — AES;
  `cpu/c_keccak.{c,h}` — CPU Keccak; `mman.{c,h}`, `int128_c.h` — platform shims.

Per-file CUDA settings preserved: `MaxRegCount` 64 (cryptonight-core,
cryptolight-core), 255 (cryptonight-extra), 128 (wildkeccak); and the autotools
`cryptonight-core.o` / `cryptonight-extra.o` special build rules were repointed.

## Move fixes

The two external consumers of the folder — `ccminer.cpp` and `core/util.cpp`, each
`#include "crypto/xmr-rpc.h"` — were repointed to
`"algos/cryptonight/xmr-rpc.h"`. All in-folder includes are own-folder /
own-subdir (`cn_aes.cuh`, `cpu/c_keccak.h`, `oaes_lib.h`, ...) or
project-include-dir (`miner.h`, `compat/bitcoin/bignum.hpp`, `compat/...`), with no `../`
includes, so the rest of the move is source-transparent. GhostRider's `*_gr`
externs are resolved by the linker and are unaffected by the path change.

## Validation

Rebuild + benchmark re-validation owed (relocation + rename; correctness follows
from the diff). CryptoNight is CPU-re-verified per candidate before submit.
