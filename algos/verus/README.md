# verus - VerusHash 2.2 (Verus Coin and PBaaS chains)

`-a verus` (alias `verushash`).

## Provenance and licences

| file | origin | licence |
|---|---|---|
| `haraka.c`, `haraka.h`, `haraka_portable.c`, `haraka_portable.h` | kste/haraka via the VerusCoin ccminer and cpuminer-opt `algo/verus` | MIT |
| `verus_clhash.c`, `verus_clhash.h`, `verus_clhash_iter.h` | Michael Toutonghi (VerusCoin); CLHash by Lemire and Kaser; via cpuminer-opt | Apache-2.0 |
| `verus-kat.h` | Verus mainnet block 4174000 as serialised, and its published hash | data |
| everything else | this repository | GPL-3.0 |

Apache-2.0 is compatible with GPLv3, so the combined binary is GPLv3. Local changes to the
vendored files are marked `EDIT (ccminer-kudaraidee)`.

## How it works

The preimage is the 140-byte header, `fd 40 05` and the 1344-byte solution (1487 bytes =
46 * 32 + 15). The first 1472 bytes are constant for a job; the miner varies only the last 15:
7 bytes of the pool extranonce, `work->data[32]` (rolled per loop and per thread) and a
32-bit counter. Jobs are hashed in PBaaS canonical form and shares are submitted that way
(zero header nonce, nonce bytes in the solution tail). Other jobs are refused.

* Host, once per job (`verus_host.c`, AES-NI): 46 Haraka512 blocks, then 276 Haraka256 blocks
  for the 8832-byte key. Every candidate is re-hashed in full on the host before it is
  submitted.
* Device, per nonce (`verus_warp.cuh` over `verus_portable.h`): the 32-iteration CLHash core,
  then the keyed Haraka512 truncated to word 7, the word the target test reads.
  * One hash per warp: the core picks one of 8 cases from the data, so 32 hashes in a warp
    would run all of them. The lanes split AES, carry-less multiply and unpack instead.
  * The key is never copied per hash: every warp reads one pristine key in shared memory
    through a copy-on-write log of the at most 64 slots a hash writes.
* The hash arithmetic exists once: `verus_portable.h` is templated over the value type, and
  the scalar host path and the warp path run the same code.

Launch shape per arch: sm_61 12 warps x 3 blocks, sm_7x 24 x 1, sm_8x 32 x 1. About 12 MH/s
on a GTX 1080 Ti (pool run) and 15 MH/s on an RTX 3060.

## Self-tests (fail closed)

At start-up, per GPU: the host chain against block 4174000's published hash; the device on the
same block; the device against the host on a fixed random job over 1024 nonces (the KAT block
alone exercises only one of the 8 cases); and a flipped bit that must change the result.
With `-D`, every new job also compares 4133 device results with the host.
