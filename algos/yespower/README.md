# yespower 1.0 -- coin variants

Every yespower 1.0 coin runs the same hash and differs only in `(N, r, pers)`, so
the coins are **aliases of `-a yespower`**: `algo_to_int()` in `algos.h` maps the
name, and the table in `yespower.cu` supplies the parameters.

`pers` is hashed, and one wrong byte gives hashes that look healthy locally and are
all rejected by the pool. The strings are cross-checked against two independent
sources and length-checked against a number written beside each one. Do not retype
them from rendered text; copy them from a raw source.

## Supported aliases

| `-a` name | N | r | pers | bytes |
|---|---|---|---|---|
| `yespower` | 2048 | 32 | (none, or `--yespower-key`) | 0 |
| `yespowerr16`, `yenten` | 4096 | 16 | (none) | 0 |
| `yespowerSUGAR`, `sugarchain` | 2048 | 32 | `Satoshi Nakamoto 31/Oct/2008 Proof-of-work is essentially one-CPU-one-vote` | 74 |
| `yespowerURX` | 2048 | 32 | `UraniumX` | 8 |
| `yespowerLTNCG` | 2048 | 32 | `LTNCGYES` | 8 |
| `yespowerMGPC` | 2048 | 32 | `Magpies are birds of the Corvidae family.` | 41 |
| `yespowerTIDE` | 2048 | 8 | (none) | 0 |
| `yespowerSMT` | **256** | 8 | (none) | 0 |
| `yespowerARWN` | 2048 | 32 | `ARWN` | 4 |
| `yespowerADVC` | 2048 | 32 | `Let the quest begin` | 19 |
| `yespowerIC` | 2048 | 32 | `IsotopeC` | 8 |
| `yespowerIOTS` | 2048 | 32 | `Iots is committed to the development of IOT` | 43 |
| `yespowerLITB` | 2048 | 32 | `LITBpower: The number of LITB working or available for proof-of-work mini` | 73 |
| `cpupower` (see below) | 2048 | 32 | `CPUpower: The number of CPU working or available for proof-of-work mining` | 73 |

Names are case-insensitive. The startup line `yespower 1.0: N=... r=... key=... (n bytes)`
shows what was selected.

## Separate algos, not aliases

- **`power2b`** (MicroBitcoin, alias `yespower-b2b`): N=2048, r=32, 46-byte `pers`,
  BLAKE2b instead of SHA-256 in the head and tail.
- **`yespowereqpay`** (EqPay): ordinary 1.0 parameters (N=2048, r=32, 88-byte
  `pers`), but the proof-of-work covers a **181-byte** header, which no
  `(N, r, pers)` triple can express. See the EqPay section below.

## Precedence

An explicit `--yespower-param` / `--yespower-key` clears the alias preset, wherever
it sits on the command line. A per-pool `"algo"` in the config re-selects the variant
on every pool switch.

## Validation

Only an accepted pool share proves a `pers` string: the GPU and the host re-verify
read the same table and would agree on a wrong one.

| `-a` | Pool (zpool, `.na.mine.zpool.ca`) | Shares |
|---|---|---|
| `yespower` (keyless) | `yespower:6234` | accepted, both RTX 3060 and GTX 1080 Ti |
| `yespowerr16` | zpool | accepted (GTX 1080 Ti) |
| `power2b` | `power2b:6242` | accepted, both cards |
| `yespowerSUGAR` | `yespowerSUGAR:6241` | 6/6 |
| `yespowerURX` | `yespowerURX:6236` | 8/8 |
| `yespowerLTNCG` | `yespowerLTNCG:6245` | 5/5 |
| `yespowerMGPC` | `yespowerMGPC:6247` | 13/13 |
| `yespowerTIDE` | `yespowerTIDE:6239` | 3/3 (job diff 0.095) |
| `yespowerADVC` | `yespowerADVC:6248` | 6/6 |
| `yespowereqpay` | `yespowerEQPAY:6249` | 3/3 |

Not yet confirmed by a pool: `ARWN`, `IC`, `IOTS`, `LITB` (each double-sourced from
cpuminer-opt and zpool's published parameters).

**`cpupower` cannot be validated any more.** CPUchain moved to an EVM chain and
mining on the classic chain is disabled, so no pool serves this algo. Its announced
successor, "YespowerEVM", is a different variant: do not point `-a cpupower` at it.

Kernel shapes r=8, 16 and 32 are all pool-confirmed (r=8 via `yespowerTIDE`).

Every GPU candidate is re-hashed on the host before submission, so a kernel bug
costs a local reject, never a bad share. A fail-closed self-test runs at start-up,
and `-D` adds a per-job GPU-vs-host differential.

## EqPay (`-a yespowereqpay`)

The hashed header is 181 bytes:

| offset | size | field | offset | size | field |
|---:|---:|---|---:|---:|---|
| 0 | 4 | nVersion | 80 | 32 | `hashStateRoot` (pool) |
| 4 | 32 | hashPrevBlock | 112 | 32 | `hashUTXORoot` (pool) |
| 36 | 32 | hashMerkleRoot | 144 | 32 | `prevoutStake.hash`, zero |
| 68 | 4 | nTime | 176 | 4 | `prevoutStake.n` = `0xffffffff` |
| 72 | 4 | nBits | 180 | 1 | signature length, 0 |
| 76 | 4 | nNonce | | | |

Only the head changes: yespower 1.0 uses `pers` as the PBKDF2 salt, so the header
never reaches the salt and smix/pwxform are untouched. The device side is
`yp_sha256_181()` (three SHA-256 blocks instead of two).

Easy to get wrong:

- `prevoutStake.n` must be `0xffffffff`; zero gives a well-formed digest the pool
  rejects.
- The extended tail is stored as the big-endian decode of the wire, like the other
  header words, so one `be32enc()` loop reproduces the wire for all 46 words.
- The state roots arrive with each 32-bit word byte-swapped: undo that per word and
  keep the word order.

### The notify

`mining.notify` carries 10 parameters, with both roots as one 128-hex (64-byte)
value in slot 3, right after prevhash:

```
[ job_id, prevhash, <64 bytes: hashStateRoot || hashUTXORoot>,
  coinb1, coinb2, merkle[], version, nbits, ntime, clean ]
```

A job without the roots is refused rather than mined with a default.

### Self-test

`yespower_eqpay_genesis_kat()` checks the digest of EqPay's genesis header against
the chain's constant at start-up and fails closed. Two more legs flip byte 80 and
byte 180, which a head hashing only 80 or 180 bytes would miss.
