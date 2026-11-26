# ccminer-kudaraidee

A CUDA miner for NVIDIA GPUs. This branch carries the restructured `algos/`,
`core/` and `cuda/` source tree and the algorithm set from
[tpfuemp/ccminer-tpfuemp](https://github.com/tpfuemp/ccminer-tpfuemp), itself
based on [ccminer-kudaraidee](https://github.com/Kudaraidee/ccminer-kudaraidee)
and [tpruvot/ccminer](https://github.com/tpruvot/ccminer). It targets **CUDA 11.8**
and modern NVIDIA hardware (Pascal sm_61 and newer).

Current version: **1.4.0**.

See [README.txt](README.txt) for the upstream feature/change history, and
[CREDITS.txt](CREDITS.txt) for lineage, contributor credits, and the donation
address.

Requirements
------------

> **This fork targets modern NVIDIA hardware. CUDA 11.8 ships; CUDA 12.9 builds.**

- **CUDA Toolkit:** 11.8 for the shipping build. Older toolkits (10.x, 11.0-11.7)
  are not supported. A second project, `ccminer-kudaraidee-cuda12.vcxproj`, builds the same
  sources against **CUDA 12.9** and is validated against a live pool; 11.8 remains
  the shipping configuration because the two measured within noise of each other.
  The texture-reference API that CUDA 12 removed is gone from the tree.
- **Build toolchain (Windows):** Visual Studio 2022, project file `ccminer-kudaraidee.vcxproj`
  (or `ccminer-kudaraidee-cuda12.vcxproj` for the CUDA 12.9 build).
- **Supported GPU architectures:** Pascal (`sm_61`, GTX 10-series) and newer —
  Turing (`sm_75`) and Ampere (`sm_86`). The default build ships native SASS for
  `sm_61 / sm_75 / sm_86` plus a `compute_86` PTX fallback for later cards.
- **Dropped architectures:** Maxwell (`sm_50/52`), Kepler and Fermi are no longer
  supported; all architecture-specific code paths below `sm_61` have been removed.

**On a pre-Pascal GPU (Maxwell/Kepler/Fermi), or need an older CUDA Toolkit?**
This fork has intentionally dropped that support. Use the upstream project, which
retains the wider hardware and toolkit range:
**https://github.com/tpruvot/ccminer**

Supported algorithms
---------------------

Select with `-a <name>`. Common aliases are shown in parentheses.

| `-a` name | Coin / description |
|---|---|
| `0x10` | ChainOX |
| `allium` | Blake-256 + Keccak + Lyra2 x2 + CubeHash + Skein + Groestl (Garlicoin) |
| `anime` | Animecoin |
| `argon2d500` | Dynamic (DYN) |
| `argon2d1000` | Zero Dynamics Cash |
| `argon2d4096` | Argentum / Myriad (XMY) |
| `argon2d16000` | Alterdot (ADOT) |
| `argon2id1024` | Bitweb (BTW) |
| `balloon` | Balloon hash |
| `bastion` | Hefty bastion |
| `bc3hash` (`bc3`) | BC3Hash (Bitcoin III, from block 64360) |
| `bitcore` (`timetravel10`) | Timetravel-10 |
| `blake` | Blake-256 (SFR) |
| `blake2b` | Blake2-B 512 (BCX) |
| `blake2s` | Blake2-S 256 (NEVA) |
| `blakecoin` | Fast Blake-256 (8 rounds) |
| `bmw` | BMW-256 |
| `bmw512` | BMW-512 |
| `c11` (`flax`) | X11 variant |
| `cryptolight` (`cryptonight-lite`) | AEON CryptoNight (MEM/2) |
| `cryptonight` | Monero-style CryptoNight |
| `curvehash` | CurvehashCoin (secp256k1 EC) |
| `decred` | Decred BLAKE3 (DCP-0011) |
| `deep` | Deepcoin |
| `dmd-gr` (`diamond`) | Diamond-Groestl |
| `equihash` (`equi`) | Equihash 200/9 (Zcash) |
| `equihash144` (`equihash144_5`) | Equihash 144/5 |
| `equihash192` (`equihash192_7`) | Equihash 192/7 |
| `evohash` | EvoAI |
| `evrprogpow` | EvrProgPow (Evrmore) |
| `firopow` | FiroPoW (Firo, StakeCube) |
| `flex` | Flex (Kylacoin, Lyncoin) |
| `fresh` | Freshcoin (Shavite-80) |
| `fugue256` | Fuguecoin |
| `ghostrider` (`gr`) | GhostRider (Raptoreum) |
| `gostcoin` | Double GOST R 34.11 |
| `groestl` | Groestlcoin |
| `heavy` | Heavycoin *(build-gated: `WITH_HEAVY_ALGO`)* |
| `heavyhash` | HeavyHash (oBTC) |
| `hmq1725` (`hmq17`) | HMQ1725 (Doubloons / Espers) |
| `homescrypt` | HomeScrypt v1.2 (Lumenite) |
| `hoohash` (`hoohashv110`, `pepepow`) | HoohashV110 (PEPEPOW) |
| `hsr` (`hshare`) | HShare / HSR (X13 + SM3) |
| `jackpot` | JHA v8 |
| `jha` | JHA |
| `kawpow` | KawPoW (Ravencoin) |
| `keccak` | Keccak-256 (deprecated) |
| `keccakc` | Keccak-256 (CreativeCoin) |
| `lbry` | LBRY Credits (SHA/RIPEMD) |
| `luffa` (`doom`) | Joincoin |
| `lyra2` (`lyra2re`) | Lyra2RE (CryptoCoin) |
| `lyra2v2` (`lyra2rev2`) | Lyra2REv2 (VertCoin) |
| `lyra2z` | Lyra2Z (ZeroCoin) |
| `lyra2z330` | Lyra2Z330 |
| `meowpow` | MeowPow (Meowcoin) |
| `meraki` | Meraki (Telestai) |
| `mike` | Mike (VKAX, FortuneBlock) |
| `minotaurx` | MinotaurX (Avian, Pulsar): yespower 1.0 r=8 behind an x16 walk |
| `mjollnir` | Mjollnir (Hefty hash) |
| `myr-gr` | Myriad-Groestl |
| `neoscrypt` | NeoScrypt (FeatherCoin, Phoenix, UFO…) |
| `xaya` (`neoscrypt-xaya`) | NeoScrypt (XAYA variant) |
| `nist5` | NIST5 (TalkCoin) |
| `odo` (`odocrypt`) | Odocrypt (DigiByte) |
| `pearlhash` (`pearl`) | Pearl (PRL), noisy int8 GEMM; rate in MAC/s |
| `penta` | Pentablake (5× Blake-512) |
| `phi` (`phi1612`) | PHI1612 (BHCoin) |
| `polytimos` | Polytimos |
| `quark` | Quark |
| `qubit` | Qubit |
| `rinhash` | RinHash (Blake3 + Argon2d + SHA3-256) |
| `s3` | S3 (1Coin) |
| `scrypt` | Scrypt |
| `scrypt-jane` | Scrypt-Jane (ChaCha) |
| `sha256csm` | SHA256csm (Galleoncoin) |
| `sha256d` (`bitcoin`, `sha256`) | SHA256d (Bitcoin) |
| `sha256dv` | SHA256d (Veil) |
| `sha256t` | SHA256 ×3 |
| `sha3d` | BSHA3 (Yilacoin, Kylacoin) |
| `sha3t` (`sha3-256t`) | SHA3-256T (Fjarcode, Bitcoin III before block 64360) |
| `sha512256d` | Double SHA-512/256 (Radiant) |
| `sia` | SIA (Blake2B) |
| `sib` | Sibcoin (X11 + Streebog) |
| `skein` | Skein-SHA2 (Skeincoin) |
| `skein2` | Double Skein (Woodcoin) |
| `skunk` | Skein-Cube-Fugue-Streebog |
| `skydoge` | SkyDoge |
| `soterg` | Soteria (X12R core-rotation) |
| `timetravel` | Timetravel (Machinecoin, permuted ×8) |
| `tribus` | Tribus (Denarius) |
| `vanilla` | Blake256-8 (VNL) |
| `veltor` (`thorsriddle`) | Veltor (Thorsriddle + Streebog) |
| `verthash` | Verthash (Vertcoin) |
| `verus` (`verushash`) | VerusHash 2.2 (Verus, PBaaS) |
| `whirlcoin` | Old Whirlcoin (Whirlpool) |
| `whirlpool` (`whirl`) | Whirlpool |
| `whirlpoolx` | WhirlpoolX |
| `whirlpoolx2` | WhirlpoolX2 (CapStash) |
| `wildkeccak` | Boolberry |
| `x11` | X11 (DarkCoin) |
| `x11evo` | Permuted X11 (Revolver) |
| `x13` | X13 (MaruCoin) |
| `x14` | X14 |
| `x15` | X15 |
| `x16r` | X16R (Ravencoin) |
| `x16rt` | X16RT (Veil) |
| `x16rv2` | X16Rv2 |
| `x16s` | X16S (Pigeoncoin) |
| `x17` | X17 |
| `x21s` | X21S |
| `x25x` | X25X (SUQA/SIN) |
| `yescrypt` | Yescrypt / Globalboost-Y (BSTY), custom `--yescrypt-param` |
| `yescryptr8` | BitZeny (ZNY) |
| `yescryptr16` | yescrypt r=16 (Yenten before it moved to `yespowerr16`) |
| `yescryptr16v2` | PPTP |
| `yescryptr24` | JagariCoinR |
| `yescryptr32` | WAVI |
| `yespower` | yespower 1.0, generic (r=32); coin presets below |
| `yespowerr16` | Yenten (YTN), yespower r=16 |
| `yespowereqpay` | EqPay: yespower 1.0 over a 181-byte Qtum-style header |
| `yespoweradvc` | AdventureCoin (ADVC) |
| `yespowersmt` | Smartiecoin (SMT), yespower N=256 r=8 |
| `power2b` (`yespower-b2b`) | yespower-b2b: BLAKE2b head/tail (MicroBitcoin) |
| `zr5` (`ziftr`) | ZR5 (ZiftrCoin) |

The yespower coins share one `-a` name and differ only in `(N, r, pers)`, which
the algo name alone cannot carry. Select one either with its own alias
(`yespowersugar`, `sugarchain`, `yespowerurx`, `yespowerltncg`, `yespowermgpc`,
`yespowertide`, `yespowersmt`, `yespowerarwn`, `yespoweradvc`, `yespoweric`, `yespoweriots`, `yespowerlitb`,
`cpupower`) or with the pool's `"algo"` field, which is honoured on a pool switch.

Run `ccminer --help` for the authoritative list and per-algo notes.

Building — Windows
-------------------

Open `ccminer-kudaraidee.vcxproj` in Visual Studio 2022 with the CUDA 11.8 toolkit
installed, and build the `Release|x64` configuration. From the command line:

    msbuild ccminer-kudaraidee.vcxproj /p:Configuration=Release /p:Platform=x64 /t:Build /m

The executable is produced at `x64\Release\ccminer.exe`.

The default targets are `sm_61 / sm_75 / sm_86` (the `CudaArchList` property in
the project). For a faster single-arch development build, pass `DevArch`:

    msbuild ccminer-kudaraidee.vcxproj /p:Configuration=Release /p:Platform=x64 /p:DevArch=86 /m

(or `build.cmd 86`). It builds into `x64\Release-sm86\ccminer-sm86.exe`, separate
from the release build, and runs only on that architecture.

Building — Linux
----------------

Requires the CUDA 11.8 toolkit and a C/C++ build toolchain
(`build-essential`, `autoconf`/`automake`, and the `curl`, `openssl` and `zlib`
development packages). Then:

    ./build.sh

`build.sh` regenerates the autotools configure script and builds **out-of-tree**
into `build-linux/`, so the resulting binary is `build-linux/ccminer` and the
source tree stays free of build artefacts.

- To change the target GPU architectures, edit `nvcc_ARCH` in `Makefile.am`
  (default `sm_61 / sm_75 / sm_86`) and re-run `./build.sh`.
- For a faster single-arch development build, run `./build.sh 86` (or `61`,
  `75`). It builds into `build-linux-sm86/` and runs only on that architecture.
- **WSL2:** the CUDA driver library `libcuda.so` lives in `/usr/lib/wsl/lib`
  rather than under the toolkit; `configure.sh` already adds it to `LDFLAGS`.

Stratum over TLS
----------------

Pool URLs may use any of `stratum+tcp://`, `stratum+ssl://`,
`stratum+tcps://`, `http://` or `https://`. Both SSL spellings are mapped to
libcurl's `https://` transport, so the connection is a real TLS socket rather
than a plaintext one with an encrypted-looking scheme name. This also works
for URLs delivered by an `X-Stratum` header, and `mining.reconnect` keeps the
original scheme instead of falling back to plaintext. Use `--cert` to supply a
custom CA bundle; libcurl 7.18.2 or newer is required (for
`curl_easy_send()`/`curl_easy_recv()`).

Nebhash
-------

`-a nebhash` mines NEB's `neb-pool/1` protocol (KEP1 convolutional model scan).
The dataset file is consensus-pinned: download `neb-datasets.bin` from the NEB
release page (verify it with `SHA256SUMS`), place it next to the binary or pass
`--neb-dataset=/path/to/neb-datasets.bin`. The miner compares the
pool's `dataset_hash` with the local file at subscribe time and refuses to mine
on a mismatch. It accepts `mining.set_target`, `mining.set_model` and the
object-form `mining.notify`, and re-checks GPU hits against the portable C
reference before submitting `mining.submit`. A pool may also send a
`mining.block` notification after its node accepts a candidate; ccminer then
counts and logs a real solved block instead of inferring one from share
difficulty.

The reference implementation and CUDA kernel live in `algos/nebhash/` under
Apache-2.0 (`LICENSE.NEB`). Full documentation: [English](algos/nebhash/README.md),
[ภาษาไทย](algos/nebhash/README.th.md).

Monitoring and control
----------------------

Two APIs are available and can run at the same time:

- **Binary/telnet API** on `--api-bind` (default `127.0.0.1:4068`), the upstream
  protocol, unchanged.
- **HTTP/JSON REST API**, added in 2026.09 and enabled with `--api-mode http` or
  `--api-mode both` (both protocols on the `--api-bind` port, or HTTP on a
  separate port with `--api-http-port`). Read-only status,
  plus optional **runtime control**: switch pool or algorithm, pause and resume
  mining, and apply per-algo parameters without restarting. It also exposes
  **Prometheus** metrics and a `/health` endpoint for container probes.

Control endpoints are refused unless enabled with `--api-control`;
`--api-token` requires a bearer token on every HTTP route, and `--api-remote`
separates "may read" from "may change" so a dashboard can be given status access
without the ability to redirect hashrate.

The contract is documented in [docs/api-rest.md](docs/api-rest.md) with an
OpenAPI schema in [docs/openapi.yaml](docs/openapi.yaml).

Source-code dependencies
------------------------

The build links against OpenSSL, libcurl and pthreads. For Windows, recent
prebuilt OpenSSL/curl `.lib` files for x86 and x64 are already included in the
tree (`compat/`). To rebuild curl from source, clone its submodule:

    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows

Credits & donation
------------------

Lineage, contributor credits, and the donation address are in
[CREDITS.txt](CREDITS.txt).
