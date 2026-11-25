
ccminer-kudaraidee 1.4.0   "Hoohash up to 2.9x; equihash 144/5 and 192/7 +16-38%; share-handling fixes"
---------------------------------------------------------------

Donation addresses and contributor credits are in CREDITS.txt.

>>> Introduction <<<

This is a CUDA accelerated mining application for NVIDIA GPUs. It supports a
large set of proof-of-work algorithms -- see the complete, current list under
the -a option below (or run "ccminer --help", or see README.md).

Some of these coins have a VERY NOTABLE nVidia advantage over competing AMD
(OpenCL only) implementations.

We did not take a big effort on improving usability, so please set
your parameters carefuly.

THIS PROGRAMM IS PROVIDED "AS-IS", USE IT AT YOUR OWN RISK!

If you're interessted and read the source-code, please excuse
that the most of our comments are in german.

>>> Command Line Interface <<<

This code is based on the pooler cpuminer and inherits
its command line interface and options.

  -a, --algo=ALGO       specify the algorithm to use
                          0x10                ChainOX
                          allium              Blake/Keccak/Lyra2/CubeHash/Skein/Groestl (Garlicoin)
                          anime               Animecoin
                          argon2d500          Dynamic (DYN)
                          argon2d1000         Zero Dynamics Cash
                          argon2d4096         Argentum / Myriad (XMY)
                          argon2d16000        Alterdot (ADOT)
                          argon2id1024        Bitweb (BTW)
                          balloon             Balloon hash
                          bastion             Hefty bastion
                          bc3hash / bc3       BC3Hash (Bitcoin III, from block 64360)
                          bitcore             Timetravel-10
                          blake               Blake-256 (SFR)
                          blake2b             Blake2-B 512 (BCX)
                          blake2s             Blake2-S 256 (NEVA)
                          blakecoin           Fast Blake-256 (8 rounds)
                          bmw                 BMW-256
                          bmw512              BMW-512
                          c11 / flax          X11 variant
                          cryptolight         AEON CryptoNight (MEM/2)
                          cryptonight         Monero-style CryptoNight
                          cpupower            CPUchain (CPU), a yespower preset
                          curvehash           CurvehashCoin (secp256k1 EC)
                          decred              Decred BLAKE3 (DCP-0011)
                          deep                Deepcoin
                          dmd-gr              Diamond-Groestl
                          equihash / equi     Equihash 200/9 (Zcash)
                          equihash144         Equihash 144/5 (alias equihash144_5)
                          equihash192         Equihash 192/7 (alias equihash192_7)
                          evohash             EvoAI
                          evrprogpow          EvrProgPow (Evrmore)
                          firopow             FiroPoW (Firo, StakeCube)
                          flex                Flex (Kylacoin, Lyncoin)
                          fresh               Freshcoin (Shavite-80)
                          fugue256            Fuguecoin
                          ghostrider / gr     GhostRider (Raptoreum)
                          gostcoin            Double GOST R 34.11
                          groestl             Groestlcoin
                          heavy               Heavycoin (build-gated WITH_HEAVY_ALGO)
                          heavyhash           HeavyHash (oBTC)
                          hmq1725 / hmq17     HMQ1725 (Doubloons / Espers)
                          homescrypt          HomeScrypt v1.2 (Lumenite)
                          hoohash / pepepow   HoohashV110 (PEPEPOW)
                          hsr / hshare        HShare / HSR (X13 + SM3)
                          jackpot             JHA v8
                          jha                 JHA
                          kawpow              KawPoW (Ravencoin)
                          keccak              Keccak-256 (deprecated)
                          keccakc             Keccak-256 (CreativeCoin)
                          lbry                LBRY Credits (SHA/RIPEMD)
                          luffa / doom        Joincoin
                          lyra2 / lyra2re     Lyra2RE (CryptoCoin)
                          lyra2v2             Lyra2REv2 (VertCoin)
                          lyra2z              Lyra2Z (ZeroCoin)
                          lyra2z330           Lyra2Z330
                          meowpow             MeowPow (Meowcoin)
                          meraki              Meraki (Telestai)
                          mike                Mike (VKAX, FortuneBlock)
                          minotaurx           MinotaurX (Avian, Pulsar)
                          mjollnir            Mjollnir (Hefty hash)
                          myr-gr              Myriad-Groestl
                          neoscrypt           NeoScrypt (FeatherCoin, Phoenix, UFO...)
                          xaya                NeoScrypt (XAYA variant)
                          nist5               NIST5 (TalkCoin)
                          odo / odocrypt      Odocrypt (DigiByte)
                          pearlhash / pearl   Pearl (PRL), noisy int8 GEMM (rate in MAC/s)
                          penta               Pentablake (5x Blake-512)
                          phi / phi1612       PHI1612 (BHCoin)
                          polytimos           Polytimos
                          power2b             yespower-b2b (MicroBitcoin)
                          quark               Quark
                          qubit               Qubit
                          rinhash             RinHash (Blake3 + Argon2d + SHA3-256)
                          s3                  S3 (1Coin)
                          scrypt              Scrypt (Litecoin, Dogecoin, ...)
                          scrypt-jane         Scrypt-Jane (ChaCha)
                          sha256csm           SHA256csm (Galleoncoin)
                          sha256d             SHA256d (Bitcoin)
                          sha256dv            SHA256d (Veil)
                          sha256t             SHA256 x3
                          sha3d               BSHA3 (Yilacoin, Kylacoin)
                          sha3t               SHA3-256T (Fjarcode, Bitcoin III before block 64360)
                          sha512256d          Double SHA-512/256 (Radiant)
                          sia                 SIA (Blake2B)
                          sib                 Sibcoin (X11 + Streebog)
                          skein               Skein-SHA2 (Skeincoin)
                          skein2              Double Skein (Woodcoin)
                          skunk               Skein-Cube-Fugue-Streebog
                          sugarchain          alias of yespowersugar
                          skydoge             SkyDoge
                          soterg              Soteria (X12R core-rotation)
                          timetravel          Timetravel (Machinecoin, permuted x8)
                          tribus              Tribus (Denarius)
                          vanilla             Blake256-8 (VNL)
                          veltor              Veltor (Thorsriddle + Streebog)
                          verthash            Verthash (Vertcoin)
                          verus / verushash   VerusHash 2.2 (Verus, PBaaS)
                          whirlcoin           Old Whirlcoin (Whirlpool)
                          whirlpool           Whirlpool
                          whirlpoolx          WhirlpoolX
                          whirlpoolx2         WhirlpoolX2 (CapStash)
                          wildkeccak          Boolberry (Stratum only)
                          x11                 X11 (DarkCoin)
                          x11evo              Permuted X11 (Revolver)
                          x13                 X13 (MaruCoin)
                          x14                 X14
                          x15                 X15
                          x16r                X16R (Ravencoin)
                          x16rt               X16RT (Veil)
                          x16rv2              X16Rv2
                          x16s                X16S (Pigeoncoin)
                          x17                 X17
                          x21s                X21S
                          x25x                X25X (SUQA/SIN)
                          yescrypt            Globalboost-Y (BSTY) or custom --yescrypt-param
                          yescryptr8          BitZeny (ZNY)
                          yescryptr16         yescrypt r=16 (Yenten before yespowerr16)
                          yescryptr16v2       PPTP
                          yescryptr24         JagariCoinR
                          yescryptr32         WAVI
                          yespower            yespower 1.0, generic (r=32)
                          yespowerr16         Yenten (YTN), yespower r=16
                          yespoweradvc        AdventureCoin (ADVC)
                          yespowerarwn        Arowana (ARWN)
                          yespowereqpay       EqPay (Qtum-style 181-byte header)
                          yespoweric          IsotopeC
                          yespoweriots        IOTS
                          yespowerlitb        LightBit (LITB)
                          yespowerltncg       LightningCash-Gold
                          yespowermgpc        MagpieCoin (MGPC)
                          yespowersmt         Smartiecoin (SMT)
                          yespowersugar       Sugarchain (SUGAR)
                          yespowertide        Tidecoin (TDC)
                          yespowerurx         UraniumX (URX)
                          zr5 / ziftr         ZR5 (ZiftrCoin)

  -d, --devices         gives a comma separated list of CUDA device IDs
                        to operate on. Device IDs start counting from 0!
                        Alternatively give string names of your card like
                        gtx780ti or gt640#2 (matching 2nd gt640 in the PC).

  -i, --intensity=N[,N] GPU threads per call 8-25 (2^N + F, default: 0=auto)
                        Decimals and multiple values are allowed for fine tuning
      --cuda-schedule   Set device threads scheduling mode (default: auto)
  -f, --diff-factor     Divide difficulty by this factor (default 1.0)
  -m, --diff-multiplier Multiply difficulty by this value (default 1.0)
  -o, --url=URL         URL of mining server
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
      --cert=FILE       certificate for mining server using SSL
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy
  -t, --threads=N       number of miner threads (default: number of nVidia GPUs in your system)
  -r, --retries=N       number of times to retry if a network call fails
                          (default: retry indefinitely)
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 15)
      --shares-limit    maximum shares to mine before exiting the program.
      --time-limit      maximum time [s] to mine before exiting the program.
  -T, --timeout=N       network timeout, in seconds (default: 300)
  -s, --scantime=N      upper bound on time spent scanning current work when
                        long polling is unavailable, in seconds (default: 5)
      --submit-stale    ignore stale job checks, may create more rejected shares
  -n, --ndevs           list cuda devices
  -N, --statsavg        number of samples used to display hashrate (default: 30)
      --no-gbt          disable getblocktemplate support (height check in solo)
      --no-longpoll     disable X-Long-Polling support
      --no-stratum      disable X-Stratum support
  -q, --quiet           disable per-thread hashmeter output
      --no-color        disable colored output
  -D, --debug           enable debug output
  -P, --protocol-dump   verbose dump of protocol-level activities
  -b, --api-bind=port   IP:port for the miner API (default: 127.0.0.1:4068), 0 disabled
      --api-remote      Allow remote control, like pool switching, imply --api-allow=0/0
      --api-allow=...   IP/mask of the allowed api client(s), 0/0 for all
      --api-mode=MODE   API protocol: binary (default), http, or both
      --api-token=TOK   require 'Authorization: Bearer TOK' on every HTTP route
      --api-cors=ORIG   value for Access-Control-Allow-Origin, enables OPTIONS
      --api-http-port=P serve HTTP on its own port instead of sniffing one port
      --api-control     Allow the API to pause/resume/stop mining at runtime
      --api-control-min-interval=N  Seconds between accepted state changes (default: 15)
      --api-control-park-timeout=N  Milliseconds to wait for threads to idle (default: 30000)
      --max-temp=N      Only mine if gpu temp is less than specified value
      --max-rate=N[KMG] Only mine if net hashrate is less than specified value
      --max-diff=N      Only mine if net difficulty is less than specified value
      --max-log-rate    Interval to reduce per gpu hashrate logs (default: 3)
      --pstate=0        will force the Geforce 9xx to run in P0 P-State
      --plimit=150W     set the gpu power limit, allow multiple values for N cards
                          on windows this parameter use percentages (like OC tools)
      --tlimit=85       Set the gpu thermal limit (windows only)
      --keep-clocks     prevent reset clocks and/or power limit on exit
      --hide-diff       Hide submitted shares diff and net difficulty
  -B, --background      run the miner in the background
      --benchmark       run in offline benchmark mode
      --cputest         debug hashes from cpu algorithms
      --cpu-affinity    set process affinity to specific cpu core(s) mask
      --cpu-priority    set process priority (default: 0 idle, 2 normal to 5 highest)
  -c, --config=FILE     load a JSON-format configuration file
                        can be from an url with the http:// prefix
  -V, --version         display version information and exit
  -h, --help            display this help text and exit


Scrypt specific options:
  -l, --launch-config   gives the launch configuration for each kernel
                        in a comma separated list, one per device.
      --interactive     comma separated list of flags (0/1) specifying
                        which of the CUDA device you need to run at inter-
                        active frame rates (because it drives a display).
  -L, --lookup-gap      Divides the per-hash memory requirement by this factor
                        by storing only every N'th value in the scratchpad.
                        Default is 1.
      --texture-cache   comma separated list of flags (0/1/2) specifying
                        which of the CUDA devices shall use the texture
                        cache for mining. Kepler devices may profit.
      --no-autotune     disable auto-tuning of kernel launch parameters

CryptoNight specific options:
  -l, --launch-config   gives the launch configuration for each kernel
                        in a comma separated list, one per device.
      --bfactor=[0-12]  Run Cryptonight core kernel in smaller pieces,
                        From 0 (ui freeze) to 12 (smooth), win default is 6
                        This is a per-device setting like the launch config.

Wildkeccak specific:
  -l, --launch-config   gives the launch configuration for each kernel
                        in a comma separated list, one per device.
  -k, --scratchpad url  Url used to download the scratchpad cache.


>>> Examples <<<


Example for Heavycoin Mining on heavycoinpool.com with a single gpu in your system
    ccminer -t 1 -a heavy -o stratum+tcp://stratum01.heavycoinpool.com:5333 -u <<username.worker>> -p <<workerpassword>> -v 8


Example for Heavycoin Mining on hvc.1gh.com with a dual gpu in your system
    ccminer -t 2 -a heavy -o stratum+tcp://hvcpool.1gh.com:5333/ -u <<WALLET>> -p x -v 8


Example for Fuguecoin solo-mining with 4 gpu's in your system and a Fuguecoin-wallet running on localhost
    ccminer -q -s 1 -t 4 -a fugue256 -o http://localhost:9089/ -u <<myusername>> -p <<mypassword>>


Example for Fuguecoin pool mining on dwarfpool.com with all your GPUs
    ccminer -q -a fugue256 -o stratum+tcp://erebor.dwarfpool.com:3340/ -u YOURWALLETADDRESS.1 -p YOUREMAILADDRESS


Example for Groestlcoin solo mining
    ccminer -q -s 1 -a groestl -o http://127.0.0.1:1441/ -u USERNAME -p PASSWORD

Example for Boolberry
    ccminer -a wildkeccak -o stratum+tcp://bbr.suprnova.cc:7777 -u tpruvot.donate -p x -k http://bbr.suprnova.cc/scratchpad.bin -l 64x360

Example for Scrypt-N (2048) on Nicehash
    ccminer -a scrypt:10 -o stratum+tcp://stratum.nicehash.com:3335 -u 3EujYFcoBzWvpUEvbe3obEG95mBuU88QBD -p x

For solo-mining you typically use -o http://127.0.0.1:xxxx where xxxx represents
the rpcport number specified in your wallet's .conf file and you have to pass the same username
and password with -O (or -u -p) as specified in the wallet config.

The wallet must also be started with the -server option and/or with the server=1 flag in the .conf file

>>> Configuration files <<<

With the -c parameter you can use a json config file to set your prefered settings.
An example is present in source tree, and is also the default one when no command line parameters are given.
This allow you to run the miner without batch/script.


>>> API and Monitoring <<<

With the -b parameter you can open your ccminer to your network, use -b 0.0.0.0:4068 if required.
On windows, setting 0.0.0.0 will ask firewall permissions on the first launch. Its normal.

Default API feature is only enabled for localhost queries by default, on port 4068.

You can test this api on linux with "telnet <miner-ip> 4068" and type "help" to list the commands.
Default api format is delimited text. If required a php json wrapper is present in api/ folder.

An HTTP/JSON REST API is available as well: --api-mode http (or both) serves
status, Prometheus metrics and /health, and with --api-control also pool,
algorithm and pause/resume control. See docs/api-rest.md.


>>> Additional Notes <<<

This fork targets modern NVIDIA hardware and CUDA 11.8 only. It builds with the
CUDA 11.8 Toolkit (Visual Studio 2022 on Windows, project file ccminer-kudaraidee.vcxproj)
and runs on GPUs of compute capability 6.1 and newer -- Pascal (sm_61, the
GTX 10-series), Turing (sm_75) and Ampere (sm_86). The default build ships
native SASS for sm_61/75/86 plus a compute_86 PTX fallback for later cards.

Support for Maxwell (sm_50/52), Kepler and Fermi has been dropped, along with
all architecture-specific code paths below sm_61, so we can implement and
optimise new algorithms for the latest hardware without carrying legacy
fallbacks.

If you have a pre-Pascal GPU (Maxwell/Kepler/Fermi) or need an older CUDA
Toolkit, use the upstream project instead -- it retains that wider hardware and
toolkit range: https://github.com/tpruvot/ccminer

>>> RELEASE HISTORY <<<
  Nov. 23rd 2026 ccminer-tpfuemp 2026.11.1

                  Speed
                    hoohash    RTX 3060 ~464 -> ~1350 kH/s, GTX 1080 Ti
                               ~1130 -> ~2760 kH/s
                    equihash   144/5: GTX 1080 Ti 61 -> 81, RTX 3060 39 -> 54
                               Sol/s; 192/7: 37 -> 43 and 25 -> 30 Sol/s

                  Fixes
                    hoohash    every share is re-checked on the CPU with the
                               node's reference code before submit
                    equihash   144/5 and 192/7: on multi-GPU rigs every solver
                               ran on GPU 0; shares lost after a pause, an
                               algo switch or a pool personalization change
                    rinhash    multi-GPU rigs shared one GPU buffer; a second
                               valid nonce in a launch was lost
                    bc3hash    duplicate share submits ("already sent")
                    verthash   every hash of a job was wrong when the header
                               version's low byte was 0xf8 or higher
                    yescrypt   variants: up to 31 nonces at the end of a range
                               were lost
                    timetravel wrong hash order for a GPU still on the old job
                               (multi-GPU)
                    x16r, x16rt, x16rv2, x16s, x17, hmq1725, skydoge, x21s,
                    soterg, whirlpoolx2, anime, heavyhash: a candidate that
                    failed the CPU check skipped the rest of the launch, and
                    half of these faults were not logged

  Nov. 13th 2026 ccminer-tpfuemp 2026.11

                  New algorithms
                    pearlhash        Pearl (PRL), noisy int8 GEMM (alias
                                     pearl): dp4a kernel on sm_61+, IMMA on
                                     sm_80+; rate in int8 MAC/s, about
                                     37 T on an RTX 3060, 17-18 T on a
                                     GTX 1080 Ti

                  Speed
                    equihash   144/5 and 192/7: about 3x on a GTX 1080 Ti
                               (144/5 ~20 -> ~60 Sol/s) and 1.8x on an
                               RTX 3060; GPU memory 3.75 GB (144/5) and
                               5.97 GB (192/7), was 4.28 / 6.50 GB; no more
                               "solutions are being LOST" warnings on 192/7
                    bc3hash    +27% on an RTX 3060, +6% on a GTX 1080 Ti;
                               one CPU core per GPU no longer spins
                               (--cuda-schedule overrides)
                    curvehash  RTX 3060 4.0 -> ~7.6 MH/s, GTX 1080 Ti
                               ~1.3 -> ~3.1 MH/s
                    qubit, deep, luffa: single-kernel SIMD, fused
                               cubehash+shavite, faster luffa-80
                    fugue256   faster kernels
                    yespower   no re-hash of the rest of a batch after a find

                  Fixes
                    fugue256   submitted a stale nonce
                    x11 family, quark, x16, x17 and others: the second GPU
                    nonce was not checked against the target before submit
                    lbry, skunk, veltor, sia, quark, anime: second nonce check
                    blake2s, decred: every candidate is checked; more than
                    two valid candidates were partly lost
                    whirlpool, whirlcoin: second nonce reported; no batch
                    skipped after a find
                    blake256, blake2s, decred, cryptonight: a scan now runs
                    up to the requested end of the range
                    equihash   --time-limit and the share limit are honoured
                               on a quiet pool
                    yescrypt   variants no longer share the first client key
                    lyra2, lyra2v2: partial last warp handled correctly
                    neoscrypt  memory leak per scan call
                    multi-GPU: the CUDA sleep table was shared without a lock

                  Verification
                    fail-closed self-tests at startup for flex, fugue256,
                    fugue512-80, luffa-80 and pearlhash; gr and mike
                    re-hash sampled lanes on the CPU (GR_VERIFY / MIKE_VERIFY)

                  Build
                    msbuild -p:DevArch=86, build.cmd 86, build.sh 86: one-arch
                    developer build; source tree reorganised (cuda/, core/)

  Oct. 17th 2026 ccminer-tpfuemp 2026.10-p2 (patch, binaries added to the 2026.10 release)

                  Fixes
                    bc3hash    too low rate on large GPUs at a very low pool
                               difficulty and in --benchmark

                  Logging
                    CUDA errors name the GPU even when several fail at once;
                    startup lists GPUs with a display watchdog

  Oct. 17th 2026 ccminer-tpfuemp 2026.10-p1 (patch, binaries added to the 2026.10 release)

                  Logging
                    CUDA errors that stop the miner now name the GPU and its PCI
                    bus id ("GPU #n (PCI 0000:01:00.0): Cuda error ..."), which
                    matches the NVIDIA driver's Xid lines in the kernel log

                  Speed
                    bc3hash    GTX 10-series cards with less than 3.9 GB free
                               (16-bit curve tables): +4% (measured on a
                               GTX 1080 Ti forced to 16-bit tables)

  Oct. 16th 2026 ccminer-tpfuemp 2026.10

                  New algorithms
                    bc3hash          BC3Hash (Bitcoin III, from block 64360):
                                     about 360 kH/s on an RTX 3060, 127 kH/s
                                     on a GTX 1080 Ti; curve tables use 3 GB
                                     of GPU memory when available (256 MB
                                     otherwise)

                  Speed
                    blake256   3-input adds on the IMAD pipe (sm_80+): blake,
                               blakecoin, vanilla +18-20% (RTX 3060); blake2s
                               +5%; GTX 10-series now defaults to -i 30 on
                               Windows too
                    balloon    register-resident kernel: about 4x on a GTX
                               1080 Ti; default batch sized to the GPU
                    bmw        target screen inside the kernel, per-job
                               precompute, default intensity 27
                    bastion, heavy: reworked HEFTY1 core

                  Fixes
                    allium, lyra2, lyra2v2, lyra2z, lyra2z330: the first launch
                    after a new job could scan past the requested range
                    argon2d    race in the finalize step; both reported
                               candidates are now checked separately
                    blake, blakecoin, vanilla, blake2b: a valid second
                    candidate in a launch is now submitted too
                    bmw, bmw512: the second candidate is verified before it
                    is submitted
                    heavy, bastion, balloon: scan resume after a find
                    bastion    two waits shared one sleep slot and overslept

                  Verification
                    fail-closed GPU self-tests at startup for argon2d (all five
                    variants), blake256, vanilla, bmw, bmw512 and bastion;
                    -D runs a GPU-vs-CPU check for them

  Oct. 2nd 2026  ccminer-tpfuemp 2026.09.2

                  New algorithms
                    verus            VerusHash 2.2 (Verus, PBaaS chains)
                    homescrypt       HomeScrypt v1.2 (Lumenite)
                    minotaurx        MinotaurX (Avian, Pulsar)
                    yespowersmt      Smartiecoin (SMT), yespower N=256 r=8

                  Speed
                    whirlpool  four hashes per thread, bitsliced: 1.2-2.9x
                    groestl    several hashes per thread, bitsliced (x-family
                               stages, groestlcoin, myr-gr, allium, lyra2)
                    x-family   fused kernels compiled per fixed stage order;
                               faster jh, shavite, groestl, fugue, hamsi, echo,
                               simd and groestl256 stages
                    lyra2      matrix columns held in registers: allium +39%,
                               lyra2 +37%, lyra2v2 +6% (RTX 3060), lyra2v2
                               +13% (GTX 1080 Ti)
                    cryptonight, flex, gr, mike: faster CN core, and a batch
                               now stops at a job change

                  Fixes
                    streebog   1 in 256 inputs hashed wrong (gost, polytimos,
                               skunk, veltor, sib, phi, x21s, x25x)
                    x16rv2     some stage orders hashed wrong (the stage list
                               was too short)
                    lyra2v2    batches whose thread count was not a multiple
                               of 64 hashed wrong
                    shared stage finals report the two lowest candidates
                    atomically; a third concurrent candidate could be lost
                    missing barriers after shared table fills (echo512,
                    streebog); bmw512's last launch wrapped past the range
                    REST API: an algo switch now resets speed and share
                    counters, as a pool switch already did

                  Build
                    Linux binaries link the CUDA runtime statically, so they
                    need only the NVIDIA driver

  Sep. 22nd 2026  ccminer-tpfuemp 2026.09.1

                  New algorithms
                    flex             Flex (Kylacoin, Lyncoin)
                    mike             Mike (VKAX, FortuneBlock)
                    yespowereqpay    EqPay, yespower 1.0 over a 181-byte header
                    yespoweradvc     AdventureCoin (ADVC), a yespower preset
                    equihash 144/5 and 192/7 -- a new DIGITBITS=24 solver; 192/7
                    could not be mined at all before (research use, see notes)

                  Speed
                    hoohash    ~8% faster, bit-exact against the reference
                    verthash   32-byte aligned datafile reads, per-card intensity
                    keccak, sha3t, sha3d, sha512256d: per-arch launch bounds
                    progpow    one kernel generator and DAG state machine now
                               shared by kawpow, meowpow, evrprogpow, firopow
                               and meraki

                  Fixes
                    hoohash    stratum difficulty factor (2^16, as the coin's own
                               reference miner uses). Without it the share target
                               was harder than the network target at the pool's
                               default difficulty and almost nothing was submitted
                    candidate reporting and share-difficulty attribution in the
                    shared screen used by most algorithms
                    uninitialised buffer readbacks in the hoohash self-test and
                    in the shared candidate collector

  Sep. 14th 2026  ccminer-tpfuemp 2026.09
                  New algorithms: the yespower 1.0 family -- yespower, yespowerr16
                  (Yenten), power2b / yespower-b2b -- with 11 coin presets selected by
                  -a name or a pool "algo" field (Sugarchain, UraniumX, CPUchain,
                  Tidecoin, LightningCash-Gold, MagpieCoin, Arowana, IsotopeC, IOTS,
                  LightBit); argon2id1024 (Bitweb)

                  Revived: -a scrypt and -a scrypt-jane, which had no dispatch entry and
                  could not run at all; -a lyra2v2, which faulted on every launch after a
                  shared stage changed its buffer layout for another algo; -a lyra2z330

                  Consensus fix: -a decred now mines BLAKE3 (DCP-0011), not Blake-256.
                  It had been computing Blake-256 since the fork inherited it and could
                  not produce an acceptable share.

                  Share-loss fixes: 64-bit target compare in blake / blakecoin / vanilla
                  (the old screen dropped valid shares below difficulty 1); atomic
                  candidate reporting in groestl256 -- whose second-nonce channel was
                  entirely dead -- balloon, blake2b, sia and heavyhash; second-nonce
                  re-verify guards in ghostrider, lyra2RE and allium; duplicate-submit
                  nonce cursors; UINT32_MAX sentinel across 28 files, since 0 is a legal
                  nonce

                  Speedups on Ampere, verified against a live pool:
                    rinhash     6.9x    shared argon2d cooperative fill
                    heavyhash   4.9x    __dp4a nibble matrix, 16 -> 4 KB of shared
                    curvehash   2.3x    safegcd inversion, then a 16-bit ecmult_gen window
                    yespower    2.3x    S-box placement and a 32-thread block, over the
                                        initial port earlier in this same cycle
                    odocrypt    1.9x    funnel-shift rotate + per-epoch NVRTC JIT
                    vanilla     1.46x   default intensity
                    blake2s     1.14x   default intensity
                    allium and lyra2 1.14x, lyra2z 1.03x, also default intensity
                  scrypt, scrypt-jane, lyra2v2 and lyra2z330 carry no factor: they could
                  not run at all before (see Revived)

                  Pool switching: several fixes, all of which only bite when more than one
                  pool is configured --
                    A switch did not close the previous pool's stratum session, so the miner
                    kept mining the OLD pool's jobs under the NEW pool's algorithm and every
                    share after the first switch was rejected. Rotating between two pools now
                    mines and submits on the pool it actually selected.
                    --time-limit and --shares-limit never stopped the miner: every pool
                    inherited the value, which made them rotate for ever and left the
                    documented "timeout reached, exiting" unreachable. A limit given on the
                    command line now ends the run, and its deadline is measured across the
                    whole session, so repeated pool rotation cannot keep deferring it. A
                    pool that sets its own time-limit or shares-limit in the config still
                    rotates, as before.
                    A requested shutdown was treated as a dead pool and triggered a failover
                    into the teardown, which could crash on exit.
                    "shares-limit" is now accepted as a per-pool config key; it had a setter
                    but was missing from the key table, so it was silently ignored.
                  CUDA 12: the texture-reference API that CUDA 12 removed is gone from
                  the four files that still used it. A validated ccminer-kudaraidee-cuda12.vcxproj
                  (CUDA 12.9) builds alongside the shipping CUDA 11.8 project; 11.8 still
                  ships, since the two measured within noise of each other

                  REST API: an HTTP/JSON API alongside the legacy binary protocol, with
                  runtime control (pool and algo switching, pause/resume), Prometheus
                  metrics, a /health endpoint and --api-remote privilege separation.
                  See docs/api-rest.md and docs/openapi.yaml

                  Reliability: device self-tests now fail closed instead of logging a
                  warning and mining on; an algo switch frees GPU buffers, drops stale
                  shares and resets per-pool stats; NVRTC is linked statically

  Jul. 22nd 2026  ccminer-tpfuemp 2026.07.2
                  New algorithms: kawpow (Ravencoin), meowpow (Meowcoin),
                  evrprogpow (Evrmore), firopow (Firo / StakeCube), meraki
                  (Telestai) -- the ProgPoW-over-ethash family; verthash
                  (Vertcoin); curvehash (CurvehashCoin, secp256k1 EC on GPU);
                  whirlpoolx2 (CapStash); argon2d500 (Dynamic) and argon2d4096
                  (Argentum / Myriad)
                  ProgPoW family: shared progpow_multi core reusing the kawpow
                  ethash library, with per-period NVRTC JIT and on-GPU DAG
                  generation; one cnt_dag knob parameterises the DAG-loop count
                  verthash: on-GPU hashing, plus --generate-verthash-dat to
                  build verthash.dat
                  argon2d: fix a last-column/cache aliasing race (~2.8x on
                  lanes=1 coins); argon2d16000 (Alterdot) now works on GPU
                  Fix: Linux release is now a portable baseline build
                  (-mtune=generic) to avoid "Illegal instruction" on older CPUs

  Jul. 17th 2026  ccminer-tpfuemp 2026.07.1
                  New algorithms: x25x (SUQA/SIN), soterg (Soteria / X12R),
                  sha512256d (Radiant, double SHA-512/256)
                  algos/ refactor: all algorithm families migrated into the algos/ tree;
                  shared CUDA device-primitive libraries (x11 / x16 / keccak / SHA-512)
                  with de-branded bare stage launchers and consolidated headers
                  Register-resident fused multi-stage kernels for the x16 / x-family chains
                  Linux (autotools) build refreshed
                  Optimise: keccak sha3d +20%; x25x shared-memory shuffle + slot-major
                  accumulator (~3.7x over the first working build)
                  Fix: BMW-256 (-a bmw) out-of-bounds crash

  Jul. 12th 2026  ccminer-tpfuemp 2026.07
                  Require the CUDA 11.8 Toolkit (Visual Studio 2022); drop 10.x/11.0-11.7
                  Build floor is now sm_61 (Pascal): default gencode sm_61/75/86 + compute_86 PTX
                  Drop Maxwell (sm_50/52), Kepler and Fermi, and all arch-specific code below sm_61
                  Remove the stale CUDA 10 project files (ccminer-cuda10.*)
                  Pre-Pascal GPU or older CUDA needed: use upstream https://github.com/tpruvot/ccminer

  Jan. 04th 2017  v2.2.4
                  Improve lyra2v2
                  Higher keccak default intensity
                  Drop SM 2.x support by default, for CUDA 9 and more recent

  Dec. 04th 2017  v2.2.3
                  Polytimos Algo
                  Handle keccakc variant (with refreshed sha256d merkle)
                  Optimised keccak for SM5+, based on alexis improvements

  Oct. 09th 2017  v2.2.2
                  Import and clean the hsr algo (x13 + custom hash)
                  Import and optimise phi algo from LuxCoin repository
                  Improve sib algo too for maxwell and pascal cards
                  Small fix to handle more than 9 cards on linux (-d 10+)
                  Attempt to free equihash memory "properly"
                  --submit-stale parameter for supernova pool (which change diff too fast)

  Sep. 01st 2017  v2.2.1
                  Improve tribus algo on recent cards (up to +10%)

  Aug. 13th 2017  v2.2
                  New skunk algo, using the heavy streebog algorithm
                  Enhance tribus algo (+10%)
                  equihash protocol enhancement on yiimp.ccminer.org and zpool.ca

  June 16th 2017  v2.1-tribus
                  Interface equihash algo with djeZo solver (from nheqminer 0.5c)
                  New api parameters (and multicast announces for local networks)
                  New tribus algo

  May. 14th 2017  v2.0
                  Handle cryptonight, wildkeccak and cryptonight-lite
                  Add a serie of new algos: timetravel, bastion, hmq1725, sha256t
                  Import lyra2z from djm34 work...
                  Rework the common skein512 (used in most algos except skein ;)
                  Upgrade whirlpool algo with alexis version (2x faster)
                  Store the share diff of second nonce(s) in most algos
                  Hardware monitoring thread to get more accurate power readings
                  Small changes for the quiet mode & max-log-rate to reduce logs
                  Add bitcore and a compatible jha algo

  Dec. 21th 2016  v1.8.4
                  Improve streebog based algos, veltor and sib (from alexis work)
                  Blake2s greetly improved (3x), thanks to alexis too...

  Sep. 28th 2016  v1.8.3
                  show intensity on startup for each cards
                  show-diff is now used by default, use --hide-diff if not wanted

  Sep. 22th 2016  v1.8.2
                  lbry improvements by Alexis Provos
                  Prevent Windows hibernate while mining
                  veltor algo (basic implementation)

  Aug. 10th 2016  v1.8.1
                  SIA Blake2-B Algo (getwork over stratum for Suprnova)
                  SIA Nanopool RPC (getwork over http)
                  Update also the older lyra2 with Nanashi version

  July 20th 2016  v1.8.0
                  Pascal support with cuda 8
                  lbry new multi sha / ripemd algo (LBC)
                  x11evo algo (XRE)
                  Lyra2v2, Neoscrypt and Decred improvements
                  Enhance windows NVAPI clock and power limits
                  Led support for mining/shares activity on windows

  May  18th 2016  v1.7.6
                  Decred vote support
                  X17 cleanup and improvement
                  Add mining.ping stratum method and handle unknown methods
                  Implement a pool stats/benchmark mode (-p stats on yiimp)
                  Add --shares-limit parameter, can be used for benchmarks

  Mar. 13th 2016  v1.7.5
                  Blake2S Algo (NEVA/OXEN)

  Feb. 28th 2016  v1.7.4 (1.7.3 was a preview, not official)
                  Decred simplified stratum (getwork over stratum)
                  Vanilla kernel by MrMad
                  Drop/Disable WhirlpoolX

  Feb. 11th 2016  v1.7.2
                  Decred Algo (longpoll only)
                  Blake256 improvements/cleanup

  Jan. 26th 2016  v1.7.1
                  Implement sib algo (X11 + Russian Streebog-512/GOST)
                  Whirlpool speed x2 with the midstate precompute
                  Small bug fixes about device ids mapping (and vendor names)
                  Add Vanilla algo (Blake256 8-rounds - double sha256)

  Nov. 06th 2015  v1.7
                  Improve old devices compatibility (x11, lyra2v2, quark, qubit...)
                  Add windows support for SM 2.1 and drop SM 3.5 (x86)
                  Improve lyra2 (v1/v2) cuda implementations
                  Improve most common algos on SM5+ with sp blake kernel
                  Restore whirlpool algo (and whirlcoin variant)
                  Prepare algo/pool switch ability, trivial method
                  Add --benchmark alone to run a benchmark for all algos
                  Add --cuda-schedule parameter
                  Add --show-diff parameter, which display shares diff,
                    and is able to detect real solved blocks on pools.

  Aug. 28th 2015  v1.6.6
                  Allow to load remote config with curl (-c http://...)
                  Add Lyra2REv2 algo (Vertcoin/Zoom)
                  Restore WhirlpoolX algo (VNL)
                  Drop Animecoin support
                  Add bmw (Midnight) algo

  July 06th 2015  v1.6.5-C11
                  Nvml api power limits
                  Add chaincoin c11 algo (used by Flaxscript too)
                  Remove pluck algo

  June 23th 2015  v1.6.5
                  Handle Ziftrcoin PoK solo mining
                  Basic compatibility with CUDA 7.0 (generally slower hashrate)
                  Show gpus vendor names on linux (windows test branch is pciutils)
                  Remove -v and -m short params specific to heavycoin
                  Add --diff-multiplier (-m) and rename --diff to --diff-factor (-f)
                  First steps to handle nvml application clocks and P0 on the GTX9xx
                  Various improvements on multipool and cmdline parameters
                  Optimize a bit qubit, deep, luffa, x11 and quark algos

  May 26th 2015   v1.6.4
                  Implement multi-pool support (failover and time rotate)
                    try "ccminer -c pools.conf" to test the sample config
                  Update the API to allow remote pool switching and pool stats
                  Auto bind the api port to the first available when using default
                  Try to compute network difficulty on pools too (for most algos)
                  Drop Whirlpool and whirpoolx algos, no more used...

  May 15th 2015   v1.6.3
                  Import and adapt Neoscrypt from djm34 work (SM 5+ only)
                  Conditional mining options based on gpu temp, network diff and rate
                  background option implementation for windows too
                  "Multithreaded" devices (-d 0,0) intensity and stats changes
                  SM5+ Optimisation of skein based on sp/klaus method (+20%)

  Apr. 21th 2015  v1.6.2
                  Import Scrypt, Scrypt:N and Scrypt-jane from Cudaminer
                  Add the --time-limit command line parameter

  Apr. 14th 2015  v1.6.1
                  Add the Double Skein Algo for Woodcoin
                  Skein/Skein2 SM 3.0 devices support

  Mar. 27th 2015  v1.6.0
                  Add the ZR5 Algo for Ziftcoin
                  Implement Skeincoin algo (skein + sha)
                  Import pluck (djm34) and whirlpoolx (alexis78) algos
                  Hashrate units based on hashing rate values (Hs/kHs/MHs/GHs)
                  Default config file (also help to debug without command line)
                  Various small fixes

  Feb. 11th 2015  v1.5.3
                  Fix anime algo
                  Allow a default config file in user or ccminer folder
                  SM 2.1 windows binary (lyra2 and blake/blakecoin for the moment)

  Jan. 24th 2015  v1.5.2
                  Allow per device intensity, example: -i 20,19.5
                  Add process CPU priority and affinity mask parameters
                  Intelligent duplicate shares check feature (enabled if needed)
                  api: Fan RPM (windows), Cuda threads count, linux kernel ver.
                  More X11 optimisations from sp and KlausT
                  SM 3.0 enhancements

  Dec. 16th 2014  v1.5.1
                  Add lyra2RE algo for Vertcoin based on djm34/vtc code
                  Multiple shares support (2 for the moment)
                  X11 optimisations (From klaust and sp-hash)
                  HTML5 WebSocket api compatibility (see api/websocket.htm)
                  Solo mode height checks with getblocktemplate rpc calls

  Nov. 27th 2014  v1.5.0
                  Upgrade compat jansson to 2.6 (for windows)
                  Add pool mining.set_extranonce support
                  Allow intermediate intensity with decimals
                  Update prebuilt x86 openssl lib to 1.0.1i
                  Fix heavy algo on linux (broken since 1.4)
                  Some internal changes to use the C++ compiler
                  New API 1.2 with some new commands (read only)
                  Add some of sp x11/x15 optimisations (and tsiv x13)

  Nov. 15th 2014  v1.4.9
                  Support of nvml and nvapi(windows) to monitor gpus
                  Fix (again) displayed hashrate for multi gpus systems
                    Average is now made by card (30 scans of the card)
                  Final API v1.1 (new fields + histo command)
                  Add support of telnet queries "telnet 127.0.0.1 4068"
                  add histo api command to get performance debug details
                  Add a rig sample php ui using json wrapper (php)
                  Restore quark/jackpot previous speed (differently)

  Nov. 12th 2014  v1.4.8
                  Add a basic API and a sample php json wrapper
                  Add statsavg (def 20) and api-bind parameters

  Nov. 11th 2014  v1.4.7
                  Average hashrate (based on the 20 last scans)
                  Rewrite blake algo
                  Add the -i (gpu threads/intensity parameter)
                  Add some X11 optimisations based on sp_ commits
                  Fix quark reported hashrate and benchmark mode for some algos
                  Enhance json config file param (int/float/false) (-c config.json)
                  Update windows prebuilt curl to 7.38.0

  Oct. 26th 2014  v1.4.6
                  Add S3 algo reusing existing code (onecoin)
                  Small X11 (simd512) enhancement

  Oct. 20th 2014  v1.4.5
                  Add keccak algo from djm34 repo (maxcoin)
                  Curl 7.35 and OpenSSL are now included in the binary (and win tree)
                  Enhance windows terminal support (--help was broken)

  Sep. 27th 2014  v1.4.4
                  First SM 5.2 Release (GTX 970 & 980)
                  CUDA Runtime included in binary
                  Colors enabled by default

  Sep. 10th 2014  v1.4.3
                  Add algos from djm34 repo (deep, doom, qubit)
                  Goalcoin seems to be dead, not imported.
                  Create also the pentablake algo (5x Blake 512)

  Sept  6th 2014  Almost twice the speed on blake256 algos with the "midstate" cache

  Sep.  1st 2014  add X17, optimized x15 and whirl
                  add blake (256 variant)
                  color support on Windows,
                  remove some dll dependencies (pthreads, msvcp)

  Aug. 18th 2014  add X14, X15, Whirl, and Fresh algos,
                  also add colors and nvprof cmd line support

  June 15th 2014  add X13 and Diamond Groestl support.
                  Thanks to tsiv and to Bombadil for the contributions!

  June 14th 2014  released Killer Groestl quad version which I deem
                  sufficiently hard to port over to AMD. It isn't
                  the fastest option for Compute 3.5 and 5.0 cards,
                  but it is still much faster than the table based
                  versions.

  May 10th 2014   added X11, but without the bells & whistles
                  (no killer Groestl, SIMD hash quite slow still)

  May 6th 2014    this adds the quark and animecoin algorithms.

  May 3rd 2014    add the MjollnirCoin hash algorithm for the upcomin
                  MjollnirCoin relaunch.

                  Add the -f (--diff) option to adjust the difficulty
                  e.g. for the erebor Dwarfpool myr-gr SaffronCoin pool.
                  Use -f 256 there.

  May 1st 2014    adapt the Jackpot algorithms to changes made by the
                  coin developers. We keep our unique nVidia advantage
                  because we have a way to break up the divergence.
                  NOTE: Jackpot Hash now requires Compute 3.0 or later.

  April, 27 2014  this release adds Myriad-Groestl and Jackpot Coin.
                  we apply an optimization to Jackpot that turns this
                  into a Keccak-only CUDA coin ;) Jackpot is tested with
                  solo--mining only at the moment.

  March, 27 2014  Heavycoin exchange rates soar, and as a result this coin
                  gets some love: We greatly optimized the Hefty1 kernel
                  for speed. Expect some hefty gains, especially on 750Ti's!

                  By popular demand, we added the -d option as known from
                  cudaminer.

                  different compute capability builds are now provided until
                  we figure out how to pack everything into a single executable
                  in a Windows build.

  March, 24 2014  fixed Groestl pool support

                  went back to Compute 1.x for cuda_hefty1.cu kernel by
                  default after numerous reports of ccminer v0.2/v0.3
                  not working with HeavyCoin for some people.

  March, 23 2014  added Groestlcoin support. stratum status unknown
                  (the only pool is currently down for fixing issues)

  March, 21 2014  use of shared memory in Fugue256 kernel boosts hash rates
                  on Fermi and Maxwell devices. Kepler may suffer slightly
                  (3-5%)

                  Fixed Stratum for Fuguecoin. Tested on dwarfpool.

  March, 18 2014  initial release.


>>> AUTHORS <<<

Notable contributors to this application are:

Christian Buchner, Christian H. (Germany): Initial CUDA implementation

djm34, tsiv, sp and klausT for cuda algos implementation and optimisation

Tanguy Pruvot : 750Ti tuning, blake, colors, zr5, skein, general code cleanup
                API monitoring, linux Config/Makefile and vstudio libs...

and also many thanks to anyone else who contributed to the original
cpuminer application (Jeff Garzik, pooler), it's original HVC-fork
and the HVC-fork available at hvc.1gh.com

Source code is included to satisfy GNU GPL V3 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
   Christian H. ( Chris84 )
   Tanguy Pruvot ( tpruvot@github )
