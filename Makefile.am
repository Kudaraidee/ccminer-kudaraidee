# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.md README.txt LICENSE.txt \
			  ccminer-kudaraidee.sln ccminer-kudaraidee.vcxproj ccminer-kudaraidee.vcxproj.filters \
			  ccminer-kudaraidee-cuda12.vcxproj \
			  compat/gettimeofday.c compat/getopt/getopt_long.c

SUBDIRS = compat

bin_PROGRAMS = ccminer

ccminer_SOURCES	= compat/elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  algos/heavy/hefty1.c \
			  ccminer.cpp core/pools.cpp core/util.cpp core/bench.cpp core/bignum.cpp \
			  core/api/api.cpp core/api/api_control.cpp core/api/api_control.h core/api/api_model.cpp core/api/api_model.h core/api/api_http.c core/api/api_http.h core/api/api_metrics.c core/api/api_metrics.h core/api/api_routes.cpp core/api/api_routes.h core/hashlog.cpp core/gpu/nvml.cpp core/stats.cpp core/sysinfos.cpp core/cuda.cpp \
			  core/gpu/nvsettings.cpp \
			  algos/equihash/equi-stratum.cpp algos/equihash/equi.cpp algos/equihash/equi_verify_144.cpp algos/equihash/equi_verify_192.cpp algos/equihash/blake2/blake2bx.cpp \
			  algos/equihash/equihash.cpp algos/equihash/cuda_equi.cu \
			  algos/equihash/cuda_equi24b.cu algos/equihash/cuda_equi24b_192.cu \
			  algos/allium/allium.cu \
			  algos/balloon/balloon.cpp algos/balloon/balloon_scan.cpp \
			  algos/balloon/balloon_selftest.cpp \
			  algos/balloon/sha256-ref.c algos/balloon/cuda_balloon.cu \
			  algos/balloon/balloon.h algos/balloon/sha256.h \
			  algos/rinhash/rinhash_scanhash.cpp algos/rinhash/rinhash_coop.cu \
			  algos/curvehash/curvehash_oracle.cpp algos/curvehash/curvehash.cu algos/curvehash/secp256k1_unity.c \
			  algos/bc3hash/bc3hash_oracle.cpp algos/bc3hash/bc3hash.cu \
			  algos/pearlhash/pearlhash.cu \
			  algos/nebhash/nebhash.cpp algos/nebhash/nebhash.h \
			  algos/nebhash/nebhash2.c algos/nebhash/nebhash2.h \
			  algos/nebhash/shares.c algos/nebhash/shares.h \
			  algos/nebhash/chacha8.c algos/nebhash/chacha8.h \
			  algos/nebhash/nebhash2_cuda.cu algos/nebhash/nebhash2_cuda.h \
			  algos/nebhash/nebhash2_table_cuda.cu algos/nebhash/nebhash2_table_cuda.h \
			  algos/nebhash/LICENSE.NEB \
			  algos/verthash/verthash.cu algos/verthash/cuda_verthash.cu \
			  algos/verthash/verthash-cpu.c algos/verthash/verthash-data.cpp \
			  cuda/sha3_device.cuh \
			  cuda/intensity_autotune.cuh \
			  algos/kawpow/cuda_kawpow.cu \
			  algos/kawpow/kawpow.cpp algos/kawpow/kawpow_core.cpp \
			  algos/xhash/xhash.cpp algos/xhash/xhash_core.cpp algos/xhash/xhash_cuda.cu \
			  algos/xhash/vendor/xhash/lib/xhash/xhash.cpp algos/xhash/vendor/xhash/lib/xhash/primes.c \
			  algos/xhash/vendor/xhash/lib/sha3/sha3.c algos/xhash/vendor/xhash/lib/keccak/keccak.c \
			  algos/kawpow/ethash/ethash.cpp algos/kawpow/ethash/progpow.cpp algos/kawpow/ethash/managed.cpp \
			  algos/kawpow/ethash/primes.c algos/kawpow/ethash/keccak.c \
			  algos/kawpow/ethash/keccakf1600.c algos/kawpow/ethash/keccakf800.c \
			  algos/progpow_multi/progpow_pp.cpp algos/progpow_multi/ppmulti_jit.cpp \
			  algos/progpow_multi/ppmulti_dag.cpp algos/progpow_multi/ppmulti_epoch.cpp \
			  algos/progpow_multi/ppmulti_core.cpp \
			  algos/progpow_multi/ppmulti_bridge.cpp \
			  algos/hoohash/cuda_hoohashv110.cu algos/hoohash/hoohash.cu \
			  algos/hoohash/hoohash_device.cuh \
			  algos/evohash/evohash.cu \
			  algos/ghostrider/ghostrider.cu \
			  algos/mike/mike.cu algos/mike/mike_hash.cpp \
			  algos/flex/flex.cu algos/flex/flex_hash.cpp \
			  algos/flex/flex.h algos/flex/flex_kat.h algos/flex/flex_schedule.h \
			  algos/minotaurx/minotaurx.cu algos/minotaurx/minotaurx_hash.cpp \
			  algos/minotaurx/minotaurx.h algos/minotaurx/minotaurx_kat.h \
			  algos/homescrypt/homescrypt.cu algos/homescrypt/homescrypt_hash.cpp \
			  algos/homescrypt/homescrypt.h algos/homescrypt/homescrypt_kat.h \
			  algos/verus/verus.cu algos/verus/verus_host.c algos/verus/haraka.c \
			  algos/verus/haraka_portable.c algos/verus/verus_clhash.c \
			  algos/verus/verus_host.h algos/verus/verus_portable.h algos/verus/verus_warp.cuh algos/verus/verus-kat.h algos/verus/haraka.h \
			  algos/verus/haraka_portable.h algos/verus/verus_clhash.h algos/verus/verus_clhash_iter.h algos/verus/verus-simd.h algos/verus/verus_host_target.h \
			  algos/mike/mike.h algos/mike/mike_kat.h \
			  algos/argon2d/argon2d.cu algos/argon2d/argon2d_fill.cu \
			  algos/argon2d/blake2b_kernels.cu \
			  algos/argon2d/argon2ref/argon2.c algos/argon2d/argon2ref/core.c \
			  algos/argon2d/argon2ref/encoding.c algos/argon2d/argon2ref/opt.c \
			  algos/argon2d/argon2ref/thread.c algos/argon2d/argon2ref/blake2/blake2b.c \
			  algos/argon2d/argon2d_kernel.h algos/argon2d/cudaexception.h \
			  algos/argon2d/argon2d_blake2b_device.cuh \
			  algos/argon2d/argon2ref/argon2.h algos/argon2d/argon2ref/core.h \
			  algos/argon2d/argon2ref/encoding.h algos/argon2d/argon2ref/thread.h \
			  algos/argon2d/argon2ref/blake2/blake2.h algos/argon2d/argon2ref/blake2/blake2-impl.h \
			  algos/argon2d/argon2ref/blake2/blamka-round-opt.h algos/argon2d/argon2ref/blake2/blamka-round-ref.h \
			  algos/yescrypt/yescrypt.cu algos/yescrypt/cuda_yescrypt.cu \
			  sph/yescrypt-common.c sph/yescrypt-opt.c sph/sha256_Y.c \
			  sph/yescrypt.h sph/yescrypt-platform.h \
algos/yespower/yespower.cu cuda/salsa_device.cuh cuda/pwxform_device.cuh cuda/blake2b_hash_device.cuh algos/yespower/yespower_smix.cuh algos/yespower/yespower_head_tail.cuh algos/yespower/yespower_hash.cuh sph/yespower_ref.c sph/yespower_b2b_ref.c sph/hmac_blake2b.c sph/hmac_blake2b.h sph/yespower.h \
			  algos/odocrypt/odocrypt_host.cpp algos/odocrypt/odocrypt_jit.cpp algos/odocrypt/cuda_odocrypt.cu \
			  algos/heavyhash/heavyhash.cu \
			  algos/heavyhash/cuda_heavyhash.cu \
			  algos/heavyhash/keccak_tiny.c \
			  algos/heavyhash/heavyhash-gate.c \
			  algos/heavy/heavy.cu \
			  algos/heavy/cuda_heavy_blake512.cu \
			  algos/heavy/cuda_combine.cu \
			  algos/heavy/cuda_heavy_groestl512.cu \
			  algos/heavy/cuda_hefty1.cu \
			  algos/heavy/cuda_heavy_keccak512.cu \
			  algos/heavy/cuda_sha256.cu \
			  algos/heavy/bastion.cu algos/heavy/cuda_bastion.cu \
			  algos/fugue/fuguecoin.cpp algos/stages/cuda_fugue256.cu sph/fugue.c compat/bitcoin/uint256.h \
			  algos/groestlcoin/groestlcoin.cpp algos/groestlcoin/cuda_groestl.cu algos/groestlcoin/cuda_groestl.h \
			  algos/myriadgroestl/myriadgroestl.cpp algos/myriadgroestl/cuda_myriadgroestl.cu \
			  algos/lyra2/Lyra2.c algos/lyra2/Sponge.c \
			  algos/lyra2/cuda_lyra2_selftest.cu algos/lyra2/lyra2RE.cu algos/stages/cuda_lyra2.cu \
			  algos/lyra2/cuda_lyra2v2_selftest.cu algos/lyra2/lyra2REv2.cu algos/stages/cuda_lyra2v2.cu \
			  algos/lyra2/Lyra2Z.c algos/lyra2/cuda_lyra2Z_selftest.cu algos/lyra2/lyra2Z.cu algos/stages/cuda_lyra2Z.cu \
			  algos/lyra2/lyra2Z330.cu algos/stages/cuda_lyra2Z330.cu \
			  cuda/blake2b_device.cuh cuda/blake3_device.cuh cuda/lyra2_device.cuh \
			  algos/stages/cuda_bmw256.cu algos/stages/cuda_cubehash256.cu \
			  algos/stages/cuda_blake256.cu algos/stages/cuda_groestl256.cu \
			  algos/stages/cuda_skein256.cu \
			  algos/blake256/blake256.cu algos/blake256/vanilla.cu \
			  algos/decred/decred_blake3.cu \
			  algos/blake256/cuda_blake256_selftest.cu \
			  algos/blake2s/blake2s.cu sph/blake2s.c \
			  algos/blake2s/cuda_blake2s_selftest.cu \
			  algos/bmw/bmw.cu algos/bmw/bmw512.cu algos/stages/cuda_bmw.cu \
			  algos/gost/cuda_gosthash.cu algos/gost/gost.cu \
			  algos/cryptonight/xmr-rpc.cpp algos/cryptonight/wildkeccak-cpu.cpp algos/cryptonight/wildkeccak.cu \
			  algos/cryptonight/cryptolight.cu algos/cryptonight/cryptolight-core.cu algos/cryptonight/cryptolight-cpu.cpp \
			  algos/cryptonight/cryptonight.cu algos/cryptonight/cryptonight-core.cu algos/cryptonight/cryptonight-extra.cu \
			  algos/cryptonight/cryptonight-cpu.cpp algos/cryptonight/oaes_lib.cpp algos/cryptonight/aesb.cpp algos/cryptonight/cpu/c_keccak.c \
			  algos/jha/jha.cu algos/jha/jackpot.cu algos/stages/cuda_jha_keccak512.cu \
			  algos/jha/cuda_jha_compactionTest.cu algos/common/cuda_checkhash.cu \
			  algos/stages/cuda_jh512.cu algos/stages/cuda_blake512.cu algos/stages/cuda_groestl512.cu algos/stages/cuda_skein512.cu \
			  algos/stages/cuda_bmw512.cu algos/stages/cuda_keccak512.cu \
			  algos/quark/anime.cu algos/quark/nist5.cu \
			  algos/quark/quark.cu algos/stages/cuda_quark_compactionTest.cu \
			  algos/neoscrypt/neoscrypt.cpp algos/neoscrypt/neoscrypt-cpu.c algos/neoscrypt/cuda_neoscrypt.cu \
			  algos/pentablake/pentablake.cu algos/skein/skein.cu algos/skein/cuda_skein.cu algos/skein/skein2.cpp algos/zr5/zr5.cu \
			  algos/skunk/skunk.cu algos/skunk/cuda_skunk.cu algos/stages/cuda_streebog.cu \
			  algos/sha256d/sha256d.cu algos/sha256d/cuda_sha256d.cu algos/sha256d/sha256t.cu algos/sha256d/cuda_sha256t.cu \
			  algos/sha256d/cuda_sha256_selftest.cu cuda/sha256_device.cuh \
			  algos/sha256d/sha256csm.cu algos/sha256d/cuda_sha256csm.cu \
			  algos/sha512256d/sha512256d.cu algos/sha512256d/cuda_sha512256d.cu \
			  algos/sha512256d/cuda_sha512256d_selftest.cu cuda/sha512_device.cuh \
			  algos/sha256dv/sha256dv.cu \
			  algos/blake2b/blake2b.cu \
			  algos/blake2b/cuda_blake2b_selftest.cu \
			  algos/keccak/keccak256.cu algos/keccak/cuda_keccak256.cu \
			  algos/keccak/cuda_keccak_selftest.cu cuda/keccak_device.cuh \
			  cuda/xfamily_selftest.cu cuda/selftest_gate.cuh cuda/blake512_device.cuh \
			  cuda/jh512_device.cuh cuda/bmw512_device.cuh cuda/skein512_device.cuh \
			  cuda/luffa512_device.cuh cuda/shabal512_device.cuh cuda/cubehash512_device.cuh \
			  cuda/hamsi512_device.cuh cuda/fugue512_device.cuh cuda/groestl512_x2_device.cuh cuda/echo512_device.cuh cuda/shavite512_device.cuh cuda/shavite512_sp_device.cuh cuda/whirlpool512_x4_device.cuh cuda/tiger192_device.cuh \
			  cuda/aes_device.cuh cuda/aes_sp_device.cuh cuda/candidate_report.cuh \
			  algos/sha3d/sha3d.cu algos/sha3d/cuda_sha3d.cu \
			  algos/sha3t/sha3t.cu algos/sha3t/cuda_sha3t.cu \
			  algos/sia/sia.cu algos/sia/sia-rpc.cpp sph/blake2b.c \
			  sph/bmw.c sph/blake.c sph/groestl.c sph/jh.c sph/keccak.c sph/skein.c \
			  sph/cubehash.c sph/echo.c sph/luffa.c sph/sha2.c sph/shavite.c sph/simd.c \
			  sph/hamsi.c sph/hamsi_helper.c sph/streebog.c sph/tiger.c \
			  sph/shabal.c sph/whirlpool.c sph/sha2big.c sph/haval.c \
			  sph/ripemd.c sph/sph_sha2.c sph/sha3d.c sph/sha3.c \
			  algos/polytimos/polytimos.cu \
			  algos/lbry/lbry.cu algos/lbry/cuda_lbry_merged.cu \
			  algos/qubit/qubit.cu algos/qubit/deep.cu algos/qubit/luffa.cu \
			  algos/tribus/tribus.cu algos/stages/cuda_echo512_final.cu \
			  algos/x11/x11.cu algos/x11/fresh.cu algos/stages/cuda_luffa512.cu algos/stages/cuda_luffa512_80.cu algos/stages/cuda_cubehash512.cu algos/stages/cuda_cubehash512_80.cu \
			  algos/stages/cuda_shavite512_legacy.cu algos/stages/cuda_shavite512.cu algos/stages/cuda_simd512.cu algos/stages/cuda_echo512_compat.cu \
			  algos/x11/x11evo.cu algos/x11/timetravel.cu algos/x11/bitcore.cu \
			  algos/x13/x13.cu algos/stages/cuda_hamsi512.cu algos/stages/cuda_fugue512.cu \
			  algos/hsr/hsr.cu algos/hsr/cuda_hsr_sm3.cu algos/hsr/sm3.c \
			  algos/x15/x14.cu algos/x15/x15.cu algos/stages/cuda_shabal512.cu algos/stages/cuda_whirlpool512.cu \
			  algos/stages/cuda_whirlpool512_80.cu \
			  algos/x15/whirlpool.cu algos/x15/whirlpoolx.cu algos/x15/cuda_whirlpoolx.cu \
			  algos/whirlpoolx2/whirlpoolx2.cu algos/whirlpoolx2/cuda_whirlpoolx2.cu \
			  algos/x16/x16r.cu algos/x16/x16rt.cu algos/x16/x16rv2.cu algos/x16/x16s.cu algos/stages/cuda_echo512_80.cu algos/stages/cuda_fugue512_80.cu \
			  algos/stages/cuda_shabal512_80.cu algos/stages/cuda_shavite512_80.cu algos/stages/cuda_simd512_80.cu \
			  algos/stages/cuda_echo512_64.cu algos/common/cuda_x_fused.cu \
			  algos/x17/x17.cu algos/x17/skydoge.cu algos/x17/hmq1725.cu algos/soterg/soterg.cu algos/stages/cuda_haval256.cu algos/stages/cuda_sha512.cu \
			  algos/x11/phi.cu \
			  algos/x11/c11.cu algos/x11/s3.cu algos/x11/sib.cu algos/x11/veltor.cu algos/x11/0x10.cu \
			  algos/x21/x21s.cu algos/stages/cuda_sha256_2.cu algos/stages/cuda_tiger192.cu \
			  algos/x25x/x25x.cu algos/x25x/cuda_x25x_panama.cu algos/x25x/cuda_x25x_lane.cu \
			  algos/x25x/cuda_x25x_blake2s.cu algos/x25x/cuda_x25x_shuffle.cu algos/x25x/cuda_x25x_swifftx.cu \
			  algos/x25x/swifftx.c sph/sph_panama.c algos/x25x/lane.c

# scrypt
# Only the two texture-free kernels are wired up; test/fermi/kepler/nv target
# sm_2x/sm_30, below this build's floor, and nothing references them.
ccminer_SOURCES += algos/scrypt/scrypt.cpp algos/scrypt/scrypt-jane.cpp \
    algos/scrypt/blake.cu algos/scrypt/keccak.cu algos/scrypt/sha256.cu \
    algos/scrypt/salsa_kernel.cu \
    algos/scrypt/nv_kernel2.cu algos/scrypt/titan_kernel.cu

if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

if HAVE_WINDOWS
ccminer_SOURCES += compat/winansi.c
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
# WITH_HEAVY_ALGO must be on the C++ side as well as nvcc_FLAGS: the -a heavy /
# -a mjollnir dispatch lives behind #ifdef in ccminer.cpp, which g++ compiles.
# With it only on the CUDA side, heavy.cu gets the real body and the dispatch is
# still compiled out, so the algo reports "not available in this build".
ccminer_CPPFLAGS = -DWITH_HEAVY_ALGO @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs) -I$(top_srcdir)/algos/curvehash/secp256k1 \
	-I$(top_srcdir)/algos/curvehash/secp256k1/include -I$(top_srcdir)/algos/bc3hash/ref -I$(top_srcdir)/algos/bc3hash/ref/include \
	-I$(top_srcdir)/algos/xhash/vendor/xhash/include -I$(top_srcdir)/algos/xhash/vendor/xhash/lib/xhash

if HAVE_OSX
ccminer_CPPFLAGS += -I/usr/local/llvm/lib/clang/4.0.0/include
ccminer_LDFLAGS += -L/usr/local/llvm/lib
ccminer_LDADD += -lomp
endif

#ccminer_CPPFLAGS += -DUSE_LIBSODIUM
#ccminer_LDFLAGS += -Lalgos/equihash/lib
#ccminer_LDADD += -lsodium
ccminer_LDADD += -lcuda
# NVRTC: per-period ProgPoW kernel JIT (ppmulti_jit). Linked
# STATICALLY so the binary self-contains the JIT and runs on driver-only rigs
# (no libnvrtc.so.11.2 needed at load time); mirror the vcxproj which uses the
# nvrtc*_static.lib set. Order matters: nvrtc -> builtins -> ptxcompiler.
ccminer_LDADD += -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static -ldl

nvcc_ARCH :=
nvcc_ARCH += -gencode=arch=compute_86,code=\"sm_86,compute_86\"
nvcc_ARCH += -gencode=arch=compute_75,code=\"sm_75,compute_75\"
nvcc_ARCH += -gencode=arch=compute_61,code=\"sm_61,compute_61\"
#nvcc_ARCH += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
#nvcc_ARCH += -gencode=arch=compute_50,code=\"sm_50,compute_50\"
#nvcc_ARCH += -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#nvcc_ARCH += -gencode=arch=compute_30,code=\"sm_30,compute_30\"

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. -I$(srcdir) @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES) --ptxas-options="-v"
# -a heavy / -a mjollnir: algos/heavy/ is complete and already in the sources below,
# but heavy.cu compiles to a stub without this symbol.
nvcc_FLAGS += -DWITH_HEAVY_ALGO

# Keep the build-machine path out of __FILE__ (and out of the shipped
# binary): it otherwise lands in .rodata and survives strip.
TRIMPATH = -ffile-prefix-map=$(abs_top_srcdir)=. -ffile-prefix-map=$(abs_top_builddir)=.
nvcc_FLAGS += -Xcompiler "$(TRIMPATH)"
ccminer_CPPFLAGS += $(TRIMPATH)

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

# equi24b: the few-large-buckets heap for the same DIGITBITS=24 family. Same
# variant selection as the TUs above and the same ptxas cache hints.
algos/equihash/cuda_equi24b.o: algos/equihash/cuda_equi24b.cu algos/equihash/equi24b_params.h algos/equihash/cuda_equi24b.h
	$(NVCC) $(nvcc_FLAGS) -DEQ_WN=144 -DEQ_WK=5 -Xptxas -dscm=cs -o $@ -c $<

# the 192/7 instantiation. cuda_equi24b_192.cu is a wrapper whose only content
# is `#include "cuda_equi24b.cu"`, so the real source AND its headers must be
# prerequisites here: an explicit rule gets no automatic dependency scanning,
# and without them an edit to cuda_equi24b.cu rebuilds the 144/5 object and
# leaves this one STALE -- one binary, two variants running different code.
algos/equihash/cuda_equi24b_192.o: algos/equihash/cuda_equi24b_192.cu algos/equihash/cuda_equi24b.cu algos/equihash/equi24b_params.h algos/equihash/cuda_equi24b.h
	$(NVCC) $(nvcc_FLAGS) -DEQ_WN=192 -DEQ_WK=7 -Xptxas -dscm=cs -o $@ -c $<

# HoohashV110 device TU: consensus-critical strict FP (must match the glibc oracle).
# No fast-math; disable FMA contraction; precise div/sqrt; no flush-to-zero.
algos/hoohash/cuda_hoohashv110.o: algos/hoohash/cuda_hoohashv110.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false -o $@ -c $<

# homescrypt: its consensus binary32 FMA chain gets the same strict-FP flags.
algos/homescrypt/homescrypt.o: algos/homescrypt/homescrypt.cu algos/homescrypt/homescrypt.h algos/homescrypt/homescrypt_kat.h
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false -o $@ -c $<

# bc3hash: the mix kernels need ~255 registers.
algos/bc3hash/bc3hash.o: algos/bc3hash/bc3hash.cu algos/bc3hash/cuda/bc3_constants.cuh \
		algos/bc3hash/cuda/bc3_uint_device.cuh algos/bc3hash/cuda/bc3_exp_device.cuh \
		algos/bc3hash/cuda/bc3_ec_device.cuh algos/bc3hash/cuda/bc3_mix_device.cuh \
		algos/bc3hash/cuda/bc3_hashes_device.cuh algos/bc3hash/cuda/bc3_modinv_device.cuh \
		algos/bc3hash/cuda/bc3_secp_device.cuh
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=255 -o $@ -c $<

# pearlhash: list the headers, the .cu is mostly them (header-only oracle + kernels)
algos/pearlhash/pearlhash.o: algos/pearlhash/pearlhash.cu algos/pearlhash/pearl_device.cuh \
		algos/pearlhash/pearl_ref.h algos/pearlhash/pearl_proof.h cuda/blake3_device.cuh
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

algos/nebhash/nebhash2_cuda.o: algos/nebhash/nebhash2_cuda.cu algos/nebhash/nebhash2_cuda.h \
		algos/nebhash/nebhash2.h cuda/blake3_device.cuh
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

algos/nebhash/nebhash2_table_cuda.o: algos/nebhash/nebhash2_table_cuda.cu algos/nebhash/nebhash2_table_cuda.h \
		algos/nebhash/nebhash2.h cuda/blake3_device.cuh
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

algos/blake256/blake256.o: algos/blake256/blake256.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

algos/stages/cuda_bmw.o: algos/stages/cuda_bmw.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=76 -o $@ -c $<

algos/cryptonight/cryptonight-core.o: algos/cryptonight/cryptonight-core.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

algos/cryptonight/cryptonight-extra.o: algos/cryptonight/cryptonight-extra.cu
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

algos/heavy/cuda_hefty1.o: algos/heavy/cuda_hefty1.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

# Luffa and Echo are faster with 80 registers than 128
algos/stages/cuda_luffa512.o: algos/stages/cuda_luffa512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_simd512.o: algos/stages/cuda_simd512.cu
	$(NVCC) $(nvcc_FLAGS) -Xcompiler -Wno-unused-variable -o $@ -c $<

algos/stages/cuda_hamsi512.o: algos/stages/cuda_hamsi512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=72 -o $@ -c $<

algos/stages/cuda_sha512.o: algos/stages/cuda_sha512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_blake512.o: algos/stages/cuda_blake512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_jh512.o: algos/stages/cuda_jh512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_keccak512.o: algos/stages/cuda_keccak512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=88 -o $@ -c $<


# Per-file register caps, kept equal to ccminer-kudaraidee.vcxproj.
algos/neoscrypt/cuda_neoscrypt.o: algos/neoscrypt/cuda_neoscrypt.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=160 -o $@ -c $<

algos/pentablake/pentablake.o: algos/pentablake/pentablake.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/skein/cuda_skein.o: algos/skein/cuda_skein.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=48 -o $@ -c $<

algos/skunk/cuda_skunk.o: algos/skunk/cuda_skunk.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

algos/stages/cuda_shavite512_legacy.o: algos/stages/cuda_shavite512_legacy.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/skein/skein.o: algos/skein/skein.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

