# To change the cuda arch, edit Makefile.am and run ./build.sh
#
# Resolves ./configure relative to this script's location, so it works whether
# run in the source tree or from a separate (out-of-tree) build dir.
# LDFLAGS adds the WSL2 driver lib dir (libcuda.so lives in /usr/lib/wsl/lib,
# not cuda/lib64); harmless on native Linux (just an extra, empty search path).

srcdir="$(cd "$(dirname "$0")" && pwd)"

# NOTE: use a portable baseline (-mtune=generic, default -march=x86-64 / SSE2),
# NOT -march=native. Host-side code here is only reference/verification (the GPU
# does the mining), so there is no perf reason to specialise, and -march=native
# bakes in the build machine's ISA (AVX/BMI2/SHA-NI on newer CPUs) which makes
# the distributed binary SIGILL ("Illegal instruction") on older target CPUs.
extracflags="-mtune=generic -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"

CUDA_CFLAGS="-O3 -lineno -Xcompiler -Wall  -D_FORCE_INLINES" \
	"$srcdir/configure" CXXFLAGS="-O3 $extracflags" LDFLAGS="-L/usr/lib/wsl/lib" \
	--with-cuda=/usr/local/cuda --with-nvml=libnvidia-ml.so
