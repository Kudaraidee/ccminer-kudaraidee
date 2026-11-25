#!/bin/bash

# Usage: ./build.sh       release build, CUDA sm_61/75/86 -> build-linux/ccminer
#        ./build.sh 86    dev build for one arch (61, 75 or 86) -> build-linux-sm86/ccminer
#
# Out-of-tree build into build-linux/. This keeps the (Windows-shared) source
# tree free of build artefacts, and links the binary to build-linux/ccminer so
# it does not collide with the Windows MSBuild IntDir directory ./ccminer/.

# export PATH="$PATH:/usr/local/cuda/bin/"

srcdir="$(cd "$(dirname "$0")" && pwd)"
cd "$srcdir"

arch="${1:-}"
builddir=build-linux
[ -n "$arch" ] && builddir="build-linux-sm$arch"

# Clear any prior IN-TREE configure state, else an out-of-tree configure refuses
# with "source directory already configured; run make distclean there first".
[ -f Makefile ] && make distclean >/dev/null 2>&1
rm -f config.status Makefile

./autogen.sh || echo done

# Fresh out-of-tree build dir.
rm -rf "$builddir"
mkdir -p "$builddir"
cd "$builddir"

"$srcdir/configure.sh"

if [ -n "$arch" ]; then
	make -j 8 nvcc_ARCH="-gencode=arch=compute_$arch,code=sm_$arch"
else
	make -j 8
fi
