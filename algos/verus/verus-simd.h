/* Decides whether the vendored VerusHash host sources can be built and pulls the
 * intrinsic header. x86-64 only: MSVC exposes AES-NI/PCLMUL without flags, GCC
 * gets the ISA from verus_host_target.h. Otherwise the code compiles out and
 * the host API refuses at run time.
 *
 * EDIT (ccminer-kudaraidee): replaces cpuminer-opt's version (no aarch64 path). */
#ifndef VERUS_SIMD_H
#define VERUS_SIMD_H

#if (defined(_MSC_VER) && defined(_M_X64)) || \
    (defined(__AES__) && defined(__PCLMUL__) && defined(__SSSE3__))

#define VERUS_HAVE_SIMD 1
#include <immintrin.h>

#endif

#endif  /* VERUS_SIMD_H */
