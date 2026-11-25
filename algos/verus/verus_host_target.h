/* Enables AES-NI, PCLMUL and SSSE3 for the rest of the including TU on GCC, so
 * no per-file build flags are needed. Include it first, and only from VerusHash
 * host TUs; the miner checks CPUID (verus_host_cpu_ok) before calling them.
 * MSVC needs nothing. */
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#pragma GCC target("aes,pclmul,ssse3")
#endif
