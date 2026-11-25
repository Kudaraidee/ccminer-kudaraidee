#pragma once

/* Shim for pre-C99 compilers with no <stdbool.h>. Must stay inert for C++ and
 * C99+: macroizing bool/true/false makes the MSVC standard library reject the
 * translation unit (xkeycheck.h, error C1189). Reachable because compat/ can
 * precede the system include path. */
#if !defined(__cplusplus) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L)

#define false   0
#define true    1

#define bool int

#endif
