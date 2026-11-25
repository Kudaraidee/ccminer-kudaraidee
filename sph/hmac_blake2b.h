/* HMAC-BLAKE2b + PBKDF2-BLAKE2b for yespower-b2b (`-a power2b`).
 * Vendored from cpuminer-opt algo/yespower/crypto/hmac-blake2b.h; the only
 * edits are the include paths and the sph_ prefix (this tree calls the same
 * API blake2b_*). See sph/yespower_b2b_ref.c for why it is the normative
 * reference rather than a re-derivation.
 * /!\ The HMAC pad here is 64 bytes, NOT BLAKE2b's 128-byte block. That is
 * non-standard but it is what the network accepts. */
#pragma once
#ifndef __HMAC_BLAKE2B_H__
#define __HMAC_BLAKE2B_H__

#include <stddef.h>
#include <stdint.h>
#include "blake2b.h"

#if defined(_MSC_VER) || defined(__x86_64__) || defined(__x86__)
#define NATIVE_LITTLE_ENDIAN
#endif

typedef struct
{
    blake2b_ctx inner;
    blake2b_ctx outer;
} hmac_blake2b_ctx;

#if defined(__cplusplus)
extern "C" {
#endif

void hmac_blake2b_hash( void *out, const void *key, size_t keylen,
                        const void *in, size_t inlen );

void pbkdf2_blake2b( const uint8_t * passwd, size_t passwdlen,
                     const uint8_t * salt, size_t saltlen, uint64_t c,
                     uint8_t * buf, size_t dkLen );

#if defined(__cplusplus)
}
#endif

#endif
