/*
 * This uses veriations of the clhash algorithm for Verus Coin, licensed
 * with the Apache-2.0 open source license.
 * 
 * Copyright (c) 2018 Michael Toutonghi
 * Distributed under the Apache 2.0 software license, available in the original form for clhash
 * here: https://github.com/lemire/clhash/commit/934da700a2a54d8202929a826e2763831bd43cf7#diff-9879d6db96fd29134fc802214163b95a
 * 
 * CLHash is a very fast hashing function that uses the
 * carry-less multiplication and SSE instructions.
 *
 * Original CLHash code (C) 2017, 2018 Daniel Lemire and Owen Kaser
 * Faster 64-bit universal hashing
 * using carry-less multiplications, Journal of Cryptographic Engineering (to appear)
 *
 * Best used on recent x64 processors (Haswell or better).
 *
 **/

#ifndef INCLUDE_VERUS_CLHASH_H
#define INCLUDE_VERUS_CLHASH_H


#include "verus-simd.h"
#ifdef _WIN32
#include <intrin.h>
#endif


#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>   /* EDIT (ccminer-kudaraidee): bool in C */
#include <stddef.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "haraka.h"
#include "haraka_portable.h"


uint64_t verusclhashv2_2(void * random, const unsigned char buf[64], uint64_t keyMask, uint32_t *fixrand, uint32_t *fixrandex,
	u128 *g_prand, u128 *g_prandex);

__m128i  lazyLengthHash(uint64_t keylength, uint64_t length);
uint64_t precompReduction64(__m128i A);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // INCLUDE_VERUS_CLHASH_H
