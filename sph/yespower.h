/*-
 * Copyright 2009 Colin Percival
 * Copyright 2013-2018 Alexander Peslyak
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This is a proof-of-work focused fork of yescrypt.  yespower 0.5 is the
 * obsolete yescrypt 0.5 re-released under a new name and is bit-compatible
 * with it; yespower 1.0 is a new proof-of-work specific variation.  The
 * version is selected through the parameter struct below.
 */

/* Trimmed from the upstream yespower.h: this miner vendors only the reference
 * implementation, so the SIMD includes, the global parameter struct and the
 * thread-local prehash context of the optimized path are all omitted. */

#ifndef SPH_YESPOWER_H__
#define SPH_YESPOWER_H__

#include <stdint.h>
#include <stdlib.h>   /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Internal type used by the memory allocator; use yespower_local_t instead. */
typedef struct {
	void *base, *aligned;
	size_t base_size, aligned_size;
} yespower_region_t;

/* Thread-local (RAM) data structure. */
typedef yespower_region_t yespower_local_t;

/* yespower algorithm version numbers. */
typedef enum { YESPOWER_0_5 = 5, YESPOWER_1_0 = 10 } yespower_version_t;

/* yespower parameters combined into one struct.
 *
 * `pers` is hashed, so one wrong byte silently produces a valid-looking hash
 * that every pool rejects.  Callers must pass exactly the coin's bytes and
 * exactly its length -- see algos/yespower/yespower.cu for the verified table. */
typedef struct {
	yespower_version_t version;
	uint32_t N, r;
	const uint8_t *pers;
	size_t perslen;
} yespower_params_t;

/* A 256-bit yespower hash. */
typedef struct {
	unsigned char uc[32];
} yespower_binary_t;

/* yespower_init_local(local) / yespower_free_local(local):
 * Initialize / free the thread-local (RAM) data structure.
 * The reference implementation allocates per call and ignores `local`, but the
 * calls are kept so a future optimized path can reuse an allocation.
 * Return 0 on success, -1 on error.  MT-safe as long as `local` is per-thread. */
extern int yespower_init_local_ref(yespower_local_t *local);
extern int yespower_free_local_ref(yespower_local_t *local);

/* yespower_ref(local, src, srclen, params, dst):
 * Compute yespower(src[0 .. srclen-1]) into dst, to be checked against target.
 *
 * NOTE: RETURNS 1 ON SUCCESS, -1 ON ERROR.
 * Upstream's own header documents "0 on success", but the reference body this
 * was vendored from initialises retval to -1 and sets it to 1 on success, so the
 * documented convention is wrong and always has been.  Test `== 1`, never `!rc`:
 * a caller that trusts the upstream comment treats every good hash as a failure.
 *
 * This is the deliberately unoptimized reference implementation -- it is the
 * normative spec and the oracle, not a production hash path. */
extern int yespower_ref(yespower_local_t *local,
    const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst);

/* As above, using a thread-local allocation. */
extern int yespower_tls_ref(const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst);

/* yespower-b2b: same structure, BLAKE2b in the head and tail. `-a power2b`. */
extern int yespower_b2b_tls_ref(const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst);

#ifdef __cplusplus
}
#endif

#endif /* SPH_YESPOWER_H__ */
