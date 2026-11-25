/*
 * The BLAKE3 hasher API that the vendored reference (ref/hoohash.c, unchanged)
 * includes as "../blake3.h", mapped onto the tree's single-chunk BLAKE3
 * (cuda/blake3_device.cuh, host + device, anchored to the published vectors).
 * The reference only hashes 80- and 32-byte inputs, well inside one chunk.
 */
#ifndef HOOHASH_REF_BLAKE3_SHIM_H
#define HOOHASH_REF_BLAKE3_SHIM_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "cuda/blake3_device.cuh"

typedef struct {
	uint8_t buf[1024];   /* one BLAKE3 chunk */
	size_t  len;
	int     overflow;
} blake3_hasher;

static inline void blake3_hasher_init(blake3_hasher *h)
{
	h->len = 0;
	h->overflow = 0;
}

static inline void blake3_hasher_update(blake3_hasher *h, const void *in, size_t n)
{
	if (n > sizeof(h->buf) - h->len) { h->overflow = 1; return; }
	memcpy(h->buf + h->len, in, n);
	h->len += n;
}

static inline void blake3_hasher_finalize(const blake3_hasher *h, uint8_t *out, size_t outlen)
{
	uint8_t d[32];
	if (h->overflow) memset(d, 0xff, sizeof(d));   /* never pass a compare */
	else blake3_256(h->buf, h->len, d);
	memcpy(out, d, outlen < sizeof(d) ? outlen : sizeof(d));
}

#endif
