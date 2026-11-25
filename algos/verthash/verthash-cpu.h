// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verthash (Vertcoin) CPU reference / verify oracle.
// Provenance: cpuminer-opt algo/verthash/Verthash.c (CryptoGraphics, GPLv2).
// Scalar path only; this is the authoritative host re-verify used by scanhash.

#ifndef VERTHASH_CPU_H
#define VERTHASH_CPU_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define VH_HASH_OUT_SIZE  32
#define VH_BYTE_ALIGNMENT 16
#define VH_HEADER_SIZE    80

// Self-contained Verthash hash: header80 is the 80-byte block header (with the
// nonce already placed at bytes 76..79). blob/blob_size is the resident
// verthash.dat image. Writes 32 bytes to output. Recomputes the 8x SHA3-512
// prehash internally (no per-job state), so it is safe to call from any thread.
void verthash_hash_oracle(const uint8_t *blob, size_t blob_size,
                          const void *header80, void *output);

#if defined(__cplusplus)
}
#endif

#endif // VERTHASH_CPU_H
