// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verthash mining-data-file (verthash.dat) host management: locate / load /
// optionally generate the fixed ~1.19 GiB data image, compute mdiv, and verify
// its SHA-256 digest. Self-contained host code (no miner.h) so it never trips
// the bool-macro/STL trap. Provenance: cpuminer-opt algo/verthash (GPLv2).

#ifndef VERTHASH_DATA_H
#define VERTHASH_DATA_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

// Load the whole file at `path` into a freshly malloc'd host buffer.
// On success returns 0 and sets *out_buf (caller frees) and *out_size.
// Non-zero on error (message printed to stderr).
int verthash_data_load(const char *path, uint8_t **out_buf, size_t *out_size);

// mdiv = ((size - 32) / 16) + 1  (the index modulus for the IO pass).
uint32_t verthash_data_mdiv(size_t size);

// Verify the buffer's SHA-256 against the known-good Vertcoin datafile digest.
// Returns 1 if it matches the canonical file, 0 otherwise.
int verthash_data_verify(const uint8_t *buf, size_t size);

// Deterministically generate verthash.dat at `path` (index=17 graph, minutes on
// CPU, ~1.19 GiB). Returns 0 on success. Ported from verthash_generate_data_file.
int verthash_generate_data_file(const char *path);

#if defined(__cplusplus)
}
#endif

#endif // VERTHASH_DATA_H
