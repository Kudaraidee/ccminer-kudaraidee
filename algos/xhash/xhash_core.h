/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * XHash host core: epoch/light-cache/DAG lifecycle and GPU search with host
 * re-verification. Plain C interface so it never mixes ccminer's miner.h
 * (bool macro) world with the C++ STL used by the reference xhash library.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *xhash_core_create(void);
void xhash_core_destroy(void *handle);

/* Recover the epoch number from a pool-supplied seed hash (-1 if unknown). */
int xhash_core_epoch_from_seed(const unsigned char seed[32]);

/* Build/keep the light cache and GPU DAG for this epoch. */
int xhash_core_ensure(void *handle, int epoch, int *regenerated);

/* Compare one GPU-found nonce/mix against the CPU reference. */
int xhash_core_selftest(void *handle, int epoch);

/* Search [start_nonce, start_nonce + throughput). Returns 1 on a share that
 * also validates on the CPU, 0 on no share, -1 on launch failure, -2 when the
 * GPU flagged candidates that the CPU rejected. */
int xhash_core_search(void *handle, const unsigned char header[32], uint64_t start_nonce,
	const unsigned char target[32], uint32_t throughput, uint64_t *nonce_out,
	unsigned char mix_out[32], unsigned char final_out[32]);

#ifdef __cplusplus
}
#endif
