/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Host-callable side of the XHash CUDA kernel. This header is plain C++ (no
 * CUDA types) so the host core can include it without nvcc.
 *
 * CUDA kernels and SHA3/FNV device code are adapted from
 * ParallaxProtocol/hashwarp (GPL-3.0), itself derived from ethminer.
 */
#pragma once

#include <stdint.h>

#define XHASH_MAX_SEARCH_RESULTS 4U

struct xhash_search_result
{
	uint32_t gid;		/* offset from start_nonce */
	uint32_t mix[8];	/* mix digest, little-endian words */
};

struct xhash_search_results
{
	xhash_search_result result[XHASH_MAX_SEARCH_RESULTS];
	uint32_t count;
};

#ifdef __cplusplus
extern "C" {
#endif

void xhash_cuda_set_constants(void *dag, uint32_t dag_items, void *light, uint32_t light_items);
void xhash_cuda_set_header(const unsigned char header[32]);
void xhash_cuda_set_target(uint64_t target);
void xhash_cuda_generate_dag(uint64_t dag_bytes, uint32_t blocks, uint32_t threads);
void xhash_cuda_search(uint32_t grid, uint32_t block, xhash_search_results *output,
	uint64_t start_nonce, unsigned kernel);

#ifdef __cplusplus
}
#endif
