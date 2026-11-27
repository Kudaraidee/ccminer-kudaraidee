/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * XHash CUDA kernels and host launch wrappers.
 *
 * Ported from ParallaxProtocol/hashwarp, libxhash-cuda/
 *   xhash_cuda_miner_kernel.cu, dagger_shuffled.cuh, fnv.cuh, keccak.cuh
 * (GPL-3.0, ethminer-derived). Only ccminer-specific plumbing and naming
 * changed; the consensus-critical Keccak/SHA3 padding, byte-wise FNV-1 and
 * DAG layout are kept identical.
 */

#include "xhash_keccak.cuh"
#include "xhash_dagger.cuh"

#define xhash_copy(dst, src, count)  \
	for (int xhash_i = 0; xhash_i != (count); ++xhash_i) \
		(dst)[xhash_i] = (src)[xhash_i]

template <bool DedupOffsets>
__global__ void xhash_search_kernel(volatile xhash_search_results *g_output, uint64_t start_nonce)
{
	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint2 mix[4];

	// compute_hash() returns true when the hash misses the target.
	if (compute_hash<DedupOffsets>(start_nonce + gid, mix))
		return;

	uint32_t index = atomicInc((uint32_t *)&g_output->count, 0xffffffff);
	if (index >= XHASH_MAX_SEARCH_RESULTS)
		return;

	g_output->result[index].gid = gid;
	for (int i = 0; i < 4; i++)
	{
		g_output->result[index].mix[2 * i]     = mix[i].x;
		g_output->result[index].mix[2 * i + 1] = mix[i].y;
	}
}

/* DAG generation: each thread builds one 64-byte half of a 128-byte item,
 * four threads (a CUDA warp group) cooperate on the parent folding via shuffles.
 */
__global__ void xhash_calculate_dag_item(uint32_t start)
{
	uint32_t const node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
	if (((node_index >> 1) & (~1)) >= d_dag_size)
		return;

	union {
		hash128_t dag_node;
		uint2 dag_node_mem[25];
	};

	xhash_copy(dag_node.uint4s, d_light[node_index % d_light_size].uint4s, 4);
	dag_node.words[0] ^= node_index;
	SHA3_512(dag_node_mem);

	const int thread_id = threadIdx.x & 3;

	for (uint32_t i = 0; i < 256; ++i)
	{
		uint32_t parent_index = fnv(node_index ^ i, dag_node.words[i % 16]) % d_light_size;
		for (uint32_t t = 0; t < 4; t++)
		{
			uint32_t shuffle_index = XHASH_SHFL(parent_index, t, 4);
			uint4 p4 = d_light[shuffle_index].uint4s[thread_id];
			for (int w = 0; w < 4; w++)
			{
				uint4 s4 = make_uint4(XHASH_SHFL(p4.x, w, 4), XHASH_SHFL(p4.y, w, 4),
					XHASH_SHFL(p4.z, w, 4), XHASH_SHFL(p4.w, w, 4));
				if (t == thread_id)
					dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
			}
		}
	}
	SHA3_512(dag_node_mem);

	// Threads exit above in groups of 4; the last group may cover nodes past
	// the end of the dataset (the guard at the top rounds down by two items).
	if (node_index >= 2 * d_dag_size)
		return;

	hash64_t *dag_nodes = (hash64_t *)d_dag;
	xhash_copy(dag_nodes[node_index].uint4s, dag_node.uint4s, 4);
}

#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
	fprintf(stderr, "XHash CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
	return; } } while (0)

extern "C" void xhash_cuda_set_constants(void *dag, uint32_t dag_items, void *light, uint32_t light_items)
{
	CHECK_CUDA(cudaMemcpyToSymbol(d_dag, &dag, sizeof(dag)));
	CHECK_CUDA(cudaMemcpyToSymbol(d_dag_size, &dag_items, sizeof(dag_items)));
	CHECK_CUDA(cudaMemcpyToSymbol(d_light, &light, sizeof(light)));
	CHECK_CUDA(cudaMemcpyToSymbol(d_light_size, &light_items, sizeof(light_items)));
}

extern "C" void xhash_cuda_set_header(const unsigned char header[32])
{
	CHECK_CUDA(cudaMemcpyToSymbol(d_header, header, 32));
}

extern "C" void xhash_cuda_set_target(uint64_t target)
{
	CHECK_CUDA(cudaMemcpyToSymbol(d_target, &target, sizeof(target)));
}

extern "C" void xhash_cuda_generate_dag(uint64_t dag_bytes, uint32_t blocks, uint32_t threads)
{
	const uint32_t work = (uint32_t)(dag_bytes / sizeof(hash64_t));
	const uint32_t run = blocks * threads;

	uint32_t base;
	for (base = 0; base + run <= work; base += run)
	{
		xhash_calculate_dag_item<<<blocks, threads>>>(base);
		CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
	}
	if (base < work)
	{
		uint32_t last_grid = work - base;
		last_grid = (last_grid + threads - 1) / threads;
		xhash_calculate_dag_item<<<last_grid, threads>>>(base);
		CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
	}
}

extern "C" void xhash_cuda_search(uint32_t grid, uint32_t block, xhash_search_results *output,
	uint64_t start_nonce, unsigned kernel)
{
	output->count = 0;
	if (kernel == 1)
		xhash_search_kernel<true><<<grid, block>>>((volatile xhash_search_results *)output, start_nonce);
	else
		xhash_search_kernel<false><<<grid, block>>>((volatile xhash_search_results *)output, start_nonce);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
}
