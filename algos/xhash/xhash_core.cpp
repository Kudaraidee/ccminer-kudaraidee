// SPDX-License-Identifier: GPL-3.0-or-later
//
// XHash host core. The C/C++ reference implementation (Apache-2.0, vendored
// under vendor/xhash) owns the light cache and CPU re-verification; the CUDA
// side (GPL-3.0, adapted from ParallaxProtocol/hashwarp) owns the DAG and the
// search. No ccminer headers are included here: miner.h macroizes `bool` in a
// way the C++ standard library rejects.

#include "xhash_core.h"
#include "xhash_cuda.h"

#include <xhash/xhash.hpp>
#include <xhash/xhash.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <memory>

namespace
{
constexpr uint32_t DAG_BLOCKS = 1024;
constexpr uint32_t DAG_THREADS = 128;
constexpr uint32_t SEARCH_THREADS = 256;

struct core_state
{
	xhash::epoch_context_ptr ctx{nullptr, xhash_destroy_epoch_context};
	void *d_light = nullptr;
	void *d_dag = nullptr;
	xhash_search_results *h_results = nullptr;
	bool light_on_host = false;
	size_t light_cap = 0;
	size_t dag_cap = 0;
	uint32_t light_items = 0;
	uint32_t dag_items = 0;
	int epoch = -1;

	~core_state()
	{
		if (d_light)
		{
			if (light_on_host) cudaFreeHost(d_light);
			else cudaFree(d_light);
		}
		if (d_dag) cudaFree(d_dag);
		if (h_results) cudaFreeHost(h_results);
	}
};

uint64_t top64_be(const unsigned char target[32])
{
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v = (v << 8) | target[i];
	return v;
}

void mix_words_to_bytes(const uint32_t words[8], unsigned char out[32])
{
	for (int i = 0; i < 8; ++i)
		for (int b = 0; b < 4; ++b)
			out[4 * i + b] = (unsigned char)(words[i] >> (8 * b));
}

bool mix_matches(const xhash_search_result &r, const xhash::hash256 &expected)
{
	unsigned char mix[32];
	mix_words_to_bytes(r.mix, mix);
	return std::memcmp(mix, expected.bytes, 32) == 0;
}

int report_cuda_error(core_state *s, cudaError_t err, const char *what)
{
	if (err == cudaSuccess) return 1;
	std::fprintf(stderr, "XHash %s failed: %s\n", what, cudaGetErrorString(err));
	// Start from clean buffers on the next attempt.
	if (s->d_light) { if (s->light_on_host) cudaFreeHost(s->d_light); else cudaFree(s->d_light); s->d_light = nullptr; }
	if (s->d_dag) { cudaFree(s->d_dag); s->d_dag = nullptr; }
	s->light_cap = s->dag_cap = 0;
	s->epoch = -1;
	return 0;
}
} // namespace

extern "C" int xhash_core_epoch_from_seed(const unsigned char seed[32])
{
	xhash::hash256 h;
	std::memcpy(h.bytes, seed, 32);
	return xhash::find_epoch_number(h);
}

extern "C" void *xhash_core_create(void)
{
	core_state *s = new core_state();
	if (cudaMallocHost(&s->h_results, sizeof(xhash_search_results)) != cudaSuccess)
	{
		std::fprintf(stderr, "XHash: could not allocate pinned result buffer\n");
		delete s;
		return nullptr;
	}
	return s;
}

extern "C" void xhash_core_destroy(void *handle)
{
	if (handle) delete static_cast<core_state *>(handle);
}

extern "C" int xhash_core_ensure(void *handle, int epoch, int *regenerated)
{
	core_state *s = static_cast<core_state *>(handle);
	if (!s || epoch < 0 || epoch > XHASH_MAX_EPOCH_NUMBER) return 0;
	if (regenerated) *regenerated = 0;
	if (s->epoch == epoch && s->d_dag && (s->d_light || s->light_on_host)) return 1;

	xhash::epoch_context_ptr ctx = xhash::create_epoch_context(epoch);
	if (!ctx)
	{
		std::fprintf(stderr, "XHash: could not build epoch %d light cache\n", epoch);
		return 0;
	}

	const uint32_t light_items = (uint32_t)ctx->light_cache_num_items;
	const uint32_t dag_items = (uint32_t)ctx->full_dataset_num_items;
	const size_t light_bytes = (size_t)light_items * XHASH_LIGHT_CACHE_ITEM_SIZE;
	const size_t dag_bytes = (size_t)dag_items * XHASH_FULL_DATASET_ITEM_SIZE;

	// Epoch change: release old buffers BEFORE the memory check so the
	// previous DAG's 1 GiB is available to the new epoch.
	if (s->d_light) { if (s->light_on_host) cudaFreeHost(s->d_light); else cudaFree(s->d_light); s->d_light = nullptr; }
	if (s->d_dag) { cudaFree(s->d_dag); s->d_dag = nullptr; }
	s->light_cap = s->dag_cap = 0;
	s->ctx.reset();

	size_t free_mem = 0, total_mem = 0;
	if (!report_cuda_error(s, cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo")) return 0;

	bool light_on_host = false;
	if (free_mem < dag_bytes + light_bytes)
	{
		if (free_mem < dag_bytes + (64u << 20))
		{
			std::fprintf(stderr, "XHash epoch %d needs %.2f GiB, only %.2f GiB free\n",
				epoch, (double)dag_bytes / (1 << 30), (double)free_mem / (1 << 30));
			return 0;
		}
		light_on_host = true;
		std::fprintf(stderr, "XHash: not enough VRAM for light cache; generating DAG via host cache\n");
	}
	s->light_on_host = light_on_host;

	cudaError_t err;
	if (light_on_host) err = cudaHostAlloc(&s->d_light, light_bytes, cudaHostAllocDefault);
	else err = cudaMalloc(&s->d_light, light_bytes);
	if (!report_cuda_error(s, err, "light cache allocation")) return 0;
	s->light_cap = light_bytes;

	err = cudaMalloc(&s->d_dag, dag_bytes);
	if (!report_cuda_error(s, err, "DAG allocation")) return 0;
	s->dag_cap = dag_bytes;

	err = cudaMemcpy(s->d_light, ctx->light_cache, light_bytes,
		light_on_host ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice);
	if (!report_cuda_error(s, err, "light cache upload")) return 0;

	s->light_items = light_items;
	s->dag_items = dag_items;
	xhash_cuda_set_constants(s->d_dag, dag_items, s->d_light, light_items);
	xhash_cuda_generate_dag(dag_bytes, DAG_BLOCKS, DAG_THREADS);
	if (!report_cuda_error(s, cudaGetLastError(), "DAG generation")) return 0;

	s->ctx = std::move(ctx);
	s->epoch = epoch;
	if (regenerated) *regenerated = 1;
	return 1;
}

extern "C" int xhash_core_selftest(void *handle, int epoch)
{
	core_state *s = static_cast<core_state *>(handle);
	if (!s) return 0;
	if (!xhash_core_ensure(handle, epoch < 0 ? 0 : epoch, nullptr)) return 0;

	// Synthetic job validated against the CPU reference for the *current*
	// epoch. Comparing GPU and CPU on the live DAG checks the whole pipeline
	// without needing a second (1 GiB+) epoch-0 DAG resident.
	xhash::hash256 header;
	for (int i = 0; i < 32; ++i) header.bytes[i] = (unsigned char)(i * 7 + 1);
	const uint64_t nonce = 0x0123456789abcd00ULL;

	const xhash::result expected = xhash::hash(*s->ctx, header, nonce);
	uint64_t target = 0;
	for (int i = 0; i < 8; ++i) target = (target << 8) | expected.final_hash.bytes[i];

	xhash_cuda_set_header(header.bytes);
	xhash_cuda_set_target(target);
	xhash_cuda_search(1, 32, s->h_results, nonce, 0);
	if (!report_cuda_error(s, cudaGetLastError(), "self-test search")) return 0;

	const uint32_t found = s->h_results->count < XHASH_MAX_SEARCH_RESULTS ?
		s->h_results->count : XHASH_MAX_SEARCH_RESULTS;
	for (uint32_t i = 0; i < found; ++i)
	{
		const xhash_search_result &r = s->h_results->result[i];
		if (r.gid == 0 && mix_matches(r, expected.mix_hash)) return 1;
	}
	std::fprintf(stderr, "XHash self-test FAILED (GPU mix != CPU reference)\n");
	return 0;
}

extern "C" int xhash_core_search(void *handle, const unsigned char header[32], uint64_t start_nonce,
	const unsigned char target[32], uint32_t throughput, uint64_t *nonce_out,
	unsigned char mix_out[32], unsigned char final_out[32])
{
	core_state *s = static_cast<core_state *>(handle);
	if (!s || !s->ctx) return -1;

	xhash_cuda_set_header(header);
	xhash_cuda_set_target(top64_be(target));

	const uint32_t block = SEARCH_THREADS;
	const uint32_t grid = (throughput + block - 1) / block;
	xhash_cuda_search(grid, block, s->h_results, start_nonce, 0);
	if (!report_cuda_error(s, cudaGetLastError(), "search")) return -1;

	const uint32_t found = s->h_results->count < XHASH_MAX_SEARCH_RESULTS ?
		s->h_results->count : XHASH_MAX_SEARCH_RESULTS;
	bool gpu_rejected = false;
	xhash::hash256 hh, mix_hash, boundary;
	std::memcpy(hh.bytes, header, 32);
	std::memcpy(boundary.bytes, target, 32);
	for (uint32_t i = 0; i < found; ++i)
	{
		const xhash_search_result &r = s->h_results->result[i];
		const uint64_t nonce = start_nonce + r.gid;
		unsigned char mix_bytes[32];
		mix_words_to_bytes(r.mix, mix_bytes);

		std::memcpy(mix_hash.bytes, mix_bytes, 32);
		// Canonical verification: compares the full 256-bit boundary and checks
		// the mix against a freshly calculated one (light-cache dataset items).
		if (xhash_verify_against_boundary(s->ctx.get(), &hh, &mix_hash, nonce, &boundary) != XHASH_SUCCESS)
		{
			gpu_rejected = true;
			continue;
		}
		*nonce_out = nonce;
		std::memcpy(mix_out, mix_bytes, 32);
		// Final hash for --show-diff. The canonical verify above already checked
		// it; recomputing it only happens on a valid share, so the cost is moot.
		std::memcpy(final_out, xhash::hash(*s->ctx, hh, nonce).final_hash.bytes, 32);
		return 1;
	}
	return gpu_rejected ? -2 : 0;
}
