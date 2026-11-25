// SPDX-License-Identifier: GPL-3.0-or-later
//
// ProgPoW-family epoch/DAG state machine (see ppmulti_dag.h). Builds the ethash
// light cache on the host with KawPoW's vendored reference, uploads it, and
// generates the full DAG on the GPU via KawPoW's kawpow_generate_dag_cpu kernel
// (reused verbatim -- the DAG is standard ethash and variant-independent).

#include "ppmulti_dag.h"
#include "ppmulti_epoch.h"

#include "../kawpow/ethash/ethash.hpp"
#include "../kawpow/ethash/ethash-internal.hpp"
#include "../kawpow/ethash/ethash.h"

#include <cstdio>
#include <cuda_runtime.h>

// Reused from algos/kawpow/cuda_kawpow.cu (single definition in the binary).
extern "C" cudaError_t kawpow_generate_dag_cpu(
    const uint32_t* d_light, uint32_t* d_dag, uint32_t dag_nodes, uint32_t light_nodes);

ppmulti_dag::~ppmulti_dag() { release(); }

void ppmulti_dag::release()
{
    if (d_dag_)   { cudaFree(d_dag_);   d_dag_ = nullptr; }
    if (d_light_) { cudaFree(d_light_); d_light_ = nullptr; }
    epoch_ = -1;
    items_ = 0;
}

bool ppmulti_dag::ensure(int seed_epoch, int light_epoch, int full_epoch, bool* regenerated)
{
    if (regenerated) *regenerated = false;
    if (d_dag_ && seed_epoch == epoch_)
        return true;  // already resident

    // Build the host epoch context: light cache sized for light_epoch and seeded
    // with seed_epoch, full dataset sized for full_epoch (per-variant DAG sizing;
    // see ppmulti_epoch.h). Its l1_cache is host-side only, used by the host
    // reference hash -- the kernel stages its L1 out of the DAG instead.
    ethash::epoch_context_ptr ctx{
        pp_create_epoch_context(seed_epoch, light_epoch, full_epoch), ethash_destroy_epoch_context};
    if (!ctx)
    {
        fprintf(stderr, "ppmulti_dag: create_epoch_context(seed %d light %d full %d) OOM\n",
            seed_epoch, light_epoch, full_epoch);
        return false;
    }

    const uint32_t light_nodes = (uint32_t)ctx->light_cache_num_items;
    const uint32_t items = (uint32_t)ctx->full_dataset_num_items;  // 128-byte items
    const uint32_t dag_nodes = items * 2u;                         // 64-byte nodes
    const size_t light_bytes = (size_t)light_nodes * 64u;
    const size_t dag_bytes = (size_t)dag_nodes * 64u;

    release();

    cudaError_t e;
    if ((e = cudaMalloc(&d_light_, light_bytes)) != cudaSuccess) goto fail;
    if ((e = cudaMemcpy(d_light_, ctx->light_cache, light_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto fail;
    if ((e = cudaMalloc(&d_dag_, dag_bytes)) != cudaSuccess) goto fail;
    if ((e = kawpow_generate_dag_cpu(d_light_, d_dag_, dag_nodes, light_nodes)) != cudaSuccess) goto fail;

    // The light cache is only needed during DAG generation.
    cudaFree(d_light_);
    d_light_ = nullptr;

    epoch_ = seed_epoch;
    items_ = items;
    if (regenerated) *regenerated = true;
    return true;

fail:
    fprintf(stderr, "ppmulti_dag: epoch %d build failed: %s\n", seed_epoch, cudaGetErrorString(e));
    release();
    return false;
}
