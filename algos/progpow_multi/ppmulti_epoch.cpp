// SPDX-License-Identifier: GPL-3.0-or-later
//
// See ppmulti_epoch.h. Builds an ethash light context with decoupled seed /
// light-cache / full-dataset epochs, so one path reproduces each coin's DAG:
//   * KawPoW/RVN            : seed=light=full=epoch (standard).
//   * MeowPow (MEWC)        : epoch>=110 -> light=full=epoch*4 (seed stays epoch).
//   * EvrProgPow (EVR)      : full=epoch+256 (3x full_dataset_init_size).
//   * FiroPoW (FIRO/SCC)    : full=epoch+64  (1.5x full_dataset_init_size).
// KawPoW's own ethash create_epoch_context is left untouched.

#include "ppmulti_epoch.h"

#include "../kawpow/ethash/ethash.h"
#include "../kawpow/ethash/ethash-internal.hpp"

#include <cstdlib>
#include <new>

void pp_dag_epochs(const pp_params& p, int epoch,
    int* seed_epoch, int* light_epoch, int* full_epoch)
{
    // MeowPow scales light+full together at/above dagchange_epoch.
    const int scaled = (p.dagchange_epoch > 0 && epoch >= p.dagchange_epoch)
                           ? epoch * p.dag_epoch_mul : epoch;
    *seed_epoch  = epoch;                          // seed always the real epoch
    *light_epoch = scaled;
    *full_epoch  = scaled + p.dag_full_epoch_offset;  // Evr/Firo/SCC bump full only
}

ethash::epoch_context* pp_create_epoch_context(int seed_epoch, int light_epoch, int full_epoch)
{
    using namespace ethash;

    static constexpr size_t l1_size = 16 * 1024;              // progpow::l1_cache_size
    static constexpr size_t context_alloc_size = sizeof(hash512);

    const int light_cache_num_items = ethash_calculate_light_cache_num_items(light_epoch);
    const int full_dataset_num_items = ethash_calculate_full_dataset_num_items(full_epoch);
    const size_t light_cache_size = get_light_cache_size(light_cache_num_items);
    // Light context (no resident full dataset); l1 cache holds the first 16 KB.
    const size_t alloc_size = context_alloc_size + light_cache_size + l1_size;

    char* const alloc_data = static_cast<char*>(std::calloc(1, alloc_size));
    if (!alloc_data)
        return nullptr;

    hash512* const light_cache = reinterpret_cast<hash512*>(alloc_data + context_alloc_size);
    const hash256 epoch_seed = ethash_calculate_epoch_seed(seed_epoch);  // REAL epoch
    ethash::build_light_cache(light_cache, light_cache_num_items, epoch_seed);

    uint32_t* const l1_cache =
        reinterpret_cast<uint32_t*>(alloc_data + context_alloc_size + light_cache_size);

    ethash_epoch_context_full* const context = new (alloc_data) ethash_epoch_context_full{
        seed_epoch,             // stored epoch_number = real epoch
        light_cache_num_items,
        light_cache,
        l1_cache,
        full_dataset_num_items, // may be sized for a different (larger) epoch
        nullptr,
    };

    auto* full_dataset_2048 = reinterpret_cast<hash2048*>(l1_cache);
    for (uint32_t i = 0; i < l1_size / sizeof(hash2048); ++i)
        full_dataset_2048[i] = ethash::calculate_dataset_item_2048(*context, i);

    return context;
}
