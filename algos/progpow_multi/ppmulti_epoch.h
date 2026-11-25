// SPDX-License-Identifier: GPL-3.0-or-later
//
// ProgPoW-family epoch-context builder with decoupled seed / light / full epochs.
//
// Coins in this family tweak the ethash DAG size while keeping the SEED at the
// real epoch (so the pool seedhash matches). Two mechanisms, both handled here:
//   * MeowPow: at/above epoch 110, light cache + full dataset sized for epoch*4.
//   * EvrProgPow/FiroPoW: a larger full_dataset_init_size (3x / 1.5x),
//     equivalent to sizing the FULL dataset (only) for epoch + offset (init/growth
//     ratio is 128, so 3x -> +256, 1.5x -> +64). Light cache stays standard.
// KawPoW/RVN is fully standard. This builder mirrors each coin's create_epoch_context.

#pragma once

#include "../kawpow/ethash/ethash.hpp"
#include "pp_params.h"

// Resolve the three epochs the DAG build needs for variant `p` at ethash `epoch`
// (== height / p.epoch_length):
//   seed_epoch  — seeds the ethash light cache (always the real epoch)
//   light_epoch — sizes the light cache
//   full_epoch  — sizes the full dataset (num_items) the GPU DAG holds
void pp_dag_epochs(const pp_params& p, int epoch,
    int* seed_epoch, int* light_epoch, int* full_epoch);

// Build a light epoch context: light cache sized for light_epoch and seeded with
// seed_epoch, full_dataset_num_items sized for full_epoch. Free with
// ethash_destroy_epoch_context.
ethash::epoch_context* pp_create_epoch_context(int seed_epoch, int light_epoch, int full_epoch);
