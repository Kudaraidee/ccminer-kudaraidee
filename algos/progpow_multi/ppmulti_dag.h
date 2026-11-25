// SPDX-License-Identifier: GPL-3.0-or-later
//
// ProgPoW-family epoch/DAG state machine, serving kawpow, meowpow, evrprogpow,
// firopow and meraki. Keyed on an explicit epoch number rather than a height --
// the core derives it with the variant's epoch_length -- so one class covers
// every epoch length and the offset/multiplier variants. Reuses KawPoW's ethash
// library and its GPU DAG-gen kernel (kawpow_generate_dag_cpu); no second copy
// of either is compiled.

#pragma once

#include <cstdint>

class ppmulti_dag
{
public:
    ppmulti_dag() = default;
    ~ppmulti_dag();

    // Ensure the DAG for `seed_epoch` is resident in VRAM, building it if the
    // epoch changed (or on first call). The light cache is sized for light_epoch
    // and seeded with seed_epoch; the full dataset (DAG) is sized for full_epoch
    // (see ppmulti_epoch.h for the per-variant epochs). Returns true on success;
    // sets *regenerated to whether a (re)build happened this call (optional).
    bool ensure(int seed_epoch, int light_epoch, int full_epoch, bool* regenerated = nullptr);

    void release();

    const uint32_t* dag() const { return d_dag_; }         // full DAG, flat 64-byte nodes
    uint32_t dataset_items_128() const { return items_; }  // full_dataset_num_items
    int epoch() const { return epoch_; }

private:
    int epoch_ = -1;
    uint32_t* d_dag_ = nullptr;
    uint32_t* d_light_ = nullptr;
    uint32_t items_ = 0;
};
