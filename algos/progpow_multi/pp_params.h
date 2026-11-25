// SPDX-License-Identifier: GPL-3.0-or-later
//
// ProgPoW variant parameters (MeowPow / EvrProgPow / FiroPoW).
//
// All four are ProgPoW-over-ethash variants that differ from the shipped KawPoW
// port (algos/kawpow/) only in a handful of constants and the keccak seal. This
// module reuses KawPoW's ethash library (DAG generation, keccak, epoch_context)
// and its GPU DAG-gen kernel verbatim (compiled once, linked, no second copy),
// and parameterizes the per-variant deltas through this POD struct.
//
// POD only -- no STL, no bool -- so it is safe to include both in the ethash /
// C++-STL core AND in the ccminer bridge translation unit (which pulls miner.h,
// and miner.h macroizes `bool` via compat/stdbool.h; see kawpow.cpp).

#pragma once

#include <stdint.h>

#define PP_SEAL_SEEDWORDS 0  // inject 15 branded words (KawPoW/MeowPow/EvrProgPow)
#define PP_SEAL_VANILLA   1  // original ProgPoW 0.9.x padding (FiroPoW)

typedef struct pp_params
{
    int      epoch_length;    // ethash blocks per DAG epoch: epoch = height / epoch_length
    int      period_length;   // blocks per ProgPoW program:   period = height / period_length
    int      num_regs;        // uint32 registers per lane (16 or 32; <= 32)
    int      cnt_cache;       // random cache accesses per loop
    int      cnt_math;        // random math operations per loop
    int      cnt_dag;         // DAG accesses == ProgPoW main-loop iterations (standard 64; Meraki 32)
    int      seal_mode;       // PP_SEAL_SEEDWORDS or PP_SEAL_VANILLA
    uint32_t seed_words[15];  // keccak seed words, used only when seal_mode == SEEDWORDS
    const char* name;         // short algo name (for logs)
    // --- Non-standard DAG sizing ---------------------------------------------
    // Coins tweak the ethash DAG size in two independent ways; the SEED always
    // stays at the real epoch (so seedhash matches), only the cache/dataset
    // item COUNTS change.
    //
    // (a) MeowPow "dagchange": at/above dagchange_epoch, BOTH the light cache and
    //     full dataset are sized for (epoch * dag_epoch_mul). dagchange_epoch==0
    //     disables it.
    int dagchange_epoch;
    int dag_epoch_mul;
    // (b) EvrProgPow/FiroPoW: a larger full_dataset_init_size (Nx the standard
    //     1<<30), which — since init/growth == 128 — is exactly equivalent to
    //     sizing the FULL dataset (only) for (epoch + offset).
    //     offset = (init_mul - 1) * 128: EvrProgPow 3x -> 256, Firo 1.5x -> 64.
    //     The light cache stays standard. 0 = standard full dataset.
    int dag_full_epoch_offset;
} pp_params;
