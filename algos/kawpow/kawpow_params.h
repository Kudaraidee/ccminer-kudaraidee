// SPDX-License-Identifier: GPL-3.0-or-later
//
// kawpow expressed as a pp_params instance.
//
// A header rather than a file-static in kawpow_core.cpp so there is exactly one
// definition: a second copy could drift from the one the miner uses.

#pragma once

#include "../progpow_multi/pp_params.h"

// PROGPOW_LANES 16 / REGS 32 / DAG_LOADS 4 / CACHE_WORDS 4096, 11 cache + 18
// math ops, a 64-iteration DAG loop, period = height/3, epoch = height/7500,
// and the RAVENCOINKAWPOW keccak seal (note the leading word is 0x72, 'r').
static const pp_params kPpKawpow = {
    /*epoch_length */ 7500,
    /*period_length*/ 3,
    /*num_regs     */ 32,
    /*cnt_cache    */ 11,
    /*cnt_math     */ 18,
    /*cnt_dag      */ 64,
    /*seal_mode    */ PP_SEAL_SEEDWORDS,
    /*seed_words   */ { 0x00000072, 0x00000041, 0x00000056, 0x00000045, 0x0000004E,   // rAVEN
                        0x00000043, 0x0000004F, 0x00000049, 0x0000004E, 0x0000004B,   // COINK
                        0x00000041, 0x00000057, 0x00000050, 0x0000004F, 0x00000057 }, // AWPOW
    /*name         */ "kawpow",
    /*dagchange    */ 0,     // standard ethash epoch sizing
    /*dag_epoch_mul*/ 1,
    /*dag_full_off */ 0,
};
