// SPDX-License-Identifier: GPL-3.0-or-later
//
// Parameterized ProgPoW reference for the MeowPow/EvrProgPow/FiroPoW/SccPow
// family. This is algos/kawpow/ethash/progpow.cpp reworked so that the four
// per-variant deltas (period length, register/cache/math counts, and the keccak
// seal) are runtime inputs on a pp_params struct instead of compile-time
// constants. It lives in namespace `progpow_pp` (distinct from KawPoW's
// `progpow`) and reuses KawPoW's ethash library types/functions, so both link
// into one binary with a single shared copy of ethash.

#pragma once

#include "../kawpow/ethash/ethash.hpp"
#include "pp_params.h"

namespace progpow_pp
{
using namespace ethash;  // Reuse KawPoW's ethash namespace (single compiled copy).

// ProgPoW hash (light epoch context) for host self-test + share re-verification.
// Bit-identical to the reference for the variant described by `p`.
result hash(const epoch_context& context, const pp_params& p, int block_number,
    const hash256& header_hash, uint64_t nonce) noexcept;

}  // namespace progpow_pp
