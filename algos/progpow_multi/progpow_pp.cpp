// SPDX-License-Identifier: GPL-3.0-or-later
//
// Parameterized ProgPoW reference (see progpow_pp.hpp). Derived operation-for-
// operation from algos/kawpow/ethash/progpow.cpp (Apache-2.0, Pawel Bylica);
// the only changes are: period_length / num_regs / cnt_cache / cnt_math are read
// from pp_params instead of file-scope constants, register arrays are sized to
// the maximum (32) and iterated to p.num_regs, and the keccak seal is either the
// 15 branded words (KawPoW-style) or the original ProgPoW padding (FiroPoW-style)
// per p.seal_mode. The RNG walk order is preserved exactly, so output is
// bit-identical to each variant's reference.

#include "progpow_pp.hpp"

#include "../kawpow/ethash/attributes.h"
#include "../kawpow/ethash/bit_manipulation.h"
#include "../kawpow/ethash/endianness.hpp"
#include "../kawpow/ethash/ethash-internal.hpp"
#include "../kawpow/ethash/kiss99.hpp"
#include "../kawpow/ethash/keccak.hpp"

#include <algorithm>
#include <utility>  // std::swap

namespace progpow_pp
{
namespace
{
// Fixed across all supported variants (only regs/cache/math/period/seal vary).
constexpr size_t num_lanes = 16;
constexpr size_t max_regs = 32;                       // sizing bound; p.num_regs <= 32
constexpr size_t l1_cache_size = 16 * 1024;
constexpr size_t l1_cache_num_items = l1_cache_size / sizeof(uint32_t);

void keccak_progpow_256(uint32_t* st) noexcept { ethash_keccakf800(st); }
inline void keccak_progpow_64(uint32_t* st) noexcept { keccak_progpow_256(st); }

// Apply the keccak seal to a pre-zeroed 25-word state. seed phase fills [10..24]
// (15 words) / final phase fills [16..24] (9 words) for the branded variants; the
// vanilla variants (FiroPoW) set the two original ProgPoW padding words.
inline void apply_seal(uint32_t state[25], const pp_params& p, bool final_phase) noexcept
{
    if (p.seal_mode == PP_SEAL_SEEDWORDS)
    {
        if (!final_phase)
            for (int i = 10; i < 25; i++) state[i] = p.seed_words[i - 10];
        else
            for (int i = 16; i < 25; i++) state[i] = p.seed_words[i - 16];
    }
    else  // PP_SEAL_VANILLA
    {
        if (!final_phase) { state[10] = 0x00000001; state[18] = 0x80008081; }
        else              { state[17] = 0x00000001; state[24] = 0x80008081; }
    }
}

// ProgPoW mix RNG state (KISS99 + Fisher-Yates permutation of register indices).
// Matches progpow::mix_rng_state; register count is runtime (p.num_regs).
class mix_rng_state
{
public:
    inline mix_rng_state(uint32_t* seed, uint32_t regs) noexcept;

    uint32_t next_dst() noexcept { return dst_seq[(dst_counter++) % num_regs]; }
    uint32_t next_src() noexcept { return src_seq[(src_counter++) % num_regs]; }

    kiss99 rng;
    uint32_t num_regs;

private:
    size_t dst_counter = 0;
    uint32_t dst_seq[max_regs];
    size_t src_counter = 0;
    uint32_t src_seq[max_regs];
};

mix_rng_state::mix_rng_state(uint32_t* hash_seed, uint32_t regs) noexcept : num_regs(regs)
{
    const auto seed_lo = static_cast<uint32_t>(hash_seed[0]);
    const auto seed_hi = static_cast<uint32_t>(hash_seed[1]);

    const auto z = fnv1a(fnv_offset_basis, seed_lo);
    const auto w = fnv1a(z, seed_hi);
    const auto jsr = fnv1a(w, seed_lo);
    const auto jcong = fnv1a(jsr, seed_hi);

    rng = kiss99{z, w, jsr, jcong};

    // Create random permutations of mix destinations / sources (Fisher-Yates).
    for (uint32_t i = 0; i < num_regs; ++i)
    {
        dst_seq[i] = i;
        src_seq[i] = i;
    }

    for (uint32_t i = num_regs; i > 1; --i)
    {
        std::swap(dst_seq[i - 1], dst_seq[rng() % i]);
        std::swap(src_seq[i - 1], src_seq[rng() % i]);
    }
}

NO_SANITIZE("unsigned-integer-overflow")
inline uint32_t random_math(uint32_t a, uint32_t b, uint32_t selector) noexcept
{
    switch (selector % 11)
    {
    default:
    case 0: return a + b;
    case 1: return a * b;
    case 2: return mul_hi32(a, b);
    case 3: return std::min(a, b);
    case 4: return rotl32(a, b);
    case 5: return rotr32(a, b);
    case 6: return a & b;
    case 7: return a | b;
    case 8: return a ^ b;
    case 9: return clz32(a) + clz32(b);
    case 10: return popcount32(a) + popcount32(b);
    }
}

NO_SANITIZE("unsigned-integer-overflow")
inline void random_merge(uint32_t& a, uint32_t b, uint32_t selector) noexcept
{
    const auto x = (selector >> 16) % 31 + 1;
    switch (selector % 4)
    {
    case 0: a = (a * 33) + b; break;
    case 1: a = (a ^ b) * 33; break;
    case 2: a = rotl32(a, x) ^ b; break;
    case 3: a = rotr32(a, x) ^ b; break;
    }
}

using lookup_fn = hash2048 (*)(const epoch_context&, uint32_t);

// Per-nonce mix state: 16 lanes x up to 32 registers (only p.num_regs used).
struct mix_t { uint32_t v[num_lanes][max_regs]; };

void round(const epoch_context& context, uint32_t r, mix_t& mix, mix_rng_state state,
    lookup_fn lookup, const pp_params& p)
{
    const uint32_t num_items = static_cast<uint32_t>(context.full_dataset_num_items / 2);
    const uint32_t item_index = mix.v[r % num_lanes][0] % num_items;
    const hash2048 item = lookup(context, item_index);

    constexpr size_t num_words_per_lane = sizeof(item) / (sizeof(uint32_t) * num_lanes);
    const int max_operations = p.cnt_cache > p.cnt_math ? p.cnt_cache : p.cnt_math;

    // Process lanes.
    for (int i = 0; i < max_operations; ++i)
    {
        if (i < p.cnt_cache)  // Random access to cached memory.
        {
            const auto src = state.next_src();
            const auto dst = state.next_dst();
            const auto sel = state.rng();

            for (size_t l = 0; l < num_lanes; ++l)
            {
                const size_t offset = mix.v[l][src] % l1_cache_num_items;
                random_merge(mix.v[l][dst], le::uint32(context.l1_cache[offset]), sel);
            }
        }
        if (i < p.cnt_math)  // Random math.
        {
            const auto src_rnd = state.rng() % (p.num_regs * (p.num_regs - 1));
            const auto src1 = src_rnd % p.num_regs;  // 0 <= src1 < num_regs
            auto src2 = src_rnd / p.num_regs;        // 0 <= src2 < num_regs - 1
            if (src2 >= src1)
                ++src2;

            const auto sel1 = state.rng();
            const auto dst = state.next_dst();
            const auto sel2 = state.rng();

            for (size_t l = 0; l < num_lanes; ++l)
            {
                const uint32_t data = random_math(mix.v[l][src1], mix.v[l][src2], sel1);
                random_merge(mix.v[l][dst], data, sel2);
            }
        }
    }

    // DAG access pattern.
    uint32_t dsts[num_words_per_lane];
    uint32_t sels[num_words_per_lane];
    for (size_t i = 0; i < num_words_per_lane; ++i)
    {
        dsts[i] = i == 0 ? 0 : state.next_dst();
        sels[i] = state.rng();
    }

    // DAG access.
    for (size_t l = 0; l < num_lanes; ++l)
    {
        const auto offset = ((l ^ r) % num_lanes) * num_words_per_lane;
        for (size_t i = 0; i < num_words_per_lane; ++i)
        {
            const auto word = le::uint32(item.word32s[offset + i]);
            random_merge(mix.v[l][dsts[i]], word, sels[i]);
        }
    }
}

void init_mix(uint32_t* hash_seed, mix_t& mix, uint32_t num_regs)
{
    const uint32_t z = fnv1a(fnv_offset_basis, static_cast<uint32_t>(hash_seed[0]));
    const uint32_t w = fnv1a(z, static_cast<uint32_t>(hash_seed[1]));

    for (uint32_t l = 0; l < num_lanes; ++l)
    {
        const uint32_t jsr = fnv1a(w, l);
        const uint32_t jcong = fnv1a(jsr, l);
        kiss99 rng{z, w, jsr, jcong};

        for (uint32_t i = 0; i < num_regs; ++i)
            mix.v[l][i] = rng();
    }
}

hash256 hash_mix(const epoch_context& context, const pp_params& p, int block_number,
    uint32_t* seed, lookup_fn lookup) noexcept
{
    mix_t mix;
    init_mix(seed, mix, static_cast<uint32_t>(p.num_regs));

    auto number = uint64_t(block_number / p.period_length);
    uint32_t new_state[2];
    new_state[0] = (uint32_t)number;
    new_state[1] = (uint32_t)(number >> 32);
    mix_rng_state state{new_state, static_cast<uint32_t>(p.num_regs)};

    for (uint32_t i = 0; i < static_cast<uint32_t>(p.cnt_dag); ++i)
        round(context, i, mix, state, lookup, p);

    // Reduce mix data to a single per-lane result.
    uint32_t lane_hash[num_lanes];
    for (size_t l = 0; l < num_lanes; ++l)
    {
        lane_hash[l] = fnv_offset_basis;
        for (int i = 0; i < p.num_regs; ++i)
            lane_hash[l] = fnv1a(lane_hash[l], mix.v[l][i]);
    }

    // Reduce all lanes to a single 256-bit result.
    static constexpr size_t num_words = sizeof(hash256) / sizeof(uint32_t);
    hash256 mix_hash;
    for (uint32_t& w : mix_hash.word32s)
        w = fnv_offset_basis;
    for (size_t l = 0; l < num_lanes; ++l)
        mix_hash.word32s[l % num_words] = fnv1a(mix_hash.word32s[l % num_words], lane_hash[l]);
    return le::uint32s(mix_hash);
}
}  // namespace

result hash(const epoch_context& context, const pp_params& p, int block_number,
    const hash256& header_hash, uint64_t nonce) noexcept
{
    uint32_t hash_seed[2];  // KISS99 initiator
    uint32_t state2[8];

    {
        uint32_t state[25] = {0x0};  // Keccak's state

        // Absorb phase for initial round of keccak: header (8 words) + nonce (2).
        for (int i = 0; i < 8; i++)
            state[i] = header_hash.word32s[i];
        state[8] = (uint32_t)nonce;
        state[9] = (uint32_t)(nonce >> 32);

        // Apply the variant's seed-phase keccak seal.
        apply_seal(state, p, /*final_phase=*/false);

        keccak_progpow_64(state);

        for (int i = 0; i < 8; i++)
            state2[i] = state[i];
    }

    hash_seed[0] = state2[0];
    hash_seed[1] = state2[1];
    const hash256 mix_hash =
        hash_mix(context, p, block_number, hash_seed, calculate_dataset_item_2048);

    uint32_t state[25] = {0x0};  // Keccak's state

    // Absorb phase for last round of keccak (256 bits): carry-over + mix.
    for (int i = 0; i < 8; i++)
        state[i] = state2[i];
    for (int i = 8; i < 16; i++)
        state[i] = mix_hash.word32s[i - 8];

    // Apply the variant's final-phase keccak seal.
    apply_seal(state, p, /*final_phase=*/true);

    keccak_progpow_256(state);

    hash256 output;
    for (int i = 0; i < 8; ++i)
        output.word32s[i] = le::uint32(state[i]);

    return {output, mix_hash};
}

}  // namespace progpow_pp
