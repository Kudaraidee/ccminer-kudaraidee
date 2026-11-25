#include "standard_hashes.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace bc3::standard {
namespace {
template <class W>
constexpr W left(W v, unsigned n)
{
    constexpr unsigned bits = 8 * sizeof(W);
    return (v << (n % bits)) | (v >> ((bits - n) % bits));
}
template <class W>
constexpr W right(W v, unsigned n)
{
    return left(v, 8 * sizeof(W) - n);
}
template <class W, bool Big = false>
W read(const std::uint8_t* p)
{
    W v = 0;
    for (unsigned i = 0; i < sizeof(W); ++i)
        v |= W(p[i]) << (8 * (Big ? sizeof(W) - i - 1 : i));
    return v;
}
template <class W, bool Big = false>
void write(std::uint8_t* p, W v)
{
    for (unsigned i = 0; i < sizeof(W); ++i)
        p[i] = std::uint8_t(v >> (8 * (Big ? sizeof(W) - i - 1 : i)));
}
template <std::size_t N, class Compress>
void blocks(std::array<std::uint8_t, N>& buffer, std::size_t& used,
            const void* ptr, std::size_t length, Compress compress)
{
    if (!length) return;
    const auto* data = static_cast<const std::uint8_t*>(ptr);
    if (used) {
        const auto take = std::min(N - used, length);
        std::memcpy(buffer.data() + used, data, take);
        used += take;
        data += take;
        length -= take;
        if (used == N) {
            compress(buffer.data());
            used = 0;
        }
    }
    while (length >= N) {
        compress(data);
        data += N;
        length -= N;
    }
    if (length) {
        std::memcpy(buffer.data(), data, length);
        used = length;
    }
}
constexpr std::array<std::uint64_t, 8> iv64{{0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                                             0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179}};
constexpr std::array<std::uint32_t, 8> iv32{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};

// FIPS 202: derive rho offsets and iota constants at compile time.
constexpr auto rho_offsets()
{
    std::array<unsigned, 25> rho{};
    unsigned x = 1, y = 0;
    for (unsigned t = 0; t < 24; ++t) {
        rho[x + 5 * y] = ((t + 1) * (t + 2) / 2) % 64;
        const auto nx = y;
        y = (2 * x + 3 * y) % 5;
        x = nx;
    }
    return rho;
}
constexpr auto iota_constants()
{
    std::array<std::uint64_t, 24> result{};
    unsigned lfsr = 1;
    for (auto& rc : result)
        for (unsigned j = 0; j < 7; ++j) {
            if (lfsr & 1) rc ^= std::uint64_t{1} << ((1u << j) - 1);
            lfsr = ((lfsr << 1) ^ ((lfsr & 128) ? 0x71 : 0)) & 255;
        }
    return result;
}
constexpr auto rho = rho_offsets();
constexpr auto iota = iota_constants();

// Fractional cube roots of the first 80 primes, as specified by FIPS 180-4.
constexpr std::uint64_t sha512_round[80] = {
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817};
} // namespace

Sha3::Sha3(unsigned bits) : rate_(200 - bits / 4), output_(bits / 8)
{
    if (bits != 224 && bits != 256 && bits != 384 && bits != 512) throw std::invalid_argument("SHA3 output size");
}
void Sha3::permute()
{
    for (const auto rc : iota) {
        std::array<std::uint64_t, 5> parity;
        std::array<std::uint64_t, 25> moved;
        for (unsigned x = 0; x < 5; ++x)
            parity[x] = lanes_[x] ^ lanes_[x + 5] ^ lanes_[x + 10] ^ lanes_[x + 15] ^ lanes_[x + 20];
        for (unsigned y = 0; y < 5; ++y)
            for (unsigned x = 0; x < 5; ++x) {
                const auto corrected = lanes_[x + 5 * y] ^ parity[(x + 4) % 5] ^ left(parity[(x + 1) % 5], 1);
                moved[y + 5 * ((2 * x + 3 * y) % 5)] = left(corrected, rho[x + 5 * y]);
            }
        for (unsigned y = 0; y < 5; ++y)
            for (unsigned x = 0; x < 5; ++x)
                lanes_[x + 5 * y] = moved[x + 5 * y] ^ ((~moved[(x + 1) % 5 + 5 * y]) & moved[(x + 2) % 5 + 5 * y]);
        lanes_[0] ^= rc;
    }
}
void Sha3::update(const void* ptr, std::size_t length)
{
    const auto* data = static_cast<const std::uint8_t*>(ptr);
    while (length) {
        if (position_ == 0 && length >= rate_) {
            for (std::size_t i = 0; i < rate_ / 8; ++i)
                lanes_[i] ^= read<std::uint64_t>(data + 8 * i);
            data += rate_;
            length -= rate_;
            permute();
        } else {
            const auto take = std::min(length, rate_ - position_);
            for (std::size_t i = 0; i < take; ++i) {
                const auto p = position_ + i;
                lanes_[p / 8] ^= std::uint64_t(data[i]) << (8 * (p % 8));
            }
            data += take;
            length -= take;
            position_ += take;
            if (position_ == rate_) {
                permute();
                position_ = 0;
            }
        }
    }
}
void Sha3::finish(std::uint8_t* output)
{
    lanes_[position_ / 8] ^= std::uint64_t{6} << (8 * (position_ % 8));
    lanes_[rate_ / 8 - 1] ^= std::uint64_t{0x80} << 56;
    permute();
    for (std::size_t i = 0; i < output_; ++i)
        output[i] = std::uint8_t(lanes_[i / 8] >> (8 * (i % 8)));
}

Sha512::Sha512() : chain_(iv64) {}
void Sha512::compress(const std::uint8_t* data)
{
    std::array<std::uint64_t, 80> schedule;
    for (unsigned i = 0; i < 16; ++i)
        schedule[i] = read<std::uint64_t, true>(data + 8 * i);
    for (unsigned i = 16; i < 80; ++i) {
        const auto a = schedule[i - 15], b = schedule[i - 2];
        schedule[i] = schedule[i - 16] + schedule[i - 7] + (right(a, 1) ^ right(a, 8) ^ (a >> 7)) + (right(b, 19) ^ right(b, 61) ^ (b >> 6));
    }
    auto s = chain_;
    for (unsigned i = 0; i < 80; ++i) {
        const auto head = s[7] + (right(s[4], 14) ^ right(s[4], 18) ^ right(s[4], 41)) +
                          (s[6] ^ (s[4] & (s[5] ^ s[6]))) + sha512_round[i] + schedule[i];
        const auto extra = (right(s[0], 28) ^ right(s[0], 34) ^ right(s[0], 39)) +
                           ((s[0] & s[1]) | (s[2] & (s[0] | s[1])));
        for (unsigned j = 7; j > 0; --j)
            s[j] = s[j - 1];
        s[4] += head;
        s[0] = head + extra;
    }
    for (unsigned i = 0; i < 8; ++i)
        chain_[i] += s[i];
}
void Sha512::update(const void* data, std::size_t n)
{
    const auto old = bytes_low_;
    bytes_low_ += n;
    bytes_high_ += (bytes_low_ < old);
    if (bytes_high_ >> 61) throw std::length_error("SHA512 length exceeds specification");
    blocks(pending_, used_, data, n, [this](const auto* p) { compress(p); });
}
std::array<std::uint8_t, 64> Sha512::finish()
{
    pending_[used_++] = 0x80;
    if (used_ > 112) {
        std::fill(pending_.begin() + std::ptrdiff_t(used_), pending_.end(), 0);
        compress(pending_.data());
        used_ = 0;
    }
    std::fill(pending_.begin() + std::ptrdiff_t(used_), pending_.begin() + 112, 0);
    write<std::uint64_t, true>(pending_.data() + 112, (bytes_high_ << 3) | (bytes_low_ >> 61));
    write<std::uint64_t, true>(pending_.data() + 120, bytes_low_ << 3);
    compress(pending_.data());
    std::array<std::uint8_t, 64> output;
    for (unsigned i = 0; i < 8; ++i)
        write<std::uint64_t, true>(output.data() + 8 * i, chain_[i]);
    return output;
}

namespace {
// Message permutations and rotations defined by the RIPEMD-160 specification.
constexpr unsigned ripemd_word[2][80] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
     3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
     4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13},
    {5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
     15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
     12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11}};
constexpr unsigned ripemd_rotation[2][80] = {
    {11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
     11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, 11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
     9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6},
    {8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
     9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
     8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11}};
constexpr std::uint32_t ripemd_constant[2][5] = {{0, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e},
                                                 {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0}};
std::uint32_t ripemd_boolean(unsigned round, std::uint32_t x, std::uint32_t y, std::uint32_t z)
{
    switch (round) {
    case 0: return x ^ y ^ z;
    case 1: return z ^ (x & (y ^ z));
    case 2: return (x | ~y) ^ z;
    case 3: return y ^ (z & (x ^ y));
    default: return x ^ (y | ~z);
    }
}
} // namespace
void Ripemd160::compress(const std::uint8_t* data)
{
    std::array<std::uint32_t, 16> message;
    for (unsigned i = 0; i < 16; ++i)
        message[i] = read<std::uint32_t>(data + 4 * i);
    std::array<std::array<std::uint32_t, 5>, 2> path{{chain_, chain_}};
    for (unsigned i = 0; i < 80; ++i)
        for (unsigned p = 0; p < 2; ++p) {
            auto& s = path[p];
            const auto sum = s[0] + ripemd_boolean(p ? 4 - i / 16 : i / 16, s[1], s[2], s[3]) +
                             message[ripemd_word[p][i]] + ripemd_constant[p][i / 16];
            s = {s[4], left(sum, ripemd_rotation[p][i]) + s[4], s[1], left(s[2], 10), s[3]};
        }
    const auto before = chain_;
    for (unsigned i = 0; i < 5; ++i)
        chain_[i] = before[(i + 1) % 5] + path[0][(i + 2) % 5] + path[1][(i + 3) % 5];
}
void Ripemd160::update(const void* data, std::size_t n)
{
    if (n > (UINT64_MAX >> 3) - bytes_) throw std::length_error("RIPEMD160 length exceeds specification");
    bytes_ += n;
    blocks(pending_, used_, data, n, [this](const auto* p) { compress(p); });
}
std::array<std::uint8_t, 20> Ripemd160::finish()
{
    pending_[used_++] = 0x80;
    if (used_ > 56) {
        std::fill(pending_.begin() + std::ptrdiff_t(used_), pending_.end(), 0);
        compress(pending_.data());
        used_ = 0;
    }
    std::fill(pending_.begin() + std::ptrdiff_t(used_), pending_.begin() + 56, 0);
    write(pending_.data() + 56, bytes_ << 3);
    compress(pending_.data());
    std::array<std::uint8_t, 20> output;
    for (unsigned i = 0; i < 5; ++i)
        write(output.data() + 4 * i, chain_[i]);
    return output;
}

namespace {
// BLAKE2 sigma permutations (the last two BLAKE2b rounds reuse rows 0,1).
constexpr unsigned sigma[10][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}, {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8}, {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9}, {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10}, {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}};
template <class W>
constexpr std::array<W, 8> blake_iv()
{
    if constexpr (sizeof(W) == 4)
        return iv32;
    else
        return iv64;
}
} // namespace
template <class W>
Blake2<W>::Blake2(const std::array<std::uint8_t, 2 * sizeof(W)>& personal) : chain_(blake_iv<W>())
{
    chain_[0] ^= W(0x01010000u + 8 * sizeof(W));
    chain_[6] ^= read<W>(personal.data());
    chain_[7] ^= read<W>(personal.data() + sizeof(W));
}
template <class W>
void Blake2<W>::compress(bool last)
{
    const auto old = bytes_low_;
    bytes_low_ += W(used_);
    bytes_high_ += W(bytes_low_ < old);
    std::array<W, 16> words, state;
    const auto iv = blake_iv<W>();
    for (unsigned i = 0; i < 16; ++i)
        words[i] = read<W>(pending_.data() + sizeof(W) * i);
    for (unsigned i = 0; i < 8; ++i) {
        state[i] = chain_[i];
        state[i + 8] = iv[i];
    }
    state[12] ^= bytes_low_;
    state[13] ^= bytes_high_;
    if (last) state[14] = ~state[14];
    constexpr unsigned rotations[4] = {sizeof(W) == 8 ? 32u : 16u, sizeof(W) == 8 ? 24u : 12u, sizeof(W) == 8 ? 16u : 8u, sizeof(W) == 8 ? 63u : 7u};
    for (unsigned round = 0; round < (sizeof(W) == 8 ? 12u : 10u); ++round) {
        const auto& order = sigma[round % 10];
        for (unsigned diagonal = 0; diagonal < 2; ++diagonal)
            for (unsigned col = 0; col < 4; ++col) {
                auto& a = state[col];
                auto& b = state[4 + (col + diagonal) % 4];
                auto& c = state[8 + (col + 2 * diagonal) % 4];
                auto& d = state[12 + (col + 3 * diagonal) % 4];
                const unsigned pair = 8 * diagonal + 2 * col;
                a += b + words[order[pair]];
                d = right(W(d ^ a), rotations[0]);
                c += d;
                b = right(W(b ^ c), rotations[1]);
                a += b + words[order[pair + 1]];
                d = right(W(d ^ a), rotations[2]);
                c += d;
                b = right(W(b ^ c), rotations[3]);
            }
    }
    for (unsigned i = 0; i < 8; ++i)
        chain_[i] ^= state[i] ^ state[i + 8];
}
template <class W>
void Blake2<W>::update(const void* ptr, std::size_t n)
{
    const auto* data = static_cast<const std::uint8_t*>(ptr);
    while (n) {
        if (used_ == BLOCK) {
            compress(false);
            used_ = 0;
        }
        const auto take = std::min(n, BLOCK - used_);
        std::memcpy(pending_.data() + used_, data, take);
        used_ += take;
        data += take;
        n -= take;
    }
}
template <class W>
std::array<std::uint8_t, 8 * sizeof(W)> Blake2<W>::finish()
{
    std::fill(pending_.begin() + std::ptrdiff_t(used_), pending_.end(), 0);
    compress(true);
    std::array<std::uint8_t, 8 * sizeof(W)> output;
    for (unsigned i = 0; i < 8; ++i)
        write(output.data() + sizeof(W) * i, chain_[i]);
    return output;
}
template class Blake2<std::uint32_t>;
template class Blake2<std::uint64_t>;
} // namespace bc3::standard
