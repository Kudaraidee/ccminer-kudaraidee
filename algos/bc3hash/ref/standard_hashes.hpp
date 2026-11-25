// Independently implemented from FIPS 180-4, FIPS 202, the RIPEMD-160
// specification, and the BLAKE2 specification. No library source is used.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace bc3::standard {
class Sha3
{
    std::array<std::uint64_t, 25> lanes_{};
    std::size_t rate_, position_ = 0, output_;
    void permute();

public:
    explicit Sha3(unsigned output_bits);
    void update(const void*, std::size_t);
    void finish(std::uint8_t*);
};

class Sha512
{
    std::array<std::uint64_t, 8> chain_;
    std::array<std::uint8_t, 128> pending_{};
    std::size_t used_ = 0;
    std::uint64_t bytes_low_ = 0, bytes_high_ = 0;
    void compress(const std::uint8_t*);

public:
    Sha512();
    void update(const void*, std::size_t);
    std::array<std::uint8_t, 64> finish();
};

class Ripemd160
{
    std::array<std::uint32_t, 5> chain_{{0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0}};
    std::array<std::uint8_t, 64> pending_{};
    std::size_t used_ = 0;
    std::uint64_t bytes_ = 0;
    void compress(const std::uint8_t*);

public:
    void update(const void*, std::size_t);
    std::array<std::uint8_t, 20> finish();
};

// The BC3 profile is unkeyed, sequential, with full-length output and a
// fixed-size personalization field. Deliberately no unused tree/key API.
template <class Word>
class Blake2
{
    static constexpr std::size_t BLOCK = 16 * sizeof(Word);
    std::array<Word, 8> chain_;
    std::array<std::uint8_t, BLOCK> pending_{};
    std::size_t used_ = 0;
    Word bytes_low_ = 0, bytes_high_ = 0;
    void compress(bool last);

public:
    explicit Blake2(const std::array<std::uint8_t, 2 * sizeof(Word)>& personal = {});
    void update(const void*, std::size_t);
    std::array<std::uint8_t, 8 * sizeof(Word)> finish();
};
using Blake2s = Blake2<std::uint32_t>;
using Blake2b = Blake2<std::uint64_t>;
extern template class Blake2<std::uint32_t>;
extern template class Blake2<std::uint64_t>;
} // namespace bc3::standard
