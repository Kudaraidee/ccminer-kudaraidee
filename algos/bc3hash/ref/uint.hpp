#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "bc3_shim.h"

namespace bc3::detail {
using Wide = bc3_u128;
template <std::size_t L>
struct UInt {
    std::array<std::uint64_t, L> v{}; // least significant limb first
    constexpr UInt() = default;
    constexpr UInt(std::uint64_t x) { v[0] = x; }
    template <std::size_t M>
    explicit constexpr UInt(const UInt<M>& x)
    {
        for (std::size_t i = 0; i < L && i < M; ++i)
            v[i] = x.v[i];
    }
    static constexpr UInt hex(const char* s)
    {
        UInt r;
        for (; *s; ++s) {
            unsigned d = *s >= '0' && *s <= '9' ? unsigned(*s - '0') : unsigned(*s - 'a' + 10);
            for (std::size_t i = L - 1; i > 0; --i)
                r.v[i] = (r.v[i] << 4) | (r.v[i - 1] >> 60);
            r.v[0] = (r.v[0] << 4) | d;
        }
        return r;
    }
    static UInt load(const void* p)
    {
        const auto* b = static_cast<const std::uint8_t*>(p);
        UInt r;
        for (std::size_t i = 0; i < L; ++i)
            for (unsigned j = 0; j < 8; ++j)
                r.v[L - 1 - i] = (r.v[L - 1 - i] << 8) | b[i * 8 + j];
        return r;
    }
    void store(void* p) const
    {
        auto* b = static_cast<std::uint8_t*>(p);
        for (std::size_t i = 0; i < L; ++i)
            for (unsigned j = 0; j < 8; ++j)
                b[i * 8 + j] = std::uint8_t(v[L - 1 - i] >> (56 - j * 8));
    }
    bool zero() const
    {
        std::uint64_t r = 0;
        for (auto x : v)
            r |= x;
        return r == 0;
    }
    unsigned byte(std::size_t i) const { return unsigned((v[i / 8] >> (8 * (i % 8))) & 255); }
    friend bool operator==(const UInt& a, const UInt& b) { return a.v == b.v; }
    friend bool operator!=(const UInt& a, const UInt& b) { return !(a == b); }
    friend bool operator<(const UInt& a, const UInt& b)
    {
        for (std::size_t i = L; i--;) {
            if (a.v[i] != b.v[i]) return a.v[i] < b.v[i];
        }
        return false;
    }
    friend bool operator>=(const UInt& a, const UInt& b) { return !(a < b); }
    friend UInt operator+(const UInt& a, const UInt& b)
    {
        UInt r;
        Wide c = 0;
        for (std::size_t i = 0; i < L; ++i) {
            c = Wide(a.v[i]) + b.v[i] + std::uint64_t(c >> 64);
            r.v[i] = std::uint64_t(c);
        }
        return r;
    }
    friend UInt operator-(const UInt& a, const UInt& b)
    {
        UInt r;
        std::uint64_t borrow = 0;
        for (std::size_t i = 0; i < L; ++i) {
            Wide t = Wide(b.v[i]) + borrow;
            r.v[i] = a.v[i] - std::uint64_t(t);
            borrow = Wide(a.v[i]) < t;
        }
        return r;
    }
    friend UInt operator^(const UInt& a, const UInt& b)
    {
        UInt r;
        for (std::size_t i = 0; i < L; ++i)
            r.v[i] = a.v[i] ^ b.v[i];
        return r;
    }
    friend UInt operator&(const UInt& a, const UInt& b)
    {
        UInt r;
        for (std::size_t i = 0; i < L; ++i)
            r.v[i] = a.v[i] & b.v[i];
        return r;
    }
    friend UInt operator|(const UInt& a, const UInt& b)
    {
        UInt r;
        for (std::size_t i = 0; i < L; ++i)
            r.v[i] = a.v[i] | b.v[i];
        return r;
    }
    friend UInt operator~(const UInt& a)
    {
        UInt r;
        for (std::size_t i = 0; i < L; ++i)
            r.v[i] = ~a.v[i];
        return r;
    }
};
using U64 = UInt<1>;
using U128 = UInt<2>;
using U256 = UInt<4>;
using U512 = UInt<8>;

template <unsigned N, std::size_t L>
inline UInt<L> shr(const UInt<L>& a)
{
    UInt<L> r;
    for (std::size_t i = 0; i < L; ++i) {
        if (i + N / 64 < L) r.v[i] = a.v[i + N / 64] >> (N % 64);
        if constexpr (N % 64)
            if (i + N / 64 + 1 < L) r.v[i] |= a.v[i + N / 64 + 1] << (64 - N % 64);
    }
    return r;
}
template <unsigned N, std::size_t L>
inline UInt<L> ror(const UInt<L>& a)
{
    UInt<L> r;
    for (std::size_t i = 0; i < L; ++i) {
        r.v[i] = a.v[(i + N / 64) % L] >> (N % 64);
        if constexpr (N % 64) r.v[i] |= a.v[(i + N / 64 + 1) % L] << (64 - N % 64);
    }
    return r;
}
template <unsigned N, std::size_t L>
inline UInt<L> rol(const UInt<L>& a)
{
    return ror<unsigned(L * 64) - N % (L * 64)>(a);
}
template <std::size_t L>
inline UInt<L> ch(const UInt<L>& x, const UInt<L>& y, const UInt<L>& z)
{
    return z ^ (x & (y ^ z));
}
template <std::size_t L>
inline UInt<L> maj(const UInt<L>& x, const UInt<L>& y, const UInt<L>& z)
{
    return (x & y) | (z & (x | y));
}
template <std::size_t L>
inline UInt<L> Sig0(const UInt<L>& x)
{
    if constexpr (L == 1) return ror<28>(x) ^ ror<34>(x) ^ ror<39>(x);
    if constexpr (L == 2) return ror<57>(x) ^ ror<63>(x) ^ ror<92>(x);
    if constexpr (L == 4) return ror<113>(x) ^ ror<146>(x) ^ ror<171>(x);
}
template <std::size_t L>
inline UInt<L> Sig1(const UInt<L>& x)
{
    if constexpr (L == 1) return ror<14>(x) ^ ror<18>(x) ^ ror<41>(x);
    if constexpr (L == 2) return ror<35>(x) ^ ror<44>(x) ^ ror<51>(x);
    if constexpr (L == 4) return ror<88>(x) ^ ror<97>(x) ^ ror<201>(x);
}
template <std::size_t L>
inline UInt<L> sig0(const UInt<L>& x)
{
    if constexpr (L == 1) return ror<1>(x) ^ ror<8>(x) ^ shr<7>(x);
    if constexpr (L == 2) return ror<42>(x) ^ ror<83>(x) ^ shr<20>(x);
    if constexpr (L == 4) return ror<131>(x) ^ ror<57>(x) ^ shr<56>(x);
}
template <std::size_t L>
inline UInt<L> sig1(const UInt<L>& x)
{
    if constexpr (L == 1) return ror<19>(x) ^ ror<61>(x) ^ shr<6>(x);
    if constexpr (L == 2) return ror<24>(x) ^ ror<119>(x) ^ shr<18>(x);
    if constexpr (L == 4) return ror<53>(x) ^ ror<247>(x) ^ shr<26>(x);
}
template <std::size_t L>
inline UInt<L> beta(const UInt<L>& x)
{
    constexpr unsigned rotation{L == 1 ? 51 : L == 2 ? 54 :
                                                       81};
    return x ^ Sig0(x) ^ Sig1(x) ^ sig0(x) ^ sig1(x) ^ rol<rotation>(x);
}
template <std::size_t L>
inline UInt<L> gamma(const UInt<L>& x, const UInt<L>& y)
{
    constexpr unsigned right_rotation{L == 1 ? 30 : L == 2 ? 48 :
                                                             58};
    constexpr unsigned left_rotation{L == 1 ? 38 : L == 2 ? 99 :
                                                            183};
    constexpr unsigned shift{L == 1 ? 12 : L == 2 ? 22 :
                                                    32};
    return x ^ beta(y) ^ ror<right_rotation>(y) ^ rol<left_rotation>(sig1(x)) ^ shr<shift>(y);
}
// All algorithm moduli have the form 2^(64*L)-C, with small C.
// Two carry folds suffice: the first carry is <= C, the next <= 1.
template <std::size_t L, std::uint64_t C>
struct Mod {
    using U = UInt<L>;
    static constexpr U modulus()
    {
        U r;
        for (auto& x : r.v)
            x = UINT64_MAX;
        r.v[0] -= C - 1;
        return r;
    }
    static U normalize(U a) { return a >= modulus() ? a - modulus() : a; }
    static U reduce(const std::array<std::uint64_t, 2 * L>& t)
    {
        U r;
        Wide carry = 0;
        for (std::size_t i = 0; i < L; ++i) {
            carry = Wide(t[L + i]) * C + t[i] + std::uint64_t(carry >> 64);
            r.v[i] = std::uint64_t(carry);
        }
        carry = Wide(std::uint64_t(carry >> 64)) * C;
        for (std::size_t i = 0; i < L; ++i) {
            carry += r.v[i];
            r.v[i] = std::uint64_t(carry);
            carry >>= 64;
        }
        // If the second fold overflowed, adding C cannot overflow again.
        if (carry) r = r + U(C);
        return normalize(r);
    }
    static U mul(const U& a, const U& b)
    {
        std::array<std::uint64_t, 2 * L> t{};
        for (std::size_t i = 0; i < L; ++i) {
            Wide carry = 0;
            for (std::size_t j = 0; j < L; ++j) {
                carry = Wide(a.v[i]) * b.v[j] + t[i + j] + std::uint64_t(carry >> 64);
                t[i + j] = std::uint64_t(carry);
            }
            t[i + L] = std::uint64_t(carry >> 64);
        }
        return reduce(t);
    }
    static U square(const U& a) { return mul(a, a); }
    // Sliding window, exponent is public and fixed for these calls.
    static U pow(U a, const U& e)
    {
        a = normalize(a);
        std::array<U, 8> odd;
        odd[0] = a;
        const U a2 = square(a);
        for (unsigned i = 1; i < 8; ++i)
            odd[i] = mul(odd[i - 1], a2);
        U r(1);
        int i = int(64 * L) - 1;
        while (i >= 0) {
            if (((e.v[unsigned(i) / 64] >> (unsigned(i) % 64)) & 1) == 0) {
                r = square(r);
                --i;
                continue;
            }
            int low = i > 3 ? i - 3 : 0;
            while (((e.v[unsigned(low) / 64] >> (unsigned(low) % 64)) & 1) == 0)
                ++low;
            unsigned index = 0;
            for (int j = i; j >= low; --j) {
                index = (index << 1) | unsigned((e.v[unsigned(j) / 64] >> (unsigned(j) % 64)) & 1);
                r = square(r);
            }
            r = mul(r, odd[index / 2]);
            i = low - 1;
        }
        return r;
    }
};
using M64 = Mod<1, 59>;
using M128 = Mod<2, 159>;
using M256 = Mod<4, 189>;
using M512 = Mod<8, 569>;
using F64 = Mod<1, 1469>;
using F128 = Mod<2, 15449>;
using F256 = Mod<4, 36113>;
using F512 = Mod<8, 38117>;
// Build the sliding-window schedule at compile time. Paired evaluation exposes
// two independent carry chains to the CPU; it does not create worker threads.
template <class M, unsigned Window = 4>
struct FixedExponent {
    using U = typename M::U;
    struct Step {
        unsigned squares, index;
    };
    std::array<Step, sizeof(U) * 8> steps{};
    unsigned count = 0, tail = 0;
    constexpr explicit FixedExponent(const U& exponent)
    {
        int i = int(sizeof(U) * 8) - 1;
        auto bit = [&](int n) constexpr { return unsigned((exponent.v[unsigned(n) / 64] >> (unsigned(n) % 64)) & 1); };
        while (i >= 0 && !bit(i))
            --i;
        while (i >= 0) {
            if (!bit(i)) {
                ++tail;
                --i;
                continue;
            }
            int low = i >= int(Window - 1) ? i - int(Window - 1) : 0;
            while (!bit(low))
                ++low;
            unsigned value = 0;
            for (int j = i; j >= low; --j)
                value = (value << 1) | bit(j);
            steps[count++] = {tail + unsigned(i - low + 1), value / 2};
            tail = 0;
            i = low - 1;
        }
    }
    U operator()(U a) const
    {
        if (!count) return U(1);
        std::array<U, 1u << (Window - 1)> odd;
        odd[0] = M::normalize(a);
        const U a2 = M::square(odd[0]);
        for (unsigned i = 1; i < odd.size(); ++i)
            odd[i] = M::mul(odd[i - 1], a2);
        U r = odd[steps[0].index];
        for (unsigned i = 1; i < count; ++i) {
            for (unsigned j = 0; j < steps[i].squares; ++j)
                r = M::square(r);
            r = M::mul(r, odd[steps[i].index]);
        }
        for (unsigned i = 0; i < tail; ++i)
            r = M::square(r);
        return r;
    }
    std::pair<U, U> pair(U a, U b) const
    {
        if (!count) return {U(1), U(1)};
        std::array<U, 1u << (Window - 1)> odda, oddb;
        odda[0] = M::normalize(a);
        oddb[0] = M::normalize(b);
        const U a2 = M::square(odda[0]), b2 = M::square(oddb[0]);
        for (unsigned i = 1; i < odda.size(); ++i) {
            odda[i] = M::mul(odda[i - 1], a2);
            oddb[i] = M::mul(oddb[i - 1], b2);
        }
        U r = odda[steps[0].index], s = oddb[steps[0].index];
        for (unsigned i = 1; i < count; ++i) {
            for (unsigned j = 0; j < steps[i].squares; ++j) {
                r = M::square(r);
                s = M::square(s);
            }
            r = M::mul(r, odda[steps[i].index]);
            s = M::mul(s, oddb[steps[i].index]);
        }
        for (unsigned i = 0; i < tail; ++i) {
            r = M::square(r);
            s = M::square(s);
        }
        return {r, s};
    }
};
// g^x = product table[position][byte]. Tables are immutable and shared.
template <class M>
struct alignas(64) FixedPower {
    using U = typename M::U;
    static constexpr std::size_t bytes = sizeof(U);
    std::array<std::array<U, 256>, bytes> table;
    explicit FixedPower(U base)
    {
        for (auto& row : table) {
            row[0] = U(1);
            row[1] = base;
            for (unsigned j = 2; j < 256; ++j)
                row[j] = M::mul(row[j - 1], base);
            base = M::mul(row[255], base);
        }
    }
    U operator()(const U& e) const
    {
        U r = table[0][e.byte(0)];
        for (std::size_t i = 1; i < bytes; ++i)
            r = M::mul(r, table[i][e.byte(i)]);
        return r;
    }
    std::pair<U, U> pair(const U& a, const U& b) const
    {
        U r = table[0][a.byte(0)], s = table[0][b.byte(0)];
        for (std::size_t i = 1; i < bytes; ++i) {
            r = M::mul(r, table[i][a.byte(i)]);
            s = M::mul(s, table[i][b.byte(i)]);
        }
        return {r, s};
    }
};
} // namespace bc3::detail
