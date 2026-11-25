// Portability shim for the vendored BitcoinIII crypto/bc3 sources (see vendor.py).
// The coin's code needs unsigned __int128 for its 64x64->128 carry chains and
// __builtin_prefetch; MSVC has neither. gcc/clang keep the native types, so the
// Linux build is the coin's code byte for byte apart from the patched names.
#pragma once
#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>

// Unsigned 128-bit value, arithmetic mod 2^128. Only the operations crypto/bc3
// uses on its Wide type: construct from u64, +, * (low 128 bits), >>, >>=, +=,
// <, truncating u64 cast and truth test.
struct bc3_u128 {
	std::uint64_t lo, hi;
	bc3_u128(std::uint64_t x = 0) : lo(x), hi(0) {}
	bc3_u128(std::uint64_t l, std::uint64_t h) : lo(l), hi(h) {}
	explicit operator std::uint64_t() const { return lo; }
	explicit operator bool() const { return (lo | hi) != 0; }
	friend bc3_u128 operator+(bc3_u128 a, bc3_u128 b)
	{
		bc3_u128 r;
		unsigned char c = _addcarry_u64(0, a.lo, b.lo, &r.lo);
		_addcarry_u64(c, a.hi, b.hi, &r.hi);
		return r;
	}
	friend bc3_u128 operator*(bc3_u128 a, bc3_u128 b)
	{
		bc3_u128 r;
		r.lo = _umul128(a.lo, b.lo, &r.hi);
		r.hi += a.lo * b.hi + a.hi * b.lo;
		return r;
	}
	friend bc3_u128 operator>>(bc3_u128 a, unsigned n)
	{
		if (n >= 128) return bc3_u128();
		if (n >= 64) return bc3_u128(a.hi >> (n - 64), 0);
		if (n == 0) return a;
		return bc3_u128(__shiftright128(a.lo, a.hi, (unsigned char)n), a.hi >> n);
	}
	friend bool operator<(bc3_u128 a, bc3_u128 b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; }
	bc3_u128& operator+=(bc3_u128 b) { return *this = *this + b; }
	bc3_u128& operator>>=(unsigned n) { return *this = *this >> n; }
};
#define bc3_prefetch(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#else
typedef unsigned __int128 bc3_u128;
#define bc3_prefetch(p) __builtin_prefetch((p), 0, 3)
#endif
