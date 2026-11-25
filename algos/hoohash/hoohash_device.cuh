// HoohashV110 (PEPEPOW) - CUDA device port
//
// Faithful translation of the consensus C reference
// (Hoosat crypto/hoohash/hoohash.c,
//  commit 9634f11410a2d71be21086e813263fa007fb6810, MIT).
//
// BLAKE3-256 comes from the shared cuda/blake3_device.cuh: single chunk,
// integer-only, all static, so several TUs may include it.
//
// Consensus-critical notes:
//  * The 64x64 scan is sequential per nonce: the row sum and `sw` carry across all
//    iterations and the branch (sw <= 0.02) depends on them.
//  * SafeComplexTransform's while-loop never recomputes `transformedValue`; on the
//    NaN/Inf path it only shrinks `input` until it returns 0. Net effect:
//    finite ? ComplexNonLinear(input) : 0  (rounds always ends 1 on the finite path).
//    Replicated verbatim so any future edge case matches the reference bit-for-bit.
//  * (uint64_t)rowsum: every row sum is provably >= 0 and small (every transform
//    branch returns >= 0; vector,multiplier,divider,mat all >= 0), so the GPU's
//    saturating float->uint conversion cannot diverge from x86 here.
//  * The FP-determinism risk is sin/cos of LARGE arguments (up to ~6e16:
//    mat<=1e6 * hashXor<=~4e9 * vector<=15). Native double transcendentals are
//    used; their parity with glibc is checked by the init self-test.
#pragma once
#include <stdint.h>
#include <math.h>
#include "cuda/blake3_device.cuh"

#define HOO_PI  3.14159265358979323846
#define HOO_EPS 1e-9
#define HOO_CTM 0.000001   // COMPLEX_TRANSFORM_MULTIPLIER

typedef struct { uint64_t s0, s1, s2, s3; } hoo_xoshiro;

__device__ __forceinline__ uint64_t hoo_rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

__device__ __forceinline__ uint64_t hoo_read_u64le(const uint8_t* d) {
    uint64_t r = 0;
    for (int i = 0; i < 8; i++) r |= ((uint64_t)d[i]) << (i * 8);
    return r;
}
__device__ __forceinline__ uint32_t hoo_read_u32le(const uint8_t* d) {
    return (uint32_t)d[0] | ((uint32_t)d[1] << 8) |
           ((uint32_t)d[2] << 16) | ((uint32_t)d[3] << 24);
}
__device__ __forceinline__ uint32_t hoo_read_u32be(const uint8_t* d) {
    return ((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) |
           ((uint32_t)d[2] << 8) | (uint32_t)d[3];
}

__device__ __forceinline__ uint64_t hoo_xoshiro_gen(hoo_xoshiro* x) {
    uint64_t res = hoo_rotl64(x->s0 + x->s3, 23) + x->s0;
    uint64_t t   = x->s1 << 17;
    x->s2 ^= x->s0;
    x->s3 ^= x->s1;
    x->s1 ^= x->s2;
    x->s0 ^= x->s3;
    x->s2 ^= t;
    x->s3 = hoo_rotl64(x->s3, 45);
    return res;
}

// --- nonlinear transforms (native transcendentals) ---
__device__ __forceinline__ double hoo_Medium(double x) {
    // One argument reduction for both; bit-identical to separate sin()/cos() calls
    // (libdevice shares the reduction and polynomials).
    double s, c;
    sincos(x, &s, &c);
    return exp(s + c);
}
__device__ __forceinline__ double hoo_Intermediate(double x) {
    if (fabs(x - HOO_PI / 2) < HOO_EPS || fabs(x - 3 * HOO_PI / 2) < HOO_EPS)
        return 0.0; // avoid singularity
    double s = sin(x); // explicit CSE: one reduction, squared (bit-exact vs sin(x)*sin(x))
    return s * s;
}
__device__ __forceinline__ double hoo_High(double x) {
    return 1.0 / sqrt(fabs(x) + 1.0);
}

__device__ double hoo_ComplexNonLinear(double x) {
    // f1 = frac(u), f2 = frac(2u) for u = x*CTM/8. Deriving f2 from f1 is exact,
    // not an approximation: 2u and 2*f1 are exact, and subtracting 1 from [1,2) is
    // exact. Do not replace with a second floor of x*CTM/4 -- same value, more work.
    const double u  = (x * HOO_CTM) / 8.0;
    const double f1 = u - floor(u);
    const double t  = f1 + f1;
    const double f2 = (t >= 1.0) ? (t - 1.0) : t;
    if (f1 < 0.33) {
        if      (f2 < 0.25) return hoo_Medium(x + (1 + f2));
        else if (f2 < 0.5)  return hoo_Medium(x - (1 + f2));
        else if (f2 < 0.75) return hoo_Medium(x * (1 + f2));
        else                return hoo_Medium(x / (1 + f2));
    } else if (f1 < 0.66) {
        if      (f2 < 0.25) return hoo_Intermediate(x + (1 + f2));
        else if (f2 < 0.5)  return hoo_Intermediate(x - (1 + f2));
        else if (f2 < 0.75) return hoo_Intermediate(x * (1 + f2));
        else                return hoo_Intermediate(x / (1 + f2));
    } else {
        if      (f2 < 0.25) return hoo_High(x + (1 + f2));
        else if (f2 < 0.5)  return hoo_High(x - (1 + f2));
        else if (f2 < 0.75) return hoo_High(x * (1 + f2));
        else                return hoo_High(x / (1 + f2));
    }
}

__device__ double hoo_SafeComplexTransform(double input) {
    double transformedValue;
    double rounds = 1;
    transformedValue = hoo_ComplexNonLinear(input);
    while (isnan(transformedValue) || isinf(transformedValue)) {
        input = input * 0.1;
        if (input <= 0.0000000000001) return 0;
        rounds++;
    }
    return transformedValue * rounds;
}

// sw = frac(p / 1024) <= 0.02 for finite p >= 0, evaluated exactly on the bit pattern:
// p / 1024 and frac are exact, so this is (p mod 1024) <= 1024 * 0.02.
__device__ __forceinline__ bool hoo_sw_low(double p) {
    const uint64_t b  = (uint64_t)__double_as_longlong(p);
    const uint64_t cb = (uint64_t)__double_as_longlong(0.02 * 1024.0);
    const int e = (int)(b >> 52) - 1023;
    if (e < 10) return b <= cb;       // p < 1024: the remainder is p itself
    if (e >= 62) return true;
    const uint64_t m   = (b & 0xFFFFFFFFFFFFFull) | (1ull << 52);
    const uint64_t rem = m & ((1ull << (62 - e)) - 1);
    const uint64_t cm  = (cb & 0xFFFFFFFFFFFFFull) | (1ull << 52);
    return rem <= (cm >> (e - ((int)(cb >> 52) - 1023)));
}

// Fills mat[][] and matdiv[][] = mat[][] * divider (the cheap step's intermediate).
__device__ void hoo_generateMatrix(const uint8_t* seed, double mat[64][64],
                                   double matdiv[64][64]) {
    hoo_xoshiro st;
    st.s0 = hoo_read_u64le(seed + 0);
    st.s1 = hoo_read_u64le(seed + 8);
    st.s2 = hoo_read_u64le(seed + 16);
    st.s3 = hoo_read_u64le(seed + 24);
    const double normalize = 1000000.0;
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            uint64_t val = hoo_xoshiro_gen(&st);
            uint32_t lo  = (uint32_t)(val & 0xFFFFFFFFu);
            mat[i][j] = (double)lo / (double)0xFFFFFFFFu * normalize;
            matdiv[i][j] = mat[i][j] * 0.0001;   // == the reference's `divider`
        }
    }
}
