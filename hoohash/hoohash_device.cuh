// HoohashV110 (PEPEPOW) — CUDA device port
//
// Faithful translation of the consensus C reference
// (internal-docs/hoohash-reference/crypto/hoohash/hoohash.c,
//  Hoosat commit 9634f11410a2d71be21086e813263fa007fb6810, MIT).
//
// BLAKE3-256 is provided by the bundled, self-contained blake3_hoo_device.cuh
// (static __device__, integer-only) so this path links into its own TU without
// colliding with rinhash/blake3_device.cuh.
//
// Consensus-critical notes:
//  * The 64x64 double matmul is INHERENTLY SEQUENTIAL: the running scalar `sw`
//    and the accumulator `product[i]` carry across all 64*64 iterations, and the
//    branch (sw <= 0.02) depends on them. => one thread per nonce candidate.
//  * SafeComplexTransform's while-loop never recomputes `transformedValue`; on the
//    NaN/Inf path it only shrinks `input` until it returns 0. Net effect:
//    finite ? ComplexNonLinear(input) : 0  (rounds always ends 1 on the finite path).
//    Replicated verbatim so any future edge case matches the reference bit-for-bit.
//  * (uint64_t)product[i]: product[i] is provably >= 0 and small (every transform
//    branch returns >= 0; vector,multiplier,divider,mat all >= 0), so the GPU's
//    saturating float->uint conversion cannot diverge from x86 here.
//  * THE FP-determinism risk is sin/cos of LARGE arguments (input can reach ~6e16:
//    mat<=1e6 * hashXor<=~4e9 * vector<=15). exp/sqrt/floor/fabs are safe. This
//    header uses NATIVE double transcendentals; consensus parity vs the glibc
//    oracle is validated separately (a real-block KAT) before any software sin/cos
//    replacement is introduced.
#pragma once
#include <stdint.h>
#include <math.h>
#include "hoohash/blake3_hoo_device.cuh"

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
    // sincos() does ONE Payne-Hanek argument reduction for both sin and cos, vs two
    // separate reductions for sin(x)+cos(x). libdevice __nv_sincos shares the core
    // reduction/polynomials with __nv_sin/__nv_cos, so it is BIT-IDENTICAL to the
    // separate calls here — verified by a 1M-nonce old-vs-new GPU differential (0 diffs)
    // plus the real-block KAT tying the baseline to the consensus (glibc) digest.
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
    double f1 = (x * HOO_CTM) / 8.0 - floor((x * HOO_CTM) / 8.0);
    double f2 = (x * HOO_CTM) / 4.0 - floor((x * HOO_CTM) / 4.0);
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

__device__ __forceinline__ double hoo_TransformFactor(double x) {
    const double granularity = 1024.0;
    return x / granularity - floor(x / granularity);
}

__device__ void hoo_generateMatrix(const uint8_t* seed, double mat[64][64]) {
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
        }
    }
}

// hashBytes = firstPass (BLAKE3 of full 80-byte header). nonce = u32 LE @ offset 76.
__device__ void hoo_matmul(double mat[64][64], const uint8_t* hashBytes,
                           uint8_t* output, uint64_t nonce) {
    uint8_t  scaledValues[32];
    uint8_t  vector[64];
    double   product[64];
    uint8_t  result[32];
    uint32_t H[8];

    for (int i = 0; i < 32; i++) scaledValues[i] = 0;
    for (int i = 0; i < 64; i++) { vector[i] = 0; product[i] = 0.0; }

    for (int i = 0; i < 8; i++) H[i] = hoo_read_u32be(hashBytes + i * 4);
    double hashXor    = (double)(H[0]^H[1]^H[2]^H[3]^H[4]^H[5]^H[6]^H[7]);
    double nonceMod   = (double)(nonce & 0xFF);
    double divider    = 0.0001;
    double multiplier = 1234;
    double sw         = 0.0;

    for (int i = 0; i < 32; i++) {
        vector[2 * i]     = hashBytes[i] >> 4;
        vector[2 * i + 1] = hashBytes[i] & 0x0F;
    }

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            if (sw <= 0.02) {
                double input     = (mat[i][j] * hashXor * (double)vector[j] + nonceMod);
                double out_val   = hoo_SafeComplexTransform(input) * (double)vector[j] * multiplier;
                product[i] += out_val;
            } else {
                double out_val   = mat[i][j] * divider * (double)vector[j];
                product[i] += out_val;
            }
            sw = hoo_TransformFactor(product[i]);
        }
    }

    for (int i = 0; i < 64; i += 2) {
        uint64_t pval     = (uint64_t)product[i] + (uint64_t)product[i + 1];
        scaledValues[i / 2] = (uint8_t)(pval & 0xFF);
    }
    for (int i = 0; i < 32; i++) result[i] = hashBytes[i] ^ scaledValues[i];

    hoo_blake3_256(result, 32, output);
}

// Full HoohashV110 of an 80-byte header -> 32-byte digest (compared big-endian).
__device__ void hoohashv110_device(const uint8_t* header80, uint8_t* output) {
    uint8_t firstPass[32];
    uint8_t matrixSeed[32];
    uint8_t masked[80];

    // firstPass = BLAKE3(full 80-byte header)
    hoo_blake3_256(header80, 80, firstPass);

    // matrixSeed = BLAKE3(header with nonce bytes [76..79] zeroed)
    for (int i = 0; i < 80; i++) masked[i] = header80[i];
    masked[76] = masked[77] = masked[78] = masked[79] = 0;
    hoo_blake3_256(masked, 80, matrixSeed);

    double mat[64][64];
    hoo_generateMatrix(matrixSeed, mat);

    uint64_t nonce = (uint64_t)hoo_read_u32le(header80 + 76);
    hoo_matmul(mat, firstPass, output, nonce);
}
