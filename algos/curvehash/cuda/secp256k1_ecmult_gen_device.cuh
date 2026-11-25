#ifndef SECP256K1_ECMULT_GEN_DEVICE_CUH
#define SECP256K1_ECMULT_GEN_DEVICE_CUH

/*
 * Fixed-base scalar multiplication k*G on device.
 *
 * Method: 8-bit windowed fixed base, 32 windows. Precomputed table (host-built,
 * uploaded once) holds, per window j and byte value i in 1..255, the affine
 * point (i * 256^j) * G as raw 32-byte big-endian X||Y (64 bytes/entry; entry
 * i=0 is the point at infinity and is skipped). k = sum_j window_j * 256^j, so
 * k*G = sum_j (window_j * 256^j)*G = sum_j table[j][window_j]. Table is
 * 32*256*64 = 512 KB. (8-bit windows halve the point-adds vs 4-bit.)
 *
 * No blinding (a side-channel defense irrelevant to mining): the accumulator
 * starts at infinity and gej_add_ge_var handles infinity / doubling / negation,
 * so the running sum is always correct. The result must equal libsecp256k1's
 * k*G bit-for-bit (validated by the ecmult KAT vs secp256k1_ec_pubkey_create).
 *
 * Scalar byte order: k32 is 32-byte big-endian (k32[0] = MSB), matching the
 * curvehash phash / secp256k1 seckey convention. Window j (byte j from the LSB)
 * is simply k32[31 - j].
 */

#include "secp256k1_group_device.cuh"

/* 16-bit windows: 16 over the 256-bit scalar, 65536 entries each. Half the point
 * additions of an 8-bit window, for a 64 MB table instead of 512 KB. */
#define SECP256K1_ECMULT_GEN_WBITS   16
#define SECP256K1_ECMULT_GEN_WINDOWS (256 / SECP256K1_ECMULT_GEN_WBITS)
#define SECP256K1_ECMULT_GEN_ENTRIES (1u << SECP256K1_ECMULT_GEN_WBITS)
#define SECP256K1_ECMULT_GEN_ENTRY   64  /* bytes per table entry: X[32]||Y[32] */

__device__ static void secp256k1_ecmult_gen(secp256k1_gej *r, const unsigned char *k32,
                                            const unsigned char *gtable) {
    secp256k1_ge add;
    add.infinity = 0;
    secp256k1_gej_set_infinity(r);
    for (int j = 0; j < SECP256K1_ECMULT_GEN_WINDOWS; j++) {
        /* window j = the j-th 16-bit chunk counting from the LSB; k32 is
         * big-endian, so its low byte is k32[31 - 2j] and its high k32[30 - 2j] */
        uint32_t bits = (uint32_t)k32[31 - 2 * j] | ((uint32_t)k32[30 - 2 * j] << 8);
        if (bits == 0) {
            continue; /* table[j][0] == infinity: additive no-op */
        }
        const unsigned char *p = gtable +
            ((size_t)j * SECP256K1_ECMULT_GEN_ENTRIES + bits) * SECP256K1_ECMULT_GEN_ENTRY;
        secp256k1_fe_set_b32(&add.x, p);
        secp256k1_fe_set_b32(&add.y, p + 32);
        add.infinity = 0;
        secp256k1_gej_add_ge_var(r, r, &add, NULL);
    }
}

/* k*G -> affine 32-byte big-endian X,Y. Returns 0 iff the result is infinity
 * (only when k == 0 mod n, which curvehash never hashes to). */
__device__ static int secp256k1_ecmult_gen_xy(unsigned char *x32, unsigned char *y32,
                                              const unsigned char *k32, const unsigned char *gtable) {
    secp256k1_gej rj;
    secp256k1_ge ra;
    secp256k1_ecmult_gen(&rj, k32, gtable);
    secp256k1_ge_set_gej_var(&ra, &rj);
    if (ra.infinity) {
        return 0;
    }
    secp256k1_fe_normalize_var(&ra.x);
    secp256k1_fe_normalize_var(&ra.y);
    secp256k1_fe_get_b32(x32, &ra.x);
    secp256k1_fe_get_b32(y32, &ra.y);
    return 1;
}

/*
 * The same k*G on bc3hash's 8 x 32-bit field and safegcd inverse, same table.
 * Windows come from a shifted copy of k (no run-time byte index); the output is
 * canonical, so the bytes equal secp256k1_ecmult_gen_xy's. 0 iff infinity.
 */
#include "../../bc3hash/cuda/bc3_secp_device.cuh"

__device__ static int secp256k1_ecmult_gen_xy_8x32(unsigned char *x32, unsigned char *y32,
                                                   const unsigned char *k32, const unsigned char *gtable) {
    bc3u256 k = bc3_load_be<8>(k32);
    bc3_jac r;
    r.inf = true;
    r.x = r.y = r.z = bc3_small<8>(0);
    #pragma unroll 1
    for (int j = 0; j < SECP256K1_ECMULT_GEN_WINDOWS; j++) {
        const uint32_t bits = k.w[0] & (SECP256K1_ECMULT_GEN_ENTRIES - 1);
        k = bc3_shr<SECP256K1_ECMULT_GEN_WBITS>(k);
        if (bits == 0) {
            continue; /* table[j][0] == infinity: additive no-op */
        }
        bc3u256 x2, y2;
        bc3_ec_load(x2, y2, gtable +
            ((size_t)j * SECP256K1_ECMULT_GEN_ENTRIES + bits) * SECP256K1_ECMULT_GEN_ENTRY);
        bc3_jac_add_affine(r, x2, y2);
    }
    if (r.inf) {
        return 0;
    }
    const bc3u256 zi = bc3_modinv<BC3_MOD_SECP>(bc3_fp_canon(r.z));
    const bc3u256 z2 = bc3_fp_sqr(zi), z3 = bc3_fp_mul(z2, zi);
    bc3_store_be<8>(x32, bc3_fp_canon(bc3_fp_mul(r.x, z2)));
    bc3_store_be<8>(y32, bc3_fp_canon(bc3_fp_mul(r.y, z3)));
    return 1;
}

/* As above on limbs (LS-first bc3u256 in and out), for curvehash_full_w. */
__device__ __forceinline__ int secp256k1_ecmult_gen_8x32(bc3u256 &x, bc3u256 &y, bc3u256 k,
                                                         const unsigned char *gtable) {
    bc3_jac r;
    r.inf = true;
    r.x = r.y = r.z = bc3_small<8>(0);
    #pragma unroll 1
    for (int j = 0; j < SECP256K1_ECMULT_GEN_WINDOWS; j++) {
        const uint32_t bits = k.w[0] & (SECP256K1_ECMULT_GEN_ENTRIES - 1);
        k = bc3_shr<SECP256K1_ECMULT_GEN_WBITS>(k);
        if (bits == 0) {
            continue; /* table[j][0] == infinity: additive no-op */
        }
        bc3u256 x2, y2;
        bc3_ec_load(x2, y2, gtable +
            ((size_t)j * SECP256K1_ECMULT_GEN_ENTRIES + bits) * SECP256K1_ECMULT_GEN_ENTRY);
        bc3_jac_add_affine(r, x2, y2);
    }
    if (r.inf) {
        return 0;
    }
    const bc3u256 zi = bc3_modinv<BC3_MOD_SECP>(bc3_fp_canon(r.z));
    const bc3u256 z2 = bc3_fp_sqr(zi), z3 = bc3_fp_mul(z2, zi);
    x = bc3_fp_canon(bc3_fp_mul(r.x, z2));
    y = bc3_fp_canon(bc3_fp_mul(r.y, z3));
    return 1;
}

#endif /* SECP256K1_ECMULT_GEN_DEVICE_CUH */
