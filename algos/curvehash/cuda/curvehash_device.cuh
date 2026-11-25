#ifndef CURVEHASH_DEVICE_CUH
#define CURVEHASH_DEVICE_CUH

/*
 * Full curvehash on device: SHA256(header80) then 8 rounds of
 *   pubkey = phash*G ; pub = 0x04||X||Y (65B) ; phash = SHA256(pub).
 * SHA-256 mirrors the host curve_sha256 (termux-derived) so it is bit-exact.
 * EC via secp256k1_ecmult_gen_xy (fixed-base, no blinding). Result must equal
 * the CurvehashCoin consensus hash (validated by the full-hash KAT vs the
 * textbook-secp256k1 + hashlib oracle).
 */

#include "secp256k1_ecmult_gen_device.cuh"

__device__ __constant__ static uint32_t CH_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ static __forceinline__ uint32_t ch_ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ static void ch_sha256_transform(uint32_t *S, const uint32_t *Win)
{
    uint32_t W[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = Win[i];
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ch_ror(W[i-15],7) ^ ch_ror(W[i-15],18) ^ (W[i-15] >> 3);
        uint32_t s1 = ch_ror(W[i-2],17) ^ ch_ror(W[i-2],19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint32_t a=S[0],b=S[1],c=S[2],d=S[3],e=S[4],f=S[5],g=S[6],h=S[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ch_ror(e,6) ^ ch_ror(e,11) ^ ch_ror(e,25);
        uint32_t chh = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + chh + CH_K[i] + W[i];
        uint32_t S0 = ch_ror(a,2) ^ ch_ror(a,13) ^ ch_ror(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    S[0]+=a; S[1]+=b; S[2]+=c; S[3]+=d; S[4]+=e; S[5]+=f; S[6]+=g; S[7]+=h;
}

/* SHA-256 of a short byte buffer (len <= 119 suffices here: 65 and 80).
 * Mirrors the host curve_sha256 block loop; big-endian digest out. */
__device__ static void ch_sha256(unsigned char *out, const unsigned char *data, int len)
{
    uint32_t S[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (int r = len; r > -9; r -= 64) {
        unsigned char blk[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) blk[i] = 0;
        int copy = (r > 64) ? 64 : (r < 0 ? 0 : r);
        for (int i = 0; i < copy; i++) blk[i] = data[len - r + i];
        if (r >= 0 && r < 64) blk[r] = 0x80;
        uint32_t W[16];
        #pragma unroll
        for (int i = 0; i < 16; i++)
            W[i] = ((uint32_t)blk[4*i] << 24) | ((uint32_t)blk[4*i+1] << 16) |
                   ((uint32_t)blk[4*i+2] << 8) | (uint32_t)blk[4*i+3];
        if (r < 56) W[15] = (uint32_t)(8 * len);
        ch_sha256_transform(S, W);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[4*i]   = (unsigned char)(S[i] >> 24);
        out[4*i+1] = (unsigned char)(S[i] >> 16);
        out[4*i+2] = (unsigned char)(S[i] >> 8);
        out[4*i+3] = (unsigned char)(S[i]);
    }
}

/* n = group order; a 32-byte big-endian scalar is a valid seckey iff 1<=k<n. */
__device__ __constant__ static unsigned char CH_N[32] = {
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe,
    0xba,0xae,0xdc,0xe6,0xaf,0x48,0xa0,0x3b,0xbf,0xd2,0x5e,0x8c,0xd0,0x36,0x41,0x41
};

__device__ static __forceinline__ int ch_seckey_valid(const unsigned char *k)
{
    int nz = 0, lt = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) nz |= k[i];
    if (nz == 0) return 0;                 /* k == 0 */
    /* k < n ? (big-endian compare, constant scan) */
    int decided = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (!decided) {
            if (k[i] < CH_N[i]) { lt = 1; decided = 1; }
            else if (k[i] > CH_N[i]) { lt = 0; decided = 1; }
        }
    }
    if (!decided) return 0;                /* k == n */
    return lt;
}

/*
 * out32 = curvehash(header80). Returns 1 if computed, 0 if a round hit an
 * invalid seckey (k==0 or k>=n); consensus asserts this never happens for a
 * valid block, so the caller must treat such a nonce as non-winning (never
 * submit). On the 0 path out32 is left as 0xFF.. so any target compare rejects.
 */
template <bool F8X32>
__device__ static int curvehash_full_t(unsigned char *out32, const unsigned char *header80,
                                       const unsigned char *gtable)
{
    unsigned char phash[32];
    unsigned char pub[65];
    ch_sha256(phash, header80, 80);
    #pragma unroll 1
    for (int round = 0; round < 8; round++) {
        if (!ch_seckey_valid(phash)) {
            #pragma unroll
            for (int i = 0; i < 32; i++) out32[i] = 0xFF;
            return 0;
        }
        pub[0] = 0x04;
        if (F8X32)
            secp256k1_ecmult_gen_xy_8x32(pub + 1, pub + 33, phash, gtable);
        else
            secp256k1_ecmult_gen_xy(pub + 1, pub + 33, phash, gtable);
        ch_sha256(phash, pub, 65);
    }
    #pragma unroll
    for (int i = 0; i < 32; i++) out32[i] = phash[i];
    return 1;
}

/* EC field: bc3hash's 8 x 32-bit one; -DCURVEHASH_FIELD_10X26 selects the
 * 10 x 26 libsecp256k1 port (and -DCURVEHASH_INV_ADDCHAIN its inversion). */
#ifdef CURVEHASH_FIELD_10X26
#define CURVEHASH_F8X32 false
#else
#define CURVEHASH_F8X32 true
#endif

__device__ static int curvehash_full(unsigned char *out32, const unsigned char *header80,
                                     const unsigned char *gtable)
{
    return curvehash_full_t<CURVEHASH_F8X32>(out32, header80, gtable);
}

/*
 * The same chain on 32-bit words and the 8 x 32 field, without byte arrays.
 * In: the 76 header bytes and nonce_be (header bytes 76..79 as a big-endian
 * word). Out: the final SHA-256 state S[8]; digest byte 4i = S[i] >> 24.
 */
__device__ __forceinline__ uint32_t ch_be32(const unsigned char *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

__device__ __forceinline__ void ch_sha256_iv(uint32_t S[8])
{
    S[0] = 0x6a09e667; S[1] = 0xbb67ae85; S[2] = 0x3c6ef372; S[3] = 0xa54ff53a;
    S[4] = 0x510e527f; S[5] = 0x9b05688c; S[6] = 0x1f83d9ab; S[7] = 0x5be0cd19;
}

/* 1 <= k < n on LS-first limbs */
__device__ __forceinline__ bool ch_seckey_valid_w(const bc3u256 &k)
{
    const uint32_t n[8] = { 0xd0364141u, 0xbfd25e8cu, 0xaf48a03bu, 0xbaaedce6u,
                            0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu };
    uint32_t borrow = 0, nz = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint64_t d = (uint64_t)k.w[i] - n[i] - borrow;
        borrow = (uint32_t)(d >> 63);
        nz |= k.w[i];
    }
    return nz != 0 && borrow; /* k - n borrows iff k < n */
}

__device__ static int curvehash_full_w(uint32_t S[8], const unsigned char *header76, uint32_t nonce_be,
                                       const unsigned char *gtable)
{
    uint32_t W[16];
    ch_sha256_iv(S);
    #pragma unroll
    for (int i = 0; i < 16; i++) W[i] = ch_be32(header76 + 4 * i);
    ch_sha256_transform(S, W);
    W[0] = ch_be32(header76 + 64);
    W[1] = ch_be32(header76 + 68);
    W[2] = ch_be32(header76 + 72);
    W[3] = nonce_be;
    W[4] = 0x80000000u;
    #pragma unroll
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 80 * 8;
    ch_sha256_transform(S, W);

    #pragma unroll 1
    for (int round = 0; round < 8; round++) {
        bc3u256 k, x, y;
        #pragma unroll
        for (int i = 0; i < 8; i++) k.w[7 - i] = S[i];
        if (!ch_seckey_valid_w(k)) {
            #pragma unroll
            for (int i = 0; i < 8; i++) S[i] = 0xffffffffu;
            return 0;
        }
        secp256k1_ecmult_gen_8x32(x, y, k, gtable);
        /* pub = 04 || X || Y: the coordinate words shifted one byte right */
        W[0] = 0x04000000u | (x.w[7] >> 8);
        #pragma unroll
        for (int i = 1; i < 8; i++) W[i] = __funnelshift_l(x.w[7 - i], x.w[8 - i], 24);
        W[8] = __funnelshift_l(y.w[7], x.w[0], 24);
        #pragma unroll
        for (int i = 9; i < 16; i++) W[i] = __funnelshift_l(y.w[7 - (i - 8)], y.w[7 - (i - 9)], 24);
        ch_sha256_iv(S);
        ch_sha256_transform(S, W);
        W[0] = (y.w[0] << 24) | 0x00800000u; /* last Y byte, then the 0x80 pad */
        #pragma unroll
        for (int i = 1; i < 15; i++) W[i] = 0;
        W[15] = 65 * 8;
        ch_sha256_transform(S, W);
    }
    return 1;
}

#endif /* CURVEHASH_DEVICE_CUH */
