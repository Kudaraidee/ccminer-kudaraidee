/*
 * Argon2 device helpers shared between the argon2d family and rinhash.
 *
 * Moved verbatim out of blake2b_kernels.cu so a second algorithm can supply
 * its own initialize/finalize kernels while reusing this code; the fill
 * kernel itself needs no such treatment, being coin-agnostic and launchable
 * across translation units. Nothing here is coin-specific: the region was
 * checked to contain no __global__ and no reference to the header symbol
 * before it was moved, and the move is gated on byte-identical SASS for
 * argon2_initialize / argon2_fill / argon2_finalize.
 *
 * computeInitialHash() travels with it but is NOT generic -- it hardcodes
 * pwdlen = saltlen = 80 with salt == password, which is the argon2d-coin
 * convention. A consumer whose password or salt differs needs its own.
 */

#ifndef ARGON2D_BLAKE2B_DEVICE_CUH
#define ARGON2D_BLAKE2B_DEVICE_CUH

#include "argon2d_kernel.h"

#define CAT(x, y) CAT_(x, y)
#define CAT_(x, y) x ## y

#define G(a,b,c,d,x,col) { \
    ref1=sigma[r][col]>>16*x;\
    ref2=sigma[r][col]>>(16*x+8);\
    CAT(v,a) += CAT(v,b)+m[ref1]; \
    CAT(v,d) = rotate64(CAT(v,d) ^ CAT(v,a),32); \
    CAT(v,c) += CAT(v,d); \
    CAT(v,b) = rotate64(CAT(v,b) ^ CAT(v,c), 24); \
    CAT(v,a) +=CAT(v,b)+m[ref2]; \
    CAT(v,d) = rotate64( CAT(v,d) ^ CAT(v,a), 16); \
    CAT(v,c) += CAT(v,d); \
    CAT(v,b) = rotate64( CAT(v,b) ^ CAT(v,c), 63); \
}

__device__ __forceinline__
void enc32(void *pp, const uint32_t x) {
    uint8_t *p = (uint8_t *) pp;

    p[3] = x & 0xff;
    p[2] = (x >> 8) & 0xff;
    p[1] = (x >> 16) & 0xff;
    p[0] = (x >> 24) & 0xff;
}


/* These are given INTERNAL linkage on purpose. A plain __device__ function in a
 * header also emits a HOST-side copy with external linkage in every including
 * translation unit, so two consumers collide at link time with LNK2005 -- which
 * is exactly what happened the first time this header was used from a second
 * TU. __device__ __forceinline__ escapes it by leaving no out-of-line copy;
 * `static` is the explicit fix and does not change the device code. */
static __device__ void load_block(uint32_t* dest, uint32_t* src, uint32_t idx) {

    uint32_t i, j;

    for (i = 0; i < 64; i++) {
        j = idx + i * 4;
        dest[j] = src[j];
    }

}

static __device__
void blake2b_compress_1w(
    uint64x8* state, const uint64_t* m,
    const uint32_t step, const bool lastChunk = false,
    const size_t lastChunkSize = 0)
{

    uint64_t v0, v1, v2, v3, v4, v5, v6,
             v7, v8, v9, v10, v11, v12,
             v13, v14, v15;

    v0 = state->s0;
    v1 = state->s1;
    v2 = state->s2;
    v3 = state->s3;
    v4 = state->s4;
    v5 = state->s5;
    v6 = state->s6;
    v7 = state->s7;
    v8 = blake2b_IV[0];
    v9 = blake2b_IV[1];
    v10 = blake2b_IV[2];
    v11 = blake2b_IV[3];

    if (lastChunk) {
        v12 = blake2b_IV[4] ^ (step - 1) * BLAKE_BLOCKBYTES + lastChunkSize;
        v14 = blake2b_IV[6] ^ (uint64_t) -1;

    } else {
        v12 = blake2b_IV[4] ^ step * BLAKE_BLOCKBYTES;
        v14 = blake2b_IV[6];
    }

    v13 = blake2b_IV[5];
    v15 = blake2b_IV[7];

#pragma unroll 12
    for (int r = 0; r < 12; r++) {
        uint8_t ref1, ref2;

        /* column step */
        G(0, 4, 8, 12, 0, 0);
        G(1, 5, 9, 13, 1, 0);
        G(2, 6, 10, 14, 2, 0);
        G(3, 7, 11, 15, 3, 0);

        /* diagonal step */
        G(0, 5, 10, 15, 0, 1);
        G(1, 6, 11, 12, 1, 1);
        G(2, 7, 8, 13, 2, 1);
        G(3, 4, 9, 14, 3, 1);
    }

    state->s0 ^= v0 ^ v8;
    state->s1 ^= v1 ^ v9;
    state->s2 ^= v2 ^ v10;
    state->s3 ^= v3 ^ v11;
    state->s4 ^= v4 ^ v12;
    state->s5 ^= v5 ^ v13;
    state->s6 ^= v6 ^ v14;
    state->s7 ^= v7 ^ v15;

}


static __device__ void blake2b_compress_4w(
    struct partialState* state, uint64_t* m,
    uint32_t step, uint32_t idx,
    bool lastChunk = false, size_t lastChunkSize = 0)
{

    uint64_t a, b, c, d;

    uint64_t counter = (idx == 0 ? step : 0);

    a = state->a;
    b = state->b;
    c = blake2b_IV[idx];

    if (lastChunk) {
        if (idx == 0)
            d = blake2b_IV[4] ^ (step - 1) * BLAKE_BLOCKBYTES + lastChunkSize;
        else if (idx == 2)
            d = blake2b_IV[6] ^ (uint64_t) -1;
        else
            d = blake2b_IV[idx + 4];
    } else {
        d = blake2b_IV[idx + 4] ^ counter * BLAKE_BLOCKBYTES;
    }

    __syncthreads();

    for (uint32_t r = 0; r < 12; ++r) {

        uint8_t ref1, ref2;

        ref1 = sigma[r][0] >> 8 * 2 * idx;
        ref2 = sigma[r][0] >> 8 * (2 * idx + 1);

        g_shuffle(&a, &b, &c, &d, &m[ref1], &m[ref2]);

        b = __shfl_sync(0xffffffff, b, idx + 1, 4);
        c = __shfl_sync(0xffffffff, c, idx + 2, 4);
        d = __shfl_sync(0xffffffff, d, idx + 3, 4);

        ref1 = sigma[r][1] >> 8 * 2 * idx;
        ref2 = sigma[r][1] >> 8 * (2 * idx + 1);

        g_shuffle(&a, &b, &c, &d, &m[ref1], &m[ref2]);

        b = __shfl_sync(0xffffffff, b, idx - 1, 4);
        c = __shfl_sync(0xffffffff, c, idx - 2, 4);
        d = __shfl_sync(0xffffffff, d, idx - 3, 4);

    }

    state->a = state->a ^ a ^ c;
    state->b = state->b ^ b ^ d;

}


static __device__ void computeInitialHash(
    const uint32_t* input, uint32_t* buffer,
    uint32_t nonce, uint32_t mcost, uint32_t lanes, uint32_t passes,
    uint32_t version, uint32_t type)
{

    uint64x8 state;

#pragma unroll
    for (int i = 0; i < 32; i++)
        buffer[i] = 0;

    state.s0 = blake2b_Init[0];
    state.s1 = blake2b_Init[1];
    state.s2 = blake2b_Init[2];
    state.s3 = blake2b_Init[3];
    state.s4 = blake2b_Init[4];
    state.s5 = blake2b_Init[5];
    state.s6 = blake2b_Init[6];
    state.s7 = blake2b_Init[7];

    /* H0 pre-hash input, in argon2ref/core.c initial_hash() order:
     * lanes, outlen, m_cost, t_cost, version, type, pwdlen, pwd,
     * saltlen, salt, secretlen(0), adlen(0). */
    buffer[0] = lanes;
    buffer[1] = ALGO_OUTLEN;
    buffer[2] = mcost;
    buffer[3] = passes;
    buffer[4] = version;
    buffer[5] = type;
    buffer[6] = 80;

#pragma unroll
    for (int i = 0; i < 19; i++)
        buffer[7 + i] = input[i];

    enc32(&buffer[26],nonce);
    buffer[27] = 80;

#pragma unroll
    for (int i = 0; i < 4; i++)
        buffer[28 + i] = input[i];

    blake2b_compress_1w(&state, (uint64_t*) buffer, 1);


#pragma unroll
    for (int i = 0; i < 15; i++)
        buffer[i] = input[i + 4];

    enc32(&buffer[15],nonce);

#pragma unroll
    for (int i = 16; i < 32; i++)
        buffer[i] = 0;

    blake2b_compress_1w(&state, (uint64_t*) buffer, 2, true, 72);

#pragma unroll
    for (int i = 0; i < 32; i++)
        buffer[i] = 0;


    memcpy(&buffer[1], &state, 64);

}

static __device__ void fillFirstBlock(struct block* memory, uint32_t* buffer,
    uint32_t lanes, uint32_t total_blocks) {

    uint32_t row = threadIdx.x / lanes;
    uint32_t column = threadIdx.x % lanes;

    struct block* memCell = (memory + (blockIdx.x * blockDim.y + threadIdx.y) * total_blocks)
                            + row * lanes + column;

    uint64_t* buffer_64 = (uint64_t*) buffer;
    uint64x8 state;

    state.s0 = blake2b_Init[0];
    state.s1 = blake2b_Init[1];
    state.s2 = blake2b_Init[2];
    state.s3 = blake2b_Init[3];
    state.s4 = blake2b_Init[4];
    state.s5 = blake2b_Init[5];
    state.s6 = blake2b_Init[6];
    state.s7 = blake2b_Init[7];

    buffer[0] = 1024;
    buffer[17] = row;
    buffer[18] = column;

    blake2b_compress_1w(&state, buffer_64, 1, true, 76);

    memCell->data[0] = state.s0;
    memCell->data[1] = state.s1;
    memCell->data[2] = state.s2;
    memCell->data[3] = state.s3;

    for (int i = 0; i < 8; i++) {
        buffer_64[i + 8] = 0;
    }

    buffer_64[0] = state.s0;
    buffer_64[1] = state.s1;
    buffer_64[2] = state.s2;
    buffer_64[3] = state.s3;
    buffer_64[4] = state.s4;
    buffer_64[5] = state.s5;
    buffer_64[6] = state.s6;
    buffer_64[7] = state.s7;

    for (uint8_t i = 1; i < 31; i++) {

        state.s0 = blake2b_Init[0];
        state.s1 = blake2b_Init[1];
        state.s2 = blake2b_Init[2];
        state.s3 = blake2b_Init[3];
        state.s4 = blake2b_Init[4];
        state.s5 = blake2b_Init[5];
        state.s6 = blake2b_Init[6];
        state.s7 = blake2b_Init[7];

        blake2b_compress_1w(&state, buffer_64, 1, true, 64);

        buffer_64[0] = state.s0;
        buffer_64[1] = state.s1;
        buffer_64[2] = state.s2;
        buffer_64[3] = state.s3;
        buffer_64[4] = state.s4;
        buffer_64[5] = state.s5;
        buffer_64[6] = state.s6;
        buffer_64[7] = state.s7;

        memCell->data[(i << 2) + 0] = state.s0;
        memCell->data[(i << 2) + 1] = state.s1;
        memCell->data[(i << 2) + 2] = state.s2;
        memCell->data[(i << 2) + 3] = state.s3;

    }

    memCell->data[124] = state.s4;
    memCell->data[125] = state.s5;
    memCell->data[126] = state.s6;
    memCell->data[127] = state.s7;

}


#endif /* ARGON2D_BLAKE2B_DEVICE_CUH */
