// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Shared Luffa-512 primitive library (device side) — x-family stage function.
 *
 * Extracted bit-identically from x11/cuda_x11_luffa512.cu (luffa_for_32.c
 * lineage, sp/alexis era). The IV and round-constant tables are statically
 * initialized here (the donor uploaded them at init — a header __constant__
 * is per-TU, so static init is required for multi-TU correctness; values
 * are byte-identical).
 *
 * Interface convention: Hash[16] holds the 64-byte value exactly as it
 * sits in the inter-stage d_hash buffer (little-endian uint32 words).
 */

#ifndef CUDA_LUFFA512_DEVICE_CUH
#define CUDA_LUFFA512_DEVICE_CUH

#include <stdint.h>
#include "cuda/helper.cuh"

#ifdef __CUDACC__


#include "cuda/helper.cuh"

typedef unsigned char BitSequence;

typedef struct {
    uint32_t buffer[8]; /* Buffer to be hashed */
    uint32_t chainv[40];   /* Chaining values */
} luffa512_state;

#define LUFFA_MULT2(a,j)\
    tmp = a[7+(8*j)];\
    a[7+(8*j)] = a[6+(8*j)];\
    a[6+(8*j)] = a[5+(8*j)];\
    a[5+(8*j)] = a[4+(8*j)];\
    a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
    a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
    a[2+(8*j)] = a[1+(8*j)];\
    a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
    a[0+(8*j)] = tmp;

#if __CUDA_ARCH__ < 350
#define LUFFA_LROT(x,bits) ((x << bits) | (x >> (32 - bits)))
#else
#define LUFFA_LROT(x, bits) __funnelshift_l(x, x, bits)
#endif

#define LUFFA_TWEAK(a0,a1,a2,a3,j)\
    a0 = LUFFA_LROT(a0,j);\
    a1 = LUFFA_LROT(a1,j);\
    a2 = LUFFA_LROT(a2,j);\
    a3 = LUFFA_LROT(a3,j);

#define LUFFA_STEP(c0,c1)\
    LUFFA_SUBCRUMB(chainv[0],chainv[1],chainv[2],chainv[3],tmp);\
    LUFFA_SUBCRUMB(chainv[5],chainv[6],chainv[7],chainv[4],tmp);\
    LUFFA_MIXWORD(chainv[0],chainv[4]);\
    LUFFA_MIXWORD(chainv[1],chainv[5]);\
    LUFFA_MIXWORD(chainv[2],chainv[6]);\
    LUFFA_MIXWORD(chainv[3],chainv[7]);\
    LUFFA_ADD_CONSTANT(chainv[0],chainv[4],c0,c1);

#define LUFFA_SUBCRUMB(a0,a1,a2,a3,a4)\
    a4  = a0;\
    a0 |= a1;\
    a2 ^= a3;\
    a1  = ~a1;\
    a0 ^= a3;\
    a3 &= a4;\
    a1 ^= a3;\
    a3 ^= a2;\
    a2 &= a0;\
    a0  = ~a0;\
    a2 ^= a1;\
    a1 |= a3;\
    a4 ^= a1;\
    a3 ^= a2;\
    a2 &= a1;\
    a1 ^= a0;\
    a0  = a4;

#define LUFFA_MIXWORD(a0,a4)\
    a4 ^= a0;\
    a0  = LUFFA_LROT(a0,2);\
    a0 ^= a4;\
    a4  = LUFFA_LROT(a4,14);\
    a4 ^= a0;\
    a0  = LUFFA_LROT(a0,10);\
    a0 ^= a4;\
    a4  = LUFFA_LROT(a4,1);

#define LUFFA_ADD_CONSTANT(a0,b0,c0,c1)\
    a0 ^= c0;\
    b0 ^= c1;

/* initial values of chaining variables */
static __constant__ uint32_t c_luffa_IV[40] = {
    0x6d251e69,0x44b051e0,0x4eaa6fb4,0xdbf78465,
    0x6e292011,0x90152df4,0xee058139,0xdef610bb,
    0xc3b44b95,0xd9d2f256,0x70eee9a0,0xde099fa3,
    0x5d9b0557,0x8fc944b3,0xcf1ccf0e,0x746cd581,
    0xf7efc89d,0x5dba5781,0x04016ce5,0xad659c05,
    0x0306194f,0x666d1836,0x24aa230a,0x8b264ae7,
    0x858075d5,0x36d79cce,0xe571f7d7,0x204b1f67,
    0x35870c6a,0x57e9e923,0x14bcb808,0x7cde72ce,
    0x6c68e9be,0x5ec41e22,0xc825b7c7,0xaffb4363,
    0xf5df3999,0x0fc688f1,0xb07224cc,0x03e86cea};

static __constant__ uint32_t c_luffa_CNS[80] = {
    0x303994a6,0xe0337818,0xc0e65299,0x441ba90d,
    0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f,
    0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4,
    0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
    0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4,
    0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28,
    0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b,
    0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
    0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72,
    0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7,
    0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719,
    0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
    0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91,
    0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be,
    0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5,
    0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
    0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab,
    0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0,
    0x78602649,0x29131ab6,0x8edae952,0x0fc053c3,
    0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31};



/***************************************************/
__device__ __forceinline__
void luffa512_rnd(luffa512_state *state)
{
    int i,j;
    uint32_t t[40];
    uint32_t chainv[8];
    uint32_t tmp;

#pragma unroll 8
    for(i=0;i<8;i++) {
        t[i]=0;
#pragma unroll 5
        for(j=0;j<5;j++) {
            t[i] ^= state->chainv[i+8*j];
        }
    }

    LUFFA_MULT2(t, 0);

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            state->chainv[i+8*j] ^= t[i];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            t[i+8*j] = state->chainv[i+8*j];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
        LUFFA_MULT2(state->chainv, j);
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            state->chainv[8*j+i] ^= t[8*((j+1)%5)+i];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            t[i+8*j] = state->chainv[i+8*j];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
        LUFFA_MULT2(state->chainv, j);
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            state->chainv[8*j+i] ^= t[8*((j+4)%5)+i];
        }
    }

#pragma unroll 5
    for(j=0;j<5;j++) {
#pragma unroll 8
        for(i=0;i<8;i++) {
            state->chainv[i+8*j] ^= state->buffer[i];
        }
        LUFFA_MULT2(state->buffer, 0);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        chainv[i] = state->chainv[i];
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        LUFFA_STEP(c_luffa_CNS[(2*i)],c_luffa_CNS[(2*i)+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i] = chainv[i];
        chainv[i] = state->chainv[i+8];
    }

    LUFFA_TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],1);

#pragma unroll 8
    for(i=0;i<8;i++) {
        LUFFA_STEP(c_luffa_CNS[(2*i)+16],c_luffa_CNS[(2*i)+16+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i+8] = chainv[i];
        chainv[i] = state->chainv[i+16];
    }

    LUFFA_TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],2);

#pragma unroll 8
    for(i=0;i<8;i++) {
        LUFFA_STEP(c_luffa_CNS[(2*i)+32],c_luffa_CNS[(2*i)+32+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i+16] = chainv[i];
        chainv[i] = state->chainv[i+24];
    }

    LUFFA_TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],3);

#pragma unroll 8
    for(i=0;i<8;i++) {
        LUFFA_STEP(c_luffa_CNS[(2*i)+48],c_luffa_CNS[(2*i)+48+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i+24] = chainv[i];
        chainv[i] = state->chainv[i+32];
    }

    LUFFA_TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],4);

#pragma unroll 8
    for(i=0;i<8;i++) {
        LUFFA_STEP(c_luffa_CNS[(2*i)+64],c_luffa_CNS[(2*i)+64+1]);
    }

#pragma unroll 8
    for(i=0;i<8;i++) {
        state->chainv[i+32] = chainv[i];
    }
}


__device__ __forceinline__
void luffa512_update_64(luffa512_state *state, const BitSequence *data)
{
#pragma unroll 8
    for(int i=0;i<8;i++) state->buffer[i] = cuda_swab32(((uint32_t*)data)[i]);
    luffa512_rnd(state);

#pragma unroll 8
    for(int i=0;i<8;i++) state->buffer[i] = cuda_swab32(((uint32_t*)(data+32))[i]);
    luffa512_rnd(state);
}


/***************************************************/
__device__ __forceinline__
void luffa512_finalize(luffa512_state *state, uint32_t *b)
{
    int i,j;

    state->buffer[0] = 0x80000000;
#pragma unroll 7
    for(int i=1;i<8;i++) state->buffer[i] = 0;
    luffa512_rnd(state);

    /*---- blank round with m=0 ----*/
#pragma unroll 8
    for(i=0;i<8;i++) state->buffer[i] =0;
    luffa512_rnd(state);

#pragma unroll 8
    for(i=0;i<8;i++) {
        b[i] = 0;
#pragma unroll 5
        for(j=0;j<5;j++) {
            b[i] ^= state->chainv[i+8*j];
        }
        b[i] = cuda_swab32((b[i]));
    }

#pragma unroll 8
    for(i=0;i<8;i++) state->buffer[i]=0;
    luffa512_rnd(state);

#pragma unroll 8
    for(i=0;i<8;i++) {
        b[8+i] = 0;
#pragma unroll 5
        for(j=0;j<5;j++) {
            b[8+i] ^= state->chainv[i+8*j];
        }
        b[8 + i] = cuda_swab32((b[8 + i]));
    }
}


/***************************************************/

#undef LUFFA_MULT2
#undef LUFFA_TWEAK
#undef LUFFA_STEP
#undef LUFFA_SUBCRUMB
#undef LUFFA_MIXWORD
#undef LUFFA_ADD_CONSTANT
#undef LUFFA_LROT

/* Luffa-512 of a 64-byte input, in place, d_hash word order in and out. */
__device__ __forceinline__
void luffa512_hash_64(uint32_t *Hash)
{
	luffa512_state state;

	#pragma unroll 40
	for (int i = 0; i < 40; i++)
		state.chainv[i] = c_luffa_IV[i];
	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		state.buffer[i] = 0;

	luffa512_update_64(&state, (const BitSequence*)Hash);
	luffa512_finalize(&state, Hash);
}

#endif /* __CUDACC__ */

#endif /* CUDA_LUFFA512_DEVICE_CUH */
