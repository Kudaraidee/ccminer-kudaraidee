// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Shared Shabal-512 primitive library (device side) — x-family stage
 * function. Extracted bit-identically from algos/stages/cuda_shabal512.cu
 * (sph shabal.c lineage, the auto-generated permutation part). Constants
 * are statically initialized (per-TU header __constant__).
 *
 * Interface convention: Hash[16] holds the 64-byte value exactly as it
 * sits in the inter-stage d_hash buffer (little-endian uint32 words).
 */

#ifndef CUDA_SHABAL512_DEVICE_CUH
#define CUDA_SHABAL512_DEVICE_CUH

#include <stdint.h>
#include "cuda/helper.cuh"  // SPH_C32/SPH_T32/SPH_ROTL32

#ifdef __CUDACC__

#define SHABAL_sM    16

#define SHABAL_C32   SPH_C32
#define SHABAL_T32   SPH_T32

#define SHABAL_O1   13
#define SHABAL_O2    9
#define SHABAL_O3    6

/*
 * We copy the state into local variables, so that the compiler knows
 * that it can optimize them at will.
 */

/* BEGIN -- automatically generated code. */

#define SHABAL_INPUT_BLOCK_ADD   do { \
		B0 = SHABAL_T32(B0 + M0); \
		B1 = SHABAL_T32(B1 + M1); \
		B2 = SHABAL_T32(B2 + M2); \
		B3 = SHABAL_T32(B3 + M3); \
		B4 = SHABAL_T32(B4 + M4); \
		B5 = SHABAL_T32(B5 + M5); \
		B6 = SHABAL_T32(B6 + M6); \
		B7 = SHABAL_T32(B7 + M7); \
		B8 = SHABAL_T32(B8 + M8); \
		B9 = SHABAL_T32(B9 + M9); \
		BA = SHABAL_T32(BA + MA); \
		BB = SHABAL_T32(BB + MB); \
		BC = SHABAL_T32(BC + MC); \
		BD = SHABAL_T32(BD + MD); \
		BE = SHABAL_T32(BE + ME); \
		BF = SHABAL_T32(BF + MF); \
	} while (0)

#define SHABAL_INPUT_BLOCK_SUB   do { \
		C0 = SHABAL_T32(C0 - M0); \
		C1 = SHABAL_T32(C1 - M1); \
		C2 = SHABAL_T32(C2 - M2); \
		C3 = SHABAL_T32(C3 - M3); \
		C4 = SHABAL_T32(C4 - M4); \
		C5 = SHABAL_T32(C5 - M5); \
		C6 = SHABAL_T32(C6 - M6); \
		C7 = SHABAL_T32(C7 - M7); \
		C8 = SHABAL_T32(C8 - M8); \
		C9 = SHABAL_T32(C9 - M9); \
		CA = SHABAL_T32(CA - MA); \
		CB = SHABAL_T32(CB - MB); \
		CC = SHABAL_T32(CC - MC); \
		CD = SHABAL_T32(CD - MD); \
		CE = SHABAL_T32(CE - ME); \
		CF = SHABAL_T32(CF - MF); \
	} while (0)

#define SHABAL_XOR_W   do { \
		A00 ^= Wlow; \
		A01 ^= Whigh; \
	} while (0)

#define SWAP(v1, v2)   do { \
		uint32_t tmp = (v1); \
		(v1) = (v2); \
		(v2) = tmp; \
	} while (0)

#define SHABAL_SWAP_BC   do { \
		SWAP(B0, C0); \
		SWAP(B1, C1); \
		SWAP(B2, C2); \
		SWAP(B3, C3); \
		SWAP(B4, C4); \
		SWAP(B5, C5); \
		SWAP(B6, C6); \
		SWAP(B7, C7); \
		SWAP(B8, C8); \
		SWAP(B9, C9); \
		SWAP(BA, CA); \
		SWAP(BB, CB); \
		SWAP(BC, CC); \
		SWAP(BD, CD); \
		SWAP(BE, CE); \
		SWAP(BF, CF); \
	} while (0)

#define SHABAL_PERM_ELT(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)   do { \
		xa0 = SHABAL_T32((xa0 \
			^ (ROTL32(xa1, 15) * 5U) \
			^ xc) * 3U) \
			^ xb1 ^ (xb2 & ~xb3) ^ xm; \
		xb0 = SHABAL_T32(~(ROTL32(xb0, 1) ^ xa0)); \
	} while (0)

#define SHABAL_PERM_STEP_0   do { \
		SHABAL_PERM_ELT(A00, A0B, B0, BD, B9, B6, C8, M0); \
		SHABAL_PERM_ELT(A01, A00, B1, BE, BA, B7, C7, M1); \
		SHABAL_PERM_ELT(A02, A01, B2, BF, BB, B8, C6, M2); \
		SHABAL_PERM_ELT(A03, A02, B3, B0, BC, B9, C5, M3); \
		SHABAL_PERM_ELT(A04, A03, B4, B1, BD, BA, C4, M4); \
		SHABAL_PERM_ELT(A05, A04, B5, B2, BE, BB, C3, M5); \
		SHABAL_PERM_ELT(A06, A05, B6, B3, BF, BC, C2, M6); \
		SHABAL_PERM_ELT(A07, A06, B7, B4, B0, BD, C1, M7); \
		SHABAL_PERM_ELT(A08, A07, B8, B5, B1, BE, C0, M8); \
		SHABAL_PERM_ELT(A09, A08, B9, B6, B2, BF, CF, M9); \
		SHABAL_PERM_ELT(A0A, A09, BA, B7, B3, B0, CE, MA); \
		SHABAL_PERM_ELT(A0B, A0A, BB, B8, B4, B1, CD, MB); \
		SHABAL_PERM_ELT(A00, A0B, BC, B9, B5, B2, CC, MC); \
		SHABAL_PERM_ELT(A01, A00, BD, BA, B6, B3, CB, MD); \
		SHABAL_PERM_ELT(A02, A01, BE, BB, B7, B4, CA, ME); \
		SHABAL_PERM_ELT(A03, A02, BF, BC, B8, B5, C9, MF); \
	} while (0)

#define SHABAL_PERM_STEP_1   do { \
		SHABAL_PERM_ELT(A04, A03, B0, BD, B9, B6, C8, M0); \
		SHABAL_PERM_ELT(A05, A04, B1, BE, BA, B7, C7, M1); \
		SHABAL_PERM_ELT(A06, A05, B2, BF, BB, B8, C6, M2); \
		SHABAL_PERM_ELT(A07, A06, B3, B0, BC, B9, C5, M3); \
		SHABAL_PERM_ELT(A08, A07, B4, B1, BD, BA, C4, M4); \
		SHABAL_PERM_ELT(A09, A08, B5, B2, BE, BB, C3, M5); \
		SHABAL_PERM_ELT(A0A, A09, B6, B3, BF, BC, C2, M6); \
		SHABAL_PERM_ELT(A0B, A0A, B7, B4, B0, BD, C1, M7); \
		SHABAL_PERM_ELT(A00, A0B, B8, B5, B1, BE, C0, M8); \
		SHABAL_PERM_ELT(A01, A00, B9, B6, B2, BF, CF, M9); \
		SHABAL_PERM_ELT(A02, A01, BA, B7, B3, B0, CE, MA); \
		SHABAL_PERM_ELT(A03, A02, BB, B8, B4, B1, CD, MB); \
		SHABAL_PERM_ELT(A04, A03, BC, B9, B5, B2, CC, MC); \
		SHABAL_PERM_ELT(A05, A04, BD, BA, B6, B3, CB, MD); \
		SHABAL_PERM_ELT(A06, A05, BE, BB, B7, B4, CA, ME); \
		SHABAL_PERM_ELT(A07, A06, BF, BC, B8, B5, C9, MF); \
	} while (0)

#define SHABAL_PERM_STEP_2   do { \
		SHABAL_PERM_ELT(A08, A07, B0, BD, B9, B6, C8, M0); \
		SHABAL_PERM_ELT(A09, A08, B1, BE, BA, B7, C7, M1); \
		SHABAL_PERM_ELT(A0A, A09, B2, BF, BB, B8, C6, M2); \
		SHABAL_PERM_ELT(A0B, A0A, B3, B0, BC, B9, C5, M3); \
		SHABAL_PERM_ELT(A00, A0B, B4, B1, BD, BA, C4, M4); \
		SHABAL_PERM_ELT(A01, A00, B5, B2, BE, BB, C3, M5); \
		SHABAL_PERM_ELT(A02, A01, B6, B3, BF, BC, C2, M6); \
		SHABAL_PERM_ELT(A03, A02, B7, B4, B0, BD, C1, M7); \
		SHABAL_PERM_ELT(A04, A03, B8, B5, B1, BE, C0, M8); \
		SHABAL_PERM_ELT(A05, A04, B9, B6, B2, BF, CF, M9); \
		SHABAL_PERM_ELT(A06, A05, BA, B7, B3, B0, CE, MA); \
		SHABAL_PERM_ELT(A07, A06, BB, B8, B4, B1, CD, MB); \
		SHABAL_PERM_ELT(A08, A07, BC, B9, B5, B2, CC, MC); \
		SHABAL_PERM_ELT(A09, A08, BD, BA, B6, B3, CB, MD); \
		SHABAL_PERM_ELT(A0A, A09, BE, BB, B7, B4, CA, ME); \
		SHABAL_PERM_ELT(A0B, A0A, BF, BC, B8, B5, C9, MF); \
	} while (0)

#define SHABAL_APPLY_P   do { \
		B0 = SHABAL_T32(B0 << 17) | (B0 >> 15); \
		B1 = SHABAL_T32(B1 << 17) | (B1 >> 15); \
		B2 = SHABAL_T32(B2 << 17) | (B2 >> 15); \
		B3 = SHABAL_T32(B3 << 17) | (B3 >> 15); \
		B4 = SHABAL_T32(B4 << 17) | (B4 >> 15); \
		B5 = SHABAL_T32(B5 << 17) | (B5 >> 15); \
		B6 = SHABAL_T32(B6 << 17) | (B6 >> 15); \
		B7 = SHABAL_T32(B7 << 17) | (B7 >> 15); \
		B8 = SHABAL_T32(B8 << 17) | (B8 >> 15); \
		B9 = SHABAL_T32(B9 << 17) | (B9 >> 15); \
		BA = SHABAL_T32(BA << 17) | (BA >> 15); \
		BB = SHABAL_T32(BB << 17) | (BB >> 15); \
		BC = SHABAL_T32(BC << 17) | (BC >> 15); \
		BD = SHABAL_T32(BD << 17) | (BD >> 15); \
		BE = SHABAL_T32(BE << 17) | (BE >> 15); \
		BF = SHABAL_T32(BF << 17) | (BF >> 15); \
		SHABAL_PERM_STEP_0; \
		SHABAL_PERM_STEP_1; \
		SHABAL_PERM_STEP_2; \
		A0B = SHABAL_T32(A0B + C6); \
		A0A = SHABAL_T32(A0A + C5); \
		A09 = SHABAL_T32(A09 + C4); \
		A08 = SHABAL_T32(A08 + C3); \
		A07 = SHABAL_T32(A07 + C2); \
		A06 = SHABAL_T32(A06 + C1); \
		A05 = SHABAL_T32(A05 + C0); \
		A04 = SHABAL_T32(A04 + CF); \
		A03 = SHABAL_T32(A03 + CE); \
		A02 = SHABAL_T32(A02 + CD); \
		A01 = SHABAL_T32(A01 + CC); \
		A00 = SHABAL_T32(A00 + CB); \
		A0B = SHABAL_T32(A0B + CA); \
		A0A = SHABAL_T32(A0A + C9); \
		A09 = SHABAL_T32(A09 + C8); \
		A08 = SHABAL_T32(A08 + C7); \
		A07 = SHABAL_T32(A07 + C6); \
		A06 = SHABAL_T32(A06 + C5); \
		A05 = SHABAL_T32(A05 + C4); \
		A04 = SHABAL_T32(A04 + C3); \
		A03 = SHABAL_T32(A03 + C2); \
		A02 = SHABAL_T32(A02 + C1); \
		A01 = SHABAL_T32(A01 + C0); \
		A00 = SHABAL_T32(A00 + CF); \
		A0B = SHABAL_T32(A0B + CE); \
		A0A = SHABAL_T32(A0A + CD); \
		A09 = SHABAL_T32(A09 + CC); \
		A08 = SHABAL_T32(A08 + CB); \
		A07 = SHABAL_T32(A07 + CA); \
		A06 = SHABAL_T32(A06 + C9); \
		A05 = SHABAL_T32(A05 + C8); \
		A04 = SHABAL_T32(A04 + C7); \
		A03 = SHABAL_T32(A03 + C6); \
		A02 = SHABAL_T32(A02 + C5); \
		A01 = SHABAL_T32(A01 + C4); \
		A00 = SHABAL_T32(A00 + C3); \
	} while (0)

#define SHABAL_INCR_W   do { \
		if ((Wlow = SHABAL_T32(Wlow + 1)) == 0) \
			Whigh = SHABAL_T32(Whigh + 1); \
	} while (0)


static __constant__
uint32_t c_shabal_A512[] = {
	SHABAL_C32(0x20728DFD), SHABAL_C32(0x46C0BD53), SHABAL_C32(0xE782B699), SHABAL_C32(0x55304632),
	SHABAL_C32(0x71B4EF90), SHABAL_C32(0x0EA9E82C), SHABAL_C32(0xDBB930F1), SHABAL_C32(0xFAD06B8B),
	SHABAL_C32(0xBE0CAE40), SHABAL_C32(0x8BD14410), SHABAL_C32(0x76D2ADAC), SHABAL_C32(0x28ACAB7F)
};

static __constant__
uint32_t c_shabal_B512[] = {
	SHABAL_C32(0xC1099CB7), SHABAL_C32(0x07B385F3), SHABAL_C32(0xE7442C26), SHABAL_C32(0xCC8AD640),
	SHABAL_C32(0xEB6F56C7), SHABAL_C32(0x1EA81AA9), SHABAL_C32(0x73B9D314), SHABAL_C32(0x1DE85D08),
	SHABAL_C32(0x48910A5A), SHABAL_C32(0x893B22DB), SHABAL_C32(0xC5A0DF44), SHABAL_C32(0xBBC4324E),
	SHABAL_C32(0x72D2F240), SHABAL_C32(0x75941D99), SHABAL_C32(0x6D8BDE82), SHABAL_C32(0xA1A7502B)
};

static __constant__
uint32_t c_shabal_C512[] = {
	SHABAL_C32(0xD9BF68D1), SHABAL_C32(0x58BAD750), SHABAL_C32(0x56028CB2), SHABAL_C32(0x8134F359),
	SHABAL_C32(0xB5D469D8), SHABAL_C32(0x941A8CC2), SHABAL_C32(0x418B2A6E), SHABAL_C32(0x04052780),
	SHABAL_C32(0x7F07D787), SHABAL_C32(0x5194358F), SHABAL_C32(0x3C60D665), SHABAL_C32(0xBE97D79A),
	SHABAL_C32(0x950C3434), SHABAL_C32(0xAED9A06D), SHABAL_C32(0x2537DC8D), SHABAL_C32(0x7CDB5969)
};

/***************************************************/

/* Shabal-512 of a 64-byte input, in place, d_hash word order in and out —
 * the donor shabal512_gpu_hash_64 kernel's interior, verbatim. */
__device__ __forceinline__
void shabal512_hash_64(uint32_t *Hash)
{
		uint32_t A00 = c_shabal_A512[0], A01 = c_shabal_A512[1], A02 = c_shabal_A512[2], A03 = c_shabal_A512[3],
			A04 = c_shabal_A512[4], A05 = c_shabal_A512[5], A06 = c_shabal_A512[6], A07 = c_shabal_A512[7],
			A08 = c_shabal_A512[8], A09 = c_shabal_A512[9], A0A = c_shabal_A512[10], A0B = c_shabal_A512[11];
		uint32_t B0 = c_shabal_B512[0], B1 = c_shabal_B512[1], B2 = c_shabal_B512[2], B3 = c_shabal_B512[3],
			B4 = c_shabal_B512[4], B5 = c_shabal_B512[5], B6 = c_shabal_B512[6], B7 = c_shabal_B512[7],
			B8 = c_shabal_B512[8], B9 = c_shabal_B512[9], BA = c_shabal_B512[10], BB = c_shabal_B512[11],
			BC = c_shabal_B512[12], BD = c_shabal_B512[13], BE = c_shabal_B512[14], BF = c_shabal_B512[15];
		uint32_t C0 = c_shabal_C512[0], C1 = c_shabal_C512[1], C2 = c_shabal_C512[2], C3 = c_shabal_C512[3],
			C4 = c_shabal_C512[4], C5 = c_shabal_C512[5], C6 = c_shabal_C512[6], C7 = c_shabal_C512[7],
			C8 = c_shabal_C512[8], C9 = c_shabal_C512[9], CA = c_shabal_C512[10], CB = c_shabal_C512[11],
			CC = c_shabal_C512[12], CD = c_shabal_C512[13], CE = c_shabal_C512[14], CF = c_shabal_C512[15];
		uint32_t M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;
		uint32_t Wlow = 1, Whigh = 0;

		M0 = Hash[0];
		M1 = Hash[1];
		M2 = Hash[2];
		M3 = Hash[3];
		M4 = Hash[4];
		M5 = Hash[5];
		M6 = Hash[6];
		M7 = Hash[7];

		M8 = Hash[8];
		M9 = Hash[9];
		MA = Hash[10];
		MB = Hash[11];
		MC = Hash[12];
		MD = Hash[13];
		ME = Hash[14];
		MF = Hash[15];

		SHABAL_INPUT_BLOCK_ADD;
		SHABAL_XOR_W;
		SHABAL_APPLY_P;
		SHABAL_INPUT_BLOCK_SUB;
		SHABAL_SWAP_BC;
		SHABAL_INCR_W;

		M0 = 0x80;
		M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

		SHABAL_INPUT_BLOCK_ADD;
		SHABAL_XOR_W;
		SHABAL_APPLY_P;

		for (uint8_t i = 0; i < 3; i ++)
		{
			SHABAL_SWAP_BC;
			SHABAL_XOR_W;
			SHABAL_APPLY_P;
		}

		Hash[0] = B0;
		Hash[1] = B1;
		Hash[2] = B2;
		Hash[3] = B3;
		Hash[4] = B4;
		Hash[5] = B5;
		Hash[6] = B6;
		Hash[7] = B7;

		Hash[8] = B8;
		Hash[9] = B9;
		Hash[10] = BA;
		Hash[11] = BB;
		Hash[12] = BC;
		Hash[13] = BD;
		Hash[14] = BE;
		Hash[15] = BF;
}

/* Shabal-512 of an 80-byte header, digest OUT in Hash[0..15]. msg points at
 * the 20-word padded block constant: words 0..15 are the first message block,
 * words 16..18 are the head of the second block (word 3 of that block is the
 * nonce, word 4 is the 0x80 pad). The x16-family 80-byte first-stage path. */
__device__ __forceinline__
void shabal512_hash_80(const uint32_t *msg, const uint32_t nonce, uint32_t *Hash)
{
		uint32_t A00 = c_shabal_A512[0], A01 = c_shabal_A512[1], A02 = c_shabal_A512[2], A03 = c_shabal_A512[3],
			A04 = c_shabal_A512[4], A05 = c_shabal_A512[5], A06 = c_shabal_A512[6], A07 = c_shabal_A512[7],
			A08 = c_shabal_A512[8], A09 = c_shabal_A512[9], A0A = c_shabal_A512[10], A0B = c_shabal_A512[11];
		uint32_t B0 = c_shabal_B512[0], B1 = c_shabal_B512[1], B2 = c_shabal_B512[2], B3 = c_shabal_B512[3],
			B4 = c_shabal_B512[4], B5 = c_shabal_B512[5], B6 = c_shabal_B512[6], B7 = c_shabal_B512[7],
			B8 = c_shabal_B512[8], B9 = c_shabal_B512[9], BA = c_shabal_B512[10], BB = c_shabal_B512[11],
			BC = c_shabal_B512[12], BD = c_shabal_B512[13], BE = c_shabal_B512[14], BF = c_shabal_B512[15];
		uint32_t C0 = c_shabal_C512[0], C1 = c_shabal_C512[1], C2 = c_shabal_C512[2], C3 = c_shabal_C512[3],
			C4 = c_shabal_C512[4], C5 = c_shabal_C512[5], C6 = c_shabal_C512[6], C7 = c_shabal_C512[7],
			C8 = c_shabal_C512[8], C9 = c_shabal_C512[9], CA = c_shabal_C512[10], CB = c_shabal_C512[11],
			CC = c_shabal_C512[12], CD = c_shabal_C512[13], CE = c_shabal_C512[14], CF = c_shabal_C512[15];
		uint32_t M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;
		uint32_t Wlow = 1, Whigh = 0;

		M0 = msg[0];
		M1 = msg[1];
		M2 = msg[2];
		M3 = msg[3];
		M4 = msg[4];
		M5 = msg[5];
		M6 = msg[6];
		M7 = msg[7];
		M8 = msg[8];
		M9 = msg[9];
		MA = msg[10];
		MB = msg[11];
		MC = msg[12];
		MD = msg[13];
		ME = msg[14];
		MF = msg[15];

		SHABAL_INPUT_BLOCK_ADD;
		SHABAL_XOR_W;
		SHABAL_APPLY_P;
		SHABAL_INPUT_BLOCK_SUB;
		SHABAL_SWAP_BC;
		SHABAL_INCR_W;

		M0 = msg[16];
		M1 = msg[17];
		M2 = msg[18];
		M3 = cuda_swab32(nonce);
		M4 = 0x80;
		M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

		SHABAL_INPUT_BLOCK_ADD;
		SHABAL_XOR_W;
		SHABAL_APPLY_P;

		for (unsigned i = 0; i < 3; i++) {
			SHABAL_SWAP_BC;
			SHABAL_XOR_W;
			SHABAL_APPLY_P;
		}

		Hash[0] = B0;
		Hash[1] = B1;
		Hash[2] = B2;
		Hash[3] = B3;
		Hash[4] = B4;
		Hash[5] = B5;
		Hash[6] = B6;
		Hash[7] = B7;
		Hash[8] = B8;
		Hash[9] = B9;
		Hash[10] = BA;
		Hash[11] = BB;
		Hash[12] = BC;
		Hash[13] = BD;
		Hash[14] = BE;
		Hash[15] = BF;
}

#undef SHABAL_sM
#undef SHABAL_C32
#undef SHABAL_T32
#undef SHABAL_O1
#undef SHABAL_O2
#undef SHABAL_O3
#undef SHABAL_INPUT_BLOCK_ADD
#undef SHABAL_INPUT_BLOCK_SUB
#undef SHABAL_XOR_W
#undef SHABAL_APPLY_P
#undef SHABAL_PERM_ELT
#undef SHABAL_PERM_STEP_0
#undef SHABAL_PERM_STEP_1
#undef SHABAL_PERM_STEP_2
#undef SHABAL_INCR_W
#undef SHABAL_SWAP_BC

#endif /* __CUDACC__ */

#endif /* CUDA_SHABAL512_DEVICE_CUH */
