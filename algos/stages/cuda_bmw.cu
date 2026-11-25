/**
 * BMW-256 CUDA Implementation - tpruvot 2015
 *
 * Not optimal but close to the sph version and easier to adapt.
 */

#include <stdio.h>
#include <memory.h>

extern "C" {
#include "sph/sph_bmw.h"
}

#include "cuda/helper.cuh"
#include "cuda/candidate_report.cuh"

__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

__constant__ static sph_u32 final_s[16] = {
	0xaaaaaaa0, 0xaaaaaaa1, 0xaaaaaaa2, 0xaaaaaaa3,
	0xaaaaaaa4, 0xaaaaaaa5, 0xaaaaaaa6, 0xaaaaaaa7,
	0xaaaaaaa8, 0xaaaaaaa9, 0xaaaaaaaa, 0xaaaaaaab,
	0xaaaaaaac, 0xaaaaaaad, 0xaaaaaaae, 0xaaaaaaaf
};

#define I16_16    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
#define I16_17    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
#define I16_18    2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17
#define I16_19    3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18
#define I16_20    4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
#define I16_21    5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#define I16_22    6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
#define I16_23    7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
#define I16_24    8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
#define I16_25    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
#define I16_26   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
#define I16_27   11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
#define I16_28   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
#define I16_29   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
#define I16_30   14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
#define I16_31   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30

#define M16_18    2,  3,  5,  6,  9, 12, 13
#define M16_19    3,  4,  6,  7, 10, 13, 14
#define M16_20    4,  5,  7,  8, 11, 14, 15
#define M16_21    5,  6,  8,  9, 12, 15, 16
#define M16_22    6,  7,  9, 10, 13,  0,  1
#define M16_23    7,  8, 10, 11, 14,  1,  2
#define M16_24    8,  9, 11, 12, 15,  2,  3
#define M16_25    9, 10, 12, 13,  0,  3,  4
#define M16_26   10, 11, 13, 14,  1,  4,  5
#define M16_27   11, 12, 14, 15,  2,  5,  6
#define M16_28   12, 13, 15, 16,  3,  6,  7
#define M16_29   13, 14,  0,  1,  4,  7,  8
#define M16_30   14, 15,  1,  2,  5,  8,  9
#define M16_31   15, 16,  2,  3,  6,  9, 10

#define ss0(x)    (((x) >> 1) ^ ((x) << 3) ^ ROTL32(x,  4) ^ ROTL32(x, 19))
#define ss1(x)    (((x) >> 1) ^ ((x) << 2) ^ ROTL32(x,  8) ^ ROTL32(x, 23))
#define ss2(x)    (((x) >> 2) ^ ((x) << 1) ^ ROTL32(x, 12) ^ ROTL32(x, 25))
#define ss3(x)    (((x) >> 2) ^ ((x) << 2) ^ ROTL32(x, 15) ^ ROTL32(x, 29))
#define ss4(x)    (((x) >> 1) ^ (x))
#define ss5(x)    (((x) >> 2) ^ (x))

#define rs1(x)    ROTL32(x,  3)
#define rs2(x)    ROTL32(x,  7)
#define rs3(x)    ROTL32(x, 13)
#define rs4(x)    ROTL32(x, 16)
#define rs5(x)    ROTL32(x, 19)
#define rs6(x)    ROTL32(x, 23)
#define rs7(x)    ROTL32(x, 27)


__device__
static void gpu_compress_small(const sph_u32 *data, const sph_u32 h[16], sph_u32 dh[16])
{
		// FOLD MAKE_Qas;

		sph_u32 dx[16];
		for (int i=0; i<16; i++)
			dx[i] = data[i] ^ h[i];

		sph_u32 qt[32];
		qt[ 0] = dx[ 5] - dx[7] + dx[10] + dx[13] + dx[14]; // Ws0
		qt[ 1] = dx[ 6] - dx[8] + dx[11] + dx[14] - dx[15]; // Ws1
		qt[ 2] = dx[ 0] + dx[7] + dx[ 9] - dx[12] + dx[15]; // Ws2
		qt[ 3] = dx[ 0] - dx[1] + dx[ 8] - dx[10] + dx[13]; // Ws3
		qt[ 4] = dx[ 1] + dx[2] + dx[ 9] - dx[11] - dx[14]; // Ws4;
		qt[ 5] = dx[ 3] - dx[2] + dx[10] - dx[12] + dx[15]; // Ws5;
		qt[ 6] = dx[ 4] - dx[0] - dx[ 3] - dx[11] + dx[13]; // Ws6;
		qt[ 7] = dx[ 1] - dx[4] - dx[ 5] - dx[12] - dx[14]; // Ws7;
		qt[ 8] = dx[ 2] - dx[5] - dx[ 6] + dx[13] - dx[15]; // Ws8;
		qt[ 9] = dx[ 0] - dx[3] + dx[ 6] - dx[ 7] + dx[14]; // Ws9;
		qt[10] = dx[ 8] - dx[1] - dx[ 4] - dx[ 7] + dx[15]; // Ws10;
		qt[11] = dx[ 8] - dx[0] - dx[ 2] - dx[ 5] + dx[ 9]; // Ws11;
		qt[12] = dx[ 1] + dx[3] - dx[ 6] - dx[ 9] + dx[10]; // Ws12;
		qt[13] = dx[ 2] + dx[4] + dx[ 7] + dx[10] + dx[11]; // Ws13;
		qt[14] = dx[ 3] - dx[5] + dx[ 8] - dx[11] - dx[12]; // Ws14;
		qt[15] = dx[12] - dx[4] - dx[ 6] - dx[ 9] + dx[13]; // Ws15;

		qt[ 0] = ss0(qt[ 0]) + h[ 1];
		qt[ 1] = ss1(qt[ 1]) + h[ 2];
		qt[ 2] = ss2(qt[ 2]) + h[ 3];
		qt[ 3] = ss3(qt[ 3]) + h[ 4];
		qt[ 4] = ss4(qt[ 4]) + h[ 5];

		qt[ 5] = ss0(qt[ 5]) + h[ 6];
		qt[ 6] = ss1(qt[ 6]) + h[ 7];
		qt[ 7] = ss2(qt[ 7]) + h[ 8];
		qt[ 8] = ss3(qt[ 8]) + h[ 9];
		qt[ 9] = ss4(qt[ 9]) + h[10];

		qt[10] = ss0(qt[10]) + h[11];
		qt[11] = ss1(qt[11]) + h[12];
		qt[12] = ss2(qt[12]) + h[13];
		qt[13] = ss3(qt[13]) + h[14];
		qt[14] = ss4(qt[14]) + h[15];

		qt[15] = ss0(qt[15]) + h[ 0];

		//MAKE_Qbs;
		#define Ks(j)   ((sph_u32)(0x05555555UL * j))
		#define Qs(j)   (qt[j])

		#define expand1s_in(i16, \
				i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, \
				i0m, i1m, i3m, i4m, i7m, i10m, i11m) \
			(ss1(qt[i0]) + ss2(qt[i1]) + ss3(qt[i2]) + ss0(qt[i3]) + ss1(qt[i4]) + ss2(qt[i5]) + ss3(qt[i6]) + ss0(qt[i7]) \
				+ ss1(qt[i8]) + ss2(qt[i9]) + ss3(qt[i10]) + ss0(qt[i11]) + ss1(qt[i12]) + ss2(qt[i13]) + ss3(qt[i14]) + ss0(qt[i15]) \
				+ ((ROTL32(data[i0m], i1m) + ROTL32(data[i3m], i4m)  - ROTL32(data[i10m], i11m) + Ks(i16)) ^ h[i7m]))

		qt[16] = expand1s_in(16,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0, 1, 3, 4, 7, 10, 11);
		qt[17] = expand1s_in(17,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  1, 2, 4, 5, 8, 11, 12);

		#define expand2s_inner(qf, i16, \
				i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, \
				i0m, i1m, i3m, i4m, i7m, i10m, i11m) \
			(qf(i0) + rs1(qf(i1)) + qf(i2) + rs2(qf(i3)) \
				+ qf(i4) + rs3(qf(i5)) + qf(i6) + rs4(qf(i7)) + qf(i8) + rs5(qf(i9)) + qf(i10) + rs6(qf(i11)) \
				+ qf(i12) + rs7(qf(i13)) + ss4(qf(i14)) + ss5(qf(i15)) \
				+ ((ROTL32(data[i0m], i1m) + ROTL32(data[i3m], i4m) - ROTL32(data[i10m], i11m) + Ks(i16)) ^ h[i7m]))

#ifdef _MSC_VER
		#define LPAR   (
		#define expand2s(i16) \
			expand2s_(Qs, i16, I16_ ## i16, M16_ ## i16)
		#define expand2s_(qf, i16, ix, iy) \
			expand2s_inner LPAR qf, i16, ix, iy)
#else
		#define expand2s_(i16, ix, iy) \
			expand2s_inner(Qs, i16, ix, iy)
		#define expand2s(i16) \
			expand2s_(i16, I16_ ## i16, M16_ ## i16)
#endif

		qt[18] = expand2s(18);
		qt[19] = expand2s(19);
		qt[20] = expand2s(20);
		qt[21] = expand2s(21);
		qt[22] = expand2s(22);
		qt[23] = expand2s(23);
		qt[24] = expand2s(24);
		qt[25] = expand2s(25);
		qt[26] = expand2s(26);
		qt[27] = expand2s(27);
		qt[28] = expand2s(28);
		qt[29] = expand2s(29);
		qt[30] = expand2s(30);
		qt[31] = expand2s(31);

		sph_u32 xl, xh;
		xl = Qs(16) ^ Qs(17) ^ Qs(18) ^ Qs(19) ^ Qs(20) ^ Qs(21) ^ Qs(22) ^ Qs(23);

		xh = xl ^ Qs(24) ^ Qs(25) ^ Qs(26) ^ Qs(27)	^ Qs(28) ^ Qs(29) ^ Qs(30) ^ Qs(31);

		dh[ 0] = ((xh <<  5) ^ (Qs(16) >>  5) ^ data[ 0]) + (xl ^ Qs(24) ^ Qs(0));
		dh[ 1] = ((xh >>  7) ^ (Qs(17) <<  8) ^ data[ 1]) + (xl ^ Qs(25) ^ Qs(1));
		dh[ 2] = ((xh >>  5) ^ (Qs(18) <<  5) ^ data[ 2]) + (xl ^ Qs(26) ^ Qs(2));
		dh[ 3] = ((xh >>  1) ^ (Qs(19) <<  5) ^ data[ 3]) + (xl ^ Qs(27) ^ Qs(3));
		dh[ 4] = ((xh >>  3) ^ (Qs(20) <<  0) ^ data[ 4]) + (xl ^ Qs(28) ^ Qs(4));
		dh[ 5] = ((xh <<  6) ^ (Qs(21) >>  6) ^ data[ 5]) + (xl ^ Qs(29) ^ Qs(5));
		dh[ 6] = ((xh >>  4) ^ (Qs(22) <<  6) ^ data[ 6]) + (xl ^ Qs(30) ^ Qs(6));
		dh[ 7] = ((xh >> 11) ^ (Qs(23) <<  2) ^ data[ 7]) + (xl ^ Qs(31) ^ Qs(7));

		dh[ 8] = ROTL32(dh[4],  9) + (xh ^ Qs(24) ^ data[ 8]) + ((xl << 8) ^ Qs(23) ^ Qs( 8));
		dh[ 9] = ROTL32(dh[5], 10) + (xh ^ Qs(25) ^ data[ 9]) + ((xl >> 6) ^ Qs(16) ^ Qs( 9));
		dh[10] = ROTL32(dh[6], 11) + (xh ^ Qs(26) ^ data[10]) + ((xl << 6) ^ Qs(17) ^ Qs(10));
		dh[11] = ROTL32(dh[7], 12) + (xh ^ Qs(27) ^ data[11]) + ((xl << 4) ^ Qs(18) ^ Qs(11));
		dh[12] = ROTL32(dh[0], 13) + (xh ^ Qs(28) ^ data[12]) + ((xl >> 3) ^ Qs(19) ^ Qs(12));
		dh[13] = ROTL32(dh[1], 14) + (xh ^ Qs(29) ^ data[13]) + ((xl >> 4) ^ Qs(20) ^ Qs(13));
		dh[14] = ROTL32(dh[2], 15) + (xh ^ Qs(30) ^ data[14]) + ((xl >> 7) ^ Qs(21) ^ Qs(14));
		dh[15] = ROTL32(dh[3], 16) + (xh ^ Qs(31) ^ data[15]) + ((xl >> 2) ^ Qs(22) ^ Qs(15));
}

/* sm_80+: move adds from the integer ALU pipe to the idle IMAD pipe. A multiply-add by a
 * constant-bank 1 cannot be folded back into IADD3. Rotate + add stays a plain add (one LEA.HI).
 * Older arches keep the original expressions. */
__constant__ uint32_t c_bmw256_one = 1, c_bmw256_minus1 = 0xffffffffu;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define BMW256_IMAD 1
__device__ __forceinline__ static uint32_t bmw_imad_add(const uint32_t x, const uint32_t y)
{
	uint32_t r;
	asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(y), "r"(c_bmw256_one), "r"(x));
	return r;
}
__device__ __forceinline__ static uint32_t bmw_imad_sub(const uint32_t x, const uint32_t y)
{
	uint32_t r;
	asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(y), "r"(c_bmw256_minus1), "r"(x));
	return r;
}
#define BMW_ADD(x, y) bmw_imad_add((x), (y))
#define BMW_SUB(x, y) bmw_imad_sub((x), (y))
#else
#define BMW_ADD(x, y) ((x) + (y))
#define BMW_SUB(x, y) ((x) - (y))
#endif

/* Block 2 of an 80-byte header with its nonce-free work done once per job on the host.
 * Only word 3 (the nonce) varies; the W_j that read it are j = 5, 6, 9, 12, 14, and the
 * AddElement terms that rotate it are j = 16, 19, 25.
 *   [ 0..15] q[j], or for those five j the partial sum without word 3
 *   [16..31] the constant part of q[16..31]
 *   [32..34] AddElement(16, 19, 25) without word 3, before the XOR; [35..37] its XOR words
 *   [38..43] H[3], then H[(j+1) % 16] for j = 5, 6, 9, 12, 14 */
__constant__ uint32_t c_bmw256_pre[44];

#define BMW256_PRE_D(i) ((i) == 5 || (i) == 6 || (i) == 9 || (i) == 12 || (i) == 14)

/* f0's W_j as (index, sign) - the BMW specification's table */
static const int8_t bmw256_w_idx[16][5] = {
	{ 5, 7, 10, 13, 14 }, { 6, 8, 11, 14, 15 }, { 0, 7, 9, 12, 15 }, { 0, 1, 8, 10, 13 },
	{ 1, 2, 9, 11, 14 }, { 3, 2, 10, 12, 15 }, { 4, 0, 3, 11, 13 }, { 1, 4, 5, 12, 14 },
	{ 2, 5, 6, 13, 15 }, { 0, 3, 6, 7, 14 }, { 8, 1, 4, 7, 15 }, { 8, 0, 2, 5, 9 },
	{ 1, 3, 6, 9, 10 }, { 2, 4, 7, 10, 11 }, { 3, 5, 8, 11, 12 }, { 12, 4, 6, 9, 13 }
};
static const int8_t bmw256_w_sgn[16][5] = {
	{ 1, -1, 1, 1, 1 }, { 1, -1, 1, 1, -1 }, { 1, 1, 1, -1, 1 }, { 1, -1, 1, -1, 1 },
	{ 1, 1, 1, -1, -1 }, { 1, -1, 1, -1, 1 }, { 1, -1, -1, -1, 1 }, { 1, -1, -1, -1, -1 },
	{ 1, -1, -1, 1, -1 }, { 1, -1, 1, -1, 1 }, { 1, -1, -1, -1, 1 }, { 1, -1, -1, -1, 1 },
	{ 1, 1, -1, -1, 1 }, { 1, 1, 1, 1, 1 }, { 1, -1, 1, -1, -1 }, { 1, -1, -1, -1, 1 }
};

static inline uint32_t bmw256_rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

/* s0..s5 of the specification, 32-bit variant (the ss0..ss5 macros above) */
static uint32_t bmw256_s(int k, uint32_t x)
{
	switch (k) {
	case 0: return (x >> 1) ^ (x << 3) ^ bmw256_rol(x, 4) ^ bmw256_rol(x, 19);
	case 1: return (x >> 1) ^ (x << 2) ^ bmw256_rol(x, 8) ^ bmw256_rol(x, 23);
	case 2: return (x >> 2) ^ (x << 1) ^ bmw256_rol(x, 12) ^ bmw256_rol(x, 25);
	case 3: return (x >> 2) ^ (x << 2) ^ bmw256_rol(x, 15) ^ bmw256_rol(x, 29);
	case 4: return (x >> 1) ^ x;
	default: return (x >> 2) ^ x;
	}
}

/* expansion term k of q[j] (k = 0..15 reads q[j - 16 + k]) */
static uint32_t bmw256_exp_term(int j, int k, uint32_t q)
{
	static const int r[7] = { 3, 7, 13, 16, 19, 23, 27 };
	if (j < 18) return bmw256_s((k + 1) % 4, q);  // expand1: s1 s2 s3 s0 ...
	if (k == 14) return bmw256_s(4, q);
	if (k == 15) return bmw256_s(5, q);
	return (k & 1) ? bmw256_rol(q, r[k / 2]) : q;  // expand2
}

/* AddElement(j) without its XOR, word 3 omitted when skip3 */
static uint32_t bmw256_add_element(int j, const uint32_t *m, bool skip3)
{
	const int a = j - 16, b = (j - 13) % 16, c = (j - 6) % 16;
	uint32_t v = 0x05555555u * (uint32_t) j;
	if (!(skip3 && a == 3)) v += bmw256_rol(m[a], a + 1);
	if (!(skip3 && b == 3)) v += bmw256_rol(m[b], b + 1);
	if (!(skip3 && c == 3)) v -= bmw256_rol(m[c], c + 1);
	return v;
}

/* m = block 2 with word 3 unset, H = the midstate */
static void bmw256_precompute(const uint32_t *m, const uint32_t *H, uint32_t *pre)
{
	uint32_t q[16];
	for (int j = 0; j < 16; j++) {
		uint32_t t = 0;
		for (int k = 0; k < 5; k++) {
			const int i = bmw256_w_idx[j][k];
			if (i == 3) continue;
			const uint32_t v = m[i] ^ H[i];
			t += (bmw256_w_sgn[j][k] > 0) ? v : (0u - v);
		}
		q[j] = BMW256_PRE_D(j) ? 0 : bmw256_s(j % 5, t) + H[(j + 1) % 16];
		pre[j] = BMW256_PRE_D(j) ? t : q[j];
	}
	int ae = 32;
	for (int j = 16; j < 32; j++) {
		uint32_t p = 0;
		for (int k = 0; k < 16; k++) {
			const int i = j - 16 + k;
			if (i < 16 && !BMW256_PRE_D(i))
				p += bmw256_exp_term(j, k, q[i]);
		}
		if (j == 16 || j == 19 || j == 25)
			pre[ae++] = bmw256_add_element(j, m, true);
		else
			p += bmw256_add_element(j, m, false) ^ H[(j - 9) % 16];
		pre[j] = p;
	}
	pre[35] = H[7]; pre[36] = H[10]; pre[37] = H[0];
	pre[38] = H[3];
	pre[39] = H[6]; pre[40] = H[7]; pre[41] = H[10]; pre[42] = H[13]; pre[43] = H[15];
}

__device__ __forceinline__
static uint32_t bmw256_dev_s(const int k, const uint32_t x)
{
	switch (k) {
	case 0: return ss0(x);
	case 1: return ss1(x);
	case 2: return ss2(x);
	case 3: return ss3(x);
	case 4: return ss4(x);
	default: return ss5(x);
	}
}

__device__ __forceinline__
static uint32_t bmw256_dev_exp_term(const int j, const int k, const uint32_t q)
{
	if (j < 18) return bmw256_dev_s((k + 1) % 4, q);
	if (k == 14) return ss4(q);
	if (k == 15) return ss5(q);
	if (!(k & 1)) return q;
	switch (k / 2) {
	case 0: return rs1(q);
	case 1: return rs2(q);
	case 2: return rs3(q);
	case 3: return rs4(q);
	case 4: return rs5(q);
	case 5: return rs6(q);
	default: return rs7(q);
	}
}

/* gpu_compress_small(data, midstate, dh) with c_bmw256_pre. */
__device__ __forceinline__
static void gpu_compress_small_80pre(const sph_u32 *data, sph_u32 dh[16])
{
	sph_u32 qt[32];
	const sph_u32 x3 = data[3] ^ c_bmw256_pre[38];

	#pragma unroll
	for (int j = 0; j < 16; j++) {
		if (!BMW256_PRE_D(j)) qt[j] = c_bmw256_pre[j];
	}
	qt[ 5] = BMW_ADD(ss0(BMW_ADD(c_bmw256_pre[ 5], x3)), c_bmw256_pre[39]);
	qt[ 6] = BMW_ADD(ss1(BMW_SUB(c_bmw256_pre[ 6], x3)), c_bmw256_pre[40]);
	qt[ 9] = BMW_ADD(ss4(BMW_SUB(c_bmw256_pre[ 9], x3)), c_bmw256_pre[41]);
	qt[12] = BMW_ADD(ss2(BMW_ADD(c_bmw256_pre[12], x3)), c_bmw256_pre[42]);
	qt[14] = BMW_ADD(ss4(BMW_ADD(c_bmw256_pre[14], x3)), c_bmw256_pre[43]);

	const sph_u32 r3 = ROTL32(data[3], 4);
	#pragma unroll
	for (int j = 16; j < 32; j++) {
		sph_u32 v = c_bmw256_pre[j];
		#pragma unroll
		for (int k = 0; k < 16; k++) {
			const int i = j - 16 + k;
			if (i >= 16 || BMW256_PRE_D(i)) {
				if (j >= 18 && (k & 1) && k < 14)
					v += bmw256_dev_exp_term(j, k, qt[i]);  // a rotate: LEA.HI
				else
					v = BMW_ADD(v, bmw256_dev_exp_term(j, k, qt[i]));
			}
		}
		if (j == 16) v = BMW_ADD(v, (c_bmw256_pre[32] + r3) ^ c_bmw256_pre[35]);
		if (j == 19) v = BMW_ADD(v, (c_bmw256_pre[33] + r3) ^ c_bmw256_pre[36]);
		if (j == 25) v = BMW_ADD(v, (c_bmw256_pre[34] - r3) ^ c_bmw256_pre[37]);
		qt[j] = v;
	}

	sph_u32 xl, xh;
	xl = Qs(16) ^ Qs(17) ^ Qs(18) ^ Qs(19) ^ Qs(20) ^ Qs(21) ^ Qs(22) ^ Qs(23);
	xh = xl ^ Qs(24) ^ Qs(25) ^ Qs(26) ^ Qs(27) ^ Qs(28) ^ Qs(29) ^ Qs(30) ^ Qs(31);

	dh[ 0] = ((xh <<  5) ^ (Qs(16) >>  5) ^ data[ 0]) + (xl ^ Qs(24) ^ Qs(0));
	dh[ 1] = ((xh >>  7) ^ (Qs(17) <<  8) ^ data[ 1]) + (xl ^ Qs(25) ^ Qs(1));
	dh[ 2] = ((xh >>  5) ^ (Qs(18) <<  5) ^ data[ 2]) + (xl ^ Qs(26) ^ Qs(2));
	dh[ 3] = ((xh >>  1) ^ (Qs(19) <<  5) ^ data[ 3]) + (xl ^ Qs(27) ^ Qs(3));
	dh[ 4] = ((xh >>  3) ^ (Qs(20) <<  0) ^ data[ 4]) + (xl ^ Qs(28) ^ Qs(4));
	dh[ 5] = ((xh <<  6) ^ (Qs(21) >>  6) ^ data[ 5]) + (xl ^ Qs(29) ^ Qs(5));
	dh[ 6] = ((xh >>  4) ^ (Qs(22) <<  6) ^ data[ 6]) + (xl ^ Qs(30) ^ Qs(6));
	dh[ 7] = ((xh >> 11) ^ (Qs(23) <<  2) ^ data[ 7]) + (xl ^ Qs(31) ^ Qs(7));

	dh[ 8] = BMW_ADD(ROTL32(dh[4],  9) + (xh ^ Qs(24) ^ data[ 8]), ((xl << 8) ^ Qs(23) ^ Qs( 8)));
	dh[ 9] = BMW_ADD(ROTL32(dh[5], 10) + (xh ^ Qs(25) ^ data[ 9]), ((xl >> 6) ^ Qs(16) ^ Qs( 9)));
	dh[10] = BMW_ADD(ROTL32(dh[6], 11) + (xh ^ Qs(26) ^ data[10]), ((xl << 6) ^ Qs(17) ^ Qs(10)));
	dh[11] = BMW_ADD(ROTL32(dh[7], 12) + (xh ^ Qs(27) ^ data[11]), ((xl << 4) ^ Qs(18) ^ Qs(11)));
	dh[12] = BMW_ADD(ROTL32(dh[0], 13) + (xh ^ Qs(28) ^ data[12]), ((xl >> 3) ^ Qs(19) ^ Qs(12)));
	dh[13] = BMW_ADD(ROTL32(dh[1], 14) + (xh ^ Qs(29) ^ data[13]), ((xl >> 4) ^ Qs(20) ^ Qs(13)));
	dh[14] = BMW_ADD(ROTL32(dh[2], 15) + (xh ^ Qs(30) ^ data[14]), ((xl >> 7) ^ Qs(21) ^ Qs(14)));
	dh[15] = BMW_ADD(ROTL32(dh[3], 16) + (xh ^ Qs(31) ^ data[15]), ((xl >> 2) ^ Qs(22) ^ Qs(15)));
}

#ifdef BMW256_IMAD
/* gpu_compress_small(data, h, dh) for sm_80+, same arithmetic with the non-rotate adds on IMAD. */
__device__ __forceinline__
static void gpu_compress_small_imad(const sph_u32 *data, const sph_u32 h[16], sph_u32 dh[16])
{
	sph_u32 dx[16];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		dx[i] = data[i] ^ h[i];

	sph_u32 qt[32];
	qt[ 0] = BMW_ADD(BMW_ADD(BMW_ADD(BMW_SUB(dx[ 5], dx[7]), dx[10]), dx[13]), dx[14]);
	qt[ 1] = BMW_SUB(BMW_ADD(BMW_ADD(BMW_SUB(dx[ 6], dx[8]), dx[11]), dx[14]), dx[15]);
	qt[ 2] = BMW_ADD(BMW_SUB(BMW_ADD(BMW_ADD(dx[ 0], dx[7]), dx[ 9]), dx[12]), dx[15]);
	qt[ 3] = BMW_ADD(BMW_SUB(BMW_ADD(BMW_SUB(dx[ 0], dx[1]), dx[ 8]), dx[10]), dx[13]);
	qt[ 4] = BMW_SUB(BMW_SUB(BMW_ADD(BMW_ADD(dx[ 1], dx[2]), dx[ 9]), dx[11]), dx[14]);
	qt[ 5] = BMW_ADD(BMW_SUB(BMW_ADD(BMW_SUB(dx[ 3], dx[2]), dx[10]), dx[12]), dx[15]);
	qt[ 6] = BMW_ADD(BMW_SUB(BMW_SUB(BMW_SUB(dx[ 4], dx[0]), dx[ 3]), dx[11]), dx[13]);
	qt[ 7] = BMW_SUB(BMW_SUB(BMW_SUB(BMW_SUB(dx[ 1], dx[4]), dx[ 5]), dx[12]), dx[14]);
	qt[ 8] = BMW_SUB(BMW_ADD(BMW_SUB(BMW_SUB(dx[ 2], dx[5]), dx[ 6]), dx[13]), dx[15]);
	qt[ 9] = BMW_ADD(BMW_SUB(BMW_ADD(BMW_SUB(dx[ 0], dx[3]), dx[ 6]), dx[ 7]), dx[14]);
	qt[10] = BMW_ADD(BMW_SUB(BMW_SUB(BMW_SUB(dx[ 8], dx[1]), dx[ 4]), dx[ 7]), dx[15]);
	qt[11] = BMW_ADD(BMW_SUB(BMW_SUB(BMW_SUB(dx[ 8], dx[0]), dx[ 2]), dx[ 5]), dx[ 9]);
	qt[12] = BMW_ADD(BMW_SUB(BMW_SUB(BMW_ADD(dx[ 1], dx[3]), dx[ 6]), dx[ 9]), dx[10]);
	qt[13] = BMW_ADD(BMW_ADD(BMW_ADD(BMW_ADD(dx[ 2], dx[4]), dx[ 7]), dx[10]), dx[11]);
	qt[14] = BMW_SUB(BMW_SUB(BMW_ADD(BMW_SUB(dx[ 3], dx[5]), dx[ 8]), dx[11]), dx[12]);
	qt[15] = BMW_ADD(BMW_SUB(BMW_SUB(BMW_SUB(dx[12], dx[4]), dx[ 6]), dx[ 9]), dx[13]);

	#pragma unroll
	for (int j = 0; j < 16; j++)
		qt[j] = BMW_ADD(bmw256_dev_s(j % 5, qt[j]), h[(j + 1) % 16]);

	#pragma unroll
	for (int j = 16; j < 32; j++) {
		const int a = j - 16, b = (j - 13) % 16, c = (j - 6) % 16;
		sph_u32 v = (ROTL32(data[a], a + 1) + ROTL32(data[b], b + 1) - ROTL32(data[c], c + 1)
			+ (sph_u32) (0x05555555UL * j)) ^ h[(j - 9) % 16];
		#pragma unroll
		for (int k = 0; k < 16; k++) {
			if (j >= 18 && (k & 1) && k < 14)
				v += bmw256_dev_exp_term(j, k, qt[j - 16 + k]);  // a rotate: LEA.HI
			else
				v = BMW_ADD(v, bmw256_dev_exp_term(j, k, qt[j - 16 + k]));
		}
		qt[j] = v;
	}

	sph_u32 xl, xh;
	xl = Qs(16) ^ Qs(17) ^ Qs(18) ^ Qs(19) ^ Qs(20) ^ Qs(21) ^ Qs(22) ^ Qs(23);
	xh = xl ^ Qs(24) ^ Qs(25) ^ Qs(26) ^ Qs(27) ^ Qs(28) ^ Qs(29) ^ Qs(30) ^ Qs(31);

	dh[ 0] = ((xh <<  5) ^ (Qs(16) >>  5) ^ data[ 0]) + (xl ^ Qs(24) ^ Qs(0));
	dh[ 1] = ((xh >>  7) ^ (Qs(17) <<  8) ^ data[ 1]) + (xl ^ Qs(25) ^ Qs(1));
	dh[ 2] = ((xh >>  5) ^ (Qs(18) <<  5) ^ data[ 2]) + (xl ^ Qs(26) ^ Qs(2));
	dh[ 3] = ((xh >>  1) ^ (Qs(19) <<  5) ^ data[ 3]) + (xl ^ Qs(27) ^ Qs(3));
	dh[ 4] = ((xh >>  3) ^ (Qs(20) <<  0) ^ data[ 4]) + (xl ^ Qs(28) ^ Qs(4));
	dh[ 5] = ((xh <<  6) ^ (Qs(21) >>  6) ^ data[ 5]) + (xl ^ Qs(29) ^ Qs(5));
	dh[ 6] = ((xh >>  4) ^ (Qs(22) <<  6) ^ data[ 6]) + (xl ^ Qs(30) ^ Qs(6));
	dh[ 7] = ((xh >> 11) ^ (Qs(23) <<  2) ^ data[ 7]) + (xl ^ Qs(31) ^ Qs(7));

	dh[ 8] = BMW_ADD(ROTL32(dh[4],  9) + (xh ^ Qs(24) ^ data[ 8]), ((xl << 8) ^ Qs(23) ^ Qs( 8)));
	dh[ 9] = BMW_ADD(ROTL32(dh[5], 10) + (xh ^ Qs(25) ^ data[ 9]), ((xl >> 6) ^ Qs(16) ^ Qs( 9)));
	dh[10] = BMW_ADD(ROTL32(dh[6], 11) + (xh ^ Qs(26) ^ data[10]), ((xl << 6) ^ Qs(17) ^ Qs(10)));
	dh[11] = BMW_ADD(ROTL32(dh[7], 12) + (xh ^ Qs(27) ^ data[11]), ((xl << 4) ^ Qs(18) ^ Qs(11)));
	dh[12] = BMW_ADD(ROTL32(dh[0], 13) + (xh ^ Qs(28) ^ data[12]), ((xl >> 3) ^ Qs(19) ^ Qs(12)));
	dh[13] = BMW_ADD(ROTL32(dh[1], 14) + (xh ^ Qs(29) ^ data[13]), ((xl >> 4) ^ Qs(20) ^ Qs(13)));
	dh[14] = BMW_ADD(ROTL32(dh[2], 15) + (xh ^ Qs(30) ^ data[14]), ((xl >> 7) ^ Qs(21) ^ Qs(14)));
	dh[15] = BMW_ADD(ROTL32(dh[3], 16) + (xh ^ Qs(31) ^ data[15]), ((xl >> 2) ^ Qs(22) ^ Qs(15)));
}
#endif

/* Digest bytes 24..31 (vhash[7]:vhash[6]) for one nonce; mining and checksum kernels share it. */
__device__ __forceinline__
static uint64_t bmw256_80_high64(const uint32_t nonce)
{
	sph_u32 m[16];
	const uint2 w16 = vectorize(c_PaddedMessage80[8]);
	m[0] = w16.x; m[1] = w16.y;
	m[2] = (sph_u32) c_PaddedMessage80[9];
	m[3] = cuda_swab32(nonce);
	m[4] = 0x80;  // padding of an 80-byte message: 0x80, zeros, then the bit count 640
	#pragma unroll
	for (int i = 5; i < 14; i++) m[i] = 0;
	m[14] = 640; m[15] = 0;

	sph_u32 h2[16], h1[16];
	gpu_compress_small_80pre(m, h2);
#ifdef BMW256_IMAD
	gpu_compress_small_imad(h2, final_s, h1);
#else
	gpu_compress_small(h2, final_s, h1);
#endif
	return ((uint64_t) h1[15] << 32) | h1[14];
}

/* Screens the top 64 bits against the target (superset of fulltest), keeps the two lowest. */
__global__
void bmw256_gpu_hash_80_final(uint32_t threads, uint32_t startNonce, uint32_t *resNonce, const uint64_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		if (bmw256_80_high64(startNonce + thread) <= target)
			report_candidate_2(resNonce, thread);
	}
}

__global__
void bmw256_gpu_checksum_80(uint32_t threads, uint32_t startNonce, uint64_t *acc)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		const uint32_t nonce = startNonce + thread;
		const uint64_t w = bmw256_80_high64(nonce);
		atomicXor((unsigned long long*) &acc[0], (unsigned long long) w);
		atomicXor((unsigned long long*) &acc[1],
			(unsigned long long) (w * (2ULL * (uint64_t) nonce + 1ULL)));
	}
}

#define BMW256_TPB 256

/* d_resNonce must be armed to 0xff by the caller; returns offsets from startNonce. */
__host__
void bmw256_cpu_hash_80_final(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_resNonce, const uint64_t target)
{
	dim3 grid((threads + BMW256_TPB-1)/BMW256_TPB);
	dim3 block(BMW256_TPB);

	bmw256_gpu_hash_80_final<<<grid, block>>>(threads, startNonce, d_resNonce, target);
}

/* GPU half of the GPU==CPU differential over [startNonce, startNonce + threads) for the
 * uploaded job; acc stays {0, 0} if it could not run. */
__host__
void bmw256_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc)
{
	uint64_t *d_acc = NULL;
	acc[0] = acc[1] = 0;

	if (cudaMalloc(&d_acc, 2 * sizeof(uint64_t)) != cudaSuccess)
		return;
	if (cudaMemset(d_acc, 0, 2 * sizeof(uint64_t)) == cudaSuccess) {
		dim3 grid((threads + BMW256_TPB-1)/BMW256_TPB);
		dim3 block(BMW256_TPB);
		bmw256_gpu_checksum_80<<<grid, block>>>(threads, startNonce, d_acc);
		if (cudaDeviceSynchronize() == cudaSuccess)
			cudaMemcpy(acc, d_acc, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	}
	cudaFree(d_acc);
}

__host__
void bmw256_setBlock_80(int thr_id, void *pdata)
{
	uint64_t PaddedMessage[16];
	memcpy(PaddedMessage, pdata, 80);
	memset(&PaddedMessage[10], 0, 48);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, sizeof(PaddedMessage), 0, cudaMemcpyHostToDevice));

	sph_bmw256_context ctx;
	sph_bmw256_init(&ctx);
	sph_bmw256(&ctx, (void*) PaddedMessage, 80);

	// block 2 as the device builds it, word 3 (the nonce) left out
	uint32_t m[16] = { 0 }, pre[44];
	memcpy(m, (const uint32_t*) pdata + 16, 12);
	m[4] = 0x80; m[14] = 640;
	bmw256_precompute(m, (const uint32_t*) ctx.H, pre);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_bmw256_pre, pre, sizeof(pre), 0, cudaMemcpyHostToDevice));
}
