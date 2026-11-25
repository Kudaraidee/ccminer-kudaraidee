#include <stdio.h>
#include <memory.h>

#include "sph/sph_fugue.h"

#include "cuda/helper.cuh"
#include <host_defines.h>

#include "cuda/candidate_report.cuh"
#include "cuda/selftest_gate.cuh"

/* res[0] < res[1]: the two lowest candidates of a launch (UINT32_MAX = none) */
static uint32_t *d_resNonce[MAX_GPUS];
/* false once a launch reported something: the buffer is re-armed before the next one */
static bool armed[MAX_GPUS];

__constant__ uint32_t pTarget[8]; // Single GPU
/* sph state after the 19 nonce-free words, advanced through the nonce-free part of
 * sub-round 4, and the folded x0..x2 lookups of its first SMIX (see SMIX_PRE). */
__constant__ uint32_t GPUpre[30];
__constant__ uint32_t GPUpreCR[7];

/* T0 of the super-mix. The other three tables are its byte rotations,
 * T_k(x) = ROTR(T0(x), 8k), so only T0 is stored. */
static __device__ uint32_t d_mixTab0[256];

/* T0 replicated per lane group (lane l reads copy l % 16): a warp's random lookups hit
 * at most two words per bank, against ~3 with a single table. 16 KB per block. */
#define FUGUE_TAB_COPIES 16
#define FUGUE_SHARED_BYTES (256 * FUGUE_TAB_COPIES * sizeof(uint32_t))

/* x * c_fugue_shl[k] == x << 8k, c_fugue_row = byte stride of one entry; constant-bank
 * operands keep ptxas from folding the IMADs back into shift-and-mask. */
static __constant__ uint32_t c_fugue_shl[4] = { 1u, 1u << 8, 1u << 16, 1u << 24 };
static __constant__ uint32_t c_fugue_row = FUGUE_TAB_COPIES * sizeof(uint32_t);

/* T_k applied to byte (3 - k) of x, i.e. mixtab0(x >> 24) ... mixtab3(x & 0xFF). */
template <bool IMADIDX>
__device__ __forceinline__ uint32_t fugue_mix(const uint32_t *tab, const uint32_t lane, const int k, const uint32_t x)
{
	uint32_t v;
#if __CUDA_ARCH__ >= 800
	if (IMADIDX) {
		/* sm_80+: (x << 8k) >> 24 with the shift and the scale-and-add on the
		 * IMAD pipe, one ALU-pipe op per lookup instead of two */
		const uint32_t t = k ? x * c_fugue_shl[k] : x;
		const uint32_t off = (t >> 24) * c_fugue_row + lane * (uint32_t)sizeof(uint32_t);
		v = *(const uint32_t *)((const char *)tab + off);
	} else
#endif
	v = tab[((x >> (24 - 8 * k)) & 0xFF) * FUGUE_TAB_COPIES + lane];
	return k ? __funnelshift_r(v, v, 8 * k) : v;
}

#define FMIX(k, x) fugue_mix<IMADIDX>(mixtabs, lane, (k), (x))

static const uint32_t mixtab0_cpu[] = {
	SPH_C32(0x63633297), SPH_C32(0x7c7c6feb), SPH_C32(0x77775ec7),
	SPH_C32(0x7b7b7af7), SPH_C32(0xf2f2e8e5), SPH_C32(0x6b6b0ab7),
	SPH_C32(0x6f6f16a7), SPH_C32(0xc5c56d39), SPH_C32(0x303090c0),
	SPH_C32(0x01010704), SPH_C32(0x67672e87), SPH_C32(0x2b2bd1ac),
	SPH_C32(0xfefeccd5), SPH_C32(0xd7d71371), SPH_C32(0xabab7c9a),
	SPH_C32(0x767659c3), SPH_C32(0xcaca4005), SPH_C32(0x8282a33e),
	SPH_C32(0xc9c94909), SPH_C32(0x7d7d68ef), SPH_C32(0xfafad0c5),
	SPH_C32(0x5959947f), SPH_C32(0x4747ce07), SPH_C32(0xf0f0e6ed),
	SPH_C32(0xadad6e82), SPH_C32(0xd4d41a7d), SPH_C32(0xa2a243be),
	SPH_C32(0xafaf608a), SPH_C32(0x9c9cf946), SPH_C32(0xa4a451a6),
	SPH_C32(0x727245d3), SPH_C32(0xc0c0762d), SPH_C32(0xb7b728ea),
	SPH_C32(0xfdfdc5d9), SPH_C32(0x9393d47a), SPH_C32(0x2626f298),
	SPH_C32(0x363682d8), SPH_C32(0x3f3fbdfc), SPH_C32(0xf7f7f3f1),
	SPH_C32(0xcccc521d), SPH_C32(0x34348cd0), SPH_C32(0xa5a556a2),
	SPH_C32(0xe5e58db9), SPH_C32(0xf1f1e1e9), SPH_C32(0x71714cdf),
	SPH_C32(0xd8d83e4d), SPH_C32(0x313197c4), SPH_C32(0x15156b54),
	SPH_C32(0x04041c10), SPH_C32(0xc7c76331), SPH_C32(0x2323e98c),
	SPH_C32(0xc3c37f21), SPH_C32(0x18184860), SPH_C32(0x9696cf6e),
	SPH_C32(0x05051b14), SPH_C32(0x9a9aeb5e), SPH_C32(0x0707151c),
	SPH_C32(0x12127e48), SPH_C32(0x8080ad36), SPH_C32(0xe2e298a5),
	SPH_C32(0xebeba781), SPH_C32(0x2727f59c), SPH_C32(0xb2b233fe),
	SPH_C32(0x757550cf), SPH_C32(0x09093f24), SPH_C32(0x8383a43a),
	SPH_C32(0x2c2cc4b0), SPH_C32(0x1a1a4668), SPH_C32(0x1b1b416c),
	SPH_C32(0x6e6e11a3), SPH_C32(0x5a5a9d73), SPH_C32(0xa0a04db6),
	SPH_C32(0x5252a553), SPH_C32(0x3b3ba1ec), SPH_C32(0xd6d61475),
	SPH_C32(0xb3b334fa), SPH_C32(0x2929dfa4), SPH_C32(0xe3e39fa1),
	SPH_C32(0x2f2fcdbc), SPH_C32(0x8484b126), SPH_C32(0x5353a257),
	SPH_C32(0xd1d10169), SPH_C32(0x00000000), SPH_C32(0xededb599),
	SPH_C32(0x2020e080), SPH_C32(0xfcfcc2dd), SPH_C32(0xb1b13af2),
	SPH_C32(0x5b5b9a77), SPH_C32(0x6a6a0db3), SPH_C32(0xcbcb4701),
	SPH_C32(0xbebe17ce), SPH_C32(0x3939afe4), SPH_C32(0x4a4aed33),
	SPH_C32(0x4c4cff2b), SPH_C32(0x5858937b), SPH_C32(0xcfcf5b11),
	SPH_C32(0xd0d0066d), SPH_C32(0xefefbb91), SPH_C32(0xaaaa7b9e),
	SPH_C32(0xfbfbd7c1), SPH_C32(0x4343d217), SPH_C32(0x4d4df82f),
	SPH_C32(0x333399cc), SPH_C32(0x8585b622), SPH_C32(0x4545c00f),
	SPH_C32(0xf9f9d9c9), SPH_C32(0x02020e08), SPH_C32(0x7f7f66e7),
	SPH_C32(0x5050ab5b), SPH_C32(0x3c3cb4f0), SPH_C32(0x9f9ff04a),
	SPH_C32(0xa8a87596), SPH_C32(0x5151ac5f), SPH_C32(0xa3a344ba),
	SPH_C32(0x4040db1b), SPH_C32(0x8f8f800a), SPH_C32(0x9292d37e),
	SPH_C32(0x9d9dfe42), SPH_C32(0x3838a8e0), SPH_C32(0xf5f5fdf9),
	SPH_C32(0xbcbc19c6), SPH_C32(0xb6b62fee), SPH_C32(0xdada3045),
	SPH_C32(0x2121e784), SPH_C32(0x10107040), SPH_C32(0xffffcbd1),
	SPH_C32(0xf3f3efe1), SPH_C32(0xd2d20865), SPH_C32(0xcdcd5519),
	SPH_C32(0x0c0c2430), SPH_C32(0x1313794c), SPH_C32(0xececb29d),
	SPH_C32(0x5f5f8667), SPH_C32(0x9797c86a), SPH_C32(0x4444c70b),
	SPH_C32(0x1717655c), SPH_C32(0xc4c46a3d), SPH_C32(0xa7a758aa),
	SPH_C32(0x7e7e61e3), SPH_C32(0x3d3db3f4), SPH_C32(0x6464278b),
	SPH_C32(0x5d5d886f), SPH_C32(0x19194f64), SPH_C32(0x737342d7),
	SPH_C32(0x60603b9b), SPH_C32(0x8181aa32), SPH_C32(0x4f4ff627),
	SPH_C32(0xdcdc225d), SPH_C32(0x2222ee88), SPH_C32(0x2a2ad6a8),
	SPH_C32(0x9090dd76), SPH_C32(0x88889516), SPH_C32(0x4646c903),
	SPH_C32(0xeeeebc95), SPH_C32(0xb8b805d6), SPH_C32(0x14146c50),
	SPH_C32(0xdede2c55), SPH_C32(0x5e5e8163), SPH_C32(0x0b0b312c),
	SPH_C32(0xdbdb3741), SPH_C32(0xe0e096ad), SPH_C32(0x32329ec8),
	SPH_C32(0x3a3aa6e8), SPH_C32(0x0a0a3628), SPH_C32(0x4949e43f),
	SPH_C32(0x06061218), SPH_C32(0x2424fc90), SPH_C32(0x5c5c8f6b),
	SPH_C32(0xc2c27825), SPH_C32(0xd3d30f61), SPH_C32(0xacac6986),
	SPH_C32(0x62623593), SPH_C32(0x9191da72), SPH_C32(0x9595c662),
	SPH_C32(0xe4e48abd), SPH_C32(0x797974ff), SPH_C32(0xe7e783b1),
	SPH_C32(0xc8c84e0d), SPH_C32(0x373785dc), SPH_C32(0x6d6d18af),
	SPH_C32(0x8d8d8e02), SPH_C32(0xd5d51d79), SPH_C32(0x4e4ef123),
	SPH_C32(0xa9a97292), SPH_C32(0x6c6c1fab), SPH_C32(0x5656b943),
	SPH_C32(0xf4f4fafd), SPH_C32(0xeaeaa085), SPH_C32(0x6565208f),
	SPH_C32(0x7a7a7df3), SPH_C32(0xaeae678e), SPH_C32(0x08083820),
	SPH_C32(0xbaba0bde), SPH_C32(0x787873fb), SPH_C32(0x2525fb94),
	SPH_C32(0x2e2ecab8), SPH_C32(0x1c1c5470), SPH_C32(0xa6a65fae),
	SPH_C32(0xb4b421e6), SPH_C32(0xc6c66435), SPH_C32(0xe8e8ae8d),
	SPH_C32(0xdddd2559), SPH_C32(0x747457cb), SPH_C32(0x1f1f5d7c),
	SPH_C32(0x4b4bea37), SPH_C32(0xbdbd1ec2), SPH_C32(0x8b8b9c1a),
	SPH_C32(0x8a8a9b1e), SPH_C32(0x70704bdb), SPH_C32(0x3e3ebaf8),
	SPH_C32(0xb5b526e2), SPH_C32(0x66662983), SPH_C32(0x4848e33b),
	SPH_C32(0x0303090c), SPH_C32(0xf6f6f4f5), SPH_C32(0x0e0e2a38),
	SPH_C32(0x61613c9f), SPH_C32(0x35358bd4), SPH_C32(0x5757be47),
	SPH_C32(0xb9b902d2), SPH_C32(0x8686bf2e), SPH_C32(0xc1c17129),
	SPH_C32(0x1d1d5374), SPH_C32(0x9e9ef74e), SPH_C32(0xe1e191a9),
	SPH_C32(0xf8f8decd), SPH_C32(0x9898e556), SPH_C32(0x11117744),
	SPH_C32(0x696904bf), SPH_C32(0xd9d93949), SPH_C32(0x8e8e870e),
	SPH_C32(0x9494c166), SPH_C32(0x9b9bec5a), SPH_C32(0x1e1e5a78),
	SPH_C32(0x8787b82a), SPH_C32(0xe9e9a989), SPH_C32(0xcece5c15),
	SPH_C32(0x5555b04f), SPH_C32(0x2828d8a0), SPH_C32(0xdfdf2b51),
	SPH_C32(0x8c8c8906), SPH_C32(0xa1a14ab2), SPH_C32(0x89899212),
	SPH_C32(0x0d0d2334), SPH_C32(0xbfbf10ca), SPH_C32(0xe6e684b5),
	SPH_C32(0x4242d513), SPH_C32(0x686803bb), SPH_C32(0x4141dc1f),
	SPH_C32(0x9999e252), SPH_C32(0x2d2dc3b4), SPH_C32(0x0f0f2d3c),
	SPH_C32(0xb0b03df6), SPH_C32(0x5454b74b), SPH_C32(0xbbbb0cda),
	SPH_C32(0x16166258)
};


#define TIX2(q, x00, x01, x08, x10, x24) { \
		x10 ^= x00; \
		x00 = (q); \
		x08 ^= x00; \
		x01 ^= x24; \
	}

#define TIX3(q, x00, x01, x04, x08, x16, x27, x30) { \
		x16 ^= x00; \
		x00 = (q); \
		x08 ^= x00; \
		x01 ^= x27; \
		x04 ^= x30; \
	}

#define TIX4(q, x00, x01, x04, x07, x08, x22, x24, x27, x30) { \
		x22 ^= x00; \
		x00 = (q); \
		x08 ^= x00; \
		x01 ^= x24; \
		x04 ^= x27; \
		x07 ^= x30; \
	}

#define CMIX30(x00, x01, x02, x04, x05, x06, x15, x16, x17) { \
		x00 ^= x04; \
		x01 ^= x05; \
		x02 ^= x06; \
		x15 ^= x04; \
		x16 ^= x05; \
		x17 ^= x06; \
	}

#define CMIX36(x00, x01, x02, x04, x05, x06, x18, x19, x20) { \
		x00 ^= x04; \
		x01 ^= x05; \
		x02 ^= x06; \
		x18 ^= x04; \
		x19 ^= x05; \
		x20 ^= x06; \
	}

#define SMIX(x0, x1, x2, x3) { \
		uint32_t c0 = 0; \
		uint32_t c1 = 0; \
		uint32_t c2 = 0; \
		uint32_t c3 = 0; \
		uint32_t r0 = 0; \
		uint32_t r1 = 0; \
		uint32_t r2 = 0; \
		uint32_t r3 = 0; \
		uint32_t tmp; \
		tmp = FMIX(0, x0); \
		c0 ^= tmp; \
		tmp = FMIX(1, x0); \
		c0 ^= tmp; \
		r1 ^= tmp; \
		tmp = FMIX(2, x0); \
		c0 ^= tmp; \
		r2 ^= tmp; \
		tmp = FMIX(3, x0); \
		c0 ^= tmp; \
		r3 ^= tmp; \
		tmp = FMIX(0, x1); \
		c1 ^= tmp; \
		r0 ^= tmp; \
		tmp = FMIX(1, x1); \
		c1 ^= tmp; \
		tmp = FMIX(2, x1); \
		c1 ^= tmp; \
		r2 ^= tmp; \
		tmp = FMIX(3, x1); \
		c1 ^= tmp; \
		r3 ^= tmp; \
		tmp = FMIX(0, x2); \
		c2 ^= tmp; \
		r0 ^= tmp; \
		tmp = FMIX(1, x2); \
		c2 ^= tmp; \
		r1 ^= tmp; \
		tmp = FMIX(2, x2); \
		c2 ^= tmp; \
		tmp = FMIX(3, x2); \
		c2 ^= tmp; \
		r3 ^= tmp; \
		tmp = FMIX(0, x3); \
		c3 ^= tmp; \
		r0 ^= tmp; \
		tmp = FMIX(1, x3); \
		c3 ^= tmp; \
		r1 ^= tmp; \
		tmp = FMIX(2, x3); \
		c3 ^= tmp; \
		r2 ^= tmp; \
		tmp = FMIX(3, x3); \
		c3 ^= tmp; \
		x0 = ((c0 ^ r0) & SPH_C32(0xFF000000)) \
			| ((c1 ^ r1) & SPH_C32(0x00FF0000)) \
			| ((c2 ^ r2) & SPH_C32(0x0000FF00)) \
			| ((c3 ^ r3) & SPH_C32(0x000000FF)); \
		x1 = ((c1 ^ (r0 << 8)) & SPH_C32(0xFF000000)) \
			| ((c2 ^ (r1 << 8)) & SPH_C32(0x00FF0000)) \
			| ((c3 ^ (r2 << 8)) & SPH_C32(0x0000FF00)) \
			| ((c0 ^ (r3 >> 24)) & SPH_C32(0x000000FF)); \
		x2 = ((c2 ^ (r0 << 16)) & SPH_C32(0xFF000000)) \
			| ((c3 ^ (r1 << 16)) & SPH_C32(0x00FF0000)) \
			| ((c0 ^ (r2 >> 16)) & SPH_C32(0x0000FF00)) \
			| ((c1 ^ (r3 >> 16)) & SPH_C32(0x000000FF)); \
		x3 = ((c3 ^ (r0 << 24)) & SPH_C32(0xFF000000)) \
			| ((c0 ^ (r1 >> 8)) & SPH_C32(0x00FF0000)) \
			| ((c1 ^ (r2 >> 8)) & SPH_C32(0x0000FF00)) \
			| ((c2 ^ (r3 >> 8)) & SPH_C32(0x000000FF)); \
		/* */ \
	}

/* SMIX of sub-round 4 with x0..x2 per-job constants: the twelve lookups on
 * x0..x2 are folded on the host into GPUpreCR = c0 c1 c2 r0 r1 r2 r3, so only
 * the four on x3 (the nonce word) remain. Same combine as SMIX. */
#define SMIX_PRE(x0, x1, x2, x3) { \
		uint32_t c0 = GPUpreCR[0]; \
		uint32_t c1 = GPUpreCR[1]; \
		uint32_t c2 = GPUpreCR[2]; \
		uint32_t c3 = 0; \
		uint32_t r0 = GPUpreCR[3]; \
		uint32_t r1 = GPUpreCR[4]; \
		uint32_t r2 = GPUpreCR[5]; \
		uint32_t r3 = GPUpreCR[6]; \
		uint32_t tmp; \
		tmp = FMIX(0, x3); \
		c3 ^= tmp; \
		r0 ^= tmp; \
		tmp = FMIX(1, x3); \
		c3 ^= tmp; \
		r1 ^= tmp; \
		tmp = FMIX(2, x3); \
		c3 ^= tmp; \
		r2 ^= tmp; \
		tmp = FMIX(3, x3); \
		c3 ^= tmp; \
		x0 = ((c0 ^ r0) & SPH_C32(0xFF000000)) \
			| ((c1 ^ r1) & SPH_C32(0x00FF0000)) \
			| ((c2 ^ r2) & SPH_C32(0x0000FF00)) \
			| ((c3 ^ r3) & SPH_C32(0x000000FF)); \
		x1 = ((c1 ^ (r0 << 8)) & SPH_C32(0xFF000000)) \
			| ((c2 ^ (r1 << 8)) & SPH_C32(0x00FF0000)) \
			| ((c3 ^ (r2 << 8)) & SPH_C32(0x0000FF00)) \
			| ((c0 ^ (r3 >> 24)) & SPH_C32(0x000000FF)); \
		x2 = ((c2 ^ (r0 << 16)) & SPH_C32(0xFF000000)) \
			| ((c3 ^ (r1 << 16)) & SPH_C32(0x00FF0000)) \
			| ((c0 ^ (r2 >> 16)) & SPH_C32(0x0000FF00)) \
			| ((c1 ^ (r3 >> 16)) & SPH_C32(0x000000FF)); \
		x3 = ((c3 ^ (r0 << 24)) & SPH_C32(0xFF000000)) \
			| ((c0 ^ (r1 >> 8)) & SPH_C32(0x00FF0000)) \
			| ((c1 ^ (r2 >> 8)) & SPH_C32(0x0000FF00)) \
			| ((c2 ^ (r3 >> 8)) & SPH_C32(0x000000FF)); \
	}

#define S00   (sc[ 0])
#define S01   (sc[ 1])
#define S02   (sc[ 2])
#define S03   (sc[ 3])
#define S04   (sc[ 4])
#define S05   (sc[ 5])
#define S06   (sc[ 6])
#define S07   (sc[ 7])
#define S08   (sc[ 8])
#define S09   (sc[ 9])
#define S10   (sc[10])
#define S11   (sc[11])
#define S12   (sc[12])
#define S13   (sc[13])
#define S14   (sc[14])
#define S15   (sc[15])
#define S16   (sc[16])
#define S17   (sc[17])
#define S18   (sc[18])
#define S19   (sc[19])
#define S20   (sc[20])
#define S21   (sc[21])
#define S22   (sc[22])
#define S23   (sc[23])
#define S24   (sc[24])
#define S25   (sc[25])
#define S26   (sc[26])
#define S27   (sc[27])
#define S28   (sc[28])
#define S29   (sc[29])
#define S30   (sc[30])
#define S31   (sc[31])
#define S32   (sc[32])
#define S33   (sc[33])
#define S34   (sc[34])
#define S35   (sc[35])

#define SWAB32(x)		( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )

template <int TPB, bool IMADIDX>
__global__ void  __launch_bounds__(TPB)
fugue256_gpu_hash(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	extern __shared__ uint32_t mixtabs[];

	for (uint32_t i = threadIdx.x; i < 256 * FUGUE_TAB_COPIES; i += blockDim.x)
		mixtabs[i] = __ldg(&d_mixTab0[i / FUGUE_TAB_COPIES]);

	__syncthreads();

	const uint32_t lane = threadIdx.x % FUGUE_TAB_COPIES;

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		/* state after word 18; word 19 is the nonce */
		uint32_t sc[30];

	#pragma unroll 30
		for(int i=0;i<30;i++)
			sc[i] = GPUpre[i];

		uint32_t nounce = startNounce + thread;
		uint32_t q;


		// sub-rounds 4, 0, 1: the nonce, then the 64-bit bit count (0, 640)

		// sub-round 4
		/* TIX2 and the first CMIX30 minus the nonce-free part done by fugue256_cpu_setBlock */
		q = nounce;
		S06 = q;
		S14 ^= q;
		SMIX_PRE(S03, S04, S05, S06);
		CMIX30(S00, S01, S02, S04, S05, S06, S15, S16, S17);
		SMIX(S00, S01, S02, S03);

		// sub-round 0
		q = 0;
		TIX2(q, S00, S01, S08, S10, S24);
		CMIX30(S27, S28, S29, S01, S02, S03, S12, S13, S14);
		SMIX(S27, S28, S29, S00);
		CMIX30(S24, S25, S26, S28, S29, S00, S09, S10, S11);
		SMIX(S24, S25, S26, S27);

		// sub-round 1
		q = 0x280;
		TIX2(q, S24, S25, S02, S04, S18);
		CMIX30(S21, S22, S23, S25, S26, S27, S06, S07, S08);
		SMIX(S21, S22, S23, S24);
		CMIX30(S18, S19, S20, S22, S23, S24, S03, S04, S05);
		SMIX(S18, S19, S20, S21);

		// final rounds

	#pragma unroll 10
		for(int i=0;i<10;i++)
		{
			//ROR(3, 30);
			uint32_t tmp[3];
			#pragma unroll 3
			for(int k=0;k<3;k++)
				tmp[k] = sc[27+k];
			#pragma unroll 27
			for(int k=26;k>=0;k--)
				sc[k+3] = sc[k];
			#pragma unroll 3
			for(int k=0;k<3;k++)
				sc[k] = tmp[k];


			CMIX30(sc[18], sc[19], sc[20], sc[22], sc[23], sc[24], sc[3], sc[4], sc[5]);
			SMIX(sc[18], sc[19], sc[20], sc[21]);
		}

	#pragma unroll 13
		for(int i=0;i<13;i++)
		{
			sc[22] ^= sc[18];
			sc[3] ^= sc[18];

			// ROR(15, 30); BEGIN
			uint32_t tmp1[15];
			#pragma unroll 15
			for(int k=0;k<15;k++)
				tmp1[k] = sc[15+k];
			#pragma unroll 15
			for(int k=14;k>=0;k--)
				sc[k+15] = sc[k];
			#pragma unroll 15
			for(int k=0;k<15;k++)
				sc[k] = tmp1[k];
			// ROR(15, 30); END

			SMIX(sc[18], sc[19], sc[20], sc[21]);
			sc[22] ^= sc[18];
			sc[4] ^= sc[18];

			// ROR(14, 30); BEGIN
			uint32_t tmp2[14];
			#pragma unroll 14
			for(int k=0;k<14;k++)
				tmp2[k] = sc[16+k];
			#pragma unroll 16
			for(int k=15;k>=0;k--)
				sc[k+14] = sc[k];
			#pragma unroll 14
			for(int k=0;k<14;k++)
				sc[k] = tmp2[k];
			// ROR(14, 30); END

			if (i == 12)
				break; // the last SMIX runs below, only if word 7 passes
			SMIX(sc[18], sc[19], sc[20], sc[21]);
		}

		/* Digest word 7 is SWAB32(sc[6]), and the last SMIX writes only
		 * sc[18..21]: settle the common case before paying for it. */
		if (SWAB32(sc[6]) > pTarget[7])
			return;
		SMIX(sc[18], sc[19], sc[20], sc[21]);

		sc[22] ^= sc[18];
		sc[3] ^= sc[18];

		uint32_t hash[8];
	#pragma unroll 4
		for(int i=0;i<4;i++)
			((uint32_t*)hash)[i] = SWAB32(sc[19+i]);

	#pragma unroll 4
		for(int i=0;i<4;i++)
			((uint32_t*)hash)[i+4] = SWAB32(sc[3+i]);

		int i;
		bool rc = true;

		for (i = 7; i >= 0; i--) {
			if (hash[i] > pTarget[i]) {
				rc = false;
				break;
			}
			if (hash[i] < pTarget[i]) {
				rc = true;
				break;
			}
		}

		if (rc)
			report_candidate_2(resNounce, nounce);
	}
}


__host__
void fugue256_cpu_init(int thr_id)
{
	/* Upload the mixing table once per device. */
	cudaMemcpyToSymbol(d_mixTab0, mixtab0_cpu, sizeof(uint32_t) * 256, 0, cudaMemcpyHostToDevice);

	CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));
	armed[thr_id] = false;
}

__host__
void fugue256_cpu_free(int thr_id)
{
	cudaFree(d_resNonce[thr_id]);
	d_resNonce[thr_id] = NULL;
}

__host__
void fugue256_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
{
	/* sph keeps the last 4-byte word (the nonce) pending, so this state has
	 * absorbed 19 words and the kernel resumes at sub-round 4 with the nonce. */
	sph_fugue256_context ctx_fugue_const;
	sph_fugue256_init(&ctx_fugue_const);
	sph_fugue256 (&ctx_fugue_const, data, 80);

	/* nonce-free part of sub-round 4: TIX2 without q, then CMIX30 */
	uint32_t pre[30], cr[7] = { 0 };
	memcpy(pre, ctx_fugue_const.S, sizeof(pre));
	pre[16] ^= pre[6];
	pre[7] ^= pre[0];
	pre[3] ^= pre[7]; pre[4] ^= pre[8]; pre[5] ^= pre[9];
	pre[18] ^= pre[7]; pre[19] ^= pre[8]; pre[20] ^= pre[9];
	/* the x0..x2 lookups of SMIX(S03, S04, S05, S06): c0 c1 c2 r0 r1 r2 r3 */
	for (int w = 0; w < 3; w++) {
		const uint32_t x = pre[3 + w];
		for (int k = 0; k < 4; k++) {
			const uint32_t t0 = mixtab0_cpu[(x >> (24 - 8 * k)) & 0xFF];
			const uint32_t t = k ? ((t0 >> (8 * k)) | (t0 << (32 - 8 * k))) : t0;
			cr[w] ^= t;                 /* c_w takes all four */
			if (k != w) cr[3 + k] ^= t; /* r_k takes every word but its own */
		}
	}
	cudaMemcpyToSymbol(GPUpre, pre, sizeof(pre));
	cudaMemcpyToSymbol(GPUpreCR, cr, sizeof(cr));

	cudaMemcpyToSymbol(pTarget, pTargetIn, 32);
}

/* Returns the lowest candidate of the launch (UINT32_MAX if none) and the
 * second lowest in *secNonce. */
__host__
uint32_t fugue256_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *secNonce)
{
	if (!armed[thr_id]) {
		cudaMemset(d_resNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));
		armed[thr_id] = true;
	}

	if (device_sm[device_map[thr_id]] < 700) {
		/* sm_6x: TPB 512 is faster at the same residency; newer arches tie */
		const uint32_t tpb = 512;
		fugue256_gpu_hash<512, false><<<(threads + tpb-1)/tpb, tpb, FUGUE_SHARED_BYTES>>>(threads, startNounce, d_resNonce[thr_id]);
	} else if (device_sm[device_map[thr_id]] >= 800) {
		const uint32_t tpb = 256;
		fugue256_gpu_hash<256, true><<<(threads + tpb-1)/tpb, tpb, FUGUE_SHARED_BYTES>>>(threads, startNounce, d_resNonce[thr_id]);
	} else {
		const uint32_t tpb = 256;
		fugue256_gpu_hash<256, false><<<(threads + tpb-1)/tpb, tpb, FUGUE_SHARED_BYTES>>>(threads, startNounce, d_resNonce[thr_id]);
	}

	// sleep-based sync, keeps CPU load low
	MyStreamSynchronize(NULL, 0, thr_id);

	uint32_t res[2] = { UINT32_MAX, UINT32_MAX };
	cudaMemcpy(res, d_resNonce[thr_id], sizeof(res), cudaMemcpyDeviceToHost);
	if (res[0] != UINT32_MAX)
		armed[thr_id] = false;
	*secNonce = res[1];
	return res[0];
}

/* ---- init-time self-test (fail-closed, cuda/selftest_gate.cuh) ---- */

#define FUGUE_ST_SPAN  299 /* two blocks, the second one ragged */
#define FUGUE_ST_START 0x51u

static void fugue256_st_hash(const uint32_t *endiandata, uint32_t nonce, uint32_t *digest)
{
	uint32_t data[20];
	memcpy(data, endiandata, sizeof(data));
	data[19] = SWAB32(nonce); // the kernel reads the nonce word big-endian
	sph_fugue256_context ctx;
	sph_fugue256_init(&ctx);
	sph_fugue256(&ctx, data, 80);
	sph_fugue256_close(&ctx, digest);
}

static bool fugue256_st_lower(const uint32_t *a, const uint32_t *b) // a < b as 256-bit LE words
{
	for (int i = 7; i >= 0; i--)
		if (a[i] != b[i]) return a[i] < b[i];
	return false;
}

bool fugue256_device_selftest(int thr_id)
{
	/* Fugue-256("") -- the designers' published empty-message digest */
	static const uint8_t fugue_empty[32] = {
		0xd6,0xec,0x52,0x89,0x80,0xc1,0x30,0xaa, 0xd1,0xd1,0xac,0xd2,0x8b,0x9d,0xd8,0xdb,
		0xde,0xae,0x0d,0x79,0xed,0xed,0x1f,0xca, 0x72,0xc2,0xaf,0x9f,0x37,0xc2,0x24,0x6f
	};
	uint8_t d[32];
	sph_fugue256_context ctx;
	sph_fugue256_init(&ctx);
	sph_fugue256(&ctx, "", 0);
	sph_fugue256_close(&ctx, d);
	const bool host_ok = memcmp(d, fugue_empty, 32) == 0;

	uint32_t data[20];
	for (int i = 0; i < 20; i++)
		data[i] = 0x01020304u * (uint32_t)(i + 1);

	/* The target is the SECOND lowest digest of the span, so exactly two nonces
	 * pass, and the device must report both, lowest nonce first. */
	uint32_t lo[8], hi[8], dig[8];
	uint32_t n_lo = UINT32_MAX, n_hi = UINT32_MAX;
	memset(lo, 0xff, 32); memset(hi, 0xff, 32);
	for (uint32_t t = 0; t < FUGUE_ST_SPAN; t++) {
		const uint32_t n = FUGUE_ST_START + t;
		fugue256_st_hash(data, n, dig);
		if (fugue256_st_lower(dig, lo)) {
			memcpy(hi, lo, 32); n_hi = n_lo;
			memcpy(lo, dig, 32); n_lo = n;
		} else if (fugue256_st_lower(dig, hi)) {
			memcpy(hi, dig, 32); n_hi = n;
		}
	}
	const uint32_t want0 = min(n_lo, n_hi), want1 = max(n_lo, n_hi);

	fugue256_cpu_setBlock(thr_id, data, hi);
	uint32_t got1 = 0;
	const uint32_t got0 = fugue256_cpu_hash(thr_id, FUGUE_ST_SPAN, FUGUE_ST_START, &got1);
	const bool kat_ok = (got0 == want0) && (got1 == want1);

	/* Same target, one header bit flipped: the answer must not survive. */
	data[0] ^= 1u;
	fugue256_cpu_setBlock(thr_id, data, hi);
	uint32_t neg1 = 0;
	const uint32_t neg0 = fugue256_cpu_hash(thr_id, FUGUE_ST_SPAN, FUGUE_ST_START, &neg1);
	const bool neg_ok = !(neg0 == want0 && neg1 == want1);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		gpulog(LOG_ERR, thr_id, "fugue256 self-test: %s", cudaGetErrorString(err));
		selftest_cuda_fault();
	}

	const bool passed = host_ok && kat_ok && neg_ok && err == cudaSuccess;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "fugue256 self-test FAILED (host %d kat %d neg %d)"
			" want=%08x/%08x got=%08x/%08x", (int)host_ok, (int)kat_ok, (int)neg_ok,
			want0, want1, got0, got1);

	return selftest_gate(thr_id, "fugue256", passed);
}
