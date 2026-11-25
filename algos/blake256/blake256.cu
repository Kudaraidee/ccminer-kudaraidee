/**
 * Blake-256 Cuda Kernel (Tested on SM 5/5.2)
 *
 * Tanguy Pruvot / SP - Jan 2016
 */

#include <stdint.h>
#include <memory.h>

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

/* threads per block */
#define TPB 512

/* hash by cpu with blake 256 */
extern "C" void blake256hash(void *output, const void *input, int8_t rounds = 14)
{
	uchar hash[64];
	sph_blake256_context ctx;

	sph_blake256_set_rounds(rounds);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);

	memcpy(output, hash, 32);
}

#include "cuda/helper.cuh"
#include "cuda/candidate_report.cuh"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

__constant__ uint32_t _ALIGN(32) d_data[12];

/* 8 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* max count of found nonces in one call */
#define NBN 2
/* Result buffer: the two LOWEST candidates of a launch, res[0] < res[1],
 * UINT32_MAX when absent (report_candidate_2). The host resumes after the
 * higher one, so any further candidate lies above it and is rescanned. */
static __thread uint32_t extra_results[NBN] = { UINT32_MAX };

__constant__ uint32_t c_one = 1;

/* sm_80+ issues LOP3/SHF/PRMT/IADD3 on the integer ALU pipe and IMAD on the FMA
 * pipe, concurrently, and this kernel is bound by the ALU pipe. ptxas already
 * turns the two-input adds into IMAD; the three-input a + (m ^ c) + b stays one
 * IADD3, so it is split into two multiply-adds by a constant-bank 1, which
 * ptxas cannot fold back into an add. Older arches keep the original expression. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t imad_add(const uint32_t x, const uint32_t y)
{
	uint32_t r;
	asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(c_one), "r"(y));
	return r;
}
#define ADD3(a,b,e) a = imad_add(e, imad_add(b, a))
#else
#define ADD3(a,b,e) a += e + b
#endif

#define GSPREC(a,b,c,d,x,y) { \
	ADD3(v[a], v[b], (m[x] ^ c_u256[y])); \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	ADD3(v[a], v[b], (m[y] ^ c_u256[x])); \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
	}

__device__ __forceinline__
void blake256_compress_14(uint32_t *h, const uint32_t *block, const uint32_t T0)
{
	uint32_t /*_ALIGN(8)*/ m[16];
	uint32_t v[16];

	m[0] = block[0];
	m[1] = block[1];
	m[2] = block[2];
	m[3] = block[3];

	const uint32_t c_u256[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	const uint32_t c_Padding[12] = {
		0x80000000UL, 0, 0, 0,
		0, 0, 0, 0,
		0, 1, 0, 640,
	};

	#pragma unroll
	for (uint32_t i = 0; i < 12; i++) {
		m[i+4] = c_Padding[i];
	}

	//#pragma unroll 8
	for(uint32_t i = 0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = c_u256[0];
	v[ 9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ T0;
	v[13] = c_u256[5] ^ T0;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC,0,1);
	GSPREC(1, 5, 0x9, 0xD,2,3);
	GSPREC(2, 6, 0xA, 0xE, 4,5);
	GSPREC(3, 7, 0xB, 0xF, 6,7);
	GSPREC(0, 5, 0xA, 0xF, 8,9);
	GSPREC(1, 6, 0xB, 0xC, 10,11);
	GSPREC(2, 7, 0x8, 0xD, 12,13);
	GSPREC(3, 4, 0x9, 0xE, 14,15);
	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);
	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);
	//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	GSPREC(0, 4, 0x8, 0xC, 9, 0);
	GSPREC(1, 5, 0x9, 0xD, 5, 7);
	GSPREC(2, 6, 0xA, 0xE, 2, 4);
	GSPREC(3, 7, 0xB, 0xF, 10, 15);
	GSPREC(0, 5, 0xA, 0xF, 14, 1);
	GSPREC(1, 6, 0xB, 0xC, 11, 12);
	GSPREC(2, 7, 0x8, 0xD, 6, 8);
	GSPREC(3, 4, 0x9, 0xE, 3, 13);
	//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	GSPREC(0, 4, 0x8, 0xC, 2, 12);
	GSPREC(1, 5, 0x9, 0xD, 6, 10);
	GSPREC(2, 6, 0xA, 0xE, 0, 11);
	GSPREC(3, 7, 0xB, 0xF, 8, 3);
	GSPREC(0, 5, 0xA, 0xF, 4, 13);
	GSPREC(1, 6, 0xB, 0xC, 7, 5);
	GSPREC(2, 7, 0x8, 0xD, 15, 14);
	GSPREC(3, 4, 0x9, 0xE, 1, 9);
	//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	GSPREC(0, 4, 0x8, 0xC, 12, 5);
	GSPREC(1, 5, 0x9, 0xD, 1, 15);
	GSPREC(2, 6, 0xA, 0xE, 14, 13);
	GSPREC(3, 7, 0xB, 0xF, 4, 10);
	GSPREC(0, 5, 0xA, 0xF, 0, 7);
	GSPREC(1, 6, 0xB, 0xC, 6, 3);
	GSPREC(2, 7, 0x8, 0xD, 9, 2);
	GSPREC(3, 4, 0x9, 0xE, 8, 11);
	//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	GSPREC(0, 4, 0x8, 0xC, 13, 11);
	GSPREC(1, 5, 0x9, 0xD, 7, 14);
	GSPREC(2, 6, 0xA, 0xE, 12, 1);
	GSPREC(3, 7, 0xB, 0xF, 3, 9);
	GSPREC(0, 5, 0xA, 0xF, 5, 0);
	GSPREC(1, 6, 0xB, 0xC, 15, 4);
	GSPREC(2, 7, 0x8, 0xD, 8, 6);
	GSPREC(3, 4, 0x9, 0xE, 2, 10);
	//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	GSPREC(0, 4, 0x8, 0xC, 6, 15);
	GSPREC(1, 5, 0x9, 0xD, 14, 9);
	GSPREC(2, 6, 0xA, 0xE, 11, 3);
	GSPREC(3, 7, 0xB, 0xF, 0, 8);
	GSPREC(0, 5, 0xA, 0xF, 12, 2);
	GSPREC(1, 6, 0xB, 0xC, 13, 7);
	GSPREC(2, 7, 0x8, 0xD, 1, 4);
	GSPREC(3, 4, 0x9, 0xE, 10, 5);
	//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	GSPREC(0, 4, 0x8, 0xC, 10, 2);
	GSPREC(1, 5, 0x9, 0xD, 8, 4);
	GSPREC(2, 6, 0xA, 0xE, 7, 6);
	GSPREC(3, 7, 0xB, 0xF, 1, 5);
	GSPREC(0, 5, 0xA, 0xF, 15, 11);
	GSPREC(1, 6, 0xB, 0xC, 9, 14);
	GSPREC(2, 7, 0x8, 0xD, 3, 12);
	GSPREC(3, 4, 0x9, 0xE, 13, 0);
	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC, 0, 1);
	GSPREC(1, 5, 0x9, 0xD, 2, 3);
	GSPREC(2, 6, 0xA, 0xE, 4, 5);
	GSPREC(3, 7, 0xB, 0xF, 6, 7);
	GSPREC(0, 5, 0xA, 0xF, 8, 9);
	GSPREC(1, 6, 0xB, 0xC, 10, 11);
	GSPREC(2, 7, 0x8, 0xD, 12, 13);
	GSPREC(3, 4, 0x9, 0xE, 14, 15);
	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);
	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);

	// only compute h6 & 7
	h[6U] ^= v[6U] ^ v[14U];
	h[7U] ^= v[7U] ^ v[15U];
}

/* ############################################################################################################################### */
/* Precalculated 1st 64-bytes block (midstate) method */

__global__ __launch_bounds__(1024,1)
void blake256_gpu_hash_16(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint64_t highTarget)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t _ALIGN(16) h[8];

		#pragma unroll
		for(int i=0; i < 8; i++) {
			h[i] = d_data[i];
		}

		// ------ Close: Bytes 64 to 80 ------

		uint32_t _ALIGN(16) ending[4];
		ending[0] = d_data[8];
		ending[1] = d_data[9];
		ending[2] = d_data[10];
		ending[3] = nonce; /* our tested value */

		blake256_compress_14(h, ending, 640);

		/* Real 64-bit compare on the top two target words. The old
		 * (h[7] == 0 && ...) form dropped valid shares below share diff 1. */
		const uint64_t high = ((uint64_t) cuda_swab32(h[7]) << 32) | cuda_swab32(h[6]);
		if (high <= highTarget)
			report_candidate_2(resNonce, nonce);
	}
}

/* Diagnostic checksum over a nonce range: two order-independent
 * accumulators over the (h7,h6) pair the mining kernel screens. Calls the
 * same compress body as the mining kernel, so it validates the shipping
 * arithmetic. See blake256_differential() for why acc[1] is needed. */
__global__
void blake256_gpu_checksum_14(const uint32_t threads, const uint32_t startNonce, uint64_t *acc)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t _ALIGN(16) h[8];

		#pragma unroll
		for (int i = 0; i < 8; i++)
			h[i] = d_data[i];

		uint32_t _ALIGN(16) ending[4];
		ending[0] = d_data[8];
		ending[1] = d_data[9];
		ending[2] = d_data[10];
		ending[3] = nonce;

		blake256_compress_14(h, ending, 640);

		const uint64_t w = ((uint64_t) h[7] << 32) | h[6];
		/* Cast to unsigned long long, not uint64_t: on Linux uint64_t is
		 * unsigned long and would not match the CUDA atomic overload. */
		atomicXor((unsigned long long*) &acc[0], (unsigned long long) w);
		atomicXor((unsigned long long*) &acc[1],
			(unsigned long long) (w * (2ULL * (uint64_t) nonce + 1ULL)));
	}
}

/* 8 rounds over the last 16 bytes (h = midstate, m = the padded last block),
 * split where the mining kernel tests h7:
 * blake256_8_head() runs everything but the final G, which only feeds h6,
 * and blake256_8_tail() runs that G. The mining kernel keeps the tail behind
 * its h7 early-out, blake256_gpu_checksum_8() runs both, so the two inline
 * the same arithmetic. (A single helper with an early return measured
 * +8..12 instructions: it added a reconvergence block to the mining kernel.) */
__device__ __forceinline__
void blake256_8_head(const uint32_t *h, const uint32_t *m, uint32_t *v)
{
	const uint32_t c_u256[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	#pragma unroll
	for (uint32_t i = 0; i < 8; i++)
		v[i] = h[i];

	v[8]  = c_u256[0];
	v[9]  = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ 640U;
	v[13] = c_u256[5] ^ 640U;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC, 0, 1);
	GSPREC(1, 5, 0x9, 0xD, 2, 3);
	GSPREC(2, 6, 0xA, 0xE, 4, 5);
	GSPREC(3, 7, 0xB, 0xF, 6, 7);
	GSPREC(0, 5, 0xA, 0xF, 8, 9);
	GSPREC(1, 6, 0xB, 0xC, 10, 11);
	GSPREC(2, 7, 0x8, 0xD, 12, 13);
	GSPREC(3, 4, 0x9, 0xE, 14, 15);
	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);
	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);
	//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	GSPREC(0, 4, 0x8, 0xC, 9, 0);
	GSPREC(1, 5, 0x9, 0xD, 5, 7);
	GSPREC(2, 6, 0xA, 0xE, 2, 4);
	GSPREC(3, 7, 0xB, 0xF, 10, 15);
	GSPREC(0, 5, 0xA, 0xF, 14, 1);
	GSPREC(1, 6, 0xB, 0xC, 11, 12);
	GSPREC(2, 7, 0x8, 0xD, 6, 8);
	GSPREC(3, 4, 0x9, 0xE, 3, 13);
	//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	GSPREC(0, 4, 0x8, 0xC, 2, 12);
	GSPREC(1, 5, 0x9, 0xD, 6, 10);
	GSPREC(2, 6, 0xA, 0xE, 0, 11);
	GSPREC(3, 7, 0xB, 0xF, 8, 3);
	GSPREC(0, 5, 0xA, 0xF, 4, 13);
	GSPREC(1, 6, 0xB, 0xC, 7, 5);
	GSPREC(2, 7, 0x8, 0xD, 15, 14);
	GSPREC(3, 4, 0x9, 0xE, 1, 9);
	//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	GSPREC(0, 4, 0x8, 0xC, 12, 5);
	GSPREC(1, 5, 0x9, 0xD, 1, 15);
	GSPREC(2, 6, 0xA, 0xE, 14, 13);
	GSPREC(3, 7, 0xB, 0xF, 4, 10);
	GSPREC(0, 5, 0xA, 0xF, 0, 7);
	GSPREC(1, 6, 0xB, 0xC, 6, 3);
	GSPREC(2, 7, 0x8, 0xD, 9, 2);
	GSPREC(3, 4, 0x9, 0xE, 8, 11);
	//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	GSPREC(0, 4, 0x8, 0xC, 13, 11);
	GSPREC(1, 5, 0x9, 0xD, 7, 14);
	GSPREC(2, 6, 0xA, 0xE, 12, 1);
	GSPREC(3, 7, 0xB, 0xF, 3, 9);
	GSPREC(0, 5, 0xA, 0xF, 5, 0);
	GSPREC(1, 6, 0xB, 0xC, 15, 4);
	GSPREC(2, 7, 0x8, 0xD, 8, 6);
	//GSPREC(3, 4, 0x9, 0xE, 2, 10);
	//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },

	// only compute h6 & 7
	//h[6] ^= v[6] ^ v[14];
	//h[7] ^= v[7] ^ v[15];

}

__device__ __forceinline__
void blake256_8_tail(const uint32_t *m, uint32_t *v)
{
	const uint32_t c_u256[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};
	GSPREC(3, 4, 0x9, 0xE, 2, 10);
}

__global__
/* The bounds are load-bearing: they are what earns this kernel 100%
 * occupancy where its 14-round sibling gets 66.7%. */
__launch_bounds__(512, 3) /* 40 regs */
void blake256_gpu_hash_16_8(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint64_t highTarget)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t h[8], v[16];

		#pragma unroll
		for (int i = 0; i < 8; i++) {
			h[i] = d_data[i];
		}

		uint32_t m[16] = {
			d_data[8], d_data[9], d_data[10], nonce,
			0x80000000UL, 0, 0, 0,
			0, 0, 0, 0,
			0, 1, 0, 640,
		};

		blake256_8_head(h, m, v);

		/* Target-derived early-out, then a real 64-bit compare: the final GSPREC is
		 * only needed for h6, so it stays behind a test on h7 alone. */
		const uint32_t h7 = cuda_swab32(h[7]^v[7]^v[15]);
		if (h7 <= (uint32_t) (highTarget >> 32))
		{
			blake256_8_tail(m, v);
			const uint64_t high = ((uint64_t) h7 << 32) | cuda_swab32(h[6]^v[6]^v[14]);
			if (high <= highTarget)
				report_candidate_2(resNonce, nonce);
		}
	}
}

/* The 8-round twin of blake256_gpu_checksum_14(), over the same head and
 * tail the mining kernel inlines, with the tail run for every nonce. */
__global__
void blake256_gpu_checksum_8(const uint32_t threads, const uint32_t startNonce, uint64_t *acc)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t h[8], v[16];

		#pragma unroll
		for (int i = 0; i < 8; i++) {
			h[i] = d_data[i];
		}

		uint32_t m[16] = {
			d_data[8], d_data[9], d_data[10], nonce,
			0x80000000UL, 0, 0, 0,
			0, 0, 0, 0,
			0, 1, 0, 640,
		};

		blake256_8_head(h, m, v);
		blake256_8_tail(m, v);

		const uint64_t w = ((uint64_t) (h[7]^v[7]^v[15]) << 32) | (h[6]^v[6]^v[14]);
		atomicXor((unsigned long long*) &acc[0], (unsigned long long) w);
		atomicXor((unsigned long long*) &acc[1],
			(unsigned long long) (w * (2ULL * (uint64_t) nonce + 1ULL)));
	}
}

__host__
static uint32_t blake256_cpu_hash_16(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint64_t highTarget,
	const int8_t rounds)
{
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	/* Both slots armed before every launch: the report keeps a running minimum.
	 * Errors are checked so Ctrl+C cannot segfault on exit. */
	if (cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	if (rounds == 8)
		blake256_gpu_hash_16_8 <<<grid, block>>> (threads, startNonce, d_resNonce[thr_id], highTarget);
	else
		blake256_gpu_hash_16  <<<grid, block>>> (threads, startNonce, d_resNonce[thr_id], highTarget);

	if (cudaSuccess == cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		result = h_resNonce[thr_id][0];
		/* Always reset the extra slot. Leaving it stale is how a nonce from an
		 * earlier launch gets submitted against the current job. */
		extra_results[0] = h_resNonce[thr_id][1];
	}
	return result;
}

/* The shipping launcher, for the self-test's report-ordering leg. */
__host__
uint32_t blake256_report_selftest(const int thr_id, const uint32_t threads, const uint32_t startNonce,
	const uint64_t highTarget, const int8_t rounds, uint32_t *second)
{
	const uint32_t first = blake256_cpu_hash_16(thr_id, threads, startNonce, highTarget, rounds);
	*second = extra_results[0];
	extra_results[0] = UINT32_MAX;
	return first;
}

__host__
static void blake256mid(uint32_t *output, const uint32_t *input, int8_t rounds = 14)
{
	sph_blake256_context ctx;

	sph_blake256_set_rounds(rounds);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 64);

	memcpy(output, (void*)ctx.H, 32);
}

__host__
void blake256_cpu_setBlock_16(uint32_t *penddata, const uint32_t *midstate, const uint32_t *ptarget)
{
	uint32_t _ALIGN(64) data[11];
	memcpy(data, midstate, 32);
	data[8] = penddata[0];
	data[9] = penddata[1];
	data[10]= penddata[2];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, data, 32 + 12, 0, cudaMemcpyHostToDevice));
}

extern bool blake256_device_selftest(int thr_id);

/* Host side of the differential. Allocates its own scratch so it cannot disturb
 * the mining result buffer, and is safe to call before scanhash has run once. */
__host__
void blake256_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc, int8_t rounds)
{
	uint64_t *d_acc = NULL;
	acc[0] = acc[1] = 0;

	if (cudaMalloc(&d_acc, 2 * sizeof(uint64_t)) != cudaSuccess)
		return;
	if (cudaMemset(d_acc, 0, 2 * sizeof(uint64_t)) == cudaSuccess) {
		const dim3 grid((threads + TPB - 1) / TPB);
		const dim3 block(TPB);
		if (rounds == 8)
			blake256_gpu_checksum_8 <<<grid, block>>> (threads, startNonce, d_acc);
		else
			blake256_gpu_checksum_14 <<<grid, block>>> (threads, startNonce, d_acc);
		if (cudaDeviceSynchronize() == cudaSuccess)
			cudaMemcpy(acc, d_acc, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	}
	cudaFree(d_acc);
}

/* Upload the job words the kernels read, from a raw pdata[] - the self-test needs
 * this because blake256mid() is static and the midstate must match the rounds. */
__host__
void blake256_setBlock_selftest(const uint32_t *pdata_words, int8_t rounds)
{
	uint32_t _ALIGN(64) ed[16], mid[8], pend[3];

	for (int k = 0; k < 16; k++)
		be32enc(&ed[k], pdata_words[k]);
	blake256mid(mid, ed, rounds);

	pend[0] = pdata_words[16];
	pend[1] = pdata_words[17];
	pend[2] = pdata_words[18];
	blake256_cpu_setBlock_16(pend, mid, NULL);
}

/* Permanent -D consistency check over a bounded nonce range, using the job
 * words already uploaded for the live job. Costs `count` CPU hashes, so it
 * runs once per job rather than per launch. d_data[0..7] is a midstate
 * computed with the job's own round count, so `rounds` must be that count. */
__host__
static void blake256_check_range(int thr_id, const uint32_t *pdata, uint32_t startNonce, uint32_t count,
	int8_t rounds)
{
	uint32_t _ALIGN(64) ed[20], vhash[8];
	uint64_t gpu[2] = { 0, 0 }, cpu[2] = { 0, 0 };

	blake256_differential(thr_id, count, startNonce, gpu, rounds);

	/* Build the header from pdata directly: scanhash fills endiandata[16..19]
	 * only inside its candidate branch. */
	for (int k = 0; k < 19; k++)
		be32enc(&ed[k], pdata[k]);

	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		be32enc(&ed[19], nonce);
		blake256hash(vhash, ed, rounds);
		/* un-swap to the device's state-word order: sph writes the digest
		 * big-endian, so a little-endian host reads back swab32(H[k]). */
		const uint64_t w = ((uint64_t) swab32(vhash[7]) << 32) | swab32(vhash[6]);
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t) nonce + 1ull);
	}

	if (gpu[0] != cpu[0] || gpu[1] != cpu[1]) {
		gpulog(LOG_ERR, thr_id, "differential MISMATCH (%d rounds) over %u nonces from %08x",
			(int) rounds, count, startNonce);
		gpulog(LOG_ERR, thr_id, "  gpu %016llx / %016llx  cpu %016llx / %016llx",
			(unsigned long long) gpu[0], (unsigned long long) gpu[1],
			(unsigned long long) cpu[0], (unsigned long long) cpu[1]);
	} else {
		gpulog(LOG_DEBUG, thr_id, "differential OK (%d rounds): %u nonces from %08x (xor %016llx)",
			(int) rounds, count, startNonce, (unsigned long long) gpu[0]);
	}
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int8_t blakerounds=14)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t _ALIGN(64) midstate[8];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	uint64_t targetHigh = ((uint64_t*)ptarget)[3];

	int dev_id = device_map[thr_id];
	/* Pascal needs the large batch to stay saturated on Windows too: on a
	 * GTX 1080 Ti 26 measured -3.5% (8 rounds) / -0.7% (14) per MHz against 30.
	 * sm_86 is flat from 26 up, so newer cards keep the shorter Windows batch. */
	const bool sm6x = device_sm[dev_id] >= 600 && device_sm[dev_id] < 700;
	int intensity = (device_sm[dev_id] > 500 && (!is_windows() || sm6x)) ? 30 : 26;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	int rc = 0;

	if (opt_benchmark) {
		/* Loosen the target itself, then derive the device's 64-bit bound from it,
		 * so the screen and fulltest() are governed by the same words. ptarget[7]
		 * must be non-zero, both to exercise the region the old screen discarded
		 * and to lift the hit rate off 2^-32 - at 2^-32 a single swept window most
		 * likely finds nothing at all. Keep the LOWER word wide: a narrow
		 * ptarget[6] under a loose ptarget[7] would make fulltest()'s accept region
		 * a strict subset of the screen's, and the same boundary nonce would then
		 * be re-found and re-rejected for ever. */
		ptarget[7] = 0x000000ffu;
		ptarget[6] = 0xffffffffu;
		targetHigh = ((uint64_t*) ptarget)[3];
	}

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		blake256_device_selftest(thr_id); // fail-closed, exits on mismatch
		init[thr_id] = true;
	}

	for (int k = 0; k < 16; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256mid(midstate, endiandata, blakerounds);
	blake256_cpu_setBlock_16(&pdata[16], midstate, ptarget);

	/* After setBlock, never before: the differential reads the job words the
	 * device already holds. */
	if (opt_debug)
		blake256_check_range(thr_id, pdata, pdata[19], 2048, blakerounds);

	do {
		// GPU HASH (second block only, first is midstate)
		work->nonces[0] = blake256_cpu_hash_16(thr_id, throughput, pdata[19], targetHigh, blakerounds);
		work->nonces[1] = extra_results[0];
		extra_results[0] = UINT32_MAX;

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhashcpu[8];
			/* The kernel reports the two lowest candidates, so everything up to the
			 * higher one has been screened, whichever of them validates. */
			const uint32_t scanned = (work->nonces[1] != UINT32_MAX) ? work->nonces[1] : work->nonces[0];

			for (int k=16; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);

			/* fulltest() alone is the authoritative check - MSW-first over all eight
			 * words. A `vhashcpu[6] <= Htarg &&` prefix would repeat the narrowing. */
			work->valid_nonces = 0;
			be32enc(&endiandata[19], work->nonces[0]);
			blake256hash(vhashcpu, endiandata, blakerounds);
			if (fulltest(vhashcpu, ptarget)) {
				work_set_target_ratio(work, vhashcpu);
				work->valid_nonces++;
			} else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
			}
#if NBN > 1
			if (work->nonces[1] != UINT32_MAX) {
				be32enc(&endiandata[19], work->nonces[1]);
				blake256hash(vhashcpu, endiandata, blakerounds);
				if (fulltest(vhashcpu, ptarget)) {
					if (!work->valid_nonces) {
						/* The first failed: this one is the share, and the submit
						 * loop starts at slot 0. */
						work->nonces[0] = work->nonces[1];
						work_set_target_ratio(work, vhashcpu);
					} else if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio[0]) {
						work->sharediff[1] = work->sharediff[0];
						work->shareratio[1] = work->shareratio[0];
						xchg(work->nonces[1], work->nonces[0]);
						work_set_target_ratio(work, vhashcpu);
					} else {
						bn_set_target_ratio(work, vhashcpu, 1);
					}
					work->valid_nonces++;
				} else {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[1]);
				}
			}
#endif
			/* Resume after the higher reported candidate either way: a third one
			 * lies above it, so a full-throughput step could skip it. */
			pdata[19] = scanned + 1;
			if (work->valid_nonces)
				return work->valid_nonces;
			continue;
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

		// the range check also catches the resume-after-candidates `continue`
	} while (!work_restart[thr_id].restart && pdata[19] < max_nonce);

	*hashes_done = pdata[19] - first_nonce;

	MyStreamSynchronize(NULL, 0, device_map[thr_id]);
	return rc;
}

// cleanup
extern "C" void free_blake256(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}

