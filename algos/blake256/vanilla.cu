/**
 * Optimized Blake-256 8-rounds Cuda Kernel (Tested on SM >3.0)
 * Based upon Blake-256 implementation of Tanguy Pruvot - Nov. 2014
 *
 * Provos Alexis - Jan. 2016
 *
 * Fixed CUDA 7.5 flaw
 * minor code changes
 * code cleanup
 * increased nonces per thread
 * removed SSE2 midstate computation
 * Provos Alexis - Mar 2016
 */

#include <stdint.h>
#include <memory.h>

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

#include "cuda/helper.cuh"
#include "cuda/candidate_report.cuh"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

/* threads per block and nonces per thread */
#define TPB 768
#define NPT 384
/* Result buffer: the two LOWEST candidates of a launch, res[0] < res[1],
 * UINT32_MAX when absent (report_candidate_2). */
#define NBN 2

__constant__ uint32_t _ALIGN(16) d_data[21];

/* 16 gpu threads max */
static uint32_t		*d_resNonce[MAX_GPUS];
static uint32_t		*h_resNonce[MAX_GPUS];
static cudaStream_t	streams[MAX_GPUS];

/* hash by cpu with blake 256 */
extern "C" void vanillahash(void *output, const void *input, int8_t blakerounds){
	uchar hash[64];
	sph_blake256_context ctx;

	sph_blake256_set_rounds(blakerounds);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);

	memcpy(output, hash, 32);
}
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

#define GS4(a,b,c,d,x,y,a1,b1,c1,d1,x1,y1,a2,b2,c2,d2,x2,y2,a3,b3,c3,d3,x3,y3) { \
	ADD3(v[ a], v[ b], (m[ x] ^ z[ y])); \
	ADD3(v[a1], v[b1], (m[x1] ^ z[y1])); \
	ADD3(v[a2], v[b2], (m[x2] ^ z[y2])); \
	ADD3(v[a3], v[b3], (m[x3] ^ z[y3])); \
	\
	v[ d] = __byte_perm(v[ d] ^ v[ a], 0, 0x1032); \
	v[d1] = __byte_perm(v[d1] ^ v[a1], 0, 0x1032); \
	v[d2] = __byte_perm(v[d2] ^ v[a2], 0, 0x1032); \
	v[d3] = __byte_perm(v[d3] ^ v[a3], 0, 0x1032); \
	\
	v[ c]+= v[ d]; \
	v[c1]+= v[d1]; \
	v[c2]+= v[d2]; \
	v[c3]+= v[d3]; \
	\
	v[ b] = ROTR32(v[ b] ^ v[ c], 12); \
	v[b1] = ROTR32(v[b1] ^ v[c1], 12); \
	v[b2] = ROTR32(v[b2] ^ v[c2], 12); \
	v[b3] = ROTR32(v[b3] ^ v[c3], 12); \
	\
	ADD3(v[ a], v[ b], (m[ y] ^ z[ x])); \
	ADD3(v[a1], v[b1], (m[y1] ^ z[x1])); \
	ADD3(v[a2], v[b2], (m[y2] ^ z[x2])); \
	ADD3(v[a3], v[b3], (m[y3] ^ z[x3])); \
	\
	v[ d] = __byte_perm(v[ d] ^ v[ a], 0, 0x0321); \
	v[d1] = __byte_perm(v[d1] ^ v[a1], 0, 0x0321); \
	v[d2] = __byte_perm(v[d2] ^ v[a2], 0, 0x0321); \
	v[d3] = __byte_perm(v[d3] ^ v[a3], 0, 0x0321); \
	\
	v[ c]+= v[ d]; \
	v[c1]+= v[d1]; \
	v[c2]+= v[d2]; \
	v[c3]+= v[d3]; \
	\
	v[ b] = ROTR32(v[ b] ^ v[ c], 7); \
	v[b1] = ROTR32(v[b1] ^ v[c1], 7); \
	v[b2] = ROTR32(v[b2] ^ v[c2], 7); \
	v[b3] = ROTR32(v[b3] ^ v[c3], 7); \
}

#define GS3(a,b,c,d,x,y,a1,b1,c1,d1,x1,y1,a2,b2,c2,d2,x2,y2) { \
	ADD3(v[ a], v[ b], (m[ x] ^ z[ y])); \
	ADD3(v[a1], v[b1], (m[x1] ^ z[y1])); \
	ADD3(v[a2], v[b2], (m[x2] ^ z[y2])); \
	\
	v[ d] = __byte_perm(v[ d] ^ v[ a], 0, 0x1032); \
	v[d1] = __byte_perm(v[d1] ^ v[a1], 0, 0x1032); \
	v[d2] = __byte_perm(v[d2] ^ v[a2], 0, 0x1032); \
	\
	v[ c]+= v[ d]; \
	v[c1]+= v[d1]; \
	v[c2]+= v[d2]; \
	\
	v[ b] = ROTR32(v[ b] ^ v[ c], 12); \
	v[b1] = ROTR32(v[b1] ^ v[c1], 12); \
	v[b2] = ROTR32(v[b2] ^ v[c2], 12); \
	\
	ADD3(v[ a], v[ b], (m[ y] ^ z[ x])); \
	ADD3(v[a1], v[b1], (m[y1] ^ z[x1])); \
	ADD3(v[a2], v[b2], (m[y2] ^ z[x2])); \
	\
	v[ d] = __byte_perm(v[ d] ^ v[ a], 0, 0x0321); \
	v[d1] = __byte_perm(v[d1] ^ v[a1], 0, 0x0321); \
	v[d2] = __byte_perm(v[d2] ^ v[a2], 0, 0x0321); \
	\
	v[ c]+= v[ d]; \
	v[c1]+= v[d1]; \
	v[c2]+= v[d2]; \
	\
	v[ b] = ROTR32(v[ b] ^ v[ c], 7); \
	v[b1] = ROTR32(v[b1] ^ v[c1], 7); \
	v[b2] = ROTR32(v[b2] ^ v[c2], 7); \
}

#define GS2(a,b,c,d,x,y,a1,b1,c1,d1,x1,y1) { \
	ADD3(v[ a], v[ b], (m[ x] ^ z[ y])); \
	ADD3(v[a1], v[b1], (m[x1] ^ z[y1])); \
	\
	v[ d] = __byte_perm(v[ d] ^ v[ a], 0, 0x1032); \
	v[d1] = __byte_perm(v[d1] ^ v[a1], 0, 0x1032); \
	\
	v[ c]+= v[ d]; \
	v[c1]+= v[d1]; \
	\
	v[ b] = ROTR32(v[ b] ^ v[ c], 12); \
	v[b1] = ROTR32(v[b1] ^ v[c1], 12); \
	\
	ADD3(v[ a], v[ b], (m[ y] ^ z[ x])); \
	ADD3(v[a1], v[b1], (m[y1] ^ z[x1])); \
	\
	v[ d] = __byte_perm(v[ d] ^ v[ a], 0, 0x0321); \
	v[d1] = __byte_perm(v[d1] ^ v[a1], 0, 0x0321); \
	\
	v[ c]+= v[ d]; \
	v[c1]+= v[d1]; \
	\
	v[ b] = ROTR32(v[ b] ^ v[ c], 7); \
	v[b1] = ROTR32(v[b1] ^ v[c1], 7); \
}

#define GS(a,b,c,d,x,y) { \
	ADD3(v[a], v[b], (m[x] ^ z[y])); \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
	ADD3(v[a], v[b], (m[y] ^ z[x])); \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}

/* One nonce of the kernel loop, split where it tests h7: vanilla_8_head()
 * runs the 8 rounds up to the h7 word, vanilla_8_tail() the remaining G steps,
 * which only feed h6. tmp[] is the host-advanced state (vanilla_cpu_setBlock_16),
 * m[3] the nonce. The mining kernel keeps the tail behind its early-out,
 * vanilla_gpu_checksum() runs both, so the two inline the same arithmetic. */
__device__ __forceinline__
void vanilla_8_head(uint32_t *v, const uint32_t *tmp, const uint32_t *m, const uint32_t *z,
	const int8_t (*r)[16])
{
	#pragma unroll
	for(int i=0;i<16;i++)
		v[ i] = tmp[ i];

	v[ 1]+= m[3] ^ z[2];
	v[13] = __byte_perm(v[13] ^ v[1],0, 0x0321);
	v[ 9]+= v[13];
	v[ 5] = ROTR32(v[5] ^ v[9], 7);
	v[ 0]+= v[5];
	v[15] = __byte_perm(v[15] ^ v[0],0, 0x1032);

	v[10]+= v[15];
	v[ 5] = ROTR32(v[5] ^ v[10], 12);
	ADD3(v[ 0], v[5], z[8]);
	v[15] = __byte_perm(v[15] ^ v[0],0, 0x0321);
	v[10]+= v[15];
	v[ 5] = ROTR32(v[5] ^ v[10], 7);

	GS3( 1, 6,11,12,10,11, 2, 7, 8,13,12,13, 3, 4, 9,14,14,15);

	#pragma unroll
	for(int i=0;i<6;i++){
		GS4(0, 4, 8,12,r[i][ 0],r[i][ 1], 1, 5, 9,13,r[i][ 2],r[i][ 3], 2, 6,10,14,r[i][ 4],r[i][ 5], 3, 7,11,15,r[i][ 6],r[i][ 7]);
		GS4(0, 5,10,15,r[i][ 8],r[i][ 9], 1, 6,11,12,r[i][10],r[i][11], 2, 7, 8,13,r[i][12],r[i][13], 3, 4, 9,14,r[i][14],r[i][15]);
	}
	GS4(0, 4, 8,12,r[6][ 0],r[6][ 1], 1, 5, 9,13,r[6][ 2],r[6][ 3], 2, 6,10,14,r[6][ 4],r[6][ 5], 3, 7,11,15,r[6][ 6],r[6][ 7]);

	ADD3(v[ 0], v[5], (m[ 5] ^ z[0]));
	ADD3(v[ 2], v[7], (m[ 8] ^ z[6]));
	v[13] = __byte_perm(v[13] ^ v[2],0, 0x1032);
	v[15] = __byte_perm(v[15] ^ v[0],0, 0x1032);

	v[ 8] += v[13];
	v[10] += v[15];

	v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);

	ADD3(v[ 0], v[5], (m[ 0] ^ z[5]));
	ADD3(v[ 2], v[7], (m[ 6] ^ z[8]));

	v[15] = __byte_perm(v[15] ^ v[ 0],0, 0x0321);
	v[13] = __byte_perm(v[13] ^ v[ 2],0, 0x0321);

	v[8] += v[13];
	v[7]  = ROTR32(v[7] ^ v[8], 7);
}

__device__ __forceinline__
void vanilla_8_tail(uint32_t *v, const uint32_t *m, const uint32_t *z)
{
	ADD3(v[ 1], v[6], (m[15] ^ z[ 4]));
	ADD3(v[ 3], v[4], (m[2] ^ z[10]));
	v[12]  = __byte_perm(v[12] ^ v[ 1],0, 0x1032);
	v[14]  = __byte_perm(v[14] ^ v[3],0, 0x1032);
	v[11] += v[12];
	v[ 9] += v[14];
	v[ 6]  = ROTR32(v[ 6] ^ v[11], 12);

	ADD3(v[ 1], v[ 6], (m[ 4] ^ z[15]));
	ADD3(v[ 3], ROTR32(v[ 4] ^ v[ 9],12), (m[10] ^ z[ 2]));
	v[12]  = __byte_perm(v[12] ^ v[ 1],0, 0x0321);
	v[14]  = __byte_perm(v[14] ^ v[ 3],0, 0x0321);

	v[11] += v[12];
	v[ 6] = ROTR32(v[ 6] ^ v[11], 7);
}

__global__ __launch_bounds__(TPB,1)
void vanilla_gpu_hash_16_8(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce,const uint64_t highTarget){
	uint32_t _ALIGN(16) v[16];
	uint32_t _ALIGN(16) tmp[16];

	const size_t thread   = blockDim.x * blockIdx.x + threadIdx.x;
	const uint64_t step     = gridDim.x * blockDim.x;
	const uint64_t maxNonce = startNonce + threads;

	const int8_t r[][16] = {
		{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
		{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
		{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
		{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 }
	};
	const uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

//PREFETCH
	#pragma unroll
	for(int i=0;i<16;i++){
		tmp[ i] = d_data[ i];
	}

	uint32_t m[16] = {
				  d_data[16],	d_data[17],	d_data[18],	0,
				  0x80000000UL,	0,		0,		0,
				  0,		0,		0,		0,
				  0,		1,		0,		640
	};

	const uint32_t h7 = d_data[19];
	const uint32_t h6 = d_data[20];
//END OF PREFETCH
	uint64_t m3 = startNonce + thread;
loopstart:
		if(m3>=maxNonce)return;
		m[3]  = m3;

		vanilla_8_head(v, tmp, m, z, r);

		// only compute h6 & 7
		/* h6 is only needed if h7 already passes, so keep the tail behind a test
		 * on h7 alone. Derived from the target, not hard-coded to zero. */
		const uint32_t hh7 = cuda_swab32(h7^v[7]^v[15]);
		if (hh7 <= (uint32_t) (highTarget >> 32)) {
			vanilla_8_tail(v, m, z);

			const uint64_t high = ((uint64_t) hh7 << 32) | cuda_swab32(h6^v[6]^v[14]);
			/* No return after a hit: keep scanning this thread's remaining strided
			 * nonces, or a lower one later in the stride is never examined. */
			if (high <= highTarget)
				report_candidate_2(resNonce, (uint32_t) m[3]);
		}
	m3+=step;
	goto loopstart;
}

/* Diagnostic checksum over a nonce range, one nonce per thread: the two
 * order-independent accumulators blake256.cu uses, over the raw (h7,h6) state
 * words, through the same head and tail the mining kernel inlines. */
__global__
void vanilla_gpu_checksum(const uint32_t threads, const uint32_t startNonce, uint64_t *acc)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	const int8_t r[][16] = {
		{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
		{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
		{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
		{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 }
	};
	const uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};
	uint32_t _ALIGN(16) v[16];
	uint32_t _ALIGN(16) tmp[16];

	#pragma unroll
	for (int i = 0; i < 16; i++)
		tmp[i] = d_data[i];

	const uint32_t nonce = startNonce + thread;
	const uint32_t m[16] = {
		d_data[16], d_data[17], d_data[18], nonce,
		0x80000000UL, 0, 0, 0,
		0, 0, 0, 0,
		0, 1, 0, 640
	};

	vanilla_8_head(v, tmp, m, z, r);
	vanilla_8_tail(v, m, z);

	const uint64_t w = ((uint64_t) (d_data[19] ^ v[7] ^ v[15]) << 32) | (d_data[20] ^ v[6] ^ v[14]);
	/* unsigned long long, not uint64_t: on Linux uint64_t is unsigned long and
	 * would not match the CUDA atomic overload. */
	atomicXor((unsigned long long*) &acc[0], (unsigned long long) w);
	atomicXor((unsigned long long*) &acc[1],
		(unsigned long long) (w * (2ULL * (uint64_t) nonce + 1ULL)));
}

__host__
void vanilla_cpu_setBlock_16(const int thr_id,const uint32_t* endiandata, uint32_t *penddata){

	const uint32_t _ALIGN(64) z[16] = {
		SPH_C32(0x243F6A88), SPH_C32(0x85A308D3), SPH_C32(0x13198A2E), SPH_C32(0x03707344),
		SPH_C32(0xA4093822), SPH_C32(0x299F31D0), SPH_C32(0x082EFA98), SPH_C32(0xEC4E6C89),
		SPH_C32(0x452821E6), SPH_C32(0x38D01377), SPH_C32(0xBE5466CF), SPH_C32(0x34E90C6C),
		SPH_C32(0xC0AC29B7), SPH_C32(0xC97C50DD), SPH_C32(0x3F84D5B5), SPH_C32(0xB5470917)
	};
	uint32_t _ALIGN(64) h[22];

	sph_blake256_context ctx;

	sph_blake256_set_rounds(8);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, endiandata, 64);

	h[ 0] = ctx.H[0];	h[ 1] = ctx.H[1];
	h[ 2] = ctx.H[2];	h[21] = ctx.H[3];
	h[ 4] = ctx.H[4];	h[20] = ctx.H[5];
	h[19] = ctx.H[6];	h[16] = ctx.H[7];

	uint32_t tmp = h[20];
	h[20] = h[19];
	h[19] = h[16];
	h[16] = penddata[ 0];
	h[17] = penddata[ 1];
	h[18] = penddata[ 2];
	h[12] = z[ 4] ^ 640;
	h[ 8] = z[ 0];

	h[ 0] += (h[16] ^ z[ 1]) + h[ 4];
	h[12]  = SPH_ROTR32(h[12] ^ h[0],16);
	h[ 8] += h[12];
	h[ 4]  = SPH_ROTR32(h[ 4] ^ h[ 8], 12);
	h[ 0] += (h[17] ^ z[ 0]) + h[ 4];
	h[12]  = SPH_ROTR32(h[12] ^ h[0],8);
	h[ 8] += h[12];
	h[ 4]  = SPH_ROTR32(h[ 4] ^ h[ 8], 7);

	h[1] += (h[18] ^ z[ 3]) + tmp;

	h[13] = SPH_ROTR32(z[ 5] ^ 640 ^ h[1],16);
	h[ 5] = ROTR32(tmp ^ (z[ 1] + h[13]), 12);

	h[ 1] += h[ 5];
	h[ 2] += (0x80000000UL ^ z[ 5]) + h[20];

	h[14]  = SPH_ROTR32(z[ 6] ^ h[2], 16);
	h[ 6]  = z[ 2] + h[14];
	h[ 6]  = SPH_ROTR32(h[20] ^ h[ 6], 12);

	h[21] += z[ 7] + h[19];
	h[ 0] += z[ 9];

	h[ 2] += z[ 4] + h[ 6];

	h[ 9] = z[ 1] + h[13];
	h[10] = z[ 2] + h[14];

	h[14] = SPH_ROTR32(h[14] ^ h[2],8); //0x0321
	h[10]+=h[14];

	h[ 6] = SPH_ROTR32(h[ 6] ^ h[10],7);
	h[15] = SPH_ROTR32(z[ 7] ^ h[21],16);

	h[11] = z[ 3] + h[15];
	h[ 7] = SPH_ROTR32(h[19] ^ h[11], 12);
	h[ 3] = h[21] + h[ 7] + z[ 6];

	h[15] = SPH_ROTR32(h[15] ^ h[ 3],8);
	h[11]+= h[15];
	h[ 7] = ROTR32(h[ 7] ^ h[11],7);

	cudaMemcpyToSymbolAsync(d_data, h, 21*sizeof(uint32_t), 0, cudaMemcpyHostToDevice, streams[thr_id]);
}

/* Host side of the differential: uploads the job from pdata (so it can run
 * before scanhash and from the self-test), then checksums `count` nonces from
 * `startNonce` on the GPU. acc stays {0,0} if the GPU part could not run. */
__host__
void vanilla_differential(int thr_id, const uint32_t *pdata, uint32_t count, uint32_t startNonce, uint64_t *acc)
{
	uint32_t _ALIGN(64) endiandata[16];
	uint64_t *d_acc = NULL;
	acc[0] = acc[1] = 0;

	for (int k = 0; k < 16; k++)
		be32enc(&endiandata[k], pdata[k]);
	vanilla_cpu_setBlock_16(thr_id, endiandata, (uint32_t*) &pdata[16]);

	if (cudaMalloc(&d_acc, 2 * sizeof(uint64_t)) != cudaSuccess)
		return;
	if (cudaMemsetAsync(d_acc, 0, 2 * sizeof(uint64_t), streams[thr_id]) == cudaSuccess) {
		vanilla_gpu_checksum <<<(count + TPB - 1) / TPB, TPB, 0, streams[thr_id]>>> (count, startNonce, d_acc);
		if (cudaStreamSynchronize(streams[thr_id]) == cudaSuccess)
			cudaMemcpy(acc, d_acc, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	}
	cudaFree(d_acc);
}

/* Permanent -D consistency check, once per job: GPU checksum against the
 * host reference over `count` nonces. */
static void vanilla_check_range(int thr_id, const uint32_t *pdata, uint32_t startNonce, uint32_t count)
{
	uint32_t _ALIGN(64) ed[20], vhash[8];
	uint64_t gpu[2], cpu[2] = { 0, 0 };

	vanilla_differential(thr_id, pdata, count, startNonce, gpu);

	for (int k = 0; k < 19; k++)
		be32enc(&ed[k], pdata[k]);
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		be32enc(&ed[19], nonce);
		vanillahash(vhash, ed, 8);
		/* sph writes the digest big-endian: un-swap to the state words. */
		const uint64_t w = ((uint64_t) swab32(vhash[7]) << 32) | swab32(vhash[6]);
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t) nonce + 1ull);
	}

	if (gpu[0] != cpu[0] || gpu[1] != cpu[1]) {
		gpulog(LOG_ERR, thr_id, "differential MISMATCH over %u nonces from %08x", count, startNonce);
		gpulog(LOG_ERR, thr_id, "  gpu %016llx / %016llx  cpu %016llx / %016llx",
			(unsigned long long) gpu[0], (unsigned long long) gpu[1],
			(unsigned long long) cpu[0], (unsigned long long) cpu[1]);
	} else {
		gpulog(LOG_DEBUG, thr_id, "differential OK: %u nonces from %08x (xor %016llx)",
			count, startNonce, (unsigned long long) gpu[0]);
	}
}

extern bool vanilla_device_selftest(int thr_id);

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_vanilla(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, const int8_t blakerounds)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce  = pdata[19];
	uint64_t targetHigh = ((uint64_t*) ptarget)[3];

	if (opt_benchmark) {
		/* Loosen the target itself, not just the device bound, and keep the low
		 * word wide so fulltest() still accepts everything the screen passes. */
		ptarget[7] = 0x000000ffu;
		ptarget[6] = 0xffffffffu;
		targetHigh = ((uint64_t*) ptarget)[3];
	}
	int dev_id = device_map[thr_id];

	/* NPT*TPB is large, so a low intensity leaves too few blocks to fill the
	 * device and per-launch cost dominates. 27 is the smallest size on the
	 * throughput plateau on sm_86; going higher only lengthens the batch.
	 * Pascal is not on it at 27 (GTX 1080 Ti: -10.9% per MHz against 30),
	 * so it takes 30 on Windows as well. */
	const bool sm6x = device_sm[dev_id] >= 600 && device_sm[dev_id] < 700;
	int intensity = (device_sm[dev_id] > 500 && (!is_windows() || sm6x)) ? 30 : 27;
	if (device_sm[dev_id] < 350) intensity = 22;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id]) {
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
		cudaStreamCreate(&streams[thr_id]);
		vanilla_device_selftest(thr_id); // fail-closed, exits on mismatch
		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];

	for (int k = 0; k < 16; k++)
		be32enc(&endiandata[k], pdata[k]);

	/* Before the job upload: the differential uploads the same words itself. */
	if (opt_debug)
		vanilla_check_range(thr_id, pdata, pdata[19], 2048);

	vanilla_cpu_setBlock_16(thr_id,endiandata,&pdata[16]);

	const dim3 grid((throughput + (NPT*TPB)-1)/(NPT*TPB));
	const dim3 block(TPB);
	int rc = 0;

	do {
		/* Both slots armed before every launch, same stream as the kernel. */
		cudaMemsetAsync(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t), streams[thr_id]);
		vanilla_gpu_hash_16_8<<<grid,block, 0, streams[thr_id]>>>(throughput, pdata[19], d_resNonce[thr_id], targetHigh);
		cudaMemcpyAsync(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost,streams[thr_id]);
		*hashes_done = pdata[19] - first_nonce + throughput;
		cudaStreamSynchronize(streams[thr_id]);

		work->nonces[0] = h_resNonce[thr_id][0];
		work->nonces[1] = h_resNonce[thr_id][1];
		if (work->nonces[0] != UINT32_MAX) {
			uint32_t vhashcpu[8];
			/* The kernel reports the two lowest candidates, so everything up to the
			 * higher one has been screened, whichever of them validates. */
			const uint32_t scanned = (work->nonces[1] != UINT32_MAX) ? work->nonces[1] : work->nonces[0];

			for (int k=0; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);

			/* fulltest() alone is authoritative; the old vhashcpu[6] <= Htarg
			 * prefix repeated the device's narrowing and dropped the same shares. */
			work->valid_nonces = 0;
			be32enc(&endiandata[19], work->nonces[0]);
			vanillahash(vhashcpu, endiandata, blakerounds);
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
				/* The second nonce needs the same re-verify as the first, or a
				 * kernel fault here becomes a bad share instead of a reject. */
				be32enc(&endiandata[19], work->nonces[1]);
				vanillahash(vhashcpu, endiandata, blakerounds);
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

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	MyStreamSynchronize(NULL, 0, dev_id);
	return rc;
}

// cleanup
extern "C" void free_vanilla(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
