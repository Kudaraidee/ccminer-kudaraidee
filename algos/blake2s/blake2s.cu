/**
 * Based on the SPH implementation of blake2s
 * Provos Alexis - 2016
 */

#include "miner.h"

#include <string.h>
#include <stdint.h>

#include "sph/blake2s.h"
#include "sph/sph_types.h"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

#include "cuda/helper.cuh"

#ifdef __CUDA_ARCH__

__device__ __forceinline__
uint32_t ROR8(const uint32_t a) {
	return __byte_perm(a, 0, 0x0321);
}

__device__ __forceinline__
uint32_t ROL16(const uint32_t a) {
	return __byte_perm(a, 0, 0x1032);
}

#else
#define ROR8(u)  (((u) >> 8) | ((u) << 24))
#define ROL16(u) (((u) << 16) | ((u) >> 16))
#endif

static const uint32_t blake2s_IV[8] = {
	0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
	0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

static const uint8_t blake2s_sigma[10][16] = {
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 },
};

#define G(r,i,a,b,c,d) \
	do { \
		a = a + b + m[blake2s_sigma[r][2*i+0]]; \
		d = SPH_ROTR32(d ^ a, 16); \
		c = c + d; \
		b = SPH_ROTR32(b ^ c, 12); \
		a = a + b + m[blake2s_sigma[r][2*i+1]]; \
		d = SPH_ROTR32(d ^ a, 8); \
		c = c + d; \
		b = SPH_ROTR32(b ^ c, 7); \
	} while(0)
#define ROUND(r)  \
	do { \
		G(r,0,v[0],v[4],v[ 8],v[12]); \
		G(r,1,v[1],v[5],v[ 9],v[13]); \
		G(r,2,v[2],v[6],v[10],v[14]); \
		G(r,3,v[3],v[7],v[11],v[15]); \
		G(r,4,v[0],v[5],v[10],v[15]); \
		G(r,5,v[1],v[6],v[11],v[12]); \
		G(r,6,v[2],v[7],v[ 8],v[13]); \
		G(r,7,v[3],v[4],v[ 9],v[14]); \
	} while(0)

extern "C" void blake2s_hash(void *output, const void *input)
{
	uint32_t m[16];
	uint32_t v[16];
	uint32_t h[8];

	uint32_t *in = (uint32_t*)input;
//	COMPRESS
	for(int i = 0; i < 16; ++i )
		m[i] = in[i];

	h[0] = 0x01010020 ^ blake2s_IV[0];
	h[1] = blake2s_IV[1];
	h[2] = blake2s_IV[2];
	h[3] = blake2s_IV[3];
	h[4] = blake2s_IV[4];
	h[5] = blake2s_IV[5];
	h[6] = blake2s_IV[6];
	h[7] = blake2s_IV[7];

	for(int i = 0; i < 8; ++i )
		v[i] = h[i];

	v[ 8] = blake2s_IV[0];		v[ 9] = blake2s_IV[1];
	v[10] = blake2s_IV[2];		v[11] = blake2s_IV[3];
	v[12] = 64 ^ blake2s_IV[4];	v[13] = blake2s_IV[5];
	v[14] = blake2s_IV[6];		v[15] = blake2s_IV[7];

	ROUND( 0 ); ROUND( 1 );
	ROUND( 2 ); ROUND( 3 );
	ROUND( 4 ); ROUND( 5 );
	ROUND( 6 ); ROUND( 7 );
	ROUND( 8 ); ROUND( 9 );

	for(size_t i = 0; i < 8; ++i)
		h[i] ^= v[i] ^ v[i + 8];

//	COMPRESS
	m[0] = in[16]; m[1] = in[17];
	m[2] = in[18]; m[3] = in[19];
	for(size_t i = 4; i < 16; ++i)
		m[i] = 0;

	for(size_t i = 0; i < 8; ++i)
		v[i] = h[i];

	v[ 8] = blake2s_IV[0];		v[ 9] = blake2s_IV[1];
	v[10] = blake2s_IV[2];		v[11] = blake2s_IV[3];
	v[12] = 0x50 ^ blake2s_IV[4];	v[13] = blake2s_IV[5];
	v[14] = ~blake2s_IV[6];		v[15] = blake2s_IV[7];

	ROUND( 0 ); ROUND( 1 );
	ROUND( 2 ); ROUND( 3 );
	ROUND( 4 ); ROUND( 5 );
	ROUND( 6 ); ROUND( 7 );
	ROUND( 8 ); ROUND( 9 );

	for(size_t i = 0; i < 8; ++i)
		h[i] ^= v[i] ^ v[i + 8];

	memcpy(output, h, 32);
}

/* 1024. sm_86 caps at 1536 threads/SM, which 1024 does not tile, so this is
 * 66.7% occupancy where 512 would give 100% - measured as no difference,
 * because the kernel is ALU-pipe-bound with no latency left to hide. Kept
 * at 1024, the shape the live pool run validated. */
#define TPB 1024
#define NPT 256
#define maxResults 16

__constant__ uint32_t _ALIGN(32) midstate[20];
__constant__ uint32_t c_one = 1;

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* sm_80+ issues LOP3/SHF/PRMT/IADD3 on the integer ALU pipe and IMAD on the FMA
 * pipe, concurrently, and this kernel is bound by the ALU pipe. ptxas already
 * turns the two-input adds into IMAD; the three-input a + b + m stays one IADD3,
 * so it is split into two multiply-adds by a constant-bank 1, which ptxas
 * cannot fold back into an add. Older arches keep the original expression. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t imad_add(const uint32_t x, const uint32_t y)
{
	uint32_t r;
	asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(c_one), "r"(y));
	return r;
}
/* the literal-0 message words of the padding select the two-input form */
__device__ __forceinline__ uint32_t add3(const uint32_t a, const uint32_t b, const int) { return b + a; }
__device__ __forceinline__ uint32_t add3(const uint32_t a, const uint32_t b, const uint32_t e) {
	return imad_add(e, imad_add(b, a));
}
#define ADD3(a,b,e) a = add3(a, b, e)
#else
#define ADD3(a,b,e) a += b + e
#endif

#define GS4(a,b,c,d,e,f,a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3){ \
	ADD3(a,b,e);		ADD3(a1,b1,e1);		ADD3(a2,b2,e2);		ADD3(a3,b3,e3); \
	d  = ROL16( d ^ a);	d1 = ROL16(d1 ^ a1);	d2 = ROL16(d2 ^ a2);	d3 = ROL16(d3 ^ a3); \
	c +=d; 			c1+=d1;			c2+=d2;			c3+=d3;\
	b  = ROTR32(b ^ c, 12); b1 = ROTR32(b1^c1, 12);	b2 = ROTR32(b2^c2, 12);	b3 = ROTR32(b3^c3, 12); \
	ADD3(a,b,f);		ADD3(a1,b1,f1);		ADD3(a2,b2,f2);		ADD3(a3,b3,f3); \
	d  = ROR8(d ^ a);	d1 = ROR8(d1^a1);	d2 = ROR8(d2^a2);	d3 = ROR8(d3^a3); \
	c  += d;		c1 += d1;		c2 += d2;		c3 += d3;\
	b  = ROTR32(b ^ c, 7);	b1 = ROTR32(b1^c1, 7);	b2 = ROTR32(b2^c2, 7);	b3 = ROTR32(b3^c3, 7); \
}

/* Output word 7 of the job's BLAKE2s-256 for one nonce. Shared verbatim by
 * the mining kernel and the diagnostic checksum kernel; the caller passes
 * its own loop-invariant midstate words, which keeps the hoist where it was.
 * Only word 7 exists on the device - the final round prunes the ops that do
 * not feed the target screen, so h0..h6 are never formed. */
__device__ __forceinline__
uint32_t blake2s_word7(const uint32_t nonce, const uint32_t m0, const uint32_t m1,
	const uint32_t m2, const uint32_t h7)
{
	const uint32_t m[3] = { m0, m1, m2 };
	uint32_t v[16];

	#pragma unroll
	for(int i=0;i<16;i++){
		v[ i] = midstate[ i];
	}

//		Round( 0 );
	v[ 1] += nonce;
	v[13] = ROR8(v[13] ^ v[ 1]);
	v[ 9] += v[13];
	v[ 5] = ROTR32(v[ 5] ^ v[ 9], 7);

	v[ 1]+= v[ 6];
	v[ 0]+= v[ 5];

	v[12] = ROL16(v[12] ^ v[ 1]);
	v[13] = ROL16(v[13] ^ v[ 2]);
	v[15] = ROL16(v[15] ^ v[ 0]);

	v[11]+= v[12];				v[ 8]+= v[13];				v[ 9]+= v[14];				v[10]+= v[15];
	v[ 6] = ROTR32(v[ 6] ^ v[11], 12);	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 12);	v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
	v[ 1]+= v[ 6];				v[ 2]+= v[ 7];				v[ 3]+= v[ 4];				v[ 0]+= v[ 5];
	v[12] = ROR8(v[12] ^ v[ 1]);		v[13] = ROR8(v[13] ^ v[ 2]);		v[14] = ROR8(v[14] ^ v[ 3]);		v[15] = ROR8(v[15] ^ v[ 0]);
	v[11]+= v[12]; 				v[ 8]+= v[13];				v[ 9]+= v[14];				v[10]+= v[15];
	v[ 6] = ROTR32(v[ 6] ^ v[11], 7);	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 7);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 7);	v[ 5] = ROTR32(v[ 5] ^ v[10], 7);

	GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
	GS4(v[ 0],v[ 5],v[10],v[15],m[ 1],0,	v[ 1],v[ 6],v[11],v[12],m[ 0],m[ 2],	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],0,nonce);
	GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,m[ 0],	v[ 2],v[ 6],v[10],v[14],0,m[ 2],	v[ 3],v[ 7],v[11],v[15],0,0);
	GS4(v[ 0],v[ 5],v[10],v[15],0,0,	v[ 1],v[ 6],v[11],v[12],nonce,0,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 1],	v[ 3],v[ 4],v[ 9],v[14],0,0);
	GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],nonce,m[ 1],	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
	GS4(v[ 0],v[ 5],v[10],v[15],m[ 2],0,	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 0],	v[ 3],v[ 4],v[ 9],v[14],0,0);
	GS4(v[ 0],v[ 4],v[ 8],v[12],0,m[ 0],	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],m[ 2],0,	v[ 3],v[ 7],v[11],v[15],0,0);
	GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 1],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],nonce,0);
	GS4(v[ 0],v[ 4],v[ 8],v[12],m[ 2],0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],m[ 0],0,	v[ 3],v[ 7],v[11],v[15],0,nonce);
	GS4(v[ 0],v[ 5],v[10],v[15],0,0,	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],m[ 1],0);
	GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],m[ 1],0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
	GS4(v[ 0],v[ 5],v[10],v[15],m[ 0],0,	v[ 1],v[ 6],v[11],v[12],0,nonce,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 2],	v[ 3],v[ 4],v[ 9],v[14],0,0);
	GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,m[ 1],	v[ 3],v[ 7],v[11],v[15],nonce,0);
	GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 0],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],m[ 2],0);
	GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,nonce,	v[ 3],v[ 7],v[11],v[15],m[ 0],0);
	GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 2],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],m[ 1],0,	v[ 3],v[ 4],v[ 9],v[14],0,0);
	GS4(v[ 0],v[ 4],v[ 8],v[12],0,m[ 2],	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],m[ 1],0);

//		GS(9,4,v[ 0],v[ 5],v[10],v[15]);
	v[ 0] += v[ 5];
	ADD3(v[ 2], v[ 7], nonce);
	v[15] = ROL16(v[15] ^ v[ 0]);
	v[13] = ROL16(v[13] ^ v[ 2]);
	v[10] += v[15];
	v[ 8] += v[13];
	v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);
	v[ 0] += v[ 5];
	v[ 2] += v[ 7];
	v[15] = ROR8(v[15] ^ v[ 0]);
	v[13] = ROR8(v[13] ^ v[ 2]);

	v[ 8] += v[13];
	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 7);


	return xor3x(h7,v[7],v[15]);
}

__global__ __launch_bounds__(TPB,1)
void blake2s_gpu_hash_nonce(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint32_t ptarget7)
{
	const uint32_t step = gridDim.x * blockDim.x;

	const uint32_t m0 = midstate[16], m1 = midstate[17], m2 = midstate[18];
	const uint32_t h7 = midstate[19];

	for(uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x ; thread <threads; thread+=step){
		const uint32_t nonce = cuda_swab32(startNonce + thread);

		if (blake2s_word7(nonce, m0, m1, m2, h7) <= ptarget7){
			uint32_t pos = atomicInc(&resNonce[0],0xffffffff)+1;
			if(pos < maxResults)
				resNonce[pos] = nonce;
			return;
		}
	}
}

/* Diagnostic kernel, never launched while mining: accumulates the WHOLE nonce
 * range instead of screening it, so unlike the host re-verify it can see a
 * nonce the mining kernel never reported.
 *   acc[0] = XOR of every word 7          -> a wrong, missing or duplicated word
 *   acc[1] = XOR of word7 * (2*nonce+1)   -> binds each word to its own nonce
 * acc[1] is not redundant: a plain XOR sum is permutation-blind. The weight
 * must be injective and odd - nonce|1 clears bit 0, so an aligned pair
 * (2k, 2k+1) would share a weight and a swap would cancel. */
__global__ __launch_bounds__(TPB,1)
void blake2s_gpu_checksum(const uint32_t threads, const uint32_t startNonce, uint64_t *acc)
{
	const uint32_t step = gridDim.x * blockDim.x;

	const uint32_t m0 = midstate[16], m1 = midstate[17], m2 = midstate[18];
	const uint32_t h7 = midstate[19];

	for(uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x ; thread <threads; thread+=step){
		const uint32_t nonce = startNonce + thread;
		const uint64_t w = blake2s_word7(cuda_swab32(nonce), m0, m1, m2, h7);

		atomicXor((unsigned long long*)&acc[0], (unsigned long long) w);
		atomicXor((unsigned long long*)&acc[1], (unsigned long long)(w * (2ull * (uint64_t)nonce + 1ull)));
	}
}

/* Replays `threads` consecutive nonces for the host to reproduce. Own buffer, so
 * it cannot disturb d_resNonce or its armed state. */
__host__
void blake2s_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc)
{
	uint64_t *d_acc = NULL;
	const dim3 grid((threads + TPB-1)/TPB);
	const dim3 block(TPB);

	acc[0] = acc[1] = 0;
	if (cudaMalloc(&d_acc, 2 * sizeof(uint64_t)) != cudaSuccess) {
		gpulog(LOG_WARNING, thr_id, "differential: cudaMalloc failed, skipped");
		return;
	}
	CUDA_SAFE_CALL(cudaMemset(d_acc, 0, 2 * sizeof(uint64_t)));

	blake2s_gpu_checksum <<<grid, block>>> (threads, startNonce, d_acc);

	CUDA_SAFE_CALL(cudaMemcpy(acc, d_acc, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	cudaFree(d_acc);
}

static void blake2s_setBlock(const uint32_t* input)
{
	uint32_t _ALIGN(64) m[16];
	uint32_t _ALIGN(64) v[16];
	uint32_t _ALIGN(64) h[21];

//	COMPRESS
	for(int i = 0; i < 16; ++i )
		m[i] = input[i];

	h[0] = 0x01010020 ^ blake2s_IV[0];
	h[1] = blake2s_IV[1];
	h[2] = blake2s_IV[2]; h[3] = blake2s_IV[3];
	h[4] = blake2s_IV[4]; h[5] = blake2s_IV[5];
	h[6] = blake2s_IV[6]; h[7] = blake2s_IV[7];

	for(int i = 0; i < 8; ++i )
		v[i] = h[i];

	v[ 8] = blake2s_IV[0];		v[ 9] = blake2s_IV[1];
	v[10] = blake2s_IV[2];		v[11] = blake2s_IV[3];
	v[12] = 64 ^ blake2s_IV[4];	v[13] = blake2s_IV[5];
	v[14] = blake2s_IV[6];		v[15] = blake2s_IV[7];

	ROUND( 0 ); ROUND( 1 );
	ROUND( 2 ); ROUND( 3 );
	ROUND( 4 ); ROUND( 5 );
	ROUND( 6 ); ROUND( 7 );
	ROUND( 8 ); ROUND( 9 );

	for(int i = 0; i < 8; ++i )
		h[i] ^= v[i] ^ v[i + 8];

	h[16] = input[16];
	h[17] = input[17];
	h[18] = input[18];

	h[ 8] = 0x6A09E667; h[ 9] = 0xBB67AE85;
	h[10] = 0x3C6EF372; h[11] = 0xA54FF53A;
	h[12] = 0x510E522F; h[13] = 0x9B05688C;
	h[14] =~0x1F83D9AB; h[15] = 0x5BE0CD19;

	h[ 0]+= h[ 4] + h[16];
	h[12] = SPH_ROTR32(h[12] ^ h[ 0],16);
	h[ 8]+= h[12];
	h[ 4] = SPH_ROTR32(h[ 4] ^ h[ 8],12);
	h[ 0]+= h[ 4] + h[17];
	h[12] = SPH_ROTR32(h[12] ^ h[ 0],8);
	h[ 8]+= h[12];
	h[ 4] = SPH_ROTR32(h[ 4] ^ h[ 8],7);

	h[ 1]+= h[ 5] + h[18];
	h[13] = SPH_ROTR32(h[13] ^ h[ 1], 16);
	h[ 9]+= h[13];
	h[ 5] = ROTR32(h[ 5] ^ h[ 9], 12);

	h[ 2]+= h[ 6];
	h[14] = SPH_ROTR32(h[14] ^ h[ 2],16);
	h[10]+= h[14];
	h[ 6] = SPH_ROTR32(h[ 6] ^ h[10], 12);
	h[ 2]+= h[ 6];
	h[14] = SPH_ROTR32(h[14] ^ h[ 2],8);
	h[10]+= h[14];
	h[ 6] = SPH_ROTR32(h[ 6] ^ h[10], 7);

	h[19] = h[7]; //constant h[7] for nonce check

	h[ 3]+= h[ 7];
	h[15] = SPH_ROTR32(h[15] ^ h[ 3],16);
	h[11]+= h[15];
	h[ 7] = SPH_ROTR32(h[ 7] ^ h[11], 12);
	h[ 3]+= h[ 7];
	h[15] = SPH_ROTR32(h[15] ^ h[ 3],8);
	h[11]+= h[15];
	h[ 7] = SPH_ROTR32(h[ 7] ^ h[11], 7);

	h[ 1]+= h[ 5];
	h[ 3]+= h[ 4];
	h[14] = SPH_ROTR32(h[14] ^ h[ 3],16);

	h[ 2]+= h[ 7];
	cudaMemcpyToSymbol(midstate, h, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

/* GPU-vs-CPU differential over a nonce range. Both sides accumulate the same
 * two values; see blake2s_gpu_checksum for why acc[1] is load-bearing. Word
 * 7 only - that is all the device computes. */
/* For the self-test TU, which needs the kernel to see a header of its own
 * choosing. Not used by the mining path. */
void blake2s_setBlock_selftest(const uint32_t *input)
{
	blake2s_setBlock(input);
}

extern bool blake2s_device_selftest(int thr_id);

static bool blake2s_check_range(int thr_id, const uint32_t *endiandata, uint32_t startNonce, uint32_t count)
{
	uint32_t _ALIGN(64) vhash[8], td[20];
	uint64_t gpu[2] = { 0, 0 }, cpu[2] = { 0, 0 };

	blake2s_differential(thr_id, count, startNonce, gpu);

	memcpy(td, endiandata, sizeof(td));
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		be32enc(&td[19], nonce);
		blake2s_hash(vhash, td);
		const uint64_t w = vhash[7];
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t)nonce + 1ull);
	}

	if (gpu[0] == cpu[0] && gpu[1] == cpu[1]) {
		gpulog(LOG_BLUE, thr_id, "differential OK: %u nonces from %08x (xor %016llx)",
			count, startNonce, (unsigned long long) cpu[0]);
		return true;
	}

	gpulog(LOG_ERR, thr_id, "DIFFERENTIAL FAILED over %u nonces from %08x - the GPU is not "
		"hashing this range as the CPU does", count, startNonce);
	gpulog(LOG_ERR, thr_id, "  gpu %016llx / %016llx", (unsigned long long) gpu[0], (unsigned long long) gpu[1]);
	gpulog(LOG_ERR, thr_id, "  cpu %016llx / %016llx", (unsigned long long) cpu[0], (unsigned long long) cpu[1]);
	if (gpu[0] == cpu[0])
		gpulog(LOG_ERR, thr_id, "  words agree but their nonces do not: an index/permutation bug");
	return false;
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake2s(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *resNonces;

	const uint32_t first_nonce = pdata[19];

	const int dev_id = device_map[thr_id];
	int rc = 0;
	/* 28 on both platforms. Below this the grid does not fill the device (at 25
	 * it is 128 blocks over 28 SMs, so the last wave uses 16 of them) and the
	 * per-launch cost is spread over too few nonces. Higher adds little and
	 * lengthens the batch; a long batch is discarded work on a job change. */
	int intensity = 28;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark) {
		// Loosen the target so candidates are both found and accepted. The GPU screens
		// word 7 only (h7 <= ptarget[7]); fulltest then compares the full 256 bits, so the
		// lower words must not be left at zero or every h7 == ptarget[7] hit is rejected
		// and the run looks like a GPU/CPU mismatch.
		ptarget[7] = 0x03;
		ptarget[6] = 0xffffffff;
	}

	const dim3 grid((throughput + (NPT*TPB)-1)/(NPT*TPB));
	const dim3 block(TPB);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], maxResults * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], maxResults * sizeof(uint32_t)), -1);

		/* Gates itself via cuda/selftest_gate.cuh: a device that cannot reproduce the
		 * consensus hash produces nothing but local rejects, silently, for a whole
		 * session. Overwrites the midstate; the real one is uploaded below. */
		blake2s_device_selftest(thr_id);

		init[thr_id] = true;
	}
	resNonces = h_resNonce[thr_id];

	for (int i=0; i < 19; i++) {
		be32enc(&endiandata[i], pdata[i]);
	}
	blake2s_setBlock(endiandata);

	/* -D only: GPU-vs-CPU over a nonce RANGE. The re-verify below sees only the
	 * nonces the GPU chose to report, so it cannot catch one the kernel never
	 * hashed; this can. Cheap enough to run once per job. */
	if (opt_debug)
		blake2s_check_range(thr_id, endiandata, pdata[19], 4096);

	cudaMemset(d_resNonce[thr_id], 0x00, maxResults*sizeof(uint32_t));

	do {
		blake2s_gpu_hash_nonce<<<grid, block>>>(throughput,pdata[19],d_resNonce[thr_id],ptarget[7]);
		cudaMemcpy(resNonces, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if(resNonces[0])
		{
			cudaMemcpy(resNonces, d_resNonce[thr_id], maxResults*sizeof(uint32_t), cudaMemcpyDeviceToHost);
			cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));

			if(resNonces[0] >= maxResults) {
				gpulog(LOG_WARNING, thr_id, "candidates flood: %u", resNonces[0]);
				resNonces[0] = maxResults-1;
			}

			*hashes_done = pdata[19] - first_nonce + throughput;

			/* CPU-check every slot and keep the two lowest valid nonces. The GPU
			 * screens word 7 only: h7 <= ptarget[7] failing fulltest is no reject. */
			uint32_t valid[2], vhash[2][8];
			int nvalid = 0;
			for (uint32_t j = 1; j <= resNonces[0]; j++)
			{
				uint32_t vhashcpu[8];
				const uint32_t nonce = sph_bswap32(resNonces[j]);
				be32enc(&endiandata[19], nonce);
				blake2s_hash(vhashcpu, endiandata);
				if (fulltest(vhashcpu, ptarget)) {
					if (nvalid == 0 || nonce < valid[0]) {
						valid[1] = valid[0]; memcpy(vhash[1], vhash[0], 32);
						valid[0] = nonce; memcpy(vhash[0], vhashcpu, 32);
					} else if (nvalid == 1 || nonce < valid[1]) {
						valid[1] = nonce; memcpy(vhash[1], vhashcpu, 32);
					}
					nvalid++;
				} else if (vhashcpu[7] > ptarget[7]) {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", nonce);
				}
			}

			if (nvalid) {
				if (nvalid > 2)
					gpulog(LOG_DEBUG, thr_id, "%d valid nonces, reporting the two lowest", nvalid);
				work->nonces[0] = valid[0];
				work_set_target_ratio(work, vhash[0]);
				rc = work->valid_nonces = 1;
				if (nvalid > 1) {
					work->nonces[1] = valid[1];
					bn_set_target_ratio(work, vhash[1], 1);
					rc = work->valid_nonces = 2;
				}
				// the rest lie above the resume point and are found again
				pdata[19] = valid[rc - 1] + 1;
				return rc;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	return rc;
}

// cleanup
extern "C" void free_blake2s(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
