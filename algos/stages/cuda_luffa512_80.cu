/*******************************************************************************
 * Luffa-512 for an 80-byte input (Doomcoin lineage, midstate precalc by klausT).
 *
 * The x-family 80-byte first stage for the qubit chains (qubit/deep/luffa) and
 * the mid-chain luffa-80 of x16/x21s/ghostrider/timetravel. The first two
 * message rounds and the message injection of the third (nonce) block are
 * precomputed on the host (luffa80_precalc); the kernel XORs the nonce into
 * that state, runs the rest of the third round and finalises. The 64-byte luffa lives in cuda/luffa512_device.cuh -
 * this kernel keeps its own round constants because the midstate precalc needs
 * a host-side copy anyway.
 *
 * Bare names (luffa512_setBlock_80 / luffa512_cpu_hash_80) are the real symbols;
 * the round-constant upload is folded into setBlock_80 (no separate init). The
 * qubit_luffa512_* names remain as legacy forwarders until the not-yet-migrated
 * consumers (x16/x21s/ghostrider/timetravel) adopt the bare names.
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda/helper.cuh"
#include "cuda/selftest_gate.cuh"
#include "cuda/candidate_report.cuh"

static __constant__ uint32_t c_statemi[40]; /* round-1 state after message injection, nonce word = 0 */

#define MULT2(a,j) {\
	tmp = a[7+(8*j)];\
	a[7+(8*j)] = a[6+(8*j)];\
	a[6+(8*j)] = a[5+(8*j)];\
	a[5+(8*j)] = a[4+(8*j)];\
	a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
	a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
	a[2+(8*j)] = a[1+(8*j)];\
	a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
	a[0+(8*j)] = tmp;\
}

#define TWEAK(a0,a1,a2,a3,j) { \
	a0 = (a0<<(j))|(a0>>(32-j));\
	a1 = (a1<<(j))|(a1>>(32-j));\
	a2 = (a2<<(j))|(a2>>(32-j));\
	a3 = (a3<<(j))|(a3>>(32-j));\
}

#define STEP(c0,c1) { \
	SUBCRUMB(chainv[0],chainv[1],chainv[2],chainv[3],tmp);\
	SUBCRUMB(chainv[5],chainv[6],chainv[7],chainv[4],tmp);\
	MIXWORD(chainv[0],chainv[4]);\
	MIXWORD(chainv[1],chainv[5]);\
	MIXWORD(chainv[2],chainv[6]);\
	MIXWORD(chainv[3],chainv[7]);\
	ADD_CONSTANT(chainv[0],chainv[4],c0,c1);\
}

#define SUBCRUMB(a0,a1,a2,a3,a4)\
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

#define MIXWORD(a0,a4)\
	a4 ^= a0;\
	a0  = (a0<<2) | (a0>>(30));\
	a0 ^= a4;\
	a4  = (a4<<14) | (a4>>(18));\
	a4 ^= a0;\
	a0  = (a0<<10) | (a0>>(22));\
	a0 ^= a4;\
	a4  = (a4<<1) | (a4>>(31));

#define ADD_CONSTANT(a0,b0,c0,c1)\
	a0 ^= c0;\
	b0 ^= c1;

/* round constants (device + host copy for the midstate precalc); the initial
 * chaining values are folded into luffa80_precalc's statechainv seed */
__constant__ uint32_t c_CNS[80];
static const uint32_t h_CNS[80] = {
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
void luffa80_mi(uint32_t *statebuffer, uint32_t *statechainv)
{
	int i,j;
	uint32_t t[40];
	uint32_t tmp;

	#pragma unroll 8
	for(i=0; i<8; i++) {
		t[i]=0;
		#pragma unroll 5
		for(j=0; j<5; j++)
			t[i] ^= statechainv[i+8*j];
	}

	MULT2(t, 0);

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			statechainv[i+8*j] ^= t[i];
	}

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			t[i+8*j] = statechainv[i+8*j];
	}

	#pragma unroll
	for(j=0; j<5; j++)
		MULT2(statechainv, j);

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			statechainv[8*j+i] ^= t[8*((j+1)%5)+i];
	}

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			t[i+8*j] = statechainv[i+8*j];
	}

	#pragma unroll
	for(j=0; j<5; j++)
		MULT2(statechainv, j);

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			statechainv[8*j+i] ^= t[8*((j+4)%5)+i];
	}

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			statechainv[i+8*j] ^= statebuffer[i];
		MULT2(statebuffer, 0);
	}
}

__device__ __forceinline__
void luffa80_perm(uint32_t *statechainv)
{
	int i;
	uint32_t chainv[8];
	uint32_t tmp;

	#pragma unroll
	for(i=0; i<8; i++)
		chainv[i] = statechainv[i];

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)], c_CNS[(2*i)+1]);

	#pragma unroll
	for(i=0; i<8; i++) {
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i+8];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],1);

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)+16], c_CNS[(2*i)+16+1]);

	#pragma unroll
	for(i=0; i<8; i++) {
		statechainv[i+8] = chainv[i];
		chainv[i] = statechainv[i+16];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],2);

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)+32],c_CNS[(2*i)+32+1]);

	#pragma unroll
	for(i=0; i<8; i++) {
		statechainv[i+16] = chainv[i];
		chainv[i] = statechainv[i+24];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],3);

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)+48],c_CNS[(2*i)+48+1]);

	#pragma unroll
	for(i=0; i<8; i++) {
		statechainv[i+24] = chainv[i];
		chainv[i] = statechainv[i+32];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],4);

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)+64],c_CNS[(2*i)+64+1]);

	#pragma unroll 8
	for(i=0; i<8; i++)
		statechainv[i+32] = chainv[i];
}

__device__ __forceinline__
void luffa80_rnd(uint32_t *statebuffer, uint32_t *statechainv)
{
	luffa80_mi(statebuffer, statechainv);
	luffa80_perm(statechainv);
}

static void luffa80_mi_cpu(uint32_t *statebuffer, uint32_t *statechainv)
{
	int i, j;
	uint32_t t[40];
	uint32_t tmp;

	for (i = 0; i<8; i++) {
		t[i] = statechainv[i];
		for (j = 1; j<5; j++)
			t[i] ^= statechainv[i + 8 * j];
	}

	MULT2(t, 0);

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			statechainv[i + 8 * j] ^= t[i];
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			t[i + 8 * j] = statechainv[i + 8 * j];
	}

	for (j = 0; j<5; j++)
		MULT2(statechainv, j);

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			t[i + 8 * j] = statechainv[i + 8 * j];
	}

	for (j = 0; j<5; j++)
		MULT2(statechainv, j);

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			statechainv[i + 8 * j] ^= statebuffer[i];
		MULT2(statebuffer, 0);
	}

}

static void luffa80_rnd_cpu(uint32_t *statebuffer, uint32_t *statechainv)
{
	int i;
	uint32_t chainv[8];
	uint32_t tmp;

	luffa80_mi_cpu(statebuffer, statechainv);

	for (i = 0; i<8; i++)
		chainv[i] = statechainv[i];

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i)], h_CNS[(2 * i) + 1]);

	for (i = 0; i<8; i++) {
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i) + 16], h_CNS[(2 * i) + 16 + 1]);

	for (i = 0; i<8; i++) {
		statechainv[i + 8] = chainv[i];
		chainv[i] = statechainv[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i) + 32], h_CNS[(2 * i) + 32 + 1]);

	for (i = 0; i<8; i++) {
		statechainv[i + 16] = chainv[i];
		chainv[i] = statechainv[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i) + 48], h_CNS[(2 * i) + 48 + 1]);

	for (i = 0; i<8; i++) {
		statechainv[i + 24] = chainv[i];
		chainv[i] = statechainv[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i) + 64], h_CNS[(2 * i) + 64 + 1]);

	for (i = 0; i<8; i++)
		statechainv[i + 32] = chainv[i];
}

/***************************************************/
__device__ __forceinline__
void luffa80_finalize(uint32_t* statebuffer, uint32_t *statechainv, uint32_t *b)
{
	int i,j;

	luffa80_perm(statechainv);

	/*---- blank round with m=0 ----*/
	#pragma unroll
	for(i=0; i<8; i++)
		statebuffer[i] =0;
	luffa80_rnd(statebuffer, statechainv);

	#pragma unroll
	for(i=0; i<8; i++) {
		b[i] = 0;
		#pragma unroll 5
		for(j=0; j<5; j++)
			b[i] ^= statechainv[i+8*j];
		b[i] = cuda_swab32((b[i]));
	}

	#pragma unroll
	for(i=0; i<8; i++)
		statebuffer[i]=0;
	luffa80_rnd(statebuffer, statechainv);

	#pragma unroll
	for(i=0; i<8; i++)
	{
		b[8+i] = 0;
		#pragma unroll 5
		for(j=0; j<5; j++)
			b[8+i] ^= statechainv[i+8*j];
		b[8+i] = cuda_swab32((b[8+i]));
	}
}


/***************************************************/
/* 56 instead of 64 registers; the terminal kernel below is slower with this bound */
__global__ __launch_bounds__(128)
void luffa512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;
		uint32_t statebuffer[8], statechainv[40];

		#pragma unroll
		for (int i = 0; i<40; i++)
			statechainv[i] = c_statemi[i];
		statechainv[3] ^= nounce; statechainv[12] ^= nounce; statechainv[21] ^= nounce;
		statechainv[30] ^= nounce; statechainv[39] ^= nounce;

		uint32_t *outHash = &outputHash[thread * 16];
		luffa80_finalize(statebuffer, statechainv, outHash);
	}
}

__host__
void luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash,int order)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	luffa512_gpu_hash_80 <<<grid, block, shared_size>>> (threads, startNounce, d_outputHash);
}

/***************************************************/
/* Terminal variant for a 32-byte luffa-80 hash (-a luffa): no third round, no
 * store; screens digest words 7..6 (a superset of the full compare, the host
 * re-verifies) and keeps the two lowest passing thread indices. */
__global__
void luffa512_gpu_hash_80_final(uint32_t threads, uint32_t startNounce, uint32_t *resNonce, const uint64_t target)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;
		uint32_t statebuffer[8], statechainv[40];

		#pragma unroll
		for (int i = 0; i<40; i++)
			statechainv[i] = c_statemi[i];
		statechainv[3] ^= nounce; statechainv[12] ^= nounce; statechainv[21] ^= nounce;
		statechainv[30] ^= nounce; statechainv[39] ^= nounce;

		luffa80_perm(statechainv);

		/*---- blank round with m=0 ----*/
		#pragma unroll
		for (int i = 0; i<8; i++)
			statebuffer[i] = 0;
		luffa80_rnd(statebuffer, statechainv);

		uint32_t h6 = 0, h7 = 0;
		#pragma unroll 5
		for (int j = 0; j<5; j++) {
			h6 ^= statechainv[6 + 8*j];
			h7 ^= statechainv[7 + 8*j];
		}
		const uint64_t check = ((uint64_t) cuda_swab32(h7) << 32) | cuda_swab32(h6);
		if (check <= target)
			report_candidate_2(resNonce, thread);
	}
}

__host__
void luffa512_cpu_hash_80_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_resNonce, const uint64_t target)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	luffa512_gpu_hash_80_final <<<grid, block>>> (threads, startNounce, d_resNonce, target);
}

__host__
static void luffa80_precalc(uint32_t* message)
{
	uint32_t statebuffer[8];
	uint32_t statechainv[40] =
	{
		0x6d251e69, 0x44b051e0, 0x4eaa6fb4, 0xdbf78465,
		0x6e292011, 0x90152df4, 0xee058139, 0xdef610bb,
		0xc3b44b95, 0xd9d2f256, 0x70eee9a0, 0xde099fa3,
		0x5d9b0557, 0x8fc944b3, 0xcf1ccf0e, 0x746cd581,
		0xf7efc89d, 0x5dba5781, 0x04016ce5, 0xad659c05,
		0x0306194f, 0x666d1836, 0x24aa230a, 0x8b264ae7,
		0x858075d5, 0x36d79cce, 0xe571f7d7, 0x204b1f67,
		0x35870c6a, 0x57e9e923, 0x14bcb808, 0x7cde72ce,
		0x6c68e9be, 0x5ec41e22, 0xc825b7c7, 0xaffb4363,
		0xf5df3999, 0x0fc688f1, 0xb07224cc, 0x03e86cea
	};

	for (int i = 0; i<8; i++)
		statebuffer[i] = cuda_swab32(message[i]);
	luffa80_rnd_cpu(statebuffer, statechainv);

	for (int i = 0; i<8; i++)
		statebuffer[i] = cuda_swab32(message[i+8]);

	luffa80_rnd_cpu(statebuffer, statechainv);


	/* nonce block: the injection is linear and only word 3 (the nonce) varies;
	 * the kernel XORs the nonce into word 3+j of sub-state j */
	uint32_t sb3[8] = { cuda_swab32(message[16]), cuda_swab32(message[17]), cuda_swab32(message[18]), 0, 0x80000000U, 0, 0, 0 };
	luffa80_mi_cpu(sb3, statechainv);
	cudaMemcpyToSymbol(c_statemi, statechainv, sizeof(statechainv), 0, cudaMemcpyHostToDevice);
}

__host__
void luffa512_setBlock_80(void *pdata)
{
	unsigned char PaddedMessage[128];

	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	PaddedMessage[80] = 0x80;
	PaddedMessage[111] = 1;
	PaddedMessage[126] = 0x02;
	PaddedMessage[127] = 0x80;

	/* round constants are job-invariant; the once-per-block upload here keeps the
	 * stage self-contained (no separate cpu_init to collide with the 64-byte one) */
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_CNS, h_CNS, sizeof(h_CNS), 0, cudaMemcpyHostToDevice));
	luffa80_precalc((uint32_t*) PaddedMessage);
}

/* Legacy forwarders - x16/x21s/ghostrider/timetravel still call the branded
 * qubit_luffa512_* names; removed once they migrate to the bare luffa512_*_80
 * ones. The old cpu_init only uploaded round constants (now folded into
 * setBlock_80), so it is a no-op. */
__host__ void qubit_luffa512_cpu_init(int thr_id, uint32_t threads) {}
__host__ void qubit_luffa512_cpu_setBlock_80(void *pdata){ luffa512_setBlock_80(pdata); }
__host__ void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order){
	luffa512_cpu_hash_80(thr_id, threads, startNounce, d_hash, order);
}

/* ------------------------------------------------------------------ self-test
 * Init-time check (docs/coding-guideline.md, section 7), fail-closed via
 * cuda/selftest_gate.cuh: sph against the published empty-message vector, both
 * kernels against sph over a ragged batch wrapping past 2^32, and a flipped-bit
 * negative. Overwrites the job constants: call it at init only.
 */
/* sph_types.h defines its own SPH_C32/C64/T32/T64; the ones from cuda_helper.h
 * are only needed by the device code above */
#undef SPH_C32
#undef SPH_C64
#undef SPH_T32
#undef SPH_T64
extern "C" {
#include "sph/sph_luffa.h"
}

#define LUFFA80_ST_THREADS 1000        /* not a multiple of the 256-thread block */
#define LUFFA80_ST_ALLOC   1024
#define LUFFA80_ST_START   0xFFFFFE80u /* the batch wraps past 2^32 */

/* Luffa submission appendix: Luffa-512 of the empty message. */
static const uint8_t kat_luffa512_empty[64] = {
	0x6E, 0x7D, 0xE4, 0x50, 0x11, 0x89, 0xB3, 0xCA, 0x58, 0xF3, 0xAC, 0x11, 0x49, 0x16, 0x65, 0x4B,
	0xBC, 0xD4, 0x92, 0x20, 0x24, 0xB4, 0xCC, 0x1C, 0xD7, 0x64, 0xAC, 0xFE, 0x8A, 0xB4, 0xB7, 0x80,
	0x5D, 0xF1, 0x33, 0xEA, 0xB3, 0x45, 0xFF, 0xDB, 0x1C, 0x41, 0x45, 0x64, 0xC9, 0x24, 0xF4, 0x8E,
	0x0A, 0x30, 0x18, 0x24, 0xE2, 0xAC, 0x4C, 0x34, 0xBD, 0x4E, 0xFD, 0xE2, 0xE4, 0x3D, 0xA9, 0x0E
};

/* Luffa-512 of the 80-byte pattern 00 01 .. 4F (from the anchored sph reference) */
static const uint8_t kat_luffa512_pat80[64] = {
	0x52, 0x24, 0xF8, 0xBC, 0x83, 0x35, 0xD5, 0xEA, 0x30, 0xE9, 0xAA, 0xA4, 0x15, 0xEA, 0xFB, 0x14,
	0xB4, 0x9F, 0x13, 0x92, 0x1B, 0x5A, 0xAA, 0x08, 0x5B, 0x5C, 0x9E, 0xB2, 0xBA, 0x4E, 0x68, 0x05,
	0xDF, 0xB1, 0x7B, 0x78, 0x16, 0xB2, 0x4B, 0x02, 0x7C, 0x8C, 0x8B, 0x4A, 0x1B, 0x4E, 0xFB, 0xDE,
	0x2D, 0xA2, 0x35, 0x9C, 0xC7, 0x90, 0x7E, 0x34, 0x8F, 0xB1, 0xC8, 0xE5, 0x47, 0xD5, 0x2F, 0x24
};

static void luffa80_selftest_ref(const uint8_t header[80], uint32_t nonce, uint8_t dig[64])
{
	uint8_t msg[80];
	sph_luffa512_context ctx;
	memcpy(msg, header, 80);
	be32enc((uint32_t*) &msg[76], nonce);
	sph_luffa512_init(&ctx);
	sph_luffa512(&ctx, msg, 80);
	sph_luffa512_close(&ctx, dig);
}

static bool luffa80_selftest_run(const uint8_t header[80], uint32_t *out /* [LUFFA80_ST_ALLOC * 16] */)
{
	uint32_t *d_hash = NULL;
	if (cudaMalloc(&d_hash, (size_t) LUFFA80_ST_ALLOC * 64) != cudaSuccess)
		return selftest_cuda_fault();
	bool ok = (cudaMemset(d_hash, 0xA5, (size_t) LUFFA80_ST_ALLOC * 64) == cudaSuccess);
	luffa512_setBlock_80((void*) header);
	luffa512_cpu_hash_80(0, LUFFA80_ST_THREADS, LUFFA80_ST_START, d_hash, 0);
	ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
	ok = ok && (cudaMemcpy(out, d_hash, (size_t) LUFFA80_ST_ALLOC * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_hash);
	return ok ? true : selftest_cuda_fault();
}

static bool luffa80_final_selftest_run(const uint8_t header[80], uint64_t target, uint32_t res[2])
{
	uint32_t *d_res = NULL;
	if (cudaMalloc(&d_res, 2 * sizeof(uint32_t)) != cudaSuccess)
		return selftest_cuda_fault();
	bool ok = (cudaMemset(d_res, 0xFF, 2 * sizeof(uint32_t)) == cudaSuccess);
	luffa512_setBlock_80((void*) header);
	luffa512_cpu_hash_80_final(0, LUFFA80_ST_THREADS, LUFFA80_ST_START, d_res, target);
	ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
	ok = ok && (cudaMemcpy(res, d_res, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d_res);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool luffa512_80_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	sph_luffa512_context ctx;
	uint8_t header[80], dig[64], ref0[64];
	for (int i = 0; i < 80; i++) header[i] = (uint8_t) i;

	// --- anchor sph: published empty-message vector, then the fixed 80-byte pattern ---
	sph_luffa512_init(&ctx);
	sph_luffa512(&ctx, header, 0);
	sph_luffa512_close(&ctx, dig);
	const bool sph_ok = (memcmp(dig, kat_luffa512_empty, 64) == 0);
	sph_luffa512_init(&ctx);
	sph_luffa512(&ctx, header, 80);
	sph_luffa512_close(&ctx, dig);
	const bool kat_ok = (memcmp(dig, kat_luffa512_pat80, 64) == 0);

	uint32_t *out = (uint32_t*) malloc((size_t) LUFFA80_ST_ALLOC * 64);
	if (!out) return false;

	// --- every GPU digest of the batch vs sph, then the slots past the batch ---
	bool gpu_ok = luffa80_selftest_run(header, out);
	for (uint32_t t = 0; gpu_ok && t < LUFFA80_ST_THREADS; t++) {
		luffa80_selftest_ref(header, LUFFA80_ST_START + t, dig);
		if (t == 0) memcpy(ref0, dig, 64);
		gpu_ok = (memcmp(dig, &out[t * 16], 64) == 0);
	}
	bool tail_ok = gpu_ok;
	for (uint32_t i = LUFFA80_ST_THREADS * 16; tail_ok && i < LUFFA80_ST_ALLOC * 16; i++)
		tail_ok = (out[i] == 0xA5A5A5A5u);

	// --- terminal kernel: every nonce passing must report thread indices 0 and 1;
	//     a target admitting exactly three nonces must report the lowest two ---
	bool final_ok = gpu_ok;
	if (final_ok) {
		uint32_t res[2];
		final_ok = luffa80_final_selftest_run(header, UINT64_MAX, res) && res[0] == 0 && res[1] == 1;
		uint64_t top[4] = { UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX }; /* 4 smallest, ascending */
		uint32_t idx[4] = { 0 };
		for (uint32_t t = 0; final_ok && t < LUFFA80_ST_THREADS; t++) {
			const uint64_t v = ((uint64_t) out[t * 16 + 7] << 32) | out[t * 16 + 6];
			for (int k = 0; k < 4; k++) {
				if (v < top[k]) {
					for (int m = 3; m > k; m--) { top[m] = top[m - 1]; idx[m] = idx[m - 1]; }
					top[k] = v; idx[k] = t;
					break;
				}
			}
		}
		if (final_ok) {
			uint32_t s[3] = { idx[0], idx[1], idx[2] }; /* the three passing indices, sorted */
			for (int a = 0; a < 2; a++)
				for (int b = a + 1; b < 3; b++)
					if (s[b] < s[a]) { const uint32_t x = s[a]; s[a] = s[b]; s[b] = x; }
			final_ok = (top[2] < top[3]) && luffa80_final_selftest_run(header, top[3] - 1, res)
			        && res[0] == s[0] && res[1] == s[1];
		}
	}

	// --- negative test: one flipped header bit must change the first digest ---
	header[0] ^= 0x01;
	const bool neg_ok = gpu_ok && luffa80_selftest_run(header, out)
	                 && (memcmp(&out[0], ref0, 64) != 0);

	free(out);

	passed = sph_ok && kat_ok && gpu_ok && tail_ok && final_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "luffa512_80 device self-test FAILED (sph %d kat %d gpu %d tail %d final %d neg %d)",
			(int) sph_ok, (int) kat_ok, (int) gpu_ok, (int) tail_ok, (int) final_ok, (int) neg_ok);
	else
		gpulog(LOG_DEBUG, thr_id, "luffa512_80 device self-test passed");
	return selftest_gate(thr_id, "luffa512_80", passed);
}
