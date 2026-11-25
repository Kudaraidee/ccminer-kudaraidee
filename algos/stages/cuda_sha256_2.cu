#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "miner.h"

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    (x)
#define ROTR(x, n)    (((x) >> (n)) | ((x) << (32 - (n))))
#include "cuda/helper.cuh"


static __constant__ const uint32_t H256[8] = {
	SPH_C32(0x6A09E667), SPH_C32(0xBB67AE85), SPH_C32(0x3C6EF372),
	SPH_C32(0xA54FF53A), SPH_C32(0x510E527F), SPH_C32(0x9B05688C),
	SPH_C32(0x1F83D9AB), SPH_C32(0x5BE0CD19)
};


__device__ __forceinline__
uint32_t Maj(const uint32_t a, const uint32_t b, const uint32_t c) { //Sha256 - Maj - andor
	uint32_t result;
	asm ("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(result) : "r"(a), "r"(b),"r"(c)); // 0xE8 = ((0xF0 & (0xCC | 0xAA)) | (0xCC & 0xAA))
	return result;
}

#define MAJ(X, Y, Z)   (((X) & (Y)) | (((X) | (Y)) & (Z)))


static __device__ __forceinline__ void sha2_step1(uint32_t a,uint32_t b,uint32_t c,uint32_t &d,uint32_t e,uint32_t f,uint32_t g,uint32_t &h,
	                                              uint32_t in,const uint32_t Kshared) {
	uint32_t t1,t2;
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 =ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	uint32_t bsg20 =ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	uint32_t andorv = Maj(a, b, c);		//((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

static __device__ __forceinline__ void sha2_step2(uint32_t a,uint32_t b,uint32_t c,uint32_t &d,uint32_t e,uint32_t f,uint32_t g,uint32_t &h,
	                                              uint32_t* in,uint32_t pc,const uint32_t Kshared) {
	uint32_t t1,t2;

	int pcidx1 = (pc-2) & 0xF;
	int pcidx2 = (pc-7) & 0xF;
	int pcidx3 = (pc-15) & 0xF;
	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];


	uint32_t ssg21 = ROTR(inx1, 17) ^ ROTR(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
	uint32_t ssg20 = ROTR(inx3, 7) ^ ROTR(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 =ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	uint32_t bsg20 =ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	uint32_t andorv = Maj(a, b, c);		//((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	in[pc] = ssg21+inx2+ssg20+inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d =  d + t1;
	h = t1 + t2;
}


static __device__ __forceinline__ void sha2_round_body(uint32_t* in, uint32_t* r) {
	uint32_t a = r[0];
	uint32_t b = r[1];
	uint32_t c = r[2];
	uint32_t d = r[3];
	uint32_t e = r[4];
	uint32_t f = r[5];
	uint32_t g = r[6];
	uint32_t h = r[7];

	sha2_step1(a,b,c,d,e,f,g,h,in[ 0],0x428A2F98);
	sha2_step1(h,a,b,c,d,e,f,g,in[ 1],0x71374491);
	sha2_step1(g,h,a,b,c,d,e,f,in[ 2],0xB5C0FBCF);
	sha2_step1(f,g,h,a,b,c,d,e,in[ 3],0xE9B5DBA5);
	sha2_step1(e,f,g,h,a,b,c,d,in[ 4],0x3956C25B);
	sha2_step1(d,e,f,g,h,a,b,c,in[ 5],0x59F111F1);
	sha2_step1(c,d,e,f,g,h,a,b,in[ 6],0x923F82A4);
	sha2_step1(b,c,d,e,f,g,h,a,in[ 7],0xAB1C5ED5);
	sha2_step1(a,b,c,d,e,f,g,h,in[ 8],0xD807AA98);
	sha2_step1(h,a,b,c,d,e,f,g,in[ 9],0x12835B01);
	sha2_step1(g,h,a,b,c,d,e,f,in[10],0x243185BE);
	sha2_step1(f,g,h,a,b,c,d,e,in[11],0x550C7DC3);
	sha2_step1(e,f,g,h,a,b,c,d,in[12],0x72BE5D74);
	sha2_step1(d,e,f,g,h,a,b,c,in[13],0x80DEB1FE);
	sha2_step1(c,d,e,f,g,h,a,b,in[14],0x9BDC06A7);
	sha2_step1(b,c,d,e,f,g,h,a,in[15],0xC19BF174);

	sha2_step2(a,b,c,d,e,f,g,h,in, 0,0xE49B69C1);
	sha2_step2(h,a,b,c,d,e,f,g,in, 1,0xEFBE4786);
	sha2_step2(g,h,a,b,c,d,e,f,in, 2,0x0FC19DC6);
	sha2_step2(f,g,h,a,b,c,d,e,in, 3,0x240CA1CC);
	sha2_step2(e,f,g,h,a,b,c,d,in, 4,0x2DE92C6F);
	sha2_step2(d,e,f,g,h,a,b,c,in, 5,0x4A7484AA);
	sha2_step2(c,d,e,f,g,h,a,b,in, 6,0x5CB0A9DC);
	sha2_step2(b,c,d,e,f,g,h,a,in, 7,0x76F988DA);
	sha2_step2(a,b,c,d,e,f,g,h,in, 8,0x983E5152);
	sha2_step2(h,a,b,c,d,e,f,g,in, 9,0xA831C66D);
	sha2_step2(g,h,a,b,c,d,e,f,in,10,0xB00327C8);
	sha2_step2(f,g,h,a,b,c,d,e,in,11,0xBF597FC7);
	sha2_step2(e,f,g,h,a,b,c,d,in,12,0xC6E00BF3);
	sha2_step2(d,e,f,g,h,a,b,c,in,13,0xD5A79147);
	sha2_step2(c,d,e,f,g,h,a,b,in,14,0x06CA6351);
	sha2_step2(b,c,d,e,f,g,h,a,in,15,0x14292967);

	sha2_step2(a,b,c,d,e,f,g,h,in, 0,0x27B70A85);
	sha2_step2(h,a,b,c,d,e,f,g,in, 1,0x2E1B2138);
	sha2_step2(g,h,a,b,c,d,e,f,in, 2,0x4D2C6DFC);
	sha2_step2(f,g,h,a,b,c,d,e,in, 3,0x53380D13);
	sha2_step2(e,f,g,h,a,b,c,d,in, 4,0x650A7354);
	sha2_step2(d,e,f,g,h,a,b,c,in, 5,0x766A0ABB);
	sha2_step2(c,d,e,f,g,h,a,b,in, 6,0x81C2C92E);
	sha2_step2(b,c,d,e,f,g,h,a,in, 7,0x92722C85);
	sha2_step2(a,b,c,d,e,f,g,h,in, 8,0xA2BFE8A1);
	sha2_step2(h,a,b,c,d,e,f,g,in, 9,0xA81A664B);
	sha2_step2(g,h,a,b,c,d,e,f,in,10,0xC24B8B70);
	sha2_step2(f,g,h,a,b,c,d,e,in,11,0xC76C51A3);
	sha2_step2(e,f,g,h,a,b,c,d,in,12,0xD192E819);
	sha2_step2(d,e,f,g,h,a,b,c,in,13,0xD6990624);
	sha2_step2(c,d,e,f,g,h,a,b,in,14,0xF40E3585);
	sha2_step2(b,c,d,e,f,g,h,a,in,15,0x106AA070);

	sha2_step2(a,b,c,d,e,f,g,h,in, 0,0x19A4C116);
	sha2_step2(h,a,b,c,d,e,f,g,in, 1,0x1E376C08);
	sha2_step2(g,h,a,b,c,d,e,f,in, 2,0x2748774C);
	sha2_step2(f,g,h,a,b,c,d,e,in, 3,0x34B0BCB5);
	sha2_step2(e,f,g,h,a,b,c,d,in, 4,0x391C0CB3);
	sha2_step2(d,e,f,g,h,a,b,c,in, 5,0x4ED8AA4A);
	sha2_step2(c,d,e,f,g,h,a,b,in, 6,0x5B9CCA4F);
	sha2_step2(b,c,d,e,f,g,h,a,in, 7,0x682E6FF3);
	sha2_step2(a,b,c,d,e,f,g,h,in, 8,0x748F82EE);
	sha2_step2(h,a,b,c,d,e,f,g,in, 9,0x78A5636F);
	sha2_step2(g,h,a,b,c,d,e,f,in,10,0x84C87814);
	sha2_step2(f,g,h,a,b,c,d,e,in,11,0x8CC70208);
	sha2_step2(e,f,g,h,a,b,c,d,in,12,0x90BEFFFA);
	sha2_step2(d,e,f,g,h,a,b,c,in,13,0xA4506CEB);
	sha2_step2(c,d,e,f,g,h,a,b,in,14,0xBEF9A3F7);
	sha2_step2(b,c,d,e,f,g,h,a,in,15,0xC67178F2);

	r[0] = r[0] + a;
	r[1] = r[1] + b;
	r[2] = r[2] + c;
	r[3] = r[3] + d;
	r[4] = r[4] + e;
	r[5] = r[5] + f;
	r[6] = r[6] + g;
	r[7] = r[7] + h;
}


// launch_bounds min-blocks was (512,2): the donor `,2` hint provisioned ptxas
// for only 2 resident blocks = 67% occupancy on Ampere even though this kernel
// is 40 reg / 0 smem (3 blocks of 512, or 6 of 256, all fit = 100%). (256,4)
// at tpb=256 reaches full occupancy; event-timed +~9-10% on RTX 3060 vs the
// shipped 512-thread config.
__global__ void __launch_bounds__(256,4) sha256_gpu_hash_64(int threads, uint32_t *g_hash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
    uint32_t in[16], in2[16], buf[8];
    uint32_t* inout = &g_hash[thread<<4];

    #pragma unroll
		for (int i = 0; i < 8; i++) buf[i] = H256[i];

		#pragma unroll
		for (int i = 0; i < 16; i++) in[i] = cuda_swab32(inout[i]);
		sha2_round_body(in,buf);

		in2[0] = 0x80000000;
		#pragma unroll
		for (int i = 1 ; i < 15; i++) in2[i] = 0;
		in2[15] = 0x200;
		sha2_round_body(in2,buf);

		#pragma unroll
		for (int i = 0; i < 8; i++) inout[i] = cuda_swab32(buf[i]);
	}
}


bool sha256x_device_selftest(int thr_id);

__host__
void sha256_cpu_hash_64(int thr_id, int threads, uint32_t *d_hash) {
	sha256x_device_selftest(thr_id); // stage has no init function; runs once
	// tpb 512->256: the kernel's (256,4) bound reaches 100% occupancy vs the old
	// 512/(512,2) = 67% (see kernel comment; +~9-10% event-timed).
	const int threadsperblock = 256;
	// Ceiling division: a truncating `threads/threadsperblock` drops the final
	// partial block (up to 255 tail nonces silently never hashed on a ragged
	// final scan batch) — the kernel's `if (thread < threads)` guard exists
	// precisely to make the round-up safe.
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	sha256_gpu_hash_64<<<grid, block>>>(threads, d_hash);
}


// Variant that also zeros the high 32 bytes (words 8..15) of each 64-byte slot,
// so a following 64-byte hash consumes (32 bytes digest || 32 bytes zero).
// Used by SkyDoge for its sha256 -> haval256 finalize (saves a separate launch).
// (256,4): same launch-shape retune as sha256_gpu_hash_64 (see its comment).
__global__ void __launch_bounds__(256,4) sha256_gpu_hash_64z(int threads, uint32_t *g_hash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
    uint32_t in[16], in2[16], buf[8];
    uint32_t* inout = &g_hash[thread<<4];

    #pragma unroll
		for (int i = 0; i < 8; i++) buf[i] = H256[i];

		#pragma unroll
		for (int i = 0; i < 16; i++) in[i] = cuda_swab32(inout[i]);
		sha2_round_body(in,buf);

		in2[0] = 0x80000000;
		#pragma unroll
		for (int i = 1 ; i < 15; i++) in2[i] = 0;
		in2[15] = 0x200;
		sha2_round_body(in2,buf);

		#pragma unroll
		for (int i = 0; i < 8; i++) inout[i] = cuda_swab32(buf[i]);
		#pragma unroll
		for (int i = 8; i < 16; i++) inout[i] = 0;
	}
}

__host__
void sha256_cpu_hash_64z(int thr_id, int threads, uint32_t *d_hash) {
	sha256x_device_selftest(thr_id);
	const int threadsperblock = 256; // (256,4) 100%-occ retune (see _64).
	// Ceiling division (see sha256_cpu_hash_64) — truncating grid drops tail
	// nonces; the `if (thread < threads)` guard makes the round-up safe.
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	sha256_gpu_hash_64z<<<grid, block>>>(threads, d_hash);
}

/* ------------------------------------------------------------------ self-test
 * Init-time KAT (cuda/stage_selftest.cuh) of sha256_cpu_hash_64 (high 32 bytes
 * passed through) and _64z (high 32 zeroed) against sph_sha256. `tested` is set
 * first, so the test's own launcher calls do not recurse. Fail-closed. */
extern "C" {
#include "sph/sph_sha2.h"
}
#include "cuda/stage_selftest.cuh"

static int s_sha256x_st_thr;
static int s_sha256x_st_zero;

static bool sha256x_st_gpu(const uint8_t *in, uint8_t *out, int n)
{
	uint32_t *d = NULL;
	if (cudaMalloc(&d, (size_t)n * 64) != cudaSuccess) return selftest_cuda_fault();
	bool ok = (cudaMemcpy(d, in, (size_t)n * 64, cudaMemcpyHostToDevice) == cudaSuccess);
	if (s_sha256x_st_zero) sha256_cpu_hash_64z(s_sha256x_st_thr, n, d);
	else                   sha256_cpu_hash_64(s_sha256x_st_thr, n, d);
	ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
	ok = ok && (cudaMemcpy(out, d, (size_t)n * 64, cudaMemcpyDeviceToHost) == cudaSuccess);
	cudaFree(d);
	return ok ? true : selftest_cuda_fault();
}

static void sha256x_st_ref(const uint8_t *in, uint8_t *out)
{
	sph_sha256_context c;
	sph_sha256_init(&c);
	sph_sha256(&c, in, 64);
	sph_sha256_close(&c, out);
	if (s_sha256x_st_zero) memset(out + 32, 0, 32);
	else                   memcpy(out + 32, in + 32, 32);
}

/* SHA-256 of the 64 bytes 00 01 .. 3f */
static const uint8_t kat_sha256_pat64[32] = {
	0xfd,0xea,0xb9,0xac,0xf3,0x71,0x03,0x62,0xbd,0x26,0x58,0xcd,0xc9,0xa2,0x9e,0x8f,
	0x9c,0x75,0x7f,0xcf,0x98,0x11,0x60,0x3a,0x8c,0x44,0x7c,0xd1,0xd9,0x15,0x11,0x08
};

__host__
bool sha256x_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;
	s_sha256x_st_thr = thr_id;

	uint8_t pat[64], dig[64];
	for (int i = 0; i < 64; i++) pat[i] = (uint8_t)i;
	s_sha256x_st_zero = 0;
	sha256x_st_ref(pat, dig);
	const bool anchor_ok = (memcmp(dig, kat_sha256_pat64, 32) == 0);

	const bool keep_high = stkat_run(thr_id, "sha256x", 64, 64, 256, 0x53484132u, sha256x_st_gpu, sha256x_st_ref, anchor_ok);
	s_sha256x_st_zero = 1;
	const bool zero_high = stkat_run(thr_id, "sha256x-z", 64, 64, 256, 0x53484133u, sha256x_st_gpu, sha256x_st_ref, anchor_ok);
	passed = keep_high && zero_high;
	return passed;
}
