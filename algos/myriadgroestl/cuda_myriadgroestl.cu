// Auf Myriadcoin spezialisierte Version von Groestl inkl. Bitslice (two hashes per thread)

#include <stdio.h>
#include <memory.h>

#include "cuda/helper.cuh"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#define __funnelshift_r(x,y,n) (x >> n)
#define atomicExch(p,x) x
#endif

#include "cuda/groestl512_x2_device.cuh"

/* groestl kernel: two hashes per thread (~250 registers, 64 words of shared stash per thread) */
#define G_TPB 128
#define G_MINB 2
#include "cuda/sha256_device.cuh"   /* c_sha256_K / h_sha256_K */
#include "miner.h"
#include "cuda/selftest_gate.cuh"
#include "sph/sph_groestl.h"

// globaler Speicher für alle HeftyHashes aller Threads
static uint32_t *d_outputHashes[MAX_GPUS];
static uint32_t *d_resultNonces[MAX_GPUS];

__constant__ uint32_t pTarget[2]; // Same for all GPU
__constant__ uint32_t myriadgroestl_gpu_msg[32];

/* K comes from cuda/sha256_device.cuh; this is the algo-specific K+W fold for the
 * fully constant second block, expanded at init. */
__constant__ uint32_t myr_sha256_gpu_constantTable2[64];

const uint32_t myr_sha256_cpu_w2Table[] = {
	0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
	0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
	0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549,
	0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
	0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7,
	0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
	0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484
};

#define SWAB32(x) cuda_swab32(x)

/* ROTR32 comes from cuda_helper.h (identical definition). */

#define R(x, n)         ((x) >> (n))
#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define S0(x)           (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)           (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)           (ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)           (ROTR32(x, 17) ^ ROTR32(x, 19) ^ R(x, 10))

__device__ __forceinline__
void myriadgroestl_gpu_sha256(uint32_t *message)
{
	uint32_t W1[16];
	#pragma unroll
	for(int k=0; k<16; k++)
		W1[k] = SWAB32(message[k]);

	uint32_t regs[8] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
	};

	// Progress W1
	#pragma unroll
	for(int j=0; j<16; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256_K[j] + W1[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	// Progress W2...W3
	uint32_t W2[16];

	////// PART 1
	#pragma unroll
	for(int j=0; j<2; j++)
		W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll 5
	for(int j=2; j<7;j++)
		W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll
	for(int j=7; j<15; j++)
		W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

	W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

	// Round function
	#pragma unroll
	for(int j=0; j<16; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256_K[j + 16] + W2[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	////// PART 2
	#pragma unroll
	for(int j=0; j<2; j++)
		W1[j] = s1(W2[14+j]) + W2[9+j] + s0(W2[1+j]) + W2[j];
	#pragma unroll 5
	for(int j=2; j<7; j++)
		W1[j] = s1(W1[j-2]) + W2[9+j] + s0(W2[1+j]) + W2[j];

	#pragma unroll
	for(int j=7; j<15; j++)
		W1[j] = s1(W1[j-2]) + W1[j-7] + s0(W2[1+j]) + W2[j];

	W1[15] = s1(W1[13]) + W1[8] + s0(W1[0]) + W2[15];

	// Round function
	#pragma unroll
	for(int j=0; j<16; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256_K[j + 32] + W1[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	////// PART 3
	#pragma unroll
	for(int j=0; j<2; j++)
		W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll 5
	for(int j=2; j<7; j++)
		W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll
	for(int j=7; j<15; j++)
		W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

	W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

	// Round function
	#pragma unroll
	for(int j=0; j<16; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256_K[j + 48] + W2[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	uint32_t hash[8] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
	};

	#pragma unroll 8
	for(int k=0; k<8; k++)
		hash[k] += regs[k];

	/////
	///// 2nd Round (wegen Msg-Padding)
	/////
	#pragma unroll
	for(int k=0; k<8; k++)
		regs[k] = hash[k];

	// Progress W1
	#pragma unroll
	for(int j=0; j<64; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable2[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	/* only the top two words are needed: the caller compares them against the target */
	message[6] = SWAB32(hash[6] + regs[6]);
	message[7] = SWAB32(hash[7] + regs[7]);
}

__global__
//__launch_bounds__(256, 6) // we want <= 40 regs
void myriadgroestl_gpu_hash_sha(uint32_t threads, uint32_t startNounce, uint32_t *hashBuffer, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNounce + thread;

		uint32_t out_state[16];
		uint32_t *inpHash = &hashBuffer[16 * thread];

		#pragma unroll 16
		for (int i=0; i < 16; i++)
			out_state[i] = inpHash[i];

		myriadgroestl_gpu_sha256(out_state);

		// Proper 64-bit compare. The old `s7 <= t1 && s6 <= t0` also demanded the low
		// word be under target when the high word was strictly below, losing a
		// t1*(2^32-t0)/(t1*2^32+t0) share of valid nonces — nil when t1==0, but a third
		// of them at low difficulty and nearly all under --benchmark.
		if (out_state[7] < pTarget[1] || (out_state[7] == pTarget[1] && out_state[6] <= pTarget[0]))
		{
			uint32_t tmp = atomicExch(&resNonces[0], nonce);
			if (tmp != UINT32_MAX)
				resNonces[1] = tmp;
		}
	}
}

__global__ __launch_bounds__(G_TPB, G_MINB)
void myriadgroestl_gpu_hash_x2(uint32_t threads, uint32_t startNounce, uint32_t *hashBuffer)
{
	extern __shared__ uint32_t stash[];                     /* [64][G_TPB] */
	const uint32_t i0 = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
	if (i0 >= threads) return;
	const bool two = i0 + 1 < threads;

	// GROESTL
	uint32_t m[2][20];
	#pragma unroll
	for (int w = 0; w < 19; w++) m[0][w] = m[1][w] = myriadgroestl_gpu_msg[w];
	m[0][19] = SWAB32(startNounce + i0);
	m[1][19] = SWAB32(startNounce + i0 + 1);

	uint32_t s[8][8], out_state[2][16];
	groestl512_x2_load80(m, s);
	groestl512_x2_compress(s, &stash[threadIdx.x], blockDim.x);
	groestl512_x2_store(s, out_state);

	#pragma unroll
	for (int h = 0; h < 2; h++) {
		if (h == 0 || two) {
			uint4 *outpHash = (uint4*) &hashBuffer[16 * (i0 + h)];
			#pragma unroll
			for (int k = 0; k < 4; k++)
				outpHash[k] = make_uint4(out_state[h][4*k], out_state[h][4*k+1], out_state[h][4*k+2], out_state[h][4*k+3]);
		}
	}
}

/* Init self-test of the Groestl kernel: 3 nonces (a thread's two + a lone one) vs sph, the next
 * slot untouched, and a flipped header bit must change the digest */
static bool myriadgroestl_selftest_run(const uint32_t *hdr20, uint32_t start, uint32_t *out64)
{
	uint32_t msgBlock[32] = { 0 };
	memcpy(msgBlock, hdr20, 80);
	uint32_t *d_out = NULL;
	if (cudaMemcpyToSymbol(myriadgroestl_gpu_msg, msgBlock, 128) != cudaSuccess
	 || cudaMalloc(&d_out, 4 * 64) != cudaSuccess)
		return selftest_cuda_fault();
	bool ok = cudaMemset(d_out, 0x5a, 4 * 64) == cudaSuccess;
	myriadgroestl_gpu_hash_x2 <<< 1, G_TPB, 64 * G_TPB * sizeof(uint32_t) >>> (3, start, d_out);
	ok = ok && cudaDeviceSynchronize() == cudaSuccess
	        && cudaMemcpy(out64, d_out, 4 * 64, cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d_out);
	return ok ? true : selftest_cuda_fault();
}

__host__
bool myriadgroestl_device_selftest(int thr_id)
{
	uint32_t hdr[20], gpu[4][16], gpu_flipped[4][16];
	const uint32_t start = 0xdeadbeefu;
	bool kat = false, tail = false, neg = false;

	for (int i = 0; i < 20; i++)
		hdr[i] = 0x04030201u * (uint32_t)(i + 1);

	if (myriadgroestl_selftest_run(hdr, start, &gpu[0][0])) {
		kat = true;
		for (int v = 0; v < 3; v++) {
			uint32_t m80[20], ref[16];
			memcpy(m80, hdr, 76);
			m80[19] = swab32(start + v);
			sph_groestl512_context ctx;
			sph_groestl512_init(&ctx);
			sph_groestl512(&ctx, m80, 80);
			sph_groestl512_close(&ctx, ref);
			kat = kat && memcmp(gpu[v], ref, 64) == 0;
		}
		tail = true;
		for (int w = 0; w < 16; w++) tail = tail && gpu[3][w] == 0x5a5a5a5au;
	}

	hdr[0] ^= 1u;
	if (myriadgroestl_selftest_run(hdr, start, &gpu_flipped[0][0]))
		neg = memcmp(gpu_flipped[0], gpu[0], 64) != 0;
	hdr[0] ^= 1u;

	const bool passed = kat && tail && neg;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "myr-gr self-test FAILED (kat %d tail %d neg %d)",
			(int)kat, (int)tail, (int)neg);
	else
		gpulog(LOG_DEBUG, thr_id, "myr-gr self-test passed");

	return selftest_gate(thr_id, "myr-gr", passed);
}

// Setup Function
__host__
void myriadgroestl_cpu_init(int thr_id, uint32_t threads)
{
	uint32_t temp[64];
	for(int i=0; i<64; i++)
		temp[i] = myr_sha256_cpu_w2Table[i] + h_sha256_K[i];

	cudaMemcpyToSymbol( myr_sha256_gpu_constantTable2, temp, sizeof(uint32_t) * 64 );

	cuda_get_arch(thr_id);

	cudaMalloc(&d_outputHashes[thr_id], (size_t) 64 * threads);
	cudaMalloc(&d_resultNonces[thr_id], 2 * sizeof(uint32_t));

	myriadgroestl_device_selftest(thr_id);
}

__host__
void myriadgroestl_cpu_free(int thr_id)
{
	cudaFree(d_outputHashes[thr_id]);
	cudaFree(d_resultNonces[thr_id]);
}

__host__
void myriadgroestl_cpu_setBlock(int thr_id, void *data, uint32_t *pTargetIn)
{
	uint32_t msgBlock[32] = { 0 };
	memcpy(&msgBlock[0], data, 80);
	msgBlock[20] = 0x80;
	msgBlock[31] = 0x01000000;

	cudaMemcpyToSymbol(myriadgroestl_gpu_msg, msgBlock, 128);
	cudaMemcpyToSymbol(pTarget, &pTargetIn[6], 2 * sizeof(uint32_t));
}

__host__
void myriadgroestl_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	uint32_t threadsperblock = 256;

	cudaMemset(d_resultNonces[thr_id], 0xFF, 2 * sizeof(uint32_t));

	const uint32_t nthr = (threads + 1) / 2;   /* two hashes per groestl thread */
	dim3 grid((nthr + G_TPB - 1) / G_TPB);
	myriadgroestl_gpu_hash_x2 <<< grid, G_TPB, 64 * G_TPB * sizeof(uint32_t) >>> (threads, startNounce, d_outputHashes[thr_id]);

	dim3 block(threadsperblock);
	dim3 grid2((threads + threadsperblock-1)/threadsperblock);
	myriadgroestl_gpu_hash_sha <<< grid2, block >>> (threads, startNounce, d_outputHashes[thr_id], d_resultNonces[thr_id]);

	cudaMemcpy(resNounce, d_resultNonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
