/*
Based on SP's BMW kernel
Provos Alexis - 2016
Optimized for pascal sp - may 2018
*/

#include <stdio.h>
#include <memory.h>

#define WANT_BMW512_80

#include "cuda/helper_alexis.cuh"
#include "cuda/vectors_alexis.cuh"
#include "cuda/bmw512_device.cuh"
#include "cuda/candidate_report.cuh"

// The BMW-512 device library (first-round + final compression for 64-byte
// inputs) lives in cuda/bmw512_device.cuh; the kernels below are thin
// wrappers / the 80-byte first-stage paths.

__global__ __launch_bounds__(256, 2)
void bmw512_gpu_hash_64(uint32_t threads, const uint32_t startNounce, uint64_t *const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){

		/* branch nonce vectors (quark/anime compaction) store absolute nonces,
		 * so map back to the hash-buffer slot: subtract startNounce (matches
		 * skein/jh/groestl). NULL vector => the slot is just the thread index. */
		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread] - startNounce;

		uint64_t *inpHash = &g_hash[8 * hashPosition];

		uint64_t __align__(16) msg[16];

		uint2x4* phash = (uint2x4*)inpHash;
		uint2x4* outpt = (uint2x4*)msg;
		outpt[0] = __ldg4(&phash[0]);
		outpt[1] = __ldg4(&phash[1]);

		bmw512_hash_64(msg);

		*(uint2x4*)&inpHash[0] = *(uint2x4*)&msg[8];
		*(uint2x4*)&inpHash[4] = *(uint2x4*)&msg[12];
	}
}

/* Unit self-test for cuda/bmw512_device.cuh (docs/coding-guideline.md section 7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool bmw512_device_selftest(int thr_id);

__host__
void bmw512_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
	bmw512_device_selftest(thr_id);
}



__host__ void bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	bmw512_gpu_hash_64 << <grid, block >> >(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
}


__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)


#undef SHL
#undef SHR
#undef CONST_EXP2

#define SHR(x, n) SHR2(x, n)
#define SHL(x, n) SHL2(x, n)
#define ROL(x, n) ROL2(x, n)


#define CONST_EXP2(i) \
	q[i+0] + ROL(q[i+1], 5)  + q[i+2] + ROL(q[i+3], 11) + \
	q[i+4] + ROL(q[i+5], 27) + q[i+6] + SWAPUINT2(q[i+7]) + \
	q[i+8] + ROL(q[i+9], 37) + q[i+10] + ROL(q[i+11], 43) + \
	q[i+12] + ROL(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

__device__
void Compression512(uint2 *msg, uint2 *hash)
{
	// Compression ref. implementation
	uint2 q[32];
	uint2 tmp;

	tmp = (msg[5] ^ hash[5]) - (msg[7] ^ hash[7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[1];
	tmp = (msg[6] ^ hash[6]) - (msg[8] ^ hash[8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
	q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[2];
	tmp = (msg[0] ^ hash[0]) + (msg[7] ^ hash[7]) + (msg[9] ^ hash[9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[3];
	tmp = (msg[0] ^ hash[0]) - (msg[1] ^ hash[1]) + (msg[8] ^ hash[8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
	q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[4];
	tmp = (msg[1] ^ hash[1]) + (msg[2] ^ hash[2]) + (msg[9] ^ hash[9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
	q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
	tmp = (msg[3] ^ hash[3]) - (msg[2] ^ hash[2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[6];
	tmp = (msg[4] ^ hash[4]) - (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
	q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[7];
	tmp = (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[5] ^ hash[5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
	q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[8];
	tmp = (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) - (msg[6] ^ hash[6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
	q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[9];
	tmp = (msg[0] ^ hash[0]) - (msg[3] ^ hash[3]) + (msg[6] ^ hash[6]) - (msg[7] ^ hash[7]) + (msg[14] ^ hash[14]);
	q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
	tmp = (msg[8] ^ hash[8]) - (msg[1] ^ hash[1]) - (msg[4] ^ hash[4]) - (msg[7] ^ hash[7]) + (msg[15] ^ hash[15]);
	q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[11];
	tmp = (msg[8] ^ hash[8]) - (msg[0] ^ hash[0]) - (msg[2] ^ hash[2]) - (msg[5] ^ hash[5]) + (msg[9] ^ hash[9]);
	q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[12];
	tmp = (msg[1] ^ hash[1]) + (msg[3] ^ hash[3]) - (msg[6] ^ hash[6]) - (msg[9] ^ hash[9]) + (msg[10] ^ hash[10]);
	q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[13];
	tmp = (msg[2] ^ hash[2]) + (msg[4] ^ hash[4]) + (msg[7] ^ hash[7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
	q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[14];
	tmp = (msg[3] ^ hash[3]) - (msg[5] ^ hash[5]) + (msg[8] ^ hash[8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
	q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[4] ^ hash[4]) - (msg[6] ^ hash[6]) - (msg[9] ^ hash[9]) + (msg[13] ^ hash[13]);
	q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[0];

	q[0 + 16] =
		(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROL(q[0], 13) ^ ROL(q[0], 43)) +
		(SHR(q[0 + 1], 2) ^ SHL(q[0 + 1], 1) ^ ROL(q[0 + 1], 19) ^ ROL(q[0 + 1], 53)) +
		(SHR(q[0 + 2], 2) ^ SHL(q[0 + 2], 2) ^ ROL(q[0 + 2], 28) ^ ROL(q[0 + 2], 59)) +
		(SHR(q[0 + 3], 1) ^ SHL(q[0 + 3], 3) ^ ROL(q[0 + 3], 4) ^ ROL(q[0 + 3], 37)) +
		(SHR(q[0 + 4], 1) ^ SHL(q[0 + 4], 2) ^ ROL(q[0 + 4], 13) ^ ROL(q[0 + 4], 43)) +
		(SHR(q[0 + 5], 2) ^ SHL(q[0 + 5], 1) ^ ROL(q[0 + 5], 19) ^ ROL(q[0 + 5], 53)) +
		(SHR(q[0 + 6], 2) ^ SHL(q[0 + 6], 2) ^ ROL(q[0 + 6], 28) ^ ROL(q[0 + 6], 59)) +
		(SHR(q[0 + 7], 1) ^ SHL(q[0 + 7], 3) ^ ROL(q[0 + 7], 4) ^ ROL(q[0 + 7], 37)) +
		(SHR(q[0 + 8], 1) ^ SHL(q[0 + 8], 2) ^ ROL(q[0 + 8], 13) ^ ROL(q[0 + 8], 43)) +
		(SHR(q[0 + 9], 2) ^ SHL(q[0 + 9], 1) ^ ROL(q[0 + 9], 19) ^ ROL(q[0 + 9], 53)) +
		(SHR(q[0 + 10], 2) ^ SHL(q[0 + 10], 2) ^ ROL(q[0 + 10], 28) ^ ROL(q[0 + 10], 59)) +
		(SHR(q[0 + 11], 1) ^ SHL(q[0 + 11], 3) ^ ROL(q[0 + 11], 4) ^ ROL(q[0 + 11], 37)) +
		(SHR(q[0 + 12], 1) ^ SHL(q[0 + 12], 2) ^ ROL(q[0 + 12], 13) ^ ROL(q[0 + 12], 43)) +
		(SHR(q[0 + 13], 2) ^ SHL(q[0 + 13], 1) ^ ROL(q[0 + 13], 19) ^ ROL(q[0 + 13], 53)) +
		(SHR(q[0 + 14], 2) ^ SHL(q[0 + 14], 2) ^ ROL(q[0 + 14], 28) ^ ROL(q[0 + 14], 59)) +
		(SHR(q[0 + 15], 1) ^ SHL(q[0 + 15], 3) ^ ROL(q[0 + 15], 4) ^ ROL(q[0 + 15], 37)) +
		((make_uint2(0x55555550ul, 0x55555555) + ROL(msg[0], 0 + 1) +
		ROL(msg[0 + 3], 0 + 4) - ROL(msg[0 + 10], 0 + 11)) ^ hash[0 + 7]);

	q[1 + 16] =
		(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROL(q[1], 13) ^ ROL(q[1], 43)) +
		(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROL(q[1 + 1], 19) ^ ROL(q[1 + 1], 53)) +
		(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROL(q[1 + 2], 28) ^ ROL(q[1 + 2], 59)) +
		(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROL(q[1 + 3], 4) ^ ROL(q[1 + 3], 37)) +
		(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROL(q[1 + 4], 13) ^ ROL(q[1 + 4], 43)) +
		(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROL(q[1 + 5], 19) ^ ROL(q[1 + 5], 53)) +
		(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROL(q[1 + 6], 28) ^ ROL(q[1 + 6], 59)) +
		(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROL(q[1 + 7], 4) ^ ROL(q[1 + 7], 37)) +
		(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROL(q[1 + 8], 13) ^ ROL(q[1 + 8], 43)) +
		(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROL(q[1 + 9], 19) ^ ROL(q[1 + 9], 53)) +
		(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROL(q[1 + 10], 28) ^ ROL(q[1 + 10], 59)) +
		(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROL(q[1 + 11], 4) ^ ROL(q[1 + 11], 37)) +
		(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROL(q[1 + 12], 13) ^ ROL(q[1 + 12], 43)) +
		(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROL(q[1 + 13], 19) ^ ROL(q[1 + 13], 53)) +
		(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROL(q[1 + 14], 28) ^ ROL(q[1 + 14], 59)) +
		(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROL(q[1 + 15], 4) ^ ROL(q[1 + 15], 37)) +
		((make_uint2(0xAAAAAAA5, 0x5AAAAAAA) + ROL(msg[1], 1 + 1) +
		ROL(msg[1 + 3], 1 + 4) - ROL(msg[1 + 10], 1 + 11)) ^ hash[1 + 7]);

	q[2 + 16] = CONST_EXP2(2) +
		((make_uint2(0xFFFFFFFA, 0x5FFFFFFF) + ROL(msg[2], 2 + 1) +
		ROL(msg[2 + 3], 2 + 4) - ROL(msg[2 + 10], 2 + 11)) ^ hash[2 + 7]);
	q[3 + 16] = CONST_EXP2(3) +
		((make_uint2(0x5555554F, 0x65555555) + ROL(msg[3], 3 + 1) +
		ROL(msg[3 + 3], 3 + 4) - ROL(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((make_uint2(0xAAAAAAA4, 0x6AAAAAAA) + ROL(msg[4], 4 + 1) +
		ROL(msg[4 + 3], 4 + 4) - ROL(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	q[5 + 16] = CONST_EXP2(5) +
		((make_uint2(0xFFFFFFF9, 0x6FFFFFFF) + ROL(msg[5], 5 + 1) +
		ROL(msg[5 + 3], 5 + 4) - ROL(msg[5 + 10], 5 + 11)) ^ hash[5 + 7]);
	q[6 + 16] = CONST_EXP2(6) +
		((make_uint2(0x5555554E, 0x75555555) + ROL(msg[6], 6 + 1) +
		ROL(msg[6 + 3], 6 + 4) - ROL(msg[6 - 6], (6 - 6) + 1)) ^ hash[6 + 7]);
	q[7 + 16] = CONST_EXP2(7) +
		((make_uint2(0xAAAAAAA3, 0x7AAAAAAA) + ROL(msg[7], 7 + 1) +
		ROL(msg[7 + 3], 7 + 4) - ROL(msg[7 - 6], (7 - 6) + 1)) ^ hash[7 + 7]);
	q[8 + 16] = CONST_EXP2(8) +
		((make_uint2(0xFFFFFFF8, 0x7FFFFFFF) + ROL(msg[8], 8 + 1) +
		ROL(msg[8 + 3], 8 + 4) - ROL(msg[8 - 6], (8 - 6) + 1)) ^ hash[8 + 7]);
	q[9 + 16] = CONST_EXP2(9) +
		((make_uint2(0x5555554D, 0x85555555) + ROL(msg[9], 9 + 1) +
		ROL(msg[9 + 3], 9 + 4) - ROL(msg[9 - 6], (9 - 6) + 1)) ^ hash[9 - 9]);
	q[10 + 16] = CONST_EXP2(10) +
		((make_uint2(0xAAAAAAA2, 0x8AAAAAAA) + ROL(msg[10], 10 + 1) +
		ROL(msg[10 + 3], 10 + 4) - ROL(msg[10 - 6], (10 - 6) + 1)) ^ hash[10 - 9]);
	q[11 + 16] = CONST_EXP2(11) +
		((make_uint2(0xFFFFFFF7, 0x8FFFFFFF) + ROL(msg[11], 11 + 1) +
		ROL(msg[11 + 3], 11 + 4) - ROL(msg[11 - 6], (11 - 6) + 1)) ^ hash[11 - 9]);
	q[12 + 16] = CONST_EXP2(12) +
		((make_uint2(0x5555554C, 0x95555555) + ROL(msg[12], 12 + 1) +
		ROL(msg[12 + 3], 12 + 4) - ROL(msg[12 - 6], (12 - 6) + 1)) ^ hash[12 - 9]);
	q[13 + 16] = CONST_EXP2(13) +
		((make_uint2(0xAAAAAAA1, 0x9AAAAAAA) + ROL(msg[13], 13 + 1) +
		ROL(msg[13 - 13], (13 - 13) + 1) - ROL(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((make_uint2(0xFFFFFFF6, 0x9FFFFFFF) + ROL(msg[14], 14 + 1) +
		ROL(msg[14 - 13], (14 - 13) + 1) - ROL(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((make_uint2(0x5555554B, 0xA5555555) + ROL(msg[15], 15 + 1) +
		ROL(msg[15 - 13], (15 - 13) + 1) - ROL(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);

	uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ msg[0]) + (XL64 ^ q[24] ^ q[0]);
	hash[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ msg[1]) + (XL64 ^ q[25] ^ q[1]);
	hash[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ msg[2]) + (XL64 ^ q[26] ^ q[2]);
	hash[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ msg[3]) + (XL64 ^ q[27] ^ q[3]);
	hash[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + (XL64 ^ q[28] ^ q[4]);
	hash[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ msg[5]) + (XL64 ^ q[29] ^ q[5]);
	hash[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ msg[6]) + (XL64 ^ q[30] ^ q[6]);
	hash[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ msg[7]) + (XL64 ^ q[31] ^ q[7]);

	hash[8] = ROL(hash[4], 9) + (XH64 ^ q[24] ^ msg[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);
	hash[9] = ROL(hash[5], 10) + (XH64 ^ q[25] ^ msg[9]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
	hash[10] = ROL(hash[6], 11) + (XH64 ^ q[26] ^ msg[10]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
	hash[11] = ROL(hash[7], 12) + (XH64 ^ q[27] ^ msg[11]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
	hash[12] = ROL(hash[0], 13) + (XH64 ^ q[28] ^ msg[12]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
	hash[13] = ROL(hash[1], 14) + (XH64 ^ q[29] ^ msg[13]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
	hash[14] = ROL(hash[2], 15) + (XH64 ^ q[30] ^ msg[14]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
	hash[15] = ROL(hash[3], 16) + (XH64 ^ q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}



__global__ __launch_bounds__(256, 2)
void bmw512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;

		// Init
		uint2 h[16] = {
			{ 0x84858687UL, 0x80818283UL },
			{ 0x8C8D8E8FUL, 0x88898A8BUL },
			{ 0x94959697UL, 0x90919293UL },
			{ 0x9C9D9E9FUL, 0x98999A9BUL },
			{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
			{ 0xACADAEAFUL, 0xA8A9AAABUL },
			{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
			{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
			{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
			{ 0xCCCDCECFUL, 0xC8C9CACBUL },
			{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
			{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
			{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
			{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
			{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
			{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
		};
		// Nachricht kopieren (Achtung, die Nachricht hat 64 Byte,
		// BMW arbeitet mit 128 Byte!!!
		uint2 message[16];
#pragma unroll 16
		for (int i = 0; i<16; i++)
			message[i] = vectorize(c_PaddedMessage80[i]);

		// die Nounce durch die thread-spezifische ersetzen
		message[9].y = cuda_swab32(nounce);	//REPLACE_HIDWORD(message[9], cuda_swab32(nounce));

		// Compression 1
		Compression512(message, h);
#pragma unroll 16
		for (int i = 0; i<16; i++)
			message[i] = make_uint2(0xaaaaaaa0 + i, 0xaaaaaaaa);


		Compression512(h, message);

		// fertig
		uint64_t *outpHash = &g_hash[thread * 8];

#pragma unroll 8
		for (int i = 0; i<8; i++)
			outpHash[i] = devectorize(message[i + 8]);
	}
}


__host__
void bmw512_cpu_setBlock_80(void *pdata)
{
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage + 80, 0, 48);
	uint64_t *message = (uint64_t*)PaddedMessage;
	message[10] = SPH_C64(0x80);
	message[15] = SPH_C64(640);
	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}



__host__
void bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	bmw512_gpu_hash_80 << <grid, block >> >(threads, startNounce, (uint64_t*)d_hash);
}

/* First compression of an 80-byte header with its nonce-free work done once per job on the
 * host. Only word 9 (high half = nonce) varies; the W_j that read it are j = 2, 4, 11, 12, 15,
 * and the AddElement terms that rotate it are j = 22, 25, 31.
 *   [ 0..15] q[j], or for those five j the partial sum without word 9
 *   [16..31] the constant part of q[16..31]
 *   [32..34] AddElement(22, 25, 31) without word 9, before the XOR */
__constant__ uint64_t c_bmw512_pre[35];

#define BMW512_PRE_D(i) ((i) == 2 || (i) == 4 || (i) == 11 || (i) == 12 || (i) == 15)
#define bmw512_iv_dev(i) (0x8081828384858687ull + (uint64_t) (i) * 0x0808080808080808ull)

static const uint64_t bmw512_iv[16] = {
	0x8081828384858687, 0x88898A8B8C8D8E8F, 0x9091929394959697, 0x98999A9B9C9D9E9F,
	0xA0A1A2A3A4A5A6A7, 0xA8A9AAABACADAEAF, 0xB0B1B2B3B4B5B6B7, 0xB8B9BABBBCBDBEBF,
	0xC0C1C2C3C4C5C6C7, 0xC8C9CACBCCCDCECF, 0xD0D1D2D3D4D5D6D7, 0xD8D9DADBDCDDDEDF,
	0xE0E1E2E3E4E5E6E7, 0xE8E9EAEBECEDEEEF, 0xF0F1F2F3F4F5F6F7, 0xF8F9FAFBFCFDFEFF
};

/* f0's W_j as (index, sign) - the BMW specification's table */
static const int8_t bmw512_w_idx[16][5] = {
	{ 5, 7, 10, 13, 14 }, { 6, 8, 11, 14, 15 }, { 0, 7, 9, 12, 15 }, { 0, 1, 8, 10, 13 },
	{ 1, 2, 9, 11, 14 }, { 3, 2, 10, 12, 15 }, { 4, 0, 3, 11, 13 }, { 1, 4, 5, 12, 14 },
	{ 2, 5, 6, 13, 15 }, { 0, 3, 6, 7, 14 }, { 8, 1, 4, 7, 15 }, { 8, 0, 2, 5, 9 },
	{ 1, 3, 6, 9, 10 }, { 2, 4, 7, 10, 11 }, { 3, 5, 8, 11, 12 }, { 12, 4, 6, 9, 13 }
};
static const int8_t bmw512_w_sgn[16][5] = {
	{ 1, -1, 1, 1, 1 }, { 1, -1, 1, 1, -1 }, { 1, 1, 1, -1, 1 }, { 1, -1, 1, -1, 1 },
	{ 1, 1, 1, -1, -1 }, { 1, -1, 1, -1, 1 }, { 1, -1, -1, -1, 1 }, { 1, -1, -1, -1, -1 },
	{ 1, -1, -1, 1, -1 }, { 1, -1, 1, -1, 1 }, { 1, -1, -1, -1, 1 }, { 1, -1, -1, -1, 1 },
	{ 1, 1, -1, -1, 1 }, { 1, 1, 1, 1, 1 }, { 1, -1, 1, -1, -1 }, { 1, -1, -1, -1, 1 }
};

static inline uint64_t bmw512_rol(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

/* s0..s5 of the specification, 64-bit variant */
static uint64_t bmw512_s(int k, uint64_t x)
{
	switch (k) {
	case 0: return (x >> 1) ^ (x << 3) ^ bmw512_rol(x, 4) ^ bmw512_rol(x, 37);
	case 1: return (x >> 1) ^ (x << 2) ^ bmw512_rol(x, 13) ^ bmw512_rol(x, 43);
	case 2: return (x >> 2) ^ (x << 1) ^ bmw512_rol(x, 19) ^ bmw512_rol(x, 53);
	case 3: return (x >> 2) ^ (x << 2) ^ bmw512_rol(x, 28) ^ bmw512_rol(x, 59);
	case 4: return (x >> 1) ^ x;
	default: return (x >> 2) ^ x;
	}
}

/* expansion term k of q[j] (k = 0..15 reads q[j - 16 + k]) */
static uint64_t bmw512_exp_term(int j, int k, uint64_t q)
{
	static const int r[7] = { 5, 11, 27, 32, 37, 43, 53 };
	if (j < 18) return bmw512_s((k + 1) % 4, q);  // expand1: s1 s2 s3 s0 ...
	if (k == 14) return bmw512_s(4, q);
	if (k == 15) return bmw512_s(5, q);
	return (k & 1) ? bmw512_rol(q, r[k / 2]) : q;  // expand2
}

/* AddElement(j) without its XOR, word 9 omitted when skip9 */
static uint64_t bmw512_add_element(int j, const uint64_t *m, bool skip9)
{
	const int a = j - 16, b = (j - 13) % 16, c = (j - 6) % 16;
	uint64_t v = 0x0555555555555555ull * (uint64_t) j;
	if (!(skip9 && a == 9)) v += bmw512_rol(m[a], a + 1);
	if (!(skip9 && b == 9)) v += bmw512_rol(m[b], b + 1);
	if (!(skip9 && c == 9)) v -= bmw512_rol(m[c], c + 1);
	return v;
}

/* bmw512_cpu_setBlock_80 plus the precomputed terms; pdata = the 20 be32enc'd header words. */
__host__
void bmw512_final_setBlock_80(void *pdata)
{
	bmw512_cpu_setBlock_80(pdata);

	uint64_t m[16], q[16], pre[35];
	memcpy(m, pdata, 80);
	m[9] = 0; // never read below: every use of word 9 is left to the device
	m[10] = 0x80; m[11] = m[12] = m[13] = m[14] = 0; m[15] = 640;

	for (int j = 0; j < 16; j++) {
		uint64_t t = 0;
		for (int k = 0; k < 5; k++) {
			const int i = bmw512_w_idx[j][k];
			if (i == 9) continue;
			const uint64_t v = m[i] ^ bmw512_iv[i];
			t += (bmw512_w_sgn[j][k] > 0) ? v : (0 - v);
		}
		q[j] = BMW512_PRE_D(j) ? 0 : bmw512_s(j % 5, t) + bmw512_iv[(j + 1) % 16];
		pre[j] = BMW512_PRE_D(j) ? t : q[j];
	}
	int ae = 32;
	for (int j = 16; j < 32; j++) {
		uint64_t p = 0;
		for (int k = 0; k < 16; k++) {
			const int i = j - 16 + k;
			if (i < 16 && !BMW512_PRE_D(i))
				p += bmw512_exp_term(j, k, q[i]);
		}
		if (j == 22 || j == 25 || j == 31)
			pre[ae++] = bmw512_add_element(j, m, true);
		else
			p += bmw512_add_element(j, m, false) ^ bmw512_iv[(j - 9) % 16];
		pre[j] = p;
	}
	cudaMemcpyToSymbol(c_bmw512_pre, pre, sizeof(pre), 0, cudaMemcpyHostToDevice);
}

__device__ __forceinline__
static uint2 bmw512_dev_s(const int k, const uint2 x)
{
	switch (k) {
	case 0: return SHR(x, 1) ^ SHL(x, 3) ^ ROL(x, 4) ^ ROL(x, 37);
	case 1: return SHR(x, 1) ^ SHL(x, 2) ^ ROL(x, 13) ^ ROL(x, 43);
	case 2: return SHR(x, 2) ^ SHL(x, 1) ^ ROL(x, 19) ^ ROL(x, 53);
	case 3: return SHR(x, 2) ^ SHL(x, 2) ^ ROL(x, 28) ^ ROL(x, 59);
	case 4: return SHR(x, 1) ^ x;
	default: return SHR(x, 2) ^ x;
	}
}

__device__ __forceinline__
static uint2 bmw512_dev_exp_term(const int j, const int k, const uint2 q)
{
	if (j < 18) return bmw512_dev_s((k + 1) % 4, q);
	if (k == 14) return bmw512_dev_s(4, q);
	if (k == 15) return bmw512_dev_s(5, q);
	if (!(k & 1)) return q;
	switch (k / 2) {
	case 0: return ROL(q, 5);
	case 1: return ROL(q, 11);
	case 2: return ROL(q, 27);
	case 3: return SWAPUINT2(q);
	case 4: return ROL(q, 37);
	case 5: return ROL(q, 43);
	default: return ROL(q, 53);
	}
}

/* Compression512(msg, IV) with c_bmw512_pre; same output in hash[] (the IV on entry is not read). */
__device__ __forceinline__
static void Compression512_80_pre(const uint2 *msg, uint2 *hash)
{
	uint2 q[32];
	const uint2 x9 = msg[9] ^ vectorize(bmw512_iv_dev(9));

	#pragma unroll
	for (int j = 0; j < 16; j++) {
		if (!BMW512_PRE_D(j)) q[j] = vectorize(c_bmw512_pre[j]);
	}
	q[2]  = bmw512_dev_s(2, vectorize(c_bmw512_pre[2]) + x9)  + vectorize(bmw512_iv_dev(3));
	q[4]  = bmw512_dev_s(4, vectorize(c_bmw512_pre[4]) + x9)  + vectorize(bmw512_iv_dev(5));
	q[11] = bmw512_dev_s(1, vectorize(c_bmw512_pre[11]) + x9) + vectorize(bmw512_iv_dev(12));
	q[12] = bmw512_dev_s(2, vectorize(c_bmw512_pre[12]) - x9) + vectorize(bmw512_iv_dev(13));
	q[15] = bmw512_dev_s(0, vectorize(c_bmw512_pre[15]) - x9) + vectorize(bmw512_iv_dev(0));

	const uint2 r9 = ROL(msg[9], 10);
	#pragma unroll
	for (int j = 16; j < 32; j++) {
		uint2 v = vectorize(c_bmw512_pre[j]);
		#pragma unroll
		for (int k = 0; k < 16; k++) {
			const int i = j - 16 + k;
			if (i >= 16 || BMW512_PRE_D(i))
				v = v + bmw512_dev_exp_term(j, k, q[i]);
		}
		if (j == 22) v = v + ((vectorize(c_bmw512_pre[32]) + r9) ^ vectorize(bmw512_iv_dev(13)));
		if (j == 25) v = v + ((vectorize(c_bmw512_pre[33]) + r9) ^ vectorize(bmw512_iv_dev(0)));
		if (j == 31) v = v + ((vectorize(c_bmw512_pre[34]) - r9) ^ vectorize(bmw512_iv_dev(6)));
		q[j] = v;
	}

	uint2 XL64 = q[16] ^ q[17] ^ q[18] ^ q[19] ^ q[20] ^ q[21] ^ q[22] ^ q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL(XH64, 5) ^ SHR(q[16], 5) ^ msg[0]) + (XL64 ^ q[24] ^ q[0]);
	hash[1] = (SHR(XH64, 7) ^ SHL(q[17], 8) ^ msg[1]) + (XL64 ^ q[25] ^ q[1]);
	hash[2] = (SHR(XH64, 5) ^ SHL(q[18], 5) ^ msg[2]) + (XL64 ^ q[26] ^ q[2]);
	hash[3] = (SHR(XH64, 1) ^ SHL(q[19], 5) ^ msg[3]) + (XL64 ^ q[27] ^ q[3]);
	hash[4] = (SHR(XH64, 3) ^ q[20] ^ msg[4]) + (XL64 ^ q[28] ^ q[4]);
	hash[5] = (SHL(XH64, 6) ^ SHR(q[21], 6) ^ msg[5]) + (XL64 ^ q[29] ^ q[5]);
	hash[6] = (SHR(XH64, 4) ^ SHL(q[22], 6) ^ msg[6]) + (XL64 ^ q[30] ^ q[6]);
	hash[7] = (SHR(XH64, 11) ^ SHL(q[23], 2) ^ msg[7]) + (XL64 ^ q[31] ^ q[7]);

	hash[8] = ROL(hash[4], 9) + (XH64 ^ q[24] ^ msg[8]) + (SHL(XL64, 8) ^ q[23] ^ q[8]);
	hash[9] = ROL(hash[5], 10) + (XH64 ^ q[25] ^ msg[9]) + (SHR(XL64, 6) ^ q[16] ^ q[9]);
	hash[10] = ROL(hash[6], 11) + (XH64 ^ q[26] ^ msg[10]) + (SHL(XL64, 6) ^ q[17] ^ q[10]);
	hash[11] = ROL(hash[7], 12) + (XH64 ^ q[27] ^ msg[11]) + (SHL(XL64, 4) ^ q[18] ^ q[11]);
	hash[12] = ROL(hash[0], 13) + (XH64 ^ q[28] ^ msg[12]) + (SHR(XL64, 3) ^ q[19] ^ q[12]);
	hash[13] = ROL(hash[1], 14) + (XH64 ^ q[29] ^ msg[13]) + (SHR(XL64, 4) ^ q[20] ^ q[13]);
	hash[14] = ROL(hash[2], 15) + (XH64 ^ q[30] ^ msg[14]) + (SHR(XL64, 7) ^ q[21] ^ q[14]);
	hash[15] = ROL(hash[3], 16) + (XH64 ^ q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}

/* Digest word 3 (vhash[7]:vhash[6]) for one nonce; mining and checksum kernels share it. */
__device__ __forceinline__
static uint64_t bmw512_80_word3(const uint32_t nounce)
{
	// Init
	uint2 h[16] = {
		{ 0x84858687UL, 0x80818283UL },
		{ 0x8C8D8E8FUL, 0x88898A8BUL },
		{ 0x94959697UL, 0x90919293UL },
		{ 0x9C9D9E9FUL, 0x98999A9BUL },
		{ 0xA4A5A6A7UL, 0xA0A1A2A3UL },
		{ 0xACADAEAFUL, 0xA8A9AAABUL },
		{ 0xB4B5B6B7UL, 0xB0B1B2B3UL },
		{ 0xBCBDBEBFUL, 0xB8B9BABBUL },
		{ 0xC4C5C6C7UL, 0xC0C1C2C3UL },
		{ 0xCCCDCECFUL, 0xC8C9CACBUL },
		{ 0xD4D5D6D7UL, 0xD0D1D2D3UL },
		{ 0xDCDDDEDFUL, 0xD8D9DADBUL },
		{ 0xE4E5E6E7UL, 0xE0E1E2E3UL },
		{ 0xECEDEEEFUL, 0xE8E9EAEBUL },
		{ 0xF4F5F6F7UL, 0xF0F1F2F3UL },
		{ 0xFCFDFEFFUL, 0xF8F9FAFBUL }
	};
	uint2 message[16];
	#pragma unroll
	for (int i=0;i<10;i++) message[i] = vectorize(c_PaddedMessage80[i]);
	/* fixed padding of an 80-byte message, as literals so it folds */
	message[10] = make_uint2(0x80, 0);
	#pragma unroll
	for (int i=11;i<15;i++) message[i] = make_uint2(0, 0);
	message[15] = make_uint2(640, 0);

	message[9].y = cuda_swab32(nounce);	//REPLACE_HIDWORD(message[9], cuda_swab32(nounce));

	Compression512_80_pre(message, h);

	#pragma unroll
	for (int i=0;i<16;i++) message[i] = make_uint2(0xaaaaaaa0+i,0xaaaaaaaa);

	Compression512(h, message);

	return devectorize(message[3+8]);
}

/* more blocks/SM spill and ran slower in sustained mining (sm_61) */
__global__ __launch_bounds__(128, 4)
void bmw512_gpu_hash_80_final(uint32_t threads, uint32_t startNounce, uint32_t *resNonce, const uint64_t target)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		if(bmw512_80_word3(startNounce + thread) <= target){
			report_candidate_2(resNonce, thread);
		}
	}
}

__global__ __launch_bounds__(128, 4)
void bmw512_gpu_checksum_80(uint32_t threads, uint32_t startNounce, uint64_t *acc)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		const uint32_t nonce = startNounce + thread;
		const uint64_t w = bmw512_80_word3(nonce);
		atomicXor((unsigned long long*) &acc[0], (unsigned long long) w);
		atomicXor((unsigned long long*) &acc[1],
			(unsigned long long) (w * (2ULL * (uint64_t) nonce + 1ULL)));
	}
}

__host__
void bmw512_cpu_hash_80_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_resNonce, const uint64_t target)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	bmw512_gpu_hash_80_final<<<grid, block>>>(threads, startNounce, d_resNonce, target);
}

/* GPU half of the GPU==CPU differential over [startNounce, startNounce + threads) for the
 * uploaded header; acc stays {0, 0} if it could not run. */
__host__
void bmw512_80_differential(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *acc)
{
	uint64_t *d_acc = NULL;
	acc[0] = acc[1] = 0;

	if (cudaMalloc(&d_acc, 2 * sizeof(uint64_t)) != cudaSuccess)
		return;
	if (cudaMemset(d_acc, 0, 2 * sizeof(uint64_t)) == cudaSuccess) {
		const uint32_t threadsperblock = 128;
		dim3 grid((threads + threadsperblock-1)/threadsperblock);
		dim3 block(threadsperblock);
		bmw512_gpu_checksum_80<<<grid, block>>>(threads, startNounce, d_acc);
		if (cudaDeviceSynchronize() == cudaSuccess)
			cudaMemcpy(acc, d_acc, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	}
	cudaFree(d_acc);
}

