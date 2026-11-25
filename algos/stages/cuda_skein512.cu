/* SKEIN 64 and 80 based on Alexis Provos version */

#define TPB52 512
/* Ampere retune of the hash_64 / _final workhorse: the alexis donor
 * __launch_bounds__(512,3) budgets 65536/(512*3)=42 reg/thread, so ptxas caps
 * skein512_hash_64 at 40 reg and SPILLS 16 B (STACK:16, _final STACK:24) on
 * sm_86. The kernel's natural footprint is 48 reg; TPB128/minb6 gives a
 * 48-reg/0-spill build at ~83% occupancy. The eliminated local-memory
 * round-trips beat the 100%-vs-83% occupancy delta (~+3% event-timed on RTX
 * 3060) — a throughput/spill-bound kernel, so 83% (spill-free) > 100%
 * (spilling); the spill-free TPB512/minb2 at 67% only ties the donor, so the
 * win is spill-elimination at recovered mid occupancy, not block shape. */
#define TPB64 128

#include <stdio.h>

/* ************************ */

/*
 * M9_ ## s ## _ ## i  evaluates to s+i mod 9 (0 <= s <= 18, 0 <= i <= 7).
 */

#include "cuda/skein512_device.cuh"
#include "cuda/candidate_report.cuh"

// The Skein-512 device library (Threefish macros, first-block subkey
// schedule and the 64-byte compression) lives in cuda/skein512_device.cuh;
// skein512_gpu_hash_64 below is a thin wrapper. The *_final and
// 80-byte kernels are unchanged.

__global__
__launch_bounds__(TPB64, 6)
void skein512_gpu_hash_64(const uint32_t threads, const uint32_t startNonce, uint64_t* __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		// Skein
		uint2 p[8];

		const uint32_t hashPosition = (g_nonceVector == NULL) ? thread : g_nonceVector[thread] - startNonce;

		uint64_t *Hash = &g_hash[hashPosition<<3];

		uint2x4 *phash = (uint2x4*)Hash;
		*(uint2x4*)&p[0] = __ldg4(&phash[0]);
		*(uint2x4*)&p[4] = __ldg4(&phash[1]);

		skein512_hash_64(p);

		phash = (uint2x4*)p;
		uint2x4 *outpt = (uint2x4*)Hash;
		outpt[0] = phash[0];
		outpt[1] = phash[1];
	}
}


__global__ __launch_bounds__(TPB64, 6)
void skein512_gpu_hash_64_final(const uint32_t threads, uint64_t* g_hash, uint32_t* resNonce, uint64_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		// Skein
		uint2 p[8], h[9];

		const uint32_t hashPosition =  thread;

		uint64_t *Hash = &g_hash[hashPosition << 3];

		uint2x4 *phash = (uint2x4*)Hash;
		*(uint2x4*)&p[0] = __ldg4(&phash[0]);
		*(uint2x4*)&p[4] = __ldg4(&phash[1]);

		h[0] = p[0];	h[1] = p[1];	h[2] = p[2];	h[3] = p[3];
		h[4] = p[4];	h[5] = p[5];	h[6] = p[6];	h[7] = p[7];

		p[0] += buffer[0];	p[1] += buffer[1];	p[2] += buffer[2];	p[3] += buffer[3];
		p[4] += buffer[4];	p[5] += buffer[5];	p[6] += buffer[6];	p[7] += buffer[7];
		macro1();
		p[0] += buffer[8];	p[1] += buffer[9];	p[2] += buffer[10];	p[3] += buffer[11];
		p[4] += buffer[12];	p[5] += buffer[13];	p[6] += buffer[14];	p[7] += buffer[15];
		macro2();
		p[0] += buffer[16];	p[1] += buffer[17];	p[2] += buffer[18];	p[3] += buffer[19];
		p[4] += buffer[20];	p[5] += buffer[21];	p[6] += buffer[22];	p[7] += buffer[23];
		macro1();
		p[0] += buffer[24];	p[1] += buffer[25];	p[2] += buffer[26];	p[3] += buffer[27];
		p[4] += buffer[28];	p[5] += buffer[29];	p[6] += buffer[30];	p[7] += buffer[31];
		macro2();
		p[0] += buffer[32];	p[1] += buffer[33];	p[2] += buffer[34];	p[3] += buffer[35];
		p[4] += buffer[36];	p[5] += buffer[37];	p[6] += buffer[38];	p[7] += buffer[39];
		macro1();
		p[0] += buffer[40];	p[1] += buffer[41];	p[2] += buffer[42];	p[3] += buffer[43];
		p[4] += buffer[44];	p[5] += buffer[45];	p[6] += buffer[46];	p[7] += buffer[47];
		macro2();
		p[0] += buffer[48];	p[1] += buffer[49];	p[2] += buffer[50];	p[3] += buffer[51];
		p[4] += buffer[52];	p[5] += buffer[53];	p[6] += buffer[54];	p[7] += buffer[55];
		macro1();
		p[0] += buffer[56];	p[1] += buffer[57];	p[2] += buffer[58];	p[3] += buffer[59];
		p[4] += buffer[60];	p[5] += buffer[61];	p[6] += buffer[62];	p[7] += buffer[63];
		macro2();
		p[0] += buffer[64];	p[1] += buffer[65];	p[2] += buffer[66];	p[3] += buffer[67];
		p[4] += buffer[68];	p[5] += buffer[69];	p[6] += buffer[70];	p[7] += buffer[71];
		macro1();
		p[0] += buffer[72];	p[1] += buffer[73];	p[2] += buffer[74];	p[3] += buffer[75];
		p[4] += buffer[76];	p[5] += buffer[77];	p[6] += buffer[78];	p[7] += buffer[79];
		macro2();
		p[0] += buffer[80];	p[1] += buffer[81];	p[2] += buffer[82];	p[3] += buffer[83];
		p[4] += buffer[84];	p[5] += buffer[85];	p[6] += buffer[86];	p[7] += buffer[87];
		macro1();
		p[0] += buffer[88];	p[1] += buffer[89];	p[2] += buffer[90];	p[3] += buffer[91];
		p[4] += buffer[92];	p[5] += buffer[93];	p[6] += buffer[94];	p[7] += buffer[95];
		macro2();
		p[0] += buffer[96];	p[1] += buffer[97];	p[2] += buffer[98];	p[3] += buffer[99];
		p[4] += buffer[100];	p[5] += buffer[101];	p[6] += buffer[102];	p[7] += buffer[103];
		macro1();
		p[0] += buffer[104];	p[1] += buffer[105];	p[2] += buffer[106];	p[3] += buffer[107];
		p[4] += buffer[108];	p[5] += buffer[109];	p[6] += buffer[110];	p[7] += buffer[111];
		macro2();
		p[0] += make_uint2(0xA9D5C3F4, 0xEABE394C);	p[1] += make_uint2(0x1A75B523, 0x991112C7);
		p[2] += make_uint2(0x660FCC33, 0xAE18A40B);	p[3] += make_uint2(0x98173EC4, 0xCAB2076D);
		p[4] += make_uint2(0x749C51CE, 0x4903ADFF);	p[5] += make_uint2(0x9746DF43, 0xFD95DE39);
		p[6] += make_uint2(0x27C79C0E, 0x8FD19341);	p[7] += make_uint2(0xFF352CBF, 0x9A255629);
		macro1();
		p[0] += make_uint2(0x1A75B523, 0x991112C7);	p[1] += make_uint2(0x660FCC33, 0xAE18A40B);
		p[2] += make_uint2(0x98173EC4, 0xCAB2076D);	p[3] += make_uint2(0x749C51CE, 0x4903ADFF);
		p[4] += make_uint2(0x9746DF03, 0x0D95DE39);	p[5] += make_uint2(0x27C79C0E, 0x8FD19341);
		p[6] += make_uint2(0xFF352CB1, 0x8A255629);	p[7] += make_uint2(0xDF6CA7BF, 0x5DB62599);
		macro2();
		p[0] += vectorize(0xAE18A40B660FCC33);		p[1] += vectorize(0xcab2076d98173ec4);
		p[2] += vectorize(0x4903ADFF749C51CE);		p[3] += vectorize(0x0D95DE399746DF03);
		p[4] += vectorize(0x8FD1934127C79BCE);		p[5] += vectorize(0x8A255629FF352CB1);
		p[6] += vectorize(0x4DB62599DF6CA7F0);		p[7] += vectorize(0xEABE394CA9D5C3F4 + 16);
		macro1();
		p[0] += vectorize(0xcab2076d98173ec4);		p[1] += vectorize(0x4903ADFF749C51CE);
		p[2] += vectorize(0x0D95DE399746DF03);		p[3] += vectorize(0x8FD1934127C79BCE);
		p[4] += vectorize(0x9A255629FF352CB1);		p[5] += vectorize(0x4DB62599DF6CA7F0);
		p[6] += vectorize(0xEABE394CA9D5C3F4 + 0x0000000000000040);
		p[7] += vectorize(0x991112C71A75B523 + 17);
		macro2();
		p[0] += vectorize(0x4903ADFF749C51CE);		p[1] += vectorize(0x0D95DE399746DF03);
		p[2] += vectorize(0x8FD1934127C79BCE);		p[3] += vectorize(0x9A255629FF352CB1);
		p[4] += vectorize(0x5DB62599DF6CA7B0);		p[5] += vectorize(0xEABE394CA9D5C3F4 + 0x0000000000000040);
		p[6] += vectorize(0x891112C71A75B523);		p[7] += vectorize(0xAE18A40B660FCC33 + 18);

#define h0 p[0]
#define h1 p[1]
#define h2 p[2]
#define h3 p[3]
#define h4 p[4]
#define h5 p[5]
#define h6 p[6]
#define h7 p[7]

		h0 ^= h[0];	h1 ^= h[1];	h2 ^= h[2];	h3 ^= h[3];
		h4 ^= h[4];	h5 ^= h[5];	h6 ^= h[6];	h7 ^= h[7];

		uint2 skein_h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ vectorize(0x1BD11BDAA9FC1A22);

		uint2 hash64[8];

		hash64[5] = h5 + 8;

		hash64[0] = h0 + h1;
		hash64[1] = ROL2(h1, 46) ^ hash64[0];
		hash64[2] = h2 + h3;
		hash64[3] = ROL2(h3, 36) ^ hash64[2];
		hash64[4] = h4 + hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] = (h6 + h7 + make_uint2(0, 0xff000000));
		hash64[7] = ROL2(h7, 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];

		hash64[0] += h1; hash64[1] += h2; hash64[2] += h3; hash64[3] += h4;
		hash64[4] += h5;
		hash64[5] += h6 + make_uint2(0, 0xff000000);
		hash64[6] += h7 + vectorize(0xff00000000000008);
		hash64[7] += skein_h8 + 1;
		macro3();
		hash64[0] += h2; hash64[1] += h3; hash64[2] += h4; hash64[3] += h5;
		hash64[4] += h6;
		hash64[5] += h7 + vectorize(0xff00000000000008);
		hash64[6] += skein_h8 + 8;
		hash64[7] += h0 + 2;
		macro4();
		hash64[0] = (hash64[0] + h3);			hash64[1] = (hash64[1] + h4);
		hash64[2] = (hash64[2] + h5);			hash64[3] = (hash64[3] + h6);
		hash64[4] = (hash64[4] + h7);			hash64[5] = (hash64[5] + skein_h8 + 8);
		hash64[6] = (hash64[6] + h0 + make_uint2(0, 0xff000000));
		hash64[7] = (hash64[7] + h1 + 3);
		macro3();
		hash64[0] = (hash64[0] + h4);			hash64[1] = (hash64[1] + h5);
		hash64[2] = (hash64[2] + h6);			hash64[3] = (hash64[3] + h7);
		hash64[4] = (hash64[4] + skein_h8);		hash64[5] = (hash64[5] + h0 + make_uint2(0, 0xff000000));
		hash64[6] = (hash64[6] + h1 + vectorize(0xff00000000000008));
		hash64[7] = (hash64[7] + h2 + 4);
		macro4();
		hash64[0] = (hash64[0] + h5);			hash64[1] = (hash64[1] + h6);
		hash64[2] = (hash64[2] + h7);			hash64[3] = (hash64[3] + skein_h8);
		hash64[4] = (hash64[4] + h0);			hash64[5] = (hash64[5] + h1 + vectorize(0xff00000000000008));
		hash64[6] = (hash64[6] + h2 + 8);		hash64[7] = (hash64[7] + h3 + 5);
		macro3();
		hash64[0] = (hash64[0] + h6);			hash64[1] = (hash64[1] + h7);
		hash64[2] = (hash64[2] + skein_h8);		hash64[3] = (hash64[3] + h0);
		hash64[4] = (hash64[4] + h1);			hash64[5] = (hash64[5] + h2 + 8);
		hash64[6] = (hash64[6] + h3 + make_uint2(0, 0xff000000));
		hash64[7] = (hash64[7] + h4 + 6);
		macro4();
		hash64[0] = (hash64[0] + h7);			hash64[1] = (hash64[1] + skein_h8);
		hash64[2] = (hash64[2] + h0);			hash64[3] = (hash64[3] + h1);
		hash64[4] = (hash64[4] + h2);			hash64[5] = (hash64[5] + h3 + make_uint2(0, 0xff000000));
		hash64[6] = (hash64[6] + h4 + vectorize(0xff00000000000008));
		hash64[7] = (hash64[7] + h5 + 7);
		macro3();
		hash64[0] = (hash64[0] + skein_h8);		hash64[1] = (hash64[1] + h0);
		hash64[2] = (hash64[2] + h1);			hash64[3] = (hash64[3] + h2);
		hash64[4] = (hash64[4] + h3);			hash64[5] = (hash64[5] + h4 + vectorize(0xff00000000000008));
		hash64[6] = (hash64[6] + h5 + 8);		hash64[7] = (hash64[7] + h6 + 8);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h0));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h1));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h2));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h3));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h4));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h5) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h6) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h7) + 9);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h1));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h2));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h3));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h4));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h5));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h6) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h7) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(skein_h8) + 10);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h2));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h3));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h4));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h5));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h6));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h7) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(skein_h8) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h0) + 11);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h3));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h4));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h5));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h6));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h7));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(skein_h8) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h0) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h1) + 12);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h4));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h5));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h6));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h7));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(skein_h8));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h0) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h1) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h2) + 13);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h5));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h6));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h7));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(skein_h8));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h0));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h1) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h2) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h3) + 14);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h6));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h7));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(skein_h8));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h0));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h1));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h2) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h3) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h4) + 15);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h7));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(skein_h8));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h0));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h1));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h2));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h3) + 0xff00000000000000);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h4) + 0xff00000000000008);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h5) + 16);
		macro4();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(skein_h8));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h0));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h1));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h2));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h3));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h4) + 0xff00000000000008);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h5) + 8);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h6) + 17);
		macro3();
		hash64[0] = vectorize(devectorize(hash64[0]) + devectorize(h0));
		hash64[1] = vectorize(devectorize(hash64[1]) + devectorize(h1));
		hash64[2] = vectorize(devectorize(hash64[2]) + devectorize(h2));
		hash64[3] = vectorize(devectorize(hash64[3]) + devectorize(h3));
		hash64[4] = vectorize(devectorize(hash64[4]) + devectorize(h4));
		hash64[5] = vectorize(devectorize(hash64[5]) + devectorize(h5) + 8);
		hash64[6] = vectorize(devectorize(hash64[6]) + devectorize(h6) + 0xff00000000000000);
		hash64[7] = vectorize(devectorize(hash64[7]) + devectorize(h7) + 18);

//		phash = (uint2x4*)hash64;
//		uint2x4 *outpt = (uint2x4*)Hash;
//		outpt[0] = phash[0];
//		outpt[1] = phash[1];

#undef h0
#undef h1
#undef h2
#undef h3
#undef h4
#undef h5
#undef h6
#undef h7

		if (devectorize(hash64[3]) <= target)
		{
			report_candidate_2(resNonce, hashPosition);
		}
	}
}


__host__
//void skein512_cpu_hash_64(int thr_id,uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash)
void skein512_cpu_hash_64(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t tpb = TPB64;
	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);
	skein512_gpu_hash_64 <<<grid, block >>>(threads, startNonce, (uint64_t*)d_hash, d_nonceVector);

}

__host__
void skein512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint64_t target, uint32_t *d_resNonce)
{
	const uint32_t tpb = TPB64;
	const dim3 grid((threads + tpb - 1) / tpb);
	const dim3 block(tpb);
	skein512_gpu_hash_64_final << <grid, block >> >(threads, (uint64_t*)d_hash, d_resNonce, target);
}



// 120 * 8 = 960 ... too big ?
static __constant__ uint2 c_buffer[120]; // padded message (80 bytes + 72*8 bytes midstate + align)

__global__
__launch_bounds__(TPB52, 3)
void skein512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *output64)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;

		uint32_t nonce = cuda_swab32(startNounce + thread);
		uint2 nonce2 = make_uint2(c_buffer[0].x, nonce);

		uint2 p[8];
		p[1] = nonce2;

		h0 = c_buffer[ 1];
		h1 = c_buffer[ 2];
		h2 = c_buffer[ 3];
		h3 = c_buffer[ 4];
		h4 = c_buffer[ 5];
		h5 = c_buffer[ 6];
		h6 = c_buffer[ 7];
		h7 = c_buffer[ 8];
		h8 = c_buffer[ 9];

		t0 = vectorize(0x50ull);
		t1 = vectorize(0xB000000000000000ull);
		t2 = t0^t1;

		p[1]=nonce2 + h1;	p[0]= c_buffer[10] + p[1];
		p[2]=c_buffer[11];
		p[3]=c_buffer[12];
		p[4]=c_buffer[13];
		p[5]=c_buffer[14];
		p[6]=c_buffer[15];
		p[7]=c_buffer[16];

//		macro1();
		p[1] = ROL2(p[1], 46) ^ p[0];
		p[2] += p[1];
		p[0] += p[3];
		p[1] = ROL2(p[1], 33) ^ p[2];
		p[3] = c_buffer[17] ^ p[0];
		p[4] += p[1];
		p[6] += p[3];
		p[0] += p[5];
		p[2] += p[7];
		p[1] = ROL2(p[1], 17) ^ p[4];
		p[3] = ROL2(p[3], 49) ^ p[6];
		p[5] = c_buffer[18] ^ p[0];
		p[7] = c_buffer[19] ^ p[2];
		p[6] += p[1];
		p[0] += p[7];
		p[2] += p[5];
		p[4] += p[3];
		p[1] = ROL2(p[1], 44) ^ p[6];
		p[7] = ROL2(p[7], 9) ^ p[0];
		p[5] = ROL2(p[5], 54) ^ p[2];
		p[3] = ROR8(p[3]) ^ p[4];

		p[0]+=h1;	p[1]+=h2;	p[2]+=h3;	p[3]+=h4;	p[4]+=h5;
		p[5]+=c_buffer[20];	p[7]+=c_buffer[21];	p[6]+=c_buffer[22];
		macro2();
		p[0]+=h2;	p[1]+=h3;	p[2]+=h4;	p[3]+=h5;	p[4]+=h6;
		p[5]+=c_buffer[22];	p[7]+=c_buffer[23];	p[6]+=c_buffer[24];
		macro1();
		p[0]+=h3;	p[1]+=h4;	p[2]+=h5;	p[3]+=h6;	p[4]+=h7;
		p[5]+=c_buffer[24];	p[7]+=c_buffer[25];	p[6]+=c_buffer[26];
		macro2();
		p[0]+=h4;	p[1]+=h5;	p[2]+=h6;	p[3]+=h7;	p[4]+=h8;
		p[5]+=c_buffer[26];	p[7]+=c_buffer[27];	p[6]+=c_buffer[28];
		macro1();
		p[0]+=h5;	p[1]+=h6;	p[2]+=h7;	p[3]+=h8;	p[4]+=h0;
		p[5]+=c_buffer[28];	p[7]+=c_buffer[29];	p[6]+=c_buffer[30];
		macro2();
		p[0]+=h6;	p[1]+=h7;	p[2]+=h8;	p[3]+=h0;	p[4]+=h1;
		p[5]+=c_buffer[30];	p[7]+=c_buffer[31];	p[6]+=c_buffer[32];
		macro1();
		p[0]+=h7;	p[1]+=h8;	p[2]+=h0;	p[3]+=h1;	p[4]+=h2;
		p[5]+=c_buffer[32];	p[7]+=c_buffer[33];	p[6]+=c_buffer[34];
		macro2();
		p[0]+=h8;	p[1]+=h0;	p[2]+=h1;	p[3]+=h2;	p[4]+=h3;
		p[5]+=c_buffer[34];	p[7]+=c_buffer[35];	p[6]+=c_buffer[36];
		macro1();
		p[0]+=h0;	p[1]+=h1;	p[2]+=h2;	p[3]+=h3;	p[4]+=h4;
		p[5]+=c_buffer[36];	p[7]+=c_buffer[37];	p[6]+=c_buffer[38];
		macro2();
		p[0]+=h1;	p[1]+=h2;	p[2]+=h3;	p[3]+=h4;	p[4]+=h5;
		p[5]+=c_buffer[38];	p[7]+=c_buffer[39];	p[6]+=c_buffer[40];
		macro1();
		p[0]+=h2;	p[1]+=h3;	p[2]+=h4;	p[3]+=h5;	p[4]+=h6;
		p[5]+=c_buffer[40];	p[7]+=c_buffer[41];	p[6]+=c_buffer[42];
		macro2();
		p[0]+=h3;	p[1]+=h4;	p[2]+=h5;	p[3]+=h6;	p[4]+=h7;
		p[5]+=c_buffer[42];	p[7]+=c_buffer[43];	p[6]+=c_buffer[44];
		macro1();
		p[0]+=h4;	p[1]+=h5;	p[2]+=h6;	p[3]+=h7;	p[4]+=h8;
		p[5]+=c_buffer[44];	p[7]+=c_buffer[45];	p[6]+=c_buffer[46];
		macro2();
		p[0]+=h5;	p[1]+=h6;	p[2]+=h7;	p[3]+=h8;	p[4]+=h0;
		p[5]+=c_buffer[46];	p[7]+=c_buffer[47];	p[6]+=c_buffer[48];
		macro1();
		p[0]+=h6;	p[1]+=h7;	p[2]+=h8;	p[3]+=h0;	p[4]+=h1;
		p[5]+=c_buffer[48];	p[7]+=c_buffer[49];	p[6]+=c_buffer[50];
		macro2();
		p[0]+=h7;	p[1]+=h8;	p[2]+=h0;	p[3]+=h1;	p[4]+=h2;
		p[5]+=c_buffer[50];	p[7]+=c_buffer[51];	p[6]+=c_buffer[52];
		macro1();
		p[0]+=h8;	p[1]+=h0;	p[2]+=h1;	p[3]+=h2;	p[4]+=h3;
		p[5]+=c_buffer[52];	p[7]+=c_buffer[53];	p[6]+=c_buffer[54];
		macro2();
		p[0]+=h0;	p[1]+=h1;	p[2]+=h2;	p[3]+=h3;	p[4]+=h4;
		p[5]+=c_buffer[54];	p[7]+=c_buffer[55];	p[6]+=c_buffer[56];

		p[0]^= c_buffer[57];
		p[1]^= nonce2;

		t0 = vectorize(8); // extra
		t1 = vectorize(0xFF00000000000000ull); // etype
//		t2 = vectorize(0xB000000000000050ull);

		h0 = p[0];
		h1 = p[1];
		h2 = p[2];
		h3 = p[3];
		h4 = p[4];
		h5 = p[5];
		h6 = p[6];
		h7 = p[7];

		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);

		p[0] = p[1] = p[2] = p[3] = p[4] =p[5] =p[6] = p[7] = vectorize(0);

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint64_t *outpHash = &output64[thread<<3];
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			outpHash[i] = devectorize(p[i]);
	}
}

__host__
void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap)
{
	const uint32_t tpb = TPB52;

	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);

	// hash function is cut in 2 parts to reduce kernel size
	skein512_gpu_hash_80 <<< grid, block >>> (threads, startNounce, (uint64_t*)d_hash);
}

__host__
void skein512_cpu_setBlock_80(void *pdata)
{
	uint64_t message[20];
	memcpy(&message[0], pdata, 80);

	uint64_t p[8];
	uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
	uint64_t t0, t1, t2;

	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	// h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h8 = 0xcab2076d98173ec4ULL;

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	memcpy(&p[0], &message[0], 64);

	TFBIG_4e_PRE(0);
	TFBIG_4o_PRE(1);
	TFBIG_4e_PRE(2);
	TFBIG_4o_PRE(3);
	TFBIG_4e_PRE(4);
	TFBIG_4o_PRE(5);
	TFBIG_4e_PRE(6);
	TFBIG_4o_PRE(7);
	TFBIG_4e_PRE(8);
	TFBIG_4o_PRE(9);
	TFBIG_4e_PRE(10);
	TFBIG_4o_PRE(11);
	TFBIG_4e_PRE(12);
	TFBIG_4o_PRE(13);
	TFBIG_4e_PRE(14);
	TFBIG_4o_PRE(15);
	TFBIG_4e_PRE(16);
	TFBIG_4o_PRE(17);
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	message[10] = message[0] ^ p[0];
	message[11] = message[1] ^ p[1];
	message[12] = message[2] ^ p[2];
	message[13] = message[3] ^ p[3];
	message[14] = message[4] ^ p[4];
	message[15] = message[5] ^ p[5];
	message[16] = message[6] ^ p[6];
	message[17] = message[7] ^ p[7];

	message[18] = t2;

	uint64_t buffer[128];

//	buffer[ 0] = message[ 8];
	buffer[ 0] = message[ 9];
	h0 = buffer[ 1] = message[10];
	h1 = buffer[ 2] = message[11];
	h2 = buffer[ 3] = message[12];
	h3 = buffer[ 4] = message[13];
	h4 = buffer[ 5] = message[14];
	h5 = buffer[ 6] = message[15];
	h6 = buffer[ 7] = message[16];
	h7 = buffer[ 8] = message[17];
	h8 = buffer[ 9] = h0^h1^h2^h3^h4^h5^h6^h7^0x1BD11BDAA9FC1A22ULL;

	t0 = 0x50ull;
	t1 = 0xB000000000000000ull;
	t2 = t0^t1;

	p[0] = message[ 8] + h0;
	p[2] = h2; p[3] = h3; p[4] = h4;
	p[5] = h5 + t0; p[6] = h6 + t1; p[7] = h7;
	p[2] += p[3]; p[4] += p[5]; p[6] += p[7];
	p[3] = ROTL64(p[3], 36) ^ p[2];
	p[5] = ROTL64(p[5], 19) ^ p[4];
	p[7] = ROTL64(p[7], 37) ^ p[6];
	p[4] += p[7];
	p[6] += p[5];
	p[7] = ROTL64(p[7], 27) ^ p[4];
	p[5] = ROTL64(p[5], 14) ^ p[6];

	buffer[10] = p[0];
	buffer[11] = p[2];
	buffer[12] = p[3];
	buffer[13] = p[4];
	buffer[14] = p[5];
	buffer[15] = p[6];
	buffer[16] = p[7];
	buffer[17] = ROTL64(p[3], 42);
	buffer[18] = ROTL64(p[5], 36);
	buffer[19] = ROTL64(p[7], 39);

	buffer[20] = h6+t1;
	buffer[21] = h8+1;
	buffer[22] = h7+t2;
	buffer[23] = h0+2;
	buffer[24] = h8+t0;
	buffer[25] = h1+3;
	buffer[26] = h0+t1;
	buffer[27] = h2+4;
	buffer[28] = h1+t2;
	buffer[29] = h3+5;
	buffer[30] = h2+t0;
	buffer[31] = h4+6;
	buffer[32] = h3+t1;
	buffer[33] = h5+7;
	buffer[34] = h4+t2;
	buffer[35] = h6+8;
	buffer[36] = h5+t0;
	buffer[37] = h7+9;
	buffer[38] = h6+t1;
	buffer[39] = h8+10;
	buffer[40] = h7+t2;
	buffer[41] = h0+11;
	buffer[42] = h8+t0;
	buffer[43] = h1+12;
	buffer[44] = h0+t1;
	buffer[45] = h2+13;
	buffer[46] = h1+t2;
	buffer[47] = h3+14;
	buffer[48] = h2+t0;
	buffer[49] = h4+15;
	buffer[50] = h3+t1;
	buffer[51] = h5+16;
	buffer[52] = h4+t2;
	buffer[53] = h6+17;
	buffer[54] = h5+t0;
	buffer[55] = h7+18;
	buffer[56] = h6+t1;

	buffer[57] = message[8];

	cudaMemcpyToSymbol(c_buffer, buffer, sizeof(c_buffer), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaGetLastError());
}

/* Unit self-test for cuda/skein512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool skein512_device_selftest(int thr_id);

__host__
void skein512_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
	skein512_device_selftest(thr_id);
}


