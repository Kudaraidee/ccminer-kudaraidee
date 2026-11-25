#include <memory.h>

#include "cuda/helper.cuh"

static __forceinline__ __device__
void Round512v35(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7,
	const int ROT0, const int ROT1, const int ROT2, const int ROT3)
{
	p0 += p1; p1 = ROL2(p1, ROT0) ^ p0;
	p2 += p3; p3 = ROL2(p3, ROT1) ^ p2;
	p4 += p5; p5 = ROL2(p5, ROT2) ^ p4;
	p6 += p7; p7 = ROL2(p7, ROT3) ^ p6;
}

__forceinline__ __device__
void Round_8_512v35(const uint2 *const __restrict__ ks, const uint2 *const __restrict__ ts,
	uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, int R)
{
	Round512v35(p0, p1, p2, p3, p4, p5, p6, p7, 46, 36, 19, 37);
	Round512v35(p2, p1, p4, p7, p6, p5, p0, p3, 33, 27, 14, 42);
	Round512v35(p4, p1, p6, p3, p0, p5, p2, p7, 17, 49, 36, 39);
	Round512v35(p6, p1, p0, p7, p2, p5, p4, p3, 44,  9, 54, 56);

	p0 += ks[(R+0) % 9];
	p1 += ks[(R+1) % 9];
	p2 += ks[(R+2) % 9];
	p3 += ks[(R+3) % 9];
	p4 += ks[(R+4) % 9];
	p5 += ks[(R+5) % 9] + ts[(R+0) % 3];
	p6 += ks[(R+6) % 9] + ts[(R+1) % 3];
	p7 += ks[(R+7) % 9] + make_uint2(R, 0);

	Round512v35(p0, p1, p2, p3, p4, p5, p6, p7, 39, 30, 34, 24);
	Round512v35(p2, p1, p4, p7, p6, p5, p0, p3, 13, 50, 10, 17);
	Round512v35(p4, p1, p6, p3, p0, p5, p2, p7, 25, 29, 39, 43);
	Round512v35(p6, p1, p0, p7, p2, p5, p4, p3, 8,  35, 56, 22);

	p0 += ks[(R+1) % 9];
	p1 += ks[(R+2) % 9];
	p2 += ks[(R+3) % 9];
	p3 += ks[(R+4) % 9];
	p4 += ks[(R+5) % 9];
	p5 += ks[(R+6) % 9] + ts[(R+1) % 3];
	p6 += ks[(R+7) % 9] + ts[(R+2) % 3];
	p7 += ks[(R+8) % 9] + make_uint2(R+1, 0);
}

__forceinline__ __device__
void Round_8_512v35_final(const uint2 *const __restrict__ ks, const uint2 *const __restrict__ ts,
	uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7)
{
	Round512v35(p0, p1, p2, p3, p4, p5, p6, p7, 46, 36, 19, 37);
	Round512v35(p2, p1, p4, p7, p6, p5, p0, p3, 33, 27, 14, 42);
	Round512v35(p4, p1, p6, p3, p0, p5, p2, p7, 17, 49, 36, 39);
	Round512v35(p6, p1, p0, p7, p2, p5, p4, p3, 44, 9, 54, 56);

	p0 += ks[8];
	p1 += ks[0];
	p2 += ks[1];
	p3 += ks[2];
	p4 += ks[3];
	p5 += ks[4] + ts[2];
	p6 += ks[5] + ts[0];
	p7 += ks[6] + make_uint2(17, 0);

	Round512v35(p0, p1, p2, p3, p4, p5, p6, p7, 39, 30, 34, 24);
	Round512v35(p2, p1, p4, p7, p6, p5, p0, p3, 13, 50, 10, 17);
	Round512v35(p4, p1, p6, p3, p0, p5, p2, p7, 25, 29, 39, 43);
	Round512v35(p6, p1, p0, p7, p2, p5, p4, p3, 8,  35, 56, 22);

	p0 += ks[0];
	p1 += ks[1];
	p2 += ks[2];
	p3 += ks[3];
}

/* Launch shape per arch; Pascal's (512,2) spills a little but wins on resident warps.
 * The host block size in skein256_cpu_hash_32 must match (a larger block fails). */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
#define SKEIN256_TPB 512
#define SKEIN256_MINB 2
#else
#define SKEIN256_TPB 128
#define SKEIN256_MINB 6
#endif

__global__ __launch_bounds__(SKEIN256_TPB, SKEIN256_MINB)
void skein256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		const uint2 skein_ks_parity = { 0xA9FC1A22, 0x1BD11BDA };
		const uint2 t12[6] = {
			{ 0x20, 0 },
			{ 0,    0xf0000000 },
			{ 0x20, 0xf0000000 },
			{ 0x08, 0 },
			{ 0,    0xff000000 },
			{ 0x08, 0xff000000 }
		};

		uint2 h[9] = {
			{ 0x2FDB3E13, 0xCCD044A1 },
			{ 0x1A79A9EB, 0xE8359030 },
			{ 0x4F816E6F, 0x55AEA061 },
			{ 0xAE9B94DB, 0x2A2767A4 },
			{ 0x74DD7683, 0xEC06025E },
			{ 0xC4746251, 0xE7A436CD },
			{ 0x393AD185, 0xC36FBAF9 },
			{ 0x33EDFC13, 0x3EEDBA18 },
			{ 0xC73A4E2A, 0xB69D3CFC }
		};
		uint2 dt0,dt1,dt2,dt3;
		uint2 p0, p1, p2, p3, p4, p5, p6, p7;

		LOHI(dt0.x,dt0.y,outputHash[thread]);
		LOHI(dt1.x,dt1.y,outputHash[threads+thread]);
		LOHI(dt2.x,dt2.y,outputHash[2*threads+thread]);
		LOHI(dt3.x,dt3.y,outputHash[3*threads+thread]);

		p0 = h[0] + dt0;
		p1 = h[1] + dt1;
		p2 = h[2] + dt2;
		p3 = h[3] + dt3;
		p4 = h[4];
		p5 = h[5] + t12[0];
		p6 = h[6] + t12[1];
		p7 = h[7];

		// forced unroll required
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 1);
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 3);
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 5);
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 7);
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 9);
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 11);
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 13);
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 15);
		Round_8_512v35(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 17);

		p0 ^= dt0;
		p1 ^= dt1;
		p2 ^= dt2;
		p3 ^= dt3;

		h[0] = p0;
		h[1] = p1;
		h[2] = p2;
		h[3] = p3;
		h[4] = p4;
		h[5] = p5;
		h[6] = p6;
		h[7] = p7;
		h[8] = skein_ks_parity ^ h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7];

		const uint2 *t = t12+3;
		p5 += t12[3];  //p5 already equal h[5]
		p6 += t12[4];

		// forced unroll
		Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 1);
		Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 3);
		Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 5);
		Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 7);
		Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 9);
		Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 11);
		Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 13);
		Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 15);
		Round_8_512v35_final(h, t, p0, p1, p2, p3, p4, p5, p6, p7);

		outputHash[thread]           = devectorize(p0);
		outputHash[threads+thread]   = devectorize(p1);
		outputHash[2*threads+thread] = devectorize(p2);
		outputHash[3*threads+thread] = devectorize(p3);
	}
}

bool skein256_device_selftest(int thr_id);

__host__
void skein256_cpu_init(int thr_id, uint32_t threads)
{
	skein256_device_selftest(thr_id);
}

__host__
void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	// must agree with SKEIN256_TPB
	const uint32_t threadsperblock = (device_sm[device_map[thr_id]] < 700) ? 512 : 128;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	skein256_gpu_hash_32<<<grid, block>>>(threads, startNounce, d_outputHash);

	MyStreamSynchronize(NULL, order, thr_id);
}

/* ------------------------------------------------------------------ self-test
 * Init-time KAT (cuda/stage_selftest.cuh) of the launcher vs sph_skein256
 * (Skein-512-256) over 32-byte SoA slots. Fail-closed. */
extern "C" {
#include "sph/sph_skein.h"
}
#include "cuda/stage_selftest.cuh"

static int s_skein256_st_thr;

static bool skein256_st_gpu(const uint8_t *in, uint8_t *out, int n)
{
	uint64_t *h = (uint64_t*)malloc((size_t)n * 32);
	uint64_t *d = NULL;
	if (!h || cudaMalloc(&d, (size_t)n * 32) != cudaSuccess) { free(h); return selftest_cuda_fault(); }
	stkat_to_soa32(in, h, n);
	bool ok = (cudaMemcpy(d, h, (size_t)n * 32, cudaMemcpyHostToDevice) == cudaSuccess);
	skein256_cpu_hash_32(s_skein256_st_thr, n, 0, d, 0);
	ok = ok && (cudaDeviceSynchronize() == cudaSuccess);
	ok = ok && (cudaMemcpy(h, d, (size_t)n * 32, cudaMemcpyDeviceToHost) == cudaSuccess);
	if (ok) stkat_from_soa32(h, out, n);
	cudaFree(d); free(h);
	return ok ? true : selftest_cuda_fault();
}

static void skein256_st_ref(const uint8_t *in, uint8_t *out)
{
	sph_skein256_context c;
	sph_skein256_init(&c);
	sph_skein256(&c, in, 32);
	sph_skein256_close(&c, out);
}

__host__
bool skein256_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;
	s_skein256_st_thr = thr_id;
	passed = stkat_run(thr_id, "skein256", 32, 32, 256, 0x534B4E32u, skein256_st_gpu, skein256_st_ref, true);
	return passed;
}
