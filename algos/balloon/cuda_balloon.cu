#include "miner.h"
#include <string.h>
#include <stdint.h>
#include <stdint.h>
#include "cuda/helper.cuh"
#include "balloon.h"
#include "sha256.h"
// The mined header is always 80 bytes; compile-time so the copies fold.
#define BALLOON_INLEN 80

// Mix pass count (t_cost); 4 is consensus. Overridable for timing builds only.
#ifndef BALLOON_MIXROUNDS
#define BALLOON_MIXROUNDS 4
#endif

// Space cost in KB (4096 blocks of 32 bytes); consensus. Both costs seed the
// index stream and block 0, so host and kernel must take them from here.
#define BALLOON_SCOST 128

// Candidates the kernel keeps per launch.
#define BALLOON_SLOTS MAX_NONCES

__global__ void balloon_gpu_hash(const uint64_t *prebuf_le, const uint8_t *input, uint32_t max_nonce, uint32_t *resNonce, const uint32_t *device_target);
__device__ void device_sha256_osol(const uint32_t *w, uint32_t *ctx);
__device__ void device_sha256_bytes(const uint8_t *data, uint32_t len, uint32_t *ctx);
__device__ void balloon_fill(uint32_t *buf, const uint8_t *in, uint32_t *prev);
__device__ void balloon_mix(uint32_t *buf, const uint64_t *prebuf_le, uint32_t *prev);
void host_sha256_osol(const __sha256_block_t blk, __sha256_hash_t ctx);
void update_device_data(int gpuid);


#ifdef _MSC_VER
#define THREAD __declspec(thread)
#else
#define THREAD __thread
#endif

#ifdef __INTELLISENSE__
#define __launch_bounds__(blocksize)
#endif

void update_device_data(int gpuid);
__constant__ const uint32_t __sha256_init[] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};
#define UINT2(x,y) make_uint2(x,y);
#define PREBUF_LEN 409600
// The mix reads 3 indices per block per round from this table.
static_assert(3 * 4096 * BALLOON_MIXROUNDS <= PREBUF_LEN / 8,
	"prebuf_le too small for BALLOON_MIXROUNDS");
static_assert(BALLOON_SCOST * 1024 / 32 == 4096,
	"the kernel's block count is fixed at 4096");
uint64_t host_prebuf_le[20][PREBUF_LEN / 8];
uint8_t host_prebuf_filled[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
#define BLOCK_SIZE (32)

uint64_t *device_prebuf_le[20];
uint32_t *device_winning_nonce[20];
uint32_t *device_target[20];
uint8_t *device_input[20];

uint8_t balloon_inited[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
uint8_t syncmode_set[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

void fill_prebuf(struct hash_state *s, int gpuid) {
	uint8_t host_prebuf[PREBUF_LEN];
	if (!host_prebuf_filled[gpuid]) {
		bitstream_fill_buffer(&s->bstream, host_prebuf, PREBUF_LEN);
		host_prebuf_filled[gpuid] = 1;
		uint8_t *buf = host_prebuf;
		uint64_t *lebuf = host_prebuf_le[gpuid];
		for (int i = 0; i < PREBUF_LEN; i += 8) {
			bytes_to_littleend8_uint64(buf, lebuf);
			*lebuf %= 4096;
			*lebuf <<= 5; // multiply by 32
			lebuf++;
			buf += 8;
		}
		update_device_data(gpuid);
	}
}

// Scans `threads` nonces from the one in input[76..79] (big-endian); returns the
// lowest passing the device screen, or UINT32_MAX. The caller re-verifies.
uint32_t balloon_cpu_hash(int thr_id, unsigned char *input, uint32_t threads, uint32_t max_nonce)
{
	struct balloon_options opts;
	struct hash_state s;
	// One nonce per thread, 64 threads per block.
	uint32_t num_threads = 64;
	uint32_t num_blocks = threads / num_threads;
	if (num_blocks == 0) num_blocks = 1;
	balloon_init(&opts, (int64_t)BALLOON_SCOST, (int32_t)BALLOON_MIXROUNDS);
	hash_state_init(&s, &opts, input);
	fill_prebuf(&s, thr_id);

	// Only the index stream is needed here; the kernel builds its own buffer.
	free(s.buffer);
	s.buffer = NULL;

	CUDA_SAFE_CALL(cudaMemcpy((void**)device_input[thr_id], (void**)input, BALLOON_INLEN, cudaMemcpyHostToDevice));

	// Result buffer: [0] lowest candidate, [1] count, then BALLOON_SLOTS candidates.
	// UINT32_MAX means none (0 is a legal nonce). Armed in full: all are read back.
	const uint32_t arm[2 + BALLOON_SLOTS] = { UINT32_MAX, 0 };
	uint32_t host_winning_nonce = UINT32_MAX;

	CUDA_SAFE_CALL(cudaMemcpy((void**)device_winning_nonce[thr_id], (void**)arm, sizeof(arm), cudaMemcpyHostToDevice));
	balloon_gpu_hash << <num_blocks, num_threads >> > (device_prebuf_le[thr_id], device_input[thr_id], max_nonce, device_winning_nonce[thr_id], device_target[thr_id]);

	CUDA_SAFE_CALL(cudaPeekAtLastError());

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy((void*)&host_winning_nonce, (void*)device_winning_nonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost));

	hash_state_free(&s);

	return host_winning_nonce;
}

// Candidates of the last launch, ascending; returns how many the kernel found.
// Above BALLOON_SLOTS out[] is a subset and only the lowest is guaranteed.
uint32_t balloon_get_candidates(int thr_id, uint32_t *out, uint32_t max_out)
{
	uint32_t buf[1 + BALLOON_SLOTS];
	CUDA_SAFE_CALL(cudaMemcpy(buf, device_winning_nonce[thr_id] + 1, sizeof(buf), cudaMemcpyDeviceToHost));
	const uint32_t found = buf[0];
	uint32_t n = found < BALLOON_SLOTS ? found : BALLOON_SLOTS;
	if (n > max_out) n = max_out;
	for (uint32_t i = 0; i < n; i++) {
		uint32_t v = buf[1 + i], j = i;
		for (; j > 0 && out[j - 1] > v; j--)
			out[j] = out[j - 1];
		out[j] = v;
	}
	return found;
}



// The index stream depends on the first SALT_LEN header bytes only; reset it
// for every new work unit.
extern "C" void reset_host_prebuf(int thr_id)
{
	host_prebuf_filled[thr_id] = 0;
}

void balloon_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc((void**)&device_prebuf_le[thr_id], (PREBUF_LEN / 8) * sizeof(uint64_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&device_winning_nonce[thr_id], (2 + BALLOON_SLOTS) * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&device_target[thr_id], 8 * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&device_input[thr_id], BALLOON_INLEN));
}

// Threads the device keeps resident at once for this kernel (one "wave"), from
// the occupancy API, so it follows the arch. 0 if the query fails.
uint32_t balloon_wave_threads(int thr_id)
{
	int dev = device_map[thr_id];
	int sms = 0, blocks = 0;
	if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
	    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, balloon_gpu_hash, 64, 0) != cudaSuccess)
		return 0;
	return (uint32_t)sms * (uint32_t)blocks * 64U;
}

__host__ void balloon_setBlock_80(int thr_id, void *pdata, const void *pTargetIn)
{

	// Upload the host target (a device symbol read on the host is garbage).
	CUDA_SAFE_CALL(cudaMemcpy(device_target[thr_id], pTargetIn, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

	if (opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void update_device_data(int thr_id) {

	CUDA_SAFE_CALL(cudaMemcpy(device_prebuf_le[thr_id], host_prebuf_le[thr_id], (PREBUF_LEN / 8) * sizeof(uint64_t), cudaMemcpyHostToDevice));

}

__global__ void balloon_gpu_hash(const uint64_t *prebuf_le, const uint8_t *input, uint32_t max_nonce, uint32_t *resNonce, const uint32_t *device_target) {

	uint32_t id = blockDim.x*blockIdx.x + threadIdx.x;
	uint32_t nonce = ((input[76] << 24) | (input[77] << 16) | (input[78] << 8) | input[79]) + id;
	// Range guard only: no early exit once a candidate is found, or nonces the
	// host counts as scanned would be skipped.
	if (nonce > max_nonce)
		return;

	uint8_t local_input[BALLOON_INLEN];
	memcpy(local_input, input, BALLOON_INLEN);
	((uint32_t*)local_input)[19] = cuda_swab32(nonce);

	// 4096 blocks of 8 state words, 128 KB per thread (local memory).
	__align__(16) uint32_t buf[4096 * 8];
	uint32_t prev[8];
	balloon_fill(buf, local_input, prev);
	balloon_mix(buf, prebuf_le, prev);

	// prev = last block; compare digest bytes 28..31 as a little-endian word
	if (cuda_swab32(prev[7]) < device_target[7]) {
		// Lowest candidate atomically, plus up to BALLOON_SLOTS in arrival order.
		atomicMin(resNonce, nonce);
		const uint32_t k = atomicAdd(resNonce + 1, 1);
		if (k < BALLOON_SLOTS)
			resNonce[2 + k] = nonce;
	}
}


/**************** SHA256 from github sha256-sse ***************/

#define	SHA256_CONST(x)		(SHA256_CONST_ ## x)

/* constants, as provided in FIPS 180-2 */

#define	SHA256_CONST_0		0x428a2f98U
#define	SHA256_CONST_1		0x71374491U
#define	SHA256_CONST_2		0xb5c0fbcfU
#define	SHA256_CONST_3		0xe9b5dba5U
#define	SHA256_CONST_4		0x3956c25bU
#define	SHA256_CONST_5		0x59f111f1U
#define	SHA256_CONST_6		0x923f82a4U
#define	SHA256_CONST_7		0xab1c5ed5U

#define	SHA256_CONST_8		0xd807aa98U
#define	SHA256_CONST_9		0x12835b01U
#define	SHA256_CONST_10		0x243185beU
#define	SHA256_CONST_11		0x550c7dc3U
#define	SHA256_CONST_12		0x72be5d74U
#define	SHA256_CONST_13		0x80deb1feU
#define	SHA256_CONST_14		0x9bdc06a7U
#define	SHA256_CONST_15		0xc19bf174U

#define	SHA256_CONST_16		0xe49b69c1U
#define	SHA256_CONST_17		0xefbe4786U
#define	SHA256_CONST_18		0x0fc19dc6U
#define	SHA256_CONST_19		0x240ca1ccU
#define	SHA256_CONST_20		0x2de92c6fU
#define	SHA256_CONST_21		0x4a7484aaU
#define	SHA256_CONST_22		0x5cb0a9dcU
#define	SHA256_CONST_23		0x76f988daU

#define	SHA256_CONST_24		0x983e5152U
#define	SHA256_CONST_25		0xa831c66dU
#define	SHA256_CONST_26		0xb00327c8U
#define	SHA256_CONST_27		0xbf597fc7U
#define	SHA256_CONST_28		0xc6e00bf3U
#define	SHA256_CONST_29		0xd5a79147U
#define	SHA256_CONST_30		0x06ca6351U
#define	SHA256_CONST_31		0x14292967U

#define	SHA256_CONST_32		0x27b70a85U
#define	SHA256_CONST_33		0x2e1b2138U
#define	SHA256_CONST_34		0x4d2c6dfcU
#define	SHA256_CONST_35		0x53380d13U
#define	SHA256_CONST_36		0x650a7354U
#define	SHA256_CONST_37		0x766a0abbU
#define	SHA256_CONST_38		0x81c2c92eU
#define	SHA256_CONST_39		0x92722c85U

#define	SHA256_CONST_40		0xa2bfe8a1U
#define	SHA256_CONST_41		0xa81a664bU
#define	SHA256_CONST_42		0xc24b8b70U
#define	SHA256_CONST_43		0xc76c51a3U
#define	SHA256_CONST_44		0xd192e819U
#define	SHA256_CONST_45		0xd6990624U
#define	SHA256_CONST_46		0xf40e3585U
#define	SHA256_CONST_47		0x106aa070U

#define	SHA256_CONST_48		0x19a4c116U
#define	SHA256_CONST_49		0x1e376c08U
#define	SHA256_CONST_50		0x2748774cU
#define	SHA256_CONST_51		0x34b0bcb5U
#define	SHA256_CONST_52		0x391c0cb3U
#define	SHA256_CONST_53		0x4ed8aa4aU
#define	SHA256_CONST_54		0x5b9cca4fU
#define	SHA256_CONST_55		0x682e6ff3U

#define	SHA256_CONST_56		0x748f82eeU
#define	SHA256_CONST_57		0x78a5636fU
#define	SHA256_CONST_58		0x84c87814U
#define	SHA256_CONST_59		0x8cc70208U
#define	SHA256_CONST_60		0x90befffaU
#define	SHA256_CONST_61		0xa4506cebU
#define	SHA256_CONST_62		0xbef9a3f7U
#define	SHA256_CONST_63		0xc67178f2U

/* Ch and Maj are the basic SHA2 functions. */
#define	Ch(b, c, d)	(((b) & (c)) ^ ((~b) & (d)))
#define	Maj(b, c, d)	(((b) & (c)) ^ ((b) & (d)) ^ ((c) & (d)))

/* Rotates x right n bits. */
#define	ROTR(x, n)	\
(((x) >> (n)) | ((x) << ((sizeof (x) * 8)-(n))))

/* Shift x right n bits */
#define	SHR(x, n)	((x) >> (n))

/* SHA256 Functions */
#define	BIGSIGMA0_256(x)	(ROTR((x), 2) ^ ROTR((x), 13) ^ ROTR((x), 22))
#define	BIGSIGMA1_256(x)	(ROTR((x), 6) ^ ROTR((x), 11) ^ ROTR((x), 25))
#define	SIGMA0_256(x)		(ROTR((x), 7) ^ ROTR((x), 18) ^ SHR((x), 3))
#define	SIGMA1_256(x)		(ROTR((x), 17) ^ ROTR((x), 19) ^ SHR((x), 10))

// sm_80+: issue adds as x * 1 + y on the idle IMAD pipe instead of the busy ALU;
// the 1 is in constant memory so ptxas cannot fold it back to an ALU add.
__constant__ uint32_t c_balloon_one = 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t madd(uint32_t x, uint32_t y)
{
	uint32_t r;
	asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(c_balloon_one), "r"(y));
	return r;
}
#define	T1_SUM(e, f, g, h, i, w)	madd(madd(h, SHA256_CONST(i)), madd(madd(BIGSIGMA1_256(e), Ch(e, f, g)), w))
#define	ADD2(x, y)		madd(x, y)
#define	W_NEXT(s1, a, s0, b)	madd(madd(SIGMA1_256(s1), a), madd(SIGMA0_256(s0), b))
#else
#define	T1_SUM(e, f, g, h, i, w)	(h + BIGSIGMA1_256(e) + Ch(e, f, g) + SHA256_CONST(i) + w)
#define	ADD2(x, y)		((x) + (y))
#define	W_NEXT(s1, a, s0, b)	(SIGMA1_256(s1) + a + SIGMA0_256(s0) + b)
#endif

#define	SHA256ROUND(a, b, c, d, e, f, g, h, i, w)			\
T1 = T1_SUM(e, f, g, h, i, w);					\
d += T1;							\
T2 = BIGSIGMA0_256(a) + Maj(a, b, c);				\
h = ADD2(T1, T2)


/*
* sparc optimization:
*
* on the sparc, we can load big endian 32-bit data easily.  note that
* special care must be taken to ensure the address is 32-bit aligned.
* in the interest of speed, we don't check to make sure, since
* careful programming can guarantee this for us.
*/

#if	defined(_BIG_ENDIAN)
#define	LOAD_BIG_32(addr)	(*(uint32_t *)(addr))
#define	LOAD_BIG_64(addr)	(*(uint64_t *)(addr))

#elif	defined(HAVE_HTONL)
#define	LOAD_BIG_32(addr) htonl(*((uint32_t *)(addr)))
#define	LOAD_BIG_64(addr) htonll(*((uint64_t *)(addr)))

#else
/* little endian -- will work on big endian, but slowly */
#define	LOAD_BIG_32(addr)	\
(((addr)[0] << 24) | ((addr)[1] << 16) | ((addr)[2] << 8) | (addr)[3])
#define	LOAD_BIG_64(addr)	\
(((uint64_t)(addr)[0] << 56) | ((uint64_t)(addr)[1] << 48) |	\
((uint64_t)(addr)[2] << 40) | ((uint64_t)(addr)[3] << 32) |	\
((uint64_t)(addr)[4] << 24) | ((uint64_t)(addr)[5] << 16) |	\
((uint64_t)(addr)[6] << 8) | (uint64_t)(addr)[7])
#endif	/* _BIG_ENDIAN */

#if 0
#define dumpstate() printf("%s: %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", __func__, w0, a, b, c, d, e, f, g, h);
#else
#define dumpstate()
#endif
void host_sha256_osol(const __sha256_block_t blk, __sha256_hash_t ctx) {
	uint32_t a = ctx[0];
	uint32_t b = ctx[1];
	uint32_t c = ctx[2];
	uint32_t d = ctx[3];
	uint32_t e = ctx[4];
	uint32_t f = ctx[5];
	uint32_t g = ctx[6];
	uint32_t h = ctx[7];

	uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
	uint32_t w8, w9, w10, w11, w12, w13, w14, w15;
	uint32_t T1, T2;

	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w0 = LOAD_BIG_32(blk + 4 * 0);
	dumpstate();
	SHA256ROUND(a, b, c, d, e, f, g, h, 0, w0);
	dumpstate();

	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w1 = LOAD_BIG_32(blk + 4 * 1);
	SHA256ROUND(h, a, b, c, d, e, f, g, 1, w1);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w2 = LOAD_BIG_32(blk + 4 * 2);
	SHA256ROUND(g, h, a, b, c, d, e, f, 2, w2);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w3 = LOAD_BIG_32(blk + 4 * 3);
	SHA256ROUND(f, g, h, a, b, c, d, e, 3, w3);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w4 = LOAD_BIG_32(blk + 4 * 4);
	SHA256ROUND(e, f, g, h, a, b, c, d, 4, w4);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w5 = LOAD_BIG_32(blk + 4 * 5);
	SHA256ROUND(d, e, f, g, h, a, b, c, 5, w5);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w6 = LOAD_BIG_32(blk + 4 * 6);
	SHA256ROUND(c, d, e, f, g, h, a, b, 6, w6);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w7 = LOAD_BIG_32(blk + 4 * 7);
	SHA256ROUND(b, c, d, e, f, g, h, a, 7, w7);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w8 = LOAD_BIG_32(blk + 4 * 8);
	SHA256ROUND(a, b, c, d, e, f, g, h, 8, w8);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w9 = LOAD_BIG_32(blk + 4 * 9);
	SHA256ROUND(h, a, b, c, d, e, f, g, 9, w9);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w10 = LOAD_BIG_32(blk + 4 * 10);
	SHA256ROUND(g, h, a, b, c, d, e, f, 10, w10);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w11 = LOAD_BIG_32(blk + 4 * 11);
	SHA256ROUND(f, g, h, a, b, c, d, e, 11, w11);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w12 = LOAD_BIG_32(blk + 4 * 12);
	SHA256ROUND(e, f, g, h, a, b, c, d, 12, w12);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w13 = LOAD_BIG_32(blk + 4 * 13);
	SHA256ROUND(d, e, f, g, h, a, b, c, 13, w13);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w14 = LOAD_BIG_32(blk + 4 * 14);
	SHA256ROUND(c, d, e, f, g, h, a, b, 14, w14);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w15 = LOAD_BIG_32(blk + 4 * 15);
	SHA256ROUND(b, c, d, e, f, g, h, a, 15, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 16, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 17, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 18, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 19, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 20, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 21, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 22, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 23, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 24, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 25, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 26, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 27, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 28, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 29, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 30, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 31, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 32, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 33, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 34, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 35, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 36, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 37, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 38, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 39, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 40, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 41, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 42, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 43, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 44, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 45, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 46, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 47, w15);

	w0 = SIGMA1_256(w14) + w9 + SIGMA0_256(w1) + w0;
	SHA256ROUND(a, b, c, d, e, f, g, h, 48, w0);
	w1 = SIGMA1_256(w15) + w10 + SIGMA0_256(w2) + w1;
	SHA256ROUND(h, a, b, c, d, e, f, g, 49, w1);
	w2 = SIGMA1_256(w0) + w11 + SIGMA0_256(w3) + w2;
	SHA256ROUND(g, h, a, b, c, d, e, f, 50, w2);
	w3 = SIGMA1_256(w1) + w12 + SIGMA0_256(w4) + w3;
	SHA256ROUND(f, g, h, a, b, c, d, e, 51, w3);
	w4 = SIGMA1_256(w2) + w13 + SIGMA0_256(w5) + w4;
	SHA256ROUND(e, f, g, h, a, b, c, d, 52, w4);
	w5 = SIGMA1_256(w3) + w14 + SIGMA0_256(w6) + w5;
	SHA256ROUND(d, e, f, g, h, a, b, c, 53, w5);
	w6 = SIGMA1_256(w4) + w15 + SIGMA0_256(w7) + w6;
	SHA256ROUND(c, d, e, f, g, h, a, b, 54, w6);
	w7 = SIGMA1_256(w5) + w0 + SIGMA0_256(w8) + w7;
	SHA256ROUND(b, c, d, e, f, g, h, a, 55, w7);
	w8 = SIGMA1_256(w6) + w1 + SIGMA0_256(w9) + w8;
	SHA256ROUND(a, b, c, d, e, f, g, h, 56, w8);
	w9 = SIGMA1_256(w7) + w2 + SIGMA0_256(w10) + w9;
	SHA256ROUND(h, a, b, c, d, e, f, g, 57, w9);
	w10 = SIGMA1_256(w8) + w3 + SIGMA0_256(w11) + w10;
	SHA256ROUND(g, h, a, b, c, d, e, f, 58, w10);
	w11 = SIGMA1_256(w9) + w4 + SIGMA0_256(w12) + w11;
	SHA256ROUND(f, g, h, a, b, c, d, e, 59, w11);
	w12 = SIGMA1_256(w10) + w5 + SIGMA0_256(w13) + w12;
	SHA256ROUND(e, f, g, h, a, b, c, d, 60, w12);
	w13 = SIGMA1_256(w11) + w6 + SIGMA0_256(w14) + w13;
	SHA256ROUND(d, e, f, g, h, a, b, c, 61, w13);
	w14 = SIGMA1_256(w12) + w7 + SIGMA0_256(w15) + w14;
	SHA256ROUND(c, d, e, f, g, h, a, b, 62, w14);
	w15 = SIGMA1_256(w13) + w8 + SIGMA0_256(w0) + w15;
	SHA256ROUND(b, c, d, e, f, g, h, a, 63, w15);


	ctx[0] += a;
	ctx[1] += b;
	ctx[2] += c;
	ctx[3] += d;
	ctx[4] += e;
	ctx[5] += f;
	ctx[6] += g;
	ctx[7] += h;

}

__device__ void device_sha256_osol(const uint32_t *w, uint32_t *ctx) {
	uint32_t a = ctx[0];
	uint32_t b = ctx[1];
	uint32_t c = ctx[2];
	uint32_t d = ctx[3];
	uint32_t e = ctx[4];
	uint32_t f = ctx[5];
	uint32_t g = ctx[6];
	uint32_t h = ctx[7];

	uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
	uint32_t w8, w9, w10, w11, w12, w13, w14, w15;
	uint32_t T1, T2;

	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w0 = w[0];
	dumpstate();
	SHA256ROUND(a, b, c, d, e, f, g, h, 0, w0);
	dumpstate();

	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w1 = w[1];
	SHA256ROUND(h, a, b, c, d, e, f, g, 1, w1);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w2 = w[2];
	SHA256ROUND(g, h, a, b, c, d, e, f, 2, w2);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w3 = w[3];
	SHA256ROUND(f, g, h, a, b, c, d, e, 3, w3);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w4 = w[4];
	SHA256ROUND(e, f, g, h, a, b, c, d, 4, w4);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w5 = w[5];
	SHA256ROUND(d, e, f, g, h, a, b, c, 5, w5);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w6 = w[6];
	SHA256ROUND(c, d, e, f, g, h, a, b, 6, w6);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w7 = w[7];
	SHA256ROUND(b, c, d, e, f, g, h, a, 7, w7);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w8 = w[8];
	SHA256ROUND(a, b, c, d, e, f, g, h, 8, w8);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w9 = w[9];
	SHA256ROUND(h, a, b, c, d, e, f, g, 9, w9);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w10 = w[10];
	SHA256ROUND(g, h, a, b, c, d, e, f, 10, w10);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w11 = w[11];
	SHA256ROUND(f, g, h, a, b, c, d, e, 11, w11);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w12 = w[12];
	SHA256ROUND(e, f, g, h, a, b, c, d, 12, w12);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w13 = w[13];
	SHA256ROUND(d, e, f, g, h, a, b, c, 13, w13);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w14 = w[14];
	SHA256ROUND(c, d, e, f, g, h, a, b, 14, w14);
	/* LINTED E_BAD_PTR_CAST_ALIGN */
	w15 = w[15];
	SHA256ROUND(b, c, d, e, f, g, h, a, 15, w15);

	w0 = W_NEXT(w14, w9, w1, w0);
	SHA256ROUND(a, b, c, d, e, f, g, h, 16, w0);
	w1 = W_NEXT(w15, w10, w2, w1);
	SHA256ROUND(h, a, b, c, d, e, f, g, 17, w1);
	w2 = W_NEXT(w0, w11, w3, w2);
	SHA256ROUND(g, h, a, b, c, d, e, f, 18, w2);
	w3 = W_NEXT(w1, w12, w4, w3);
	SHA256ROUND(f, g, h, a, b, c, d, e, 19, w3);
	w4 = W_NEXT(w2, w13, w5, w4);
	SHA256ROUND(e, f, g, h, a, b, c, d, 20, w4);
	w5 = W_NEXT(w3, w14, w6, w5);
	SHA256ROUND(d, e, f, g, h, a, b, c, 21, w5);
	w6 = W_NEXT(w4, w15, w7, w6);
	SHA256ROUND(c, d, e, f, g, h, a, b, 22, w6);
	w7 = W_NEXT(w5, w0, w8, w7);
	SHA256ROUND(b, c, d, e, f, g, h, a, 23, w7);
	w8 = W_NEXT(w6, w1, w9, w8);
	SHA256ROUND(a, b, c, d, e, f, g, h, 24, w8);
	w9 = W_NEXT(w7, w2, w10, w9);
	SHA256ROUND(h, a, b, c, d, e, f, g, 25, w9);
	w10 = W_NEXT(w8, w3, w11, w10);
	SHA256ROUND(g, h, a, b, c, d, e, f, 26, w10);
	w11 = W_NEXT(w9, w4, w12, w11);
	SHA256ROUND(f, g, h, a, b, c, d, e, 27, w11);
	w12 = W_NEXT(w10, w5, w13, w12);
	SHA256ROUND(e, f, g, h, a, b, c, d, 28, w12);
	w13 = W_NEXT(w11, w6, w14, w13);
	SHA256ROUND(d, e, f, g, h, a, b, c, 29, w13);
	w14 = W_NEXT(w12, w7, w15, w14);
	SHA256ROUND(c, d, e, f, g, h, a, b, 30, w14);
	w15 = W_NEXT(w13, w8, w0, w15);
	SHA256ROUND(b, c, d, e, f, g, h, a, 31, w15);

	w0 = W_NEXT(w14, w9, w1, w0);
	SHA256ROUND(a, b, c, d, e, f, g, h, 32, w0);
	w1 = W_NEXT(w15, w10, w2, w1);
	SHA256ROUND(h, a, b, c, d, e, f, g, 33, w1);
	w2 = W_NEXT(w0, w11, w3, w2);
	SHA256ROUND(g, h, a, b, c, d, e, f, 34, w2);
	w3 = W_NEXT(w1, w12, w4, w3);
	SHA256ROUND(f, g, h, a, b, c, d, e, 35, w3);
	w4 = W_NEXT(w2, w13, w5, w4);
	SHA256ROUND(e, f, g, h, a, b, c, d, 36, w4);
	w5 = W_NEXT(w3, w14, w6, w5);
	SHA256ROUND(d, e, f, g, h, a, b, c, 37, w5);
	w6 = W_NEXT(w4, w15, w7, w6);
	SHA256ROUND(c, d, e, f, g, h, a, b, 38, w6);
	w7 = W_NEXT(w5, w0, w8, w7);
	SHA256ROUND(b, c, d, e, f, g, h, a, 39, w7);
	w8 = W_NEXT(w6, w1, w9, w8);
	SHA256ROUND(a, b, c, d, e, f, g, h, 40, w8);
	w9 = W_NEXT(w7, w2, w10, w9);
	SHA256ROUND(h, a, b, c, d, e, f, g, 41, w9);
	w10 = W_NEXT(w8, w3, w11, w10);
	SHA256ROUND(g, h, a, b, c, d, e, f, 42, w10);
	w11 = W_NEXT(w9, w4, w12, w11);
	SHA256ROUND(f, g, h, a, b, c, d, e, 43, w11);
	w12 = W_NEXT(w10, w5, w13, w12);
	SHA256ROUND(e, f, g, h, a, b, c, d, 44, w12);
	w13 = W_NEXT(w11, w6, w14, w13);
	SHA256ROUND(d, e, f, g, h, a, b, c, 45, w13);
	w14 = W_NEXT(w12, w7, w15, w14);
	SHA256ROUND(c, d, e, f, g, h, a, b, 46, w14);
	w15 = W_NEXT(w13, w8, w0, w15);
	SHA256ROUND(b, c, d, e, f, g, h, a, 47, w15);

	w0 = W_NEXT(w14, w9, w1, w0);
	SHA256ROUND(a, b, c, d, e, f, g, h, 48, w0);
	w1 = W_NEXT(w15, w10, w2, w1);
	SHA256ROUND(h, a, b, c, d, e, f, g, 49, w1);
	w2 = W_NEXT(w0, w11, w3, w2);
	SHA256ROUND(g, h, a, b, c, d, e, f, 50, w2);
	w3 = W_NEXT(w1, w12, w4, w3);
	SHA256ROUND(f, g, h, a, b, c, d, e, 51, w3);
	w4 = W_NEXT(w2, w13, w5, w4);
	SHA256ROUND(e, f, g, h, a, b, c, d, 52, w4);
	w5 = W_NEXT(w3, w14, w6, w5);
	SHA256ROUND(d, e, f, g, h, a, b, c, 53, w5);
	w6 = W_NEXT(w4, w15, w7, w6);
	SHA256ROUND(c, d, e, f, g, h, a, b, 54, w6);
	w7 = W_NEXT(w5, w0, w8, w7);
	SHA256ROUND(b, c, d, e, f, g, h, a, 55, w7);
	w8 = W_NEXT(w6, w1, w9, w8);
	SHA256ROUND(a, b, c, d, e, f, g, h, 56, w8);
	w9 = W_NEXT(w7, w2, w10, w9);
	SHA256ROUND(h, a, b, c, d, e, f, g, 57, w9);
	w10 = W_NEXT(w8, w3, w11, w10);
	SHA256ROUND(g, h, a, b, c, d, e, f, 58, w10);
	w11 = W_NEXT(w9, w4, w12, w11);
	SHA256ROUND(f, g, h, a, b, c, d, e, 59, w11);
	w12 = W_NEXT(w10, w5, w13, w12);
	SHA256ROUND(e, f, g, h, a, b, c, d, 60, w12);
	w13 = W_NEXT(w11, w6, w14, w13);
	SHA256ROUND(d, e, f, g, h, a, b, c, 61, w13);
	w14 = W_NEXT(w12, w7, w15, w14);
	SHA256ROUND(c, d, e, f, g, h, a, b, 62, w14);
	w15 = W_NEXT(w13, w8, w0, w15);
	SHA256ROUND(b, c, d, e, f, g, h, a, 63, w15);


	ctx[0] += a;
	ctx[1] += b;
	ctx[2] += c;
	ctx[3] += d;
	ctx[4] += e;
	ctx[5] += f;
	ctx[6] += g;
	ctx[7] += h;

}

// SHA-256 of a short byte message (len <= 183), result as native state words.
__device__ void device_sha256_bytes(const uint8_t *data, uint32_t len, uint32_t *ctx)
{
	uint8_t block[3 * 64];
	const uint32_t num_blocks = len / 64 + 1;
	const uint32_t bits = len * 8;
	memcpy(block, data, len);
	block[len] = 0x80;
	memset(block + len + 1, 0, num_blocks * 64 - 8 - (len + 1));
	uint8_t *lp = block + num_blocks * 64 - 8;
	lp[0] = 0; lp[1] = 0; lp[2] = 0; lp[3] = 0;
	lp[4] = 0; lp[5] = 0; lp[6] = (bits >> 8) & 0xff; lp[7] = bits & 0xff;

	for (int k = 0; k < 8; k++) ctx[k] = __sha256_init[k];
	for (uint32_t b = 0; b < num_blocks; b++) {
		uint32_t w[16];
		const uint8_t *bp = block + 64 * b;
		#pragma unroll
		for (int k = 0; k < 16; k++)
			w[k] = LOAD_BIG_32(bp + 4 * k);
		device_sha256_osol(w, ctx);
	}
}

__device__ __forceinline__ void sha256_iv(uint32_t *st)
{
	st[0] = 0x6a09e667; st[1] = 0xbb67ae85; st[2] = 0x3c6ef372; st[3] = 0xa54ff53a;
	st[4] = 0x510e527f; st[5] = 0x9b05688c; st[6] = 0x1f83d9ab; st[7] = 0x5be0cd19;
}

__device__ __forceinline__ void load_block(const uint32_t *buf, uint32_t off, uint32_t *v)
{
	const uint4 a = *(const uint4 *)(buf + off), b = *(const uint4 *)(buf + off + 4);
	v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
	v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

__device__ __forceinline__ void store_block(uint32_t *buf, uint32_t off, const uint32_t *v)
{
	*(uint4 *)(buf + off) = make_uint4(v[0], v[1], v[2], v[3]);
	*(uint4 *)(buf + off + 4) = make_uint4(v[4], v[5], v[6], v[7]);
}

// Blocks are stored as SHA-256 state words (a digest read big-endian), so each
// message is built in registers and the previous block stays in registers. The
// 64-bit little-endian counter stays below 2^15, so its high word is zero.
//
// Fill: block 0 = H(counter=0 || salt || input || s_cost || t_cost),
// block i = H(counter=i || block i-1). Returns the last block in prev[].
__device__ void balloon_fill(uint32_t *buf, const uint8_t *in, uint32_t *prev)
{
	const int64_t s_cost = BALLOON_SCOST;
	const int32_t t_cost = BALLOON_MIXROUNDS;
	uint8_t data[132];
	memset(data, 0, 8);
	memcpy(data + 8, in, SALT_LEN);
	memcpy(data + 8 + SALT_LEN, in, BALLOON_INLEN);
	memcpy(data + 8 + SALT_LEN + BALLOON_INLEN, &s_cost, 8);
	memcpy(data + 16 + SALT_LEN + BALLOON_INLEN, &t_cost, 4);
	device_sha256_bytes(data, 132, prev);
	store_block(buf, 0, prev);

	for (uint32_t i = 1; i < 4096; i++) {
		uint32_t m[16];
		m[0] = cuda_swab32(i); m[1] = 0;
		#pragma unroll
		for (int k = 0; k < 8; k++) m[2 + k] = prev[k];
		m[10] = 0x80000000U; m[11] = 0; m[12] = 0; m[13] = 0; m[14] = 0;
		m[15] = 40 * 8;
		sha256_iv(prev);
		device_sha256_osol(m, prev);
		store_block(buf, i * 8, prev);
	}
}

// Mix: block i = H(counter || block i-1 || block i || 3 indexed blocks), 168 bytes
// = 3 transforms. prebuf_le holds byte offsets, hence >> 2 to word offsets.
__device__ void balloon_mix(uint32_t *buf, const uint64_t *prebuf_le, uint32_t *prev)
{
	const uint64_t *idx = prebuf_le;
	uint32_t counter = 4096;
	for (int32_t r = 0; r < BALLOON_MIXROUNDS; r++) {
		for (uint32_t i = 0; i < 4096; i++) {
			uint32_t cur[8], b2[8], b3[8], b4[8], m[16];
			const uint32_t o2 = (uint32_t)idx[0] >> 2;
			const uint32_t o3 = (uint32_t)idx[1] >> 2;
			const uint32_t o4 = (uint32_t)idx[2] >> 2;
			idx += 3;
			load_block(buf, i * 8, cur);
			load_block(buf, o2, b2);
			load_block(buf, o3, b3);
			load_block(buf, o4, b4);

			m[0] = cuda_swab32(counter); m[1] = 0;
			#pragma unroll
			for (int k = 0; k < 8; k++) m[2 + k] = prev[k];
			#pragma unroll
			for (int k = 0; k < 6; k++) m[10 + k] = cur[k];
			sha256_iv(prev);
			device_sha256_osol(m, prev);

			m[0] = cur[6]; m[1] = cur[7];
			#pragma unroll
			for (int k = 0; k < 8; k++) m[2 + k] = b2[k];
			#pragma unroll
			for (int k = 0; k < 6; k++) m[10 + k] = b3[k];
			device_sha256_osol(m, prev);

			m[0] = b3[6]; m[1] = b3[7];
			#pragma unroll
			for (int k = 0; k < 8; k++) m[2 + k] = b4[k];
			m[10] = 0x80000000U; m[11] = 0; m[12] = 0; m[13] = 0; m[14] = 0;
			m[15] = 168 * 8;
			device_sha256_osol(m, prev);

			store_block(buf, i * 8, prev);
			counter++;
		}
	}
}
