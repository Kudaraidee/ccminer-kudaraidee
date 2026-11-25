/**
 * X25X algorithm (SUQA / SIN) — 25-stage accumulate-all chain.
 *
 * Unlike the x16 family this is a fixed-order pipeline: every stage writes its
 * own 64-byte slot hash[0..24] (rather than feeding the next stage in place),
 * then a 12-round byte shuffle mixes all 24 produced slots and a final BLAKE2s
 * over those 1536 bytes yields slot 24, whose first 32 bytes are the result.
 *
 * Reference chain transcribed from cpuminer-opt algo/x22/x25x.c.
 *
 * The GPU pipeline runs the linear sub-chains through the shared x-family stage
 * kernels on a flat working buffer, snapshotting each result into a per-thread
 * 24-slot accumulator; SWIFFTX consumes accumulator slots 12..15 and the shuffle
 * + BLAKE2s finaliser consume all 24. x25x_hash() is the host reference and the
 * safety-net re-verify for every GPU candidate.
 */

#include <stdio.h>
#include <memory.h>
#include <stdint.h>

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"

#include "sph/sph_haval.h"
#include "sph/sph_tiger.h"
#include "sph/sph_streebog.h"
#include "sph/blake2s.h"

#include "algos/lyra2/Lyra2.h"
#include "algos/x25x/swifftx.h"
#include "sph/sph_panama.h"
#include "algos/x25x/lane.h"
}

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

// New x25x GPU stage launchers (each validated in isolation; see M2/M3).
extern void x25x_panama_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x25x_lane_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x25x_blake2s_cpu_hash(int thr_id, uint32_t threads, uint32_t *d_acc, uint32_t *d_out);
extern void x25x_shuffle_cpu(int thr_id, uint32_t threads, uint32_t *d_acc);
extern void x25x_swifftx_cpu_hash_acc(int thr_id, uint32_t threads, uint32_t *d_acc, uint32_t *d_hash);
extern void x25x_swifftx_cpu_init(int thr_id);

// Tail-stage launchers not declared in the shared bridge header (as in x21s).
extern void streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern bool streebog_device_selftest(int thr_id);
extern void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, int order);
extern void sha256_cpu_hash_64(int thr_id, int threads, uint32_t *d_hash);

// Fixed-order chain, stages 0..23 into hash[0..23] (pre-shuffle). Split out so
// the M4 pipeline debug can compare per-slot against the GPU accumulator.
static void x25x_intermediates(const void *input, unsigned char hash[25][64])
{
	memset(hash, 0, 25 * 64);

	sph_blake512_context   ctx_blake;
	sph_bmw512_context     ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_skein512_context   ctx_skein;
	sph_jh512_context      ctx_jh;
	sph_keccak512_context  ctx_keccak;
	sph_luffa512_context   ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context    ctx_simd;
	sph_echo512_context    ctx_echo;
	sph_hamsi512_context   ctx_hamsi;
	sph_fugue512_context   ctx_fugue;
	sph_shabal512_context  ctx_shabal;
	sph_whirlpool_context  ctx_whirlpool;
	sph_sha512_context     ctx_sha512;
	sph_haval256_5_context ctx_haval;
	sph_tiger_context      ctx_tiger;
	sph_gost512_context    ctx_gost;
	sph_sha256_context     ctx_sha256;
	sph_panama_context     ctx_panama;

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, hash[0]);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, hash[0], 64);
	sph_bmw512_close(&ctx_bmw, hash[1]);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, hash[1], 64);
	sph_groestl512_close(&ctx_groestl, hash[2]);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash[2], 64);
	sph_skein512_close(&ctx_skein, hash[3]);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, hash[3], 64);
	sph_jh512_close(&ctx_jh, hash[4]);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, hash[4], 64);
	sph_keccak512_close(&ctx_keccak, hash[5]);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, hash[5], 64);
	sph_luffa512_close(&ctx_luffa, hash[6]);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, hash[6], 64);
	sph_cubehash512_close(&ctx_cubehash, hash[7]);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, hash[7], 64);
	sph_shavite512_close(&ctx_shavite, hash[8]);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, hash[8], 64);
	sph_simd512_close(&ctx_simd, hash[9]);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, hash[9], 64);
	sph_echo512_close(&ctx_echo, hash[10]);

	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, hash[10], 64);
	sph_hamsi512_close(&ctx_hamsi, hash[11]);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, hash[11], 64);
	sph_fugue512_close(&ctx_fugue, hash[12]);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash[12], 64);
	sph_shabal512_close(&ctx_shabal, hash[13]);

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, hash[13], 64);
	sph_whirlpool_close(&ctx_whirlpool, hash[14]);

	sph_sha512_init(&ctx_sha512);
	sph_sha512(&ctx_sha512, hash[14], 64);
	sph_sha512_close(&ctx_sha512, hash[15]);

	// SWIFFTX consumes the 256-byte window hash[12..15], not a single 64B slot.
	ComputeSingleSWIFFTX((unsigned char*)hash[12], (unsigned char*)hash[16]);

	sph_haval256_5_init(&ctx_haval);
	sph_haval256_5(&ctx_haval, hash[16], 64);
	sph_haval256_5_close(&ctx_haval, hash[17]);

	sph_tiger_init(&ctx_tiger);
	sph_tiger(&ctx_tiger, hash[17], 64);
	sph_tiger_close(&ctx_tiger, hash[18]);

	LYRA2(hash[19], 32, hash[18], 32, hash[18], 32, 1, 4, 4);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, hash[19], 64);
	sph_gost512_close(&ctx_gost, hash[20]);

	sph_sha256_init(&ctx_sha256);
	sph_sha256(&ctx_sha256, hash[20], 64);
	sph_sha256_close(&ctx_sha256, hash[21]);

	sph_panama_init(&ctx_panama);
	sph_panama(&ctx_panama, hash[21], 64);
	sph_panama_close(&ctx_panama, hash[22]);

	laneHash(512, (const BitSequence*)hash[22], 512, (BitSequence*)hash[23]);
}

// X25X CPU hash (fixed-order accumulate-all reference / validation).
extern "C" void x25x_hash(void *output, const void *input)
{
	unsigned char _ALIGN(64) hash[25][64];
	x25x_intermediates(input, hash);

	// 12-round byte shuffle over the 24 produced slots (uint16 view).
	#define X25X_SHUFFLE_BLOCKS (24 * 64 / 2)
	#define X25X_SHUFFLE_ROUNDS 12

	static const uint16_t x25x_round_const[X25X_SHUFFLE_ROUNDS] = {
		0x142c, 0x5830, 0x678c, 0xe08c, 0x3c67, 0xd50d, 0xb1d8, 0xecb2,
		0xd7ee, 0x6783, 0xfa6c, 0x4b9c
	};

	uint16_t *block_pointer = (uint16_t*)hash;
	for (int r = 0; r < X25X_SHUFFLE_ROUNDS; r++) {
		for (int i = 0; i < X25X_SHUFFLE_BLOCKS; i++) {
			uint16_t block_value = block_pointer[X25X_SHUFFLE_BLOCKS - i - 1];
			block_pointer[i] ^= block_pointer[block_value % X25X_SHUFFLE_BLOCKS]
			                    + (x25x_round_const[r] << (i % 16));
		}
	}

	#undef X25X_SHUFFLE_BLOCKS
	#undef X25X_SHUFFLE_ROUNDS

	blake2s_simple((uint8_t*)hash[24], (const void*)hash[0], 64 * 24);

	memcpy(output, hash[24], 32);
}

// ===================== M4: full GPU pipeline =====================
// Per-thread buffers: a flat 64-byte working slot d_hash carries the linear
// sub-chains through the existing x-family stage kernels; each stage's output
// is snapshotted into a 24-slot (1536-byte) accumulator d_acc. SWIFFTX reads
// slots 12..15 of d_acc; the shuffle and blake2s finaliser read all of d_acc.

#define X25X_ACC_U64 (24 * 8)   // 24 slots x 8 uint64 = 1536 bytes / thread

static uint32_t *d_hash[MAX_GPUS]     = { 0 };
static uint32_t *d_acc[MAX_GPUS]      = { 0 };
static uint32_t *d_final[MAX_GPUS]    = { 0 };
static uint64_t *d_matrix[MAX_GPUS]   = { 0 };
static uint32_t *d_resNonce[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { 0 };

// snapshot the 64-byte working hash into accumulator slot `slot`.
// d_acc is SLOT-MAJOR: plane s occupies [s*threads*64 .. (s+1)*threads*64), with
// thread t at +t*64. So this write is COALESCED across threads (consecutive t ->
// consecutive 64 B), unlike a thread-major layout; the shuffle/swifftx/blake2s
// readers gather per-thread across the 24 planes (also coalesced per plane).
__global__ void x25x_snap_gpu(uint32_t threads, const uint64_t *d_h, uint64_t *d_a, uint32_t slot)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t < threads) {
		const uint64_t *s = &d_h[(size_t)t << 3];
		uint64_t *d = &d_a[(size_t)slot * threads * 8 + (size_t)t * 8];
		#pragma unroll
		for (int i = 0; i < 8; i++) d[i] = s[i];
	}
}
__host__ static void x25x_snap(uint32_t threads, uint32_t *d_h, uint32_t *d_a, uint32_t slot)
{
	const uint32_t tpb = 256; dim3 grid((threads + tpb - 1) / tpb), block(tpb);
	x25x_snap_gpu<<<grid, block>>>(threads, (uint64_t*)d_h, (uint64_t*)d_a, slot);
}

// zero working-hash 32-bit words [keep..15] (zero-pad a short digest to 64 bytes)
__global__ void x25x_zeropad_gpu(uint32_t threads, uint32_t *d_h, uint32_t keep)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t < threads) {
		uint32_t *h = &d_h[(size_t)t << 4];
		for (uint32_t i = keep; i < 16; i++) h[i] = 0;
	}
}
__host__ static void x25x_zeropad(uint32_t threads, uint32_t *d_h, uint32_t keep)
{
	const uint32_t tpb = 256; dim3 grid((threads + tpb - 1) / tpb), block(tpb);
	x25x_zeropad_gpu<<<grid, block>>>(threads, d_h, keep);
}

// screen the 32-byte results: record threads whose top 64 bits <= target.
// A necessary condition for hash<=target, so no solution is missed; the host
// re-verifies every hit with x25x_hash + fulltest.
__global__ void x25x_final_check_gpu(uint32_t threads, uint32_t startNonce, const uint64_t *d_f, uint64_t target, uint32_t *resNonce)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t < threads) {
		const uint64_t hi = d_f[((size_t)t << 2) + 3];   // words[6],[7] of the 32-byte result
		if (hi <= target) {
			uint32_t tmp = atomicExch(&resNonce[0], startNonce + t);
			if (tmp != UINT32_MAX) resNonce[1] = tmp;
		}
	}
}

// Run the full 25-stage chain over `threads` nonces; leaves 32-byte results in df.
static void x25x_pipeline(int thr_id, uint32_t threads, uint32_t startNonce,
                          uint32_t *dh, uint32_t *da, uint32_t *df)
{
	int order = 0;

	blake512_cpu_hash_80(thr_id, threads, startNonce, dh);                      x25x_snap(threads, dh, da, 0);
	bmw512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);         x25x_snap(threads, dh, da, 1);
	groestl512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);     x25x_snap(threads, dh, da, 2);
	skein512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);       x25x_snap(threads, dh, da, 3);
	jh512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);          x25x_snap(threads, dh, da, 4);
	keccak512_cpu_hash_64(thr_id, threads, NULL, dh);                           x25x_snap(threads, dh, da, 5);
	luffa512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);       x25x_snap(threads, dh, da, 6);
	cubehash512_cpu_hash_64(thr_id, threads, dh);                               x25x_snap(threads, dh, da, 7);
	shavite512_cpu_hash_64(thr_id, threads, dh);                                x25x_snap(threads, dh, da, 8);
	simd512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);        x25x_snap(threads, dh, da, 9);
	echo512_cpu_hash_64(thr_id, threads, dh);                                   x25x_snap(threads, dh, da, 10);
	hamsi512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);       x25x_snap(threads, dh, da, 11);
	fugue512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);       x25x_snap(threads, dh, da, 12);
	shabal512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);      x25x_snap(threads, dh, da, 13);
	whirlpool512_cpu_hash_64(thr_id, threads, startNonce, NULL, dh, order++);   x25x_snap(threads, dh, da, 14);
	sha512_cpu_hash_64(thr_id, threads, startNonce, dh);                        x25x_snap(threads, dh, da, 15);

	// SWIFFTX consumes the 256-byte window (slots 12..15) straight from d_acc.
	x25x_swifftx_cpu_hash_acc(thr_id, threads, da, dh);                         x25x_snap(threads, dh, da, 16);

	haval256_cpu_hash_64(thr_id, threads, startNonce, dh, 512);
	x25x_zeropad(threads, dh, 8);                                              x25x_snap(threads, dh, da, 17);
	tiger192_cpu_hash_64(thr_id, threads, 0, dh);
	x25x_zeropad(threads, dh, 6);                                              x25x_snap(threads, dh, da, 18);
	lyra2v2_cpu_hash_32(thr_id, threads, startNonce, (uint64_t*)dh, order++);
	x25x_zeropad(threads, dh, 8);                                              x25x_snap(threads, dh, da, 19);
	streebog_cpu_hash_64(thr_id, threads, dh);                                 x25x_snap(threads, dh, da, 20);
	sha256_cpu_hash_64(thr_id, threads, dh);
	x25x_zeropad(threads, dh, 8);                                              x25x_snap(threads, dh, da, 21);
	x25x_panama_cpu_hash_64(thr_id, threads, dh);                              x25x_snap(threads, dh, da, 22);
	x25x_lane_cpu_hash_64(thr_id, threads, dh);                                x25x_snap(threads, dh, da, 23);

	x25x_shuffle_cpu(thr_id, threads, da);
	x25x_blake2s_cpu_hash(thr_id, threads, da, df);
}

// Init-time device self-test for the 5 new primitives (KAT vs the CPU refs +
// a negative bit-flip check that proves the comparison isn't vacuous). Runs
// once per GPU; a mismatch logs a loud error. Uses the d_hash/d_acc/d_final
// scratch (threads=1: slot-major == contiguous, so the window/gather line up).
static void x25x_selftest(int thr_id, uint32_t *dh, uint32_t *da, uint32_t *df)
{
	static bool done[MAX_GPUS] = { 0 };
	if (done[thr_id]) return;
	done[thr_id] = true;

	static const uint16_t rc[12] = {
		0x142c,0x5830,0x678c,0xe08c,0x3c67,0xd50d,0xb1d8,0xecb2,0xd7ee,0x6783,0xfa6c,0x4b9c };
	unsigned char in[1536], got[1536], gold[1536];
	bool ok = true;

	// panama: 64B -> 32B  (+ negative: a 1-bit-flipped input must change the digest)
	for (int i = 0; i < 64; i++) in[i] = (unsigned char)(i * 7 + 1);
	sph_panama_context pc; sph_panama_init(&pc); sph_panama(&pc, in, 64); sph_panama_close(&pc, gold);
	cudaMemcpy(dh, in, 64, cudaMemcpyHostToDevice);
	x25x_panama_cpu_hash_64(thr_id, 1, dh);
	cudaMemcpy(got, dh, 64, cudaMemcpyDeviceToHost);
	ok &= (memcmp(gold, got, 32) == 0);
	in[0] ^= 1;
	cudaMemcpy(dh, in, 64, cudaMemcpyHostToDevice);
	x25x_panama_cpu_hash_64(thr_id, 1, dh);
	unsigned char neg[64]; cudaMemcpy(neg, dh, 64, cudaMemcpyDeviceToHost);
	ok &= (memcmp(got, neg, 32) != 0);   // not vacuous

	// lane-512: 64B -> 64B
	for (int i = 0; i < 64; i++) in[i] = (unsigned char)(i * 11 + 3);
	laneHash(512, (const BitSequence*)in, 512, (BitSequence*)gold);
	cudaMemcpy(dh, in, 64, cudaMemcpyHostToDevice);
	x25x_lane_cpu_hash_64(thr_id, 1, dh);
	cudaMemcpy(got, dh, 64, cudaMemcpyDeviceToHost);
	ok &= (memcmp(gold, got, 64) == 0);

	// shuffle: 1536B in place
	for (int i = 0; i < 1536; i++) in[i] = (unsigned char)(i * 131 + 7);
	memcpy(gold, in, 1536);
	{
		uint16_t *bp = (uint16_t*)gold;
		for (int r = 0; r < 12; r++)
			for (int i = 0; i < 768; i++) {
				uint16_t bv = bp[768 - i - 1];
				bp[i] ^= bp[bv % 768] + (rc[r] << (i % 16));
			}
	}
	cudaMemcpy(da, in, 1536, cudaMemcpyHostToDevice);
	x25x_shuffle_cpu(thr_id, 1, da);
	cudaMemcpy(got, da, 1536, cudaMemcpyDeviceToHost);
	ok &= (memcmp(gold, got, 1536) == 0);

	// blake2s: 1536B -> 32B
	for (int i = 0; i < 1536; i++) in[i] = (unsigned char)(i * 97 + 5);
	blake2s_simple(gold, in, 1536);
	cudaMemcpy(da, in, 1536, cudaMemcpyHostToDevice);
	x25x_blake2s_cpu_hash(thr_id, 1, da, df);
	cudaMemcpy(got, df, 32, cudaMemcpyDeviceToHost);
	ok &= (memcmp(gold, got, 32) == 0);

	// swifftx: 256B window (accumulator slots 12..15) -> 64B
	for (int i = 0; i < 256; i++) in[i] = (unsigned char)(i * 29 + 3);
	ComputeSingleSWIFFTX(in, gold);
	cudaMemcpy((char*)da + 12 * 64, in, 256, cudaMemcpyHostToDevice);
	x25x_swifftx_cpu_hash_acc(thr_id, 1, da, dh);
	cudaMemcpy(got, dh, 64, cudaMemcpyDeviceToHost);
	ok &= (memcmp(gold, got, 64) == 0);

	if (ok)
		applog(LOG_INFO, "GPU #%d: x25x device self-test passed (swifftx/panama/lane/blake2s/shuffle)", device_map[thr_id]);
	else
		applog(LOG_ERR, "GPU #%d: x25x device self-test FAILED", device_map[thr_id]);
}

extern "C" int scanhash_x25x(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	uint32_t _ALIGN(64) endiandata[20];

	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 19;
	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 20;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x08ff;

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		// the lyra2v2 wander matrix is on chip; d_matrix only carries the 4 x uint2x4
		// sponge state, at the init/final kernels' 64-thread padded stride
		const size_t state_sz = 4 * 4 * sizeof(uint64_t);
		const size_t padded = ((size_t)throughput + 63) & ~(size_t)63;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_matrix[thr_id], state_sz * padded), -1);

		cuda_get_arch(thr_id);

		blake512_cpu_init(thr_id, throughput);
		bmw512_cpu_init(thr_id, throughput);
		groestl512_cpu_init(thr_id, throughput);
		skein512_cpu_init(thr_id, throughput);
		jh512_cpu_init(thr_id, throughput);
		keccak512_cpu_init(thr_id, throughput);
		luffa512_cpu_init(thr_id, throughput);
		shavite512_cpu_init(thr_id, throughput);
		simd512_cpu_init(thr_id, throughput);
		x16_echo512_cpu_init(thr_id, throughput);
		hamsi512_cpu_init(thr_id, throughput);
		fugue512_cpu_init(thr_id, throughput);
		shabal512_cpu_init(thr_id, throughput);
		whirlpool512_cpu_init(thr_id, throughput, 0);
		sha512_cpu_init(thr_id, throughput);
		haval256_cpu_init(thr_id, throughput);
		lyra2v2_cpu_init(thr_id, throughput, d_matrix[thr_id]);
		x25x_swifftx_cpu_init(thr_id);   // host InitializeSWIFFTX + upload tables

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id],   (size_t)64   * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_acc[thr_id],    (size_t)1536 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_final[thr_id],  (size_t)32   * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)), -1);

		x25x_selftest(thr_id, d_hash[thr_id], d_acc[thr_id], d_final[thr_id]);
		streebog_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake512_cpu_setBlock_80(thr_id, endiandata);

	const uint64_t target64 = ((uint64_t)ptarget[7] << 32) | ptarget[6];
	const uint32_t tpb = 256;

	do {
		cudaMemset(d_resNonce[thr_id], 0xff, 2 * sizeof(uint32_t));

		x25x_pipeline(thr_id, throughput, pdata[19], d_hash[thr_id], d_acc[thr_id], d_final[thr_id]);

		dim3 grid((throughput + tpb - 1) / tpb), block(tpb);
		x25x_final_check_gpu<<<grid, block>>>(throughput, pdata[19], (uint64_t*)d_final[thr_id], target64, d_resNonce[thr_id]);

		uint32_t resNonce[2] = { UINT32_MAX, UINT32_MAX };
		cudaMemcpy(resNonce, d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (resNonce[0] != UINT32_MAX) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], resNonce[0]);
			x25x_hash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[0] = resNonce[0];
				work_set_target_ratio(work, vhash);
				if (resNonce[1] != UINT32_MAX) {
					be32enc(&endiandata[19], resNonce[1]);
					x25x_hash(vhash, endiandata);
					work->nonces[1] = resNonce[1];
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(resNonce[0], resNonce[1]) + 1;
				} else {
					pdata[19] = resNonce[0] + 1;
				}
				return work->valid_nonces;
			} else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", resNonce[0]);
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_x25x(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_acc[thr_id]);
	cudaFree(d_final[thr_id]);
	cudaFree(d_matrix[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);
	simd512_cpu_free(thr_id);
	fugue512_cpu_free(thr_id);
	whirlpool512_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
