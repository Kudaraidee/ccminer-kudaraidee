/**
 * GhostRider algorithm (Raptoreum / PEPEW "ghostrider")
 *
 * Consensus hash = 15 core rounds (the x16 set minus SHA-512) interleaved with
 * 3 CryptoNight-v1 rounds, in three groups of (5 core + 1 CN). Both the core
 * order (15 distinct algos) and the CN triple (first 3 of 6 distinct variants)
 * are derived from header bytes [4..36) and are therefore CONSTANT for an entire
 * job (the nonce at byte 76 does not affect the order) -- exactly like x16r.
 *
 * This first milestone provides the CPU reference + self-test; the GPU scanhash
 * pipeline is added on top of the existing crypto/cryptonight CUDA core next.
 *
 * Reference: cpuminer-opt algo/gr (WyvernTKC cpuminer-gr).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

extern "C" {
#include <sph/sph_blake.h>
#include <sph/sph_bmw.h>
#include <sph/sph_groestl.h>
#include <sph/sph_skein.h>
#include <sph/sph_jh.h>
#include <sph/sph_keccak.h>

#include <sph/sph_luffa.h>
#include <sph/sph_cubehash.h>
#include <sph/sph_shavite.h>
#include <sph/sph_simd.h>
#include <sph/sph_echo.h>

#include <sph/sph_hamsi.h>
#include <sph/sph_fugue.h>
#include <sph/sph_shabal.h>
#include <sph/sph_whirlpool.h>
}

#include "miner.h"
#include "cuda_helper.h"
#include "x16/cuda_x16.h"

// CryptoNight-v1 variant wrappers (crypto/cryptonight-cpu.cpp) for CPU verify.
extern "C" void cryptonight_gr_dark      (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_darklite  (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_fast      (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_lite      (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_turtle    (void* output, const void* input, size_t len);
extern "C" void cryptonight_gr_turtlelite(void* output, const void* input, size_t len);

// GhostRider CryptoNight-v1 GPU path (crypto/cryptonight-core.cu, -extra.cu).
extern "C" void cryptonight_core_cuda_gr(int thr_id, int blocks, int threads, int variant, uint32_t stride64,
	uint64_t *d_long_state, uint64_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b,
	uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, uint64_t *d_ctx_tweak);
extern "C" void cryptonight_extra_cpu_prepare_gr(int thr_id, uint32_t threads, uint64_t *d_hash,
	uint64_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b,
	uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, uint64_t *d_ctx_tweak);
extern "C" void cryptonight_extra_cpu_final_gr(int thr_id, uint32_t threads, uint64_t *d_ctx_state,
	uint64_t *d_hash, int zero_high);

// Per-thread device buffers.
static uint32_t *d_hash[MAX_GPUS]      = { 0 };
static uint64_t *d_long_state[MAX_GPUS] = { 0 };
static uint64_t *d_ctx_state[MAX_GPUS] = { 0 };
static uint32_t *d_ctx_a[MAX_GPUS]     = { 0 };
static uint32_t *d_ctx_b[MAX_GPUS]     = { 0 };
static uint32_t *d_ctx_key1[MAX_GPUS]  = { 0 };
static uint32_t *d_ctx_key2[MAX_GPUS]  = { 0 };
static uint64_t *d_ctx_tweak[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { 0 };

enum Algo {
	BLAKE = 0, BMW, GROESTL, JH, KECCAK, SKEIN, LUFFA, CUBEHASH,
	SHAVITE, SIMD, ECHO, HAMSI, FUGUE, SHABAL, WHIRLPOOL,
	HASH_FUNC_COUNT
};

enum CNAlgo {
	CNDark = 0, CNDarklite, CNFast, CNLite, CNTurtle, CNTurtlelite,
	CN_HASH_FUNC_COUNT
};

// ----------------------------------------------------------------------------
// Order derivation: deduplicated nibble walk over header bytes [4..36)
// (x16s-style). getAlgoString reads size/2 bytes; for GhostRider size = 64 so
// 32 bytes [4..36) are consumed, low nibble then high nibble of each, % count,
// keeping the first occurrence of each distinct algo until `count` are chosen.
// ----------------------------------------------------------------------------
static void selectAlgo(unsigned char nibble, bool* selectedAlgos,
                       uint8_t* selectedIndex, int algoCount, int* currentCount)
{
	uint8_t algoDigit = (nibble & 0x0F) % algoCount;
	if (!selectedAlgos[algoDigit]) {
		selectedAlgos[algoDigit] = true;
		selectedIndex[currentCount[0]] = algoDigit;
		currentCount[0] += 1;
	}
	algoDigit = (nibble >> 4) % algoCount;
	if (!selectedAlgos[algoDigit]) {
		selectedAlgos[algoDigit] = true;
		selectedIndex[currentCount[0]] = algoDigit;
		currentCount[0] += 1;
	}
}

static void getAlgoString(const void* mem, unsigned int size,
                          uint8_t* selectedAlgoOutput, int algoCount)
{
	unsigned char* p = (unsigned char*)mem;
	unsigned int len = size / 2;
	bool selectedAlgo[HASH_FUNC_COUNT] = { false };
	int selectedCount = 0;

	for (unsigned int i = 0; i < len; i++) {
		selectAlgo(p[i], selectedAlgo, selectedAlgoOutput, algoCount, &selectedCount);
		if (selectedCount == algoCount) break;
	}
	if (selectedCount < algoCount)
		for (uint8_t i = 0; i < algoCount; i++)
			if (!selectedAlgo[i])
				selectedAlgoOutput[selectedCount++] = i;
}

// ----------------------------------------------------------------------------
// Core / CN dispatch. doCoreAlgo writes 64 bytes (512-bit), doCNAlgo writes 32.
// ----------------------------------------------------------------------------
static void doCoreAlgo(int algo, const void* in, void* out, size_t size)
{
	switch (algo) {
	case BLAKE: {
		sph_blake512_context ctx; sph_blake512_init(&ctx);
		sph_blake512(&ctx, in, size); sph_blake512_close(&ctx, out); break;
	}
	case BMW: {
		sph_bmw512_context ctx; sph_bmw512_init(&ctx);
		sph_bmw512(&ctx, in, size); sph_bmw512_close(&ctx, out); break;
	}
	case GROESTL: {
		sph_groestl512_context ctx; sph_groestl512_init(&ctx);
		sph_groestl512(&ctx, in, size); sph_groestl512_close(&ctx, out); break;
	}
	case JH: {
		sph_jh512_context ctx; sph_jh512_init(&ctx);
		sph_jh512(&ctx, in, size); sph_jh512_close(&ctx, out); break;
	}
	case KECCAK: {
		sph_keccak512_context ctx; sph_keccak512_init(&ctx);
		sph_keccak512(&ctx, in, size); sph_keccak512_close(&ctx, out); break;
	}
	case SKEIN: {
		sph_skein512_context ctx; sph_skein512_init(&ctx);
		sph_skein512(&ctx, in, size); sph_skein512_close(&ctx, out); break;
	}
	case LUFFA: {
		sph_luffa512_context ctx; sph_luffa512_init(&ctx);
		sph_luffa512(&ctx, in, size); sph_luffa512_close(&ctx, out); break;
	}
	case CUBEHASH: {
		sph_cubehash512_context ctx; sph_cubehash512_init(&ctx);
		sph_cubehash512(&ctx, in, size); sph_cubehash512_close(&ctx, out); break;
	}
	case SHAVITE: {
		sph_shavite512_context ctx; sph_shavite512_init(&ctx);
		sph_shavite512(&ctx, in, size); sph_shavite512_close(&ctx, out); break;
	}
	case SIMD: {
		sph_simd512_context ctx; sph_simd512_init(&ctx);
		sph_simd512(&ctx, in, size); sph_simd512_close(&ctx, out); break;
	}
	case ECHO: {
		sph_echo512_context ctx; sph_echo512_init(&ctx);
		sph_echo512(&ctx, in, size); sph_echo512_close(&ctx, out); break;
	}
	case HAMSI: {
		sph_hamsi512_context ctx; sph_hamsi512_init(&ctx);
		sph_hamsi512(&ctx, in, size); sph_hamsi512_close(&ctx, out); break;
	}
	case FUGUE: {
		sph_fugue512_context ctx; sph_fugue512_init(&ctx);
		sph_fugue512(&ctx, in, size); sph_fugue512_close(&ctx, out); break;
	}
	case SHABAL: {
		sph_shabal512_context ctx; sph_shabal512_init(&ctx);
		sph_shabal512(&ctx, in, size); sph_shabal512_close(&ctx, out); break;
	}
	case WHIRLPOOL: {
		sph_whirlpool_context ctx; sph_whirlpool_init(&ctx);
		sph_whirlpool(&ctx, in, size); sph_whirlpool_close(&ctx, out); break;
	}
	}
}

static void doCNAlgo(int cnAlgo, const void* in, void* out, size_t size)
{
	switch (cnAlgo) {
	case CNDark:       cryptonight_gr_dark(out, in, size); break;
	case CNDarklite:   cryptonight_gr_darklite(out, in, size); break;
	case CNFast:       cryptonight_gr_fast(out, in, size); break;
	case CNLite:       cryptonight_gr_lite(out, in, size); break;
	case CNTurtle:     cryptonight_gr_turtle(out, in, size); break;
	case CNTurtlelite: cryptonight_gr_turtlelite(out, in, size); break;
	}
}

// ----------------------------------------------------------------------------
// Full GhostRider hash. `input` is the 80-byte header in the same big-endian
// (per-32-bit-word byteswapped) frame the GPU/CPU scanhash prepares.
// ----------------------------------------------------------------------------
extern "C" void ghostrider_hash(void* output, const void* input)
{
	uint8_t coreOrder[HASH_FUNC_COUNT];
	uint8_t cnOrder[CN_HASH_FUNC_COUNT];
	uint8_t hash_1[64] = { 0 };
	uint8_t hash_2[64] = { 0 };

	getAlgoString((const uint8_t*)input + 4, 64, coreOrder, 15);
	getAlgoString((const uint8_t*)input + 4, 64, cnOrder, 6);

	// Group 1: first core round consumes the full 80-byte header.
	doCoreAlgo(coreOrder[0], input,  hash_1, 80);
	doCoreAlgo(coreOrder[1], hash_1, hash_2, 64);
	doCoreAlgo(coreOrder[2], hash_2, hash_1, 64);
	doCoreAlgo(coreOrder[3], hash_1, hash_2, 64);
	doCoreAlgo(coreOrder[4], hash_2, hash_1, 64);
	doCNAlgo(cnOrder[0], hash_1, hash_2, 64);
	memset(hash_2 + 32, 0, 32);

	// Group 2
	doCoreAlgo(coreOrder[5], hash_2, hash_1, 64);
	doCoreAlgo(coreOrder[6], hash_1, hash_2, 64);
	doCoreAlgo(coreOrder[7], hash_2, hash_1, 64);
	doCoreAlgo(coreOrder[8], hash_1, hash_2, 64);
	doCoreAlgo(coreOrder[9], hash_2, hash_1, 64);
	doCNAlgo(cnOrder[1], hash_1, hash_2, 64);
	memset(hash_2 + 32, 0, 32);

	// Group 3
	doCoreAlgo(coreOrder[10], hash_2, hash_1, 64);
	doCoreAlgo(coreOrder[11], hash_1, hash_2, 64);
	doCoreAlgo(coreOrder[12], hash_2, hash_1, 64);
	doCoreAlgo(coreOrder[13], hash_1, hash_2, 64);
	doCoreAlgo(coreOrder[14], hash_2, hash_1, 64);
	doCNAlgo(cnOrder[2], hash_1, hash_2, 64);

	memcpy(output, hash_2, 32);
}

// ----------------------------------------------------------------------------
// Known-answer self-test (cpuminer-opt gr vector).
// ----------------------------------------------------------------------------
static const uint8_t gr_test_input[80] = {
	0x70,0x00,0x00,0x00,0x5d,0x38,0x5b,0xa1,0x14,0xd0,0x79,0x97,0x0b,0x29,0xa9,0x41,
	0x8f,0xd0,0x54,0x9e,0x7d,0x68,0xa9,0x5c,0x7f,0x16,0x86,0x21,0xa3,0x14,0x20,0x10,
	0x00,0x00,0x00,0x00,0x57,0x85,0x86,0xd1,0x49,0xfd,0x07,0xb2,0x2f,0x3a,0x8a,0x34,
	0x7c,0x51,0x6d,0xe7,0x05,0x2f,0x03,0x4d,0x2b,0x76,0xff,0x68,0xe0,0xd6,0xec,0xff,
	0x9b,0x77,0xa4,0x54,0x89,0xe3,0xfd,0x51,0x17,0x32,0x01,0x1d,0xf0,0x73,0x10,0x00
};

static const uint8_t gr_test_expected[32] = {
	0x57,0x28,0x99,0x3c,0x46,0xe9,0x78,0x21,0x1b,0x52,0x84,0xc4,0x6d,0xc2,0x89,0x3f,
	0x51,0x1b,0x28,0x79,0x4a,0x25,0x14,0x98,0x67,0xec,0x8c,0x33,0xa5,0xef,0xb5,0x69
};

extern "C" bool ghostrider_self_test(void)
{
	uint8_t hash[32];
	ghostrider_hash(hash, gr_test_input);
	return memcmp(hash, gr_test_expected, 32) == 0;
}

// ----------------------------------------------------------------------------
// GPU core-round dispatch. The 15 core algos reuse the existing x16 CUDA
// kernels (GhostRider's algo indices 0..14 match x16r's exactly). The first
// round consumes the 80-byte header; the rest hash the 64-byte d_hash in place.
// ----------------------------------------------------------------------------
static void gr_core_setBlock_80(int algo, int thr_id, uint32_t *endiandata, uint32_t *pdata)
{
	switch (algo) {
	case BLAKE:     quark_blake512_cpu_setBlock_80(thr_id, endiandata); break;
	case BMW:       quark_bmw512_cpu_setBlock_80(endiandata); break;
	case GROESTL:   groestl512_setBlock_80(thr_id, endiandata); break;
	case JH:        jh512_setBlock_80(thr_id, endiandata); break;
	case KECCAK:    keccak512_setBlock_80(thr_id, endiandata); break;
	case SKEIN:     skein512_cpu_setBlock_80((void*)endiandata); break;
	case LUFFA:     qubit_luffa512_cpu_setBlock_80((void*)endiandata); break;
	case CUBEHASH:  cubehash512_setBlock_80(thr_id, endiandata); break;
	case SHAVITE:   x16_shavite512_setBlock_80((void*)endiandata); break;
	case SIMD:      x16_simd512_setBlock_80((void*)endiandata); break;
	case ECHO:      x16_echo512_setBlock_80((void*)endiandata); break;
	case HAMSI:     x16_hamsi512_setBlock_80((void*)endiandata); break;
	case FUGUE:     x16_fugue512_setBlock_80((void*)pdata); break; // fugue byteswaps internally
	case SHABAL:    x16_shabal512_setBlock_80((void*)endiandata); break;
	case WHIRLPOOL: x16_whirlpool512_setBlock_80((void*)endiandata); break;
	}
}

static void gr_core_hash_80(int algo, int thr_id, uint32_t throughput, uint32_t nonce, uint32_t *d_h)
{
	switch (algo) {
	case BLAKE:     quark_blake512_cpu_hash_80(thr_id, throughput, nonce, d_h); break;
	case BMW:       quark_bmw512_cpu_hash_80(thr_id, throughput, nonce, d_h, 0); break;
	case GROESTL:   groestl512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case JH:        jh512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case KECCAK:    keccak512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case SKEIN:     skein512_cpu_hash_80(thr_id, throughput, nonce, d_h, 1); break;
	case LUFFA:     qubit_luffa512_cpu_hash_80(thr_id, throughput, nonce, d_h, 0); break;
	case CUBEHASH:  cubehash512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case SHAVITE:   x16_shavite512_cpu_hash_80(thr_id, throughput, nonce, d_h, 0); break;
	case SIMD:      x16_simd512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case ECHO:      x16_echo512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case HAMSI:     x16_hamsi512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case FUGUE:     x16_fugue512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case SHABAL:    x16_shabal512_cuda_hash_80(thr_id, throughput, nonce, d_h); break;
	case WHIRLPOOL: x16_whirlpool512_hash_80(thr_id, throughput, nonce, d_h); break;
	}
}

static void gr_core_hash_64(int algo, int thr_id, uint32_t throughput, uint32_t nonce, uint32_t *d_h, int order)
{
	switch (algo) {
	case BLAKE:     quark_blake512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case BMW:       quark_bmw512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case GROESTL:   quark_groestl512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case JH:        quark_jh512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case KECCAK:    quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_h); break;
	case SKEIN:     quark_skein512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case LUFFA:     x11_luffa512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case CUBEHASH:  x11_cubehash512_cpu_hash_64(thr_id, throughput, d_h); break;
	case SHAVITE:   x11_shavite512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case SIMD:      x11_simd512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case ECHO:      x16_echo512_cpu_hash_64(thr_id, throughput, d_h); break;
	case HAMSI:     x13_hamsi512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case FUGUE:     x13_fugue512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case SHABAL:    x14_shabal512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	case WHIRLPOOL: x15_whirlpool_cpu_hash_64(thr_id, throughput, nonce, NULL, d_h, order); break;
	}
}

// Per-variant scratchpad sizes (bytes): dark, darklite, fast, lite, turtle, turtlelite.
static const uint32_t gr_cn_mem[CN_HASH_FUNC_COUNT] = { 524288u, 524288u, 2097152u, 1048576u, 262144u, 262144u };

// Scratchpad budget (bytes) and the thread count the per-thread buffers were
// sized for, established once at init.
static size_t   gr_scratch_bytes[MAX_GPUS]  = { 0 };
static uint32_t gr_max_throughput[MAX_GPUS] = { 0 };

// One CryptoNight-v1 round: 64-byte d_hash -> 32-byte d_hash (+ zero high 32).
// stride64 = the job's per-thread slot (largest CN variant) in uint64 words.
static void gr_cn_round(int thr_id, int blocks, int threads, int variant, uint32_t stride64, uint32_t *d_h, int zero_high)
{
	const uint32_t throughput = (uint32_t)(blocks * threads);
	cryptonight_extra_cpu_prepare_gr(thr_id, throughput, (uint64_t*)d_h,
		d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id],
		d_ctx_key1[thr_id], d_ctx_key2[thr_id], d_ctx_tweak[thr_id]);
	cryptonight_core_cuda_gr(thr_id, blocks, threads, variant, stride64,
		d_long_state[thr_id], d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id],
		d_ctx_key1[thr_id], d_ctx_key2[thr_id], d_ctx_tweak[thr_id]);
	cryptonight_extra_cpu_final_gr(thr_id, throughput, d_ctx_state[thr_id], (uint64_t*)d_h, zero_high);
}

extern "C" int scanhash_ghostrider(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	uint32_t* pdata = work->data;
	uint32_t* ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	uint32_t _ALIGN(64) endiandata[20];

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		if (ghostrider_self_test())
			gpulog(LOG_INFO, thr_id, "ghostrider: CPU self-test OK");
		else
			gpulog(LOG_ERR, thr_id, "ghostrider: CPU SELF-TEST FAILED");

		// Fixed throughput sized from free VRAM at the worst-case 2 MiB/thread, so
		// the kernels are init and run at the SAME thread count (init==run, like
		// the smallest CN variant (256 KiB). Per job we pick stride = the job's
		// largest CN variant and threads = budget/stride, so light jobs (turtle/
		// dark) run far more threads than heavy (fast/lite) ones.
		size_t freeMem = 0, totalMem = 0;
		cudaMemGetInfo(&freeMem, &totalMem);
		size_t reserve = 768ULL << 20; // headroom for ctx/core/check buffers + driver
		size_t budget = (freeMem > reserve) ? (freeMem - reserve) : (freeMem / 2);
		budget = (budget / 2097152) * 2097152; // whole 2 MiB slots
		gr_scratch_bytes[thr_id] = budget;

		uint32_t max_throughput = (uint32_t) min((size_t)(budget / 262144u), (size_t)(1U << 18));
		if (gpus_intensity[thr_id] > 0 && gpus_intensity[thr_id] < max_throughput)
			max_throughput = gpus_intensity[thr_id]; // -i N caps thread count
		max_throughput = (max_throughput / 128) * 128;
		if (max_throughput < 128) max_throughput = 128;
		gr_max_throughput[thr_id] = max_throughput;

		gpulog(LOG_INFO, thr_id, "ghostrider: %.0f MiB scratchpad, up to %u threads",
			(double)budget / (1024*1024), max_throughput);

		quark_blake512_cpu_init(thr_id, max_throughput);
		quark_bmw512_cpu_init(thr_id, max_throughput);
		quark_groestl512_cpu_init(thr_id, max_throughput);
		quark_skein512_cpu_init(thr_id, max_throughput);
		quark_jh512_cpu_init(thr_id, max_throughput);
		quark_keccak512_cpu_init(thr_id, max_throughput);
		qubit_luffa512_cpu_init(thr_id, max_throughput);
		x11_luffa512_cpu_init(thr_id, max_throughput);
		x11_shavite512_cpu_init(thr_id, max_throughput);
		x11_simd512_cpu_init(thr_id, max_throughput);
		x11_echo512_cpu_init(thr_id, max_throughput);
		x13_hamsi512_cpu_init(thr_id, max_throughput);
		x13_fugue512_cpu_init(thr_id, max_throughput);
		x14_shabal512_cpu_init(thr_id, max_throughput);
		x15_whirlpool_cpu_init(thr_id, max_throughput, 0);
		x16_echo512_cuda_init(thr_id, max_throughput);     // needed by x16_echo512_cuda_hash_80
		x16_fugue512_cpu_init(thr_id, max_throughput);     // needed by x16_fugue512_cuda_hash_80
		x16_whirlpool512_init(thr_id, max_throughput);     // needed by x16_whirlpool512_hash_80

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t)64 * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_long_state[thr_id], budget), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_state[thr_id], (size_t)26 * sizeof(uint64_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_a[thr_id], (size_t)4 * sizeof(uint32_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_b[thr_id], (size_t)4 * sizeof(uint32_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_key1[thr_id], (size_t)40 * sizeof(uint32_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_key2[thr_id], (size_t)40 * sizeof(uint32_t) * max_throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_ctx_tweak[thr_id], (size_t)sizeof(uint64_t) * max_throughput), -1);

		cuda_check_cpu_init(thr_id, max_throughput);

		gpulog(LOG_INFO, thr_id, "ghostrider: running startup self-test (kernel + pipeline), ~10s...");

		// Guard 1 (races): every core kernel must be self-consistent at full
		// throughput. Variation across repeats => an intra-kernel race (e.g. a
		// shared AES table read before it is cooperatively filled).
		{
			uint8_t tv[80];
			for (int i = 0; i < 80; i++) tv[i] = (uint8_t)(i * 13 + 5);
			const uint32_t tp = max_throughput;
			int races = 0;
			for (int a = 0; a < HASH_FUNC_COUNT; a++) {
				uint8_t first[64], g[64]; int v64 = 0, v80 = 0;
				for (int rep = 0; rep < 8; rep++) {
					cudaMemcpy(d_hash[thr_id], tv, 64, cudaMemcpyHostToDevice);
					gr_core_hash_64(a, thr_id, tp, 0, d_hash[thr_id], 0);
					cudaMemcpy(g, d_hash[thr_id], 64, cudaMemcpyDeviceToHost);
					if (rep == 0) memcpy(first, g, 64); else if (memcmp(g, first, 64)) v64++;
				}
				gr_core_setBlock_80(a, thr_id, (uint32_t*)tv, (uint32_t*)tv);
				for (int rep = 0; rep < 8; rep++) {
					gr_core_hash_80(a, thr_id, tp, 0, d_hash[thr_id]);
					cudaMemcpy(g, d_hash[thr_id], 64, cudaMemcpyDeviceToHost);
					if (rep == 0) memcpy(first, g, 64); else if (memcmp(g, first, 64)) v80++;
				}
				if (v64 || v80) { races++; gpulog(LOG_ERR, thr_id, "ghostrider: RACE in algo %d (hash64=%d hash80=%d)", a, v64, v80); }
			}
			if (!races) gpulog(LOG_INFO, thr_id, "ghostrider: kernels race-free");
		}

		// Guard 2 (chain logic): verify the full GPU pipeline against the CPU
		// reference across several header orders (order derivation, CN interleaving,
		// first-round init). Run at low throughput so startup stays fast.
		{
			uint32_t lcg = 0x12345678u;
			int diffs = 0;
			const int NTEST = 6;
			for (int t = 0; t < NTEST; t++) {
				uint32_t pd[20], edata[20];
				for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
				for (int k = 0; k < 20; k++) be32enc(&edata[k], pd[k]); // mirror scanhash

				uint8_t cOrd[HASH_FUNC_COUNT], nOrd[CN_HASH_FUNC_COUNT];
				getAlgoString(&edata[1], 64, cOrd, 15);
				getAlgoString(&edata[1], 64, nOrd, 6);
				uint32_t vmx = 0;
				for (int g = 0; g < 3; g++) vmx = max(vmx, gr_cn_mem[nOrd[g]]);
				uint32_t s64 = vmx >> 3;
				// This guard validates chain logic (order derivation, CN interleaving),
				// not kernel races (covered separately), so run at a small fixed thread
				// count to keep startup fast regardless of the job's variant mix.
				uint32_t tp = 512;
				uint32_t bl = tp / 128;

				uint32_t *dh = d_hash[thr_id];
				const int seq[18]   = { 1,0,0,0,0, 2, 0,0,0,0,0, 2, 0,0,0,0,0, 3 };
				const int cnstep[18] = { 0,0,0,0,0, 0, 0,0,0,0,0, 1, 0,0,0,0,0, 2 };
				uint8_t cbuf[64] = {0}, tmp[64], gbuf[64];
				be32enc(&edata[19], pd[19]);
				gr_core_setBlock_80(cOrd[0], thr_id, edata, pd);
				int od = 0, ci = 0;
				bool orderDiff = false;
				for (int s = 0; s < 18; s++) {
					int algo;
					if (seq[s] == 1) { algo = cOrd[ci++]; gr_core_hash_80(algo, thr_id, tp, pd[19], dh);
						doCoreAlgo(algo, edata, cbuf, 80); }
					else if (seq[s] == 0) { algo = cOrd[ci++]; gr_core_hash_64(algo, thr_id, tp, pd[19], dh, od++);
						doCoreAlgo(algo, cbuf, tmp, 64); memcpy(cbuf, tmp, 64); }
					else { algo = nOrd[cnstep[s]]; gr_cn_round(thr_id, bl, 128, algo, s64, dh, seq[s] == 2 ? 1 : 0);
						doCNAlgo(algo, cbuf, tmp, 64); memcpy(cbuf, tmp, 32); if (seq[s] == 2) memset(cbuf + 32, 0, 32); }
					int cl = (seq[s] == 3) ? 32 : 64;
					cudaMemcpy(gbuf, dh, cl, cudaMemcpyDeviceToHost);
					if (memcmp(gbuf, cbuf, cl) != 0) {
						gpulog(LOG_ERR, thr_id, "gr DIFF t=%d tp=%u stage=%d algo=%d core=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d cn=%d,%d,%d",
							t, tp, s, algo, cOrd[0],cOrd[1],cOrd[2],cOrd[3],cOrd[4],cOrd[5],cOrd[6],cOrd[7],cOrd[8],cOrd[9],
							cOrd[10],cOrd[11],cOrd[12],cOrd[13],cOrd[14], nOrd[0],nOrd[1],nOrd[2]);
						orderDiff = true; break;
					}
				}
				if (orderDiff) { diffs++; if (diffs >= 8) break; }
			}
			if (diffs == 0) gpulog(LOG_INFO, thr_id, "ghostrider: GPU==CPU on %d random orders", NTEST);
			else gpulog(LOG_ERR, thr_id, "ghostrider: %d/%d orders DIFF", diffs, NTEST);
		}

		gpulog(LOG_INFO, thr_id, "ghostrider: self-test complete, starting mining");

		init[thr_id] = true;
	}

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	// Order is constant across the nonce batch (depends only on header [4..36)).
	uint8_t coreOrder[HASH_FUNC_COUNT];
	uint8_t cnOrder[CN_HASH_FUNC_COUNT];
	getAlgoString(&endiandata[1], 64, coreOrder, 15);
	getAlgoString(&endiandata[1], 64, cnOrder, 6);

	// Pack threads for this job: per-thread slot = the job's largest CN variant.
	uint32_t vmax = 0;
	for (int g = 0; g < 3; g++)
		vmax = max(vmax, gr_cn_mem[cnOrder[g]]);
	const uint32_t stride64 = vmax >> 3;
	const uint32_t threads = 128;
	uint32_t throughput = (uint32_t) min((size_t)(gr_scratch_bytes[thr_id] / vmax), (size_t)gr_max_throughput[thr_id]);
	throughput = (throughput / threads) * threads;
	if (throughput < threads) throughput = threads;
	const uint32_t blocks = throughput / threads;

	// Bail out before launching the 18-kernel pipeline if the job changed or the
	// miner is shutting down (avoids wasted work and a teardown-race CUDA error).
	if (work_restart[thr_id].restart) {
		*hashes_done = 0;
		return 0;
	}

	gr_core_setBlock_80(coreOrder[0], thr_id, endiandata, pdata);
	cuda_check_cpu_setTarget(ptarget);

	// One batch per call: the GPU pipeline dominates, and per-call setup (order
	// derive + setBlock + setTarget) is microseconds, so returning each batch lets
	// the hashrate meter update without measurable overhead.
	uint32_t *dh = d_hash[thr_id];
	int order = 0;

	// Group 1
	gr_core_hash_80(coreOrder[0], thr_id, throughput, pdata[19], dh);
	gr_core_hash_64(coreOrder[1], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[2], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[3], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[4], thr_id, throughput, pdata[19], dh, order++);
	gr_cn_round(thr_id, blocks, threads, cnOrder[0], stride64, dh, 1);

	// Group 2
	gr_core_hash_64(coreOrder[5], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[6], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[7], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[8], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[9], thr_id, throughput, pdata[19], dh, order++);
	gr_cn_round(thr_id, blocks, threads, cnOrder[1], stride64, dh, 1);

	// Group 3
	gr_core_hash_64(coreOrder[10], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[11], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[12], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[13], thr_id, throughput, pdata[19], dh, order++);
	gr_core_hash_64(coreOrder[14], thr_id, throughput, pdata[19], dh, order++);
	gr_cn_round(thr_id, blocks, threads, cnOrder[2], stride64, dh, 0);

	// One-time GPU vs CPU correctness check on the batch's first nonce.
	static bool gr_gpu_checked = false;
	if (!gr_gpu_checked) {
		gr_gpu_checked = true;
		uint32_t _ALIGN(64) ghash[8], chash[8], ed[20];
		cudaMemcpy(ghash, dh, 32, cudaMemcpyDeviceToHost);
		memcpy(ed, endiandata, 80);
		be32enc(&ed[19], pdata[19]);
		ghostrider_hash(chash, ed);
		if (memcmp(ghash, chash, 32) == 0)
			gpulog(LOG_INFO, thr_id, "ghostrider: GPU matches CPU reference");
		else
			gpulog(LOG_ERR, thr_id, "ghostrider: GPU != CPU! gpu[0]=%08x cpu[0]=%08x", ghash[0], chash[0]);
	}

	work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], dh);
	if (work->nonces[0] != UINT32_MAX) {
		uint32_t _ALIGN(64) vhash[8];
		be32enc(&endiandata[19], work->nonces[0]);
		ghostrider_hash(vhash, endiandata);

		if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
			work->valid_nonces = 1;
			work_set_target_ratio(work, vhash);
			work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], dh, 1);
			if (work->nonces[1] != 0) {
				be32enc(&endiandata[19], work->nonces[1]);
				ghostrider_hash(vhash, endiandata);
				bn_set_target_ratio(work, vhash, 1);
				work->valid_nonces++;
				pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
			} else {
				pdata[19] = work->nonces[0] + 1;
			}
			*hashes_done = pdata[19] - first_nonce;
			return work->valid_nonces;
		} else {
			gpu_increment_reject(thr_id);
			if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
		}
	}

	pdata[19] += throughput;
	if (pdata[19] > max_nonce || pdata[19] < first_nonce)
		pdata[19] = max_nonce;
	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_ghostrider(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_long_state[thr_id]);
	cudaFree(d_ctx_state[thr_id]);
	cudaFree(d_ctx_a[thr_id]);
	cudaFree(d_ctx_b[thr_id]);
	cudaFree(d_ctx_key1[thr_id]);
	cudaFree(d_ctx_key2[thr_id]);
	cudaFree(d_ctx_tweak[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id);
	x15_whirlpool_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
