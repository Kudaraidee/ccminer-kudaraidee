/**
 * SkyDoge algorithm (fixed-order 20-step chained hash)
 *
 * Chain (each round consumes the previous 64-byte output; the first consumes
 * the 80-byte header):
 *
 *   1 blake512   2 skein512   3 bmw512     4 groestl512  5 jh512
 *   6 luffa512   7 keccak512  8 simd512    9 echo512    10 cubehash512
 *  11 shavite512 12 hamsi512 13 fugue512  14 shabal512  15 whirlpool
 *  16 sha512    17 simd512   18 whirlpool 19 sha256     20 haval256-5
 *
 * (simd and whirlpool each run twice.) Finalize: sha256 writes 32 bytes; the
 * high 32 bytes of the 64-byte buffer are zeroed; haval256-5 hashes the full
 * 64 bytes; the first 32 bytes of its output are the result.
 *
 * Port of tpfuemp/yiimp-skydoge (and cpuminer-opt algo/x17/skydoge.c) to CUDA.
 * Reuses the x16/x17/x21 family kernels, which are bit-identical to the sph
 * reference (proven by the pool-confirmed x17 family).
 */

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
}

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

static uint32_t *d_hash[MAX_GPUS];

/* SkyDoge has two consecutive fusible runs (the rest of the chain is broken up
 * by the non-fusible groestl/simd/echo/shavite/fugue/whirlpool stages):
 *   steps 2-3  : skein -> bmw
 *   steps 5-7  : jh -> luffa -> keccak */

// sha256 over a 64-byte chained value, writing the 32-byte digest and zeroing
// the high 32 bytes of the slot in the same kernel (algos/stages/cuda_sha256_2.cu), so
// the final haval consumes the same (32 bytes digest || 32 bytes zero) buffer
// as the CPU reference -- without a separate zeroing launch.
extern void sha256_cpu_hash_64z(int thr_id, int threads, uint32_t *d_hash);

// SkyDoge CPU Hash (validation + self-test reference)
extern "C" void skydoge_hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	uint32_t _ALIGN(64) hashA[16];

	sph_blake512_context   ctx_blake;
	sph_skein512_context   ctx_skein;
	sph_bmw512_context     ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context      ctx_jh;
	sph_luffa512_context   ctx_luffa;
	sph_keccak512_context  ctx_keccak;
	sph_simd512_context    ctx_simd;
	sph_echo512_context    ctx_echo;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_hamsi512_context   ctx_hamsi;
	sph_fugue512_context   ctx_fugue;
	sph_shabal512_context  ctx_shabal;
	sph_whirlpool_context  ctx_whirlpool;
	sph_sha512_context     ctx_sha512;
	sph_sha256_context     ctx_sha256;
	sph_haval256_5_context ctx_haval;

	sph_blake512_init(&ctx_blake);                                  // 1
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, hash);

	sph_skein512_init(&ctx_skein);                                  // 2
	sph_skein512(&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, hash);

	sph_bmw512_init(&ctx_bmw);                                      // 3
	sph_bmw512(&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, hash);

	sph_groestl512_init(&ctx_groestl);                             // 4
	sph_groestl512(&ctx_groestl, (const void*) hash, 64);
	sph_groestl512_close(&ctx_groestl, hash);

	sph_jh512_init(&ctx_jh);                                       // 5
	sph_jh512(&ctx_jh, (const void*) hash, 64);
	sph_jh512_close(&ctx_jh, hash);

	sph_luffa512_init(&ctx_luffa);                                 // 6
	sph_luffa512(&ctx_luffa, (const void*) hash, 64);
	sph_luffa512_close(&ctx_luffa, hash);

	sph_keccak512_init(&ctx_keccak);                              // 7
	sph_keccak512(&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, hash);

	sph_simd512_init(&ctx_simd);                                  // 8
	sph_simd512(&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, hash);

	sph_echo512_init(&ctx_echo);                                  // 9
	sph_echo512(&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, hash);

	sph_cubehash512_init(&ctx_cubehash);                         // 10
	sph_cubehash512(&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, hash);

	sph_shavite512_init(&ctx_shavite);                           // 11
	sph_shavite512(&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, hash);

	sph_hamsi512_init(&ctx_hamsi);                               // 12
	sph_hamsi512(&ctx_hamsi, (const void*) hash, 64);
	sph_hamsi512_close(&ctx_hamsi, hash);

	sph_fugue512_init(&ctx_fugue);                               // 13
	sph_fugue512(&ctx_fugue, (const void*) hash, 64);
	sph_fugue512_close(&ctx_fugue, hash);

	sph_shabal512_init(&ctx_shabal);                             // 14
	sph_shabal512(&ctx_shabal, (const void*) hash, 64);
	sph_shabal512_close(&ctx_shabal, hash);

	sph_whirlpool_init(&ctx_whirlpool);                          // 15
	sph_whirlpool(&ctx_whirlpool, (const void*) hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	sph_sha512_init(&ctx_sha512);                                // 16
	sph_sha512(&ctx_sha512, (const void*) hash, 64);
	sph_sha512_close(&ctx_sha512, hash);

	sph_simd512_init(&ctx_simd);                                 // 17
	sph_simd512(&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, hash);

	sph_whirlpool_init(&ctx_whirlpool);                          // 18
	sph_whirlpool(&ctx_whirlpool, (const void*) hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	sph_sha256_init(&ctx_sha256);                                // 19
	sph_sha256(&ctx_sha256, (const void*) hash, 64);
	sph_sha256_close(&ctx_sha256, hashA);

	// Zero the high 32 bytes (uint32 indices 8..15) before the final haval.
	for (int i = 8; i < 16; i++)
		hashA[i] = 0;

	sph_haval256_5_init(&ctx_haval);                             // 20
	sph_haval256_5(&ctx_haval, (const void*) hashA, 64);
	sph_haval256_5_close(&ctx_haval, hash);

	memcpy(output, hash, 32);
}

// Consensus KAT from a real pool-accepted share (zpool.ca SkyDoge job 30b2,
// 2026-06-26). The input is the 80-byte be32enc'd header exactly as
// skydoge_hash receives it; the expected output is the 32-byte digest the pool
// accepted, so it is consensus-correct.
static const uint8_t skydoge_test_input[80] =
{
	0x00,0x00,0x00,0x20, 0x2e,0x4c,0xf8,0x2c, 0x5a,0xc6,0x4f,0xf3, 0xc0,0xcc,0x0d,0x6d,
	0x8c,0x0d,0x0e,0xb5, 0x45,0x32,0x1b,0x2c, 0x85,0x9f,0x8a,0x78, 0xa4,0xd3,0x0e,0x01,
	0x00,0x00,0x00,0x00, 0xd2,0xe7,0x08,0x8b, 0xbd,0xf7,0x3f,0x0d, 0x6b,0xfa,0x2c,0xf9,
	0x22,0x48,0x32,0x36, 0xfb,0x65,0x99,0x4c, 0x22,0x10,0x73,0xe9, 0x85,0x0d,0x36,0xaf,
	0x3f,0xdf,0x1f,0xfd, 0x5c,0xdf,0x3e,0x6a, 0x3d,0x20,0x02,0x1c, 0xf0,0x02,0x51,0xaf
};

static const uint8_t skydoge_test_expected[32] =
{
	0x72,0xd6,0x05,0x0e, 0x7d,0xfa,0x96,0x04, 0x80,0x1a,0x9d,0x73, 0xbc,0xd5,0x44,0xe0,
	0x7b,0xed,0xd5,0xd2, 0xc1,0x49,0xb4,0xd1, 0xf1,0x33,0xff,0xbc, 0x1a,0x00,0x00,0x00
};

static bool skydoge_self_test(void)
{
	uint8_t hash[32];
	skydoge_hash(hash, skydoge_test_input);
	return memcmp(hash, skydoge_test_expected, 32) == 0;
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_skydoge(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << 19); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		if (!skydoge_self_test())
			gpulog(LOG_WARNING, thr_id, "SkyDoge CPU self-test FAILED (consensus KAT mismatch)");

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id])
			echo512_cpu_init_compat(thr_id, throughput);

		blake512_cpu_init(thr_id, throughput);
		skein512_cpu_init(thr_id, throughput);
		bmw512_cpu_init(thr_id, throughput);
		groestl512_cpu_init(thr_id, throughput);
		jh512_cpu_init(thr_id, throughput);
		keccak512_cpu_init(thr_id, throughput);
		luffa512_cpu_init(thr_id, throughput);
		//cubehash512 needs no runtime init (self-test runs from the hash launcher)
		shavite512_cpu_init(thr_id, throughput);
		simd512_cpu_init(thr_id, throughput);
		hamsi512_cpu_init(thr_id, throughput);
		fugue512_cpu_init(thr_id, throughput);
		shabal512_cpu_init(thr_id, throughput);
		whirlpool512_cpu_init(thr_id, throughput, 0);
		sha512_cpu_init(thr_id, throughput);
		haval256_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		x_fused_device_selftest(thr_id);

		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Hash with CUDA
		blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;                        // 1
		/* fused run: skein -> bmw (steps 2-3, register-resident) */
		x_fused_fixed_cpu_hash_64(throughput, XF_SKEIN_BMW, d_hash[thr_id]); order += 2;                     // 2-3
		groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);                // 4
		/* fused run: jh -> luffa -> keccak (steps 5-7, register-resident) */
		x_fused_fixed_cpu_hash_64(throughput, XF_JH_LUFFA_KECCAK, d_hash[thr_id]); order += 3;               // 5-7
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);                   // 8
		if (use_compat_kernels[thr_id])                                                                      // 9
			echo512_cpu_hash_64_compat(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		else {
			echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		}
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;                                // 10
		shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;                                 // 11
		hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);                  // 12
		fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);                  // 13
		shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);                 // 14
		whirlpool512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);              // 15
		sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;                          // 16
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);                   // 17
		whirlpool512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);              // 18
		sha256_cpu_hash_64z(thr_id, throughput, d_hash[thr_id]); order++;                                    // 19 (digest + zero high 32 bytes)
		haval256_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], 256); order++;                   // 20

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			skydoge_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					skydoge_hash(vhash, endiandata);
					work_check_second_nonce(thr_id, work, vhash);
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else {
				// failed candidate or top-word tie: count and log it, scan on from the next nonce
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_skydoge(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);

	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);
	simd512_cpu_free(thr_id);
	fugue512_cpu_free(thr_id);
	whirlpool512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
