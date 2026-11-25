#include <stdio.h>
#include <memory.h>
#include <unistd.h>

extern "C" {
#include "sph/sph_groestl.h"
#include "sph/sph_keccak.h"
#include "sph/sph_jh.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_luffa.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_shabal.h"
#include "algos/lyra2/Lyra2.h"
}

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

static uint64_t* d_matrix[MAX_GPUS];
static uint32_t* d_hash[MAX_GPUS];
static uint64_t* d_hash_256[MAX_GPUS];

extern void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern void lyra2_cuda_hash_64(int thr_id, const uint32_t threads, uint64_t* d_hash_256, uint32_t* d_hash);

extern "C" void evohash(void *state, const void *input)
{
	sph_groestl512_context   ctx_groestl;
	sph_keccak512_context    ctx_keccak;
	sph_cubehash512_context  ctx_cubehash;
	sph_luffa512_context     ctx_luffa;
	sph_echo512_context      ctx_echo;
	sph_simd512_context      ctx_simd;
	sph_shavite512_context   ctx_shavite;
	sph_hamsi512_context     ctx_hamsi;
	sph_fugue512_context     ctx_fugue;
	sph_whirlpool_context    ctx_whirlpool;
	sph_skein512_context     ctx_skein;
	sph_shabal512_context    ctx_shabal;
	sph_bmw512_context       ctx_bmw;
	sph_jh512_context        ctx_jh;

	unsigned char hash[128] = { 0 };
	unsigned char hashA[64] = { 0 };
	unsigned char hashB[64] = { 0 };

    // CUBE512-80
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, input, 80);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // BMW512
    sph_bmw512_init(&ctx_bmw);
    sph_bmw512(&ctx_bmw, hash, 64);
    sph_bmw512_close(&ctx_bmw, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Hamsi512
    sph_hamsi512_init(&ctx_hamsi);
    sph_hamsi512(&ctx_hamsi, hashA, 64);
    sph_hamsi512_close(&ctx_hamsi, hash);

    // Fugue512
    sph_fugue512_init(&ctx_fugue);
    sph_fugue512(&ctx_fugue, hash, 64);
    sph_fugue512_close(&ctx_fugue, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // SIMD512
    sph_simd512_init(&ctx_simd);
    sph_simd512(&ctx_simd, hashA, 64);
    sph_simd512_close(&ctx_simd, hash);

    // Echo512
    sph_echo512_init(&ctx_echo);
    sph_echo512(&ctx_echo, hash, 64);
    sph_echo512_close(&ctx_echo, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // CubeHash512
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hashA, 64);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // Shavite512
    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, 64);
    sph_shavite512_close(&ctx_shavite, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Luffa512
    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hashA, 64);
    sph_luffa512_close(&ctx_luffa, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // CubeHash512
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hashA, 64);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // Shavite512
    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, 64);
    sph_shavite512_close(&ctx_shavite, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Luffa512
    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hashA, 64);
    sph_luffa512_close(&ctx_luffa, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Hamsi512
    sph_hamsi512_init(&ctx_hamsi);
    sph_hamsi512(&ctx_hamsi, hashA, 64);
    sph_hamsi512_close(&ctx_hamsi, hash);

    // Fugue512
    sph_fugue512_init(&ctx_fugue);
    sph_fugue512(&ctx_fugue, hash, 64);
    sph_fugue512_close(&ctx_fugue, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // SIMD512
    sph_simd512_init(&ctx_simd);
    sph_simd512(&ctx_simd, hashA, 64);
    sph_simd512_close(&ctx_simd, hash);

    // Echo512
    sph_echo512_init(&ctx_echo);
    sph_echo512(&ctx_echo, hash, 64);
    sph_echo512_close(&ctx_echo, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // CubeHash512
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hashA, 64);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // Shavite512
    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, 64);
    sph_shavite512_close(&ctx_shavite, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Luffa512
    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hashA, 64);
    sph_luffa512_close(&ctx_luffa, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // CubeHash512
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hashA, 64);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // Shavite512
    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, 64);
    sph_shavite512_close(&ctx_shavite, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Luffa512
    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hashA, 64);
    sph_luffa512_close(&ctx_luffa, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Hamsi512
    sph_hamsi512_init(&ctx_hamsi);
    sph_hamsi512(&ctx_hamsi, hashA, 64);
    sph_hamsi512_close(&ctx_hamsi, hash);

    // Fugue512
    sph_fugue512_init(&ctx_fugue);
    sph_fugue512(&ctx_fugue, hash, 64);
    sph_fugue512_close(&ctx_fugue, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // SIMD512
    sph_simd512_init(&ctx_simd);
    sph_simd512(&ctx_simd, hashA, 64);
    sph_simd512_close(&ctx_simd, hash);

    // Echo512
    sph_echo512_init(&ctx_echo);
    sph_echo512(&ctx_echo, hash, 64);
    sph_echo512_close(&ctx_echo, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // CubeHash512
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hashA, 64);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // Shavite512
    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, 64);
    sph_shavite512_close(&ctx_shavite, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Luffa512
    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hashA, 64);
    sph_luffa512_close(&ctx_luffa, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // CubeHash512
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hashA, 64);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // Shavite512
    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, 64);
    sph_shavite512_close(&ctx_shavite, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Luffa512
    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hashA, 64);
    sph_luffa512_close(&ctx_luffa, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Hamsi512
    sph_hamsi512_init(&ctx_hamsi);
    sph_hamsi512(&ctx_hamsi, hashA, 64);
    sph_hamsi512_close(&ctx_hamsi, hash);

    // Fugue512
    sph_fugue512_init(&ctx_fugue);
    sph_fugue512(&ctx_fugue, hash, 64);
    sph_fugue512_close(&ctx_fugue, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // SIMD512
    sph_simd512_init(&ctx_simd);
    sph_simd512(&ctx_simd, hashA, 64);
    sph_simd512_close(&ctx_simd, hash);

    // Echo512
    sph_echo512_init(&ctx_echo);
    sph_echo512(&ctx_echo, hash, 64);
    sph_echo512_close(&ctx_echo, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // CubeHash512
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hashA, 64);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // Shavite512
    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, 64);
    sph_shavite512_close(&ctx_shavite, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Luffa512
    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hashA, 64);
    sph_luffa512_close(&ctx_luffa, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // CubeHash512
    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512(&ctx_cubehash, hashA, 64);
    sph_cubehash512_close(&ctx_cubehash, hash);

    // Shavite512
    sph_shavite512_init(&ctx_shavite);
    sph_shavite512(&ctx_shavite, hash, 64);
    sph_shavite512_close(&ctx_shavite, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Luffa512
    sph_luffa512_init(&ctx_luffa);
    sph_luffa512(&ctx_luffa, hashA, 64);
    sph_luffa512_close(&ctx_luffa, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Whirlpool
    sph_whirlpool_init(&ctx_whirlpool);
    sph_whirlpool(&ctx_whirlpool, hashA, 64);
    sph_whirlpool_close(&ctx_whirlpool, hash);

    // Shabal512
    sph_shabal512_init(&ctx_shabal);
    sph_shabal512(&ctx_shabal, hash, 64);
    sph_shabal512_close(&ctx_shabal, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // JH512
    sph_jh512_init(&ctx_jh);
    sph_jh512(&ctx_jh, hashA, 64);
    sph_jh512_close(&ctx_jh, hash);

    // Keccak512
    sph_keccak512_init(&ctx_keccak);
    sph_keccak512(&ctx_keccak, hash, 64);
    sph_keccak512_close(&ctx_keccak, hashB);

    // LYRA2
    LYRA2(&hashA[0], 32, &hashB[0], 32, &hashB[0], 32, 1, 8, 8);
    LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

    // Skein512
    sph_skein512_init(&ctx_skein);
    sph_skein512(&ctx_skein, hashA, 64);
    sph_skein512_close(&ctx_skein, hash);

    // Groestl512
    sph_groestl512_init(&ctx_groestl);
    sph_groestl512(&ctx_groestl, hash, 64);
    sph_groestl512_close(&ctx_groestl, hash);

    for (int i=0; i<32; i++)
        hash[i] ^= hash[i+32];

    memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_evohash(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	
	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << 18); 

	if (opt_benchmark)
		ptarget[7] = 0x00FF;
		
	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		
		cuda_get_arch(thr_id);

		size_t matrix_sz = device_sm[dev_id] > 500 ? sizeof(uint64_t) * 16 : sizeof(uint64_t) * 8 * 8 * 3 * 4;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput), 0);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash_256[thr_id], (size_t)32 * throughput), 0);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t)64 * throughput), 0);

		x11_echo512_cpu_init(thr_id, throughput);

		groestl512_cpu_init(thr_id, throughput);
		bmw512_cpu_init(thr_id, throughput);
		skein512_cpu_init(thr_id, throughput);
		keccak512_cpu_init(thr_id, throughput);
		jh512_cpu_init(thr_id, throughput);
		luffa512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		simd512_cpu_init(thr_id, throughput);
		hamsi512_cpu_init(thr_id, throughput);
		fugue512_cpu_init(thr_id, throughput);
		shabal512_cpu_init(thr_id, throughput);
		whirlpool512_cpu_init(thr_id, throughput, 0);
		lyra2_cpu_init(thr_id, throughput, d_matrix[thr_id]);
		
		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	cubehash512_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;
		cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash[thr_id]); order++;
		skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			/* check now with the CPU to confirm */
			be32enc(&endiandata[19], work->nonces[0]);
			evohash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					evohash(vhash, endiandata);
					work_check_second_nonce(thr_id, work, vhash);
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_evohash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);
	cudaFree(d_hash_256[thr_id]);
	
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
