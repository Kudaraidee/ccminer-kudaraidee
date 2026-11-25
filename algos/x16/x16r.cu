/**
 * X16R algorithm (X16 with Randomized chain order)
 *
 * tpruvot 2018 - GPL code
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

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
}

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

static uint32_t *d_hash[MAX_GPUS];

enum Algo {
	BLAKE = 0,
	BMW,
	GROESTL,
	JH,
	KECCAK,
	SKEIN,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HAMSI,
	FUGUE,
	SHABAL,
	WHIRLPOOL,
	SHA512,
	HASH_FUNC_COUNT
};

static const char* algo_strings[] = {
	"blake",
	"bmw512",
	"groestl",
	"jh512",
	"keccak",
	"skein",
	"luffa",
	"cube",
	"shavite",
	"simd",
	"echo",
	"hamsi",
	"fugue",
	"shabal",
	"whirlpool",
	"sha512",
	NULL
};

static __thread uint32_t s_ntime = UINT32_MAX;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

static __thread uint8_t fused_run[16] = { 0 }; /* fused run length starting at stage i, 0 = none */

static void getAlgoString(const uint32_t* prevblock, char *output)
{
	char *sptr = output;
	uint8_t* data = (uint8_t*)prevblock;

	for (uint8_t j = 0; j < HASH_FUNC_COUNT; j++) {
		uint8_t b = (15 - j) >> 1; // 16 ascii hex chars, reversed
		uint8_t algoDigit = (j & 1) ? data[b] & 0xF : data[b] >> 4;
		if (algoDigit >= 10)
			sprintf(sptr, "%c", 'A' + (algoDigit - 10));
		else
			sprintf(sptr, "%u", (uint32_t) algoDigit);
		sptr++;
	}
	*sptr = '\0';
}

// X16R CPU Hash (Validation)
extern "C" void x16r_hash(void *output, const void *input)
{
	unsigned char _ALIGN(64) hash[128];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;

	void *in = (void*) input;
	int size = 80;

	uint32_t *in32 = (uint32_t*) input;
	getAlgoString(&in32[1], hashOrder);

	for (int i = 0; i < 16; i++)
	{
		const char elem = hashOrder[i];
		const uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

		switch (algo) {
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, in, size);
			sph_blake512_close(&ctx_blake, hash);
			break;
		case BMW:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, in, size);
			sph_bmw512_close(&ctx_bmw, hash);
			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, in, size);
			sph_groestl512_close(&ctx_groestl, hash);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, in, size);
			sph_skein512_close(&ctx_skein, hash);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, in, size);
			sph_jh512_close(&ctx_jh, hash);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, in, size);
			sph_keccak512_close(&ctx_keccak, hash);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, in, size);
			sph_luffa512_close(&ctx_luffa, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, in, size);
			sph_cubehash512_close(&ctx_cubehash, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, in, size);
			sph_shavite512_close(&ctx_shavite, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, in, size);
			sph_simd512_close(&ctx_simd, hash);
			break;
		case ECHO:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, in, size);
			sph_echo512_close(&ctx_echo, hash);
			break;
		case HAMSI:
			sph_hamsi512_init(&ctx_hamsi);
			sph_hamsi512(&ctx_hamsi, in, size);
			sph_hamsi512_close(&ctx_hamsi, hash);
			break;
		case FUGUE:
			sph_fugue512_init(&ctx_fugue);
			sph_fugue512(&ctx_fugue, in, size);
			sph_fugue512_close(&ctx_fugue, hash);
			break;
		case SHABAL:
			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, in, size);
			sph_shabal512_close(&ctx_shabal, hash);
			break;
		case WHIRLPOOL:
			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, in, size);
			sph_whirlpool_close(&ctx_whirlpool, hash);
			break;
		case SHA512:
			sph_sha512_init(&ctx_sha512);
			sph_sha512(&ctx_sha512,(const void*) in, size);
			sph_sha512_close(&ctx_sha512,(void*) hash);
			break;
		}
		in = (void*) hash;
		size = 64;
	}
	memcpy(output, hash, 32);
}

void whirlpool_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;

	sph_whirlpool_init(&ctx);
	sph_whirlpool(&ctx, input, 64);

	memcpy(state, ctx.state, 64);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "x16r-"
#include "cuda/debug.cuh"

//static int algo80_tests[HASH_FUNC_COUNT] = { 0 };
//static int algo64_tests[HASH_FUNC_COUNT] = { 0 };
static int algo80_fails[HASH_FUNC_COUNT] = { 0 };

extern "C" int scanhash_x16r(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 19;
	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 20;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id])
			echo512_cpu_init_compat(thr_id, throughput);

		blake512_cpu_init(thr_id, throughput);
		bmw512_cpu_init(thr_id, throughput);
		groestl512_cpu_init(thr_id, throughput);
		skein512_cpu_init(thr_id, throughput);
		jh512_cpu_init(thr_id, throughput);
		keccak512_cpu_init(thr_id, throughput);
		qubit_luffa512_cpu_init(thr_id, throughput);
		luffa512_cpu_init(thr_id, throughput); // 64
		shavite512_cpu_init(thr_id, throughput);
		simd512_cpu_init(thr_id, throughput); // 64
		echo512_cpu_init(thr_id, throughput);
		hamsi512_cpu_init(thr_id, throughput);
		fugue512_cpu_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		shabal512_cpu_init(thr_id, throughput);
		whirlpool512_cpu_init(thr_id, throughput, 0);
		x16_whirlpool512_cpu_init(thr_id, throughput);
		sha512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	if (opt_benchmark) {
		((uint32_t*)ptarget)[7] = 0x003f;
		/* benchmark header runs all 16 stages exactly once (hashOrder
		 * "0123456789ABCDEF"): deterministic full-chain rate, comparable
		 * across runs; replaces the degenerate 0xAA echo/skein-only order */
		((uint8_t*)pdata)[4]  = 0x89;
		((uint8_t*)pdata)[5]  = 0xAB;
		((uint8_t*)pdata)[6]  = 0xCD;
		((uint8_t*)pdata)[7]  = 0xEF;
		((uint8_t*)pdata)[8]  = 0x01;
		((uint8_t*)pdata)[9]  = 0x23;
		((uint8_t*)pdata)[10] = 0x45;
		((uint8_t*)pdata)[11] = 0x67;
	}
	uint32_t _ALIGN(64) endiandata[20];

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	uint32_t ntime = swab32(pdata[17]);
	if (s_ntime != ntime) {
		getAlgoString(&endiandata[1], hashOrder);
		s_ntime = ntime;
		if (opt_debug && !thr_id) applog(LOG_DEBUG, "hash order %s (%08x)", hashOrder, ntime);

		/* find maximal runs of >= 2 fusible 64-byte stages; upload the id
		 * sequence once (kernels index it by (start,len)) */
		uint8_t ids[16];
		memset(fused_run, 0, sizeof(fused_run));
		for (int i = 1; i < 16; i++) {
			const char elem = hashOrder[i];
			ids[i] = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
		}
		for (int i = 1; i < 16; ) {
			int len = 0;
			while (i + len < 16 && x_fusible[ids[i + len]]) len++;
			if (len >= 2) fused_run[i] = (uint8_t) len;
			i += (len > 0) ? len : 1;
		}
		/* fused-kernel unit test (clobbers the order constant, so run it
		 * before the real upload) */
		x_fused_device_selftest(thr_id);

		x_fused_setOrder(&ids[1], 15);
		if (opt_debug) applog(LOG_DEBUG, "fused runs: %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u",
			fused_run[0],fused_run[1],fused_run[2],fused_run[3],fused_run[4],fused_run[5],fused_run[6],fused_run[7],
			fused_run[8],fused_run[9],fused_run[10],fused_run[11],fused_run[12],fused_run[13],fused_run[14],fused_run[15]);
	}

	cuda_check_cpu_setTarget(ptarget);

	char elem = hashOrder[0];
	const uint8_t algo80 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

	switch (algo80) {
		case BLAKE:
			blake512_cpu_setBlock_80(thr_id, endiandata);
			break;
		case BMW:
			bmw512_cpu_setBlock_80(endiandata);
			break;
		case GROESTL:
			groestl512_setBlock_80(thr_id, endiandata);
			break;
		case JH:
			jh512_setBlock_80(thr_id, endiandata);
			break;
		case KECCAK:
			keccak512_setBlock_80(thr_id, endiandata);
			break;
		case SKEIN:
			skein512_cpu_setBlock_80((void*)endiandata);
			break;
		case LUFFA:
			qubit_luffa512_cpu_setBlock_80((void*)endiandata);
			break;
		case CUBEHASH:
			cubehash512_setBlock_80(thr_id, endiandata);
			break;
		case SHAVITE:
			shavite512_setBlock_80((void*)endiandata);
			break;
		case SIMD:
			simd512_setBlock_80((void*)endiandata);
			break;
		case ECHO:
			echo512_setBlock_80((void*)endiandata);
			break;
		case HAMSI:
			x16_hamsi512_setBlock_80((void*)endiandata);
			break;
		case FUGUE:
			fugue512_setBlock_80((void*)pdata);
			break;
		case SHABAL:
			shabal512_setBlock_80((void*)endiandata);
			break;
		case WHIRLPOOL:
			x16_whirlpool512_setBlock_80((void*)endiandata);
			break;
		case SHA512:
			x16_sha512_setBlock_80(endiandata);
			break;
		default: {
			return -1;
		}
	}

	do {
		int order = 0;

		// Hash with CUDA

		switch (algo80) {
			case BLAKE:
				blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("blake80:");
				break;
			case BMW:
				bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				TRACE("bmw80  :");
				break;
			case GROESTL:
				groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("grstl80:");
				break;
			case JH:
				jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("jh51280:");
				break;
			case KECCAK:
				keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("kecck80:");
				break;
			case SKEIN:
				skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
				TRACE("skein80:");
				break;
			case LUFFA:
				qubit_luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				TRACE("luffa80:");
				break;
			case CUBEHASH:
				cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("cube 80:");
				break;
			case SHAVITE:
				shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				TRACE("shavite:");
				break;
			case SIMD:
				simd512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("simd512:");
				break;
			case ECHO:
				echo512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("echo   :");
				break;
			case HAMSI:
				x16_hamsi512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("hamsi  :");
				break;
			case FUGUE:
				fugue512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("fugue  :");
				break;
			case SHABAL:
				shabal512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("shabal :");
				break;
			case WHIRLPOOL:
				x16_whirlpool512_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("whirl  :");
				break;
			case SHA512:
				x16_sha512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("sha512 :");
				break;
		}

		for (int i = 1; i < 16; )
		{
			if (fused_run[i] >= 2) {
				const int len = fused_run[i];
				x_fused_cpu_hash_64(thr_id, throughput, i - 1, len, 0, d_hash[thr_id]);
				order += len;
				i += len;
				TRACE("fused  :");
				continue;
			}

			const char elem = hashOrder[i];
			const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
			i++;

			switch (algo64) {
			case BLAKE:
				blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("blake  :");
				break;
			case BMW:
				bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("bmw    :");
				break;
			case GROESTL:
				groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("groestl:");
				break;
			case JH:
				jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("jh512  :");
				break;
			case KECCAK:
				keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
				TRACE("keccak :");
				break;
			case SKEIN:
				skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("skein  :");
				break;
			case LUFFA:
				luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("luffa  :");
				break;
			case CUBEHASH:
				cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
				TRACE("cube   :");
				break;
			case SHAVITE:
				shavite512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
				TRACE("shavite:");
				break;
			case SIMD:
				simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("simd   :");
				break;
			case ECHO:
				if (use_compat_kernels[thr_id])
					echo512_cpu_hash_64_compat(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				else {
					echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
				}
				TRACE("echo   :");
				break;
			case HAMSI:
				hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("hamsi  :");
				break;
			case FUGUE:
				fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("fugue  :");
				break;
			case SHABAL:
				shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("shabal :");
				break;
			case WHIRLPOOL:
				whirlpool512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("shabal :");
				break;
			case SHA512:
				sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("sha512 :");
				break;
			}
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
#ifdef _DEBUG
		uint32_t _ALIGN(64) dhash[8];
		be32enc(&endiandata[19], pdata[19]);
		x16r_hash(dhash, endiandata);
		applog_hash(dhash);
		return -1;
#endif
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			x16r_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					x16r_hash(vhash, endiandata);
					work_check_second_nonce(thr_id, work, vhash);
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
#if 0
				gpulog(LOG_INFO, thr_id, "hash found with %s 80!", algo_strings[algo80]);

				algo80_tests[algo80] += work->valid_nonces;
				char oks64[128] = { 0 };
				char oks80[128] = { 0 };
				char fails[128] = { 0 };
				for (int a = 0; a < HASH_FUNC_COUNT; a++) {
					const char elem = hashOrder[a];
					const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
					if (a > 0) algo64_tests[algo64] += work->valid_nonces;
					sprintf(&oks64[strlen(oks64)], "|%X:%2d", a, algo64_tests[a] < 100 ? algo64_tests[a] : 99);
					sprintf(&oks80[strlen(oks80)], "|%X:%2d", a, algo80_tests[a] < 100 ? algo80_tests[a] : 99);
					sprintf(&fails[strlen(fails)], "|%X:%2d", a, algo80_fails[a] < 100 ? algo80_fails[a] : 99);
				}
				applog(LOG_INFO, "K64: %s", oks64);
				applog(LOG_INFO, "K80: %s", oks80);
				applog(LOG_ERR,  "F80: %s", fails);
#endif
				return work->valid_nonces;
			}
			else {
				// failed candidate or top-word tie: count and log it, scan on from the next nonce
				gpu_increment_reject(thr_id);
				algo80_fails[algo80]++;
				if (!opt_quiet) gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %s",
						work->nonces[0], algo_strings[algo80], hashOrder);
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
extern "C" void free_x16r(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);

	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);
	simd512_cpu_free(thr_id);
	fugue512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id); // to merge with x13_fugue512 ?
	whirlpool512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
