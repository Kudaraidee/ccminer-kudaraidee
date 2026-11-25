/**
 * X11EVO algo implementation
 * Cuda implementation by tpruvot@github - May 2016
 *
 * Migrated to the shared x-family machinery (docs/coding-guideline.md §2/§3):
 * the 11 stages (blake..echo, permuted per block-epoch) call the bare <prim>512
 * device-launcher names through the cuda_x_stages.h bridge (sp shavite, alexis
 * echo). This is a VARIABLE-ORDER chain (like x16r/bitcore) with a fixed blake
 * 80-byte first stage: the fusible-run structure is recomputed whenever the
 * order changes (fused_run[] + an x11evo->fused-kernel id map, since the stage
 * enums differ) and consecutive register-resident stages run via the shared
 * fused kernel. The terminal is order-dependent so the terminal-compare
 * (echo/skein _final) is NOT applied (like x16/bitcore).
 */
#include <stdio.h>
#include <memory.h>

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
}

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos/common/cuda_x_stages.h"

static uint32_t *d_hash[MAX_GPUS];

enum Algo {
	BLAKE = 0,
	BMW,
	GROESTL,
	SKEIN,
	JH,
	KECCAK,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HASH_FUNC_COUNT
};

/* x11evo's stage enum differs from the shared fused kernel's (x11evo has
 * SKEIN=3,JH=4,KECCAK=5 vs the fused kernel's JH=3,KECCAK=4,SKEIN=5); map
 * x11evo ids -> fused-kernel ids so the shared fused kernel + x_fusible[]
 * predicate select the right stages. */
static const uint8_t evo_to_fused[HASH_FUNC_COUNT] = { 0, 1, 2, 5, 3, 4, 6, 7, 8, 9, 10 };
static __thread int s_fused_seq = -1;
static __thread uint8_t fused_run[HASH_FUNC_COUNT] = { 0 };

static void swap8(uint8_t *a, uint8_t *b)
{
	uint8_t t = *a;
	*a = *b;
	*b = t;
}

static void initPerm(uint8_t n[], int count)
{
	for (int i = 0; i < count; i++)
		n[i] = i;
}

static int nextPerm(uint8_t n[], int count)
{
	int tail, i, j;

	if (count <= 1)
		return 0;

	for (i = count - 1; i>0 && n[i - 1] >= n[i]; i--);
	tail = i;

	if (tail > 0) {
		for (j = count - 1; j>tail && n[j] <= n[tail - 1]; j--);
		swap8(&n[tail - 1], &n[j]);
	}

	for (i = tail, j = count - 1; i<j; i++, j--)
		swap8(&n[i], &n[j]);

	return (tail != 0);
}

static void getAlgoString(char *str, int seq)
{
	uint8_t algoList[HASH_FUNC_COUNT];
	char *sptr;

	initPerm(algoList, HASH_FUNC_COUNT);

	for (int k = 0; k < seq; k++) {
		nextPerm(algoList, HASH_FUNC_COUNT);
	}

	sptr = str;
	for (int j = 0; j < HASH_FUNC_COUNT; j++) {
		if (algoList[j] >= 10)
			sprintf(sptr, "%c", 'A' + (algoList[j] - 10));
		else
			sprintf(sptr, "%u", (uint32_t) algoList[j]);
		sptr++;
	}
	*sptr = '\0';
	//applog(LOG_DEBUG, "nextPerm %s", str);
}

static __thread uint32_t s_ntime = 0;
static char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };
static int  s_sequence = -1;

#define INITIAL_DATE 0x57254700
static inline int getCurrentAlgoSeq(uint32_t current_time)
{
	// change once per day
	return (int) (current_time - INITIAL_DATE) / (60 * 60 * 24);
}

static void evo_twisted_code(uint32_t ntime, char *permstr)
{
	int seq = getCurrentAlgoSeq(ntime);
	if (s_sequence != seq) {
		getAlgoString(permstr, seq);
		s_sequence = seq;
	}
}

// X11evo CPU Hash
extern "C" void x11evo_hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[64/4] = { 0 };

	sph_blake512_context     ctx_blake;
	sph_bmw512_context       ctx_bmw;
	sph_groestl512_context   ctx_groestl;
	sph_skein512_context     ctx_skein;
	sph_jh512_context        ctx_jh;
	sph_keccak512_context    ctx_keccak;
	sph_luffa512_context     ctx_luffa1;
	sph_cubehash512_context  ctx_cubehash1;
	sph_shavite512_context   ctx_shavite1;
	sph_simd512_context      ctx_simd1;
	sph_echo512_context      ctx_echo1;

	if (s_sequence == -1) {
		uint32_t *data = (uint32_t*) input;
		const uint32_t ntime = data[17];
		evo_twisted_code(ntime, hashOrder);
	}

	void *in = (void*) input;
	int size = 80;

	const int hashes = (int) strlen(hashOrder);

	for (int i = 0; i < hashes; i++)
	{
		const char elem = hashOrder[i];
		uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

		if (i > 0) {
			in = (void*) hash;
			size = 64;
		}

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
			sph_luffa512_init(&ctx_luffa1);
			sph_luffa512(&ctx_luffa1, in, size);
			sph_luffa512_close(&ctx_luffa1, hash);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash1);
			sph_cubehash512(&ctx_cubehash1, in, size);
			sph_cubehash512_close(&ctx_cubehash1, hash);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite1);
			sph_shavite512(&ctx_shavite1, in, size);
			sph_shavite512_close(&ctx_shavite1, hash);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd1);
			sph_simd512(&ctx_simd1, in, size);
			sph_simd512_close(&ctx_simd1, hash);
			break;
		case ECHO:
			sph_echo512_init(&ctx_echo1);
			sph_echo512(&ctx_echo1, in, size);
			sph_echo512_close(&ctx_echo1, hash);
			break;
		}
	}

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "evo"
#include "cuda/debug.cuh"

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_x11evo(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 20 : 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_debug || s_ntime != pdata[17] || s_sequence == -1) {
		uint32_t ntime = swab32(work->data[17]);
		evo_twisted_code(ntime, hashOrder);
		s_ntime = pdata[17];
		if (opt_debug) {
			int secs = (int) (ntime - INITIAL_DATE) % (60 * 60 * 24);
			secs = (60 * 60 * 24) - secs;
			applog(LOG_DEBUG, "evo hash order %s, next in %d mn", hashOrder, secs/60);
		}
	}

	if (opt_benchmark)
		ptarget[7] = 0x5;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
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
		keccak512_cpu_init(thr_id, throughput);
		jh512_cpu_init(thr_id, throughput);
		luffa512_cpu_init(thr_id, throughput); // 64
		shavite512_cpu_init(thr_id, throughput);
		if (simd512_cpu_init(thr_id, throughput) != 0) {
			return 0;
		}
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	cuda_check_cpu_setTarget(ptarget);
	blake512_cpu_setBlock_80(thr_id, endiandata);

	const int hashes = (int) strlen(hashOrder);

	/* (re)build the per-permutation fused-run structure + upload the mapped
	 * 64-byte stage ids whenever the order changes (rarely — once per block
	 * epoch); done post-init so the device is ready. Positions 1..hashes-1 are
	 * the 64-byte stages (pos 0 is the blake-80). */
	if (s_fused_seq != s_sequence) {
		uint8_t ids[HASH_FUNC_COUNT];
		memset(fused_run, 0, sizeof(fused_run));
		for (int i = 1; i < hashes; i++) {
			const char e = hashOrder[i];
			const uint8_t a = e >= 'A' ? e - 'A' + 10 : e - '0';
			ids[i] = evo_to_fused[a];
		}
		for (int i = 1; i < hashes; ) {
			int len = 0;
			while (i + len < hashes && x_fusible[ids[i + len]]) len++;
			if (len >= 2) fused_run[i] = (uint8_t) len;
			i += (len > 0) ? len : 1;
		}
		/* selftest clobbers the order constant, so run it before the real upload */
		x_fused_device_selftest(thr_id);
		x_fused_setOrder(&ids[1], hashes - 1);
		s_fused_seq = s_sequence;
	}

	do {
		int order = 1;

		// Hash with CUDA
		blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		TRACE("blake80:");

		for (int i = 1; i < hashes; )
		{
			/* fused run of >= 2 consecutive register-resident stages */
			if (fused_run[i] >= 2) {
				const int len = fused_run[i];
				x_fused_cpu_hash_64(thr_id, throughput, i - 1, len, 0, d_hash[thr_id]);
				TRACE("fused  :");
				i += len;
				order += len;
				continue;
			}

			const char elem = hashOrder[i];
			const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

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
			case SKEIN:
				skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("skein  :");
				break;
			case JH:
				jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("jh512  :");
				break;
			case KECCAK:
				keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
				TRACE("keccak :");
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
			}
			i++;
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			x11evo_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				pdata[19] = work->nonces[0] + 1; // cursor
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					x11evo_hash(vhash, endiandata);
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
					gpulog(LOG_DEBUG, thr_id, "second nonce %08x! cursor %08x", work->nonces[1], pdata[19]);
					work_check_second_nonce(thr_id, work, vhash);
				}
				return work->valid_nonces;
			} else if (vhash[7] > Htarg) {
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

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}

// cleanup
extern "C" void free_x11evo(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);

	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);
	simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
