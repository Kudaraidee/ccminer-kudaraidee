/**
 * Timetravel (timetravel8, Machinecoin) CUDA implementation
 *  by tpruvot@github - March 2017
 *
 * Migrated to the shared x-family machinery (docs/coding-guideline.md §2/§3):
 * the 8 stages (blake,bmw,groestl,skein,jh,keccak,luffa,cubehash, permuted per
 * ntime) call the bare <prim>512 device-launcher names through the
 * cuda_x_stages.h bridge. This is a VARIABLE-ORDER chain (like x16r/bitcore):
 * the fusible-run structure is recomputed per permutation (fused_run[] + a
 * timetravel->fused-kernel id map, since the stage enums differ) and consecutive
 * register-resident stages run via the shared fused kernel. Only groestl is a
 * fusion boundary here, so 7 of 8 stages fuse. The terminal is order-dependent
 * so the terminal-compare (echo/skein _final) is NOT applied (like x16/bitcore).
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

#define HASH_FUNC_BASE_TIMESTAMP 1389040865U // Machinecoin Genesis Timestamp
#define HASH_FUNC_COUNT 8
#define HASH_FUNC_COUNT_PERMUTATIONS 40320U

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
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
	MAX_ALGOS_COUNT
};

static const char* algo_strings[] = {
	"blake",
	"bmw512",
	"groestl",
	"skein",
	"jh512",
	"keccak",
	"luffa",
	"cube",
	NULL
};

inline void swap8(uint8_t *a, uint8_t *b)
{
	uint8_t t = *a;
	*a = *b;
	*b = t;
}

inline void initPerm(uint8_t n[], int count)
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
}

/* The scan's order is per miner thread: at a job change every thread keeps
 * hashing (and CPU-checking) its own job with that job's order until it sees
 * work_restart. A process-global order let the first thread on the new job
 * rewrite it under the others. */
static __thread uint32_t s_ntime = 0;
static __thread uint32_t s_sequence = UINT32_MAX;
static __thread uint8_t s_firstalgo = 0xFF;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

/* the CPU hash derives its own order from its input's ntime (cached per thread) */
static __thread uint32_t s_cpu_sequence = UINT32_MAX;
static __thread char s_cpu_order[HASH_FUNC_COUNT + 1] = { 0 };

#define INITIAL_DATE HASH_FUNC_BASE_TIMESTAMP
static inline uint32_t getCurrentAlgoSeq(uint32_t ntime)
{
	// unlike x11evo, the permutation changes often (with ntime)
	return (uint32_t) (ntime - INITIAL_DATE) % HASH_FUNC_COUNT_PERMUTATIONS;
}

// To finish...
static void get_travel_order(uint32_t ntime, char *permstr)
{
	uint32_t seq = getCurrentAlgoSeq(ntime);
	if (s_sequence != seq) {
		getAlgoString(permstr, seq);
		s_sequence = seq;
	}
}

// CPU Hash
extern "C" void timetravel_hash(void *output, const void *input)
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

	/* The order is a function of the header being hashed: word 17 of the
	 * byte-swapped header is the ntime the scan uses (swab32(pdata[17])). */
	{
		const uint32_t seq = getCurrentAlgoSeq(((const uint32_t*) input)[17]);
		if (seq != s_cpu_sequence) {
			getAlgoString(s_cpu_order, seq);
			s_cpu_sequence = seq;
		}
	}
	const char *hashOrder = s_cpu_order;

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
			//applog_hex((void*)hash, 32);
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
		}
	}

	memcpy(output, hash, 32);
}

static uint32_t get_next_time(uint32_t ntime, char* curOrder)
{
	char nextOrder[HASH_FUNC_COUNT + 1] = { 0 };
	uint32_t secs = 15;
	do {
		uint32_t nseq = getCurrentAlgoSeq(ntime+secs);
		getAlgoString(nextOrder, nseq);
		secs += 15;
	} while (curOrder[0] == nextOrder[0]);
	return secs;
}

//#define _DEBUG
#define _DEBUG_PREFIX "tt-"
#include "cuda/debug.cuh"

/* timetravel's stage enum differs from the shared fused kernel's (timetravel has
 * SKEIN=3,JH=4,KECCAK=5 vs the fused kernel's JH=3,KECCAK=4,SKEIN=5); map
 * timetravel ids -> fused-kernel ids so the shared fused kernel + x_fusible[]
 * predicate select the right stages. */
static const uint8_t tt_to_fused[HASH_FUNC_COUNT] = { 0, 1, 2, 5, 3, 4, 6, 7 };
static __thread uint32_t s_fused_seq = UINT32_MAX;
static __thread uint8_t fused_run[HASH_FUNC_COUNT] = { 0 };

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_timetravel(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 20 : 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 19=256*256*8;
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	// if (opt_benchmark) pdata[17] = swab32(0x5886a4be); // TO DEBUG GROESTL 80

	if (opt_debug || s_ntime != pdata[17] || s_sequence == UINT32_MAX) {
		uint32_t ntime = swab32(work->data[17]);
		get_travel_order(ntime, hashOrder);
		s_ntime = pdata[17];
		if (opt_debug && !thr_id) {
			applog(LOG_DEBUG, "timetravel hash order %s (%08x)", hashOrder, ntime);
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

		blake512_cpu_init(thr_id, throughput);
		bmw512_cpu_init(thr_id, throughput);
		groestl512_cpu_init(thr_id, throughput);
		skein512_cpu_init(thr_id, throughput);
		keccak512_cpu_init(thr_id, throughput);
		jh512_cpu_init(thr_id, throughput);
		qubit_luffa512_cpu_init(thr_id, throughput); // 80-byte luffa (only constants, 480 bytes)
		luffa512_cpu_init(thr_id, throughput);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMemset(d_hash[thr_id], 0, (size_t) 64 * throughput), -1);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	cuda_check_cpu_setTarget(ptarget);

	const int hashes = (int) strlen(hashOrder);
	const char first = hashOrder[0];
	const uint8_t algo80 = first >= 'A' ? first - 'A' + 10 : first - '0';
	if (algo80 != s_firstalgo) {
		s_firstalgo = algo80;
		if (!thr_id) applog(LOG_INFO, "Timetravel first algo is now %s", algo_strings[algo80 % HASH_FUNC_COUNT]);
	}

	/* (re)build the per-permutation fused-run structure + upload the mapped
	 * 64-byte stage ids whenever the order changes; done post-init so the device
	 * is ready. Positions 1..hashes-1 are the 64-byte stages (pos 0 is hashed at
	 * 80 bytes by the algo80 switch below). */
	if (s_fused_seq != s_sequence) {
		uint8_t ids[HASH_FUNC_COUNT];
		memset(fused_run, 0, sizeof(fused_run));
		for (int i = 1; i < hashes; i++) {
			const char e = hashOrder[i];
			const uint8_t a = e >= 'A' ? e - 'A' + 10 : e - '0';
			ids[i] = tt_to_fused[a];
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
		case SKEIN:
			skein512_cpu_setBlock_80((void*)endiandata);
			break;
		case JH:
			jh512_setBlock_80(thr_id, endiandata);
			break;
		case KECCAK:
			keccak512_setBlock_80(thr_id, endiandata);
			break;
		case LUFFA:
			qubit_luffa512_cpu_setBlock_80((void*)endiandata);
			break;
		case CUBEHASH:
			cubehash512_setBlock_80(thr_id, endiandata);
			break;
		default: {
			uint32_t next = get_next_time(swab32(s_ntime), hashOrder);
			if (!thr_id)
				applog(LOG_WARNING, "kernel %c unimplemented, next in %u mn", first, next/60);
			sleep(next > 30 ? 60 : 10);
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
			case SKEIN:
				skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
				TRACE("skein80:");
				break;
			case JH:
				jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("jh51280:");
				break;
			case KECCAK:
				keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("kecck80:");
				break;
			case LUFFA:
				qubit_luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				TRACE("luffa80:");
				break;
			case CUBEHASH:
				cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("cube 80:");
				break;
		}

		for (int i = 1; i < hashes; )
		{
			/* fused run of >= 2 consecutive register-resident stages */
			if (fused_run[i] >= 2) {
				const int len = fused_run[i];
				x_fused_cpu_hash_64(thr_id, throughput, i - 1, len, 0, d_hash[thr_id]);
				TRACE("fused  :");
				i += len;
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
			}
			i++;
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			be32enc(&endiandata[19], work->nonces[0]);
			timetravel_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				pdata[19] = work->nonces[0];
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					timetravel_hash(vhash, endiandata);
					if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						bn_set_target_ratio(work, vhash, 1);
						work->valid_nonces++;
					}
					pdata[19] = max(pdata[19], work->nonces[1]) + 1;
				}
				return work->valid_nonces;
			} else if (vhash[7] > Htarg) {
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
extern "C" void free_timetravel(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);

	blake512_cpu_free(thr_id);
	groestl512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
