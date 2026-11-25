#include "miner.h"
#include "cuda/helper.cuh"
extern "C" {
#include "sph/yescrypt.h"
}

#include <math.h>
extern "C" {
#include "sph/SHA3api_ref.h"
}

extern void yescrypt_cpu_init(int thr_id, int threads, uint32_t *d_hash1, uint32_t *d_hash2, uint32_t *d_hash3, uint32_t *d_hash4);
extern void yescrypt_setTarget(int thr_id, uint32_t pdata[20], char *key, uint32_t key_len);
extern uint32_t yescrypt_cpu_hash_32(int thr_id, uint32_t threads, uint32_t valid, uint32_t startNounce, uint32_t *resultnonces, uint32_t max_results, uint32_t target, const uint32_t N, const uint32_t r, const uint32_t p);
extern void yescrypt_cpu_free(int thr_id);

/* candidates the launcher hands back per batch (it keeps a counted slot array) */
#define YS_HOST_SLOTS 8

extern char *yescrypt_key;
extern size_t yescrypt_key_len;
extern uint32_t yescrypt_param_N;
extern uint32_t yescrypt_param_r;
extern uint32_t yescrypt_param_p;

void yescrypt_hash_base(void *state, const void *input, const uint32_t N, const uint32_t r, const uint32_t p, char *key, const size_t key_len)
{
	client_key = key;
	client_key_len = key_len;
	yescrypt_bsty((unsigned char*)input, 80, (unsigned char*)input, 80, N, r, p, (unsigned char *)state, 32);
}

void yescrypt_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 2048, 8, 1, NULL, 0);
}

void yescryptr8_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 2048, 8, 1, "Client Key", 10);
}

void yescryptr16_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 4096, 16, 1, "Client Key", 10);
}

void yescryptr16v2_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 4096, 16, 4, "PPTPPubKey", 10);
}

void yescryptr24_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 4096, 24, 1, "Jagaricoin", 10);
}

void yescryptr32_hash(void *state, const void *input)
{
	yescrypt_hash_base(state, input, 4096, 32, 1, "WaviBanana", 10);
}

int scanhash_yescrypt_base(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done,
	const uint32_t N, const uint32_t r, const uint32_t p,
	char *key, const size_t key_len) {
	static THREAD uint32_t *d_hash1 = nullptr;
	static THREAD uint32_t *d_hash2 = nullptr;
	static THREAD uint32_t *d_hash3 = nullptr;
	static THREAD uint32_t *d_hash4 = nullptr;

	uint32_t* pdata = work->data;
	uint32_t* ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (key && key_len > YESCRYPT_KEY_MAX) {
		gpulog(LOG_ERR, thr_id, "yescrypt key is %u bytes, at most %d are supported",
		       (unsigned)key_len, YESCRYPT_KEY_MAX);
		proper_exit(EXIT_CODE_USAGE);
	}

	int dev_id = device_map[thr_id];
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, dev_id);

	uint32_t CUDAcore_count;
	if (device_sm[dev_id] == 600)		// Pascal(P100)
		CUDAcore_count = props.multiProcessorCount * 64;
	else								// Maxwell/Pascal/Volta and everything newer; the Kepler and
										// Fermi arms were unreachable under a sm_61 floor
		CUDAcore_count = props.multiProcessorCount * 128;

	uint32_t throughputmax;
#if defined WIN32 && !defined _WIN64
	// 2GB limit for cudaMalloc
	uint32_t max_thread_multiple = (min(0x7fffffffULL, props.totalGlobalMem) - 256 * 1024 * 1024) / (((520 + 2 * r * (N + 16 * p)) * sizeof(uint32_t)) * CUDAcore_count);
#else
	uint32_t max_thread_multiple = (props.totalGlobalMem - 256 * 1024 * 1024) / (((520 + 2 * r * (N + 16 * p)) * sizeof(uint32_t)) * CUDAcore_count);
#endif

	if (device_sm[dev_id] > 500)		// Maxwell(GTX9xx)/Pascal/Volta
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * min(3, max_thread_multiple));
	else								// sm_61 floor: nothing below 500 can load this build,
										// but keep an else so the value is always initialised
		throughputmax = device_intensity(dev_id, __func__, CUDAcore_count * min(2, max_thread_multiple));

	throughputmax = (throughputmax / CUDAcore_count) * CUDAcore_count;
	if (throughputmax == 0) throughputmax = CUDAcore_count;

	// The launcher tiles a batch three ways (grid = threads/32, k1's 4-way and
	// SMix's 16-way splits) and needs a multiple of 32; every launch below is
	// padded up to one, so the allocation must be one too (CUDAcore_count is).
	throughputmax &= ~31u;
	if (throughputmax == 0) throughputmax = 32;

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;

	static THREAD bool init = false;
	if (!init)
	{
		applog(LOG_WARNING, "Using intensity %.3f (%d threads)", throughput2intensity(throughputmax), throughputmax);
		CUDA_SAFE_CALL(cudaSetDevice(dev_id));
		CUDA_SAFE_CALL(cudaDeviceReset());
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		//CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
#if defined WIN32 && !defined _WIN64
		// 2GB limit for cudaMalloc
		if (throughputmax > (min(0x7fffffffULL, props.totalGlobalMem) - 256 * 1024 * 1024) / ((520 + 2 * r * (N + 16 * p)) * sizeof(uint32_t)))
#else
		if (throughputmax > (props.totalGlobalMem - 256 * 1024 * 1024) / ((520 + 2 * r * (N + 16 * p)) * sizeof(uint32_t)))
#endif
		{
			applog(LOG_ERR, "Memory Error");
//			mining_has_stopped[thr_id] = true;
//			cudaStreamDestroy(gpustream[thr_id]);
			proper_exit(2);
		}

		size_t hash1_sz = 2 * 16 * r * p * sizeof(uint32_t);	// B
		size_t hash2_sz = 512 * sizeof(uint32_t);				// S(4way)
		size_t hash3_sz = 2 * N * r * sizeof(uint32_t);			// V(16way)
		size_t hash4_sz = 8 * sizeof(uint32_t);					// sha256
		CUDA_SAFE_CALL(cudaMalloc(&d_hash1, hash1_sz * throughputmax));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash2, hash2_sz * throughputmax));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash3, hash3_sz * throughputmax));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash4, hash4_sz * throughputmax));

		yescrypt_cpu_init(thr_id, throughputmax, d_hash1, d_hash2, d_hash3, d_hash4);

		cuda_check_cpu_init(thr_id, throughputmax);
//		mining_has_stopped[thr_id] = false;

		init = true;
	}

	uint32_t endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	yescrypt_setTarget(thr_id, pdata, key, key_len);
	const uint32_t Htarg = ptarget[7];
	do {
		if (pdata[19] >= max_nonce)
			break;
		// n = real nonces of this launch, clamped to the end of the range. The
		// launch is padded to a multiple of 32 (<= throughputmax, both are); the
		// result kernel only reports lanes below n, so padding cannot report.
		const uint32_t n = (uint32_t) min((uint64_t) throughputmax, (uint64_t) max_nonce - pdata[19]);
		const uint32_t launch = (n + 31u) & ~31u;

		uint32_t found[YS_HOST_SLOTS];
		const uint32_t nfound = yescrypt_cpu_hash_32(thr_id, launch, n, pdata[19], found, YS_HOST_SLOTS,
		                                             Htarg, N, r, p);

		if (nfound) {
			// slots are in atomicAdd order: check in nonce order, so the two
			// reported are the lowest valid ones and every later candidate lies
			// above the resume point (the next scan finds it again)
			for (uint32_t i = 1; i < nfound; i++)
				for (uint32_t j = i; j > 0 && found[j - 1] > found[j]; j--) {
					const uint32_t t = found[j]; found[j] = found[j - 1]; found[j - 1] = t;
				}

			work->valid_nonces = 0;
			for (uint32_t i = 0; i < nfound && work->valid_nonces < 2; i++) {
				uint32_t _ALIGN(64) vhash64[8] = { 0 };
				be32enc(&endiandata[19], found[i]);
				yescrypt_hash_base(vhash64, endiandata, N, r, p, key, key_len);
				if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
					work->nonces[work->valid_nonces] = found[i];
					bn_set_target_ratio(work, vhash64, work->valid_nonces);
					work->valid_nonces++;
					if (opt_benchmark)
						gpulog(LOG_INFO, thr_id, "found nonce %08x", found[i]);
				} else if (vhash64[7] > Htarg) {
					// the screen compares word 7 only, so a word-7 miss is a
					// wrong GPU hash (equal word 7 failing fulltest is legitimate)
					gpu_increment_reject(thr_id);
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", found[i]);
				}
			}

			if (work->valid_nonces) {
				*hashes_done = pdata[19] - first_nonce + n;
				// resume at the highest reported nonce; the core steps past it
				// (nonceptr[0]++ on the same work)
				pdata[19] = work->nonces[work->valid_nonces - 1];
				return work->valid_nonces;
			}
		}

		if ((uint64_t) n + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += n;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

int scanhash_yescrypt(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	if (yescrypt_param_N == 0) yescrypt_param_N = 2048;
	if (yescrypt_param_r == 0) yescrypt_param_r = 8;
	if (yescrypt_param_p == 0) yescrypt_param_p = 1;
	return  scanhash_yescrypt_base(thr_id, work, max_nonce, hashes_done, yescrypt_param_N, yescrypt_param_r, yescrypt_param_p, yescrypt_key, yescrypt_key_len);
}

int scanhash_yescryptr8(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	return  scanhash_yescrypt_base(thr_id, work, max_nonce, hashes_done, 2048, 8, 1, "Client Key", 10);
}

int scanhash_yescryptr16(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	return  scanhash_yescrypt_base(thr_id, work, max_nonce, hashes_done, 4096, 16, 1, "Client Key", 10);
}

int scanhash_yescryptr16v2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	return  scanhash_yescrypt_base(thr_id, work, max_nonce, hashes_done, 4096, 16, 4, "PPTPPubKey", 10);
}

int scanhash_yescryptr24(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	return  scanhash_yescrypt_base(thr_id, work, max_nonce, hashes_done, 4096, 24, 1, "Jagaricoin", 10);
}

int scanhash_yescryptr32(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	return  scanhash_yescrypt_base(thr_id, work, max_nonce, hashes_done, 4096, 32, 1, "WaviBanana", 10);
}
