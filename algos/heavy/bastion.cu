/**
 * bastion cuda implemention tpruvot@github 2017
 */

#include <stdio.h>
#include <string.h>
//#include <openssl/sha.h>
#include <stdint.h>
#include <miner.h>
#include "cuda/helper.cuh"
#include "cuda/selftest_gate.cuh"

static uint32_t *d_hash[MAX_GPUS];
static uint32_t* d_hash_br[MAX_GPUS]; // branch 1 at [0, n), branch 2 at [n, threads)

extern void skein512_cpu_init(int thr_id, uint32_t threads);
extern void skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void luffa512_cpu_init(int thr_id, uint32_t threads);
extern void luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void fugue512_cpu_init(int thr_id, uint32_t threads);
extern void fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void fugue512_cpu_free(int thr_id);

extern void shabal512_cpu_init(int thr_id, uint32_t threads);
extern void shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void whirlpool512_cpu_init(int thr_id, uint32_t threads, int mode);
extern void whirlpool512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void whirlpool512_cpu_free(int thr_id);

extern void echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash); // alexis 64-byte echo (no cpu_init)

extern void bastion_cpu_init(const int thr_id, const uint32_t threads);
extern void bastion_free(const int thr_id);

extern uint32_t bastion_filter2(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_branch);
extern void bastion_merge2(const int thr_id, const uint32_t threads, uint32_t *outpHashes, const uint32_t* d_branch);

extern void hefty_cpu_hash_64(int thr_id, uint32_t threads, int startNounce, uint32_t *d_outputHash);
extern void hefty_cpu_setBlock(int thr_id, uint32_t threads, void *data, int len);
extern void hefty_cpu_init_tables(int thr_id);

static bool init[MAX_GPUS] = { 0 };

// Full chain for one batch; the PoW hash is the first 32 bytes of each d_hash record.
static void bastion_hash_batch(int thr_id, uint32_t threads, uint32_t startNonce)
{
	uint32_t branchNonces;
	// MyStreamSynchronize sleep slots: 0 belongs to hefty_cpu_hash_64, so start at 1
	int order = 1;

	// hefty
	hefty_cpu_hash_64(thr_id, threads, startNonce, d_hash[thr_id]);

	luffa512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], order++);

	// fugue or skein
	branchNonces = bastion_filter2(thr_id, threads, d_hash[thr_id], d_hash_br[thr_id]);
	fugue512_cpu_hash_64(thr_id, branchNonces, startNonce, NULL, d_hash_br[thr_id], order++);
	skein512_cpu_hash_64(thr_id, threads-branchNonces, startNonce, NULL, &d_hash_br[thr_id][(size_t) branchNonces * 16], order++);
	bastion_merge2(thr_id, threads, d_hash[thr_id], d_hash_br[thr_id]);

	whirlpool512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], order++);
	fugue512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], order++);

	// echo or luffa
	branchNonces = bastion_filter2(thr_id, threads, d_hash[thr_id], d_hash_br[thr_id]);
	echo512_cpu_hash_64(thr_id, branchNonces, d_hash_br[thr_id]); order++;
	luffa512_cpu_hash_64(thr_id, threads-branchNonces, startNonce, NULL, &d_hash_br[thr_id][(size_t) branchNonces * 16], order++);
	bastion_merge2(thr_id, threads, d_hash[thr_id], d_hash_br[thr_id]);

	shabal512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], order++);
	skein512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], order++);

	// shabal or whirlpool
	branchNonces = bastion_filter2(thr_id, threads, d_hash[thr_id], d_hash_br[thr_id]);
	shabal512_cpu_hash_64(thr_id, branchNonces, startNonce, NULL, d_hash_br[thr_id], order++);
	whirlpool512_cpu_hash_64(thr_id, threads-branchNonces, startNonce, NULL, &d_hash_br[thr_id][(size_t) branchNonces * 16], order++);
	bastion_merge2(thr_id, threads, d_hash[thr_id], d_hash_br[thr_id]);

	shabal512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], order++);

	// hamsi or luffa
	branchNonces = bastion_filter2(thr_id, threads, d_hash[thr_id], d_hash_br[thr_id]);
	hamsi512_cpu_hash_64(thr_id, branchNonces, startNonce, NULL, d_hash_br[thr_id], order++);
	luffa512_cpu_hash_64(thr_id, threads-branchNonces, startNonce, NULL, &d_hash_br[thr_id][(size_t) branchNonces * 16], order++);
	bastion_merge2(thr_id, threads, d_hash[thr_id], d_hash_br[thr_id]);
}

// Mismatches between d_hash records [offset, offset+span) of the batch hashed from `start`
// and bastionhash(endiandata); UINT32_MAX if the readback failed.
static uint32_t bastion_compare_slice(int thr_id, uint32_t *endiandata, uint32_t start, uint32_t offset, uint32_t span)
{
	uint32_t vhash[8], bad = 0;
	uint32_t *h_hash = (uint32_t*) malloc((size_t) 64 * span);
	if (!h_hash)
		return UINT32_MAX;
	if (cudaMemcpy(h_hash, &d_hash[thr_id][(size_t) offset * 16], (size_t) 64 * span, cudaMemcpyDeviceToHost) != cudaSuccess) {
		free(h_hash);
		return UINT32_MAX;
	}
	const uint32_t saved = endiandata[19];
	for (uint32_t i = 0; i < span; i++) {
		endiandata[19] = start + offset + i;
		bastionhash(vhash, (uchar*) endiandata);
		bad += memcmp(vhash, &h_hash[i * 16], 32) != 0;
	}
	endiandata[19] = saved;
	free(h_hash);
	return bad;
}

// Init self-test: 256 nonces across a byte carry (both sides of every branch occur),
// plus a flipped-header negative.
static bool bastion_device_selftest(int thr_id, uint32_t threads)
{
	const uint32_t span = min(threads, 256U);
	const uint32_t start = 0x0001ff80U;
	uint32_t endiandata[20];

	for (int k = 0; k < 20; k++)
		endiandata[k] = 0x9e3779b9U * (k + 1);
	hefty_cpu_setBlock(thr_id, span, endiandata, 80);
	bastion_hash_batch(thr_id, span, start);
	const uint32_t bad = bastion_compare_slice(thr_id, endiandata, start, 0, span);
	endiandata[1] ^= 1; // the GPU hashed the unflipped header: every digest must differ
	const uint32_t neg_bad = bastion_compare_slice(thr_id, endiandata, start, 0, span);
	if (bad == UINT32_MAX || neg_bad == UINT32_MAX)
		return selftest_gate(thr_id, "bastion", selftest_cuda_fault());

	const bool passed = bad == 0 && neg_bad == span;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "bastion self-test FAILED (%u of %u digests wrong, negative leg %u of %u)",
			bad, span, neg_bad, span);
	else
		gpulog(LOG_DEBUG, thr_id, "bastion self-test passed (%u nonces)", span);
	return selftest_gate(thr_id, "bastion", passed);
}

// -D only, once per job: a full-size batch of the real header, first and last
// BASTION_DIFF_SPAN digests checked against bastionhash().
#define BASTION_DIFF_SPAN 515U
#define BASTION_DIFF_SKEW 0x51U
static uint32_t diff_sig[MAX_GPUS];
static bool diff_seen[MAX_GPUS];

static void bastion_debug_differential(int thr_id, const uint32_t *pdata, uint32_t *endiandata, uint32_t threads)
{
	uint32_t sig = 0x811c9dc5u;
	for (int i = 0; i < 19; i++)
		sig = (sig ^ pdata[i]) * 0x01000193u;
	if (diff_seen[thr_id] && diff_sig[thr_id] == sig)
		return;
	diff_sig[thr_id] = sig;
	diff_seen[thr_id] = true;

	const uint32_t span = min(threads, BASTION_DIFF_SPAN);
	const uint32_t start = pdata[19] + BASTION_DIFF_SKEW;
	bastion_hash_batch(thr_id, threads, start);
	const uint32_t head = bastion_compare_slice(thr_id, endiandata, start, 0, span);
	const uint32_t tail = bastion_compare_slice(thr_id, endiandata, start, threads - span, span);
	if (head == UINT32_MAX || tail == UINT32_MAX)
		gpulog(LOG_WARNING, thr_id, "differential could not run (CUDA resource failure)");
	else if (head || tail)
		gpulog(LOG_ERR, thr_id, "DIFFERENTIAL MISMATCH in a %u-nonce batch from %08x: head %u/%u, tail %u/%u digests differ from the CPU",
			threads, start, head, span, tail, span);
	else
		gpulog(LOG_DEBUG, thr_id, "differential OK: head+tail %u+%u of a %u-nonce batch from %08x, job %08x",
			span, span, threads, start, sig);
}


int scanhash_bastion(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	// 2^22 measured on GTX 1080 Ti (sm_61) and RTX 3060 (sm_86); sm_75 not measured
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 22);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

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


		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash_br[thr_id], (size_t) 64 * throughput));

		bastion_cpu_init(thr_id, throughput);
		hefty_cpu_init_tables(thr_id);
		luffa512_cpu_init(thr_id, throughput);

		skein512_cpu_init(thr_id, throughput);
		hamsi512_cpu_init(thr_id, throughput);
		fugue512_cpu_init(thr_id, throughput);
		shabal512_cpu_init(thr_id, throughput);
		whirlpool512_cpu_init(thr_id, throughput, 0);

		cuda_check_cpu_init(thr_id, throughput);
		bastion_device_selftest(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	hefty_cpu_setBlock(thr_id, throughput, endiandata, 80);

	cuda_check_cpu_setTarget(ptarget);

	if (opt_debug)
		bastion_debug_differential(thr_id, pdata, endiandata, throughput);

	do {
		bastion_hash_batch(thr_id, throughput, pdata[19]);

		*hashes_done = pdata[19] - first_nonce + throughput;

		CUDA_LOG_ERROR();

		// the kernels scan n = pdata[19] + thread and hash it byte-swapped: n is the cursor,
		// swab32(n) the submitted nonce
		const uint32_t n0 = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (n0 != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			endiandata[19] = n0;
			bastionhash(vhash, (uchar*) endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				uint32_t last = n0;
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[0] = swab32(n0);
				const uint32_t n1 = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (n1 != UINT32_MAX) {
					endiandata[19] = n1;
					bastionhash(vhash, (uchar*) endiandata);
					if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						bn_set_target_ratio(work, vhash, 1);
						work->nonces[1] = swab32(n1);
						work->valid_nonces++;
						last = n1;
					}
				}
				pdata[19] = last + 1;
				return work->valid_nonces;
			}
			gpu_increment_reject(thr_id);
			if (!opt_quiet) gpulog(LOG_WARNING, thr_id,
				"result for %08x does not validate on CPU!", swab32(n0));
			pdata[19] = n0 + 1;
			continue;
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	CUDA_LOG_ERROR();

	return 0;
}

// cleanup
extern "C" void free_bastion(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_hash_br[thr_id]);

	fugue512_cpu_free(thr_id);
	whirlpool512_cpu_free(thr_id);

	bastion_free(thr_id);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}

#undef SPH_C32
#undef SPH_T32
#undef SPH_C64
#undef SPH_T64
extern "C" {
#include "hefty1.h"
#include "sph/sph_luffa.h"
#include "sph/sph_fugue.h"
#include "sph/sph_skein.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_shabal.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
}

__host__
void bastionhash(void* output, const uchar* input)
{
	unsigned char _ALIGN(128) hash[64] = { 0 };

	sph_echo512_context ctx_echo;
	sph_luffa512_context ctx_luffa;
	sph_fugue512_context ctx_fugue;
	sph_whirlpool_context ctx_whirlpool;
	sph_shabal512_context ctx_shabal;
	sph_skein512_context ctx_skein;
	sph_hamsi512_context ctx_hamsi;

	HEFTY1(input, 80, hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, hash, 64);
	sph_luffa512_close(&ctx_luffa, hash);

	if (hash[0] & 0x8)
	{
		sph_fugue512_init(&ctx_fugue);
		sph_fugue512(&ctx_fugue, hash, 64);
		sph_fugue512_close(&ctx_fugue, hash);
	} else {
		sph_skein512_init(&ctx_skein);
		sph_skein512(&ctx_skein, hash, 64);
		sph_skein512_close(&ctx_skein, hash);
	}

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, hash, 64);
	sph_fugue512_close(&ctx_fugue, hash);

	if (hash[0] & 0x8)
	{
		sph_echo512_init(&ctx_echo);
		sph_echo512(&ctx_echo, hash, 64);
		sph_echo512_close(&ctx_echo, hash);
	} else {
		sph_luffa512_init(&ctx_luffa);
		sph_luffa512(&ctx_luffa, hash, 64);
		sph_luffa512_close(&ctx_luffa, hash);
	}

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash, 64);
	sph_shabal512_close(&ctx_shabal, hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, 64);
	sph_skein512_close(&ctx_skein, hash);

	if (hash[0] & 0x8)
	{
		sph_shabal512_init(&ctx_shabal);
		sph_shabal512(&ctx_shabal, hash, 64);
		sph_shabal512_close(&ctx_shabal, hash);
	} else {
		sph_whirlpool_init(&ctx_whirlpool);
		sph_whirlpool(&ctx_whirlpool, hash, 64);
		sph_whirlpool_close(&ctx_whirlpool, hash);
	}

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash, 64);
	sph_shabal512_close(&ctx_shabal, hash);

	if (hash[0] & 0x8)
	{
		sph_hamsi512_init(&ctx_hamsi);
		sph_hamsi512(&ctx_hamsi, hash, 64);
		sph_hamsi512_close(&ctx_hamsi, hash);
	} else {
		sph_luffa512_init(&ctx_luffa);
		sph_luffa512(&ctx_luffa, hash, 64);
		sph_luffa512_close(&ctx_luffa, hash);
	}

	memcpy(output, hash, 32);
}
