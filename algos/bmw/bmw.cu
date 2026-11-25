/**
 * bmw-256 MDT
 * tpruvot - 2015
 */
#include <algorithm> // before miner.h, whose min/max macros break it

extern "C" {
#include "sph/sph_bmw.h"
}

#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/selftest_gate.cuh"

#define NBN 2

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern void bmw256_setBlock_80(int thr_id, void *pdata);
extern void bmw256_cpu_hash_80_final(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_resNonce, const uint64_t target);
extern void bmw256_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc);

// CPU Hash
extern "C" void bmw_hash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_bmw256_context ctx;

	sph_bmw256_init(&ctx);
	sph_bmw256(&ctx, input, 80);
	sph_bmw256_close(&ctx, (void*) hash);

	memcpy(state, hash, 32);
}

/* One launch through the shipping kernel: returns the lowest candidate
 * (absolute nonce) and the second-lowest in *second, UINT32_MAX for none. */
static uint32_t bmw_scan_launch(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_res, uint32_t *h_res, uint32_t *second)
{
	*second = UINT32_MAX;
	if (cudaMemset(d_res, 0xff, NBN * sizeof(uint32_t)) != cudaSuccess)
		return UINT32_MAX;
	bmw256_cpu_hash_80_final(thr_id, threads, startNonce, d_res, target);
	if (cudaMemcpy(h_res, d_res, NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
		return UINT32_MAX;
	if (h_res[1] != UINT32_MAX) *second = startNonce + h_res[1];
	return (h_res[0] != UINT32_MAX) ? startNonce + h_res[0] : UINT32_MAX;
}

/* Init-time self-test, fails closed:
 *   KAT    host reference vs a digest from an independent implementation of the spec
 *   NEG    one flipped header bit must change the digest
 *   DIFF   GPU vs CPU over a nonce range through the mining arithmetic (bmw256_80_high64)
 *   NEGD   the same with one host word perturbed must fail
 *   REPORT the launcher returns the two lowest candidates (all nonces, and five in one block) */
static const uint32_t bmw_kat_digest[8] = {
	0x4632BD8Bu, 0x15275D91u, 0x20E4AF72u, 0xFD18E4A1u,
	0x3BD4C178u, 0x6499EDFFu, 0x5210DCB6u, 0xAFE23EA8u,
};
#define BMW_KAT_NONCE 0x1234abcdu
#define BMW_ST_TPB 256

/* BMW_SELFTEST_FAULT=<leg> (kat, neg, diff, negd, report, cuda, jobdiff) breaks that one
 * check on purpose, to prove the gate fires. Testing only. */
static bool bmw_fault(int thr_id, const char *leg)
{
	const char *m = getenv("BMW_SELFTEST_FAULT");
	if (!m || strcmp(m, leg)) return false;
	static bool warned = false;
	if (!warned) {
		gpulog(LOG_WARNING, thr_id, "bmw: SELF-TEST FAULT INJECTION ACTIVE (%s) -- for testing only", m);
		warned = true;
	}
	return true;
}

static void bmw_selftest_header(uint32_t *ed, uint32_t nonce)
{
	for (int i = 0; i < 19; i++)
		be32enc(&ed[i], 0x01020304u + i);
	be32enc(&ed[19], nonce);
}

/* vhash[7]:vhash[6], the word pair the device screens and accumulates. */
static uint64_t bmw_high64(const uint32_t *vhash)
{
	return ((uint64_t) vhash[7] << 32) | vhash[6];
}

static void bmw_cpu_acc(uint32_t *ed, uint32_t startNonce, uint32_t count, uint64_t *cpu, uint32_t perturb_at)
{
	uint32_t _ALIGN(64) vhash[8];
	cpu[0] = cpu[1] = 0;
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		be32enc(&ed[19], nonce);
		bmw_hash(vhash, ed);
		uint64_t w = bmw_high64(vhash);
		if (i == perturb_at)
			w ^= 1ull;
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t) nonce + 1ull);
	}
}

static bool bmw_selftest_report(int thr_id, uint32_t *ed, uint32_t start, uint32_t count,
	uint32_t *d_res, uint32_t *h_res)
{
	uint32_t _ALIGN(64) vhash[8];
	uint64_t w[BMW_ST_TPB], sorted[BMW_ST_TPB];
	uint32_t sec, lo = UINT32_MAX, hi = UINT32_MAX;

	// every nonce a candidate: the two lowest are the first two
	uint32_t first = bmw_scan_launch(thr_id, count, start + (bmw_fault(thr_id, "report") ? 1 : 0), UINT64_MAX,
		d_res, h_res, &sec);
	const bool sat_ok = (first == start && sec == start + 1);

	// exactly five candidates inside one block, the lowest not first in thread order
	for (uint32_t i = 0; i < BMW_ST_TPB; i++) {
		be32enc(&ed[19], start + i);
		bmw_hash(vhash, ed);
		sorted[i] = w[i] = bmw_high64(vhash);
	}
	std::sort(sorted, sorted + BMW_ST_TPB);
	const uint64_t target = sorted[4];
	for (uint32_t i = 0; i < BMW_ST_TPB; i++) {
		if (w[i] > target) continue;
		if (lo == UINT32_MAX) lo = start + i;
		else if (hi == UINT32_MAX) hi = start + i;
	}
	first = bmw_scan_launch(thr_id, BMW_ST_TPB, start, target, d_res, h_res, &sec);
	return sat_ok && first == lo && sec == hi;
}

static bool bmw_device_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res)
{
	uint32_t _ALIGN(64) ed[20], dig[8];

	bmw_selftest_header(ed, BMW_KAT_NONCE);
	bmw_hash(dig, ed);
	if (bmw_fault(thr_id, "kat")) dig[0] ^= 1u;
	const bool kat_ok = (memcmp(dig, bmw_kat_digest, sizeof(dig)) == 0);

	const uint32_t flip = bmw_fault(thr_id, "neg") ? 0u : 0x00000001u;
	ed[0] ^= flip;
	bmw_hash(dig, ed);
	const bool neg_ok = (memcmp(dig, bmw_kat_digest, sizeof(dig)) != 0);
	ed[0] ^= flip;

	/* Uploads the test header; scanhash uploads the real one on every call. */
	bmw256_setBlock_80(thr_id, (void*) ed);

	const uint32_t start = 0x00010000u, count = 2048u + 77u; // not a whole number of blocks
	uint64_t gpu[2], cpu[2], cpu_bad[2];
	if (bmw_fault(thr_id, "cuda"))
		gpu[0] = gpu[1] = 0; // as if the differential could not run
	else
		bmw256_differential(thr_id, count, start + (bmw_fault(thr_id, "diff") ? 1 : 0), gpu);
	if (gpu[0] == 0 && gpu[1] == 0)
		selftest_cuda_fault();
	bmw_cpu_acc(ed, start, count, cpu, UINT32_MAX);
	const bool diff_ok = (gpu[0] == cpu[0] && gpu[1] == cpu[1]);
	bmw_cpu_acc(ed, start, count, cpu_bad, bmw_fault(thr_id, "negd") ? UINT32_MAX : count / 2);
	const bool negd_ok = !(gpu[0] == cpu_bad[0] && gpu[1] == cpu_bad[1]);

	const bool rep_ok = bmw_selftest_report(thr_id, ed, start, count, d_res, h_res);

	const bool passed = kat_ok && neg_ok && diff_ok && negd_ok && rep_ok;
	if (!passed) {
		gpulog(LOG_ERR, thr_id, "bmw self-test FAILED (kat %d neg %d diff %d negd %d report %d)",
			(int) kat_ok, (int) neg_ok, (int) diff_ok, (int) negd_ok, (int) rep_ok);
		if (!diff_ok)
			gpulog(LOG_ERR, thr_id, "  gpu %016llx / %016llx  cpu %016llx / %016llx",
				(unsigned long long) gpu[0], (unsigned long long) gpu[1],
				(unsigned long long) cpu[0], (unsigned long long) cpu[1]);
	}
	else
		gpulog(LOG_DEBUG, thr_id, "bmw self-test passed (kat, neg, diff %u nonces, negd, report)", count);
	return selftest_gate(thr_id, "bmw", passed);
}

/* -D: GPU==CPU over a bounded range of the live job, once per job. */
static void bmw_check_range(int thr_id, const uint32_t *endiandata, uint32_t startNonce, uint32_t count)
{
	uint32_t _ALIGN(64) ed[20];
	uint64_t gpu[2], cpu[2];

	memcpy(ed, endiandata, sizeof(ed));
	bmw256_differential(thr_id, count, startNonce, gpu);
	bmw_cpu_acc(ed, startNonce, count, cpu, bmw_fault(thr_id, "jobdiff") ? count / 2 : UINT32_MAX);
	if (gpu[0] != cpu[0] || gpu[1] != cpu[1])
		gpulog(LOG_ERR, thr_id, "differential MISMATCH over %u nonces from %08x:"
			" gpu %016llx / %016llx  cpu %016llx / %016llx", count, startNonce,
			(unsigned long long) gpu[0], (unsigned long long) gpu[1],
			(unsigned long long) cpu[0], (unsigned long long) cpu[1]);
	else
		gpulog(LOG_DEBUG, thr_id, "differential OK: %u nonces from %08x (xor %016llx)",
			count, startNonce, (unsigned long long) gpu[0]);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_bmw(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	/* 2^27 keeps the GPU ~99% busy (tuned on sm_61 and sm_86) */
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 27);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x0005;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
                        // reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)));
		bmw_device_selftest(thr_id, d_resNonce[thr_id], h_resNonce[thr_id]); // fail-closed

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
	}

	cudaGetLastError();
	bmw256_setBlock_80(thr_id, (void*)endiandata);
	const uint64_t target = ((uint64_t*) ptarget)[3];

	if (opt_debug)
		bmw_check_range(thr_id, endiandata, pdata[19], 2048);

	do {
		// clamp each launch to the range left; a reject-continue can step past max_nonce
		const uint32_t n = (pdata[19] < max_nonce) ? min(throughput, max_nonce - pdata[19]) : 0;
		if (n == 0) break;
		uint32_t second;
		work->nonces[0] = bmw_scan_launch(thr_id, n, pdata[19], target, d_resNonce[thr_id], h_resNonce[thr_id], &second);
		work->nonces[1] = second;

		*hashes_done = pdata[19] - first_nonce + n;

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			/* the two lowest are reported, so everything below the higher one is screened */
			const uint32_t scanned = (second != UINT32_MAX) ? second : work->nonces[0];

			work->valid_nonces = 0;
			be32enc(&endiandata[19], work->nonces[0]);
			bmw_hash(vhash, endiandata);
			if (fulltest(vhash, ptarget)) {
				work_set_target_ratio(work, vhash);
				work->valid_nonces++;
			} else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
			}
			if (second != UINT32_MAX) {
				be32enc(&endiandata[19], second);
				bmw_hash(vhash, endiandata);
				if (fulltest(vhash, ptarget)) {
					if (!work->valid_nonces) {
						work->nonces[0] = second;
						work_set_target_ratio(work, vhash);
					} else {
						bn_set_target_ratio(work, vhash, 1);
					}
					work->valid_nonces++;
				} else {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", second);
				}
			}
			pdata[19] = scanned + 1;
			if (work->valid_nonces)
				return work->valid_nonces;
			continue;
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

// cleanup
extern "C" void free_bmw(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
