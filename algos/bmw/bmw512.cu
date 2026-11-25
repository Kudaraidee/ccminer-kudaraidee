/**
 * BMW512
 */
#include <algorithm> // before miner.h, whose min/max macros break it

extern "C" {
#include "sph/sph_bmw.h"
}
#include "miner.h"
#include "cuda/helper.cuh"
#include "cuda/selftest_gate.cuh"

#define NBN 2

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];
extern void bmw512_cpu_init(int thr_id, uint32_t threads);
extern void bmw512_final_setBlock_80(void *pdata);
void bmw512_cpu_hash_80_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_resNonce, const uint64_t target);
extern void bmw512_80_differential(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *acc);


extern "C" void bmw512_hash(void *state, const void *input) {
	sph_bmw512_context ctx_bmw;
	unsigned char hash[64];

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, input, 80);
	sph_bmw512_close(&ctx_bmw, hash);
	memcpy(state, hash, 32);
}

/* One launch through the shipping kernel: returns the lowest candidate
 * (absolute nonce) and the second-lowest in *second, UINT32_MAX for none. */
static uint32_t bmw512_scan_launch(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_res, uint32_t *h_res, uint32_t *second)
{
	*second = UINT32_MAX;
	if (cudaMemset(d_res, 0xff, NBN * sizeof(uint32_t)) != cudaSuccess)
		return UINT32_MAX;
	bmw512_cpu_hash_80_final(thr_id, threads, startNonce, d_res, target);
	if (cudaMemcpy(h_res, d_res, NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
		return UINT32_MAX;
	if (h_res[1] != UINT32_MAX) *second = startNonce + h_res[1];
	return (h_res[0] != UINT32_MAX) ? startNonce + h_res[0] : UINT32_MAX;
}

/* Init-time self-test of the 80-byte mining kernel, fails closed (bmw512_cpu_init()
 * already covers the 64-byte device library):
 *   KAT    host reference vs a digest from an independent implementation of the spec
 *   NEG    one flipped header bit must change the digest
 *   DIFF   GPU vs CPU over a nonce range through the mining arithmetic (bmw512_80_word3)
 *   NEGD   the same with one host word perturbed must fail
 *   REPORT the launcher returns the two lowest candidates (all nonces, and five in one block) */
static const uint32_t bmw512_kat_digest[8] = {
	0xD257DF6Du, 0x0C58061Bu, 0x5099D121u, 0x7A420563u,
	0xDDD1419Bu, 0xFFC47576u, 0x9F6CAB0Fu, 0x9472ED22u,
};
#define BMW512_KAT_NONCE 0x1234abcdu
#define BMW512_ST_TPB 128

/* BMW_SELFTEST_FAULT=<leg> (kat, neg, diff, negd, report, cuda, jobdiff) breaks that one
 * check on purpose, to prove the gate fires. Testing only. */
static bool bmw512_fault(int thr_id, const char *leg)
{
	const char *m = getenv("BMW_SELFTEST_FAULT");
	if (!m || strcmp(m, leg)) return false;
	static bool warned = false;
	if (!warned) {
		gpulog(LOG_WARNING, thr_id, "bmw512: SELF-TEST FAULT INJECTION ACTIVE (%s) -- for testing only", m);
		warned = true;
	}
	return true;
}

static void bmw512_selftest_header(uint32_t *ed, uint32_t nonce)
{
	for (int i = 0; i < 19; i++)
		be32enc(&ed[i], 0x01020304u + i);
	be32enc(&ed[19], nonce);
}

/* vhash[7]:vhash[6] = digest word 3, the word the device screens and accumulates. */
static uint64_t bmw512_word3(const uint32_t *vhash)
{
	return ((uint64_t) vhash[7] << 32) | vhash[6];
}

static void bmw512_cpu_acc(uint32_t *ed, uint32_t startNonce, uint32_t count, uint64_t *cpu, uint32_t perturb_at)
{
	uint32_t _ALIGN(64) vhash[8];
	cpu[0] = cpu[1] = 0;
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		be32enc(&ed[19], nonce);
		bmw512_hash(vhash, ed);
		uint64_t w = bmw512_word3(vhash);
		if (i == perturb_at)
			w ^= 1ull;
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t) nonce + 1ull);
	}
}

static bool bmw512_selftest_report(int thr_id, uint32_t *ed, uint32_t start, uint32_t count,
	uint32_t *d_res, uint32_t *h_res)
{
	uint32_t _ALIGN(64) vhash[8];
	uint64_t w[BMW512_ST_TPB], sorted[BMW512_ST_TPB];
	uint32_t sec, lo = UINT32_MAX, hi = UINT32_MAX;

	uint32_t first = bmw512_scan_launch(thr_id, count, start + (bmw512_fault(thr_id, "report") ? 1 : 0), UINT64_MAX,
		d_res, h_res, &sec);
	const bool sat_ok = (first == start && sec == start + 1);

	for (uint32_t i = 0; i < BMW512_ST_TPB; i++) {
		be32enc(&ed[19], start + i);
		bmw512_hash(vhash, ed);
		sorted[i] = w[i] = bmw512_word3(vhash);
	}
	std::sort(sorted, sorted + BMW512_ST_TPB);
	const uint64_t target = sorted[4];
	for (uint32_t i = 0; i < BMW512_ST_TPB; i++) {
		if (w[i] > target) continue;
		if (lo == UINT32_MAX) lo = start + i;
		else if (hi == UINT32_MAX) hi = start + i;
	}
	first = bmw512_scan_launch(thr_id, BMW512_ST_TPB, start, target, d_res, h_res, &sec);
	return sat_ok && first == lo && sec == hi;
}

static bool bmw512_80_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res)
{
	uint32_t _ALIGN(64) ed[20], dig[8];

	bmw512_selftest_header(ed, BMW512_KAT_NONCE);
	bmw512_hash(dig, ed);
	if (bmw512_fault(thr_id, "kat")) dig[0] ^= 1u;
	const bool kat_ok = (memcmp(dig, bmw512_kat_digest, sizeof(dig)) == 0);

	const uint32_t flip = bmw512_fault(thr_id, "neg") ? 0u : 0x00000001u;
	ed[0] ^= flip;
	bmw512_hash(dig, ed);
	const bool neg_ok = (memcmp(dig, bmw512_kat_digest, sizeof(dig)) != 0);
	ed[0] ^= flip;

	/* Uploads the test header; scanhash uploads the real one on every call. */
	bmw512_final_setBlock_80((void*) ed);

	const uint32_t start = 0x00010000u, count = 2048u + 77u; // not a whole number of blocks
	uint64_t gpu[2], cpu[2], cpu_bad[2];
	if (bmw512_fault(thr_id, "cuda"))
		gpu[0] = gpu[1] = 0; // as if the differential could not run
	else
		bmw512_80_differential(thr_id, count, start + (bmw512_fault(thr_id, "diff") ? 1 : 0), gpu);
	if (gpu[0] == 0 && gpu[1] == 0)
		selftest_cuda_fault();
	bmw512_cpu_acc(ed, start, count, cpu, UINT32_MAX);
	const bool diff_ok = (gpu[0] == cpu[0] && gpu[1] == cpu[1]);
	bmw512_cpu_acc(ed, start, count, cpu_bad, bmw512_fault(thr_id, "negd") ? UINT32_MAX : count / 2);
	const bool negd_ok = !(gpu[0] == cpu_bad[0] && gpu[1] == cpu_bad[1]);

	const bool rep_ok = bmw512_selftest_report(thr_id, ed, start, count, d_res, h_res);

	const bool passed = kat_ok && neg_ok && diff_ok && negd_ok && rep_ok;
	if (!passed) {
		gpulog(LOG_ERR, thr_id, "bmw512 self-test FAILED (kat %d neg %d diff %d negd %d report %d)",
			(int) kat_ok, (int) neg_ok, (int) diff_ok, (int) negd_ok, (int) rep_ok);
		if (!diff_ok)
			gpulog(LOG_ERR, thr_id, "  gpu %016llx / %016llx  cpu %016llx / %016llx",
				(unsigned long long) gpu[0], (unsigned long long) gpu[1],
				(unsigned long long) cpu[0], (unsigned long long) cpu[1]);
	}
	else
		gpulog(LOG_DEBUG, thr_id, "bmw512 self-test passed (kat, neg, diff %u nonces, negd, report)", count);
	return selftest_gate(thr_id, "bmw512", passed);
}

/* -D: GPU==CPU over a bounded range of the live job, once per job. */
static void bmw512_check_range(int thr_id, const uint32_t *endiandata, uint32_t startNonce, uint32_t count)
{
	uint32_t _ALIGN(64) ed[20];
	uint64_t gpu[2], cpu[2];

	memcpy(ed, endiandata, sizeof(ed));
	bmw512_80_differential(thr_id, count, startNonce, gpu);
	bmw512_cpu_acc(ed, startNonce, count, cpu, bmw512_fault(thr_id, "jobdiff") ? count / 2 : UINT32_MAX);
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


extern "C" int scanhash_bmw512(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done) {
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t _ALIGN(64) endiandata[20];

	if (opt_benchmark) ptarget[7] = 0x00ff;

	for (int k=0; k < 20; k++) be32enc(&endiandata[k], pdata[k]);

	uint32_t throughput =  cuda_default_throughput(thr_id, 1 << 28);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}

		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)));
		bmw512_cpu_init(thr_id, throughput);
		bmw512_80_selftest(thr_id, d_resNonce[thr_id], h_resNonce[thr_id]); // fail-closed

		init[thr_id] = true;
	}

	bmw512_final_setBlock_80((void*)endiandata);
	const uint64_t target = *(uint64_t*)&ptarget[6];
	*hashes_done = 0;

	if (opt_debug)
		bmw512_check_range(thr_id, endiandata, pdata[19], 2048);

	do {
		// clamp each launch to the range left, or the last one runs past max_nonce and wraps
		const uint32_t n = (pdata[19] < max_nonce) ? min(throughput, max_nonce - pdata[19]) : 0;
		if (n == 0) break;
		uint32_t second;
		work->nonces[0] = bmw512_scan_launch(thr_id, n, pdata[19], target, d_resNonce[thr_id], h_resNonce[thr_id], &second);
		work->nonces[1] = second;
		*hashes_done += n;

		if (work->nonces[0] != UINT32_MAX) {
			uint32_t _ALIGN(64) vhash[8];
			/* the two lowest are reported, so everything below the higher one is screened */
			const uint32_t scanned = (second != UINT32_MAX) ? second : work->nonces[0];

			work->valid_nonces = 0;
			be32enc(&endiandata[19], work->nonces[0]);
			bmw512_hash(vhash, endiandata);
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
				bmw512_hash(vhash, endiandata);
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

		if ((uint64_t)n + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += n;

	} while (!work_restart[thr_id].restart);
	return 0;
}


extern "C" void free_bmw512(int thr_id) {
	if (!init[thr_id]) return;

	cudaSetDevice(device_map[thr_id]);

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
