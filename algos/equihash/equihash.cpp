/**
 * Equihash solver interface for ccminer (compatible with linux and windows)
 * Solver taken from nheqminer, by djeZo (and NiceHash)
 * tpruvot - 2017 (GPL v3)
 */
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include <stdexcept>
#include <vector>

#include <sph/sph_sha2.h>

#include "eqcuda.hpp"
#include "equihash.h" // equi_verify()

#include <miner.h>

// All solutions (BLOCK_HEADER_LEN + SOLSIZE_LEN + SOL_LEN) sha256d should be under the target
extern "C" void equi_hash(const void* input, void* output, int len)
{
	uint8_t _ALIGN(64) hash0[32], hash1[32];

	sph_sha256_context ctx_sha256;

	sph_sha256_init(&ctx_sha256);
	sph_sha256(&ctx_sha256, input, len);
	sph_sha256_close(&ctx_sha256, hash0);
	sph_sha256(&ctx_sha256, hash0, 32);
	sph_sha256_close(&ctx_sha256, hash1);

	memcpy(output, hash1, 32);
}

// input here is 140 for the header and 1344 for the solution (equi.cpp)
extern "C" int equi_verify_sol(void * const hdr, void * const sol)
{
	bool res = equi_verify((uint8_t*) hdr, (uint8_t*) sol);

	//applog_hex((void*)hdr, 140);
	//applog_hex((void*)sol, 1344);

	return res ? 1 : 0;
}

#include "cuda/helper.cuh"

//#define EQNONCE_OFFSET 30 /* 27:34 */
#define NONCE_OFT EQNONCE_OFFSET

static bool init[MAX_GPUS] = { 0 };
static int valid_sols[MAX_GPUS] = { 0 };
static uint8_t _ALIGN(64) data_sols[MAX_GPUS][MAXREALSOLS][1536] = { 0 }; // 140+3+1344 required
static eq_cuda_context_interface* solvers[MAX_GPUS] = { NULL };

// --- equihash variant (n,k)+personalization dispatch ------------------------
// The djeZo solver above is 200/9-only; equi24b handles 144/5 and 192/7.
// Default is Zcash 200/9 ("ZcashPoW"). The (n,k) variant is set
// ONLY by the `-a` algo parameter (equihash / equihash144) -> eq_set_variant_144()
// -- it fixes the CUDA solver, so it never changes at runtime. The pool's
// mining.notify may then set the personalization (eq_set_variant_params: personal
// only; its (n,k) is validate-only). Solution size + personalization are
// parametrized via eq_solsize()/eq_personal.
#include "cuda_equi24b.h"
#include "equi_verify.h"
#include "equi_pack.h"    // eq_minimal_from_indices: indices -> submitted bytes
static int   eq_wn = 200, eq_wk = 9;
static char  eq_personal[16] = "ZcashPoW";


// equi24b (cuda_equi24b.cu) is the only solver for the DIGITBITS=24 variants.
// It retains every layer, so its arena is large and variant-dependent; a card
// that cannot hold it cannot mine the variant.
//
// Every solution is re-verified on the host before submit, so a solver defect
// costs a local reject rather than a bad share.
static void* equi24b_ctx[MAX_GPUS] = { NULL };
// Queued solves (oldest first) and the headers they were launched for. They
// survive a return from scanhash and are kept if the next call's header differs
// at most in the nonce words.
static void*    equi24b_pipe[MAX_GPUS] = { NULL };
static uint32_t eq_pend_hdr[MAX_GPUS][2][35];
// The personalization each queued solve was launched with: the pool can switch it
// between notifies (zpool moves equihash144 between coins), and it is not part of
// the header, so a solve is verified with its own and the queue drops on a change.
static char     eq_pend_pers[MAX_GPUS][2][16];
static int      eq_pend_n[MAX_GPUS] = { 0 };

// Throughput meter for the 144/5 and 192/7 path, rate-limited to one line per 5 s.
// sol/nonce (expect ~2) is reported too: dropped pairs lower it, Sol/s cannot show that.
static uint32_t eq_m_sols[MAX_GPUS]   = { 0 };
static uint32_t eq_m_nonces[MAX_GPUS] = { 0 };
static uint32_t eq_m_bad[MAX_GPUS]    = { 0 };  // solutions that failed the host re-verify
static uint32_t eq_m_clamped[MAX_GPUS] = { 0 }; // solutions found but discarded by the MAXSOLS cap
static time_t   eq_m_since[MAX_GPUS]  = { 0 };

static inline int eq_cbitlen()   { return eq_wn / (eq_wk + 1); }               // 200/9: 20, 144/5 + 192/7: 24
static inline int eq_proofsize() { return 1 << eq_wk; }                        // 512 / 32 / 128
static inline int eq_solsize()   { return eq_proofsize() * (eq_cbitlen() + 1) / 8; } // 1344 / 100 / 400

// Bitcoin CompactSize prefix length for the solution byte count.
static inline int eq_solprefix() { int s = eq_solsize(); return s < 253 ? 1 : (s <= 0xffff ? 3 : 5); }

// Shared accessors for the stratum layer (equi-stratum.cpp) -- the number of
// bytes stored in work->extra to hex-encode on submit: compactSize + solution
// (1347 for 200/9, 101 for 144/5).
extern "C" int eq_variant_storelen() { return eq_solprefix() + eq_solsize(); }
extern "C" int eq_variant_wk()       { return eq_wk; }

// Select the 144/5 (BitcoinZ) variant explicitly (from the -a alias).
extern "C" void eq_set_variant_144()
{
	eq_wn = 144; eq_wk = 5;
	snprintf(eq_personal, sizeof(eq_personal), "%s", "BitcoinZ");
	applog(LOG_NOTICE, "equihash variant %d/%d personal=\"%s\" (sol %d bytes)",
	       eq_wn, eq_wk, eq_personal, eq_solsize());
}

// Select the 192/7 (ZeroClassic-class) variant explicitly (from the -a alias).
// Same solver as 144/5 -- both are DIGITBITS=24 -- differing only in the round
// count and PROOFSIZE, which are compile-time in the solver TU pair.
extern "C" void eq_set_variant_192()
{
	eq_wn = 192; eq_wk = 7;
	snprintf(eq_personal, sizeof(eq_personal), "%s", "ZcashPoW");
	applog(LOG_NOTICE, "equihash variant %d/%d personal=\"%s\" (sol %d bytes)",
	       eq_wn, eq_wk, eq_personal, eq_solsize());
}

// Apply the equihash params the POOL advertises in mining.notify (zpool /
// cpuminer-opt convention: trailing "<n>_<k>" and 8-char personalization).
//
// (n,k) is FIXED by the -a algo parameter -- it defines the CUDA solver/kernel,
// so we never switch it at runtime (that would force a kernel unload/reload on a
// job change). The pool-advertised (n,k) is therefore validation-only: warn on
// mismatch (miner pointed at the wrong-variant pool) and ignore it. Only the
// personalization (a runtime BLAKE2b param, no kernel impact) is adopted from
// the pool -- this is what lets a 144/5 pool select e.g. "ZcashPoW". Logs only on
// change to avoid per-notify spam.
extern "C" void eq_set_variant_params(int wn, int wk, const char* personal)
{
	if (wn > 0 && wk > 0 && (wn != eq_wn || wk != eq_wk)) {
		static bool warned = false;
		if (!warned) {
			applog(LOG_WARNING, "pool advertises equihash %d/%d but miner is %d/%d "
			       "(fixed by -a); ignoring pool (n,k) -- use the matching -a algo",
			       wn, wk, eq_wn, eq_wk);
			warned = true;
		}
		return; // wrong variant; don't adopt this pool's personalization either
	}
	if (personal && *personal) {
		char pers[16];
		snprintf(pers, sizeof(pers), "%.8s", personal);
		if (strncmp(pers, eq_personal, 8) != 0) {
			snprintf(eq_personal, sizeof(eq_personal), "%s", pers);
			applog(LOG_NOTICE, "equihash personalization=\"%s\" (from pool)", eq_personal);
		}
	}
}


// solver callbacks
static void cb_solution(int thr_id, const std::vector<uint32_t>& solutions, size_t cbitlen, const unsigned char *compressed_sol)
{
	std::vector<unsigned char> nSolution;
	if (!compressed_sol) {
		nSolution = eq_minimal_from_indices(solutions, cbitlen);
	} else {
		gpulog(LOG_INFO, thr_id, "compressed_sol");
		nSolution = std::vector<unsigned char>(1344);
		for (size_t i = 0; i < cbitlen; i++)
			nSolution[i] = compressed_sol[i];
	}
	int nsol = valid_sols[thr_id];
	if (nsol < 0) nsol = 0;
	if(nSolution.size() == 1344) {
		// todo, only store solution data here...
		le32enc(&data_sols[thr_id][nsol][140], 0x000540fd); // sol sz header
		memcpy(&data_sols[thr_id][nsol][143], nSolution.data(), 1344);
		valid_sols[thr_id] = nsol + 1;
	}
}
static void cb_hashdone(int thr_id) {
	if (!valid_sols[thr_id]) valid_sols[thr_id] = -1;
}
// Return to the miner loop at least once a second so it re-checks the time and
// share limits (max_nonce cannot bound the scan: it comes from a rate counted in
// solutions). The nonce carries forward through pdata.
static bool eq_slice_done(const struct timeval *start)
{
	struct timeval now, d;
	gettimeofday(&now, NULL);
	timeval_subtract(&d, &now, (struct timeval *) start);
	return d.tv_sec >= 1;
}

static bool cb_cancel(int thr_id) {
	if (work_restart[thr_id].restart)
		valid_sols[thr_id] = -1;
	return work_restart[thr_id].restart;
}

// --- DIGITBITS=24 scan path (144/5 and 192/7, equi24b solver) ---------------
// Proof buffers are sized for the largest proof (128 indices, 192/7); the emit
// callback takes the size from the solver.
#define EQ_MAX_PROOF 128
static uint32_t tromp_idx[MAX_GPUS][MAXREALSOLS][EQ_MAX_PROOF];
static int      tromp_ns[MAX_GPUS];

static void tromp_emit(void* ud, const uint32_t* idx, uint32_t proofsize)
{
	int thr = *(int*)ud;
	int n = tromp_ns[thr];
	if (n < MAXREALSOLS && proofsize <= EQ_MAX_PROOF) {
		memcpy(tromp_idx[thr][n], idx, proofsize * sizeof(uint32_t));
		tromp_ns[thr] = n + 1;
	}
}

static bool eq24_launch(int thr_id, const uint32_t *hdr)
{
	const int n = eq_pend_n[thr_id];
	// one snapshot for the launch and the queue: the stratum thread may change it
	char pers[16];
	memcpy(pers, eq_personal, sizeof(pers));
	pers[sizeof(pers) - 1] = '\0';
	const int rc = (eq_wk == 7)
		? equi24b_192_launch(equi24b_ctx[thr_id], equi24b_pipe[thr_id], (const char*) hdr, pers)
		: equi24b_144_launch(equi24b_ctx[thr_id], equi24b_pipe[thr_id], (const char*) hdr, pers);
	if (rc) {
		gpulog(LOG_WARNING, thr_id, "equi24b launch failed (%d)", rc);
		return false;
	}
	memcpy(eq_pend_hdr[thr_id][n], hdr, 140);
	memcpy(eq_pend_pers[thr_id][n], pers, sizeof(pers));
	eq_pend_n[thr_id] = n + 1;
	return true;
}

static void eq24_reset(int thr_id)
{
	if (eq_wk == 7) equi24b_192_pipe_reset(equi24b_pipe[thr_id]);
	else            equi24b_144_pipe_reset(equi24b_pipe[thr_id]);
	eq_pend_n[thr_id] = 0;
}

static int scanhash_equihash_dig24(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[35];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t nonce_increment = (rand() & 0xFF) | 1; // odd step (never a 0 re-grind)
	struct timeval tv_start, tv_end, diff;
	double secs;
	uint32_t soluce_count = 0;
	const int cbl   = eq_cbitlen();  // 24
	const int solsz = eq_solsize();  // 100 (144/5) or 400 (192/7)

	if (opt_benchmark)
		ptarget[7] = 0xfffff;

	// the CUDA current device is per host thread and defaults to 0; select ours
	// before equi24b allocates its arena, or every thread lands on GPU 0
	const cudaError_t dev_err = cudaSetDevice(device_map[thr_id]);
	if (dev_err != cudaSuccess) {
		gpulog(LOG_ERR, thr_id, "equihash%d/%d: cannot select CUDA device %d: %s",
		       eq_wn, eq_wk, device_map[thr_id], cudaGetErrorString(dev_err));
		proper_exit(EXIT_CODE_CUDA_ERROR);
		return -1;
	}

	if (!init[thr_id]) {
		// equi24b is the only solver for these variants: a card that cannot hold
		// the arena cannot mine them, and says so.
		const size_t need = (eq_wk == 7) ? equi24b_192_arena_needed()
		                                 : equi24b_144_arena_needed();
		equi24b_ctx[thr_id] = (eq_wk == 7) ? equi24b_192_init() : equi24b_144_init();
		equi24b_pipe[thr_id] = (eq_wk == 7) ? equi24b_192_pipe_new() : equi24b_144_pipe_new();
		if (!equi24b_ctx[thr_id] || !equi24b_pipe[thr_id]) {
			gpulog(LOG_ERR, thr_id, "equihash%d/%d needs a %.2f GiB arena and it could "
			       "not be allocated", eq_wn, eq_wk, (double)need / (1 << 30));
			proper_exit(EXIT_CODE_CUDA_ERROR);
			return -1;
		}
		gpulog(LOG_INFO, thr_id, "equihash%d/%d: equi24b solver (%.2f GiB arena)",
		       eq_wn, eq_wk, (double)need / (1 << 30));

		gpus_intensity[thr_id] = 8192;
		api_set_throughput(thr_id, gpus_intensity[thr_id]);
		cuda_get_arch(thr_id);
		init[thr_id] = true;
	}

	gettimeofday(&tv_start, NULL);
	memcpy(endiandata, pdata, 140);
	work->valid_nonces = 0;

	// Queued solves carry over when only the two nonce words the miner loop steps
	// between calls differ (data[30], data[31] with the GPU id in its top byte):
	// they continue their own nonce sequence, and a reset starts a fresh one.
	if (eq_pend_n[thr_id]) {
		const uint32_t *q = eq_pend_hdr[thr_id][0];
		if (memcmp(q, endiandata, NONCE_OFT * 4) ||
		    memcmp(q + NONCE_OFT + 2, endiandata + NONCE_OFT + 2, 140 - (NONCE_OFT + 2) * 4) ||
		    strncmp(eq_pend_pers[thr_id][0], eq_personal, 8))
			eq24_reset(thr_id);
	}
	if (!eq_pend_n[thr_id] && !eq24_launch(thr_id, endiandata))
		goto out;

	do {
		// queue the next nonce first, so the GPU works while this one is checked
		if (eq_pend_n[thr_id] == 1 && !work_restart[thr_id].restart) {
			uint32_t next[35];
			memcpy(next, eq_pend_hdr[thr_id][0], 140);
			next[NONCE_OFT] += nonce_increment;
			eq24_launch(thr_id, next);
		}
		memcpy(endiandata, eq_pend_hdr[thr_id][0], 140);
		char pers[16];
		memcpy(pers, eq_pend_pers[thr_id][0], sizeof(pers));
		memcpy(eq_pend_hdr[thr_id][0], eq_pend_hdr[thr_id][1], 140);
		memcpy(eq_pend_pers[thr_id][0], eq_pend_pers[thr_id][1], sizeof(pers));
		eq_pend_n[thr_id]--;

		tromp_ns[thr_id] = 0;
		unsigned dr[4] = { 0, 0, 0, 0 };
		int nsol;
		nsol = (eq_wk == 7)
			? equi24b_192_collect(equi24b_ctx[thr_id], equi24b_pipe[thr_id], tromp_emit, &thr_id, dr)
			: equi24b_144_collect(equi24b_ctx[thr_id], equi24b_pipe[thr_id], tromp_emit, &thr_id, dr);
		soluce_count += (nsol > 0 ? nsol : 0);
		eq_m_sols[thr_id] += (uint32_t) (nsol > 0 ? nsol : 0); // every solver solution, not just submitted ones
		*hashes_done = soluce_count;

		// Candidates past the device cap are discarded in arrival order (real
		// solutions included), and the heap counters report dropped items.
		unsigned clamped = 0;
		{
			clamped = dr[3];
			// Scatter overflow is the expected bucket-fill tail (negligible loss);
			// staging overflow means the shared-memory sizing is wrong.
			if (dr[1] || dr[2])
				gpulog(LOG_WARNING, thr_id, "equi24b heap overflow: scatter %u, "
				       "round staging %u, final staging %u -- solutions are being LOST",
				       dr[0], dr[1], dr[2]);
			else if (dr[0])
				gpulog(LOG_DEBUG, thr_id, "equi24b: %u item(s) over bucket capacity", dr[0]);
		}
		if (clamped) {
			eq_m_clamped[thr_id] += clamped;
			gpulog(LOG_WARNING, thr_id, "solution buffer overflow: %u candidate(s) discarded "
			       "this solve, %u total -- solutions are being LOST", clamped, eq_m_clamped[thr_id]);
		}

		if (tromp_ns[thr_id] > 0) {
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			uint8_t  _ALIGN(64) full_data[140 + 3 + 1344] = { 0 };   // 1344 >= 400, fits both
			uint8_t* sol_data = &full_data[140];

			for (int s = 0; s < tromp_ns[thr_id]; s++) {
				std::vector<u32> idx(tromp_idx[thr_id][s], tromp_idx[thr_id][s] + eq_proofsize());
				std::vector<unsigned char> minimal = eq_minimal_from_indices(idx, cbl); // solsz bytes

				memcpy(full_data, endiandata, 140);
				// compactSize: one byte below 253 (144/5: 100 B), else 0xfd + u16 LE
				// (192/7: 400 B).
				const int pfx = eq_solprefix();
				if (pfx == 1) {
					sol_data[0] = (uint8_t) solsz;
				} else {
					sol_data[0] = 0xfd;
					sol_data[1] = (uint8_t)(solsz & 0xff);
					sol_data[2] = (uint8_t)((solsz >> 8) & 0xff);
				}
				memcpy(&sol_data[pfx], minimal.data(), solsz);
				equi_hash(full_data, vhash, 140 + pfx + solsz);

				if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
					// host re-verify with the verifier of THIS variant
					int rc = (eq_wk == 7)
						? eq_verify_192((const char*) endiandata, pers,
						                tromp_idx[thr_id][s])
						: eq_verify_144((const char*) endiandata, pers,
						                tromp_idx[thr_id][s]);
					if (rc != 0) {
						// never drop silently: the only signal of a wrong GPU hash
						eq_m_bad[thr_id]++;
						gpulog(LOG_WARNING, thr_id, "solution failed host re-verify (rc=%d), %u so far",
						       rc, eq_m_bad[thr_id]);
					} else if (work->valid_nonces < MAX_NONCES) {
						work->valid_nonces++;
						memcpy(work->data, endiandata, 140);
						equi_store_work_solution(work, vhash, sol_data);
						work->nonces[work->valid_nonces - 1] = endiandata[NONCE_OFT];
						pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;
						goto out;
					}
				}
				if (work->valid_nonces == MAX_NONCES) goto out;
			}
			if (work->valid_nonces) goto out;
		}

		// rate-limited: per-solve I/O would skew the measurement
		eq_m_nonces[thr_id]++;
		{
			const time_t now = time(NULL);
			if (!eq_m_since[thr_id]) {
				eq_m_since[thr_id] = now;
			} else if (now - eq_m_since[thr_id] >= 5) {
				const double el = (double) (now - eq_m_since[thr_id]);
				gpulog(LOG_INFO, thr_id, "equihash%d/%d: %.2f Sol/s, %.2f sol/nonce (%u nonces/%.0fs)%s%s",
				       eq_wn, eq_wk, eq_m_sols[thr_id] / el,
				       eq_m_nonces[thr_id] ? (double) eq_m_sols[thr_id] / eq_m_nonces[thr_id] : 0.0,
				       eq_m_nonces[thr_id], el,
				       eq_m_bad[thr_id] ? " *** re-verify failures, see above ***" : "",
				       eq_m_clamped[thr_id] ? " *** solutions discarded by MAXSOLS, see above ***" : "");
				eq_m_sols[thr_id] = eq_m_nonces[thr_id] = 0;
				eq_m_since[thr_id] = now;
			}
		}

		if (eq_slice_done(&tv_start)) break;
	} while (!work_restart[thr_id].restart);

out:
	gettimeofday(&tv_end, NULL);
	timeval_subtract(&diff, &tv_end, &tv_start);
	secs = (1.0 * diff.tv_sec) + (0.000001 * diff.tv_usec);
	gpulog(LOG_DEBUG, thr_id, "%d solutions in %.2f s (%.2f Sol/s)",
	       soluce_count, secs, secs > 0 ? soluce_count / secs : 0.0);
	*hashes_done = soluce_count;
	// resume at the queued nonce, so the next call collects that solve
	pdata[NONCE_OFT] = eq_pend_n[thr_id] ? eq_pend_hdr[thr_id][0][NONCE_OFT]
	                                     : endiandata[NONCE_OFT] + nonce_increment;
	return work->valid_nonces;
}

extern "C" int scanhash_equihash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	if (eq_wk == 5 || eq_wk == 7)      // both are DIGITBITS=24
		return scanhash_equihash_dig24(thr_id, work, max_nonce, hashes_done);

	uint32_t _ALIGN(64) endiandata[35];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[NONCE_OFT];
	uint32_t nonce_increment = (rand() & 0xFF) | 1; // nonce randomizer; force odd so the step is never 0 (a 0 step re-grinds the same nonce until restart)
	struct timeval tv_start, tv_end, diff;
	double secs, solps;
	uint32_t soluce_count = 0;

	if (opt_benchmark)
		ptarget[7] = 0xfffff;

	if (!init[thr_id]) {
		try {
			int mode = 1;
			switch (mode) {
			case 1:
				solvers[thr_id] = new eq_cuda_context<CONFIG_MODE_1>(thr_id, device_map[thr_id]);
				break;
			default:
				proper_exit(EXIT_CODE_SW_INIT_ERROR);
				return -1;
			}
			size_t memSz = solvers[thr_id]->equi_mem_sz / (1024*1024);
			gpus_intensity[thr_id] = (uint32_t) solvers[thr_id]->throughput;
			api_set_throughput(thr_id, gpus_intensity[thr_id]);
			gpulog(LOG_DEBUG, thr_id, "Allocated %u MB of context memory", (u32) memSz);
			cuda_get_arch(thr_id);
			init[thr_id] = true;
		} catch (const std::exception & e) {
			CUDA_LOG_ERROR();
			gpulog(LOG_ERR, thr_id, "init: %s", e.what());
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}
	}

	gettimeofday(&tv_start, NULL);
	memcpy(endiandata, pdata, 140);
	work->valid_nonces = 0;

	do {

		try {

			valid_sols[thr_id] = 0;
			solvers[thr_id]->solve(
				(const char *) endiandata, (unsigned int) (140 - 32),
				(const char *) &endiandata[27], (unsigned int) 32,
				&cb_cancel, &cb_solution, &cb_hashdone
			);

			*hashes_done = soluce_count;

			// solutions past MAXREALSOLS are dropped; report it (expected ~2 per
			// solve against a cap of 9)
			if (solvers[thr_id]->sols_overflow) {
				eq_m_clamped[thr_id] += solvers[thr_id]->sols_overflow;
				gpulog(LOG_WARNING, thr_id, "solution buffer overflow: %u solution(s) discarded "
				       "this solve, %u total -- raise MAXREALSOLS",
				       solvers[thr_id]->sols_overflow, eq_m_clamped[thr_id]);
			}

		} catch (const std::exception & e) {
			gpulog(LOG_WARNING, thr_id, "solver: %s", e.what());
			free_equihash(thr_id);
			sleep(1);
			return -1;
		}

		if (valid_sols[thr_id] > 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			uint8_t _ALIGN(64) full_data[140+3+1344] = { 0 };
			uint8_t* sol_data = &full_data[140];

			soluce_count += valid_sols[thr_id];

			for (int nsol=0; nsol < valid_sols[thr_id]; nsol++)
			{
				memcpy(full_data, endiandata, 140);
				memcpy(sol_data, &data_sols[thr_id][nsol][140], 1347);
				equi_hash(full_data, vhash, 140+3+1344);

				if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
				{
					bool valid = equi_verify_sol(endiandata, &sol_data[3]);
					if (valid && work->valid_nonces < MAX_NONCES) {
						work->valid_nonces++;
						memcpy(work->data, endiandata, 140);
						equi_store_work_solution(work, vhash, sol_data);
						work->nonces[work->valid_nonces-1] = endiandata[NONCE_OFT];
						pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;
						//applog_hex(vhash, 32);
						//applog_hex(&work->data[27], 32);
						goto out; // second solution storage not handled..
					}
				}
				if (work->valid_nonces == MAX_NONCES) goto out;
			}
			if (work->valid_nonces)
				goto out;

			valid_sols[thr_id] = 0;
		}

		endiandata[NONCE_OFT] += nonce_increment;

		if (eq_slice_done(&tv_start)) break;
	} while (!work_restart[thr_id].restart);

out:
	gettimeofday(&tv_end, NULL);
	timeval_subtract(&diff, &tv_end, &tv_start);
	secs = (1.0 * diff.tv_sec) + (0.000001 * diff.tv_usec);
	solps = (double)soluce_count / secs;
	gpulog(LOG_DEBUG, thr_id, "%d solutions in %.2f s (%.2f Sol/s)", soluce_count, secs, solps);

	// H/s
	*hashes_done = soluce_count;

	pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;

	return work->valid_nonces;
}

// cleanup
void free_equihash(int thr_id)
{
	if (!init[thr_id])
		return;

	// the frees below must hit this thread's GPU, whichever thread calls us
	cudaSetDevice(device_map[thr_id]);

	if (equi24b_ctx[thr_id]) {            // 144/5 + 192/7 (equi24b) path
		if (eq_wk == 7) equi24b_192_free(equi24b_ctx[thr_id]);
		else            equi24b_144_free(equi24b_ctx[thr_id]);
		equi24b_ctx[thr_id] = NULL;
		if (equi24b_pipe[thr_id]) {
			if (eq_wk == 7) equi24b_192_pipe_free(equi24b_pipe[thr_id]);
			else            equi24b_144_pipe_free(equi24b_pipe[thr_id]);
			equi24b_pipe[thr_id] = NULL;
		}
		eq_pend_n[thr_id] = 0;            // the queued solves died with the pipe
	} else if (solvers[thr_id]) {         // 200/9 (djeZo) path
		// assume config 1 was used... interface destructor seems bad
		eq_cuda_context<CONFIG_MODE_1>* ptr = dynamic_cast<eq_cuda_context<CONFIG_MODE_1>*>(solvers[thr_id]);
		ptr->freemem();
		solvers[thr_id] = NULL;
	}

	init[thr_id] = false;
}

// mmm... viva c++ junk
void eq_cuda_context_interface::solve(const char *tequihash_header, unsigned int tequihash_header_len,
	const char* nonce, unsigned int nonce_len,
	fn_cancel cancelf, fn_solution solutionf, fn_hashdone hashdonef) { }
