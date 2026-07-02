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

#include <cuda_helper.h>

//#define EQNONCE_OFFSET 30 /* 27:34 */
#define NONCE_OFT EQNONCE_OFFSET

static bool init[MAX_GPUS] = { 0 };
static int valid_sols[MAX_GPUS] = { 0 };
static uint8_t _ALIGN(64) data_sols[MAX_GPUS][MAXREALSOLS][1536] = { 0 }; // 140+3+1344 required
static eq_cuda_context_interface* solvers[MAX_GPUS] = { NULL };

// --- equihash variant (n,k)+personalization dispatch ------------------------
// The djeZo solver above is 200/9-only; the tromp144 solver (cuda_equi_tromp.cu)
// handles 144/5. Default is Zcash 200/9 ("ZcashPoW"). The (n,k) variant is set
// ONLY by the `-a` algo parameter (equihash / equihash144) -> eq_set_variant_144()
// — it fixes the CUDA solver, so it never changes at runtime. The pool's
// mining.notify may then set the personalization (eq_set_variant_params: personal
// only; its (n,k) is validate-only). Solution size + personalization are
// parametrized via eq_solsize()/eq_personal.
#include "cuda_equi_tromp.h"
static int   eq_wn = 200, eq_wk = 9;
static char  eq_personal[16] = "ZcashPoW";
static void* tromp_ctx[MAX_GPUS] = { NULL };

static inline int eq_cbitlen()   { return eq_wn / (eq_wk + 1); }               // 20 / 24
static inline int eq_proofsize() { return 1 << eq_wk; }                        // 512 / 32
static inline int eq_solsize()   { return eq_proofsize() * (eq_cbitlen() + 1) / 8; } // 1344 / 100

// Bitcoin CompactSize prefix length for the solution byte count.
static inline int eq_solprefix() { int s = eq_solsize(); return s < 253 ? 1 : (s <= 0xffff ? 3 : 5); }

// Shared accessors for the stratum layer (equi-stratum.cpp) — the number of
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

// Apply the equihash params the POOL advertises in mining.notify (zpool /
// cpuminer-opt convention: trailing "<n>_<k>" and 8-char personalization).
//
// (n,k) is FIXED by the -a algo parameter — it defines the CUDA solver/kernel,
// so we never switch it at runtime (that would force a kernel unload/reload on a
// job change). The pool-advertised (n,k) is therefore validation-only: warn on
// mismatch (miner pointed at the wrong-variant pool) and ignore it. Only the
// personalization (a runtime BLAKE2b param, no kernel impact) is adopted from
// the pool — this is what lets a 144/5 pool select e.g. "ZcashPoW". Logs only on
// change to avoid per-notify spam.
extern "C" void eq_set_variant_params(int wn, int wk, const char* personal)
{
	if (wn > 0 && wk > 0 && (wn != eq_wn || wk != eq_wk)) {
		static bool warned = false;
		if (!warned) {
			applog(LOG_WARNING, "pool advertises equihash %d/%d but miner is %d/%d "
			       "(fixed by -a); ignoring pool (n,k) — use the matching -a algo",
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

static void CompressArray(const unsigned char* in, size_t in_len,
	unsigned char* out, size_t out_len, size_t bit_len, size_t byte_pad)
{
	assert(bit_len >= 8);
	assert(8 * sizeof(uint32_t) >= 7 + bit_len);

	size_t in_width = (bit_len + 7) / 8 + byte_pad;
	assert(out_len == bit_len*in_len / (8 * in_width));

	uint32_t bit_len_mask = (1UL << bit_len) - 1;

	// The acc_bits least-significant bits of acc_value represent a bit sequence
	// in big-endian order.
	size_t acc_bits = 0;
	uint32_t acc_value = 0;

	size_t j = 0;
	for (size_t i = 0; i < out_len; i++) {
		// When we have fewer than 8 bits left in the accumulator, read the next
		// input element.
		if (acc_bits < 8) {
			acc_value = acc_value << bit_len;
			for (size_t x = byte_pad; x < in_width; x++) {
				acc_value = acc_value | (
					(
					// Apply bit_len_mask across byte boundaries
					in[j + x] & ((bit_len_mask >> (8 * (in_width - x - 1))) & 0xFF)
					) << (8 * (in_width - x - 1))); // Big-endian
			}
			j += in_width;
			acc_bits += bit_len;
		}

		acc_bits -= 8;
		out[i] = (acc_value >> acc_bits) & 0xFF;
	}
}

#ifndef htobe32
#define htobe32(x) swab32(x)
#endif

static void EhIndexToArray(const u32 i, unsigned char* arr)
{
	u32 bei = htobe32(i);
	memcpy(arr, &bei, sizeof(u32));
}

static std::vector<unsigned char> GetMinimalFromIndices(std::vector<u32> indices, size_t cBitLen)
{
	assert(((cBitLen + 1) + 7) / 8 <= sizeof(u32));
	size_t lenIndices = indices.size()*sizeof(u32);
	size_t minLen = (cBitLen + 1)*lenIndices / (8 * sizeof(u32));
	size_t bytePad = sizeof(u32) - ((cBitLen + 1) + 7) / 8;
	std::vector<unsigned char> array(lenIndices);
	for (size_t i = 0; i < indices.size(); i++) {
		EhIndexToArray(indices[i], array.data() + (i*sizeof(u32)));
	}
	std::vector<unsigned char> ret(minLen);
	CompressArray(array.data(), lenIndices, ret.data(), minLen, cBitLen + 1, bytePad);
	return ret;
}

// solver callbacks
static void cb_solution(int thr_id, const std::vector<uint32_t>& solutions, size_t cbitlen, const unsigned char *compressed_sol)
{
	std::vector<unsigned char> nSolution;
	if (!compressed_sol) {
		nSolution = GetMinimalFromIndices(solutions, cbitlen);
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
static bool cb_cancel(int thr_id) {
	if (work_restart[thr_id].restart)
		valid_sols[thr_id] = -1;
	return work_restart[thr_id].restart;
}

// --- 144/5 scan path, driven by the tromp144 solver -------------------------
// FOR TECHNICAL STUDY ONLY: tromp's reference solver — correct + live-validated,
// but not performance-optimized and not comparable to dedicated Equihash miners.
static uint32_t tromp_idx[MAX_GPUS][MAXREALSOLS][32]; // emitted solution indices
static int      tromp_ns[MAX_GPUS];

static void tromp_emit(void* ud, const uint32_t* idx, uint32_t proofsize)
{
	int thr = *(int*)ud;
	int n = tromp_ns[thr];
	if (n < MAXREALSOLS && proofsize == 32) {
		memcpy(tromp_idx[thr][n], idx, 32 * sizeof(uint32_t));
		tromp_ns[thr] = n + 1;
	}
}

static int scanhash_equihash_144_5(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[35];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t nonce_increment = (rand() & 0xFF) | 1; // odd step (never a 0 re-grind)
	struct timeval tv_start, tv_end, diff;
	double secs;
	uint32_t soluce_count = 0;
	const int cbl   = eq_cbitlen();  // 24
	const int solsz = eq_solsize();  // 100

	if (opt_benchmark)
		ptarget[7] = 0xfffff;

	if (!init[thr_id]) {
		tromp_ctx[thr_id] = tromp144_init(8192, 0);
		if (!tromp_ctx[thr_id]) {
			gpulog(LOG_ERR, thr_id, "tromp144_init failed");
			proper_exit(EXIT_CODE_CUDA_ERROR);
			return -1;
		}
		gpus_intensity[thr_id] = 8192;
		api_set_throughput(thr_id, gpus_intensity[thr_id]);
		cuda_get_arch(thr_id);
		init[thr_id] = true;
	}

	gettimeofday(&tv_start, NULL);
	memcpy(endiandata, pdata, 140);
	work->valid_nonces = 0;

	do {
		tromp_ns[thr_id] = 0;
		int nsol = tromp144_solve(tromp_ctx[thr_id], (const char*) endiandata,
		                          eq_personal, tromp_emit, &thr_id);
		soluce_count += (nsol > 0 ? nsol : 0);
		*hashes_done = soluce_count;

		if (tromp_ns[thr_id] > 0) {
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			uint8_t  _ALIGN(64) full_data[140 + 3 + 1344] = { 0 };
			uint8_t* sol_data = &full_data[140];

			for (int s = 0; s < tromp_ns[thr_id]; s++) {
				std::vector<u32> idx(tromp_idx[thr_id][s], tromp_idx[thr_id][s] + 32);
				std::vector<unsigned char> minimal = GetMinimalFromIndices(idx, cbl); // 100 bytes

				memcpy(full_data, endiandata, 140);
				sol_data[0] = (uint8_t) solsz;             // compactSize 0x64 (solsz < 253)
				memcpy(&sol_data[1], minimal.data(), solsz);
				equi_hash(full_data, vhash, 140 + 1 + solsz);

				if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
					int rc = tromp144_verify((const char*) endiandata, eq_personal,
					                         tromp_idx[thr_id][s]);
					if (rc == 0 && work->valid_nonces < MAX_NONCES) {
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

		endiandata[NONCE_OFT] += nonce_increment;

	} while (!work_restart[thr_id].restart);

out:
	gettimeofday(&tv_end, NULL);
	timeval_subtract(&diff, &tv_end, &tv_start);
	secs = (1.0 * diff.tv_sec) + (0.000001 * diff.tv_usec);
	gpulog(LOG_DEBUG, thr_id, "%d solutions in %.2f s (%.2f Sol/s)",
	       soluce_count, secs, secs > 0 ? soluce_count / secs : 0.0);
	*hashes_done = soluce_count;
	pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;
	return work->valid_nonces;
}

extern "C" int scanhash_equihash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	if (eq_wk == 5)
		return scanhash_equihash_144_5(thr_id, work, max_nonce, hashes_done);

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
#ifdef CONFIG_MODE_2
			case 2:
				solvers[thr_id] = new eq_cuda_context<CONFIG_MODE_2>(thr_id, device_map[thr_id]);
				break;
#endif
#ifdef CONFIG_MODE_3
			case 3:
				solvers[thr_id] = new eq_cuda_context<CONFIG_MODE_3>(thr_id, device_map[thr_id]);
				break;
#endif
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

	if (tromp_ctx[thr_id]) {              // 144/5 (tromp) path
		tromp144_free(tromp_ctx[thr_id]);
		tromp_ctx[thr_id] = NULL;
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
