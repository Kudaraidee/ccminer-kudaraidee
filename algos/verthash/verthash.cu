// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verthash (Vertcoin VTC) ccminer bridge: scanhash_verthash + init/free.
//
// Standard Bitcoin-fork mining path: 80-byte header, 32-bit nonce at data[19],
// generic stratum/submit. The one new capability is a persistent ~1.19 GiB
// verthash.dat image in VRAM, loaded once per device from a shared host copy.
// Every candidate is re-verified on the host before submit, so a kernel bug can
// only cost a local reject, never a bad share.
//
// Provenance & design: algos/verthash/README.md; device kernels in
// cuda_verthash.cu; datafile management in verthash-data.cpp; CPU oracle in
// verthash-cpu.c. Ported from VerthashMiner (GPLv2) + cpuminer-opt (GPLv2).

#include <cuda_runtime.h>
#include <stdint.h>
#include <unistd.h>   // sleep() on Linux (Windows shim: compat/unistd.h + compat.h)

extern "C" {
#include "miner.h"
}
#include "cuda/helper.cuh"
#include "cuda/intensity_autotune.cuh"
#include "cuda/selftest_gate.cuh"

extern "C" {
#include "algos/verthash/verthash-data.h"
#include "algos/verthash/verthash-cpu.h"
}

// cuda_verthash.cu launchers
extern "C" void verthash_cuda_set_header(const uint32_t header19[19]);
extern "C" void verthash_cuda_set_mdiv(uint32_t mdiv);
extern "C" void verthash_cuda_precompute(uint2 *d_kstates);
extern "C" void verthash_cuda_hash(uint2 *d_iohashes, const uint2 *d_kstates, const uint2 *d_memory,
                                   const uint2 *d_memory_odd,
                                   uint32_t in18, uint32_t firstNonce, uint32_t nonces,
                                   uint32_t *d_results, uint32_t target);

// opt_verthash_data (--verthash-data <path>) comes from miner.h; default
// "verthash.dat" in the cwd.

// -------- shared host datafile (loaded once for all GPUs) --------
static uint8_t  *s_dat      = NULL;
static size_t    s_dat_size = 0;
static uint32_t  s_mdiv     = 0;
static pthread_mutex_t s_dat_lock = PTHREAD_MUTEX_INITIALIZER;

// -------- per-device state --------
static bool       s_init[MAX_GPUS]       = { 0 };
static bool       s_selftested[MAX_GPUS] = { 0 };
static int        s_selftest_tries[MAX_GPUS] = { 0 };
static uint2     *d_memory[MAX_GPUS]     = { 0 };
// 16-byte-shifted copy of the datafile: makes every 32-byte item read
// 32-byte-aligned, removing the sector straddle. NULL = single-buffer path.
static uint2     *d_memory_odd[MAX_GPUS] = { 0 };
static uint2     *d_iohashes[MAX_GPUS]   = { 0 };
static uint2     *d_kstates[MAX_GPUS]    = { 0 };
static uint32_t  *d_results[MAX_GPUS]    = { 0 };
static uint32_t   s_throughput[MAX_GPUS] = { 0 };   // allocated size (tuner maximum)
static intensity_tuner_t s_tuner[MAX_GPUS];

// Load + verify the host datafile exactly once, shared across GPU threads.
static bool ensure_host_datafile(int thr_id)
{
	bool ok = true;
	pthread_mutex_lock(&s_dat_lock);
	if (!s_dat) {
		const char *path = opt_verthash_data ? opt_verthash_data : "verthash.dat";
		gpulog(LOG_INFO, thr_id, "Verthash: loading data file '%s' (~1.2 GiB)...", path);
		if (verthash_data_load(path, &s_dat, &s_dat_size) != 0) {
			gpulog(LOG_ERR, thr_id, "Verthash: failed to load data file '%s'", path);
			gpulog(LOG_NOTICE, thr_id, "Verthash: set --verthash-data <path> to your verthash.dat");
			ok = false;
		} else {
			s_mdiv = verthash_data_mdiv(s_dat_size);
			if (verthash_data_verify(s_dat, s_dat_size))
				gpulog(LOG_INFO, thr_id, "Verthash: data file verified (%zu bytes, mdiv %u)", s_dat_size, s_mdiv);
			else {
				// Fatal: a non-canonical datafile hashes a different function,
				// so every share would be rejected.
				gpulog(LOG_ERR, thr_id, "Verthash: data file digest MISMATCH -- this is not the"
				       " canonical Verthash.dat (%zu bytes, mdiv %u). Mining it would produce"
				       " nothing but rejected shares; refusing to start.", s_dat_size, s_mdiv);
				proper_exit(EXIT_CODE_SW_INIT_ERROR);
			}
		}
	}
	pthread_mutex_unlock(&s_dat_lock);
	return ok;
}

// Lets the self-test gate be made to fail on demand, since a passing gate and a
// gate that never ran look identical. VERTHASH_SELFTEST_FAULT =
// kat | neg | cuda. Logged loudly; never a healthy state.
static const char *verthash_fault_mode(int thr_id)
{
	static int announced[MAX_GPUS] = { 0 };
	const char *m = getenv("VERTHASH_SELFTEST_FAULT");
	if (m && *m && !announced[thr_id]) {
		announced[thr_id] = 1;
		gpulog(LOG_WARNING, thr_id, "Verthash: SELF-TEST FAULT INJECTION ACTIVE (%s) -- test build only", m);
	}
	return (m && *m) ? m : NULL;
}

// Returns false only when the test could not RUN (CUDA resource failure) --
// not a wrong answer. The verdict comes back in *mism / *negative_ok.
static bool verthash_selftest_run(int thr_id, uint32_t N, int *mism, bool *negative_ok)
{
	const char *fault = verthash_fault_mode(thr_id);
	uint8_t header[80], ref0[32];
	uint32_t *hw = (uint32_t *) header;
	uint32_t *gpu;

	*mism = 0;
	*negative_ok = false;

	for (int i = 0; i < 80; i++) header[i] = (uint8_t)(i * 7 + 1);
	// Byte 0 near the top of its range: the 8 subset seeds byte0+1..byte0+8 then
	// cover both the plain lanes (0xfb..0xff) and the wrap lanes (0x00..0x02),
	// so a seed add that carries into byte 1 fails the test.
	header[0] = 0xfa;

	if (fault && !strcmp(fault, "cuda")) return selftest_cuda_fault();

	verthash_cuda_set_header(hw);
	verthash_cuda_set_mdiv(s_mdiv);
	verthash_cuda_precompute(d_kstates[thr_id]);
	cudaMemset(d_results[thr_id], 0, sizeof(uint32_t));
	verthash_cuda_hash(d_iohashes[thr_id], d_kstates[thr_id], d_memory[thr_id],
	                   d_memory_odd[thr_id], hw[18], 0, N, d_results[thr_id], 0xffffffffu);
	if (cudaDeviceSynchronize() != cudaSuccess) return selftest_cuda_fault();

	gpu = (uint32_t *) malloc(N * 32);
	if (!gpu) return selftest_cuda_fault();
	if (cudaMemcpy(gpu, d_iohashes[thr_id], N * 32, cudaMemcpyDeviceToHost) != cudaSuccess) {
		free(gpu);
		return selftest_cuda_fault();
	}

	if (fault && !strcmp(fault, "kat")) gpu[0] ^= 1u;   // one wrong digest

	for (uint32_t k = 0; k < N; k++) {
		uint8_t ref[32];
		hw[19] = k;
		verthash_hash_oracle(s_dat, s_dat_size, header, ref);
		if (memcmp(ref, gpu + k * 8, 32) != 0) (*mism)++;
	}

	// Negative control: a bit-flipped reference for nonce 0 must NOT match the
	// GPU hash -- proves the comparison above is not vacuous.
	hw[19] = 0;
	verthash_hash_oracle(s_dat, s_dat_size, header, ref0);
	if (!(fault && !strcmp(fault, "neg"))) ref0[0] ^= 0x01;
	*negative_ok = (memcmp(ref0, gpu, 32) != 0);

	free(gpu);
	return true;
}

// GPU vs CPU-oracle self-test plus a negative control. Fail-closed: a mismatch
// refuses to start, since a card that cannot reproduce the consensus hash mines
// a whole session producing only local rejects. A resource failure only warns.
static bool verthash_selftest(int thr_id)
{
	const uint32_t N = 256;
	int mism = 0;
	bool negative_ok = false;
	bool ran = verthash_selftest_run(thr_id, N, &mism, &negative_ok);
	bool passed = ran && mism == 0 && negative_ok;

	if (passed)
		gpulog(LOG_INFO, thr_id, "Verthash self-test OK (GPU==CPU on %u nonces, negative control passed)", N);
	else
		gpulog(LOG_ERR, thr_id, "Verthash self-test FAILED (%d/%u mismatch, neg=%d, ran=%d)",
		       mism, N, (int) negative_ok, (int) ran);

	return selftest_gate(thr_id, "verthash", passed);
}

static int verthash_cmp_u32(const void *a, const void *b)
{
	const uint32_t x = *(const uint32_t *) a, y = *(const uint32_t *) b;
	return (x > y) - (x < y);
}

extern "C" int scanhash_verthash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!ensure_host_datafile(thr_id)) { *hashes_done = 0; sleep(1); return 0; }

	// throughput: memory-latency bound (4096 random reads/nonce). Round DOWN to a
	// multiple of 256 (exact grids; the IO kernel cannot early-return).
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 15);
	throughput &= ~255u;
	if (throughput < 256) throughput = 256;

	// Launch size is auto-tuned on the running card unless the user passed -i.
	// Buffers are sized for the tuner's maximum so the batch size can vary.
	if (!s_init[thr_id])
		throughput = intensity_tuner_init(&s_tuner[thr_id], thr_id, throughput, 256, 250.0);

	if (!s_init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		cuda_get_arch(thr_id);

		// resident datafile + scratch buffers
		if (cudaMalloc(&d_memory[thr_id], s_dat_size) != cudaSuccess) {
			gpulog(LOG_ERR, thr_id, "Verthash: cudaMalloc %zu bytes for datafile failed", s_dat_size);
			return -1;
		}
		cudaMemcpy(d_memory[thr_id], s_dat, s_dat_size, cudaMemcpyHostToDevice);

		// Second copy shifted 16 bytes, so an odd datafile index reads the same
		// logical bytes from a 32-byte-aligned address (see cuda_verthash.cu).
		// Costs ~1.19 GiB; skipped when VRAM is short or VERTHASH_NO_DUAL is set.
		if (!getenv("VERTHASH_NO_DUAL")) {
			size_t freemem = 0, totalmem = 0;
			cudaMemGetInfo(&freemem, &totalmem);
			if (freemem > s_dat_size + (256u << 20) &&
			    cudaMalloc(&d_memory_odd[thr_id], s_dat_size) == cudaSuccess) {
				cudaMemcpy(d_memory_odd[thr_id], s_dat + 16, s_dat_size - 16,
				           cudaMemcpyHostToDevice);
				gpulog(LOG_INFO, thr_id, "Verthash: 32B-aligned dual buffer enabled (+%zu MB)",
				       s_dat_size >> 20);
			} else {
				// A warning, not info: this card silently runs the slower path and
				// this line is the operator's only notice.
				d_memory_odd[thr_id] = NULL;
				gpulog(LOG_WARNING, thr_id, "Verthash: NOT enough VRAM for the 32B-aligned dual buffer"
				       " (%zu MB free, need %zu) -- falling back to the slower single-buffer path",
				       freemem >> 20, (s_dat_size + (256u << 20)) >> 20);
			}
		}
		cudaMalloc(&d_iohashes[thr_id], (size_t) throughput * 4 * sizeof(uint2));
		cudaMalloc(&d_kstates[thr_id], 8 * 25 * sizeof(uint2));
		cudaMalloc(&d_results[thr_id], (size_t)(throughput + 1) * sizeof(uint32_t));
		s_throughput[thr_id] = throughput;
		CUDA_LOG_ERROR();

		if (intensity_tuner_running(&s_tuner[thr_id]))
			gpulog(LOG_INFO, thr_id, "Intensity auto, buffers sized for %g / %u threads",
			       throughput2intensity(throughput), throughput);
		else
			gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			       throughput2intensity(throughput), throughput);
		s_init[thr_id] = true;
	}
	throughput = s_throughput[thr_id];  // allocated maximum; per-batch size below

	// Only mark done on a pass; the gate exits on a real mismatch. Retry is
	// bounded: each attempt costs a launch plus 256 CPU oracle hashes.
	#define VH_SELFTEST_TRIES 3
	if (!s_selftested[thr_id]) {
		if (verthash_selftest(thr_id)) {
			s_selftested[thr_id] = true;
		} else if (++s_selftest_tries[thr_id] >= VH_SELFTEST_TRIES) {
			s_selftested[thr_id] = true;   // stop retrying, but it never passed
			gpulog(LOG_WARNING, thr_id, "Verthash: self-test could not run in %d attempts --"
			       " MINING UNVERIFIED on this device", VH_SELFTEST_TRIES);
		}
	}

	// Byteswapped header, the order the CPU oracle consumes. Words 0..18 are
	// fixed; the raw nonce goes at word 19.
	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	verthash_cuda_set_header(endiandata);
	verthash_cuda_set_mdiv(s_mdiv);
	verthash_cuda_precompute(d_kstates[thr_id]);

	const uint32_t Htarg = ptarget[7];

	do {
		if (pdata[19] >= max_nonce)
			break;
		const uint32_t tuned = intensity_tuner_running(&s_tuner[thr_id])
		                     ? intensity_tuner_size(&s_tuner[thr_id])
		                     : intensity_tuner_chosen(&s_tuner[thr_id]);
		// n = real nonces of this launch, clamped to the end of the range. The
		// launch itself stays a multiple of 256 (exact grids, no bounds guard in the
		// IO kernel); tuned is one, so batch <= tuned <= the allocated maximum.
		// Candidates at offset >= n (past max_nonce) are dropped below.
		const uint32_t n = (uint32_t) min((uint64_t) tuned, (uint64_t) max_nonce - pdata[19]);
		const uint32_t batch = (n + 255u) & ~255u;
		const double t_batch = tuner_now();

		cudaMemset(d_results[thr_id], 0, sizeof(uint32_t));
		verthash_cuda_hash(d_iohashes[thr_id], d_kstates[thr_id], d_memory[thr_id],
		                   d_memory_odd[thr_id], endiandata[18], pdata[19], batch,
		                   d_results[thr_id], Htarg);
		cudaDeviceSynchronize();

		uint32_t nres = 0;
		cudaMemcpy(&nres, d_results[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (nres > batch) nres = batch;

		// A batch that found something pays a host re-verify, and a clamped last
		// batch is not the tuned size, so neither is a fair sample; feed them tainted.
		intensity_tuner_sample(&s_tuner[thr_id], tuner_now() - t_batch, nres != 0 || batch != tuned);

		if (nres && bench_algo < 0) {
			uint32_t *offs = (uint32_t *) malloc((size_t) nres * sizeof(uint32_t));
			cudaMemcpy(offs, d_results[thr_id] + 1, (size_t) nres * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			// slots are in atomicAdd order: check in nonce order, so the two
			// reported are the lowest valid ones and every later candidate lies
			// above the resume point (the next scan finds it again)
			qsort(offs, nres, sizeof(uint32_t), verthash_cmp_u32);

			work->valid_nonces = 0;
			for (uint32_t r = 0; r < nres && work->valid_nonces < 2; r++) {
				if (offs[r] >= n)
					break;   // padding of a clamped launch: past max_nonce
				const uint32_t nonce = pdata[19] + offs[r];
				uint32_t _ALIGN(64) vhash[8];
				((uint32_t *) endiandata)[19] = nonce;   // raw nonce (verthash hashes it LE)
				verthash_hash_oracle(s_dat, s_dat_size, endiandata, vhash);
				if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
					work->nonces[work->valid_nonces] = nonce;
					bn_set_target_ratio(work, vhash, work->valid_nonces);
					work->valid_nonces++;
				} else if (vhash[7] > Htarg) {
					// the screen is a safe superset (word 7 <= target), so a word-7
					// miss here is a wrong GPU hash: count it, never stay silent
					gpu_increment_reject(thr_id);
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", nonce);
				}
				// word 7 == target but the lower words fail: a legitimate screen pass
			}
			free(offs);

			if (work->valid_nonces) {
				*hashes_done = pdata[19] - first_nonce + n;
				// resume at the highest reported nonce; the core steps past it
				// (nonceptr[0]++ on the same work), candidates above it are re-found
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

extern "C" void free_verthash(int thr_id)
{
	if (!s_init[thr_id]) return;
	cudaDeviceSynchronize();
	if (d_memory[thr_id])   { cudaFree(d_memory[thr_id]);   d_memory[thr_id] = NULL; }
	if (d_memory_odd[thr_id]) { cudaFree(d_memory_odd[thr_id]); d_memory_odd[thr_id] = NULL; }
	if (d_iohashes[thr_id]) { cudaFree(d_iohashes[thr_id]); d_iohashes[thr_id] = NULL; }
	if (d_kstates[thr_id])  { cudaFree(d_kstates[thr_id]);  d_kstates[thr_id] = NULL; }
	if (d_results[thr_id])  { cudaFree(d_results[thr_id]);  d_results[thr_id] = NULL; }
	s_selftested[thr_id] = false;
	s_selftest_tries[thr_id] = 0;
	s_init[thr_id] = false;
}
