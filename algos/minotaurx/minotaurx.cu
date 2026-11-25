/* minotaurx (Avian / Pulsar) -- host driver and mining kernel.
 *
 * The six-node x16 walk runs on the host (minotaurx_hash.cpp), overlapped with
 * the kernel; the GPU runs only the final yespower 1.0 node (N=2048, r=8) on a
 * 64-byte input per instance, using the shared yespower body.
 *
 * Difficulty is the 2^32 scale, not yespower's 65536.  Every candidate is
 * re-hashed on the host before submit. */

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos.h"

extern "C" {
#include "sph/yespower.h"
}

#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "algos/yespower/yespower_hash.cuh"
#include "cuda/selftest_gate.cuh"

#include "minotaurx.h"
#include "minotaurx_kat.h"

#define MX_N         2048u
#define MX_R         8u
#define MX_IN_WORDS  16u              /* the 64-byte yespower input, per instance */

/* Kernel shape: the same as yespower.cu's r=8 path. */
#define MX_WIDTH     32
#define MX_WMASK     0xffffffffu
#define MX_MINBLK    1
#define MX_X_SHARED(PL)  ((PL) == PWX_S_GLOBAL)
#define MX_IPB_AMPERE 8u
#define MX_IPB_OTHER  16u
#define MX_MAX_RES   8u

static const char MX_PERS[] = "et in arcadia ego";   /* 17 bytes */

__constant__ static uint32_t c_mx_target[8];

/* Exact 256-bit compare, MSW first, as fulltest() does. */
__device__ __forceinline__ bool mx_below_target(const uint32_t h[8])
{
#pragma unroll
	for (int i = 7; i >= 0; i--) {
		if (h[i] > c_mx_target[i]) return false;
		if (h[i] < c_mx_target[i]) return true;
	}
	return true;
}

/* One block = one instance; `in` holds MX_IN_WORDS per instance in SHA-256
 * input order.  `digests` is NULL when mining; the self-tests pass a buffer. */
template<int PLACE>
__global__ __launch_bounds__(MX_WIDTH, MX_MINBLK)
void minotaurx_gpu_hash(const uint32_t startNonce, const uint32_t *__restrict__ in,
                        uint32_t *__restrict__ Vs, uint32_t *__restrict__ Bs,
                        uint32_t *__restrict__ Xs, uint4 *__restrict__ Sg,
                        uint32_t *__restrict__ resNonces, uint32_t *__restrict__ digests)
{
	extern __shared__ uint4 s_S[];
	__shared__ uint32_t s_X[MX_X_SHARED(PLACE) ? 32u * MX_R : 1u];

	const int j = threadIdx.x & 3;
	const uint32_t inst = blockIdx.x;
	const uint32_t nonce = startNonce + inst;
	const uint32_t bw = 32u * MX_R;
	uint32_t out[8];

	uint4 *S = (PLACE == PWX_S_SHARED) ? s_S : (Sg + (size_t)inst * YP_SBOX_UINT4);
	uint32_t *Xp = MX_X_SHARED(PLACE) ? s_X : (Xs + (size_t)inst * bw);

	yespower_hash_1_0<MX_R, PLACE, YP_HEAD_SHA256_64, MX_WIDTH>(
		in + (size_t)inst * MX_IN_WORDS, 0u, MX_N, S,
		Bs + (size_t)inst * bw, Xp, Vs + (size_t)inst * bw * MX_N,
		out, j, 0xfu, (int)threadIdx.x, MX_WMASK);

	/* threadIdx.x, not j: only tid < 4 computes out[] */
	if (threadIdx.x == 0) {
		if (digests) {
#pragma unroll
			for (int i = 0; i < 8; i++) digests[inst * 8u + i] = out[i];
		}
		if (mx_below_target(out)) {
			const uint32_t pos = atomicAdd(&resNonces[0], 1u);
			if (pos < MX_MAX_RES) resNonces[1u + pos] = nonce;
		}
	}
}

static THREAD uint32_t *d_V[MAX_GPUS]   = { 0 };
static THREAD uint32_t *d_B[MAX_GPUS]   = { 0 };
static THREAD uint32_t *d_X[MAX_GPUS]   = { 0 };
static THREAD uint4    *d_S[MAX_GPUS]   = { 0 };   /* global S arena; NULL on the shared path */
static THREAD uint32_t *d_in[MAX_GPUS]  = { 0 };
static THREAD uint32_t *d_res[MAX_GPUS] = { 0 };
static THREAD uint32_t *h_in[MAX_GPUS]  = { 0 };   /* host chain output, one batch */
static THREAD bool      mx_sglobal[MAX_GPUS]   = { false };
static THREAD uint32_t  mx_instances[MAX_GPUS] = { 0 };
static THREAD uint32_t  mx_diff_prev[MAX_GPUS][8];        /* -D: prevhash last checked */
static THREAD time_t    mx_diff_time[MAX_GPUS] = { 0 };   /* -D: when; 0 = never */
static bool init[MAX_GPUS] = { false };

static void mx_launch(int dev, uint32_t instances, uint32_t startNonce, uint32_t *d_digests)
{
	if (mx_sglobal[dev])
		minotaurx_gpu_hash<PWX_S_GLOBAL><<<instances, MX_WIDTH, 0>>>(startNonce, d_in[dev],
			d_V[dev], d_B[dev], d_X[dev], d_S[dev], d_res[dev], d_digests);
	else
		minotaurx_gpu_hash<PWX_S_SHARED><<<instances, MX_WIDTH, YP_SBOX_UINT4 * 16u>>>(startNonce,
			d_in[dev], d_V[dev], d_B[dev], d_X[dev], d_S[dev], d_res[dev], d_digests);
}

/* 64 chain bytes -> SHA-256 input words.  MX_FAULT_LE_WORDS is a test-only fault. */
static inline void mx_pack(uint32_t *dst, const uint8_t st[64])
{
	for (uint32_t k = 0; k < MX_IN_WORDS; k++)
#ifdef MX_FAULT_LE_WORDS
		dst[k] = le32dec(st + 4 * k);
#else
		dst[k] = be32dec(st + 4 * k);
#endif
}

/* Host chain for `count` nonces from `start`, into h_in[dev]. */
static void mx_chain_batch(int dev, uint32_t *endiandata, uint32_t start, uint32_t count)
{
	uint8_t st[64];
	for (uint32_t i = 0; i < count; i++) {
		be32enc(&endiandata[19], start + i);
		minotaurx_chain(st, endiandata, NULL);
		mx_pack(h_in[dev] + (size_t)i * MX_IN_WORDS, st);
	}
}

/* Fail-closed init gate: the CPU reference on four Pulsar mainnet blocks, then
 * the mining kernel on their chain states, on the published yespower-node
 * vector, and on its own target report. */
static bool minotaurx_device_selftest(int thr_id, int dev)
{
	char det[MINOTAURX_SELFTEST_DETAIL];
	if (!minotaurx_self_test(det, sizeof det)) {
		gpulog(LOG_ERR, thr_id, "minotaurx CPU self-test FAILED: %s", det);
		return selftest_gate(thr_id, "minotaurx", false);
	}
	gpulog(LOG_INFO, thr_id, "minotaurx CPU self-test OK (%s)", det);

	/* inst 0..3: KAT chain states; 4: flipped nonce (must differ); 5: node-21 vector */
	enum { NI = 6 };
	const uint32_t start = 0x51u;             /* nonzero: exercises startNonce + inst */
	uint32_t digests[NI * 8], res[1u + MX_MAX_RES] = { 0 }, target[8];
	uint32_t *d_dig = NULL;
	uint8_t st[64], hdr[80];

	for (int k = 0; k < 4; k++) {
		minotaurx_chain(st, minotaurx_kats[k].header, NULL);
		mx_pack(h_in[dev] + k * MX_IN_WORDS, st);
	}
	memcpy(hdr, minotaurx_kats[0].header, 80);
	hdr[79] ^= 1;
	minotaurx_chain(st, hdr, NULL);
	mx_pack(h_in[dev] + 4 * MX_IN_WORDS, st);
	mx_pack(h_in[dev] + 5 * MX_IN_WORDS, minotaurx_kat_node21_in);

	/* the loosest of the four block targets; inst 4 clears it with p ~ 2^-43 */
	memcpy(target, minotaurx_kats[0].target, 32);
	for (int k = 1; k < 4; k++)
		for (int w = 7; w >= 0; w--) {
			uint32_t t; memcpy(&t, minotaurx_kats[k].target + 4 * w, 4);
			if (t == target[w]) continue;
			if (t > target[w]) memcpy(target, minotaurx_kats[k].target, 32);
			break;
		}

	if (cudaMalloc(&d_dig, sizeof(digests)) != cudaSuccess)
		return selftest_gate(thr_id, "minotaurx", selftest_cuda_fault());
	bool ran = cudaMemcpy(d_in[dev], h_in[dev], NI * MX_IN_WORDS * 4, cudaMemcpyHostToDevice) == cudaSuccess
	        && cudaMemcpy(d_res[dev], res, sizeof(res), cudaMemcpyHostToDevice) == cudaSuccess
	        && cudaMemcpyToSymbol(c_mx_target, target, 32) == cudaSuccess;
	if (ran) {
		mx_launch(dev, NI, start, d_dig);
		ran = cudaGetLastError() == cudaSuccess
		   && cudaMemcpy(digests, d_dig, sizeof(digests), cudaMemcpyDeviceToHost) == cudaSuccess
		   && cudaMemcpy(res, d_res[dev], sizeof(res), cudaMemcpyDeviceToHost) == cudaSuccess;
	}
	cudaFree(d_dig);
	if (!ran)
		return selftest_gate(thr_id, "minotaurx", selftest_cuda_fault());

	const bool node_ok = memcmp(digests + 5 * 8, minotaurx_kats[0].digest, 32) == 0;
	bool kat_ok = true;
	for (int k = 0; k < 4; k++)
		kat_ok = kat_ok && memcmp(digests + k * 8, minotaurx_kats[k].digest, 32) == 0;
	const bool nonvac_ok = memcmp(digests + 4 * 8, minotaurx_kats[0].digest, 32) != 0;

	/* Report path: exactly {0,1,2,3,5} + start, in any order. */
	uint32_t seen = 0;
	bool report_ok = (res[0] == 5u);
	for (uint32_t s = 0; report_ok && s < res[0]; s++) {
		const uint32_t i = res[1u + s] - start;
		report_ok = (i < NI && i != 4u && !(seen & (1u << i)));
		seen |= 1u << i;
	}

	const bool passed = node_ok && kat_ok && nonvac_ok && report_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "minotaurx GPU self-test FAILED: yespower-node=%d kat=%d "
		       "non-vacuity=%d report=%d (count %u)", (int) node_ok, (int) kat_ok,
		       (int) nonvac_ok, (int) report_ok, res[0]);
	else
		gpulog(LOG_INFO, thr_id, "minotaurx GPU self-test OK (yespower node, 4 mainnet "
		       "headers, target report)");
	return selftest_gate(thr_id, "minotaurx", passed);
}

/* `-D`: the kernel's digest for MX_DIFF_SPAN nonces against the CPU reference,
 * folded into a plain and a (2*nonce+1)-weighted XOR.  This catches a missed or
 * misplaced nonce, which the host re-verify cannot.  Runs on a new prevhash,
 * otherwise at most every MX_DIFF_EVERY s (the merkle root changes every few
 * seconds).  The span is prime and larger than a batch, so it ends on a short
 * launch.  MX_FAULT_DIFF_CONST/_PERM are test-only faults. */
#define MX_DIFF_SPAN  457u
#define MX_DIFF_OFF   0x51u
#define MX_DIFF_EVERY 60

static void mx_debug_differential(int thr_id, int dev, const uint32_t pdata[20],
                                  const uint32_t *endiandata_in, uint32_t base)
{
	const time_t now = time(NULL);
	const bool new_block = memcmp(mx_diff_prev[dev], &pdata[1], 32) != 0;
	if (!new_block && mx_diff_time[dev] && now - mx_diff_time[dev] < MX_DIFF_EVERY)
		return;
	memcpy(mx_diff_prev[dev], &pdata[1], 32);
	mx_diff_time[dev] = now;

	uint32_t _ALIGN(64) endian[20];
	memcpy(endian, endiandata_in, sizeof(endian));
	const uint32_t start = base + MX_DIFF_OFF;
	const uint32_t inst = mx_instances[dev];
	unsigned long long g[2] = { 0ull, 0ull }, c[2] = { 0ull, 0ull };
	uint32_t *d_dig = NULL;
	uint32_t *dig = (uint32_t *) malloc((size_t) inst * 8u * sizeof(uint32_t));
	bool ok = dig && cudaMalloc(&d_dig, (size_t) inst * 8u * sizeof(uint32_t)) == cudaSuccess;

	for (uint32_t done = 0; ok && done < MX_DIFF_SPAN; ) {
		uint32_t chunk = MX_DIFF_SPAN - done;
		if (chunk > inst) chunk = inst;
		mx_chain_batch(dev, endian, start + done, chunk);
		ok = cudaMemcpy(d_in[dev], h_in[dev], (size_t) chunk * MX_IN_WORDS * 4,
		                cudaMemcpyHostToDevice) == cudaSuccess;
		if (ok) {
			mx_launch(dev, chunk, start + done, d_dig);
			ok = cudaGetLastError() == cudaSuccess &&
			     cudaMemcpy(dig, d_dig, (size_t) chunk * 8u * sizeof(uint32_t),
			                cudaMemcpyDeviceToHost) == cudaSuccess;
		}
		for (uint32_t i = 0; ok && i < chunk; i++) {
			const uint32_t nonce = start + done + i;
			unsigned long long q = ((unsigned long long) dig[i * 8u + 7u] << 32) | dig[i * 8u + 6u];
			unsigned long long w = 2ull * (unsigned long long) nonce + 1ull;
#ifdef MX_FAULT_DIFF_CONST
			q ^= 1ull;
#endif
#ifdef MX_FAULT_DIFF_PERM
			w = 2ull * (unsigned long long) (nonce ^ 1u) + 1ull;
#endif
			g[0] ^= q;
			g[1] ^= q * w;
		}
		done += chunk;
	}
	if (d_dig) cudaFree(d_dig);
	free(dig);
	if (!ok) {
		gpulog(LOG_WARNING, thr_id, "minotaurx differential: could not run (CUDA resource "
		       "failure) -- not evidence of a wrong hash");
		return;
	}

	for (uint32_t i = 0; i < MX_DIFF_SPAN; i++) {
		const uint32_t nonce = start + i;
		uint32_t h[8];
		be32enc(&endian[19], nonce);
		if (!minotaurx_hash(h, endian)) {
			gpulog(LOG_WARNING, thr_id, "minotaurx differential: CPU reference refused its parameters");
			return;
		}
		const unsigned long long q = ((unsigned long long) h[7] << 32) | h[6];
		c[0] ^= q;
		c[1] ^= q * (2ull * (unsigned long long) nonce + 1ull);
	}

	if (g[0] == c[0] && g[1] == c[1])
		gpulog(LOG_DEBUG, thr_id, "minotaurx differential ok: %u nonces from %08x, "
		       "acc %016llx/%016llx", MX_DIFF_SPAN, start, g[0], g[1]);
	else
		gpulog(LOG_ERR, thr_id, "minotaurx DIFFERENTIAL MISMATCH over %u nonces from %08x "
		       "(acc0 %s, acc1 %s): gpu %016llx/%016llx != cpu %016llx/%016llx",
		       MX_DIFF_SPAN, start, g[0] == c[0] ? "same" : "MOVED",
		       g[1] == c[1] ? "same" : "MOVED", g[0], g[1], c[0], c[1]);
}

extern "C" int scanhash_minotaurx(int thr_id, struct work *work, uint32_t max_nonce,
                                  unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t vhash[8];
	uint32_t n = first_nonce;
	const int dev = device_map[thr_id];
	time_t t_start;

	/* the benchmark rescans one small window, so make a hit in it likely */
	if (opt_benchmark)
		ptarget[7] = 0x07ffffff;

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	if (!init[dev]) {
		const uint32_t shneed = YP_SBOX_UINT4 * 16u;
		int optin = 0;
		CUDA_CALL_OR_RET_X(cudaSetDevice(device_map[thr_id]), -1);
		cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
		                       device_map[thr_id]);

		const size_t bw      = 32u * (size_t) MX_R;
		const size_t v_inst  = bw * 4 * (size_t) MX_N;
		const size_t bx_inst = bw * 4 * 2;
		const size_t in_inst = MX_IN_WORDS * 4;
		const size_t per_inst = v_inst + bx_inst + in_inst + shneed;
		size_t avail = (size_t) cuda_available_memory(thr_id) * 1024u * 1024u;
		avail = (avail > (256u << 20)) ? (avail - (256u << 20)) : 0u;
		uint32_t fit = (uint32_t) (avail / per_inst);

		/* global S whenever it fits, as in yespower.cu */
		mx_sglobal[dev] = fit >= (uint32_t) device_mpcount[dev] * 4u || (uint32_t) optin < shneed;

		if (!mx_sglobal[dev]) {
			if (cudaFuncSetAttribute(minotaurx_gpu_hash<PWX_S_SHARED>,
			        cudaFuncAttributeMaxDynamicSharedMemorySize, shneed) != cudaSuccess) {
				applog(LOG_ERR, "minotaurx: could not raise the dynamic shared limit to %u B", shneed);
				proper_exit(EXIT_CODE_CUDA_ERROR);
			}
			mx_instances[dev] = (uint32_t) device_mpcount[dev];
		} else {
			const uint32_t ipb = (device_sm[dev] >= 700) ? MX_IPB_AMPERE : MX_IPB_OTHER;
			const uint32_t want = (uint32_t) device_mpcount[dev] * ipb;
			fit -= fit % 4u;
			mx_instances[dev] = (want < fit) ? want : fit;
			if (mx_instances[dev] < 4u) {
				applog(LOG_ERR, "minotaurx: GPU #%d needs %.0f MB per instance and has %d MB free",
				       device_map[thr_id], (double) per_inst / (1024.0 * 1024.0),
				       cuda_available_memory(thr_id));
				proper_exit(EXIT_CODE_CUDA_ERROR);
			}
			if (mx_instances[dev] < want)
				applog(LOG_WARNING, "minotaurx: GPU #%d VRAM caps this at %u instances "
				       "(wanted %u) -- expect a lower rate", device_map[thr_id],
				       mx_instances[dev], want);
		}
		/* the self-test launches 6 instances */
		if (mx_instances[dev] < 6u) mx_instances[dev] = 6u;

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_V[dev], v_inst * mx_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_B[dev], bw * 4 * mx_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_X[dev], bw * 4 * mx_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_in[dev], in_inst * mx_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_res[dev], (1u + MX_MAX_RES) * sizeof(uint32_t)), -1);
		if (mx_sglobal[dev])
			CUDA_CALL_OR_RET_X(cudaMalloc(&d_S[dev], (size_t) shneed * mx_instances[dev]), -1);
		h_in[dev] = (uint32_t *) malloc(in_inst * mx_instances[dev]);
		if (!h_in[dev]) {
			applog(LOG_ERR, "minotaurx: out of host memory");
			return -1;
		}

		{
			uint8_t persbuf[YP_PERS_MAX] = { 0 };
			uint32_t plen = (uint32_t) (sizeof(MX_PERS) - 1);
			memcpy(persbuf, MX_PERS, plen);
			cudaMemcpyToSymbol(c_yp_pers, persbuf, YP_PERS_MAX);
			cudaMemcpyToSymbol(c_yp_perslen, &plen, sizeof(plen));
		}

		applog(LOG_INFO, "GPU #%d: minotaurx %u instances, %.0f MB of V, S in %s",
		       device_map[thr_id], mx_instances[dev],
		       (double) (v_inst * (double) mx_instances[dev]) / (1024.0 * 1024.0),
		       mx_sglobal[dev] ? "global" : "shared");
		minotaurx_device_selftest(thr_id, dev);
		init[dev] = true;
	}

	cudaMemcpyToSymbol(c_mx_target, ptarget, 32);

	/* before the first chain batch: it borrows h_in/d_in */
	if (opt_debug)
		mx_debug_differential(thr_id, dev, pdata, endiandata, first_nonce);

	const uint32_t batch = mx_instances[dev];
	t_start = time(NULL);

	/* Later batches are chained while the kernel runs.  Batch k+1 is uploaded
	 * after batch k is read back, so one device input buffer is enough. */
	mx_chain_batch(dev, endiandata, n, batch);

	do {
		uint32_t res[1u + MX_MAX_RES] = { 0 };

		if (cudaMemcpy(d_in[dev], h_in[dev], (size_t) batch * MX_IN_WORDS * 4,
		               cudaMemcpyHostToDevice) != cudaSuccess ||
		    cudaMemcpy(d_res[dev], res, sizeof(res), cudaMemcpyHostToDevice) != cudaSuccess) {
			applog(LOG_ERR, "minotaurx: GPU #%d upload failed", device_map[thr_id]);
			return -1;
		}
		mx_launch(dev, batch, n, NULL);

		/* overlapped with the kernel */
		mx_chain_batch(dev, endiandata, n + batch, batch);

		if (cudaGetLastError() != cudaSuccess ||
		    cudaMemcpy(res, d_res[dev], sizeof(res), cudaMemcpyDeviceToHost) != cudaSuccess) {
			applog(LOG_ERR, "minotaurx: GPU #%d launch failed: %s",
			       device_map[thr_id], cudaGetErrorString(cudaGetLastError()));
			return -1;
		}

		*hashes_done = n - first_nonce + batch;

		if (res[0] != 0u) {
			uint32_t cand[MX_MAX_RES];
			uint32_t ncand = res[0];
			int found = 0;

			if (ncand > MX_MAX_RES) {
				applog(LOG_WARNING, "GPU #%d: minotaurx candidates flood: %u (keeping %u)",
				       device_map[thr_id], ncand, (uint32_t) MX_MAX_RES);
				ncand = MX_MAX_RES;
			}
			for (uint32_t s = 0; s < ncand; s++) cand[s] = res[1u + s];
			/* ascending: the resume cursor needs the two lowest */
			for (uint32_t a = 1; a < ncand; a++) {
				const uint32_t v = cand[a];
				uint32_t b = a;
				while (b > 0 && cand[b - 1] > v) { cand[b] = cand[b - 1]; b--; }
				cand[b] = v;
			}

			for (uint32_t s = 0; s < ncand && found < 2; s++) {
				be32enc(&endiandata[19], cand[s]);
				if (!minotaurx_hash(vhash, endiandata)) {
					applog(LOG_ERR, "minotaurx: host re-verify failed at nonce %08x", cand[s]);
					return -1;
				}
				if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
					work->nonces[found] = cand[s];
					if (found == 0) work_set_target_ratio(work, vhash);
					else            bn_set_target_ratio(work, vhash, found);
					found++;
				} else {
					gpu_increment_reject(thr_id);
					applog(LOG_WARNING, "GPU #%d: minotaurx result %08x does not validate",
					       device_map[thr_id], cand[s]);
				}
			}
			if (found) {
				/* nonces[0] is what is submitted; resume past the highest one. */
				work->valid_nonces = found;
				pdata[19] = ((found > 1 && work->nonces[1] > work->nonces[0])
				             ? work->nonces[1] : work->nonces[0]) + 1;
				return found;
			}
		}

		n += batch;
		if (time(NULL) - t_start >= 1)
			break;                       /* yield: report the rate, poll for work */
	} while ((uint64_t) n + batch < (uint64_t) max_nonce &&
	         !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce;
	pdata[19] = n;
	return 0;
}

/* Must stay registered in algo_free_all(): an algo switch re-arms init here. */
extern "C" void free_minotaurx(int thr_id)
{
	const int dev = device_map[thr_id];

	if (!init[dev])
		return;

	cudaSetDevice(dev);
	cudaDeviceSynchronize();

	if (d_V[dev])   cudaFree(d_V[dev]);
	if (d_B[dev])   cudaFree(d_B[dev]);
	if (d_X[dev])   cudaFree(d_X[dev]);
	if (d_S[dev])   cudaFree(d_S[dev]);
	if (d_in[dev])  cudaFree(d_in[dev]);
	if (d_res[dev]) cudaFree(d_res[dev]);
	free(h_in[dev]);

	d_V[dev] = NULL; d_B[dev] = NULL; d_X[dev] = NULL; d_S[dev] = NULL;
	d_in[dev] = NULL; d_res[dev] = NULL; h_in[dev] = NULL;

	mx_instances[dev] = 0;
	mx_sglobal[dev]   = false;
	mx_diff_time[dev] = 0;
	init[dev] = false;
}
