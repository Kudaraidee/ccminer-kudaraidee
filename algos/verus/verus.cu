/*
 * VerusHash 2.2 (Verus Coin / PBaaS) - ccminer scan loop, kernel launch and
 * self-tests.
 *
 * One hash per warp (verus_warp.cuh): the selector of the CLHash core is
 * warp-uniform, so every branch is uniform and the lanes split the AES, CLMUL
 * and unpack primitives. The 8832-byte key is never copied per hash: every
 * hash of a job reads one pristine key in shared memory through a copy-on-write
 * overlay of at most 64 slots.
 *
 * The host does the job-constant prologue (46 x Haraka512 + 276 x Haraka256,
 * AES-NI, verus_host.c), the device the per-nonce part, and every candidate is
 * re-hashed in full on the host before it is submitted.
 *
 * Preimage = 140-byte header || fd 40 05 || 1344-byte solution (from the pool),
 * PBaaS canonical form: prevhash, merkle, sapling root, nBits, nNonce and the two
 * MMR roots are zeroed, and the miner varies only the last 15 solution bytes:
 *   [0..6]  header bytes 108..114 (pool extranonce1)
 *   [7..10] work->data[32]       (per-thread roll, never in the header submit)
 *   [11..14] the 32-bit counter work->data[30], little-endian
 * The solution template lives in work->extra as CompactSize + bytes.
 */
#include "miner.h"
#include "cuda/helper.cuh"

#include <string.h>
#ifndef _WIN32
#include <unistd.h>   /* usleep; Windows has it from compat.h */
#endif
#include <stdlib.h>
#include <time.h>

#include "cuda/selftest_gate.cuh"
#include "cuda/candidate_report.cuh"

#include "verus_warp.cuh"
#include "verus_host.h"
#include "verus-kat.h"

#define VR_PREIMAGE   (VERUS_BASE_SIZE + VERUS_SOLUTION_FIXED)   /* 1487 */
#define VR_HALF_LEN   (VR_PREIMAGE - VERUS_NONCE_SPACE)          /* 1472 */
#define VR_DIFF_SPAN  4133u   /* -D differential: not a multiple of any block size */

/* Launch shape per arch: the register budget decides how many hashes an SM
 * keeps in flight.
 *   sm_61: 12 warps x 3 blocks, <= 56 registers (96 KB shared per SM)
 *   sm_7x: 24 warps x 1 block  (64 KB shared per SM caps the warps per block)
 *   sm_8x: 32 warps x 1 block, <= 64 registers
 * The host must launch with exactly these block sizes (vr_shape). */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define VR_TPB  384
#define VR_MINB 3
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define VR_TPB  768
#define VR_MINB 1
#else
#define VR_TPB  1024
#define VR_MINB 1
#endif

static void vr_shape(int dev, int *tpb, int *blocks, size_t *smem)
{
	const long sm = device_sm[dev];
	int t, b;
	if (sm < 700)      { t = 384;  b = 3; }
	else if (sm < 800) { t = 768;  b = 1; }
	else               { t = 1024; b = 1; }
	*tpb = t;
	*blocks = device_mpcount[dev] * b;
	*smem = 1024u * (size_t) vw_aes_tables((int) (sm / 100)) + VP_KEY_SLOTS * 16u +
	        (size_t) (t / 32) * VW_WARP_SMEM;
}

/* job: curBuf as 16 little-endian words, half[64] (VerusHashHalf output) with
 * the 11 job-constant nonce bytes over bytes 32..42; the counter bytes 43..46
 * are zero and filled in per nonce */
struct vr_job { uint32_t cb[16]; };

/* curBuf for counter n. volatile, so the reload after clhash is not merged
 * with the first load. */
__device__ __forceinline__ void vr_load_cb(uint32_t cb[16], const vr_job *job, uint32_t n)
{
	for (int k = 0; k < 4; k++) {
		uint32_t x, y, z, w;
		asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
		             : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(job->cb + 4 * k));
		cb[4 * k] = x; cb[4 * k + 1] = y; cb[4 * k + 2] = z; cb[4 * k + 3] = w;
	}
	cb[10] |= n << 24;
	cb[11] |= n >> 8;
}

/* dynamic shared: AES tables | pristine key | per warp: ov[64][4] idx[64] map[512] */
__global__ void __launch_bounds__(VR_TPB, VR_MINB)
verus_gpu_hash(const uint32_t *gK, const uint32_t *gT0, const vr_job *job,
               uint32_t first, uint32_t count, uint32_t target7,
               uint32_t *res, uint32_t *w7out)
{
	extern __shared__ __align__(16) uint8_t smem[];
	uint32_t *sT = (uint32_t *) smem;
	uint32_t *sK = sT + VW_T_WORDS;
	uint8_t *wsm = (uint8_t *) (sK + VP_KEY_SLOTS * 4);

	/* table t = T0 rotated left by 8t (the row-t contribution of a column) */
	for (int i = threadIdx.x; i < VW_T_WORDS; i += blockDim.x) {
		const uint32_t v = gT0[i & 255], r = 8 * (i >> 8);
		sT[i] = r ? (v << r) | (v >> (32 - r)) : v;
	}
	for (int i = threadIdx.x; i < VP_KEY_SLOTS * 4; i += blockDim.x) sK[i] = gK[i];
	__syncthreads();

	uint8_t *wbase = wsm + (threadIdx.x >> 5) * VW_WARP_SMEM;
	uint32_t *ov  = (uint32_t *) wbase;
	uint16_t *idx = (uint16_t *) (wbase + VP_OVL_SLOTS * 16);
	uint8_t  *map = wbase + VP_OVL_SLOTS * 18;
	vw_key o;
	vw_key_init(&o, sK, map, idx, ov);

	const uint32_t gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	const uint32_t nw = (gridDim.x * blockDim.x) >> 5;
	for (uint32_t i = gw; i < count; i += nw) {      /* warp-uniform */
		const uint32_t n = first + i;
		uint32_t cb[16];
		uint8_t out[32];
		vr_load_cb(cb, job, n);
		const uint64_t inter = vp_clhash(&o, cb, sT, (vp_stats *) 0);
		vr_load_cb(cb, job, n);        /* reloaded: no registers held across clhash */
		vp_hash_tail(out, 0, cb, inter, &o, sT, (vp_stats *) 0);
		if (vw_lane() == 0) {
			const uint32_t w7 = (uint32_t) out[28] | (uint32_t) out[29] << 8 |
			                    (uint32_t) out[30] << 16 | (uint32_t) out[31] << 24;
			if (w7out) w7out[i] = w7;
			if (w7 <= target7) report_candidate_2(res, n);
		}
	}
}

/* ------------------------------------------------------------------------- */

static bool init[MAX_GPUS] = { false };
static uint32_t *d_K[MAX_GPUS], *d_T[MAX_GPUS], *d_res[MAX_GPUS], *d_w7[MAX_GPUS];
static vr_job *d_job[MAX_GPUS];
static int vr_tpb[MAX_GPUS], vr_blocks[MAX_GPUS];
static size_t vr_smem[MAX_GPUS];

/* the job-constant host state, per device (one miner thread per device);
 * keybuf is read with aligned SSE loads */
struct alignas(16) vr_host {
	uint8_t keybuf[VERUS_KEYBUF_BYTES];
	uint8_t half[64];
	uint8_t prefix[VR_HALF_LEN];
	bool    valid;
	bool    warned;
};
static vr_host *vr_h[MAX_GPUS];

static uint32_t vr_le32(const uint8_t *p)
{
	return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}

/* nonceSpace for counter n: base[0..10] then n little-endian */
static void vr_nonce(uint8_t nonce[15], const uint8_t base[11], uint32_t n)
{
	memcpy(nonce, base, 11);
	nonce[11] = (uint8_t) n;         nonce[12] = (uint8_t) (n >> 8);
	nonce[13] = (uint8_t) (n >> 16); nonce[14] = (uint8_t) (n >> 24);
}

static bool vr_set_base(int dev, const uint8_t half[64], const uint8_t base[11])
{
	uint8_t b[64];
	vr_job j;
	memcpy(b, half, 64);
	memcpy(b + 32, base, 11);
	memset(b + 43, 0, 4);
	for (int k = 0; k < 16; k++) j.cb[k] = vr_le32(b + 4 * k);
	return cudaMemcpy(d_job[dev], &j, sizeof j, cudaMemcpyHostToDevice) == cudaSuccess;
}

static bool vr_upload(int dev, const uint8_t *keybuf, const uint8_t half[64], const uint8_t base[11])
{
	return cudaMemcpy(d_K[dev], keybuf, VP_KEY_SLOTS * 16, cudaMemcpyHostToDevice) == cudaSuccess &&
	       vr_set_base(dev, half, base);
}

/* res[0..1] are armed here; w7out (optional) receives word 7 of every hash */
static bool vr_launch(int dev, uint32_t first, uint32_t count, uint32_t target7, uint32_t *w7out)
{
	if (cudaMemset(d_res[dev], 0xff, 2 * sizeof(uint32_t)) != cudaSuccess)
		return false;
	verus_gpu_hash<<<vr_blocks[dev], vr_tpb[dev], vr_smem[dev]>>>(
		d_K[dev], d_T[dev], d_job[dev], first, count, target7, d_res[dev], w7out);
	return cudaGetLastError() == cudaSuccess;
}

static uint64_t s_rng;
static uint64_t vr_splitmix(void)
{
	uint64_t z = (s_rng += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

/* Device word 7 against the host over `span` nonces of the job in h. */
static bool vr_compare(int dev, vr_host *h, const uint8_t base[11], uint32_t first,
                       uint32_t span, uint32_t *bad)
{
	uint32_t *w7 = (uint32_t *) malloc(span * sizeof(uint32_t));
	if (!w7) return false;
	bool ran = vr_launch(dev, first, span, 0, d_w7[dev]) &&
	           cudaMemcpy(w7, d_w7[dev], span * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess;
	*bad = 0;
	for (uint32_t i = 0; ran && i < span; i++) {
		uint8_t nonce[15], hash[32] = { 0 };
		vr_nonce(nonce, base, first + i);
		verus_host_hash(hash, h->half, nonce, h->keybuf, 0);
		if (vr_le32(hash + 28) != w7[i]) (*bad)++;
	}
	free(w7);
	return ran;
}

/* Fail-closed self-test:
 *   1. host chain against the KAT block's published hash
 *   2. device, same block
 *   3. negative: a flipped bit must change the device result
 *   4. device against host on a fixed random job, 1024 nonces
 * Leg 4 is needed because the KAT block is a degenerate job: its first selector
 * repeats for all 32 iterations, so it runs only one of the 8 mixer cases. */
static void verus_device_selftest(int thr_id, int dev)
{
	static vr_host t;      /* this function's own state */
	uint8_t pre[VR_PREIMAGE], base[11], hash[32];
	bool host_ok, kat_ok = false, rnd_ok = false, neg_ok = false, ran = true;
	uint32_t bad = 0, w7 = 0;

	/* leg 1 */
	memcpy(pre, verus_kat_block, VR_PREIMAGE);
	verus_host_clear_noncanonical(pre);
	host_ok = verus_host_full(hash, pre, VR_PREIMAGE) && !memcmp(hash, verus_kat_hash, 32);

	/* leg 2: the block's own nonce, target open */
	if (verus_host_prologue(pre, VR_HALF_LEN, t.half, t.keybuf)) {
		const uint32_t n = vr_le32(pre + VR_HALF_LEN + 11);
		uint32_t res[2] = { 0, 0 };
		memcpy(base, pre + VR_HALF_LEN, 11);
		ran = vr_upload(dev, t.keybuf, t.half, base) &&
		      vr_launch(dev, n, 1, 0xffffffffu, d_w7[dev]) &&
		      cudaMemcpy(&w7, d_w7[dev], 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
		      cudaMemcpy(res, d_res[dev], sizeof res, cudaMemcpyDeviceToHost) == cudaSuccess;
		kat_ok = ran && w7 == vr_le32(verus_kat_hash + 28) && res[0] == n;

		/* leg 3: one bit of the half state flipped */
		uint8_t badhalf[64];
		memcpy(badhalf, t.half, 64);
		badhalf[3] ^= 0x10;
		uint32_t w7n = w7;
		ran = ran && vr_upload(dev, t.keybuf, badhalf, base) &&
		      vr_launch(dev, n, 1, 0, d_w7[dev]) &&
		      cudaMemcpy(&w7n, d_w7[dev], 4, cudaMemcpyDeviceToHost) == cudaSuccess;
		neg_ok = ran && w7n != w7;
	}

	/* leg 4: reaches every mixer case */
	s_rng = 0x2545f4914f6cdd1dULL;
	for (int i = 0; i < VR_PREIMAGE; i += 8) {
		const uint64_t z = vr_splitmix();
		memcpy(pre + i, &z, VR_PREIMAGE - i < 8 ? VR_PREIMAGE - i : 8);
	}
	pre[140] = 0xfd; pre[141] = 0x40; pre[142] = 0x05;
	if (ran && verus_host_prologue(pre, VR_HALF_LEN, t.half, t.keybuf)) {
		memcpy(base, pre + VR_HALF_LEN, 11);
		ran = vr_upload(dev, t.keybuf, t.half, base) && vr_compare(dev, &t, base, 0, 1024, &bad);
		rnd_ok = ran && bad == 0;
	}
	if (!ran) selftest_cuda_fault();

	if (vr_h[dev]) vr_h[dev]->valid = false;   /* the device job is the test's now */

	const bool passed = host_ok && kat_ok && rnd_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "verus self-test FAILED (host KAT %s, device KAT %s, "
		       "random job %u/1024 wrong, negative %s)", host_ok ? "ok" : "BAD",
		       kat_ok ? "ok" : "BAD", bad, neg_ok ? "ok" : "BAD");
	else if (opt_debug)
		gpulog(LOG_DEBUG, thr_id, "verus self-test OK (block %d KAT host + device, "
		       "1024-nonce random job, negative)", VERUS_KAT_HEIGHT);
	selftest_gate(thr_id, "verus", passed);
}

/* -D: a fresh job's first VR_DIFF_SPAN nonces, device word 7 against the host */
static void vr_debug_differential(int thr_id, int dev, vr_host *h, const uint8_t base[11],
                                  uint32_t first)
{
	uint32_t bad = 0;
	if (!vr_compare(dev, h, base, first, VR_DIFF_SPAN, &bad))
		gpulog(LOG_WARNING, thr_id, "verus differential could not run");
	else if (bad)
		gpulog(LOG_ERR, thr_id, "verus DIFFERENTIAL MISMATCH: %u of %u nonces from %08x",
		       bad, VR_DIFF_SPAN, first);
	else
		gpulog(LOG_DEBUG, thr_id, "verus differential ok: %u nonces from %08x", VR_DIFF_SPAN, first);
}

extern "C" int scanhash_verus(int thr_id, struct work *work, uint32_t max_nonce,
                              unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[30];
	const int dev = device_map[thr_id];
	uint8_t pre[VR_PREIMAGE], base[11];
	uint32_t n = first_nonce;
	time_t t_start;

	*hashes_done = 0;
	if (n >= max_nonce)
		return 0;
	/* Benchmark target: about one candidate per 2^24 nonces, so the host
	 * re-verify runs without distorting the rate. The lower words are open
	 * because the device screens word 7 only. */
	if (opt_benchmark) {
		for (int k = 0; k < 7; k++) ptarget[k] = 0xffffffffu;
		ptarget[7] = 0x000000ff;
	}

	if (!init[dev]) {
		if (!verus_host_cpu_ok()) {
			applog(LOG_ERR, "verus: this CPU lacks AES-NI/PCLMULQDQ/SSSE3, which the "
			       "job prologue and the share re-verify need");
			proper_exit(EXIT_CODE_SW_INIT_ERROR);
		}
		CUDA_CALL_OR_RET_X(cudaSetDevice(dev), -1);
		vr_shape(dev, &vr_tpb[dev], &vr_blocks[dev], &vr_smem[dev]);
		if (cudaFuncSetAttribute(verus_gpu_hash, cudaFuncAttributeMaxDynamicSharedMemorySize,
		                         (int) vr_smem[dev]) != cudaSuccess) {
			applog(LOG_ERR, "verus: GPU #%d cannot give %u B of shared memory per block",
			       dev, (unsigned) vr_smem[dev]);
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}
		uint32_t T0[256];
		vp_build_t0(T0);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_K[dev], VP_KEY_SLOTS * 16), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_T[dev], sizeof T0), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_job[dev], sizeof(vr_job)), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_res[dev], 2 * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_w7[dev], VR_DIFF_SPAN * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMemcpy(d_T[dev], T0, sizeof T0, cudaMemcpyHostToDevice), -1);
		if (!vr_h[dev]) {
			vr_h[dev] = (vr_host *) calloc(1, sizeof(vr_host) + 64);
			if (!vr_h[dev]) { applog(LOG_ERR, "verus: out of host memory"); return -1; }
		}
		vr_h[dev]->valid = false;
		applog(LOG_INFO, "GPU #%d: verus %d warps x %d blocks, %u B shared per block",
		       dev, vr_tpb[dev] / 32, vr_blocks[dev], (unsigned) vr_smem[dev]);
		verus_device_selftest(thr_id, dev);
		init[dev] = true;
	}
	vr_host *h = vr_h[dev];

	/* the preimage: header from work->data, solution template from work->extra */
	const uint8_t *ex = work->extra;
	const int sol_len = ex[0] == 0xfd ? (int) (ex[1] | ex[2] << 8) : 0;
	if (opt_benchmark && sol_len == 0) {
		/* no pool: the KAT block (a real PBaaS solution) with one nTime bit
		 * flipped; the block itself runs one mixer case only and would
		 * overstate the rate */
		memcpy(pre, verus_kat_block, VR_PREIMAGE);
		pre[100] ^= 1;
		memcpy(base, verus_kat_block + VR_HALF_LEN, 11);
	} else {
		if (sol_len != VERUS_SOLUTION_FIXED) {
			if (!h->warned)
				gpulog(LOG_WARNING, thr_id, "verus: no usable pool solution in this job (%d bytes)",
				       sol_len);
			h->warned = true;
			usleep(100 * 1000);
			return 0;
		}
		memcpy(pre, pdata, VERUS_HEADER_SIZE);
		memcpy(pre + VERUS_HEADER_SIZE, ex, 3 + VERUS_SOLUTION_FIXED);
		memcpy(base, ((const uint8_t *) pdata) + 108, 7);
		memcpy(base + 7, &pdata[32], 4);
	}
	{
		const uint8_t *sol = pre + VERUS_BASE_SIZE;
		if (!(sol[0] >= 7 && sol[5] > 0)) {
			/* not a PBaaS job: solution versions below 7 hash with older VerusHash
			 * variants, and nothing here could be checked against a real block */
			if (!h->warned)
				gpulog(LOG_ERR, thr_id, "verus: job is not in PBaaS form (solution version %u, "
				       "%u PBaaS headers) - not mining it", sol[0], sol[5]);
			h->warned = true;
			usleep(100 * 1000);
			return 0;
		}
	}
	h->warned = false;
	verus_host_clear_noncanonical(pre);

	/* the prologue only when the 1472 job-constant bytes change */
	const bool fresh = !h->valid || memcmp(h->prefix, pre, VR_HALF_LEN) != 0;
	if (fresh) {
		memcpy(h->prefix, pre, VR_HALF_LEN);
		if (!verus_host_prologue(pre, VR_HALF_LEN, h->half, h->keybuf) ||
		    !vr_upload(dev, h->keybuf, h->half, base)) {
			applog(LOG_ERR, "verus: GPU #%d job upload failed", dev);
			return -1;
		}
		h->valid = true;
		if (opt_debug)
			vr_debug_differential(thr_id, dev, h, base, first_nonce);
	} else if (!vr_set_base(dev, h->half, base)) {
		applog(LOG_ERR, "verus: GPU #%d job upload failed", dev);
		return -1;
	}

	const uint32_t batch = cuda_default_throughput(thr_id, 1U << 20);
	t_start = time(NULL);

	do {
		const uint32_t cnt = (uint64_t) n + batch > (uint64_t) max_nonce ? max_nonce - n : batch;
		uint32_t res[2];
		if (!vr_launch(dev, n, cnt, ptarget[7], NULL) ||
		    cudaMemcpy(res, d_res[dev], sizeof res, cudaMemcpyDeviceToHost) != cudaSuccess) {
			applog(LOG_ERR, "verus: GPU #%d launch failed: %s", dev,
			       cudaGetErrorString(cudaGetLastError()));
			return -1;
		}

		if (res[0] != UINT32_MAX) {
			int found = 0;
			for (int s = 0; s < 2 && res[s] != UINT32_MAX; s++) {
				uint8_t nonce[15];
				uint32_t vhash[8];
				vr_nonce(nonce, base, res[s]);
				memcpy(pre + VR_HALF_LEN, nonce, VERUS_NONCE_SPACE);
				if (!verus_host_full((uint8_t *) vhash, pre, VR_PREIMAGE)) {
					applog(LOG_ERR, "verus: host re-verify failed at nonce %08x", res[s]);
					return -1;
				}
				if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
					work->nonces[found] = res[s];
					if (found == 0) work_set_target_ratio(work, vhash);
					else            bn_set_target_ratio(work, vhash, found);
					found++;
				} else {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result %08x does not validate on CPU!", res[s]);
				}
			}
			/* res[0] < res[1]; anything above the resume point is rescanned */
			const uint32_t hi = res[1] != UINT32_MAX ? res[1] : res[0];
			if (found) {
				work->valid_nonces = found;
				pdata[30] = hi + 1;
				*hashes_done = pdata[30] - first_nonce;
				return found;
			}
			n = hi + 1;
		} else {
			n += cnt;
		}
		*hashes_done = n - first_nonce;
		if (time(NULL) - t_start >= 1)
			break;                       /* yield: report the rate, poll for work */
	} while ((uint64_t) n + 1 < (uint64_t) max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce;
	pdata[30] = n;
	return 0;
}

/* Must stay registered in algo_free_all(): an algo switch re-arms init here. */
extern "C" void free_verus(int thr_id)
{
	const int dev = device_map[thr_id];
	if (!init[dev])
		return;
	cudaSetDevice(dev);
	cudaDeviceSynchronize();
	cudaFree(d_K[dev]); cudaFree(d_T[dev]); cudaFree(d_job[dev]);
	cudaFree(d_res[dev]); cudaFree(d_w7[dev]);
	d_K[dev] = d_T[dev] = d_res[dev] = d_w7[dev] = NULL;
	d_job[dev] = NULL;
	if (vr_h[dev]) vr_h[dev]->valid = false;
	init[dev] = false;
	cudaDeviceSynchronize();
}
