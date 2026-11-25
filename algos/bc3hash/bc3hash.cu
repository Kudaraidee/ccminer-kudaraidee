/*
 * bc3hash (BitcoinIII BC3, from block 64 360) -- GPU scanhash.
 *
 * bc3hash(header80) = SHA3-256 over five tagged hashes of the header and
 * bc3mix512(SHA3-512 of it); bc3mix512 is >99% of the work (fixed-exponent
 * modexps, fixed-base powers, 64 secp256k1 fixed-base mults, 64 inversions).
 * The device code in cuda/ is a transcription of the coin's crypto/bc3; the host
 * oracle (bc3hash_oracle.cpp, the coin's own code) re-verifies every candidate,
 * so a kernel defect can only cause a local reject, never a bad share.
 *
 * One thread per nonce, state in global memory between kernels:
 *   k_front_ab : front hashes + mix phases A/B
 *   k_c_exp1, k_c_exp2, k_c_ec : mix phase C, three launches per round (16 rounds)
 *   k_d_final  : mix phase D + final SHA3-256 + target compare + report
 *
 * Tables per device: pow64/128/256 (336 KB, built by the oracle) and four
 * secp256k1 window tables for H, J, K, L (built on the device; 16- or 20-bit
 * windows, 256 MB or 3 GB).
 */

#include <algorithm>
#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/bc3_hashes_device.cuh"
#include "cuda/bc3_mix_device.cuh"
#include "../../cuda/candidate_report.cuh"
#include "../../cuda/selftest_gate.cuh"

extern "C" void bc3hash_hash(void* output, const void* input80);
extern "C" const void* bc3hash_pow_table(int which, size_t* bytes);
extern "C" int bc3hash_window_bases(int point, int wbits, int count, unsigned char* out);
extern "C" int bc3hash_curve_ref(int point, const uint32_t* k, uint32_t* x, uint32_t* y);

#define BC3_TPB 64
/* d_res: the two lowest candidates, the count, then up to BC3_SLOTS candidates */
#define BC3_SLOTS MAX_NONCES
#define BC3_RES_WORDS (3 + BC3_SLOTS)

/* Pipeline state per nonce, SoA by 32-bit word: U256 slots h1, h2, m256[48..63],
 * j1..j4 of the current phase C round, and the 57 words of the front digests the
 * final hash needs (s384 | sh | r160 | bs | bb). */
#define BC3_ST_SLOTS 22
#define BC3_FRONT_WORDS 57

static bool init_done[MAX_GPUS] = { 0 };
static uint8_t* d_curve[MAX_GPUS][4];
static int ec_w[MAX_GPUS]; /* EC window width: 16 or 20 */
static void* d_pow[MAX_GPUS][3];
static bc3u256* d_pow_w[MAX_GPUS]; /* F256 powers with pow_rb-bit rows (NULL for 8) */
static int pow_rb[MAX_GPUS];
static uint32_t* d_state[MAX_GPUS];
static uint32_t* d_front[MAX_GPUS];
static uint32_t* d_res[MAX_GPUS];
static uint32_t max_batch[MAX_GPUS];
static uint32_t wave[MAX_GPUS]; /* nonces per k_d_final wave (0: unknown) */

/* header bytes 0..75 (be32enc of pdata[0..18]) as words, and the full target */
static __constant__ uint32_t c_header[19];
static __constant__ uint32_t c_target[8];

__device__ __forceinline__ void bc3_st_put(uint32_t* st, uint32_t n, int slot, uint32_t t, const bc3u256& v)
{
	#pragma unroll
	for (int i = 0; i < 8; i++) st[(size_t)(slot * 8 + i) * n + t] = v.w[i];
}

__device__ __forceinline__ bc3u256 bc3_st_get(const uint32_t* st, uint32_t n, int slot, uint32_t t)
{
	bc3u256 v;
	#pragma unroll
	for (int i = 0; i < 8; i++) v.w[i] = st[(size_t)(slot * 8 + i) * n + t];
	return v;
}

/* launch bounds on sm_70+ only (k_front_ab, k_d_final) */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define BC3_SM70_BOUNDS __launch_bounds__(BC3_TPB, 10)
#else
#define BC3_SM70_BOUNDS
#endif
__global__ void BC3_SM70_BOUNDS bc3_k_front_ab(bc3_tables tb, uint32_t threads, uint32_t startNonce, uint32_t* st, uint32_t* front)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= threads) return;
	const uint32_t nonce = startNonce + t;
	uint8_t hdr[80];
	#pragma unroll
	for (int i = 0; i < 19; i++) {
		const uint32_t w = c_header[i];
		hdr[4 * i] = (uint8_t)w;
		hdr[4 * i + 1] = (uint8_t)(w >> 8);
		hdr[4 * i + 2] = (uint8_t)(w >> 16);
		hdr[4 * i + 3] = (uint8_t)(w >> 24);
	}
	/* nonce hashed big-endian, matching the host's be32enc(&endiandata[19], nonce) */
	hdr[76] = (uint8_t)(nonce >> 24);
	hdr[77] = (uint8_t)(nonce >> 16);
	hdr[78] = (uint8_t)(nonce >> 8);
	hdr[79] = (uint8_t)nonce;

	bc3_front f;
	bc3h_front(f, hdr);
	const uint8_t* keep = f.s384; /* s384..bb are contiguous in bc3_front */
	for (int i = 0; i < BC3_FRONT_WORDS; i++)
		front[(size_t)i * threads + t] = bc3h_le32(keep + 4 * i);

	bc3u256 h1, h2, m[16];
	bc3_mix_ab(h1, h2, m, f.s512, tb);
	bc3_st_put(st, threads, 0, t, h1);
	bc3_st_put(st, threads, 1, t, h2);
	for (int i = 0; i < 16; i++) bc3_st_put(st, threads, 2 + i, t, m[i]);
}

/* Phase C, one round in three kernels; slots 18..21 carry j1..j4 between them.
 * The launch bounds keep each step's register count (and so occupancy) in check. */
__global__ void __launch_bounds__(BC3_TPB, 10) bc3_k_c_exp1(uint32_t threads, uint32_t* st)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= threads) return;
	bc3u256 j1, j2;
	bc3_mix_c_exp1(j1, j2, bc3_st_get(st, threads, 0, t), bc3_st_get(st, threads, 1, t));
	bc3_st_put(st, threads, 18, t, j1);
	bc3_st_put(st, threads, 19, t, j2);
}

__global__ void __launch_bounds__(BC3_TPB, 10) bc3_k_c_exp2(uint32_t threads, uint32_t* st, int round)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= threads) return;
	bc3u256 j3, j4;
	bc3_mix_c_exp2(j3, j4, bc3_st_get(st, threads, 1, t), bc3_st_get(st, threads, 18, t),
	               bc3_st_get(st, threads, 19, t), bc3_st_get(st, threads, 2 + round, t), round);
	bc3_st_put(st, threads, 20, t, j3);
	bc3_st_put(st, threads, 21, t, j4);
}

/* sm_61 with 16-bit windows runs faster at 128 registers, everything else at 168 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define BC3_EC_MINB(W) ((W) == 16 ? 8 : 5)
#else
#define BC3_EC_MINB(W) 5
#endif
template <int W>
__global__ void __launch_bounds__(BC3_TPB, BC3_EC_MINB(W)) bc3_k_c_ec(bc3_tables tb, uint32_t threads, uint32_t* st)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= threads) return;
	bc3u256 h1 = bc3_st_get(st, threads, 0, t), h2 = bc3_st_get(st, threads, 1, t);
	bc3_mix_c_ec<W>(h1, h2, bc3_st_get(st, threads, 18, t), bc3_st_get(st, threads, 19, t),
	             bc3_st_get(st, threads, 20, t), bc3_st_get(st, threads, 21, t), tb);
	bc3_st_put(st, threads, 0, t, h1);
	bc3_st_put(st, threads, 1, t, h2);
}

/* Final SHA3-256 over the front digests and the mix, target compare, report.
 * digests != NULL (self-test only): every digest is stored, as 32 bytes per nonce. */
__device__ __forceinline__ void bc3_final_tail(uint32_t t, uint32_t threads, uint32_t startNonce, const uint32_t* front,
                                               uint32_t* resNonce, uint8_t* digests, const uint8_t* mix)
{
	bc3_front f;
	uint8_t* keep = f.s384;
	for (int i = 0; i < BC3_FRONT_WORDS; i++) {
		const uint32_t w = front[(size_t)i * threads + t];
		keep[4 * i] = (uint8_t)w;
		keep[4 * i + 1] = (uint8_t)(w >> 8);
		keep[4 * i + 2] = (uint8_t)(w >> 16);
		keep[4 * i + 3] = (uint8_t)(w >> 24);
	}
	uint8_t d[32];
	bc3h_final(d, f, mix);
	if (digests)
		for (int i = 0; i < 32; i++) digests[32 * (size_t)t + i] = d[i];

	/* Core copies the digest bytes into a uint256 as-is: little-endian words,
	 * word 7 most significant, with no byte reversal. Full 256-bit compare. */
	bool below = true;
	for (int i = 7; i >= 0; i--) {
		const uint32_t h = bc3h_le32(d + 4 * i);
		if (h != c_target[i]) {
			below = h < c_target[i];
			break;
		}
	}
	if (below) {
		report_candidate_2(resNonce, startNonce + t);
		const uint32_t k = atomicAdd(&resNonce[2], 1u);
		if (k < BC3_SLOTS) resNonce[3 + k] = startNonce + t;
	}
}

template <int RB>
__global__ void BC3_SM70_BOUNDS bc3_k_d_final(bc3_tables tb, uint32_t threads, uint32_t startNonce, const uint32_t* st,
                              const uint32_t* front, uint32_t* resNonce, uint8_t* digests)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= threads) return;
	uint8_t mix[32];
	bc3_mix_d<RB>(mix, bc3_st_get(st, threads, 0, t), bc3_st_get(st, threads, 1, t), tb);
	bc3_final_tail(t, threads, startNonce, front, resNonce, digests, mix);
}

/* Phase D as one launch per round (u1, u2 in slots 0, 1), then the final hash;
 * used on sm_6x, sm_70+ runs k_d_final. */
template <int RB>
__global__ void __launch_bounds__(BC3_TPB, 10) bc3_k_d_round(bc3_tables tb, uint32_t threads, uint32_t* st, int i)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= threads) return;
	bc3u256 u1 = bc3_st_get(st, threads, 0, t), u2 = bc3_st_get(st, threads, 1, t), k1, k2;
	bc3_mix_d_pow<RB>(k1, k2, u1, u2, tb);
	bc3_mix_d_inv(u1, u2, k1, k2, i);
	bc3_st_put(st, threads, 0, t, u1);
	bc3_st_put(st, threads, 1, t, u2);
}
__global__ void bc3_k_final(uint32_t threads, uint32_t startNonce, const uint32_t* st, const uint32_t* front,
                            uint32_t* resNonce, uint8_t* digests)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= threads) return;
	uint8_t mix[32];
	bc3_mix_d_out(mix, bc3_st_get(st, threads, 0, t), bc3_st_get(st, threads, 1, t));
	bc3_final_tail(t, threads, startNonce, front, resNonce, digests, mix);
}

static bool d_split[MAX_GPUS]; /* phase D per round (sm_6x); BC3HASH_DSPLIT=0|1 overrides */

static bc3_tables bc3_device_tables(int thr_id)
{
	bc3_tables tb;
	tb.pow64 = (const bc3u64*)d_pow[thr_id][0];
	tb.pow128 = (const bc3u128*)d_pow[thr_id][1];
	tb.pow256 = pow_rb[thr_id] == 8 ? (const bc3u256*)d_pow[thr_id][2] : d_pow_w[thr_id];
	for (int p = 0; p < 4; p++) tb.curve[p] = d_curve[thr_id][p];
	return tb;
}

/* One batch through all kernels; leaves the candidates in d_res. */
static void bc3_launch(int thr_id, uint32_t threads, uint32_t startNonce, uint8_t* d_digests)
{
	const bc3_tables tb = bc3_device_tables(thr_id);
	const uint32_t grid = (threads + BC3_TPB - 1) / BC3_TPB;
	bc3_k_front_ab <<< grid, BC3_TPB >>> (tb, threads, startNonce, d_state[thr_id], d_front[thr_id]);
	for (int r = 0; r < 16; r++) {
		bc3_k_c_exp1 <<< grid, BC3_TPB >>> (threads, d_state[thr_id]);
		bc3_k_c_exp2 <<< grid, BC3_TPB >>> (threads, d_state[thr_id], r);
		if (ec_w[thr_id] == 20) bc3_k_c_ec<20> <<< grid, BC3_TPB >>> (tb, threads, d_state[thr_id]);
		else bc3_k_c_ec<16> <<< grid, BC3_TPB >>> (tb, threads, d_state[thr_id]);
	}
	uint32_t* st = d_state[thr_id];
	uint32_t* fr = d_front[thr_id];
	if (d_split[thr_id]) {
		for (int i = 0; i < 64; i++) switch (pow_rb[thr_id]) {
			case 12: bc3_k_d_round<12> <<< grid, BC3_TPB >>> (tb, threads, st, i); break;
			case 11: bc3_k_d_round<11> <<< grid, BC3_TPB >>> (tb, threads, st, i); break;
			default: bc3_k_d_round<8> <<< grid, BC3_TPB >>> (tb, threads, st, i); break;
			}
		bc3_k_final <<< grid, BC3_TPB >>> (threads, startNonce, st, fr, d_res[thr_id], d_digests);
		return;
	}
	switch (pow_rb[thr_id]) {
	case 12: bc3_k_d_final<12> <<< grid, BC3_TPB >>> (tb, threads, startNonce, st, fr, d_res[thr_id], d_digests); break;
	case 11: bc3_k_d_final<11> <<< grid, BC3_TPB >>> (tb, threads, startNonce, st, fr, d_res[thr_id], d_digests); break;
	default: bc3_k_d_final<8> <<< grid, BC3_TPB >>> (tb, threads, startNonce, st, fr, d_res[thr_id], d_digests); break;
	}
}

/* Default batch: a whole number of waves of the final and front kernels, short
 * enough per launch for the Windows TDR. */
static uint32_t bc3_default_batch(int thr_id)
{
	const int dev = device_map[thr_id];
	int sms = 0, bab = 0, bd = 0;
	cudaSetDevice(dev);
	if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
	    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bab, bc3_k_front_ab, BC3_TPB, 0) != cudaSuccess ||
	    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bd, bc3_k_d_final<8>, BC3_TPB, 0) != cudaSuccess ||
	    sms <= 0 || bab <= 0 || bd <= 0)
		return 1U << 16;

	/* k_d_final waves; a multiple of `step` also ends k_front_ab on whole waves */
	const int want = device_sm[dev] >= 800 ? 5 : 6;
	auto gcd = [](int a, int b) { while (b) { const int r = a % b; a = b; b = r; } return a; };
	const int step = bab / gcd(bab, bd);
	int waves = (want + step - 1) / step * step;
	if (waves > want + want / 2) waves = want;
	wave[thr_id] = (uint32_t)bd * BC3_TPB * sms;
	return (uint32_t)waves * wave[thr_id];
}

static void bc3_set_header(const uint8_t* header76, const uint32_t* target8)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_header, header76, 76, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_target, target8, 32, 0, cudaMemcpyHostToDevice));
}

/* ------------------------------------------------------------------ self-test */

/* BC3HASH_SELFTEST_FAULT=<leg> (tables, kat, report, neg, scan, cuda) breaks that one
 * check on purpose, to prove the gate fires. Testing only. */
static bool bc3_fault(int thr_id, const char* leg)
{
	const char* m = getenv("BC3HASH_SELFTEST_FAULT");
	if (!m || strcmp(m, leg)) return false;
	static bool warned = false;
	if (!warned) {
		gpulog(LOG_WARNING, thr_id, "bc3hash: SELF-TEST FAULT INJECTION ACTIVE (%s) -- for testing only", m);
		warned = true;
	}
	return true;
}

static int hexval(char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }
static void unhex(uint8_t* out, const char* s, int n)
{
	for (int i = 0; i < n; i++) out[i] = (uint8_t)(hexval(s[2 * i]) << 4 | hexval(s[2 * i + 1]));
}

/* The coin's 80-byte test vectors (two bc3hash headers, two reference rows). */
static const char* const kat_hdr[4] = {
	"00080020000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f00d2496bffff7f202a000000",
	"00180020000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f00d2496bffff7f202a000000",
	"d28b013747f66f8346796bf89e30680a51e28aaa9b615315571c1363218c446211bb8f609777a89394571669895e0f22c124d4d8ec0cee4b34fcbb5587289baac99cdfbe3960e4b4473764d38de89394",
	"2e3752eda11570b4ab3c630da852d2ce221dcc97c9bc3da05b22305ef28b2bb53d500ad7fdb7865efe8ae85846e2a0c0c5d6ec058541f5a7f6d2391d3cf62a4258645d10a0e744e2f251bccef7cd7078"};
static const char* const kat_digest[4] = {
	"eef9a6934aa1719f19bdea11824927d4cd58b0995033b99edb4bf2370d77160e",
	"102b13f7ba825a57fdd460fc6a7caaa473ea2cd7e312f5988d8d5c41438d3975",
	"5a8696fbdc18033a66704e835c2fd336f8629563ae4fdc1b783ccd207047b544",
	"b405f924e81156d1957784a38b362ba9bf52b83b37f486b6ff936c1a9fee16cd"};

/* Hash `count` nonces from `start` of one 76-byte header through the real scan
 * kernels, digests and the first `nres` words of d_res back to the host. */
static bool bc3_selftest_scan(int thr_id, const uint8_t* hdr76, const uint32_t* target8, uint32_t start,
                              uint32_t count, uint8_t* digests, uint32_t* res, int nres = 2)
{
	uint8_t* d_dig = NULL;
	if (cudaMalloc(&d_dig, 32 * (size_t)count) != cudaSuccess) return selftest_cuda_fault();
	bc3_set_header(hdr76, target8);
	const uint32_t arm[3] = { UINT32_MAX, UINT32_MAX, 0 };
	cudaMemcpy(d_res[thr_id], arm, sizeof(arm), cudaMemcpyHostToDevice);
	bc3_launch(thr_id, count, start, d_dig);
	const bool ok = cudaMemcpy(digests, d_dig, 32 * (size_t)count, cudaMemcpyDeviceToHost) == cudaSuccess &&
	                cudaMemcpy(res, d_res[thr_id], 4 * (size_t)nres, cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d_dig);
	return ok ? true : selftest_cuda_fault();
}

static bool bc3_device_selftest(int thr_id)
{
	const uint32_t all_ones[8] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
	bool ran = bc3_fault(thr_id, "cuda") ? selftest_cuda_fault() : true;

	/* tbl: pow tables read back byte-exact; each EC table spot-checked against
	 * the coin's curve_multiply at 64 entries spread over all windows. */
	bool tbl_ok = true;
	for (int w = 0; w < 3 && ran; w++) {
		size_t bytes;
		const void* h = bc3hash_pow_table(w, &bytes);
		void* rb = malloc(bytes);
		ran = rb && cudaMemcpy(rb, d_pow[thr_id][w], bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
		if (ran && w == 1 && bc3_fault(thr_id, "tables")) ((uint8_t*)rb)[bytes / 2] ^= 1;
		if (ran && memcmp(rb, h, bytes)) tbl_ok = false;
		free(rb);
	}
	for (int p = 0; p < 4 && ran; p++)
		for (int s = 0; s < 64 && ran; s++) {
			const int W = ec_w[thr_id], nwin = (256 + W - 1) / W, j = s % nwin;
			const int jbits = j == nwin - 1 ? 256 - W * (nwin - 1) : W;
			const uint32_t i = 1 + (uint32_t)(((uint64_t)s * 2654435761u) & ((1u << jbits) - 2));
			uint8_t ent[64];
			ran = cudaMemcpy(ent, d_curve[thr_id][p] + (((size_t)j << W) + i) * BC3_EC_ENTRY, 64,
			                 cudaMemcpyDeviceToHost) == cudaSuccess;
			uint32_t k[8] = { 0 }, x[8], y[8];
			const uint64_t kv = (uint64_t)i << ((j * W) & 31); /* i * 2^(jW) */
			k[(j * W) >> 5] = (uint32_t)kv;
			if (((j * W) >> 5) + 1 < 8) k[((j * W) >> 5) + 1] = (uint32_t)(kv >> 32);
			if (!bc3hash_curve_ref(p, k, x, y)) { tbl_ok = false; continue; }
			for (int b = 0; b < 32; b++) {
				const uint8_t xb = (uint8_t)(x[7 - b / 4] >> (24 - 8 * (b % 4)));
				const uint8_t yb = (uint8_t)(y[7 - b / 4] >> (24 - 8 * (b % 4)));
				if (ent[b] != xb || ent[32 + b] != yb) tbl_ok = false;
			}
		}
	if (!ran) selftest_cuda_fault();

	/* kat: each vector is hashed at its own nonce through the scan kernels; the
	 * all-ones target makes it a candidate, so the report path is exercised too.
	 * neg: the same scan with one header bit flipped must change the digest. */
	bool kat_ok = true, rep_ok = true, neg_ok = true;
	for (int v = 0; v < 4 && ran; v++) {
		uint8_t hdr[80], want[32], got[32];
		uint32_t res[2];
		unhex(hdr, kat_hdr[v], 80);
		unhex(want, kat_digest[v], 32);
		const uint32_t nonce = (uint32_t)hdr[76] << 24 | (uint32_t)hdr[77] << 16 | (uint32_t)hdr[78] << 8 | hdr[79];
		/* the faults change what the device is given, one leg each */
		if (v == 1 && bc3_fault(thr_id, "kat")) hdr[10] ^= 1;
		const uint32_t zeros[8] = { 0 };
		const uint32_t* tgt = (v == 2 && bc3_fault(thr_id, "report")) ? zeros : all_ones;
		ran = bc3_selftest_scan(thr_id, hdr, tgt, nonce, 1, got, res);
		if (!ran) break;
		if (memcmp(got, want, 32)) kat_ok = false;
		if (res[0] != nonce || res[1] != UINT32_MAX) rep_ok = false;
		if (v == 0) {
			if (!bc3_fault(thr_id, "neg")) hdr[5] ^= 0x10;
			ran = bc3_selftest_scan(thr_id, hdr, all_ones, nonce, 1, got, res);
			if (ran && !memcmp(got, want, 32)) neg_ok = false;
		}
	}

	/* scan: 64 consecutive nonces of a random-looking header against a target
	 * that ~1/8 of hashes meet; the two reported nonces must be exactly the two
	 * lowest CPU hits, the count and slots must match the CPU hits, and every
	 * GPU digest must equal the oracle's. */
	bool scan_ok = true;
	if (ran) {
		uint8_t hdr[80], dig[64 * 32];
		for (int i = 0; i < 80; i++) hdr[i] = (uint8_t)(i * 37 + 11);
		hdr[1] |= 0x08; /* BC3_VBIT, as a real job carries it */
		uint32_t target[8];
		for (int i = 0; i < 7; i++) target[i] = UINT32_MAX;
		target[7] = 0x1fffffffu;
		const uint32_t start = 0x12345670u;
		uint32_t res[BC3_RES_WORDS];
		ran = bc3_selftest_scan(thr_id, hdr, target, start + (bc3_fault(thr_id, "scan") ? 1 : 0), 64, dig, res,
		                        BC3_RES_WORDS);
		uint32_t lo[2] = { UINT32_MAX, UINT32_MAX };
		uint64_t hits = 0; /* bit n: nonce start + n meets the target */
		for (uint32_t n = 0; ran && n < 64; n++) {
			const uint32_t nonce = start + n;
			hdr[76] = (uint8_t)(nonce >> 24), hdr[77] = (uint8_t)(nonce >> 16);
			hdr[78] = (uint8_t)(nonce >> 8), hdr[79] = (uint8_t)nonce;
			uint32_t h[8];
			bc3hash_hash(h, hdr);
			if (memcmp(h, dig + 32 * n, 32)) scan_ok = false;
			if (h[7] <= target[7]) { /* target words 0..6 are all-ones */
				hits |= 1ull << n;
				if (lo[0] == UINT32_MAX) lo[0] = nonce;
				else if (lo[1] == UINT32_MAX) lo[1] = nonce;
			}
		}
		if (ran && (res[0] != lo[0] || res[1] != lo[1] || lo[1] == UINT32_MAX)) scan_ok = false;
		uint32_t nhits = 0;
		for (uint64_t m = hits; m; m &= m - 1) nhits++;
		if (ran && res[2] != nhits) scan_ok = false;
		uint64_t seen = 0; /* each slot a distinct hit; all hits when they fit */
		for (uint32_t i = 0; ran && i < min(res[2], (uint32_t)BC3_SLOTS); i++) {
			const uint32_t n = res[3 + i] - start;
			if (n >= 64 || !(hits >> n & 1) || (seen >> n & 1)) scan_ok = false;
			else seen |= 1ull << n;
		}
		if (ran && nhits <= BC3_SLOTS && seen != hits) scan_ok = false;
		gpulog(LOG_DEBUG, thr_id, "bc3hash self-test scan: %u of 64 nonces are candidates", nhits);
	}

	const bool passed = ran && tbl_ok && kat_ok && rep_ok && neg_ok && scan_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "bc3hash self-test FAILED (tables %s, vectors %s, report %s, negative %s, scan %s)",
		       tbl_ok ? "ok" : "MISMATCH", kat_ok ? "ok" : "MISMATCH", rep_ok ? "ok" : "MISMATCH",
		       neg_ok ? "ok" : "MISMATCH", scan_ok ? "ok" : "MISMATCH");
	else
		gpulog(LOG_INFO, thr_id, "bc3hash self-test passed (tables, 4 vectors, report, negative, 64-nonce scan)");
	return selftest_gate(thr_id, "bc3hash", passed);
}

/* ------------------------------------------------------------------ init */

/* A batch runs about a second: wait for it asleep, not spinning a CPU core per
 * GPU (that starves the host on rigs with few cores). --cuda-schedule wins. */
static bool sched_set[MAX_GPUS];
static void bc3_blocking_sync(int thr_id)
{
	unsigned flags = 0;
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaGetLastError();
	if (cudaGetDeviceFlags(&flags) != cudaSuccess ||
	    (flags & cudaDeviceScheduleMask) != cudaDeviceScheduleBlockingSync)
		cuda_reset_device(thr_id, NULL); /* context already active: a reset applies it */
	sched_set[thr_id] = true;
	gpulog(LOG_DEBUG, thr_id, "bc3hash: blocking sync (--cuda-schedule overrides)");
}

static void bc3hash_init(int thr_id, uint32_t throughput)
{
	cudaSetDevice(device_map[thr_id]);
	if (opt_cudaschedule < 0) bc3_blocking_sync(thr_id);

	for (int w = 0; w < 3; w++) {
		size_t bytes;
		const void* h = bc3hash_pow_table(w, &bytes);
		CUDA_SAFE_CALL(cudaMalloc(&d_pow[thr_id][w], bytes));
		CUDA_SAFE_CALL(cudaMemcpy(d_pow[thr_id][w], h, bytes, cudaMemcpyHostToDevice));
	}

	/* Phase D powers with 12-bit rows by default, built on the device from the
	 * coin's 8-bit table; BC3HASH_POW_BITS=8|11|12 picks the width. */
	{
		const char* fb = getenv("BC3HASH_POW_BITS");
		const int rb = fb ? atoi(fb) : 12;
		pow_rb[thr_id] = rb == 8 || rb == 11 ? rb : 12;
		d_pow_w[thr_id] = NULL;
		if (pow_rb[thr_id] != 8) {
			const size_t ent = pow_rb[thr_id] == 12 ? bc3_powrows<12>::entries() : bc3_powrows<11>::entries();
			CUDA_SAFE_CALL(cudaMalloc(&d_pow_w[thr_id], ent * sizeof(bc3u256)));
			const bc3u256* t8 = (const bc3u256*)d_pow[thr_id][2];
			for (size_t first = 0; first < ent; first += 1u << 16) {
				const uint32_t end = (uint32_t)min(first + (1u << 16), ent);
				const uint32_t grid = (end - (uint32_t)first + 127) / 128;
				if (pow_rb[thr_id] == 12)
					bc3_pow_table_kernel<36113u, 12> <<< grid, 128 >>> (d_pow_w[thr_id], t8, (uint32_t)first, end);
				else
					bc3_pow_table_kernel<36113u, 11> <<< grid, 128 >>> (d_pow_w[thr_id], t8, (uint32_t)first, end);
				CUDA_SAFE_CALL(cudaGetLastError());
			}
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
		}
	}

	/* EC window: 20 bits (3 GB of tables) when the card has the memory, else 16
	 * (256 MB). BC3HASH_EC_WINDOW=16|20 forces it. */
	{
		size_t freeb = 0, totalb = 0;
		cudaMemGetInfo(&freeb, &totalb);
		const char* fw = getenv("BC3HASH_EC_WINDOW");
		const size_t need20 = 4 * bc3_ecw<20>::bytes() + ((size_t)768 << 20);
		ec_w[thr_id] = fw ? (atoi(fw) == 20 ? 20 : 16) : (freeb >= need20 ? 20 : 16);
	}
	const int W = ec_w[thr_id], nwin = (256 + W - 1) / W;
	const size_t tbytes = W == 20 ? bc3_ecw<20>::bytes() : bc3_ecw<16>::bytes();
	const size_t tentries = tbytes / BC3_EC_ENTRY;
	uint8_t wb[16 * 64], *d_wb;
	CUDA_SAFE_CALL(cudaMalloc(&d_wb, sizeof(wb)));
	for (int p = 0; p < 4; p++) {
		if (!bc3hash_window_bases(p, W, nwin, wb)) {
			gpulog(LOG_ERR, thr_id, "bc3hash: libsecp256k1 could not derive the curve window bases");
			proper_exit(EXIT_CODE_SW_INIT_ERROR);
		}
		CUDA_SAFE_CALL(cudaMalloc(&d_curve[thr_id][p], tbytes));
		CUDA_SAFE_CALL(cudaMemcpy(d_wb, wb, 64 * nwin, cudaMemcpyHostToDevice));
		/* fixed 65536-entry chunks keep each launch short whatever the SM count */
		for (size_t first = 0; first < tentries; first += 1u << 16) {
			const uint32_t end = (uint32_t)min(first + (1u << 16), tentries);
			const uint32_t grid = (end - (uint32_t)first + 127) / 128;
			if (W == 20) bc3_ec_table_kernel<20> <<< grid, 128 >>> (d_curve[thr_id][p], d_wb, (uint32_t)first, end);
			else bc3_ec_table_kernel<16> <<< grid, 128 >>> (d_curve[thr_id][p], d_wb, (uint32_t)first, end);
			CUDA_SAFE_CALL(cudaGetLastError());
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
		}
	}
	cudaFree(d_wb);

	max_batch[thr_id] = throughput;
	CUDA_SAFE_CALL(cudaMalloc(&d_state[thr_id], (size_t)BC3_ST_SLOTS * 8 * 4 * throughput));
	CUDA_SAFE_CALL(cudaMalloc(&d_front[thr_id], (size_t)BC3_FRONT_WORDS * 4 * throughput));
	CUDA_SAFE_CALL(cudaMalloc(&d_res[thr_id], BC3_RES_WORDS * sizeof(uint32_t)));

	{
		const char* fd = getenv("BC3HASH_DSPLIT");
		d_split[thr_id] = fd ? atoi(fd) != 0 : device_sm[device_map[thr_id]] < 700;
	}
	gpulog(LOG_INFO, thr_id, "bc3hash: %d-bit curve windows (%u MB), %d-bit power rows, phase D %s, batch %u nonces",
	       W, (unsigned)(4 * (tbytes >> 20)), pow_rb[thr_id], d_split[thr_id] ? "per round" : "merged", throughput);

	bc3_device_selftest(thr_id);
	init_done[thr_id] = true;
}

/* ------------------------------------------------------------------ -D differential */

/* -D: BC3_DIFF_N consecutive nonces of the live job through the scan kernels
 * against the oracle: each digest, a plain and a 2*nonce+1 weighted accumulator
 * (catches swapped nonces), and the report against a target 5 of them meet.
 * BC3HASH_DIFF_FAULT=1..4 (shifted start, swap, bit flip, low target) breaks it
 * on purpose. Testing only. */
#define BC3_DIFF_N 251
#define BC3_DIFF_PERIOD 60

static bool bc3_differential(int thr_id, const uint32_t* endiandata76, uint32_t start, uint32_t fault)
{
	const uint32_t N = BC3_DIFF_N;
	static uint8_t cpu[BC3_DIFF_N * 32], gpu[BC3_DIFF_N * 32];
	uint32_t _ALIGN(64) ed[20];
	memcpy(ed, endiandata76, 76);
	for (uint32_t i = 0; i < N; i++) {
		be32enc(&ed[19], start + i);
		bc3hash_hash(cpu + 32 * i, ed);
	}

	/* target = the 5th lowest digest (compared as the device does: LE words, word 7 high) */
	uint32_t order[BC3_DIFF_N];
	for (uint32_t i = 0; i < N; i++) order[i] = i;
	auto word = [](const uint8_t* d, int i) { return (uint32_t)d[4 * i] | (uint32_t)d[4 * i + 1] << 8 |
	                                                 (uint32_t)d[4 * i + 2] << 16 | (uint32_t)d[4 * i + 3] << 24; };
	auto less = [&](uint32_t a, uint32_t b) {
		for (int i = 7; i >= 0; i--) {
			const uint32_t x = word(cpu + 32 * a, i), y = word(cpu + 32 * b, i);
			if (x != y) return x < y;
		}
		return false;
	};
	std::sort(order, order + N, less);
	uint32_t target[8], dev_target[8];
	const uint32_t tsel = order[4];
	for (int i = 0; i < 8; i++) {
		target[i] = word(cpu + 32 * tsel, i);
		dev_target[i] = fault == 4 ? word(cpu + 32 * order[0], i) : target[i];
	}
	uint32_t want[2] = { UINT32_MAX, UINT32_MAX };
	for (uint32_t i = 0; i < N; i++)
		if (!less(tsel, i)) { /* digest <= target */
			if (want[0] == UINT32_MAX) want[0] = start + i;
			else if (want[1] == UINT32_MAX) want[1] = start + i;
		}

	uint32_t res[2];
	if (!bc3_selftest_scan(thr_id, (const uint8_t*)ed, dev_target, start + (fault == 1 ? 1 : 0), N, gpu, res)) {
		s_selftest_cuda_fault = false;
		gpulog(LOG_WARNING, thr_id, "bc3hash differential could not run (CUDA resource failure), skipped");
		return true;
	}
	if (fault == 2) for (int b = 0; b < 32; b++) { const uint8_t t = gpu[32 * 7 + b]; gpu[32 * 7 + b] = gpu[32 * 8 + b]; gpu[32 * 8 + b] = t; }
	if (fault == 3) gpu[32 * 100 + 17] ^= 0x04;

	uint32_t acc[2][2][8] = { { { 0 } } }, bad = 0, first_bad = UINT32_MAX;
	for (uint32_t i = 0; i < N; i++) {
		const uint32_t w = 2u * (start + i) + 1u;
		if (memcmp(cpu + 32 * i, gpu + 32 * i, 32)) { bad++; if (first_bad == UINT32_MAX) first_bad = i; }
		for (int k = 0; k < 8; k++) {
			acc[0][0][k] ^= word(cpu + 32 * i, k), acc[0][1][k] ^= word(cpu + 32 * i, k) * w;
			acc[1][0][k] ^= word(gpu + 32 * i, k), acc[1][1][k] ^= word(gpu + 32 * i, k) * w;
		}
	}
	const bool d_ok = !memcmp(acc[0][0], acc[1][0], 32), n_ok = !memcmp(acc[0][1], acc[1][1], 32);
	const bool r_ok = res[0] == want[0] && res[1] == want[1];
	if (bad || !d_ok || !n_ok || !r_ok) {
		gpulog(LOG_ERR, thr_id, "bc3hash differential FAILED over %u nonces from %08x: %u digests differ (first +%u)",
		       N, start, bad, first_bad == UINT32_MAX ? 0 : first_bad);
		gpulog(LOG_ERR, thr_id, "bc3hash differential: accumulators %s / weighted %s, report %s",
		       d_ok ? "ok" : "MISMATCH", n_ok ? "ok" : "MISMATCH", r_ok ? "ok" : "MISMATCH");
		if (!r_ok)
			gpulog(LOG_ERR, thr_id, "bc3hash differential: report %08x %08x, want %08x %08x", res[0], res[1], want[0], want[1]);
		if (d_ok && !n_ok)
			gpulog(LOG_ERR, thr_id, "bc3hash differential: digests agree but their NONCE pairing does not");
		return false;
	}
	gpulog(LOG_DEBUG, thr_id, "bc3hash differential OK: %u nonces from %08x, digests + both accumulators + report",
	       N, start);
	return true;
}

/* ------------------------------------------------------------------ scanhash */

extern "C" int scanhash_bc3hash(int thr_id, struct work* work, uint32_t max_nonce, unsigned long* hashes_done)
{
	uint32_t* pdata = work->data;
	uint32_t* ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	static uint32_t def_batch[MAX_GPUS];
	if (!def_batch[thr_id]) def_batch[thr_id] = bc3_default_batch(thr_id);
	uint32_t throughput = cuda_default_throughput(thr_id, def_batch[thr_id]);
	if (!init_done[thr_id]) bc3hash_init(thr_id, throughput);
	throughput = min(throughput, max_batch[thr_id]);
	throughput = min(throughput, max_nonce - first_nonce);

	/* benchmark: rare candidates (2^-24), so batches are not cut short */
	if (opt_benchmark && ptarget[7] < 0x000000ffU)
		ptarget[7] = 0x000000ffU;

	/* easy target: at most ~3 expected candidates per batch, in whole waves */
	{
		const double p = ((double)ptarget[7] + ptarget[6] / 4294967296.0 + 1.0) / 4294967296.0;
		const double cap = 3.0 / p;
		if (cap < throughput) {
			const uint32_t w = wave[thr_id] ? wave[thr_id] : BC3_TPB;
			throughput = min(throughput, max(w, (uint32_t)(cap / w) * w));
		}
		static uint32_t logged[MAX_GPUS];
		if (opt_debug && throughput != logged[thr_id]) {
			logged[thr_id] = throughput;
			gpulog(LOG_DEBUG, thr_id, "bc3hash batch %u nonces (~%.1f candidates)", throughput, throughput * p);
		}
	}

	/* 76-byte base header = be32enc(pdata[0..18]); the kernel reads these bytes
	 * back as little-endian words, so upload the bytes, not pdata. */
	uint32_t _ALIGN(64) endiandata[20];
	for (int i = 0; i < 19; i++)
		be32enc(&endiandata[i], pdata[i]);

	/* -D: once per new prevhash, and otherwise at most every BC3_DIFF_PERIOD s */
	if (opt_debug) {
		static uint32_t last_prev[MAX_GPUS][8];
		static time_t last_run[MAX_GPUS];
		const time_t now = time(NULL);
		if (memcmp(last_prev[thr_id], &pdata[1], 32) || now - last_run[thr_id] >= BC3_DIFF_PERIOD) {
			memcpy(last_prev[thr_id], &pdata[1], 32);
			last_run[thr_id] = now;
			const char* fenv = getenv("BC3HASH_DIFF_FAULT");
			const uint32_t fault = fenv ? (uint32_t)atoi(fenv) : 0;
			if (fault)
				gpulog(LOG_WARNING, thr_id, "bc3hash differential: FAULT %u injected on purpose", fault);
			if (!bc3_differential(thr_id, endiandata, pdata[19], fault)) {
				gpulog(LOG_ERR, thr_id, "bc3hash: refusing to mine after a failed differential");
				proper_exit(EXIT_CODE_SW_INIT_ERROR);
			}
		}
	}
	bc3_set_header((const uint8_t*)endiandata, ptarget);

	const uint32_t arm[3] = { UINT32_MAX, UINT32_MAX, 0 };
	do {
		/* never past max_nonce, so a resume at the batch end is above every candidate */
		const uint32_t n = (uint32_t)min((uint64_t)throughput, (uint64_t)max_nonce - pdata[19]);
		CUDA_SAFE_CALL(cudaMemcpy(d_res[thr_id], arm, sizeof(arm), cudaMemcpyHostToDevice));
		bc3_launch(thr_id, n, pdata[19], NULL);
		uint32_t res[BC3_RES_WORDS];
		CUDA_SAFE_CALL(cudaMemcpy(res, d_res[thr_id], sizeof(res), cudaMemcpyDeviceToHost));

		if (res[0] != UINT32_MAX) {
			/* all candidates and resume at the batch end, else the two lowest */
			const bool all = res[2] <= BC3_SLOTS;
			/* nc, not n: n is the batch size the resume below needs */
			uint32_t cand[BC3_SLOTS], nc = 0;
			if (all) {
				for (uint32_t i = 0; i < res[2]; i++) {
					uint32_t j = nc++;
					for (; j > 0 && cand[j - 1] > res[3 + i]; j--) cand[j] = cand[j - 1];
					cand[j] = res[3 + i];
				}
			} else {
				cand[nc++] = res[0];
				cand[nc++] = res[1];
			}
			work->valid_nonces = 0;
			for (uint32_t s = 0; s < nc; s++) {
				uint32_t _ALIGN(64) vhash[8];
				be32enc(&endiandata[19], cand[s]);
				bc3hash_hash(vhash, endiandata);
				if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
					work->nonces[work->valid_nonces] = cand[s];
					if (work->valid_nonces == 0)
						work_set_target_ratio(work, vhash);
					else
						bn_set_target_ratio(work, vhash, work->valid_nonces);
					work->valid_nonces++;
				} else {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", cand[s]);
				}
			}
			if (work->valid_nonces) {
				pdata[19] = all ? pdata[19] + n : res[1] + 1;
				*hashes_done = pdata[19] - first_nonce;
				return work->valid_nonces;
			}
		}

		if ((uint64_t)n + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += n;
	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_bc3hash(int thr_id)
{
	if (!init_done[thr_id]) return;
	cudaSetDevice(device_map[thr_id]);
	for (int p = 0; p < 4; p++) cudaFree(d_curve[thr_id][p]);
	for (int w = 0; w < 3; w++) cudaFree(d_pow[thr_id][w]);
	cudaFree(d_pow_w[thr_id]);
	cudaFree(d_state[thr_id]);
	cudaFree(d_front[thr_id]);
	cudaFree(d_res[thr_id]);
	if (sched_set[thr_id]) {
		cudaSetDeviceFlags(cudaDeviceScheduleAuto); /* the next algo gets the default back */
		cudaGetLastError();
		sched_set[thr_id] = false;
	}
	init_done[thr_id] = false;
}
