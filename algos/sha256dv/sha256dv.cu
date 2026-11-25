/**
 * SHA256Dv (Veil) - double SHA-256 with a coin-specific Stratum protocol.
 *
 * The hash is ordinary SHA-256d over an 80-byte "stage2" buffer:
 *
 *   off 0  : version  (LE, 4)
 *   off 4  : midstate (BE, 32)  - supplied pre-hashed by the pool
 *   off 36 : merkle   (LE, 32)  - byte-reversed merkle_be from the pool
 *   off 68 : ntime    (LE, 4)
 *   off 72 : nonce_lo (LE, 4)   - the searched 32-bit word
 *   off 76 : nonce_hi (LE, 4)   - base high word of the 64-bit nonce
 *
 *   hash = SHA256( SHA256( stage2 ) )
 *
 * The first 64-byte SHA-256 block (version + midstate + merkle[0:28]) is fixed
 * for a job, so its midstate is precomputed on the host once per job; the GPU
 * kernel only runs the second block (which holds nonce_lo) plus the second
 * SHA-256. nonce_lo sits at word index 2 of the second block (byte 72), with
 * nonce_hi fixed at word index 3 (byte 76).
 *
 * Bespoke notify/submit live in util.cpp (stratum_notify) and ccminer.cpp
 * (submit path); see also algo/sha/sha256dv.c in cpuminer-opt.
 */

#include "miner.h"
#include "cuda/helper.cuh"

#include "cuda/sha256_device.cuh"

#include <stdint.h>
#include <string.h>

// Per-job constants uploaded via sha256dv_setBlock().
__device__ __constant__ static uint32_t d_midstate[8]; // SHA-256 midstate of stage2 block 1
__device__ __constant__ static uint32_t d_block2[4];    // w0=merkle tail, w1=ntime, [2] unused, w3=nonce_hi
__device__ __constant__ static uint32_t d_target[8];    // work->target (index 7 = most significant)
// Block-2 compression state after rounds 0,1,2 (only nonce word w[2] varies, so
// those rounds depend on the nonce only as "const + w2"). Layout matches a..h
// after round 2 with the two nonce-linear words pre-split: [0]=const part of a,
// [1]=b, [2]=c, [3]=d, [4]=const part of e, [5]=f, [6]=g, [7]=h. The kernel adds
// w2 back into [0]/[4] and resumes compression at round 3 (cf. cgminer SHA256d).
__device__ __constant__ static uint32_t d_pre[8];

static uint32_t *d_resNonce[MAX_GPUS];
static bool      init_done[MAX_GPUS] = { 0 };

// Veil mines a 64-bit nonce = nonce_hi:nonce_lo. The pool supplies a starting
// nonce_hi; the miner searches the full 32-bit nonce_lo and, when that space is
// used up, rolls nonce_hi forward for a fresh 2^32 range (cf. cpuminer-opt). So
// the search space is effectively unbounded and we never idle.
//
// A job's fixed identity (midstate/ntime/target) lives in work->veil_*, which
// miner_thread's data[]-based job-change check can't see; it therefore resets
// work->data[19] (and re-copies the pool's base nonce_hi) on every scantime
// regen of an *unchanged* job. We keep the running (nonce_hi, nonce_lo) cursor
// ourselves, keyed on a signature of the fixed inputs, so those resets don't
// rescan covered ground (which would resubmit duplicate shares) or rewind the
// rolled nonce_hi.
static uint32_t dv_hi[MAX_GPUS]      = { 0 };  // current (rolled) nonce_hi
static uint32_t dv_lo[MAX_GPUS]      = { 0 };  // nonce_lo cursor within dv_hi
static uint32_t dv_sig[MAX_GPUS]     = { 0 };
static bool     dv_sig_set[MAX_GPUS] = { 0 };

// ---------------------------------------------------------------------------
// Host SHA-256 (for midstate precompute, CPU validation and self-test)
// ---------------------------------------------------------------------------

static void host_sha256_transform(uint32_t state[8], const uint32_t in[16])
{
	uint32_t w[16]; // sha256_transform_full consumes the block
	for (int i = 0; i < 16; i++) w[i] = in[i];
	sha256_transform_full(w, state, h_sha256_K);
}

// Precompute block-2 compression state through round 2, where only the nonce
// word w[2] varies. Rounds 0,1 are fully constant; round 2's t1/t2 reduce to
// "const + w2", so the resulting a and e are "const + w2". We hand the kernel
// the constant parts (a/e split out) and let it resume compression at round 3.
static void sha256dv_precompute_pre(const uint32_t ms[8], const uint32_t block2[4],
                                    uint32_t pre[8])
{
	// rounds 0,1 constant, round 2 (w2 = nonce_lo word) split into T1c/T2c
	sha256_prehash_split_host(ms, block2, 2, pre);
}

// Double SHA-256 of an 80-byte stage2 buffer -> 8 LE words (index 7 = MSW,
// matching the target convention).
static void host_sha256dv(uint32_t hash_le[8], const uint8_t stage2[80])
{
	uint32_t blk[16], st[8];

	// First SHA-256 (two blocks).
	memcpy(st, h_sha256_H, sizeof(st));
	for (int i = 0; i < 16; i++) blk[i] = be32dec(stage2 + i*4);
	host_sha256_transform(st, blk);
	for (int i = 0; i < 4;  i++) blk[i] = be32dec(stage2 + 64 + i*4);
	blk[4] = 0x80000000;
	for (int i = 5; i < 15; i++) blk[i] = 0;
	blk[15] = 640;
	host_sha256_transform(st, blk);

	// Second SHA-256 over the 32-byte first digest.
	uint32_t st2[8];
	memcpy(st2, h_sha256_H, sizeof(st2));
	for (int i = 0; i < 8;  i++) blk[i] = st[i];
	blk[8] = 0x80000000;
	for (int i = 9; i < 15; i++) blk[i] = 0;
	blk[15] = 256;
	host_sha256_transform(st2, blk);

	for (int i = 0; i < 8; i++) hash_le[i] = st2[i];
}

// Build the 80-byte stage2 buffer from the work fields for a given nonce pair.
static void sha256dv_build_stage2(uint8_t out[80], const struct work *work,
                                  uint32_t nonce_lo, uint32_t nonce_hi)
{
	uint8_t *p = out;
	le32enc(p, work->data[0]); p += 4;                  // version (LE)
	memcpy(p, work->veil_midstate_be, 32); p += 32;     // midstate (BE)
	for (int i = 0; i < 32; i++) p[i] = work->veil_merkle_be[31 - i];
	p += 32;                                            // merkle (LE = reversed BE)
	le32enc(p, work->veil_ntime); p += 4;               // ntime (LE)
	le32enc(p, nonce_lo); p += 4;                       // nonce_lo (LE)
	le32enc(p, nonce_hi);                               // nonce_hi (LE)
}

static bool sha256dv_meets_target(const uint32_t *hash, const uint32_t *target)
{
	for (int i = 7; i >= 0; i--) {
		if (hash[i] > target[i]) return false;
		if (hash[i] < target[i]) return true;
	}
	return true;
}

// ---------------------------------------------------------------------------
// Device SHA-256
// ---------------------------------------------------------------------------
// Both blocks are inlined into the kernel below (the first transform resumes
// from the host-precomputed round-2 state; the second uses a round-60 early-out),
// so there is no shared device transform helper.

__global__
void sha256dv_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce)
{
	const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < threads)
	{
		const uint32_t nonce_lo = startNonce + tid;
		const uint32_t w2 = cuda_swab32(nonce_lo);   // nonce_lo message word

		// Second block of the first SHA-256 (continues from the job midstate).
		// Only message word w[2] depends on the nonce, so rounds 0,1,2 of the
		// compression are precomputed on the host (d_pre); we resume at round 3.
		uint32_t w[64];
		w[0] = d_block2[0];                  // merkle tail
		w[1] = d_block2[1];                  // ntime
		w[2] = w2;
		w[3] = d_block2[3];                  // nonce_hi
		w[4] = 0x80000000;
		#pragma unroll
		for (int i = 5; i < 15; i++) w[i] = 0;
		w[15] = 640;
		#pragma unroll
		for (int i = 16; i < 64; i++) {
			uint32_t s0 = ROTR32(w[i-15], 7) ^ ROTR32(w[i-15], 18) ^ (w[i-15] >> 3);
			uint32_t s1 = ROTR32(w[i-2], 17) ^ ROTR32(w[i-2], 19) ^ (w[i-2] >> 10);
			w[i] = w[i-16] + s0 + w[i-7] + s1;
		}

		uint32_t a = d_pre[0] + w2, b = d_pre[1], c = d_pre[2], d = d_pre[3];
		uint32_t e = d_pre[4] + w2, f = d_pre[5], g = d_pre[6], h = d_pre[7];
		#pragma unroll
		for (int i = 3; i < 64; i++) {
			uint32_t S1 = ROTR32(e,6) ^ ROTR32(e,11) ^ ROTR32(e,25);
			uint32_t ch = (e & f) ^ (~e & g);
			uint32_t t1 = h + S1 + ch + c_sha256_K[i] + w[i];
			uint32_t S0 = ROTR32(a,2) ^ ROTR32(a,13) ^ ROTR32(a,22);
			uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
			uint32_t t2 = S0 + maj;
			h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
		}

		uint32_t st[8];
		st[0] = d_midstate[0] + a; st[1] = d_midstate[1] + b;
		st[2] = d_midstate[2] + c; st[3] = d_midstate[3] + d;
		st[4] = d_midstate[4] + e; st[5] = d_midstate[5] + f;
		st[6] = d_midstate[6] + g; st[7] = d_midstate[7] + h;

		// Second SHA-256 over the 32-byte first digest, inlined with a round-60
		// early-out. The compared MSW st2[7] = 0x5be0cd19 + e60 is fully
		// determined after round 60 (register lineage gives h63 = e60), so for
		// any target whose high word is 0 (pool diff >= 1) almost every nonce is
		// rejected here -- skipping rounds 61-63, the feed-forward and the full
		// compare. Rare survivors (incl. the exact-match self-test) finish normally.
		uint32_t w2s[64];
		#pragma unroll
		for (int i = 0; i < 8; i++) w2s[i] = st[i];
		w2s[8] = 0x80000000;
		#pragma unroll
		for (int i = 9; i < 15; i++) w2s[i] = 0;
		w2s[15] = 256;
		#pragma unroll
		for (int i = 16; i <= 60; i++) {
			uint32_t s0 = ROTR32(w2s[i-15],7) ^ ROTR32(w2s[i-15],18) ^ (w2s[i-15] >> 3);
			uint32_t s1 = ROTR32(w2s[i-2],17) ^ ROTR32(w2s[i-2],19) ^ (w2s[i-2] >> 10);
			w2s[i] = w2s[i-16] + s0 + w2s[i-7] + s1;
		}

		uint32_t A = 0x6a09e667, B = 0xbb67ae85, C = 0x3c6ef372, D = 0xa54ff53a;
		uint32_t E = 0x510e527f, F = 0x9b05688c, G = 0x1f83d9ab, H = 0x5be0cd19;
		#pragma unroll
		for (int i = 0; i <= 60; i++) {
			uint32_t S1 = ROTR32(E,6) ^ ROTR32(E,11) ^ ROTR32(E,25);
			uint32_t ch = (E & F) ^ (~E & G);
			uint32_t t1 = H + S1 + ch + c_sha256_K[i] + w2s[i];
			uint32_t S0 = ROTR32(A,2) ^ ROTR32(A,13) ^ ROTR32(A,22);
			uint32_t maj = (A & B) ^ (A & C) ^ (B & C);
			uint32_t t2 = S0 + maj;
			H=G; G=F; F=E; E=D+t1; D=C; C=B; B=A; A=t1+t2;
		}

		// st2[7] == 0x5be0cd19 + e60 (== h after round 63). Reject unless it can
		// still meet the target high word; otherwise finish and full-compare.
		if (0x5be0cd19 + E <= d_target[7]) {
			#pragma unroll
			for (int i = 61; i < 64; i++) {
				uint32_t s0 = ROTR32(w2s[i-15],7) ^ ROTR32(w2s[i-15],18) ^ (w2s[i-15] >> 3);
				uint32_t s1 = ROTR32(w2s[i-2],17) ^ ROTR32(w2s[i-2],19) ^ (w2s[i-2] >> 10);
				w2s[i] = w2s[i-16] + s0 + w2s[i-7] + s1;
			}
			#pragma unroll
			for (int i = 61; i < 64; i++) {
				uint32_t S1 = ROTR32(E,6) ^ ROTR32(E,11) ^ ROTR32(E,25);
				uint32_t ch = (E & F) ^ (~E & G);
				uint32_t t1 = H + S1 + ch + c_sha256_K[i] + w2s[i];
				uint32_t S0 = ROTR32(A,2) ^ ROTR32(A,13) ^ ROTR32(A,22);
				uint32_t maj = (A & B) ^ (A & C) ^ (B & C);
				uint32_t t2 = S0 + maj;
				H=G; G=F; F=E; E=D+t1; D=C; C=B; B=A; A=t1+t2;
			}

			uint32_t st2[8];
			st2[0]=0x6a09e667+A; st2[1]=0xbb67ae85+B; st2[2]=0x3c6ef372+C; st2[3]=0xa54ff53a+D;
			st2[4]=0x510e527f+E; st2[5]=0x9b05688c+F; st2[6]=0x1f83d9ab+G; st2[7]=0x5be0cd19+H;

			// MSW-first compare against target.
			bool ok = true;
			#pragma unroll
			for (int i = 7; i >= 0; i--) {
				if (st2[i] > d_target[i]) { ok = false; break; }
				if (st2[i] < d_target[i]) break;
			}
			if (ok)
				resNonce[0] = nonce_lo;
		}
	}
}

__host__
static void sha256dv_setBlock(const uint32_t midstate[8], const uint32_t block2[4],
                              const uint32_t pre[8], const uint32_t target[8])
{
	cudaMemcpyToSymbol(d_midstate, midstate, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_block2,   block2,   4 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_pre,      pre,      8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_target,   target,   8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// Startup self-test: GPU kernel must reproduce the host SHA-256d on a fixed
// input (the host path uses an independent transform; per-share validation in
// scanhash re-checks every candidate against it as well).
// ---------------------------------------------------------------------------
static bool sha256dv_self_test(int thr_id)
{
	struct work w;
	memset(&w, 0, sizeof(w));
	w.data[0] = 0x20000000;
	for (int i = 0; i < 32; i++) { w.veil_midstate_be[i] = (uint8_t)(i + 1); w.veil_merkle_be[i] = (uint8_t)(0xa0 + i); }
	w.veil_ntime = 0x66000000;
	const uint32_t nonce_hi = 0x12345678, nonce_lo = 0x0000abcd;

	uint8_t stage2[80];
	uint32_t exp[8], ms[8], blk[16], block2[4];

	sha256dv_build_stage2(stage2, &w, nonce_lo, nonce_hi);
	host_sha256dv(exp, stage2);

	// midstate of block 1, plus the block-2 fixed words.
	memcpy(ms, h_sha256_H, sizeof(ms));
	for (int i = 0; i < 16; i++) blk[i] = be32dec(stage2 + i*4);
	host_sha256_transform(ms, blk);
	block2[0] = be32dec(stage2 + 64);
	block2[1] = be32dec(stage2 + 68);
	block2[2] = 0;
	block2[3] = be32dec(stage2 + 76);

	// Use the expected host digest as the target: the kernel reports the test
	// nonce only if its GPU-computed hash is <= exp. Since the host hash equals
	// exp exactly, a correct kernel reports it; a kernel that computes a larger
	// digest fails the test (got stays UINT32_MAX).
	uint32_t pre[8];
	sha256dv_precompute_pre(ms, block2, pre);
	sha256dv_setBlock(ms, block2, pre, exp);

	cudaMemset(d_resNonce[thr_id], 0xff, sizeof(uint32_t));
	sha256dv_gpu_hash <<< 1, 1 >>> (1, nonce_lo, d_resNonce[thr_id]);
	uint32_t got = UINT32_MAX;
	cudaMemcpy(&got, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	(void)nonce_hi;
	return got == nonce_lo;
}

// ---------------------------------------------------------------------------
// scanhash
// ---------------------------------------------------------------------------

extern "C" int scanhash_sha256dv(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const int dev_id = device_map[thr_id];

	// --benchmark has no Stratum job; synthesize a deterministic one so the
	// kernel (and startup self-test) can run and report a hashrate.
	if (opt_benchmark && !work->veil_sha256dv) {
		work->veil_sha256dv = true;
		work->data[0] = 0x20000000;
		for (int i = 0; i < 32; i++) {
			work->veil_midstate_be[i] = (uint8_t)(i + 1);
			work->veil_merkle_be[i]   = (uint8_t)(0xa0 + i);
		}
		work->veil_ntime    = 0x66000000;
		work->veil_nonce_hi = 0;
	}

	if (!work->veil_sha256dv) { *hashes_done = 0; return 0; }

	// Live Veil jobs manage their own nonce_lo cursor (see dv_cursor note); only
	// --benchmark uses the host-supplied pdata[19]/max_nonce window directly.
	const bool dv_track = !opt_benchmark;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 24);
	if (init_done[thr_id] && !dv_track) throughput = min(throughput, max_nonce - pdata[19]);

	// 0x03 like the rest of the sha256 family: an easier target floods the log
	// with one "share found" scanhash return (and Total: line) per ~2^24 hashes.
	if (opt_benchmark)
		ptarget[7] = 0x00000003;

	if (!init_done[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)), 0);

		if (!sha256dv_self_test(thr_id))
			gpulog(LOG_WARNING, thr_id, "SHA256Dv self-test FAILED (GPU/CPU mismatch)");

		init_done[thr_id] = true;
	}

	// Resolve the running (nonce_hi, nonce_lo) cursor before building stage2, so
	// the kernel encodes the rolled nonce_hi. Signature = FNV-1a over the fixed
	// inputs (midstate, ntime, target) -- NOT nonce_hi, which we roll ourselves.
	// A changed signature is a genuinely new job: restart from the pool's base
	// nonce_hi (striped by thr_id). An unchanged signature resumes our cursor,
	// ignoring stratum_gen_work's per-regen reset of pdata[19]/veil_nonce_hi.
	const uint32_t hi_stride = (opt_n_threads > 0) ? (uint32_t)opt_n_threads : 1u;
	uint32_t nonce_hi = work->veil_nonce_hi;
	if (dv_track) {
		uint32_t sig = 2166136261u;
		for (int i = 0; i < 32; i++) { sig ^= work->veil_midstate_be[i]; sig *= 16777619u; }
		sig ^= work->veil_ntime;   sig *= 16777619u;
		for (int i = 0; i < 8; i++) { sig ^= ptarget[i]; sig *= 16777619u; }

		if (!dv_sig_set[thr_id] || sig != dv_sig[thr_id]) {
			dv_sig[thr_id] = sig; dv_sig_set[thr_id] = true;
			dv_hi[thr_id] = work->veil_nonce_hi + (uint32_t)thr_id;
			dv_lo[thr_id] = 0;
		}
		nonce_hi  = dv_hi[thr_id];
		pdata[19] = dv_lo[thr_id];
	}

	// Per-job constants: stage2 block-1 midstate + block-2 fixed words.
	uint8_t  stage2[80];
	uint32_t ms[8], blk[16], block2[4];

	sha256dv_build_stage2(stage2, work, pdata[19], nonce_hi);
	memcpy(ms, h_sha256_H, sizeof(ms));
	for (int i = 0; i < 16; i++) blk[i] = be32dec(stage2 + i*4);
	host_sha256_transform(ms, blk);
	block2[0] = be32dec(stage2 + 64);
	block2[1] = be32dec(stage2 + 68);
	block2[2] = 0;
	block2[3] = be32dec(stage2 + 76);

	uint32_t pre[8];
	sha256dv_precompute_pre(ms, block2, pre);
	sha256dv_setBlock(ms, block2, pre, ptarget);

	const uint32_t first_nonce = pdata[19];
	// Scan the full nonce_lo space (0..2^32); cap this call to one host-sized
	// window so we stay responsive to new work. scan_end bounds the kernel; the
	// per-call cap bounds how far we sweep before returning.
	const uint64_t scan_end = dv_track ? 0x100000000ull : (uint64_t)max_nonce;
	const uint64_t call_end = dv_track
		? ((uint64_t)first_nonce + 0x40000000ull < scan_end
			? (uint64_t)first_nonce + 0x40000000ull : scan_end)
		: scan_end;

	// Reset the result slot once. The kernel only writes it on a find (which
	// returns below), so a no-find iteration leaves it at UINT32_MAX -- no need
	// to re-clear every launch. The rare reject-continue path re-arms it inline.
	cudaMemset(d_resNonce[thr_id], 0xff, sizeof(uint32_t));

	do {
		// Clamp the final window so the cursor never wraps past the 2^32 nonce_lo
		// space (dv_track) or the host max_nonce (benchmark).
		uint32_t span = throughput;
		if ((uint64_t)pdata[19] + span > scan_end)
			span = (uint32_t)(scan_end - pdata[19]);

		const uint32_t tpb = 128;
		dim3 grid((span + tpb - 1) / tpb);
		dim3 block(tpb);
		sha256dv_gpu_hash <<< grid, block >>> (span, pdata[19], d_resNonce[thr_id]);

		uint32_t resNonce = UINT32_MAX;
		cudaMemcpy(&resNonce, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

		*hashes_done = pdata[19] - first_nonce + span;

		if (resNonce != UINT32_MAX) {
			uint32_t hash_le[8];
			sha256dv_build_stage2(stage2, work, resNonce, nonce_hi);
			host_sha256dv(hash_le, stage2);

			if (sha256dv_meets_target(hash_le, ptarget)) {
				work->veil_nonce_lo = resNonce;
				work->veil_nonce_hi = nonce_hi;
				work->nonces[0] = resNonce;
				work->valid_nonces = 1;
				memcpy(work->target, ptarget, sizeof(work->target));
				for (int i = 0; i < 8; i++) ((uint32_t*)work->extra)[i] = hash_le[i];
				bn_set_target_ratio(work, hash_le, 0); // share diff for --show-diff
				pdata[19] = resNonce + 1;
				if (dv_track) {
					dv_lo[thr_id] = resNonce + 1;
					if (resNonce == 0xFFFFFFFFu) {       // nonce_lo wrapped: roll nonce_hi
						dv_hi[thr_id] += hi_stride;
						dv_lo[thr_id] = 0;
					}
				}
				return 1;
			} else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for nonce_lo %08x does not validate on CPU!", resNonce);
				// Re-arm the slot: the kernel wrote a (rejected) nonce into it, so
				// clear it before continuing or the next read would re-trigger.
				cudaMemset(d_resNonce[thr_id], 0xff, sizeof(uint32_t));
			}
		}

		if ((uint64_t)pdata[19] + span >= scan_end) {
			// nonce_lo space used up for this nonce_hi: roll nonce_hi and restart
			// nonce_lo at 0. Never idle -- the next call gets a fresh 2^32 range.
			*hashes_done = pdata[19] - first_nonce + span;
			if (dv_track) {
				dv_hi[thr_id] += hi_stride;
				dv_lo[thr_id] = 0;
				pdata[19] = 0;
			} else {
				pdata[19] = (uint32_t)scan_end;
			}
			return 0;
		}
		pdata[19] += span;
		if (dv_track) dv_lo[thr_id] = pdata[19];

	} while (!work_restart[thr_id].restart && (uint64_t)pdata[19] < call_end);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_sha256dv(int thr_id)
{
	if (!init_done[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_resNonce[thr_id]);
	init_done[thr_id]  = false;
	dv_sig_set[thr_id] = false;
	dv_hi[thr_id]      = 0;
	dv_lo[thr_id]      = 0;
	cudaDeviceSynchronize();
}
