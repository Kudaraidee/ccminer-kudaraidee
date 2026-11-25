#include <memory.h>

#include "cuda/helper.cuh"
#include "cuda/groestl512_x2_device.cuh"   /* bitsliced S-box, GF(2^8) doubling, plane transpose */

/* Groestl-256 of a 32-byte hash, four hashes per thread, bitsliced in registers: the
 * groestl512_x2 layout at 8 columns x 4 hashes (bit p = 4*col + h). Terminal stage: only digest
 * word 7 is un-transposed, for the target screen. ~250 registers + 64-word shared stash. */
#define TPB 128
#define MINB 2

static uint32_t *h_GNonces[MAX_GPUS];
static uint32_t *d_GNonces[MAX_GPUS];

__constant__ uint32_t pTarget[8];

__device__ __forceinline__
void groestl256_x4_round(uint32_t (&s)[8][8], const uint32_t r, const bool q, const uint32_t (&rot)[8])
{
	if (q) {
		#pragma unroll
		for (int i = 0; i < 8; i++)
			#pragma unroll
			for (int b = 0; b < 8; b++) s[i][b] = ~s[i][b];
		#pragma unroll
		for (int b = 0; b < 4; b++) s[7][b] ^= 0u - ((r >> b) & 1u);
		s[7][4] ^= 0xF0F0F0F0u; s[7][5] ^= 0xFF00FF00u; s[7][6] ^= 0xFFFF0000u;
	} else {
		#pragma unroll
		for (int b = 0; b < 4; b++) s[0][b] ^= 0u - ((r >> b) & 1u);
		s[0][4] ^= 0xF0F0F0F0u; s[0][5] ^= 0xFF00FF00u; s[0][6] ^= 0xFFFF0000u;
	}
	/* ShiftBytes before SubBytes (they commute) */
	#pragma unroll
	for (int i = 0; i < 8; i++)
		#pragma unroll
		for (int b = 0; b < 8; b++) s[i][b] = __funnelshift_r(s[i][b], s[i][b], rot[i]);
	#pragma unroll
	for (int i = 0; i < 8; i++) groestl512_x2_sbox(s[i]);
	/* MixBytes: the groestl512_x2 network (same matrix for both widths) */
	groestl512_x2_mix(s);
}

__device__ __forceinline__
void groestl256_x4_perm(uint32_t (&s)[8][8], const bool q)
{
	/* ShiftBytes sigma, P: 0..7, Q: 1 3 5 7 0 2 4 6 (columns) -> rotate 4*sigma bits */
	uint32_t rot[8];
	rot[0] = q ? 4 : 0;   rot[1] = q ? 12 : 4;  rot[2] = q ? 20 : 8;  rot[3] = q ? 28 : 12;
	rot[4] = q ? 0 : 16;  rot[5] = q ? 8 : 20;  rot[6] = q ? 16 : 24; rot[7] = q ? 24 : 28;
	#pragma unroll 1
	for (uint32_t r = 0; r < 10; r++) groestl256_x4_round(s, r, q, rot);
}

/* digest word 7 (bytes 28..31) of the four hashes of this thread; in[h] = 8 LE words */
__device__ __forceinline__
void groestl256_x4_hash_32_w7(const uint32_t (&in)[4][8], uint32_t *st, const uint32_t stride, uint32_t (&w7)[4])
{
	/* row r: W[k] byte n = byte(row r, col 2n + (k>>2)) of hash k&3; cols 4..7 = padding */
	uint32_t s[8][8];
	#pragma unroll
	for (int r = 0; r < 8; r++) {
		#pragma unroll
		for (int k = 0; k < 8; k++) {
			const int h = k & 3, cb = k >> 2, rw = r >> 2, rb = r & 3;
			s[r][k] = __byte_perm(in[h][2 * cb + rw], in[h][4 + 2 * cb + rw], 0x4400 + rb + ((4 + rb) << 4)) & 0xFFFFu;
		}
		if (r == 0) { s[0][0] |= 0x00800000u; s[0][1] |= 0x00800000u; s[0][2] |= 0x00800000u; s[0][3] |= 0x00800000u; } /* col 4 row 0 = 0x80 */
		if (r == 7) { s[7][4] |= 0x01000000u; s[7][5] |= 0x01000000u; s[7][6] |= 0x01000000u; s[7][7] |= 0x01000000u; } /* col 7 row 7 = 0x01 */
		groestl512_x2_transpose(s[r]);
	}
	/* P(m ^ IV), Q(m), P(H) with H = P ^ Q ^ IV; IV = 0x01 at col 7 row 6 */
	#define ST(i, b) st[((i) * 8 + (b)) * stride]
	#pragma unroll 1
	for (int p = 0; p < 3; p++) {
		if (p == 0) {
			#pragma unroll
			for (int i = 0; i < 8; i++)
				#pragma unroll
				for (int b = 0; b < 8; b++) ST(i, b) = s[i][b];
			s[6][0] ^= 0xF0000000u;
		} else if (p == 1) {
			#pragma unroll
			for (int i = 0; i < 8; i++)
				#pragma unroll
				for (int b = 0; b < 8; b++) { const uint32_t v = ST(i, b); ST(i, b) = s[i][b]; s[i][b] = v; }
		} else {
			#pragma unroll
			for (int i = 0; i < 8; i++)
				#pragma unroll
				for (int b = 0; b < 8; b++) { s[i][b] ^= ST(i, b); ST(i, b) = s[i][b] ^ ((i == 6 && b == 0) ? 0xF0000000u : 0u); }
			s[6][0] ^= 0xF0000000u;
		}
		groestl256_x4_perm(s, p == 1);
	}
	/* feed-forward and un-transpose rows 4..7 only: word 7 = col 7 = W_row[4 + h] byte 3 */
	#pragma unroll
	for (int i = 4; i < 8; i++) {
		#pragma unroll
		for (int b = 0; b < 8; b++) s[i][b] ^= ST(i, b);
		groestl512_x2_transpose(s[i]);
	}
	#undef ST
	#pragma unroll
	for (int h = 0; h < 4; h++) {
		const uint32_t lo = __byte_perm(s[4][4 + h], s[5][4 + h], 0x0073);
		const uint32_t hi = __byte_perm(s[6][4 + h], s[7][4 + h], 0x0073);
		w7[h] = __byte_perm(lo, hi, 0x5410);
	}
}

__global__ __launch_bounds__(TPB, MINB)
void groestl256_gpu_hash_32_x4(uint32_t threads, uint32_t startNounce, uint64_t *outputHash, uint32_t *resNonces)
{
	extern __shared__ uint32_t stash[];                     /* [64][TPB] */
	const uint32_t i0 = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
	if (i0 >= threads) return;

	/* SoA input, as the lyra2 stages write it; slots past the batch reload the first hash */
	uint32_t in[4][8];
	#pragma unroll
	for (int h = 0; h < 4; h++) {
		const uint32_t i = (i0 + h < threads) ? i0 + h : i0;
		#pragma unroll
		for (int k = 0; k < 4; k++)
			LOHI(in[h][2 * k], in[h][2 * k + 1], outputHash[k * threads + i]);
	}

	uint32_t w7[4];
	groestl256_x4_hash_32_w7(in, &stash[threadIdx.x], blockDim.x, w7);

	#pragma unroll
	for (int h = 0; h < 4; h++) {
		if (i0 + h < threads && w7[h] <= pTarget[7]) {
			const uint32_t nonce = startNounce + i0 + h;
			// Keep the two lowest candidates in one pass. atomicMin returns the previous minimum, so
			// the displaced value moves to slot 1; reading resNonces[0] outside the atomic loses one.
			const uint32_t prev = atomicMin(&resNonces[0], nonce);
			if (prev != UINT32_MAX)
				atomicMin(&resNonces[1], max(prev, nonce));
		}
	}
}

bool groestl256_device_selftest(int thr_id);

__host__
void groestl256_cpu_init(int thr_id, uint32_t threads)
{
	cudaMalloc(&d_GNonces[thr_id], 2*sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2*sizeof(uint32_t));
	groestl256_device_selftest(thr_id);
}

__host__
void groestl256_cpu_free(int thr_id)
{
	cudaFree(d_GNonces[thr_id]);
	cudaFreeHost(h_GNonces[thr_id]);
}

__host__
uint32_t groestl256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	uint32_t result = UINT32_MAX;
	cudaMemset(d_GNonces[thr_id], 0xff, 2*sizeof(uint32_t));

	const uint32_t nthr = (threads + 3) / 4;   /* four hashes per thread */
	dim3 grid((nthr + TPB - 1) / TPB);
	dim3 block(TPB);

	groestl256_gpu_hash_32_x4<<<grid, block, 64 * TPB * sizeof(uint32_t)>>>(threads, startNounce, d_outputHash, d_GNonces[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);

	// get first found nonce
	cudaMemcpy(h_GNonces[thr_id], d_GNonces[thr_id], 1*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	result = *h_GNonces[thr_id];

	return result;
}

__host__
uint32_t groestl256_getSecNonce(int thr_id, int num)
{
	uint32_t results[2];
	memset(results, 0xFF, sizeof(results));
	cudaMemcpy(results, d_GNonces[thr_id], sizeof(results), cudaMemcpyDeviceToHost);
	if (results[1] == results[0])
		return UINT32_MAX;
	return results[num];
}

__host__
void groestl256_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32, 0, cudaMemcpyHostToDevice);
}

/* ------------------------------------------------------------------ self-test
 * Init-time KAT (the stage returns nonces): each slot alone must report at target v and not
 * at v - 1 (v = sph_groestl256 bytes 28..31), and a ragged batch must return the two lowest
 * candidates. Clobbers pTarget, which scanhash sets before every launch. Fail-closed. */
extern "C" {
#include "sph/sph_groestl.h"
}
#include "cuda/stage_selftest.cuh"

#define GROESTL256_ST_N     256
#define GROESTL256_ST_EXACT 64

static bool groestl256_st_launch(int thr_id, uint32_t threads, uint32_t start, uint64_t *d,
	uint32_t target, uint32_t *res)
{
	const uint32_t t[8] = { 0, 0, 0, 0, 0, 0, 0, target };
	groestl256_setTarget(t);
	res[0] = groestl256_cpu_hash_32(thr_id, threads, start, d, -1);
	res[1] = groestl256_getSecNonce(thr_id, 1);
	return cudaGetLastError() == cudaSuccess;
}

static int groestl256_st_cmp(const void *a, const void *b)
{
	const uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
	return (x > y) - (x < y);
}

__host__
bool groestl256_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	const int n = GROESTL256_ST_N, nr = GROESTL256_ST_N - 57;
	const uint32_t start = 0x7fffff00u;
	uint8_t *in = (uint8_t*)malloc((size_t)n * 32);
	uint64_t *soa = (uint64_t*)malloc((size_t)n * 32);
	uint32_t v[GROESTL256_ST_N];
	uint64_t *d_aos = NULL, *d_soa = NULL;
	if (!in || !soa || cudaMalloc(&d_aos, (size_t)n * 32) != cudaSuccess || cudaMalloc(&d_soa, (size_t)nr * 32) != cudaSuccess) {
		free(in); free(soa); cudaFree(d_aos);
		return selftest_gate(thr_id, "groestl256", selftest_cuda_fault());
	}
	stkat_fill(in, n, 32, 0x47523536u);
	for (int i = 0; i < n; i++) {
		uint8_t dg[32];
		sph_groestl256_context c;
		sph_groestl256_init(&c);
		sph_groestl256(&c, in + i * 32, 32);
		sph_groestl256_close(&c, dg);
		memcpy(&v[i], dg + 28, 4);
	}
	stkat_to_soa32(in, soa, nr);
	bool cuda_ok = cudaMemcpy(d_aos, in, (size_t)n * 32, cudaMemcpyHostToDevice) == cudaSuccess
		&& cudaMemcpy(d_soa, soa, (size_t)nr * 32, cudaMemcpyHostToDevice) == cudaSuccess;

	int bad_exact = 0, first = -1;
	uint32_t res[2];
	for (int i = 0; cuda_ok && i < GROESTL256_ST_EXACT; i++) {
		cuda_ok = groestl256_st_launch(thr_id, 1, start + i, d_aos + 4 * i, v[i], res);
		bool ok = res[0] == start + (uint32_t)i && res[1] == UINT32_MAX;
		if (cuda_ok && v[i] != 0) {
			cuda_ok = groestl256_st_launch(thr_id, 1, start + i, d_aos + 4 * i, v[i] - 1, res);
			ok = ok && res[0] == UINT32_MAX;
		}
		if (!ok) { if (first < 0) first = i; bad_exact++; }
	}

	/* targets: none, all, and the values of a few ranks */
	uint32_t sorted[GROESTL256_ST_N];
	memcpy(sorted, v, sizeof(uint32_t) * nr);
	qsort(sorted, nr, sizeof(uint32_t), groestl256_st_cmp);
	const uint32_t targets[] = { sorted[0] ? sorted[0] - 1 : 0, UINT32_MAX,
		sorted[0], sorted[1], sorted[2], sorted[7], sorted[31], sorted[nr / 2] };
	int bad_batch = 0;
	for (int t = 0; cuda_ok && t < (int)(sizeof(targets) / sizeof(targets[0])); t++) {
		uint32_t e[2] = { UINT32_MAX, UINT32_MAX };
		for (int i = 0; i < nr; i++)
			if (v[i] <= targets[t]) {
				if (e[0] == UINT32_MAX) e[0] = start + i;
				else if (e[1] == UINT32_MAX) e[1] = start + i;
			}
		cuda_ok = groestl256_st_launch(thr_id, nr, start, d_soa, targets[t], res);
		if (res[0] != e[0] || res[1] != e[1]) bad_batch++;
	}

	cudaFree(d_aos); cudaFree(d_soa); free(in); free(soa);
	if (!cuda_ok)
		return selftest_gate(thr_id, "groestl256", selftest_cuda_fault());
	passed = bad_exact == 0 && bad_batch == 0;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "groestl256 device self-test FAILED (exact: %d of %d slots wrong, first %d; batch: %d targets wrong)",
			bad_exact, GROESTL256_ST_EXACT, first, bad_batch);
	else
		gpulog(LOG_DEBUG, thr_id, "groestl256 device self-test passed");
	passed = selftest_gate(thr_id, "groestl256", passed);
	return passed;
}
