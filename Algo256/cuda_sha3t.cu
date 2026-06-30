/**
 * SHA3-256t (triple NIST SHA3-256) CUDA implementation
 *
 * Author: Pkules (https://github.com/Pkules)
 *
 * Three sequential rounds of NIST FIPS 202 SHA3-256 over an 80-byte
 * block header. Uses midstate precomputation for the first round and
 * a compact full-permutation device function for rounds 2 and 3.
 *
 * Donations: bc3 / bc1 — address in your miner config
 */

#include <miner.h>

extern "C" {
#include <stdint.h>
#include <memory.h>
}

#include <cuda_helper.h>
#include <cuda_vectors.h>

#define TPB52 512
#define TPB50 256
#define NPT   2
#define NBN   2

static uint32_t *d_sha3t_nonces[MAX_GPUS];
static uint32_t *h_sha3t_nonces[MAX_GPUS];

/* per-GPU constant memory — precomputed from the first 72 bytes of the header */
__constant__ uint2 c_sha3t_mid[17];
__constant__ uint2 c_sha3t_msg[6];

__constant__ uint2 sha3t_round_constants[24] = {
	{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 },
	{ 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
	{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 },
	{ 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
	{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 },
	{ 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
	{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 },
	{ 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
	{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 },
	{ 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
	{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 },
	{ 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

/* ── device helpers ─────────────────────────────────────────────────────────── */

__device__ __forceinline__
uint2 chi(const uint2 a, const uint2 b, const uint2 c) {
	uint2 r;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm("lop3.b32 %0,%1,%2,%3,0xD2;" : "=r"(r.x) : "r"(a.x),"r"(b.x),"r"(c.x));
	asm("lop3.b32 %0,%1,%2,%3,0xD2;" : "=r"(r.y) : "r"(a.y),"r"(b.y),"r"(c.y));
#else
	r = a ^ (~b & c);
#endif
	return r;
}

__device__ __forceinline__
uint64_t xor5(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {
	uint64_t r;
	asm("xor.b64 %0,%1,%2;" : "=l"(r) : "l"(d),"l"(e));
	asm("xor.b64 %0,%0,%1;" : "+l"(r) : "l"(c));
	asm("xor.b64 %0,%0,%1;" : "+l"(r) : "l"(b));
	asm("xor.b64 %0,%0,%1;" : "+l"(r) : "l"(a));
	return r;
}

/* Full 24-round Keccak-f[1600] permutation.
 * Used for rounds 2 and 3 (32-byte input with SHA3-256 NIST padding).
 * Caller must initialise s[0..24] before calling. */
__device__ __forceinline__
void sha3t_keccakf(uint2 s[25])
{
	uint2 t[5], v, w, u[5];
#pragma unroll 24
	for (int i = 0; i < 24; i++) {
		/* theta — column parities */
#pragma unroll 5
		for (int j = 0; j < 5; j++)
			t[j] = vectorize(xor5(devectorize(s[j]),   devectorize(s[j+5]),
			                      devectorize(s[j+10]), devectorize(s[j+15]),
			                      devectorize(s[j+20])));
#pragma unroll 5
		for (int j = 0; j < 5; j++) u[j] = ROL2(t[j], 1);

		s[ 4]=xor3x(s[ 4],t[3],u[0]); s[ 9]=xor3x(s[ 9],t[3],u[0]);
		s[14]=xor3x(s[14],t[3],u[0]); s[19]=xor3x(s[19],t[3],u[0]); s[24]=xor3x(s[24],t[3],u[0]);
		s[ 0]=xor3x(s[ 0],t[4],u[1]); s[ 5]=xor3x(s[ 5],t[4],u[1]);
		s[10]=xor3x(s[10],t[4],u[1]); s[15]=xor3x(s[15],t[4],u[1]); s[20]=xor3x(s[20],t[4],u[1]);
		s[ 1]=xor3x(s[ 1],t[0],u[2]); s[ 6]=xor3x(s[ 6],t[0],u[2]);
		s[11]=xor3x(s[11],t[0],u[2]); s[16]=xor3x(s[16],t[0],u[2]); s[21]=xor3x(s[21],t[0],u[2]);
		s[ 2]=xor3x(s[ 2],t[1],u[3]); s[ 7]=xor3x(s[ 7],t[1],u[3]);
		s[12]=xor3x(s[12],t[1],u[3]); s[17]=xor3x(s[17],t[1],u[3]); s[22]=xor3x(s[22],t[1],u[3]);
		s[ 3]=xor3x(s[ 3],t[2],u[4]); s[ 8]=xor3x(s[ 8],t[2],u[4]);
		s[13]=xor3x(s[13],t[2],u[4]); s[18]=xor3x(s[18],t[2],u[4]); s[23]=xor3x(s[23],t[2],u[4]);

		/* rho-pi */
		v = s[1];
		s[ 1]=ROL2(s[ 6],44); s[ 6]=ROL2(s[ 9],20); s[ 9]=ROL2(s[22],61); s[22]=ROL2(s[14],39);
		s[14]=ROL2(s[20],18); s[20]=ROL2(s[ 2],62); s[ 2]=ROL2(s[12],43); s[12]=ROL2(s[13],25);
		s[13]=ROL8(s[19]);    s[19]=ROR8(s[23]);     s[23]=ROL2(s[15],41); s[15]=ROL2(s[ 4],27);
		s[ 4]=ROL2(s[24],14); s[24]=ROL2(s[21], 2); s[21]=ROL2(s[ 8],55); s[ 8]=ROL2(s[16],45);
		s[16]=ROL2(s[ 5],36); s[ 5]=ROL2(s[ 3],28); s[ 3]=ROL2(s[18],21); s[18]=ROL2(s[17],15);
		s[17]=ROL2(s[11],10); s[11]=ROL2(s[ 7], 6); s[ 7]=ROL2(s[10], 3); s[10]=ROL2(v, 1);

		/* chi */
#pragma unroll 5
		for (int j = 0; j < 25; j += 5) {
			v=s[j]; w=s[j+1];
			s[j]  =chi(v,    w,    s[j+2]);
			s[j+1]=chi(w,    s[j+2],s[j+3]);
			s[j+2]=chi(s[j+2],s[j+3],s[j+4]);
			s[j+3]=chi(s[j+3],s[j+4],v);
			s[j+4]=chi(s[j+4],v,    w);
		}

		/* iota */
		s[0] ^= sha3t_round_constants[i];
	}
}

/* ── main GPU kernel ─────────────────────────────────────────────────────────── */

#if __CUDA_ARCH__ <= 500
__global__ __launch_bounds__(TPB50, 2)
#else
__global__ __launch_bounds__(TPB52, 1)
#endif
void sha3t_gpu_hash_80(uint32_t threads, uint32_t startNonce,
                       uint32_t *resNounce, const uint2 highTarget)
{
	uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	uint2 s[25], t[5], v, w, u[5];

#if __CUDA_ARCH__ > 500
	uint64_t step     = (uint64_t)gridDim.x * blockDim.x;
	uint64_t maxNonce = (uint64_t)startNonce + threads;
	for (uint64_t nounce = startNonce + thread; nounce < maxNonce; nounce += step) {
#else
	uint32_t nounce = startNonce + thread;
	if (thread < threads) {
#endif
		/* ── round 1: 80-byte header → 32-byte hash1 ──────────────────────────
		 *
		 * Midstate covers the first 72 bytes (s[0..8]).  s[9] carries the bits
		 * field in .x and the per-thread nonce (byte-swapped) in .y.
		 * s[10] = 0x06  ← NIST SHA3-256 domain separator (NOT 0x01 Keccak).
		 * s[16] = {0, 0x80000000}  ← end-of-rate bit.
		 * All other lanes initialised to 0 by the midstate constants. */

		s[ 9] = make_uint2(c_sha3t_msg[0].x, cuda_swab32(nounce));
		s[10] = make_uint2(6, 0);   /* SHA3-256 padding — critical difference */

		t[ 4] = c_sha3t_msg[1] ^ s[ 9];
		u[ 0] = t[4] ^ c_sha3t_mid[ 0];
		u[ 1] = c_sha3t_mid[ 1] ^ ROL2(t[4], 1);
		u[ 2] = c_sha3t_mid[ 2];

		s[ 7] = ROL2(s[10] ^ u[0], 3);   /* rho-pi of padded lane 10 */
		s[10] = c_sha3t_mid[ 3];
		     w = c_sha3t_mid[ 4];
		s[20] = c_sha3t_mid[ 5];
		s[ 6] = ROL2(s[ 9] ^ u[2], 20);
		s[ 9] = c_sha3t_mid[ 6];
		s[22] = c_sha3t_mid[ 7];
		s[14] = ROL2(u[0], 18);
		s[ 2] = c_sha3t_mid[ 8];
		s[12] = ROL2(u[1], 25);
		s[13] = c_sha3t_mid[ 9];
		s[19] = ROR8(u[1]);
		s[23] = ROR2(u[0], 23);
		s[15] = c_sha3t_mid[10];
		s[ 4] = c_sha3t_mid[11];
		s[24] = c_sha3t_mid[12];
		s[21] = ROR2(c_sha3t_msg[2] ^ u[1], 9);
		s[ 8] = c_sha3t_mid[13];
		s[16] = ROR2(c_sha3t_msg[3] ^ u[0], 28);
		s[ 5] = ROL2(c_sha3t_msg[4] ^ u[1], 28);
		s[ 3] = ROL2(u[1], 21);
		s[18] = c_sha3t_mid[14];
		s[17] = c_sha3t_mid[15];
		s[11] = c_sha3t_mid[16];

		v = c_sha3t_msg[5] ^ u[0];
		s[ 0] = chi(v,    w,    s[ 2]);
		s[ 1] = chi(w,    s[ 2], s[ 3]);
		s[ 2] = chi(s[ 2],s[ 3], s[ 4]);
		s[ 3] = chi(s[ 3],s[ 4], v   );
		s[ 4] = chi(s[ 4],v,     w   );
		v=s[ 5];w=s[ 6]; s[ 5]=chi(v,w,s[ 7]); s[ 6]=chi(w,s[ 7],s[ 8]); s[ 7]=chi(s[ 7],s[ 8],s[ 9]); s[ 8]=chi(s[ 8],s[ 9],v); s[ 9]=chi(s[ 9],v,w);
		v=s[10];w=s[11]; s[10]=chi(v,w,s[12]); s[11]=chi(w,s[12],s[13]); s[12]=chi(s[12],s[13],s[14]); s[13]=chi(s[13],s[14],v); s[14]=chi(s[14],v,w);
		v=s[15];w=s[16]; s[15]=chi(v,w,s[17]); s[16]=chi(w,s[17],s[18]); s[17]=chi(s[17],s[18],s[19]); s[18]=chi(s[18],s[19],v); s[19]=chi(s[19],v,w);
		v=s[20];w=s[21]; s[20]=chi(v,w,s[22]); s[21]=chi(w,s[22],s[23]); s[22]=chi(s[22],s[23],s[24]); s[23]=chi(s[23],s[24],v); s[24]=chi(s[24],v,w);
		s[ 0] ^= sha3t_round_constants[0];

		/* rounds 1-23: full permutation (24 rounds total for round 1) */
#if __CUDA_ARCH__ > 500
#pragma unroll 23
#else
#pragma unroll 4
#endif
		for (int i = 1; i < 24; i++) {
#pragma unroll 5
			for (int j = 0; j < 5; j++)
				t[j] = vectorize(xor5(devectorize(s[j]),   devectorize(s[j+5]),
				                      devectorize(s[j+10]), devectorize(s[j+15]),
				                      devectorize(s[j+20])));
#pragma unroll 5
			for (int j = 0; j < 5; j++) u[j] = ROL2(t[j], 1);

			s[ 4]=xor3x(s[ 4],t[3],u[0]); s[ 9]=xor3x(s[ 9],t[3],u[0]); s[14]=xor3x(s[14],t[3],u[0]); s[19]=xor3x(s[19],t[3],u[0]); s[24]=xor3x(s[24],t[3],u[0]);
			s[ 0]=xor3x(s[ 0],t[4],u[1]); s[ 5]=xor3x(s[ 5],t[4],u[1]); s[10]=xor3x(s[10],t[4],u[1]); s[15]=xor3x(s[15],t[4],u[1]); s[20]=xor3x(s[20],t[4],u[1]);
			s[ 1]=xor3x(s[ 1],t[0],u[2]); s[ 6]=xor3x(s[ 6],t[0],u[2]); s[11]=xor3x(s[11],t[0],u[2]); s[16]=xor3x(s[16],t[0],u[2]); s[21]=xor3x(s[21],t[0],u[2]);
			s[ 2]=xor3x(s[ 2],t[1],u[3]); s[ 7]=xor3x(s[ 7],t[1],u[3]); s[12]=xor3x(s[12],t[1],u[3]); s[17]=xor3x(s[17],t[1],u[3]); s[22]=xor3x(s[22],t[1],u[3]);
			s[ 3]=xor3x(s[ 3],t[2],u[4]); s[ 8]=xor3x(s[ 8],t[2],u[4]); s[13]=xor3x(s[13],t[2],u[4]); s[18]=xor3x(s[18],t[2],u[4]); s[23]=xor3x(s[23],t[2],u[4]);

			v = s[1];
			s[ 1]=ROL2(s[ 6],44); s[ 6]=ROL2(s[ 9],20); s[ 9]=ROL2(s[22],61); s[22]=ROL2(s[14],39);
			s[14]=ROL2(s[20],18); s[20]=ROL2(s[ 2],62); s[ 2]=ROL2(s[12],43); s[12]=ROL2(s[13],25);
			s[13]=ROL8(s[19]);    s[19]=ROR8(s[23]);     s[23]=ROL2(s[15],41); s[15]=ROL2(s[ 4],27);
			s[ 4]=ROL2(s[24],14); s[24]=ROL2(s[21], 2); s[21]=ROL2(s[ 8],55); s[ 8]=ROL2(s[16],45);
			s[16]=ROL2(s[ 5],36); s[ 5]=ROL2(s[ 3],28); s[ 3]=ROL2(s[18],21); s[18]=ROL2(s[17],15);
			s[17]=ROL2(s[11],10); s[11]=ROL2(s[ 7], 6); s[ 7]=ROL2(s[10], 3); s[10]=ROL2(v, 1);

#pragma unroll 5
			for (int j = 0; j < 25; j += 5) {
				v=s[j]; w=s[j+1];
				s[j]  =chi(v,    w,    s[j+2]);
				s[j+1]=chi(w,    s[j+2],s[j+3]);
				s[j+2]=chi(s[j+2],s[j+3],s[j+4]);
				s[j+3]=chi(s[j+3],s[j+4],v);
				s[j+4]=chi(s[j+4],v,    w);
			}
			s[0] ^= sha3t_round_constants[i];
		}

		/* hash1 = first 32 bytes of state = s[0..3] */
		uint2 h0 = s[0], h1 = s[1], h2 = s[2], h3 = s[3];

		/* ── round 2: SHA3-256(hash1) ─────────────────────────────────────── */
		s[ 0]=h0; s[ 1]=h1; s[ 2]=h2; s[ 3]=h3;
		s[ 4]=make_uint2(6,0);              /* padding at byte 32 */
		s[ 5]=make_uint2(0,0); s[ 6]=make_uint2(0,0); s[ 7]=make_uint2(0,0);
		s[ 8]=make_uint2(0,0); s[ 9]=make_uint2(0,0); s[10]=make_uint2(0,0);
		s[11]=make_uint2(0,0); s[12]=make_uint2(0,0); s[13]=make_uint2(0,0);
		s[14]=make_uint2(0,0); s[15]=make_uint2(0,0);
		s[16]=make_uint2(0,0x80000000);     /* end-of-rate bit at byte 135 */
		s[17]=make_uint2(0,0); s[18]=make_uint2(0,0); s[19]=make_uint2(0,0);
		s[20]=make_uint2(0,0); s[21]=make_uint2(0,0); s[22]=make_uint2(0,0);
		s[23]=make_uint2(0,0); s[24]=make_uint2(0,0);

		sha3t_keccakf(s);

		h0=s[0]; h1=s[1]; h2=s[2]; h3=s[3];

		/* ── round 3: SHA3-256(hash2) ─────────────────────────────────────── */
		s[ 0]=h0; s[ 1]=h1; s[ 2]=h2; s[ 3]=h3;
		s[ 4]=make_uint2(6,0);
		s[ 5]=make_uint2(0,0); s[ 6]=make_uint2(0,0); s[ 7]=make_uint2(0,0);
		s[ 8]=make_uint2(0,0); s[ 9]=make_uint2(0,0); s[10]=make_uint2(0,0);
		s[11]=make_uint2(0,0); s[12]=make_uint2(0,0); s[13]=make_uint2(0,0);
		s[14]=make_uint2(0,0); s[15]=make_uint2(0,0);
		s[16]=make_uint2(0,0x80000000);
		s[17]=make_uint2(0,0); s[18]=make_uint2(0,0); s[19]=make_uint2(0,0);
		s[20]=make_uint2(0,0); s[21]=make_uint2(0,0); s[22]=make_uint2(0,0);
		s[23]=make_uint2(0,0); s[24]=make_uint2(0,0);

		sha3t_keccakf(s);

		/* final_hash[6..7] == devectorize(s[3]).  Compare 64 bits for precision. */
		if (devectorize(s[3]) <= devectorize(highTarget)) {
			const uint32_t tmp = atomicExch(&resNounce[0], (uint32_t)nounce);
			if (tmp != UINT32_MAX)
				resNounce[1] = tmp;
		}
	}
}

/* ── host functions ──────────────────────────────────────────────────────────── */

__host__
void sha3t_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce,
                       uint32_t *resNonces, const uint2 highTarget)
{
	uint32_t tpb;
	dim3 grid;
	if (device_sm[device_map[thr_id]] <= 500) {
		tpb = TPB50;
		grid.x = (threads + tpb - 1) / tpb;
	} else {
		tpb = TPB52;
		grid.x = (threads + (NPT * tpb) - 1) / (NPT * tpb);
	}
	const dim3 block(tpb);

	sha3t_gpu_hash_80<<<grid, block>>>(threads, startNonce,
	                                    d_sha3t_nonces[thr_id], highTarget);
	cudaMemcpy(h_sha3t_nonces[thr_id], d_sha3t_nonces[thr_id],
	           NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	memcpy(resNonces, h_sha3t_nonces[thr_id], NBN * sizeof(uint32_t));
}

/* Precompute midstate from the first 72 bytes of the block header.
 * endiandata[0..9] = 80-byte header in LE uint64 pairs (already endian-swapped).
 *
 * SHA3-256 differs from Keccak-256 only in s[10] = 6 (vs 1).
 * Everything else is structurally identical to keccak256_setBlock_80. */
__host__
void sha3t_setBlock_80(uint64_t *endiandata)
{
	uint64_t mid[17], s[25];
	uint64_t t[5], u[5];

	s[10] = 6;                          /* NIST SHA3-256 domain separator */
	s[16] = ((uint64_t)1) << 63;       /* end-of-rate: bit 1087 */

	t[0] = endiandata[0] ^ endiandata[5] ^ s[10];
	t[1] = endiandata[1] ^ endiandata[6] ^ s[16];
	t[2] = endiandata[2] ^ endiandata[7];
	t[3] = endiandata[3] ^ endiandata[8];
	/* t[4] depends on endiandata[9] which contains the nonce — done per-thread */

	mid[ 0] = ROTL64(t[1], 1);
	     u[1] = t[0] ^ ROTL64(t[2], 1);
	     u[2] = t[1] ^ ROTL64(t[3], 1);
	mid[ 1] = t[2];
	mid[ 2] = t[3] ^ ROTL64(t[0], 1);
	mid[ 3] = ROTL64(endiandata[1] ^ u[1], 1);
	mid[ 4] = ROTL64(endiandata[6] ^ u[1], 44);
	mid[ 5] = ROTL64(endiandata[2] ^ u[2], 62);
	mid[ 6] = ROTL64(u[2], 61);
	mid[ 7] = ROTL64(mid[2], 39);
	mid[ 8] = ROTL64(u[2], 43);
	mid[ 9] = ROTL64(mid[2], 8);
	mid[10] = ROTL64(endiandata[4] ^ mid[2], 27);
	mid[11] = ROTL64(mid[2], 14);
	mid[12] = ROTL64(u[1], 2);
	mid[13] = ROTL64(s[16] ^ u[1], 45);
	mid[14] = ROTL64(u[2], 15);
	mid[15] = ROTL64(u[1], 10);
	mid[16] = ROTL64(endiandata[7] ^ u[2], 6);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sha3t_mid, mid,
	               17 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));

	uint64_t msg[6];
	msg[0] = endiandata[9];
	msg[1] = endiandata[4];
	msg[2] = endiandata[8];
	msg[3] = endiandata[5];
	msg[4] = endiandata[3];
	msg[5] = endiandata[0];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sha3t_msg, msg,
	               6 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void sha3t_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_sha3t_nonces[thr_id], NBN * sizeof(uint32_t)));
	h_sha3t_nonces[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
	if (!h_sha3t_nonces[thr_id]) {
		gpulog(LOG_ERR, thr_id, "Host memory allocation failed");
		exit(EXIT_FAILURE);
	}
}

__host__
void sha3t_setOutput(int thr_id)
{
	CUDA_SAFE_CALL(cudaMemset(d_sha3t_nonces[thr_id], 0xff, NBN * sizeof(uint32_t)));
}

__host__
void sha3t_cpu_free(int thr_id)
{
	cudaFree(d_sha3t_nonces[thr_id]);
	free(h_sha3t_nonces[thr_id]);
}
