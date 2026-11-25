// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * sha3d CUDA kernel -- double NIST SHA3-256 over an 80-byte header
 * (BSHA3 / Yilacoin).
 *
 * Provenance: Algo256/cuda_keccak256_sha3d.cu, the tpruvot-era "compat"
 * kernel that was the only correct path (the pre-sm_35 branch hashed a
 * single permutation with Keccak 0x01 padding and is deleted). 2026-07:
 * permutation replaced by the shared cuda/keccak_device.cuh building
 * blocks; dual-nonce result buffer like the sibling kernels.
 */

#include <miner.h>

extern "C" {
#include <stdint.h>
#include <memory.h>
}

#include "cuda/helper.cuh"
#include "cuda/keccak_device.cuh"

extern bool keccak_device_selftest(int thr_id);

/* launch shape (swept 2026-07-12 on RTX 3060, see README) */
#define TPB 128
/* min blocks/SM, per-arch: Pascal wants 8 (64 regs, 50% occupancy), Ampere
 * measured worse there and keeps 5. sm_75 untested. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
#define BPM 8
#else
#define BPM 5
#endif
#define NPT 1   /* nonces per thread; the kernel below assumes exactly one */
static_assert(NPT == 1, "kernel handles one nonce per thread: raise the grid, not NPT");
#define NBN 2

static uint32_t *d_sha3d_nonces[MAX_GPUS];
static uint32_t *h_sha3d_nonces[MAX_GPUS];

/* per-GPU constant memory -- precomputed from the first 72 bytes of the
 * header (sha3t-style absorb midstate; same 0x06/0x80 padding layout).
 * Keep this block textually in sync with algos/sha3t/cuda_sha3t.cu and
 * algos/keccak/cuda_keccak256.cu. */
__constant__ uint2 c_sha3d_mid[17];
__constant__ uint2 c_sha3d_msg[6];

__global__ __launch_bounds__(TPB, BPM)
void sha3d_gpu_hash_80(uint32_t threads, uint32_t startNonce, uint32_t *resNonces, const uint2 highTarget)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint2 s[25], t[5], u[5], v, w;

	/* one nonce per thread; the grid rounds up, so this drops the ragged tail */
	if (thread < threads) {
	const uint32_t nonce = startNonce + thread;

	/* -- first SHA3-256: 80-byte header -> 32-byte digest -----------------
	 *
	 * Round 0 from the 72-byte midstate: s[9] carries the bits field in .x
	 * and the per-thread nonce (byte-swapped) in .y.
	 * s[10] = 0x06 -- NIST SHA3-256 domain separator (NOT 0x01 Keccak);
	 * s[16] = end-of-rate bit (byte 135). */

	s[ 9] = make_uint2(c_sha3d_msg[0].x, cuda_swab32(nonce));
	s[10] = make_uint2(0x06, 0);

	t[ 4] = c_sha3d_msg[1] ^ s[ 9];
	u[ 0] = t[4] ^ c_sha3d_mid[ 0];
	u[ 1] = c_sha3d_mid[ 1] ^ ROL2(t[4], 1);
	u[ 2] = c_sha3d_mid[ 2];

	s[ 7] = ROL2(s[10] ^ u[0], 3);   /* rho-pi of padded lane 10 */
	s[10] = c_sha3d_mid[ 3];
	     w = c_sha3d_mid[ 4];
	s[20] = c_sha3d_mid[ 5];
	s[ 6] = ROL2(s[ 9] ^ u[2], 20);
	s[ 9] = c_sha3d_mid[ 6];
	s[22] = c_sha3d_mid[ 7];
	s[14] = ROL2(u[0], 18);
	s[ 2] = c_sha3d_mid[ 8];
	s[12] = ROL2(u[1], 25);
	s[13] = c_sha3d_mid[ 9];
	s[19] = ROR8(u[1]);
	s[23] = ROR2(u[0], 23);
	s[15] = c_sha3d_mid[10];
	s[ 4] = c_sha3d_mid[11];
	s[24] = c_sha3d_mid[12];
	s[21] = ROR2(c_sha3d_msg[2] ^ u[1], 9);
	s[ 8] = c_sha3d_mid[13];
	s[16] = ROR2(c_sha3d_msg[3] ^ u[0], 28);
	s[ 5] = ROL2(c_sha3d_msg[4] ^ u[1], 28);
	s[ 3] = ROL2(u[1], 21);
	s[18] = c_sha3d_mid[14];
	s[17] = c_sha3d_mid[15];
	s[11] = c_sha3d_mid[16];

	v = c_sha3d_msg[5] ^ u[0];
	s[ 0] = keccak_chi(v,    w,    s[ 2]);
	s[ 1] = keccak_chi(w,    s[ 2], s[ 3]);
	s[ 2] = keccak_chi(s[ 2],s[ 3], s[ 4]);
	s[ 3] = keccak_chi(s[ 3],s[ 4], v   );
	s[ 4] = keccak_chi(s[ 4],v,     w   );
	v=s[ 5];w=s[ 6]; s[ 5]=keccak_chi(v,w,s[ 7]); s[ 6]=keccak_chi(w,s[ 7],s[ 8]); s[ 7]=keccak_chi(s[ 7],s[ 8],s[ 9]); s[ 8]=keccak_chi(s[ 8],s[ 9],v); s[ 9]=keccak_chi(s[ 9],v,w);
	v=s[10];w=s[11]; s[10]=keccak_chi(v,w,s[12]); s[11]=keccak_chi(w,s[12],s[13]); s[12]=keccak_chi(s[12],s[13],s[14]); s[13]=keccak_chi(s[13],s[14],v); s[14]=keccak_chi(s[14],v,w);
	v=s[15];w=s[16]; s[15]=keccak_chi(v,w,s[17]); s[16]=keccak_chi(w,s[17],s[18]); s[17]=keccak_chi(s[17],s[18],s[19]); s[18]=keccak_chi(s[18],s[19],v); s[19]=keccak_chi(s[19],v,w);
	v=s[20];w=s[21]; s[20]=keccak_chi(v,w,s[22]); s[21]=keccak_chi(w,s[22],s[23]); s[22]=keccak_chi(s[22],s[23],s[24]); s[23]=keccak_chi(s[23],s[24],v); s[24]=keccak_chi(s[24],v,w);
	s[ 0] ^= c_keccak_rc[0];

	/* rounds 1-23 */
	#pragma unroll 23
	for (int i = 1; i < 24; i++)
		keccak_round(s, c_keccak_rc[i]);

	/* second SHA3-256 over the 32-byte digest s[0..3]; the last round is
	 * truncated to lane 3, which is all the target compare reads */
	#pragma unroll
	for (int i = 5; i < 25; i++)
		s[i] = make_uint2(0, 0);
	s[ 4] = make_uint2(0x06, 0);
	s[16] = make_uint2(0, 0x80000000);

	#pragma unroll 23
	for (int i = 0; i < 23; i++)
		keccak_round(s, c_keccak_rc[i]);

	/* final_hash[6..7] == lane 3: 64-bit target compare;
	 * every candidate is re-hashed on the CPU before submit */
	if (devectorize(keccak_final_lane3(s)) <= devectorize(highTarget)) {
		/* Keep the two LOWEST nonces: the host resumes from max(slot0,slot1)+1,
		 * so anything higher steps over an unreported candidate. Both slots
		 * atomic; the loser comes from atomicMin's return, not a re-read. */
		const uint32_t prev = atomicMin(&resNonces[0], nonce);
		if (prev != UINT32_MAX)
			atomicMin(&resNonces[1], (prev > nonce) ? prev : nonce);
	}
	}
}

__host__
void sha3d_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces, const uint2 highTarget)
{
	const dim3 grid((threads + (NPT * TPB) - 1) / (NPT * TPB));
	const dim3 block(TPB);

	sha3d_gpu_hash_80 <<<grid, block>>> (threads, startNonce, d_sha3d_nonces[thr_id], highTarget);

	cudaMemcpy(h_sha3d_nonces[thr_id], d_sha3d_nonces[thr_id], NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	memcpy(resNonces, h_sha3d_nonces[thr_id], NBN * sizeof(uint32_t));
}

/* Precompute midstate from the first 72 bytes of the block header.
 * endiandata[0..9] = 80-byte header in LE uint64 pairs (already
 * endian-swapped). Identical to sha3t_setBlock_80 (same 0x06 padding). */
__host__
void sha3d_setBlock_80(uint64_t *endiandata)
{
	uint64_t mid[17], s[25];
	uint64_t t[5], u[5];

	s[10] = 6;                          /* NIST SHA3-256 domain separator */
	s[16] = ((uint64_t)1) << 63;       /* end-of-rate: bit 1087 */

	t[0] = endiandata[0] ^ endiandata[5] ^ s[10];
	t[1] = endiandata[1] ^ endiandata[6] ^ s[16];
	t[2] = endiandata[2] ^ endiandata[7];
	t[3] = endiandata[3] ^ endiandata[8];
	/* t[4] depends on endiandata[9] which contains the nonce -- done per-thread */

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

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sha3d_mid, mid,
	               17 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));

	uint64_t msg[6];
	msg[0] = endiandata[9];
	msg[1] = endiandata[4];
	msg[2] = endiandata[8];
	msg[3] = endiandata[5];
	msg[4] = endiandata[3];
	msg[5] = endiandata[0];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sha3d_msg, msg,
	               6 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void sha3d_setOutput(int thr_id)
{
	CUDA_SAFE_CALL(cudaMemset(d_sha3d_nonces[thr_id], 0xff, NBN * sizeof(uint32_t)));
}

__host__
void sha3d_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_sha3d_nonces[thr_id], NBN * sizeof(uint32_t)));
	h_sha3d_nonces[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
	if (h_sha3d_nonces[thr_id] == NULL) {
		gpulog(LOG_ERR, thr_id, "Host memory allocation failed");
		exit(EXIT_FAILURE);
	}
	keccak_device_selftest(thr_id);
}

__host__
void sha3d_cpu_free(int thr_id)
{
	cudaFree(d_sha3d_nonces[thr_id]);
	free(h_sha3d_nonces[thr_id]);
}
