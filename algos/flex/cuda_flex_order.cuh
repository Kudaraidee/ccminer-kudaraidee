/**
 * Per-nonce chain derivation on device, for Flex.
 *
 * Flex seeds its selection from SHA3-512 of the WHOLE 80-byte header, so the
 * core order and the CN triple change with every nonce.  That is what breaks
 * the x16r/ghostrider "derive once per job, then run 18 whole-batch kernels"
 * model, and it is why this file exists: it computes each lane's chain on the
 * GPU so the host can sort lanes into per-algo buckets and still call the
 * existing shared stage kernels unmodified.
 *
 * Header-only on purpose (the `cuda/intensity_autotune.cuh` precedent): every
 * kernel and launcher here is `static`, so including it from a second TU costs
 * nothing and needs no entry in any of the four build files.
 *
 * Everything is gated against the CPU reference in algos/flex/flex_hash.cpp --
 * see the project notes
 */

#ifndef CUDA_FLEX_ORDER_CUH
#define CUDA_FLEX_ORDER_CUH

#include <stdint.h>
#include "algos/flex/cuda_flex_sha3.cuh"
#include "algos/flex/flex.h"

#ifdef __CUDACC__

/* Header words 0..18.  Word 19 is the nonce and is computed per lane, so it is
 * deliberately NOT part of the symbol. */
static __constant__ uint32_t c_flex_head[19];

/* Copy sizeof(the SYMBOL), never sizeof(the source): CUDA 11.8 rejects an
 * oversized cudaMemcpyToSymbol with cudaErrorInvalidValue, copies nothing, and
 * surfaces the error at the NEXT cudaGetLastError -- i.e. one launch late, on
 * an unrelated line. */
static void flex_seed_setBlock_80(const uint32_t *endiandata)
{
	cudaMemcpyToSymbol(c_flex_head, endiandata, sizeof(c_flex_head), 0, cudaMemcpyHostToDevice);
}

/* ---------------------------------------------------------------------------
 * SHA3-512 of the 80-byte header for one lane.
 *
 * Block 1 is header bytes 0..71 = words 0..17, which are job-constant, so the
 * whole first permutation is a per-job midstate and could be hoisted to the
 * host.  Not done yet -- it needs a host keccak-f1600 with its own gate, and
 * it buys one permutation per lane against a chain of 15 core rounds and 3 CN
 * rounds.  Recorded in the plan as a a separate item so it is not re-discovered.
 *
 * `rawNonce` is in the **pdata frame** -- the same frame every x16
 * `*_cuda_hash_80` takes -- NOT the byteswapped `endiandata` frame the rest of
 * the header arrives in.  Both frames are live in this tree at once:
 * ghostrider derives its order from `endiandata` while handing `pdata[19]` to
 * hash_80.  Feeding one value to both is invisible -- the miner runs, the
 * hashrate is stable, every digest is wrong.  Header bytes 76..79 hold the
 * BIG-endian nonce, so read back as a little-endian lane word they are
 * `swab32(rawNonce)`; that conversion happens here, once, so every caller can
 * pass the raw nonce exactly like every other 80-byte stage.
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void flex_sha3_512_seed_80(uint32_t rawNonce, uint2 seed[8])
{
	uint2 s[25];
	const uint32_t nonce = cuda_swab32(rawNonce);   /* -> endiandata frame */

	#pragma unroll 9
	for (int i = 0; i < 9; i++)
		s[i] = make_uint2(c_flex_head[2 * i], c_flex_head[2 * i + 1]);
	#pragma unroll 16
	for (int i = 9; i < 25; i++)
		s[i] = make_uint2(0, 0);

	keccakf1600_full(s);

	/* Block 2: bytes 72..79 = (word 18, nonce) -> lane 0, then the padding.
	 * Lanes 9..24 carry over from the first permutation and must not be
	 * cleared; only the rate lanes are XORed. */
	s[0] = make_uint2(s[0].x ^ c_flex_head[18], s[0].y ^ nonce);
	s[1] = make_uint2(s[1].x ^ 6u, s[1].y);            /* SHA-3 domain byte */
	s[8] = make_uint2(s[8].x, s[8].y ^ 0x80000000u);   /* end of rate */

	keccakf1600_full(s);

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		seed[i] = s[i];
}

/* ---------------------------------------------------------------------------
 * The nibble walk.
 *
 * Mirrors flex_get_algo_string(): low nibble of each byte first, each reduced
 * % count, first occurrence wins, then any value that never appeared is
 * back-filled in ascending order.  Reads size/2 = 32 of the 64 seed bytes.
 *
 * The "seen" set is a BITMASK rather than a bool[16] so nothing lands in local
 * memory; a dynamically indexed array here would put a frame on every lane of
 * the seed kernel.
 * ------------------------------------------------------------------------- */
__device__ __forceinline__
void flex_walk_nibbles(const uint8_t * __restrict__ seed, uint8_t * __restrict__ out,
                       uint32_t count, uint32_t stride)
{
	uint32_t seen = 0;
	uint32_t n = 0;

	for (uint32_t i = 0; i < 32 && n < count; i++) {
		const uint8_t b = seed[i];
		uint32_t d = (uint32_t)(b & 0x0F) % count;
		if (!((seen >> d) & 1u)) { seen |= 1u << d; out[n * stride] = (uint8_t)d; n++; }
		d = (uint32_t)(b >> 4) % count;
		if (!((seen >> d) & 1u)) { seen |= 1u << d; out[n * stride] = (uint8_t)d; n++; }
	}
	/* Back-fill. The CPU reference does this too; it fires whenever 32 bytes
	 * of nibbles did not cover the whole pool. */
	for (uint32_t d = 0; d < count && n < count; d++)
		if (!((seen >> d) & 1u)) { out[n * stride] = (uint8_t)d; n++; }
}

/* One lane -> its chain.
 *
 * d_core is [FLEX_CORE_CHAIN_LEN][stride], d_cn is [FLEX_CN_ALGO_COUNT][stride],
 * transposed so device writes coalesce and the host reads one contiguous run
 * per round.
 *
 * stride is the CAPACITY the arrays were allocated with; threads is how many
 * lanes this batch derives. They differ whenever a caller derives fewer lanes
 * than the allocation, and conflating them hands every lane another round's
 * algorithm. Keep them separate.
 *
 * Slot 14 of the core order is 0 (blake512) explicitly: selection only fills
 * 0..13, but the chain reads slot 14. That is consensus.
 *
 * `d_nonces` is the nonce filter's compacted nonce list. When it is NULL the
 * batch is the contiguous run startNonce..startNonce+threads, which is the
 * unfiltered path; when it is set, lane t hashes d_nonces[t] and the lanes
 * are SCATTERED in nonce space. Everything downstream is unaffected because
 * each lane's chain depends only on its own nonce. */
static __global__
void flex_seed_order_gpu(uint32_t threads, uint32_t stride, uint32_t startNonce,
                         uint8_t * __restrict__ d_core, uint8_t * __restrict__ d_cn,
                         uint2 * __restrict__ d_seed,
                         const uint32_t * __restrict__ d_nonces)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	uint2 seed[8];
	flex_sha3_512_seed_80(d_nonces ? d_nonces[thread] : (startNonce + thread), seed);

	/* The nibble walk reads the seed as bytes, little-endian per lane, which
	 * is how the CPU reference reads its sph output. */
	uint8_t sb[64];
	#pragma unroll 8
	for (int i = 0; i < 8; i++) {
		sb[8 * i + 0] = (uint8_t)(seed[i].x);
		sb[8 * i + 1] = (uint8_t)(seed[i].x >> 8);
		sb[8 * i + 2] = (uint8_t)(seed[i].x >> 16);
		sb[8 * i + 3] = (uint8_t)(seed[i].x >> 24);
		sb[8 * i + 4] = (uint8_t)(seed[i].y);
		sb[8 * i + 5] = (uint8_t)(seed[i].y >> 8);
		sb[8 * i + 6] = (uint8_t)(seed[i].y >> 16);
		sb[8 * i + 7] = (uint8_t)(seed[i].y >> 24);
	}

	flex_walk_nibbles(sb, d_core + thread, FLEX_CORE_ALGO_COUNT, stride);
	d_core[(size_t)FLEX_CORE_ALGO_COUNT * stride + thread] = 0;   /* slot 14 */
	flex_walk_nibbles(sb, d_cn + thread, FLEX_CN_ALGO_COUNT, stride);

	if (d_seed) {
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			d_seed[(size_t)thread * 8 + i] = seed[i];
	}
}

static void flex_seed_order_cpu(uint32_t threads, uint32_t startNonce,
                                uint8_t *d_core, uint8_t *d_cn, uint2 *d_seed)
{
	const uint32_t tpb = 128;
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);
	flex_seed_order_gpu <<<grid, block>>> (threads, threads, startNonce, d_core, d_cn, d_seed, NULL);
}

/* Same, over the nonce filter's compacted nonce list. Deliberately the SAME
 * kernel: the walk and the SoA order layout are gated by
 * the test harness and a second copy for the filtered path
 * would be a second thing to keep correct. */
static void flex_seed_order_nonces_cpu(uint32_t threads, uint32_t stride,
                                       const uint32_t *d_nonces,
                                       uint8_t *d_core, uint8_t *d_cn, uint2 *d_seed)
{
	const uint32_t tpb = 128;
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);
	flex_seed_order_gpu <<<grid, block>>> (threads, stride, 0, d_core, d_cn, d_seed, d_nonces);
}

/* ---------------------------------------------------------------------------
 * Gather / scatter of the 64-byte chain state.
 *
 * Each round, lanes are compacted by the algo they need so the existing shared
 * stage kernels can be called EXACTLY as x16r calls them -- base pointer at
 * offset 0, `count` threads, no index vector, no assumptions about how a given
 * stage indexes its private per-thread scratch.
 *
 * Both move 64 bytes, not 32. In flex the high half of a CN result is the
 * round's own input (the blake branch writes only the low 32) and the final
 * SHA3-256 hashes all 64, so dropping the high half changes the digest of
 * roughly half of all nonces.
 * ------------------------------------------------------------------------- */
static __global__
void flex_gather64_gpu(uint32_t n, uint4 * __restrict__ dst,
                       const uint4 * __restrict__ src, const uint32_t * __restrict__ idx)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= n)
		return;
	const uint4 *s = src + (size_t)idx[t] * 4;
	uint4 *d = dst + (size_t)t * 4;
	d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
}

static __global__
void flex_scatter64_gpu(uint32_t n, uint4 * __restrict__ dst,
                        const uint4 * __restrict__ src, const uint32_t * __restrict__ idx)
{
	const uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= n)
		return;
	const uint4 *s = src + (size_t)t * 4;
	uint4 *d = dst + (size_t)idx[t] * 4;
	d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
}

static void flex_gather64(uint32_t n, void *dst, const void *src, const uint32_t *d_idx)
{
	const uint32_t tpb = 128;
	flex_gather64_gpu <<<dim3((n + tpb - 1) / tpb), dim3(tpb)>>>
		(n, (uint4*)dst, (const uint4*)src, d_idx);
}

static void flex_scatter64(uint32_t n, void *dst, const void *src, const uint32_t *d_idx)
{
	const uint32_t tpb = 128;
	flex_scatter64_gpu <<<dim3((n + tpb - 1) / tpb), dim3(tpb)>>>
		(n, (uint4*)dst, (const uint4*)src, d_idx);
}

#endif /* __CUDACC__ */

#endif /* CUDA_FLEX_ORDER_CUH */
