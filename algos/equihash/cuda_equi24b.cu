// ---------------------------------------------------------------------------
// equi24b -- Equihash solver for the DIGITBITS=24 variants (144/5, 192/7) on a
// heap of few large buckets. See equi24b_params.h for the geometry and why.
//
// Stages: digitH builds layer 0, digitR<R> collides each subsequent layer,
// digitK tests the truncated last one for candidates, expandSols walks the
// retained layers back to leaf indices and checks every hash bit.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "blake2/blake2.h"
#include "equi24b_params.h"

// blake2b_tromp.cuh uses these unqualified, so they must precede it
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;
typedef unsigned char uchar;


#define EQ24B_CAT2(a, b) a##b
#define EQ24B_CAT(a, b)  EQ24B_CAT2(a, b)
#define EQ24B_NS         EQ24B_CAT(equi24b_, EQ_WN)
#define EQ24B_API(name)  EQ24B_CAT(EQ24B_CAT(equi24b_, EQ_WN), EQ24B_CAT(_, name))

#define EQ24B_BYTE(w, j) ((u32)((w)[(j) >> 3] >> (((j) & 7) * 8)) & 0xffu)

#define EQ24B_HASHESPERBLAKE (512 / EQ_WN)
#define EQ24B_NHASHES        (2u * (1u << EQ24B_DIGITBITS))
#define EQ24B_NBLAKES        ((EQ24B_NHASHES + EQ24B_HASHESPERBLAKE - 1) / EQ24B_HASHESPERBLAKE)
#define EQ24B_L0_STRIDE_W    (EQ24B_STRIDE(0) / 4u)

namespace EQ24B_NS {

// Included INSIDE the namespace, and necessarily BELOW the EQ24B_NS macro:
// this header defines __device__ functions at file scope with external
// linkage, so at global scope the 144/5 and 192/7 objects both emit them and
// the link fails. The u32/u64/uchar typedefs stay global -- the header finds
// them by outward lookup, and the extern "C" wrappers below need them too.
#include "blake2b_tromp.cuh"   // ROR2/G/ROUND + blake2b_gpu_hash_regs

// Warp-aggregated append: lanes hitting the same counter elect one leader that
// does a single atomicAdd for the group.
__device__ __forceinline__ u32 eq24b_append(u32 *ctr, u32 key, u32 lane)
{
#if __CUDA_ARCH__ >= 700
	// MUST be the active mask, not 0xffffffff: both call sites sit after a
	// `continue`, so naming absent lanes is undefined behaviour (it can hang).
	const u32 act = __activemask();
	const u32 peers = __match_any_sync(act, key);
	const u32 leader = __ffs(peers) - 1u;
	const u32 rank = __popc(peers & ((1u << lane) - 1u));
	u32 base = 0u;
	if (lane == leader) base = atomicAdd(ctr, (u32)__popc(peers));
	base = __shfl_sync(peers, base, leader);
	return base + rank;
#else
	// Pascal has no __match_any_sync: ballot repeatedly over the distinct keys
	// the warp holds. sm_61 is a shipped target, so this path is not optional.
	u32 active = __ballot(1);
	u32 peers = 0u, base = 0u;
	while (active) {
		const u32 ldr = (u32)__ffs(active) - 1u;
		const u32 k = __shfl(key, ldr);
		const u32 same = __ballot(k == key);
		if (k == key) peers = same;
		active &= ~same;
	}
	const u32 leader = (u32)__ffs(peers) - 1u;
	const u32 rank = __popc(peers & ((1u << lane) - 1u));
	if (lane == leader) base = atomicAdd(ctr, (u32)__popc(peers));
	base = __shfl(base, leader);
	return base + rank;
#endif
}

// Slot in a random destination bucket: a warp's keys are nearly all distinct,
// so on sm_6x (ballot loop above) a plain atomic is cheaper.
#ifndef EQ24B_PLAIN_SCATTER
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define EQ24B_PLAIN_SCATTER 1
#else
#define EQ24B_PLAIN_SCATTER 0
#endif
#endif
__device__ __forceinline__ u32 eq24b_scatter_slot(u32 *ctr, u32 key, u32 lane)
{
#if EQ24B_PLAIN_SCATTER
	(void)key; (void)lane;
	return atomicAdd(ctr, 1u);
#else
	return eq24b_append(ctr, key, lane);
#endif
}

// ---------------------------------------------------------------------------
// digitH -- blake2b over the nonce space, bucketed by the top EQ24B_BUCKBITS
// bits of digit 0. One independent layer array: no stagger, no partitions.
// ---------------------------------------------------------------------------
__device__ u64 eq24b_mid_h[8];   // blake2b midstate of the current header

__global__ void digitH(const u64 *__restrict__ blake_h,
                       u64 m0, u64 m1lo, u64 counter,
                       u32 *__restrict__ layer0,
                       u32 *__restrict__ counts,
                       u32 *__restrict__ overflow)
{
	const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 stride = gridDim.x * blockDim.x;

	u64 h[8], hh[7];
#pragma unroll
	for (int i = 0; i < 8; i++) h[i] = blake_h[i];

	for (u32 blk = id; blk < EQ24B_NBLAKES; blk += stride) {
		blake2b_gpu_hash_regs(h, m0, m1lo | ((u64)blk << 32), counter, hh);

#pragma unroll
		for (u32 i = 0; i < EQ24B_HASHESPERBLAKE; i++) {
			const int b = (EQ_WN / 8) * (int)i;
			const u32 index = blk * EQ24B_HASHESPERBLAKE + i;
			// NBLAKES rounds up, so the last blake emits hashes past the end of
			// the nonce space when HASHESPERBLAKE does not divide NHASHES. Those
			// indices do not fit the solution encoding; drop them here rather
			// than let a proof our own verifier accepts be rejected by a pool.
			if (index >= EQ24B_NHASHES) continue;

			const u32 d0 = (EQ24B_BYTE(hh, b) << 16) | (EQ24B_BYTE(hh, b + 1) << 8)
			             | EQ24B_BYTE(hh, b + 2);
			const u32 bucket = (d0 >> EQ24B_RESTBITS) & EQ24B_BUCKMASK;

			const u32 slot = eq24b_scatter_slot(&counts[bucket], bucket, threadIdx.x & 31);
			if (slot >= EQ24B_NSLOTS) {          // counted, never silent
				atomicAdd(overflow, 1u);
				continue;
			}

			// Layer-0 record (equi24b_params.h), written as whole uint4 stores: a
			// scattered store costs a sector transaction whatever its width.
			u32 *rec = layer0 + EQ24B_RIDX(0u, bucket, slot) * EQ24B_L0_STRIDE_W;
			u32 w[8];
#pragma unroll
			for (u32 k = 0; k < 8u; k++) w[k] = 0u;
			w[0] = index;
			// Payload: the bytes after the one holding the bucket index.
			// Bytes past the end of THIS hash must be zeroed, not taken -- the
			// blake output holds HASHESPERBLAKE hashes back to back, so reading
			// past byte EQ_WN/8 silently pulls in the neighbour's bits. The
			// payload is 17 B for 144/5, so the last word is 1 valid byte + 3
			// that belong to the next hash.
#pragma unroll
			for (u32 k = 0; k < 6u; k++) {
				if (k * 4u >= EQ24B_PAYLOAD(0)) break;
				u32 v = 0u;
#pragma unroll
				for (u32 j = 0; j < 4u; j++) {
					const u32 bi = 1u + k * 4u + j;          // byte within this hash
					if (bi < (u32)(EQ_WN / 8) && bi <= EQ24B_PAYLOAD(0))
						v |= (u32)EQ24B_BYTE(hh, b + (int)bi) << (8u * j);
				}
				w[2 + k] = v;
			}
			if (EQ24B_L0C) {   // 16 B record + side word (equi24b_params.h)
				const u32 b17 = w[6] & 0xffu;
				*(uint4 *)rec = make_uint4((w[2] & ~0xf0u) | ((b17 & 0xfu) << 4), w[3], w[4], w[5]);
				layer0[EQ24B_L0C_SIDE_OFF_W + EQ24B_RIDX(0u, bucket, slot)] = index | ((b17 >> 4) << 25);
			} else {
				*(uint4 *)(rec + 0) = make_uint4(w[0], w[1], w[2], w[3]);
				*(uint4 *)(rec + 4) = make_uint4(w[4], w[5], w[6], w[7]);
			}
		}
	}
}


// Record layout of layer L (see EQ24B_PACKED): stride in words and whether the
// ref is split around the payload. Device side, for expandSols.
struct eq24b_layer { u32 stride_w; bool packed, logged, trunc; };

__device__ __forceinline__ eq24b_layer eq24b_layer_info(int layer)
{
#define EQ24B_LI(L) case L: return { EQ24B_STRIDE(L##u) / 4u, EQ24B_PACKED(L##u), EQ24B_LOGGED(L##u), EQ24B_TRUNC(L##u) };
	switch (layer) {
	EQ24B_LI(0) EQ24B_LI(1) EQ24B_LI(2) EQ24B_LI(3) EQ24B_LI(4)
#if EQ_WK > 5
	EQ24B_LI(5) EQ24B_LI(6)
#endif
	default: return { 0u, false, false, false };
	}
#undef EQ24B_LI
}

// Payload word w of a layer-L record. Packed and logged layers keep non-payload
// bits in their last payload word; plain layers are written zero-padded. (A
// logged layer's word-0 pointer nibble is never read as payload: readers use
// bytes 0-1 as the rest and stage over them.)
template <u32 L>
__device__ __forceinline__ u32 eq24b_pay(const u32 *rec, u32 w)
{
	const u32 v = rec[EQ24B_POFF_W(L) + w];
	return ((EQ24B_PACKED(L) || EQ24B_LOGGED(L)) && w == (EQ24B_PAYLOAD(L) + 3u) / 4u - 1u)
		? (v & EQ24B_PMASK(L)) : v;
}

// Logged layer: the 26-bit log pointer kept in a record (layout: equi24b_params.h).
__device__ __forceinline__ u32 eq24b_get_lptr(const u32 *rec)
{
	return (rec[3] >> 16) | (((rec[3] >> 8) & 0x3fu) << 16) | (((rec[0] >> 4) & 0xfu) << 22);
}

// Store the ref into the assembled words of a layer-L record (payload already in).
template <u32 L>
__device__ __forceinline__ void eq24b_put_ref(u32 *o, eq_u64 ref)
{
	o[0] = (u32)ref;
	if (EQ24B_PACKED(L)) o[3] |= (u32)(ref >> 32) << 24;   // bits 32-37, byte 15
	else                 o[1] = (u32)(ref >> 32);
}

__device__ __forceinline__ eq_u64 eq24b_get_ref(const u32 *rec, bool packed)
{
	return packed ? ((eq_u64)(rec[3] >> 24) << 32) | rec[0]
	              : ((eq_u64)rec[1] << 32) | rec[0];
}

// Truncated last layer (layout: equi24b_params.h, EQ24B_TRUNCL).
__device__ __forceinline__ eq_u64 eq24b_get_tref(const u32 *rec)
{
	return ((eq_u64)(((rec[0] >> 4) & 0xfu) | ((rec[0] >> 30) << 4)) << 32) | rec[1];
}

// ---------------------------------------------------------------------------
// digitR<R> -- one collision round.
//
// Layout convention, fixed by digitH and preserved every round: a payload's
// first two bytes carry the 12 REST bits of the digit just consumed, in the low
// nibble of byte 0 and all of byte 1. Two items in the same bucket collide iff
// those match, so XORing a colliding pair zeroes payload bytes 0 and 1; the next
// digit is then payload bytes 2,3,4, its bucket is the top 12 of those, and the
// next payload starts at byte 3. PAYLOAD(r) = PAYLOAD(r-1) - 3 falls out of it.
//
// A bucket is processed in EQ24B_CHUNKS_R(R) chunks split by rest PREFIX, which
// is lossless because items whose rest differs in the high bits cannot collide.
// Each chunk is its own block and a bucket's chunks have adjacent block ids, so
// they run together: every one streams the whole bucket, which DRAM delivers
// about once and L2 serves to the others.
// Within a chunk, collisions are found by a per-rest singly-linked list built
// with atomicExch -- each item stores the previous head, so walking from an
// item's link visits exactly the items staged before it in its group and every
// unordered pair is enumerated once.
// ---------------------------------------------------------------------------
template <u32 R>
__global__ void digitR(const u32 *__restrict__ heapIn, u32 *__restrict__ heapOut,
                       const u32 *__restrict__ cntIn, u32 *__restrict__ cntOut,
                       u32 *__restrict__ overflow)
{
	const u32 CH   = EQ24B_CHUNKS_R(R);
	const u32 bucket = blockIdx.x / CH, chunk = blockIdx.x % CH;
	const u32 tid = threadIdx.x, lane = tid & 31u;
	const u32 nin = min(cntIn[bucket], EQ24B_NSLOTS);

	const u32 PW   = EQ24B_STAGE_W(R);          // payload words staged
	const u32 PH   = EQ24B_STAGE_H(R);          // plus a u16 tail
	const u32 SB   = EQ24B_SBYTES(R);           // payload bytes staged
	const u32 SLOTS = EQ24B_STAGE_SLOTS(R);
	const u32 HEADS = EQ24B_HEADS(R);
	const u32 PIN  = EQ24B_POFF_W(R - 1u);      // payload word offset, input / output
	const u32 POUT = EQ24B_POFF_W(R);
	const u32 INW  = EQ24B_STRIDE(R - 1u) / 4u;  // input record stride, words
	const u32 OUTW = EQ24B_STRIDE(R) / 4u;

	extern __shared__ u32 sh[];
	u32 *s_pay  = sh;                              // [SLOTS][PW], word 0 low 16 bits = origin slot
	u16 *s_next = (u16 *)(sh + SLOTS * PW);        // [SLOTS]
	u16 *s_tail = s_next + SLOTS;                  // [SLOTS] if PH
	u32 *s_head = sh + EQ24B_HEAD_OFF_W(R);        // [HEADS], u32: no 16-bit atomicExch
	__shared__ u32 s_n, s_lg;
	u32 *const lg = heapOut + EQ24B_LOG_OFF_W + (size_t)blockIdx.x * EQ24B_LOGCAP;   // LOGGED(R) only

	for (u32 i = tid; i < HEADS; i += blockDim.x) s_head[i] = EQ24B_LIST_END;
	if (tid == 0) { s_n = 0u; s_lg = 0u; }
	__syncthreads();

	// ---- pass A: stage the items whose rest prefix selects this chunk ----
	if (INW == 4u) {
		// 16 B records: one load per record and two records in flight per
		// thread, so the filter never waits on a dependent load. More records
		// raise registers past two blocks per SM on sm_6x.
		const u32 U = 2u;
		for (u32 base = tid; base < nin; base += blockDim.x * U) {
			uint4 rv[U];
#pragma unroll
			for (u32 u = 0; u < U; u++) {
				const u32 slot = base + u * blockDim.x;
				if (slot < nin) rv[u] = __ldg((const uint4 *)(heapIn + EQ24B_RIDX(R - 1u, bucket, slot) * INW));
			}
#pragma unroll
			for (u32 u = 0; u < U; u++) {
				const u32 slot = base + u * blockDim.x;
				if (slot >= nin) continue;
				const u32 rw[4] = { rv[u].x, rv[u].y, rv[u].z, rv[u].w };
				const u32 p0 = rw[PIN];
				const u32 rest = ((p0 & 0x0fu) << 8) | ((p0 >> 8) & 0xffu);
				if (rest / HEADS != chunk) continue;
				const u32 s = eq24b_append(&s_n, chunk, lane);
				if (s >= SLOTS) { if (lane == 0) atomicAdd(overflow + 1, 1u); continue; }
				s_pay[s * PW] = (p0 & 0xffff0000u) | slot;
#pragma unroll
				for (u32 w = 1; w < PW; w++) s_pay[s * PW + w] = eq24b_pay<R - 1u>(rw, w);
				if (PH) s_tail[s] = (u16)eq24b_pay<R - 1u>(rw, PW);
				s_next[s] = (u16)atomicExch(&s_head[rest % HEADS], s);
			}
		}
	} else
	for (u32 slot = tid; slot < nin; slot += blockDim.x) {
		const u32 *rec = heapIn + EQ24B_RIDX(R - 1u, bucket, slot) * INW;
		const u32 p0 = rec[PIN];                     // payload word 0
		const u32 rest = ((p0 & 0x0fu) << 8) | ((p0 >> 8) & 0xffu);
		if (rest / HEADS != chunk) continue;

		const u32 s = eq24b_append(&s_n, chunk, lane);
		if (s >= SLOTS) { if (lane == 0) atomicAdd(overflow + 1, 1u); continue; }

		// payload bytes 0-1 hold the rest (equal within a list, never emitted):
		// they carry the origin slot instead
		s_pay[s * PW] = (p0 & 0xffff0000u) | slot;
		for (u32 w = 1; w < PW; w++) s_pay[s * PW + w] = eq24b_pay<R - 1u>(rec, w);
		if (PH) s_tail[s] = (u16)eq24b_pay<R - 1u>(rec, PW);
		s_next[s] = (u16)atomicExch(&s_head[rest % HEADS], s);
	}
	__syncthreads();

	// ---- pass B: walk each item's list and emit its pairs ----------------
	const u32 nst = min(s_n, SLOTS);
	for (u32 s = tid; s < nst; s += blockDim.x) {
		for (u32 t = s_next[s]; t != EQ24B_LIST_END; t = s_next[t]) {
			u32 x[8];
			for (u32 w = 0; w < PW; w++) x[w] = s_pay[s * PW + w] ^ s_pay[t * PW + w];
			if (PH) x[PW] = (u32)(s_tail[s] ^ s_tail[t]);
			x[0] &= 0xffff0000u;                     // slots, not payload
			// A pair whose payload cancels ENTIRELY agrees on every remaining bit,
			// which only happens when the two subtrees share leaves. Dropping it
			// is not an optimisation: such a child has rest 0, so every one of
			// them lands in bucket 0 and pairs with the others, growing
			// quadratically per round until it swamps the final round.
			u32 nz = 0u;
			for (u32 w = 0; w < PW + PH; w++) nz |= x[w];
			if (!nz) continue;
			// bytes 0,1 of the xor are zero by construction (same rest)
			const u32 b2 = (x[0] >> 16) & 0xffu, b3 = (x[0] >> 24) & 0xffu;
			const u32 dst = ((b2 << 4) | (b3 >> 4)) & EQ24B_BUCKMASK;

			// logged layer: claim the log entry first, so a full region drops
			// the item before it holds a record slot
			u32 lk = 0u;
			if (EQ24B_LOGGED(R)) {
				lk = eq24b_append(&s_lg, 0u, lane);
				if (lk >= EQ24B_LOGCAP) { atomicAdd(overflow + 1, 1u); continue; }
			}
			const u32 os = eq24b_scatter_slot(&cntOut[dst], dst, lane);
			if (os >= EQ24B_NSLOTS) { atomicAdd(overflow, 1u); continue; }

			u32 *out = heapOut + EQ24B_RIDX(R, dst, os) * OUTW;
			// reference: source bucket + Cantor of the two ORIGIN slots
			const u32 os_ = s_pay[s * PW] & 0xffffu, ot_ = s_pay[t * PW] & 0xffffu;
			const u32 a = max(os_, ot_), bb = min(os_, ot_);
			const u32 cantor = (a * (a - 1u)) / 2u + bb;
			const eq_u64 ref = ((eq_u64)bucket << EQ24B_CANTORBITS) | cantor;
			u32 o[8];
			for (u32 w = 0; w < 8u; w++) o[w] = 0u;
			// next payload starts at byte 3 of the xor
			for (u32 w = 0; w * 4u < EQ24B_PAYLOAD(R) && w < 6u; w++) {
				u32 v = 0u;
				for (u32 j = 0; j < 4u; j++) {
					const u32 bi = 3u + w * 4u + j;
					// bytes past PAYLOAD(R) and past the staged bytes must be
					// ZERO, not whatever the xor left there: a later zero test
					// would see them.
					if (w * 4u + j < EQ24B_PAYLOAD(R) && bi < SB)
						v |= ((x[bi >> 2] >> ((bi & 3u) * 8u)) & 0xffu) << (8u * j);
				}
				o[POUT + w] = v;
			}
			if (EQ24B_LOGGED(R)) {
				const u32 ptr = blockIdx.x * EQ24B_LOGCAP + lk;
				lg[lk] = cantor | (((o[3] >> 8) & 0x3fu) << 26);
				o[0] = (o[0] & ~0xf0u) | (((ptr >> 22) & 0xfu) << 4);
				o[3] = (o[3] & 0xc0ffu) | (((ptr >> 16) & 0x3fu) << 8) | (ptr << 16);
			} else if (EQ24B_TRUNC(R)) {
				o[0] = (o[0] & EQ24B_TRUNC_MASK) | (((u32)(ref >> 32) & 0xfu) << 4)
				     | (((u32)(ref >> 36) & 3u) << 30);
				o[1] = (u32)ref;
			} else {
				eq24b_put_ref<R>(o, ref);
			}
			if (OUTW == 2u) {
				*(uint2 *)out = make_uint2(o[0], o[1]);
			} else {
				*(uint4 *)(out + 0) = make_uint4(o[0], o[1], o[2], o[3]);
				if (OUTW > 4u) *(uint4 *)(out + 4) = make_uint4(o[4], o[5], o[6], o[7]);
			}
		}
	}
}


// ---------------------------------------------------------------------------
// digitK -- the final round in ONE pass: one block per bucket, staging index
// == slot. Shared: u32 heads per rest, per slot a u16 [link 14 | key bits 8-9]
// and a u8 [key bits 0-7], key = the first 10 kept final-digit bits. A key
// match re-reads both records and tests all kept bits before it becomes a
// candidate.
// ---------------------------------------------------------------------------
#define EQ24B_K_END    0x3fffu
#define EQ24B_K_SHARED (EQ24B_NRESTS * 4u + EQ24B_NSLOTS * 3u)
__global__ void digitK(const u32 *__restrict__ heapIn,
                        const u32 *__restrict__ cntIn,
                        u32 *__restrict__ sols, u32 *__restrict__ nsols, u32 maxsols)
{
	const u32 bucket = blockIdx.x, tid = threadIdx.x;
	const u32 nin = min(cntIn[bucket], EQ24B_NSLOTS);
	const u32 INW = EQ24B_STRIDE(EQ24B_TRUNCL) / 4u;
	const u32 KEEP = EQ24B_TRUNC_MASK & 0xffff0000u;

	extern __shared__ u32 sh[];
	u32   *s_head = sh;                                  // [NRESTS]
	u16   *s_link = (u16 *)(sh + EQ24B_NRESTS);          // [NSLOTS]
	uchar *s_key  = (uchar *)(s_link + EQ24B_NSLOTS);    // [NSLOTS]

	for (u32 i = tid; i < EQ24B_NRESTS; i += blockDim.x) s_head[i] = EQ24B_K_END;
	__syncthreads();

	for (u32 slot = tid; slot < nin; slot += blockDim.x) {
		const u32 p0 = heapIn[EQ24B_RIDX(EQ24B_TRUNCL, bucket, slot) * INW];
		const u32 rest = ((p0 & 0x0fu) << 8) | ((p0 >> 8) & 0xffu);
		const u32 key = (p0 >> 16) & 0x3ffu;
		s_key[slot] = (uchar)key;
		const u32 prev = atomicExch(&s_head[rest], slot);
		s_link[slot] = (u16)((prev & EQ24B_K_END) | ((key >> 8) << 14));
	}
	__syncthreads();

	for (u32 sl = tid; sl < nin; sl += blockDim.x) {
		const u32 ls = s_link[sl];
		const u32 ks = s_key[sl] | ((ls >> 14) << 8);
		for (u32 t = ls & EQ24B_K_END; t != EQ24B_K_END; ) {
			const u32 lt = s_link[t];
			if ((s_key[t] | ((lt >> 14) << 8)) == ks) {
				const u32 a = heapIn[EQ24B_RIDX(EQ24B_TRUNCL, bucket, sl) * INW];
				const u32 b = heapIn[EQ24B_RIDX(EQ24B_TRUNCL, bucket, t) * INW];
				if (!((a ^ b) & KEEP)) {
					const u32 at = atomicAdd(nsols, 1u);
					if (at < maxsols) {
						sols[at * 3 + 0] = bucket;
						sols[at * 3 + 1] = sl;
						sols[at * 3 + 2] = t;
					}
				}
			}
			t = lt & EQ24B_K_END;
		}
	}
}


// ---------------------------------------------------------------------------
// expandSols -- walk the reference tree to the blake indices.
//
// The arena retains every layer, so a node can be read in full; only the
// reference is needed (layouts: equi24b_params.h). A candidate passes only if
// the full hashes of its leaves, kept in layer 0, cancel; passing proofs are
// appended to out via *nout. Ordering is consensus-relevant:
// each pair of sub-lists must lead with the smaller index, done bottom-up after
// a raw expansion.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void eq24b_expand(u32 t, u32 *const *__restrict__ layers,
                                             const u32 *__restrict__ sols, u32 nsols,
                                             u32 *__restrict__ out, u32 *__restrict__ nout)
{

	u32 idx[EQ24B_PROOFSIZE], bkt[EQ24B_PROOFSIZE], slt[EQ24B_PROOFSIZE];
	bkt[0] = sols[t * 3 + 0]; slt[0] = sols[t * 3 + 1];
	bkt[1] = sols[t * 3 + 0]; slt[1] = sols[t * 3 + 2];
	u32 n = 2, lbits = 0u;

	for (int layer = (int)EQ_WK - 1; layer >= 1; layer--) {
		const u32 *L = layers[layer];
		const eq24b_layer li = eq24b_layer_info(layer);
		for (int i = (int)n - 1; i >= 0; i--) {
			const u32 *rec = L + EQ24B_RIDX(0u, bkt[i], slt[i]) * li.stride_w;
			u32 src;
			eq_u64 c;
			if (li.logged) {
				const u32 ptr = eq24b_get_lptr(rec);
				const u32 lw = L[EQ24B_LOG_OFF_W + ptr];
				src = ptr / (EQ24B_LOGCAP * EQ24B_CHUNKS_R(EQ24B_LOGL));
				c = lw & EQ24B_CANTORMASK;
				lbits ^= lw >> EQ24B_CANTORBITS;
			} else {
				const eq_u64 ref = li.trunc ? eq24b_get_tref(rec) : eq24b_get_ref(rec, li.packed);
				src = (u32)(ref >> EQ24B_CANTORBITS);
				c = ref & EQ24B_CANTORMASK;
			}
			u32 a = (u32)((1.0 + sqrt(1.0 + 8.0 * (double)c)) * 0.5);
			while (a > 1u && (eq_u64)a * (a - 1u) / 2u > c) a--;
			while ((eq_u64)(a + 1u) * a / 2u <= c) a++;
			const u32 b0 = (u32)(c - (eq_u64)a * (a - 1u) / 2u);
			bkt[2*i] = src; slt[2*i] = b0;
			bkt[2*i+1] = src; slt[2*i+1] = a;
		}
		n *= 2;
	}
	if (lbits) return;   // the logged layer's moved bits must cancel too

	// every hash bit: the bucket (top 12) and the payload (all bytes after the
	// first) of each leaf, as layer 0 holds them
	const u32 *L0 = layers[0];
	const u32 SW0 = eq24b_layer_info(0).stride_w;
	const u32 NW0 = EQ24B_L0C ? 4u : (EQ24B_PAYLOAD(0) + 3u) / 4u;
	u32 acc[6] = { 0u, 0u, 0u, 0u, 0u, 0u }, bx = 0u;
	for (u32 i = 0; i < n; i++) {
		const size_t ri = EQ24B_RIDX(0u, bkt[i], slt[i]);
		const u32 *rec = L0 + ri * SW0;
		if (EQ24B_L0C) {   // every stored bit is a hash bit (equi24b_params.h)
			const u32 sw = L0[EQ24B_L0C_SIDE_OFF_W + ri];
			idx[i] = sw & 0x1ffffffu;
			bx ^= (sw >> 25) << 12;
		} else {
			idx[i] = rec[0];
		}
		bx ^= bkt[i];
		for (u32 w = 0; w < NW0; w++) acc[w] ^= rec[EQ24B_POFF_W(0) + w];
	}
	for (u32 w = 0; w < NW0; w++) bx |= acc[w];
	if (bx) return;
	// a proof repeating a leaf satisfies every XOR condition and is never a
	// share: drop it here rather than copy and sort it on the host
	for (u32 i = 1; i < n; i++)
		for (u32 j = 0; j < i; j++)
			if (idx[i] == idx[j]) return;

	for (u32 blk = 2; blk <= EQ24B_PROOFSIZE; blk <<= 1)
		for (u32 base = 0; base < EQ24B_PROOFSIZE; base += blk)
			if (idx[base] > idx[base + blk / 2])
				for (u32 k = 0; k < blk / 2; k++) {
					const u32 tmp = idx[base + k];
					idx[base + k] = idx[base + blk / 2 + k];
					idx[base + blk / 2 + k] = tmp;
				}
	const u32 at = atomicAdd(nout, 1u);
	if (at >= nsols) return;
	for (u32 i = 0; i < EQ24B_PROOFSIZE; i++) out[at * EQ24B_PROOFSIZE + i] = idx[i];
}

// The candidate count is read on the device, so the launch needs no host copy.
__global__ void expandSols(u32 *const *__restrict__ layers,
                           const u32 *__restrict__ sols, const u32 *__restrict__ found, u32 maxsol,
                           u32 *__restrict__ out, u32 *__restrict__ nout)
{
	const u32 n = min(*found, maxsol);
	for (u32 t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x)
		eq24b_expand(t, layers, sols, n, out, nout);
}

}  // namespace EQ24B_NS

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
extern "C" unsigned long long EQ24B_API(layer_bytes)(unsigned r)
{
	switch (r) {
	case 0: return (unsigned long long)EQ24B_LAYER_BYTES(0);
	case 1: return (unsigned long long)EQ24B_LAYER_BYTES(1);
	case 2: return (unsigned long long)EQ24B_LAYER_BYTES(2);
	case 3: return (unsigned long long)EQ24B_LAYER_BYTES(3);
	case 4: return (unsigned long long)EQ24B_LAYER_BYTES(4);
#if EQ_WK > 5
	case 5: return (unsigned long long)EQ24B_LAYER_BYTES(5);
	case 6: return (unsigned long long)EQ24B_LAYER_BYTES(6);
#endif
	default: return 0;
	}
}

extern "C" unsigned EQ24B_API(nbuckets)(void) { return EQ24B_NBUCKETS; }
extern "C" unsigned EQ24B_API(nslots)(void)   { return EQ24B_NSLOTS; }
extern "C" unsigned EQ24B_API(stride0)(void)  { return EQ24B_STRIDE(0); }

#include <stdlib.h>
#include <string.h>

// forward declarations: the solve path below calls these, and they are
// defined further down in this file.
extern "C" unsigned long long EQ24B_API(layer_bytes)(unsigned);
extern "C" unsigned EQ24B_API(shared)(unsigned);
extern "C" void EQ24B_API(free)(void *);
extern "C" int EQ24B_API(digitH)(const void *, unsigned *, unsigned *, unsigned *, int, int);
extern "C" int EQ24B_API(digitR)(int, const unsigned *, unsigned *, const unsigned *, unsigned *, unsigned *, int);

// ---------------------------------------------------------------------------
// solver context: every layer retained, which is what lets expandSols read a
// record instead of only a self-sufficient reference word.
// ---------------------------------------------------------------------------
namespace EQ24B_NS {
struct ctx {
	u32 *layer[EQ_WK];
	u32 *cnt[EQ_WK];
	u32 **d_layers;
	u32 *over, *sols, *nsols, *idx;
	u32 *hidx;
	u32 maxsol;
};
static void setheader(blake2b_state *c, const char *hdr, const char *personal)
{
	char p16[16];
	memcpy(p16, personal, 8);
	u32 n = EQ_WN, k = EQ_WK;
	memcpy(p16 + 8, &n, 4); memcpy(p16 + 12, &k, 4);
	blake2b_param P[1];
	memset(P, 0, sizeof(blake2b_param));
	P->digest_length = (512 / EQ_WN) * EQ_WN / 8;
	P->fanout = P->depth = 1;
	memcpy(P->personal, p16, 16);
	eq_blake2b_init_param(c, P);
	eq_blake2b_update(c, (const uint8_t *)hdr, 140);
}
}  // namespace EQ24B_NS

extern "C" void *EQ24B_API(init)(void)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)calloc(1, sizeof(EQ24B_NS::ctx));
	if (!c) return NULL;
	// Candidate capacity for digitK, NOT a solution count: digitK emits every
	// colliding pair (large rest groups make that many per nonce) and the host
	// filter removes the trivial ones. Too small a cap discards real solutions
	// in arrival order; this is ~10x the observed peak.
	c->maxsol = 262144;
	bool okk = true;
	u32 *hostptr[EQ_WK];
	for (unsigned r = 0; r < EQ_WK; r++) {
		const unsigned long long lb = EQ24B_API(layer_bytes)(r);
		if (!lb) { okk = false; break; }        // a layer with no size is a wiring bug
		okk &= cudaMalloc(&c->layer[r], (size_t)lb) == cudaSuccess;
		okk &= cudaMalloc(&c->cnt[r], (size_t)EQ24B_NBUCKETS * sizeof(u32)) == cudaSuccess;
		hostptr[r] = c->layer[r];
	}
	okk &= cudaMalloc(&c->d_layers, EQ_WK * sizeof(u32 *)) == cudaSuccess;
	okk &= cudaMalloc(&c->over, 4 * sizeof(u32)) == cudaSuccess;
	okk &= cudaMalloc(&c->sols, c->maxsol * 3 * sizeof(u32)) == cudaSuccess;
	okk &= cudaMalloc(&c->nsols, sizeof(u32)) == cudaSuccess;
	okk &= cudaMalloc(&c->idx, (size_t)c->maxsol * EQ24B_PROOFSIZE * sizeof(u32)) == cudaSuccess;
	c->hidx = (u32 *)malloc((size_t)c->maxsol * EQ24B_PROOFSIZE * sizeof(u32));
	if (!okk || !c->hidx) { EQ24B_API(free)(c); return NULL; }
	cudaMemcpy(c->d_layers, hostptr, EQ_WK * sizeof(u32 *), cudaMemcpyHostToDevice);
	return c;
}

extern "C" void EQ24B_API(free)(void *p)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)p;
	if (!c) return;
	for (unsigned r = 0; r < EQ_WK; r++) { if (c->layer[r]) cudaFree(c->layer[r]); if (c->cnt[r]) cudaFree(c->cnt[r]); }
	if (c->d_layers) cudaFree(c->d_layers);
	if (c->over) cudaFree(c->over);
	if (c->sols) cudaFree(c->sols);
	if (c->nsols) cudaFree(c->nsols);
	if (c->idx) cudaFree(c->idx);
	if (c->hidx) ::free(c->hidx);
	::free(c);
}

// Loss counters from the last solve, all silent without this accessor:
//   [0] digitR scatter overflow (destination bucket full)
//   [1] digitR staging overflow (chunk full)
//   [2] unused (digitK stages every slot)
//   [3] digitK candidates counted but not stored (maxsol clamp)
extern "C" void EQ24B_API(drops)(void *p, unsigned *out4)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)p;
	out4[0] = out4[1] = out4[2] = out4[3] = 0u;
	if (!c) return;
	cudaMemcpy(out4, c->over, 3 * sizeof(u32), cudaMemcpyDeviceToHost);
	u32 found = 0; cudaMemcpy(&found, c->nsols, sizeof(u32), cudaMemcpyDeviceToHost);
	out4[3] = found > c->maxsol ? found - c->maxsol : 0u;
}

// Threads per block for collision round r (r == EQ_WK is digitK). Round blocks
// are shared-bound at 2 per SM, so the block size is what sets the resident
// warps; sm_6x needs more of them than sm_8x. sm_7x takes the sm_6x values
// (untested): Turing's 64 KB of shared holds only one such block per SM.
static int eq24b_round_tpb(unsigned r)
{
	int dev = 0, major = 8;
	cudaGetDevice(&dev);
	cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
	(void)r;
	return major >= 8 ? 512 : 1024;
}

// Launch helpers: no counter reset and no sync, so a solve's kernels run back
// to back on the default stream (defined below). A pinned midstate buffer makes
// the digitH upload asynchronous.
static int eq24b_launch_digitH(const blake2b_state *st, u64 *pinned, unsigned *d_layer0,
                               unsigned *d_counts, unsigned *d_overflow, int blocks, int tpb);
static int eq24b_launch_digitR(int r, const unsigned *d_in, unsigned *d_out,
                               const unsigned *d_cntIn, unsigned *d_cntOut, unsigned *d_over, int tpb);

// digitH grid: sm_6x runs faster with fewer, larger blocks.
static void eq24b_dh_shape(int *blocks, int *tpb)
{
	int dev = 0, major = 8;
	cudaGetDevice(&dev);
	cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
	*blocks = major == 6 ? 1024 : 2048;
	*tpb    = major == 6 ? 512 : 128;
}

// Reset the counters and queue digitH .. digitK for one header. Every launch is
// checked; a kernel fault surfaces at the caller's first sync.
static int eq24b_queue(EQ24B_NS::ctx *c, const char *headernonce, const char *personal, u64 *pinned)
{
	blake2b_state mid;
	EQ24B_NS::setheader(&mid, headernonce, personal);
	cudaMemset(c->over, 0, 4 * sizeof(u32));
	cudaMemset(c->nsols, 0, sizeof(u32));
	for (unsigned r = 0; r < EQ_WK; r++)   // each counter array is written by one kernel
		cudaMemset(c->cnt[r], 0, (size_t)EQ24B_NBUCKETS * sizeof(u32));
	int hb, ht;
	eq24b_dh_shape(&hb, &ht);
	const int tpb = eq24b_round_tpb(1u);
	if (eq24b_launch_digitH(&mid, pinned, c->layer[0], c->cnt[0], c->over, hb, ht)) return -4;
	for (unsigned r = 1; r < EQ_WK; r++)
		if (eq24b_launch_digitR((int)r, c->layer[r-1], c->layer[r], c->cnt[r-1], c->cnt[r], c->over, tpb))
			return -4;
	EQ24B_NS::digitK<<<EQ24B_NBUCKETS, tpb, EQ24B_K_SHARED>>>(c->layer[EQ_WK-1], c->cnt[EQ_WK-1],
	                                               c->sols, c->nsols, c->maxsol);
	return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

// Trivial-solution filter, a backstop: expandSols already drops proofs that
// repeat a leaf. Emits the rest and returns their count.
static int eq24b_emit(const u32 *hidx, u32 nok, void (*emit)(void *, const uint32_t *, uint32_t), void *ud)
{
	int emitted = 0;
	for (u32 s = 0; s < nok; s++) {
		const u32 *q = hidx + (size_t)s * EQ24B_PROOFSIZE;
		u32 sorted[EQ24B_PROOFSIZE];
		memcpy(sorted, q, sizeof(sorted));
		for (u32 i = 1; i < EQ24B_PROOFSIZE; i++)
			for (u32 j = i; j && sorted[j-1] > sorted[j]; j--) {
				const u32 tv = sorted[j]; sorted[j] = sorted[j-1]; sorted[j-1] = tv;
			}
		bool dup = false;
		for (u32 i = 1; i < EQ24B_PROOFSIZE && !dup; i++) dup = sorted[i] == sorted[i-1];
		if (dup) continue;
		if (emit) emit(ud, q, EQ24B_PROOFSIZE);
		emitted++;
	}
	return emitted;
}

extern "C" int EQ24B_API(solve)(void *p, const char *headernonce, const char *personal,
                                void (*emit)(void *, const uint32_t *, uint32_t), void *ud)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)p;
	if (!c) return -1;
	if (eq24b_queue(c, headernonce, personal, NULL)) return -4;

	u32 found = 0;
	if (cudaMemcpy(&found, c->nsols, sizeof(u32), cudaMemcpyDeviceToHost) != cudaSuccess) return -3;
	const u32 store = found < c->maxsol ? found : c->maxsol;
	if (!store) return 0;

	// over[3] counts the candidates that pass expandSols' full check
	EQ24B_NS::expandSols<<<(store + 63) / 64, 64>>>(c->d_layers, c->sols, c->nsols, c->maxsol, c->idx, c->over + 3);
	if (cudaGetLastError() != cudaSuccess) return -4;
	if (cudaDeviceSynchronize() != cudaSuccess) return -3;
	u32 nok = 0;
	cudaMemcpy(&nok, c->over + 3, sizeof(u32), cudaMemcpyDeviceToHost);
	if (nok > store) nok = store;
	if (!nok) return 0;
	cudaMemcpy(c->hidx, c->idx, (size_t)nok * EQ24B_PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
	return eq24b_emit(c->hidx, nok, emit, ud);
}

// ---- pipelined solve ---------------------------------------------------------
// launch() queues a whole solve, expandSols and the copy back included, and
// returns at once; collect() waits for the oldest queued solve and emits its
// proofs. Two slots, so the next header can be queued before the last result
// is collected and the GPU does not idle while the host checks proofs.
#define EQ24B_PIPE_CAP 64u   // proofs copied back per solve; more count as clamped
namespace EQ24B_NS {
struct pipe {
	u32 *h[2];           // pinned: over[4], found, CAP proofs
	u64 *mid[2];         // pinned digitH midstate
	cudaEvent_t ev[2];
	unsigned nl, nc;     // solves launched / collected
};
}

extern "C" void EQ24B_API(pipe_free)(void *pq)
{
	EQ24B_NS::pipe *q = (EQ24B_NS::pipe *)pq;
	if (!q) return;
	for (int i = 0; i < 2; i++) {
		if (q->ev[i]) cudaEventDestroy(q->ev[i]);
		if (q->h[i]) cudaFreeHost(q->h[i]);
		if (q->mid[i]) cudaFreeHost(q->mid[i]);
	}
	::free(q);
}

extern "C" void *EQ24B_API(pipe_new)(void)
{
	EQ24B_NS::pipe *q = (EQ24B_NS::pipe *)calloc(1, sizeof(EQ24B_NS::pipe));
	if (!q) return NULL;
	for (int i = 0; i < 2; i++)
		if (cudaHostAlloc((void **)&q->h[i], (5u + EQ24B_PIPE_CAP * EQ24B_PROOFSIZE) * sizeof(u32), cudaHostAllocDefault) != cudaSuccess
		    || cudaHostAlloc((void **)&q->mid[i], 8 * sizeof(u64), cudaHostAllocDefault) != cudaSuccess
		    || cudaEventCreateWithFlags(&q->ev[i], cudaEventDisableTiming) != cudaSuccess) {
			EQ24B_API(pipe_free)(q);
			return NULL;
		}
	return q;
}

extern "C" int EQ24B_API(launch)(void *p, void *pq, const char *headernonce, const char *personal)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)p;
	EQ24B_NS::pipe *q = (EQ24B_NS::pipe *)pq;
	if (!c || !q) return -1;
	if (q->nl - q->nc >= 2u) return -5;   // both slots in flight
	const unsigned k = q->nl & 1u;
	if (eq24b_queue(c, headernonce, personal, q->mid[k])) return -4;
	EQ24B_NS::expandSols<<<64, 64>>>(c->d_layers, c->sols, c->nsols, c->maxsol, c->idx, c->over + 3);
	if (cudaGetLastError() != cudaSuccess) return -4;
	cudaMemcpyAsync(q->h[k], c->over, 4 * sizeof(u32), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(q->h[k] + 4, c->nsols, sizeof(u32), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(q->h[k] + 5, c->idx, (size_t)EQ24B_PIPE_CAP * EQ24B_PROOFSIZE * sizeof(u32),
	                cudaMemcpyDeviceToHost, 0);
	if (cudaEventRecord(q->ev[k], 0) != cudaSuccess) return -4;
	q->nl++;
	return 0;
}

// drops4 (optional) receives that solve's loss counters, as drops() describes.
extern "C" int EQ24B_API(collect)(void *p, void *pq, void (*emit)(void *, const uint32_t *, uint32_t),
                                  void *ud, unsigned *drops4)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)p;
	EQ24B_NS::pipe *q = (EQ24B_NS::pipe *)pq;
	if (!c || !q) return -1;
	if (q->nl == q->nc) return -5;        // nothing queued
	const unsigned k = q->nc & 1u;
	q->nc++;
	if (cudaEventSynchronize(q->ev[k]) != cudaSuccess) return -3;
	const u32 *h = q->h[k];
	const u32 found = h[4];
	const u32 store = found < c->maxsol ? found : c->maxsol;
	u32 nok = h[3] < store ? h[3] : store;
	const u32 over = nok > EQ24B_PIPE_CAP ? nok - EQ24B_PIPE_CAP : 0u;
	if (drops4) {
		drops4[0] = h[0]; drops4[1] = h[1]; drops4[2] = h[2];
		drops4[3] = (found - store) + over;
	}
	if (over) nok = EQ24B_PIPE_CAP;
	return eq24b_emit(h + 5, nok, emit, ud);
}

// Forget queued solves (the caller's header changed): waits for them, emits nothing.
extern "C" void EQ24B_API(pipe_reset)(void *pq)
{
	EQ24B_NS::pipe *q = (EQ24B_NS::pipe *)pq;
	if (!q) return;
	while (q->nc != q->nl) cudaEventSynchronize(q->ev[q->nc++ & 1u]);
}

extern "C" unsigned EQ24B_API(chunks)(unsigned r)
{
	if (r == EQ_WK) return 1u;   // digitK: one block per bucket
	switch (r) { case 1: return EQ24B_CHUNKS_R(1u); case 2: return EQ24B_CHUNKS_R(2u);
	             case 3: return EQ24B_CHUNKS_R(3u); case 4: return EQ24B_CHUNKS_R(4u);
#if EQ_WK > 5
	             case 5: return EQ24B_CHUNKS_R(5u); case 6: return EQ24B_CHUNKS_R(6u);
#endif
	             default: return 0; }
}

// shared bytes a round needs (layout: equi24b_params.h, EQ24B_HEAD_OFF_W)
static unsigned eq24b_shared(unsigned r)
{
	if (r == EQ_WK) return EQ24B_K_SHARED;
	switch (r) {
#define SH(R) case R: return EQ24B_SHARED_BYTES(R##u);
	SH(1) SH(2) SH(3) SH(4)
#if EQ_WK > 5
	SH(5) SH(6)
#endif
#undef SH
	default: return 0;
	}
}

extern "C" unsigned EQ24B_API(shared)(unsigned r) { return eq24b_shared(r); }

// Device bytes a context will try to allocate. Every layer is RETAINED, which
// is what lets expandSols read a record rather than only a reference word, so
// this grows with the round count and differs between the variants.
extern "C" unsigned long long EQ24B_API(arena_needed)(void)
{
	unsigned long long t = 0;
	for (unsigned r = 0; r < EQ_WK; r++)
		t += EQ24B_API(layer_bytes)(r) + (unsigned long long)EQ24B_NBUCKETS * sizeof(unsigned);
	// candidate buffers: pair records + expanded proofs + the host mirror's twin
	t += (unsigned long long)262144u * 3u * sizeof(unsigned);
	t += (unsigned long long)262144u * EQ24B_PROOFSIZE * sizeof(unsigned);
	return t;
}

static int eq24b_launch_digitR(int r, const unsigned *d_in, unsigned *d_out,
                               const unsigned *d_cntIn, unsigned *d_cntOut, unsigned *d_over, int tpb)
{
	const unsigned sh = eq24b_shared((unsigned)r);
	if (!sh) return -1;
	switch (r) {
	case 1: EQ24B_NS::digitR<1><<<EQ24B_NBUCKETS * EQ24B_CHUNKS_R(1u), tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
	case 2: EQ24B_NS::digitR<2><<<EQ24B_NBUCKETS * EQ24B_CHUNKS_R(2u), tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
	case 3: EQ24B_NS::digitR<3><<<EQ24B_NBUCKETS * EQ24B_CHUNKS_R(3u), tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
	case 4: EQ24B_NS::digitR<4><<<EQ24B_NBUCKETS * EQ24B_CHUNKS_R(4u), tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
#if EQ_WK > 5
	case 5: EQ24B_NS::digitR<5><<<EQ24B_NBUCKETS * EQ24B_CHUNKS_R(5u), tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
	case 6: EQ24B_NS::digitR<6><<<EQ24B_NBUCKETS * EQ24B_CHUNKS_R(6u), tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
#endif
	default: return -1;
	}
	// A launch that never starts (bad shared size, bad geometry) leaves
	// cudaDeviceSynchronize happy. Check the LAUNCH separately or a broken
	// kernel reads as "ran, produced nothing".
	return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

extern "C" int EQ24B_API(digitR)(int r, const unsigned *d_in, unsigned *d_out,
                                 const unsigned *d_cntIn, unsigned *d_cntOut,
                                 unsigned *d_over, int tpb)
{
	cudaMemset(d_cntOut, 0, (size_t)EQ24B_NBUCKETS * sizeof(unsigned));
	const int e = eq24b_launch_digitR(r, d_in, d_out, d_cntIn, d_cntOut, d_over, tpb);
	if (e) return e;
	return cudaDeviceSynchronize() == cudaSuccess ? 0 : -3;
}

static int eq24b_launch_digitH(const blake2b_state *st, u64 *pinned, unsigned *d_layer0,
                               unsigned *d_counts, unsigned *d_overflow, int blocks, int tpb)
{
	if (st->buflen != 12) return -1;
	const u64 m0 = *(const u64 *)(st->buf);
	const u64 m1lo = *(const u32 *)(st->buf + 8);
	const u64 counter = st->counter + 16;      // 12 buffered + the 4-byte index
	u64 *d_h = NULL;
	cudaError_t e;
	if (pinned) {
		memcpy(pinned, st->h, 8 * sizeof(u64));
		e = cudaMemcpyToSymbolAsync(EQ24B_NS::eq24b_mid_h, pinned, 8 * sizeof(u64), 0, cudaMemcpyHostToDevice, 0);
	} else {
		e = cudaMemcpyToSymbol(EQ24B_NS::eq24b_mid_h, st->h, 8 * sizeof(u64));
	}
	if (e != cudaSuccess || cudaGetSymbolAddress((void **)&d_h, EQ24B_NS::eq24b_mid_h) != cudaSuccess) return -2;
	EQ24B_NS::digitH<<<blocks, tpb>>>(d_h, m0, m1lo, counter,
	                                  d_layer0, d_counts, d_overflow);
	// a launch that never starts (e.g. too many threads for the register count)
	// passes cudaDeviceSynchronize and leaves layer 0 empty
	return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

extern "C" int EQ24B_API(digitH)(const void *blake_state,
                                 unsigned *d_layer0,
                                 unsigned *d_counts, unsigned *d_overflow,
                                 int blocks, int tpb)
{
	cudaMemset(d_counts, 0, (size_t)EQ24B_NBUCKETS * sizeof(unsigned));
	cudaMemset(d_overflow, 0, sizeof(unsigned));
	const int e = eq24b_launch_digitH((const blake2b_state *)blake_state, NULL, d_layer0, d_counts, d_overflow, blocks, tpb);
	if (e) return e;
	return cudaDeviceSynchronize() == cudaSuccess ? 0 : -3;
}
