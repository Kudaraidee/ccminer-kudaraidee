/**
 * Reduced-round BLAKE2b permutation as Lyra2 uses it (the "G function" with
 * 32/24/16/63 rotations), shared by every lyra2 matrix stage.
 *
 * Two state layouts are supported, both of them the 16-word BLAKE2b state seen
 * as a 4x4 matrix of uint2:
 *
 *  - warp-cooperative: 4 lanes per hash, lane L holding column L, i.e. words
 *    { L, 4+L, 8+L, 12+L } as s[0..3]. The diagonal step needs cross-lane
 *    rotations, hence the shuffles. Launch with blockDim.x == 4.
 *  - single-thread: one thread holds all 16 words as uint2x4 s[4], so the
 *    diagonal step is just a different operand pattern and no shuffle is needed.
 *
 * Include after cuda_lyra2_vectors.h (needs uint2x4, ROR24/ROR16/ROR2).
 */
#ifndef CUDA_BLAKE2B_DEVICE_CUH
#define CUDA_BLAKE2B_DEVICE_CUH

static __device__ __forceinline__
void Gfunc(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; uint2 tmp = d; d.y = a.x ^ tmp.x; d.x = a.y ^ tmp.y;
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);
}

// ---- warp-cooperative layout (4 lanes per hash) -------------------------------

__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	return __shfl(a, b, c);
}

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
}

__device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	a1 = WarpShuffle(a1, b1, c);
	a2 = WarpShuffle(a2, b2, c);
	a3 = WarpShuffle(a3, b3, c);
}

// one ROUND_LYRA: G over the columns, then over the diagonals
__device__ __forceinline__ void round_lyra(uint2 s[4])
{
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

// Quad-scoped variants, for kernels whose launch can leave part of a warp
// inactive (a tail quad that returned on a thread-count guard): the shuffles
// then must name only the 4 lanes of this hash, because a __shfl_sync mask that
// names an exited thread is undefined behaviour. quad_mask() assumes the same
// blockDim.x == 4 layout.
__device__ __forceinline__ uint32_t quad_mask()
{
	const uint32_t laneid = (threadIdx.y * 4 + threadIdx.x) & 31;
	return 0xfu << (laneid & ~3u);
}

__device__ __forceinline__ uint2 QuadShuffle(uint2 a, uint32_t srcLane, uint32_t mask)
{
	return make_uint2(__shfl_sync(mask, a.x, srcLane, 4), __shfl_sync(mask, a.y, srcLane, 4));
}

__device__ __forceinline__ void QuadShuffle3(uint2 &a1, uint2 &a2, uint2 &a3,
	uint32_t b1, uint32_t b2, uint32_t b3, uint32_t mask)
{
	a1 = QuadShuffle(a1, b1, mask);
	a2 = QuadShuffle(a2, b2, mask);
	a3 = QuadShuffle(a3, b3, mask);
}

__device__ __forceinline__ void round_lyra_quad(uint2 s[4], const uint32_t qmask)
{
	Gfunc(s[0], s[1], s[2], s[3]);
	QuadShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, qmask);
	Gfunc(s[0], s[1], s[2], s[3]);
	QuadShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, qmask);
}

// ---- single-thread layout (all 16 words in one thread) ------------------------

static __device__ __forceinline__
void round_lyra(uint2x4* s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}

#endif // CUDA_BLAKE2B_DEVICE_CUH
