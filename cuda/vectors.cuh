#ifndef CUDA_VECTORS_H
#define CUDA_VECTORS_H

#include "cuda/helper.cuh"

/* Macros for uint2 operations (used by skein) */


static __device__ __forceinline__ uint2 operator+ (const uint2 a, const uint32_t b)
{
#if 0 && defined(__CUDA_ARCH__) && CUDA_VERSION < 7000
	uint2 result;
	asm(
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
	: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
#else
	return vectorize(devectorize(a) + b);
#endif
}

/* ulonglong2to8 — 256-bit vector (4x ulonglong2). Backported from
 * neoscrypt/cuda_vectors.h (the only symbol x11/cuda_streebog.cu needed from
 * that header); implemented member-wise so it is self-contained and does not
 * pull a global ulonglong2 operator overload into this shared header. The
 * neoscrypt copy is removed when neoscrypt migrates. */
typedef struct __align__(64) ulonglong2to8 {
	ulonglong2 l0, l1, l2, l3;
} ulonglong2to8;

static __inline__ __device__ ulonglong2to8 make_ulonglong2to8(ulonglong2 s0, ulonglong2 s1, ulonglong2 s2, ulonglong2 s3)
{
	ulonglong2to8 t; t.l0 = s0; t.l1 = s1; t.l2 = s2; t.l3 = s3;
	return t;
}

static __forceinline__ __device__
ulonglong2to8 operator^ (const ulonglong2to8 &a, const ulonglong2to8 &b)
{
	return make_ulonglong2to8(
		make_ulonglong2(a.l0.x ^ b.l0.x, a.l0.y ^ b.l0.y),
		make_ulonglong2(a.l1.x ^ b.l1.x, a.l1.y ^ b.l1.y),
		make_ulonglong2(a.l2.x ^ b.l2.x, a.l2.y ^ b.l2.y),
		make_ulonglong2(a.l3.x ^ b.l3.x, a.l3.y ^ b.l3.y));
}

static __forceinline__ __device__
ulonglong2to8 operator+ (const ulonglong2to8 &a, const ulonglong2to8 &b)
{
	return make_ulonglong2to8(
		make_ulonglong2(a.l0.x + b.l0.x, a.l0.y + b.l0.y),
		make_ulonglong2(a.l1.x + b.l1.x, a.l1.y + b.l1.y),
		make_ulonglong2(a.l2.x + b.l2.x, a.l2.y + b.l2.y),
		make_ulonglong2(a.l3.x + b.l3.x, a.l3.y + b.l3.y));
}

static __forceinline__ __device__ void operator^= (ulonglong2to8 &a, const ulonglong2to8 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (ulonglong2to8 &a, const ulonglong2to8 &b) { a = a + b; }

#endif // CUDA_VECTORS_H
