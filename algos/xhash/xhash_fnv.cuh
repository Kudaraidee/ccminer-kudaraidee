// SPDX-License-Identifier: GPL-3.0-or-later
// Adapted from ParallaxProtocol/hashwarp libxhash-cuda/fnv.cuh (GPL-3.0).
#pragma once
#define FNV_PRIME 0x01000193u

// XHash uses true FNV-1: for each 32-bit b, multiply by prime then xor each byte of b (LE)
DEV_INLINE uint32_t fnv_u32(uint32_t a, uint32_t b)
{
    uint32_t h = a;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        h = h * FNV_PRIME;
        h ^= (b & 0xFFu);
        b >>= 8;
    }
    return h;
}

// Keep macro name for existing call sites
#undef fnv
#define fnv(x, y) fnv_u32((x), (y))

DEV_INLINE uint4 fnv4(uint4 a, uint4 b)
{
    uint4 h = a;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        h.x = h.x * FNV_PRIME;
        h.y = h.y * FNV_PRIME;
        h.z = h.z * FNV_PRIME;
        h.w = h.w * FNV_PRIME;

        h.x ^= (b.x & 0xFFu);
        h.y ^= (b.y & 0xFFu);
        h.z ^= (b.z & 0xFFu);
        h.w ^= (b.w & 0xFFu);

        b.x >>= 8;
        b.y >>= 8;
        b.z >>= 8;
        b.w >>= 8;
    }
    return h;
}

DEV_INLINE uint32_t fnv_reduce(uint4 v)
{
    uint32_t r = fnv_u32(v.x, v.y);
    r = fnv_u32(r, v.z);
    r = fnv_u32(r, v.w);
    return r;
}
