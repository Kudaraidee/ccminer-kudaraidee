/* For IDE: */
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "argon2d_kernel.h"
#include "cudaexception.h"

#include <stdexcept>
#ifndef NDEBUG
#include <iostream>
#endif

/* The other ARGON2_* constants come from argon2d_kernel.h. These three cannot:
 * argon2ref/argon2.h, which argon2d.cu includes, declares the versions as an
 * enum and ARGON2_SYNC_POINTS as a macro, so moving either to the shared
 * header breaks that translation unit. */
#define ARGON2_VERSION_10 0x10
#define ARGON2_VERSION_13 0x13
#define ARGON2_SYNC_POINTS 4

#define THREADS_PER_LANE 32
#define QWORDS_PER_THREAD (ARGON2_QWORDS_IN_BLOCK / 32)


using namespace std;


__device__ uint64_t u64_build(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ uint32_t u64_lo(uint64_t x)
{
    return (uint32_t)x;
}

__device__ uint32_t u64_hi(uint64_t x)
{
    return (uint32_t)(x >> 32);
}


static __device__ __forceinline__ uint64_t __ldL1(const uint64_t* ptr)
{
    uint32_t a,b;
    asm("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(a), "=r"(b) : "l"(ptr));
    return u64_build(b,a);

}

static __device__ __forceinline__ void __stL1(const uint64_t *ptr, const uint64_t value)
{
    uint32_t* val=(uint32_t*)&value;
    asm("st.global.wb.v2.u32 [%0], {%1, %2};" ::"l"(ptr), "r"(val[0]), "r"(val[1]));
}

__device__ uint64_t u64_shuffle(uint64_t v, uint32_t thread)
{
    uint32_t lo = u64_lo(v);
    uint32_t hi = u64_hi(v);
    lo = __shfl_sync(0xffffffff, lo, thread);
    hi = __shfl_sync(0xffffffff, hi, thread);
    return u64_build(hi, lo);
}


__device__ uint64_t cmpeq_mask(uint32_t test, uint32_t ref)
{
    uint32_t x = -(uint32_t)(test == ref);
    return u64_build(x, x);
}

__device__ uint64_t block_th_get(const struct block_th *b, uint32_t idx)
{
    uint64_t res = 0;
    res ^= cmpeq_mask(idx, 0) & b->a;
    res ^= cmpeq_mask(idx, 1) & b->b;
    res ^= cmpeq_mask(idx, 2) & b->c;
    res ^= cmpeq_mask(idx, 3) & b->d;
    return res;
}

__device__ void block_th_set(struct block_th *b, uint32_t idx, uint64_t v)
{
    b->a ^= cmpeq_mask(idx, 0) & (v ^ b->a);
    b->b ^= cmpeq_mask(idx, 1) & (v ^ b->b);
    b->c ^= cmpeq_mask(idx, 2) & (v ^ b->c);
    b->d ^= cmpeq_mask(idx, 3) & (v ^ b->d);
}

__device__ void move_block(struct block_th *dst, const struct block_th *src)
{
    *dst = *src;
}

__device__ void xor_block(struct block_th *dst, const struct block_th *src)
{
    dst->a ^= src->a;
    dst->b ^= src->b;
    dst->c ^= src->c;
    dst->d ^= src->d;
}

__device__ void load_block(struct block_th *dst, const struct block_g *src,
                           uint32_t thread)
{
    dst->a = __ldL1(&src->data[0 * THREADS_PER_LANE + thread]);
    dst->b = __ldL1(&src->data[1 * THREADS_PER_LANE + thread]);
    dst->c = __ldL1(&src->data[2 * THREADS_PER_LANE + thread]);
    dst->d = __ldL1(&src->data[3 * THREADS_PER_LANE + thread]);
}
__device__ void shared_load_block(struct block_th *dst, const struct block_g *src,
                           uint32_t thread)
{
    dst->a = src->data[0 * THREADS_PER_LANE + thread];
    dst->b = src->data[1 * THREADS_PER_LANE + thread];
    dst->c = src->data[2 * THREADS_PER_LANE + thread];
    dst->d = src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void load_block_xor(struct block_th *dst, const struct block_g *src,
                               uint32_t thread)
{

    dst->a ^= __ldL1(&src->data[0 * THREADS_PER_LANE + thread]);
    dst->b ^= __ldL1(&src->data[1 * THREADS_PER_LANE + thread]);
    dst->c ^= __ldL1(&src->data[2 * THREADS_PER_LANE + thread]);
    dst->d ^= __ldL1(&src->data[3 * THREADS_PER_LANE + thread]);
}
__device__ void shared_load_block_xor(struct block_th *dst, const struct block_g *src,
                               uint32_t thread)
{

    dst->a ^= src->data[0 * THREADS_PER_LANE + thread];
    dst->b ^= src->data[1 * THREADS_PER_LANE + thread];
    dst->c ^= src->data[2 * THREADS_PER_LANE + thread];
    dst->d ^= src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void shared_move_block(struct block_g *dst, const struct block_g *src,
                            uint32_t thread)
{
    dst->data[0 * THREADS_PER_LANE + thread] = src->data[0 * THREADS_PER_LANE + thread];
    dst->data[1 * THREADS_PER_LANE + thread] = src->data[1 * THREADS_PER_LANE + thread];
    dst->data[2 * THREADS_PER_LANE + thread] = src->data[2 * THREADS_PER_LANE + thread];
    dst->data[3 * THREADS_PER_LANE + thread] = src->data[3 * THREADS_PER_LANE + thread];

}


__device__ void store_block(struct block_g *dst, const struct block_th *src,
                            uint32_t thread)
{
    __stL1(&dst->data[0 * THREADS_PER_LANE + thread] , src->a);
    __stL1(&dst->data[1 * THREADS_PER_LANE + thread] , src->b);
    __stL1(&dst->data[2 * THREADS_PER_LANE + thread] , src->c);
    __stL1(&dst->data[3 * THREADS_PER_LANE + thread] , src->d);

}

__device__ void shared_store_block(struct block_g *dst, const struct block_th *src,
                            uint32_t thread)
{
    dst->data[0 * THREADS_PER_LANE + thread] = src->a;
    dst->data[1 * THREADS_PER_LANE + thread] = src->b;
    dst->data[2 * THREADS_PER_LANE + thread] = src->c;
    dst->data[3 * THREADS_PER_LANE + thread] = src->d;

}

__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ uint64_t f(uint64_t x, uint64_t y)
{
    uint32_t xlo = u64_lo(x);
    uint32_t ylo = u64_lo(y);
    return x + y + 2 * u64_build(__umulhi(xlo, ylo), xlo * ylo);
}

__device__ void g(struct block_th *block)
{
    uint64_t a, b, c, d;
    a = block->a;
    b = block->b;
    c = block->c;
    d = block->d;

    a = f(a, b);
    d = rotr64(d ^ a, 32);
    c = f(c, d);
    b = rotr64(b ^ c, 24);
    a = f(a, b);
    d = rotr64(d ^ a, 16);
    c = f(c, d);
    b = rotr64(b ^ c, 63);

    block->a = a;
    block->b = b;
    block->c = c;
    block->d = d;
}

template<class shuffle>
__device__ void apply_shuffle(struct block_th *block, uint32_t thread)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t src_thr = shuffle::apply(thread, i);

        uint64_t v = block_th_get(block, i);
        v = u64_shuffle(v, src_thr);
        block_th_set(block, i, v);
    }
}

__device__ void transpose(struct block_th *block, uint32_t thread)
{
    uint32_t thread_group = (thread & 0x0C) >> 2;
    for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
        uint32_t thr = (i << 2) ^ thread;
        uint32_t idx = thread_group ^ i;

        uint64_t v = block_th_get(block, idx);
        v = u64_shuffle(v, thr);
        block_th_set(block, idx, v);
    }
}

struct identity_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return thread;
    }
};

struct shift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct unshift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct shift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

struct unshift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

__device__ void shuffle_block(struct block_th *block, uint32_t thread)
{
    transpose(block, thread);

    g(block);

    apply_shuffle<shift1_shuffle>(block, thread);

    g(block);

    apply_shuffle<unshift1_shuffle>(block, thread);
    transpose(block, thread);

    g(block);

    apply_shuffle<shift2_shuffle>(block, thread);

    g(block);

    apply_shuffle<unshift2_shuffle>(block, thread);
}

__device__ void next_addresses(struct block_th *addr, struct block_th *tmp,
                               uint32_t thread_input, uint32_t thread)
{
    addr->a = u64_build(0, thread_input);
    addr->b = 0;
    addr->c = 0;
    addr->d = 0;

    shuffle_block(addr, thread);

    addr->a ^= u64_build(0, thread_input);
    move_block(tmp, addr);

    shuffle_block(addr, thread);

    xor_block(addr, tmp);
}

__device__ void compute_ref_pos(
        uint32_t lanes, uint32_t segment_blocks,
        uint32_t pass, uint32_t lane, uint32_t slice, uint32_t offset,
        uint32_t *ref_lane, uint32_t *ref_index)
{
    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    /* lanes is a power of two (argon2d.cu); avoids a runtime division */
    *ref_lane = *ref_lane & (lanes - 1);

    uint32_t base;
    if (pass != 0) {
        base = lane_blocks - segment_blocks;
    } else {
        if (slice == 0) {
            *ref_lane = lane;
        }
        base = slice * segment_blocks;
    }

    uint32_t ref_area_size = base + offset - 1;
    if (*ref_lane != lane) {
        ref_area_size = min(ref_area_size, base);
    }

    *ref_index = __umulhi(*ref_index, *ref_index);
    *ref_index = ref_area_size - 1 - __umulhi(ref_area_size, *ref_index);

    if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
        *ref_index += (slice + 1) * segment_blocks;
        if (*ref_index >= lane_blocks) {
            *ref_index -= lane_blocks;
        }
    }
}

struct ref {
    uint32_t ref_lane;
    uint32_t ref_index;
};



__device__ void argon2_core(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_th *tmp,struct block_g *c,
        uint32_t lanes, uint32_t thread, uint32_t pass, uint32_t version,
        uint32_t ref_index, uint32_t ref_lane,uint32_t curr_index,uint32_t lane)
{
    struct block_g *mem_ref = memory + ref_index * lanes + ref_lane;

    /* v1.3 on passes after the first: B[i] = B[i] XOR G(B[i-1], B[ref]),
     * where v1.0 overwrites. Dead for every t_cost=1 or v0x10 variant.
     * The old B[i] must be read the way the store side below writes it -
     * columns 2..7 live only in the shared cache, and their global slots
     * hold garbage from argon2_initialize (16 first-blocks at lanes=1). */
    if (version != ARGON2_VERSION_10 && pass != 0) {
        if (curr_index < 8 && curr_index > 1)
            shared_load_block(tmp, &c[lane * ARGON2_SHARED_BLOCKS_PER_LANE + curr_index-2], thread);
        else
            load_block(tmp, mem_curr, thread);

        if (ref_index<8 && ref_index > 1){
            shared_load_block_xor(prev, &c[ref_lane * ARGON2_SHARED_BLOCKS_PER_LANE + ref_index-2], thread);
        }else{
            load_block_xor(prev, mem_ref, thread);
        }

        xor_block(tmp, prev);
    } else {
        if (ref_index<8 && ref_index > 1){
            shared_load_block_xor(prev, &c[ref_lane * ARGON2_SHARED_BLOCKS_PER_LANE + ref_index-2], thread);
        }else{
            load_block_xor(prev, mem_ref, thread);
        }

        move_block(tmp, prev);
    }

    shuffle_block(prev, thread);

    xor_block(prev, tmp);

    /* Columns 2..7 live in the shared cache (their global slots are never
     * read); everything else, including the last column, goes to global.
     * The last column must NOT be staged into shared: its natural slot
     * aliases the column-2 cache entry, and other lanes may still legally
     * ref column 2 within the final slice (no barrier until slice end) -
     * overwriting it early made those refs read the final block instead
     * (rare wrong hashes, caught by the host re-verify since the donor). */
    if(curr_index<8 && curr_index >1){
        shared_store_block(&c[lane * ARGON2_SHARED_BLOCKS_PER_LANE + curr_index-2], prev, thread);
    }else{
        store_block(mem_curr, prev, thread);
    }
}


__device__ void argon2_step(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_th *tmp, struct block_th *addr,
        struct block_g *c,
        uint32_t lanes, uint32_t segment_blocks, uint32_t thread,
        uint32_t *thread_input, uint32_t lane, uint32_t pass, uint32_t slice,
        uint32_t offset, uint32_t version, uint32_t type)
{
    uint32_t ref_index, ref_lane;

    if (type == ARGON2_I || (type == ARGON2_ID && pass == 0 &&
            slice < ARGON2_SYNC_POINTS / 2)) {
        /* Data-independent addressing: J1/J2 come from a generated address
         * block, not from the previous block. Segments longer than
         * ARGON2_ADDRESSES_IN_BLOCK regenerate mid-segment. */
        uint32_t addr_index = offset % ARGON2_ADDRESSES_IN_BLOCK;
        if (addr_index == 0) {
            if (thread == 6) {
                ++*thread_input;
            }
            next_addresses(addr, tmp, *thread_input, thread);
        }

        uint32_t thr = addr_index % THREADS_PER_LANE;
        uint32_t idx = addr_index / THREADS_PER_LANE;

        uint64_t v = block_th_get(addr, idx);
        v = u64_shuffle(v, thr);
        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);
    } else {
        uint64_t v = u64_shuffle(prev->a, 0);
        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);
    }

    compute_ref_pos(lanes, segment_blocks, pass, lane, slice, offset, &ref_lane, &ref_index);

    argon2_core(memory, mem_curr, prev, tmp,c, lanes, thread, pass, version, ref_index, ref_lane,slice*segment_blocks+offset,lane);
}



__global__ void argon2_fill(
        struct block_g *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks, uint32_t version, uint32_t type)
{
    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t lane   = threadIdx.y;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

    memory += (size_t)job_id * lanes * lane_blocks;

    struct block_th prev, addr, tmp;

    /* lanes * ARGON2_SHARED_BLOCKS_PER_LANE blocks, sized at launch. */
    extern __shared__ block_g c[];

    /* Address-generator seed, one word per lane-thread: 0=pass, 1=lane,
     * 2=slice, 3=memory blocks, 4=passes, 5=type, 6=address-block counter.
     * pass and slice are 0 here, so threads 0 and 2 take the default. */
    uint32_t thread_input;

    if (type == ARGON2_I || type == ARGON2_ID) {
        switch (thread) {
        case 1:
            thread_input = lane;
            break;
        case 3:
            thread_input = lanes * lane_blocks;
            break;
        case 4:
            thread_input = passes;
            break;
        case 5:
            thread_input = type;
            break;
        default:
            thread_input = 0;
            break;
        }

        /* Offsets 0 and 1 are already filled, so argon2_step's
         * addr_index==0 path never runs for them: seed the first block. */
        if (segment_blocks > 2) {
            if (thread == 6) {
                ++thread_input;
            }
            next_addresses(&addr, &tmp, thread_input, thread);
        }
    }

    struct block_g *mem_lane = memory + lane;

    struct block_g *mem_prev = mem_lane + 1 * lanes;

    struct block_g *mem_curr = mem_lane + 2 * lanes;


    load_block(&prev, mem_prev, thread);

    uint32_t skip = 2;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
            for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
                if (skip > 0) {
                    --skip;
                    continue;
                }

                argon2_step(
                            memory, mem_curr, &prev, &tmp, &addr, c, lanes,
                            segment_blocks, thread, &thread_input, lane, pass,
                            slice, offset, version, type);

                mem_curr += lanes;
            }

            __syncthreads();

            if (type == ARGON2_I || type == ARGON2_ID) {
                if (thread == 2) {
                    ++thread_input;   /* next slice */
                }
                if (thread == 6) {
                    thread_input = 0; /* restart the address-block counter */
                }
            }
        }

        /* Argon2id stops using addresses after pass 0, so only Argon2i
         * advances the pass counter. */
        if (type == ARGON2_I) {
            if (thread == 0) {
                ++thread_input;
            }
            if (thread == 2) {
                thread_input = 0;
            }
        }

        mem_curr = mem_lane;
    }


    __syncthreads();

    /* Final block = XOR of each lane's last-column block, read from global
     * (__syncthreads() above orders the stores block-wide) and written into
     * the job's block 0 where argon2_finalize reads it. The block has
     * lanes*32 threads for 256 output words, so stride-loop for lanes < 8
     * (e.g. lanes=1 has 32 threads covering 8 words each). */
    thread=threadIdx.x + blockDim.x * threadIdx.y;
    uint32_t* last_col = (uint32_t*)(memory + (size_t)(lane_blocks - 1) * lanes);

    for (uint32_t j = thread; j < 256; j += blockDim.x * blockDim.y) {
        uint32_t buf = 0;

        for (uint32_t i=0; i<lanes; i++){
            buf ^= last_col[j + i*256];
        }

        ((uint32_t*)memory)[j] = buf;
    }
}
