// SPDX-License-Identifier: GPL-3.0-or-later
// Adapted from ParallaxProtocol/hashwarp libxhash-cuda/dagger_shuffled.cuh (GPL-3.0).
#pragma once
#include "xhash_cuda.cuh"
#include "xhash_fnv.cuh"
#include "xhash_keccak.cuh"

// Hashes each group of THREADS_PER_HASH lanes computes at once
#define _PARALLEL_HASH 4

// Development builds (-DHASHWARP_COMPUTE_BENCH=ON) confine DAG reads to 32 MB, which stays
// in L2: memory stops being the bottleneck and the hashrate measures the kernel's compute
// efficiency, as on compute bound GPUs. The hashes are wrong: never use it to mine.
#ifdef XHASH_COMPUTE_BENCH
#define DAG_ACCESS_ITEMS (d_dag_size < 262144u ? d_dag_size : 262144u)
#else
#define DAG_ACCESS_ITEMS d_dag_size
#endif

// DedupOffsets: every lane computes the next DAG address of one hash and shares it,
// instead of all lanes computing the addresses of all the hashes. Fewer instructions
// (faster on compute bound GPUs) but the hashes wait on each other, which leaves fewer
// memory requests in flight (slower on memory bound GPUs). The miner times both.
template <bool DedupOffsets>
DEV_INLINE bool compute_hash(uint64_t nonce, uint2* mix_hash)
{
    // sha3_512(header .. nonce)
    uint2 state[12];

    state[4] = vectorize(nonce);

    keccak_f1600_init(state);

    // Threads work together in this phase in groups of 8.
    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
    const int mix_idx = thread_id & 3;

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
    {
        uint4 mix[_PARALLEL_HASH];
        uint32_t offset[_PARALLEL_HASH];
        uint32_t init0[_PARALLEL_HASH];

        // share init among threads
        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[8];
            for (int j = 0; j < 8; j++)
            {
                shuffle[j].x = XHASH_SHFL(state[j].x, i + p, THREADS_PER_HASH);
                shuffle[j].y = XHASH_SHFL(state[j].y, i + p, THREADS_PER_HASH);
            }
            switch (mix_idx)
            {
            case 0:
                mix[p] = vectorize2(shuffle[0], shuffle[1]);
                break;
            case 1:
                mix[p] = vectorize2(shuffle[2], shuffle[3]);
                break;
            case 2:
                mix[p] = vectorize2(shuffle[4], shuffle[5]);
                break;
            case 3:
                mix[p] = vectorize2(shuffle[6], shuffle[7]);
                break;
            }
            init0[p] = XHASH_SHFL(shuffle[0].x, 0, THREADS_PER_HASH);
        }

        if (DedupOffsets)
        {
            // The next DAG address of hash p only depends on one word of lane t's mix:
            // lane p computes it for hash p and shares it
            uint32_t own_init0 = init0[0];
            for (int p = 1; p < _PARALLEL_HASH; p++)
                own_init0 = (thread_id == p) ? init0[p] : own_init0;

            for (uint32_t a = 0; a < ACCESSES; a += 4)
            {
                int t = (int)((a >> 2) & 7u);

                for (uint32_t b = 0; b < 4; b++)
                {
                    uint32_t own_word = 0;
                    for (int p = 0; p < _PARALLEL_HASH; p++)
                    {
                        uint32_t word = XHASH_SHFL(((uint32_t*)&mix[p])[b], t, THREADS_PER_HASH);
                        own_word = (thread_id == p) ? word : own_word;
                    }
                    uint32_t own_offset = fnv(own_init0 ^ (a + b), own_word) % DAG_ACCESS_ITEMS;

                    for (int p = 0; p < _PARALLEL_HASH; p++)
                    {
                        offset[p] = XHASH_SHFL(own_offset, p, THREADS_PER_HASH);
                        mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
                    }
                }
            }
        }
        else
        {
            for (uint32_t a = 0; a < ACCESSES; a += 4)
            {
                int t = (int)((a >> 2) & 7u);

                for (uint32_t b = 0; b < 4; b++)
                {
                    for (int p = 0; p < _PARALLEL_HASH; p++)
                    {
                        offset[p] =
                            fnv(init0[p] ^ (a + b), ((uint32_t*)&mix[p])[b]) % DAG_ACCESS_ITEMS;
                        offset[p] = XHASH_SHFL(offset[p], t, THREADS_PER_HASH);
                        mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
                    }
                }
            }
        }

        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[4];
            uint32_t thread_mix = fnv_reduce(mix[p]);

            // update mix across threads
            shuffle[0].x = XHASH_SHFL(thread_mix, 0, THREADS_PER_HASH);
            shuffle[0].y = XHASH_SHFL(thread_mix, 1, THREADS_PER_HASH);
            shuffle[1].x = XHASH_SHFL(thread_mix, 2, THREADS_PER_HASH);
            shuffle[1].y = XHASH_SHFL(thread_mix, 3, THREADS_PER_HASH);
            shuffle[2].x = XHASH_SHFL(thread_mix, 4, THREADS_PER_HASH);
            shuffle[2].y = XHASH_SHFL(thread_mix, 5, THREADS_PER_HASH);
            shuffle[3].x = XHASH_SHFL(thread_mix, 6, THREADS_PER_HASH);
            shuffle[3].y = XHASH_SHFL(thread_mix, 7, THREADS_PER_HASH);

            if ((i + p) == thread_id)
            {
                // move mix into state:
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }

    // keccak_256(keccak_512(header..nonce) .. mix);
    if (cuda_swab64(keccak_f1600_final(state)) > d_target)
        return true;

    mix_hash[0] = state[8];
    mix_hash[1] = state[9];
    mix_hash[2] = state[10];
    mix_hash[3] = state[11];

    return false;
}
