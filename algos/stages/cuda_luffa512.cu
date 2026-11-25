/*
 * luffa_for_32.c lineage — Luffa-512 for the x-family 64-byte chaining.
 *
 * The device implementation (state struct, round/permutation macros, IV and
 * round-constant tables, luffa512_hash_64) lives in cuda/luffa512_device.cuh
 * (docs/coding-guideline.md §3); the kernel below is a thin wrapper. The
 * tables are statically initialized in the header, so no init-time upload
 * remains.
 */

#include "cuda/helper.cuh"

#include "cuda/luffa512_device.cuh"

// Die Hash-Funktion
__global__ void luffa512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[8 * hashPosition];

#if __CUDA_ARCH__ == 610
        // Pascal: plain 32-bit loads bypass L1, so copy the record with 128-bit access
        { uint4 *p_ = (uint4*)(Hash); uint4 l_[4];
		#pragma unroll
		for (int i_ = 0; i_ < 4; i_++) l_[i_] = __ldg(&p_[i_]);
		luffa512_hash_64((uint32_t*)l_);
		#pragma unroll
		for (int i_ = 0; i_ < 4; i_++) p_[i_] = l_[i_]; }
#else
        luffa512_hash_64(Hash);
#endif
    }
}

/* Unit self-test for cuda/luffa512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool luffa512_device_selftest(int thr_id);

// Setup Function
__host__
void luffa512_cpu_init(int thr_id, uint32_t threads)
{
    luffa512_device_selftest(thr_id);
}

__host__ void luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const uint32_t threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    luffa512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
    MyStreamSynchronize(NULL, order, thr_id);
}
