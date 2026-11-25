// SPDX-License-Identifier: GPL-3.0-or-later
//
// KawPoW core (see kawpow_core.h): the ethash/C++-STL side of the port. Owns the
// epoch/DAG state machine, the per-period NVRTC kernel, device buffers, and the
// host ProgPoW reference used to re-verify every GPU candidate before it is
// reported to the ccminer bridge. No ccminer headers here (bool-macro world).

#include "kawpow_core.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include "../progpow_multi/ppmulti_jit.h"
#include "kawpow_params.h"   // the single kawpow pp_params definition
#include "../progpow_multi/ppmulti_dag.h"
#include "../progpow_multi/ppmulti_epoch.h"   // pp_dag_epochs

#include "ethash/ethash.hpp"
#include "ethash/ethash.h"
#include "ethash/ethash-internal.hpp"  // ethash::is_less_or_equal
#include "ethash/progpow.hpp"

#include <cstdio>
#include <cstring>

namespace {



struct core_state
{
    int sm_arch = 86;
    ppmulti_dag dag;
    ppmulti_jit* jit = nullptr;
    uint32_t* d_header = nullptr;
    ppmulti_result* d_result = nullptr;
    const ethash::epoch_context* host_ctx = nullptr;
    int host_epoch = -1;

    ~core_state()
    {
        if (jit) delete jit;
        if (d_header) cudaFree(d_header);
        if (d_result) cudaFree(d_result);
        if (host_ctx)
            ethash_destroy_epoch_context(const_cast<ethash_epoch_context*>(host_ctx));
    }
};

// Top 64 bits of a 256-bit big-endian (MSB-first) target, as the GPU screen.
static uint64_t target_top64(const unsigned char* t)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | t[i];
    return v;
}

// Cached host epoch context (light path) for reverify + self-test.
const ethash::epoch_context& host_ctx_for(core_state* s, int height)
{
    const int epoch = ethash::get_epoch_number(height);
    if (!s->host_ctx || s->host_epoch != epoch) {
        if (s->host_ctx)
            ethash_destroy_epoch_context(const_cast<ethash_epoch_context*>(s->host_ctx));
        s->host_ctx = ethash::create_epoch_context(epoch).release();
        s->host_epoch = epoch;
    }
    return *s->host_ctx;
}

ethash::hash256 header_from_bytes(const unsigned char* b)
{
    ethash::hash256 h;
    memcpy(h.bytes, b, 32);
    return h;
}

} // namespace

extern "C" void* kawpow_core_create(int sm_arch)
{
    core_state* s = new core_state();
    s->sm_arch = sm_arch;
    s->jit = new ppmulti_jit(sm_arch, kPpKawpow);
    if (cudaMalloc(&s->d_header, 32) != cudaSuccess) { delete s; return nullptr; }
    if (cudaMalloc(&s->d_result, sizeof(ppmulti_result)) != cudaSuccess) { delete s; return nullptr; }
    return s;
}

extern "C" void kawpow_core_destroy(void* h)
{
    if (h) delete static_cast<core_state*>(h);
}

extern "C" int kawpow_core_ensure(void* h, int height, int* regenerated)
{
    core_state* s = static_cast<core_state*>(h);
    bool re = false;
    // Standard ethash sizing collapses the triple to one epoch; derived through
    // pp_dag_epochs anyway so the path cannot drift from pp_params.
    const int epoch = height / kPpKawpow.epoch_length;
    int seed_epoch = 0, light_epoch = 0, full_epoch = 0;
    pp_dag_epochs(kPpKawpow, epoch, &seed_epoch, &light_epoch, &full_epoch);
    bool ok = s->dag.ensure(seed_epoch, light_epoch, full_epoch, &re);
    if (regenerated) *regenerated = re ? 1 : 0;
    return ok ? 1 : 0;
}

extern "C" int kawpow_core_selftest(void* h, int height)
{
    core_state* s = static_cast<core_state*>(h);
    const uint64_t period = (uint64_t)height / 3u;
    const uint32_t items = s->dag.dataset_items_128() / 2u;  // 256-byte items
    CUfunction fn;
    if (!s->jit->get(period, items, &fn)) return 0;

    unsigned char hdr[32];
    for (int i = 0; i < 32; i++) hdr[i] = (unsigned char)(i * 7 + 1);
    const uint64_t start = 0x0123456789abcd00ULL;

    cudaMemcpy(s->d_header, hdr, 32, cudaMemcpyHostToDevice);
    cudaMemset(s->d_result, 0, sizeof(ppmulti_result));

    CUdeviceptr dhdr = (CUdeviceptr)s->d_header;
    CUdeviceptr ddag = (CUdeviceptr)s->dag.dag();
    CUdeviceptr dres = (CUdeviceptr)s->d_result;
    uint64_t start_nonce = start;
    uint64_t target = ~0ull;  // accept anything
    uint32_t hack_false = 0;
    // Launch a full warp (32 = 2 lane-groups) so the __shfl_sync(0xffffffff,...)
    // full-warp mask in the generated kernel matches the active threads.
    void* args[] = { &dhdr, &start_nonce, &ddag, &target, &dres, &hack_false };
    if (cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS)
        return 0;
    cuCtxSynchronize();

    ppmulti_result r;
    cudaMemcpy(&r, s->d_result, sizeof(r), cudaMemcpyDeviceToHost);
    if (!r.found) return 0;

    // Verify whichever lane's nonce the kernel reported against the host.
    const uint64_t nonce = start + r.nonce_lo;
    ethash::hash256 h256 = header_from_bytes(hdr);
    ethash::result hr = progpow::hash(host_ctx_for(s, height), height, h256, nonce);
    if (memcmp(r.mix, hr.mix_hash.bytes, 32) != 0 ||
        memcmp(r.final, hr.final_hash.bytes, 32) != 0)
        return 0;
    // Negative: a flipped header bit must change the final hash.
    h256.bytes[0] ^= 0x01;
    ethash::result hr2 = progpow::hash(host_ctx_for(s, height), height, h256, nonce);
    if (memcmp(hr2.final_hash.bytes, hr.final_hash.bytes, 32) == 0)
        return 0;
    return 1;
}

extern "C" int kawpow_core_search(void* h, const unsigned char* header32, uint64_t start_nonce,
    const unsigned char* target32, int height, uint32_t throughput,
    uint64_t* nonce_out, unsigned char* mix_out, unsigned char* final_out)
{
    core_state* s = static_cast<core_state*>(h);
    const uint64_t period = (uint64_t)height / 3u;
    const uint32_t items = s->dag.dataset_items_128() / 2u;  // 256-byte items
    CUfunction fn;
    if (!s->jit->get(period, items, &fn)) return -1;

    cudaMemcpy(s->d_header, header32, 32, cudaMemcpyHostToDevice);
    cudaMemset(s->d_result, 0, sizeof(ppmulti_result));

    const uint32_t tpb = 256;  // multiple of 16 (PROGPOW_LANES); 256 shares the
                               // 16 KB c_dag across more threads -> ~67% occ vs
                               // 50% at 128 (shared-mem-limited kernel)
    const uint32_t grid = (throughput + tpb - 1) / tpb;

    CUdeviceptr dhdr = (CUdeviceptr)s->d_header;
    CUdeviceptr ddag = (CUdeviceptr)s->dag.dag();
    CUdeviceptr dres = (CUdeviceptr)s->d_result;
    uint64_t start = start_nonce;
    uint64_t target64 = target_top64(target32);
    uint32_t hack_false = 0;
    void* args[] = { &dhdr, &start, &ddag, &target64, &dres, &hack_false };
    if (cuLaunchKernel(fn, grid, 1, 1, tpb, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS)
        return -1;
    cuCtxSynchronize();

    ppmulti_result r;
    cudaMemcpy(&r, s->d_result, sizeof(r), cudaMemcpyDeviceToHost);
    if (!r.found) return 0;

    const uint64_t nonce = start_nonce + r.nonce_lo;
    ethash::hash256 hh = header_from_bytes(header32);
    ethash::result hr = progpow::hash(host_ctx_for(s, height), height, hh, nonce);
    ethash::hash256 boundary;
    memcpy(boundary.bytes, target32, 32);
    // Authoritative full 256-bit target check on the freshly-recomputed hash
    // (the GPU's own final/mix are only trusted by the self-test). Using
    // is_less_or_equal here avoids progpow::verify recomputing the whole mix.
    if (!ethash::is_less_or_equal(hr.final_hash, boundary))
        return -2;  // GPU flagged but host rejected (local reject, never submitted)

    *nonce_out = nonce;
    memcpy(mix_out, hr.mix_hash.bytes, 32);
    memcpy(final_out, hr.final_hash.bytes, 32);
    return 1;
}
