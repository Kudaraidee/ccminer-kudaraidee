// SPDX-License-Identifier: GPL-3.0-or-later
//
// ProgPoW-family core (see ppmulti_core.h). The ethash/C++-STL side of the port:
// epoch/DAG state machine, per-period NVRTC kernel, device buffers, and the host
// ProgPoW reference (progpow_pp) used to re-verify every GPU candidate before it
// is reported. Mirrors algos/kawpow/kawpow_core.cpp; the variant is selected by
// the pp_params passed to ppmulti_core_create. No ccminer headers here.

#include "ppmulti_core.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include "ppmulti_jit.h"
#include "ppmulti_dag.h"
#include "ppmulti_epoch.h"
#include "progpow_pp.hpp"

#include "../kawpow/ethash/ethash.hpp"
#include "../kawpow/ethash/ethash.h"
#include "../kawpow/ethash/ethash-internal.hpp"  // ethash::is_less_or_equal

#include <cstdio>
#include <cstring>

namespace {

struct core_state
{
    pp_params params;
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

static uint64_t target_top64(const unsigned char* t)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | t[i];
    return v;
}

// Cached host epoch context (light path) for reverify + self-test.
const ethash::epoch_context& host_ctx_for(core_state* s, int epoch)
{
    if (!s->host_ctx || s->host_epoch != epoch) {
        if (s->host_ctx)
            ethash_destroy_epoch_context(const_cast<ethash_epoch_context*>(s->host_ctx));
        // Same per-variant seed/light/full epochs as the GPU DAG (ppmulti_dag), so
        // host re-verification matches the kernel exactly.
        int se, le, fe;
        pp_dag_epochs(s->params, epoch, &se, &le, &fe);
        s->host_ctx = pp_create_epoch_context(se, le, fe);
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

extern "C" void* ppmulti_core_create(int sm_arch, const pp_params* p)
{
    core_state* s = new core_state();
    s->params = *p;
    s->sm_arch = sm_arch;
    s->jit = new ppmulti_jit(sm_arch, s->params);
    if (cudaMalloc(&s->d_header, 32) != cudaSuccess) { delete s; return nullptr; }
    if (cudaMalloc(&s->d_result, sizeof(ppmulti_result)) != cudaSuccess) { delete s; return nullptr; }
    return s;
}

extern "C" void ppmulti_core_destroy(void* h)
{
    if (h) delete static_cast<core_state*>(h);
}

extern "C" int ppmulti_core_ensure(void* h, int height, int* regenerated)
{
    core_state* s = static_cast<core_state*>(h);
    const int epoch = height / s->params.epoch_length;
    int se, le, fe;
    pp_dag_epochs(s->params, epoch, &se, &le, &fe);
    bool re = false;
    bool ok = s->dag.ensure(se, le, fe, &re);
    if (regenerated) *regenerated = re ? 1 : 0;
    return ok ? 1 : 0;
}

extern "C" int ppmulti_core_selftest(void* h, int height)
{
    core_state* s = static_cast<core_state*>(h);
    const int epoch = height / s->params.epoch_length;
    const uint64_t period = (uint64_t)height / (uint64_t)s->params.period_length;
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
    void* args[] = { &dhdr, &start_nonce, &ddag, &target, &dres, &hack_false };
    if (cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS)
        return 0;
    cuCtxSynchronize();

    ppmulti_result r;
    cudaMemcpy(&r, s->d_result, sizeof(r), cudaMemcpyDeviceToHost);
    if (!r.found) return 0;

    const uint64_t nonce = start + r.nonce_lo;
    ethash::hash256 h256 = header_from_bytes(hdr);
    ethash::result hr = progpow_pp::hash(host_ctx_for(s, epoch), s->params, height, h256, nonce);
    if (memcmp(r.mix, hr.mix_hash.bytes, 32) != 0 ||
        memcmp(r.final, hr.final_hash.bytes, 32) != 0)
        return 0;
    // Negative: a flipped header bit must change the final hash.
    h256.bytes[0] ^= 0x01;
    ethash::result hr2 = progpow_pp::hash(host_ctx_for(s, epoch), s->params, height, h256, nonce);
    if (memcmp(hr2.final_hash.bytes, hr.final_hash.bytes, 32) == 0)
        return 0;
    return 1;
}

extern "C" int ppmulti_core_search(void* h, const unsigned char* header32, uint64_t start_nonce,
    const unsigned char* target32, int height, uint32_t throughput,
    uint64_t* nonce_out, unsigned char* mix_out, unsigned char* final_out)
{
    core_state* s = static_cast<core_state*>(h);
    const int epoch = height / s->params.epoch_length;
    const uint64_t period = (uint64_t)height / (uint64_t)s->params.period_length;
    const uint32_t items = s->dag.dataset_items_128() / 2u;  // 256-byte items
    CUfunction fn;
    if (!s->jit->get(period, items, &fn)) return -1;

    cudaMemcpy(s->d_header, header32, 32, cudaMemcpyHostToDevice);
    cudaMemset(s->d_result, 0, sizeof(ppmulti_result));

    const uint32_t tpb = 256;  // multiple of 16 (PROGPOW_LANES)
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
    ethash::result hr = progpow_pp::hash(host_ctx_for(s, epoch), s->params, height, hh, nonce);
    ethash::hash256 boundary;
    memcpy(boundary.bytes, target32, 32);
    if (!ethash::is_less_or_equal(hr.final_hash, boundary))
        return -2;  // GPU flagged but host rejected (local reject, never submitted)

    *nonce_out = nonce;
    memcpy(mix_out, hr.mix_hash.bytes, 32);
    memcpy(final_out, hr.final_hash.bytes, 32);
    return 1;
}
