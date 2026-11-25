// SPDX-License-Identifier: GPL-3.0-or-later
//
// Parameterized ProgPoW per-period JIT (see ppmulti_jit.h). The register, cache
// and math counts and the keccak seal are emitted from pp_params, so this one
// generator serves kawpow, meowpow, evrprogpow, firopow and meraki.
//
// The host-side RNG walk mirrors progpow_pp::mix_rng_state (progpow_pp.cpp),
// consuming the KISS99 stream in the exact order of round(), so the baked
// register indices and selectors match the reference bit-for-bit for whichever
// variant is being generated.
//
// The parts that must be identical for every variant live in
// ../kawpow/progpow_gen_shared.h: the RNG walk, the merge() and math() emitters,
// and the device primitives baked into each kernel (keccak_f800 and its
// __constant__ round constants, kiss99, fnv1a_d, fill_mix). Do not re-declare
// them here -- they decide WHICH program is generated, so a local copy that
// drifts changes the hash rather than the performance.
//
// Genuinely per-variant, and so not shared: the PROGPOW_* #define block, the
// kernel entry, the keccak seal, and mix_rng_state's register count.

#include "ppmulti_jit.h"
#include "../kawpow/progpow_gen_shared.h"

#include <cstdio>
#include <sstream>
#include <vector>
#include <nvrtc.h>

namespace {

// Pulled in so the call sites below stay unqualified.
using progpow_gen::fnv1a;
using progpow_gen::kiss99;
using progpow_gen::kiss99_t;
using progpow_gen::math;
using progpow_gen::merge;

constexpr uint32_t PP_MAX_REGS = 32;
constexpr uint32_t PP_LANES = 16;
constexpr int PP_DAG_LOADS = 4;




// Host mirror of progpow_pp::mix_rng_state (runtime register count).
struct mix_rng_state
{
    kiss99_t rng;
    uint32_t num_regs;
    uint32_t dst_seq[PP_MAX_REGS];
    uint32_t src_seq[PP_MAX_REGS];
    uint32_t dst_cnt = 0;
    uint32_t src_cnt = 0;

    mix_rng_state(uint64_t period, uint32_t regs) : num_regs(regs)
    {
        const uint32_t seed0 = (uint32_t)period;
        const uint32_t seed1 = (uint32_t)(period >> 32);
        uint32_t h = 0x811c9dc5u;
        const uint32_t z = fnv1a(h, seed0);
        const uint32_t w = fnv1a(h, seed1);
        const uint32_t jsr = fnv1a(h, seed0);
        const uint32_t jcong = fnv1a(h, seed1);
        rng = kiss99_t{z, w, jsr, jcong};

        for (uint32_t i = 0; i < num_regs; ++i) { dst_seq[i] = i; src_seq[i] = i; }
        for (uint32_t i = num_regs; i > 1; --i)
        {
            uint32_t j = kiss99(rng) % i;
            uint32_t t = dst_seq[i - 1]; dst_seq[i - 1] = dst_seq[j]; dst_seq[j] = t;
            j = kiss99(rng) % i;
            t = src_seq[i - 1]; src_seq[i - 1] = src_seq[j]; src_seq[j] = t;
        }
    }

    uint32_t next_dst() { return dst_seq[(dst_cnt++) % num_regs]; }
    uint32_t next_src() { return src_seq[(src_cnt++) % num_regs]; }
    uint32_t next_rng() { return kiss99(rng); }
};

// Baked merge: a = f(a, b). Mirrors progpow_pp::random_merge.

// Baked math: d = g(a, b). Mirrors progpow_pp::random_math.

inline std::string lane(uint32_t reg) { return "mix[" + std::to_string(reg) + "]"; }

// Period-specialized progPowLoop (one round; run 64x at runtime). RNG walk is
// identical to progpow_pp::round() for the given register/cache/math counts.
std::string gen_progpowloop(const pp_params& p, uint64_t period)
{
    mix_rng_state st(period, (uint32_t)p.num_regs);
    std::stringstream s;

    s << "__device__ __forceinline__ void progPowLoop(const uint32_t loop,\n"
      << "        uint32_t* mix,\n"
      << "        const dag_t* __restrict__ g_dag,\n"
      << "        const uint32_t* __restrict__ c_dag,\n"
      << "        const uint32_t hack_false)\n{\n";
    s << "    dag_t data_dag;\n";
    s << "    uint32_t offset, data;\n";
    s << "    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);\n";
    s << "    offset = __shfl_sync(0xFFFFFFFFu, mix[0], loop % PROGPOW_LANES, PROGPOW_LANES);\n";
    s << "    offset %= PROGPOW_DAG_ELEMENTS;\n";
    s << "    offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;\n";
    s << "    data_dag = g_dag[offset];\n";
    s << "    if (hack_false) __threadfence_block();\n";

    const int max_ops = p.cnt_cache > p.cnt_math ? p.cnt_cache : p.cnt_math;
    for (int i = 0; i < max_ops; ++i)
    {
        if (i < p.cnt_cache)
        {
            const uint32_t src = st.next_src();
            const uint32_t dst = st.next_dst();
            const uint32_t sel = st.next_rng();
            s << "    // cache " << i << "\n";
            s << "    data = c_dag[" << lane(src) << " % PROGPOW_CACHE_WORDS];\n";
            s << "    " << merge(lane(dst), "data", sel);
        }
        if (i < p.cnt_math)
        {
            const uint32_t src_rnd = st.next_rng() % ((uint32_t)p.num_regs * ((uint32_t)p.num_regs - 1));
            const uint32_t src1 = src_rnd % (uint32_t)p.num_regs;
            uint32_t src2 = src_rnd / (uint32_t)p.num_regs;
            if (src2 >= src1) ++src2;
            const uint32_t sel1 = st.next_rng();
            const uint32_t dst = st.next_dst();
            const uint32_t sel2 = st.next_rng();
            s << "    // math " << i << "\n";
            s << "    " << math("data", lane(src1), lane(src2), sel1);
            s << "    " << merge(lane(dst), "data", sel2);
        }
    }

    s << "    if (hack_false) __threadfence_block();\n";
    for (int i = 0; i < PP_DAG_LOADS; ++i)
    {
        const uint32_t dst = (i == 0) ? 0 : st.next_dst();
        const uint32_t sel = st.next_rng();
        const std::string src = "data_dag.s[" + std::to_string(i) + "]";
        s << "    " << merge(lane(dst), src, sel);
    }
    s << "}\n";
    return s.str();
}

// Emit the keccak seal snippet for the device kernel. `indent` prefixes each line.
// `off` is the state fill offset (10 = seed phase, 16 = final phase).
std::string gen_seal(const pp_params& p, const char* indent, int off)
{
    std::stringstream s;
    if (p.seal_mode == PP_SEAL_SEEDWORDS)
        s << indent << "for (int i = " << off << "; i < 25; ++i) state[i] = sw[i - " << off << "];\n";
    else  // PP_SEAL_VANILLA: zero the fill range, then set the two padding words.
    {
        s << indent << "for (int i = " << off << "; i < 25; ++i) state[i] = 0u;\n";
        if (off == 10)
            s << indent << "state[10] = 0x00000001u; state[18] = 0x80008081u;\n";
        else
            s << indent << "state[17] = 0x00000001u; state[24] = 0x80008081u;\n";
    }
    return s.str();
}

// Warp-cooperative search entry (16 threads share a nonce). Structure follows
// kawpow_search; the keccak seal is emitted per variant.
std::string gen_device_entry(const pp_params& p)
{
    std::stringstream s;
    s << "struct ppmulti_result { uint32_t found; uint32_t nonce_lo; uint32_t mix[8]; uint32_t final[8]; };\n\n";
    s << "extern \"C\" __global__ void progpow_search(\n"
      << "    const uint32_t* __restrict__ header, uint64_t start_nonce,\n"
      << "    const dag_t* __restrict__ g_dag,\n"
      << "    uint64_t target, ppmulti_result* __restrict__ result, uint32_t hack_false)\n{\n";
    s << "    __shared__ uint32_t c_dag[PROGPOW_CACHE_WORDS];\n";
    s << "    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    s << "    const uint64_t nonce = start_nonce + gid;\n";
    s << "    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);\n";

    if (p.seal_mode == PP_SEAL_SEEDWORDS)
    {
        s << "    const uint32_t sw[15] = {";
        for (int i = 0; i < 15; ++i)
        {
            char buf[16];
            snprintf(buf, sizeof(buf), "0x%08Xu", p.seed_words[i]);
            s << (i ? "," : "") << buf;
        }
        s << "};\n";
    }

    s << "    for (uint32_t word = threadIdx.x * PROGPOW_DAG_LOADS; word < PROGPOW_CACHE_WORDS;\n"
      << "         word += blockDim.x * PROGPOW_DAG_LOADS) {\n"
      << "        dag_t load = g_dag[word / PROGPOW_DAG_LOADS];\n"
      << "        #pragma unroll\n"
      << "        for (int i = 0; i < PROGPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];\n"
      << "    }\n"
      << "    __syncthreads();\n\n";

    // Initial keccak seed for this thread's own nonce.
    s << "    uint32_t state2[8];\n";
    s << "    {\n";
    s << "        uint32_t state[25];\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < 8; ++i) state[i] = header[i];\n";
    s << "        state[8] = (uint32_t)nonce;\n";
    s << "        state[9] = (uint32_t)(nonce >> 32);\n";
    s << gen_seal(p, "        ", 10);
    s << "        keccak_f800(state);\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < 8; ++i) state2[i] = state[i];\n";
    s << "    }\n\n";

    s << "    uint32_t digest[8];\n";
    s << "    #pragma unroll 1\n";
    s << "    for (uint32_t h = 0; h < PROGPOW_LANES; ++h) {\n";
    s << "        uint32_t mix[PROGPOW_REGS];\n";
    s << "        const uint32_t s0 = __shfl_sync(0xFFFFFFFFu, state2[0], h, PROGPOW_LANES);\n";
    s << "        const uint32_t s1 = __shfl_sync(0xFFFFFFFFu, state2[1], h, PROGPOW_LANES);\n";
    s << "        fill_mix(s0, s1, lane_id, mix);\n\n";
    s << "        #pragma unroll 1\n";
    s << "        for (uint32_t loop = 0; loop < PROGPOW_CNT_DAG; ++loop)\n";
    s << "            progPowLoop(loop, mix, g_dag, c_dag, hack_false);\n\n";
    s << "        uint32_t digest_lane = FNV_OFFSET_BASIS;\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < PROGPOW_REGS; ++i) FNV1A(digest_lane, mix[i]);\n\n";
    s << "        uint32_t dt[8];\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < 8; ++i) dt[i] = FNV_OFFSET_BASIS;\n";
    s << "        for (int i = 0; i < PROGPOW_LANES; i += 8)\n";
    s << "            #pragma unroll\n";
    s << "            for (int j = 0; j < 8; ++j)\n";
    s << "                FNV1A(dt[j], __shfl_sync(0xFFFFFFFFu, digest_lane, i + j, PROGPOW_LANES));\n\n";
    s << "        if (h == lane_id)\n";
    s << "            #pragma unroll\n";
    s << "            for (int i = 0; i < 8; ++i) digest[i] = dt[i];\n";
    s << "    }\n\n";

    // Final keccak for this thread's nonce.
    s << "    uint32_t state[25];\n";
    s << "    #pragma unroll\n";
    s << "    for (int i = 0; i < 8; ++i) state[i] = state2[i];\n";
    s << "    #pragma unroll\n";
    s << "    for (int i = 8; i < 16; ++i) state[i] = digest[i - 8];\n";
    s << gen_seal(p, "    ", 16);
    s << "    keccak_f800(state);\n\n";
    s << "    const uint64_t res = ((uint64_t)cuda_swab32(state[0]) << 32) | cuda_swab32(state[1]);\n";
    s << "    if (res > target) return;\n\n";
    s << "    if (atomicExch(&result->found, 1u) == 0u) {\n";
    s << "        result->nonce_lo = gid;\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < 8; ++i) { result->mix[i] = digest[i]; result->final[i] = state[i]; }\n";
    s << "    }\n}\n";
    return s.str();
}

} // anonymous namespace

std::string ppmulti_progpow_source(const pp_params& p, uint64_t period, uint32_t num_items)
{
    std::stringstream s;
    s << "#define PROGPOW_DAG_ELEMENTS " << num_items << "u\n";
    s << "#define PROGPOW_LANES        " << PP_LANES << "\n";
    s << "#define PROGPOW_REGS         " << p.num_regs << "\n";
    s << "#define PROGPOW_CNT_DAG      " << p.cnt_dag << "\n";
    s << "#define PROGPOW_DAG_LOADS    " << PP_DAG_LOADS << "\n";
    s << "#define PROGPOW_CACHE_WORDS  4096\n";
    s << "#define FNV_OFFSET_BASIS     0x811c9dc5u\n";
    s << progpow_gen::device_preamble();
    s << progpow_gen::device_body();
    s << "\n// ProgPoW program (" << (p.name ? p.name : "progpow") << ") for period " << period << "\n";
    s << gen_progpowloop(p, period);
    s << gen_device_entry(p);
    return s.str();
}

// ---- JIT compile + cache ----------------------------------------------------

ppmulti_jit::~ppmulti_jit()
{
    if (cached_module_)
        cuModuleUnload(cached_module_);
}

bool ppmulti_jit::get(uint64_t period, uint32_t num_items, CUfunction* fn)
{
    if (cached_module_ && cached_period_ == period)
    {
        *fn = cached_fn_;
        return true;
    }

    const std::string src = ppmulti_progpow_source(params_, period, num_items);
    ++compiles_;

    nvrtcProgram prog;
    nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "progpow.cu", 0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS)
    {
        fprintf(stderr, "ppmulti_jit: nvrtcCreateProgram: %s\n", nvrtcGetErrorString(nr));
        return false;
    }

    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d", sm_arch_);
    const char* opts[] = { arch_opt, "--std=c++14" };
    nr = nvrtcCompileProgram(prog, 2, opts);
    if (nr != NVRTC_SUCCESS)
    {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size ? log_size : 1);
        nvrtcGetProgramLog(prog, log.data());
        fprintf(stderr, "ppmulti_jit: nvrtcCompileProgram failed:\n%s\n", log.data());
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptx_size = 0;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    CUmodule mod = nullptr;
    CUresult cr = cuModuleLoadData(&mod, ptx.data());
    if (cr != CUDA_SUCCESS)
    {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        fprintf(stderr, "ppmulti_jit: cuModuleLoadData: %s\n", es ? es : "?");
        return false;
    }
    CUfunction f = nullptr;
    cr = cuModuleGetFunction(&f, mod, "progpow_search");
    if (cr != CUDA_SUCCESS)
    {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        fprintf(stderr, "ppmulti_jit: cuModuleGetFunction: %s\n", es ? es : "?");
        cuModuleUnload(mod);
        return false;
    }

    if (cached_module_)
        cuModuleUnload(cached_module_);
    cached_module_ = mod;
    cached_fn_ = f;
    cached_period_ = period;
    *fn = f;
    return true;
}
