// SPDX-License-Identifier: GPL-3.0-or-later
//
// Shared ProgPoW generator parts, used by algos/progpow_multi/ppmulti_jit.cpp to
// build the kernel for kawpow, meowpow, evrprogpow, firopow and meraki.
//
// What belongs here: anything that must be identical for every variant to emit
// the same program. The host RNG walk and the merge()/math() emitters decide
// WHICH ProgPoW program is generated, so an edit here changes the hash itself.
//
// What does not: anything per-variant. The PROGPOW_* #define block sits between
// device_preamble() and device_body() because it is derived from pp_params, as
// are the kernel entry, the keccak seal and mix_rng_state's register count.

#pragma once

#include <cstdint>
#include <string>

namespace progpow_gen {

// ---------------------------------------------------------------------------
// Host-side RNG. This is the walk that selects the period's program: it must
// consume the RNG in exactly progpow::mix_rng_state's order, in both callers.
// ---------------------------------------------------------------------------

inline uint32_t fnv1a(uint32_t& h, uint32_t d) { return h = (h ^ d) * 0x01000193u; }
struct kiss99_t { uint32_t z, w, jsr, jcong; };
inline uint32_t kiss99(kiss99_t& st)
{
    st.z = 36969 * (st.z & 0xffff) + (st.z >> 16);
    st.w = 18000 * (st.w & 0xffff) + (st.w >> 16);
    st.jcong = 69069 * st.jcong + 1234567;
    st.jsr ^= (st.jsr << 17);
    st.jsr ^= (st.jsr >> 13);
    st.jsr ^= (st.jsr << 5);
    return (((st.z << 16) + st.w) ^ st.jcong) + st.jsr;
}
// ---------------------------------------------------------------------------
// Operator emitters. Consensus-defined: the case numbering IS the protocol.
// ---------------------------------------------------------------------------

inline std::string merge(const std::string& a, const std::string& b, uint32_t r)
{
    const uint32_t x = ((r >> 16) % 31) + 1;
    switch (r % 4)
    {
    case 0: return a + " = (" + a + " * 33u) + " + b + ";\n";
    case 1: return a + " = (" + a + " ^ " + b + ") * 33u;\n";
    case 2: return a + " = ROTL32(" + a + ", " + std::to_string(x) + ") ^ " + b + ";\n";
    default: return a + " = ROTR32(" + a + ", " + std::to_string(x) + ") ^ " + b + ";\n";
    }
}
inline std::string math(const std::string& d, const std::string& a, const std::string& b, uint32_t r)
{
    switch (r % 11)
    {
    case 0: return d + " = " + a + " + " + b + ";\n";
    case 1: return d + " = " + a + " * " + b + ";\n";
    case 2: return d + " = mul_hi(" + a + ", " + b + ");\n";
    case 3: return d + " = min(" + a + ", " + b + ");\n";
    case 4: return d + " = ROTL32(" + a + ", (" + b + ") % 32u);\n";
    case 5: return d + " = ROTR32(" + a + ", (" + b + ") % 32u);\n";
    case 6: return d + " = " + a + " & " + b + ";\n";
    case 7: return d + " = " + a + " | " + b + ";\n";
    case 8: return d + " = " + a + " ^ " + b + ";\n";
    case 9: return d + " = clz(" + a + ") + clz(" + b + ");\n";
    default: return d + " = popcount(" + a + ") + popcount(" + b + ");\n";
    }
}
// ---------------------------------------------------------------------------
// Device source, part 1 of 2: typedefs and intrinsic macros. The caller emits
// its own PROGPOW_* #define block after this, then appends device_body().
// ---------------------------------------------------------------------------

inline const char* device_preamble()
{
    return R"CUDA(
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
// NOTE: do NOT typedef size_t here -- NVRTC's builtin header already declares it
// (as unsigned long). A conflicting typedef is only a warning under dynamically
// loaded NVRTC but a hard "invalid redeclaration" error under the statically
// linked NVRTC (which force-includes __nv_nvrtc_builtin_header.h). size_t is not
// used in the emitted device code anyway.

#define ROTL32(x,n) __funnelshift_l((x), (x), (n))
#define ROTR32(x,n) __funnelshift_r((x), (x), (n))
#define min(a,b) ((a) < (b) ? (a) : (b))
#define mul_hi(a,b) __umulhi((a), (b))
#define clz(a) __clz((a))
#define popcount(a) __popc((a))

)CUDA";
}

// ---------------------------------------------------------------------------
// Device source, part 2 of 2: dag_t and the primitives baked into every
// generated kernel -- rol32, cuda_swab32, fnv1a_d/FNV1A, kiss99, fill_mix, and
// keccak_f800 with its __constant__ round constants.
// ---------------------------------------------------------------------------

inline const char* device_body()
{
    return R"CUDA(typedef struct __align__(16) { uint32_t s[PROGPOW_DAG_LOADS]; } dag_t;

__device__ __forceinline__ uint32_t rol32(uint32_t x, unsigned s)
{
    return (x << s) | (x >> (32 - s));
}

__device__ __forceinline__ uint32_t cuda_swab32(uint32_t x)
{
    return __byte_perm(x, x, 0x0123);
}

__device__ __forceinline__ uint32_t fnv1a_d(uint32_t h, uint32_t d) { return (h ^ d) * 0x01000193u; }
#define FNV1A(h, d) ((h) = ((h) ^ (d)) * 0x01000193u)

typedef struct { uint32_t z, w, jsr, jcong; } kiss99_t;
__device__ uint32_t kiss99(kiss99_t* st)
{
    st->z = 36969 * (st->z & 0xffff) + (st->z >> 16);
    st->w = 18000 * (st->w & 0xffff) + (st->w >> 16);
    st->jcong = 69069 * st->jcong + 1234567;
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    return (((st->z << 16) + st->w) ^ st->jcong) + st->jsr;
}

// Expand the per-nonce keccak seed into this lane's 32-register mix state.
// Identical to progpow::init_mix's per-lane KISS99 seeding.
__device__ __forceinline__ void fill_mix(uint32_t seed0, uint32_t seed1, uint32_t lane_id, uint32_t* mix)
{
    const uint32_t z = fnv1a_d(FNV_OFFSET_BASIS, seed0);
    const uint32_t w = fnv1a_d(z, seed1);
    kiss99_t st;
    st.z = z; st.w = w;
    st.jsr = fnv1a_d(w, lane_id);
    st.jcong = fnv1a_d(st.jsr, lane_id);
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(&st);
}

// Round constants at file scope and in __constant__, NOT inside the function:
// an array declared inside keccak_f800 and indexed by the loop variable is
// materialised in LOCAL memory (88-byte stack frame, 22 STL + 2 LDL per call,
// and the function is called twice per nonce). Same form as cuda_kawpow.cu's
// kF800RC -- keep the two in step.
__constant__ static const uint32_t kF800RC[22] = {
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080 };

__device__ void keccak_f800(uint32_t state[25])
{
    uint32_t Aba, Abe, Abi, Abo, Abu, Aga, Age, Agi, Ago, Agu;
    uint32_t Aka, Ake, Aki, Ako, Aku, Ama, Ame, Ami, Amo, Amu;
    uint32_t Asa, Ase, Asi, Aso, Asu;
    uint32_t Eba, Ebe, Ebi, Ebo, Ebu, Ega, Ege, Egi, Ego, Egu;
    uint32_t Eka, Eke, Eki, Eko, Eku, Ema, Eme, Emi, Emo, Emu;
    uint32_t Esa, Ese, Esi, Eso, Esu;
    uint32_t Ba, Be, Bi, Bo, Bu, Da, De, Di, Do, Du;
    Aba = state[0];  Abe = state[1];  Abi = state[2];  Abo = state[3];  Abu = state[4];
    Aga = state[5];  Age = state[6];  Agi = state[7];  Ago = state[8];  Agu = state[9];
    Aka = state[10]; Ake = state[11]; Aki = state[12]; Ako = state[13]; Aku = state[14];
    Ama = state[15]; Ame = state[16]; Ami = state[17]; Amo = state[18]; Amu = state[19];
    Asa = state[20]; Ase = state[21]; Asi = state[22]; Aso = state[23]; Asu = state[24];
    for (int round = 0; round < 22; round += 2) {
        Ba = Aba ^ Aga ^ Aka ^ Ama ^ Asa; Be = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        Bi = Abi ^ Agi ^ Aki ^ Ami ^ Asi; Bo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        Bu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;
        Da = Bu ^ rol32(Be, 1); De = Ba ^ rol32(Bi, 1); Di = Be ^ rol32(Bo, 1);
        Do = Bi ^ rol32(Bu, 1); Du = Bo ^ rol32(Ba, 1);
        Ba = Aba ^ Da; Be = rol32(Age ^ De, 12); Bi = rol32(Aki ^ Di, 11);
        Bo = rol32(Amo ^ Do, 21); Bu = rol32(Asu ^ Du, 14);
        Eba = Ba ^ (~Be & Bi) ^ kF800RC[round]; Ebe = Be ^ (~Bi & Bo);
        Ebi = Bi ^ (~Bo & Bu); Ebo = Bo ^ (~Bu & Ba); Ebu = Bu ^ (~Ba & Be);
        Ba = rol32(Abo ^ Do, 28); Be = rol32(Agu ^ Du, 20); Bi = rol32(Aka ^ Da, 3);
        Bo = rol32(Ame ^ De, 13); Bu = rol32(Asi ^ Di, 29);
        Ega = Ba ^ (~Be & Bi); Ege = Be ^ (~Bi & Bo); Egi = Bi ^ (~Bo & Bu);
        Ego = Bo ^ (~Bu & Ba); Egu = Bu ^ (~Ba & Be);
        Ba = rol32(Abe ^ De, 1); Be = rol32(Agi ^ Di, 6); Bi = rol32(Ako ^ Do, 25);
        Bo = rol32(Amu ^ Du, 8); Bu = rol32(Asa ^ Da, 18);
        Eka = Ba ^ (~Be & Bi); Eke = Be ^ (~Bi & Bo); Eki = Bi ^ (~Bo & Bu);
        Eko = Bo ^ (~Bu & Ba); Eku = Bu ^ (~Ba & Be);
        Ba = rol32(Abu ^ Du, 27); Be = rol32(Aga ^ Da, 4); Bi = rol32(Ake ^ De, 10);
        Bo = rol32(Ami ^ Di, 15); Bu = rol32(Aso ^ Do, 24);
        Ema = Ba ^ (~Be & Bi); Eme = Be ^ (~Bi & Bo); Emi = Bi ^ (~Bo & Bu);
        Emo = Bo ^ (~Bu & Ba); Emu = Bu ^ (~Ba & Be);
        Ba = rol32(Abi ^ Di, 30); Be = rol32(Ago ^ Do, 23); Bi = rol32(Aku ^ Du, 7);
        Bo = rol32(Ama ^ Da, 9); Bu = rol32(Ase ^ De, 2);
        Esa = Ba ^ (~Be & Bi); Ese = Be ^ (~Bi & Bo); Esi = Bi ^ (~Bo & Bu);
        Eso = Bo ^ (~Bu & Ba); Esu = Bu ^ (~Ba & Be);
        Ba = Eba ^ Ega ^ Eka ^ Ema ^ Esa; Be = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        Bi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi; Bo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        Bu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;
        Da = Bu ^ rol32(Be, 1); De = Ba ^ rol32(Bi, 1); Di = Be ^ rol32(Bo, 1);
        Do = Bi ^ rol32(Bu, 1); Du = Bo ^ rol32(Ba, 1);
        Ba = Eba ^ Da; Be = rol32(Ege ^ De, 12); Bi = rol32(Eki ^ Di, 11);
        Bo = rol32(Emo ^ Do, 21); Bu = rol32(Esu ^ Du, 14);
        Aba = Ba ^ (~Be & Bi) ^ kF800RC[round + 1]; Abe = Be ^ (~Bi & Bo);
        Abi = Bi ^ (~Bo & Bu); Abo = Bo ^ (~Bu & Ba); Abu = Bu ^ (~Ba & Be);
        Ba = rol32(Ebo ^ Do, 28); Be = rol32(Egu ^ Du, 20); Bi = rol32(Eka ^ Da, 3);
        Bo = rol32(Eme ^ De, 13); Bu = rol32(Esi ^ Di, 29);
        Aga = Ba ^ (~Be & Bi); Age = Be ^ (~Bi & Bo); Agi = Bi ^ (~Bo & Bu);
        Ago = Bo ^ (~Bu & Ba); Agu = Bu ^ (~Ba & Be);
        Ba = rol32(Ebe ^ De, 1); Be = rol32(Egi ^ Di, 6); Bi = rol32(Eko ^ Do, 25);
        Bo = rol32(Emu ^ Du, 8); Bu = rol32(Esa ^ Da, 18);
        Aka = Ba ^ (~Be & Bi); Ake = Be ^ (~Bi & Bo); Aki = Bi ^ (~Bo & Bu);
        Ako = Bo ^ (~Bu & Ba); Aku = Bu ^ (~Ba & Be);
        Ba = rol32(Ebu ^ Du, 27); Be = rol32(Ega ^ Da, 4); Bi = rol32(Eke ^ De, 10);
        Bo = rol32(Emi ^ Di, 15); Bu = rol32(Eso ^ Do, 24);
        Ama = Ba ^ (~Be & Bi); Ame = Be ^ (~Bi & Bo); Ami = Bi ^ (~Bo & Bu);
        Amo = Bo ^ (~Bu & Ba); Amu = Bu ^ (~Ba & Be);
        Ba = rol32(Ebi ^ Di, 30); Be = rol32(Ego ^ Do, 23); Bi = rol32(Eku ^ Du, 7);
        Bo = rol32(Ema ^ Da, 9); Bu = rol32(Ese ^ De, 2);
        Asa = Ba ^ (~Be & Bi); Ase = Be ^ (~Bi & Bo); Asi = Bi ^ (~Bo & Bu);
        Aso = Bo ^ (~Bu & Ba); Asu = Bu ^ (~Ba & Be);
    }
    state[0] = Aba;  state[1] = Abe;  state[2] = Abi;  state[3] = Abo;  state[4] = Abu;
    state[5] = Aga;  state[6] = Age;  state[7] = Agi;  state[8] = Ago;  state[9] = Agu;
    state[10] = Aka; state[11] = Ake; state[12] = Aki; state[13] = Ako; state[14] = Aku;
    state[15] = Ama; state[16] = Ame; state[17] = Ami; state[18] = Amo; state[19] = Amu;
    state[20] = Asa; state[21] = Ase; state[22] = Asi; state[23] = Aso; state[24] = Asu;
}
)CUDA";
}

} // namespace progpow_gen
