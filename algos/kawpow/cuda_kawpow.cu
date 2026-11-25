// SPDX-License-Identifier: GPL-3.0-or-later
//
// KawPoW (Ravencoin) CUDA device code.
// Provenance and design: algos/kawpow/README.md.
//
// Phase 2: Ethash DAG generation on the GPU. One thread builds one 64-byte DAG
// node, bit-for-bit identical to the host reference
// ethash::calculate_dataset_item_512() (algos/kawpow/ethash/ethash.cpp).
// Ported from xmrig's kawpow_dag.cl (one-thread-per-node variant).

#include <cstdint>
#include <cuda_runtime.h>

namespace {

// --- Keccak-f[1600], ethash "SHA3-512" flavour (64-byte in, 64-byte out) -----
// Message occupies lanes 0..7; lane 8 carries the 0x01 pad start and the 0x80
// final bit at the top of the 576-bit (72-byte, 9-lane) rate. Matches the
// keccak512 used by build_light_cache / calculate_dataset_item_512.

__constant__ static const uint64_t kKeccakRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ __forceinline__ uint64_t rol64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

// In-place Keccak-f[1600] permutation over 25 lanes.
__device__ void keccak_f1600(uint64_t s[25])
{
    #pragma unroll 1
    for (int round = 0; round < 24; ++round)
    {
        uint64_t c[5], d[5], t;

        // Theta
        c[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        c[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        c[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        c[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        c[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];
        d[0] = c[4] ^ rol64(c[1], 1);
        d[1] = c[0] ^ rol64(c[2], 1);
        d[2] = c[1] ^ rol64(c[3], 1);
        d[3] = c[2] ^ rol64(c[4], 1);
        d[4] = c[3] ^ rol64(c[0], 1);
        #pragma unroll
        for (int i = 0; i < 25; i += 5)
        {
            s[i + 0] ^= d[0];
            s[i + 1] ^= d[1];
            s[i + 2] ^= d[2];
            s[i + 3] ^= d[3];
            s[i + 4] ^= d[4];
        }

        // Rho + Pi
        t = s[1];
        uint64_t b;
        b = s[10]; s[10] = rol64(t, 1);  t = b;
        b = s[7];  s[7]  = rol64(t, 3);  t = b;
        b = s[11]; s[11] = rol64(t, 6);  t = b;
        b = s[17]; s[17] = rol64(t, 10); t = b;
        b = s[18]; s[18] = rol64(t, 15); t = b;
        b = s[3];  s[3]  = rol64(t, 21); t = b;
        b = s[5];  s[5]  = rol64(t, 28); t = b;
        b = s[16]; s[16] = rol64(t, 36); t = b;
        b = s[8];  s[8]  = rol64(t, 45); t = b;
        b = s[21]; s[21] = rol64(t, 55); t = b;
        b = s[24]; s[24] = rol64(t, 2);  t = b;
        b = s[4];  s[4]  = rol64(t, 14); t = b;
        b = s[15]; s[15] = rol64(t, 27); t = b;
        b = s[23]; s[23] = rol64(t, 41); t = b;
        b = s[19]; s[19] = rol64(t, 56); t = b;
        b = s[13]; s[13] = rol64(t, 8);  t = b;
        b = s[12]; s[12] = rol64(t, 25); t = b;
        b = s[2];  s[2]  = rol64(t, 43); t = b;
        b = s[20]; s[20] = rol64(t, 62); t = b;
        b = s[14]; s[14] = rol64(t, 18); t = b;
        b = s[22]; s[22] = rol64(t, 39); t = b;
        b = s[9];  s[9]  = rol64(t, 61); t = b;
        b = s[6];  s[6]  = rol64(t, 20); t = b;
                   s[1]  = rol64(t, 44);

        // Chi
        #pragma unroll
        for (int j = 0; j < 25; j += 5)
        {
            uint64_t a0 = s[j + 0], a1 = s[j + 1], a2 = s[j + 2], a3 = s[j + 3], a4 = s[j + 4];
            s[j + 0] = a0 ^ ((~a1) & a2);
            s[j + 1] = a1 ^ ((~a2) & a3);
            s[j + 2] = a2 ^ ((~a3) & a4);
            s[j + 3] = a3 ^ ((~a4) & a0);
            s[j + 4] = a4 ^ ((~a0) & a1);
        }

        // Iota
        s[0] ^= kKeccakRC[round];
    }
}

// 64-byte -> 64-byte ethash keccak512. in/out are 8 lanes (16 x uint32).
__device__ __forceinline__ void sha3_512_64(uint64_t out[8], const uint64_t in[8])
{
    uint64_t s[25];
    #pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = in[i];
    s[8] = 0x8000000000000001ULL;  // 0x01 pad start + 0x80 final bit @ rate end
    #pragma unroll
    for (int i = 9; i < 25; ++i) s[i] = 0;
    keccak_f1600(s);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = s[i];
}

#define FNV_PRIME 0x01000193u

__device__ __forceinline__ uint32_t fnv1(uint32_t u, uint32_t v)
{
    return (u * FNV_PRIME) ^ v;
}

} // anonymous namespace

// One 64-byte DAG node per thread. g_light / g_dag are arrays of 16 x uint32
// (64-byte) nodes. Mirrors ethash::calculate_dataset_item_512(index).
extern "C" __global__ void kawpow_gpu_calculate_dag_item(
    uint32_t start, const uint32_t* __restrict__ g_light, uint32_t* __restrict__ g_dag,
    uint32_t dag_nodes, uint32_t light_nodes)
{
    const uint32_t node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (node_index >= dag_nodes)
        return;

    // mix = light[index % light_nodes]; mix.word32s[0] ^= index; mix = keccak512(mix)
    uint32_t mix[16];
    const uint32_t* src = g_light + (size_t)(node_index % light_nodes) * 16u;
    #pragma unroll
    for (int i = 0; i < 16; ++i) mix[i] = src[i];
    mix[0] ^= node_index;

    uint64_t* mix64 = reinterpret_cast<uint64_t*>(mix);
    sha3_512_64(mix64, mix64);

    // 512 parent mixes.
    #pragma unroll 1
    for (uint32_t j = 0; j < 512; ++j)
    {
        const uint32_t t = fnv1(node_index ^ j, mix[j % 16]);
        const uint32_t parent = t % light_nodes;
        const uint32_t* p = g_light + (size_t)parent * 16u;
        #pragma unroll
        for (int w = 0; w < 16; ++w)
            mix[w] = fnv1(mix[w], p[w]);
    }

    sha3_512_64(mix64, mix64);

    uint32_t* dst = g_dag + (size_t)node_index * 16u;
    #pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = mix[i];
}

// Host launcher: fill the whole DAG (dag_nodes 64-byte nodes) in chunks. Blocks
// until done. Returns the last CUDA error.
extern "C" cudaError_t kawpow_generate_dag_cpu(
    const uint32_t* d_light, uint32_t* d_dag, uint32_t dag_nodes, uint32_t light_nodes)
{
    const int tpb = 256;
    const uint32_t chunk = tpb * 4096u;  // ~1M nodes per launch
    for (uint32_t base = 0; base < dag_nodes; base += chunk)
    {
        const uint32_t n = (dag_nodes - base < chunk) ? (dag_nodes - base) : chunk;
        const uint32_t grid = (n + tpb - 1) / tpb;
        kawpow_gpu_calculate_dag_item<<<grid, tpb>>>(base, d_light, d_dag, dag_nodes, light_nodes);
    }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    return e;
}

// ============================================================================
// Phase 3: ProgPoW main hash (KawPoW). Single thread computes one nonce, with
// all 16 lanes held in local arrays. Mirrors progpow::hash() in
// algos/kawpow/ethash/progpow.cpp bit-for-bit; device is little-endian so the
// host's le::uint32 conversions are identities and dropped. Warp-cooperative
// 16-lane execution is a later optimization (plan Phase 7).
// ============================================================================

namespace {

constexpr uint32_t PP_NUM_REGS = 32;
constexpr uint32_t PP_NUM_LANES = 16;
constexpr int PP_CACHE_ACCESSES = 11;
constexpr int PP_MATH_OPS = 18;
constexpr uint32_t PP_L1_ITEMS = 16 * 1024 / 4;  // 4096
constexpr uint32_t PP_PERIOD_LENGTH = 3;
constexpr uint32_t FNV_OFFSET_BASIS = 0x811c9dc5u;

__device__ __forceinline__ uint32_t fnv1a(uint32_t u, uint32_t v)
{
    return (u ^ v) * FNV_PRIME;
}

// --- Keccak-f[800] (ProgPoW seed + finalize), verbatim from ethash_keccakf800.
__constant__ static const uint32_t kF800RC[22] = {
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080,
};

__device__ __forceinline__ uint32_t rol32(uint32_t x, unsigned s)
{
    return (x << s) | (x >> (32 - s));
}

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

    for (int round = 0; round < 22; round += 2)
    {
        Ba = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        Be = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        Bi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        Bo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        Bu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;
        Da = Bu ^ rol32(Be, 1); De = Ba ^ rol32(Bi, 1); Di = Be ^ rol32(Bo, 1);
        Do = Bi ^ rol32(Bu, 1); Du = Bo ^ rol32(Ba, 1);

        Ba = Aba ^ Da;         Be = rol32(Age ^ De, 12); Bi = rol32(Aki ^ Di, 11);
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

        Ba = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        Be = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        Bi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        Bo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        Bu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;
        Da = Bu ^ rol32(Be, 1); De = Ba ^ rol32(Bi, 1); Di = Be ^ rol32(Bo, 1);
        Do = Bi ^ rol32(Bu, 1); Du = Bo ^ rol32(Ba, 1);

        Ba = Eba ^ Da;         Be = rol32(Ege ^ De, 12); Bi = rol32(Eki ^ Di, 11);
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

// ravencoin_kawpow padding constants (progpow.cpp).
__constant__ static const uint32_t kRvnPad[15] = {
    0x00000072, 0x00000041, 0x00000056, 0x00000045, 0x0000004E,
    0x00000043, 0x0000004F, 0x00000049, 0x0000004E, 0x0000004B,
    0x00000041, 0x00000057, 0x00000050, 0x0000004F, 0x00000057,
};

// --- KISS99 -----------------------------------------------------------------
struct kiss99_t
{
    uint32_t z, w, jsr, jcong;
    __device__ __forceinline__ uint32_t next()
    {
        z = 36969 * (z & 0xffff) + (z >> 16);
        w = 18000 * (w & 0xffff) + (w >> 16);
        jcong = 69069 * jcong + 1234567;
        jsr ^= (jsr << 17);
        jsr ^= (jsr >> 13);
        jsr ^= (jsr << 5);
        return (((z << 16) + w) ^ jcong) + jsr;
    }
};

// mix_rng_state: KISS99 + Fisher-Yates permutations of src/dst register indices.
struct mix_rng_state
{
    kiss99_t rng;
    uint32_t dst_seq[PP_NUM_REGS];
    uint32_t src_seq[PP_NUM_REGS];
    uint32_t dst_counter;
    uint32_t src_counter;

    __device__ mix_rng_state(const uint32_t seed[2])
    {
        const uint32_t z = fnv1a(FNV_OFFSET_BASIS, seed[0]);
        const uint32_t w = fnv1a(z, seed[1]);
        const uint32_t jsr = fnv1a(w, seed[0]);
        const uint32_t jcong = fnv1a(jsr, seed[1]);
        rng = kiss99_t{z, w, jsr, jcong};

        for (uint32_t i = 0; i < PP_NUM_REGS; ++i) { dst_seq[i] = i; src_seq[i] = i; }
        for (uint32_t i = PP_NUM_REGS; i > 1; --i)
        {
            uint32_t j = rng.next() % i;
            uint32_t t = dst_seq[i - 1]; dst_seq[i - 1] = dst_seq[j]; dst_seq[j] = t;
            j = rng.next() % i;
            t = src_seq[i - 1]; src_seq[i - 1] = src_seq[j]; src_seq[j] = t;
        }
        dst_counter = 0;
        src_counter = 0;
    }

    __device__ __forceinline__ uint32_t next_dst() { return dst_seq[(dst_counter++) % PP_NUM_REGS]; }
    __device__ __forceinline__ uint32_t next_src() { return src_seq[(src_counter++) % PP_NUM_REGS]; }
};

__device__ __forceinline__ uint32_t random_math(uint32_t a, uint32_t b, uint32_t sel)
{
    switch (sel % 11)
    {
    default:
    case 0: return a + b;
    case 1: return a * b;
    case 2: return __umulhi(a, b);
    case 3: return min(a, b);
    case 4: return (a << (b & 31)) | (a >> ((32 - (b & 31)) & 31));
    case 5: return (a >> (b & 31)) | (a << ((32 - (b & 31)) & 31));
    case 6: return a & b;
    case 7: return a | b;
    case 8: return a ^ b;
    case 9: return __clz(a) + __clz(b);
    case 10: return __popc(a) + __popc(b);
    }
}

__device__ __forceinline__ void random_merge(uint32_t& a, uint32_t b, uint32_t sel)
{
    const uint32_t x = (sel >> 16) % 31 + 1;
    switch (sel % 4)
    {
    case 0: a = (a * 33) + b; break;
    case 1: a = (a ^ b) * 33; break;
    case 2: a = rol32(a, x) ^ b; break;
    case 3: a = (a >> x | a << ((32 - x) & 31)) ^ b; break;
    }
}

// One ProgPoW round over all 16 lanes. mix is [16][32]. num_items counts
// 256-byte dataset items. dag is a flat array of 64-byte nodes (16 words each).
__device__ void pp_round(const uint32_t* __restrict__ dag, const uint32_t* __restrict__ l1,
    uint32_t num_items, uint32_t r, uint32_t mix[PP_NUM_LANES][PP_NUM_REGS], mix_rng_state state)
{
    const uint32_t item_index = mix[r % PP_NUM_LANES][0] % num_items;
    // 256-byte item = 4 consecutive 64-byte nodes at node offset item_index*4.
    const uint32_t* item = dag + (size_t)item_index * 64u;  // 64 words

    const int num_words_per_lane = 4;  // 256 / (4 * 16)
    const int max_ops = PP_CACHE_ACCESSES > PP_MATH_OPS ? PP_CACHE_ACCESSES : PP_MATH_OPS;

    for (int i = 0; i < max_ops; ++i)
    {
        if (i < PP_CACHE_ACCESSES)
        {
            const uint32_t src = state.next_src();
            const uint32_t dst = state.next_dst();
            const uint32_t sel = state.rng.next();
            for (uint32_t l = 0; l < PP_NUM_LANES; ++l)
            {
                const uint32_t offset = mix[l][src] % PP_L1_ITEMS;
                random_merge(mix[l][dst], l1[offset], sel);
            }
        }
        if (i < PP_MATH_OPS)
        {
            const uint32_t src_rnd = state.rng.next() % (PP_NUM_REGS * (PP_NUM_REGS - 1));
            const uint32_t src1 = src_rnd % PP_NUM_REGS;
            uint32_t src2 = src_rnd / PP_NUM_REGS;
            if (src2 >= src1) ++src2;
            const uint32_t sel1 = state.rng.next();
            const uint32_t dst = state.next_dst();
            const uint32_t sel2 = state.rng.next();
            for (uint32_t l = 0; l < PP_NUM_LANES; ++l)
            {
                const uint32_t data = random_math(mix[l][src1], mix[l][src2], sel1);
                random_merge(mix[l][dst], data, sel2);
            }
        }
    }

    uint32_t dsts[num_words_per_lane];
    uint32_t sels[num_words_per_lane];
    for (int i = 0; i < num_words_per_lane; ++i)
    {
        dsts[i] = (i == 0) ? 0 : state.next_dst();
        sels[i] = state.rng.next();
    }
    for (uint32_t l = 0; l < PP_NUM_LANES; ++l)
    {
        const uint32_t offset = ((l ^ r) % PP_NUM_LANES) * num_words_per_lane;
        for (int i = 0; i < num_words_per_lane; ++i)
            random_merge(mix[l][dsts[i]], item[offset + i], sels[i]);
    }
}

} // anonymous namespace

// Single thread computes one nonce's KawPoW hash. header = 8 words. dag = full
// DAG (flat 64-byte nodes). l1 = 4096-word L1 cache. dataset_items_128 =
// full_dataset_num_items (128-byte items). out_mix / out_final = 8 words each.
extern "C" __global__ void kawpow_gpu_hash(
    const uint32_t* __restrict__ header, uint64_t nonce,
    const uint32_t* __restrict__ dag, const uint32_t* __restrict__ l1,
    uint32_t dataset_items_128, int block_number,
    uint32_t* __restrict__ out_mix, uint32_t* __restrict__ out_final)
{
    // --- Initial keccak_f800: seed = keccak(header || nonce || rvn_pad). ---
    uint32_t state[25];
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[i] = header[i];
    state[8] = (uint32_t)nonce;
    state[9] = (uint32_t)(nonce >> 32);
    #pragma unroll
    for (int i = 10; i < 25; ++i) state[i] = kRvnPad[i - 10];
    keccak_f800(state);

    uint32_t state2[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) state2[i] = state[i];

    // --- ProgPoW mix. ---
    uint32_t seed[2] = { state2[0], state2[1] };

    // init_mix: per-lane KISS99 seeded from the keccak seed.
    uint32_t mix[PP_NUM_LANES][PP_NUM_REGS];
    {
        const uint32_t z = fnv1a(FNV_OFFSET_BASIS, seed[0]);
        const uint32_t w = fnv1a(z, seed[1]);
        for (uint32_t l = 0; l < PP_NUM_LANES; ++l)
        {
            const uint32_t jsr = fnv1a(w, l);
            const uint32_t jcong = fnv1a(jsr, l);
            kiss99_t rng{z, w, jsr, jcong};
            for (uint32_t i = 0; i < PP_NUM_REGS; ++i)
                mix[l][i] = rng.next();
        }
    }

    // mix_rng_state seeded from the period (block_number / period_length).
    const uint64_t number = (uint64_t)block_number / PP_PERIOD_LENGTH;
    uint32_t period_seed[2] = { (uint32_t)number, (uint32_t)(number >> 32) };
    mix_rng_state state_rng(period_seed);

    const uint32_t num_items = dataset_items_128 / 2;  // 256-byte items
    for (uint32_t i = 0; i < 64; ++i)
        pp_round(dag, l1, num_items, i, mix, state_rng);

    // Reduce mix to per-lane hashes, then cross-lane to 256-bit mix_hash.
    uint32_t lane_hash[PP_NUM_LANES];
    for (uint32_t l = 0; l < PP_NUM_LANES; ++l)
    {
        lane_hash[l] = FNV_OFFSET_BASIS;
        for (uint32_t i = 0; i < PP_NUM_REGS; ++i)
            lane_hash[l] = fnv1a(lane_hash[l], mix[l][i]);
    }
    uint32_t mix_hash[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) mix_hash[i] = FNV_OFFSET_BASIS;
    for (uint32_t l = 0; l < PP_NUM_LANES; ++l)
        mix_hash[l % 8] = fnv1a(mix_hash[l % 8], lane_hash[l]);

    // --- Final keccak_f800: final = keccak(state2 || mix_hash || rvn_pad). ---
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[i] = state2[i];
    #pragma unroll
    for (int i = 8; i < 16; ++i) state[i] = mix_hash[i - 8];
    #pragma unroll
    for (int i = 16; i < 25; ++i) state[i] = kRvnPad[i - 16];
    keccak_f800(state);

    #pragma unroll
    for (int i = 0; i < 8; ++i) { out_mix[i] = mix_hash[i]; out_final[i] = state[i]; }
}
