#include <stdio.h>
#include <memory.h>

#include "miner.h"

#include "cuda/helper.cuh"


// globaler Speicher für alle HeftyHashes aller Threads
uint32_t *heavy_heftyHashes[MAX_GPUS];

/* Hash-Tabellen */
__constant__ uint32_t hefty_gpu_constantTable[64];
// K, staged per block in shared memory by the kernel
#define HEFTY_SHARED_BYTES (64 * sizeof(uint32_t))
#define heftyLookUp(x) (*((uint32_t*)heftytab + (x)))

// muss expandiert werden
__constant__ uint32_t hefty_gpu_blockHeader[16]; // 2x512 Bit Message
__constant__ uint32_t hefty_gpu_register[8];
__constant__ uint32_t hefty_gpu_sponge[4];

uint32_t hefty_cpu_hashTable[] = {
    0x6a09e667UL,
    0xbb67ae85UL,
    0x3c6ef372UL,
    0xa54ff53aUL,
    0x510e527fUL,
    0x9b05688cUL,
    0x1f83d9abUL,
    0x5be0cd19UL
};

uint32_t hefty_cpu_constantTable[] = {
    0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL,
    0x3956c25bUL, 0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL,
    0xd807aa98UL, 0x12835b01UL, 0x243185beUL, 0x550c7dc3UL,
    0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL, 0xc19bf174UL,
    0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
    0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL,
    0x983e5152UL, 0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL,
    0xc6e00bf3UL, 0xd5a79147UL, 0x06ca6351UL, 0x14292967UL,
    0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL, 0x53380d13UL,
    0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
    0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL,
    0xd192e819UL, 0xd6990624UL, 0xf40e3585UL, 0x106aa070UL,
    0x19a4c116UL, 0x1e376c08UL, 0x2748774cUL, 0x34b0bcb5UL,
    0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL, 0x682e6ff3UL,
    0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
    0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL
};

__host__ __device__
static uint32_t S(uint32_t x, int n)
{
    return (((x) >> (n)) | ((x) << (32 - (n))));
}

#define R(x, n)          ((x) >> (n))
#define Ch(x, y, z)      ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)     ((x & (y | z)) | (y & z))
#define S0(x)            (S(x, 2) ^ S(x, 13) ^ S(x, 22))
#define S1(x)            (S(x, 6) ^ S(x, 11) ^ S(x, 25))
#define s0(x)            (S(x, 7) ^ S(x, 18) ^ R(x, 3))
#define s1(x)            (S(x, 17) ^ S(x, 19) ^ R(x, 10))

#define SWAB32(x)        ( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )

// uint8_t
#define smoosh4(x)       ( ((x)>>4) ^ ((x) & 0x0F) )

__host__ __forceinline__ __device__
uint8_t smoosh2(uint32_t x)
{
    uint16_t w = (x >> 16) ^ (x & 0xffff);
    uint8_t n = smoosh4( (uint8_t)( (w >> 8) ^ (w & 0xFF) ) );
    return 24 - (((n >> 2) ^ (n & 0x03)) << 3);
}
// 4 auf einmal
#define smoosh4Quad(x)   ( (((x)>>4) ^ (x)) & 0x0F0F0F0F )
#define getByte(x,y)     ( ((x) >> (y)) & 0xFF )

// sm_80+: adds issued as x * 1 + y on the IMAD pipe (the ALU pipe is the busy one);
// the 1 lives in constant memory so ptxas cannot fold it back into an add.
__constant__ uint32_t c_hefty_one = 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t hefty_madd(uint32_t x, uint32_t y)
{
    uint32_t r;
    asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(x), "r"(c_hefty_one), "r"(y));
    return r;
}
#endif

// kernel-body sums: IMAD on sm_80+, the original expression elsewhere
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define HEFTY_ADD2(a, b)       hefty_madd(a, b)
#define HEFTY_ADD4(a, b, c, d) hefty_madd(hefty_madd(a, b), hefty_madd(c, d))
#else
#define HEFTY_ADD2(a, b)       ((a) + (b))
#define HEFTY_ADD4(a, b, c, d) ((a) + (b) + (c) + (d))
#endif

#ifdef __CUDA_ARCH__
// Device form of one Mangle half, same function as the host code below. With k the XOR
// of the 2-bit groups of `from`, sel = k ^ 3 selects both the case and the rotate byte of r:
// sel 2 add, sel 1 and-not, otherwise xor.
__device__ __forceinline__
uint32_t hefty_mangle_half(uint32_t s0, uint32_t target, uint32_t from, uint32_t r)
{
    uint32_t f = from ^ (from >> 16);
    f ^= f >> 8;
    f ^= f >> 4;
    const uint32_t sel = ~(f ^ (f >> 2)) & 3;
    const uint32_t rot = __funnelshift_r(s0, s0, __byte_perm(r, 0, sel));
    // computed and selected, not branched: the case differs per nonce
#if __CUDA_ARCH__ >= 800
    const uint32_t add = hefty_madd(~rot, target); // target + ~rot
    const uint32_t andn = target & ~rot;
    const uint32_t x = target ^ rot;
    return sel == 2 ? add : (sel == 1 ? andn : x);
#else
    const uint32_t add = target - rot - 1; // target + ~rot
    const uint32_t andn = target & ~rot;
    const uint32_t x = target ^ rot;
    return sel == 2 ? add : (sel == 1 ? andn : x);
#endif
}
#endif

__host__ __forceinline__ __device__
void Mangle(uint32_t *inp)
{
#ifdef __CUDA_ARCH__
    const uint32_t r = smoosh4Quad(inp[0]);
    inp[1] ^= __funnelshift_r(inp[0], inp[0], r >> 24);
#if __CUDA_ARCH__ >= 800
    inp[2] = hefty_mangle_half(inp[0], inp[2], inp[1], hefty_madd(r, 0x01010101));
    inp[3] = hefty_mangle_half(inp[0], inp[3], inp[1] ^ inp[2], hefty_madd(r, 0x02020202));
    inp[0] ^= hefty_madd(inp[1] ^ inp[2], inp[3]);
#else
    inp[2] = hefty_mangle_half(inp[0], inp[2], inp[1], r + 0x01010101);
    inp[3] = hefty_mangle_half(inp[0], inp[3], inp[1] ^ inp[2], r + 0x02020202);
    inp[0] ^= (inp[1] ^ inp[2]) + inp[3];
#endif
#else
    uint32_t r = smoosh4Quad(inp[0]);
    uint32_t inp0org;
    uint32_t tmp0Mask, tmp1Mask;
    uint32_t in1, in2, isAddition;
    int32_t tmp;
    uint8_t b;

    inp[1] = inp[1] ^ S(inp[0], getByte(r, 24));

    r += 0x01010101;
    tmp = smoosh2(inp[1]);
    b = getByte(r,tmp);
    inp0org = S(inp[0], b);
    tmp0Mask = (uint32_t) -((tmp >> 3) & 1); // Bit 3 an Position 0
    tmp1Mask = (uint32_t) -((tmp >> 4) & 1); // Bit 4 an Position 0

    in1 =    (inp[2] & ~inp0org) |
            (tmp1Mask & ~inp[2] & inp0org) |
            (~tmp0Mask & ~inp[2] & inp0org);
    in2 = inp[2] += ~inp0org;
    isAddition = ~tmp0Mask & tmp1Mask;
    inp[2] = isAddition ? in2 : in1;

    r += 0x01010101;
    tmp = smoosh2(inp[1] ^ inp[2]);
    b = getByte(r,tmp);
    inp0org = S(inp[0], b);
    tmp0Mask = (uint32_t) -((tmp >> 3) & 1); // Bit 3 an Position 0
    tmp1Mask = (uint32_t) -((tmp >> 4) & 1); // Bit 4 an Position 0

    in1 =    (inp[3] & ~inp0org) |
            (tmp1Mask & ~inp[3] & inp0org) |
            (~tmp0Mask & ~inp[3] & inp0org);
    in2 = inp[3] += ~inp0org;
    isAddition = ~tmp0Mask & tmp1Mask;
    inp[3] = isAddition ? in2 : in1;

    inp[0] ^= (inp[1] ^ inp[2]) + inp[3];
#endif
}

__host__ __forceinline__ __device__
void Absorb(uint32_t *inp, uint32_t x)
{
    inp[0] ^= x;
    Mangle(inp);
}

__host__ __forceinline__ __device__
uint32_t Squeeze(uint32_t *inp)
{
    uint32_t y = inp[0];
    Mangle(inp);
    return y;
}

__host__ __forceinline__ __device__
uint32_t Br(uint32_t *sponge, uint32_t x)
{
    uint32_t r = Squeeze(sponge);
    uint32_t t = ((r >> 8) & 0x1F);
#ifdef __CUDA_ARCH__
    // bit t of x kept (r&3 == 0), cleared (1), set (2) or flipped (3)
    const uint32_t b0 = (r & 1) << t, b1 = ((r >> 1) & 1) << t;
    return (x & ~b0 & ~b1) | (b1 & ~b0) | (~x & b0 & b1);
#else
    uint32_t y = 1 << t;

    uint32_t a = (((r>>1) & 0x01) << t) & y;
    uint32_t b = ((r & 0x01) << t) & y;
    uint32_t c = x & y;

    uint32_t retVal = (x & ~y) | (~b & c) | (a & ~c);
    return retVal;
#endif
}

__device__ __forceinline__
void hefty_gpu_round(uint32_t *regs, uint32_t W, uint32_t K, uint32_t *sponge)
{
    uint32_t tmpBr;

    uint32_t brG = Br(sponge, regs[6]);
    uint32_t brF = Br(sponge, regs[5]);
#if __CUDA_ARCH__ >= 800
    uint32_t tmp1 = hefty_madd(hefty_madd(Ch(regs[4], brF, brG), regs[7]), hefty_madd(W, K));
#else
    uint32_t tmp1 = Ch(regs[4], brF, brG) + regs[7] + W + K;
#endif
    uint32_t brE = Br(sponge, regs[4]);
#if __CUDA_ARCH__ >= 800
    uint32_t tmp2 = hefty_madd(tmp1, S1(brE));
#else
    uint32_t tmp2 = tmp1 + S1(brE);
#endif
    uint32_t brC = Br(sponge, regs[2]);
    uint32_t brB = Br(sponge, regs[1]);
    uint32_t brA = Br(sponge, regs[0]);
    uint32_t tmp3 = Maj(brA, brB, brC);
    tmpBr = Br(sponge, regs[0]);
#if __CUDA_ARCH__ >= 800
    uint32_t tmp4 = hefty_madd(tmp3, S0(tmpBr));
#else
    uint32_t tmp4 = tmp3 + S0(tmpBr);
#endif
    tmpBr = Br(sponge, tmp2);

    #pragma unroll 7
    for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
#if __CUDA_ARCH__ >= 800
    regs[0] = hefty_madd(tmp2, tmp4);
    regs[4] = hefty_madd(regs[4], tmpBr);
#else
    regs[0] = tmp2 + tmp4;
    regs[4] += tmpBr;
#endif
}

__host__
void hefty_cpu_round(uint32_t *regs, uint32_t W, uint32_t K, uint32_t *sponge)
{
    uint32_t tmpBr;

    uint32_t brG = Br(sponge, regs[6]);
    uint32_t brF = Br(sponge, regs[5]);
    uint32_t tmp1 = Ch(regs[4], brF, brG) + regs[7] + W + K;
    uint32_t brE = Br(sponge, regs[4]);
    uint32_t tmp2 = tmp1 + S1(brE);
    uint32_t brC = Br(sponge, regs[2]);
    uint32_t brB = Br(sponge, regs[1]);
    uint32_t brA = Br(sponge, regs[0]);
    uint32_t tmp3 = Maj(brA, brB, brC);
    tmpBr = Br(sponge, regs[0]);
    uint32_t tmp4 = tmp3 + S0(tmpBr);
    tmpBr = Br(sponge, tmp2);

    for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
    regs[0] = tmp2 + tmp4;
    regs[4] += tmpBr;
}

// One HEFTY1 per thread. OUT_WORDS: 8 = 32-byte digests, 16 = digest zero-padded to 64 bytes.
template <int OUT_WORDS>
__device__ __forceinline__
void hefty_hash_body(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
    extern __shared__ unsigned char heftytab[];
    if(threadIdx.x < 64)
    {
        *((uint32_t*)heftytab + threadIdx.x) = hefty_gpu_constantTable[threadIdx.x];
    }

    __syncthreads();

    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        // bestimme den aktuellen Zähler
        uint32_t nounce = startNounce + thread;

        // jeder thread in diesem  Block bekommt sein eigenes W Array im Shared memory
        // reduktion von 256 byte auf 128 byte
        uint32_t W1[16];
        uint32_t W2[16];

        // Initialisiere die register a bis h mit der Hash-Tabelle
        uint32_t regs[8];
        uint32_t hash[8];
        uint32_t sponge[4];

#pragma unroll 4
        for(int k=0; k < 4; k++)
            sponge[k] = hefty_gpu_sponge[k];

        // pre
#pragma unroll 8
        for (int k=0; k < 8; k++)
        {
            regs[k] = hefty_gpu_register[k];
            hash[k] = regs[k];
        }

        //memcpy(W, &hefty_gpu_blockHeader[0], sizeof(uint32_t) * 16); // verbleibende 20 bytes aus Block 2 plus padding
#pragma unroll 16
        for(int k=0;k<16;k++)
            W1[k] = hefty_gpu_blockHeader[k];
        W1[3] = cuda_swab32(nounce);

        // 2. Runde
#pragma unroll 16
        for(int j=0;j<16;j++)
            Absorb(sponge, W1[j] ^ heftyLookUp(j));

// Progress W1 (Bytes 0...63)
#pragma unroll 16
        for(int j=0;j<16;j++)
        {
            Absorb(sponge, regs[3] ^ regs[7]);
            hefty_gpu_round(regs, W1[j], heftyLookUp(j), sponge);
        }

// Progress W2 (Bytes 64...127) then W3 (Bytes 128...191) ...


        for(int k=0;k<3;k++)
        {

            for(int j=0;j<2;j++)
                W2[j] = HEFTY_ADD4(s1(W1[14+j]), W1[9+j], s0(W1[1+j]), W1[j]);

            for(int j=2;j<7;j++)
                W2[j] = HEFTY_ADD4(s1(W2[j-2]), W1[9+j], s0(W1[1+j]), W1[j]);


            for(int j=7;j<15;j++)
                W2[j] = HEFTY_ADD4(s1(W2[j-2]), W2[j-7], s0(W1[1+j]), W1[j]);

            W2[15] = HEFTY_ADD4(s1(W2[13]), W2[8], s0(W2[0]), W1[15]);


            for(int j=0;j<16;j++)
            {
                Absorb(sponge, HEFTY_ADD2(regs[3], regs[7]));
                hefty_gpu_round(regs, W2[j], heftyLookUp(j + ((k+1)<<4)), sponge);
            }

            for(int j=0;j<16;j++)
                W1[j] = W2[j];
        }

#pragma unroll 8
        for(int k=0;k<8;k++)
            hash[k] += regs[k];

        if (OUT_WORDS == 8) {
#pragma unroll 8
            for(int k=0;k<8;k++)
                ((uint32_t*)outputHash)[(thread<<3)+k] = cuda_swab32(hash[k]);
        } else {
            uint4 *pdst = (uint4*) &outputHash[(size_t) thread * OUT_WORDS];
            pdst[0] = make_uint4(cuda_swab32(hash[0]), cuda_swab32(hash[1]), cuda_swab32(hash[2]), cuda_swab32(hash[3]));
            pdst[1] = make_uint4(cuda_swab32(hash[4]), cuda_swab32(hash[5]), cuda_swab32(hash[6]), cuda_swab32(hash[7]));
            pdst[2] = make_uint4(0, 0, 0, 0);
            pdst[3] = make_uint4(0, 0, 0, 0);
        }
    }
}

__global__
void hefty_gpu_hash(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
    hefty_hash_body<8>(threads, startNounce, outputHash);
}

__global__
void hefty_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
    hefty_hash_body<16>(threads, startNounce, outputHash);
}

// constant tables only (hefty_cpu_hash_64 writes into the caller's buffer)
__host__
void hefty_cpu_init_tables(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);

    // Kopiere die Hash-Tabellen in den GPU-Speicher
    cudaMemcpyToSymbol( hefty_gpu_constantTable,
                        hefty_cpu_constantTable,
                        sizeof(uint32_t) * 64 );
}

__host__
void hefty_cpu_init(int thr_id, uint32_t threads)
{
    hefty_cpu_init_tables(thr_id);

    // Speicher für alle Hefty1 hashes belegen
    CUDA_SAFE_CALL(cudaMalloc(&heavy_heftyHashes[thr_id], (size_t) 32 * threads));
}

__host__
void hefty_cpu_free(int thr_id)
{
    cudaFree(heavy_heftyHashes[thr_id]);
}

__host__
void hefty_cpu_setBlock(int thr_id, uint32_t threads, void *data, int len)
// data muss 80/84-Byte haben!
{
    // Nachricht expandieren und setzen
    uint32_t msgBlock[32];

    memset(msgBlock, 0, sizeof(msgBlock));
    memcpy(&msgBlock[0], data, len);
    if (len == 84) {
        msgBlock[21] |= 0x80;
        msgBlock[31] = 672; // bitlen
    } else if (len == 80) {
        msgBlock[20] |= 0x80;
        msgBlock[31] = 640; // bitlen
    }

    for(int i=0;i<31;i++) // Byteorder drehen
        msgBlock[i] = SWAB32(msgBlock[i]);

    // die erste Runde wird auf der CPU durchgeführt, da diese für
    // alle Threads gleich ist. Der Hash wird dann an die Threads
    // übergeben

    // Erstelle expandierten Block W
    uint32_t W[64];
    memcpy(W, &msgBlock[0], sizeof(uint32_t) * 16);
    for(int j=16;j<64;j++)
        W[j] = s1(W[j-2]) + W[j-7] + s0(W[j-15]) + W[j-16];

    // Initialisiere die register a bis h mit der Hash-Tabelle
    uint32_t regs[8];
    uint32_t hash[8];
    uint32_t sponge[4];

    // pre
    memset(sponge, 0, sizeof(uint32_t) * 4);
    for (int k=0; k < 8; k++)
    {
        regs[k] = hefty_cpu_hashTable[k];
        hash[k] = regs[k];
    }

    // 1. Runde
    for(int j=0;j<16;j++)
        Absorb(sponge, W[j] ^ hefty_cpu_constantTable[j]);

    for(int j=0;j<16;j++)
    {
        Absorb(sponge, regs[3] ^ regs[7]);
        hefty_cpu_round(regs, W[j], hefty_cpu_constantTable[j], sponge);
    }

    for(int j=16;j<64;j++)
    {
        Absorb(sponge, regs[3] + regs[7]);
        hefty_cpu_round(regs, W[j], hefty_cpu_constantTable[j], sponge);
    }

    for(int k=0;k<8;k++)
        hash[k] += regs[k];

    // sponge speichern
    cudaMemcpyToSymbol(hefty_gpu_sponge, sponge, 16);
    // hash speichern
    cudaMemcpyToSymbol(hefty_gpu_register, hash, 32);
    // Blockheader setzen (korrekte Nonce fehlt da drin noch)
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(hefty_gpu_blockHeader, &msgBlock[16], 64));
}

__host__
void hefty_cpu_hash(int thr_id, uint32_t threads, int startNounce)
{
    uint32_t threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    const int shared_size = HEFTY_SHARED_BYTES;

    hefty_gpu_hash <<< grid, block, shared_size >>> (threads, startNounce, heavy_heftyHashes[thr_id]);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, 0, thr_id);
}

// digests written as 64-byte records (upper half zero) straight into the caller's buffer
__host__
void hefty_cpu_hash_64(int thr_id, uint32_t threads, int startNounce, uint32_t *d_outputHash)
{
    const uint32_t threadsperblock = 256;
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);
    const int shared_size = HEFTY_SHARED_BYTES;
    hefty_gpu_hash_64 <<< grid, block, shared_size >>> (threads, startNounce, d_outputHash);
    MyStreamSynchronize(NULL, 0, thr_id);
}

