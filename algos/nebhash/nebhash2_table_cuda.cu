// nebhash v2 attempts on CUDA from a table of per-sample gradients. The contract is in nebhash2_table_cuda.h.
//
// The gradient of an attempt is the sum of the gradients of its 16 samples, clipped afterwards, and the gradient of a
// sample depends on the model and on that sample only. For one model state the host evaluates every sample once with
// neb2_sample_grad of the C reference and uploads one row per sample. A row holds, per layer: every gradient entry of
// a convolution (8 or 16 bits, or 32 when a value does not fit), and for a linear layer its cout errors then its cin
// inputs, whose outer product is the gradient of the layer. One thread block evaluates one attempt: it gathers the
// factors of its 16 rows, transposed so that one entry of a linear layer is four dp4a, sums, clips, and hashes.
#include "nebhash2_table_cuda.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include "nebhash2.h"
#include "cuda/blake3_device.cuh"

#ifndef THREADS
#define THREADS 128
#endif
#define CHUNK 65536u  // attempts per launch
// sm_6x: the launch shape (THREADS or 32 threads per attempt) is chosen per model. After each build, launches of at
// least TUNE_MIN attempts alternate the two shapes, timed after one warm-up, until each has TUNE_ATTEMPTS; the faster
// is kept. Both shapes give the same tickets, and the timed launches are real work.
#define TUNE_MIN 1024u
#define TUNE_ATTEMPTS 262144u

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

#define QR(a, b, c, d) \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8);  x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);

static __device__ void chacha8_block0(const uint32_t key[8], uint32_t out[16]) {
  uint32_t s[16], x[16];
  s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
  for (int i = 0; i < 8; i++) s[4 + i] = key[i];
  s[12] = 0; s[13] = 0; s[14] = 0; s[15] = 0;
  for (int i = 0; i < 16; i++) x[i] = s[i];
  for (int r = 0; r < 4; r++) {
    QR(0, 4, 8, 12) QR(1, 5, 9, 13) QR(2, 6, 10, 14) QR(3, 7, 11, 15)
    QR(0, 5, 10, 15) QR(1, 6, 11, 12) QR(2, 7, 8, 13) QR(3, 4, 9, 14)
  }
  for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
}

enum { NARROW = 0, WIDE = 1, LINEAR = 2, NARROW8 = 3 };

// The layout of a row and of the shared memory of a block.
struct TPlan {
  int nlayers;
  int kind[NEB2_MAX_LAYERS], nw[NEB2_MAX_LAYERS], cin[NEB2_MAX_LAYERS], cout[NEB2_MAX_LAYERS];
  int at[NEB2_MAX_LAYERS];                             // offset of the layer in a row
  int sh_d[NEB2_MAX_LAYERS], sh_in[NEB2_MAX_LAYERS];   // linear: error words in shared; sh_in is unused (rows are reloaded)
  int row, shared_words;
  uint32_t linear_zero[8];
  // conv layers: blk = weights per output channel; am = offset of the layer's channel-activity word in a row (bit o:
  // some weight of output channel o has a non-zero gradient for this sample), -1 when cout > 32 (no mask)
  int blk[NEB2_MAX_LAYERS], am[NEB2_MAX_LAYERS];
};
static __device__ __constant__ TPlan P;

#ifdef NEB_NEGATIVE_CONTROL_TABLE_CLIP
#define CLIPV (NEB_CLIP - 1)  // wrong on purpose: the check must fail
#else
#define CLIPV NEB_CLIP
#endif

__device__ __forceinline__ uint32_t warp_sum(uint32_t v)
{
  for (int off = 16; off; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__global__ void header_prefix(const uint32_t *header, uint32_t *cv) {
  uint32_t first[16];
  for (int i = 0; i < 16; i++) first[i] = header[i];
  blake3_compress(blake3_iv(), first, 0, 64, BLAKE3_CHUNK_START, cv);
}

// One linear layer of an attempt (NT = threads per block). With FIXED (cin % 8 == 0; NT % 8 == 0), every entry f = o * cin + q of
// this thread has f & 7 == q & 7 == tid & 7, so it adds into ONE lane: a plain accumulator replaces the runtime-indexed
// lane[f & 7], which ptxas turned into an 8-way compare + predicated read and write per entry. Sums are mod 2^32, so
// the order of the additions does not change the ticket. !FIXED keeps the per-entry lane index.
static_assert(THREADS % 8 == 0, "the fixed lane index needs THREADS % 8 == 0");  // NT: 128 or 32
template <int NT, bool FIXED>
__device__ __forceinline__ void linear_layer(const uint8_t *__restrict__ rows, const uint32_t *rb, const int32_t *S,
                                             int i, int tid, const int *act, int nact, uint32_t base, uint32_t *lane) {
  const int cin = P.cin[i];
  const int xat = P.at[i] + P.cout[i];
  uint32_t acc = 0;
  for (int q = tid; q < cin; q += NT) {
    int32_t x[4];
    for (int w = 0; w < 4; w++) {
      uint32_t v = 0;
      for (int b = 0; b < 4; b++)
        v |= (uint32_t)__ldg(&rows[rb[4 * w + b] + xat + q]) << (8 * b);
      x[w] = (int32_t)v;
    }
    for (int j = 0; j < nact; j++) {
      const int o = act[j];
      const int32_t *d = S + P.sh_d[i] + 4 * o;
      int G = 0;
      G = __dp4a(d[0], x[0], G);
      G = __dp4a(d[1], x[1], G);
      G = __dp4a(d[2], x[2], G);
      G = __dp4a(d[3], x[3], G);
      const uint32_t f = (uint32_t)(o * cin + q);
      const int cg = G < -CLIPV ? -CLIPV : (G > CLIPV ? CLIPV : G);
      const uint32_t v = ((uint32_t)cg ^ (f + base)) - (f + base);
      if (FIXED) acc += v; else lane[f & 7] += v;
    }
  }
  if (FIXED) lane[tid & 7] += acc;
}

// Linear layer, two inputs per thread (needs cin % 8 == 0 and an even xat): one 16-bit load per sample brings
// x[q] and x[q+1] of that sample, so 16 loads serve two input columns instead of 32 byte loads. Bytes of four
// samples are transposed with PRMT into the sample-packed words the dp4a takes (byte k = sample 4w + k), exactly the
// words the byte-load path builds. Lane of every entry: q & 7 = (2 * tid + j) & 7, two fixed accumulators.
template <int NT>
__device__ __forceinline__ void linear_layer_pairs(const uint8_t *__restrict__ rows, const uint32_t *rb, const int32_t *S,
                                                   int i, int tid, const int *act, int nact, uint32_t base, uint32_t *lane) {
  const int cin = P.cin[i];
  const int xat = P.at[i] + P.cout[i];
  uint32_t acc0 = 0, acc1 = 0;
  for (int q = 2 * tid; q < cin; q += 2 * NT) {
    int32_t x0[4], x1[4];
#pragma unroll
    for (int w = 0; w < 4; w++) {
      const uint32_t a = __ldg((const unsigned short *)(rows + rb[4 * w + 0] + xat + q));
      const uint32_t b = __ldg((const unsigned short *)(rows + rb[4 * w + 1] + xat + q));
      const uint32_t c = __ldg((const unsigned short *)(rows + rb[4 * w + 2] + xat + q));
      const uint32_t d = __ldg((const unsigned short *)(rows + rb[4 * w + 3] + xat + q));
      const uint32_t t01 = a | (b << 16), t23 = c | (d << 16);  // bytes: s0q s0q1 s1q s1q1 | s2q s2q1 s3q s3q1
      x0[w] = (int32_t)__byte_perm(t01, t23, 0x6420);
      x1[w] = (int32_t)__byte_perm(t01, t23, 0x7531);
    }
    for (int j = 0; j < nact; j++) {
      const int o = act[j];
      const int32_t *d = S + P.sh_d[i] + 4 * o;
      const int32_t d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3];
      int G0 = __dp4a(d0, x0[0], 0), G1 = __dp4a(d0, x1[0], 0);
      G0 = __dp4a(d1, x0[1], G0); G1 = __dp4a(d1, x1[1], G1);
      G0 = __dp4a(d2, x0[2], G0); G1 = __dp4a(d2, x1[2], G1);
      G0 = __dp4a(d3, x0[3], G0); G1 = __dp4a(d3, x1[3], G1);
      const uint32_t f0 = (uint32_t)(o * cin + q), f1 = f0 + 1;
      const int c0 = G0 < -CLIPV ? -CLIPV : (G0 > CLIPV ? CLIPV : G0);
      const int c1 = G1 < -CLIPV ? -CLIPV : (G1 > CLIPV ? CLIPV : G1);
      acc0 += ((uint32_t)c0 ^ (f0 + base)) - (f0 + base);
      acc1 += ((uint32_t)c1 ^ (f1 + base)) - (f1 + base);
    }
  }
  lane[(2 * tid) & 7] += acc0;
  lane[(2 * tid + 1) & 7] += acc1;
}

// Launch bounds: sm_6x runs 8 blocks of 128 threads per SM (64 registers; this file is built without --maxrregcount,
// see the build rules) or 32 blocks of 32; sm_75 and newer keep the plain bound.
#ifndef NEB_SM6X_MINB
#define NEB_SM6X_MINB 8
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define NEB_TABLE_BOUNDS __launch_bounds__(NT, NT == 32 ? 32 : NEB_SM6X_MINB)
#else
#define NEB_TABLE_BOUNDS __launch_bounds__(NT)
#endif
template <int NT>
__global__ void NEB_TABLE_BOUNDS table_kernel(const uint8_t *__restrict__ rows, uint32_t n,
                                                        const uint32_t *__restrict__ h0, uint32_t start,
                                                        const uint32_t *__restrict__ bound, uint32_t *hits,
                                                        uint32_t max_hits, const uint32_t *__restrict__ cv) {
  extern __shared__ int32_t S[];  // per linear layer: cout x 4 words of errors (inputs reloaded from rows)
  __shared__ uint32_t key[8], idx[NEB_NB], lanes[8];
  __shared__ uint32_t wsum[8][NT / 32];
  __shared__ int active[NEB2_MAX_LAYERS][32], active_count[NEB2_MAX_LAYERS];
  // conv: bit b of cm[i][o] = sample b has a non-zero gradient in output channel o of layer i; a sample whose channel
  // is inactive contributes exact zeros to that channel's weights, so its loads are skipped
  __shared__ uint32_t cm[NEB2_MAX_LAYERS][32], smask[NEB2_MAX_LAYERS][NEB_NB];
  const int tid = threadIdx.x;
  const uint32_t nonce = start + blockIdx.x;
  if (tid < 8) lanes[tid] = 0;
  if (tid == 0) {
    uint32_t b1[16] = { h0[16], h0[17], h0[18], nonce };
    uint32_t k[8], u[16];
    blake3_compress(cv, b1, 0, 16, BLAKE3_CHUNK_END | BLAKE3_ROOT, k);
    for (int i = 0; i < 8; i++) key[i] = k[i];
    chacha8_block0(k, u);
    for (int j = 0; j < NEB_NB; j++) idx[j] = (uint32_t)(((unsigned long long)u[j] * n) >> 32);
  }
  __syncthreads();
  // Row offsets fit in 32 bits for the official dataset (build() rejects a
  // larger table), which avoids a 64-bit multiply on every row byte load.
  uint32_t rb[NEB_NB];
#pragma unroll
  for (int b = 0; b < NEB_NB; b++) rb[b] = idx[b] * (uint32_t)P.row;
  // Only the cout error words are staged; the cin input words are loaded once per input element and reused across
  // all outputs, so shared memory stays small. Lane sums are mod 2^32, so the loop order does not matter.
  for (int i = 0; i < P.nlayers; i++) {
    if (P.kind[i] != LINEAR) continue;
    for (int o = tid; o < P.cout[i]; o += NT) {
      for (int w = 0; w < 4; w++) {
        uint32_t v = 0;
        for (int b = 0; b < 4; b++)
          v |= (uint32_t)__ldg(&rows[rb[4 * w + b] + P.at[i] + o]) << (8 * b);
        S[P.sh_d[i] + 4 * o + w] = (int32_t)v;
      }
    }
  }
  if (tid < NEB_NB)
    for (int i = 0; i < P.nlayers; i++)
      if (P.kind[i] != LINEAR && P.am[i] >= 0)
        smask[i][tid] = __ldg((const uint32_t *)(rows + idx[tid] * (uint32_t)P.row + P.am[i]));
  __syncthreads();
  if (tid < P.nlayers && P.kind[tid] == LINEAR) {
    int count = 0;
    for (int o = 0; o < P.cout[tid]; o++) {
      const int32_t *d = S + P.sh_d[tid] + 4 * o;
      if ((d[0] | d[1] | d[2] | d[3]) != 0) active[tid][count++] = o;
    }
    active_count[tid] = count;
  }
  // the channel masks: warps 1.. while warp 0 compacts the linear errors; a one-warp block does both in turn
  if (NT == 32 || tid >= 32) {
    const int o = NT == 32 ? tid : tid - 32;
    for (int i = 0; i < P.nlayers; i++) {
      if (P.kind[i] == LINEAR || P.am[i] < 0 || o >= P.cout[i]) continue;
      uint32_t m = 0;
#pragma unroll
      for (int b = 0; b < NEB_NB; b++) m |= ((smask[i][b] >> o) & 1u) << b;
      cm[i][o] = m;
    }
  }
  __syncthreads();
  uint32_t lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < P.nlayers; i++) {
    const uint32_t base = 7919u * (uint32_t)(i + 1);
    if (P.kind[i] == LINEAR) {
      if ((P.cin[i] & 7) == 0 && ((P.at[i] + P.cout[i]) & 1) == 0)
        linear_layer_pairs<NT>(rows, rb, S, i, tid, active[i], active_count[i], base, lane);
      else if ((P.cin[i] & 7) == 0)
        linear_layer<NT, true>(rows, rb, S, i, tid, active[i], active_count[i], base, lane);
      else
        linear_layer<NT, false>(rows, rb, S, i, tid, active[i], active_count[i], base, lane);
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
      if (P.kind[i] == NARROW && P.am[i] >= 0 && (P.blk[i] & 7) == 0 && P.blk[i] <= 64) {
        // Pascal: one warp per channel, lane ln holds entries f, f + 1 (f = o * blk + 2 * ln), one 32-bit load per
        // sample active in the channel.
        const int ln = tid & 31, half = P.blk[i] >> 1;
        uint32_t acc0 = 0, acc1 = 0;
        if (ln < half) {
          for (int o = tid >> 5; o < P.cout[i]; o += NT / 32) {
            const int f = o * P.blk[i] + 2 * ln;
            const uint8_t *rp = rows + P.at[i] + 2 * f;
            int G0 = 0, G1 = 0;
            uint32_t m = cm[i][o];
            while (m) {
              int bb[4];
#pragma unroll
              for (int u = 0; u < 4; u++) { bb[u] = m ? __ffs(m) - 1 : -1; m &= m - 1; }
              uint32_t v[4];
#pragma unroll
              for (int u = 0; u < 4; u++)
                v[u] = bb[u] >= 0 ? __ldg((const uint32_t *)(rp + idx[bb[u]] * (uint32_t)P.row)) : 0u;
#pragma unroll
              for (int u = 0; u < 4; u++) { G0 += (int)(int16_t)(v[u] & 0xffffu); G1 += (int)v[u] >> 16; }
            }
            const int c0 = G0 < -CLIPV ? -CLIPV : (G0 > CLIPV ? CLIPV : G0);
            const int c1 = G1 < -CLIPV ? -CLIPV : (G1 > CLIPV ? CLIPV : G1);
            acc0 += (uint32_t)c0 ^ ((uint32_t)f + base);
            acc1 += (uint32_t)c1 ^ ((uint32_t)(f + 1) + base);
          }
        }
        lane[(2 * ln) & 7] += acc0;
        lane[(2 * ln + 1) & 7] += acc1;
      } else
#else
      if (P.kind[i] == NARROW && P.am[i] >= 0 && (P.blk[i] & 7) == 0 && P.blk[i] <= 128) {
        // A sample whose output channel o is inactive (no non-zero gradient in the channel's blk weights) adds exact
        // zeros there, so each channel loads only the samples active in it (cm[i][o]). A channel is blk / 4 lanes of
        // one 64-bit load each (entries f .. f + 3, f = o * blk + 4 * l), and a warp takes 32 / (blk / 4) channels
        // side by side. blk % 8 == 0 keeps the lanes (4 * l + j) & 7 fixed per thread.
        const int ln = tid & 31, q = P.blk[i] >> 2, groups = 32 / q;
        const int g = ln / q, l = ln - g * q;
        uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        if (g < groups) {
          for (int o = (tid >> 5) * groups + g; o < P.cout[i]; o += (NT / 32) * groups) {
            const int f = o * P.blk[i] + 4 * l;
            const uint8_t *rp = rows + P.at[i] + 2 * f;
            int G0 = 0, G1 = 0, G2 = 0, G3 = 0;
            uint32_t m = cm[i][o];
            while (m) {
              int bb[4];
#pragma unroll
              for (int u = 0; u < 4; u++) { bb[u] = m ? __ffs(m) - 1 : -1; m &= m - 1; }
              uint2 v[4];
#pragma unroll
              for (int u = 0; u < 4; u++)
                v[u] = bb[u] >= 0 ? __ldg((const uint2 *)(rp + idx[bb[u]] * (uint32_t)P.row)) : make_uint2(0u, 0u);
#pragma unroll
              for (int u = 0; u < 4; u++) {
                G0 += (int)(int16_t)(v[u].x & 0xffffu);
                G1 += (int)v[u].x >> 16;
                G2 += (int)(int16_t)(v[u].y & 0xffffu);
                G3 += (int)v[u].y >> 16;
              }
            }
            const int c0 = G0 < -CLIPV ? -CLIPV : (G0 > CLIPV ? CLIPV : G0);
            const int c1 = G1 < -CLIPV ? -CLIPV : (G1 > CLIPV ? CLIPV : G1);
            const int c2 = G2 < -CLIPV ? -CLIPV : (G2 > CLIPV ? CLIPV : G2);
            const int c3 = G3 < -CLIPV ? -CLIPV : (G3 > CLIPV ? CLIPV : G3);
            acc0 += (uint32_t)c0 ^ ((uint32_t)f + base);
            acc1 += (uint32_t)c1 ^ ((uint32_t)(f + 1) + base);
            acc2 += (uint32_t)c2 ^ ((uint32_t)(f + 2) + base);
            acc3 += (uint32_t)c3 ^ ((uint32_t)(f + 3) + base);
          }
        }
        lane[(4 * l) & 7] += acc0;
        lane[(4 * l + 1) & 7] += acc1;
        lane[(4 * l + 2) & 7] += acc2;
        lane[(4 * l + 3) & 7] += acc3;
      } else
#endif
      if (P.kind[i] == NARROW8) {
        // Four consecutive int8 factors occupy one aligned word. Transpose the
        // bytes of four rows so each resulting word holds one factor from four
        // samples, then dp4a sums those four samples with no 16-bit truncation.
        uint32_t acc4[4] = {0, 0, 0, 0};  // lane (4 * tid + j) & 7, fixed per thread and j
        for (int f0 = 4 * tid; f0 < P.nw[i]; f0 += 4 * NT) {
          int G0 = 0, G1 = 0, G2 = 0, G3 = 0;
#pragma unroll
          for (int r = 0; r < NEB_NB; r += 4) {
            const uint32_t p0 = __ldg((const uint32_t *)(rows + rb[r + 0] + P.at[i] + f0));
            const uint32_t p1 = __ldg((const uint32_t *)(rows + rb[r + 1] + P.at[i] + f0));
            const uint32_t p2 = __ldg((const uint32_t *)(rows + rb[r + 2] + P.at[i] + f0));
            const uint32_t p3 = __ldg((const uint32_t *)(rows + rb[r + 3] + P.at[i] + f0));
            const uint32_t t0 = (p0 & 0xffu) | ((p1 & 0xffu) << 8) | ((p2 & 0xffu) << 16) | ((p3 & 0xffu) << 24);
            const uint32_t t1 = ((p0 >> 8) & 0xffu) | (((p1 >> 8) & 0xffu) << 8) | (((p2 >> 8) & 0xffu) << 16) | (((p3 >> 8) & 0xffu) << 24);
            const uint32_t t2 = ((p0 >> 16) & 0xffu) | (((p1 >> 16) & 0xffu) << 8) | (((p2 >> 16) & 0xffu) << 16) | (((p3 >> 16) & 0xffu) << 24);
            const uint32_t t3 = ((p0 >> 24) & 0xffu) | (((p1 >> 24) & 0xffu) << 8) | (((p2 >> 24) & 0xffu) << 16) | (((p3 >> 24) & 0xffu) << 24);
            G0 = __dp4a((int)t0, (int)0x01010101, G0);
            G1 = __dp4a((int)t1, (int)0x01010101, G1);
            G2 = __dp4a((int)t2, (int)0x01010101, G2);
            G3 = __dp4a((int)t3, (int)0x01010101, G3);
          }
          const int G[4] = {G0, G1, G2, G3};
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int f = f0 + j;
            if (f >= P.nw[i]) break;
            const int cg = G[j] < -CLIPV ? -CLIPV : (G[j] > CLIPV ? CLIPV : G[j]);
            acc4[j] += (uint32_t)cg ^ ((uint32_t)f + base);
          }
        }
#pragma unroll
        for (int j = 0; j < 4; j++) lane[(4 * tid + j) & 7] += acc4[j];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
        // quads on Pascal only (slower than the pairs below on sm_86)
      } else if (P.kind[i] == NARROW && (P.nw[i] & 3) == 0) {
        // int16 factors in quads: one 64-bit load per sample brings f .. f+3 (layers and rows are 8-byte aligned,
        // 2 * f is a multiple of 8), a quarter of the loads of the per-factor gather. Sums stay 32-bit (exact).
        // f = 4 * tid + 4 * NT * k, so the lanes are (4 * tid + j) & 7: four fixed accumulators.
        uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        for (int f = 4 * tid; f < P.nw[i]; f += 4 * NT) {
          int G0 = 0, G1 = 0, G2 = 0, G3 = 0;
#pragma unroll
          for (int b = 0; b < NEB_NB; b++) {
            const uint2 v = __ldg((const uint2 *)(rows + rb[b] + P.at[i] + 2 * f));
            G0 += (int)(int16_t)(v.x & 0xffffu);
            G1 += (int)v.x >> 16;
            G2 += (int)(int16_t)(v.y & 0xffffu);
            G3 += (int)v.y >> 16;
          }
          const int c0 = G0 < -CLIPV ? -CLIPV : (G0 > CLIPV ? CLIPV : G0);
          const int c1 = G1 < -CLIPV ? -CLIPV : (G1 > CLIPV ? CLIPV : G1);
          const int c2 = G2 < -CLIPV ? -CLIPV : (G2 > CLIPV ? CLIPV : G2);
          const int c3 = G3 < -CLIPV ? -CLIPV : (G3 > CLIPV ? CLIPV : G3);
          acc0 += (uint32_t)c0 ^ ((uint32_t)f + base);
          acc1 += (uint32_t)c1 ^ ((uint32_t)(f + 1) + base);
          acc2 += (uint32_t)c2 ^ ((uint32_t)(f + 2) + base);
          acc3 += (uint32_t)c3 ^ ((uint32_t)(f + 3) + base);
        }
        lane[(4 * tid) & 7] += acc0;
        lane[(4 * tid + 1) & 7] += acc1;
        lane[(4 * tid + 2) & 7] += acc2;
        lane[(4 * tid + 3) & 7] += acc3;
#endif
      } else if (P.kind[i] == NARROW && (P.nw[i] & 1) == 0) {
        // int16 factors in pairs: one 32-bit load per sample brings entries f and f+1 (low / high half; at, the
        // row size and 2 * f are multiples of 4), halving the loads of the gather. Sums stay 32-bit, so they are
        // exact. f = 2 * tid + 2 * NT * k, so the lanes are (2 * tid + j) & 7: two fixed accumulators.
        uint32_t acc0 = 0, acc1 = 0;
        for (int f = 2 * tid; f < P.nw[i]; f += 2 * NT) {
          int G0 = 0, G1 = 0;
#pragma unroll
          for (int b = 0; b < NEB_NB; b++) {
            const uint32_t v = __ldg((const uint32_t *)(rows + rb[b] + P.at[i] + 2 * f));
            G0 += (int)(int16_t)(v & 0xffffu);
            G1 += (int)v >> 16;
          }
          const int c0 = G0 < -CLIPV ? -CLIPV : (G0 > CLIPV ? CLIPV : G0);
          const int c1 = G1 < -CLIPV ? -CLIPV : (G1 > CLIPV ? CLIPV : G1);
          acc0 += (uint32_t)c0 ^ ((uint32_t)f + base);
          acc1 += (uint32_t)c1 ^ ((uint32_t)(f + 1) + base);
        }
        lane[(2 * tid) & 7] += acc0;
        lane[(2 * tid + 1) & 7] += acc1;
      } else {
        uint32_t acc = 0;  // lane tid & 7 for every f of this thread
        for (int f = tid; f < P.nw[i]; f += NT) {
          int G = 0;
          if (P.kind[i] == NARROW) {
            for (int b = 0; b < NEB_NB; b++) G += __ldg((const int16_t *)(rows + rb[b] + P.at[i] + 2 * f));
          } else {
            for (int b = 0; b < NEB_NB; b++) G += __ldg((const int32_t *)(rows + rb[b] + P.at[i] + 4 * f));
          }
          const int cg = G < -CLIPV ? -CLIPV : (G > CLIPV ? CLIPV : G);
          acc += (uint32_t)cg ^ ((uint32_t)f + base);
        }
        lane[tid & 7] += acc;
      }
    }
  }
  // One reduction per warp instead of 256 shared-memory atomics per lane.
  // Addition is modulo 2^32, so changing the accumulation order is safe.
  {
    const int warp = tid >> 5;
    for (int k = 0; k < 8; k++) {
      const uint32_t v = warp_sum(lane[k]);
      if (!(tid & 31)) wsum[k][warp] = v;
    }
    __syncthreads();
    if (!tid) {
      for (int k = 0; k < 8; k++) {
        uint32_t v = 0;
        for (int w = 0; w < NT / 32; w++) v += wsum[k][w];
        lanes[k] = v + P.linear_zero[k];
      }
    }
  }
  if (tid == 0) {
    uint32_t blk[16], t[8];
    for (int i = 0; i < 8; i++) { blk[i] = lanes[i]; blk[8 + i] = key[i]; }
    blake3_compress(blake3_iv(), blk, 0, 64, BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT, t);
    bool ok = true;
    for (int k = 7; k >= 0; k--) {
      if (t[k] != bound[k]) { ok = t[k] < bound[k]; break; }
    }
    if (ok) {
      const uint32_t at = atomicAdd(hits, 1u);
      if (at < max_hits) hits[1 + at] = nonce;
    }
  }
}

// Sparse scan. Many samples of a trained model have an all-zero gradient, and a zero entry adds the constant
// 0 ^ (f + base) to the digest, so the digest of an attempt depends only on its active draws (samples with a non-zero
// gradient). With none it is the zero digest dig[n], with one it is that sample's own digest dig[s], both built with
// the table. Every evaluated attempt has the same chance (the ticket is a BLAKE3 of digest and key), so this kernel
// evaluates only the attempts with at most one active draw: one thread per nonce, key + ChaCha8 + 16 bitmap lookups,
// one more BLAKE3 when selected. evald counts the evaluated attempts. DUMP writes nonce and ticket of every selected
// attempt instead of comparing (self-test).
template <bool DUMP>
__global__ void __launch_bounds__(256) sparse_kernel(const uint32_t *__restrict__ flags, const uint32_t *__restrict__ dig,
                                                     uint32_t n, const uint32_t *__restrict__ h0, uint32_t start,
                                                     uint32_t count, const uint32_t *__restrict__ bound, uint32_t *hits,
                                                     uint32_t max_hits, const uint32_t *__restrict__ cv, uint32_t *evald) {
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  bool sel = false;
  uint32_t k[8];
  uint32_t sidx = n;
  const uint32_t nonce = start + g;
  if (g < count) {
    uint32_t b1[16] = { __ldg(&h0[16]), __ldg(&h0[17]), __ldg(&h0[18]), nonce };
    uint32_t c[8], u[16];
#pragma unroll
    for (int i = 0; i < 8; i++) c[i] = __ldg(&cv[i]);
    blake3_compress(c, b1, 0, 16, BLAKE3_CHUNK_END | BLAKE3_ROOT, k);
    chacha8_block0(k, u);
    int na = 0;
#pragma unroll
    for (int j = 0; j < NEB_NB; j++) {
      const uint32_t s = (uint32_t)(((unsigned long long)u[j] * n) >> 32);
      if ((__ldg(&flags[s >> 5]) >> (s & 31)) & 1u) { na++; sidx = s; }
    }
    sel = na <= 1;
  }
  const uint32_t ball = __ballot_sync(0xffffffffu, sel);
  if ((threadIdx.x & 31) == 0 && ball) atomicAdd(evald, (uint32_t)__popc(ball));
  if (!sel) return;
  uint32_t blk[16], t[8];
#pragma unroll
  for (int i = 0; i < 8; i++) { blk[i] = __ldg(&dig[8 * sidx + i]); blk[8 + i] = k[i]; }
  blake3_compress(blake3_iv(), blk, 0, 64, BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT, t);
  if (DUMP) {
    const uint32_t at = atomicAdd(hits, 1u);
    if (at < max_hits) {
      hits[1 + at] = nonce;
      for (int i = 0; i < 8; i++) hits[1 + max_hits + 8 * at + i] = t[i];
    }
    return;
  }
  bool ok = true;
  for (int i = 7; i >= 0; i--) {
    const uint32_t b = __ldg(&bound[i]);
    if (t[i] != b) { ok = t[i] < b; break; }
  }
  if (ok) {
    const uint32_t at = atomicAdd(hits, 1u);
    if (at < max_hits) hits[1 + at] = nonce;
  }
}

struct neb2t_scan {
  int device, arch;
  // sparse scan: bit s of flags = sample s has a non-zero gradient; dig = 8 digest words per sample alone, then the
  // zero digest at index n
  std::vector<uint32_t> flags, dig;
  uint32_t nactive;
  uint32_t *d_flags, *d_dig, *d_eval;
  bool sm6x;           // compute capability 6.x: the launch shape is measured per model
  int threads;         // threads per attempt for the current model; 0 while measuring
  int tune_warm;       // the next timed launch is the warm-up
  uint32_t tune_n[2];  // attempts timed per shape (THREADS, 32)
  float tune_ms[2];
  cudaEvent_t ev[2];
  int pin;             // >= 0: every launch uses that shape (neb2t_scan_pin_shape), untimed
  TPlan plan;
  uint32_t n, capacity, max_hits;
  std::vector<uint8_t> values, labels, rows;
  uint8_t *d_rows;
  size_t d_rows_bytes;
  uint32_t *d_header, *d_bound, *d_hits, *d_cv;
  uint8_t cached_prefix[64];
  bool prefix_ready;
};

#define TRY(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) return (int)e_; } while (0)

static void make_plan(const neb2_arch *A, const int *wide, const int *narrow8, TPlan *p) {
  memset(p, 0, sizeof *p);
  p->nlayers = A->nlayers;
  int at = 0, sh = 0;
  for (int i = 0; i < A->nlayers; i++) {
    const neb2_layer &L = A->l[i];
    p->nw[i] = L.nw;
    p->cin[i] = L.cin;
    p->cout[i] = L.cout;
    p->at[i] = at;
    if (L.kind == NEB2_LINEAR) {
      p->kind[i] = LINEAR;
      for (int f = 0; f < L.nw; f++)
        p->linear_zero[f & 7] += (uint32_t)f + 7919u * (uint32_t)(i + 1);
      at += L.cout + L.cin;
      p->sh_d[i] = sh; sh += 4 * L.cout;
      p->sh_in[i] = sh; // unused: input words are reloaded per q, not staged
    } else {
      p->kind[i] = wide[i] ? WIDE : (narrow8[i] ? NARROW8 : NARROW);
      at += wide[i] ? 4 * L.nw : (narrow8[i] ? ((L.nw + 3) & ~3) : 2 * L.nw);
    }
    at = (at + 7) & ~7;  // 8-byte aligned layers and rows: the int16 quad path loads 64 bits per sample
  }
  for (int i = 0; i < A->nlayers; i++) {
    const neb2_layer &L = A->l[i];
    p->am[i] = -1;
    if (L.kind == NEB2_LINEAR) continue;
    p->blk[i] = L.cout > 0 ? L.nw / L.cout : L.nw;
    if (L.cout <= 32 && L.cout > 0 && L.nw % L.cout == 0) { p->am[i] = at; at += 4; }
  }
  at = (at + 7) & ~7;
  p->row = at;
  p->shared_words = sh;
}

// Per-sample gradients of one model: the convolution entries as int32 (the table width of each layer is chosen
// afterwards from the range actually seen) and the linear factors (cout errors then cin inputs) as int8.
struct SampleGrads {
  std::vector<int32_t> conv;
  std::vector<int8_t> lin;
  int conv_w = 0, lin_w = 0;
};

// One evaluation of every sample with neb2_sample_table_row of the C reference (neb2_sample_grad without the linear
// gradients, which the row never reads: only their factors), on every host thread, recording the range of each
// convolution layer, so the widths are chosen after a single pass.
static void sample_grads(neb2t_scan *s, const neb2_arch *A, const int8_t *weights, SampleGrads &sg,
                         int32_t lo[NEB2_MAX_LAYERS], int32_t hi[NEB2_MAX_LAYERS]) {
  neb2_model *m = (neb2_model *)calloc(1, sizeof *m);
  m->arch = s->arch;
  memcpy(m->w, weights, (size_t)A->nweights);
  for (int l = 0; l < A->nlayers; l++) {
    if (A->l[l].kind == NEB2_LINEAR) sg.lin_w += A->l[l].cout + A->l[l].cin;
    else sg.conv_w += A->l[l].nw;
  }
  sg.conv.assign((size_t)sg.conv_w * s->n, 0);
  sg.lin.assign((size_t)sg.lin_w * s->n, 0);
  unsigned nt = std::thread::hardware_concurrency();
  if (nt == 0) nt = 4;
  std::vector<int32_t> tlo((size_t)nt * NEB2_MAX_LAYERS, 0), thi((size_t)nt * NEB2_MAX_LAYERS, 0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      std::vector<int32_t> g(NEB2_MAX_WEIGHTS);
      std::vector<int8_t> fac(2 * NEB2_MAX_WEIGHTS);
      int32_t *mylo = tlo.data() + (size_t)t * NEB2_MAX_LAYERS, *myhi = thi.data() + (size_t)t * NEB2_MAX_LAYERS;
      for (uint32_t i = t; i < s->n; i += nt) {
        neb2_sample_table_row(m, s->values.data() + (size_t)i * A->nin, s->labels[i], g.data(), fac.data());
        int32_t *c = sg.conv.data() + (size_t)i * sg.conv_w;
        int8_t *f = sg.lin.data() + (size_t)i * sg.lin_w;
        const int8_t *fp = fac.data();
        for (int l = 0; l < A->nlayers; l++) {
          const neb2_layer &L = A->l[l];
          if (L.kind == NEB2_LINEAR) {
            memcpy(f, fp, (size_t)(L.cout + L.cin));
            f += L.cout + L.cin;
            fp += L.cout + L.cin;
          } else {
            for (int k = 0; k < L.nw; k++) {
              const int32_t v = g[L.w_off + k];
              c[k] = v;
              if (v < mylo[l]) mylo[l] = v;
              if (v > myhi[l]) myhi[l] = v;
            }
            c += L.nw;
          }
        }
      }
    });
  for (auto &x : th) x.join();
  free(m);
  for (int l = 0; l < NEB2_MAX_LAYERS; l++) {
    lo[l] = hi[l] = 0;
    for (unsigned t = 0; t < nt; t++) {
      lo[l] = std::min(lo[l], tlo[(size_t)t * NEB2_MAX_LAYERS + l]);
      hi[l] = std::max(hi[l], thi[(size_t)t * NEB2_MAX_LAYERS + l]);
    }
  }
}

// Writes the row of every sample in the layout of s->plan from the stored gradients, on every host thread.
static void pack_rows(neb2t_scan *s, const neb2_arch *A, const SampleGrads &sg) {
  const TPlan &p = s->plan;
  s->rows.assign((size_t)p.row * s->n, 0);
  unsigned nt = std::thread::hardware_concurrency();
  if (nt == 0) nt = 4;
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      for (uint32_t i = t; i < s->n; i += nt) {
        uint8_t *row = s->rows.data() + (size_t)i * p.row;
        const int32_t *c = sg.conv.data() + (size_t)i * sg.conv_w;
        const int8_t *f = sg.lin.data() + (size_t)i * sg.lin_w;
        for (int l = 0; l < A->nlayers; l++) {
          const neb2_layer &L = A->l[l];
          if (p.kind[l] == LINEAR) {
            memcpy(row + p.at[l], f, (size_t)(L.cout + L.cin));
            f += L.cout + L.cin;
            continue;
          }
          if (p.kind[l] == WIDE) {
            memcpy(row + p.at[l], c, (size_t)4 * L.nw);
          } else if (p.kind[l] == NARROW8) {
            uint8_t *o = row + p.at[l];
            for (int k = 0; k < L.nw; k++) o[k] = (uint8_t)(int8_t)c[k];
          } else {
            int16_t *o = (int16_t *)(row + p.at[l]);
            for (int k = 0; k < L.nw; k++) o[k] = (int16_t)c[k];
          }
          if (p.am[l] >= 0) {
            uint32_t act = 0;
            for (int ch = 0; ch < L.cout; ch++)
              for (int k = 0; k < p.blk[l]; k++)
                if (c[ch * p.blk[l] + k]) { act |= 1u << ch; break; }
            memcpy(row + p.am[l], &act, 4);
          }
          c += L.nw;
        }
      }
    });
  for (auto &x : th) x.join();
}

// Sparse scan tables: the active flag of every sample, the digest of every active sample alone and the zero digest
// (index n), as the reference computes them: lane f & 7, f the flat index in the layer, entry clip(G) ^ (f + 7919 *
// (layer + 1)). An entry of a linear layer is d[o] * x[q].
static void sparse_digests(neb2t_scan *s, const neb2_arch *A, const SampleGrads &sg) {
  const uint32_t n = s->n;
  s->flags.assign((n + 31) / 32, 0u);
  s->dig.assign((size_t)8 * (n + 1), 0u);
  std::vector<uint8_t> act(n, 0);
  unsigned nt = std::thread::hardware_concurrency();
  if (nt == 0) nt = 4;
  // the digest of one sample, or of the zero gradient (i == n)
  auto digest = [&](uint32_t i, uint32_t *lane) {
    const int32_t *c = i < n ? sg.conv.data() + (size_t)i * sg.conv_w : NULL;
    const int8_t *fa = i < n ? sg.lin.data() + (size_t)i * sg.lin_w : NULL;
    for (int k = 0; k < 8; k++) lane[k] = 0;
    for (int l = 0; l < A->nlayers; l++) {
      const neb2_layer &L = A->l[l];
      const uint32_t base = 7919u * (uint32_t)(l + 1);
      if (L.kind == NEB2_LINEAR) {
        for (int o = 0; o < L.cout; o++)
          for (int q = 0; q < L.cin; q++) {
            const uint32_t f = (uint32_t)(o * L.cin + q);
            int G = fa ? (int)fa[o] * (int)fa[L.cout + q] : 0;
            G = G < -CLIPV ? -CLIPV : (G > CLIPV ? CLIPV : G);
            lane[f & 7] += (uint32_t)G ^ (f + base);
          }
        if (fa) fa += L.cout + L.cin;
      } else {
        for (int f = 0; f < L.nw; f++) {
          int G = c ? c[f] : 0;
          G = G < -CLIPV ? -CLIPV : (G > CLIPV ? CLIPV : G);
          lane[f & 7] += (uint32_t)G ^ ((uint32_t)f + base);
        }
        if (c) c += L.nw;
      }
    }
  };
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      for (uint32_t i = t; i < n; i += nt) {
        const int32_t *c = sg.conv.data() + (size_t)i * sg.conv_w;
        bool a = false;
        for (int k = 0; k < sg.conv_w && !a; k++) a = c[k] != 0;
        const int8_t *fa = sg.lin.data() + (size_t)i * sg.lin_w;
        for (int l = 0; l < A->nlayers && !a; l++) {
          const neb2_layer &L = A->l[l];
          if (L.kind != NEB2_LINEAR) continue;
          bool dz = false, xz = false;
          for (int o = 0; o < L.cout; o++) dz |= fa[o] != 0;
          for (int q = 0; q < L.cin; q++) xz |= fa[L.cout + q] != 0;
          a = dz && xz;
          fa += L.cout + L.cin;
        }
        act[i] = a;
        if (a) digest(i, &s->dig[(size_t)8 * i]);
      }
    });
  for (auto &x : th) x.join();
  digest(n, &s->dig[(size_t)8 * n]);
  s->nactive = 0;
  for (uint32_t i = 0; i < n; i++) {
    if (act[i]) { s->flags[i >> 5] |= 1u << (i & 31); s->nactive++; }
    else memcpy(&s->dig[(size_t)8 * i], &s->dig[(size_t)8 * n], 32);
  }
}

static int build(neb2t_scan *s, const int8_t *weights) {
  const neb2_arch *A = neb2_arch_get(s->arch);
  // the kernel lists the active outputs of a linear layer in a fixed shared array of 32
  for (int i = 0; i < A->nlayers; i++)
    if (A->l[i].kind == NEB2_LINEAR && A->l[i].cout > 32) return (int)cudaErrorInvalidValue;
  SampleGrads sg;
  int32_t lo[NEB2_MAX_LAYERS], hi[NEB2_MAX_LAYERS];
  sample_grads(s, A, weights, sg, lo, hi);
  // narrowest width that holds every entry of the layer: int8 (dp4a path), else int16, else int32
  int wide[NEB2_MAX_LAYERS] = {0}, n8[NEB2_MAX_LAYERS] = {0};
  for (int i = 0; i < A->nlayers; i++) {
    if (A->l[i].kind == NEB2_LINEAR) continue;
    if (lo[i] >= -128 && hi[i] <= 127) n8[i] = 1;
    else if (lo[i] < -32768 || hi[i] > 32767) wide[i] = 1;
  }
  make_plan(A, wide, n8, &s->plan);
  s->threads = s->sm6x ? 0 : THREADS;
  s->tune_warm = 1;
  s->tune_n[0] = s->tune_n[1] = 0;
  s->tune_ms[0] = s->tune_ms[1] = 0.f;
  s->pin = -1;
  pack_rows(s, A, sg);
  sparse_digests(s, A, sg);
  if (s->rows.size() > UINT32_MAX) return (int)cudaErrorInvalidValue;
  TRY(cudaSetDevice(s->device));
  if (!s->d_flags) TRY(cudaMalloc(&s->d_flags, s->flags.size() * 4));
  if (!s->d_dig) TRY(cudaMalloc(&s->d_dig, s->dig.size() * 4));
  TRY(cudaMemcpy(s->d_flags, s->flags.data(), s->flags.size() * 4, cudaMemcpyHostToDevice));
  TRY(cudaMemcpy(s->d_dig, s->dig.data(), s->dig.size() * 4, cudaMemcpyHostToDevice));
  if (s->rows.size() != s->d_rows_bytes) {
    cudaFree(s->d_rows);
    s->d_rows = NULL;
    TRY(cudaMalloc(&s->d_rows, s->rows.size()));
    s->d_rows_bytes = s->rows.size();
  }
  return (int)cudaMemcpy(s->d_rows, s->rows.data(), s->rows.size(), cudaMemcpyHostToDevice);
}

extern "C" {

int neb2t_scan_open(int device, int arch, const uint8_t *values, const uint8_t *labels, uint32_t n, const int8_t *weights,
                   uint32_t capacity, uint32_t max_hits, neb2t_scan **out) {
  const neb2_arch *A = neb2_arch_get(arch);
  if (!A || n == 0 || capacity == 0 || max_hits == 0) return (int)cudaErrorInvalidValue;
  neb2t_scan *s = new neb2t_scan();
  s->device = device;
  s->arch = arch;
  s->n = n;
  s->capacity = capacity;
  s->max_hits = max_hits;
  s->values.assign(values, values + (size_t)n * A->nin);
  s->labels.assign(labels, labels + n);
  s->d_rows = NULL;
  s->d_rows_bytes = 0;
  s->d_header = s->d_bound = s->d_hits = NULL;
  s->d_cv = NULL;
  s->d_flags = s->d_dig = s->d_eval = NULL;
  s->nactive = 0;
  s->ev[0] = s->ev[1] = NULL;
  memset(&s->plan, 0, sizeof s->plan);
  cudaError_t e = cudaSetDevice(device);
  int major = 0;
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  s->sm6x = major == 6;
  if (e == cudaSuccess && s->sm6x) e = cudaEventCreate(&s->ev[0]);
  if (e == cudaSuccess && s->sm6x) e = cudaEventCreate(&s->ev[1]);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_header, 80);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_cv, 32);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_bound, 32);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_hits, ((size_t)max_hits + 1) * 4);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_eval, 4);
  int st = (int)e;
  if (st == 0) st = build(s, weights);
  if (st != 0) {
    neb2t_scan_close(s);
    return st;
  }
  *out = s;
  return 0;
}

int neb2t_scan_set_model(neb2t_scan *s, const int8_t *weights) { return build(s, weights); }

// header, its BLAKE3 prefix (cached), the bound and a cleared hit count
static int upload_job(neb2t_scan *s, const uint8_t *header, const uint8_t *bound) {
  TRY(cudaSetDevice(s->device));
  TRY(cudaMemcpy(s->d_header, header, 80, cudaMemcpyHostToDevice));
  if (!s->prefix_ready || memcmp(s->cached_prefix, header, 64) != 0) {
    header_prefix<<<1, 1>>>(s->d_header, s->d_cv);
    TRY(cudaGetLastError());
    memcpy(s->cached_prefix, header, 64);
    s->prefix_ready = true;
  }
  if (bound) TRY(cudaMemcpy(s->d_bound, bound, 32, cudaMemcpyHostToDevice));
  TRY(cudaMemset(s->d_hits, 0, 4));
  return 0;
}

#define SPARSE_CHUNK (1u << 22)  // nonces per sparse launch

int neb2t_scan_run_sparse(neb2t_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound,
                          uint32_t *hits, uint32_t *nhits, uint32_t *evaluated) {
  if (count == 0) return (int)cudaErrorInvalidValue;
  const int st = upload_job(s, header, bound);
  if (st) return st;
  TRY(cudaMemset(s->d_eval, 0, 4));
  for (uint32_t done = 0; done < count;) {
    const uint32_t part = count - done < SPARSE_CHUNK ? count - done : SPARSE_CHUNK;
    sparse_kernel<false><<<(part + 255) / 256, 256>>>(s->d_flags, s->d_dig, s->n, s->d_header, start + done, part,
                                                      s->d_bound, s->d_hits, s->max_hits, s->d_cv, s->d_eval);
    TRY(cudaGetLastError());
    done += part;
  }
  uint32_t found = 0;
  TRY(cudaMemcpy(&found, s->d_hits, 4, cudaMemcpyDeviceToHost));
  TRY(cudaMemcpy(evaluated, s->d_eval, 4, cudaMemcpyDeviceToHost));
  *nhits = found;
  if (found > 0)
    TRY(cudaMemcpy(hits, s->d_hits + 1, (size_t)(found < s->max_hits ? found : s->max_hits) * 4, cudaMemcpyDeviceToHost));
  return 0;
}

int neb2t_scan_sparse_dump(neb2t_scan *s, const uint8_t *header, uint32_t start, uint32_t count, uint32_t max,
                           uint32_t *nonces, uint8_t *tickets, uint32_t *ndump) {
  if (count == 0 || max == 0) return (int)cudaErrorInvalidValue;
  const int st = upload_job(s, header, NULL);
  if (st) return st;
  uint32_t *d = NULL;
  TRY(cudaMalloc(&d, ((size_t)9 * max + 1) * 4));
  cudaError_t e = cudaMemset(d, 0, 4);
  if (e == cudaSuccess) e = cudaMemset(s->d_eval, 0, 4);
  if (e == cudaSuccess) {
    sparse_kernel<true><<<(count + 255) / 256, 256>>>(s->d_flags, s->d_dig, s->n, s->d_header, start, count, s->d_bound,
                                                      d, max, s->d_cv, s->d_eval);
    e = cudaGetLastError();
  }
  uint32_t found = 0;
  if (e == cudaSuccess) e = cudaMemcpy(&found, d, 4, cudaMemcpyDeviceToHost);
  if (found > max) found = max;
  if (e == cudaSuccess && found) e = cudaMemcpy(nonces, d + 1, (size_t)found * 4, cudaMemcpyDeviceToHost);
  if (e == cudaSuccess && found) e = cudaMemcpy(tickets, d + 1 + max, (size_t)found * 32, cudaMemcpyDeviceToHost);
  cudaFree(d);
  *ndump = found;
  return (int)e;
}

uint32_t neb2t_scan_active(const neb2t_scan *s) { return s->nactive; }

int neb2t_scan_run(neb2t_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound, uint32_t *hits,
                  uint32_t *nhits) {
  if (count == 0 || count > s->capacity) return (int)cudaErrorInvalidValue;
  TRY(cudaSetDevice(s->device));
  TRY(cudaMemcpyToSymbol(P, &s->plan, sizeof s->plan));
  const int st = upload_job(s, header, bound);
  if (st) return st;
  for (uint32_t done = 0; done < count; done += CHUNK) {
    const uint32_t part = count - done < CHUNK ? count - done : CHUNK;
    const size_t shm = (size_t)s->plan.shared_words * 4;
    const bool timed = s->pin < 0 && s->threads == 0 && part >= TUNE_MIN;
    const int k = s->pin >= 0 ? s->pin
                : s->threads ? (s->threads == 32) : (s->tune_n[1] < s->tune_n[0]);  // the less measured shape
    if (timed) TRY(cudaEventRecord(s->ev[0]));
    if (k)
      table_kernel<32><<<part, 32, shm>>>(s->d_rows, s->n, s->d_header, start + done, s->d_bound, s->d_hits,
                                          s->max_hits, s->d_cv);
    else
      table_kernel<THREADS><<<part, THREADS, shm>>>(s->d_rows, s->n, s->d_header, start + done, s->d_bound, s->d_hits,
                                                    s->max_hits, s->d_cv);
    TRY(cudaGetLastError());
    if (timed) {
      float ms = 0.f;
      TRY(cudaEventRecord(s->ev[1]));
      TRY(cudaEventSynchronize(s->ev[1]));
      TRY(cudaEventElapsedTime(&ms, s->ev[0], s->ev[1]));
      if (s->tune_warm) {
        s->tune_warm = 0;
      } else {
        s->tune_ms[k] += ms;
        s->tune_n[k] += part;
        if (s->tune_n[0] >= TUNE_ATTEMPTS && s->tune_n[1] >= TUNE_ATTEMPTS)
          s->threads = s->tune_ms[1] / s->tune_n[1] < s->tune_ms[0] / s->tune_n[0] ? 32 : THREADS;
      }
    }
  }
  uint32_t found = 0;
  TRY(cudaMemcpy(&found, s->d_hits, 4, cudaMemcpyDeviceToHost));
  *nhits = found;
  if (found > 0)
    TRY(cudaMemcpy(hits, s->d_hits + 1, (size_t)(found < s->max_hits ? found : s->max_hits) * 4, cudaMemcpyDeviceToHost));
  return 0;
}

int neb2t_scan_threads(const neb2t_scan *s) { return s->threads; }
int neb2t_scan_shapes(const neb2t_scan *s) { return s->sm6x ? 2 : 1; }
void neb2t_scan_pin_shape(neb2t_scan *s, int i) { s->pin = i >= 0 && i < neb2t_scan_shapes(s) ? i : -1; }

void neb2t_scan_close(neb2t_scan *s) {
  if (!s) return;
  cudaSetDevice(s->device);
  if (s->ev[0]) cudaEventDestroy(s->ev[0]);
  if (s->ev[1]) cudaEventDestroy(s->ev[1]);
  cudaFree(s->d_rows);
  cudaFree(s->d_header);
  cudaFree(s->d_cv);
  cudaFree(s->d_bound);
  cudaFree(s->d_hits);
  cudaFree(s->d_flags);
  cudaFree(s->d_dig);
  cudaFree(s->d_eval);
  delete s;
}

}  // extern "C"
