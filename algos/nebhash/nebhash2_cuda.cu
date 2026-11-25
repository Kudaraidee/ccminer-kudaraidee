// nebhash v2 attempts on CUDA, one thread block per attempt: the contract is in nebhash2_cuda.h, the arithmetic is the
// one of pow/ref/nebhash2.c (spec/nebhash-v2.md section 4). For each of the 16 samples the block runs the forward and
// backward passes layer by layer, all threads sharing the positions of a layer, and keeps the inputs and the routed
// errors of every layer in a scratch area of the device; then each thread computes whole entries of the gradient
// (no atomics on the gradient, so no order effect), clips them and adds them to the digest lanes, which are sums modulo
// 2^32. The architecture table comes from the C reference (neb2_arch_get) and is copied to constant memory.
#include "nebhash2_cuda.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include "nebhash2.h"

#define THREADS 256
#define BATCH 512  // attempts per launch: sets the scratch size

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ __forceinline__ int rsh(int x, int n) { return n == 0 ? x : (x + (1 << (n - 1))) >> n; }

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

static __device__ __constant__ uint32_t kIV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                           0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

static __device__ __forceinline__ void g3(uint32_t *v, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  v[a] = v[a] + v[b] + mx; v[d] = rotr32(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];      v[b] = rotr32(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + my; v[d] = rotr32(v[d] ^ v[a], 8);
  v[c] = v[c] + v[d];      v[b] = rotr32(v[b] ^ v[c], 7);
}

// One BLAKE3 compression with chunk counter 0 (single-chunk inputs only). flags: CHUNK_START 1, CHUNK_END 2, ROOT 8.
static __device__ void b3_compress(const uint32_t cv[8], const uint32_t blk[16], uint32_t blen, uint32_t flags, uint32_t out[8]) {
  const uint8_t perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t m[16], v[16];
  for (int i = 0; i < 16; i++) m[i] = blk[i];
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = kIV[0]; v[9] = kIV[1]; v[10] = kIV[2]; v[11] = kIV[3];
  v[12] = 0; v[13] = 0; v[14] = blen; v[15] = flags;
  for (int r = 0; r < 7; r++) {
    g3(v, 0, 4, 8, 12, m[0], m[1]);   g3(v, 1, 5, 9, 13, m[2], m[3]);
    g3(v, 2, 6, 10, 14, m[4], m[5]);  g3(v, 3, 7, 11, 15, m[6], m[7]);
    g3(v, 0, 5, 10, 15, m[8], m[9]);  g3(v, 1, 6, 11, 12, m[10], m[11]);
    g3(v, 2, 7, 8, 13, m[12], m[13]); g3(v, 3, 4, 9, 14, m[14], m[15]);
    if (r < 6) {
      uint32_t t[16];
      for (int i = 0; i < 16; i++) t[i] = m[perm[i]];
      for (int i = 0; i < 16; i++) m[i] = t[i];
    }
  }
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

// The architecture and the scratch layout, in constant memory. Per sample: the input `in` (int8) and the routed error
// `d` (int8, on the convolution grid) of every layer, kept for the gradient phase; shared by the samples: the clipped
// activations with a flag for a clipped value (`a`: low nibble the value, bit 4 set when y was outside -8..7), the
// pooled outputs and their errors.
struct Plan {
  neb2_arch A;
  int nin_in[NEB2_MAX_LAYERS], nconv[NEB2_MAX_LAYERS], nout[NEB2_MAX_LAYERS];
  int off_in[NEB2_MAX_LAYERS], off_d[NEB2_MAX_LAYERS];  // inside one sample's part
  int off_a[NEB2_MAX_LAYERS], off_out[NEB2_MAX_LAYERS], off_e[NEB2_MAX_LAYERS];  // inside the common part
  int sample_bytes, common_bytes, block_bytes;
};
static __device__ __constant__ Plan P;

__device__ __forceinline__ int sq(int kind, int n) { return kind == NEB2_CONV2D ? n * n : n; }

// The first maximum of the pooling window of grid position g, or -1 when g is in no full window; sets *j to the pooled
// index of the window.
static __device__ int window_argmax(const neb2_layer &L, const int8_t *a, int g, int *j) {
  const int cl = L.conv_len, ol = L.out_len, p = L.pool;
  if (L.kind == NEB2_CONV1D) {
    const int o = g / cl, pos = g % cl, w = pos / p;
    if (w >= ol) return -1;
    const int s = o * cl + w * p;
    int best = s;
    for (int t = 1; t < p; t++)
      if ((int8_t)(a[s + t] << 4) > (int8_t)(a[best] << 4)) best = s + t;
    *j = o * ol + w;
    return best;
  }
  const int o = g / (cl * cl), py = (g / cl) % cl, px = g % cl, wy = py / p, wx = px / p;
  if (wy >= ol || wx >= ol) return -1;
  int best = (o * cl + wy * p) * cl + wx * p;
  for (int ty = 0; ty < p; ty++)
    for (int tx = 0; tx < p; tx++) {
      const int q = (o * cl + wy * p + ty) * cl + wx * p + tx;
      if ((int8_t)(a[q] << 4) > (int8_t)(a[best] << 4)) best = q;
    }
  *j = (o * ol + wy) * ol + wx;
  return best;
}

__global__ void __launch_bounds__(THREADS) neb2_kernel(const int8_t *__restrict__ wg, const uint8_t *__restrict__ data,
                                                       const uint8_t *__restrict__ labels, uint32_t n, uint8_t *scratch,
                                                       uint32_t h0[20], uint32_t start, uint32_t count,
                                                       const uint32_t *bound, uint32_t *hits, uint32_t max_hits) {
  extern __shared__ int8_t W[];  // the weights, then the per-block values below
  __shared__ uint32_t key[8], idx[NEB_NB], lanes[8];
  __shared__ int label[NEB_NB];
  const int tid = threadIdx.x;
  if (blockIdx.x >= count) return;
  const uint32_t nonce = start + blockIdx.x;
  const neb2_arch &A = P.A;
  for (int f = tid; f < A.nweights; f += THREADS) W[f] = wg[f];
  if (tid < 8) lanes[tid] = 0;
  if (tid == 0) {
    uint32_t b0[16], b1[16], iv[8], cv[8], k[8], u[16];
    for (int i = 0; i < 8; i++) iv[i] = kIV[i];
    for (int i = 0; i < 16; i++) { b0[i] = h0[i]; b1[i] = i < 4 ? h0[16 + i] : 0u; }
    b1[3] = nonce;  // header word 19: the nonce
    b3_compress(iv, b0, 64, 1u, cv);
    b3_compress(cv, b1, 16, 2u | 8u, k);
    for (int i = 0; i < 8; i++) key[i] = k[i];
    chacha8_block0(k, u);
    for (int j = 0; j < NEB_NB; j++) {
      idx[j] = (uint32_t)(((unsigned long long)u[j] * n) >> 32);
      label[j] = labels[idx[j]];
    }
  }
  __syncthreads();
  uint8_t *base = scratch + (size_t)blockIdx.x * P.block_bytes;
  uint8_t *common = base + (size_t)NEB_NB * P.sample_bytes;
  const int last = A.nlayers - 1;
  for (int b = 0; b < NEB_NB; b++) {
    uint8_t *smp = base + (size_t)b * P.sample_bytes;
    const uint8_t *x = data + (size_t)idx[b] * A.nin;
    // forward
    for (int i = 0; i < A.nlayers; i++) {
      const neb2_layer &L = A.l[i];
      int8_t *in = (int8_t *)(smp + P.off_in[i]);
      int8_t *a = (int8_t *)(common + P.off_a[i]);
      int8_t *out = (int8_t *)(common + P.off_out[i]);
      if (L.nsrc == 0) {
        for (int q = tid; q < P.nin_in[i]; q += THREADS) in[q] = (int8_t)x[L.in_off + q];
      } else {
        int at = 0;
        for (int s = 0; s < L.nsrc; s++) {
          const int src = L.src[s];
          const int8_t *so = (const int8_t *)(common + P.off_out[src]);
          for (int q = tid; q < P.nout[src]; q += THREADS) in[at + q] = so[q];
          at += P.nout[src];
        }
      }
      __syncthreads();
      const int8_t *w = W + L.w_off;
      for (int g = tid; g < P.nconv[i]; g += THREADS) {
        int z = 0;
        if (L.kind == NEB2_LINEAR) {
          const int8_t *wr = w + g * L.cin;
          for (int q = 0; q < L.cin; q++) z += wr[q] * in[q];
        } else if (L.kind == NEB2_CONV1D) {
          const int cl = L.conv_len, o = g / cl, p = g % cl, nn = L.in_len;
          for (int c = 0; c < L.cin; c++) {
            const int8_t *wr = w + (o * L.cin + c) * L.k;
            for (int t = 0; t < L.k; t++) {
              const int q = p * L.stride + t - L.pad;
              if (q >= 0 && q < nn) z += wr[t] * in[c * nn + q];
            }
          }
        } else {
          const int cl = L.conv_len, o = g / (cl * cl), py = (g / cl) % cl, px = g % cl, nn = L.in_len, k = L.k;
          for (int c = 0; c < L.cin; c++) {
            const int8_t *wr = w + (o * L.cin + c) * k * k;
            for (int ky = 0; ky < k; ky++) {
              const int qy = py * L.stride + ky - L.pad;
              if (qy < 0 || qy >= nn) continue;
              for (int kx = 0; kx < k; kx++) {
                const int qx = px * L.stride + kx - L.pad;
                if (qx >= 0 && qx < nn) z += wr[ky * k + kx] * in[(c * nn + qy) * nn + qx];
              }
            }
          }
        }
        const int y = rsh(z, L.s);
        const int v = clampi(y, -8, 7);
        a[g] = (int8_t)((v & 15) | ((y < -8 || y > 7) ? 16 : 0));
      }
      __syncthreads();
      for (int j = tid; j < P.nout[i]; j += THREADS) {
        if (L.pool <= 1) {
          out[j] = (int8_t)(a[j] << 4) >> 4;
          continue;
        }
        // the window of pooled index j and its first maximum
        int g0;
        if (L.kind == NEB2_CONV1D) {
          g0 = (j / L.out_len) * L.conv_len + (j % L.out_len) * L.pool;
        } else {
          const int ol = L.out_len, o = j / (ol * ol), jy = (j / ol) % ol, jx = j % ol;
          g0 = (o * L.conv_len + jy * L.pool) * L.conv_len + jx * L.pool;
        }
        int jj = 0;
        const int best = window_argmax(L, a, g0, &jj);
        out[j] = (int8_t)(a[best] << 4) >> 4;
      }
      __syncthreads();
    }
    // the error of the output layer
    {
      int8_t *e = (int8_t *)(common + P.off_e[last]);
      const int8_t *out = (const int8_t *)(common + P.off_out[last]);
      if (tid < NEB2_NOUT) e[tid] = (int8_t)clampi(out[tid] - (tid == label[b] ? NEB_TPOS : NEB_TNEG), -8, 7);
    }
    __syncthreads();
    // backward: route, mask (not on the output layer), and the error of the sources
    for (int i = last; i >= 0; i--) {
      const neb2_layer &L = A.l[i];
      const int8_t *a = (const int8_t *)(common + P.off_a[i]);
      const int8_t *e = (const int8_t *)(common + P.off_e[i]);
      int8_t *d = (int8_t *)(smp + P.off_d[i]);
      for (int g = tid; g < P.nconv[i]; g += THREADS) {
        int v = 0;
        if (L.pool <= 1) {
          v = e[g];
        } else {
          int j = 0;
          if (window_argmax(L, a, g, &j) == g) v = e[j];
        }
        if (i != last && (a[g] & 16)) v = 0;
        d[g] = (int8_t)v;
      }
      __syncthreads();
      if (L.nsrc > 0) {
        const int8_t *w = W + L.w_off;
        const int8_t *dd = d;
        for (int q = tid; q < P.nin_in[i]; q += THREADS) {
          int m = 0;
          if (L.kind == NEB2_LINEAR) {
            for (int o = 0; o < L.cout; o++) m += w[o * L.cin + q] * dd[o];
          } else if (L.kind == NEB2_CONV1D) {
            const int nn = L.in_len, c = q / nn, t = q % nn, cl = L.conv_len;
            for (int o = 0; o < L.cout; o++)
              for (int kk = 0; kk < L.k; kk++) {
                const int pp = t + L.pad - kk;
                if (pp < 0 || pp % L.stride) continue;
                const int p = pp / L.stride;
                if (p < cl) m += w[(o * L.cin + c) * L.k + kk] * dd[o * cl + p];
              }
          } else {
            const int nn = L.in_len, c = q / (nn * nn), ty = (q / nn) % nn, tx = q % nn, cl = L.conv_len, k = L.k;
            for (int o = 0; o < L.cout; o++)
              for (int ky = 0; ky < k; ky++) {
                const int ppy = ty + L.pad - ky;
                if (ppy < 0 || ppy % L.stride) continue;
                const int py = ppy / L.stride;
                if (py >= cl) continue;
                for (int kx = 0; kx < k; kx++) {
                  const int ppx = tx + L.pad - kx;
                  if (ppx < 0 || ppx % L.stride) continue;
                  const int px = ppx / L.stride;
                  if (px < cl) m += w[((o * L.cin + c) * k + ky) * k + kx] * dd[(o * cl + py) * cl + px];
                }
              }
          }
          // q is an index of the concatenated input: find its source
          int at = 0, s = 0;
          while (q >= at + P.nout[L.src[s]]) { at += P.nout[L.src[s]]; s++; }
          int8_t *es = (int8_t *)(common + P.off_e[L.src[s]]);
          es[q - at] = (int8_t)clampi(rsh(m, L.sb), -8, 7);
        }
      }
      __syncthreads();
    }
  }
  // gradient, clip and digest: thread tid computes the entries f = tid, tid + THREADS, ...
  uint32_t lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < A.nlayers; i++) {
    const neb2_layer &L = A.l[i];
    for (int f = tid; f < L.nw; f += THREADS) {
      int G = 0;
      for (int b = 0; b < NEB_NB; b++) {
        const uint8_t *smp = base + (size_t)b * P.sample_bytes;
        const int8_t *in = (const int8_t *)(smp + P.off_in[i]);
        const int8_t *d = (const int8_t *)(smp + P.off_d[i]);
        if (L.kind == NEB2_LINEAR) {
          G += d[f / L.cin] * in[f % L.cin];
        } else if (L.kind == NEB2_CONV1D) {
          const int o = f / (L.cin * L.k), c = (f / L.k) % L.cin, kk = f % L.k, cl = L.conv_len, nn = L.in_len;
          for (int p = 0; p < cl; p++) {
            const int q = p * L.stride + kk - L.pad;
            if (q >= 0 && q < nn) G += d[o * cl + p] * in[c * nn + q];
          }
        } else {
          const int k = L.k, o = f / (L.cin * k * k), c = (f / (k * k)) % L.cin, ky = (f / k) % k, kx = f % k;
          const int cl = L.conv_len, nn = L.in_len;
          for (int py = 0; py < cl; py++) {
            const int qy = py * L.stride + ky - L.pad;
            if (qy < 0 || qy >= nn) continue;
            for (int px = 0; px < cl; px++) {
              const int qx = px * L.stride + kx - L.pad;
              if (qx >= 0 && qx < nn) G += d[(o * cl + py) * cl + px] * in[(c * nn + qy) * nn + qx];
            }
          }
        }
      }
      const int cg = clampi(G, -NEB_CLIP, NEB_CLIP);
      lane[f & 7] += (uint32_t)cg ^ ((uint32_t)f + 7919u * (uint32_t)(i + 1));
    }
  }
  for (int k = 0; k < 8; k++) atomicAdd(&lanes[k], lane[k]);
  __syncthreads();
  if (tid == 0) {
    // ticket = BLAKE3(digest || key): one 64-byte block, CHUNK_START | CHUNK_END | ROOT
    uint32_t blk[16], iv[8], t[8];
    for (int i = 0; i < 8; i++) { blk[i] = lanes[i]; blk[8 + i] = key[i]; iv[i] = kIV[i]; }
    b3_compress(iv, blk, 64, 1u | 2u | 8u, t);
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

struct neb2_scan {
  int device;
  Plan plan;
  int nweights;
  uint32_t n, capacity, max_hits;
  int8_t *d_w;
  uint8_t *d_data, *d_labels, *d_scratch;
  uint32_t *d_header, *d_bound, *d_hits;
};

#define TRY(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) return (int)e_; } while (0)

static int make_plan(int arch, Plan *p) {
  const neb2_arch *A = neb2_arch_get(arch);
  if (!A) return -1;
  memset(p, 0, sizeof *p);
  p->A = *A;
  int at = 0, common = 0;
  for (int i = 0; i < A->nlayers; i++) {
    const neb2_layer &L = A->l[i];
    p->nin_in[i] = L.kind == NEB2_LINEAR ? L.cin : L.cin * (L.kind == NEB2_CONV2D ? L.in_len * L.in_len : L.in_len);
    p->nconv[i] = L.cout * (L.kind == NEB2_CONV2D ? L.conv_len * L.conv_len : L.conv_len);
    p->nout[i] = L.cout * (L.kind == NEB2_CONV2D ? L.out_len * L.out_len : L.out_len);
    p->off_in[i] = at; at += (p->nin_in[i] + 15) & ~15;
    p->off_d[i] = at; at += (p->nconv[i] + 15) & ~15;
    p->off_a[i] = common; common += (p->nconv[i] + 15) & ~15;
    p->off_out[i] = common; common += (p->nout[i] + 15) & ~15;
    p->off_e[i] = common; common += (p->nout[i] + 15) & ~15;
  }
  p->sample_bytes = at;
  p->common_bytes = common;
  p->block_bytes = (NEB_NB * at + common + 255) & ~255;
  return 0;
}

extern "C" {

int neb2_scan_open(int device, int arch, const uint8_t *values, const uint8_t *labels, uint32_t n, const int8_t *weights,
                   uint32_t capacity, uint32_t max_hits, neb2_scan **out) {
  if (n == 0 || capacity == 0 || max_hits == 0) return (int)cudaErrorInvalidValue;
  neb2_scan *s = (neb2_scan *)calloc(1, sizeof *s);
  if (!s) return (int)cudaErrorMemoryAllocation;
  if (make_plan(arch, &s->plan) != 0) { free(s); return (int)cudaErrorInvalidValue; }
  s->device = device;
  s->nweights = s->plan.A.nweights;
  s->n = n;
  s->capacity = capacity;
  s->max_hits = max_hits;
  const size_t nin = (size_t)s->plan.A.nin;
  cudaError_t e = cudaSetDevice(device);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_w, (size_t)s->nweights);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_data, n * nin);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_labels, n);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_scratch, (size_t)BATCH * s->plan.block_bytes);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_header, 80);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_bound, 32);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_hits, ((size_t)max_hits + 1) * 4);
  if (e == cudaSuccess) e = cudaMemcpy(s->d_data, values, n * nin, cudaMemcpyHostToDevice);
  if (e == cudaSuccess) e = cudaMemcpy(s->d_labels, labels, n, cudaMemcpyHostToDevice);
  if (e == cudaSuccess) e = cudaMemcpy(s->d_w, weights, (size_t)s->nweights, cudaMemcpyHostToDevice);
  if (e == cudaSuccess)
    e = cudaFuncSetAttribute(neb2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NEB2_MAX_WEIGHTS);
  if (e != cudaSuccess) {
    neb2_scan_close(s);
    return (int)e;
  }
  *out = s;
  return 0;
}

int neb2_scan_set_model(neb2_scan *s, const int8_t *weights) {
  TRY(cudaSetDevice(s->device));
  return (int)cudaMemcpy(s->d_w, weights, (size_t)s->nweights, cudaMemcpyHostToDevice);
}

int neb2_scan_run(neb2_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound, uint32_t *hits,
                  uint32_t *nhits) {
  if (count == 0 || count > s->capacity) return (int)cudaErrorInvalidValue;
  TRY(cudaSetDevice(s->device));
  TRY(cudaMemcpyToSymbol(P, &s->plan, sizeof s->plan));
  TRY(cudaMemcpy(s->d_header, header, 80, cudaMemcpyHostToDevice));
  TRY(cudaMemcpy(s->d_bound, bound, 32, cudaMemcpyHostToDevice));
  TRY(cudaMemset(s->d_hits, 0, 4));
  for (uint32_t done = 0; done < count; done += BATCH) {
    const uint32_t part = count - done < BATCH ? count - done : BATCH;
    neb2_kernel<<<part, THREADS, (size_t)s->nweights>>>(s->d_w, s->d_data, s->d_labels, s->n, s->d_scratch, s->d_header,
                                                         start + done, part, s->d_bound, s->d_hits, s->max_hits);
    TRY(cudaGetLastError());
  }
  uint32_t found = 0;
  TRY(cudaMemcpy(&found, s->d_hits, 4, cudaMemcpyDeviceToHost));
  *nhits = found;
  if (found > 0)
    TRY(cudaMemcpy(hits, s->d_hits + 1, (size_t)(found < s->max_hits ? found : s->max_hits) * 4, cudaMemcpyDeviceToHost));
  return 0;
}

void neb2_scan_close(neb2_scan *s) {
  if (!s) return;
  cudaSetDevice(s->device);
  cudaFree(s->d_w);
  cudaFree(s->d_data);
  cudaFree(s->d_labels);
  cudaFree(s->d_scratch);
  cudaFree(s->d_header);
  cudaFree(s->d_bound);
  cudaFree(s->d_hits);
  free(s);
}

}  // extern "C"
