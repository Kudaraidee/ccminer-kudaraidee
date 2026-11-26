// nebhash v2 attempts on CUDA from a table of per-sample gradients. The contract is in nebhash2_table_cuda.h.
//
// The gradient of an attempt is the sum of the gradients of its 16 samples, clipped afterwards, and the gradient of a
// sample depends on the model and on that sample only. For one model state the host evaluates every sample once with
// neb2_sample_grad of the C reference and uploads one row per sample. A row holds, per layer: every gradient entry of
// a convolution (16 bits, or 32 when a value does not fit), and for a linear layer its cout errors then its cin
// inputs, whose outer product is the gradient of the layer. One thread block evaluates one attempt: it gathers the
// factors of its 16 rows, transposed so that one entry of a linear layer is four dp4a, sums, clips, and hashes.
#include "nebhash2_table_cuda.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>
#include "nebhash2.h"

#ifndef THREADS
#define THREADS 256
#endif
#define CHUNK 65536u  // attempts per launch

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

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

enum { NARROW = 0, WIDE = 1, LINEAR = 2 };

// The layout of a row and of the shared memory of a block.
struct TPlan {
  int nlayers;
  int kind[NEB2_MAX_LAYERS], nw[NEB2_MAX_LAYERS], cin[NEB2_MAX_LAYERS], cout[NEB2_MAX_LAYERS];
  int at[NEB2_MAX_LAYERS];                             // offset of the layer in a row
  int sh_d[NEB2_MAX_LAYERS], sh_in[NEB2_MAX_LAYERS];   // linear: the transposed factors in shared memory, in words
  int row, shared_words;
};
static __device__ __constant__ TPlan P;

#ifdef NEB_NEGATIVE_CONTROL_TABLE_CLIP
#define CLIPV (NEB_CLIP - 1)  // wrong on purpose: the check must fail
#else
#define CLIPV NEB_CLIP
#endif

__global__ void __launch_bounds__(THREADS) table_kernel(const uint8_t *__restrict__ rows, uint32_t n,
                                                        const uint32_t *__restrict__ h0, uint32_t start,
                                                        const uint32_t *__restrict__ bound, uint32_t *hits,
                                                        uint32_t max_hits) {
  extern __shared__ int32_t S[];  // per linear layer: cout x 4 words of errors, cin x 4 words of inputs
  __shared__ uint32_t key[8], idx[NEB_NB], lanes[8];
  const int tid = threadIdx.x;
  const uint32_t nonce = start + blockIdx.x;
  if (tid < 8) lanes[tid] = 0;
  if (tid == 0) {
    uint32_t b0[16], b1[16], iv[8], cv[8], k[8], u[16];
    for (int i = 0; i < 8; i++) iv[i] = kIV[i];
    for (int i = 0; i < 16; i++) { b0[i] = h0[i]; b1[i] = i < 4 ? h0[16 + i] : 0u; }
    b1[3] = nonce;
    b3_compress(iv, b0, 64, 1u, cv);
    b3_compress(cv, b1, 16, 2u | 8u, k);
    for (int i = 0; i < 8; i++) key[i] = k[i];
    chacha8_block0(k, u);
    for (int j = 0; j < NEB_NB; j++) idx[j] = (uint32_t)(((unsigned long long)u[j] * n) >> 32);
  }
  __syncthreads();
  // gather the factors of the linear layers: element e of the 16 rows becomes 16 bytes, sample b in byte b
  for (int i = 0; i < P.nlayers; i++) {
    if (P.kind[i] != LINEAR) continue;
    const int total = P.cout[i] + P.cin[i];
    for (int e = tid; e < total; e += THREADS) {
      int32_t *dst = e < P.cout[i] ? S + P.sh_d[i] + 4 * e : S + P.sh_in[i] + 4 * (e - P.cout[i]);
      for (int w = 0; w < 4; w++) {
        uint32_t v = 0;
        for (int b = 0; b < 4; b++)
          v |= (uint32_t)rows[(size_t)idx[4 * w + b] * P.row + P.at[i] + e] << (8 * b);
        dst[w] = (int32_t)v;
      }
    }
  }
  __syncthreads();
  uint32_t lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < P.nlayers; i++) {
    const uint32_t base = 7919u * (uint32_t)(i + 1);
    if (P.kind[i] == LINEAR) {
      const int cin = P.cin[i], cout = P.cout[i];
      for (int o = 0; o < cout; o++) {
        const int32_t *d = S + P.sh_d[i] + 4 * o;
        const int32_t d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3];
        const bool zero = (d0 | d1 | d2 | d3) == 0;
        for (int q = tid; q < cin; q += THREADS) {
          const uint32_t f = (uint32_t)(o * cin + q);
          int G = 0;
          if (!zero) {
            const int32_t *x = S + P.sh_in[i] + 4 * q;
            G = __dp4a(d0, x[0], G);
            G = __dp4a(d1, x[1], G);
            G = __dp4a(d2, x[2], G);
            G = __dp4a(d3, x[3], G);
          }
          const int cg = G < -CLIPV ? -CLIPV : (G > CLIPV ? CLIPV : G);
          lane[f & 7] += (uint32_t)cg ^ (f + base);
        }
      }
    } else {
      for (int f = tid; f < P.nw[i]; f += THREADS) {
        int G = 0;
        if (P.kind[i] == NARROW) {
          for (int b = 0; b < NEB_NB; b++) G += *(const int16_t *)(rows + (size_t)idx[b] * P.row + P.at[i] + 2 * f);
        } else {
          for (int b = 0; b < NEB_NB; b++) G += *(const int32_t *)(rows + (size_t)idx[b] * P.row + P.at[i] + 4 * f);
        }
        const int cg = G < -CLIPV ? -CLIPV : (G > CLIPV ? CLIPV : G);
        lane[f & 7] += (uint32_t)cg ^ ((uint32_t)f + base);
      }
    }
  }
  for (int k = 0; k < 8; k++) atomicAdd(&lanes[k], lane[k]);
  __syncthreads();
  if (tid == 0) {
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

struct neb2t_scan {
  int device, arch;
  TPlan plan;
  uint32_t n, capacity, max_hits;
  std::vector<uint8_t> values, labels, rows;
  uint8_t *d_rows;
  size_t d_rows_bytes;
  uint32_t *d_header, *d_bound, *d_hits;
};

#define TRY(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) return (int)e_; } while (0)

static void make_plan(const neb2_arch *A, const int *wide, TPlan *p) {
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
      at += L.cout + L.cin;
      p->sh_d[i] = sh; sh += 4 * L.cout;
      p->sh_in[i] = sh; sh += 4 * L.cin;
    } else {
      p->kind[i] = wide[i] ? WIDE : NARROW;
      at += (wide[i] ? 4 : 2) * L.nw;
    }
    at = (at + 3) & ~3;
  }
  p->row = at;
  p->shared_words = sh;
}

// Fills the rows of every sample for the weights. Returns the layer whose values do not fit 16 bits, or -1.
static int fill_rows(neb2t_scan *s, const neb2_arch *A, const int8_t *weights) {
  neb2_model *m = (neb2_model *)calloc(1, sizeof *m);
  m->arch = s->arch;
  memcpy(m->w, weights, (size_t)A->nweights);
  const TPlan &p = s->plan;
  s->rows.assign((size_t)p.row * s->n, 0);
  unsigned nt = std::thread::hardware_concurrency();
  if (nt == 0) nt = 4;
  std::vector<int> overflow(nt, -1);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      std::vector<int32_t> g(NEB2_MAX_WEIGHTS);
      std::vector<int8_t> fac(2 * NEB2_MAX_WEIGHTS);
      for (uint32_t i = t; i < s->n; i += nt) {
        neb2_sample_grad(m, s->values.data() + (size_t)i * A->nin, s->labels[i], g.data(), fac.data());
        uint8_t *row = s->rows.data() + (size_t)i * p.row;
        const int8_t *f = fac.data();
        for (int l = 0; l < A->nlayers; l++) {
          const neb2_layer &L = A->l[l];
          if (p.kind[l] == LINEAR) {
            memcpy(row + p.at[l], f, (size_t)(L.cout + L.cin));
            f += L.cout + L.cin;
          } else if (p.kind[l] == WIDE) {
            memcpy(row + p.at[l], g.data() + L.w_off, (size_t)4 * L.nw);
          } else {
            int16_t *o = (int16_t *)(row + p.at[l]);
            for (int k = 0; k < L.nw; k++) {
              const int32_t v = g[L.w_off + k];
              if (v < -32768 || v > 32767) overflow[t] = l;
              o[k] = (int16_t)v;
            }
          }
        }
      }
    });
  for (auto &t : th) t.join();
  free(m);
  for (int o : overflow)
    if (o >= 0) return o;
  return -1;
}

static int build(neb2t_scan *s, const int8_t *weights) {
  const neb2_arch *A = neb2_arch_get(s->arch);
  int wide[NEB2_MAX_LAYERS] = {0};
  for (int i = 0; i < A->nlayers; i++) wide[i] = s->plan.kind[i] == WIDE;
  for (;;) {
    make_plan(A, wide, &s->plan);
    const int o = fill_rows(s, A, weights);
    if (o < 0) break;
    wide[o] = 1;
  }
  TRY(cudaSetDevice(s->device));
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
  memset(&s->plan, 0, sizeof s->plan);
  cudaError_t e = cudaSetDevice(device);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_header, 80);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_bound, 32);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_hits, ((size_t)max_hits + 1) * 4);
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

int neb2t_scan_run(neb2t_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound, uint32_t *hits,
                  uint32_t *nhits) {
  if (count == 0 || count > s->capacity) return (int)cudaErrorInvalidValue;
  TRY(cudaSetDevice(s->device));
  TRY(cudaMemcpyToSymbol(P, &s->plan, sizeof s->plan));
  TRY(cudaMemcpy(s->d_header, header, 80, cudaMemcpyHostToDevice));
  TRY(cudaMemcpy(s->d_bound, bound, 32, cudaMemcpyHostToDevice));
  TRY(cudaMemset(s->d_hits, 0, 4));
  for (uint32_t done = 0; done < count; done += CHUNK) {
    const uint32_t part = count - done < CHUNK ? count - done : CHUNK;
    table_kernel<<<part, THREADS, (size_t)s->plan.shared_words * 4>>>(s->d_rows, s->n, s->d_header, start + done, s->d_bound,
                                                                      s->d_hits, s->max_hits);
    TRY(cudaGetLastError());
  }
  uint32_t found = 0;
  TRY(cudaMemcpy(&found, s->d_hits, 4, cudaMemcpyDeviceToHost));
  *nhits = found;
  if (found > 0)
    TRY(cudaMemcpy(hits, s->d_hits + 1, (size_t)(found < s->max_hits ? found : s->max_hits) * 4, cudaMemcpyDeviceToHost));
  return 0;
}

void neb2t_scan_close(neb2t_scan *s) {
  if (!s) return;
  cudaSetDevice(s->device);
  cudaFree(s->d_rows);
  cudaFree(s->d_header);
  cudaFree(s->d_bound);
  cudaFree(s->d_hits);
  delete s;
}

}  // extern "C"
