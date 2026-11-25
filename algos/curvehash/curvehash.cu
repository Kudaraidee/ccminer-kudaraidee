/*
 * curvehash (CurvehashCoin) - GPU scanhash.
 *
 * One thread per nonce runs the full curvehash (device secp256k1 fixed-base
 * k*G stack + SHA-256, see cuda/curvehash_device.cuh) and screens hash[7]
 * against the target high word. The host recomputes every candidate with the
 * libsecp256k1 oracle (curvehash_host_reverify) before submit, so a kernel bug
 * can only cause a local reject, never a bad share.
 *
 * The fixed-base G-table is built on the host (libsecp256k1) once per device at
 * init and uploaded. EC is compute-bound: expect a low (kH/s-class) rate.
 */

#include <miner.h>
#include "cuda/helper.cuh"

#include "cuda/curvehash_device.cuh"

/* 16 windows of 16 bits, 65536 entries of 64 B = 64 MB. */
#define CURVE_W8_BYTES     (32 * 256 * 64)
#define CURVE_GTABLE_BYTES ((size_t)16 * 65536 * 64)

static bool      init_done[MAX_GPUS] = { 0 };
static uint32_t  curve_tpb[MAX_GPUS];
static uint8_t  *d_gtable[MAX_GPUS];
static uint8_t  *d_header[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

extern "C" void curvehash_build_gtable(unsigned char *out);
extern "C" void curvehash_build_window_table(unsigned char *out, int wbits);
extern "C" int  curvehash_check_w16_vs_w8(const unsigned char *w16, const unsigned char *w8);
extern "C" int  curvehash_host_reverify(int thr_id, const uint32_t *pdata, uint32_t nonce,
                                        const uint32_t *ptarget, uint32_t *hash);
extern "C" void curvehash_host_free(int thr_id);

/* Hash path and launch shape per arch: the word-domain chain at (320,2) on sm_61
 * and at (128,4) on sm_80+, the byte path at (512,1) on sm_70/75. The host reads
 * the block size back from the loaded kernel (curve_tpb), so a card running
 * another arch's PTX still launches the shape it was compiled for. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700 && !defined(CURVEHASH_FIELD_10X26)
#define CURVE_WORDS 1
#define CURVE_TPB   320
#define CURVE_MINB  2
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(CURVEHASH_FIELD_10X26)
#define CURVE_WORDS 1
#define CURVE_TPB   128
#define CURVE_MINB  4
#else
#define CURVE_WORDS 0
#define CURVE_TPB   512
#define CURVE_MINB  1
#endif

/* Shared by the scan kernel and the differential so the two cannot drift.
 * Inlining it leaves the scan kernel's SASS unchanged. */
__device__ __forceinline__ bool curvehash_nonce_digest(uint8_t h[32], uint32_t nonce,
    const uint8_t * __restrict__ header76, const uint8_t * __restrict__ gtable)
{
    uint8_t hdr[80];
    #pragma unroll
    for (int i = 0; i < 76; i++) hdr[i] = header76[i];
    /* nonce hashed big-endian, matching the host swab32(nonce) buffer bytes */
    hdr[76] = (uint8_t)(nonce >> 24);
    hdr[77] = (uint8_t)(nonce >> 16);
    hdr[78] = (uint8_t)(nonce >> 8);
    hdr[79] = (uint8_t)(nonce);

    return curvehash_full(h, hdr, gtable);
}

__global__ void __launch_bounds__(CURVE_TPB, CURVE_MINB)
curvehash_scan_kernel(uint32_t threads, uint32_t startNonce,
    const uint8_t * __restrict__ header76, const uint8_t * __restrict__ gtable,
    uint32_t target7, uint32_t *resNonce)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= threads) return;
    uint32_t nonce = startNonce + idx;

    /* host compares hash[7] (uint32 at byte 28, little-endian read) <= target7 */
#if CURVE_WORDS
    uint32_t S[8];
    if (!curvehash_full_w(S, header76, nonce, gtable)) return; /* invalid seckey: skip */
    const uint32_t w7 = __byte_perm(S[7], 0, 0x0123);
#else
    uint8_t h[32];
    if (!curvehash_nonce_digest(h, nonce, header76, gtable)) return; /* invalid seckey: skip */

    uint32_t w7 = ((uint32_t)h[31] << 24) | ((uint32_t)h[30] << 16) |
                  ((uint32_t)h[29] << 8)  | (uint32_t)h[28];
#endif
    if (w7 <= target7) atomicMin(resNonce, nonce);
}

/*
 * GPU-vs-CPU differential over a nonce RANGE: the candidate re-verify only sees
 * nonces the GPU reported, so it cannot catch one the kernel never hashed.
 *
 * Two order-independent accumulators. acc[0..7] (xor of the digest words) catches
 * a wrong, missing or duplicated digest; acc[8..15] weights each digest by the
 * nonce and is what binds the two, since a plain xor is permutation-blind.
 *
 * The weight must stay injective AND odd: 2*nonce+1, never nonce|1 -- clearing
 * bit 0 gives an aligned pair (2k, 2k+1) the same weight, hiding a swap of it.
 *
 * Launch shape mirrors the scan kernel so an index-math defect reproduces here.
 */
#define CURVE_DIFF_ACC 16

__global__ void __launch_bounds__(CURVE_TPB, CURVE_MINB)
curvehash_diff_kernel(uint32_t threads, uint32_t startNonce,
    const uint8_t * __restrict__ header76, const uint8_t * __restrict__ gtable,
    uint32_t *acc, uint32_t fault)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= threads) return;
    uint32_t nonce = startNonce + idx;

    /* Fault injection for the negative control: 1 = drop a nonce (both
     * accumulators must catch it), 2 = swap two (only acc[8..15] can). */
    if (fault == 1 && idx == 3) return;
    if (fault == 2 && idx == 3) nonce = startNonce + 4;
    if (fault == 2 && idx == 4) nonce = startNonce + 3;

#if CURVE_WORDS
    uint32_t S[8];
    if (!curvehash_full_w(S, header76, nonce, gtable)) return;
#else
    uint8_t h[32];
    if (!curvehash_nonce_digest(h, nonce, header76, gtable)) return;
#endif

    const uint32_t w = 2u * (startNonce + idx) + 1u;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
#if CURVE_WORDS
        const uint32_t d = __byte_perm(S[i], 0, 0x0123);
#else
        uint32_t d = ((uint32_t)h[4*i+3] << 24) | ((uint32_t)h[4*i+2] << 16) |
                     ((uint32_t)h[4*i+1] << 8)  | (uint32_t)h[4*i];
#endif
        atomicXor(&acc[i],     d);
        atomicXor(&acc[8 + i], d * w);
    }
}

/* Hash n headers (80 B each) into n digests (32 B each), so the host can run
 * several self-test vectors in one launch. */
__global__ void curvehash_selftest_kernel(const uint8_t *gtable, const uint8_t * __restrict__ hdrs,
                                          int n, uint8_t *out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
#if CURVE_WORDS
    /* same path as the scan kernel */
    const uint8_t *h = hdrs + t * 80;
    const uint32_t nonce_be = ((uint32_t)h[76] << 24) | ((uint32_t)h[77] << 16) |
                              ((uint32_t)h[78] << 8) | h[79];
    uint32_t S[8];
    curvehash_full_w(S, h, nonce_be, gtable);
    for (int i = 0; i < 8; i++) {
        out[t * 32 + 4 * i]     = (uint8_t)(S[i] >> 24);
        out[t * 32 + 4 * i + 1] = (uint8_t)(S[i] >> 16);
        out[t * 32 + 4 * i + 2] = (uint8_t)(S[i] >> 8);
        out[t * 32 + 4 * i + 3] = (uint8_t)S[i];
    }
#else
    curvehash_full(out + t * 32, hdrs + t * 80, gtable);
#endif
}

/*
 * One-time device self-test, fail-closed: a GPU that cannot reproduce the
 * consensus hash can only produce local rejects, so refuse to start. Four legs,
 * reported separately so a failure points at the right layer:
 *   tblbuild - sha256d of the host-built G-table vs a known constant (all 8192
 *              entries, not a sample; a corrupt entry causes only RARE wrong
 *              hashes, which no digest KAT can detect)
 *   tblup    - the table read back from device memory vs the host buffer
 *   kat      - three header/digest vectors from an independent reference
 *              implementation: the consensus vector plus all-00 and all-ff
 *   neg      - a flipped header bit must change the digest, computed on the
 *              device, so the comparison cannot be vacuous
 */
static bool curvehash_selftest_gpu(int thr_id, const uint8_t *d_gtable,
                                  const unsigned char *h_gtable)
{
    /* Digests from an independent reference implementation of curvehash. */
    static const uint8_t kat[3][32] = {
      { 0xb2,0x64,0x54,0x16,0xce,0x97,0xcf,0x39,0x35,0x59,0x2d,0x82,0xea,0xeb,0xf2,0x52,
        0x12,0x00,0x8e,0xbf,0x04,0xf6,0x23,0x73,0x20,0x3a,0x71,0x53,0xfa,0x1e,0x14,0x66 },
      { 0x6d,0xbb,0x54,0x4b,0x3f,0xb5,0x12,0x83,0x78,0x82,0x79,0xb1,0xbf,0x37,0x01,0xf2,
        0x5d,0x19,0x9d,0x63,0x73,0x2f,0x08,0x42,0x03,0xdf,0x04,0xb7,0x20,0xa4,0xe0,0x3d },
      { 0x07,0x3f,0x04,0xc8,0xbd,0x37,0x98,0x77,0x84,0xd4,0x8c,0x99,0x9c,0xea,0x20,0x5e,
        0x5b,0x3d,0xb4,0x0e,0xfc,0x02,0x3c,0xbd,0x8d,0xb5,0x25,0x5d,0x27,0xc0,0x05,0xe5 }
    };
    /* sha256d of the 8-bit table, from an independent implementation of it. The
     * shipped 16-bit table is checked through this one: see
     * curvehash_check_w16_vs_w8. */
    static const uint8_t tbl_sha256d[32] = {
        0x05,0xee,0x91,0x6f,0x8e,0xc8,0xb4,0xf1,0x4d,0x7c,0xd3,0xa3,0x71,0x0b,0xb5,0xff,
        0xc6,0x55,0x3c,0x10,0xc0,0x1c,0x11,0x2d,0xcb,0xb6,0x5c,0x06,0xe6,0xb1,0x6d,0x91
    };
    const int NV = 4;                     /* 3 KAT vectors + 1 negative vector */
    uint8_t h_hdr[NV * 80], h_out[NV * 32];
    uint8_t *d_hdr = NULL, *d_out = NULL;

    /* v0 = 0x00..0x4f, v1 = all zero, v2 = all ff, v3 = v0 with one bit flipped. */
    for (int i = 0; i < 80; i++) {
        h_hdr[i]        = (uint8_t)i;
        h_hdr[80 + i]   = 0x00;
        h_hdr[160 + i]  = 0xff;
        h_hdr[240 + i]  = (uint8_t)i;
    }
    h_hdr[240 + 40] ^= 0x01;

    CUDA_SAFE_CALL(cudaMalloc(&d_hdr, sizeof(h_hdr)));
    CUDA_SAFE_CALL(cudaMalloc(&d_out, sizeof(h_out)));
    CUDA_SAFE_CALL(cudaMemcpy(d_hdr, h_hdr, sizeof(h_hdr), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(d_out, 0, sizeof(h_out)));

    curvehash_selftest_kernel <<< 1, NV >>> (d_gtable, d_hdr, NV, d_out);

    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    cudaFree(d_hdr);
    cudaFree(d_out);

    /* tbl leg: the host build (all 8192 entries, via sha256d) and the upload
     * (byte-for-byte readback of the full 512 KB). */
    unsigned char dig[32];
    unsigned char *rb = (unsigned char *)malloc(CURVE_GTABLE_BYTES);

    /* Leg 1: rebuild the 8-bit table and check it against the constant.
     * Leg 2: every 16-bit entry must decompose into two 8-bit entries. */
    unsigned char *w8 = (unsigned char *)malloc(CURVE_W8_BYTES);
    bool tbl_build_ok = false, tbl_decomp_ok = true;
    if (w8) {
        curvehash_build_gtable(w8);
        sha256d(dig, w8, CURVE_W8_BYTES);
        tbl_build_ok = (memcmp(dig, tbl_sha256d, 32) == 0);
        tbl_decomp_ok = curvehash_check_w16_vs_w8(h_gtable, w8);
        free(w8);
    }
    CUDA_SAFE_CALL(cudaMemcpy(rb, d_gtable, CURVE_GTABLE_BYTES, cudaMemcpyDeviceToHost));
    const bool tbl_upload_ok = (memcmp(rb, h_gtable, CURVE_GTABLE_BYTES) == 0);
    free(rb);
    const bool tbl_ok = tbl_build_ok && tbl_decomp_ok && tbl_upload_ok;
    const bool kat0_ok = (memcmp(h_out,       kat[0], 32) == 0);
    const bool kat1_ok = (memcmp(h_out + 32,  kat[1], 32) == 0);
    const bool kat2_ok = (memcmp(h_out + 64,  kat[2], 32) == 0);
    const bool neg_ok  = (memcmp(h_out + 96,  kat[0], 32) != 0);

    if (!(tbl_ok && kat0_ok && kat1_ok && kat2_ok && neg_ok)) {
        if (!tbl_build_ok)
            gpulog(LOG_ERR, thr_id, "curvehash self-test: host-built G-table checksum mismatch "
                                    "- libsecp256k1 table build is broken, not the kernel");
        if (!tbl_decomp_ok)
            gpulog(LOG_ERR, thr_id, "curvehash self-test: 16-bit table does not decompose into the "
                                    "8-bit table - the window table build is broken, not the kernel");
        if (!tbl_upload_ok)
            gpulog(LOG_ERR, thr_id, "curvehash self-test: device G-table != host G-table "
                                    "- the 512 KB upload is broken, not the kernel");
        gpulog(LOG_ERR, thr_id, "curvehash GPU self-test FAILED (tblbuild %d tbldecomp %d tblup %d kat %d%d%d neg %d)",
               (int)tbl_build_ok, (int)tbl_decomp_ok, (int)tbl_upload_ok, (int)kat0_ok, (int)kat1_ok, (int)kat2_ok, (int)neg_ok);
        return false;
    }
    return true;
}

/*
 * Compare both accumulators against the libsecp256k1 oracle. The count is not a
 * multiple of the block size and the start is offset, so the launch has a ragged
 * tail and does not begin on a block boundary. Costs one host hash per nonce, so
 * it runs once per job under -D, never per launch.
 */
#define CURVE_DIFF_N      1000u
#define CURVE_DIFF_OFF    7u

static bool curvehash_differential(int thr_id, const uint32_t *pdata, uint32_t fault)
{
    const uint32_t start = pdata[19] + CURVE_DIFF_OFF;
    uint32_t h_acc[CURVE_DIFF_ACC], cpu[CURVE_DIFF_ACC] = { 0 };
    uint32_t *d_acc = NULL;
    const uint32_t tpb = curve_tpb[thr_id];
    const uint32_t grid = (CURVE_DIFF_N + tpb - 1) / tpb;

    CUDA_SAFE_CALL(cudaMalloc(&d_acc, sizeof(h_acc)));
    CUDA_SAFE_CALL(cudaMemset(d_acc, 0, sizeof(h_acc)));
    curvehash_diff_kernel <<< grid, tpb >>> (CURVE_DIFF_N, start, d_header[thr_id],
                                             d_gtable[thr_id], d_acc, fault);
    CUDA_SAFE_CALL(cudaMemcpy(h_acc, d_acc, sizeof(h_acc), cudaMemcpyDeviceToHost));
    cudaFree(d_acc);

    /* Same two accumulators on the host, over the same range. The oracle's
     * target test is irrelevant here, only the digest is. */
    uint32_t dummy_target[8];
    memset(dummy_target, 0, sizeof(dummy_target));
    for (uint32_t i = 0; i < CURVE_DIFF_N; i++) {
        uint32_t _ALIGN(64) vhash[8];
        const uint32_t nonce = start + i;
        curvehash_host_reverify(thr_id, pdata, nonce, dummy_target, vhash);
        const uint32_t w = 2u * nonce + 1u;
        for (int k = 0; k < 8; k++) {
            cpu[k]     ^= vhash[k];
            cpu[8 + k] ^= vhash[k] * w;
        }
    }

    const bool d_ok = (memcmp(h_acc, cpu, 8 * sizeof(uint32_t)) == 0);
    const bool n_ok = (memcmp(h_acc + 8, cpu + 8, 8 * sizeof(uint32_t)) == 0);
    if (!(d_ok && n_ok)) {
        gpulog(LOG_ERR, thr_id, "curvehash differential FAILED over %u nonces from %08x "
               "(digest %s, nonce-weighted %s)", CURVE_DIFF_N, start,
               d_ok ? "ok" : "MISMATCH", n_ok ? "ok" : "MISMATCH");
        if (d_ok && !n_ok)
            gpulog(LOG_ERR, thr_id, "curvehash differential: digests agree but their NONCE "
                                    "pairing does not - the kernel is hashing the wrong nonces");
        return false;
    }
    gpulog(LOG_DEBUG, thr_id, "curvehash differential OK: %u nonces from %08x, both accumulators",
           CURVE_DIFF_N, start);
    return true;
}

static void curvehash_init(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);

    /* block size the loaded scan kernel was compiled for (CURVE_TPB) */
    cudaFuncAttributes fa;
    CUDA_SAFE_CALL(cudaFuncGetAttributes(&fa, curvehash_scan_kernel));
    curve_tpb[thr_id] = fa.maxThreadsPerBlock >= 512 ? 512 : fa.maxThreadsPerBlock >= 320 ? 320 : 128;

    CUDA_SAFE_CALL(cudaMalloc(&d_gtable[thr_id], CURVE_GTABLE_BYTES));
    CUDA_SAFE_CALL(cudaMalloc(&d_header[thr_id], 76));
    CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], sizeof(uint32_t)));

    unsigned char *tbl = (unsigned char *)malloc(CURVE_GTABLE_BYTES);
    if (!tbl) {
        gpulog(LOG_ERR, thr_id, "curvehash: out of memory for the %.0f MB window table",
               (double)CURVE_GTABLE_BYTES / 1048576.0);
        proper_exit(EXIT_CODE_SW_INIT_ERROR);
    }
    curvehash_build_window_table(tbl, 16);
    CUDA_SAFE_CALL(cudaMemcpy(d_gtable[thr_id], tbl, CURVE_GTABLE_BYTES, cudaMemcpyHostToDevice));

    /* One-time device self-test, FAIL-CLOSED: a GPU that cannot reproduce the
     * consensus hash can only produce local rejects, so do not start. */
    const bool st_ok = curvehash_selftest_gpu(thr_id, d_gtable[thr_id], tbl);
    free(tbl);
    if (!st_ok) {
        gpulog(LOG_ERR, thr_id, "curvehash: refusing to start after a failed self-test");
        proper_exit(EXIT_CODE_SW_INIT_ERROR);
    }

    init_done[thr_id] = true;
}

extern "C" int scanhash_curvehash(int thr_id, struct work *work, uint32_t max_nonce,
                                  unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    if (opt_benchmark && ptarget[7] < 0x0000ffffU)
        ptarget[7] = 0x0000ffffU;

    if (!init_done[thr_id])
        curvehash_init(thr_id);

    /* the sm_61 shape needs 2^18 nonces per launch to fill the GPU; -i overrides */
    uint32_t throughput = cuda_default_throughput(thr_id, curve_tpb[thr_id] == 320 ? 1U << 18 : 1U << 16);
    throughput = min(throughput, max_nonce - first_nonce);

    /* 76-byte base header = be32enc(pdata[0..18]); the kernel appends the nonce
     * and consumes this as BYTES. If it is ever changed to read uint32 words,
     * upload pdata verbatim instead: reading be32enc'd bytes back as a word on a
     * little-endian device byte-swaps them a second time. */
    uint32_t _ALIGN(64) endiandata[19];
    for (int i = 0; i < 19; i++)
        be32enc(&endiandata[i], pdata[i]);
    CUDA_SAFE_CALL(cudaMemcpy(d_header[thr_id], endiandata, 76, cudaMemcpyHostToDevice));

    /* Once per job, after the header upload, under -D only. CURVEHASH_DIFF_FAULT
     * injects a defect (1 = drop a nonce, 2 = swap two) so the gate can be shown
     * to fire without a rebuild. */
    if (opt_debug) {
        static uint32_t last_hdr[MAX_GPUS][19];
        if (memcmp(last_hdr[thr_id], endiandata, sizeof(last_hdr[0])) != 0) {
            memcpy(last_hdr[thr_id], endiandata, sizeof(last_hdr[0]));
            const char *fenv = getenv("CURVEHASH_DIFF_FAULT");
            const uint32_t fault = fenv ? (uint32_t)atoi(fenv) : 0;
            if (fault)
                gpulog(LOG_WARNING, thr_id, "curvehash differential: FAULT %u injected on purpose", fault);
            if (!curvehash_differential(thr_id, pdata, fault)) {
                gpulog(LOG_ERR, thr_id, "curvehash: refusing to mine after a failed differential");
                proper_exit(EXIT_CODE_SW_INIT_ERROR);
            }
        }
    }

    const uint32_t UMAX = UINT32_MAX;
    const uint32_t tpb = curve_tpb[thr_id];

    do {
        CUDA_SAFE_CALL(cudaMemcpy(d_resNonce[thr_id], &UMAX, sizeof(uint32_t), cudaMemcpyHostToDevice));

        uint32_t grid = (throughput + tpb - 1) / tpb;
        curvehash_scan_kernel <<< grid, tpb >>> (throughput, pdata[19], d_header[thr_id],
                                                 d_gtable[thr_id], ptarget[7], d_resNonce[thr_id]);

        uint32_t win = UMAX;
        CUDA_SAFE_CALL(cudaMemcpy(&win, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost));

        *hashes_done = pdata[19] - first_nonce + throughput;

        if (win != UMAX) {
            uint32_t _ALIGN(64) vhash[8];
            if (curvehash_host_reverify(thr_id, pdata, win, ptarget, vhash)) {
                work_set_target_ratio(work, vhash);
                work->nonces[0] = win;
                work->valid_nonces = 1;
                /* Resume PAST the winner: the caller restores this cursor after
                 * submitting, so `= win` would re-find and re-submit the same
                 * nonce and never advance. */
                pdata[19] = win + 1;
                return 1;
            }
            gpu_increment_reject(thr_id);
            if (!opt_quiet)
                gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", win);
        }

        pdata[19] += throughput;

    } while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

    *hashes_done = pdata[19] - first_nonce;
    return 0;
}

extern "C" void free_curvehash(int thr_id)
{
    if (!init_done[thr_id]) return;
    cudaSetDevice(device_map[thr_id]);
    cudaFree(d_gtable[thr_id]);
    cudaFree(d_header[thr_id]);
    cudaFree(d_resNonce[thr_id]);
    curvehash_host_free(thr_id);
    init_done[thr_id] = false;
}
