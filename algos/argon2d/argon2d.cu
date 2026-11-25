#include <miner.h>
#include "argon2ref/argon2.h"
#include "argon2d_kernel.h"
#include "cuda/helper.cuh"
#include "cuda/selftest_gate.cuh"
#include <cuda_runtime.h>

#define NBN 2

static const size_t INPUT_BYTES = 80;
static const size_t OUTPUT_BYTES = 32;
static const unsigned int DEFAULT_ARGON2_FLAG = 2;

static uint32_t *d_resNonces[MAX_GPUS];
static uint32_t throughputs[MAX_GPUS] = {0};
static bool init[MAX_GPUS] = {0};
static bool diff_done[MAX_GPUS] = {0};
uint8_t* memory[MAX_GPUS];

/* Per-coin Argon2 geometry (all: outlen=32, salt=pwd=header).
 * lanes must be a power of two: compute_ref_pos() masks with lanes-1. */
static const argon2d_variant v_argon2d500   = ARGON2D_VARIANT_INIT(500, 8, 2, ARGON2_VERSION_10, ARGON2_D);    /* Dynamic (DYN) */
static const argon2d_variant v_argon2d1000  = ARGON2D_VARIANT_INIT(1000, 8, 2, ARGON2_VERSION_10, ARGON2_D);   /* Zero Dynamics Cash */
static const argon2d_variant v_argon2d4096  = ARGON2D_VARIANT_INIT(4096, 1, 1, ARGON2_VERSION_13, ARGON2_D);   /* Argentum / Myriad */
static const argon2d_variant v_argon2d16000 = ARGON2D_VARIANT_INIT(16000, 1, 1, ARGON2_VERSION_10, ARGON2_D);  /* Alterdot */
static const argon2d_variant v_argon2id1024 = ARGON2D_VARIANT_INIT(1024, 1, 3, ARGON2_VERSION_13, ARGON2_ID);  /* Bitweb (BTW) */

static void argon2d_cpu_hash(void *output, const void *input, const argon2d_variant *v)
{
	argon2_context context;
	context.out = (uint8_t *)output;
	context.outlen = (uint32_t)OUTPUT_BYTES;
	context.pwd = (uint8_t *)input;
	context.pwdlen = (uint32_t)INPUT_BYTES;
	context.salt = (uint8_t *)input; //salt = input
	context.saltlen = (uint32_t)INPUT_BYTES;
	context.secret = NULL;
	context.secretlen = 0;
	context.ad = NULL;
	context.adlen = 0;
	context.allocate_cbk = NULL;
	context.free_cbk = NULL;
	/* DEFAULT_ARGON2_FLAG is 2 = ARGON2_FLAG_CLEAR_SECRET despite its name;
	 * harmless with secret == NULL, and flags never enter H0. Never set
	 * ARGON2_FLAG_CLEAR_PASSWORD: pwd aliases the caller's header buffer. */
	context.flags = (v->type == ARGON2_ID) ? ARGON2_DEFAULT_FLAGS
	                                       : DEFAULT_ARGON2_FLAG;
	// main configurable Argon2 hash parameters
	context.m_cost = v->mcost;  // Memory in KiB
	context.lanes = v->lanes;   // Degree of Parallelism
	context.threads = 1;        // Threads
	context.t_cost = v->passes; // Iterations
	context.version = v->version;

	const int rc = argon2_ctx( &context, (argon2_type)v->type );
	if (rc != ARGON2_OK)
		applog(LOG_ERR, "argon2 error %d: %s", rc, argon2_error_message(rc));
}

void argon2d500_hash( void *output, const void *input )
{
	argon2d_cpu_hash(output, input, &v_argon2d500);
}

void argon2d1000_0dync_hash( void *output, const void *input )
{
	argon2d_cpu_hash(output, input, &v_argon2d1000);
}

void argon2d4096_hash( void *output, const void *input )
{
	argon2d_cpu_hash(output, input, &v_argon2d4096);
}

void argon2d16000_hash( void *output, const void *input )
{
	argon2d_cpu_hash(output, input, &v_argon2d16000);
}

void argon2id1024_hash( void *output, const void *input )
{
	argon2d_cpu_hash(output, input, &v_argon2id1024);
}

/*
 * One-time self-test: the CPU reference (the authoritative pre-submit
 * re-verify oracle) over a fixed 80-byte header (bytes 0x00..0x4f) must match
 * digests computed with the independent official argon2 library (argon2-cffi,
 * each variant's type and version), and a one-bit header flip must change the
 * digest (proves the test isn't vacuous).
 *
 * argon2id1024 adds a real-block vector and the RFC 9106 vector, and is the
 * only variant that fails CLOSED.
 */
static bool argon2id_ok = false;
/* Host KAT verdicts; the GPU self-test compares against the same CPU
 * reference, so it refuses on these too. */
static bool kat500_ok, kat1000_ok, kat4096_ok, kat16000_ok, host_neg_ok;

/* Bitweb mainnet block 76523, raw 80-byte header.
 * Source: https://explorer.bitwebcore.net/info
 * Its sha256d is asserted too, so a mistyped byte reports "header corrupt"
 * rather than "argon2 is broken". (On Bitweb the block hash is sha256d and
 * the PoW hash is Argon2id, so an explorer's hash field is not a PoW vector.) */
static const unsigned char kat_id_header[80] = {
	0x00,0x00,0x00,0x20, 0xb7,0xfe,0xf5,0xb3, 0x4c,0xef,0xa3,0xc9,
	0x50,0x5b,0xd3,0x46, 0x39,0xc5,0xab,0xfd, 0x8b,0x24,0xbd,0x23,
	0x2d,0xbe,0x5e,0x28, 0xa6,0x68,0xb0,0x57, 0x8d,0xd6,0x02,0xfd,
	0xc5,0x0e,0xf7,0xb4, 0xb9,0x2b,0x74,0x63, 0xd1,0x3d,0xea,0xf0,
	0x76,0xab,0x0c,0xc0, 0xc5,0x73,0x9e,0x21, 0x66,0xd9,0x75,0x88,
	0xbd,0x33,0x6a,0x01, 0x57,0x07,0x9f,0xa3, 0x36,0x87,0x6c,0x6a,
	0xa2,0x85,0x2b,0x1d, 0x10,0xd3,0x00,0x00
};
/* sha256d(header), in the byte order sha256d() emits (reverse of the
 * 11cd1ff6...b903 display form). */
static const unsigned char kat_id_header_sha256d[32] = {
	0x03,0xb9,0xdb,0xab, 0x1d,0x72,0x85,0x1f, 0x91,0x01,0x5d,0x02,
	0xce,0xba,0x93,0x67, 0xe3,0x59,0x49,0x3f, 0x6b,0xbf,0x6a,0xdf,
	0x74,0x9a,0x7a,0x1b, 0xf6,0x1f,0xcd,0x11
};
/* The Argon2id PoW hash of that header, in the order argon2id1024_hash()
 * emits it. Reversed it is 000000102da8...ecf02, which satisfies the block's
 * nBits target 0x1d2b85a2 - so these really are the consensus parameters. */
static const unsigned char kat_id_hash[32] = {
	0x02,0xcf,0x1e,0x2a, 0x74,0xa6,0xda,0x7a, 0x4a,0x35,0x48,0x03,
	0x5b,0x25,0xf5,0x90, 0x70,0x9a,0x2e,0x9f, 0xb3,0xd7,0xad,0x6f,
	0xa6,0xa6,0xa8,0x2d, 0x10,0x00,0x00,0x00
};

/*
 * RFC 9106 section 5.3 Argon2id vector: t=3, m=32, p=4, pwd 32x01, salt 16x02,
 * secret 8x03, ad 12x04. Needs a secret and associated data, so it bypasses
 * argon2d_cpu_hash. Proves the library is standard Argon2id, not merely
 * self-consistent with Bitweb.
 */
static bool argon2id_rfc9106_ok(void)
{
	static const unsigned char expect[32] = {
		0x0d,0x64,0x0d,0xf5, 0x8d,0x78,0x76,0x6c, 0x08,0xc0,0x37,0xa3,
		0x4a,0x8b,0x53,0xc9, 0xd0,0x1e,0xf0,0x45, 0x2d,0x75,0xb6,0x5e,
		0xb5,0x25,0x20,0xe9, 0x6b,0x01,0xe6,0x59
	};
	unsigned char pwd[32], salt[16], secret[8], ad[12], out[32];
	memset(pwd, 0x01, sizeof(pwd));
	memset(salt, 0x02, sizeof(salt));
	memset(secret, 0x03, sizeof(secret));
	memset(ad, 0x04, sizeof(ad));

	argon2_context ctx;
	ctx.out = out;             ctx.outlen = 32;
	ctx.pwd = pwd;             ctx.pwdlen = sizeof(pwd);
	ctx.salt = salt;           ctx.saltlen = sizeof(salt);
	ctx.secret = secret;       ctx.secretlen = sizeof(secret);
	ctx.ad = ad;               ctx.adlen = sizeof(ad);
	ctx.allocate_cbk = NULL;   ctx.free_cbk = NULL;
	ctx.flags = ARGON2_DEFAULT_FLAGS;
	ctx.m_cost = 32;           ctx.lanes = 4;
	ctx.threads = 1;           ctx.t_cost = 3;
	ctx.version = ARGON2_VERSION_13;

	if (argon2_ctx(&ctx, Argon2_id) != ARGON2_OK)
		return false;
	return memcmp(out, expect, 32) == 0;
}

static void argon2d_selftest_once(void)
{
	static bool tested = false;
	if (tested)
		return;
	tested = true;

	static const unsigned char kat500[32] = {
		0x15,0xc7,0x09,0xe0,0x67,0x8a,0xfc,0x10,0xbf,0x5a,0x39,0x63,0xe0,0x3b,0x3c,0x69,
		0x38,0xa9,0xe4,0xde,0xde,0x83,0x30,0x2b,0x6e,0xe6,0x4d,0xca,0xd5,0xfe,0x45,0xfa
	};
	static const unsigned char kat1000[32] = {
		0xf6,0x2c,0x2c,0x19,0x46,0x47,0x58,0x63,0xf8,0x78,0xc2,0xd5,0x4a,0x2f,0x79,0x36,
		0x2b,0x6a,0x0a,0x7c,0xa0,0xb2,0x6e,0xcd,0xaf,0xf3,0x08,0x52,0xb7,0x93,0x15,0xf0
	};
	static const unsigned char kat4096[32] = { /* version 0x13 */
		0xa1,0x56,0xe0,0xc0,0x2d,0xc3,0xd0,0x64,0xf4,0x77,0x16,0x7b,0x03,0x00,0xd8,0xb6,
		0xaf,0x08,0xb0,0xee,0xc6,0x8f,0x17,0x03,0x01,0x2f,0x04,0xbf,0xe8,0xe5,0x21,0x53
	};
	static const unsigned char kat_id1024[32] = { /* Argon2id, version 0x13 */
		0xf4,0xfb,0x55,0x64,0xe6,0x71,0x0f,0xd1,0xb8,0xd5,0x6a,0x3c,0x8d,0x24,0xc6,0x94,
		0x34,0x1b,0x25,0x02,0xd6,0xab,0xa3,0x92,0x99,0x3c,0x66,0x71,0x9e,0x37,0xfb,0xfb
	};
	static const unsigned char kat16000[32] = { /* version 0x10, argon2-cffi */
		0x09,0xa8,0x23,0x46,0x4a,0xca,0x85,0x66,0x73,0xae,0x65,0xcf,0x71,0x5e,0xdf,0x48,
		0xdd,0x5c,0x56,0x27,0xba,0xf9,0x07,0x36,0xb3,0xa4,0x83,0x0f,0xe5,0xa4,0x1f,0xf4
	};
	unsigned char hdr[80], out[32];
	for (int i = 0; i < 80; i++)
		hdr[i] = (unsigned char)i;

	argon2d500_hash(out, hdr);
	kat500_ok = (memcmp(out, kat500, 32) == 0);

	argon2d1000_0dync_hash(out, hdr);
	kat1000_ok = (memcmp(out, kat1000, 32) == 0);

	argon2d4096_hash(out, hdr);
	kat4096_ok = (memcmp(out, kat4096, 32) == 0);

	argon2d16000_hash(out, hdr);
	kat16000_ok = (memcmp(out, kat16000, 32) == 0);

	argon2id1024_hash(out, hdr);
	const bool kat_id1024_ok = (memcmp(out, kat_id1024, 32) == 0);

	/* Real-block leg: prove the header transcription first (sha256d is
	 * independent of Argon2), then the consensus PoW hash. */
	sha256d(out, kat_id_header, 80);
	const bool hdr_ok = (memcmp(out, kat_id_header_sha256d, 32) == 0);

	argon2id1024_hash(out, kat_id_header);
	const bool kat_id_block_ok = (memcmp(out, kat_id_hash, 32) == 0);

	const bool rfc_ok = argon2id_rfc9106_ok();

	hdr[40] ^= 0x01; /* flip one bit */
	argon2d500_hash(out, hdr);
	host_neg_ok = (memcmp(out, kat500, 32) != 0);
	const bool neg_ok = host_neg_ok;

	argon2id_ok = kat_id1024_ok && hdr_ok && kat_id_block_ok && rfc_ok && neg_ok;

	if (!(kat500_ok && kat1000_ok && kat4096_ok && kat16000_ok && neg_ok))
		applog(LOG_ERR, "argon2d self-test FAILED (kat500 %d kat1000 %d kat4096 %d kat16000 %d neg %d)",
		       (int)kat500_ok, (int)kat1000_ok, (int)kat4096_ok, (int)kat16000_ok, (int)neg_ok);

	if (!argon2id_ok) {
		if (!hdr_ok)
			applog(LOG_ERR, "argon2id1024 self-test: KAT header is corrupt (sha256d mismatch)");
		applog(LOG_ERR, "argon2id1024 self-test FAILED (synthetic %d block %d rfc9106 %d neg %d)",
		       (int)kat_id1024_ok, (int)kat_id_block_ok, (int)rfc_ok, (int)neg_ok);
	}
}

__host__
static void ar_set_throughput(int thr_id, const argon2d_variant *v){
    int avail_mem = cuda_available_memory(thr_id);
    uint32_t throughput = (avail_mem * 1024 * 0.75) / v->total_blocks;
    throughput = cuda_default_throughput(thr_id, throughput);
    throughput = (throughput / 16) * 16;

    throughputs[thr_id] = throughput;
}

__host__
static void argon2d_init(int thr_id, const argon2d_variant *v){

    size_t mem_size = (size_t)throughputs[thr_id] * v->total_blocks * ARGON2_BLOCK_SIZE;

    gpulog(LOG_INFO, thr_id,
            "batchsize: %u, trying to allocate %u MB of memory",
            throughputs[thr_id],  mem_size / (1024 * 1024));

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_resNonces[thr_id], NBN * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &memory[thr_id], mem_size));

}


/* In-pipeline per-kernel split timing, only active with -D (debug). */
static cudaEvent_t prof_ev[MAX_GPUS][4];
static bool prof_ready[MAX_GPUS] = {0};
static float prof_ms[MAX_GPUS][3] = {0};
static int prof_n[MAX_GPUS] = {0};

__host__ static void argon2d_hash_cuda(int thr_id, uint32_t throughput, uint32_t startNonce, uint32_t target, uint32_t* resNonces, const argon2d_variant *v){

    struct block_g *memory_blocks=(struct block_g *)memory[thr_id];
    const dim3 blocks = dim3(1, 1, throughput);
    const dim3 th_1 = dim3(16, 16, 1);
    const dim3 th_2 = dim3(THREADS_PER_LANE, v->lanes, 1);
    const dim3 th_3 = dim3(4, 16, 1);

    if (opt_debug && !prof_ready[thr_id]) {
        for (int i = 0; i < 4; i++)
            cudaEventCreate(&prof_ev[thr_id][i]);
        prof_ready[thr_id] = true;
    }

    CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)));

    if (opt_debug) cudaEventRecord(prof_ev[thr_id][0]);

    argon2_initialize<<<throughput/16, th_1>>>((block*) memory[thr_id], startNonce, v->mcost, v->lanes, v->passes, v->version, v->type, v->total_blocks);

    if (opt_debug) cudaEventRecord(prof_ev[thr_id][1]);

    argon2_fill<<<blocks, th_2, v->lanes * ARGON2_SHARED_BLOCKS_PER_LANE * sizeof(block_g)>>>(memory_blocks, v->passes, v->lanes, v->segment_blocks, v->version, v->type);

    if (opt_debug) cudaEventRecord(prof_ev[thr_id][2]);

    argon2_finalize<<<throughput/16, th_3, 16 * 258 * sizeof(uint32_t)>>>((block*) memory[thr_id], startNonce, target, d_resNonces[thr_id], v->total_blocks, NULL);

    if (opt_debug) cudaEventRecord(prof_ev[thr_id][3]);

    cudaDeviceSynchronize();

    if (opt_debug) {
        float ms;
        for (int i = 0; i < 3; i++) {
            cudaEventElapsedTime(&ms, prof_ev[thr_id][i], prof_ev[thr_id][i+1]);
            prof_ms[thr_id][i] += ms;
        }
        if (++prof_n[thr_id] % 20 == 0) {
            float tot = prof_ms[thr_id][0] + prof_ms[thr_id][1] + prof_ms[thr_id][2];
            gpulog(LOG_DEBUG, thr_id, "splits over %d batches: init %.1f%% fill %.1f%% final %.1f%% (%.2f ms/batch)",
                   prof_n[thr_id], 100.f*prof_ms[thr_id][0]/tot, 100.f*prof_ms[thr_id][1]/tot,
                   100.f*prof_ms[thr_id][2]/tot, tot/prof_n[thr_id]);
        }
    }

    CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (resNonces[0] == resNonces[1]) {
        resNonces[1] = UINT32_MAX;
    }

}

/*
 * GPU-vs-CPU digest differential, run once per thread under -D on the first
 * real job. The pre-submit re-verify only sees candidates, so it cannot catch
 * a kernel that MISSES valid nonces; comparing full digests for a batch of
 * consecutive nonces can. Re-run it after any change to the three kernels.
 */
#define ARGON2D_DIFF_NONCES 64

/* Runs the three shipping kernels over n consecutive nonces of the header
 * already uploaded with set_data() and copies the 32-byte digests to h.
 * n must be a multiple of 16 (the initialize/finalize block geometry) and at
 * most the allocated batch. Returns false on a CUDA failure. */
static bool argon2d_gpu_digests(int thr_id, const argon2d_variant *v,
    uint32_t startNonce, uint32_t n, uint32_t *h)
{
    uint32_t *d_digests = NULL;

    if (cudaMalloc((void**)&d_digests, n * 32) != cudaSuccess)
        return false;

    const dim3 th_1 = dim3(16, 16, 1);
    const dim3 th_2 = dim3(THREADS_PER_LANE, v->lanes, 1);
    const dim3 th_3 = dim3(4, 16, 1);

    argon2_initialize<<<n/16, th_1>>>((block*) memory[thr_id], startNonce,
        v->mcost, v->lanes, v->passes, v->version, v->type, v->total_blocks);
    argon2_fill<<<dim3(1, 1, n), th_2, v->lanes * ARGON2_SHARED_BLOCKS_PER_LANE * sizeof(block_g)>>>(
        (struct block_g *)memory[thr_id], v->passes, v->lanes, v->segment_blocks, v->version, v->type);
    /* target 0 => the candidate path never fires; we only want the digests */
    argon2_finalize<<<n/16, th_3, 16 * 258 * sizeof(uint32_t)>>>((block*) memory[thr_id],
        startNonce, 0, d_resNonces[thr_id], v->total_blocks, d_digests);

    const bool ok = cudaGetLastError() == cudaSuccess &&
        cudaMemcpy(h, d_digests, n * 32, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d_digests);
    return ok;
}

static void argon2d_gpu_differential(int thr_id, const argon2d_variant *v,
    void (*cpu_hash)(void*, const void*), uint32_t *endiandata, uint32_t startNonce)
{
    const uint32_t n = ARGON2D_DIFF_NONCES;
    uint32_t *h_digests = NULL;
    uint32_t _ALIGN(64) hdr[20];
    int bad = 0;

    if (n > throughputs[thr_id]) {
        gpulog(LOG_DEBUG, thr_id, "differential skipped: batch %u < %u",
               throughputs[thr_id], n);
        return;
    }

    h_digests = (uint32_t*)calloc(n, 32);
    if (!h_digests)
        return;

    if (argon2d_gpu_digests(thr_id, v, startNonce, n, h_digests)) {
        memcpy(hdr, endiandata, 80);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t _ALIGN(64) vhash[8];
            be32enc(&hdr[19], startNonce + i);
            cpu_hash(vhash, hdr);
            if (memcmp(vhash, &h_digests[i * 8], 32) != 0) {
                if (bad < 3) {
                    char *g = bin2hex((uchar*)&h_digests[i * 8], 32);
                    char *e = bin2hex((uchar*)vhash, 32);
                    gpulog(LOG_ERR, thr_id, "differential MISMATCH nonce %08x", startNonce + i);
                    if (g) { applog(LOG_ERR, "  gpu %s", g); free(g); }
                    if (e) { applog(LOG_ERR, "  cpu %s", e); free(e); }
                }
                bad++;
            }
        }
        if (bad)
            gpulog(LOG_ERR, thr_id, "GPU/CPU differential FAILED: %d of %u digests differ", bad, n);
        else
            gpulog(LOG_DEBUG, thr_id, "GPU/CPU differential: %u/%u digests match", n, n);
    }

    free(h_digests);
}

/*
 * Init-time GPU self-test, once per thread: the three kernels against the CPU
 * reference over ARGON2D_KAT_NONCES nonces, plus a header bit flipped on the
 * device that must change the digest and still match the CPU. Fails closed
 * on a mismatch; a CUDA resource failure only warns.
 * ARGON2D_SELFTEST_FAULT = kat | neg | cuda | host forces a failure.
 */
#define ARGON2D_KAT_NONCES 32 /* two initialize/finalize blocks */
#define ARGON2D_KAT_START  0x01000007u

static const char *argon2d_fault_mode(int thr_id)
{
    const char *m = getenv("ARGON2D_SELFTEST_FAULT");
    if (m && *m)
        gpulog(LOG_WARNING, thr_id, "argon2d: SELF-TEST FAULT INJECTION ACTIVE (%s) -- for testing only", m);
    return (m && *m) ? m : NULL;
}

static bool argon2d_gpu_selftest_run(int thr_id, const argon2d_variant *v,
    void (*cpu_hash)(void*, const void*), int *mism, bool *neg_ok)
{
    const char *fault = argon2d_fault_mode(thr_id);
    const uint32_t n = ARGON2D_KAT_NONCES;
    uint32_t _ALIGN(64) hdr[20], vhash[8];
    uint32_t _ALIGN(64) gpu[ARGON2D_KAT_NONCES * 8], gpu_neg[8 * ARGON2D_KAT_NONCES];

    *mism = 0;
    *neg_ok = false;

    for (int i = 0; i < 80; i++)
        ((uint8_t*)hdr)[i] = (uint8_t)(i * 7 + 1);

    if (fault && !strcmp(fault, "cuda"))
        return selftest_cuda_fault();

    set_data(hdr);
    if (!argon2d_gpu_digests(thr_id, v, ARGON2D_KAT_START, n, gpu))
        return selftest_cuda_fault();

    if (fault && !strcmp(fault, "kat"))
        gpu[3] ^= 1u;

    for (uint32_t i = 0; i < n; i++) {
        be32enc(&hdr[19], ARGON2D_KAT_START + i);
        cpu_hash(vhash, hdr);
        if (memcmp(vhash, &gpu[i * 8], 32) != 0)
            (*mism)++;
    }

    /* negative leg, perturbed on the device side */
    if (!(fault && !strcmp(fault, "neg")))
        hdr[1] ^= 0x00000100u;
    set_data(hdr);
    if (!argon2d_gpu_digests(thr_id, v, ARGON2D_KAT_START, n, gpu_neg))
        return selftest_cuda_fault();
    be32enc(&hdr[19], ARGON2D_KAT_START);
    cpu_hash(vhash, hdr);
    *neg_ok = memcmp(gpu_neg, gpu, 32) != 0 && memcmp(gpu_neg, vhash, 32) == 0;

    return true;
}

/* The host KAT verdict for this variant (argon2d_selftest_once). */
static bool argon2d_host_kat_ok(const argon2d_variant *v)
{
    if (v == &v_argon2d500)   return kat500_ok && host_neg_ok;
    if (v == &v_argon2d1000)  return kat1000_ok && host_neg_ok;
    if (v == &v_argon2d4096)  return kat4096_ok && host_neg_ok;
    if (v == &v_argon2d16000) return kat16000_ok && host_neg_ok;
    return argon2id_ok;
}

static void argon2d_gpu_selftest(int thr_id, const char *name, const argon2d_variant *v,
    void (*cpu_hash)(void*, const void*))
{
    int mism = 0;
    bool neg_ok = false;
    bool passed = argon2d_gpu_selftest_run(thr_id, v, cpu_hash, &mism, &neg_ok);
    const char *fault = getenv("ARGON2D_SELFTEST_FAULT");
    const bool host_ok = argon2d_host_kat_ok(v) && !(fault && !strcmp(fault, "host"));

    /* a failed host KAT is a wrong answer, never a resource fault */
    if (!host_ok)
        s_selftest_cuda_fault = false;
    passed = passed && mism == 0 && neg_ok && host_ok;
    if (passed)
        gpulog(LOG_INFO, thr_id, "%s GPU self-test OK (%d/%d digests + negative)",
               name, ARGON2D_KAT_NONCES, ARGON2D_KAT_NONCES);
    else
        gpulog(LOG_ERR, thr_id, "%s GPU self-test FAILED (%d of %d digests differ, negative %d, host KAT %d)",
               name, mism, ARGON2D_KAT_NONCES, (int)neg_ok, (int)host_ok);
    selftest_gate(thr_id, name, passed);
}

static int scanhash_argon2d( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done,
    const char *name, const argon2d_variant *v, void (*cpu_hash)(void*, const void*) )
{
    uint32_t _ALIGN(64) endiandata[20];
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    uint32_t throughput = 0;

    /* Benchmark mode restarts this window after every find, so a looser
     * screen would lower the rate. The wide lower word makes the host accept
     * everything the GPU screen passes. */
    if (opt_benchmark) {
        ptarget[7] = 0x0fff;
        ptarget[6] = 0xffffffff;
    }

    if (!init[thr_id])
    {
        cudaSetDevice(device_map[thr_id]);
        if (opt_cudaschedule == -1 && gpu_threads == 1) {
            cudaDeviceReset();
            cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            CUDA_LOG_ERROR();
        }

        argon2d_selftest_once();

        /* Fail closed before allocating. The four Argon2d variants refuse
         * on their host KAT in argon2d_gpu_selftest() below. */
        if (v->type == ARGON2_ID && !argon2id_ok) {
            applog(LOG_ERR, "argon2id1024: refusing to start after a failed self-test");
            proper_exit(EXIT_CODE_SW_INIT_ERROR);
        }

        ar_set_throughput(thr_id, v);

        argon2d_init(thr_id, v);

        argon2d_gpu_selftest(thr_id, name, v, cpu_hash);

        init[thr_id] = true;
    }

    throughput = throughputs[thr_id];

    for (int k=0; k < 20; k++)
        be32enc(&endiandata[k], pdata[k]);

    set_data(endiandata);

    if (opt_debug && !diff_done[thr_id]) {
        diff_done[thr_id] = true;
        /* unaligned start, so a nonce-mapping slip shows */
        argon2d_gpu_differential(thr_id, v, cpu_hash, endiandata, pdata[19] + 7);
    }

    do {

        argon2d_hash_cuda(thr_id, throughput, pdata[19], ptarget[7], work->nonces, v);

        *hashes_done = pdata[19] - first_nonce + throughput;

        pdata[19] += throughput;

        /* Re-verify each slot on its own: a failure is counted and logged,
         * and a valid second candidate is kept when the first one fails. */
        uint32_t valid[NBN];
        int nvalid = 0;
        for (int k = 0; k < NBN; k++) {
            const uint32_t nonce = work->nonces[k];
            if (nonce == UINT32_MAX)
                continue;

            uint32_t _ALIGN(64) vhash[8];
            be32enc(&endiandata[19], nonce);
            cpu_hash(vhash, endiandata);

            if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
                if (nvalid == 0)
                    work_set_target_ratio(work, vhash);
                else
                    bn_set_target_ratio(work, vhash, 1);
                valid[nvalid++] = nonce;
                if (opt_debug)
                    gpulog(LOG_DEBUG, thr_id, "found nonce %08x (vhash7 %08x)", nonce, vhash[7]);
            } else {
                gpu_increment_reject(thr_id);
                if (!opt_quiet) {
                    gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", nonce);
                    /* full input for offline replay of the mismatch (endiandata
                     * still holds the candidate nonce at word 19) */
                    char *hex = bin2hex((uchar*)endiandata, 80);
                    if (hex) {
                        applog(LOG_WARNING, "argon2d replay: %s", hex);
                        free(hex);
                    }
                }
            }
        }

        if (nvalid) {
            /* the submit path reads nonces[0] */
            work->nonces[0] = valid[0];
            work->nonces[1] = nvalid > 1 ? valid[1] : UINT32_MAX;
            work->valid_nonces = nvalid;
            return nvalid;
        }

        if ((uint64_t)throughput + pdata[19] >= max_nonce) {
            pdata[19] = max_nonce;
            break;
        }

    } while (!work_restart[thr_id].restart && !abort_flag);

    *hashes_done = pdata[19] - first_nonce;
    return 0;

}

int scanhash_argon2d500( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, "argon2d500", &v_argon2d500, argon2d500_hash);
}

int scanhash_argon2d1000( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, "argon2d1000", &v_argon2d1000, argon2d1000_0dync_hash);
}

int scanhash_argon2d4096( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, "argon2d4096", &v_argon2d4096, argon2d4096_hash);
}

int scanhash_argon2d16000( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, "argon2d16000", &v_argon2d16000, argon2d16000_hash);
}

int scanhash_argon2id1024( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, "argon2id1024", &v_argon2id1024, argon2id1024_hash);
}

static void free_argon2d(int thr_id)
{
    if (!init[thr_id])
        return;

    cudaDeviceSynchronize();

    cudaFree(memory[thr_id]);

    cudaFree(d_resNonces[thr_id]);

    init[thr_id] = false;
    diff_done[thr_id] = false;

    cudaDeviceSynchronize();

    cudaDeviceReset();
}

extern "C" void free_argon2d500(int thr_id)
{
    free_argon2d(thr_id);
}

extern "C" void free_argon2d1000(int thr_id)
{
    free_argon2d(thr_id);
}

extern "C" void free_argon2d4096(int thr_id)
{
    free_argon2d(thr_id);
}

extern "C" void free_argon2d16000(int thr_id)
{
    free_argon2d(thr_id);
}

extern "C" void free_argon2id1024(int thr_id)
{
    free_argon2d(thr_id);
}
