#include <miner.h>
#include "argon2ref/argon2.h"
#include "argon2d_kernel.h"
#include "cuda_helper.h"
#include <cuda_runtime.h>

#define NBN 2

static const size_t INPUT_BYTES = 80;
static const size_t OUTPUT_BYTES = 32;
static const unsigned int DEFAULT_ARGON2_FLAG = 2;

static uint32_t *d_resNonces[MAX_GPUS];
static uint32_t throughputs[MAX_GPUS] = {0};
static bool init[MAX_GPUS] = {0};
uint8_t* memory[MAX_GPUS];

/* Per-coin Argon2d geometry (all: outlen=32, salt=pwd=header). */
static const argon2d_variant v_argon2d500   = ARGON2D_VARIANT_INIT(500, 8, 2, ARGON2_VERSION_10);   /* Dynamic (DYN) */
static const argon2d_variant v_argon2d1000  = ARGON2D_VARIANT_INIT(1000, 8, 2, ARGON2_VERSION_10);  /* Zero Dynamics Cash */
static const argon2d_variant v_argon2d4096  = ARGON2D_VARIANT_INIT(4096, 1, 1, ARGON2_VERSION_13);  /* Argentum / Myriad */
static const argon2d_variant v_argon2d16000 = ARGON2D_VARIANT_INIT(16000, 1, 1, ARGON2_VERSION_10); /* Alterdot */

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
	context.flags = DEFAULT_ARGON2_FLAG; // = ARGON2_DEFAULT_FLAGS
	// main configurable Argon2 hash parameters
	context.m_cost = v->mcost;  // Memory in KiB
	context.lanes = v->lanes;   // Degree of Parallelism
	context.threads = 1;        // Threads
	context.t_cost = v->passes; // Iterations
	context.version = v->version;

	argon2_ctx( &context, Argon2_d );
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

/*
 * One-time self-test: the CPU reference (the authoritative pre-submit
 * re-verify oracle) over a fixed 80-byte header (bytes 0x00..0x4f) must match
 * digests computed with the independent official argon2 library (argon2-cffi,
 * type=D, each variant's version), and a one-bit header flip must change the
 * digest (proves the test isn't vacuous).
 */
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
	unsigned char hdr[80], out[32];
	for (int i = 0; i < 80; i++)
		hdr[i] = (unsigned char)i;

	argon2d500_hash(out, hdr);
	const bool kat500_ok = (memcmp(out, kat500, 32) == 0);

	argon2d1000_0dync_hash(out, hdr);
	const bool kat1000_ok = (memcmp(out, kat1000, 32) == 0);

	argon2d4096_hash(out, hdr);
	const bool kat4096_ok = (memcmp(out, kat4096, 32) == 0);

	hdr[40] ^= 0x01; /* flip one bit */
	argon2d500_hash(out, hdr);
	const bool neg_ok = (memcmp(out, kat500, 32) != 0);

	if (!(kat500_ok && kat1000_ok && kat4096_ok && neg_ok))
		applog(LOG_ERR, "argon2d self-test FAILED (kat500 %d kat1000 %d kat4096 %d neg %d)",
		       (int)kat500_ok, (int)kat1000_ok, (int)kat4096_ok, (int)neg_ok);
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

    argon2_initialize<<<throughput/16, th_1>>>((block*) memory[thr_id], startNonce, v->mcost, v->lanes, v->passes, v->version, v->total_blocks);

    if (opt_debug) cudaEventRecord(prof_ev[thr_id][1]);

    argon2_fill<<<blocks, th_2, v->lanes * ARGON2_SHARED_BLOCKS_PER_LANE * sizeof(block_g)>>>(memory_blocks, v->passes, v->lanes, v->segment_blocks);

    if (opt_debug) cudaEventRecord(prof_ev[thr_id][2]);

    argon2_finalize<<<throughput/16, th_3, 16 * 258 * sizeof(uint32_t)>>>((block*) memory[thr_id], startNonce, target, d_resNonces[thr_id], v->total_blocks);

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

static int scanhash_argon2d( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done,
    const argon2d_variant *v, void (*cpu_hash)(void*, const void*) )
{
    uint32_t _ALIGN(64) endiandata[20];
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    uint32_t throughput = 0;

    if (opt_benchmark)
        ptarget[7] = 0x0fff;

    if (!init[thr_id])
    {
        cudaSetDevice(device_map[thr_id]);
        if (opt_cudaschedule == -1 && gpu_threads == 1) {
            cudaDeviceReset();
            cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            CUDA_LOG_ERROR();
        }

        argon2d_selftest_once();

        ar_set_throughput(thr_id, v);

        argon2d_init(thr_id, v);

        init[thr_id] = true;
    }

    throughput = throughputs[thr_id];

    for (int k=0; k < 20; k++)
        be32enc(&endiandata[k], pdata[k]);

    set_data(endiandata);

    do {

        argon2d_hash_cuda(thr_id, throughput, pdata[19], ptarget[7], work->nonces, v);

        *hashes_done = pdata[19] - first_nonce + throughput;

        pdata[19] += throughput;

        if (work->nonces[0] != UINT32_MAX)
        {

            uint32_t _ALIGN(64) vhash[8];
            const uint32_t Htarg = ptarget[7];
            be32enc(&endiandata[19], work->nonces[0]);
            cpu_hash( vhash, endiandata );

            if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                work->valid_nonces = 1;
                work_set_target_ratio(work, vhash);
                if (opt_debug)
                    gpulog(LOG_DEBUG, thr_id, "found nonce %08x (vhash7 %08x)", work->nonces[0], vhash[7]);

                if (work->nonces[1] != UINT32_MAX) {
                    be32enc(&endiandata[19], work->nonces[1]);
                    cpu_hash(vhash, endiandata);
                    if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                        bn_set_target_ratio(work, vhash, 1);
                        work->valid_nonces++;
                    }
                }

                return work->valid_nonces;

            }
            else if (vhash[7] > Htarg) {
                gpu_increment_reject(thr_id);
                if (!opt_quiet) {
                    gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
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
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, &v_argon2d500, argon2d500_hash);
}

int scanhash_argon2d1000( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, &v_argon2d1000, argon2d1000_0dync_hash);
}

int scanhash_argon2d4096( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, &v_argon2d4096, argon2d4096_hash);
}

int scanhash_argon2d16000( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    return scanhash_argon2d(thr_id, work, max_nonce, hashes_done, &v_argon2d16000, argon2d16000_hash);
}

static void free_argon2d(int thr_id)
{
    if (!init[thr_id])
        return;

    cudaDeviceSynchronize();

    cudaFree(memory[thr_id]);

    cudaFree(d_resNonces[thr_id]);

    init[thr_id] = false;

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
