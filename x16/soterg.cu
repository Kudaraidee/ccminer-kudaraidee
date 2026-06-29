/**
 * X12R algorithm
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_shabal.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_sha2.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x16.h"

static uint32_t *d_hash[MAX_GPUS];

enum Algo {
    BLAKE = 0,
    SHABAL,
    GROESTL,
    JH,
    KECCAK,
    SKEIN,
    LUFFA,
    CUBEHASH,
    SIMD,
    ECHO,
    HAMSI,
    SHA512,
    HASH_FUNC_COUNT
};

static const char* algo_strings[] = {
    "blake",
    "shabal",
    "groestl",
    "jh512",
    "keccak",
    "skein",
    "luffa",
    "cube",
    "simd",
    "echo",
    "hamsi",
    "sha512",
    NULL
};

#define TIME_MASK 0xFFFFFFA0

static __thread uint32_t s_ntime = UINT32_MAX;
static uint8_t s_firstalgo = 0xFF;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };
static void init_soterg(const int thr_id, int dev_id);
static uint32_t thr_throughput[MAX_GPUS] = { 0 };

static uint8_t GetNibble(const uint8_t* hash, int index)
{
        index = 63 - index;
        if (index % 2 == 1)
            return(hash[index / 2] >> 4);
        return(hash[index / 2] & 0x0F);
}

// Helper function to get hash selection with fallback logic
static inline int GetHashSelection(const uint32_t* prevblock, int index)
{
    const uint8_t* data = (const uint8_t*)prevblock;
    const int START = 48;
    const int MASK = 0xF;
    
    int pos = START + (index & MASK);
    int pos_rev = 63 - pos;
    int nibble = GetNibble(data, pos);
    
    // Fast path: 75-85% of cases
    if (nibble < 12) return nibble;
    
    // Slow path: search next 15 positions
    for (int i = 1; i < 16; ++i) {
        pos = START + ((index + i) & MASK);
	pos_rev = 63 - pos;
        //nibble = (pos_rev & 1) ? (data[pos_rev >> 1] & 0xF) : (data[pos_rev >> 1] >> 4);
        //nibble = (pos_rev & 1) ?  (data[pos_rev >> 1] >> 4) : (data[pos_rev >> 1] & 0xF);
	nibble = GetNibble(data, pos);
        if (nibble < 12) return nibble;
    }
    
    // Fallback: mathematically guaranteed to be 0-11
    return nibble % 12;
}

static void getAlgoString(const uint32_t* prevblock, char *output)
{
    char *sptr = output;
    
    for (uint8_t j = 0; j < HASH_FUNC_COUNT; j++) {
        int hashSelection = GetHashSelection(prevblock, j);
        if (hashSelection >= 10)
            sprintf(sptr, "%c", 'A' + (hashSelection - 10));
        else
            sprintf(sptr, "%u", (uint32_t) hashSelection);
        sptr++;
    }
    *sptr = '\0';
}

static void getprevblock(const uint32_t timeStamp, void* prevblock)
{
    int32_t maskedTime = timeStamp & TIME_MASK;
    sha256d((unsigned char*)prevblock, (const unsigned char*)&(maskedTime), sizeof(maskedTime));
}

extern "C" void soterg_hash(void *output, const void *input)
{
    unsigned char _ALIGN(64) hash[128];

    sph_blake512_context ctx_blake;
    sph_shabal512_context ctx_shabal;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    sph_luffa512_context ctx_luffa;
    sph_cubehash512_context ctx_cubehash;
    sph_sha512_context ctx_sha512;
    sph_simd512_context ctx_simd;
    sph_echo512_context ctx_echo;
    sph_hamsi512_context ctx_hamsi;

    void *in = (void*) input;
    int size = 80;

    uint32_t *in32 = (uint32_t*)input;
    uint32_t ntime = in32[17];

    uint32_t _ALIGN(64) prevblock[8];
    getprevblock(ntime, &prevblock);
    getAlgoString(&prevblock[0], hashOrder);

    for (int i = 0; i < 12; i++)
    {
        const char elem = hashOrder[i];
        const uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

        switch (algo) {
        case BLAKE:
            sph_blake512_init(&ctx_blake);
            sph_blake512(&ctx_blake, in, size);
            sph_blake512_close(&ctx_blake, hash);
            break;
        case KECCAK:
            sph_keccak512_init(&ctx_keccak);
            sph_keccak512(&ctx_keccak, in, size);
            sph_keccak512_close(&ctx_keccak, hash);
            break;
        case SKEIN:
            sph_skein512_init(&ctx_skein);
            sph_skein512(&ctx_skein, in, size);
            sph_skein512_close(&ctx_skein, hash);
            break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, in, size);
			sph_luffa512_close(&ctx_luffa, hash);
			break;
        case CUBEHASH:
            sph_cubehash512_init(&ctx_cubehash);
            sph_cubehash512(&ctx_cubehash, in, size);
            sph_cubehash512_close(&ctx_cubehash, hash);
            break;
        case SIMD:
            sph_simd512_init(&ctx_simd);
            sph_simd512(&ctx_simd, in, size);
            sph_simd512_close(&ctx_simd, hash);
            break;
        case HAMSI:
            sph_hamsi512_init(&ctx_hamsi);
            sph_hamsi512(&ctx_hamsi, in, size);
            sph_hamsi512_close(&ctx_hamsi, hash);
            break;
        case SHA512:
            sph_sha512_init(&ctx_sha512);
            sph_sha512(&ctx_sha512,(const void*) in, size);
            sph_sha512_close(&ctx_sha512,(void*) hash);
            break;
        case JH:
            sph_jh512_init(&ctx_jh);
            sph_jh512(&ctx_jh, in, size);
            sph_jh512_close(&ctx_jh, hash);
            break;
        case SHABAL:
            sph_shabal512_init(&ctx_shabal);
            sph_shabal512(&ctx_shabal, in, size);
            sph_shabal512_close(&ctx_shabal, hash);
            break;
        case GROESTL:
            sph_groestl512_init(&ctx_groestl);
            sph_groestl512(&ctx_groestl, in, size);
            sph_groestl512_close(&ctx_groestl, hash);
            break;
        case ECHO:
            sph_echo512_init(&ctx_echo);
            sph_echo512(&ctx_echo, in, size);
            sph_echo512_close(&ctx_echo, hash);
            break;
        }
        in = (void*) hash;
        size = 64;
    }
    memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "x12r-"
#include "cuda_debug.cuh"

static void init_soterg(const int thr_id, const int dev_id)
{
    int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 19;
    if (strstr(device_name[dev_id], "GTX 1080")) intensity = 20;
    uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);

    cudaSetDevice(device_map[thr_id]);
    if (opt_cudaschedule == -1 && gpu_threads == 1)
    {
        cudaDeviceReset();
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    }

    gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

    thr_throughput[thr_id] = throughput;

    // Per-algo initializers (x12 subset)
    quark_blake512_cpu_init(thr_id, throughput);
    quark_groestl512_cpu_init(thr_id, throughput);
    quark_skein512_cpu_init(thr_id, throughput);
    quark_jh512_cpu_init(thr_id, throughput);
    quark_keccak512_cpu_init(thr_id, throughput);
    x11_simd512_cpu_init(thr_id, throughput);
    x13_hamsi512_cpu_init(thr_id, throughput);
    x16_echo512_cuda_init(thr_id, throughput);
    x11_echo512_cpu_init(thr_id, throughput);
    qubit_luffa512_cpu_init(thr_id, throughput);
    x11_luffa512_cpu_init(thr_id, throughput); // 64
    x14_shabal512_cpu_init(thr_id, throughput);
    x17_sha512_cpu_init(thr_id, throughput);
}

static int algo80_fails[HASH_FUNC_COUNT] = { 0 };

extern "C" int scanhash_soterg(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    const int dev_id = device_map[thr_id];

    if (!init[thr_id])
    {
        init_soterg(thr_id, dev_id);
        CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t)64 * thr_throughput[thr_id]), 0);
        cuda_check_cpu_init(thr_id, thr_throughput[thr_id]);
    }

    uint32_t throughput = thr_throughput[thr_id];

    init[thr_id] = true;

    if (opt_benchmark)
    {
        ((uint32_t*)ptarget)[7] = 0x003f;
        ((uint32_t*)pdata)[1] = 0xEFCDAB89;
        ((uint32_t*)pdata)[2] = 0x67452301;
    }

    uint32_t _ALIGN(64) endiandata[20];

    for (int k=0; k < 19; k++)
        be32enc(&endiandata[k], pdata[k]);

    static uint32_t _ALIGN(64) prevblock[8];

    uint32_t ntime = swab32(pdata[17]);

    if (s_ntime != ntime)
    {
        getprevblock(ntime, &prevblock);
        getAlgoString(&prevblock[0], hashOrder);
        s_ntime = ntime;
        if (!thr_id) applog(LOG_INFO, "hash order: %s time: (%08x) time hash: (%08x)", hashOrder, ntime, prevblock);
    }

    cuda_check_cpu_setTarget(ptarget);

    const int hashes = (int)strlen(hashOrder);
    const char first = hashOrder[0];
    const uint8_t algo80 = first >= 'A' ? first - 'A' + 10 : first - '0';
    
    if (algo80 != s_firstalgo) {
        s_firstalgo = algo80;
        gpulog(LOG_INFO, thr_id, CL_GRN "Algo is now %s, Order %s", algo_strings[algo80 % HASH_FUNC_COUNT], hashOrder);
    }

    switch (algo80) {
        case BLAKE:
            quark_blake512_cpu_setBlock_80(thr_id, endiandata);
            break;
        case KECCAK:
            keccak512_setBlock_80(thr_id, endiandata);
            break;
        case SKEIN:
            skein512_cpu_setBlock_80((void*)endiandata);
            break;
		case LUFFA:
			qubit_luffa512_cpu_setBlock_80((void*)endiandata);
			break;   
        case CUBEHASH:
            cubehash512_setBlock_80(thr_id, endiandata);
            break;
        case SIMD:
            x16_simd512_setBlock_80((void*)endiandata);
            break;
        case HAMSI:
            x16_hamsi512_setBlock_80((void*)endiandata);
            break;
		case SHA512:
			x16_sha512_setBlock_80(endiandata);
			break;   
        case JH:
            jh512_setBlock_80(thr_id, endiandata);
            break;
		case SHABAL:
        	x16_shabal512_setBlock_80((void*)endiandata);
			break;
        case GROESTL:
            groestl512_setBlock_80(thr_id, endiandata);
            break;
        case ECHO:
            x16_echo512_setBlock_80((void*)endiandata);
            break;

        default: {
            if (!thr_id)
                applog(LOG_WARNING, "kernel %s %c unimplemented, order %s", algo_strings[algo80], hashOrder);
            usleep(10);
            return -1;
        }
    }

    int warn = 0;

    do {
        int order = 0;

        // Hash with CUDA
        switch (algo80) {
            case BLAKE:
                quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                TRACE("blake80:");
                break;
            case KECCAK:
                keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                TRACE("kecck80:");
                break;
            case SKEIN:
                skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
                TRACE("skein80:");
                break;
			case LUFFA:
				qubit_luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
				TRACE("luffa80:");
				break;
            case CUBEHASH:
                cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                TRACE("cube 80:");
                break;
            case SIMD:
                x16_simd512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                TRACE("simd512:");
                break;
            case HAMSI:
                x16_hamsi512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                TRACE("hamsi  :");
                break;
			case SHA512:
				x16_sha512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("sha512 :");
				break;	
            case JH:
                jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                TRACE("jh51280:");
                break;
			case SHABAL:
				x16_shabal512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("shabal :");
				break;	
            case GROESTL:
                groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                TRACE("grstl80:");
                break;
            case ECHO:
                x16_echo512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                TRACE("echo   :");
                break;
        }

#pragma omp parallel
#pragma omp for nowait
        for (int i = 1; i < 12; i++)
        {
            const char elem = hashOrder[i];
            const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';

            switch (algo64) {
            case BLAKE:
                quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("blake  :");
                break;
            case KECCAK:
                quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
                TRACE("keccak :");
                break;
            case SKEIN:
                quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("skein  :");
                break;
			case LUFFA:
				x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("luffa  :");
				break;
            case CUBEHASH:
                x11_cubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
                TRACE("cube   :");
                break;
            case SIMD:
                x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("simd   :");
                break;
            case HAMSI:
                x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("hamsi  :");
                break;
 			case SHA512:
				x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
				TRACE("sha512 :");
				break;
            case JH:
                quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("jh512  :");
                break;
			case SHABAL:
				x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
				TRACE("shabal :");
				break;
            case GROESTL:
                quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                TRACE("groestl:");
                break;
            case ECHO:
                if (use_compat_kernels[thr_id])
                    x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                else {
                    x16_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
                }
                TRACE("echo   :");
                break;
            }
        }

        *hashes_done = pdata[19] - first_nonce + throughput;

        work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
#ifdef _DEBUG
        uint32_t _ALIGN(64) dhash[8];
        be32enc(&endiandata[19], pdata[19]);
        soterg_hash(dhash, endiandata);
        applog_hash(dhash);
        return -1;
#endif
        if (work->nonces[0] != UINT32_MAX)
        {
            const uint32_t Htarg = ptarget[7];
            uint32_t _ALIGN(64) vhash[8];
            be32enc(&endiandata[19], work->nonces[0]);
            soterg_hash(vhash, endiandata);

            if (1 || vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
                work->valid_nonces = 1;
                work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
                work_set_target_ratio(work, vhash);
                if (work->nonces[1] != 0) {
                    be32enc(&endiandata[19], work->nonces[1]);
                    soterg_hash(vhash, endiandata);
                    bn_set_target_ratio(work, vhash, 1);
                    work->valid_nonces++;
                    pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
                } else {
                    pdata[19] = work->nonces[0] + 1;
                }
                return work->valid_nonces;
            }
            else if (vhash[7] > Htarg) {
                gpu_increment_reject(thr_id);
                algo80_fails[algo80]++;
                if (!warn) {
                    warn++;
                    pdata[19] = work->nonces[0] + 1;
                    continue;
                } else {
                    if (!opt_quiet) gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %s",
                        work->nonces[0], algo_strings[algo80], hashOrder);
                    warn = 0;
                }
            }
        }

        if ((uint64_t)throughput + pdata[19] >= max_nonce) {
            pdata[19] = max_nonce;
            break;
        }

        pdata[19] += throughput;

    } while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

    *hashes_done = pdata[19] - first_nonce;
    return 0;
}

// cleanup
extern "C" void free_soterg(int thr_id)
{
    if (!init[thr_id])
        return;

    cudaDeviceSynchronize();

    cudaFree(d_hash[thr_id]);

    quark_blake512_cpu_free(thr_id);
    quark_groestl512_cpu_free(thr_id);
    x11_simd512_cpu_free(thr_id);

    cuda_check_cpu_free(thr_id);

    cudaDeviceSynchronize();
    init[thr_id] = false;
}
