#include <stdio.h>
#include <cstdint>
#include <memory.h>
#include <miner.h>
#include "cuda/helper.cuh" //

// Host re-verify: RinHash = BLAKE3-256 -> Argon2d -> SHA3-256. All three
// primitives already exist host-side, which is why no re-verify was a choice
// rather than a limitation.
//   blake3_256      cuda/blake3_device.cuh is __host__ __device__
//   argon2d_hash_raw the vendored reference, ARGON2_VERSION_13 like the kernel
//   sha3             sph/sha3.c, FIPS-202 (0x06), like the device sha3_256_32
#include "cuda/blake3_device.cuh"
#include "algos/argon2d/argon2ref/argon2.h"
extern "C" {
#include "sph/sha3.h"
}
#include "cuda/selftest_gate.cuh"

using namespace std;

/* The cooperative Argon2d path (algos/rinhash/rinhash_coop.cu): 32 threads per
 * nonce sharing the argon2d family's fill kernel, with the 1 KB block held in
 * registers rather than a per-thread local buffer. */
extern "C" void RinHash_mine_coop(int thr_id,
    const uint32_t* work_data, uint32_t nonce_offset, uint32_t start_nonce,
    uint32_t num_nonces, uint32_t* target, uint32_t* found_nonce,
    uint8_t* target_hash, uint8_t* best_hash, uint32_t* solution_found,
    uint32_t* hashes_actually_done);
extern "C" void rinhash_coop_reserve(int thr_id, uint32_t max_nonces);
extern "C" void rinhash_coop_cleanup(int thr_id);
extern "C" uint32_t rinhash_coop_batch_for_mps(int mpcount);

// Both defined below; scanhash uses them.
extern "C" void rinhash_hash(void *output, const void *input);
static bool rinhash_device_selftest(int thr_id);

// Release the persistent device memory. Named for algo_free_all(), which
// ccminer calls when it parks a thread, stops on a conditional-mining rule or
// switches algo. The original rinhash_free()/rinhash_init() pair was declared
// in no header and called from nowhere, so the buffers survived all three.
extern "C" void free_rinhash(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);
    rinhash_coop_cleanup(thr_id);
}

// Main scanning function that tries different nonces to find a valid hash
int scanhash_rinhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    uint32_t nonce = first_nonce;
    if (opt_benchmark)
        ptarget[7] = 0xff;

    // Nonces per scanhash call, and per kernel launch (one launch per call).
    // 256 per SM, so the footprint scales with the card (448 MB on a 28-SM
    // 3060). Throughput is insensitive to this value. Must stay a multiple of
    // the 16 nonces per CUDA block the end kernels are launched with.
    const uint32_t total_batch_size = rinhash_coop_batch_for_mps(device_mpcount[thr_id]);

    // Work is only polled for between launches, so the launch length is the
    // staleness window. At this size it is milliseconds against a job cadence
    // of tens of seconds.
    const uint32_t kernel_chunk_size = total_batch_size;

    max_nonce = min(first_nonce + total_batch_size, max_nonce);

    // Size the persistent buffers once, for the largest LAUNCH -- not the whole
    // batch, which no launch reaches. Growing them on demand would put a ~1 GB
    // realloc inside this timed call.
    rinhash_coop_reserve(thr_id, min(total_batch_size, kernel_chunk_size));

    // Fail-closed, once per thread, and after the reserve so its one-nonce
    // launch cannot size the buffers small and force a realloc.
    static bool selftested[MAX_GPUS] = { false };
    if (!selftested[thr_id]) {
        selftested[thr_id] = true;
        if (rinhash_device_selftest(thr_id))
            gpulog(LOG_INFO, thr_id, "RinHash self-test OK (SHA3 + BLAKE3 vectors, GPU==CPU, negative)");
    }

    uint32_t found_nonce = 0;
    uint32_t solution_found = 0;
    // Nonces REALLY hashed, which is not the chunk size: the kernel abandons
    // waves once a solution is found. Counting the chunk inflates the rate.
    unsigned long total_hashed = 0;
    uint32_t chunk_hashed = 0;
    uint8_t best_hash[32];
    uint8_t target_hash[32]; // (unused, but the API requires it)
    uint32_t target[8];

    // Convert target (already little-endian)
    for (int i = 0; i < 8; i++) {
        target[i] = ptarget[i];
    }

    work->valid_nonces = 0;
    cudaSetDevice(device_map[thr_id]);

    do {
        // CORRECT LOGIC: compute the chunk for this kernel call
        uint32_t current_chunk_size = min(kernel_chunk_size, max_nonce - nonce);
        
        if (current_chunk_size <= 0) {
            *hashes_done = total_hashed;
            return 0; // completed total_batch_size
        }

        solution_found = 0;
        RinHash_mine_coop(thr_id,
            pdata,
            19, // nonce offset
            nonce,
            current_chunk_size, // call the kernel with this chunk
            target,
            &found_nonce,
            target_hash, // (unused)
            best_hash,
            &solution_found,
            &chunk_hashed
        );

        total_hashed += chunk_hashed;

        *hashes_done = total_hashed;

        if (solution_found) {
            uint32_t _ALIGN(64) vhash[8];
            uint32_t _ALIGN(64) hdr[20];

            // Re-hash the candidate on the host and submit on THAT. The header
            // is uploaded verbatim and the kernel only overwrites word 19, so
            // reproduce exactly that.
            memcpy(hdr, pdata, 80);
            hdr[19] = found_nonce;
            rinhash_hash(vhash, hdr);

            // Disagreement here means the device screen and the CPU reference
            // computed different digests: a kernel or parameter bug, not a
            // near-miss. Worth a distinct message from an ordinary reject.
            if (memcmp(vhash, best_hash, 32) != 0)
                gpulog(LOG_WARNING, thr_id, "GPU/CPU digest mismatch for nonce %08x", found_nonce);

            if (fulltest(vhash, ptarget)) {
                work->valid_nonces = 1;
                work_set_target_ratio(work, vhash);
                work->nonces[0] = found_nonce;
                // the launch reports its LOWEST candidate: resume after it, not
                // past the chunk, so the others are found again (a chunk is ~7 ms)
                pdata[19] = found_nonce + 1;
                *hashes_done = total_hashed;
                return 1;
            } else {
                gpu_increment_reject(thr_id);
                if (!opt_quiet)
                    gpulog(LOG_WARNING, thr_id, "result for %08x does not validate!", found_nonce);
                nonce = found_nonce + 1;   // scan the rest of the chunk
                continue;
            }
        }
        
        // CORRECT LOGIC: advance the nonce and repeat the 'do...while' loop
        // to process the next chunk
        nonce += current_chunk_size;

    } while (nonce < max_nonce && !work_restart[thr_id].restart);

    pdata[19] = nonce;
    *hashes_done = total_hashed;
    return 0; // finished the batch without finding anything
}

static void hex32(const uint8_t *d, char *out)
{
    for (int i = 0; i < 32; i++)
        sprintf(out + i * 2, "%02x", d[i]);
    out[64] = 0;
}

// Init self-test, four legs, fail-closed.
//   sha3 - SHA3-256("") vs the published FIPS-202 digest. Pins the 0x06 domain
//          byte, which a self-generated vector cannot.
//   b3   - BLAKE3("") vs the published digest. Pins the shared header.
//   kat  - the SHIPPING launcher must agree with the CPU reference on a fixed
//          header. This is also the only check on the device argon2d, against
//          the vendored reference implementation.
//   neg  - a flipped header bit must change the device digest, or kat could
//          pass on a kernel that ignored its input.
static bool rinhash_device_selftest(int thr_id)
{
    static const char *KAT_SHA3_EMPTY =
        "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a";
    static const char *KAT_BLAKE3_EMPTY =
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";

    uint8_t d[32];
    char got[65];
    bool sha3_ok = false, b3_ok = false, kat_ok = false, neg_ok = false;

    sha3("", 0, d, 32);
    hex32(d, got);
    sha3_ok = (strcmp(got, KAT_SHA3_EMPTY) == 0);

    blake3_256((const uint8_t *)"", 0, d);
    hex32(d, got);
    b3_ok = (strcmp(got, KAT_BLAKE3_EMPTY) == 0);

    // An all-ones target makes every nonce in the span a candidate, so the
    // launcher answers immediately for a span of our choosing. The launch
    // reports its lowest candidate, which here must be the first nonce of the
    // span (the kat leg checks that, so the ordering itself is tested); the
    // digest is then held to the reference for the reported nonce. The neg leg
    // only requires a nonce inside the span.
    const uint32_t kat_span = 128;
    uint32_t target[8];
    for (int i = 0; i < 8; i++) target[i] = 0xFFFFFFFFu;

    uint32_t hdr[20];
    for (int i = 0; i < 20; i++) hdr[i] = 0x11111111u * (uint32_t)(i + 1);
    const uint32_t kat_nonce = 0x1234abcdu;
    hdr[19] = kat_nonce;

    uint8_t gpu[32], cpu[32], dummy[32];
    uint32_t found = 0, solved = 0, hashed = 0;

    RinHash_mine_coop(thr_id, hdr, 19, kat_nonce, kat_span, target,
                            &found, dummy, gpu, &solved, &hashed);
    if (!solved) {
        // The device could not answer at all: not evidence of a wrong hash.
        selftest_cuda_fault();
    } else if (found < kat_nonce || found >= kat_nonce + kat_span) {
        // It DID answer, with a nonce outside the range it was given. That is
        // a real failure, not a resource fault.
        gpulog(LOG_ERR, thr_id, "RinHash self-test: reported nonce %08x outside [%08x,%08x)",
               found, kat_nonce, kat_nonce + kat_span);
    } else if (found != kat_nonce) {
        // every nonce is a candidate, so the lowest one must be reported
        gpulog(LOG_ERR, thr_id, "RinHash self-test: reported %08x, not the lowest candidate %08x",
               found, kat_nonce);
    } else {
        uint32_t kat_hdr[20];
        memcpy(kat_hdr, hdr, 80);
        kat_hdr[19] = found;
        rinhash_hash(cpu, kat_hdr);
        kat_ok = (memcmp(cpu, gpu, 32) == 0);
    }

    // neg: flip one header bit and require the device's digest to differ from
    // the reference's digest for the UNFLIPPED header at the same nonce.
    // Comparing the two device digests would not do: their nonces can differ,
    // so they would differ for a reason that proves nothing.
    uint8_t neg_gpu[32], neg_cpu[32];
    uint32_t hdr2[20];
    memcpy(hdr2, hdr, 80);
    hdr2[3] ^= 0x00000008u;
    found = solved = 0;
    RinHash_mine_coop(thr_id, hdr2, 19, kat_nonce, kat_span, target,
                            &found, dummy, neg_gpu, &solved, &hashed);
    if (!solved) {
        selftest_cuda_fault();
    } else if (found < kat_nonce || found >= kat_nonce + kat_span) {
        gpulog(LOG_ERR, thr_id, "RinHash self-test: neg nonce %08x outside the span", found);
    } else {
        uint32_t neg_hdr[20];
        memcpy(neg_hdr, hdr, 80);      // the ORIGINAL header
        neg_hdr[19] = found;
        rinhash_hash(neg_cpu, neg_hdr);
        neg_ok = (memcmp(neg_gpu, neg_cpu, 32) != 0);
    }

    const bool passed = sha3_ok && b3_ok && kat_ok && neg_ok;
    if (!passed)
        gpulog(LOG_ERR, thr_id, "RinHash self-test FAILED (sha3 %d b3 %d kat %d neg %d)",
               (int)sha3_ok, (int)b3_ok, (int)kat_ok, (int)neg_ok);

    return selftest_gate(thr_id, "RinHash", passed);
}

// CPU reference for one 80-byte header. Parameters must track the kernel:
// t_cost 2, m_cost 64 KiB, lanes 1, salt "RinCoinSalt" -- see the
// device_argon2d_hash call in rinhash.cu.
extern "C" void rinhash_hash(void *output, const void *input)
{
    static const uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
    uint8_t blake3_out[32], argon2_out[32];

    blake3_256((const uint8_t *)input, 80, blake3_out);
    if (argon2d_hash_raw(2, 64, 1, blake3_out, sizeof(blake3_out),
                         salt, sizeof(salt), argon2_out, sizeof(argon2_out)) != ARGON2_OK) {
        memset(output, 0xFF, 32);   // cannot verify -> fail the compare, never pass it
        return;
    }
    sha3(argon2_out, sizeof(argon2_out), output, 32);
}
