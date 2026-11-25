// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verthash (Vertcoin) CPU reference / verify oracle — scalar path.
// Provenance: cpuminer-opt algo/verthash/Verthash.c + verthash-gate.c
// (CryptoGraphics, GPLv2). Rewritten self-contained (no thread-local prehash
// state) so it is reentrant and usable as the host re-verify safety net.
//
// Algorithm (per nonce):
//   hash[32]  = SHA3-256(header[80])                        -- running output
//   subset[512] = 8 x SHA3-512( header with byte0 += i+1 )  -- i = 0..7
//   acc = 0x811c9dc5
//   for r in 0..31, for i in 0..127:
//     idx      = fnv1a( rol32(subset[i], r), acc ) % mdiv   -- r=0 => no rotate
//     blob_off = blob + idx*4  (uint32 units; 16-byte-aligned seek, 32-byte read)
//     acc      = fnv1a-chain over blob_off[0..7]
//     hash[j]  = fnv1a( hash[j], blob_off[j] )   for j = 0..7
//   output = hash

#include "verthash-cpu.h"
#include "sph/sha3.h"   // vendored FIPS-202 tiny_sha3 (0x06 pad)
#include <string.h>

#define VH_P0_SIZE   64
#define VH_N_ITER    8
#define VH_N_SUBSET  (VH_P0_SIZE * VH_N_ITER)   // 512
#define VH_N_ROT     32

#define fnv1a(a, b) (((a) ^ (b)) * 0x1000193U)

static inline uint32_t rol32(uint32_t x, uint32_t n)
{
    return (x << n) | (x >> (32 - n));
}

void verthash_hash_oracle(const uint8_t *blob_bytes, size_t blob_size,
                          const void *input, void *output)
{
    uint32_t subset[VH_N_SUBSET / 4];   // 128 words = 512 bytes
    uint32_t hash[VH_HASH_OUT_SIZE / 4];
    const uint32_t *blob = (const uint32_t *) blob_bytes;
    uint32_t accumulator = 0x811c9dc5U;
    const uint32_t mdiv = (uint32_t)
        (((blob_size - VH_HASH_OUT_SIZE) / VH_BYTE_ALIGNMENT) + 1);

    // 1) SHA3-256 of the 80-byte header -> the running 32-byte output.
    sha3(input, VH_HEADER_SIZE, hash, VH_HASH_OUT_SIZE);

    // 2) 8 x SHA3-512, each over the header with byte[0] incremented by (i+1),
    //    producing 8 x 64 = 512 bytes = the 128-word subset.
    {
        uint8_t in[VH_HEADER_SIZE];
        memcpy(in, input, VH_HEADER_SIZE);
        for (int i = 0; i < VH_N_ITER; i++) {
            in[0] += 1;
            sha3(in, VH_HEADER_SIZE, ((uint8_t *) subset) + i * 64, 64);
        }
    }

    // 3) 32 rotations x 128 iterations = 4096 random 32-byte reads.
    for (uint32_t r = 0; r < VH_N_ROT; r++) {
        for (uint32_t i = 0; i < VH_N_SUBSET / 4; i++) {
            const uint32_t seek = (r == 0) ? subset[i] : rol32(subset[i], r);
            const uint32_t *blob_off =
                blob + (fnv1a(seek, accumulator) % mdiv) * 4;

            accumulator = fnv1a(accumulator, blob_off[0]);
            accumulator = fnv1a(accumulator, blob_off[1]);
            accumulator = fnv1a(accumulator, blob_off[2]);
            accumulator = fnv1a(accumulator, blob_off[3]);
            accumulator = fnv1a(accumulator, blob_off[4]);
            accumulator = fnv1a(accumulator, blob_off[5]);
            accumulator = fnv1a(accumulator, blob_off[6]);
            accumulator = fnv1a(accumulator, blob_off[7]);

            for (int j = 0; j < 8; j++)
                hash[j] = fnv1a(hash[j], blob_off[j]);
        }
    }

    memcpy(output, hash, VH_HASH_OUT_SIZE);
}
