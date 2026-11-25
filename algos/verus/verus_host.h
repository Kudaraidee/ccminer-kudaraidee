/* VerusHash 2.2 host side: a plain-C API over the vendored cpuminer-opt chain
 * (haraka.c, verus_clhash.c), so no miner TU has to see the SIMD types.
 *
 * Preimage = 140-byte header || CompactSize(1344) || solution = 1487 bytes, and
 * 1487 = 46*32 + 15: the first 1472 bytes are job-constant (the prologue), only
 * the last 15 change per nonce.
 */
#ifndef VERUS_HOST_H
#define VERUS_HOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VERUS_KEY_SIZE       8832  /* 512*16 mutable + 40*16 keyed-Haraka headroom */
#define VERUS_KEY_SIZE128     552
#define VERUS_SCRATCH_U128     64  /* FixKey journal: 32 prand + 32 prandex saves  */
#define VERUS_KEYBUF_BYTES   ( ( VERUS_KEY_SIZE128 + VERUS_SCRATCH_U128 ) * 16 )
#define VERUS_HEADER_SIZE     140
#define VERUS_BASE_SIZE       143  /* header + 3-byte CompactSize */
#define VERUS_SOLUTION_FIXED 1344  /* VRSC SOLUTION_SIZE_FIXED */
#define VERUS_SOLUTION_MAX   2048
#define VERUS_PREIMAGE_MAX   ( VERUS_BASE_SIZE + VERUS_SOLUTION_MAX )
#define VERUS_NONCE_SPACE      15  /* the last 15 bytes of the preimage */

/* 1 if this CPU has AES-NI, PCLMULQDQ and SSSE3 (and the build has the code) */
int verus_host_cpu_ok( void );

/* CBlockHeader::ClearNonCanonicalData() for PBaaS (solution[0] >= 7 and
 * solution[5] > 0): zero prevhash, merkle, sapling root, nBits, nNonce and the
 * two MMR roots; nVersion and nTime stay. */
void verus_host_clear_noncanonical( uint8_t *pre );

/* Pools send a short template; pad it to the chain's fixed size, or to the
 * 15 (mod 32) minimum if it is larger. */
int verus_host_padded_solution_size( int received );

/* Job prologue: 46 x Haraka512 over pre[0..half_len) -> half[64] (state +
 * FillExtra), then 276 x Haraka256 -> the pristine key. keybuf is
 * VERUS_KEYBUF_BYTES, 16-byte aligned. Returns 0 if the host cannot run it. */
int verus_host_prologue( const uint8_t *pre, int half_len, uint8_t half[64],
                         void *keybuf );

/* One nonce over a prologue'd key, which is restored (FixKey) on return.
 * full = 0 writes only hash[28..31], the word the target test reads; full = 1
 * writes all 32 bytes. Returns the clhash intermediate. */
uint64_t verus_host_hash( uint8_t hash[32], const uint8_t half[64],
                          const uint8_t nonce15[15], void *keybuf, int full );

/* Full 32-byte digest of a whole preimage (pre_len = 15 mod 32). */
int verus_host_full( uint8_t out[32], const uint8_t *pre, int pre_len );

#ifdef __cplusplus
}
#endif

#endif /* VERUS_HOST_H */
