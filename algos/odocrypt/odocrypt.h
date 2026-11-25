#ifndef ODOCRYPT_H__
#define ODOCRYPT_H__ 1

#include <stdint.h>
#include <stddef.h>

// Odocrypt (DigiByte) — a self-mutating SPN block cipher whose S-boxes, P-boxes,
// rotations and round keys are regenerated every "shapechange" epoch from a
// 32-bit key (derived from the block time). Scalar host table generator ported
// from DigiByte Core's src/crypto/odocrypt.cpp (forward/Encrypt path only); the
// per-nonce hash runs on the GPU in cuda_odocrypt.cu.
//
//   hash = first 32 bytes of KeccakP800_12( OdoCrypt(key).Encrypt( header||0x01 ) )
//   key  = nTime - (nTime % ODO_SHAPECHANGE_INTERVAL)

#define ODO_DIGEST_SIZE          80          // block size in bytes
#define ODO_ROUNDS               84
#define ODO_SMALL_SBOX_WIDTH     6
#define ODO_LARGE_SBOX_WIDTH     10
#define ODO_PBOX_SUBROUNDS       6
#define ODO_PBOX_M               3
#define ODO_ROTATION_COUNT       6
#define ODO_WORD_BITS            64
#define ODO_STATE_SIZE           10          // (ODO_DIGEST_SIZE*8)/ODO_WORD_BITS
#define ODO_SMALL_SBOX_COUNT     40          // (DIGEST_BITS)/(6+10)
#define ODO_LARGE_SBOX_COUNT     10          // == ODO_STATE_SIZE
#define ODO_SHAPECHANGE_INTERVAL 864000      // 10 days, in seconds

typedef struct
{
   uint64_t mask[ODO_PBOX_SUBROUNDS][ODO_STATE_SIZE / 2];
   int      rotation[ODO_PBOX_SUBROUNDS - 1][ODO_STATE_SIZE / 2];
} OdoPbox;

typedef struct
{
   uint8_t  Sbox1[ODO_SMALL_SBOX_COUNT][1 << ODO_SMALL_SBOX_WIDTH];   // [40][64]
   uint16_t Sbox2[ODO_LARGE_SBOX_COUNT][1 << ODO_LARGE_SBOX_WIDTH];   // [10][1024]
   OdoPbox  Permutation[2];
   int      Rotations[ODO_ROTATION_COUNT];
   uint16_t RoundKey[ODO_ROUNDS];
} OdoCrypt;

#ifdef __cplusplus
extern "C" {
#endif

// Build the cipher tables for a given epoch key (host).
void odocrypt_init( OdoCrypt *c, uint32_t key );

// Reference host hash (Encrypt + KeccakP800_12) used for CPU re-verification of
// GPU candidates and for the known-answer self-test. 'input' is the 80-byte
// block header; 'output' receives 32 bytes.
void odo_hash_host( const OdoCrypt *c, void *output, const void *input );

#ifdef __cplusplus
}
#endif

#endif /* ODOCRYPT_H__ */
