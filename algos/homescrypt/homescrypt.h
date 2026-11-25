/**
 * homescrypt (Lumenite) -- HomeScrypt v1.2 "FP6", CPU reference interface.
 *
 * seed  = SHA-256("HOMESCRYPT-V1.2-SEED" || header || scrypt_1024_1_1_256(header))
 * fill  a 16 MiB scratchpad (2^21 u64) with a serial splitmix chain
 * mix   2^16 rounds of three dependent random reads, a 192-FMA binary32 chain
 *       and three ordered writes (the writes can alias; the order is consensus)
 * fold  the whole scratchpad into four words
 * out   = SHA-256("HOMESCRYPT-V1.2-FINAL" || seed || fold || header)
 *
 * The only floating point is a correctly rounded fmaf.  Difficulty is on the
 * 2^32 scale.
 */
#ifndef HOMESCRYPT_H
#define HOMESCRYPT_H

#include <stdint.h>
#include <stdlib.h>

#define HOMESCRYPT_WORDS      (1u << 21)   /* scratchpad, u64 words */
#define HOMESCRYPT_MIX_ROUNDS 65536u
#define HOMESCRYPT_FP_PASSES  6u

#ifdef __cplusplus
extern "C" {
#endif

/* `input` is the 80-byte header as hashed (the be32enc'd work data).
 * 32-byte digest out, uint256 byte order. */
void homescrypt_hash(void *output, const void *input);

/* With the mix shape as a parameter; the self-test uses a short shape. */
void homescrypt_hash_tuned(void *output, const void *input, uint32_t mix_rounds, uint32_t fp_passes);

/* Known-answer self-test over homescrypt_kat.h plus a flipped-header
 * negative.  `detail` receives a short summary either way. */
#define HOMESCRYPT_SELFTEST_DETAIL 160
bool homescrypt_self_test(char *detail, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* HOMESCRYPT_H */
