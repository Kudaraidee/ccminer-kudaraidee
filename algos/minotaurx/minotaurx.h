/**
 * minotaurx (Avian / Pulsar) -- CPU reference interface.
 *
 * seed = SHA-512(header); node i hashes with algo seed[i] % 16 (minotaur's own
 * order, not x16r's); the branch is bit 0 of byte 63 of each node's output; the
 * walk visits 7 nodes and ends in yespower 1.0 (N=2048, r=8, pers "et in arcadia
 * ego") over the 64-byte running hash.  No Hamsi skip (that is plain minotaur).
 * Difficulty is the 2^32 scale.
 */

#ifndef MINOTAURX_H
#define MINOTAURX_H

#include <stdint.h>
#include <stdlib.h>

#define MINOTAURX_ALGO_COUNT 16     /* cheap algos; index 16 is yespower */
#define MINOTAURX_NODES      22
#define MINOTAURX_WALK_LEN   7      /* nodes visited per nonce, yespower included */

#ifdef __cplusplus
extern "C" {
#endif

/* `input` is the 80-byte header (be32enc'd work data).  32-byte digest out.
 * False only if the yespower reference refused; the digest is then all-ones. */
bool minotaurx_hash(void *output, const void *input);

/* The walk up to yespower: `out64` receives yespower's input.  `path`, if not
 * NULL, receives the MINOTAURX_WALK_LEN-1 cheap algo indices visited. */
void minotaurx_chain(void *out64, const void *input, uint8_t *path);

/* The yespower node alone on a 64-byte input; all-ones and false on refusal. */
bool minotaurx_yespower64(void *out32, const void *in64);

/* Chain state entering yespower for Pulsar block 7920000; its digest is that
 * block's (cpuminer-opt algo/yespower/yespower-kat.h). */
extern const uint8_t minotaurx_kat_node21_in[64];

/* Known-answer self-test on four Pulsar mainnet headers (digest and target) plus
 * a flipped-nonce check.  True only on a full pass; `detail` gets one line.
 * Logs nothing, so it links into a standalone harness. */
#define MINOTAURX_SELFTEST_DETAIL 224
bool minotaurx_self_test(char *detail, size_t detail_len);

#ifdef __cplusplus
}
#endif

#endif /* MINOTAURX_H */
