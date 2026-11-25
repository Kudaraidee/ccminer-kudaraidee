/**
 * Mike (VKAX / FortuneBlock) -- CPU reference interface.
 *
 * Mike is GhostRider with the core pool reduced from 15 algorithms to 11,
 * which shortens the third core group from five rounds to one:
 *
 * ghostrider   5 core, CN, 5 core, CN, 5 core, CN   (15 core + 3 CN)
 * mike         5 core, CN, 5 core, CN, 1 core, CN   (11 core + 3 CN)
 *
 * Everything else matches GhostRider: the core table (entries 0..10), all six
 * CryptoNight-v1 variants and their parameters, the CN finalization, the
 * post-CN zeroing of the high 32 bytes, the nibble walk, the sha256d merkle
 * root, and the 2^16 difficulty factor.
 *
 * The order is derived from header bytes [4,36) and is therefore constant for
 * a whole job -- the nonce at byte 76 does not affect it.
 *
 * WARNING: the core order is a permutation of 0..10 from reducing each header
 * nibble % 11.  It is NOT GhostRider's 15-wide permutation truncated to 11
 * entries.  Using 15 gives a wrong hash on nearly every input, with no symptom
 * until a pool rejects the share -- and a sparse test vector passes either
 * way.  MIKE_CORE_ALGO_COUNT keeps that number in one place, and
 * mike_kat_dense[] (mike_kat.h) is the only gate that can see it.
 */

#ifndef MIKE_H
#define MIKE_H

#include <stdint.h>
#include <stdlib.h>

/* Core table order.  Identical to ghostrider's entries 0..10, so a core index
 * means the same thing in both algos.  The terminator IS the pool size, so the
 * count cannot drift away from the table. */
enum MikeAlgo {
	MIKE_BLAKE = 0, MIKE_BMW, MIKE_GROESTL, MIKE_JH, MIKE_KECCAK,
	MIKE_SKEIN, MIKE_LUFFA, MIKE_CUBEHASH, MIKE_SHAVITE, MIKE_SIMD,
	MIKE_ECHO,
	MIKE_CORE_ALGO_COUNT          /* == 11 */
};

/* CryptoNight-v1 variant table order.  Identical to ghostrider's. */
enum MikeCNAlgo {
	MIKE_CN_DARK = 0, MIKE_CN_DARKLITE, MIKE_CN_FAST, MIKE_CN_LITE,
	MIKE_CN_TURTLE, MIKE_CN_TURTLELITE,
	MIKE_CN_ALGO_COUNT            /* == 6 */
};

#ifdef __cplusplus
extern "C" {
#endif

/* Consensus hash.  `input` is the 80-byte header in the byteswapped
 * (per-32-bit-word big-endian) frame scanhash prepares.  `output` receives
 * 32 bytes. */
void mike_hash(void *output, const void *input);

/* Known-answer self-test over both vector sets in mike_kat.h.  Returns true
 * only on a full pass; on failure `detail` receives a one-line description
 * (pass a buffer of at least MIKE_SELFTEST_DETAIL bytes).
 *
 * This TU deliberately does not log or include miner.h, so the CPU reference
 * links into a standalone harness with nothing but sph/ and the CryptoNight
 * CPU cores.  The caller owns the logging. */
#define MIKE_SELFTEST_DETAIL 192
bool mike_self_test(char *detail, size_t detail_len);

/* Deduplicated nibble walk over `mem`, low nibble of each byte first, each
 * reduced % algoCount, keeping the first occurrence of each distinct value
 * until `algoCount` are chosen, then back-filling any that never appeared.
 * Reads size/2 bytes. */
void mike_get_algo_string(const void *mem, unsigned int size,
                          uint8_t *selectedAlgoOutput, int algoCount);

/* Single core / CN round on the CPU.  Exposed so the GPU driver's
 * startup pipeline check can compare against them step by step. */
void mike_core_algo_cpu(int algo, const void *in, void *out, size_t size);
void mike_cn_algo_cpu(int cnAlgo, const void *in, void *out, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* MIKE_H */
