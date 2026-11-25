/**
 * Flex (Kylacoin / Lyncoin) -- CPU reference interface.
 *
 * A GhostRider sibling, but with four independent consensus differences.  Do
 * not assume anything carries over from algos/ghostrider or algos/mike:
 *
 * 1. EVERY keccak in the chain is SHA-3 (0x06 padding), not legacy Keccak
 * (0x01).  The CPU reference gets this from sph_sha3d512 / sph_sha3d256;
 * the shared device library in cuda/keccak_device.cuh bakes in 0x01 and is
 * used by nine other algos, so the GPU side needs its own kernels.
 *
 * 2. The selection seed is SHA3-512 of the WHOLE 80-byte header, so the core
 * order and the CN triple depend on the NONCE.  Every nonce takes a
 * different chain.  (ghostrider and mike seed from bytes [4,36), which is
 * nonce-independent -- that is what lets them run a whole-batch pipeline
 * and what flex cannot do.)
 *
 * 3. The CryptoNight finalization is `flex_extra_hashes[state[0] & 2]` over
 * {blake, groestl, skein-512}: JH is never used, groestl is unreachable,
 * and skein is 512-bit (64 bytes), not 256-bit.  Parameters (memory,
 * iterations, lite half-addressing) are byte-identical to ghostrider's --
 * only the ending differs, and it differs on roughly half of all inputs.
 *
 * 4. The high 32 bytes are NOT zeroed after a CN round, and a final SHA3-256
 * over all 64 bytes closes the chain.  So the high 32 bytes of a CN result
 * -- which ghostrider discards -- are consensus-load-bearing here.
 *
 * Core pool: 14 algos, the x16 set minus JH and SHA-512, REINDEXED.  A gr core
 * index does not mean the same thing as a flex one from index 3 upward.
 *
 * The chain runs 15 core rounds, not 14: the reference indexes coreOrder[14]
 * although only slots 0..13 are ever selected, so slot 14 stays zero and the
 * fifteenth core round is ALWAYS blake512.  That is consensus, confirmed by a
 * pool-accepted vector.
 *
 * Difficulty: flex leaves opt_target_factor at 1.0.  It does NOT share
 * ghostrider's 2^16 factor.
 * Merkle root: sha3d over the coinbase, then sha256d for each branch.
 */

#ifndef FLEX_H
#define FLEX_H

#include <stdint.h>
#include <stdlib.h>

/* Core table order: the x16 set minus JH and SHA-512.  The terminator IS the
 * pool size, so the count cannot drift from the table. */
enum FlexAlgo {
	FLEX_BLAKE = 0, FLEX_BMW, FLEX_GROESTL, FLEX_KECCAK, FLEX_SKEIN,
	FLEX_LUFFA, FLEX_CUBEHASH, FLEX_SHAVITE, FLEX_SIMD, FLEX_ECHO,
	FLEX_HAMSI, FLEX_FUGUE, FLEX_SHABAL, FLEX_WHIRLPOOL,
	FLEX_CORE_ALGO_COUNT          /* == 14 */
};

/* CryptoNight-v1 variant table order.  Same order as ghostrider's. */
enum FlexCNAlgo {
	FLEX_CN_DARK = 0, FLEX_CN_DARKLITE, FLEX_CN_FAST, FLEX_CN_LITE,
	FLEX_CN_TURTLE, FLEX_CN_TURTLELITE,
	FLEX_CN_ALGO_COUNT            /* == 6 */
};

/* Core rounds actually executed.  One MORE than the pool size, because the
 * chain reads coreOrder[14], which selection never fills -- see the header
 * comment.  Anything sizing a core-order array must use this, not the count. */
#define FLEX_CORE_CHAIN_LEN (FLEX_CORE_ALGO_COUNT + 1)   /* == 15 */

/* Number of CN rounds, and the core rounds that precede each. */
#define FLEX_CN_ROUNDS 3

#ifdef __cplusplus
extern "C" {
#endif

/* Consensus hash.  `input` is the 80-byte header in the byteswapped
 * (per-32-bit-word big-endian) frame scanhash prepares.  `output` receives
 * 32 bytes. */
void flex_hash(void *output, const void *input);

/* Derive the per-nonce chain for one header: `coreOrder` receives
 * FLEX_CORE_CHAIN_LEN bytes (slot 14 always 0 = blake) and `cnOrder` receives
 * FLEX_CN_ALGO_COUNT, of which the chain uses the first three.  `seed` may be
 * NULL; if not, it receives the 64-byte SHA3-512 selection seed.
 *
 * Split out of flex_hash because the GPU driver derives the same orders on
 * device and must be able to compare against this, lane by lane. */
void flex_derive_order(const void *input, uint8_t *coreOrder, uint8_t *cnOrder,
                       uint8_t *seed);

/* Known-answer self-test.  Returns true only on a full pass; on failure
 * `detail` receives a one-line description.  This TU does not log or include
 * miner.h, so the CPU reference links into a standalone harness. */
#define FLEX_SELFTEST_DETAIL 224
bool flex_self_test(char *detail, size_t detail_len);

/* Deduplicated nibble walk, low nibble of each byte first, each reduced
 * % algoCount.  Reads size/2 bytes. */
void flex_get_algo_string(const void *mem, unsigned int size,
                          uint8_t *selectedAlgoOutput, int algoCount);

/* Single core / CN round on the CPU, for the GPU driver's step-by-step check.
 *
 * flex_cn_algo_cpu requires `out` to be a 64-byte buffer and copies `in`
 * into it first: on the blake branch the CN writes only the low 32 bytes, and
 * the high 32 must then read back as the round's own input.  Passing a 32-byte
 * buffer, or skipping the pre-seed, changes the digest. */
void flex_core_algo_cpu(int algo, const void *in, void *out, size_t size);
void flex_cn_algo_cpu(int cnAlgo, const void *in, void *out, size_t size);

/* SHA-3 (0x06) helpers -- exported so the GPU kernels can be gated against
 * the same code the chain uses. */
void flex_sha3_512(void *out, const void *in, size_t len);
void flex_sha3_256(void *out, const void *in, size_t len);

/* sha3d (double SHA3-256) over arbitrary input -- flex's merkle root leaf. */
void flex_sha3d(void *out, const void *in, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* FLEX_H */
