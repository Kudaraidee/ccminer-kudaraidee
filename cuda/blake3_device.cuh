// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * BLAKE3-256, unkeyed, SINGLE CHUNK (input <= 1024 bytes), integer-only.
 *
 * One body compiled for host and device, so a consensus hash cannot drift
 * between the two sides. Everything is `static` (internal linkage), so several
 * translation units may include it without a multiple-definition link error and
 * without pulling in any dependency.
 *
 * Validated against the published BLAKE3 test vectors at the lengths that decide
 * the flag and block-length logic: 0, 1, 63, 64, 65, 127, 128, 129, 180, 512 and
 * 1024. The 180-byte case is decred's header -- three blocks, last block 52
 * bytes, and a middle block carrying NO flags.
 *
 * blake3_256() is NOT valid for input > 1024 bytes; blake3_hash() (below) is the
 * full chunk tree, keyed or unkeyed. This is the tree's ONLY BLAKE3: decred,
 * hoohash, rinhash and pearlhash all use it, so a change to the shared core lands
 * on four consensus algos at once.
 */
#ifndef CUDA_BLAKE3_DEVICE_CUH
#define CUDA_BLAKE3_DEVICE_CUH

#include <stdint.h>
#include <stddef.h>

#ifdef __CUDACC__
#define B3_HD __host__ __device__
#define B3_HDI __host__ __device__ __forceinline__
#else
#define B3_HD
#define B3_HDI inline
#endif

#define BLAKE3_OUT_LEN		32
#define BLAKE3_BLOCK_LEN	64
#define BLAKE3_CHUNK_LEN	1024

/* flag bits (the only ones a single chunk needs) */
#define BLAKE3_CHUNK_START	1u
#define BLAKE3_CHUNK_END	2u
#define BLAKE3_ROOT		8u

static B3_HDI uint32_t blake3_rotr(uint32_t w, int c)
{
	return (w >> c) | (w << (32 - c));
}

static B3_HDI void blake3_g(uint32_t *s, int a, int b, int c, int d, uint32_t mx, uint32_t my)
{
	s[a] = s[a] + s[b] + mx;
	s[d] = blake3_rotr(s[d] ^ s[a], 16);
	s[c] = s[c] + s[d];
	s[b] = blake3_rotr(s[b] ^ s[c], 12);
	s[a] = s[a] + s[b] + my;
	s[d] = blake3_rotr(s[d] ^ s[a], 8);
	s[c] = s[c] + s[d];
	s[b] = blake3_rotr(s[b] ^ s[c], 7);
}

static B3_HD void blake3_round(uint32_t *s, const uint32_t *m)
{
	blake3_g(s, 0, 4, 8,  12, m[0],  m[1]);
	blake3_g(s, 1, 5, 9,  13, m[2],  m[3]);
	blake3_g(s, 2, 6, 10, 14, m[4],  m[5]);
	blake3_g(s, 3, 7, 11, 15, m[6],  m[7]);
	blake3_g(s, 0, 5, 10, 15, m[8],  m[9]);
	blake3_g(s, 1, 6, 11, 12, m[10], m[11]);
	blake3_g(s, 2, 7, 8,  13, m[12], m[13]);
	blake3_g(s, 3, 4, 9,  14, m[14], m[15]);
}

static B3_HD const uint32_t *blake3_iv(void)
{
	static const uint32_t IV[8] = {
		0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
		0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
	};
	return IV;
}

/* One compression. out_cv gets the 8-word chaining value, which for a block
 * flagged ROOT is the 256-bit digest. */
static B3_HD void blake3_compress(const uint32_t cv[8], const uint32_t block[16],
	uint64_t counter, uint32_t block_len, uint32_t flags, uint32_t out_cv[8])
{
	const uint32_t IV[8] = {
		0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
		0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
	};
	const uint8_t SIGMA[16] = { 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8 };

	uint32_t s[16];
	s[0] = cv[0]; s[1] = cv[1]; s[2] = cv[2]; s[3] = cv[3];
	s[4] = cv[4]; s[5] = cv[5]; s[6] = cv[6]; s[7] = cv[7];
	s[8] = IV[0]; s[9] = IV[1]; s[10] = IV[2]; s[11] = IV[3];
	s[12] = (uint32_t)counter;
	s[13] = (uint32_t)(counter >> 32);
	s[14] = block_len;
	s[15] = flags;

	uint32_t m[16];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		m[i] = block[i];

	for (int r = 0; r < 7; r++) {
		blake3_round(s, m);
		if (r < 6) {
			uint32_t pm[16];
			#pragma unroll
			for (int i = 0; i < 16; i++) pm[i] = m[SIGMA[i]];
			#pragma unroll
			for (int i = 0; i < 16; i++) m[i] = pm[i];
		}
	}

	#pragma unroll
	for (int i = 0; i < 8; i++)
		out_cv[i] = s[i] ^ s[i + 8];
}

static B3_HDI void blake3_words_from_le(const uint8_t *b, uint32_t *w)
{
	#pragma unroll
	for (int i = 0; i < 16; i++)
		w[i] = (uint32_t)b[i*4] | ((uint32_t)b[i*4+1] << 8)
		     | ((uint32_t)b[i*4+2] << 16) | ((uint32_t)b[i*4+3] << 24);
}

/* Unkeyed BLAKE3-256 over a single chunk. out receives 32 bytes. */
static B3_HD void blake3_256(const uint8_t *input, size_t len, uint8_t *out)
{
	const uint32_t *IV = blake3_iv();

	uint32_t cv[8];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		cv[i] = IV[i];

	size_t pos = 0;
	uint32_t blocks_done = 0;
	for (;;) {
		const size_t remaining = len - pos;
		const uint32_t block_len = remaining >= BLAKE3_BLOCK_LEN
			? (uint32_t)BLAKE3_BLOCK_LEN : (uint32_t)remaining;
		const bool is_last = (remaining <= BLAKE3_BLOCK_LEN);

		uint8_t buf[BLAKE3_BLOCK_LEN];
		#pragma unroll
		for (int i = 0; i < BLAKE3_BLOCK_LEN; i++)
			buf[i] = (i < (int)block_len) ? input[pos + i] : 0;

		uint32_t block[16];
		blake3_words_from_le(buf, block);

		uint32_t flags = 0;
		if (blocks_done == 0) flags |= BLAKE3_CHUNK_START;
		if (is_last)          flags |= BLAKE3_CHUNK_END | BLAKE3_ROOT;

		uint32_t next[8];
		blake3_compress(cv, block, 0 /* single chunk => counter 0 */,
			block_len, flags, next);
		#pragma unroll
		for (int i = 0; i < 8; i++)
			cv[i] = next[i];

		blocks_done++;
		pos += block_len;
		if (is_last)
			break;
	}

	#pragma unroll
	for (int i = 0; i < 8; i++) {
		out[i*4]   = (uint8_t)(cv[i]);
		out[i*4+1] = (uint8_t)(cv[i] >> 8);
		out[i*4+2] = (uint8_t)(cv[i] >> 16);
		out[i*4+3] = (uint8_t)(cv[i] >> 24);
	}
}

/* ---------------------------------------------------------------------------
 * Midstate split, for a nonce that lives in the LAST block.
 *
 * The blocks before the nonce are per-job constants, so the host compresses
 * them once and the device only ever compresses the final block. Kept as
 * separate code from blake3_256() above on purpose: that one is anchored to
 * the published vectors, so it serves as the oracle for this one.
 */

/* Compress whole 64-byte blocks that are NOT the last block of the chunk.
 * nblocks must be >= 1, and there must be at least one more block after these. */
static B3_HD void blake3_prefix(const uint8_t *in, uint32_t nblocks, uint32_t cv[8])
{
	const uint32_t *IV = blake3_iv();
	for (int i = 0; i < 8; i++)
		cv[i] = IV[i];

	for (uint32_t b = 0; b < nblocks; b++) {
		uint32_t block[16], next[8];
		blake3_words_from_le(in + (size_t)b * BLAKE3_BLOCK_LEN, block);
		blake3_compress(cv, block, 0 /* single chunk */, BLAKE3_BLOCK_LEN,
			b == 0 ? BLAKE3_CHUNK_START : 0u, next);
		for (int i = 0; i < 8; i++)
			cv[i] = next[i];
	}
}

/* Compress the chunk's final block from a midstate. m holds the 16 little-endian
 * words of the block, zero-padded past block_len; the caller may patch a nonce
 * word into it directly, which is the point of the split. first_block is for the
 * degenerate one-block chunk, where CHUNK_START also belongs here. */
static B3_HD void blake3_final(const uint32_t cv[8], const uint32_t m[16],
	uint32_t block_len, bool first_block, uint8_t *out)
{
	uint32_t flags = BLAKE3_CHUNK_END | BLAKE3_ROOT;
	if (first_block)
		flags |= BLAKE3_CHUNK_START;

	uint32_t next[8];
	blake3_compress(cv, m, 0 /* single chunk */, block_len, flags, next);

	for (int i = 0; i < 8; i++) {
		out[i*4]   = (uint8_t)(next[i]);
		out[i*4+1] = (uint8_t)(next[i] >> 8);
		out[i*4+2] = (uint8_t)(next[i] >> 16);
		out[i*4+3] = (uint8_t)(next[i] >> 24);
	}
}

/* ---------------------------------------------------------------------------
 * Keyed mode and the full chunk tree. blake3_keyed_block(): keyed hash of one
 * 64-byte block. blake3_hash(): any length, keyed (key != NULL) or not.
 */
#define BLAKE3_PARENT		4u
#define BLAKE3_KEYED_HASH	16u

static B3_HDI void blake3_cv_to_bytes(const uint32_t cv[8], uint8_t *out)
{
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		out[i*4]   = (uint8_t)(cv[i]);
		out[i*4+1] = (uint8_t)(cv[i] >> 8);
		out[i*4+2] = (uint8_t)(cv[i] >> 16);
		out[i*4+3] = (uint8_t)(cv[i] >> 24);
	}
}

/* keyed BLAKE3-256 of exactly one 64-byte block (m = 16 LE words) */
static B3_HD void blake3_keyed_block(const uint32_t key[8], const uint32_t m[16], uint32_t out_cv[8])
{
	blake3_compress(key, m, 0, BLAKE3_BLOCK_LEN,
		BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT | BLAKE3_KEYED_HASH, out_cv);
}

/* A node whose compression is still pending, so the caller can decide ROOT. */
typedef struct {
	uint32_t cv[8];
	uint32_t block[16];
	uint64_t counter;
	uint32_t block_len;
	uint32_t flags;
} blake3_node;

/* All blocks of one chunk except the last are compressed; the last is left pending. */
static B3_HD void blake3_chunk_node(const uint32_t key[8], const uint8_t *in, size_t len,
	uint64_t chunk_counter, uint32_t base_flags, blake3_node *node)
{
	uint32_t cv[8];
	for (int i = 0; i < 8; i++) cv[i] = key[i];
	size_t pos = 0;
	uint32_t start = BLAKE3_CHUNK_START;
	while (len - pos > BLAKE3_BLOCK_LEN) {
		uint32_t block[16], next[8];
		blake3_words_from_le(in + pos, block);
		blake3_compress(cv, block, chunk_counter, BLAKE3_BLOCK_LEN, base_flags | start, next);
		for (int i = 0; i < 8; i++) cv[i] = next[i];
		start = 0;
		pos += BLAKE3_BLOCK_LEN;
	}
	uint8_t buf[BLAKE3_BLOCK_LEN];
	for (int i = 0; i < BLAKE3_BLOCK_LEN; i++)
		buf[i] = (pos + i < len) ? in[pos + i] : 0;
	for (int i = 0; i < 8; i++) node->cv[i] = cv[i];
	blake3_words_from_le(buf, node->block);
	node->counter = chunk_counter;
	node->block_len = (uint32_t)(len - pos);
	node->flags = base_flags | start | BLAKE3_CHUNK_END;
}

static B3_HD void blake3_parent_node(const uint32_t key[8], const uint32_t left[8],
	const uint32_t right[8], uint32_t base_flags, blake3_node *node)
{
	for (int i = 0; i < 8; i++) {
		node->cv[i] = key[i];
		node->block[i] = left[i];
		node->block[8 + i] = right[i];
	}
	node->counter = 0;
	node->block_len = BLAKE3_BLOCK_LEN;
	node->flags = base_flags | BLAKE3_PARENT;
}

static B3_HD void blake3_node_cv(const blake3_node *node, uint32_t extra_flags, uint32_t out_cv[8])
{
	blake3_compress(node->cv, node->block, node->counter, node->block_len,
		node->flags | extra_flags, out_cv);
}

/* BLAKE3-256 of any length; key == NULL for the unkeyed hash. Standard tree:
 * chunk CVs are merged on a stack whenever the chunk count gains a trailing zero
 * bit, and only the final node is compressed with ROOT. 54 stack entries cover
 * the 2^64-byte maximum. */
static B3_HD void blake3_hash(const uint8_t *in, size_t len, const uint8_t *key, uint8_t *out)
{
	uint32_t kw[8];
	uint32_t base_flags = 0;
	if (key) {
		for (int i = 0; i < 8; i++)
			kw[i] = (uint32_t)key[i*4] | ((uint32_t)key[i*4+1] << 8)
			      | ((uint32_t)key[i*4+2] << 16) | ((uint32_t)key[i*4+3] << 24);
		base_flags = BLAKE3_KEYED_HASH;
	} else {
		const uint32_t *IV = blake3_iv();
		for (int i = 0; i < 8; i++) kw[i] = IV[i];
	}

	uint32_t stack[54][8];
	int depth = 0;
	uint64_t chunk = 0;
	size_t pos = 0;
	while (len - pos > BLAKE3_CHUNK_LEN) {
		blake3_node node;
		uint32_t cv[8];
		blake3_chunk_node(kw, in + pos, BLAKE3_CHUNK_LEN, chunk, base_flags, &node);
		blake3_node_cv(&node, 0, cv);
		chunk++;
		pos += BLAKE3_CHUNK_LEN;
		for (uint64_t t = chunk; (t & 1) == 0; t >>= 1) {
			blake3_node parent;
			depth--;
			blake3_parent_node(kw, stack[depth], cv, base_flags, &parent);
			blake3_node_cv(&parent, 0, cv);
		}
		for (int i = 0; i < 8; i++) stack[depth][i] = cv[i];
		depth++;
	}

	blake3_node node;
	blake3_chunk_node(kw, in + pos, len - pos, chunk, base_flags, &node);
	while (depth > 0) {
		uint32_t right[8];
		blake3_node_cv(&node, 0, right);
		depth--;
		blake3_parent_node(kw, stack[depth], right, base_flags, &node);
	}
	uint32_t root[8];
	blake3_node_cv(&node, BLAKE3_ROOT, root);
	blake3_cv_to_bytes(root, out);
}

/* --- decred: a 180-byte header, so 64 + 64 + 52 and the nonce at byte 140 --- */
#define DECRED_HDR_LEN		180
#define DECRED_NONCE_BYTE	140
#define DECRED_PREFIX_BLOCKS	2			/* bytes 0..127 are nonce-free */
#define DECRED_TAIL_LEN		(DECRED_HDR_LEN - DECRED_PREFIX_BLOCKS * BLAKE3_BLOCK_LEN)
/* byte 140 is 12 bytes into block 3, and BLAKE3 words are little-endian, so the
 * nonce is word 3 verbatim - no byte shuffling. */
#define DECRED_NONCE_WORD	((DECRED_NONCE_BYTE - DECRED_PREFIX_BLOCKS * BLAKE3_BLOCK_LEN) / 4)

/* Host, once per job: midstate over bytes 0..127, plus the tail words with the
 * nonce slot left at whatever the header held. */
static B3_HD void decred_blake3_prepare(const uint8_t *header, uint32_t cv[8], uint32_t m[16])
{
	blake3_prefix(header, DECRED_PREFIX_BLOCKS, cv);

	uint8_t tail[BLAKE3_BLOCK_LEN];
	for (int i = 0; i < BLAKE3_BLOCK_LEN; i++)
		tail[i] = (i < DECRED_TAIL_LEN)
			? header[DECRED_PREFIX_BLOCKS * BLAKE3_BLOCK_LEN + i] : 0;
	blake3_words_from_le(tail, m);
}

/* Device, once per nonce. */
static B3_HD void decred_blake3_nonce(const uint32_t cv[8], const uint32_t m[16],
	uint32_t nonce, uint8_t *out)
{
	uint32_t mm[16];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		mm[i] = m[i];
	mm[DECRED_NONCE_WORD] = nonce;

	blake3_final(cv, mm, DECRED_TAIL_LEN, false, out);
}

#endif // CUDA_BLAKE3_DEVICE_CUH
