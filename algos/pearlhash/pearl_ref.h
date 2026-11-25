// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * pearlhash host reference (dense, certificate V3), used to re-verify candidates
 * and by the self-test. Ported from pearl-research-labs/pearl @ 0798903:
 *   zk-pow/src/ffi/mine.rs            try_mine_one (tile loop, seed chain)
 *   zk-pow/src/circuit/pearl_noise.rs noise PRNG
 *   zk-pow/src/api/seed.rs            V3 root salting
 *   zk-pow/src/api/proof_utils.rs     PeriodicPattern, serializers, nbits, jackpot hash
 *   zk-pow/src/api/sanity_checks.rs   difficulty bound
 */
#ifndef PEARL_REF_H
#define PEARL_REF_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "cuda/blake3_device.cuh"

#define PEARL_JACKPOT_WORDS	16
#define PEARL_LROT		13
#define PEARL_PENALTY_RANK	128
#define PEARL_PATTERN_DIMS	3
#define PEARL_HEADER_LEN	76
#define PEARL_CONFIG_LEN	52
#define PEARL_MAX_PATTERN	256

/* --- PeriodicPattern: 3 x (stride, length), outermost last ------------------- */
typedef struct {
	uint32_t stride[PEARL_PATTERN_DIMS];
	uint32_t length[PEARL_PATTERN_DIMS];
} pearl_pattern;

/* PeriodicPattern::from_list. Returns 0 if the list is not a valid periodic pattern. */
static int pearl_pattern_from_list(const uint32_t *list, int n, pearl_pattern *p)
{
	if (n < 1 || n > PEARL_MAX_PATTERN || list[0] != 0) return 0;
	for (int i = 1; i < n; i++) if (list[i] <= list[i - 1]) return 0;
	uint32_t q[PEARL_MAX_PATTERN];
	memcpy(q, list, n * sizeof(uint32_t));
	uint32_t s_shape[8], l_shape[8];
	int dims = 0, len = n;
	while (len > 1) {
		int found = 0;
		for (int period = 1; period < len; period++) {
			if (len % period) continue;
			uint32_t s = q[period];
			int ok = 1;
			for (int i = 0; i < len - period && ok; i++) ok = (q[i] + s == q[i + period]);
			if (ok) {
				if (dims == 8) return 0;
				s_shape[dims] = s; l_shape[dims] = (uint32_t)(len / period); dims++;
				len = period; found = 1;
				break;
			}
		}
		if (!found) return 0;
	}
	if (dims > PEARL_PATTERN_DIMS) return 0;
	/* reverse, then pad with (period, 1) */
	for (int i = 0; i < dims; i++) {
		p->stride[i] = s_shape[dims - 1 - i];
		p->length[i] = l_shape[dims - 1 - i];
	}
	uint32_t period = dims ? p->stride[dims - 1] * p->length[dims - 1] : 1;
	for (int i = dims; i < PEARL_PATTERN_DIMS; i++) { p->stride[i] = period; p->length[i] = 1; }
	return 1;
}

static uint32_t pearl_pattern_period(const pearl_pattern *p)
{
	return p->stride[PEARL_PATTERN_DIMS - 1] * p->length[PEARL_PATTERN_DIMS - 1];
}

static uint32_t pearl_pattern_size(const pearl_pattern *p)
{
	uint32_t s = 1;
	for (int i = 0; i < PEARL_PATTERN_DIMS; i++) s *= p->length[i];
	return s;
}

/* PeriodicPattern::to_list, ascending-dimension order (matches the reference) */
static int pearl_pattern_list(const pearl_pattern *p, uint32_t *out)
{
	int n = 1;
	out[0] = 0;
	for (int d = 0; d < PEARL_PATTERN_DIMS; d++) {
		uint32_t tmp[PEARL_MAX_PATTERN];
		int m = 0;
		for (uint32_t i = 0; i < p->length[d]; i++)
			for (int j = 0; j < n; j++) tmp[m++] = out[j] + i * p->stride[d];
		memcpy(out, tmp, m * sizeof(uint32_t));
		n = m;
	}
	return n;
}

static int pearl_offset_is_valid(const pearl_pattern *p, uint32_t offset)
{
	for (int d = PEARL_PATTERN_DIMS - 1; d >= 0; d--) {
		offset %= p->stride[d] * p->length[d];
		if (offset >= p->stride[d]) return 0;
	}
	return 1;
}

static void pearl_pattern_bytes(const pearl_pattern *p, uint8_t out[6])
{
	uint32_t min_stride = 1;
	for (int d = 0; d < PEARL_PATTERN_DIMS; d++) {
		out[2 * d] = (uint8_t)(p->stride[d] / min_stride - 1);
		out[2 * d + 1] = (uint8_t)(p->length[d] - 1);
		min_stride = p->stride[d] * p->length[d];
	}
}

/* hash-tile patterns of the two GPU kernels: one thread's register block each */
static const uint32_t PEARL_DP4A_PATTERN[8] = { 0, 1, 2, 3, 64, 65, 66, 67 };	/* rows == cols */
static const uint32_t PEARL_IMMA_ROWS[8] = { 0, 8, 16, 24, 32, 40, 48, 56 };
static const uint32_t PEARL_IMMA_COLS[8] = { 0, 1, 8, 9, 16, 17, 24, 25 };
/* wide IMMA kernel (64 x 64 warp tile): rows PEARL_IMMA_ROWS x these 16 cols */
static const uint32_t PEARL_IMMA_W_COLS[16] = { 0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57 };

/* --- MiningConfiguration (dense: e = top_k = 0) ------------------------------ */
typedef struct {
	uint32_t k;		/* common_dim */
	uint32_t r;		/* noise rank */
	pearl_pattern rows, cols;
} pearl_config;

static void pearl_config_bytes(const pearl_config *c, uint8_t out[PEARL_CONFIG_LEN])
{
	memset(out, 0, PEARL_CONFIG_LEN);
	out[0] = (uint8_t)c->k; out[1] = (uint8_t)(c->k >> 8);
	out[2] = (uint8_t)(c->k >> 16); out[3] = (uint8_t)(c->k >> 24);
	out[4] = (uint8_t)c->r; out[5] = (uint8_t)(c->r >> 8);
	/* mma_type 0 = Int7xInt7ToInt32 at bytes 6..7 */
	pearl_pattern_bytes(&c->rows, out + 8);
	pearl_pattern_bytes(&c->cols, out + 14);
	/* 32-byte MoE trailer stays zero */
}

static uint32_t pearl_dot_len(const pearl_config *c) { return c->k - c->k % c->r; }

/* --- seed chain ---------------------------------------------------------------- */
static const uint8_t PEARL_SEED_SALT_A[32] = {
	0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6, 0x92, 0xfc, 0xf0, 0x76, 0xf8,
	0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23, 0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b };
static const uint8_t PEARL_SEED_SALT_B[32] = {
	0x11, 0x30, 0x06, 0x32, 0xec, 0x63, 0x01, 0xca, 0x2b, 0xe2, 0xaf, 0x71, 0x8b, 0x3f, 0x4d, 0x4f,
	0x1a, 0xe9, 0xc6, 0x39, 0x88, 0xe8, 0xcc, 0x04, 0x48, 0x44, 0x30, 0x1d, 0x71, 0xb8, 0x9a, 0xa9 };

static void pearl_job_key(const uint8_t header[PEARL_HEADER_LEN], const pearl_config *c, uint8_t out[32])
{
	uint8_t msg[PEARL_HEADER_LEN + PEARL_CONFIG_LEN];
	memcpy(msg, header, PEARL_HEADER_LEN);
	pearl_config_bytes(c, msg + PEARL_HEADER_LEN);
	blake3_hash(msg, sizeof(msg), NULL, out);
}

/* keyed BLAKE3 over data zero-padded to a 1024-byte boundary (= the Merkle root) */
static void pearl_matrix_root(const uint8_t *data, size_t len, const uint8_t key[32], uint8_t out[32])
{
	size_t padded = (len + BLAKE3_CHUNK_LEN - 1) / BLAKE3_CHUNK_LEN * BLAKE3_CHUNK_LEN;
	if (padded == len) { blake3_hash(data, len, key, out); return; }
	uint8_t *buf = (uint8_t *)calloc(padded, 1);
	memcpy(buf, data, len);
	blake3_hash(buf, padded, key, out);
	free(buf);
}

/* V3: B3(root | le32(dim) | 0^28, key = salt) */
static void pearl_bind_root(const uint8_t root[32], uint32_t dim, const uint8_t salt[32], uint8_t out[32])
{
	uint8_t msg[64] = { 0 };
	memcpy(msg, root, 32);
	msg[32] = (uint8_t)dim; msg[33] = (uint8_t)(dim >> 8);
	msg[34] = (uint8_t)(dim >> 16); msg[35] = (uint8_t)(dim >> 24);
	blake3_hash(msg, 64, salt, out);
}

static void pearl_seeds(const uint8_t job_key[32], const uint8_t hash_a[32], const uint8_t hash_b[32],
	uint32_t m, uint32_t n, uint8_t b_seed[32], uint8_t a_seed[32])
{
	uint8_t ba[32], bb[32], msg[64];
	pearl_bind_root(hash_a, m, PEARL_SEED_SALT_A, ba);
	pearl_bind_root(hash_b, n, PEARL_SEED_SALT_B, bb);
	memcpy(msg, job_key, 32); memcpy(msg + 32, bb, 32);
	blake3_hash(msg, 64, NULL, b_seed);
	memcpy(msg, b_seed, 32); memcpy(msg + 32, ba, 32);
	blake3_hash(msg, 64, NULL, a_seed);
}

/* --- noise PRNG (pearl_noise.rs) --------------------------------------------- */
/* B3_HD: the GPU kernels use these same bodies */

/* get_random_hash(index, label, key=noise_seed, slot): message = le32(1+index) in
 * word `slot`, label "A_tensor"/"B_tensor" zero-padded at bytes 32..63. */
static B3_HD void pearl_random_hash(uint32_t index, char label_ab, const uint32_t key[8], int slot, uint32_t out[8])
{
	uint32_t m[16] = { 0 };
	m[slot] = index + 1;
	/* "A_tensor" little-endian words: 'A','_','t','e' | 'n','s','o','r' */
	m[8] = (uint32_t)(uint8_t)label_ab | ((uint32_t)'_' << 8) | ((uint32_t)'t' << 16) | ((uint32_t)'e' << 24);
	m[9] = (uint32_t)'n' | ((uint32_t)'s' << 8) | ((uint32_t)'o' << 16) | ((uint32_t)'r' << 24);
	blake3_keyed_block(key, m, out);
}

static B3_HD void pearl_key_words(const uint8_t b[32], uint32_t w[8])
{
	for (int i = 0; i < 8; i++)
		w[i] = (uint32_t)b[i*4] | ((uint32_t)b[i*4+1] << 8) | ((uint32_t)b[i*4+2] << 16) | ((uint32_t)b[i*4+3] << 24);
}

/* uniform noise byte: (byte & 63) - 32, in [-32, 31] */
static B3_HDI int8_t pearl_uniform_byte(uint32_t word, int byte)
{
	return (int8_t)((int)((word >> (8 * byte)) & 63) - 32);
}

/* one sparse row from one u32: +1 at first, -1 at second */
static B3_HDI void pearl_perm_pair(uint32_t u, uint32_t r, uint32_t *first, uint32_t *second)
{
	uint32_t f = u & (r - 1);
	*first = f;
	*second = f ^ (1 + (uint32_t)(((uint64_t)(r - 1) * u) >> 32));
}

/* fold the XOR of a tile's running sums after r-chunk c (0-based) into the jackpot */
static B3_HDI void pearl_fold(uint32_t jp[16], uint32_t c, uint32_t x)
{
	uint32_t t = c % 16;
	jp[t] = ((jp[t] << 13) | (jp[t] >> 19)) ^ x;
}

/* ticket digest words (LE U256, word 7 most significant) <= bound */
static B3_HDI int pearl_meets_words(const uint32_t h[8], const uint32_t bound[8])
{
	for (int i = 7; i >= 0; i--)
		if (h[i] != bound[i]) return h[i] < bound[i];
	return 1;
}

/* uniform rows x r matrix in [-32, 31]; flat index row*r + c, 32 values per hash */
static void pearl_noise_uniform(const uint8_t seed[32], char label, uint32_t rows, uint32_t r, int8_t *out)
{
	uint32_t key[8], h[8];
	pearl_key_words(seed, key);
	uint32_t total = rows * r;
	for (uint32_t blk = 0; blk < total / 32; blk++) {
		pearl_random_hash(blk, label, key, 0, h);
		for (int j = 0; j < 32; j++)
			out[blk * 32 + j] = pearl_uniform_byte(h[j >> 2], j & 3);
	}
}

/* sparse +1/-1 rows: k pairs (p0, p1), 8 per hash */
static void pearl_noise_perm(const uint8_t seed[32], char label, uint32_t k, uint32_t r, uint32_t *p0, uint32_t *p1)
{
	uint32_t key[8], h[8];
	pearl_key_words(seed, key);
	for (uint32_t blk = 0; blk * 8 < k; blk++) {
		pearl_random_hash(blk, label, key, 1, h);
		for (int j = 0; j < 8 && blk * 8 + j < k; j++)
			pearl_perm_pair(h[j], r, &p0[blk * 8 + j], &p1[blk * 8 + j]);
	}
}

/* noise for all `rows` rows (A: m x k from a_seed/'A'; B^T: n x k from b_seed/'B') */
static void pearl_noise_full(const uint8_t seed[32], char label, uint32_t rows, uint32_t k, uint32_t r, int8_t *out)
{
	int8_t *u = (int8_t *)malloc((size_t)rows * r);
	uint32_t *p0 = (uint32_t *)malloc(k * sizeof(uint32_t)), *p1 = (uint32_t *)malloc(k * sizeof(uint32_t));
	pearl_noise_uniform(seed, label, rows, r, u);
	pearl_noise_perm(seed, label, k, r, p0, p1);
	for (uint32_t i = 0; i < rows; i++)
		for (uint32_t l = 0; l < k; l++)
			out[(size_t)i * k + l] = (int8_t)((int)u[(size_t)i * r + p0[l]] - (int)u[(size_t)i * r + p1[l]]);
	free(u); free(p0); free(p1);
}

/* noise for selected rows only: out[j][l], j < nrows (compute_noise_for_indices) */
static void pearl_noise_rows(const uint8_t seed[32], char label, const uint32_t *rows, int nrows,
	uint32_t k, uint32_t r, int8_t *out)
{
	uint32_t key[8], h[8];
	pearl_key_words(seed, key);
	uint32_t *p0 = (uint32_t *)malloc(k * sizeof(uint32_t)), *p1 = (uint32_t *)malloc(k * sizeof(uint32_t));
	int8_t *u = (int8_t *)malloc(r);
	pearl_noise_perm(seed, label, k, r, p0, p1);
	for (int j = 0; j < nrows; j++) {
		for (uint32_t blk = 0; blk < r / 32; blk++) {
			pearl_random_hash(rows[j] * (r / 32) + blk, label, key, 0, h);
			for (int b = 0; b < 32; b++) u[blk * 32 + b] = pearl_uniform_byte(h[b >> 2], b & 3);
		}
		for (uint32_t l = 0; l < k; l++)
			out[(size_t)j * k + l] = (int8_t)((int)u[p0[l]] - (int)u[p1[l]]);
	}
	free(p0); free(p1); free(u);
}

/* --- ticket ------------------------------------------------------------------- */
static void pearl_jackpot_hash(const uint32_t jp[PEARL_JACKPOT_WORDS], const uint8_t a_seed[32], uint8_t out[32])
{
	uint32_t key[8], h[8];
	pearl_key_words(a_seed, key);
	blake3_keyed_block(key, jp, h);
	blake3_cv_to_bytes(h, out);
}

/* --- bound: target(nbits) * h*w*(dot/r)*128, saturating; 8 LE words ---------- */
static void pearl_bound(uint32_t nbits, const pearl_config *c, uint32_t bound[8])
{
	uint32_t t[8] = { 0 };
	uint32_t exponent = nbits >> 24, mantissa = nbits & 0x00ffffff;
	if (mantissa != 0 && exponent != 0 && !(mantissa & 0x00800000)) {
		if (exponent <= 3) {
			t[0] = mantissa >> (8 * (3 - exponent));
		} else {
			uint32_t sh = 8 * (exponent - 3);	/* bits shifted past 256 are dropped (U256 <<=) */
			uint64_t v = mantissa;
			for (int b = 0; b < 64; b++)
				if ((v >> b) & 1) {
					uint32_t pos = sh + b;
					if (pos < 256) t[pos >> 5] |= 1u << (pos & 31);
				}
		}
	}
	uint64_t factor = (uint64_t)pearl_pattern_size(&c->rows) * pearl_pattern_size(&c->cols)
		* (pearl_dot_len(c) / c->r) * PEARL_PENALTY_RANK;
	uint64_t carry = 0;
	for (int i = 0; i < 8; i++) {
		uint64_t p = (uint64_t)t[i] * factor + carry;
		bound[i] = (uint32_t)p;
		carry = p >> 32;
	}
	if (carry) for (int i = 0; i < 8; i++) bound[i] = 0xffffffffu;	/* saturate */
}

/* LE U256 hash <= bound */
static int pearl_meets(const uint8_t hash[32], const uint32_t bound[8])
{
	uint32_t w[8];
	pearl_key_words(hash, w);
	return pearl_meets_words(w, bound);
}

/* One tile's jackpot from noised operands (an: m x k, bn: n x k, int8 row-major). */
static void pearl_tile_jackpot(const int8_t *an, const int8_t *bn, uint32_t k, uint32_t r,
	const uint32_t *rows, int h, const uint32_t *cols, int w, uint32_t jp[PEARL_JACKPOT_WORDS])
{
	int32_t acc[PEARL_MAX_PATTERN];
	memset(acc, 0, sizeof(acc));
	memset(jp, 0, PEARL_JACKPOT_WORDS * sizeof(uint32_t));
	uint32_t dot = k - k % r;
	for (uint32_t ll = r; ll <= dot; ll += r) {
		uint32_t x = 0;
		for (int u = 0; u < h; u++)
			for (int v = 0; v < w; v++) {
				const int8_t *pa = an + (size_t)rows[u] * k, *pb = bn + (size_t)cols[v] * k;
				int32_t s = acc[u * w + v];
				for (uint32_t l = ll - r; l < ll; l++) s += (int32_t)pa[l] * (int32_t)pb[l];
				acc[u * w + v] = s;
				x ^= (uint32_t)s;
			}
		pearl_fold(jp, ll / r - 1, x);
	}
}

#endif /* PEARL_REF_H */
