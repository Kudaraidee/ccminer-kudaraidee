// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * pearlhash shares, host only: BLAKE3 Merkle multi-leaf proofs and the bincode
 * PlainProof (base64) sent with mining.submit. Ported from pearl @ 0798903:
 * pearl-blake3/src/merkle.rs and zk-pow/src/ffi/plain_proof.rs.
 * bincode 1.x fixint LE: usize = u64, Vec = u64 length + items, [u8; 32] = raw,
 * Option::None = one 0x00 byte. The tree root equals blake3_hash(padded, key).
 */
#ifndef PEARL_PROOF_H
#define PEARL_PROOF_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "algos/pearlhash/pearl_ref.h"

typedef struct {
	uint32_t key[8];
	size_t leaves;
	int nlayers;		/* layer[0] = chunk CVs ... layer[nlayers-1] has 1 or 2 nodes */
	uint32_t *layer[64];	/* 8 words per node */
	size_t len[64];
	uint8_t root[32];
} pearl_merkle;

/* CV of one 1024-byte chunk, NOT root (the tree has >= 2 leaves). */
static void pearl_chunk_cv(const uint32_t key[8], const uint8_t *chunk, uint64_t index, uint32_t cv[8])
{
	blake3_node node;
	blake3_chunk_node(key, chunk, BLAKE3_CHUNK_LEN, index, BLAKE3_KEYED_HASH, &node);
	blake3_node_cv(&node, 0, cv);
}

static void pearl_parent_cv(const uint32_t key[8], const uint32_t *l, const uint32_t *r, uint32_t root_flag, uint32_t cv[8])
{
	blake3_node node;
	blake3_parent_node(key, l, r, BLAKE3_KEYED_HASH, &node);
	blake3_node_cv(&node, root_flag, cv);
}

/* upper layers + root from layer[0] */
static void pearl_merkle_finish(pearl_merkle *t)
{
	int L = 0;
	while (t->len[L] > 2) {
		size_t n = t->len[L], nn = (n + 1) / 2;
		if (!t->layer[L + 1]) t->layer[L + 1] = (uint32_t *)malloc(nn * 32);
		for (size_t i = 0; i < n / 2; i++)
			pearl_parent_cv(t->key, t->layer[L] + 16 * i, t->layer[L] + 16 * i + 8, 0, t->layer[L + 1] + 8 * i);
		if (n & 1) memcpy(t->layer[L + 1] + 8 * (nn - 1), t->layer[L] + 8 * (n - 1), 32);
		t->len[++L] = nn;
	}
	t->nlayers = L + 1;
	uint32_t r[8];
	pearl_parent_cv(t->key, t->layer[L], t->layer[L] + 8, BLAKE3_ROOT, r);
	blake3_cv_to_bytes(r, t->root);
}

/* Tree over data (len bytes, conceptually zero-padded to a chunk multiple). Needs >= 2 leaves. */
static int pearl_merkle_build(pearl_merkle *t, const uint8_t *data, size_t len, const uint8_t key[32])
{
	memset(t, 0, sizeof(*t));
	pearl_key_words(key, t->key);
	t->leaves = (len + BLAKE3_CHUNK_LEN - 1) / BLAKE3_CHUNK_LEN;
	if (t->leaves < 2) return 0;
	t->layer[0] = (uint32_t *)malloc(t->leaves * 32);
	t->len[0] = t->leaves;
	uint8_t tail[BLAKE3_CHUNK_LEN];
	for (size_t i = 0; i < t->leaves; i++) {
		const uint8_t *c = data + i * BLAKE3_CHUNK_LEN;
		if ((i + 1) * BLAKE3_CHUNK_LEN > len) {
			memset(tail, 0, sizeof(tail));
			memcpy(tail, c, len - i * BLAKE3_CHUNK_LEN);
			c = tail;
		}
		pearl_chunk_cv(t->key, c, i, t->layer[0] + 8 * i);
	}
	pearl_merkle_finish(t);
	return 1;
}

/* Replace leaf i's chunk (full 1024 bytes) and recompute its path and the root. */
static void pearl_merkle_update_leaf(pearl_merkle *t, size_t i, const uint8_t *chunk)
{
	pearl_chunk_cv(t->key, chunk, i, t->layer[0] + 8 * i);
	for (int L = 0; L + 1 < t->nlayers; L++) {
		size_t p = i / 2, n = t->len[L];
		if (2 * p + 1 < n)
			pearl_parent_cv(t->key, t->layer[L] + 16 * p, t->layer[L] + 16 * p + 8, 0, t->layer[L + 1] + 8 * p);
		else
			memcpy(t->layer[L + 1] + 8 * p, t->layer[L] + 8 * (2 * p), 32);
		i = p;
	}
	uint32_t r[8];
	pearl_parent_cv(t->key, t->layer[t->nlayers - 1], t->layer[t->nlayers - 1] + 8, BLAKE3_ROOT, r);
	blake3_cv_to_bytes(r, t->root);
}

static void pearl_merkle_free(pearl_merkle *t)
{
	for (int i = 0; i < 64; i++) free(t->layer[i]);
	memset(t, 0, sizeof(*t));
}

/* --- growable byte buffer + bincode writers --------------------------------- */
typedef struct { uint8_t *p; size_t n, cap; } pearl_buf;

static void pb_put(pearl_buf *b, const void *src, size_t n)
{
	if (b->n + n > b->cap) {
		b->cap = (b->n + n) * 2 + 4096;
		b->p = (uint8_t *)realloc(b->p, b->cap);
	}
	memcpy(b->p + b->n, src, n);
	b->n += n;
}

static void pb_u64(pearl_buf *b, uint64_t v)
{
	uint8_t x[8];
	for (int i = 0; i < 8; i++) x[i] = (uint8_t)(v >> (8 * i));
	pb_put(b, x, 8);
}

static int pearl_cmp_size(const void *a, const void *b)
{
	size_t x = *(const size_t *)a, y = *(const size_t *)b;
	return x < y ? -1 : x > y;
}

/* MatrixMerkleProof for `rows` (sorted ascending) of a rows x cols int8 matrix `data`
 * whose tree is `t`. Appends bincode to b. */
/* cchunk != NULL: every proven leaf holds cchunk (a constant matrix; data is not read) */
static void pearl_matrix_proof_c(pearl_buf *b, const pearl_merkle *t, const uint8_t *data, size_t data_len,
	const uint8_t *cchunk, const uint32_t *rows, int nrows, size_t cols)
{
	/* compute_leaf_indices_from_rows: unique sorted chunk indices */
	size_t *idx = (size_t *)malloc(sizeof(size_t) * (nrows * (cols / BLAKE3_CHUNK_LEN + 2)));
	int ni = 0;
	for (int r = 0; r < nrows; r++) {
		size_t first = (size_t)rows[r] * cols / BLAKE3_CHUNK_LEN;
		size_t last = ((size_t)(rows[r] + 1) * cols - 1) / BLAKE3_CHUNK_LEN;
		for (size_t i = first; i <= last; i++) idx[ni++] = i;
	}
	qsort(idx, ni, sizeof(size_t), pearl_cmp_size);
	int u = 0;
	for (int i = 0; i < ni; i++) if (!u || idx[u - 1] != idx[i]) idx[u++] = idx[i];

	/* leaf_data: Vec<Vec<u8>> */
	pb_u64(b, u);
	uint8_t chunk[BLAKE3_CHUNK_LEN];
	for (int i = 0; i < u; i++) {
		size_t s = idx[i] * BLAKE3_CHUNK_LEN, e = s + BLAKE3_CHUNK_LEN;
		if (e > data_len) e = data_len;
		memset(chunk, 0, sizeof(chunk));
		if (cchunk) memcpy(chunk, cchunk, BLAKE3_CHUNK_LEN);
		else if (e > s) memcpy(chunk, data + s, e - s);
		pb_u64(b, BLAKE3_CHUNK_LEN);
		pb_put(b, chunk, BLAKE3_CHUNK_LEN);
	}
	/* leaf_indices, total_leaves, root */
	pb_u64(b, u);
	for (int i = 0; i < u; i++) pb_u64(b, idx[i]);
	pb_u64(b, t->leaves);
	pb_put(b, t->root, 32);

	/* siblings: level by level, ascending, exactly as get_multileaf_proof */
	size_t *cur = (size_t *)malloc(sizeof(size_t) * u), *nxt = (size_t *)malloc(sizeof(size_t) * u);
	memcpy(cur, idx, sizeof(size_t) * u);
	int nc = u;
	size_t sib_cap = 64 * (size_t)u + 64, ns = 0;
	uint32_t *sib = (uint32_t *)malloc(sib_cap * 32);
	size_t level_len = t->leaves;
	for (int L = 0; level_len > 1 && nc > 0; L++) {
		const uint32_t *nodes = t->layer[L];
		for (int j = 0; j < nc; j++) {
			size_t i = cur[j];
			int has_prev = j > 0 && cur[j - 1] == i - 1;
			int has_next = j + 1 < nc && cur[j + 1] == i + 1;
			if (i & 1) {
				if (!has_prev) memcpy(sib + 8 * ns++, nodes + 8 * (i - 1), 32);
			} else if (!has_next && i + 1 < level_len) {
				memcpy(sib + 8 * ns++, nodes + 8 * (i + 1), 32);
			}
		}
		int nn = 0;
		for (int j = 0; j < nc; j++) if (!nn || nxt[nn - 1] != cur[j] / 2) nxt[nn++] = cur[j] / 2;
		size_t *tmp = cur; cur = nxt; nxt = tmp; nc = nn;
		level_len = (level_len + 1) / 2;
	}
	pb_u64(b, ns);
	for (size_t i = 0; i < ns; i++) {
		uint8_t x[32];
		blake3_cv_to_bytes(sib + 8 * i, x);
		pb_put(b, x, 32);
	}
	/* row_indices */
	pb_u64(b, nrows);
	for (int r = 0; r < nrows; r++) pb_u64(b, rows[r]);
	free(idx); free(cur); free(nxt); free(sib);
}

static void pearl_matrix_proof(pearl_buf *b, const pearl_merkle *t, const uint8_t *data, size_t data_len,
	const uint32_t *rows, int nrows, size_t cols)
{
	pearl_matrix_proof_c(b, t, data, data_len, NULL, rows, nrows, cols);
}

/* Dense PlainProof. A: m x k row-major, Bt: n x k row-major; trees over each, keyed by job_key. */
static void pearl_plain_proof(pearl_buf *b, uint32_t m, uint32_t n, uint32_t k, uint32_t r,
	const pearl_merkle *ta, const uint8_t *A, const uint32_t *arows, int h,
	const pearl_merkle *tb, const uint8_t *Bt, const uint32_t *bcols, int w)
{
	pb_u64(b, m); pb_u64(b, n); pb_u64(b, k); pb_u64(b, r);
	pearl_matrix_proof(b, ta, A, (size_t)m * k, arows, h, k);
	pearl_matrix_proof(b, tb, Bt, (size_t)n * k, bcols, w, k);
	uint8_t none = 0;
	pb_put(b, &none, 1);
}

/* The same for constant matrices (fused geometry): every proven leaf of A holds achunk and every
 * proven leaf of B^T holds bchunk. Only valid when no proven row touches a non-constant leaf. */
static void pearl_plain_proof_const(pearl_buf *b, uint32_t m, uint32_t n, uint32_t k, uint32_t r,
	const pearl_merkle *ta, const uint8_t *achunk, const uint32_t *arows, int h,
	const pearl_merkle *tb, const uint8_t *bchunk, const uint32_t *bcols, int w)
{
	pb_u64(b, m); pb_u64(b, n); pb_u64(b, k); pb_u64(b, r);
	pearl_matrix_proof_c(b, ta, NULL, (size_t)m * k, achunk, arows, h, k);
	pearl_matrix_proof_c(b, tb, NULL, (size_t)n * k, bchunk, bcols, w, k);
	uint8_t none = 0;
	pb_put(b, &none, 1);
}

/* standard base64, NUL-terminated, caller frees */
static char *pearl_base64(const uint8_t *p, size_t n)
{
	static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	char *o = (char *)malloc((n + 2) / 3 * 4 + 1), *q = o;
	size_t i = 0;
	for (; i + 2 < n; i += 3) {
		uint32_t v = (uint32_t)p[i] << 16 | (uint32_t)p[i + 1] << 8 | p[i + 2];
		*q++ = T[v >> 18]; *q++ = T[(v >> 12) & 63]; *q++ = T[(v >> 6) & 63]; *q++ = T[v & 63];
	}
	if (i < n) {
		uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0);
		*q++ = T[v >> 18]; *q++ = T[(v >> 12) & 63];
		*q++ = i + 1 < n ? T[(v >> 6) & 63] : '=';
		*q++ = '=';
	}
	*q = 0;
	return o;
}

/* share bound: target (64 hex chars, big-endian U256, as mining.notify sends it) times the
 * penalized work factor h*w*(dot/r)*128 (penalized_target_bound). Returns 0 if unusable. */
static int pearl_share_bound(const char *hex64, const pearl_config *c, uint32_t bound[8])
{
	uint32_t t[8];
	if (strlen(hex64) != 64) return 0;
	for (int i = 0; i < 8; i++) {
		char w[9]; memcpy(w, hex64 + 8 * (7 - i), 8); w[8] = 0;
		char *end; t[i] = (uint32_t)strtoul(w, &end, 16);
		if (*end) return 0;
	}
	uint64_t factor = (uint64_t)pearl_pattern_size(&c->rows) * pearl_pattern_size(&c->cols)
		* (pearl_dot_len(c) / c->r) * PEARL_PENALTY_RANK;
	uint64_t carry = 0;
	for (int i = 0; i < 8; i++) {
		uint64_t p = (uint64_t)t[i] * factor + carry;
		bound[i] = (uint32_t)p;
		carry = p >> 32;
	}
	return carry == 0 && factor != 0;
}

#endif /* PEARL_PROOF_H */
