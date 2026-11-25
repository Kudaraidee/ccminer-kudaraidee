/*
 * Unity build of the vendored libsecp256k1 for the curvehash PoW.
 *
 * curvehash's proof-of-work runs 8 rounds of secp256k1 fixed-base scalar
 * multiplication (secp256k1_ec_pubkey_create) interleaved with SHA-256; the
 * host CPU path (and, later, the GPU candidate re-verify) needs a bit-exact
 * secp256k1_ec_pubkey_create oracle. We vendor the same libsecp256k1 the
 * upstream reference miner used and compile it as a single translation unit.
 *
 * Config (portable, MSVC-safe — no __int128, no build-time gen_context):
 *   FIELD_10X26 + SCALAR_8X32     -> 32-bit limbs, pure C
 *   NUM_NONE                      -> no GMP dependency
 *   FIELD/SCALAR_INV_BUILTIN      -> self-contained modular inverse
 *   (USE_ECMULT_STATIC_PRECOMPUTATION intentionally left undefined ->
 *    the ecmult_gen table is built at runtime in secp256k1_context_create,
 *    so no precomputed table / codegen step is required.)
 */
#define USE_NUM_NONE 1
#define USE_FIELD_INV_BUILTIN 1
#define USE_SCALAR_INV_BUILTIN 1
#define USE_FIELD_10X26 1
#define USE_SCALAR_8X32 1

// The config above is self-contained; drop any global HAVE_CONFIG_H (the Linux
// autotools build defines it project-wide) so the vendored util.h does not try
// to pull a nonexistent libsecp256k1-config.h. MSVC never defines it.
#undef HAVE_CONFIG_H

#include "secp256k1/src/secp256k1.c"

#include <stdlib.h>
#include <string.h>

/*
 * Fixed-base window table: out[(j * entries + i) * 64] = i * 2^(wbits*j) * G as
 * X[32] || Y[32] big-endian, i == 0 left zero (the device reads that as infinity
 * and skips it).
 *
 * Entry i is entry i-1 plus the window base, so the table costs one point
 * addition per entry and one batched inversion per window rather than a scalar
 * multiplication per entry. It lives in this translation unit because the public
 * API cannot add two points without inverting each time.
 */
void curvehash_build_window_table(unsigned char *out, int wbits)
{
	const size_t entries = (size_t)1 << wbits;
	const int windows = 256 / wbits;
	secp256k1_gej *gj = (secp256k1_gej *)malloc(sizeof(secp256k1_gej) * entries);
	secp256k1_ge *ga = (secp256k1_ge *)malloc(sizeof(secp256k1_ge) * entries);
	secp256k1_gej base, acc;
	int j;
	size_t i;

	if (!gj || !ga) { free(gj); free(ga); return; }
	memset(out, 0, (size_t)windows * entries * 64);
	secp256k1_gej_set_ge(&base, &secp256k1_ge_const_g);

	for (j = 0; j < windows; j++) {
		/* gj[i] = i * base, i = 1 .. entries-1 (slot 0 stays infinity) */
		acc = base;
		for (i = 1; i < entries; i++) {
			gj[i] = acc;
			secp256k1_gej_add_var(&acc, &acc, &base, NULL);
		}
		/* one inversion for the whole window instead of one per entry */
		secp256k1_ge_set_all_gej_var(ga + 1, gj + 1, entries - 1, NULL);

		for (i = 1; i < entries; i++) {
			unsigned char *p = out + ((size_t)j * entries + i) * 64;
			secp256k1_fe_normalize_var(&ga[i].x);
			secp256k1_fe_normalize_var(&ga[i].y);
			secp256k1_fe_get_b32(p, &ga[i].x);
			secp256k1_fe_get_b32(p + 32, &ga[i].y);
		}
		/* next window's base = 2^wbits * this one */
		for (i = 0; i < (size_t)wbits; i++)
			secp256k1_gej_double_var(&base, &base, NULL);
	}
	free(gj);
	free(ga);
}

/* Load one 64-byte X||Y table entry as an affine point. */
static int curvehash_entry_ge(secp256k1_ge *r, const unsigned char *p)
{
	secp256k1_fe x, y;
	if (!secp256k1_fe_set_b32(&x, p) || !secp256k1_fe_set_b32(&y, p + 32))
		return 0;
	secp256k1_ge_set_xy(r, &x, &y);
	return 1;
}

/*
 * Check every entry of the 16-bit table against the 8-bit one:
 *   T16[j][i] == T8[2j][i & 0xff] + T8[2j+1][i >> 8]
 *
 * The two are built by different methods -- a scalar multiplication per entry
 * against repeated addition -- so agreement is not self-fulfilling, and it
 * extends the 8-bit table's checksum to all 2^20 shipped entries. Returns 1 on
 * match.
 */
int curvehash_check_w16_vs_w8(const unsigned char *w16, const unsigned char *w8)
{
	const size_t N16 = 65536, N8 = 256;
	secp256k1_gej *gj = (secp256k1_gej *)malloc(sizeof(secp256k1_gej) * N16);
	secp256k1_ge  *ga = (secp256k1_ge  *)malloc(sizeof(secp256k1_ge)  * N16);
	int ok = (gj && ga);
	int j;
	size_t i;

	for (j = 0; ok && j < 16; j++) {
		for (i = 1; i < N16; i++) {
			const size_t lo = i & 0xff, hi = i >> 8;
			secp256k1_ge a, b;
			if (lo && !curvehash_entry_ge(&a, w8 + ((size_t)(2*j)   * N8 + lo) * 64)) { ok = 0; break; }
			if (hi && !curvehash_entry_ge(&b, w8 + ((size_t)(2*j+1) * N8 + hi) * 64)) { ok = 0; break; }
			if (!lo)      secp256k1_gej_set_ge(&gj[i], &b);
			else if (!hi) secp256k1_gej_set_ge(&gj[i], &a);
			else {
				secp256k1_gej_set_ge(&gj[i], &a);
				secp256k1_gej_add_ge_var(&gj[i], &gj[i], &b, NULL);
			}
		}
		if (!ok) break;
		secp256k1_ge_set_all_gej_var(ga + 1, gj + 1, N16 - 1, NULL);
		for (i = 1; i < N16; i++) {
			unsigned char xy[64];
			secp256k1_fe_normalize_var(&ga[i].x);
			secp256k1_fe_normalize_var(&ga[i].y);
			secp256k1_fe_get_b32(xy, &ga[i].x);
			secp256k1_fe_get_b32(xy + 32, &ga[i].y);
			if (memcmp(xy, w16 + ((size_t)j * N16 + i) * 64, 64)) { ok = 0; break; }
		}
	}
	free(gj);
	free(ga);
	return ok;
}
