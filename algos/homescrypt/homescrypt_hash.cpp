/* homescrypt (Lumenite): HomeScrypt v1.2 CPU reference and self-test,
 * following homescrypt_v12_tuned in lumenite-core src/crypto/homescrypt.cpp. */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "homescrypt.h"
#include "homescrypt_kat.h"

extern "C" {
#include "sph/sph_sha2.h"
}

namespace {

const uint32_t WORDS = HOMESCRYPT_WORDS;
const uint32_t MASK = WORDS - 1;

const unsigned char TAG_SEED[20]  = { 'H','O','M','E','S','C','R','Y','P','T','-','V','1','.','2','-','S','E','E','D' };
const unsigned char TAG_FINAL[21] = { 'H','O','M','E','S','C','R','Y','P','T','-','V','1','.','2','-','F','I','N','A','L' };

inline uint64_t rotl(uint64_t x, unsigned r) { return (x << r) | (x >> (64 - r)); } /* r in 1..63 */

inline uint64_t av(uint64_t x)
{
	x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
	x ^= x >> 27; x *= 0x94d049bb133111ebull;
	x ^= x >> 31;
	return x;
}

inline uint64_t ld64(const unsigned char *p)
{
	uint64_t v = 0;
	for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
	return v;
}

inline void st64(unsigned char *p, uint64_t v)
{
	for (int i = 0; i < 8; i++) { p[i] = (unsigned char) v; v >>= 8; }
}

inline uint32_t fbits(float f) { uint32_t w; memcpy(&w, &f, 4); return w; }

/* any word -> a positive normal in [2^-9, 2^7): never NaN, inf or denormal */
inline float safe(uint32_t w)
{
	const uint32_t b = ((118u + ((w >> 23) & 15u)) << 23) | (w & 0x7FFFFFu);
	float f; memcpy(&f, &b, 4);
	return f;
}

/* ---- scrypt(N=1024, r=1, p=1) over the header, Litecoin's scrypt_1024_1_1_256 ---- */

void sha256(const void *a, size_t alen, const void *b, size_t blen, unsigned char out[32])
{
	sph_sha256_context c;
	sph_sha256_init(&c);
	sph_sha256(&c, a, alen);
	if (blen) sph_sha256(&c, b, blen);
	sph_sha256_close(&c, out);
}

/* HMAC-SHA256 with the 80-byte header as key (longer than a block, so hashed) */
void hmac(const unsigned char k0[32], const void *msg, size_t len, uint32_t be_index, unsigned char out[32])
{
	unsigned char pad[64], inner[32], idx[4] = {
		(unsigned char) (be_index >> 24), (unsigned char) (be_index >> 16),
		(unsigned char) (be_index >> 8), (unsigned char) be_index };
	sph_sha256_context c;

	for (int i = 0; i < 64; i++) pad[i] = (i < 32 ? k0[i] : 0) ^ 0x36;
	sph_sha256_init(&c);
	sph_sha256(&c, pad, 64);
	sph_sha256(&c, msg, len);
	sph_sha256(&c, idx, 4);
	sph_sha256_close(&c, inner);
	for (int i = 0; i < 64; i++) pad[i] = (i < 32 ? k0[i] : 0) ^ 0x5c;
	sha256(pad, 64, inner, 32, out);
}

inline uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

void salsa8(uint32_t B[16], const uint32_t Bx[16])
{
	uint32_t x[16];
	for (int i = 0; i < 16; i++) x[i] = (B[i] ^= Bx[i]);
	for (int r = 0; r < 8; r += 2) {
		x[ 4] ^= rotl32(x[ 0] + x[12],  7); x[ 8] ^= rotl32(x[ 4] + x[ 0],  9);
		x[12] ^= rotl32(x[ 8] + x[ 4], 13); x[ 0] ^= rotl32(x[12] + x[ 8], 18);
		x[ 9] ^= rotl32(x[ 5] + x[ 1],  7); x[13] ^= rotl32(x[ 9] + x[ 5],  9);
		x[ 1] ^= rotl32(x[13] + x[ 9], 13); x[ 5] ^= rotl32(x[ 1] + x[13], 18);
		x[14] ^= rotl32(x[10] + x[ 6],  7); x[ 2] ^= rotl32(x[14] + x[10],  9);
		x[ 6] ^= rotl32(x[ 2] + x[14], 13); x[10] ^= rotl32(x[ 6] + x[ 2], 18);
		x[ 3] ^= rotl32(x[15] + x[11],  7); x[ 7] ^= rotl32(x[ 3] + x[15],  9);
		x[11] ^= rotl32(x[ 7] + x[ 3], 13); x[15] ^= rotl32(x[11] + x[ 7], 18);
		x[ 1] ^= rotl32(x[ 0] + x[ 3],  7); x[ 2] ^= rotl32(x[ 1] + x[ 0],  9);
		x[ 3] ^= rotl32(x[ 2] + x[ 1], 13); x[ 0] ^= rotl32(x[ 3] + x[ 2], 18);
		x[ 6] ^= rotl32(x[ 5] + x[ 4],  7); x[ 7] ^= rotl32(x[ 6] + x[ 5],  9);
		x[ 4] ^= rotl32(x[ 7] + x[ 6], 13); x[ 5] ^= rotl32(x[ 4] + x[ 7], 18);
		x[11] ^= rotl32(x[10] + x[ 9],  7); x[ 8] ^= rotl32(x[11] + x[10],  9);
		x[ 9] ^= rotl32(x[ 8] + x[11], 13); x[10] ^= rotl32(x[ 9] + x[ 8], 18);
		x[12] ^= rotl32(x[15] + x[14],  7); x[13] ^= rotl32(x[12] + x[15],  9);
		x[14] ^= rotl32(x[13] + x[12], 13); x[15] ^= rotl32(x[14] + x[13], 18);
	}
	for (int i = 0; i < 16; i++) B[i] += x[i];
}

/* V is borrowed from the scratchpad; the fill overwrites it */
void scrypt_1024_1_1(const unsigned char hdr[80], unsigned char out[32], uint32_t *V)
{
	unsigned char k0[32], B[128];
	uint32_t X[32];

	sha256(hdr, 80, NULL, 0, k0);
	for (uint32_t i = 0; i < 4; i++) hmac(k0, hdr, 80, i + 1, B + 32 * i);
	for (int k = 0; k < 32; k++)
		X[k] = (uint32_t) B[4 * k] | ((uint32_t) B[4 * k + 1] << 8) |
		       ((uint32_t) B[4 * k + 2] << 16) | ((uint32_t) B[4 * k + 3] << 24);
	for (uint32_t i = 0; i < 1024; i++) {
		memcpy(&V[i * 32], X, 128);
		salsa8(&X[0], &X[16]);
		salsa8(&X[16], &X[0]);
	}
	for (uint32_t i = 0; i < 1024; i++) {
		const uint32_t j = X[16] & 1023;
		for (int k = 0; k < 32; k++) X[k] ^= V[j * 32 + k];
		salsa8(&X[0], &X[16]);
		salsa8(&X[16], &X[0]);
	}
	for (int k = 0; k < 32; k++) {
		B[4 * k] = (unsigned char) X[k];         B[4 * k + 1] = (unsigned char) (X[k] >> 8);
		B[4 * k + 2] = (unsigned char) (X[k] >> 16); B[4 * k + 3] = (unsigned char) (X[k] >> 24);
	}
	hmac(k0, B, 128, 1, out);
}

} // namespace

extern "C" void homescrypt_hash_tuned(void *output, const void *input, uint32_t mix_rounds, uint32_t fp_passes)
{
	static thread_local std::vector<uint64_t> tls;
	if (tls.size() != WORDS) tls.resize(WORDS);
	uint64_t *S = tls.data();
	const unsigned char *hdr = (const unsigned char *) input;

	unsigned char sseed[32], seed[32];
	scrypt_1024_1_1(hdr, sseed, (uint32_t *) S);
	{
		sph_sha256_context c;
		sph_sha256_init(&c);
		sph_sha256(&c, TAG_SEED, 20);
		sph_sha256(&c, hdr, 80);
		sph_sha256(&c, sseed, 32);
		sph_sha256_close(&c, seed);
	}

	uint64_t s[4] = { ld64(seed), ld64(seed + 8), ld64(seed + 16), ld64(seed + 24) };
	uint64_t x = s[0] ^ rotl(s[1], 17) ^ rotl(s[2], 31) ^ rotl(s[3], 47);

	for (uint32_t i = 0; i < WORDS; i++) {
		x += 0x9e3779b97f4a7c15ull + i;
		x ^= s[i & 3] + rotl(x, (i & 31) + 1);
		x = av(x);
		S[i] = x;
		s[i & 3] ^= x + rotl(s[(i + 1) & 3], 23);
	}

	for (uint32_t r = 0; r < mix_rounds; r++) {
		const uint32_t l0 = r & 3, l1 = (r + 1) & 3, l2 = (r + 2) & 3, l3 = (r + 3) & 3;
		const uint32_t i1 = (uint32_t) (s[l0] ^ x ^ r) & MASK;
		const uint64_t a = S[i1];
		const uint32_t i2 = (uint32_t) (a ^ rotl(s[l1], 17)) & MASK;
		const uint64_t b = S[i2];
		const uint32_t i3 = (uint32_t) (b ^ rotl(a, 31) ^ s[l2]) & MASK;
		const uint64_t c = S[i3];
		const uint64_t mixed = a + rotl(b, 23) + rotl(c, 41) + s[l3] + r;

		float m0 = safe((uint32_t) (c >> 32)), m1 = safe((uint32_t) c);
		float m2 = safe((uint32_t) (mixed >> 32)), m3 = safe((uint32_t) mixed);
		float g0 = safe((uint32_t) (a >> 32)), g1 = safe((uint32_t) a);
		float g2 = safe((uint32_t) (b >> 32)), g3 = safe((uint32_t) b);

		for (uint32_t p = 0; p < fp_passes; p++) {
			for (int j = 0; j < 8; j++) {
				g0 = fmaf(g0, m0, g3);
				g1 = fmaf(g1, m1, g0);
				g2 = fmaf(g2, m2, g1);
				g3 = fmaf(g3, m3, g2);
			}
			const uint32_t w0 = fbits(g0), w1 = fbits(g1), w2 = fbits(g2), w3 = fbits(g3);
			g0 = safe(w0); g1 = safe(w1); g2 = safe(w2); g3 = safe(w3);
			const uint32_t t = fbits(m0);
			m0 = safe(fbits(m1) ^ w0);
			m1 = safe(fbits(m2) ^ w1);
			m2 = safe(fbits(m3) ^ w2);
			m3 = safe(t ^ w3);
		}

		const uint64_t fp = (((uint64_t) fbits(g0) << 32) | fbits(g1))
			^ rotl(((uint64_t) fbits(g2) << 32) | fbits(g3), 29);

		s[l0] ^= mixed ^ fp;
		s[l0] = rotl(s[l0], (unsigned) (mixed & 31) + 1);
		s[l1] += a ^ rotl(c, 13);

		/* the three indices can alias: this order is consensus */
		S[i1] = a ^ s[l0] ^ rotl(c, 7);
		S[i2] = b + s[l1] + rotl(a, 19);
		S[i3] = c ^ mixed ^ rotl(b, 37);
		x = rotl(x ^ mixed ^ fp ^ S[i3], 29);
	}

	uint64_t f0 = s[0], f1 = s[1], f2 = s[2], f3 = s[3];
	for (uint32_t i = 0; i < WORDS; i += 4) {
		f0 = av(f0 ^ S[i]);
		f1 = av(f1 + S[i + 1]);
		f2 = av(f2 ^ rotl(S[i + 2], 17));
		f3 = av(f3 + rotl(S[i + 3], 41));
	}

	unsigned char folded[32];
	st64(folded, f0); st64(folded + 8, f1); st64(folded + 16, f2); st64(folded + 24, f3);
	sph_sha256_context c;
	sph_sha256_init(&c);
	sph_sha256(&c, TAG_FINAL, 21);
	sph_sha256(&c, seed, 32);
	sph_sha256(&c, folded, 32);
	sph_sha256(&c, hdr, 80);
	sph_sha256_close(&c, output);
}

extern "C" void homescrypt_hash(void *output, const void *input)
{
	homescrypt_hash_tuned(output, input, HOMESCRYPT_MIX_ROUNDS, HOMESCRYPT_FP_PASSES);
}

/* Every KAT vector, then a flipped header that must change the digest. */
extern "C" bool homescrypt_self_test(char *detail, size_t len)
{
	int ok = 0, n = 0;
	char failed[64] = "";
	for (const homescrypt_kat_vec &v : homescrypt_kat) {
		unsigned char h[80], d[32];
		memcpy(h, v.header, 80);
		if (v.override_nonce) memcpy(h + 76, &v.nonce, 4);
		homescrypt_hash_tuned(d, h, v.mix_rounds, v.fp_passes);
		n++;
		if (!memcmp(d, v.digest, 32)) ok++;
		else if (!failed[0]) snprintf(failed, sizeof failed, ", first failure %s", v.name);
	}
	unsigned char h[80], d[32];
	memcpy(h, homescrypt_kat[0].header, 80);
	h[79] ^= 1;
	homescrypt_hash(d, h);
	const bool nonvac = memcmp(d, homescrypt_kat[0].digest, 32) != 0;
	snprintf(detail, len, "%d/%d vectors, flipped header %s%s", ok, n, nonvac ? "differs" : "DOES NOT DIFFER", failed);
	return ok == n && nonvac;
}
