/*
 * bc3hash host oracle: BitcoinIII Core's crypto/bc3 (vendored in ref/ by
 * ref/vendor.py) as ONE C++17 translation unit, plus the C entry points the
 * CUDA side needs: the CPU hash for re-verify / self-test, the FixedPower
 * tables and the EC window bases the device builds its curve tables from.
 *
 * Needs C++17 (MSVC: /std:c++17 on this file only) and the include dirs
 * ref, ref/include and the vendored libsecp256k1 headers (curvehash's).
 */
#include "ref/mix.cpp"
#include "ref/hash.cpp"
#include "ref/standard_hashes.cpp"

#include <secp256k1.h>

#include <cstring>

using namespace bc3::detail;
using namespace bc3::detail::constants;

extern "C" {

/* bc3hash of an 80-byte serialized header -> 32 digest bytes. The digest bytes
 * are Core's uint256 as-is (little-endian number): compare as 8 LE words. */
void bc3hash_hash(void* output, const void* input80)
{
	const auto d = bc3::bc3hash(input80, 80);
	std::memcpy(output, d.data(), 32);
}

void bc3hash_mix512(void* output32, const void* input64)
{
	const auto d = bc3::bc3mix512(input64);
	std::memcpy(output32, d.data(), 32);
}

/* mix.cpp Tables::pow64/pow128/pow256 (which = 0/1/2): [bytes][256] values of
 * 8/16/32 bytes, limbs least significant first = the device layout. */
const void* bc3hash_pow_table(int which, size_t* bytes)
{
	static const FixedPower<F64> p64(g64);
	static const FixedPower<F128> p128(g128);
	static const FixedPower<F256> p256(g256);
	switch (which) {
	case 0: *bytes = sizeof(p64.table); return p64.table.data();
	case 1: *bytes = sizeof(p128.table); return p128.table.data();
	default: *bytes = sizeof(p256.table); return p256.table.data();
	}
}

/* The window bases 2^(wbits*j) * P (j = 0..count-1, count <= 16) of curve point
 * H/J/K/L (0..3) as big-endian X||Y, 64 bytes each, by libsecp doublings.
 * Returns 0 on failure. */
int bc3hash_window_bases(int point, int wbits, int count, unsigned char* out)
{
	secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
	if (!ctx) return 0;
	const U256 xs[4] = {H_x, J_x, K_x, L_x}, ys[4] = {H_y, J_y, K_y, L_y};
	unsigned char enc[65] = {4};
	xs[point].store(enc + 1);
	ys[point].store(enc + 33);
	secp256k1_pubkey b;
	int ok = secp256k1_ec_pubkey_parse(ctx, &b, enc, 65);
	for (int j = 0; ok && j < count; j++) {
		size_t len = 65;
		ok = secp256k1_ec_pubkey_serialize(ctx, enc, &len, &b, SECP256K1_EC_UNCOMPRESSED);
		std::memcpy(out + 64 * j, enc + 1, 64);
		for (int d = 0; ok && d < wbits; d++) {
			const secp256k1_pubkey* pts[2] = {&b, &b};
			secp256k1_pubkey n;
			ok = secp256k1_ec_pubkey_combine(ctx, &n, pts, 2);
			b = n;
		}
	}
	secp256k1_context_destroy(ctx);
	return ok;
}

/* The coin's curve_multiply for point 0..3, scalar/x/y as 8 LE 32-bit limbs.
 * Used to spot-check the device-built tables. */
int bc3hash_curve_ref(int point, const uint32_t* k, uint32_t* x, uint32_t* y)
{
	try {
		U256 s;
		std::memcpy(s.v.data(), k, 32);
		const auto p = curve_multiply(unsigned(point), s);
		std::memcpy(x, p.first.v.data(), 32);
		std::memcpy(y, p.second.v.data(), 32);
		return 1;
	} catch (...) {
		return 0;
	}
}

} // extern "C"
