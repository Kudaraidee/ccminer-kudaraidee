/*
 * bc3hash front/final hashes on device: per-thread transcriptions of the coin's
 * standard_hashes.cpp (SHA3 FIPS 202, SHA-512, RIPEMD-160, BLAKE2s/b with
 * personalization), plus bc3hash's tagged-message assembly (hash.cpp).
 * They are <1% of the hash and written for clarity, not speed.
 */
#pragma once
#include <stdint.h>

#define BC3_DOMAIN "BC3Hash-BlockHash-Algorithm/BitcoinIII-PoW/2026"
#define BC3_DOMAIN_LEN 47

__device__ __forceinline__ uint64_t bc3h_rotl64(uint64_t v, int n) { return n ? (v << n) | (v >> (64 - n)) : v; }
__device__ __forceinline__ uint64_t bc3h_rotr64(uint64_t v, int n) { return bc3h_rotl64(v, (64 - n) & 63); }
__device__ __forceinline__ uint32_t bc3h_rotl32(uint32_t v, int n) { return n ? (v << n) | (v >> (32 - n)) : v; }
__device__ __forceinline__ uint32_t bc3h_rotr32(uint32_t v, int n) { return bc3h_rotl32(v, (32 - n) & 31); }

__device__ __forceinline__ uint64_t bc3h_le64(const uint8_t* p)
{
	uint64_t v = 0;
	for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
	return v;
}
__device__ __forceinline__ uint64_t bc3h_be64(const uint8_t* p)
{
	uint64_t v = 0;
	for (int i = 0; i < 8; i++) v = (v << 8) | p[i];
	return v;
}
__device__ __forceinline__ uint32_t bc3h_le32(const uint8_t* p)
{
	return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* ---------------------------------------------------------------- SHA3 */

__device__ __constant__ static const uint64_t BC3H_IOTA[24] = {
	0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull};
/* rho offset of lane x + 5y (standard_hashes.cpp rho_offsets()) */
__device__ __constant__ static const uint8_t BC3H_RHO[25] = {
	0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};

__device__ static void bc3h_keccakf(uint64_t* lanes)
{
	for (int r = 0; r < 24; r++) {
		uint64_t parity[5], moved[25];
#pragma unroll
		for (int x = 0; x < 5; x++)
			parity[x] = lanes[x] ^ lanes[x + 5] ^ lanes[x + 10] ^ lanes[x + 15] ^ lanes[x + 20];
#pragma unroll
		for (int y = 0; y < 5; y++)
#pragma unroll
			for (int x = 0; x < 5; x++) {
				const uint64_t corrected = lanes[x + 5 * y] ^ parity[(x + 4) % 5] ^ bc3h_rotl64(parity[(x + 1) % 5], 1);
				moved[y + 5 * ((2 * x + 3 * y) % 5)] = bc3h_rotl64(corrected, BC3H_RHO[x + 5 * y]);
			}
#pragma unroll
		for (int y = 0; y < 5; y++)
#pragma unroll
			for (int x = 0; x < 5; x++)
				lanes[x + 5 * y] = moved[x + 5 * y] ^ (~moved[(x + 1) % 5 + 5 * y] & moved[(x + 2) % 5 + 5 * y]);
		lanes[0] ^= BC3H_IOTA[r];
	}
}

/* One-shot SHA3-<bits> (0x06 domain padding). */
__device__ static void bc3h_sha3(uint8_t* out, int bits, const uint8_t* msg, int len)
{
	const int rate = 200 - bits / 4;
	uint64_t lanes[25];
	for (int i = 0; i < 25; i++) lanes[i] = 0;
	int pos = 0;
	for (int i = 0; i < len; i++) {
		lanes[pos / 8] ^= (uint64_t)msg[i] << (8 * (pos % 8));
		if (++pos == rate) {
			bc3h_keccakf(lanes);
			pos = 0;
		}
	}
	lanes[pos / 8] ^= 6ull << (8 * (pos % 8));
	lanes[rate / 8 - 1] ^= 0x80ull << 56;
	bc3h_keccakf(lanes);
	for (int i = 0; i < bits / 8; i++) out[i] = (uint8_t)(lanes[i / 8] >> (8 * (i % 8)));
}

/* ---------------------------------------------------------------- SHA-512 */

__device__ __constant__ static const uint64_t BC3H_SHA512_K[80] = {
	0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull,
	0x3956c25bf348b538ull, 0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull,
	0xd807aa98a3030242ull, 0x12835b0145706fbeull, 0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,
	0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull, 0xc19bf174cf692694ull,
	0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
	0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull,
	0x983e5152ee66dfabull, 0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull,
	0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull, 0x06ca6351e003826full, 0x142929670a0e6e70ull,
	0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull, 0x53380d139d95b3dfull,
	0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
	0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull,
	0xd192e819d6ef5218ull, 0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull,
	0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull, 0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull,
	0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull, 0x682e6ff3d6b2b8a3ull,
	0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
	0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull,
	0xca273eceea26619cull, 0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull,
	0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull, 0x113f9804bef90daeull, 0x1b710b35131c471bull,
	0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull, 0x431d67c49c100d4cull,
	0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull};
__device__ __constant__ static const uint64_t BC3H_IV64[8] = {
	0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
	0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
__device__ __constant__ static const uint32_t BC3H_IV32[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

__device__ static void bc3h_sha512_compress(uint64_t* chain, const uint8_t* data)
{
	uint64_t w[80];
	for (int i = 0; i < 16; i++) w[i] = bc3h_be64(data + 8 * i);
	for (int i = 16; i < 80; i++) {
		const uint64_t a = w[i - 15], b = w[i - 2];
		w[i] = w[i - 16] + w[i - 7] + (bc3h_rotr64(a, 1) ^ bc3h_rotr64(a, 8) ^ (a >> 7)) +
		       (bc3h_rotr64(b, 19) ^ bc3h_rotr64(b, 61) ^ (b >> 6));
	}
	uint64_t s[8];
	for (int i = 0; i < 8; i++) s[i] = chain[i];
	for (int i = 0; i < 80; i++) {
		const uint64_t head = s[7] + (bc3h_rotr64(s[4], 14) ^ bc3h_rotr64(s[4], 18) ^ bc3h_rotr64(s[4], 41)) +
		                      (s[6] ^ (s[4] & (s[5] ^ s[6]))) + BC3H_SHA512_K[i] + w[i];
		const uint64_t extra = (bc3h_rotr64(s[0], 28) ^ bc3h_rotr64(s[0], 34) ^ bc3h_rotr64(s[0], 39)) +
		                       ((s[0] & s[1]) | (s[2] & (s[0] | s[1])));
		for (int j = 7; j > 0; j--) s[j] = s[j - 1];
		s[4] += head;
		s[0] = head + extra;
	}
	for (int i = 0; i < 8; i++) chain[i] += s[i];
}

/* One-shot SHA-512 of len < 2^61 bytes. */
__device__ static void bc3h_sha512(uint8_t* out, const uint8_t* msg, int len)
{
	uint64_t chain[8];
	for (int i = 0; i < 8; i++) chain[i] = BC3H_IV64[i];
	uint8_t blk[128];
	int off = 0;
	for (; len - off >= 128; off += 128) bc3h_sha512_compress(chain, msg + off);
	int used = len - off;
	for (int i = 0; i < used; i++) blk[i] = msg[off + i];
	blk[used++] = 0x80;
	if (used > 112) {
		for (int i = used; i < 128; i++) blk[i] = 0;
		bc3h_sha512_compress(chain, blk);
		used = 0;
	}
	for (int i = used; i < 120; i++) blk[i] = 0; /* high length word is 0 */
	const uint64_t bitlen = (uint64_t)len << 3;
	for (int i = 0; i < 8; i++) blk[120 + i] = (uint8_t)(bitlen >> (56 - 8 * i));
	bc3h_sha512_compress(chain, blk);
	for (int i = 0; i < 64; i++) out[i] = (uint8_t)(chain[i / 8] >> (56 - 8 * (i % 8)));
}

/* ---------------------------------------------------------------- RIPEMD-160 */

__device__ __constant__ static const uint8_t BC3H_RMD_WORD[2][80] = {
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
	 3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
	 4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13},
	{5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
	 15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
	 12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11}};
__device__ __constant__ static const uint8_t BC3H_RMD_ROT[2][80] = {
	{11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
	 11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, 11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
	 9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6},
	{8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
	 9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
	 8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11}};
__device__ __constant__ static const uint32_t BC3H_RMD_K[2][5] = {
	{0, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e}, {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0}};

__device__ __forceinline__ uint32_t bc3h_rmd_f(int round, uint32_t x, uint32_t y, uint32_t z)
{
	switch (round) {
	case 0: return x ^ y ^ z;
	case 1: return z ^ (x & (y ^ z));
	case 2: return (x | ~y) ^ z;
	case 3: return y ^ (z & (x ^ y));
	default: return x ^ (y | ~z);
	}
}

__device__ static void bc3h_rmd_compress(uint32_t* chain, const uint8_t* data)
{
	uint32_t m[16];
	for (int i = 0; i < 16; i++) m[i] = bc3h_le32(data + 4 * i);
	uint32_t s[2][5];
	for (int p = 0; p < 2; p++)
		for (int i = 0; i < 5; i++) s[p][i] = chain[i];
	for (int i = 0; i < 80; i++)
		for (int p = 0; p < 2; p++) {
			const uint32_t sum = s[p][0] + bc3h_rmd_f(p ? 4 - i / 16 : i / 16, s[p][1], s[p][2], s[p][3]) +
			                     m[BC3H_RMD_WORD[p][i]] + BC3H_RMD_K[p][i / 16];
			const uint32_t n1 = bc3h_rotl32(sum, BC3H_RMD_ROT[p][i]) + s[p][4];
			const uint32_t n3 = bc3h_rotl32(s[p][2], 10);
			s[p][0] = s[p][4];
			s[p][2] = s[p][1];
			s[p][1] = n1;
			s[p][4] = s[p][3];
			s[p][3] = n3;
		}
	uint32_t before[5];
	for (int i = 0; i < 5; i++) before[i] = chain[i];
	for (int i = 0; i < 5; i++) chain[i] = before[(i + 1) % 5] + s[0][(i + 2) % 5] + s[1][(i + 3) % 5];
}

__device__ static void bc3h_ripemd160(uint8_t* out, const uint8_t* msg, int len)
{
	uint32_t chain[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
	uint8_t blk[64];
	int off = 0;
	for (; len - off >= 64; off += 64) bc3h_rmd_compress(chain, msg + off);
	int used = len - off;
	for (int i = 0; i < used; i++) blk[i] = msg[off + i];
	blk[used++] = 0x80;
	if (used > 56) {
		for (int i = used; i < 64; i++) blk[i] = 0;
		bc3h_rmd_compress(chain, blk);
		used = 0;
	}
	for (int i = used; i < 56; i++) blk[i] = 0;
	const uint64_t bitlen = (uint64_t)len << 3;
	for (int i = 0; i < 8; i++) blk[56 + i] = (uint8_t)(bitlen >> (8 * i));
	bc3h_rmd_compress(chain, blk);
	for (int i = 0; i < 20; i++) out[i] = (uint8_t)(chain[i / 4] >> (8 * (i % 4)));
}

/* ---------------------------------------------------------------- BLAKE2 */

__device__ __constant__ static const uint8_t BC3H_SIGMA[10][16] = {
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}};

/* W = uint32_t (BLAKE2s) or uint64_t (BLAKE2b); bytes_done counts through this block. */
template <class W>
__device__ void bc3h_blake2_compress(W* chain, const uint8_t* blk, uint64_t bytes_done, bool last)
{
	constexpr int SW = sizeof(W);
	constexpr int R0 = SW == 8 ? 32 : 16, R1 = SW == 8 ? 24 : 12, R2 = SW == 8 ? 16 : 8, R3 = SW == 8 ? 63 : 7;
	W m[16], v[16];
	for (int i = 0; i < 16; i++) {
		W x = 0;
		for (int b = 0; b < SW; b++) x |= (W)blk[SW * i + b] << (8 * b);
		m[i] = x;
	}
	for (int i = 0; i < 8; i++) {
		v[i] = chain[i];
		v[i + 8] = SW == 8 ? (W)BC3H_IV64[i] : (W)BC3H_IV32[i];
	}
	v[12] ^= (W)bytes_done;
	if (SW == 4) v[13] ^= (W)(bytes_done >> 32);
	if (last) v[14] = ~v[14];
	for (int round = 0; round < (SW == 8 ? 12 : 10); round++) {
		const uint8_t* order = BC3H_SIGMA[round % 10];
		for (int diagonal = 0; diagonal < 2; diagonal++)
			for (int col = 0; col < 4; col++) {
				W& a = v[col];
				W& b = v[4 + (col + diagonal) % 4];
				W& c = v[8 + (col + 2 * diagonal) % 4];
				W& d = v[12 + (col + 3 * diagonal) % 4];
				const int pair = 8 * diagonal + 2 * col;
				a += b + m[order[pair]];
				d = (d ^ a) >> R0 | (d ^ a) << (8 * SW - R0);
				c += d;
				b = (b ^ c) >> R1 | (b ^ c) << (8 * SW - R1);
				a += b + m[order[pair + 1]];
				d = (d ^ a) >> R2 | (d ^ a) << (8 * SW - R2);
				c += d;
				b = (b ^ c) >> R3 | (b ^ c) << (8 * SW - R3);
			}
	}
	for (int i = 0; i < 8; i++) chain[i] ^= v[i] ^ v[i + 8];
}

/* Unkeyed, sequential, full-length BLAKE2 with a 2*sizeof(W)-byte personalization.
 * A block is compressed only once more input follows it (the last block may be
 * partial, or full when len is a multiple of the block size). len > 0. */
template <class W>
__device__ void bc3h_blake2(uint8_t* out, const uint8_t* msg, int len, const uint8_t* person)
{
	constexpr int SW = sizeof(W), BLOCK = 16 * SW;
	W chain[8];
	for (int i = 0; i < 8; i++) chain[i] = SW == 8 ? (W)BC3H_IV64[i] : (W)BC3H_IV32[i];
	chain[0] ^= (W)(0x01010000u + 8 * SW);
	W p0 = 0, p1 = 0;
	for (int b = 0; b < SW; b++) {
		p0 |= (W)person[b] << (8 * b);
		p1 |= (W)person[SW + b] << (8 * b);
	}
	chain[6] ^= p0;
	chain[7] ^= p1;
	int off = 0;
	for (; len - off > BLOCK; off += BLOCK) bc3h_blake2_compress<W>(chain, msg + off, (uint64_t)(off + BLOCK), false);
	uint8_t blk[BLOCK];
	const int used = len - off;
	for (int i = 0; i < BLOCK; i++) blk[i] = i < used ? msg[off + i] : 0;
	bc3h_blake2_compress<W>(chain, blk, (uint64_t)len, true);
	for (int i = 0; i < 8 * SW; i++) out[i] = (uint8_t)(chain[i / SW] >> (8 * (i % SW)));
}

/* ---------------------------------------------------------------- bc3hash */

__device__ __constant__ static const uint8_t BC3H_PERSON_S[8] = {0x0a, 0x20, 0xe8, 0x11, 0x61, 0x09, 0x08, 0xba};
__device__ __constant__ static const uint8_t BC3H_PERSON_B[16] = {0x1e, 0x80, 0xf2, 0xfd, 0xdc, 0x58, 0x18, 0xd6,
                                                                  0xfe, 0x9b, 0x58, 0x1e, 0x3b, 0xd4, 0x93, 0x0b};

/* msg = DOMAIN | tag | data; returns length. buf must hold 47 + 11 + dlen. */
__device__ __forceinline__ int bc3h_tagged(uint8_t* buf, const char* tag, const uint8_t* data, int dlen)
{
	const char* d = BC3_DOMAIN;
	int n = 0;
	while (*d) buf[n++] = (uint8_t)*d++;
	while (*tag) buf[n++] = (uint8_t)*tag++;
	for (int i = 0; i < dlen; i++) buf[n++] = data[i];
	return n;
}

/* The front end of bc3hash (hash.cpp State, update(header80)): s512 is mix512's
 * input; the 228 bytes s384|sh|r160|bs|bb are kept for the final SHA3-256. */
struct bc3_front {
	uint8_t s512[64];
	uint8_t s384[48];
	uint8_t sh[64];
	uint8_t r160[20];
	uint8_t bs[32];
	uint8_t bb[64];
};

__device__ void bc3h_front(bc3_front& f, const uint8_t* header80)
{
	uint8_t buf[BC3_DOMAIN_LEN + 11 + 80];
	int n = bc3h_tagged(buf, "/SHA3-512", header80, 80);
	bc3h_sha3(f.s512, 512, buf, n);
	n = bc3h_tagged(buf, "/SHA3-384", header80, 80);
	bc3h_sha3(f.s384, 384, buf, n);
	n = bc3h_tagged(buf, "/SHA-512", header80, 80);
	bc3h_sha512(f.sh, buf, n);
	n = bc3h_tagged(buf, "/RIPEMD-160", header80, 80);
	bc3h_ripemd160(f.r160, buf, n);
	n = bc3h_tagged(buf, "/BLAKE2s", header80, 80);
	bc3h_blake2<uint32_t>(f.bs, buf, n, BC3H_PERSON_S);
	n = bc3h_tagged(buf, "/BLAKE2b", header80, 80);
	bc3h_blake2<uint64_t>(f.bb, buf, n, BC3H_PERSON_B);
}

/* hash = SHA3-256(DOMAIN|"/SHA3-256"|s384|mix|sh|r160|bs|bb), 316-byte message. */
__device__ void bc3h_final(uint8_t* out32, const bc3_front& f, const uint8_t* mix32)
{
	uint8_t buf[316];
	int n = bc3h_tagged(buf, "/SHA3-256", f.s384, 48);
	for (int i = 0; i < 32; i++) buf[n++] = mix32[i];
	for (int i = 0; i < 64; i++) buf[n++] = f.sh[i];
	for (int i = 0; i < 20; i++) buf[n++] = f.r160[i];
	for (int i = 0; i < 32; i++) buf[n++] = f.bs[i];
	for (int i = 0; i < 64; i++) buf[n++] = f.bb[i];
	bc3h_sha3(out32, 256, buf, n);
}
