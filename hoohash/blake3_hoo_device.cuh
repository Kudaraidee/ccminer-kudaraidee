// Self-contained BLAKE3-256 for the HoohashV110 device path.
//
// Why a private copy instead of rinhash/blake3_device.cuh: that header defines
// non-static host symbols (and a global thrust::host_vector), so it can only be
// linked into ONE translation unit. Hoohash needs BLAKE3 in its own TU, so we
// bundle a minimal, integer-only, single-chunk BLAKE3-256 with EVERYTHING marked
// `static __device__` (internal linkage -> no multiple-definition at link, and no
// thrust dependency).
//
// Standard BLAKE3, unkeyed, single chunk (input <= 1024 bytes; Hoohash only ever
// feeds 80- and 32-byte buffers). Bit-exact with the reference (validated against
// the real-block KAT). NOT valid for inputs > 1024 bytes (would need the tree).
#pragma once
#include <stdint.h>
#include <stddef.h>

static __device__ __forceinline__ uint32_t hoo_b3_rotr(uint32_t w, int c) {
	return (w >> c) | (w << (32 - c));
}

static __device__ __forceinline__ void hoo_b3_g(uint32_t* s, int a, int b, int c, int d,
                                                uint32_t mx, uint32_t my) {
	s[a] = s[a] + s[b] + mx;
	s[d] = hoo_b3_rotr(s[d] ^ s[a], 16);
	s[c] = s[c] + s[d];
	s[b] = hoo_b3_rotr(s[b] ^ s[c], 12);
	s[a] = s[a] + s[b] + my;
	s[d] = hoo_b3_rotr(s[d] ^ s[a], 8);
	s[c] = s[c] + s[d];
	s[b] = hoo_b3_rotr(s[b] ^ s[c], 7);
}

static __device__ void hoo_b3_round(uint32_t* s, const uint32_t* m) {
	hoo_b3_g(s, 0, 4, 8,  12, m[0],  m[1]);
	hoo_b3_g(s, 1, 5, 9,  13, m[2],  m[3]);
	hoo_b3_g(s, 2, 6, 10, 14, m[4],  m[5]);
	hoo_b3_g(s, 3, 7, 11, 15, m[6],  m[7]);
	hoo_b3_g(s, 0, 5, 10, 15, m[8],  m[9]);
	hoo_b3_g(s, 1, 6, 11, 12, m[10], m[11]);
	hoo_b3_g(s, 2, 7, 8,  13, m[12], m[13]);
	hoo_b3_g(s, 3, 4, 9,  14, m[14], m[15]);
}

// Returns the first 8 output words (chaining value == 256-bit root output).
static __device__ void hoo_b3_compress(const uint32_t cv[8], const uint32_t block[16],
                                       uint64_t counter, uint32_t block_len, uint32_t flags,
                                       uint32_t out_cv[8]) {
	const uint32_t IV[8] = {
		0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
		0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
	};
	const uint8_t SIGMA[16] = { 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8 };

	uint32_t s[16];
	s[0]=cv[0]; s[1]=cv[1]; s[2]=cv[2]; s[3]=cv[3];
	s[4]=cv[4]; s[5]=cv[5]; s[6]=cv[6]; s[7]=cv[7];
	s[8]=IV[0]; s[9]=IV[1]; s[10]=IV[2]; s[11]=IV[3];
	s[12]=(uint32_t)counter;
	s[13]=(uint32_t)(counter >> 32);
	s[14]=block_len;
	s[15]=flags;

	uint32_t m[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) m[i] = block[i];

	for (int r = 0; r < 7; r++) {
		hoo_b3_round(s, m);
		if (r < 6) {
			uint32_t pm[16];
			#pragma unroll
			for (int i = 0; i < 16; i++) pm[i] = m[SIGMA[i]];
			#pragma unroll
			for (int i = 0; i < 16; i++) m[i] = pm[i];
		}
	}

	#pragma unroll
	for (int i = 0; i < 8; i++) out_cv[i] = s[i] ^ s[i + 8];
}

// Unkeyed BLAKE3-256 over a single chunk. out = 32 bytes, little-endian words.
static __device__ void hoo_blake3_256(const uint8_t* input, size_t len, uint8_t* out) {
	const uint32_t IV[8] = {
		0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
		0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
	};
	const uint32_t CHUNK_START = 1, CHUNK_END = 2, ROOT = 8;

	uint32_t cv[8];
	#pragma unroll
	for (int i = 0; i < 8; i++) cv[i] = IV[i];

	size_t pos = 0;
	uint32_t blocks_done = 0;
	while (true) {
		size_t remaining = len - pos;
		uint32_t block_len = remaining >= 64 ? 64u : (uint32_t)remaining;
		bool is_last_block = (remaining <= 64);

		uint8_t buf[64];
		#pragma unroll
		for (int i = 0; i < 64; i++) buf[i] = (i < (int)block_len) ? input[pos + i] : 0;

		uint32_t block[16];
		#pragma unroll
		for (int i = 0; i < 16; i++)
			block[i] = (uint32_t)buf[i*4] | ((uint32_t)buf[i*4+1] << 8) |
			           ((uint32_t)buf[i*4+2] << 16) | ((uint32_t)buf[i*4+3] << 24);

		uint32_t flags = 0;
		if (blocks_done == 0) flags |= CHUNK_START;
		if (is_last_block)    flags |= CHUNK_END | ROOT;

		uint32_t out_cv[8];
		hoo_b3_compress(cv, block, 0 /*single-chunk counter*/, block_len, flags, out_cv);
		#pragma unroll
		for (int i = 0; i < 8; i++) cv[i] = out_cv[i];

		blocks_done++;
		pos += block_len;
		if (is_last_block) break;
	}

	#pragma unroll
	for (int i = 0; i < 8; i++) {
		out[i*4]   = (uint8_t)(cv[i]);
		out[i*4+1] = (uint8_t)(cv[i] >> 8);
		out[i*4+2] = (uint8_t)(cv[i] >> 16);
		out[i*4+3] = (uint8_t)(cv[i] >> 24);
	}
}
