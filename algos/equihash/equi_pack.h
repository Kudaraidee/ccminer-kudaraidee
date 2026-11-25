// ---------------------------------------------------------------------------
// Equihash minimal-solution packing: indices -> the bytes a share carries.
//
// A solution is transmitted as PROOFSIZE indices packed at (cBitLen + 1) bits
// each, big-endian, with no padding between them:
//
//   200/9  512 indices x 21 bits = 1344 bytes
//   144/5   32 indices x 25 bits =  100 bytes
//   192/7  128 indices x 25 bits =  400 bytes
//
// This lives in a header rather than in equihash.cpp because it is the ONLY
// step between a verified solution and the wire, and the host re-verify cannot
// see it: the re-verify checks INDICES, so a packing defect leaves every
// in-tree gate green and shows up only as pool rejects. Keeping one copy lets
// the gates exercise the same source the miner ships.
//
// Self-contained on purpose (no miner.h): the correctness gates link it
// without the miner. Everything is static, so each translation unit compiles
// its own copy of one text and the two cannot drift.
// ---------------------------------------------------------------------------
#ifndef EQUI_PACK_H
#define EQUI_PACK_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <vector>

// Pack in_len bytes of bit_len-wide big-endian elements down to out_len bytes.
// `byte_pad` is the number of leading bytes of each input element to skip, so a
// 25-bit index stored in a 4-byte slot uses byte_pad = 0 and a 21-bit index
// uses byte_pad = 1.
static void eq_compress_array(const unsigned char *in, size_t in_len,
                              unsigned char *out, size_t out_len,
                              size_t bit_len, size_t byte_pad)
{
	assert(bit_len >= 8);
	assert(8 * sizeof(uint32_t) >= 7 + bit_len);

	const size_t in_width = (bit_len + 7) / 8 + byte_pad;
	assert(out_len == bit_len * in_len / (8 * in_width));
	(void) in_len;

	const uint32_t bit_len_mask = (uint32_t)((1UL << bit_len) - 1);

	// The acc_bits least-significant bits of acc_value hold a big-endian bit
	// sequence waiting to be emitted.
	size_t acc_bits = 0, j = 0;
	uint32_t acc_value = 0;

	for (size_t i = 0; i < out_len; i++) {
		if (acc_bits < 8) {              // fewer than 8 bits left: read an element
			acc_value = acc_value << bit_len;
			for (size_t x = byte_pad; x < in_width; x++)
				acc_value |= (uint32_t)((in[j + x] &
					((bit_len_mask >> (8 * (in_width - x - 1))) & 0xFF))
					<< (8 * (in_width - x - 1)));      // big-endian
			j += in_width;
			acc_bits += bit_len;
		}
		acc_bits -= 8;
		out[i] = (unsigned char)((acc_value >> acc_bits) & 0xFF);
	}
}

// Big-endian, so that a lexicographic compare of the array equals an integer
// compare of the index -- which is what the consensus ordering rule relies on.
// Written as shifts rather than htobe32 + memcpy: the byte order is the wire
// format, not the host's, and the old form needed a per-platform shim.
static void eq_index_to_array(const uint32_t i, unsigned char *arr)
{
	arr[0] = (unsigned char)(i >> 24);
	arr[1] = (unsigned char)(i >> 16);
	arr[2] = (unsigned char)(i >> 8);
	arr[3] = (unsigned char)(i);
}

// The submitted solution for `indices` at collision-bit-length cBitLen.
static std::vector<unsigned char> eq_minimal_from_indices(const std::vector<uint32_t> &indices,
                                                          size_t cBitLen)
{
	assert(((cBitLen + 1) + 7) / 8 <= sizeof(uint32_t));
	const size_t lenIndices = indices.size() * sizeof(uint32_t);
	const size_t minLen     = (cBitLen + 1) * lenIndices / (8 * sizeof(uint32_t));
	const size_t bytePad    = sizeof(uint32_t) - ((cBitLen + 1) + 7) / 8;

	std::vector<unsigned char> array(lenIndices);
	for (size_t i = 0; i < indices.size(); i++)
		eq_index_to_array(indices[i], array.data() + i * sizeof(uint32_t));

	std::vector<unsigned char> ret(minLen);
	eq_compress_array(array.data(), lenIndices, ret.data(), minLen, cBitLen + 1, bytePad);
	return ret;
}

#endif // EQUI_PACK_H
