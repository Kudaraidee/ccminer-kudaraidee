/*
Based upon the 2 Christians,klaus_t's, Tanguy Pruvot's, tsiv and SP's work (2013-2016)
Provos Alexis - 2016
optimized by sp - 2018 (+20% faster on the gtx 1080ti)
*/

#include "miner.h"
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"
#include "cuda_x11_aes_sp.cuh"

//#define INTENSIVE_GMF
//#include "cuda_x11_aes.cuh"
#include "x15/cuda_whirlpool_tables.cuh"

__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c)
{
	asm("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a) : "r"(b), "r"(c));	// 0xEA = (F0 ^ CC) ^ AA
	return a;
}
//--------START OF WHIRLPOOL DEVICE MACROS---------------------------------------------------------------------------
__constant__ static uint2 b0[256];

__constant__ static uint2 precomputed_round_key_64[72];

__constant__ uint2 InitVector_RC[10];

__device__ __forceinline__
void static TRANSFER(uint2 *const __restrict__ dst, const uint2 *const __restrict__ src){
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
	dst[3] = src[3];
	dst[4] = src[4];
	dst[5] = src[5];
	dst[6] = src[6];
	dst[7] = src[7];
}

__device__ __forceinline__ uint2 d_ROUND_ELT(const uint32_t index, const uint2 sharedMemory[256][16], const uint2 *const __restrict__ in, const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){

	uint2 ret = sharedMemory[__byte_perm(in[i0].x, 0, 0x4440)][threadIdx.x & index];  //__ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= ROL8(sharedMemory[__byte_perm(in[i1].x, 0, 0x4441)][threadIdx.x & index]);
	ret ^= ROL16(sharedMemory[__byte_perm(in[i2].x, 0, 0x4442)][threadIdx.x &index]);
	ret ^= ROL24(sharedMemory[__byte_perm(in[i3].x, 0, 0x4443)][threadIdx.x &index]);
	ret ^= SWAPUINT2(sharedMemory[__byte_perm(in[i4].y, 0, 0x4440)][threadIdx.x &index]);
	ret ^= ROR24(sharedMemory[__byte_perm(in[i5].y, 0, 0x4441)][threadIdx.x &index]);
	ret ^= ROR16(sharedMemory[__byte_perm(in[i6].y, 0, 0x4442)][threadIdx.x &index]); //ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= ROR8(sharedMemory[__byte_perm(in[i7].y, 0, 0x4443)][threadIdx.x &index]); //__ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
uint2 d_ROUND_ELT1(const uint32_t index, const uint2 sharedMemory[256][16], const uint2 *const __restrict__ in, const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){
	uint2 ret = sharedMemory[__byte_perm(in[i0].x, 0, 0x4440)][threadIdx.x & index];
	ret ^= ROL8(sharedMemory[__byte_perm(in[i1].x, 0, 0x4441)][threadIdx.x & index]);
	ret ^= ROL16(sharedMemory[__byte_perm(in[i2].x, 0, 0x4442)][threadIdx.x & index]);
	ret ^= ROL24(sharedMemory[__byte_perm(in[i3].x, 0, 0x4443)][threadIdx.x & index]);
	ret ^= SWAPUINT2(sharedMemory[__byte_perm(in[i4].y, 0, 0x4440)][threadIdx.x & index]);
	ret ^= ROR24(sharedMemory[__byte_perm(in[i5].y, 0, 0x4441)][threadIdx.x & index]);
	ret ^= ROR16(sharedMemory[__byte_perm(in[i6].y, 0, 0x4442)][threadIdx.x & index]);//sharedMemory[6][__byte_perm(in[i6].y, 0, 0x4442)]
	ret ^= ROR8(sharedMemory[__byte_perm(in[i7].y, 0, 0x4443)][threadIdx.x & index]);//sharedMemory[7][__byte_perm(in[i7].y, 0, 0x4443)]
	ret ^= c0;
	return ret;
}
//--------END OF WHIRLPOOL DEVICE MACROS-----------------------------------------------------------------------------

//---hamsi macros---
__constant__  uint32_t d_alpha_n[] = {
	0xff00f0f0, 0xccccaaaa, 0xf0f0cccc, 0xff00aaaa, 0xccccaaaa, 0xf0f0ff00, 0xaaaacccc, 0xf0f0ff00, 0xf0f0cccc, 0xaaaaff00, 0xccccff00, 0xaaaaf0f0, 0xaaaaf0f0, 0xff00cccc, 0xccccf0f0, 0xff00aaaa,
	0xccccaaaa, 0xff00f0f0, 0xff00aaaa, 0xf0f0cccc, 0xf0f0ff00, 0xccccaaaa, 0xf0f0ff00, 0xaaaacccc, 0xaaaaff00, 0xf0f0cccc, 0xaaaaf0f0, 0xccccff00, 0xff00cccc, 0xaaaaf0f0, 0xff00aaaa, 0xccccf0f0
};

__constant__  uint32_t d_alpha_f[] = {
	0xcaf9639c, 0x0ff0f9c0, 0x639c0ff0, 0xcaf9f9c0, 0x0ff0f9c0, 0x639ccaf9, 0xf9c00ff0, 0x639ccaf9, 0x639c0ff0, 0xf9c0caf9, 0x0ff0caf9, 0xf9c0639c, 0xf9c0639c, 0xcaf90ff0, 0x0ff0639c, 0xcaf9f9c0,
	0x0ff0f9c0, 0xcaf9639c, 0xcaf9f9c0, 0x639c0ff0, 0x639ccaf9, 0x0ff0f9c0, 0x639ccaf9, 0xf9c00ff0, 0xf9c0caf9, 0x639c0ff0, 0xf9c0639c, 0x0ff0caf9, 0xcaf90ff0, 0xf9c0639c, 0xcaf9f9c0, 0x0ff0639c
};

__constant__  uint32_t c_c[] = {
	0x73746565, 0x6c706172, 0x6b204172, 0x656e6265, 0x72672031, 0x302c2062, 0x75732032, 0x3434362c,
	0x20422d33, 0x30303120, 0x4c657576, 0x656e2d48, 0x65766572, 0x6c65652c, 0x2042656c, 0x6769756d
};

__constant__  uint32_t d_T512[1024] = {
	0xef0b0270, 0x3afd0000, 0x5dae0000, 0x69490000, 0x9b0f3c06, 0x4405b5f9, 0x66140a51, 0x924f5d0a, 0xc96b0030, 0xe7250000, 0x2f840000, 0x264f0000, 0x08695bf9, 0x6dfcf137, 0x509f6984, 0x9e69af68,
	0xc96b0030, 0xe7250000, 0x2f840000, 0x264f0000, 0x08695bf9, 0x6dfcf137, 0x509f6984, 0x9e69af68, 0x26600240, 0xddd80000, 0x722a0000, 0x4f060000, 0x936667ff, 0x29f944ce, 0x368b63d5, 0x0c26f262,
	0x145a3c00, 0xb9e90000, 0x61270000, 0xf1610000, 0xce613d6c, 0xb0493d78, 0x47a96720, 0xe18e24c5, 0x23671400, 0xc8b90000, 0xf4c70000, 0xfb750000, 0x73cd2465, 0xf8a6a549, 0x02c40a3f, 0xdc24e61f,
	0x23671400, 0xc8b90000, 0xf4c70000, 0xfb750000, 0x73cd2465, 0xf8a6a549, 0x02c40a3f, 0xdc24e61f, 0x373d2800, 0x71500000, 0x95e00000, 0x0a140000, 0xbdac1909, 0x48ef9831, 0x456d6d1f, 0x3daac2da,
	0x54285c00, 0xeaed0000, 0xc5d60000, 0xa1c50000, 0xb3a26770, 0x94a5c4e1, 0x6bb0419d, 0x551b3782, 0x9cbb1800, 0xb0d30000, 0x92510000, 0xed930000, 0x593a4345, 0xe114d5f4, 0x430633da, 0x78cace29,
	0x9cbb1800, 0xb0d30000, 0x92510000, 0xed930000, 0x593a4345, 0xe114d5f4, 0x430633da, 0x78cace29, 0xc8934400, 0x5a3e0000, 0x57870000, 0x4c560000, 0xea982435, 0x75b11115, 0x28b67247, 0x2dd1f9ab,
	0x29449c00, 0x64e70000, 0xf24b0000, 0xc2f30000, 0x0ede4e8f, 0x56c23745, 0xf3e04259, 0x8d0d9ec4, 0x466d0c00, 0x08620000, 0xdd5d0000, 0xbadd0000, 0x6a927942, 0x441f2b93, 0x218ace6f, 0xbf2c0be2,
	0x466d0c00, 0x08620000, 0xdd5d0000, 0xbadd0000, 0x6a927942, 0x441f2b93, 0x218ace6f, 0xbf2c0be2, 0x6f299000, 0x6c850000, 0x2f160000, 0x782e0000, 0x644c37cd, 0x12dd1cd6, 0xd26a8c36, 0x32219526,
	0xf6800005, 0x3443c000, 0x24070000, 0x8f3d0000, 0x21373bfb, 0x0ab8d5ae, 0xcdc58b19, 0xd795ba31, 0xa67f0001, 0x71378000, 0x19fc0000, 0x96db0000, 0x3a8b6dfd, 0xebcaaef3, 0x2c6d478f, 0xac8e6c88,
	0xa67f0001, 0x71378000, 0x19fc0000, 0x96db0000, 0x3a8b6dfd, 0xebcaaef3, 0x2c6d478f, 0xac8e6c88, 0x50ff0004, 0x45744000, 0x3dfb0000, 0x19e60000, 0x1bbc5606, 0xe1727b5d, 0xe1a8cc96, 0x7b1bd6b9,
	0xf7750009, 0xcf3cc000, 0xc3d60000, 0x04920000, 0x029519a9, 0xf8e836ba, 0x7a87f14e, 0x9e16981a, 0xd46a0000, 0x8dc8c000, 0xa5af0000, 0x4a290000, 0xfc4e427a, 0xc9b4866c, 0x98369604, 0xf746c320,
	0xd46a0000, 0x8dc8c000, 0xa5af0000, 0x4a290000, 0xfc4e427a, 0xc9b4866c, 0x98369604, 0xf746c320, 0x231f0009, 0x42f40000, 0x66790000, 0x4ebb0000, 0xfedb5bd3, 0x315cb0d6, 0xe2b1674a, 0x69505b3a,
	0x774400f0, 0xf15a0000, 0xf5b20000, 0x34140000, 0x89377e8c, 0x5a8bec25, 0x0bc3cd1e, 0xcf3775cb, 0xf46c0050, 0x96180000, 0x14a50000, 0x031f0000, 0x42947eb8, 0x66bf7e19, 0x9ca470d2, 0x8a341574,
	0xf46c0050, 0x96180000, 0x14a50000, 0x031f0000, 0x42947eb8, 0x66bf7e19, 0x9ca470d2, 0x8a341574, 0x832800a0, 0x67420000, 0xe1170000, 0x370b0000, 0xcba30034, 0x3c34923c, 0x9767bdcc, 0x450360bf,
	0xe8870170, 0x9d720000, 0x12db0000, 0xd4220000, 0xf2886b27, 0xa921e543, 0x4ef8b518, 0x618813b1, 0xb4370060, 0x0c4c0000, 0x56c20000, 0x5cae0000, 0x94541f3f, 0x3b3ef825, 0x1b365f3d, 0xf3d45758,
	0xb4370060, 0x0c4c0000, 0x56c20000, 0x5cae0000, 0x94541f3f, 0x3b3ef825, 0x1b365f3d, 0xf3d45758, 0x5cb00110, 0x913e0000, 0x44190000, 0x888c0000, 0x66dc7418, 0x921f1d66, 0x55ceea25, 0x925c44e9,
	0x0c720000, 0x49e50f00, 0x42790000, 0x5cea0000, 0x33aa301a, 0x15822514, 0x95a34b7b, 0xb44b0090, 0xfe220000, 0xa7580500, 0x25d10000, 0xf7600000, 0x893178da, 0x1fd4f860, 0x4ed0a315, 0xa123ff9f,
	0xfe220000, 0xa7580500, 0x25d10000, 0xf7600000, 0x893178da, 0x1fd4f860, 0x4ed0a315, 0xa123ff9f, 0xf2500000, 0xeebd0a00, 0x67a80000, 0xab8a0000, 0xba9b48c0, 0x0a56dd74, 0xdb73e86e, 0x1568ff0f,
	0x45180000, 0xa5b51700, 0xf96a0000, 0x3b480000, 0x1ecc142c, 0x231395d6, 0x16bca6b0, 0xdf33f4df, 0xb83d0000, 0x16710600, 0x379a0000, 0xf5b10000, 0x228161ac, 0xae48f145, 0x66241616, 0xc5c1eb3e,
	0xb83d0000, 0x16710600, 0x379a0000, 0xf5b10000, 0x228161ac, 0xae48f145, 0x66241616, 0xc5c1eb3e, 0xfd250000, 0xb3c41100, 0xcef00000, 0xcef90000, 0x3c4d7580, 0x8d5b6493, 0x7098b0a6, 0x1af21fe1,
	0x75a40000, 0xc28b2700, 0x94a40000, 0x90f50000, 0xfb7857e0, 0x49ce0bae, 0x1767c483, 0xaedf667e, 0xd1660000, 0x1bbc0300, 0x9eec0000, 0xf6940000, 0x03024527, 0xcf70fcf2, 0xb4431b17, 0x857f3c2b,
	0xd1660000, 0x1bbc0300, 0x9eec0000, 0xf6940000, 0x03024527, 0xcf70fcf2, 0xb4431b17, 0x857f3c2b, 0xa4c20000, 0xd9372400, 0x0a480000, 0x66610000, 0xf87a12c7, 0x86bef75c, 0xa324df94, 0x2ba05a55,
	0x75c90003, 0x0e10c000, 0xd1200000, 0xbaea0000, 0x8bc42f3e, 0x8758b757, 0xbb28761d, 0x00b72e2b, 0xeecf0001, 0x6f564000, 0xf33e0000, 0xa79e0000, 0xbdb57219, 0xb711ebc5, 0x4a3b40ba, 0xfeabf254,
	0xeecf0001, 0x6f564000, 0xf33e0000, 0xa79e0000, 0xbdb57219, 0xb711ebc5, 0x4a3b40ba, 0xfeabf254, 0x9b060002, 0x61468000, 0x221e0000, 0x1d740000, 0x36715d27, 0x30495c92, 0xf11336a7, 0xfe1cdc7f,
	0x86790000, 0x3f390002, 0xe19ae000, 0x98560000, 0x9565670e, 0x4e88c8ea, 0xd3dd4944, 0x161ddab9, 0x30b70000, 0xe5d00000, 0xf4f46000, 0x42c40000, 0x63b83d6a, 0x78ba9460, 0x21afa1ea, 0xb0a51834,
	0x30b70000, 0xe5d00000, 0xf4f46000, 0x42c40000, 0x63b83d6a, 0x78ba9460, 0x21afa1ea, 0xb0a51834, 0xb6ce0000, 0xdae90002, 0x156e8000, 0xda920000, 0xf6dd5a64, 0x36325c8a, 0xf272e8ae, 0xa6b8c28d,
	0x14190000, 0x23ca003c, 0x50df0000, 0x44b60000, 0x1b6c67b0, 0x3cf3ac75, 0x61e610b0, 0xdbcadb80, 0xe3430000, 0x3a4e0014, 0xf2c60000, 0xaa4e0000, 0xdb1e42a6, 0x256bbe15, 0x123db156, 0x3a4e99d7,
	0xe3430000, 0x3a4e0014, 0xf2c60000, 0xaa4e0000, 0xdb1e42a6, 0x256bbe15, 0x123db156, 0x3a4e99d7, 0xf75a0000, 0x19840028, 0xa2190000, 0xeef80000, 0xc0722516, 0x19981260, 0x73dba1e6, 0xe1844257,
	0x54500000, 0x0671005c, 0x25ae0000, 0x6a1e0000, 0x2ea54edf, 0x664e8512, 0xbfba18c3, 0x7e715d17, 0xbc8d0000, 0xfc3b0018, 0x19830000, 0xd10b0000, 0xae1878c4, 0x42a69856, 0x0012da37, 0x2c3b504e,
	0xbc8d0000, 0xfc3b0018, 0x19830000, 0xd10b0000, 0xae1878c4, 0x42a69856, 0x0012da37, 0x2c3b504e, 0xe8dd0000, 0xfa4a0044, 0x3c2d0000, 0xbb150000, 0x80bd361b, 0x24e81d44, 0xbfa8c2f4, 0x524a0d59,
	0x69510000, 0xd4e1009c, 0xc3230000, 0xac2f0000, 0xe4950bae, 0xcea415dc, 0x87ec287c, 0xbce1a3ce, 0xc6730000, 0xaf8d000c, 0xa4c10000, 0x218d0000, 0x23111587, 0x7913512f, 0x1d28ac88, 0x378dd173,
	0xc6730000, 0xaf8d000c, 0xa4c10000, 0x218d0000, 0x23111587, 0x7913512f, 0x1d28ac88, 0x378dd173, 0xaf220000, 0x7b6c0090, 0x67e20000, 0x8da20000, 0xc7841e29, 0xb7b744f3, 0x9ac484f4, 0x8b6c72bd,
	0xcc140000, 0xa5630000, 0x5ab90780, 0x3b500000, 0x4bd013ff, 0x879b3418, 0x694348c1, 0xca5a87fe, 0x819e0000, 0xec570000, 0x66320280, 0x95f30000, 0x5da92802, 0x48f43cbc, 0xe65aa22d, 0x8e67b7fa,
	0x819e0000, 0xec570000, 0x66320280, 0x95f30000, 0x5da92802, 0x48f43cbc, 0xe65aa22d, 0x8e67b7fa, 0x4d8a0000, 0x49340000, 0x3c8b0500, 0xaea30000, 0x16793bfd, 0xcf6f08a4, 0x8f19eaec, 0x443d3004,
	0x78230000, 0x12fc0000, 0xa93a0b80, 0x90a50000, 0x713e2879, 0x7ee98924, 0xf08ca062, 0x636f8bab, 0x02af0000, 0xb7280000, 0xba1c0300, 0x56980000, 0xba8d45d3, 0x8048c667, 0xa95c149a, 0xf4f6ea7b,
	0x02af0000, 0xb7280000, 0xba1c0300, 0x56980000, 0xba8d45d3, 0x8048c667, 0xa95c149a, 0xf4f6ea7b, 0x7a8c0000, 0xa5d40000, 0x13260880, 0xc63d0000, 0xcbb36daa, 0xfea14f43, 0x59d0b4f8, 0x979961d0,
	0xac480000, 0x1ba60000, 0x45fb1380, 0x03430000, 0x5a85316a, 0x1fb250b6, 0xfe72c7fe, 0x91e478f6, 0x1e4e0000, 0xdecf0000, 0x6df80180, 0x77240000, 0xec47079e, 0xf4a0694e, 0xcda31812, 0x98aa496e,
	0x1e4e0000, 0xdecf0000, 0x6df80180, 0x77240000, 0xec47079e, 0xf4a0694e, 0xcda31812, 0x98aa496e, 0xb2060000, 0xc5690000, 0x28031200, 0x74670000, 0xb6c236f4, 0xeb1239f8, 0x33d1dfec, 0x094e3198,
	0xaec30000, 0x9c4f0001, 0x79d1e000, 0x2c150000, 0x45cc75b3, 0x6650b736, 0xab92f78f, 0xa312567b, 0xdb250000, 0x09290000, 0x49aac000, 0x81e10000, 0xcafe6b59, 0x42793431, 0x43566b76, 0xe86cba2e,
	0xdb250000, 0x09290000, 0x49aac000, 0x81e10000, 0xcafe6b59, 0x42793431, 0x43566b76, 0xe86cba2e, 0x75e60000, 0x95660001, 0x307b2000, 0xadf40000, 0x8f321eea, 0x24298307, 0xe8c49cf9, 0x4b7eec55,
	0x58430000, 0x807e0000, 0x78330001, 0xc66b3800, 0xe7375cdc, 0x79ad3fdd, 0xac73fe6f, 0x3a4479b1, 0x1d5a0000, 0x2b720000, 0x488d0000, 0xaf611800, 0x25cb2ec5, 0xc879bfd0, 0x81a20429, 0x1e7536a6,
	0x1d5a0000, 0x2b720000, 0x488d0000, 0xaf611800, 0x25cb2ec5, 0xc879bfd0, 0x81a20429, 0x1e7536a6, 0x45190000, 0xab0c0000, 0x30be0001, 0x690a2000, 0xc2fc7219, 0xb1d4800d, 0x2dd1fa46, 0x24314f17,
	0xa53b0000, 0x14260000, 0x4e30001e, 0x7cae0000, 0x8f9e0dd5, 0x78dfaa3d, 0xf73168d8, 0x0b1b4946, 0x07ed0000, 0xb2500000, 0x8774000a, 0x970d0000, 0x437223ae, 0x48c76ea4, 0xf4786222, 0x9075b1ce,
	0x07ed0000, 0xb2500000, 0x8774000a, 0x970d0000, 0x437223ae, 0x48c76ea4, 0xf4786222, 0x9075b1ce, 0xa2d60000, 0xa6760000, 0xc9440014, 0xeba30000, 0xccec2e7b, 0x3018c499, 0x03490afa, 0x9b6ef888,
	0x88980000, 0x1f940000, 0x7fcf002e, 0xfb4e0000, 0xf158079a, 0x61ae9167, 0xa895706c, 0xe6107494, 0x0bc20000, 0xdb630000, 0x7e88000c, 0x15860000, 0x91fd48f3, 0x7581bb43, 0xf460449e, 0xd8b61463,
	0x0bc20000, 0xdb630000, 0x7e88000c, 0x15860000, 0x91fd48f3, 0x7581bb43, 0xf460449e, 0xd8b61463, 0x835a0000, 0xc4f70000, 0x01470022, 0xeec80000, 0x60a54f69, 0x142f2a24, 0x5cf534f2, 0x3ea660f7,
	0x52500000, 0x29540000, 0x6a61004e, 0xf0ff0000, 0x9a317eec, 0x452341ce, 0xcf568fe5, 0x5303130f, 0x538d0000, 0xa9fc0000, 0x9ef70006, 0x56ff0000, 0x0ae4004e, 0x92c5cdf9, 0xa9444018, 0x7f975691,
	0x538d0000, 0xa9fc0000, 0x9ef70006, 0x56ff0000, 0x0ae4004e, 0x92c5cdf9, 0xa9444018, 0x7f975691, 0x01dd0000, 0x80a80000, 0xf4960048, 0xa6000000, 0x90d57ea2, 0xd7e68c37, 0x6612cffd, 0x2c94459e,
	0xe6280000, 0x4c4b0000, 0xa8550000, 0xd3d002e0, 0xd86130b8, 0x98a7b0da, 0x289506b4, 0xd75a4897, 0xf0c50000, 0x59230000, 0x45820000, 0xe18d00c0, 0x3b6d0631, 0xc2ed5699, 0xcbe0fe1c, 0x56a7b19f,
	0xf0c50000, 0x59230000, 0x45820000, 0xe18d00c0, 0x3b6d0631, 0xc2ed5699, 0xcbe0fe1c, 0x56a7b19f, 0x16ed0000, 0x15680000, 0xedd70000, 0x325d0220, 0xe30c3689, 0x5a4ae643, 0xe375f8a8, 0x81fdf908,
	0xb4310000, 0x77330000, 0xb15d0000, 0x7fd004e0, 0x78a26138, 0xd116c35d, 0xd256d489, 0x4e6f74de, 0xe3060000, 0xbdc10000, 0x87130000, 0xbff20060, 0x2eba0a1a, 0x8db53751, 0x73c5ab06, 0x5bd61539,
	0xe3060000, 0xbdc10000, 0x87130000, 0xbff20060, 0x2eba0a1a, 0x8db53751, 0x73c5ab06, 0x5bd61539, 0x57370000, 0xcaf20000, 0x364e0000, 0xc0220480, 0x56186b22, 0x5ca3f40c, 0xa1937f8f, 0x15b961e7,
	0x02f20000, 0xa2810000, 0x873f0000, 0xe36c7800, 0x1e1d74ef, 0x073d2bd6, 0xc4c23237, 0x7f32259e, 0xbadd0000, 0x13ad0000, 0xb7e70000, 0xf7282800, 0xdf45144d, 0x361ac33a, 0xea5a8d14, 0x2a2c18f0,
	0xbadd0000, 0x13ad0000, 0xb7e70000, 0xf7282800, 0xdf45144d, 0x361ac33a, 0xea5a8d14, 0x2a2c18f0, 0xb82f0000, 0xb12c0000, 0x30d80000, 0x14445000, 0xc15860a2, 0x3127e8ec, 0x2e98bf23, 0x551e3d6e,
	0x1e6c0000, 0xc4420000, 0x8a2e0000, 0xbcb6b800, 0x2c4413b6, 0x8bfdd3da, 0x6a0c1bc8, 0xb99dc2eb, 0x92560000, 0x1eda0000, 0xea510000, 0xe8b13000, 0xa93556a5, 0xebfb6199, 0xb15c2254, 0x33c5244f,
	0x92560000, 0x1eda0000, 0xea510000, 0xe8b13000, 0xa93556a5, 0xebfb6199, 0xb15c2254, 0x33c5244f, 0x8c3a0000, 0xda980000, 0x607f0000, 0x54078800, 0x85714513, 0x6006b243, 0xdb50399c, 0x8a58e6a4,
	0x033d0000, 0x08b30000, 0xf33a0000, 0x3ac20007, 0x51298a50, 0x6b6e661f, 0x0ea5cfe3, 0xe6da7ffe, 0xa8da0000, 0x96be0000, 0x5c1d0000, 0x07da0002, 0x7d669583, 0x1f98708a, 0xbb668808, 0xda878000,
	0xa8da0000, 0x96be0000, 0x5c1d0000, 0x07da0002, 0x7d669583, 0x1f98708a, 0xbb668808, 0xda878000, 0xabe70000, 0x9e0d0000, 0xaf270000, 0x3d180005, 0x2c4f1fd3, 0x74f61695, 0xb5c347eb, 0x3c5dfffe,
	0x01930000, 0xe7820000, 0xedfb0000, 0xcf0c000b, 0x8dd08d58, 0xbca3b42e, 0x063661e1, 0x536f9e7b, 0x92280000, 0xdc850000, 0x57fa0000, 0x56dc0003, 0xbae92316, 0x5aefa30c, 0x90cef752, 0x7b1675d7,
	0x92280000, 0xdc850000, 0x57fa0000, 0x56dc0003, 0xbae92316, 0x5aefa30c, 0x90cef752, 0x7b1675d7, 0x93bb0000, 0x3b070000, 0xba010000, 0x99d00008, 0x3739ae4e, 0xe64c1722, 0x96f896b3, 0x2879ebac,
	0x5fa80000, 0x56030000, 0x43ae0000, 0x64f30013, 0x257e86bf, 0x1311944e, 0x541e95bf, 0x8ea4db69, 0x00440000, 0x7f480000, 0xda7c0000, 0x2a230001, 0x3badc9cc, 0xa9b69c87, 0x030a9e60, 0xbe0a679e,
	0x00440000, 0x7f480000, 0xda7c0000, 0x2a230001, 0x3badc9cc, 0xa9b69c87, 0x030a9e60, 0xbe0a679e, 0x5fec0000, 0x294b0000, 0x99d20000, 0x4ed00012, 0x1ed34f73, 0xbaa708c9, 0x57140bdf, 0x30aebcf7,
	0xee930000, 0xd6070000, 0x92c10000, 0x2b9801e0, 0x9451287c, 0x3b6cfb57, 0x45312374, 0x201f6a64, 0x7b280000, 0x57420000, 0xa9e50000, 0x634300a0, 0x9edb442f, 0x6d9995bb, 0x27f83b03, 0xc7ff60f0,
	0x7b280000, 0x57420000, 0xa9e50000, 0x634300a0, 0x9edb442f, 0x6d9995bb, 0x27f83b03, 0xc7ff60f0, 0x95bb0000, 0x81450000, 0x3b240000, 0x48db0140, 0x0a8a6c53, 0x56f56eec, 0x62c91877, 0xe7e00a94
};

#define SBOX(a, b, c, d) { \
		uint32_t t; \
		t =(a); \
		a =(a & c) ^ d; \
		c =(c ^ b) ^ a; \
		d =(d | t) ^ b; \
		b = d; \
		d =((d | (t ^ c)) ^ a); \
		a&= b; \
		t^=(c ^ a); \
		b = b ^ d ^ t; \
		(a) = (c); \
		(c) = (b); \
		(b) = (d); \
		(d) = (~t); \
	}

#define HAMSI_L(a, b, c, d) { \
		(a) = ROTL32(a, 13); \
		(c) = ROTL32(c, 3); \
		(b) ^= (a) ^ (c); \
		(d) ^= (c) ^ ((a) << 3); \
		(b) = ROTL32(b, 1); \
		(d) = ROTL32(d, 7); \
		(a) = ROTL32(a ^ b ^ d, 5); \
		(c) = ROTL32(c ^ d ^ (b<<7), 22); \
			}

#define ROUND_BIG(rc, alpha) { \
		m[ 0] ^= alpha[ 0]; \
		c[ 4] ^= alpha[ 8]; \
		m[ 8] ^= alpha[16]; \
		c[12] ^= alpha[24]; \
		m[ 1] ^= alpha[ 1] ^ (rc); \
		c[ 5] ^= alpha[ 9]; \
		m[ 9] ^= alpha[17]; \
		c[13] ^= alpha[25]; \
		c[ 0] ^= alpha[ 2]; \
		m[ 4] ^= alpha[10]; \
		c[ 8] ^= alpha[18]; \
		m[12] ^= alpha[26]; \
		c[ 1] ^= alpha[ 3]; \
		m[ 5] ^= alpha[11]; \
		c[ 9] ^= alpha[19]; \
		m[13] ^= alpha[27]; \
		m[ 2] ^= alpha[ 4]; \
		c[ 6] ^= alpha[12]; \
		m[10] ^= alpha[20]; \
		c[14] ^= alpha[28]; \
		m[ 3] ^= alpha[ 5]; \
		c[ 7] ^= alpha[13]; \
		m[11] ^= alpha[21]; \
		c[15] ^= alpha[29]; \
		c[ 2] ^= alpha[ 6]; \
		m[ 6] ^= alpha[14]; \
		c[10] ^= alpha[22]; \
		m[14] ^= alpha[30]; \
		c[ 3] ^= alpha[ 7]; \
		m[ 7] ^= alpha[15]; \
		c[11] ^= alpha[23]; \
		m[15] ^= alpha[31]; \
		SBOX(m[ 0], c[ 4], m[ 8], c[12]); \
		SBOX(m[ 1], c[ 5], m[ 9], c[13]); \
		SBOX(c[ 0], m[ 4], c[ 8], m[12]); \
		SBOX(c[ 1], m[ 5], c[ 9], m[13]); \
		HAMSI_L(m[ 0], c[ 5], c[ 8], m[13]); \
		SBOX(m[ 2], c[ 6], m[10], c[14]); \
		HAMSI_L(m[ 1], m[ 4], c[ 9], c[14]); \
		SBOX(m[ 3], c[ 7], m[11], c[15]); \
		HAMSI_L(c[ 0], m[ 5], m[10], c[15]); \
		SBOX(c[ 2], m[ 6], c[10], m[14]); \
		HAMSI_L(c[ 1], c[ 6], m[11], m[14]); \
		SBOX(c[ 3], m[ 7], c[11], m[15]); \
		HAMSI_L(m[ 2], c[ 7], c[10], m[15]); \
		HAMSI_L(m[ 3], m[ 6], c[11], c[12]); \
		HAMSI_L(c[ 2], m[ 7], m[ 8], c[13]); \
		HAMSI_L(c[ 3], c[ 4], m[ 9], m[12]); \
		HAMSI_L(m[ 0], c[ 0], m[ 3], c[ 3]); \
		HAMSI_L(m[ 8], c[ 9], m[11], c[10]); \
		HAMSI_L(c[ 5], m[ 5], c[ 6], m[ 6]); \
		HAMSI_L(c[13], m[12], c[14], m[15]); \
			}



//------FUGUE MACROS--------------------------------------------------
static __constant__ const uint32_t c_S[16] = {
	0x8807a57e, 0xe616af75, 0xc5d3e4db, 0xac9ab027,
	0xd915f117, 0xb6eecc54, 0x06e8020b, 0x4a92efd1,
	0xaac6e2c9, 0xddb21398, 0xcae65838, 0x437f203f,
	0x25ea78e7, 0x951fddd6, 0xda6ed11d, 0xe13e3567
};

static __device__ uint32_t mixtab0[256] = {
	0x63633297, 0x7c7c6feb, 0x77775ec7, 0x7b7b7af7, 0xf2f2e8e5, 0x6b6b0ab7, 0x6f6f16a7, 0xc5c56d39, 0x303090c0, 0x01010704, 0x67672e87, 0x2b2bd1ac, 0xfefeccd5, 0xd7d71371, 0xabab7c9a,
	0x767659c3, 0xcaca4005, 0x8282a33e, 0xc9c94909, 0x7d7d68ef, 0xfafad0c5, 0x5959947f, 0x4747ce07, 0xf0f0e6ed, 0xadad6e82, 0xd4d41a7d, 0xa2a243be, 0xafaf608a, 0x9c9cf946, 0xa4a451a6,
	0x727245d3, 0xc0c0762d, 0xb7b728ea, 0xfdfdc5d9, 0x9393d47a, 0x2626f298, 0x363682d8, 0x3f3fbdfc, 0xf7f7f3f1, 0xcccc521d, 0x34348cd0, 0xa5a556a2, 0xe5e58db9, 0xf1f1e1e9, 0x71714cdf,
	0xd8d83e4d, 0x313197c4, 0x15156b54, 0x04041c10, 0xc7c76331, 0x2323e98c, 0xc3c37f21, 0x18184860, 0x9696cf6e, 0x05051b14, 0x9a9aeb5e, 0x0707151c, 0x12127e48, 0x8080ad36, 0xe2e298a5,
	0xebeba781, 0x2727f59c, 0xb2b233fe, 0x757550cf, 0x09093f24, 0x8383a43a, 0x2c2cc4b0, 0x1a1a4668, 0x1b1b416c, 0x6e6e11a3, 0x5a5a9d73, 0xa0a04db6, 0x5252a553, 0x3b3ba1ec, 0xd6d61475,
	0xb3b334fa, 0x2929dfa4, 0xe3e39fa1, 0x2f2fcdbc, 0x8484b126, 0x5353a257, 0xd1d10169, 0x00000000, 0xededb599, 0x2020e080, 0xfcfcc2dd, 0xb1b13af2, 0x5b5b9a77, 0x6a6a0db3, 0xcbcb4701,
	0xbebe17ce, 0x3939afe4, 0x4a4aed33, 0x4c4cff2b, 0x5858937b, 0xcfcf5b11, 0xd0d0066d, 0xefefbb91, 0xaaaa7b9e, 0xfbfbd7c1, 0x4343d217, 0x4d4df82f, 0x333399cc, 0x8585b622, 0x4545c00f,
	0xf9f9d9c9, 0x02020e08, 0x7f7f66e7, 0x5050ab5b, 0x3c3cb4f0, 0x9f9ff04a, 0xa8a87596, 0x5151ac5f, 0xa3a344ba, 0x4040db1b, 0x8f8f800a, 0x9292d37e, 0x9d9dfe42, 0x3838a8e0, 0xf5f5fdf9,
	0xbcbc19c6, 0xb6b62fee, 0xdada3045, 0x2121e784, 0x10107040, 0xffffcbd1, 0xf3f3efe1, 0xd2d20865, 0xcdcd5519, 0x0c0c2430, 0x1313794c, 0xececb29d, 0x5f5f8667, 0x9797c86a, 0x4444c70b,
	0x1717655c, 0xc4c46a3d, 0xa7a758aa, 0x7e7e61e3, 0x3d3db3f4, 0x6464278b, 0x5d5d886f, 0x19194f64, 0x737342d7, 0x60603b9b, 0x8181aa32, 0x4f4ff627, 0xdcdc225d, 0x2222ee88, 0x2a2ad6a8,
	0x9090dd76, 0x88889516, 0x4646c903, 0xeeeebc95, 0xb8b805d6, 0x14146c50, 0xdede2c55, 0x5e5e8163, 0x0b0b312c, 0xdbdb3741, 0xe0e096ad, 0x32329ec8, 0x3a3aa6e8, 0x0a0a3628, 0x4949e43f,
	0x06061218, 0x2424fc90, 0x5c5c8f6b, 0xc2c27825, 0xd3d30f61, 0xacac6986, 0x62623593, 0x9191da72, 0x9595c662, 0xe4e48abd, 0x797974ff, 0xe7e783b1, 0xc8c84e0d, 0x373785dc, 0x6d6d18af,
	0x8d8d8e02, 0xd5d51d79, 0x4e4ef123, 0xa9a97292, 0x6c6c1fab, 0x5656b943, 0xf4f4fafd, 0xeaeaa085, 0x6565208f, 0x7a7a7df3, 0xaeae678e, 0x08083820, 0xbaba0bde, 0x787873fb, 0x2525fb94,
	0x2e2ecab8, 0x1c1c5470, 0xa6a65fae, 0xb4b421e6, 0xc6c66435, 0xe8e8ae8d, 0xdddd2559, 0x747457cb, 0x1f1f5d7c, 0x4b4bea37, 0xbdbd1ec2, 0x8b8b9c1a, 0x8a8a9b1e, 0x70704bdb, 0x3e3ebaf8,
	0xb5b526e2, 0x66662983, 0x4848e33b, 0x0303090c, 0xf6f6f4f5, 0x0e0e2a38, 0x61613c9f, 0x35358bd4, 0x5757be47, 0xb9b902d2, 0x8686bf2e, 0xc1c17129, 0x1d1d5374, 0x9e9ef74e, 0xe1e191a9,
	0xf8f8decd, 0x9898e556, 0x11117744, 0x696904bf, 0xd9d93949, 0x8e8e870e, 0x9494c166, 0x9b9bec5a, 0x1e1e5a78, 0x8787b82a, 0xe9e9a989, 0xcece5c15, 0x5555b04f, 0x2828d8a0, 0xdfdf2b51,
	0x8c8c8906, 0xa1a14ab2, 0x89899212, 0x0d0d2334, 0xbfbf10ca, 0xe6e684b5, 0x4242d513, 0x686803bb, 0x4141dc1f, 0x9999e252, 0x2d2dc3b4, 0x0f0f2d3c, 0xb0b03df6, 0x5454b74b, 0xbbbb0cda,
	0x16166258
};


#define mixtab0(x) shared[0][x]
#define mixtab1(x) shared[1][x]
#define mixtab2(x) shared[2][x]
#define mixtab3(x) shared[3][x]

#define TIX4(q, x00, x01, x04, x07, x08, x22, x24, x27, x30) { \
		x22 ^= x00; \
		x00 = (q); \
		x08 ^= (q); \
		x01 ^= x24; \
		x04 ^= x27; \
		x07 ^= x30; \
	}

#define CMIX36(x00, x01, x02, x04, x05, x06, x18, x19, x20) { \
		x00 ^= x04; \
		x01 ^= x05; \
		x02 ^= x06; \
		x18 ^= x04; \
		x19 ^= x05; \
		x20 ^= x06; \
				}

__device__ __forceinline__
static void SMIX_LDG(const uint32_t shared[4][256], uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3){
	uint32_t c0 = __ldg(&mixtab0[__byte_perm(x0, 0, 0x4443)]);
	uint32_t r1 = mixtab1(__byte_perm(x0, 0, 0x4442));
	uint32_t r2 = mixtab2(__byte_perm(x0, 0, 0x4441));
	uint32_t r3 = mixtab3(__byte_perm(x0, 0, 0x4440));
	c0 = c0 ^ r1 ^ r2 ^ r3;
	uint32_t r0 = mixtab0(__byte_perm(x1, 0, 0x4443));
	uint32_t c1 = r0 ^ mixtab1(__byte_perm(x1, 0, 0x4442));
	uint32_t tmp = mixtab2(__byte_perm(x1, 0, 0x4441));
	c1 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x1, 0, 0x4440));
	c1 ^= tmp;
	r3 ^= tmp;
	uint32_t c2 = __ldg(&mixtab0[__byte_perm(x2, 0, 0x4443)]);
	r0 ^= c2;
	tmp = mixtab1(__byte_perm(x2, 0, 0x4442));
	c2 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x2, 0, 0x4441));
	c2 ^= tmp;
	tmp = mixtab3(__byte_perm(x2, 0, 0x4440));
	c2 ^= tmp;
	r3 ^= tmp;
	uint32_t c3 = __ldg(&mixtab0[__byte_perm(x3, 0, 0x4443)]);
	r0 ^= c3;
	tmp = mixtab1(__byte_perm(x3, 0, 0x4442));
	c3 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x3, 0, 0x4441));
	c3 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x3, 0, 0x4440));
	c3 ^= tmp;
	x0 = ((c0 ^ (r0 << 0)) & 0xFF000000) | ((c1 ^ (r1 << 0)) & 0x00FF0000) | ((c2 ^ (r2 << 0)) & 0x0000FF00) | ((c3 ^ (r3 << 0)) & 0x000000FF);
	x1 = ((c1 ^ (r0 << 8)) & 0xFF000000) | ((c2 ^ (r1 << 8)) & 0x00FF0000) | ((c3 ^ (r2 << 8)) & 0x0000FF00) | ((c0 ^ (r3 >> 24)) & 0x000000FF);
	x2 = ((c2 ^ (r0 << 16)) & 0xFF000000) | ((c3 ^ (r1 << 16)) & 0x00FF0000) | ((c0 ^ (r2 >> 16)) & 0x0000FF00) | ((c1 ^ (r3 >> 16)) & 0x000000FF);
	x3 = ((c3 ^ (r0 << 24)) & 0xFF000000) | ((c0 ^ (r1 >> 8)) & 0x00FF0000) | ((c1 ^ (r2 >> 8)) & 0x0000FF00) | ((c2 ^ (r3 >> 8)) & 0x000000FF);
}

__device__ __forceinline__
static void SMIX(const uint32_t shared[4][256], uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3){
	uint32_t c0 = mixtab0(__byte_perm(x0, 0, 0x4443));
	uint32_t r1 = mixtab1(__byte_perm(x0, 0, 0x4442));
	uint32_t r2 = mixtab2(__byte_perm(x0, 0, 0x4441));
	uint32_t r3 = mixtab3(__byte_perm(x0, 0, 0x4440));
	c0 = c0 ^ r1 ^ r2 ^ r3;
	uint32_t r0 = mixtab0(__byte_perm(x1, 0, 0x4443));
	uint32_t c1 = r0 ^ mixtab1(__byte_perm(x1, 0, 0x4442));
	uint32_t tmp = mixtab2(__byte_perm(x1, 0, 0x4441));
	c1 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x1, 0, 0x4440));
	c1 ^= tmp;
	r3 ^= tmp;
	uint32_t c2 = mixtab0(__byte_perm(x2, 0, 0x4443));
	r0 ^= c2;
	tmp = mixtab1(__byte_perm(x2, 0, 0x4442));
	c2 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x2, 0, 0x4441));
	c2 ^= tmp;
	tmp = mixtab3(__byte_perm(x2, 0, 0x4440));
	c2 ^= tmp;
	r3 ^= tmp;
	uint32_t c3 = mixtab0(__byte_perm(x3, 0, 0x4443));
	r0 ^= c3;
	tmp = mixtab1(__byte_perm(x3, 0, 0x4442));
	c3 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x3, 0, 0x4441));
	c3 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x3, 0, 0x4440));
	c3 ^= tmp;
	x0 = ((c0 ^ (r0 << 0)) & 0xFF000000) | ((c1 ^ (r1 << 0)) & 0x00FF0000) | ((c2 ^ (r2 << 0)) & 0x0000FF00) | ((c3 ^ (r3 << 0)) & 0x000000FF);
	x1 = ((c1 ^ (r0 << 8)) & 0xFF000000) | ((c2 ^ (r1 << 8)) & 0x00FF0000) | ((c3 ^ (r2 << 8)) & 0x0000FF00) | ((c0 ^ (r3 >> 24)) & 0x000000FF);
	x2 = ((c2 ^ (r0 << 16)) & 0xFF000000) | ((c3 ^ (r1 << 16)) & 0x00FF0000) | ((c0 ^ (r2 >> 16)) & 0x0000FF00) | ((c1 ^ (r3 >> 16)) & 0x000000FF);
	x3 = ((c3 ^ (r0 << 24)) & 0xFF000000) | ((c0 ^ (r1 >> 8)) & 0x00FF0000) | ((c1 ^ (r2 >> 8)) & 0x0000FF00) | ((c2 ^ (r3 >> 8)) & 0x000000FF);
}

#define mROR3 { \
	B[ 6] = S[33], B[ 7] = S[34], B[ 8] = S[35]; \
	S[35] = S[32]; S[34] = S[31]; S[33] = S[30]; S[32] = S[29]; S[31] = S[28]; S[30] = S[27]; S[29] = S[26]; S[28] = S[25]; S[27] = S[24]; \
	S[26] = S[23]; S[25] = S[22]; S[24] = S[21]; S[23] = S[20]; S[22] = S[19]; S[21] = S[18]; S[20] = S[17]; S[19] = S[16]; S[18] = S[15]; \
	S[17] = S[14]; S[16] = S[13]; S[15] = S[12]; S[14] = S[11]; S[13] = S[10]; S[12] = S[ 9]; S[11] = S[ 8]; S[10] = S[ 7]; S[ 9] = S[ 6]; \
	S[ 8] = S[ 5]; S[ 7] = S[ 4]; S[ 6] = S[ 3]; S[ 5] = S[ 2]; S[ 4] = S[ 1]; S[ 3] = S[ 0]; S[ 2] = B[ 8]; S[ 1] = B[ 7]; S[ 0] = B[ 6]; \
	}

#define mROR8 { \
	B[ 1] = S[28]; B[ 2] = S[29]; B[ 3] = S[30]; B[ 4] = S[31]; B[ 5] = S[32]; B[ 6] = S[33]; B[ 7] = S[34]; B[ 8] = S[35]; \
	S[35] = S[27]; S[34] = S[26]; S[33] = S[25]; S[32] = S[24]; S[31] = S[23]; S[30] = S[22]; S[29] = S[21]; S[28] = S[20]; S[27] = S[19]; \
	S[26] = S[18]; S[25] = S[17]; S[24] = S[16]; S[23] = S[15]; S[22] = S[14]; S[21] = S[13]; S[20] = S[12]; S[19] = S[11]; S[18] = S[10]; \
	S[17] = S[ 9]; S[16] = S[ 8]; S[15] = S[ 7]; S[14] = S[ 6]; S[13] = S[ 5]; S[12] = S[ 4]; S[11] = S[ 3]; S[10] = S[ 2]; S[ 9] = S[ 1]; \
	S[ 8] = S[ 0]; S[ 7] = B[ 8]; S[ 6] = B[ 7]; S[ 5] = B[ 6]; S[ 4] = B[ 5]; S[ 3] = B[ 4]; S[ 2] = B[ 3]; S[ 1] = B[ 2]; S[ 0] = B[ 1]; \
	}

#define mROR9 { \
	B[ 0] = S[27]; B[ 1] = S[28]; B[ 2] = S[29]; B[ 3] = S[30]; B[ 4] = S[31]; B[ 5] = S[32]; B[ 6] = S[33]; B[ 7] = S[34]; B[ 8] = S[35]; \
	S[35] = S[26]; S[34] = S[25]; S[33] = S[24]; S[32] = S[23]; S[31] = S[22]; S[30] = S[21]; S[29] = S[20]; S[28] = S[19]; S[27] = S[18]; \
	S[26] = S[17]; S[25] = S[16]; S[24] = S[15]; S[23] = S[14]; S[22] = S[13]; S[21] = S[12]; S[20] = S[11]; S[19] = S[10]; S[18] = S[ 9]; \
	S[17] = S[ 8]; S[16] = S[ 7]; S[15] = S[ 6]; S[14] = S[ 5]; S[13] = S[ 4]; S[12] = S[ 3]; S[11] = S[ 2]; S[10] = S[ 1]; S[ 9] = S[ 0]; \
	S[ 8] = B[ 8]; S[ 7] = B[ 7]; S[ 6] = B[ 6]; S[ 5] = B[ 5]; S[ 4] = B[ 4]; S[ 3] = B[ 3]; S[ 2] = B[ 2]; S[ 1] = B[ 1]; S[ 0] = B[ 0]; \
	}

#define FUGUE512_3(x, y, z) {  \
        TIX4(x, S[ 0], S[ 1], S[ 4], S[ 7], S[ 8], S[22], S[24], S[27], S[30]); \
        CMIX36(S[33], S[34], S[35], S[ 1], S[ 2], S[ 3], S[15], S[16], S[17]); \
        SMIX(shared, S[33], S[34], S[35], S[ 0]); \
        CMIX36(S[30], S[31], S[32], S[34], S[35], S[ 0], S[12], S[13], S[14]); \
        SMIX(shared, S[30], S[31], S[32], S[33]); \
        CMIX36(S[27], S[28], S[29], S[31], S[32], S[33], S[ 9], S[10], S[11]); \
        SMIX(shared, S[27], S[28], S[29], S[30]); \
        CMIX36(S[24], S[25], S[26], S[28], S[29], S[30], S[ 6], S[ 7], S[ 8]); \
        SMIX_LDG(shared, S[24], S[25], S[26], S[27]); \
        \
        TIX4(y, S[24], S[25], S[28], S[31], S[32], S[10], S[12], S[15], S[18]); \
        CMIX36(S[21], S[22], S[23], S[25], S[26], S[27], S[ 3], S[ 4], S[ 5]); \
        SMIX(shared, S[21], S[22], S[23], S[24]); \
        CMIX36(S[18], S[19], S[20], S[22], S[23], S[24], S[ 0], S[ 1], S[ 2]); \
        SMIX_LDG(shared, S[18], S[19], S[20], S[21]); \
        CMIX36(S[15], S[16], S[17], S[19], S[20], S[21], S[33], S[34], S[35]); \
        SMIX(shared, S[15], S[16], S[17], S[18]); \
        CMIX36(S[12], S[13], S[14], S[16], S[17], S[18], S[30], S[31], S[32]); \
        SMIX_LDG(shared, S[12], S[13], S[14], S[15]); \
        \
        TIX4(z, S[12], S[13], S[16], S[19], S[20], S[34], S[ 0], S[ 3], S[ 6]); \
        CMIX36(S[ 9], S[10], S[11], S[13], S[14], S[15], S[27], S[28], S[29]); \
        SMIX(shared, S[ 9], S[10], S[11], S[12]); \
        CMIX36(S[ 6], S[ 7], S[ 8], S[10], S[11], S[12], S[24], S[25], S[26]); \
        SMIX_LDG(shared, S[ 6], S[ 7], S[ 8], S[ 9]); \
        CMIX36(S[ 3], S[ 4], S[ 5], S[ 7], S[ 8], S[ 9], S[21], S[22], S[23]); \
        SMIX_LDG(shared, S[ 3], S[ 4], S[ 5], S[ 6]); \
        CMIX36(S[ 0], S[ 1], S[ 2], S[ 4], S[ 5], S[ 6], S[18], S[19], S[20]); \
        SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]); \
	}

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#define TPB50_1 128
#define TPB50_2 128
#define TPB52_1 128
#define TPB52_2 128

static uint4 *d_temp4[MAX_GPUS];
#include "cuda_x11_simd512_func.cuh"

//ECHO MACROS--------------------------------
#define SHIFT_ROW1(a, b, c, d)   do { \
		tmp0 = W[a+0]; \
		W[a+0] = W[b+0]; \
		W[b+0] = W[c+0]; \
		W[c+0] = W[d+0]; \
		W[d+0] = tmp0; \
\
		tmp0 = W[a+1]; \
		W[a+1] = W[b+1]; \
		W[b+1] = W[c+1]; \
		W[c+1] = W[d+1]; \
		W[d+1] = tmp0; \
\
		tmp0 = W[a+2]; \
		W[a+2] = W[b+2]; \
		W[b+2] = W[c+2]; \
		W[c+2] = W[d+2]; \
		W[d+2] = tmp0; \
\
		tmp0 = W[a+3]; \
		W[a+3] = W[b+3]; \
		W[b+3] = W[c+3]; \
		W[c+3] = W[d+3]; \
		W[d+3] = tmp0; \
				} while (0)

#define SHIFT_ROW2(a, b, c, d)   do { \
		tmp0 = W[a+0]; \
		W[a+0] = W[c+0]; \
		W[c+0] = tmp0; \
\
		tmp0 = W[a+1]; \
		W[a+1] = W[c+1]; \
		W[c+1] = tmp0; \
\
		tmp0 = W[a+2]; \
		W[a+2] = W[c+2]; \
		W[c+2] = tmp0; \
\
		tmp0 = W[a+3]; \
		W[a+3] = W[c+3]; \
		W[c+3] = tmp0; \
\
		tmp0 = W[b+0]; \
		W[b+0] = W[d+0]; \
		W[d+0] = tmp0; \
\
		tmp0 = W[b+1]; \
		W[b+1] = W[d+1]; \
		W[d+1] = tmp0; \
\
		tmp0 = W[b+2]; \
		W[b+2] = W[d+2]; \
		W[d+2] = tmp0; \
\
		tmp0 = W[b+3]; \
		W[b+3] = W[d+3]; \
		W[d+3] = tmp0; \
		} while (0)

#define MIX_COLUMN1(ia, ib, ic, id, n)   do { \
		tmp0 = W[ia+n]; \
		unsigned int tmp1 = W[ic+n]; \
		unsigned int tmp2 = tmp0 ^ W[ib+n]; \
		unsigned int tmp3 = W[ib+n] ^ tmp1; \
		unsigned int tmp4 = tmp1 ^ W[id+n]; \
		unsigned int tmp5 = (((tmp2 & (0x80808080)) >> 7) * 27 ^ ((tmp2 & (0x7F7F7F7F)) << 1));\
		unsigned int tmp6 = (((tmp3 & (0x80808080)) >> 7) * 27 ^ ((tmp3 & (0x7F7F7F7F)) << 1));\
		unsigned int tmp7 = (((tmp4 & (0x80808080)) >> 7) * 27 ^ ((tmp4 & (0x7F7F7F7F)) << 1));\
		W[ia+n] = tmp5 ^ tmp3 ^ W[id+n]; \
		W[ib+n] = tmp6 ^ tmp0 ^ tmp4; \
		W[ic+n] = tmp7 ^ tmp2 ^ W[id+n]; \
		W[id+n] = tmp5^tmp6^tmp7^tmp2^tmp1; \
		} while (0)

#define MIX_COLUMN(a, b, c, d)   do { \
		MIX_COLUMN1(a, b, c, d, 0); \
		MIX_COLUMN1(a, b, c, d, 1); \
		MIX_COLUMN1(a, b, c, d, 2); \
		MIX_COLUMN1(a, b, c, d, 3); \
		} while (0)
//END OF ECHO MACROS-------------------------

__device__
static void echo_round_sp(const uint32_t sharedMemory[8 * 1024], uint32_t *W, uint32_t &k0){
	// Big Sub Words
#pragma unroll 16
	for (int idx = 0; idx < 16; idx++)
		AES_2ROUND_32(sharedMemory, W[(idx << 2) + 0], W[(idx << 2) + 1], W[(idx << 2) + 2], W[(idx << 2) + 3], k0);

	// Shift Rows
#pragma unroll 4
	for (int i = 0; i < 4; i++){
		uint32_t t[4];
		/// 1, 5, 9, 13
		t[0] = W[i + 4];
		t[1] = W[i + 8];
		t[2] = W[i + 24];
		t[3] = W[i + 60];
		W[i + 4] = W[i + 20];
		W[i + 8] = W[i + 40];
		W[i + 24] = W[i + 56];
		W[i + 60] = W[i + 44];

		W[i + 20] = W[i + 36];
		W[i + 40] = t[1];
		W[i + 56] = t[2];
		W[i + 44] = W[i + 28];

		W[i + 28] = W[i + 12];
		W[i + 12] = t[3];
		W[i + 36] = W[i + 52];
		W[i + 52] = t[0];
	}
	// Mix Columns
#pragma unroll 4
	for (int i = 0; i < 4; i++){ // Schleife über je 2*uint32_t
#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16){ // Schleife über die elemnte
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i + 12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = (bc^ a[3] ^ abx);
			W[idx + i + 4] = xor3(a[0], cd, bcx);
			W[idx + i + 8] = xor3(ab, a[3], cdx);
			W[idx + i + 12] = xor3(ab, a[2], xor3(abx, bcx, cdx));
		}
	}
}


__global__ __launch_bounds__(128,5)
static void x11_simd512_gpu_compress_64(uint32_t threads, uint32_t *g_hash,const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x)>>3;
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)
	uint32_t IV[32];
	if (thread < threads){

		uint32_t *Hash = &g_hash[thread<<4];
//		Compression1(Hash, thread, g_fft4, g_state);
		uint32_t A[32];

		*(uint2x4*)&IV[ 0] = *(uint2x4*)&c_IV_512[ 0];
		*(uint2x4*)&IV[ 8] = *(uint2x4*)&c_IV_512[ 8];
		*(uint2x4*)&IV[16] = *(uint2x4*)&c_IV_512[16];
		*(uint2x4*)&IV[24] = *(uint2x4*)&c_IV_512[24];

		*(uint2x4*)&A[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&A[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		#pragma unroll 16
		for(uint32_t i=0;i<16;i++)
			A[ i] = A[ i] ^ IV[ i];

		#pragma unroll 16
		for(uint32_t i=16;i<32;i++)
			A[ i] = IV[ i];

		Round8(A, thr_offset, g_fft4);
		
		STEP8_IF(&IV[ 0],32, 4,13,&A[ 0],&A[ 8],&A[16],&A[24]);
		STEP8_IF(&IV[ 8],33,13,10,&A[24],&A[ 0],&A[ 8],&A[16]);
		STEP8_IF(&IV[16],34,10,25,&A[16],&A[24],&A[ 0],&A[ 8]);
		STEP8_IF(&IV[24],35,25, 4,&A[ 8],&A[16],&A[24],&A[ 0]);

		#pragma unroll 32
		for(uint32_t i=0;i<32;i++){
			IV[ i] = A[ i];
		}
		
		A[ 0] ^= 512;

		Round8_0_final(A, 3,23,17,27);
		Round8_1_final(A,28,19,22, 7);
		Round8_2_final(A,29, 9,15, 5);
		Round8_3_final(A, 4,13,10,25);
		STEP8_IF(&IV[ 0],32, 4,13, &A[ 0], &A[ 8], &A[16], &A[24]);
		STEP8_IF(&IV[ 8],33,13,10, &A[24], &A[ 0], &A[ 8], &A[16]);
		STEP8_IF(&IV[16],34,10,25, &A[16], &A[24], &A[ 0], &A[ 8]);
		STEP8_IF(&IV[24],35,25, 4, &A[ 8], &A[16], &A[24], &A[ 0]);

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&A[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&A[ 8];
	}
}

__global__
__launch_bounds__(128, 5)
void x11_simd512_gpu_compress_64_pascal_final(uint32_t threads, uint32_t startnonce, uint32_t *g_hash, const uint4 *const __restrict__ g_fft4, uint32_t *d_resNonce, const uint64_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x)>>3;
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)
	uint32_t IV[32];
	if (thread < threads){

		uint32_t *Hash = &g_hash[thread << 4];
		//		Compression1(Hash, thread, g_fft4, g_state);
		uint32_t A[32];

		*(uint2x4*)&IV[0] = *(uint2x4*)&c_IV_512[0];
		*(uint2x4*)&IV[8] = *(uint2x4*)&c_IV_512[8];
		*(uint2x4*)&IV[16] = *(uint2x4*)&c_IV_512[16];
		*(uint2x4*)&IV[24] = *(uint2x4*)&c_IV_512[24];

		*(uint2x4*)&A[0] = __ldg4((uint2x4*)&Hash[0]);
		*(uint2x4*)&A[8] = __ldg4((uint2x4*)&Hash[8]);

#pragma unroll 16
		for (uint32_t i = 0; i<16; i++)
			A[i] = A[i] ^ IV[i];

#pragma unroll 16
		for (uint32_t i = 16; i<32; i++)
			A[i] = IV[i];

		Round8(A, thr_offset, g_fft4);

		STEP8_IF(&IV[0], 32, 4, 13, &A[0], &A[8], &A[16], &A[24]);
		STEP8_IF(&IV[8], 33, 13, 10, &A[24], &A[0], &A[8], &A[16]);
		STEP8_IF(&IV[16], 34, 10, 25, &A[16], &A[24], &A[0], &A[8]);
		STEP8_IF(&IV[24], 35, 25, 4, &A[8], &A[16], &A[24], &A[0]);

#pragma unroll 32
		for (uint32_t i = 0; i<32; i++){
			IV[i] = A[i];
		}

		A[0] ^= 512;

		Round8_0_final(A, 3, 23, 17, 27);
		Round8_1_final(A, 28, 19, 22, 7);
		Round8_2_final(A, 29, 9, 15, 5);
		Round8_3_final(A, 4, 13, 10, 25);
		STEP8_IF(&IV[0], 32, 4, 13, &A[0], &A[8], &A[16], &A[24]);
		STEP8_IF(&IV[8], 33, 13, 10, &A[24], &A[0], &A[8], &A[16]);
		STEP8_IF(&IV[16], 34, 10, 25, &A[16], &A[24], &A[0], &A[8]);
		STEP8_IF(&IV[24], 35, 25, 4, &A[8], &A[16], &A[24], &A[0]);

		//		*(uint2x4*)&Hash[0] = *(uint2x4*)&A[0];
		//		*(uint2x4*)&Hash[8] = *(uint2x4*)&A[8];

		//		*(uint64_t*)&Hash[6] = *(uint64_t*)&A[6];

		//		__syncthreads();

		uint64_t check = ((uint64_t*)A)[3];
		uint32_t nonce = thread + startnonce;
		if (check <= target)
		{
			uint32_t tmp = atomicExch(&d_resNonce[0], nonce);
			if (tmp != UINT32_MAX)
				if (tmp != d_resNonce[0] ) d_resNonce[1] = tmp;
		}


	}
}




__host__
int x11_simd512_cpu_init(int thr_id, uint32_t threads)
{
	cudaMalloc(&d_temp4[thr_id], 64*sizeof(uint4)*threads);

	// whirlpool
	uint64_t* table0 = NULL;
	table0 = (uint64_t*)plain_T0;
	cudaMemcpyToSymbol(InitVector_RC, plain_RC, 10 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(precomputed_round_key_64, plain_precomputed_round_key_64, 72 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(b0, table0, 256 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	uint64_t table7[256];
	for (int i = 0; i<256; i++){
		table7[i] = ROTR64(table0[i], 8);
	}

	return 0;
}

__host__
void x11_simd512_cpu_free(int thr_id){
	cudaFree(d_temp4[thr_id]);
}
//extern void x11_simd512_cpu_free(int thr_id);

__device__ __forceinline__
static void SIMD_Compress(uint32_t *A, const uint32_t thr_offset, const uint4 *const __restrict__ g_fft4){

	uint32_t IV[32];

	*(uint2x4*)&IV[0] = *(uint2x4*)&c_IV_512[0];
	*(uint2x4*)&IV[8] = *(uint2x4*)&c_IV_512[8];
	*(uint2x4*)&IV[16] = *(uint2x4*)&c_IV_512[16];
	*(uint2x4*)&IV[24] = *(uint2x4*)&c_IV_512[24];

	Round8(A, thr_offset, g_fft4);

	const uint32_t a[4] = { 4, 13, 10, 25 };

	for (int i = 0; i<4; i++)
		STEP8_IF(&IV[i * 8], 32 + i, a[i], a[(i + 1) & 3], &A[(0 + i * 24) & 31], &A[(8 + i * 24) & 31], &A[(16 + i * 24) & 31], &A[(24 + i * 24) & 31]);

#pragma unroll 32
	for (uint32_t i = 0; i<32; i++){
		IV[i] = A[i];
	}

	A[0] ^= 512;

	Round8_0_final(A, 3, 23, 17, 27);
	Round8_1_final(A, 28, 19, 22, 7);
	Round8_2_final(A, 29, 9, 15, 5);
	Round8_3_final(A, 4, 13, 10, 25);

	for (int i = 0; i<4; i++)
		STEP8_IF(&IV[i * 8], 32 + i, a[i], a[(i + 1) & 3], &A[(0 + i * 24) & 31], &A[(8 + i * 24) & 31], &A[(16 + i * 24) & 31], &A[(24 + i * 24) & 31]);

}



__global__ //__launch_bounds__(128, 5)
static void x16_simd512_gpu_compress_64_fugue512(uint32_t *g_hash, const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)
	__shared__ uint32_t shared[4][256];

	if (threadIdx.x<128)
	{
		uint2 temp = __ldg(&((uint2*)&mixtab0)[threadIdx.x]);

		shared[0][(threadIdx.x << 1) + 0] = temp.x;
		shared[0][(threadIdx.x << 1) + 1] = temp.y;
		shared[1][(threadIdx.x << 1) + 0] = ROR8(temp.x);
		shared[1][(threadIdx.x << 1) + 1] = ROR8(temp.y);
		shared[2][(threadIdx.x << 1) + 0] = ROL16(temp.x);
		shared[2][(threadIdx.x << 1) + 1] = ROL16(temp.y);
		shared[3][(threadIdx.x << 1) + 0] = ROL8(temp.x);
		shared[3][(threadIdx.x << 1) + 1] = ROL8(temp.y);


		/*		const uint32_t tmp = mixtab0[threadIdx.x];
		shared[0][threadIdx.x] = tmp;
		shared[1][threadIdx.x] = ROR8(tmp);
		shared[2][threadIdx.x] = ROL16(tmp);
		shared[3][threadIdx.x] = ROL8(tmp);
		*/
	}


	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751, 0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7, 0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
	};
	uint32_t k0;
	uint32_t h[16];

	//	if (thread < threads){

	uint32_t *Hash = &g_hash[thread << 4];

	uint32_t A[32];

	*(uint2x4*)&A[0] = *(uint2x4*)&c_IV_512[0] ^ __ldg4((uint2x4*)&Hash[0]);
	*(uint2x4*)&A[8] = *(uint2x4*)&c_IV_512[8] ^ __ldg4((uint2x4*)&Hash[8]);
	*(uint2x4*)&A[16] = *(uint2x4*)&c_IV_512[16];
	*(uint2x4*)&A[24] = *(uint2x4*)&c_IV_512[24];

	__syncthreads();

	SIMD_Compress(A, thr_offset, g_fft4);

	/*
	#pragma unroll 16
	for (int i = 0; i<16; i++){
	h[i] = A[i];
	}
	*(uint2x4*)&p[0] = *(uint2x4*)&A[0];
	*(uint2x4*)&p[4] = *(uint2x4*)&A[8];
	*/

	//	uint32_t *Hash = &g_hash[thread << 4];
	//		uint8_t h1[64];
	uint32_t c[16], m[16];
	*(uint2x4*)&h[0] = *((uint2x4*)&A[0]);
	*(uint2x4*)&h[8] = *((uint2x4*)&A[8]);

	for (int i = 0; i < 16; i++)
	{
		h[i] = cuda_swab32(h[i]);
	}

	__syncthreads();

	//		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&h[ 0];
	//		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&h[ 8];
	uint32_t S[36];
	uint32_t B[9];

	S[0] = S[1] = S[2] = S[3] = S[4] = S[5] = S[6] = S[7] = S[8] = S[9] = S[10] = S[11] = S[12] = S[13] = S[14] = S[15] = S[16] = S[17] = S[18] = S[19] = 0;
	*(uint2x4*)&S[20] = *(uint2x4*)&c_S[0];
#pragma unroll 8
	for (int i = 0; i<8; i++){
		S[28 + i] = c_S[i + 8];
	}

	FUGUE512_3(h[0x0], h[0x1], h[0x2]);
	FUGUE512_3(h[0x3], h[0x4], h[0x5]);
	FUGUE512_3(h[0x6], h[0x7], h[0x8]);
	FUGUE512_3(h[0x9], h[0xA], h[0xB]);
	FUGUE512_3(h[0xC], h[0xD], h[0xE]);
	FUGUE512_3(h[0xF], 0U, 512U);

	for (uint32_t i = 0; i < 32; i += 2){
		mROR3;
		CMIX36(S[0], S[1], S[2], S[4], S[5], S[6], S[18], S[19], S[20]);
		SMIX_LDG(shared, S[0], S[1], S[2], S[3]);
		mROR3;
		CMIX36(S[0], S[1], S[2], S[4], S[5], S[6], S[18], S[19], S[20]);
		SMIX_LDG(shared, S[0], S[1], S[2], S[3]);
	}
	#pragma unroll 11
	for (uint32_t i = 0; i < 13; i++) {
		S[4] ^= S[0];	S[9] ^= S[0];	S[18] ^= S[0];	S[27] ^= S[0];
		mROR9;
		SMIX_LDG(shared, S[0], S[1], S[2], S[3]);
		S[4] ^= S[0];	S[10] ^= S[0];	S[18] ^= S[0];	S[27] ^= S[0];
		mROR9;
		SMIX(shared, S[0], S[1], S[2], S[3]);
		S[4] ^= S[0];	S[10] ^= S[0];	S[19] ^= S[0];	S[27] ^= S[0];
		mROR9;
		SMIX_LDG(shared, S[0], S[1], S[2], S[3]);
		S[4] ^= S[0];	S[10] ^= S[0];	S[19] ^= S[0];	S[28] ^= S[0];
		mROR8;
		SMIX_LDG(shared, S[0], S[1], S[2], S[3]);
	}
	S[4] ^= S[0];	S[9] ^= S[0];	S[18] ^= S[0];	S[27] ^= S[0];

	S[0] = cuda_swab32(S[1]);	S[1] = cuda_swab32(S[2]);	S[2] = cuda_swab32(S[3]);	S[3] = cuda_swab32(S[4]);
	S[4] = cuda_swab32(S[9]);	S[5] = cuda_swab32(S[10]);	S[6] = cuda_swab32(S[11]);	S[7] = cuda_swab32(S[12]);
	S[8] = cuda_swab32(S[18]);	S[9] = cuda_swab32(S[19]);	S[10] = cuda_swab32(S[20]);	S[11] = cuda_swab32(S[21]);
	S[12] = cuda_swab32(S[27]);	S[13] = cuda_swab32(S[28]);	S[14] = cuda_swab32(S[29]);	S[15] = cuda_swab32(S[30]);

	*(uint2x4*)&Hash[0] = *(uint2x4*)&S[0];
	*(uint2x4*)&Hash[8] = *(uint2x4*)&S[8];
	//	}
}

__global__ //__launch_bounds__(128, 5)
static void x16_simd512_gpu_compress_64_hamsi512(uint32_t *g_hash, const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)

	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751, 0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7, 0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
	};
	uint32_t k0;
	uint32_t h[16];

	//	if (thread < threads){

	uint32_t *Hash = &g_hash[thread << 4];

	uint32_t A[32];

	*(uint2x4*)&A[0] = *(uint2x4*)&c_IV_512[0] ^ __ldg4((uint2x4*)&Hash[0]);
	*(uint2x4*)&A[8] = *(uint2x4*)&c_IV_512[8] ^ __ldg4((uint2x4*)&Hash[8]);
	*(uint2x4*)&A[16] = *(uint2x4*)&c_IV_512[16];
	*(uint2x4*)&A[24] = *(uint2x4*)&c_IV_512[24];

	__syncthreads();

	SIMD_Compress(A, thr_offset, g_fft4);

	uint8_t h1[64];

	//	uint32_t c[16], m[16];
	//	*(uint2x4*)&h[0] = *((uint2x4*)&A[0]);
	//	*(uint2x4*)&h[8] = *((uint2x4*)&A[8]);

	*(uint2x4*)&h1[0] = *(uint2x4*)&A[0];
	*(uint2x4*)&h1[32] = *(uint2x4*)&A[8];

	uint32_t c[16], m[16];
	*(uint16*)&c[0] = *(uint16*)&c_c[0];
	*(uint16*)&h[0] = *(uint16*)&c_c[0];

	const uint32_t *tp;
	uint32_t dm;

	for (int i = 0; i < 64; i += 8)
	{
		tp = &d_T512[0];

		dm = -(h1[i] & 1);
		m[0] = dm & tp[0]; m[1] = dm & tp[1];
		m[2] = dm & tp[2]; m[3] = dm & tp[3];
		m[4] = dm & tp[4]; m[5] = dm & tp[5];
		m[6] = dm & tp[6]; m[7] = dm & tp[7];
		m[8] = dm & tp[8]; m[9] = dm & tp[9];
		m[10] = dm & tp[10]; m[11] = dm & tp[11];
		m[12] = dm & tp[12]; m[13] = dm & tp[13];
		m[14] = dm & tp[14]; m[15] = dm & tp[15];
		tp += 16;
		//#pragma unroll 7
		for (int v = 1; v < 8; v++) {
			dm = -((h1[i] >> v) & 1);
			m[0] ^= dm & tp[0]; m[1] ^= dm & tp[1];
			m[2] ^= dm & tp[2]; m[3] ^= dm & tp[3];
			m[4] ^= dm & tp[4]; m[5] ^= dm & tp[5];
			m[6] ^= dm & tp[6]; m[7] ^= dm & tp[7];
			m[8] ^= dm & tp[8]; m[9] ^= dm & tp[9];
			m[10] ^= dm & tp[10]; m[11] ^= dm & tp[11];
			m[12] ^= dm & tp[12]; m[13] ^= dm & tp[13];
			m[14] ^= dm & tp[14]; m[15] ^= dm & tp[15];
			tp += 16;
		}

		//#pragma unroll
		for (int u = 1; u < 8; u++) {
#pragma unroll 8
			for (int v = 0; v < 8; v++) {
				dm = -((h1[i + u] >> v) & 1);
				m[0] ^= dm & tp[0]; m[1] ^= dm & tp[1];
				m[2] ^= dm & tp[2]; m[3] ^= dm & tp[3];
				m[4] ^= dm & tp[4]; m[5] ^= dm & tp[5];
				m[6] ^= dm & tp[6]; m[7] ^= dm & tp[7];
				m[8] ^= dm & tp[8]; m[9] ^= dm & tp[9];
				m[10] ^= dm & tp[10]; m[11] ^= dm & tp[11];
				m[12] ^= dm & tp[12]; m[13] ^= dm & tp[13];
				m[14] ^= dm & tp[14]; m[15] ^= dm & tp[15];
				tp += 16;
			}
		}

		//#pragma unroll 6
		for (int r = 0; r < 6; r++) {
			ROUND_BIG(r, d_alpha_n);
		}
		/* order is (no more) important */
		h[0] ^= m[0]; h[1] ^= m[1]; h[2] ^= c[0]; h[3] ^= c[1];
		h[4] ^= m[2]; h[5] ^= m[3]; h[6] ^= c[2]; h[7] ^= c[3];
		h[8] ^= m[8]; h[9] ^= m[9]; h[10] ^= c[8]; h[11] ^= c[9];
		h[12] ^= m[10]; h[13] ^= m[11]; h[14] ^= c[10]; h[15] ^= c[11];

		*(uint16*)&c[0] = *(uint16*)&h[0];
	}

	*(uint2x4*)&m[0] = *(uint2x4*)&d_T512[112];
	*(uint2x4*)&m[8] = *(uint2x4*)&d_T512[120];

#pragma unroll 6
	for (int r = 0; r < 6; r++) {
		ROUND_BIG(r, d_alpha_n);
	}

	/* order is (no more) important */
	h[0] ^= m[0]; h[1] ^= m[1]; h[2] ^= c[0]; h[3] ^= c[1];
	h[4] ^= m[2]; h[5] ^= m[3]; h[6] ^= c[2]; h[7] ^= c[3];
	h[8] ^= m[8]; h[9] ^= m[9]; h[10] ^= c[8]; h[11] ^= c[9];
	h[12] ^= m[10]; h[13] ^= m[11]; h[14] ^= c[10]; h[15] ^= c[11];

	*(uint16*)&c[0] = *(uint16*)&h[0];

	*(uint2x4*)&m[0] = *(uint2x4*)&d_T512[784];
	*(uint2x4*)&m[8] = *(uint2x4*)&d_T512[792];

#pragma unroll 12
	for (int r = 0; r < 12; r++)
		ROUND_BIG(r, d_alpha_f);

	/* order is (no more) important */
	h[0] ^= m[0]; h[1] ^= m[1]; h[2] ^= c[0]; h[3] ^= c[1];
	h[4] ^= m[2]; h[5] ^= m[3]; h[6] ^= c[2]; h[7] ^= c[3];
	h[8] ^= m[8]; h[9] ^= m[9]; h[10] ^= c[8]; h[11] ^= c[9];
	h[12] ^= m[10]; h[13] ^= m[11]; h[14] ^= c[10]; h[15] ^= c[11];


#pragma unroll 16
	for (int i = 0; i < 16; i++)
		h[i] = cuda_swab32(h[i]);

	*(uint2x4*)&Hash[0] = *(uint2x4*)&h[0];
	*(uint2x4*)&Hash[8] = *(uint2x4*)&h[8];
}
__global__
__launch_bounds__(256, 3)
static void x16_simd512_gpu_compress_64_echo512(uint32_t *g_hash, const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)

	__shared__ uint32_t sharedMemory[8 * 1024];

	if(threadIdx.x<256) aes_gpu_init256_32(sharedMemory);


	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751, 0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7, 0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
	};
	uint32_t k0;
	uint32_t h[16];

	//	if (thread < threads){

	uint32_t *Hash = &g_hash[thread << 4];

	uint32_t A[32];

	*(uint2x4*)&A[0] = *(uint2x4*)&c_IV_512[0] ^ __ldg4((uint2x4*)&Hash[0]);
	*(uint2x4*)&A[8] = *(uint2x4*)&c_IV_512[8] ^ __ldg4((uint2x4*)&Hash[8]);
	*(uint2x4*)&A[16] = *(uint2x4*)&c_IV_512[16];
	*(uint2x4*)&A[24] = *(uint2x4*)&c_IV_512[24];

	__syncthreads();

	SIMD_Compress(A, thr_offset, g_fft4);

	*(uint2x4*)&Hash[0] = *(uint2x4*)&A[0];
	*(uint2x4*)&Hash[8] = *(uint2x4*)&A[8];

#pragma unroll 16
	for (int i = 0; i<16; i++){
		h[i] = A[i];
	}

	k0 = 512 + 8;

#pragma unroll 4
	for (uint32_t idx = 0; idx < 16; idx += 4)
		AES_2ROUND_32(sharedMemory, h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);

	k0 += 4;

	uint32_t W[64];

	//		#pragma unroll 4
	for (int i = 0; i < 4; i++){
		uint32_t a = P[i];
		uint32_t b = P[i + 4];
		uint32_t c = h[i + 8];
		uint32_t d = P[i + 8];

		uint32_t ab = a ^ b;
		uint32_t bc = b ^ c;
		uint32_t cd = c ^ d;


		uint32_t t = ((a ^ b) & 0x80808080);
		uint32_t t2 = ((b ^ c) & 0x80808080);
		uint32_t t3 = ((c ^ d) & 0x80808080);

		uint32_t abx = ((t >> 7) * 27U) ^ ((ab^t) << 1);
		uint32_t bcx = ((t2 >> 7) * 27U) ^ ((bc^t2) << 1);
		uint32_t cdx = ((t3 >> 7) * 27U) ^ ((cd^t3) << 1);

		W[0U + i] = bc ^ d ^ abx;
		W[4U + i] = a ^ cd ^ bcx;
		W[8U + i] = ab ^ d ^ cdx;
		W[12U + i] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = P[12U + i];
		b = h[i + 4U];
		c = P[12U + i + 4U];
		d = P[12U + i + 8U];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;


		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27U ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

		W[16U + i] = abx ^ bc ^ d;
		W[16U + i + 4U] = bcx ^ a ^ cd;
		W[16U + i + 8U] = cdx ^ ab ^ d;
		W[16U + i + 12U] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = h[i];
		b = P[24U + i + 0U];
		c = P[24U + i + 4U];
		d = P[24U + i + 8U];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;


		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27U ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

		W[32U + i] = abx ^ bc ^ d;
		W[32U + i + 4U] = bcx ^ a ^ cd;
		W[32U + i + 8U] = cdx ^ ab ^ d;
		W[32U + i + 12U] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = P[36U + i];
		b = P[36U + i + 4U];
		c = P[36U + i + 8U];
		d = h[i + 12U];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;

		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27U ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

		W[48U + i] = abx ^ bc ^ d;
		W[48U + i + 4U] = bcx ^ a ^ cd;
		W[48U + i + 8U] = cdx ^ ab ^ d;
		W[48U + i + 12U] = abx ^ bcx ^ cdx ^ ab ^ c;
	}

	for (int k = 1; k < 10; k++){
		echo_round_sp(sharedMemory, W, k0);
	}
#pragma unroll 4
	for (uint32_t i = 0; i < 16; i += 4)
	{
		W[i] ^= W[32 + i] ^ 512;
		W[i + 1] ^= W[32 + i + 1];
		W[i + 2] ^= W[32 + i + 2];
		W[i + 3] ^= W[32 + i + 3];
	}
	*(uint2x4*)&Hash[0] = *(uint2x4*)&Hash[0] ^ *(uint2x4*)&W[0];
	*(uint2x4*)&Hash[8] = *(uint2x4*)&Hash[8] ^ *(uint2x4*)&W[8];
}

__global__ __launch_bounds__(256, 3)
void x16_simd512_gpu_compress_64_maxwell_echo512_final(const uint32_t* __restrict__ g_hash, uint32_t startnonce, const uint4 *const __restrict__ g_fft4, uint32_t* resNonce, const uint64_t target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)

	__shared__ uint32_t sharedMemory[1024 * 8];

	aes_gpu_init256_32(sharedMemory);

	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751, 0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7, 0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
	};
	uint32_t k0;
	uint32_t h[16];

	//	if (thread < threads){

	const uint32_t* __restrict__ Hash = &g_hash[thread << 4];

	uint32_t A[32];

	*(uint2x4*)&A[0] = *(uint2x4*)&c_IV_512[0] ^ __ldg4((uint2x4*)&Hash[0]);
	*(uint2x4*)&A[8] = *(uint2x4*)&c_IV_512[8] ^ __ldg4((uint2x4*)&Hash[8]);
	*(uint2x4*)&A[16] = *(uint2x4*)&c_IV_512[16];
	*(uint2x4*)&A[24] = *(uint2x4*)&c_IV_512[24];

	SIMD_Compress(A, thr_offset, g_fft4);

#pragma unroll 16
	for (int i = 0; i<16; i++){
		h[i] = A[i];
	}

	uint64_t backup = *(uint64_t*)&h[6];

	k0 = 512 + 8;

#pragma unroll 3
	for (uint32_t idx = 0; idx < 16; idx += 4){
		AES_2ROUND_32(sharedMemory, h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);
		idx += 4;
		AES_2ROUND_32(sharedMemory, h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);
	}
	k0 += 4;

	uint32_t W[64];

	//		#pragma unroll 4
	for (int i = 0; i < 4; i++){
		uint32_t a = P[i];
		uint32_t b = P[i + 4];
		uint32_t c = h[i + 8];
		uint32_t d = P[i + 8];

		uint32_t ab = a ^ b;
		uint32_t bc = b ^ c;
		uint32_t cd = c ^ d;


		uint32_t t = ((a ^ b) & 0x80808080);
		uint32_t t2 = ((b ^ c) & 0x80808080);
		uint32_t t3 = ((c ^ d) & 0x80808080);

		uint32_t abx = ((t >> 7) * 27U) ^ ((ab^t) << 1);
		uint32_t bcx = ((t2 >> 7) * 27U) ^ ((bc^t2) << 1);
		uint32_t cdx = ((t3 >> 7) * 27U) ^ ((cd^t3) << 1);

		W[0U + i] = bc ^ d ^ abx;
		W[4U + i] = a ^ cd ^ bcx;
		W[8U + i] = ab ^ d ^ cdx;
		W[12U + i] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = P[12U + i];
		b = h[i + 4U];
		c = P[12U + i + 4U];
		d = P[12U + i + 8U];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;


		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27U ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

		W[16U + i] = abx ^ bc ^ d;
		W[16U + i + 4U] = bcx ^ a ^ cd;
		W[16U + i + 8U] = cdx ^ ab ^ d;
		W[16U + i + 12U] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = h[i];
		b = P[24U + i + 0U];
		c = P[24U + i + 4U];
		d = P[24U + i + 8U];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;


		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27U ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

		W[32U + i] = abx ^ bc ^ d;
		W[32U + i + 4U] = bcx ^ a ^ cd;
		W[32U + i + 8U] = cdx ^ ab ^ d;
		W[32U + i + 12U] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = P[36U + i];
		b = P[36U + i + 4U];
		c = P[36U + i + 8U];
		d = h[i + 12U];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;

		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27U ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

		W[48U + i] = abx ^ bc ^ d;
		W[48U + i + 4U] = bcx ^ a ^ cd;
		W[48U + i + 8U] = cdx ^ ab ^ d;
		W[48U + i + 12U] = abx ^ bcx ^ cdx ^ ab ^ c;
	}

	for (int k = 1; k < 9; k++){
		echo_round_sp(sharedMemory, W, k0);
	}

	// Big Sub Words
	uint32_t y[4];
	aes_round_32(sharedMemory, W[0], W[1], W[2], W[3], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[0], W[1], W[2], W[3]);
	aes_round_32(sharedMemory, W[4], W[5], W[6], W[7], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[4], W[5], W[6], W[7]);
	aes_round_32(sharedMemory, W[8], W[9], W[10], W[11], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[8], W[9], W[10], W[11]);
	aes_round_32(sharedMemory, W[20], W[21], W[22], W[23], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[20], W[21], W[22], W[23]);
	aes_round_32(sharedMemory, W[28], W[29], W[30], W[31], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[28], W[29], W[30], W[31]);
	aes_round_32(sharedMemory, W[32], W[33], W[34], W[35], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[32], W[33], W[34], W[35]);
	aes_round_32(sharedMemory, W[40], W[41], W[42], W[43], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[40], W[41], W[42], W[43]);
	aes_round_32(sharedMemory, W[52], W[53], W[54], W[55], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[52], W[53], W[54], W[55]);
	aes_round_32(sharedMemory, W[60], W[61], W[62], W[63], k0, y[0], y[1], y[2], y[3]);
	aes_round_32(sharedMemory, y[0], y[1], y[2], y[3], W[60], W[61], W[62], W[63]);

	uint32_t bc = W[22] ^ W[42];
	uint32_t t2 = (bc & 0x80808080);
	W[6] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

	bc = W[23] ^ W[43];
	t2 = (bc & 0x80808080);
	W[7] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

	bc = W[10] ^ W[54];
	t2 = (bc & 0x80808080);
	W[38] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

	bc = W[11] ^ W[55];
	t2 = (bc & 0x80808080);
	W[39] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

	uint64_t check = backup ^ *(uint64_t*)&W[2] ^ *(uint64_t*)&W[6] ^ *(uint64_t*)&W[10] ^ *(uint64_t*)&W[30] ^ *(uint64_t*)&W[34] ^ *(uint64_t*)&W[38] ^ *(uint64_t*)&W[42] ^ *(uint64_t*)&W[62];
	uint32_t nonce = thread + startnonce;

	if (check <= target)
	{
		uint32_t tmp = atomicExch(&resNonce[0], nonce);
		if (tmp != UINT32_MAX)
			resNonce[1] = tmp;
	}
	//	}
}

__global__ //__launch_bounds__(128, 5)
static void x16_simd512_gpu_compress_64_whirlpool512(uint32_t *g_hash, const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)

	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751, 0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7, 0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
	};

	__shared__ uint2 sharedMemory[256][16];

	if (threadIdx.x < 128)
	{
		const uint2 tmp = b0[threadIdx.x];
		const uint2 tmp2 = b0[threadIdx.x+128];
		sharedMemory[threadIdx.x][0] = tmp;
		sharedMemory[threadIdx.x][1] = tmp;
		sharedMemory[threadIdx.x][2] = tmp;
		sharedMemory[threadIdx.x][3] = tmp;
		sharedMemory[threadIdx.x][4] = tmp;
		sharedMemory[threadIdx.x][5] = tmp;
		sharedMemory[threadIdx.x][6] = tmp;
		sharedMemory[threadIdx.x][7] = tmp;
		sharedMemory[threadIdx.x][8] = tmp;
		sharedMemory[threadIdx.x][9] = tmp;
		sharedMemory[threadIdx.x][10] = tmp;
		sharedMemory[threadIdx.x][11] = tmp;
		sharedMemory[threadIdx.x][12] = tmp;
		sharedMemory[threadIdx.x][13] = tmp;
		sharedMemory[threadIdx.x][14] = tmp;
		sharedMemory[threadIdx.x][15] = tmp;

		sharedMemory[threadIdx.x + 128][0] = tmp2;
		sharedMemory[threadIdx.x + 128][1] = tmp2;
		sharedMemory[threadIdx.x + 128][2] = tmp2;
		sharedMemory[threadIdx.x + 128][3] = tmp2;
		sharedMemory[threadIdx.x + 128][4] = tmp2;
		sharedMemory[threadIdx.x + 128][5] = tmp2;
		sharedMemory[threadIdx.x + 128][6] = tmp2;
		sharedMemory[threadIdx.x + 128][7] = tmp2;
		sharedMemory[threadIdx.x + 128][8] = tmp2;
		sharedMemory[threadIdx.x + 128][9] = tmp2;
		sharedMemory[threadIdx.x + 128][10] = tmp2;
		sharedMemory[threadIdx.x + 128][11] = tmp2;
		sharedMemory[threadIdx.x + 128][12] = tmp2;
		sharedMemory[threadIdx.x + 128][13] = tmp2;
		sharedMemory[threadIdx.x + 128][14] = tmp2;
		sharedMemory[threadIdx.x + 128][15] = tmp2;
	}


	uint32_t k0;
	//	uint32_t h[16];

	//	if (thread < threads){

	uint32_t *Hash = &g_hash[thread << 4];

	uint32_t A[32];

	*(uint2x4*)&A[0] = *(uint2x4*)&c_IV_512[0] ^ __ldg4((uint2x4*)&Hash[0]);
	*(uint2x4*)&A[8] = *(uint2x4*)&c_IV_512[8] ^ __ldg4((uint2x4*)&Hash[8]);
	*(uint2x4*)&A[16] = *(uint2x4*)&c_IV_512[16];
	*(uint2x4*)&A[24] = *(uint2x4*)&c_IV_512[24];

	__syncthreads();

	SIMD_Compress(A, thr_offset, g_fft4);

	uint2 hash[8], n[8], h[8];
	uint2 tmp[8] = {
		{ 0xC0EE0B30, 0x672990AF }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 },
		{ 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }, { 0x28282828, 0x28282828 }
	};

	*(uint2x4*)&hash[0] = *((uint2x4*)&A[0]); //__ldg4((uint2x4*)&g_hash[(thread << 3) + 0]);
	*(uint2x4*)&hash[4] = *((uint2x4*)&A[8]); //__ldg4((uint2x4*)&g_hash[(thread << 3) + 4]);

	__syncthreads();

	const uint32_t index = 15; //sharedindex;

#pragma unroll 8
	for (int i = 0; i<8; i++)
		n[i] = hash[i];

	tmp[0] ^= d_ROUND_ELT(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1);
	tmp[1] ^= d_ROUND_ELT(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2);
	tmp[2] ^= d_ROUND_ELT(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3);
	tmp[3] ^= d_ROUND_ELT(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4);
	tmp[4] ^= d_ROUND_ELT(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5);
	tmp[5] ^= d_ROUND_ELT(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6);
	tmp[6] ^= d_ROUND_ELT(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7);
	tmp[7] ^= d_ROUND_ELT(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0);
	for (int i = 1; i <10; i++)
	{
		TRANSFER(n, tmp);
		tmp[0] = d_ROUND_ELT1(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i - 1) * 8 + 0]);
		tmp[1] = d_ROUND_ELT1(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i - 1) * 8 + 1]);
		tmp[2] = d_ROUND_ELT1(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i - 1) * 8 + 2]);
		tmp[3] = d_ROUND_ELT1(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i - 1) * 8 + 3]);
		tmp[4] = d_ROUND_ELT1(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i - 1) * 8 + 4]);
		tmp[5] = d_ROUND_ELT1(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i - 1) * 8 + 5]);
		tmp[6] = d_ROUND_ELT1(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i - 1) * 8 + 6]);
		tmp[7] = d_ROUND_ELT1(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i - 1) * 8 + 7]);
	}

	TRANSFER(h, tmp);
#pragma unroll 8
	for (int i = 0; i<8; i++)
		hash[i] = h[i] = h[i] ^ hash[i];

#pragma unroll 6
	for (int i = 1; i<7; i++)
		n[i] = vectorize(0);

	n[0] = vectorize(0x80);
	n[7] = vectorize(0x2000000000000);

#pragma unroll 8
	for (int i = 0; i < 8; i++) {
		n[i] = n[i] ^ h[i];
	}

//	#pragma unroll 2
	for (int i = 0; i < 10; i++)
	{
		tmp[0] = InitVector_RC[i];
		tmp[0] ^= d_ROUND_ELT(index, sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1);
		tmp[1] = d_ROUND_ELT(index, sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[2] = d_ROUND_ELT(index, sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[3] = d_ROUND_ELT(index, sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[4] = d_ROUND_ELT(index, sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[5] = d_ROUND_ELT(index, sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[6] = d_ROUND_ELT(index, sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[7] = d_ROUND_ELT(index, sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
		TRANSFER(h, tmp);
		tmp[0] = d_ROUND_ELT1(index, sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
		tmp[1] = d_ROUND_ELT1(index, sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
		tmp[2] = d_ROUND_ELT1(index, sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
		tmp[3] = d_ROUND_ELT1(index, sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
		tmp[4] = d_ROUND_ELT1(index, sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
		tmp[5] = d_ROUND_ELT1(index, sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
		tmp[6] = d_ROUND_ELT1(index, sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
		tmp[7] = d_ROUND_ELT1(index, sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
		TRANSFER(n, tmp);
	}

	hash[0] = xor3x(hash[0], n[0], vectorize(0x80));
	hash[1] = hash[1] ^ n[1];
	hash[2] = hash[2] ^ n[2];
	hash[3] = hash[3] ^ n[3];
	hash[4] = hash[4] ^ n[4];
	hash[5] = hash[5] ^ n[5];
	hash[6] = hash[6] ^ n[6];
	hash[7] = xor3x(hash[7], n[7], vectorize(0x2000000000000));

	*(uint2x4*)&Hash[0] = *(uint2x4*)&hash[0];
	*(uint2x4*)&Hash[8] = *(uint2x4*)&hash[4];
}


__host__
void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	int dev_id = device_map[thr_id];

	uint32_t tpb = TPB52_1;
	if (device_sm[dev_id] <= 500) tpb = TPB50_1;
	const dim3 grid1((8*threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	x11_simd512_gpu_expand_64 <<<grid1, block1>>> (threads, d_hash, d_temp4[thr_id]);
	x11_simd512_gpu_compress_64 <<< grid1, block1 >>> (threads, d_hash, d_temp4[thr_id]);
}


__host__
void x11_simd512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startnonce, uint32_t *d_hash, uint32_t *d_resNonce, uint64_t target)
{
	int dev_id = device_map[thr_id];

	uint32_t tpb = 32;
	if (device_sm[dev_id] <= 500) tpb = TPB50_1;
	const dim3 grid1((8 * threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	//	tpb = TPB52_2;
	//	if (device_sm[dev_id] <= 500) tpb = TPB50_2;
	//	const dim3 grid2((threads + tpb - 1) / tpb);
	//	const dim3 block2(tpb);

	x11_simd512_gpu_expand_64 << <grid1, block1 >> > (threads, d_hash, d_temp4[thr_id]);
	x11_simd512_gpu_compress_64_pascal_final << < grid1, block1 >> > (threads, startnonce, d_hash, d_temp4[thr_id], d_resNonce, target);
}


__host__
void x16_simd_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{

	int dev_id = device_map[thr_id];

	uint32_t tpb = 128;
	const dim3 grid1((8 * threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	tpb = 256;
	const dim3 grid2((threads + tpb - 1) / tpb);
	const dim3 block2(tpb);

	x11_simd512_gpu_expand_64 << <grid1, block1 >> > (threads, d_hash, d_temp4[thr_id]);
	x16_simd512_gpu_compress_64_echo512 << < grid2, block2 >> > (d_hash, d_temp4[thr_id]);
}

__host__
void x16_simd_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startnonce, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target)
{

	int dev_id = device_map[thr_id];

	uint32_t tpb = 128;
	const dim3 grid1((8 * threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	tpb = 256;
	const dim3 grid2((threads + tpb - 1) / tpb);
	const dim3 block2(tpb);

	x11_simd512_gpu_expand_64 << <grid1, block1 >> > (threads, d_hash, d_temp4[thr_id]);
	x16_simd512_gpu_compress_64_maxwell_echo512_final << < grid2, block2 >> > (d_hash, startnonce, d_temp4[thr_id], d_resNonce, target);
}


__host__
void x16_simd_whirlpool512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	int dev_id = device_map[thr_id];

	uint32_t tpb = TPB52_1;
	if (device_sm[dev_id] <= 500) tpb = TPB52_1;
	const dim3 grid1((8 * threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	tpb = 128;
	if (device_sm[dev_id] <= 500) tpb = 128;
	const dim3 grid2((threads + tpb - 1) / tpb);
	const dim3 block2(tpb);

	x11_simd512_gpu_expand_64 << <grid1, block1 >> > (threads, d_hash, d_temp4[thr_id]);
	x16_simd512_gpu_compress_64_whirlpool512 << < grid2, block2 >> > (d_hash, d_temp4[thr_id]);
}

__host__
void x16_simd_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

	int dev_id = device_map[thr_id];

	uint32_t tpb = TPB52_1;
	if (device_sm[dev_id] <= 500) tpb = TPB52_1;
	const dim3 grid1((8 * threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	tpb = 128;
	if (device_sm[dev_id] <= 500) tpb = 128;
	const dim3 grid2((threads + tpb - 1) / tpb);
	const dim3 block2(tpb);

	x11_simd512_gpu_expand_64 << <grid1, block1 >> > (threads, d_hash, d_temp4[thr_id]);
	x16_simd512_gpu_compress_64_hamsi512 << < grid2, block2 >> > (d_hash, d_temp4[thr_id]);
}

__host__
void x16_simd_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

	int dev_id = device_map[thr_id];

	uint32_t tpb = TPB52_1;
	if (device_sm[dev_id] <= 500) tpb = TPB52_1;
	const dim3 grid1((8 * threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

	tpb = 128;
	if (device_sm[dev_id] <= 500) tpb = 128;
	const dim3 grid2((threads + tpb - 1) / tpb);
	const dim3 block2(tpb);

	x11_simd512_gpu_expand_64 << <grid1, block1 >> > (threads, d_hash, d_temp4[thr_id]);
	x16_simd512_gpu_compress_64_fugue512 << < grid2, block2 >> > (d_hash, d_temp4[thr_id]);
}
