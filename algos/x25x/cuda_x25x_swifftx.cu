/**
 * SWIFFTX stage for X25X (SUQA/SIN) -- the compute tentpole.
 *
 * One thread per hash: consumes the 256-byte window (accumulator slots
 * hash[12..15]) and writes the 512-bit (64-byte) SWIFFTX digest (chain slot
 * hash[16]).  Faithful transcription of the scalar reference in
 * algos/x25x/swifftx.c (FFT() copied verbatim; SWIFFTFFT/SWIFFTSum/
 * TranslateToBase256/ComputeSingleSWIFFTX rewritten with literal sizes).
 *
 * The runtime tables (multipliers, fftTable) plus the constant As/SBox are
 * built on the host by InitializeSWIFFTX() and uploaded once to device
 * symbols by x25x_swifftx_cpu_init().  swift_int16_t is 32-bit (Z_257
 * residues stored in 32-bit slots), matching the CPU reference.
 */

#include "cuda/helper.cuh"
#include <stdint.h>

typedef int32_t swift_int16_t;
typedef int32_t swift_int32_t;

// Device copies of the SWIFFTX tables (see x25x_swifftx_cpu_init).
// multipliers (FFT twiddles) and As (SWIFFTSum coefficients) are read at
// thread-UNIFORM indices in the hot loops, so __constant__ gives a broadcast
// from the constant cache. fftTable and SBox are indexed by data-dependent
// (divergent) bytes, so they stay in global memory (constant would serialise).
// multipliers (FFT twiddles) are read at thread/lane-UNIFORM indices -> constant
// broadcast. fftTable (data-dependent byte index) and As (in the warp version,
// As[lane*64+j] is per-lane distinct) are divergent -> keep in global memory.
__constant__ swift_int16_t c_multipliers[64];
__device__   swift_int16_t c_fftTable[256 * 8];
__device__   swift_int16_t c_As[3 * 32 * 64];
__device__   unsigned char c_SBox[256];

// ---- FFT(): one 8-byte chunk -> 64 partial-reduced values. Verbatim scalar
// reference (algos/x25x/swifftx.c), with the global tables renamed to their
// device copies. --------------------------------------------------------------
__device__ static void swifftx_FFT(const unsigned char *input, swift_int32_t *output)
{
   
   swift_int16_t *mult = c_multipliers;
	swift_int16_t *table = &( c_fftTable[ input[0] << 3 ] );
   swift_int32_t F[64];

   /*
   for (int i = 0; i < 8; i++)
   {
      int j = i<<3;
      swift_int16_t *table = &(c_fftTable[input[i] << 3]);
      F[i   ] = mult[j+0] * table[0];
      F[i+ 8] = mult[j+1] * table[1];
      F[i+16] = mult[j+2] * table[2];
      F[i+24] = mult[j+3] * table[3];
      F[i+32] = mult[j+4] * table[4];
      F[i+40] = mult[j+5] * table[5];
      F[i+48] = mult[j+6] * table[6];
      F[i+56] = mult[j+7] * table[7];
   }
*/

	F[ 0] = mult[ 0] * table[0];
	F[ 8] = mult[ 1] * table[1];
	F[16] = mult[ 2] * table[2];
	F[24] = mult[ 3] * table[3];
	F[32] = mult[ 4] * table[4];
	F[40] = mult[ 5] * table[5];
	F[48] = mult[ 6] * table[6];
	F[56] = mult[ 7] * table[7];

	table = &(c_fftTable[input[1] << 3]);

	F[ 1] = mult[ 8] * table[0];
	F[ 9] = mult[ 9] * table[1];
	F[17] = mult[10] * table[2];
	F[25] = mult[11] * table[3];
	F[33] = mult[12] * table[4];
	F[41] = mult[13] * table[5];
	F[49] = mult[14] * table[6];
	F[57] = mult[15] * table[7];

	table = &(c_fftTable[input[2] << 3]);

	F[ 2] = mult[16] * table[0];
	F[10] = mult[17] * table[1];
	F[18] = mult[18] * table[2];
	F[26] = mult[19] * table[3];
	F[34] = mult[20] * table[4];
	F[42] = mult[21] * table[5];
	F[50] = mult[22] * table[6];
	F[58] = mult[23] * table[7];

	table = &(c_fftTable[input[3] << 3]);

	F[ 3] = mult[24] * table[0];
	F[11] = mult[25] * table[1];
	F[19] = mult[26] * table[2];
	F[27] = mult[27] * table[3];
	F[35] = mult[28] * table[4];
	F[43] = mult[29] * table[5];
	F[51] = mult[30] * table[6];
	F[59] = mult[31] * table[7];

	table = &(c_fftTable[input[4] << 3]);

	F[ 4] = mult[32] * table[0];
	F[12] = mult[33] * table[1];
	F[20] = mult[34] * table[2];
	F[28] = mult[35] * table[3];
	F[36] = mult[36] * table[4];
	F[44] = mult[37] * table[5];
	F[52] = mult[38] * table[6];
	F[60] = mult[39] * table[7];

	table = &(c_fftTable[input[5] << 3]);

	F[ 5] = mult[40] * table[0];
	F[13] = mult[41] * table[1];
	F[21] = mult[42] * table[2];
	F[29] = mult[43] * table[3];
	F[37] = mult[44] * table[4];
	F[45] = mult[45] * table[5];
	F[53] = mult[46] * table[6];
	F[61] = mult[47] * table[7];

	table = &(c_fftTable[input[6] << 3]);

	F[ 6] = mult[48] * table[0];
	F[14] = mult[49] * table[1];
	F[22] = mult[50] * table[2];
	F[30] = mult[51] * table[3];
	F[38] = mult[52] * table[4];
	F[46] = mult[53] * table[5];
	F[54] = mult[54] * table[6];
	F[62] = mult[55] * table[7];

	table = &(c_fftTable[input[7] << 3]);

	F[ 7] = mult[56] * table[0];
	F[15] = mult[57] * table[1];
	F[23] = mult[58] * table[2];
	F[31] = mult[59] * table[3];
	F[39] = mult[60] * table[4];
	F[47] = mult[61] * table[5];
	F[55] = mult[62] * table[6];
	F[63] = mult[63] * table[7];

   #define ADD_SUB( a, b ) \
   { \
      int temp = b; \
      b = a - b; \
      a = a + temp; \
   }
   
   #define Q_REDUCE( a ) \
      ( ( (a) & 0xff ) - ( (a) >> 8 ) )
   
/*

   for ( int i = 0; i < 8; i++ )
   {
      int j = i<<3;
      ADD_SUB( F[j  ], F[j+1] );
      ADD_SUB( F[j+2], F[j+3] );
      ADD_SUB( F[j+4], F[j+5] );
      ADD_SUB( F[j+6], F[j+7] );

      F[j+3] <<= 4;
      F[j+7] <<= 4;

      ADD_SUB( F[j  ], F[j+2] );
      ADD_SUB( F[j+1], F[j+3] );
      ADD_SUB( F[j+4], F[j+6] );
      ADD_SUB( F[j+5], F[j+7] );

      F[j+5] <<= 2;
      F[j+6] <<= 4;
      F[j+7] <<= 6;

      ADD_SUB( F[j  ], F[j+4] );
      ADD_SUB( F[j+1], F[j+5] );
      ADD_SUB( F[j+2], F[j+6] );
      ADD_SUB( F[j+3], F[j+7] );

      output[i   ] = Q_REDUCE( F[j  ] );
      output[i+ 8] = Q_REDUCE( F[j+1] );
      output[i+16] = Q_REDUCE( F[j+2] );
      output[i+24] = Q_REDUCE( F[j+3] );
      output[i+32] = Q_REDUCE( F[j+4] );
      output[i+40] = Q_REDUCE( F[j+5] );
      output[i+48] = Q_REDUCE( F[j+6] );
      output[i+56] = Q_REDUCE( F[j+7] );
   }
*/

	// Iteration 0:
	ADD_SUB( F[ 0], F[ 1] );
	ADD_SUB( F[ 2], F[ 3] );
	ADD_SUB( F[ 4], F[ 5] );
	ADD_SUB( F[ 6], F[ 7] );
	F[ 3] <<= 4;
	F[ 7] <<= 4;
	ADD_SUB( F[ 0], F[ 2] );
	ADD_SUB( F[ 1], F[ 3] );
	ADD_SUB( F[ 4], F[ 6] );
	ADD_SUB( F[ 5], F[ 7] );
	F[ 5] <<= 2;
	F[ 6] <<= 4;
	F[ 7] <<= 6;
	ADD_SUB( F[ 0], F[ 4] );
	ADD_SUB( F[ 1], F[ 5] );
	ADD_SUB( F[ 2], F[ 6] );
	ADD_SUB( F[ 3], F[ 7] );

   output[ 0] = Q_REDUCE( F[ 0] );
	output[ 8] = Q_REDUCE( F[ 1] );
	output[16] = Q_REDUCE( F[ 2] );
	output[24] = Q_REDUCE( F[ 3] );
	output[32] = Q_REDUCE( F[ 4] );
	output[40] = Q_REDUCE( F[ 5] );
	output[48] = Q_REDUCE( F[ 6] );
	output[56] = Q_REDUCE( F[ 7] );

	// Iteration 1:
	ADD_SUB( F[ 8], F[ 9] );
	ADD_SUB( F[10], F[11] );
	ADD_SUB( F[12], F[13] );
	ADD_SUB( F[14], F[15] );
	F[11] <<= 4;
	F[15] <<= 4;
	ADD_SUB( F[ 8], F[10] );
	ADD_SUB( F[ 9], F[11] );
	ADD_SUB( F[12], F[14] );
	ADD_SUB( F[13], F[15] );
	F[13] <<= 2;
	F[14] <<= 4;
	F[15] <<= 6;
	ADD_SUB( F[ 8], F[12] );
	ADD_SUB( F[ 9], F[13] );
	ADD_SUB( F[10], F[14] );
	ADD_SUB( F[11], F[15] );

	output[ 1] = Q_REDUCE( F[ 8] );
	output[ 9] = Q_REDUCE( F[ 9] );
	output[17] = Q_REDUCE( F[10] );
	output[25] = Q_REDUCE( F[11] );
	output[33] = Q_REDUCE( F[12] );
	output[41] = Q_REDUCE( F[13] );
	output[49] = Q_REDUCE( F[14] );
	output[57] = Q_REDUCE( F[15] );

	// Iteration 2:
	ADD_SUB( F[16], F[17] );
	ADD_SUB( F[18], F[19] );
	ADD_SUB( F[20], F[21] );
	ADD_SUB( F[22], F[23] );
	F[19] <<= 4;
	F[23] <<= 4;
	ADD_SUB( F[16], F[18]);
	ADD_SUB( F[17], F[19]);
	ADD_SUB( F[20], F[22]);
	ADD_SUB( F[21], F[23]);
	F[21] <<= 2;
	F[22] <<= 4;
	F[23] <<= 6;
	ADD_SUB( F[16], F[20] );
	ADD_SUB( F[17], F[21] );
	ADD_SUB( F[18], F[22] );
	ADD_SUB( F[19], F[23] );

	output[ 2] = Q_REDUCE( F[16] );
	output[10] = Q_REDUCE( F[17] );
	output[18] = Q_REDUCE( F[18] );
	output[26] = Q_REDUCE( F[19] );
	output[34] = Q_REDUCE( F[20] );
	output[42] = Q_REDUCE( F[21] );
	output[50] = Q_REDUCE( F[22] );
	output[58] = Q_REDUCE( F[23] );

	// Iteration 3:
	ADD_SUB( F[24], F[25] );
	ADD_SUB( F[26], F[27] );
	ADD_SUB( F[28], F[29] );
	ADD_SUB( F[30], F[31] );
 	F[27] <<= 4;
 	F[31] <<= 4;
	ADD_SUB( F[24], F[26] );
	ADD_SUB( F[25], F[27] );
	ADD_SUB( F[28], F[30] );
	ADD_SUB( F[29], F[31] );
	F[29] <<= 2;
	F[30] <<= 4;
	F[31] <<= 6;
	ADD_SUB( F[24], F[28] );
	ADD_SUB( F[25], F[29] );
	ADD_SUB( F[26], F[30] );
	ADD_SUB( F[27], F[31] );

	output[ 3] = Q_REDUCE( F[24] );
	output[11] = Q_REDUCE( F[25] );
	output[19] = Q_REDUCE( F[26] );
	output[27] = Q_REDUCE( F[27] );
	output[35] = Q_REDUCE( F[28] );
	output[43] = Q_REDUCE( F[29] );
	output[51] = Q_REDUCE( F[30] );
	output[59] = Q_REDUCE( F[31] );

	// Iteration 4:
	ADD_SUB( F[32], F[33] );
	ADD_SUB( F[34], F[35] );
	ADD_SUB( F[36], F[37] );
	ADD_SUB( F[38], F[39] );
	F[35] <<= 4;
	F[39] <<= 4;
	ADD_SUB( F[32], F[34] );
	ADD_SUB( F[33], F[35] );
	ADD_SUB( F[36], F[38] );
	ADD_SUB( F[37], F[39] );
	F[37] <<= 2;
	F[38] <<= 4;
	F[39] <<= 6;
	ADD_SUB( F[32], F[36] );
	ADD_SUB( F[33], F[37] );
	ADD_SUB( F[34], F[38] );
	ADD_SUB( F[35], F[39] );

	output[ 4] = Q_REDUCE( F[32] );
	output[12] = Q_REDUCE( F[33] );
	output[20] = Q_REDUCE( F[34] );
	output[28] = Q_REDUCE( F[35] );
	output[36] = Q_REDUCE( F[36] );
	output[44] = Q_REDUCE( F[37] );
	output[52] = Q_REDUCE( F[38] );
	output[60] = Q_REDUCE( F[39] );

	// Iteration 5:
	ADD_SUB( F[40], F[41] );
	ADD_SUB( F[42], F[43] );
	ADD_SUB( F[44], F[45] );
	ADD_SUB( F[46], F[47] );
	F[43] <<= 4;
	F[47] <<= 4;
	ADD_SUB( F[40], F[42] );
	ADD_SUB( F[41], F[43] );
	ADD_SUB( F[44], F[46] );
	ADD_SUB( F[45], F[47] );
	F[45] <<= 2;
	F[46] <<= 4;
	F[47] <<= 6;
	ADD_SUB( F[40], F[44] );
	ADD_SUB( F[41], F[45] );
	ADD_SUB( F[42], F[46] );
	ADD_SUB( F[43], F[47] );

	output[ 5] = Q_REDUCE( F[40] );
	output[13] = Q_REDUCE( F[41] );
	output[21] = Q_REDUCE( F[42] );
	output[29] = Q_REDUCE( F[43] );
	output[37] = Q_REDUCE( F[44] );
	output[45] = Q_REDUCE( F[45] );
	output[53] = Q_REDUCE( F[46] );
	output[61] = Q_REDUCE( F[47] );

	// Iteration 6:
	ADD_SUB( F[48], F[49] );
	ADD_SUB( F[50], F[51] );
	ADD_SUB( F[52], F[53] );
	ADD_SUB( F[54], F[55] );
	F[51] <<= 4;
	F[55] <<= 4;
	ADD_SUB( F[48], F[50] );
	ADD_SUB( F[49], F[51] );
	ADD_SUB( F[52], F[54] );
	ADD_SUB( F[53], F[55] );
	F[53] <<= 2;
	F[54] <<= 4;
	F[55] <<= 6;
	ADD_SUB( F[48], F[52] );
	ADD_SUB( F[49], F[53] );
	ADD_SUB( F[50], F[54] );
	ADD_SUB( F[51], F[55] );

	output[ 6] = Q_REDUCE( F[48] );
	output[14] = Q_REDUCE( F[49] );
	output[22] = Q_REDUCE( F[50] );
	output[30] = Q_REDUCE( F[51] );
	output[38] = Q_REDUCE( F[52] );
	output[46] = Q_REDUCE( F[53] );
	output[54] = Q_REDUCE( F[54] );
	output[62] = Q_REDUCE( F[55] );

	// Iteration 7:
	ADD_SUB( F[56], F[57] );
	ADD_SUB( F[58], F[59] );
	ADD_SUB( F[60], F[61] );
	ADD_SUB( F[62], F[63] );
	F[59] <<= 4;
	F[63] <<= 4;
	ADD_SUB( F[56], F[58] );
	ADD_SUB( F[57], F[59] );
	ADD_SUB( F[60], F[62] );
	ADD_SUB( F[61], F[63] );
	F[61] <<= 2;
	F[62] <<= 4;
	F[63] <<= 6;
	ADD_SUB( F[56], F[60] );
	ADD_SUB( F[57], F[61] );
	ADD_SUB( F[58], F[62] );
	ADD_SUB( F[59], F[63] );

	output[ 7] = Q_REDUCE( F[56] );
	output[15] = Q_REDUCE( F[57] );
	output[23] = Q_REDUCE( F[58] );
	output[31] = Q_REDUCE( F[59] );
	output[39] = Q_REDUCE( F[60] );
	output[47] = Q_REDUCE( F[61] );
	output[55] = Q_REDUCE( F[62] );
	output[63] = Q_REDUCE( F[63] );

   #undef ADD_SUB
   #undef Q_REDUCE

}

// ---- base-256 translation (verbatim reference, literal EIGHTH_N=8) -----------
__device__ static swift_int32_t swifftx_TranslateToBase256(swift_int32_t *input, unsigned char *output)
{
	swift_int32_t pairs[4];
	for (int i = 0; i < 8; i += 2)
		pairs[i >> 1] = input[i] + input[i + 1] + (input[i + 1] << 8);

	for (int i = 3; i > 0; --i)
		for (int j = i - 1; j < 3; ++j) {
			swift_int32_t temp = pairs[j] + pairs[j + 1] + (pairs[j + 1] << 9);
			pairs[j] = temp & 0xffff;
			pairs[j + 1] += (temp >> 16);
		}

	for (int i = 0; i < 8; i += 2) {
		output[i]     = (unsigned char)(pairs[i >> 1] & 0xff);
		output[i + 1] = (unsigned char)((pairs[i >> 1] >> 8) & 0xff);
	}
	return (pairs[3] >> 16);
}

__device__ static void swifftx_SWIFFTFFT(const unsigned char *input, int m, swift_int32_t *output)
{
	for (int i = 0; i < m; i++, input += 8, output += 64)
		swifftx_FFT(input, output);
}

// ---- SWIFFTSum: dot-product with A, mod-257 reduce, base-256 change ----------
__device__ static void swifftx_SWIFFTSum(const swift_int32_t *input, int m,
                                         unsigned char *output, const swift_int16_t *a)
{
	swift_int32_t result[64];
	int carry = 0;

	for (int j = 0; j < 64; ++j) {
		swift_int32_t sum = 0;
		const swift_int32_t *f = input + j;
		const swift_int16_t *k = a + j;
		for (int i = 0; i < m; i++, f += 64, k += 64)
			sum += (*f) * (*k);
		result[j] = sum;
	}

	for (int j = 0; j < 64; ++j)
		result[j] = ((257 << 22) + result[j]) % 257;

	for (int j = 0; j < 8; ++j) {
		int carryBit = swifftx_TranslateToBase256(result + (j << 3), output + (j << 3));
		carry |= carryBit << j;
	}
	output[64] = (unsigned char)carry;
}

// ---- ComputeSingleSWIFFTX: 256-byte input -> 64-byte digest ------------------
__device__ static void swifftx_Compute(const unsigned char *input, unsigned char *output)
{
	swift_int32_t fftOut[64 * 32];
	unsigned char sum[64 * 3 + 8];
	unsigned char carry0, carry1, carry2;

	swifftx_SWIFFTFFT(input, 32, fftOut);

	swifftx_SWIFFTSum(fftOut, 32, sum,        c_As);
	carry0 = sum[64];
	swifftx_SWIFFTSum(fftOut, 32, sum + 64,   c_As + 32 * 64);
	carry1 = sum[128];
	swifftx_SWIFFTSum(fftOut, 32, sum + 128,  c_As + 2 * 32 * 64);
	carry2 = sum[192];

	sum[192] = carry0;
	sum[193] = carry1;
	sum[194] = carry2;
	for (int i = 195; i < 200; ++i) sum[i] = 0;

	for (int i = 0; i < 200; ++i) sum[i] = c_SBox[sum[i]];

	swifftx_SWIFFTFFT(sum, 25, fftOut);
	swifftx_SWIFFTSum(fftOut, 25, sum, c_As);

	for (int i = 0; i < 64; ++i) output[i] = sum[i];
}

// Pipeline SWIFFTX: one thread per hash (SWIFFTX is compute-bound, so the plain
// embarrassingly-parallel mapping keeps the ALUs saturated). Reads the 256-byte
// window (accumulator slots 12..15), writes the 64-byte digest to a flat d_hash.
__global__ void x25x_swifftx_acc_gpu(uint32_t threads, const uint8_t *g_acc, uint8_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		// slot-major accumulator: gather the 256-byte window from planes 12..15.
		unsigned char in[256], out[64];
		#pragma unroll
		for (int k = 0; k < 4; k++) {
			const uint8_t *plane = g_acc + (size_t)(12 + k) * threads * 64 + (size_t)thread * 64;
			#pragma unroll
			for (int i = 0; i < 64; i++) in[k * 64 + i] = plane[i];
		}

		swifftx_Compute(in, out);

		uint8_t *pout = g_hash + (size_t)thread * 64;
		#pragma unroll
		for (int i = 0; i < 64; i++) pout[i] = out[i];
	}
}

__host__ void x25x_swifftx_cpu_hash_acc(int thr_id, uint32_t threads, uint32_t *d_acc, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x25x_swifftx_acc_gpu<<<grid, block>>>(threads, (const uint8_t*)d_acc, (uint8_t*)d_hash);
}

// Build the tables on the host once, then upload to the device symbols.
extern "C" {
	void InitializeSWIFFTX();
	extern swift_int16_t multipliers[];
	extern swift_int16_t fftTable[];
	extern const swift_int16_t As[];
	extern unsigned char SBox[];
}

__host__ void x25x_swifftx_cpu_init(int thr_id)
{
	InitializeSWIFFTX();
	cudaMemcpyToSymbol(c_multipliers, multipliers, sizeof(swift_int16_t) * 64);
	cudaMemcpyToSymbol(c_fftTable,    fftTable,    sizeof(swift_int16_t) * 256 * 8);
	cudaMemcpyToSymbol(c_As,          As,          sizeof(swift_int16_t) * 3 * 32 * 64);
	cudaMemcpyToSymbol(c_SBox,        SBox,        256);
}
