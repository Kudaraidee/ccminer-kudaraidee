/**
 * Lyra2 matrix row operations for the shared-memory-resident variants.
 *
 * The includer must define the matrix shape before including this header, and
 * include cuda/blake2b_device.cuh first (for round_lyra):
 *
 *     #define Nrow      8
 *     #define Ncol      8
 *     #define memshift  3
 *     #define BUF_COUNT 0     // rows kept in DMatrix instead of shared memory
 *     #define LYRA2_TPB 32    // the wander kernel's threads/block
 *
 * The wander kernel always runs block(4, LYRA2_TPB/4), so the includer must
 * launch it as block(LYRA2_BDIMX, LYRA2_BDIMY) rather than repeating the
 * numbers -- that is what keeps launcher and kernel in step.
 *
 * Layout: 4 lanes per hash (LYRA2_BDIMX == 4), lane L holding BLAKE2b state
 * column L, so one 96-byte matrix column is memshift uint2 per lane. LD4S/ST4S
 * address either dynamic shared memory or DMatrix depending on BUF_COUNT, which
 * is why they are macro-driven rather than templated.
 *
 * "rand" below is the sponge state after the reduced round; rotW is Lyra2's
 * one-word rotation of it, which in this layout is the preceding lane's words
 * (lane 0 additionally shifts by one index).
 */
#ifndef CUDA_LYRA2_DEVICE_CUH
#define CUDA_LYRA2_DEVICE_CUH

#ifndef LYRA2_TPB
#error "lyra2_device.cuh: define LYRA2_TPB (the wander kernel's threads/block) first"
#endif

/* 4 lanes per hash, so the block is always (4, LYRA2_TPB/4). ptxas cannot
 * strength-reduce the index multiplies unless these are compile-time. */
#define LYRA2_BDIMX 4
#define LYRA2_BDIMY (LYRA2_TPB / LYRA2_BDIMX)

/* Runtime block size on purpose: a compile-time index lets ptxas fully unroll the
 * Lyra2Z wander, which is slower (#pragma unroll does not stop it). */
#define LYRA2_IDX_X blockDim.x
#define LYRA2_IDX_Y blockDim.y

__device__ __forceinline__ void LD4S(uint2 res[3], const int row, const int col, const int thread, const int threads)
{
#if BUF_COUNT != 8
	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
#endif
#if BUF_COUNT != 0
	const int d0 = (memshift *(Ncol * row + col) * threads + thread)*LYRA2_IDX_X + threadIdx.x;
#endif

#if BUF_COUNT == 8
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = *(DMatrix + d0 + j * threads * LYRA2_IDX_X);
#elif BUF_COUNT == 0
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = shared_mem[((s0 + j) * LYRA2_IDX_Y + threadIdx.y) * LYRA2_IDX_X + threadIdx.x];
#else
	if (row < BUF_COUNT)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
			res[j] = *(DMatrix + d0 + j * threads * LYRA2_IDX_X);
	}
	else
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			res[j] = shared_mem[((s0 + j) * LYRA2_IDX_Y + threadIdx.y) * LYRA2_IDX_X + threadIdx.x];
	}
#endif
}

__device__ __forceinline__ void ST4S(const int row, const int col, const uint2 data[3], const int thread, const int threads)
{
#if BUF_COUNT != 8
	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
#endif
#if BUF_COUNT != 0
	const int d0 = (memshift *(Ncol * row + col) * threads + thread)*LYRA2_IDX_X + threadIdx.x;
#endif

#if BUF_COUNT == 8
	#pragma unroll
	for (int j = 0; j < 3; j++)
		*(DMatrix + d0 + j * threads * LYRA2_IDX_X) = data[j];

#elif BUF_COUNT == 0
	#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s0 + j) * LYRA2_IDX_Y + threadIdx.y) * LYRA2_IDX_X + threadIdx.x] = data[j];

#else
	if (row < BUF_COUNT)
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + d0 + j * threads * LYRA2_IDX_X) = data[j];
	}
	else
	{
	#pragma unroll
		for (int j = 0; j < 3; j++)
			shared_mem[((s0 + j) * LYRA2_IDX_Y + threadIdx.y) * LYRA2_IDX_X + threadIdx.x] = data[j];
	}
#endif
}

// Setup phase, rows 0 and 1: reducedSqueezeRow0 then reducedDuplexRow1
static __device__ __forceinline__
void reduceDuplex(uint2 state[4], uint32_t thread, const uint32_t threads)
{
	uint2 state1[3];

	#pragma unroll
	for (int i = 0; i < Ncol; i++)
	{
		ST4S(0, Ncol - i - 1, state, thread, threads);

		round_lyra(state);
	}

	#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
		LD4S(state1, 0, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];
		ST4S(1, Ncol - i - 1, state1, thread, threads);
	}
}

// Setup phase, rows 2..Nrow-1: M[rowOut] written back-to-front, M[rowInOut]
// rotW-updated
static __device__ __forceinline__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];

	#pragma unroll 1
	for (int i = 0; i < Ncol; i++)
	{
		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, Ncol - i - 1, state1, thread, threads);

		// simultaneously receive data from preceding thread and send data to following thread
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);
	}
}

// Wandering phase: M[rowOut] and M[rowInOut] are both XOR-updated
static __device__ __forceinline__
void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	for (int i = 0; i < Ncol; i++)
	{
		uint2 state1[3], state2[3];

		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		// simultaneously receive data from preceding thread and send data to following thread
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);

		LD4S(state1, rowOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, i, state1, thread, threads);
	}
}

#endif // CUDA_LYRA2_DEVICE_CUH
