/*
 * RinHash = BLAKE3-256 -> Argon2d -> SHA3-256, one nonce per 32 cooperating
 * threads.
 *
 *   rin_coop_initialize  BLAKE3(header) -> H0 -> blocks 0 and 1
 *   argon2_fill          blocks 2..63 over two passes; it also writes the
 *                        final XOR back into block 0
 *   rin_coop_finalize    block 0 -> blake2b-long(32) -> SHA3-256 -> compare
 *
 * argon2_fill is algos/argon2d/'s, used verbatim: its arguments are geometry
 * only and five other algorithms depend on it, so it is called, never edited.
 * Only the two ends are local, because the argon2d coins hardwire their own
 * convention there (pwdlen = saltlen = 80, salt == password).
 *
 * Memory contract is argon2_fill's: each nonce owns lanes * 4 * segment_blocks
 * contiguous 1 KB blocks -- at lanes=1, segment_blocks=16 that is the 64 blocks
 * RinHash's m_cost asks for.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

#include "algos/argon2d/argon2d_blake2b_device.cuh"
#include "cuda/blake3_device.cuh"
#include "cuda/sha3_device.cuh"
#include "miner.h"

/* RinHash's Argon2d geometry: t=2, m=64 KiB, lanes=1, v1.3, Argon2d, out 32. */
#define RIN_LANES            1u
#define RIN_PASSES           2u
#define RIN_MCOST            64u
#define RIN_SEGMENT_BLOCKS   (RIN_MCOST / (4u * RIN_LANES))   /* 16 */
#define RIN_TOTAL_BLOCKS     (RIN_SEGMENT_BLOCKS * 4u * RIN_LANES) /* 64 */
#define RIN_VERSION          0x13u

/* Nonces per CUDA block in the two end kernels. fillFirstBlock() derives the
 * job index as blockIdx.x * blockDim.y + threadIdx.y, so this is blockDim.y and
 * the nonce count must be a multiple of it. */
#define RIN_JOBS_PER_BLOCK   16u

/* Defined in algos/argon2d/argon2d_fill.cu. NOT extern "C": it ships with C++
 * linkage there, and argon2d.cu already launches it across the same boundary. */
__global__ void argon2_fill(
        struct block_g *memory, uint32_t passes, uint32_t lanes,
        uint32_t segment_blocks, uint32_t version, uint32_t type);

__constant__ uint32_t c_rin_header[20];
__constant__ uint32_t c_rin_target[8];

/* H0 = Blake2b-512 over the Argon2 pre-hash input:
 *
 *   LE32 lanes | LE32 outlen | LE32 m_cost | LE32 t_cost | LE32 version |
 *   LE32 type  | LE32 pwdlen | pwd(32) | LE32 saltlen | salt(11) |
 *   LE32 secretlen(0) | LE32 adlen(0)
 *
 * = 83 bytes, so one final Blake2b block. computeInitialHash() cannot be shared
 * because it hardcodes the coins' 80-byte password and salt.
 *
 * secretlen and adlen land at byte offsets 75 and 79, which are NOT
 * word-aligned; they are left as the zeroes the buffer already holds. */
__device__ __forceinline__
void rin_initial_hash(uint32_t nonce, uint32_t *buffer /* 32 words */)
{
	uint32_t hdr[20];
	#pragma unroll
	for (int i = 0; i < 20; i++) hdr[i] = c_rin_header[i];
	hdr[19] = nonce;

	uint8_t pwd[32];
	blake3_256((const uint8_t *)hdr, 80, pwd);

	uint8_t *p = (uint8_t *)buffer;
	#pragma unroll
	for (int i = 0; i < 32; i++) buffer[i] = 0;

	buffer[0] = RIN_LANES;
	buffer[1] = ALGO_OUTLEN;      /* 32 */
	buffer[2] = RIN_MCOST;
	buffer[3] = RIN_PASSES;
	buffer[4] = RIN_VERSION;
	buffer[5] = ARGON2_D;         /* 0 */
	buffer[6] = 32;               /* pwdlen */
	#pragma unroll
	for (int i = 0; i < 32; i++) p[28 + i] = pwd[i];
	buffer[15] = 11;              /* saltlen, byte offset 60 */
	p[64] = 'R'; p[65] = 'i'; p[66] = 'n'; p[67] = 'C'; p[68] = 'o'; p[69] = 'i';
	p[70] = 'n'; p[71] = 'S'; p[72] = 'a'; p[73] = 'l'; p[74] = 't';

	uint64x8 state;
	state.s0 = blake2b_Init[0]; state.s1 = blake2b_Init[1];
	state.s2 = blake2b_Init[2]; state.s3 = blake2b_Init[3];
	state.s4 = blake2b_Init[4]; state.s5 = blake2b_Init[5];
	state.s6 = blake2b_Init[6]; state.s7 = blake2b_Init[7];

	blake2b_compress_1w(&state, (uint64_t *)buffer, 1, true, 83);

	/* fillFirstBlock() wants H0 in buffer[1..16] with the rest zero: it adds
	 * the 1024 outlen at [0] and the row/column at [17]/[18]. */
	#pragma unroll
	for (int i = 0; i < 32; i++) buffer[i] = 0;
	uint64_t *b64 = (uint64_t *)&buffer[1];
	b64[0] = state.s0; b64[1] = state.s1; b64[2] = state.s2; b64[3] = state.s3;
	b64[4] = state.s4; b64[5] = state.s5; b64[6] = state.s6; b64[7] = state.s7;
}

/* Blocks 0 and 1 of every nonce. threadIdx.x selects the block (row), as
 * fillFirstBlock() expects; both threads recompute H0, which is one BLAKE3 of
 * 80 bytes. */
__global__ void rin_coop_initialize(struct block *memory, uint32_t start_nonce)
{
	uint32_t buffer[32];
	const uint32_t nonce = start_nonce + (blockIdx.x * blockDim.y + threadIdx.y);

	rin_initial_hash(nonce, buffer);
	fillFirstBlock(memory, buffer, RIN_LANES, RIN_TOTAL_BLOCKS);
}

/* Block 0 holds argon2_fill's final XOR. Tag = blake2b-long(block, 32), which
 * for outlen <= 64 is Blake2b(LE32(32) || block) -- 1028 bytes, so eight full
 * compressions and a 4-byte tail. Then SHA3-256 and the target compare.
 *
 * Four threads per nonce, each owning two Blake2b state words, as
 * blake2b_compress_4w() requires. */
__global__ void rin_coop_finalize(struct block *memory, uint32_t start_nonce,
                                  uint32_t span,
                                  uint32_t *solution_found, uint32_t *solution_nonce)
{
	extern __shared__ uint32_t s_buf[];
	uint32_t *input = &s_buf[threadIdx.y * 258];
	uint64_t *input_64 = (uint64_t *)input;

	const uint32_t idx = threadIdx.x;
	const uint32_t job = blockIdx.x * blockDim.y + threadIdx.y;
	const uint32_t nonce = start_nonce + job;

	/* The launch rounds up to whole 16-nonce blocks; this tail was filled but
	 * never asked for, so it must not win the report slot. Do NOT return early:
	 * that diverges the warp before the barriers below. Guard the report. */
	const bool wanted = (job < span);

	uint32_t *mem_lane = (uint32_t *)(memory + (size_t)job * RIN_TOTAL_BLOCKS);
	partialState state;

	load_block(&input[1], mem_lane, idx);
	input[0] = ALGO_OUTLEN;

	/* Each thread wrote a quarter of the block; every thread now reads all of
	 * it. Without this the code is warp-synchronous by assumption, which Volta's
	 * independent thread scheduling makes unsafe. */
	__syncwarp();

	state.a = blake2b_Init_928[idx];
	state.b = blake2b_Init_928[idx + 4];

	blake2b_compress_4w(&state, &input_64[0],   1, idx);
	blake2b_compress_4w(&state, &input_64[16],  2, idx);
	blake2b_compress_4w(&state, &input_64[32],  3, idx);
	blake2b_compress_4w(&state, &input_64[48],  4, idx);
	blake2b_compress_4w(&state, &input_64[64],  5, idx);
	blake2b_compress_4w(&state, &input_64[80],  6, idx);
	blake2b_compress_4w(&state, &input_64[96],  7, idx);
	blake2b_compress_4w(&state, &input_64[112], 8, idx);

	/* Same shape: the four threads clear the buffer between them, then all four
	 * read it. input[256] must be read before the clear can reach it. */
	const uint32_t tail = input[256];
	__syncwarp();
	zero_buffer(&input[0], idx);
	input[0] = tail;
	__syncwarp();
	blake2b_compress_4w(&state, &input_64[0], 9, idx, true, 4);

	/* The compress above read this buffer and the store below overwrites it:
	 * without a barrier, thread 0 can clobber a word thread 3 is still
	 * reading. */
	__syncwarp();
	input_64[idx] = state.a;
	__syncwarp();

	if (idx != 0 || !wanted) return;

	/* input[0..7] is the 32-byte Argon2d tag. */
	uint32_t hash_words[8];
	sha3_256_32((const uint8_t *)input, hash_words);

	bool meets = true;
	for (int i = 7; i >= 0; i--) {
		if (hash_words[i] > c_rin_target[i]) { meets = false; break; }
		if (hash_words[i] < c_rin_target[i]) break;
	}

	/* Report the LOWEST candidate of the launch (the slot is preset to
	 * 0xFFFFFFFF): the host resumes at reported + 1, so every other candidate
	 * is above it and found again. A first-come slot would lose the ones below.
	 * The digest goes to this nonce's own lane (its block 0 was read above and
	 * the next launch re-initializes it); the host fetches the reported one. */
	if (meets) {
		for (int w = 0; w < 8; w++) mem_lane[w] = hash_words[w];
		atomicMin(solution_nonce, nonce);
		atomicExch(solution_found, 1u);
	}
}

/* ------------------------------------------------------------------ *
 * Host side: persistent device memory and the three-kernel launch.
 * ------------------------------------------------------------------ */
static struct {
	struct block *d_memory;
	uint32_t *d_solution_found;
	uint32_t *d_solution_nonce;
	uint32_t *h_pinned;              /* found | nonce | hash[8] */
	uint32_t  max_nonces;
	cudaStream_t stream;
	bool init;
} g_coop[MAX_GPUS];   /* per GPU thread: each lives on its own device */

static void coop_check(const char *where)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		applog(LOG_ERR, "rinhash coop: %s: %s", where, cudaGetErrorString(e));
}

extern "C" void rinhash_coop_reserve(int thr_id, uint32_t max_nonces)
{
	max_nonces = (max_nonces / RIN_JOBS_PER_BLOCK) * RIN_JOBS_PER_BLOCK;
	if (max_nonces == 0) max_nonces = RIN_JOBS_PER_BLOCK;
	if (g_coop[thr_id].init && max_nonces <= g_coop[thr_id].max_nonces) return;

	if (g_coop[thr_id].init) {
		cudaFree(g_coop[thr_id].d_memory);
		cudaFree(g_coop[thr_id].d_solution_found);
		cudaFreeHost(g_coop[thr_id].h_pinned);
	} else {
		cudaStreamCreate(&g_coop[thr_id].stream);
	}

	const size_t bytes = (size_t)max_nonces * RIN_TOTAL_BLOCKS * ARGON2_BLOCK_SIZE;
	cudaMalloc(&g_coop[thr_id].d_memory, bytes);
	/* one allocation for the two report words: found | nonce */
	cudaMalloc(&g_coop[thr_id].d_solution_found, 2 * sizeof(uint32_t));
	g_coop[thr_id].d_solution_nonce = g_coop[thr_id].d_solution_found + 1;
	cudaMallocHost(&g_coop[thr_id].h_pinned, 10 * sizeof(uint32_t));
	g_coop[thr_id].max_nonces = max_nonces;
	g_coop[thr_id].init = true;
	coop_check("reserve");
	applog(LOG_INFO, "RinHash cooperative fill: %u nonces (%.2f MB VRAM)",
	       max_nonces, bytes / (1024.0 * 1024.0));
}

/* Nonces per call: 256 per SM, which scales the footprint with the card.
 * Throughput is insensitive to this value, because the block lives in
 * registers rather than in a per-thread cache stream. */
extern "C" uint32_t rinhash_coop_batch_for_mps(int mpcount)
{
	const uint32_t mp = (mpcount > 0) ? (uint32_t)mpcount : 28u;
	return mp * 256u;
}

extern "C" void rinhash_coop_cleanup(int thr_id)
{
	if (!g_coop[thr_id].init) return;
	cudaFree(g_coop[thr_id].d_memory);
	cudaFree(g_coop[thr_id].d_solution_found);
	cudaFreeHost(g_coop[thr_id].h_pinned);
	cudaStreamDestroy(g_coop[thr_id].stream);
	g_coop[thr_id].init = false;
	g_coop[thr_id].max_nonces = 0;
}

/* The launcher scanhash calls. */
extern "C" void RinHash_mine_coop(int thr_id,
    const uint32_t *work_data, uint32_t nonce_offset, uint32_t start_nonce,
    uint32_t num_nonces, uint32_t *target, uint32_t *found_nonce,
    uint8_t *target_hash, uint8_t *best_hash, uint32_t *solution_found,
    uint32_t *hashes_actually_done)
{
	(void)nonce_offset; (void)target_hash;

	/* Hash exactly what was asked for: the geometry rounds up to whole blocks
	 * and the finalize kernel ignores the padded tail, so no nonce outside
	 * [start, start+num_nonces) can be reported. */
	if (num_nonces == 0) return;
	uint32_t launched = ((num_nonces + RIN_JOBS_PER_BLOCK - 1) / RIN_JOBS_PER_BLOCK)
	                    * RIN_JOBS_PER_BLOCK;
	rinhash_coop_reserve(thr_id, launched);
	if (launched > g_coop[thr_id].max_nonces) {
		launched = g_coop[thr_id].max_nonces;
		if (num_nonces > launched) num_nonces = launched;
	}

	cudaStream_t st = g_coop[thr_id].stream;
	cudaMemcpyToSymbolAsync(c_rin_header, work_data, 80, 0, cudaMemcpyHostToDevice, st);
	cudaMemcpyToSymbolAsync(c_rin_target, target, 8 * sizeof(uint32_t), 0,
	                        cudaMemcpyHostToDevice, st);
	/* found = 0; nonce = 0xFFFFFFFF, the atomicMin identity */
	cudaMemsetAsync(g_coop[thr_id].d_solution_found, 0, sizeof(uint32_t), st);
	cudaMemsetAsync(g_coop[thr_id].d_solution_nonce, 0xFF, sizeof(uint32_t), st);

	const uint32_t jobs = launched / RIN_JOBS_PER_BLOCK;

	rin_coop_initialize<<<jobs, dim3(2 * RIN_LANES, RIN_JOBS_PER_BLOCK), 0, st>>>(
	        (struct block *)g_coop[thr_id].d_memory, start_nonce);

	argon2_fill<<<dim3(1, 1, launched), dim3(THREADS_PER_LANE, RIN_LANES, 1),
	              RIN_LANES * ARGON2_SHARED_BLOCKS_PER_LANE * sizeof(block_g), st>>>(
	        (struct block_g *)g_coop[thr_id].d_memory, RIN_PASSES, RIN_LANES,
	        RIN_SEGMENT_BLOCKS, RIN_VERSION, ARGON2_D);

	rin_coop_finalize<<<jobs, dim3(4, RIN_JOBS_PER_BLOCK),
	                    RIN_JOBS_PER_BLOCK * 258 * sizeof(uint32_t), st>>>(
	        (struct block *)g_coop[thr_id].d_memory, start_nonce, num_nonces,
	        g_coop[thr_id].d_solution_found, g_coop[thr_id].d_solution_nonce);

	cudaMemcpyAsync(g_coop[thr_id].h_pinned, g_coop[thr_id].d_solution_found,
	                2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
	cudaStreamSynchronize(st);
	coop_check("launch");

	/* Only reported nonces in [start, start+num_nonces) can be set (the
	 * kernel guards the padded tail); check anyway before indexing a lane. */
	*solution_found = g_coop[thr_id].h_pinned[0] &&
	                  g_coop[thr_id].h_pinned[1] - start_nonce < num_nonces;
	/* Nothing is abandoned -- no wave loop -- so this is exactly the span. */
	*hashes_actually_done = num_nonces;

	if (*solution_found) {
		*found_nonce = g_coop[thr_id].h_pinned[1];
		/* the reported nonce's digest, left in block 0 of its own lane */
		const size_t lane = (size_t)(*found_nonce - start_nonce) * RIN_TOTAL_BLOCKS;
		cudaMemcpyAsync(g_coop[thr_id].h_pinned + 2,
		                (const uint32_t *)(g_coop[thr_id].d_memory + lane),
		                8 * sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
		cudaStreamSynchronize(st);
		coop_check("digest");
		for (int i = 0; i < 8; i++)
			((uint32_t *)best_hash)[i] = g_coop[thr_id].h_pinned[2 + i];
	} else {
		*found_nonce = UINT32_MAX;
		for (int i = 0; i < 8; i++) ((uint32_t *)best_hash)[i] = 0;
	}
}
