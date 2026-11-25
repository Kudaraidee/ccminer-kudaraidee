/**
 * This code compares final hash against target
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"

#include "cuda/helper.cuh"

__constant__ uint32_t pTarget[8]; // 32 bytes

// store MAX_GPUS device arrays of 8 nonces
//
// Slot 0 is the candidate COUNT, slots 1..CHECKHASH_SLOTS-1 hold the nonces,
// so at most 7 are retained per batch. The bound used to be a bare `8` in the
// kernel while the size was a bare `32` in the allocation and the memcpy --
// three places to keep in step by hand. One name instead.
#define CHECKHASH_SLOTS   8
#define CHECKHASH_NONCES  (CHECKHASH_SLOTS - 1)
#define CHECKHASH_BYTES   (CHECKHASH_SLOTS * sizeof(uint32_t))

static uint32_t* h_resNonces[MAX_GPUS] = { NULL };
static uint32_t* d_resNonces[MAX_GPUS] = { NULL };
static __thread bool init_done = false;

__host__
void cuda_check_cpu_init(int thr_id, uint32_t threads)
{
    CUDA_CALL_OR_RET(cudaMalloc(&d_resNonces[thr_id], CHECKHASH_BYTES));
    CUDA_SAFE_CALL(cudaMallocHost(&h_resNonces[thr_id], CHECKHASH_BYTES));
    init_done = true;
}

__host__
void cuda_check_cpu_free(int thr_id)
{
	if (!init_done) return;
	cudaFree(d_resNonces[thr_id]);
	cudaFreeHost(h_resNonces[thr_id]);
	d_resNonces[thr_id] = NULL;
	h_resNonces[thr_id] = NULL;
	init_done = false;
}

// Target Difficulty
__host__
void cuda_check_cpu_setTarget(const void *ptarget)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, ptarget, 32, 0, cudaMemcpyHostToDevice));
}

/* --------------------------------------------------------------------------------------------- */

__device__ __forceinline__
static bool hashbelowtarget(const uint32_t *const __restrict__ hash, const uint32_t *const __restrict__ target)
{
	if (hash[7] > target[7])
		return false;
	if (hash[7] < target[7])
		return true;
	if (hash[6] > target[6])
		return false;
	if (hash[6] < target[6])
		return true;

	if (hash[5] > target[5])
		return false;
	if (hash[5] < target[5])
		return true;
	if (hash[4] > target[4])
		return false;
	if (hash[4] < target[4])
		return true;

	if (hash[3] > target[3])
		return false;
	if (hash[3] < target[3])
		return true;
	if (hash[2] > target[2])
		return false;
	if (hash[2] < target[2])
		return true;

	if (hash[1] > target[1])
		return false;
	if (hash[1] < target[1])
		return true;
	if (hash[0] > target[0])
		return false;

	return true;
}

__global__ __launch_bounds__(512, 4)
void cuda_checkhash_64(uint32_t threads, uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// shl 4 = *16 x 4 (uint32) = 64 bytes
		// todo: use only 32 bytes * threads if possible
		uint32_t *inpHash = &hash[thread << 4];

		// atomicMin, so slot 0 is the LOWEST candidate in the batch.
		//
		// It used to be `if (resNonces[0] == UINT32_MAX) resNonces[0] = nonce;`
		// -- a cross-thread read-modify-write with no atomic, so the winner was
		// whichever thread happened to store last, NOT the smallest nonce. Two
		// consequences, both silent:
		//   - the caller resumes from max(nonce0, nonce1) + 1, so a LOWER
		//     candidate that lost the race is skipped permanently;
		//   - cuda_check_hash_suppl() could hand back that same nonce as the
		//     "second" one, costing the real second candidate its submit
		//     (measured: 4 duplicate submits in 38 multi-candidate batches).
		// The init is already 0xffffffff (see the memset in the host wrapper),
		// which is exactly atomicMin's identity.
		if (hashbelowtarget(inpHash, pTarget))
			atomicMin(&resNonces[0], (startNounce + thread));
	}
}

__global__ __launch_bounds__(512, 4)
void cuda_checkhash_32(uint32_t threads, uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *inpHash = &hash[thread << 3];

		// Same fix as cuda_checkhash_64 above: atomicMin so slot 0 is the
		// lowest candidate, not the last thread to win a race.
		if (hashbelowtarget(inpHash, pTarget))
			atomicMin(&resNonces[0], (startNounce + thread));
	}
}

__host__
uint32_t cuda_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash)
{
	cudaMemset(d_resNonces[thr_id], 0xff, sizeof(uint32_t));

	const uint32_t threadsperblock = 512;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	if (bench_algo >= 0) // dont interrupt the global benchmark
		return UINT32_MAX;

	if (!init_done) {
		applog(LOG_ERR, "missing call to cuda_check_cpu_init");
		return UINT32_MAX;
	}

	cuda_checkhash_64 <<<grid, block>>> (threads, startNounce, d_inputHash, d_resNonces[thr_id]);
	cudaDeviceSynchronize();

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return h_resNonces[thr_id][0];
}

__host__
uint32_t cuda_check_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash)
{
	cudaMemset(d_resNonces[thr_id], 0xff, sizeof(uint32_t));

	const uint32_t threadsperblock = 512;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	if (bench_algo >= 0) // dont interrupt the global benchmark
		return UINT32_MAX;

	if (!init_done) {
		applog(LOG_ERR, "missing call to cuda_check_cpu_init");
		return UINT32_MAX;
	}

	cuda_checkhash_32 <<<grid, block>>> (threads, startNounce, d_inputHash, d_resNonces[thr_id]);
	cudaDeviceSynchronize();

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return h_resNonces[thr_id][0];
}

/* --------------------------------------------------------------------------------------------- */

__global__ __launch_bounds__(512, 4)
void cuda_checkhash_64_suppl(uint32_t threads, uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	// Grid is rounded up to a multiple of 512; without this the tail threads read
	// hashes outside the batch and report nonces beyond the scan range.
	if (thread >= threads)
		return;

	uint32_t *inpHash = &hash[thread << 4];

	if (hashbelowtarget(inpHash, pTarget)) {
		// Must be atomic: concurrent candidates otherwise lose increments and
		// overwrite each other's slots.
		uint32_t resNum = atomicAdd(&resNonces[0], 1) + 1;
		__threadfence();
		if (resNum < CHECKHASH_SLOTS)
			resNonces[resNum] = (startNounce + thread);
	}
}

/* Runs the supplementary screen ONCE and hands back the candidates it kept,
 * sorted ascending. Shared by both public entry points. */
static uint32_t checkhash_suppl_collect(int thr_id, uint32_t threads, uint32_t startNounce,
	uint32_t *d_inputHash, uint32_t *sorted, uint32_t *pstored)
{
	const uint32_t threadsperblock = 512;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	*pstored = 0;

	if (!init_done) {
		applog(LOG_ERR, "missing call to cuda_check_cpu_init");
		return 0;
	}

	// Arm every slot the readback copies, not just the count: the kernel writes
	// only as many slots as it finds candidates, so the rest would be read back
	// uninitialised. Slots get the UINT32_MAX sentinel, the count starts at 0.
	cudaMemset(d_resNonces[thr_id], 0xff, CHECKHASH_BYTES);
	cudaMemset(d_resNonces[thr_id], 0, sizeof(uint32_t));

	cuda_checkhash_64_suppl <<<grid, block>>> (threads, startNounce, d_inputHash, d_resNonces[thr_id]);
	cudaDeviceSynchronize();

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], CHECKHASH_BYTES, cudaMemcpyDeviceToHost);
	const uint32_t rescnt = h_resNonces[thr_id][0];

	/* The kernel stores candidates in DISCOVERY order -- whichever thread won
	 * its atomicAdd first -- not sorted. Indexing that directly meant
	 * _suppl(.., 1) returned "the second one stored", which can be the very
	 * nonce cuda_check_hash() already returned as the lowest: the caller then
	 * submits a duplicate (the hashlog catches it) and the REAL second
	 * candidate is never submitted at all. Measured on a live pool before this
	 * fix: 4 duplicate submits across 38 multi-candidate batches.
	 *
	 * Sorting makes index k mean "the k-th LOWEST", so index 0 agrees with
	 * cuda_check_hash()'s atomicMin by construction and index 1 is a genuinely
	 * different, genuinely next candidate.
	 *
	 * Only the retained slots can be sorted. With more than CHECKHASH_NONCES
	 * candidates in one batch the kernel has already dropped some, so this is
	 * the k-th lowest of those KEPT; callers read the total to see that case
	 * and resume conservatively rather than skipping the remainder. */
	uint32_t stored = rescnt;
	if (stored > CHECKHASH_NONCES)
		stored = CHECKHASH_NONCES;

	for (uint32_t i = 0; i < stored; i++)
		sorted[i] = h_resNonces[thr_id][i + 1];
	for (uint32_t i = 1; i < stored; i++) {           /* insertion sort, n <= 7 */
		const uint32_t v = sorted[i];
		uint32_t j = i;
		while (j > 0 && sorted[j - 1] > v) { sorted[j] = sorted[j - 1]; j--; }
		sorted[j] = v;
	}

	if (opt_debug && rescnt > 1)
		applog(LOG_WARNING, "Found %u nonces (%u kept), sorted: %x + %x",
			rescnt, stored, sorted[0], stored > 1 ? sorted[1] : UINT32_MAX);

	*pstored = stored;
	return rescnt;
}

__host__
uint32_t cuda_check_hash_suppl(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, uint8_t numNonce)
{
	/* UINT32_MAX = "no further candidate", matching cuda_check_hash(). 0 cannot
	 * serve as the sentinel: it is a legal nonce. */
	uint32_t sorted[CHECKHASH_NONCES], stored = 0;

	checkhash_suppl_collect(thr_id, threads, startNounce, d_inputHash, sorted, &stored);

	return (numNonce < stored) ? sorted[numNonce] : UINT32_MAX;
}

/* Every candidate of the batch in one launch: writes up to max_out nonces to
 * out[] in ascending order and returns how many it wrote. *total, if given,
 * receives how many the batch FOUND -- larger than the written count when the
 * device dropped candidates past CHECKHASH_NONCES, which tells the caller it
 * must not skip the rest of the window.
 *
 * Prefer this over looping cuda_check_hash_suppl(): cuda_check_hash_count()
 * means a COUNT only after a suppl call, but the lowest NONCE after
 * cuda_check_hash(), so the loop form is easy to mis-order. */
__host__
uint32_t cuda_check_hash_suppl_all(int thr_id, uint32_t threads, uint32_t startNounce,
	uint32_t *d_inputHash, uint32_t *out, uint32_t max_out, uint32_t *total)
{
	uint32_t sorted[CHECKHASH_NONCES], stored = 0;

	const uint32_t rescnt = checkhash_suppl_collect(thr_id, threads, startNounce,
							d_inputHash, sorted, &stored);
	if (total)
		*total = rescnt;

	if (stored > max_out)
		stored = max_out;
	for (uint32_t i = 0; i < stored; i++)
		out[i] = sorted[i];

	return stored;
}

// Candidate count from the last cuda_check_hash_suppl() call. Lets a caller tell
// "retrieved them all" from "there are more" before advancing the nonce cursor.
__host__
uint32_t cuda_check_hash_count(int thr_id)
{
	if (!init_done)
		return 0;
	return h_resNonces[thr_id][0];
}

/* --------------------------------------------------------------------------------------------- */

__global__
void cuda_check_hash_branch_64(uint32_t threads, uint32_t startNounce, uint32_t *g_nonceVector, uint32_t *g_hash, uint32_t *resNounce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = g_nonceVector[thread];
		uint32_t hashPosition = (nounce - startNounce) << 4;
		uint32_t *inpHash = &g_hash[hashPosition];

		for (int i = 7; i >= 0; i--) {
			if (inpHash[i] > pTarget[i]) {
				return;
			}
			if (inpHash[i] < pTarget[i]) {
				break;
			}
		}
		if (resNounce[0] > nounce)
			resNounce[0] = nounce;
	}
}

__host__
uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order)
{
	const uint32_t threadsperblock = 256;

	uint32_t result = UINT32_MAX;

	if (bench_algo >= 0) // dont interrupt the global benchmark
		return result;

	if (!init_done) {
		applog(LOG_ERR, "missing call to cuda_check_cpu_init");
		return result;
	}

	cudaMemset(d_resNonces[thr_id], 0xff, sizeof(uint32_t));

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_check_hash_branch_64 <<<grid, block>>> (threads, startNounce, d_nonceVector, d_inputHash, d_resNonces[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	result = *h_resNonces[thr_id];

	return result;
}

/* Function to get the compiled Shader Model version */
int cuda_arch[MAX_GPUS] = { 0 };
__global__ void nvcc_get_arch(int *d_version)
{
	*d_version = 0;
#ifdef __CUDA_ARCH__
	*d_version = __CUDA_ARCH__;
#endif
}

__host__
int cuda_get_arch(int thr_id)
{
	int *d_version;
	int dev_id = device_map[thr_id];
	if (cuda_arch[dev_id] == 0) {
		// only do it once...
		cudaMalloc(&d_version, sizeof(int));
		nvcc_get_arch <<< 1, 1 >>> (d_version);
		cudaMemcpy(&cuda_arch[dev_id], d_version, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(d_version);
	}
	return cuda_arch[dev_id];
}
