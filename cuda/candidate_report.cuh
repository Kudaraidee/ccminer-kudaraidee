#ifndef CUDA_CANDIDATE_REPORT_CUH
#define CUDA_CANDIDATE_REPORT_CUH

#include <stdint.h>

/*
 * Two-slot candidate report for the on-device target compare in the stage
 * *_final kernels. res[0..1] must be armed to UINT32_MAX before the launch.
 *
 * Keeps the two LOWEST candidates of the launch, res[0] < res[1], whatever
 * the thread interleaving: slot 0 takes the minimum, and whichever value it
 * did not keep (the newcomer or the one it displaced) competes for slot 1.
 * Every value except the global minimum reaches slot 1 exactly once, so
 * slot 1 ends at the second minimum. Any further candidate lies above the
 * host's max(res[0], res[1]) + 1 resume point and is rescanned.
 */
__device__ __forceinline__
void report_candidate_2(uint32_t *res, const uint32_t nonce)
{
	const uint32_t old = atomicMin(&res[0], nonce);
	const uint32_t displaced = (old > nonce) ? old : nonce;
	if (displaced != UINT32_MAX)
		atomicMin(&res[1], displaced);
}

#endif
