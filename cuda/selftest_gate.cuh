// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Fail-closed gate for the init-time device self-tests
 * (docs/coding-guideline.md section 7, layer 1).
 *
 * A KAT mismatch refuses to start: a GPU that cannot reproduce the consensus
 * hash produces nothing but local rejects, silently, for the whole session.
 *
 * A self-test that could not be *run* (out of memory) only warns: an init-time
 * resource blip must not kill a mining rig. Any other CUDA error means a broken
 * kernel and fails the test; see selftest_cuda_fault().
 *
 * Usage in a self-test TU:
 *     static bool foo_selftest_run(...) {
 *         if (cudaMalloc(...) != cudaSuccess) return selftest_cuda_fault();
 *         ...
 *         return ok ? true : selftest_cuda_fault();
 *     }
 *     bool foo_device_selftest(int thr_id) {
 *         ...
 *         passed = kat_ok && gpu_ok && neg_ok;
 *         if (!passed) gpulog(LOG_ERR, thr_id, "foo self-test FAILED (...)", ...);
 *         return selftest_gate(thr_id, "foo", passed);
 *     }
 */

#ifndef CUDA_SELFTEST_GATE_CUH
#define CUDA_SELFTEST_GATE_CUH

#include <cuda_runtime.h>
#include <miner.h> // gpulog, proper_exit, EXIT_CODE_SW_INIT_ERROR

/* Per-TU: each self-test TU owns its own state, and selftest_gate() clears it,
 * so a resource fault in one primitive cannot mask a real mismatch in the
 * next one. */
static bool s_selftest_cuda_fault = false;
static cudaError_t s_selftest_cuda_error = cudaSuccess;

/* Classify the pending CUDA error, return false: out of memory or none (a host
 * allocation failed) = could not run, anything else = fail. */
static inline bool selftest_cuda_fault(void)
{
	const cudaError_t err = cudaGetLastError();
	if (err == cudaSuccess || err == cudaErrorMemoryAllocation)
		s_selftest_cuda_fault = true;
	else if (s_selftest_cuda_error == cudaSuccess)
		s_selftest_cuda_error = err;
	return false;
}

/* Returns `passed` so it can wrap an existing `return passed;`. Does not
 * return at all when the verdict is a genuine mismatch. */
static inline bool selftest_gate(int thr_id, const char *name, bool passed)
{
	const cudaError_t err = s_selftest_cuda_error;
	const bool could_not_run = s_selftest_cuda_fault && err == cudaSuccess;
	s_selftest_cuda_fault = false;
	s_selftest_cuda_error = cudaSuccess;

	if (!passed) {
		if (err != cudaSuccess)
			gpulog(LOG_ERR, thr_id, "%s self-test: CUDA error %s", name, cudaGetErrorString(err));
		if (could_not_run) {
			/* the caller has already logged its FAILED line by now */
			gpulog(LOG_WARNING, thr_id, "%s self-test could not run (CUDA resource failure);"
				" the failure above is not evidence of a wrong hash, continuing", name);
		} else {
			gpulog(LOG_ERR, thr_id, "%s: refusing to start after a failed device self-test", name);
			proper_exit(EXIT_CODE_SW_INIT_ERROR);
		}
	}
	return passed;
}

#endif // CUDA_SELFTEST_GATE_CUH
