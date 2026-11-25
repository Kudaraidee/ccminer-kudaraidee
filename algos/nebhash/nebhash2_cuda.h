#ifndef NEB_NEBHASH2_CUDA_H
#define NEB_NEBHASH2_CUDA_H
// Nonce scan of nebhash v2 (spec/nebhash-v2.md) on a CUDA device: the same contract as scan_cuda.h for the convolutional
// architectures of pow/ref/nebhash2.h. One thread block evaluates one attempt; every sum is exact in int32, so the
// tickets equal pow/ref/nebhash2.c bit for bit (the miner checks them against it). The status values are cudaError_t
// codes; 0 is success. A handle belongs to one device and must be used from one thread at a time.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct neb2_scan neb2_scan;

// Opens a handle on a device for the architecture `arch` (NEB2_KEP1 or NEB2_RET2). values is n x nin values 0..7,
// labels n values 0..1, weights the architecture's weight count in model order. One scan covers at most capacity nonces
// and reports at most max_hits of them.
int neb2_scan_open(int device, int arch, const uint8_t *values, const uint8_t *labels, uint32_t n, const int8_t *weights,
                   uint32_t capacity, uint32_t max_hits, neb2_scan **out);
// Replaces the weights after a model step.
int neb2_scan_set_model(neb2_scan *s, const int8_t *weights);
// As neb_scan_run of scan_cuda.h: the nonces start .. start + count - 1 of the header whose ticket is not above bound.
int neb2_scan_run(neb2_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound, uint32_t *hits,
                  uint32_t *nhits);
void neb2_scan_close(neb2_scan *s);

#ifdef __cplusplus
}
#endif
#endif
