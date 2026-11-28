#ifndef NEB_NEBHASH2_TABLE_CUDA_H
#define NEB_NEBHASH2_TABLE_CUDA_H
// Nonce scan of nebhash v2 (spec/nebhash-v2.md) on a CUDA device from a table of per-sample gradients: the contract of
// nebhash2_cuda.h under other names, and the same tickets as pow/ref/nebhash2.c bit for bit (the miner checks them
// against it). For each model the host evaluates every sample of the dataset once with neb2_sample_grad and uploads
// one row per sample; an attempt then reads its 16 rows, adds them, clips the sum and hashes the digest. The status
// values are cudaError_t codes; 0 is success. A handle belongs to one device and must be used from one thread at a
// time.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct neb2t_scan neb2t_scan;

// As neb2_scan_open: values is n x nin values 0..7, labels n values 0..1, weights the architecture's weight count in
// model order. The handle keeps a copy of values and labels, and builds the table of the model.
int neb2t_scan_open(int device, int arch, const uint8_t *values, const uint8_t *labels, uint32_t n, const int8_t *weights,
                    uint32_t capacity, uint32_t max_hits, neb2t_scan **out);
// Builds and uploads the table of another model: one sample evaluation per sample, on every thread of the host.
int neb2t_scan_set_model(neb2t_scan *s, const int8_t *weights);
// As neb2_scan_run: the nonces start .. start + count - 1 of the header whose ticket is not above bound.
int neb2t_scan_run(neb2t_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound,
                   uint32_t *hits, uint32_t *nhits);
// Sparse scan: of the nonces start .. start + count - 1 (no capacity limit), evaluates only the attempts whose
// 16 draws hold at most one sample with a non-zero gradient (their digest is precomputed), and reports those whose
// ticket is not above bound. *evaluated gets the number of attempts evaluated: the work done, for the hash rate.
int neb2t_scan_run_sparse(neb2t_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound,
                          uint32_t *hits, uint32_t *nhits, uint32_t *evaluated);
// Self-test of the sparse scan: the nonce and the ticket (32 bytes, little-endian words) of up to max attempts the
// sparse scan evaluates in that range, in no particular order; *ndump gets their number.
int neb2t_scan_sparse_dump(neb2t_scan *s, const uint8_t *header, uint32_t start, uint32_t count, uint32_t max,
                           uint32_t *nonces, uint8_t *tickets, uint32_t *ndump);
// Samples of the current model with a non-zero gradient.
uint32_t neb2t_scan_active(const neb2t_scan *s);
// Threads per attempt the scan uses for the current model; 0 while it is still timing both shapes (sm_6x, after each
// model build; the timed launches are ordinary scans).
int neb2t_scan_threads(const neb2t_scan *s);
// Launch shapes the scan may choose between (1, or 2 on sm_6x), and pinning one of them (i = -1: automatic again) so
// a self-test can check each; a pinned scan does not time its launches.
int neb2t_scan_shapes(const neb2t_scan *s);
void neb2t_scan_pin_shape(neb2t_scan *s, int i);
void neb2t_scan_close(neb2t_scan *s);

#ifdef __cplusplus
}
#endif
#endif
