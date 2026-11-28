#ifndef CCMINER_ALGOS_NEBHASH_H
#define CCMINER_ALGOS_NEBHASH_H
/* Host-side glue between ccminer's stratum/work machinery and the nebhash v2
 * reference + CUDA kernel copied from https://github.com/SilasEigen/neb
 * (Apache-2.0, see LICENSE.NEB).  Main network: KEP1, neb-pool/1. */
#include <stddef.h>
#include <stdint.h>

struct work;

/* Dataset file (NMDS container).  Load once, before connecting to a pool. */
int  nebhash_load_dataset_file(const char *path);
int  nebhash_dataset_loaded(void);
void nebhash_dataset_hash(uint8_t out[32]);
void nebhash_set_dataset_option(const char *path);
const char *nebhash_dataset_option(void);

/* Model sent by mining.set_model; packed weights are nweights/2 bytes. */
int  nebhash_set_model(int slot, const uint8_t weights_hash[32],
                       const uint8_t *packed, size_t packed_len);
int  nebhash_have_model(const uint8_t weights_hash[32]);

/* --benchmark: install the seeded model of slot 0 (after the dataset is loaded). */
int  nebhash_benchmark_init(void);

/* Pool target sent by mining.set_target, as a 32-byte little-endian integer. */
double nebhash_target_to_diff(const uint8_t le[32]);

/* Assemble the 80-byte attempt header (little-endian words, nonce zeroed). */
int  nebhash_build_header(uint32_t version, const uint8_t parent[32],
                          const uint8_t *cb_prefix, size_t cb_prefix_len,
                          const uint8_t *xn1, size_t xn1_len,
                          const uint8_t *xn2, size_t xn2_len,
                          const uint8_t *cb_suffix, size_t cb_suffix_len,
                          const uint8_t *branch, size_t branch_count,
                          uint32_t tx_count, const uint8_t shares_root[32],
                          uint32_t ntime, uint32_t bits, uint8_t header[80]);

/* Miner thread entry point and teardown (registered in algo_free_all()). */
int  nebhash_scanhash(int thr_id, struct work *work, uint32_t max_nonce,
                      unsigned long *hashes_done);
void nebhash_free(int thr_id);

#endif /* CCMINER_ALGOS_NEBHASH_H */
