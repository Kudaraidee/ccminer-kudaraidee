#ifndef _CUDA_FUGUE256_H
#define _CUDA_FUGUE256_H

uint32_t fugue256_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *secNonce);
void fugue256_cpu_setBlock(int thr_id, void *data, void *pTargetIn);
void fugue256_cpu_init(int thr_id);
void fugue256_cpu_free(int thr_id);
bool fugue256_device_selftest(int thr_id);

#endif
