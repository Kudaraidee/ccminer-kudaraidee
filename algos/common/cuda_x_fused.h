#ifndef CUDA_X_FUSED_H
#define CUDA_X_FUSED_H

#include <stdint.h>

/* fused runs of the fixed-order chains, one compile-time-specialised kernel
 * per stage sequence (lists in cuda_x_fused.cu); no order upload needed */
enum {
	XF_JH_LUFFA_KECCAK_CUBE,               /* 0x10 */
	XF_JH_LUFFA_KECCAK,                    /* skydoge */
	XF_SKEIN_JH_KECCAK,                    /* sib */
	XF_SKEIN_BMW,                          /* 0x10 skydoge */
	XF_LUFFA_CUBE,                         /* hmq1725 */
	XF_JH_KECCAK,                          /* hmq1725 */
	XF_JH_CUBE,                            /* phi */
	XF_SKEIN_JH_KECCAK_LUFFA_CUBE_SHAVITE, /* x11 x13 x14 x15 x17 hsr */
	XF_JH_KECCAK_SKEIN_LUFFA_CUBE_SHAVITE, /* c11 */
	XF_LUFFA_CUBE_SHAVITE,                 /* sib */
	XF_CUBE_SHAVITE,                       /* qubit */
	XF_FIXED_COUNT
};

/* stage id of shavite in x_fused_fixed_ids() (not a runtime-fusible stage) */
#define X_FUSED_SHAVITE 17

void x_fused_fixed_cpu_hash_64(uint32_t threads, int seq, uint32_t *d_hash);
int x_fused_fixed_ids(int seq, uint8_t ids[8]);

#endif
