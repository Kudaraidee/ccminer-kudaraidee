// equi24b, 192/7 instantiation: same digit width as 144/5 (192/8 = 144/6 = 24),
// so only the round count and PROOFSIZE differ and both are compile-time.
// Built -DEQ_WN=192 -DEQ_WK=7; symbols are namespaced on EQ_WN (EQ24B_API) so
// this object links alongside the 144/5 one.
//
// This file is only the include below, so a build rule naming it must also name
// cuda_equi24b.cu and equi24b_params.h as prerequisites, or this object goes
// stale when the real source changes.
#include "cuda_equi24b.cu"
