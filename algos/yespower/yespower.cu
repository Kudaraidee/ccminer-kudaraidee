/* yespower 1.0 family -- host driver and mining kernel.
 *
 * The hash body is in yespower_hash.cuh. Every GPU candidate is re-hashed on the
 * host (sph/yespower_ref.c, or sph/yespower_b2b_ref.c for power2b), so a kernel
 * bug costs a local reject, never a bad share.
 *
 * S-box placement: S is 96 KiB per instance. It normally lives in a global arena
 * (any sm_61+ card, many instances per SM). Shared S needs a 98304 B block opt-in
 * (sm_80+) and pins one block per SM, so it is only the fallback when the arena
 * does not fit in VRAM.
 *
 * yespower 0.5 is bit-identical to yescrypt 0.5 (algos/yescrypt/) and is not
 * duplicated here. */

#include "miner.h"
#include "cuda/helper.cuh"
#include "algos.h"     /* ALGO_* and `opt_algo`; miner.h declares neither */

extern "C" {
#include "sph/yespower.h"
}

#include <string.h>
#include <stdlib.h>

#include "yespower_hash.cuh"
#include "cuda/selftest_gate.cuh"

extern char *yescrypt_key;          /* shared with --yescrypt-key / --yespower-key */
extern size_t yescrypt_key_len;
extern uint32_t yescrypt_param_N;   /* shared with --yescrypt-param / --yespower-param */
extern uint32_t yescrypt_param_r;

/* Personalisation strings. `pers` is hashed: one wrong byte gives hashes the
 * pool rejects while the miner looks healthy. Each length is written beside its
 * string in the table below and checked when selected. Sources: README.md. */
static const char PERS_POWER2B[] = "Now I am become Death, the destroyer of worlds";
static const char PERS_SUGAR[]   =
	"Satoshi Nakamoto 31/Oct/2008 Proof-of-work is essentially one-CPU-one-vote";
static const char PERS_URX[]     = "UraniumX";
static const char PERS_LTNCG[]   = "LTNCGYES";
static const char PERS_MGPC[]    = "Magpies are birds of the Corvidae family.";
static const char PERS_ARWN[]    = "ARWN";
static const char PERS_ADVC[]    = "Let the quest begin";
/* EqPay: 88 bytes, and the trailing '.' is part of it -- perslen is exactly
 * strlen(), with no NUL hashed. */
static const char PERS_EQPAY[]   =
	"The gods had gone away, and the ritual of the religion continued "
	"senselessly, uselessly.";
static const char PERS_IC[]      = "IsotopeC";
static const char PERS_IOTS[]    = "Iots is committed to the development of IOT";
static const char PERS_LITB[]    =
	"LITBpower: The number of LITB working or available for proof-of-work mini";
static const char PERS_CPUPOWER[] =
	"CPUpower: The number of CPU working or available for proof-of-work mining";

/* Coin presets for `-a yespower`: fixed (N, r, pers). algos.h maps the name,
 * this table supplies the parameters, and a name algos.h accepts without an entry
 * here is rejected (generic parameters would mine shares the pool rejects).
 * yespowerEQPAY hashes a 181-byte header and is its own algo. */
struct yespower_variant {
	const char *name;
	uint32_t    N;
	uint32_t    r;
	const char *pers;      /* NULL = keyless */
	size_t      perslen;   /* 0 iff pers == NULL */
};

static const struct yespower_variant yespower_variants[] = {
	/* name             N     r   pers            perslen */
	{ "yespowersugar",  2048, 32, PERS_SUGAR,     74 },  /* Sugarchain (SUGAR) */
	{ "sugarchain",     2048, 32, PERS_SUGAR,     74 },
	{ "yespowerurx",    2048, 32, PERS_URX,       8  },  /* UraniumX (URX)     */
	{ "yespowerltncg",  2048, 32, PERS_LTNCG,     8  },  /* LightningCash-Gold */
	{ "yespowermgpc",   2048, 32, PERS_MGPC,      41 },  /* MagpieCoin (MGPC)  */
	{ "yespowertide",   2048,  8, NULL,           0  },  /* Tidecoin (TDC)     */
	{ "yespowersmt",     256,  8, NULL,           0  },  /* Smartiecoin (SMT)  */
	{ "yespowerarwn",   2048, 32, PERS_ARWN,      4  },  /* Arowana (ARWN)     */
	{ "yespoweradvc",   2048, 32, PERS_ADVC,      19 },  /* AdventureCoin ADVC */
	{ "yespoweric",     2048, 32, PERS_IC,        8  },  /* IsotopeC           */
	{ "yespoweriots",   2048, 32, PERS_IOTS,      43 },  /* IOTS               */
	{ "yespowerlitb",   2048, 32, PERS_LITB,      73 },  /* LightBit (LITB)    */
	{ "cpupower",       2048, 32, PERS_CPUPOWER,  73 },  /* CPUchain (CPU)     */

	/* power2b is absent deliberately: it is its own algo (ALGO_POWER2B, blake2b
	 * head/tail) and takes its parameters from yespower_params_for() below. */
};

/* Names without a preset: they mine whatever --yespower-param/-key say. */
static bool yespower_name_is_generic(const char *name)
{
	return !strcasecmp(name, "yespower")     || !strcasecmp(name, "yespowerr16") ||
	       !strcasecmp(name, "yenten")       || !strcasecmp(name, "power2b")     ||
	       !strcasecmp(name, "yespower-b2b");
}

/* The variant `-a` (or a pool's "algo") selected, NULL for the generic names.
 * Not written into yescrypt_param_N/_r/yescrypt_key, which are shared with the
 * yescrypt 0.5 algos; an explicit --yespower-param/-key clears it. */
static const struct yespower_variant *yespower_selected = NULL;

extern "C" void yespower_clear_variant(void)
{
	yespower_selected = NULL;
}

/* Select the variant for an `-a` alias or a pool's "algo". False = no entry
 * (an alias-list bug): fatal for `-a`, a pool switch refuses the switch. */
extern "C" bool yespower_set_variant(const char *name)
{
	if (!name || !*name || yespower_name_is_generic(name)) {
		yespower_selected = NULL;
		return true;
	}
	for (size_t i = 0; i < ARRAY_SIZE(yespower_variants); i++) {
		const struct yespower_variant *v = &yespower_variants[i];
		if (strcasecmp(name, v->name))
			continue;
		if (v->pers ? (strlen(v->pers) != v->perslen) : (v->perslen != 0)) {
			applog(LOG_ERR, "yespower: variant '%s' pers is %u bytes, table says %u"
			       " -- the literal has been damaged, refusing to mine",
			       v->name, (uint32_t) (v->pers ? strlen(v->pers) : 0),
			       (uint32_t) v->perslen);
			return false;
		}
		yespower_selected = v;
		return true;
	}
	applog(LOG_ERR, "yespower: '%s' is an accepted algo name with no parameter entry"
	       " -- it would mine generic yespower and every share would be rejected",
	       name);
	return false;
}

/* "" when no variant is selected; lets a failed algo switch restore it. */
extern "C" const char *yespower_variant_name(void)
{
	return yespower_selected ? yespower_selected->name : "";
}

/* Fill `p` for the given algo. Returns false if the algo is not a yespower one. */
static bool yespower_params_for(int algo, yespower_params_t *p)
{
	memset(p, 0, sizeof(*p));
	p->version = YESPOWER_1_0;

	switch (algo) {
	case ALGO_YESPOWER:
		/* coin preset, else --yespower-param/-key, else the 1.0 defaults */
		if (yespower_selected) {
			p->N = yespower_selected->N;
			p->r = yespower_selected->r;
			p->pers = (const uint8_t *) yespower_selected->pers;
			p->perslen = yespower_selected->perslen;
			return true;
		}
		p->N = yescrypt_param_N ? yescrypt_param_N : 2048;
		p->r = yescrypt_param_r ? yescrypt_param_r : 32;
		p->pers = (const uint8_t *) yescrypt_key;
		p->perslen = yescrypt_key ? yescrypt_key_len : 0;
		return true;
	case ALGO_YESPOWERR16:       /* Yenten (YTN) */
		p->N = 4096; p->r = 16;
		p->pers = NULL; p->perslen = 0;
		return true;
	case ALGO_POWER2B:           /* MicroBitcoin: BLAKE2b head/tail */
		p->N = 2048; p->r = 32;
		p->pers = (const uint8_t *) PERS_POWER2B;
		p->perslen = sizeof(PERS_POWER2B) - 1;   /* 46 */
		return true;
	case ALGO_YESPOWEREQPAY:     /* EqPay: 1.0 params, 181-byte header */
		p->N = 2048; p->r = 32;
		p->pers = (const uint8_t *) PERS_EQPAY;
		p->perslen = sizeof(PERS_EQPAY) - 1;     /* 88 */
		return true;
	default:
		return false;
	}
}

/* EqPay (Qtum-derived) hashes a 181-byte header: the usual fields, then
 * hashStateRoot, hashUTXORoot, a null prevoutStake and an empty signature
 * length (layout in README.md). prevoutStake.n must be 0xffffffff. */
#define YP_EQPAY_HDR_BYTES     181u
#define YP_EQPAY_HDR_WORDS     46u       /* ceil(181/4), the work-buffer span */
#define YP_EQPAY_STATE_ROOT     80u
#define YP_EQPAY_UTXO_ROOT     112u
#define YP_EQPAY_STAKE         144u
#define YP_EQPAY_STAKE_N       176u
#define YP_EQPAY_SIGLEN        180u

/* Bytes of block header this algo's proof-of-work covers. */
static inline size_t yp_header_bytes(int algo)
{
	return (algo == ALGO_YESPOWEREQPAY) ? YP_EQPAY_HDR_BYTES : 80u;
}

/* One header in, 32-byte digest out. The reference returns 1 on success; on
 * failure the digest is all-ones, which never meets a target. */
static bool yespower_hash_hdr(void *state, const void *input, size_t len,
                              const yespower_params_t *params)
{
	yespower_binary_t out;
	/* same head/tail primitive as the kernel */
	const int rc = (opt_algo == ALGO_POWER2B)
	             ? yespower_b2b_tls_ref((const uint8_t *) input, len, params, &out)
	             : yespower_tls_ref((const uint8_t *) input, len, params, &out);
	if (rc != 1) {
		memset(state, 0xff, 32);
		return false;
	}
	memcpy(state, out.uc, 32);
	return true;
}

/* Exposed for the GPU path's host-side re-verification. */
extern "C" void yespower_hash(void *state, const void *input, int algo)
{
	yespower_params_t p;
	if (!yespower_params_for(algo, &p)) { memset(state, 0xff, 32); return; }
	(void) yespower_hash_hdr(state, input, yp_header_bytes(algo), &p);
}

/* ---------------------------------------------------------------------------
 * GPU path.
 * ------------------------------------------------------------------------- */

/* X in shared on the global-S path (all arches). */
#define YP_X_SHARED(PL)  ((PL) == PWX_S_GLOBAL)

/* minBlocks for __launch_bounds__ (a register cap). Not arch-conditional: with
 * several -gencode targets the bounds are fixed once, and a cap on every arch
 * costs spills where none is needed. */
#ifndef YP_MINBLK
#define YP_MINBLK 1
#endif

/* Block width. Four lanes run pwxform (PWXgather = 4); the others only widen the
 * V copies. At most one warp: every barrier is a __syncwarp (static_assert in
 * yespower_hash.cuh). */
#ifndef YP_WIDTH
#define YP_WIDTH 32
#endif

/* Participation mask for the block (avoids 1u << 32). */
#define YP_WMASK ((YP_WIDTH >= 32) ? 0xffffffffu : (uint32_t)((1u << (YP_WIDTH & 31)) - 1u))

/* Candidate slots: res[0] is the count, res[1..YP_MAX_RES] the nonces. */
#define YP_MAX_RES 8u

/* Instances per SM. Ampere shares one L1/shared pool per SM, so X in shared
 * costs L1 and it runs fewer instances (also less VRAM and shorter launches);
 * Pascal has a separate L1. sm_75 takes the Ampere value (16 would not fit its
 * shared budget). Clamped by VRAM at init. */
#define YP_IPB_AMPERE 8u
#define YP_IPB_OTHER  16u

/* Header words in SHA-256 input order; [19] (the nonce) is passed separately.
 * 46 words for EqPay's 181-byte header; the others use the first 20. */
#define YP_HDR_WORDS      20u                   /* yespower 1.0, 80-byte header */
#define YP_HDR_WORDS_181  46u                   /* EqPay, 181-byte header       */
__constant__ static uint32_t c_yp_hdr[YP_HDR_WORDS_181];
__constant__ static uint32_t c_yp_target[8];

/* Exact 256-bit compare, MSW first, as fulltest(). A hash[7]-only screen would
 * drop shares below diff 1. */
__device__ __forceinline__ bool yp_below_target(const uint32_t h[8])
{
#pragma unroll
	for (int i = 7; i >= 0; i--) {
		if (h[i] > c_yp_target[i]) return false;
		if (h[i] < c_yp_target[i]) return true;
	}
	return true;                     /* equal counts as a hit, as fulltest does */
}

/* One block per nonce: YP_WIDTH threads, lanes 0-3 run pwxform. */
template<uint32_t R, int PLACE, int HEAD>
__global__ __launch_bounds__(YP_WIDTH, YP_MINBLK)
void yespower_gpu_hash(const uint32_t startNonce, const uint32_t N,
                       uint32_t *__restrict__ Vs, uint32_t *__restrict__ Bs,
                       uint32_t *__restrict__ Xs, uint4 *__restrict__ Sg,
                       uint32_t *__restrict__ resNonces)
{
	/* dynamic shared S, used only on the shared-S path */
	extern __shared__ uint4 s_S[];

	/* X in shared on the global-S path only: shared S already uses 98304 of the
	 * 101376 B a block can opt into. */
	__shared__ uint32_t s_X[YP_X_SHARED(PLACE) ? 32u * R : 1u];

	const int j = threadIdx.x & 3;
	const uint32_t inst = blockIdx.x;
	const uint32_t nonce = startNonce + inst;
	const uint32_t bw = 32u * R;
	uint32_t out[8];

	uint4 *S = (PLACE == PWX_S_SHARED) ? s_S
	                                   : (Sg + (size_t)inst * YP_SBOX_UINT4);
	uint32_t *Xp = YP_X_SHARED(PLACE) ? s_X : (Xs + (size_t)inst * bw);

	yespower_hash_1_0<R, PLACE, HEAD, YP_WIDTH>(c_yp_hdr, nonce, N, S,
	                     Bs + (size_t)inst * bw,
	                     Xp,
	                     Vs + (size_t)inst * bw * N,
	                     out, j, 0xfu, (int)threadIdx.x, YP_WMASK);

	/* Candidate report: atomic count in slot 0, bounded writes after it; the host
	 * sorts. threadIdx.x == 0, not j == 0: above width 4 several threads have
	 * j == 0 but only tid < 4 hold the digest. */

	if (threadIdx.x == 0 && yp_below_target(out)) {
		const uint32_t pos = atomicAdd(&resNonces[0], 1u);
		if (pos < YP_MAX_RES) resNonces[1u + pos] = nonce;
		/* overflow is counted; the host logs it */
	}
}

/* Instruments: the fail-closed init self-test and the -D per-job differential.
 * Both inline the mining kernel's hash body. */

/* The digest folds into two order-independent accumulators: XOR, and XOR
 * weighted by 2*nonce+1. The plain one cannot see a nonce permutation or a
 * uniform perturbation over an even span; the weighted one sees both. */
template<uint32_t R, int PLACE, int HEAD>
__global__ __launch_bounds__(YP_WIDTH, YP_MINBLK)
void yespower_gpu_checksum(const uint32_t startNonce, const uint32_t N,
                           uint32_t *__restrict__ Vs, uint32_t *__restrict__ Bs,
                           uint32_t *__restrict__ Xs, uint4 *__restrict__ Sg,
                           unsigned long long *__restrict__ acc)
{
	extern __shared__ uint4 s_S[];

	/* X in shared on the global-S path only (see yespower_gpu_hash). */
	__shared__ uint32_t s_X[YP_X_SHARED(PLACE) ? 32u * R : 1u];

	const int j = threadIdx.x & 3;
	const uint32_t inst = blockIdx.x;
	const uint32_t nonce = startNonce + inst;
	const uint32_t bw = 32u * R;
	uint32_t out[8];

	uint4 *S = (PLACE == PWX_S_SHARED) ? s_S
	                                   : (Sg + (size_t)inst * YP_SBOX_UINT4);
	uint32_t *Xp = YP_X_SHARED(PLACE) ? s_X : (Xs + (size_t)inst * bw);

	yespower_hash_1_0<R, PLACE, HEAD, YP_WIDTH>(c_yp_hdr, nonce, N, S,
	                     Bs + (size_t)inst * bw,
	                     Xp,
	                     Vs + (size_t)inst * bw * N,
	                     out, j, 0xfu, (int)threadIdx.x, YP_WMASK);

	/* threadIdx.x == 0: see yespower_gpu_hash */
	if (threadIdx.x == 0) {
		unsigned long long q = ((unsigned long long) out[7] << 32) | out[6];
		unsigned long long w = 2ull * (unsigned long long) nonce + 1ull;
#ifdef YP_FAULT_DIFF_CONST
		/* test only: the same perturbation on every digest (acc[1] must see it) */
		q ^= 1ull;
#endif
#ifdef YP_FAULT_DIFF_PERM
		/* test only: wrong nonce weight (acc[1] must see it) */
		w = 2ull * (unsigned long long) (nonce ^ 1u) + 1ull;
#endif
		atomicXor(&acc[0], q);
		atomicXor(&acc[1], q * w);
	}
}

static THREAD uint32_t *d_V[MAX_GPUS]  = { 0 };
static THREAD uint32_t *d_B[MAX_GPUS]  = { 0 };
static THREAD uint32_t *d_X[MAX_GPUS]  = { 0 };
static THREAD uint4    *d_S[MAX_GPUS]  = { 0 };   /* global S arena; NULL on the shared path */
static THREAD bool      yp_sglobal[MAX_GPUS] = { false };
static THREAD int       yp_head[MAX_GPUS]    = { YP_HEAD_SHA256 };  /* head/tail primitive */
static THREAD uint32_t *d_res[MAX_GPUS] = { 0 };
static THREAD uint32_t yp_instances[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { false };

/* Dispatch on r; only the shapes the variants use (8, 16, 32) are built. */
static bool yp_launch(uint32_t r, uint32_t instances, uint32_t startNonce, uint32_t N,
                      int dev, bool sglobal, int head)
{
	const uint32_t shbytes = sglobal ? 0u : (YP_SBOX_UINT4 * 16u);

#define YP_LAUNCH_1(RR, PL, HD, SH)                                              \
	yespower_gpu_hash<RR, PL, HD><<<instances, YP_WIDTH, (SH)>>>(startNonce, N,         \
		d_V[dev], d_B[dev], d_X[dev], d_S[dev], d_res[dev])

#define YP_LAUNCH(RR)                                                            \
	do {                                                                         \
		if (sglobal && head == YP_HEAD_B2B) YP_LAUNCH_1(RR, PWX_S_GLOBAL, YP_HEAD_B2B, 0); \
		else if (sglobal)          YP_LAUNCH_1(RR, PWX_S_GLOBAL, YP_HEAD_SHA256, 0); \
		else if (head == YP_HEAD_B2B)       YP_LAUNCH_1(RR, PWX_S_SHARED, YP_HEAD_B2B, shbytes); \
		else                       YP_LAUNCH_1(RR, PWX_S_SHARED, YP_HEAD_SHA256, shbytes); \
	} while (0)

	/* EqPay: r = 32 only */
	if (head == YP_HEAD_SHA256_181) {
		if (r != 32) return false;
		if (sglobal) YP_LAUNCH_1(32, PWX_S_GLOBAL, YP_HEAD_SHA256_181, 0);
		else         YP_LAUNCH_1(32, PWX_S_SHARED, YP_HEAD_SHA256_181, shbytes);
		return true;
	}

	switch (r) {
	case 8:  YP_LAUNCH(8);  return true;
	case 16: YP_LAUNCH(16); return true;
	case 32: YP_LAUNCH(32); return true;
	default: return false;
	}
#undef YP_LAUNCH
#undef YP_LAUNCH_1
}

static bool yp_set_shared_limit(uint32_t r, int head)
{
	const uint32_t shbytes = YP_SBOX_UINT4 * 16u;
	cudaError_t e;

	/* The opt-in is per kernel: the checksum kernel needs its own. */
#define YP_OPTIN(RR)                                                                 \
	((head == YP_HEAD_B2B)                                                       \
	   ? (cudaFuncSetAttribute(yespower_gpu_hash<RR, PWX_S_SHARED, YP_HEAD_B2B>,        \
	                           cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes),   \
	      cudaFuncSetAttribute(yespower_gpu_checksum<RR, PWX_S_SHARED, YP_HEAD_B2B>,    \
	                           cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes))   \
	   : (cudaFuncSetAttribute(yespower_gpu_hash<RR, PWX_S_SHARED, YP_HEAD_SHA256>,     \
	                           cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes),   \
	      cudaFuncSetAttribute(yespower_gpu_checksum<RR, PWX_S_SHARED, YP_HEAD_SHA256>, \
	                           cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes)))

	if (head == YP_HEAD_SHA256_181) {
		if (r != 32) return false;
		e = cudaFuncSetAttribute(yespower_gpu_hash<32, PWX_S_SHARED, YP_HEAD_SHA256_181>,
		                         cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
		if (e == cudaSuccess)
			e = cudaFuncSetAttribute(yespower_gpu_checksum<32, PWX_S_SHARED, YP_HEAD_SHA256_181>,
			                         cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
		return e == cudaSuccess;
	}

	switch (r) {
	case 8:  e = YP_OPTIN(8);  break;
	case 16: e = YP_OPTIN(16); break;
	case 32: e = YP_OPTIN(32); break;
	default: return false;
	}
#undef YP_OPTIN
	return e == cudaSuccess;
}


/* Instrument spans: prime lengths, so no instance count divides them (the last
 * launch is short), starting off zero so the startNonce arithmetic is used. */
#define YP_DIFF_SPAN 53u
#define YP_DIFF_OFF  0x51u
#define YP_ST_SPAN   13u
#define YP_ST_OFF    0x51u
#define YP_ST_EVEN   12u    /* even sub-span: see the acc[0] blindness leg */

static bool yp_checksum_launch(uint32_t r, uint32_t instances, uint32_t startNonce,
                               uint32_t N, int dev, bool sglobal, int head,
                               unsigned long long *d_acc)
{
	const uint32_t shbytes = sglobal ? 0u : (YP_SBOX_UINT4 * 16u);

#define YP_CK_1(RR, PL, HD, SH)                                                  \
	yespower_gpu_checksum<RR, PL, HD><<<instances, YP_WIDTH, (SH)>>>(startNonce, N,     \
		d_V[dev], d_B[dev], d_X[dev], d_S[dev], d_acc)

#define YP_CK(RR)                                                                \
	do {                                                                         \
		if (sglobal && head == YP_HEAD_B2B) YP_CK_1(RR, PWX_S_GLOBAL, YP_HEAD_B2B, 0); \
		else if (sglobal)          YP_CK_1(RR, PWX_S_GLOBAL, YP_HEAD_SHA256, 0); \
		else if (head == YP_HEAD_B2B)       YP_CK_1(RR, PWX_S_SHARED, YP_HEAD_B2B, shbytes); \
		else                       YP_CK_1(RR, PWX_S_SHARED, YP_HEAD_SHA256, shbytes); \
	} while (0)

	/* EqPay: r = 32 only, as in yp_launch */
	if (head == YP_HEAD_SHA256_181) {
		if (r != 32) return false;
		if (sglobal) YP_CK_1(32, PWX_S_GLOBAL, YP_HEAD_SHA256_181, 0);
		else         YP_CK_1(32, PWX_S_SHARED, YP_HEAD_SHA256_181, shbytes);
		return true;
	}

	switch (r) {
	case 8:  YP_CK(8);  return true;
	case 16: YP_CK(16); return true;
	case 32: YP_CK(32); return true;
	default: return false;
	}
#undef YP_CK
#undef YP_CK_1
}

/* Accumulate over [start, start+span) in chunks of at most the allocated
 * instance count. */
static bool yp_gpu_acc(int dev, const yespower_params_t *p,
                       uint32_t start, uint32_t span, unsigned long long acc[2])
{
	unsigned long long *d_acc = NULL;
	unsigned long long zero[2] = { 0ull, 0ull };
	bool ok = true;

	if (cudaMalloc(&d_acc, sizeof(zero)) != cudaSuccess) return false;
	if (cudaMemcpy(d_acc, zero, sizeof(zero), cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(d_acc);
		return false;
	}
	for (uint32_t done = 0; done < span && ok; ) {
		uint32_t chunk = span - done;
		if (chunk > yp_instances[dev]) chunk = yp_instances[dev];
		ok = yp_checksum_launch(p->r, chunk, start + done, p->N, dev,
		                        yp_sglobal[dev], yp_head[dev], d_acc)
		     && cudaDeviceSynchronize() == cudaSuccess;
		done += chunk;
	}
	ok = ok && cudaMemcpy(acc, d_acc, sizeof(zero), cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d_acc);
	return ok;
}

/* One CPU pass over the span for every accumulator the legs need. `flip` changes
 * the header (a different job); perm/const change only the accumulation. */
static bool yp_cpu_acc(const uint32_t *pdata, const yespower_params_t *p,
                       uint32_t start, uint32_t span, bool flip,
                       unsigned long long plain[2], unsigned long long perm[2],
                       unsigned long long even_plain[2], unsigned long long even_const[2])
{
	uint32_t endian[YP_EQPAY_HDR_WORDS], vhash[8];
	const size_t hlen = yp_header_bytes(opt_algo);
	const uint32_t hwords = (uint32_t)((hlen + 3u) / 4u);

	if (plain)      plain[0]      = plain[1]      = 0ull;
	if (perm)       perm[0]       = perm[1]       = 0ull;
	if (even_plain) even_plain[0] = even_plain[1] = 0ull;
	if (even_const) even_const[0] = even_const[1] = 0ull;

	for (uint32_t k = 0; k < hwords; k++) be32enc(&endian[k], pdata[k]);
	if (flip) endian[3] ^= 0x00000001u;

	for (uint32_t i = 0; i < span; i++) {
		const uint32_t nonce = start + i;
		unsigned long long q, w;

		be32enc(&endian[19], nonce);
		if (!yespower_hash_hdr(vhash, endian, hlen, p)) return false;

		q = ((unsigned long long) vhash[7] << 32) | vhash[6];
		w = 2ull * (unsigned long long) nonce + 1ull;

		if (plain) { plain[0] ^= q; plain[1] ^= q * w; }
		if (perm)  { perm[0]  ^= q; perm[1]  ^= q * (2ull * (unsigned long long)(nonce ^ 1u) + 1ull); }
		if (i < YP_ST_EVEN) {
			if (even_plain) { even_plain[0] ^= q;        even_plain[1] ^= q * w; }
			if (even_const) { even_const[0] ^= (q ^ 1ull); even_const[1] ^= (q ^ 1ull) * w; }
		}
	}
	return true;
}

/* EqPay genesis gate: an external oracle (the chain's genesis digest). Flipping
 * byte 80 or 180 must change it, which a head hashing 80 or 180 bytes misses. */
static bool yespower_eqpay_genesis_kat(int thr_id, const yespower_params_t *p)
{
	static const uint8_t kat_hdr[181] = {
		0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0xe1, 0xa7, 0xff, 0x0f, 0x99, 0xf5, 0xe6, 0x92, 0xa1, 0x6f, 0x69, 0x69,
		0xe0, 0x09, 0xca, 0xf9, 0x51, 0x86, 0x57, 0xe6, 0x7b, 0x12, 0xa5, 0xac,
		0x15, 0xf8, 0x53, 0x39, 0x5c, 0x7f, 0xed, 0xaa, 0x25, 0xc6, 0x55, 0x61,
		0xff, 0xff, 0x3f, 0x1f, 0x53, 0x01, 0x00, 0x00, 0xe9, 0x65, 0xff, 0xd0,
		0x02, 0xcd, 0x6a, 0xd0, 0xe2, 0xdc, 0x40, 0x2b, 0x80, 0x44, 0xde, 0x83,
		0x3e, 0x06, 0xb2, 0x31, 0x27, 0xea, 0x8c, 0x3d, 0x80, 0xae, 0xc9, 0x14,
		0x10, 0x77, 0x14, 0x95, 0x56, 0xe8, 0x1f, 0x17, 0x1b, 0xcc, 0x55, 0xa6,
		0xff, 0x83, 0x45, 0xe6, 0x92, 0xc0, 0xf8, 0x6e, 0x5b, 0x48, 0xe0, 0x1b,
		0x99, 0x6c, 0xad, 0xc0, 0x01, 0x62, 0x2f, 0xb5, 0xe3, 0x63, 0xb4, 0x21,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
		0x00,
	};

	static const uint8_t kat_digest[32] = {
		0x56, 0xda, 0x66, 0x12, 0x94, 0xe4, 0xac, 0xf5, 0x60, 0xea, 0x4d, 0x73,
		0x12, 0x1a, 0xc9, 0xd8, 0xd8, 0xf8, 0xd3, 0x09, 0xf0, 0x6e, 0xe3, 0x32,
		0x4e, 0x7b, 0xe3, 0x36, 0x01, 0x56, 0x04, 0x00,
	};

	uint8_t out[32], alt[YP_EQPAY_HDR_BYTES], tmp[32];
	bool digest_ok, roots_ok, siglen_ok, passed;

	if (!yespower_hash_hdr(out, kat_hdr, YP_EQPAY_HDR_BYTES, p)) {
		gpulog(LOG_ERR, thr_id, "eqpay self-test: the CPU reference refused the "
		                        "parameters (N=%u r=%u)", p->N, (uint32_t) p->r);
		return selftest_gate(thr_id, "eqpay", false);
	}
	digest_ok = (memcmp(out, kat_digest, 32) == 0);

	memcpy(alt, kat_hdr, YP_EQPAY_HDR_BYTES);
	alt[YP_EQPAY_STATE_ROOT] ^= 0x01;
	roots_ok = yespower_hash_hdr(tmp, alt, YP_EQPAY_HDR_BYTES, p) &&
	           memcmp(tmp, out, 32) != 0;

	memcpy(alt, kat_hdr, YP_EQPAY_HDR_BYTES);
	alt[YP_EQPAY_SIGLEN] ^= 0x01;
	siglen_ok = yespower_hash_hdr(tmp, alt, YP_EQPAY_HDR_BYTES, p) &&
	            memcmp(tmp, out, 32) != 0;

	passed = digest_ok && roots_ok && siglen_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "eqpay self-test FAILED: digest=%d roots=%d siglen=%d "
		                        "(%u-byte header)",
		       (int) digest_ok, (int) roots_ok, (int) siglen_ok,
		       (unsigned) YP_EQPAY_HDR_BYTES);

	return selftest_gate(thr_id, "eqpay", passed);
}

/* Smartiecoin KAT: header and digest of a pool-accepted share (block 233709).
 * Bytes 1 and 70 lie in different SHA-256 blocks; flipping either must change it. */
static bool yespower_smt_share_kat(int thr_id, const yespower_params_t *p)
{
	static const uint8_t kat_hdr[80] = {
		0x00, 0x00, 0x00, 0x20, 0x90, 0xd9, 0xbd, 0xf0, 0x6c, 0xbb, 0x6c, 0x96,
		0xb8, 0x71, 0x7f, 0x2c, 0x62, 0xeb, 0x0b, 0xa7, 0x50, 0x6b, 0x03, 0x47,
		0x2e, 0x24, 0x08, 0x20, 0xb8, 0xd1, 0xf5, 0x93, 0x3d, 0x00, 0x00, 0x00,
		0x65, 0xe6, 0x75, 0xf4, 0xe7, 0xfa, 0x46, 0xa9, 0x54, 0x53, 0x75, 0xfd,
		0x98, 0x5a, 0xb4, 0x99, 0xfd, 0x7d, 0x64, 0x54, 0x18, 0xee, 0x94, 0x1c,
		0x28, 0x93, 0x36, 0x08, 0x25, 0x51, 0x9c, 0x58, 0x64, 0x94, 0xba, 0x6a,
		0x77, 0xfb, 0x4c, 0x1d, 0xfe, 0xff, 0xff, 0xaf,
	};

	static const uint8_t kat_digest[32] = {
		0x9e, 0x48, 0x11, 0xc5, 0x41, 0xdd, 0xe6, 0xfb, 0x3b, 0x50, 0x7d, 0xce,
		0x11, 0x0f, 0x2b, 0xcc, 0x97, 0x0d, 0xbb, 0x76, 0xe6, 0x4e, 0x21, 0x25,
		0xcd, 0xcf, 0x32, 0x90, 0x95, 0xa7, 0x73, 0x00,
	};

	uint8_t out[32], alt[80], tmp[32];
	bool digest_ok, lo_ok, hi_ok, passed;

	if (!yespower_hash_hdr(out, kat_hdr, 80, p)) {
		gpulog(LOG_ERR, thr_id, "smt self-test: the CPU reference refused the "
		                        "parameters (N=%u r=%u)", p->N, (uint32_t) p->r);
		return selftest_gate(thr_id, "smt", false);
	}
	digest_ok = (memcmp(out, kat_digest, 32) == 0);

	memcpy(alt, kat_hdr, 80);
	alt[1] ^= 0x01;
	lo_ok = yespower_hash_hdr(tmp, alt, 80, p) && memcmp(tmp, out, 32) != 0;

	memcpy(alt, kat_hdr, 80);
	alt[70] ^= 0x01;
	hi_ok = yespower_hash_hdr(tmp, alt, 80, p) && memcmp(tmp, out, 32) != 0;

	passed = digest_ok && lo_ok && hi_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "smt self-test FAILED: digest=%d byte1=%d byte70=%d "
		                        "(N=%u r=%u)", (int) digest_ok, (int) lo_ok, (int) hi_ok,
		       p->N, (uint32_t) p->r);

	return selftest_gate(thr_id, "smt", passed);
}

/* Fail-closed init gate: GPU == CPU over a fixed header (kat), GPU != CPU for a
 * different header (neg), and two CPU-only legs showing each accumulator's
 * blind spot (perm, const). */
static bool yespower_device_selftest(int thr_id, int dev, const yespower_params_t *p)
{
	/* external oracles first */
	if (opt_algo == ALGO_YESPOWEREQPAY && !yespower_eqpay_genesis_kat(thr_id, p))
		return false;
	if (opt_algo == ALGO_YESPOWER && yespower_selected &&
	    !strcasecmp(yespower_selected->name, "yespowersmt") &&
	    !yespower_smt_share_kat(thr_id, p))
		return false;

	/* Fixed header; c_yp_hdr is re-uploaded with the real job on every call. */
	static const uint32_t testhdr[YP_EQPAY_HDR_WORDS] = {
		0x20000000u, 0x0f1e2d3cu, 0x4b5a6978u, 0x8796a5b4u, 0xc3d2e1f0u,
		0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u, 0x0badc0deu,
		0xfeedfaceu, 0xcafebabeu, 0x13579bdfu, 0x2468ace0u, 0xa5a5a5a5u,
		0x5a5a5a5au, 0x00ff00ffu, 0xff00ff00u, 0x1a2b3c4du, 0x00000000u,
		/* EqPay tail, read only by the 181-byte head */
		0x01234567u, 0x89abcdefu, 0x1a2b3c4du, 0x5e6f7a8bu,
		0x00112233u, 0x44556677u, 0x8899aabbu, 0xccddeeffu,
		0xdeadbeefu, 0xfeedfaceu, 0x0badf00du, 0xcafed00du,
		0x13579bdfu, 0x02468aceu, 0xfdb97531u, 0xeca86420u,
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0xffffffffu,   /* prevoutStake.n */
		0x00000000u    /* signature length byte, then padding */
	};
	const uint32_t st_words = (uint32_t)((yp_header_bytes(opt_algo) + 3u) / 4u);
	unsigned long long g[2], cpu[2], cneg[2], perm[2], ev[2], evc[2];
	bool kat, neg, permleg, constleg, passed;

	if (cudaMemcpyToSymbol(c_yp_hdr, testhdr, st_words * sizeof(uint32_t)) != cudaSuccess)
		return selftest_gate(thr_id, "yespower", selftest_cuda_fault());

	if (!yp_gpu_acc(dev, p, YP_ST_OFF, YP_ST_SPAN, g))
		return selftest_gate(thr_id, "yespower", selftest_cuda_fault());

	if (!yp_cpu_acc(testhdr, p, YP_ST_OFF, YP_ST_SPAN, false, cpu, perm, ev, evc) ||
	    !yp_cpu_acc(testhdr, p, YP_ST_OFF, YP_ST_SPAN, true,  cneg, NULL, NULL, NULL)) {
		gpulog(LOG_ERR, thr_id, "yespower self-test: the CPU reference refused the "
		                        "parameters (N=%u r=%u)", p->N, (uint32_t) p->r);
		return selftest_gate(thr_id, "yespower", false);
	}

	/* 1. this card reproduces the CPU reference */
	kat = (g[0] == cpu[0] && g[1] == cpu[1]);
	/* 2. and not a different job (rules out a constant result) */
	neg = (g[0] != cneg[0] || g[1] != cneg[1]);
	/* 3. acc[0] is permutation-blind; acc[1] is not. */
	permleg = (perm[0] == cpu[0] && perm[1] != cpu[1]);
	/* 4. a uniform perturbation over an even span cancels in acc[0] only */
	constleg = (evc[0] == ev[0] && evc[1] != ev[1]);

	passed = kat && neg && permleg && constleg;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "yespower self-test FAILED: kat=%d neg=%d perm=%d const=%d "
		                        "(gpu %016llx/%016llx vs cpu %016llx/%016llx)",
		       (int) kat, (int) neg, (int) permleg, (int) constleg,
		       (unsigned long long) g[0], (unsigned long long) g[1],
		       (unsigned long long) cpu[0], (unsigned long long) cpu[1]);

	return selftest_gate(thr_id, "yespower", passed);
}

/* -D differential, once per job: keyed off FNV-1a of pdata[0..18] (no nonce). */
static THREAD uint32_t yp_diff_sig[MAX_GPUS] = { 0 };

static uint32_t yp_job_sig(const uint32_t pdata[20])
{
	uint32_t h = 2166136261u;
	for (int i = 0; i < 19; i++) {
		const uint32_t w = pdata[i];
		for (int b = 0; b < 4; b++) { h ^= (w >> (b * 8)) & 0xffu; h *= 16777619u; }
	}
	return h ? h : 1u;
}

static void yespower_debug_differential(int thr_id, int dev, const uint32_t pdata[20],
                                        const yespower_params_t *p, uint32_t base)
{
	unsigned long long g[2], c[2];
	const uint32_t sig = yp_job_sig(pdata);
	const uint32_t start = base + YP_DIFF_OFF;

	if (yp_diff_sig[dev] == sig) return;
	yp_diff_sig[dev] = sig;

	if (!yp_gpu_acc(dev, p, start, YP_DIFF_SPAN, g)) {
		gpulog(LOG_WARNING, thr_id, "yespower differential: could not run (CUDA resource "
		                            "failure) -- not evidence of a wrong hash");
		return;
	}
	if (!yp_cpu_acc(pdata, p, start, YP_DIFF_SPAN, false, c, NULL, NULL, NULL)) {
		gpulog(LOG_WARNING, thr_id, "yespower differential: CPU reference refused the parameters");
		return;
	}

	if (g[0] == c[0] && g[1] == c[1])
		gpulog(LOG_DEBUG, thr_id, "yespower differential ok: %u nonces from %08x, "
		                          "acc %016llx/%016llx",
		       YP_DIFF_SPAN, start, (unsigned long long) g[0], (unsigned long long) g[1]);
	else
		gpulog(LOG_ERR, thr_id, "yespower DIFFERENTIAL MISMATCH over %u nonces from %08x: "
		                        "gpu %016llx/%016llx != cpu %016llx/%016llx",
		       YP_DIFF_SPAN, start,
		       (unsigned long long) g[0], (unsigned long long) g[1],
		       (unsigned long long) c[0], (unsigned long long) c[1]);
}

extern "C" int scanhash_yespower(int thr_id, struct work *work, uint32_t max_nonce,
                                 unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[YP_EQPAY_HDR_WORDS];   /* 46: sized for the longest header */
	uint32_t vhash[8];
	yespower_params_t params;
	uint32_t n = first_nonce;
	const int dev = device_map[thr_id];
	static THREAD bool announced = false;
	time_t t_start;

	if (!yespower_params_for(opt_algo, &params)) {
		applog(LOG_ERR, "yespower: algo %d is not a yespower variant", opt_algo);
		proper_exit(EXIT_CODE_USAGE);   /* permanent */
	}

	/* Reject what the reference rejects: N a power of two in [256, 512K], r in
	 * [8, 32]. Otherwise every hash fails, nothing is found and the loop reports a
	 * fictitious rate. */
	if (params.N < 256 || params.N > 512 * 1024 || (params.N & (params.N - 1))) {
		applog(LOG_ERR, "yespower: N=%u invalid (need a power of two in [256, 524288])",
		       params.N);
		proper_exit(EXIT_CODE_USAGE);   /* permanent: the next call would fail identically */
	}
	if (params.r < 8 || params.r > 32) {
		applog(LOG_ERR, "yespower: r=%u invalid (need 8..32)", params.r);
		proper_exit(EXIT_CODE_USAGE);   /* permanent */
	}

	/* only r = 8, 16, 32 have a kernel */
	if (params.r != 8 && params.r != 16 && params.r != 32) {
		applog(LOG_ERR, "yespower: r=%u has no GPU kernel (only 8, 16 and 32 are built)",
		       params.r);
		proper_exit(EXIT_CODE_USAGE);   /* permanent */
	}

	if (!announced) {
		announced = true;
		applog(LOG_NOTICE, "yespower 1.0: N=%u r=%u key=%s (%u bytes)",
		       params.N, params.r,
		       params.perslen ? (const char *) params.pers : "<none>",
		       (uint32_t) params.perslen);
	}

	/* Benchmark work keeps one header and restarts at nonce 0, so the same
	 * ~250-nonce window is scanned every pass; 0x07ffffff (p = 1/32) makes it
	 * contain a hit. Candidates are still re-hashed on the host. */
	if (opt_benchmark)
		ptarget[7] = 0x07ffffff;

	/* Nonce convention as algos/yescrypt/yescrypt.cu: be32enc every word, submit
	 * with the default le32enc. */
	const size_t hdr_bytes  = yp_header_bytes(opt_algo);
	const uint32_t hdr_words = (uint32_t)((hdr_bytes + 3u) / 4u);

	/* endiandata is the wire byte stream. stratum_gen_work stores EqPay's tail as
	 * the big-endian decode of the wire, so no word needs a special case. */
	for (uint32_t k = 0; k < hdr_words; k++)
		be32enc(&endiandata[k], pdata[k]);

	if (!init[dev]) {
		const uint32_t shneed = YP_SBOX_UINT4 * 16u;
		int optin = 0;
		CUDA_CALL_OR_RET_X(cudaSetDevice(device_map[thr_id]), -1);

		cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
		                       device_map[thr_id]);
		/* S placement: the global arena whenever VRAM holds 4 instances/SM (already
		 * faster than shared S, which pins one block per SM); shared S only as the
		 * fallback on a card that can opt in to 98304 B. */

		const size_t bw_q      = 32u * (size_t) params.r;
		const size_t v_inst_q  = bw_q * 4 * (size_t) params.N;
		const size_t per_inst_q = v_inst_q + bw_q * 4 * 2 + (size_t) shneed;
		size_t avail_q = (size_t) cuda_available_memory(thr_id) * 1024u * 1024u;
		avail_q = (avail_q > (256u << 20)) ? (avail_q - (256u << 20)) : 0u;
		const uint32_t fit_q = (uint32_t) (avail_q / per_inst_q);

		const bool global_fits = fit_q >= (uint32_t) device_mpcount[dev] * 4u;
		yp_sglobal[dev] = global_fits || ((uint32_t) optin < shneed);

		if (!yp_sglobal[dev] && (uint32_t) optin >= shneed)
			applog(LOG_WARNING, "yespower: GPU #%d has room for only %u instances; "
			                    "falling back to the shared S-box, which is ~2.1x slower",
			       device_map[thr_id], fit_q);
		/* power2b (alias yespower-b2b): BLAKE2b head/tail */
		yp_head[dev] = (opt_algo == ALGO_POWER2B)    ? YP_HEAD_B2B
	             : (opt_algo == ALGO_YESPOWEREQPAY) ? YP_HEAD_SHA256_181
	                                                : YP_HEAD_SHA256;
		if (yp_sglobal[dev] && (uint32_t) optin < shneed)
			applog(LOG_INFO, "yespower: GPU #%d offers %d B of shared memory per block "
			                 "(needs %u), using the global S-box path",
			       device_map[thr_id], optin, shneed);
		else if (yp_sglobal[dev])
			applog(LOG_INFO, "yespower: GPU #%d using the global S-box path "
			                 "(~2.1x the shared path -- occupancy beats access cost)",
			       device_map[thr_id]);

		if (!yp_sglobal[dev] && !yp_set_shared_limit(params.r, yp_head[dev])) {
			applog(LOG_ERR, "yespower: could not raise the dynamic shared limit to %u B",
			       shneed);
			proper_exit(EXIT_CODE_CUDA_ERROR);   /* permanent */
		}

		const size_t bw      = 32u * (size_t) params.r;
		const size_t sbytes  = (size_t) YP_SBOX_UINT4 * 16u;
		const size_t v_inst  = bw * 4 * (size_t) params.N;      /* V per instance */
		const size_t bx_inst = bw * 4 * 2;                      /* B + X          */

		if (!yp_sglobal[dev]) {
			/* S fills shared memory: one block per SM */
			yp_instances[dev] = (uint32_t) device_mpcount[dev];
		} else {
			/* No 96 KiB shared block, so many instances per SM; V scales with the count,
			 * so clamp to VRAM. */
			const size_t per_inst = v_inst + bx_inst + sbytes;
			size_t avail = (size_t) cuda_available_memory(thr_id) * 1024u * 1024u;
			const size_t reserve = 256u * 1024u * 1024u;
			avail = (avail > reserve) ? (avail - reserve) : 0u;

			uint32_t fit  = (uint32_t) (avail / per_inst);
			/* device_sm[] = compute capability x10 */
			const uint32_t ipb = (device_sm[dev] >= 700) ? YP_IPB_AMPERE : YP_IPB_OTHER;
			uint32_t want = (uint32_t) device_mpcount[dev] * ipb;
			fit -= fit % 4u;                       /* whole quads of instances */
			yp_instances[dev] = (want < fit) ? want : fit;

			if (yp_instances[dev] < 4u) {
				applog(LOG_ERR, "yespower: GPU #%d needs %.0f MB per instance and has "
				                "%d MB free -- not enough for the global S-box path",
				       device_map[thr_id], (double) per_inst / (1024.0 * 1024.0),
				       cuda_available_memory(thr_id));
				proper_exit(EXIT_CODE_CUDA_ERROR);
			}
			if (yp_instances[dev] < want)
				applog(LOG_WARNING, "yespower: GPU #%d VRAM caps this at %u instances "
				                    "(wanted %u) -- expect a lower rate",
				       device_map[thr_id], yp_instances[dev], want);
		}

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_V[dev], v_inst * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_B[dev], bw * 4 * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_X[dev], bw * 4 * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_res[dev], (1u + YP_MAX_RES) * sizeof(uint32_t)), -1);
		if (yp_sglobal[dev])
			CUDA_CALL_OR_RET_X(cudaMalloc(&d_S[dev], sbytes * yp_instances[dev]), -1);

		/* pers is per variant; an algo switch re-runs this via free_yespower(). */
		if (params.perslen > YP_PERS_MAX) {
			applog(LOG_ERR, "yespower: pers is %u bytes, the kernel holds %u",
			       (uint32_t) params.perslen, (uint32_t) YP_PERS_MAX);
			proper_exit(EXIT_CODE_USAGE);   /* permanent */
		}
		{
			uint8_t persbuf[YP_PERS_MAX] = { 0 };
			uint32_t plen = (uint32_t) params.perslen;
			if (plen) memcpy(persbuf, params.pers, plen);
			cudaMemcpyToSymbol(c_yp_pers, persbuf, YP_PERS_MAX);
			cudaMemcpyToSymbol(c_yp_perslen, &plen, sizeof(plen));
		}

		applog(LOG_INFO, "GPU #%d: yespower %u instances, %.0f MB of V, S in %s",
		       device_map[thr_id], yp_instances[dev],
		       (double) (v_inst * (double) yp_instances[dev]) / (1024.0 * 1024.0),
		       yp_sglobal[dev] ? "global" : "shared");
		/* Fail-closed self-test; needs the buffers above. */
		yespower_device_selftest(thr_id, dev, &params);
		init[dev] = true;
	}

	/* Upload pdata, not endiandata: the kernel wants SHA-256 input words, i.e. the
	 * big-endian decode of the byte stream, which is pdata itself. */
	cudaMemcpyToSymbol(c_yp_hdr, pdata, hdr_words * sizeof(uint32_t));
	cudaMemcpyToSymbol(c_yp_target, ptarget, 32);

	/* -D: per-job GPU-vs-CPU differential. The host re-verify cannot see a
	 * missed nonce; this can. */
	if (opt_debug)
		yespower_debug_differential(thr_id, dev, pdata, &params, first_nonce);

	/* Return about once a second so the miner reports a rate and sees new work. */
	t_start = time(NULL);

	do {
		const uint32_t batch = yp_instances[dev];
		/* arm, read back and allocate the same size */
		uint32_t res[1u + YP_MAX_RES] = { 0 };

		cudaMemcpy(d_res[dev], res, sizeof(res), cudaMemcpyHostToDevice);

		if (!yp_launch(params.r, batch, n, params.N, dev, yp_sglobal[dev], yp_head[dev])) {
			applog(LOG_ERR, "yespower: no kernel for r=%u", params.r);
			return -1;
		}
		if (cudaGetLastError() != cudaSuccess ||
		    cudaMemcpy(res, d_res[dev], sizeof(res), cudaMemcpyDeviceToHost) != cudaSuccess) {
			applog(LOG_ERR, "yespower: GPU #%d launch failed: %s",
			       device_map[thr_id], cudaGetErrorString(cudaGetLastError()));
			return -1;
		}

		*hashes_done = n - first_nonce + batch;

		/* Every candidate is re-hashed on the host before it can become a share. */
		if (res[0] != 0u) {
			uint32_t cand[YP_MAX_RES];
			uint32_t ncand = res[0];
			int found = 0;

			if (ncand > YP_MAX_RES) {
				applog(LOG_WARNING, "GPU #%d: yespower candidates flood: %u (keeping %u)",
				       device_map[thr_id], ncand, (uint32_t) YP_MAX_RES);
				ncand = YP_MAX_RES;
			}
			for (uint32_t s = 0; s < ncand; s++) cand[s] = res[1u + s];

			/* ascending: the lowest are examined first */
			for (uint32_t a = 1; a < ncand; a++) {
				const uint32_t v = cand[a];
				uint32_t b = a;
				while (b > 0 && cand[b - 1] > v) { cand[b] = cand[b - 1]; b--; }
				cand[b] = v;
			}

			/* Examine the lowest candidates, as many as work can carry. */
			uint32_t examined = 0;
			for (uint32_t s = 0; s < ncand && found < MAX_NONCES; s++) {
				examined = s + 1;
				be32enc(&endiandata[19], cand[s]);
				if (!yespower_hash_hdr(vhash, endiandata, hdr_bytes, &params)) {
					applog(LOG_ERR, "yespower: host re-verify failed at nonce %08x", cand[s]);
					return -1;
				}
				if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
					work->nonces[found] = cand[s];
					bn_set_target_ratio(work, vhash, found);
					found++;
				} else {
					gpu_increment_reject(thr_id);
					applog(LOG_WARNING, "GPU #%d: yespower result %08x does not validate",
					       device_map[thr_id], cand[s]);
				}
			}
			if (found) {
				/* work->nonces[] is what gets submitted; pdata[19] alone would submit 0. */
				work->valid_nonces = found;
				/* The launch covered [n, n + batch): if every candidate was kept and
				 * examined, resume past it; otherwise just above the highest one examined,
				 * so the rest are found again. */
				uint32_t resume = (res[0] <= YP_MAX_RES && examined == ncand)
				                  ? n + batch
				                  : cand[examined - 1] + 1;
				if (resume > max_nonce || resume < first_nonce)
					resume = max_nonce;
				pdata[19] = resume;
				*hashes_done = resume - first_nonce;
				return found;
			}
		}

		n += batch;
		if (time(NULL) - t_start >= 1)
			break;                       /* yield: report the rate, poll for work */
	} while ((uint64_t) n + yp_instances[dev] < (uint64_t) max_nonce &&
	         !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce;
	pdata[19] = n;
	return 0;
}

/* Frees the buffers and re-arms init[dev]. Must stay in algo_free_all(): the
 * buffer sizes, pers and head/tail primitive are latched at init, so an algo
 * switch needs a re-init. */
extern "C" void free_yespower(int thr_id)
{
	const int dev = device_map[thr_id];

	if (!init[dev])
		return;

	cudaSetDevice(dev);
	cudaDeviceSynchronize();

	if (d_V[dev])   cudaFree(d_V[dev]);
	if (d_B[dev])   cudaFree(d_B[dev]);
	if (d_X[dev])   cudaFree(d_X[dev]);
	if (d_S[dev])   cudaFree(d_S[dev]);
	if (d_res[dev]) cudaFree(d_res[dev]);

	d_V[dev] = NULL; d_B[dev] = NULL; d_X[dev] = NULL;
	d_S[dev] = NULL; d_res[dev] = NULL;

	yp_instances[dev] = 0;
	yp_sglobal[dev]   = false;
	yp_head[dev]      = YP_HEAD_SHA256;
	yp_diff_sig[dev]  = 0;    /* the -D differential re-runs on new params */

	init[dev] = false;
}
