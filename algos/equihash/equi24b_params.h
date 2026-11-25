// ---------------------------------------------------------------------------
// Geometry for the DIGITBITS=24 solver (144/5 and 192/7): few large buckets.
//
// The scatter is bound by its write frontier, one open 32 B sector per
// destination bucket; with 4096 buckets that frontier stays resident in L2.
//
//   * a bucket is never fully staged in shared memory; each round processes
//     it in chunks split by REST PREFIX, which is lossless because items whose
//     rest differs in the high bits cannot collide
//   * collisions are found with a per-rest linked list in shared memory
//     (atomicExch on a head array)
//   * layers are independent arrays, so nothing is overwritten and the
//     reference may be 8 bytes
//
// Record strides are 8, 16 or 32 B only: a scattered record costs one sector
// transaction, and any other size straddles sectors.
// ---------------------------------------------------------------------------
#ifndef EQUI24B_PARAMS_H
#define EQUI24B_PARAMS_H

#if !defined(EQ_WN) || !defined(EQ_WK)
#error "equi24b_params.h needs -DEQ_WN and -DEQ_WK (144/5 or 192/7)"
#endif

#define EQ24B_NDIGITS      (EQ_WK + 1)
#define EQ24B_DIGITBITS    (EQ_WN / EQ24B_NDIGITS)

#define EQ24B_BUCKBITS     12u
#define EQ24B_RESTBITS     (EQ24B_DIGITBITS - EQ24B_BUCKBITS)
#define EQ24B_NBUCKETS     (1u << EQ24B_BUCKBITS)
#define EQ24B_NRESTS       (1u << EQ24B_RESTBITS)
#define EQ24B_BUCKMASK     (EQ24B_NBUCKETS - 1u)
#define EQ24B_RESTMASK     (EQ24B_NRESTS - 1u)

// Slots per bucket: mean 2^25 / 4096 = 8192, sigma 90.5, so 8704 is mean + 5.7 sigma.
#define EQ24B_NSLOTS       8704u

// ---- reference word: 8 bytes (source bucket + Cantor code of two slots) ------
#define EQ24B_CANTOR_MAX   (((eq_u64)EQ24B_NSLOTS * (EQ24B_NSLOTS - 1u)) / 2u)
#define EQ24B_CANTORBITS   26u
#define EQ24B_CANTORMASK   ((((eq_u64)1) << EQ24B_CANTORBITS) - 1u)
#define EQ24B_REFBITS      (EQ24B_BUCKBITS + EQ24B_CANTORBITS)

// ---- payload schedule and record layout ------------------------------------
// Bits still to cancel after round r, less the bucket index the position encodes.
#define EQ24B_HASHBITS(r)  (EQ_WN - (r) * EQ24B_DIGITBITS - EQ24B_BUCKBITS)
#define EQ24B_PAYLOAD(r)   ((EQ24B_HASHBITS(r) + 7) / 8)
// Record: [8 B ref][payload]. A layer whose payload fits 16 B only without the
// ref's 26 unused high bits is PACKED instead: [ref bits 0-31][payload][ref bits
// 32-37 in byte 15], so it takes one 16 B sector slot instead of a 32 B one.
// Layer 0 holds the blake index (4 B) and keeps the plain layout.
//
// One LOGGED layer (14-byte payload) is 16 B too: [payload bytes 0-13][bytes
// 14-15]. The last 6 hash bits (byte 13 bits 0-5) move to a pair log and the
// layer carries a 26-bit pointer to its log entry instead of a ref: pointer
// bits 0-15 in bytes 14-15, 16-21 in byte 13 bits 0-5, 22-25 in byte 0 bits
// 4-7 (the next bucket's low nibble, implied by the position). Each round block
// appends its entries to its own LOGCAP region of the log, so log writes are
// sequential; the parent bucket is the region index / CHUNKS. A log word is
// [Cantor of the origin slots, 26 bits][the 6 hash bits]. Later layers carry
// those bits as zero, so digitK tests the rest and expandSols the 6 bits.
#if EQ_WN == 144
#define EQ24B_LOGL         1u
#else
#define EQ24B_LOGL         3u
#endif
#define EQ24B_LOGGED(r)    ((r) == EQ24B_LOGL)
// The LAST layer (EQ_WK - 1) is 8 B and TRUNCATED: [word 0: payload bytes 0-3]
// [word 1: ref bits 0-31], ref bits 32-35 in byte 0 bits 4-7 (the bucket
// nibble, implied by the position) and 36-37 in byte 3 bits 6-7. It keeps the
// rest plus 14 bits of the final digit; digitK tests those, and expandSols
// checks every hash bit of a candidate against layer 0.
#define EQ24B_TRUNCL       (EQ_WK - 1u)
#define EQ24B_TRUNC(r)     ((r) == EQ24B_TRUNCL)
#define EQ24B_TRUNC_MASK   0x3fffff0fu
#define EQ24B_LOGCAP       3584u
#define EQ24B_LOG_OFF_W    (EQ24B_NBUCKETS * EQ24B_NSLOTS * 4u)   // log offset in the layer, words
#define EQ24B_LOG_WORDS    (EQ24B_NBUCKETS * EQ24B_CHUNKS_R(EQ24B_LOGL) * EQ24B_LOGCAP)
// 144/5 layer 0 is COMPACT: a 16 B record of payload bytes 0-15 (hash bytes
// 1-16) whose byte 0 bits 4-7 (the bucket nibble, implied by the position)
// carry hash byte 17 bits 0-3, plus a 4 B SIDE word at the same index in a
// separate array: [leaf index, 25 bits][hash byte 17 bits 4-7]. Both are
// slot-major. 192/7's layer 0 does not fit 16 B and keeps [index | pad | payload].
#if EQ_WN == 144
#define EQ24B_L0C          1u
#else
#define EQ24B_L0C          0u
#endif
#define EQ24B_L0C_SIDE_OFF_W (EQ24B_NBUCKETS * EQ24B_NSLOTS * 4u)   // side array offset, words
#define EQ24B_PACKED(r)    ((r) > 0u && !EQ24B_LOGGED(r) && !EQ24B_TRUNC(r) && 8u + EQ24B_PAYLOAD(r) > 16u && 4u + EQ24B_PAYLOAD(r) <= 15u)
#define EQ24B_POFF_W(r)    ((EQ24B_LOGGED(r) || EQ24B_TRUNC(r) || (EQ24B_L0C && (r) == 0u)) ? 0u : (EQ24B_PACKED(r) ? 1u : 2u))   // payload word offset
#define EQ24B_RECBYTES(r)  (EQ24B_TRUNC(r) ? 8u : (EQ24B_PACKED(r) || EQ24B_LOGGED(r)) ? 16u : 8u + EQ24B_PAYLOAD(r))
// Payload bits valid in its last word (readers mask the rest: a packed layer
// keeps ref bits there, a logged one pointer bits, and the zero-XOR test must
// not see them).
#define EQ24B_PMASK(r)     (EQ24B_LOGGED(r) ? 0x0000c0ffu : EQ24B_TRUNC(r) ? EQ24B_TRUNC_MASK : \
	((EQ24B_PAYLOAD(r) & 3u) ? ((1u << (8u * (EQ24B_PAYLOAD(r) & 3u))) - 1u) : 0xffffffffu))
#define EQ24B_STRIDE(r) ((EQ24B_L0C && (r) == 0u) ? 16u : \
	(EQ24B_RECBYTES(r) <= 8u ? 8u : (EQ24B_RECBYTES(r) <= 16u ? 16u : 32u)))
// Record index of (bucket, slot), every layer SLOT-MAJOR in groups of 32: the
// slots every bucket is filling at the same time sit in one window, so the
// scatter's write front stays compact.
#define EQ24B_RIDX(r, b, s) \
	(((size_t)((s) >> 5) * EQ24B_NBUCKETS + (b)) * 32u + ((s) & 31u))
#define EQ24B_LAYER_BYTES(r) \
	((eq_u64)EQ24B_NBUCKETS * EQ24B_NSLOTS * EQ24B_STRIDE(r) + (EQ24B_LOGGED(r) ? (eq_u64)EQ24B_LOG_WORDS * 4u : 0u) \
	 + ((EQ24B_L0C && (r) == 0u) ? (eq_u64)EQ24B_NBUCKETS * EQ24B_NSLOTS * 4u : 0u))

#ifdef __cplusplus
#include <cstdint>
typedef std::uint64_t eq_u64;

static_assert(EQ24B_DIGITBITS == 24,
	"equi24b is the DIGITBITS=24 family only (144/5, 192/7)");
static_assert(EQ24B_BUCKBITS + EQ24B_RESTBITS == EQ24B_DIGITBITS,
	"bucket and rest bits must partition a digit");
static_assert(EQ24B_CANTOR_MAX < (((eq_u64)1) << EQ24B_CANTORBITS),
	"Cantor code overflows its field at this NSLOTS");
static_assert(EQ24B_REFBITS <= 64,
	"reference word must fit 64 bits");
static_assert(EQ24B_REFBITS > 32,
	"reference exceeds 32 bits by design -- layers must NOT be staggered");
static_assert(EQ24B_REFBITS <= 40,
	"a packed record keeps ref bits 32+ in one byte");
#endif

// ---- digitR staging ---------------------------------------------------------
// One block per (bucket, chunk). Per staged item: the input payload minus its
// LAST byte (final-digit bits no layer keeps, see EQ24B_TRUNCL), as STAGE_W
// words plus, when 1-2 bytes remain, a u16 tail (STAGE_H), and a 2-byte list
// link. The ORIGIN SLOT the reference names (the staging index is an append
// order) replaces payload bytes 0-1, the rest, which is equal within a list and
// never emitted. Heads cover only the chunk's rest range and are u32 because
// there is no 16-bit atomicExch. The heads division rounds UP, or the top rests
// would select a chunk never visited.
#define EQ24B_SBYTES(r)      (EQ24B_PAYLOAD((r) - 1u) - 1u)
#define EQ24B_STAGE_W(r)     (EQ24B_SBYTES(r) / 4u + ((EQ24B_SBYTES(r) & 3u) > 2u ? 1u : 0u))
#define EQ24B_STAGE_H(r)     ((EQ24B_SBYTES(r) & 3u) != 0u && (EQ24B_SBYTES(r) & 3u) <= 2u ? 1u : 0u)

// Chunks per digitR round (digitK runs one block per bucket).
#if EQ_WN == 144
#define EQ24B_CHUNKS_R(r)    ((r) <= 2u ? 4u : 3u)
#else
#define EQ24B_CHUNKS_R(r)    ((r) == 1u ? 6u : (r) == 2u ? 5u : (r) <= 4u ? 4u : 3u)
#endif
#define EQ24B_HEADS(r)       ((EQ24B_NRESTS + EQ24B_CHUNKS_R(r) - 1u) / EQ24B_CHUNKS_R(r))

// 48 KB minus 512 B for the static counters: a dynamic request of exactly
// 49152 B fails the launch with cudaErrorInvalidValue.
#define EQ24B_SHARED_BUDGET  (49152u - 512u)
#define EQ24B_LIST_END       0xffffu
#define EQ24B_PROOFSIZE      (1u << EQ_WK)
#define EQ24B_STAGE_SLOTS(r) ((EQ24B_SHARED_BUDGET - EQ24B_HEADS(r) * 4u - 4u) / (EQ24B_STAGE_W(r) * 4u + EQ24B_STAGE_H(r) * 2u + 2u))
// Word offset of the head array: payload words, then the u16 links and the u16
// tails, padded to a word.
#define EQ24B_HEAD_OFF_W(r)  (EQ24B_STAGE_SLOTS(r) * EQ24B_STAGE_W(r) \
	+ (EQ24B_STAGE_SLOTS(r) * (1u + EQ24B_STAGE_H(r)) + 1u) / 2u)
#define EQ24B_SHARED_BYTES(r) ((EQ24B_HEAD_OFF_W(r) + EQ24B_HEADS(r)) * 4u)

#ifdef __cplusplus
// Per round: staging fits the budget, one block holds a full bucket's share of
// its chunk + 6 sigma (binomial), and its indices fit the u16 links
// (EQ24B_LIST_END is reserved).
#define EQ24B_SHARE(r)       (EQ24B_NSLOTS * EQ24B_HEADS(r) / EQ24B_NRESTS)
#define EQ24B_ROUND_OK(r) \
	(EQ24B_SHARED_BYTES(r) <= EQ24B_SHARED_BUDGET \
	 && EQ24B_STAGE_SLOTS(r) > EQ24B_SHARE(r) \
	 && (EQ24B_STAGE_SLOTS(r) - EQ24B_SHARE(r)) * (EQ24B_STAGE_SLOTS(r) - EQ24B_SHARE(r)) >= 36u * EQ24B_SHARE(r) \
	 && EQ24B_STAGE_SLOTS(r) < EQ24B_LIST_END)
static_assert(EQ24B_ROUND_OK(1u) && EQ24B_ROUND_OK(2u) && EQ24B_ROUND_OK(3u) && EQ24B_ROUND_OK(4u),
	"digitR staging out of budget or capacity");
static_assert(EQ24B_NSLOTS <= 0x10000u, "the staged origin slot is 16 bits");
static_assert(EQ24B_PAYLOAD(EQ24B_LOGL) == 14u && EQ24B_HASHBITS(EQ24B_LOGL) == 108u,
	"the logged layer layout assumes a 108-bit, 14-byte payload");
static_assert(EQ24B_PAYLOAD(EQ24B_TRUNCL) == 5u && EQ24B_REFBITS == 38u,
	"the truncated layer layout assumes a 5-byte payload and a 38-bit ref");
static_assert(!EQ24B_L0C || (EQ24B_PAYLOAD(0) == 17u && EQ24B_SBYTES(1u) == 16u && (2u << EQ24B_DIGITBITS) <= (1u << 25)),
	"the compact layer 0 assumes a 17-byte payload, 16 staged bytes and 25-bit leaf indices");
static_assert((eq_u64)EQ24B_LOG_WORDS <= ((eq_u64)1 << 26), "log pointer is 26 bits");
static_assert(EQ24B_LOGCAP * EQ24B_CHUNKS_R(EQ24B_LOGL) * 2u >= 3u * EQ24B_NSLOTS,
	"a block's log region must hold 1.5x its mean output");
#if EQ_WK > 5
static_assert(EQ24B_ROUND_OK(5u) && EQ24B_ROUND_OK(6u),
	"digitR staging out of budget or capacity");
#endif
#endif

#endif // EQUI24B_PARAMS_H
