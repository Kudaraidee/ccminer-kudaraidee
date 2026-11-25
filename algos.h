#ifndef ALGOS_H
#define ALGOS_H

#include <string.h>
#include "compat.h"

enum sha_algos {
	ALGO_ANIME = 0,
	ALGO_ARGON2D1000,
	ALGO_ARGON2D16000,
	ALGO_ARGON2D4096,
	ALGO_ARGON2D500,
	ALGO_ARGON2ID1024,
	ALGO_BALLOON,
	ALGO_BLAKECOIN,
	ALGO_BLAKE,
	ALGO_BLAKE2B,
	ALGO_BLAKE2S,
	ALGO_BMW,
	ALGO_BMW512,
	ALGO_BASTION,
	ALGO_C11,
	ALGO_CRYPTOLIGHT,
	ALGO_CRYPTONIGHT,
	ALGO_DEEP,
	ALGO_DECRED,
	ALGO_DMD_GR,
	ALGO_EQUIHASH,
	ALGO_FRESH,
	ALGO_FUGUE256,		/* Fugue256 */
	ALGO_GOSTCOIN,
	ALGO_GROESTL,
	ALGO_HEAVY,		/* Heavycoin hash */
	ALGO_HMQ1725,
	ALGO_HSR,
	ALGO_KECCAK,
	ALGO_KECCAKC,		/* refreshed Keccak with pool factor 256 */
	ALGO_JACKPOT,
	ALGO_JHA,
	ALGO_LBRY,
	ALGO_LUFFA,
	ALGO_LYRA2,
	ALGO_ALLIUM,
	ALGO_HEAVYHASH,
	ALGO_LYRA2v2,
	ALGO_LYRA2Z,
	ALGO_MJOLLNIR,		/* Hefty hash */
	ALGO_MYR_GR,
	ALGO_NEOSCRYPT,
	ALGO_XAYA,
	ALGO_NIST5,
	ALGO_PENTABLAKE,
	ALGO_PHI,
	ALGO_POLYTIMOS,
	ALGO_QUARK,
	ALGO_QUBIT,
	ALGO_SCRYPT,
	ALGO_SCRYPT_JANE,
	ALGO_SHA256CSM,
	ALGO_SHA256D,
	ALGO_SHA256T,
	ALGO_SHA256DV,
	ALGO_SHA3D,
	ALGO_SIA,
	ALGO_SIB,
	ALGO_SKEIN,
	ALGO_SKEIN2,
	ALGO_SKUNK,
	ALGO_S3,
	ALGO_TIMETRAVEL,
	ALGO_TRIBUS,
	ALGO_BITCORE,
	ALGO_X11EVO,
	ALGO_0X10,
	ALGO_X11,
	ALGO_X13,
	ALGO_X14,
	ALGO_X15,
	ALGO_X16R,
	ALGO_X16RT,
	ALGO_X16RV2,
	ALGO_X16S,
	ALGO_X17,
	ALGO_X21S,
	ALGO_VANILLA,
	ALGO_VELTOR,
	ALGO_WHIRLCOIN,
	ALGO_WHIRLPOOL,
	ALGO_WHIRLPOOLX,
	ALGO_WHIRLPOOLX2,
	ALGO_WILDKECCAK,
	ALGO_LYRA2RE,
	ALGO_LYRA2Z330,
	ALGO_YESCRYPT,
	ALGO_YESCRYPTR8,
	ALGO_YESCRYPTR16,
	ALGO_YESCRYPTR16V2,
	ALGO_YESCRYPTR24,
	ALGO_YESCRYPTR32,
	ALGO_ZR5,
	ALGO_RINHASH,
	ALGO_EVOHASH,
	ALGO_ODO,
	ALGO_SKYDOGE,
	ALGO_HOOHASH,
	ALGO_GHOSTRIDER,
	ALGO_SHA3T,
	ALGO_SHA512256D,
	ALGO_SOTERG,
	ALGO_X25X,
	ALGO_CURVEHASH,
	ALGO_KAWPOW,
	ALGO_MEOWPOW,
	ALGO_EVRPROGPOW,
	ALGO_FIROPOW,
	ALGO_MERAKI,
	ALGO_VERTHASH,
	ALGO_YESPOWER,
	ALGO_YESPOWERR16,
	ALGO_POWER2B,
	ALGO_YESPOWEREQPAY,
	ALGO_MIKE,
	ALGO_FLEX,
	ALGO_MINOTAURX,
	ALGO_HOMESCRYPT,
	ALGO_VERUS,
	ALGO_BC3HASH,
	ALGO_PEARLHASH,
	ALGO_AUTO,
	ALGO_COUNT
};

extern volatile enum sha_algos opt_algo;

// ProgPoW-over-ethash family (KawPoW + the MeowPow/EvrProgPow/FiroPoW variants
// in algos/progpow_multi/). They share the ethproxy/kawpow stratum wire format
// and the work->kawpow_* job fields, so the stratum/submit/gen_work paths treat
// them uniformly; only the scanhash dispatch differs per variant.
static inline int is_progpow_algo(int a)
{
	return a == ALGO_KAWPOW || a == ALGO_MEOWPOW || a == ALGO_EVRPROGPOW ||
	       a == ALGO_FIROPOW || a == ALGO_MERAKI;
}

static const char *algo_names[] = {
	"anime",
	"argon2d1000",
	"argon2d16000",
	"argon2d4096",
	"argon2d500",
	"argon2id1024",
	"balloon",
	"blakecoin",
	"blake",
	"blake2b",
	"blake2s",
	"bmw",
	"bmw512",
	"bastion",
	"c11",
	"cryptolight",
	"cryptonight",
	"deep",
	"decred",
	"dmd-gr",
	"equihash",
	"fresh",
	"fugue256",
	"gostcoin",
	"groestl",
	"heavy",
	"hmq1725",
	"hsr",
	"keccak",
	"keccakc",
	"jackpot",
	"jha",
	"lbry",
	"luffa",
	"lyra2",
	"allium",
	"heavyhash",
	"lyra2v2",
	"lyra2z",
	"mjollnir",
	"myr-gr",
	"neoscrypt",
	"xaya",
	"nist5",
	"penta",
	"phi",
	"polytimos",
	"quark",
	"qubit",
	"scrypt",
	"scrypt-jane",
	"sha256csm",
	"sha256d",
	"sha256t",
	"sha256dv",
	"sha3d",
	"sia",
	"sib",
	"skein",
	"skein2",
	"skunk",
	"s3",
	"timetravel",
	"tribus",
	"bitcore",
	"x11evo",
	"0x10",
	"x11",
	"x13",
	"x14",
	"x15",
	"x16r",
	"x16rt",
	"x16rv2",
	"x16s",
	"x17",
	"x21s",
	"vanilla",
	"veltor",
	"whirlcoin",
	"whirlpool",
	"whirlpoolx",
	"whirlpoolx2",
	"wildkeccak",
	/* NULL, not "lyra2re": that name ALSO exists in the alias chain below, mapping to
	 * ALGO_LYRA2 -- but algo_to_int() searches THIS table first and returned the orphan
	 * ALGO_LYRA2RE, which has no scanhash dispatch, so `-a lyra2re` printed
	 * "not available in this build" and exited. The lookup already handles NULL
	 * (`if (algo_names[i] && ...)`), so the slot is skipped and the alias now fires.
	 * A slot rather than a deletion, so the enum is NOT renumbered. */
	NULL,
	"lyra2z330",
	"yescrypt",
	"yescryptr8",
	"yescryptr16",
	"yescryptr16v2",
	"yescryptr24",
	"yescryptr32",
	"zr5",
	"rinhash",
	"evohash",
	"odo",
	"skydoge",
	"hoohash",
	"ghostrider",
	"sha3t",
	"sha512256d",
	"soterg",
	"x25x",
	"curvehash",
	"kawpow",
	"meowpow",
	"evrprogpow",
	"firopow",
	"meraki",
	"verthash",
	"yespower",
	"yespowerr16",
	"power2b",
	"yespowereqpay",
	"mike",
	"flex",
	"minotaurx",
	"homescrypt",
	"verus",
	"bc3hash",
	"pearlhash",
	"auto", /* reserved for multi algo */
	""
};

// string to int/enum
static inline int algo_to_int(char* arg)
{
	int i;

	for (i = 0; i < ALGO_COUNT; i++) {
		if (algo_names[i] && !strcasecmp(arg, algo_names[i])) {
			return i;
		}
	}

	if (i == ALGO_COUNT) {
		// some aliases...
		if (!strcasecmp("all", arg))
			i = ALGO_AUTO;
		/* Every wild yespower 1.0 coin is -a yespower with a distinct
		 * --yespower-key on the r=32 default, so these are aliases, not algos.
		 * The parameters each one needs live in the table in
		 * algos/yespower/yespower.cu, which `-a` applies via
		 * yespower_set_variant(); a name added here and not there aborts at
		 * startup rather than mining generic parameters. */
		else if (!strcasecmp("yespower-b2b", arg))
			i = ALGO_POWER2B;
		else if (!strcasecmp("yenten", arg))
			i = ALGO_YESPOWERR16;
		else if (!strcasecmp("yespowersugar", arg) ||
		         !strcasecmp("sugarchain", arg) ||
		         !strcasecmp("yespowerurx", arg) ||
		         !strcasecmp("yespowerltncg", arg) ||
		         !strcasecmp("yespowermgpc", arg) ||
		         !strcasecmp("yespowertide", arg) ||
		         !strcasecmp("yespowersmt", arg) ||
		         !strcasecmp("yespowerarwn", arg) ||
		         !strcasecmp("yespoweradvc", arg) ||
		         !strcasecmp("yespoweric", arg) ||
		         !strcasecmp("yespoweriots", arg) ||
		         !strcasecmp("yespowerlitb", arg) ||
		         !strcasecmp("cpupower", arg))
			i = ALGO_YESPOWER;
		else if (!strcasecmp("hoohashv110", arg))
			i = ALGO_HOOHASH;
		else if (!strcasecmp("verushash", arg))
			i = ALGO_VERUS;
		else if (!strcasecmp("pepepow", arg))
			i = ALGO_HOOHASH;
		else if (!strcasecmp("gr", arg))
			i = ALGO_GHOSTRIDER;
		else if (!strcasecmp("sha3-256t", arg))
			i = ALGO_SHA3T;
		else if (!strcasecmp("cryptonight-light", arg))
			i = ALGO_CRYPTOLIGHT;
		else if (!strcasecmp("cryptonight-lite", arg))
			i = ALGO_CRYPTOLIGHT;
		else if (!strcasecmp("flax", arg))
			i = ALGO_C11;
		else if (!strcasecmp("diamond", arg))
			i = ALGO_DMD_GR;
		else if (!strcasecmp("equi", arg))
			i = ALGO_EQUIHASH;
		else if (!strcasecmp("equihash144", arg) || !strcasecmp("equihash144_5", arg))
			i = ALGO_EQUIHASH; // 144/5 variant; the (n,k)+personalization is set in parse_arg 'a'
		else if (!strcasecmp("equihash192", arg) || !strcasecmp("equihash192_7", arg))
			i = ALGO_EQUIHASH; // 192/7 variant; (n,k)+personalization set in parse_arg 'a'
		else if (!strcasecmp("doom", arg))
			i = ALGO_LUFFA;
		else if (!strcasecmp("hmq17", arg))
			i = ALGO_HMQ1725;
		else if (!strcasecmp("hshare", arg))
			i = ALGO_HSR;
		else if (!strcasecmp("lyra2re", arg))
			i = ALGO_LYRA2;
		else if (!strcasecmp("lyra2rev2", arg))
			i = ALGO_LYRA2v2;
		else if (!strcasecmp("phi1612", arg))
			i = ALGO_PHI;
		else if (!strcasecmp("bitcoin", arg))
			i = ALGO_SHA256D;
		else if (!strcasecmp("sha256", arg))
			i = ALGO_SHA256D;
		else if (!strcasecmp("gostcoin", arg))
			i = ALGO_GOSTCOIN;
		else if (!strcasecmp("thorsriddle", arg))
			i = ALGO_VELTOR;
		else if (!strcasecmp("timetravel10", arg))
			i = ALGO_BITCORE;
		else if (!strcasecmp("whirl", arg))
			i = ALGO_WHIRLPOOL;
		else if (!strcasecmp("ziftr", arg))
			i = ALGO_ZR5;
		else if (!strcasecmp("neoscrypt-xaya", arg))
			i = ALGO_XAYA;
		else if (!strcasecmp("odocrypt", arg))
			i = ALGO_ODO;
		else if (!strcasecmp("curve", arg))
			i = ALGO_CURVEHASH;
		else if (!strcasecmp("bc3", arg))
			i = ALGO_BC3HASH;
		else if (!strcasecmp("pearl", arg))
			i = ALGO_PEARLHASH;
		else
			i = -1;
	}

	return i;
}

#endif
