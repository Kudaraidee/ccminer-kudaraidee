/* VerusHash 2.2 host chain, transcribed from cpuminer-opt algo/verus/verus-gate.c
 * (itself from the VerusCoin ccminer, verusscan.cpp). */
#include "verus_host_target.h"
#include "verus_host.h"
#include "verus-simd.h"

#include <string.h>

#if defined(VERUS_HAVE_SIMD)

#include "haraka.h"
#include "haraka_portable.h"
#include "verus_clhash.h"

#if defined(_MSC_VER)
#include <intrin.h>
#define VH_ALIGN(n) __declspec(align(n))
#else
#include <cpuid.h>
#define VH_ALIGN(n) __attribute__((aligned(n)))
#endif

int verus_host_cpu_ok( void )
{
	unsigned int r[4] = { 0 };
#if defined(_MSC_VER)
	int ri[4];
	__cpuid( ri, 1 );
	memcpy( r, ri, sizeof r );
#else
	if ( !__get_cpuid( 1, &r[0], &r[1], &r[2], &r[3] ) ) return 0;
#endif
	/* leaf 1 ECX: bit 1 PCLMULQDQ, bit 9 SSSE3, bit 25 AES */
	return ( r[2] & ( 1u << 1 ) ) && ( r[2] & ( 1u << 9 ) ) && ( r[2] & ( 1u << 25 ) );
}

static void GenNewCLKey( unsigned char *seedBytes32, u128 *keyback )
{
	const int n256blks = VERUS_KEY_SIZE >> 5;     /* 276, no tail */
	unsigned char *pkey = (unsigned char*)keyback;
	unsigned char *psrc = seedBytes32;

	for ( int i = 0; i < n256blks; i++ )
	{
		haraka256( pkey, psrc );
		psrc  = pkey;
		pkey += 32;
	}
}

/* Undo the <= 64 mutated slots. Reverse order: indices repeat across
 * iterations, so the oldest saved value lands last; within an iteration
 * prandex before prand, mirroring the algorithm's write order on a collision. */
static void FixKey( uint32_t *fixrand, uint32_t *fixrandex, u128 *keyback,
                    u128 *g_prand, u128 *g_prandex )
{
	for ( int i = 31; i > -1; i-- )
	{
		keyback[ fixrandex[i] ] = g_prandex[i];
		keyback[ fixrand[i]   ] = g_prand[i];
	}
}

static void VerusHashHalf( void *result2, const unsigned char *data, int len )
{
	VH_ALIGN(32) unsigned char buf1[64] = { 0 }, buf2[64];
	unsigned char *curBuf = buf1, *result = buf2, *tmp;
	int curPos = 0;

	load_constants();
	load_constants_port();

	for ( int pos = 0; pos < len; )
	{
		int room = 32 - curPos;
		if ( len - pos >= room )
		{
			memcpy( curBuf + 32 + curPos, data + pos, room );
			haraka512( result, curBuf );
			tmp = curBuf; curBuf = result; result = tmp;
			pos += room; curPos = 0;
		}
		else
		{
			memcpy( curBuf + 32 + curPos, data + pos, len - pos );
			curPos += len - pos;
			pos = len;
		}
	}
	/* FillExtra: pad[0..15] = state[0..15], pad[16] = state[0] */
	memcpy( curBuf + 47, curBuf, 16 );
	memcpy( curBuf + 63, curBuf, 1 );
	memcpy( result2, curBuf, 64 );
}

int verus_host_prologue( const uint8_t *pre, int half_len, uint8_t half[64],
                         void *keybuf )
{
	if ( !verus_host_cpu_ok() ) return 0;
	VerusHashHalf( half, pre, half_len );
	GenNewCLKey( half, (u128*)keybuf );
	return 1;
}

uint64_t verus_host_hash( uint8_t hash[32], const uint8_t half[64],
                          const uint8_t nonce15[15], void *keybuf, int full )
{
	u128 *key = (u128*)keybuf;
	u128 *g_prand   = key + VERUS_KEY_SIZE128;
	u128 *g_prandex = key + VERUS_KEY_SIZE128 + 32;
	uint32_t fixrand[32], fixrandex[32];
	VH_ALIGN(32) unsigned char curBuf[64];

	memcpy( curBuf, half, 64 );
	memcpy( curBuf + 32, nonce15, VERUS_NONCE_SPACE );

	uint64_t intermediate = verusclhashv2_2( key, curBuf, 511, fixrand, fixrandex,
	                                         g_prand, g_prandex );

	/* fill2: curBuf[47] = byte 0, curBuf[48..63] = bytes 1..7,0,1..7,0 */
	const __m128i shuf2 = _mm_setr_epi8( 1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0 );
	const __m128i fill2 =
		_mm_shuffle_epi8( _mm_loadl_epi64( (u128*)&intermediate ), shuf2 );
	_mm_store_si128( (u128*)( &curBuf[32 + 16] ), fill2 );
	curBuf[32 + 15] = *( (unsigned char*)&intermediate );

	/* round constants from the MUTATED key, before FixKey */
	if ( full )
		haraka512_port_keyed( hash, curBuf, key + ( intermediate & 511 ) );
	else
		haraka512_keyed( hash, curBuf, key + ( intermediate & 511 ) );

	FixKey( fixrand, fixrandex, key, g_prand, g_prandex );
	return intermediate;
}

int verus_host_full( uint8_t out[32], const uint8_t *pre, int pre_len )
{
	VH_ALIGN(64) uint8_t keybuf[VERUS_KEYBUF_BYTES];
	VH_ALIGN(32) uint8_t half[64];
	const int half_len = pre_len - VERUS_NONCE_SPACE;

	if ( pre_len < VERUS_BASE_SIZE || ( pre_len % 32 ) != VERUS_NONCE_SPACE )
		return 0;
	if ( !verus_host_prologue( pre, half_len, half, keybuf ) ) return 0;
	verus_host_hash( out, half, pre + half_len, keybuf, 1 );
	return 1;
}

#else  /* no AES-NI build: the API exists and refuses */

int verus_host_cpu_ok( void ) { return 0; }
int verus_host_prologue( const uint8_t *pre, int half_len, uint8_t half[64],
                         void *keybuf )
{ (void)pre; (void)half_len; (void)half; (void)keybuf; return 0; }
uint64_t verus_host_hash( uint8_t hash[32], const uint8_t half[64],
                          const uint8_t nonce15[15], void *keybuf, int full )
{ (void)hash; (void)half; (void)nonce15; (void)keybuf; (void)full; return 0; }
int verus_host_full( uint8_t out[32], const uint8_t *pre, int pre_len )
{ (void)out; (void)pre; (void)pre_len; return 0; }

#endif /* VERUS_HAVE_SIMD */

void verus_host_clear_noncanonical( uint8_t *pre )
{
	memset( pre + 4, 0, 96 );                    /* prevhash, merkle, sapling */
	memset( pre + 104, 0, 4 );                   /* nBits  */
	memset( pre + 108, 0, 32 );                  /* nNonce */
	memset( pre + VERUS_BASE_SIZE + 8, 0, 64 );  /* hashPrev/BlockMMRRoot */
}

int verus_host_padded_solution_size( int received )
{
	if ( received <= VERUS_SOLUTION_FIXED ) return VERUS_SOLUTION_FIXED;
	return received + ( 47 - ( ( received + VERUS_BASE_SIZE ) % 32 ) );
}
