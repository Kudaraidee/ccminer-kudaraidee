// Odocrypt per-epoch NVRTC recompile. See odocrypt_jit.h for what and why.
//
// The generated kernel is a textual mirror of the device half of
// cuda_odocrypt.cu, with the rotation amounts substituted in as literals. That
// duplication is gated rather than trusted: odo_jit_gate() in cuda_odocrypt.cu
// compares full digests from both kernels against the CPU reference on every
// epoch change before either screens a share, and retires this path on any
// disagreement. A drift between this string and the .cu can therefore only cost
// the speedup.

#include <stdio.h>
#include <stdlib.h>   // getenv (MSVC gets it via <string>, libstdc++ does not)
#include <string.h>
#include <string>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#include "miner.h"
#include "odocrypt_jit.h"

// ---- generated device source ----------------------------------------------

// Fixed prefix. NVRTC has no <stdint.h>. Do not add a `typedef ... size_t`:
// it is a hard error under the statically linked NVRTC.
static const char *kPrefix = R"CUDA(
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;

#define ODO_ROUNDS               84
#define ODO_STATE_SIZE           10
#define ODO_PBOX_SUBROUNDS       6
#define ODO_ROTATION_COUNT       6
#define ODO_SMALL_SBOX_WIDTH     6
#define ODO_LARGE_SBOX_WIDTH     10
#define ODO_SMALL_SBOX_COUNT     40
#define ODO_LARGE_SBOX_COUNT     10

// cuda_helper.h's ROTL64 (USE_ROT_ASM_OPT == 1), inlined. With a literal `r`
// the branch folds and this becomes two SHF instructions, which is the entire
// point of this translation unit.
__device__ __forceinline__ uint64_t rotl64( uint64_t x, int r )
{
   const uint32_t lo = (uint32_t)x, hi = (uint32_t)( x >> 32 );
   uint32_t rx, ry;
   if ( r >= 32 ) { rx = __funnelshift_l( lo, hi, r ); ry = __funnelshift_l( hi, lo, r ); }
   else           { rx = __funnelshift_l( hi, lo, r ); ry = __funnelshift_l( lo, hi, r ); }
   return ( (uint64_t)ry << 32 ) | (uint64_t)rx;
}

// Per job: header words 0..18 and the target's high word, uploaded through
// cuModuleGetGlobal so they are constant-bank reads like the static kernel's.
// Reading them out of a global buffer instead is measurably worse.
__constant__ uint32_t c_job[20];

__shared__ uint8_t  s_Sbox1[ODO_SMALL_SBOX_COUNT][1 << ODO_SMALL_SBOX_WIDTH];
__shared__ uint16_t s_Sbox2[ODO_LARGE_SBOX_COUNT][1 << ODO_LARGE_SBOX_WIDTH];

__device__ __forceinline__ void stage_sboxes( const uint8_t *g1, const uint16_t *g2 )
{
   uint32_t *dst1 = (uint32_t*)s_Sbox1;
   uint32_t *dst2 = (uint32_t*)s_Sbox2;
   const uint32_t *src1 = (const uint32_t*)g1;
   const uint32_t *src2 = (const uint32_t*)g2;
   for ( uint32_t i = threadIdx.x; i < sizeof s_Sbox1 / 4; i += blockDim.x ) dst1[i] = src1[i];
   for ( uint32_t i = threadIdx.x; i < sizeof s_Sbox2 / 4; i += blockDim.x ) dst2[i] = src2[i];
   __syncthreads();
}
)CUDA";

// The stages, with the baked accessors called instead of __constant__ reads.
static const char *kBody = R"CUDA(
__device__ void apply_pbox( uint64_t state[ODO_STATE_SIZE], int p )
{
   #pragma unroll
   for ( int i = 0; i < ODO_PBOX_SUBROUNDS - 1; i++ )
   {
      #pragma unroll
      for ( int k = 0; k < ODO_STATE_SIZE / 2; k++ )
      {
         uint64_t swp = pmask_v( p, i, k ) & ( state[2*k] ^ state[2*k+1] );
         state[2*k]   ^= swp;
         state[2*k+1] ^= swp;
      }
      uint64_t next[ODO_STATE_SIZE];
      #pragma unroll
      for ( int x = 0; x < ODO_STATE_SIZE; x++ )
         next[( 3 * x ) % ODO_STATE_SIZE] = state[x];
      #pragma unroll
      for ( int x = 0; x < ODO_STATE_SIZE; x++ ) state[x] = next[x];
      #pragma unroll
      for ( int k = 0; k < ODO_STATE_SIZE / 2; k++ )
         state[2*k] = rotl64( state[2*k], prot_amt( p, i, k ) );
   }
   #pragma unroll
   for ( int k = 0; k < ODO_STATE_SIZE / 2; k++ )
   {
      uint64_t swp = pmask_v( p, ODO_PBOX_SUBROUNDS-1, k ) & ( state[2*k] ^ state[2*k+1] );
      state[2*k]   ^= swp;
      state[2*k+1] ^= swp;
   }
}

__device__ void apply_sboxes( uint64_t state[ODO_STATE_SIZE] )
{
   const uint64_t MASK1 = ( 1 << ODO_SMALL_SBOX_WIDTH ) - 1;
   const uint64_t MASK2 = ( 1 << ODO_LARGE_SBOX_WIDTH ) - 1;
   int smallIdx = 0;
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ )
   {
      uint64_t next = 0;
      int pos = 0, largeIdx = i;
      #pragma unroll
      for ( int j = 0; j < ODO_SMALL_SBOX_COUNT / ODO_STATE_SIZE; j++ )
      {
         next |= (uint64_t)s_Sbox1[smallIdx][( state[i] >> pos ) & MASK1] << pos;
         pos += ODO_SMALL_SBOX_WIDTH;
         next |= (uint64_t)s_Sbox2[largeIdx][( state[i] >> pos ) & MASK2] << pos;
         pos += ODO_LARGE_SBOX_WIDTH;
         smallIdx++;
      }
      state[i] = next;
   }
}

__device__ void apply_rotations( uint64_t state[ODO_STATE_SIZE] )
{
   uint64_t next[ODO_STATE_SIZE];
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ )
      next[i] = state[( i + 1 ) % ODO_STATE_SIZE];
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ )
      #pragma unroll
      for ( int j = 0; j < ODO_ROTATION_COUNT; j++ )
         next[i] ^= rotl64( state[i], rot_amt( j ) );
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ ) state[i] = next[i];
}

__constant__ uint32_t kc_rc[22] =
{
   0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
   0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
   0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
   0x0000800a, 0x8000000a, 0x80008081, 0x00008080
};
__constant__ int kc_rho[25] =
{
   0,  1, 30, 28, 27,  4, 12,  6, 23, 20,  3, 10, 11, 25,  7,
   9, 13, 15, 21,  8, 18,  2, 29, 24, 14
};
#define KIDX(x,y)  ( ( (x) % 5 ) + 5 * ( (y) % 5 ) )
#define ROTL32(x,n) __funnelshift_l( (x), (x), (n) )

__device__ void keccakp800_12( uint32_t A[25] )
{
   for ( int round = 22 - 12; round < 22; round++ )
   {
      uint32_t C[5], D[5], B[25];
      #pragma unroll
      for ( int x = 0; x < 5; x++ )
         C[x] = A[KIDX(x,0)] ^ A[KIDX(x,1)] ^ A[KIDX(x,2)] ^ A[KIDX(x,3)] ^ A[KIDX(x,4)];
      #pragma unroll
      for ( int x = 0; x < 5; x++ )
         D[x] = ROTL32( C[(x+1)%5], 1 ) ^ C[(x+4)%5];
      #pragma unroll
      for ( int x = 0; x < 5; x++ )
         #pragma unroll
         for ( int y = 0; y < 5; y++ )
            A[KIDX(x,y)] ^= D[x];
      // ROTL32 is a funnel shift and __funnelshift_l(x, x, 0) is x, so the
      // zero rho offset needs no guard.
      #pragma unroll
      for ( int i = 0; i < 25; i++ )
         A[i] = ROTL32( A[i], kc_rho[i] );
      #pragma unroll
      for ( int x = 0; x < 5; x++ )
         #pragma unroll
         for ( int y = 0; y < 5; y++ )
            B[KIDX( y, 2*x + 3*y )] = A[KIDX(x,y)];
      #pragma unroll
      for ( int y = 0; y < 5; y++ )
         #pragma unroll
         for ( int x = 0; x < 5; x++ )
            A[KIDX(x,y)] = B[KIDX(x,y)] ^ ( ( ~B[KIDX(x+1,y)] ) & B[KIDX(x+2,y)] );
      A[0] ^= kc_rc[round];
   }
}

__device__ __forceinline__ void hash_nonce( uint32_t nonce, uint32_t out[8] )
{
   const uint32_t *job = c_job;
   uint64_t state[ODO_STATE_SIZE];
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE - 1; i++ )
      state[i] = (uint64_t)job[2*i] | ( (uint64_t)job[2*i+1] << 32 );
   state[ODO_STATE_SIZE-1] = (uint64_t)job[18] | ( (uint64_t)nonce << 32 );

   uint64_t total = 0;
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ ) total ^= state[i];
   total ^= total >> 32;
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ ) state[i] ^= total;

   for ( int round = 0; round < ODO_ROUNDS; round++ )
   {
      apply_pbox( state, 0 );
      apply_sboxes( state );
      apply_pbox( state, 1 );
      apply_rotations( state );
      const uint32_t rk = c_rkey[round];
      #pragma unroll
      for ( int i = 0; i < ODO_STATE_SIZE; i++ ) state[i] ^= (uint64_t)( ( rk >> i ) & 1 );
   }

   uint32_t A[25];
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ )
   {
      A[2*i]   = (uint32_t)state[i];
      A[2*i+1] = (uint32_t)( state[i] >> 32 );
   }
   A[20] = 1u;
   A[21] = A[22] = A[23] = A[24] = 0u;

   keccakp800_12( A );

   #pragma unroll
   for ( int i = 0; i < 8; i++ ) out[i] = A[i];
}

extern "C" __global__ void odo_jit_hash( uint32_t threads, uint32_t startNonce,
                                         const uint8_t *sbox1, const uint16_t *sbox2,
                                         uint32_t *resNonce )
{
   stage_sboxes( sbox1, sbox2 );
   const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
   const uint32_t nonce = startNonce + thread;
   uint32_t h[8];
   hash_nonce( nonce, h );
   if ( thread < threads && h[7] <= c_job[19] )
      atomicMin( resNonce, nonce );
}

extern "C" __global__ void odo_jit_digest( uint32_t threads, uint32_t startNonce,
                                           const uint8_t *sbox1, const uint16_t *sbox2,
                                           uint32_t *out )
{
   stage_sboxes( sbox1, sbox2 );
   const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
   uint32_t h[8];
   hash_nonce( startNonce + thread, h );
   if ( thread >= threads ) return;
   #pragma unroll
   for ( int i = 0; i < 8; i++ ) out[8 * thread + i] = h[i];
}
)CUDA";

// A switch on a value the unroller has already turned into a literal folds to
// the literal. A `const` array would not: nvcc parks it in the constant bank
// and emits a load, which is exactly what this is meant to remove.
static void emit_prot( std::ostringstream &s, const OdoCrypt *c )
{
   s << "__device__ __forceinline__ int prot_amt( int p, int i, int k )\n{\n"
        "   switch ( p * 25 + i * 5 + k )\n   {\n";
   for ( int p = 0; p < 2; p++ )
      for ( int i = 0; i < ODO_PBOX_SUBROUNDS - 1; i++ )
         for ( int k = 0; k < ODO_STATE_SIZE / 2; k++ )
            s << "      case " << ( p * 25 + i * 5 + k ) << ": return "
              << c->Permutation[p].rotation[i][k] << ";\n";
   s << "   }\n   return 1;\n}\n";
}

// The masks go into an initialised __constant__ array, deliberately NOT into
// instruction immediates: LOP3 takes a constant-bank operand for free, so
// `c_pmask[..] & (a ^ b)` is one instruction, whereas a 64-bit literal must be
// materialised into registers first. Promote a value to a literal only where it
// becomes part of an instruction, as the rotation amounts do.
static void emit_pmask( std::ostringstream &s, const OdoCrypt *c )
{
   char buf[48];
   s << "__constant__ uint64_t c_pmask[2][ODO_PBOX_SUBROUNDS][ODO_STATE_SIZE / 2] = {\n";
   for ( int p = 0; p < 2; p++ )
   {
      s << "   {";
      for ( int i = 0; i < ODO_PBOX_SUBROUNDS; i++ )
      {
         s << ( i ? ",\n    {" : "\n    {" );
         for ( int k = 0; k < ODO_STATE_SIZE / 2; k++ )
         {
            snprintf( buf, sizeof buf, "0x%016llxULL",
                      (unsigned long long)c->Permutation[p].mask[i][k] );
            s << ( k ? "," : "" ) << buf;
         }
         s << "}";
      }
      s << ( p ? "\n   }\n" : "\n   },\n" );
   }
   s << "};\n"
        "__device__ __forceinline__ uint64_t pmask_v( int p, int i, int k )\n"
        "{\n   return c_pmask[p][i][k];\n}\n";
}

static void emit_rkey( std::ostringstream &s, const OdoCrypt *c )
{
   // Indexed by the round counter, so these cannot fold into instructions; they
   // are epoch data, so they are baked into the module instead of uploaded.
   s << "__constant__ uint32_t c_rkey[ODO_ROUNDS] = {";
   for ( int i = 0; i < ODO_ROUNDS; i++ )
      s << ( i ? "," : "" ) << ( i % 12 ? " " : "\n   " ) << (unsigned)c->RoundKey[i];
   s << "\n};\n";
}

static void emit_rot( std::ostringstream &s, const OdoCrypt *c )
{
   s << "__device__ __forceinline__ int rot_amt( int j )\n{\n   switch ( j )\n   {\n";
   for ( int j = 0; j < ODO_ROTATION_COUNT; j++ )
      s << "      case " << j << ": return " << c->Rotations[j] << ";\n";
   s << "   }\n   return 1;\n}\n";
}

static std::string odocrypt_jit_source( const OdoCrypt *c )
{
   std::ostringstream s;
   s << kPrefix;
   emit_prot( s, c );
   emit_pmask( s, c );
   emit_rot( s, c );
   emit_rkey( s, c );
   s << kBody;
   return s.str();
}

// ---- compile + cache -------------------------------------------------------

struct OdoJit
{
   CUmodule   mod      = NULL;
   CUfunction fn_hash  = NULL;
   CUfunction fn_dig   = NULL;
   uint32_t   key      = 0;
   bool       have     = false;
   bool       retired  = false;   // the gate said no; do not try again
   uint32_t   compiles = 0;
};

static OdoJit s_jit[MAX_GPUS];

extern "C" bool odocrypt_jit_ready( int thr_id )
{
   return s_jit[thr_id].have && !s_jit[thr_id].retired;
}

extern "C" int odocrypt_jit_regs( int thr_id )
{
   int n = 0;
   if ( !s_jit[thr_id].fn_hash ) return 0;
   cuFuncGetAttribute( &n, CU_FUNC_ATTRIBUTE_NUM_REGS, s_jit[thr_id].fn_hash );
   return n;
}

extern "C" uint32_t odocrypt_jit_compiles( int thr_id )
{
   return s_jit[thr_id].compiles;
}

extern "C" void odocrypt_jit_disable( int thr_id, const char *why )
{
   OdoJit &j = s_jit[thr_id];
   if ( j.mod ) { cuModuleUnload( j.mod ); j.mod = NULL; }
   j.fn_hash = j.fn_dig = NULL;
   j.have = false;
   j.retired = true;
   gpulog( LOG_WARNING, thr_id, "odocrypt: JIT kernel disabled (%s); using the static kernel", why );
}

extern "C" bool odocrypt_jit_prepare( int thr_id, const OdoCrypt *c, uint32_t key, int sm_arch )
{
   OdoJit &j = s_jit[thr_id];
   if ( j.retired ) return false;
   if ( j.have && j.key == key ) return true;

   const std::string src = odocrypt_jit_source( c );

   // An NVRTC kernel never appears in the build log, so ODO_JIT_DUMP=<path>
   // writes the generated source out for an offline compile when its register
   // usage or SASS needs inspecting.
   const char *dump = getenv( "ODO_JIT_DUMP" );
   if ( dump && *dump )
   {
      FILE *f = fopen( dump, "wb" );
      if ( f ) { fwrite( src.data(), 1, src.size(), f ); fclose( f ); }
   }

   nvrtcProgram prog;
   nvrtcResult nr = nvrtcCreateProgram( &prog, src.c_str(), "odocrypt_jit.cu", 0, NULL, NULL );
   if ( nr != NVRTC_SUCCESS )
   {
      gpulog( LOG_WARNING, thr_id, "odocrypt: nvrtcCreateProgram: %s", nvrtcGetErrorString( nr ) );
      return false;
   }

   // Ask for a real architecture and take the CUBIN rather than the PTX, so the
   // ptxas inside NVRTC assembles it instead of the driver. PTX stays as the
   // fallback for anything that cannot produce a CUBIN.
   char arch_opt[64];
   bool cubin = true;
   snprintf( arch_opt, sizeof arch_opt, "--gpu-architecture=sm_%d", sm_arch );
   const char *opts[] = { arch_opt, "--std=c++14", "--maxrregcount=128" };
   nr = nvrtcCompileProgram( prog, 3, opts );
   if ( nr != NVRTC_SUCCESS )
   {
      nvrtcDestroyProgram( &prog );
      if ( nvrtcCreateProgram( &prog, src.c_str(), "odocrypt_jit.cu", 0, NULL, NULL ) != NVRTC_SUCCESS )
         return false;
      snprintf( arch_opt, sizeof arch_opt, "--gpu-architecture=compute_%d", sm_arch );
      nr = nvrtcCompileProgram( prog, 3, opts );
      cubin = false;
   }
   if ( nr != NVRTC_SUCCESS )
   {
      size_t log_size = 0;
      nvrtcGetProgramLogSize( prog, &log_size );
      std::vector<char> log( log_size ? log_size : 1 );
      nvrtcGetProgramLog( prog, log.data() );
      gpulog( LOG_WARNING, thr_id, "odocrypt: JIT compile failed:\n%s", log.data() );
      nvrtcDestroyProgram( &prog );
      return false;
   }

   std::vector<char> image;
   size_t img_size = 0;
   if ( cubin && nvrtcGetCUBINSize( prog, &img_size ) == NVRTC_SUCCESS && img_size )
   {
      image.resize( img_size );
      if ( nvrtcGetCUBIN( prog, image.data() ) != NVRTC_SUCCESS ) { image.clear(); cubin = false; }
   }
   else cubin = false;

   if ( !cubin )
   {
      nvrtcGetPTXSize( prog, &img_size );
      image.resize( img_size ? img_size : 1 );
      nvrtcGetPTX( prog, image.data() );
   }
   nvrtcDestroyProgram( &prog );

   CUmodule mod = NULL;
   CUresult cr = cuModuleLoadData( &mod, image.data() );
   if ( cr != CUDA_SUCCESS )
   {
      const char *es = NULL; cuGetErrorString( cr, &es );
      gpulog( LOG_WARNING, thr_id, "odocrypt: cuModuleLoadData: %s", es ? es : "?" );
      return false;
   }

   CUfunction fh = NULL, fd = NULL;
   if ( cuModuleGetFunction( &fh, mod, "odo_jit_hash" ) != CUDA_SUCCESS ||
        cuModuleGetFunction( &fd, mod, "odo_jit_digest" ) != CUDA_SUCCESS )
   {
      gpulog( LOG_WARNING, thr_id, "odocrypt: cuModuleGetFunction failed" );
      cuModuleUnload( mod );
      return false;
   }

   if ( j.mod ) cuModuleUnload( j.mod );
   j.mod = mod;
   j.fn_hash = fh;
   j.fn_dig = fd;
   j.key = key;
   j.have = true;
   j.compiles++;
   return true;
}

extern "C" bool odocrypt_jit_set_job( int thr_id, const uint32_t *header19, uint32_t target_hi )
{
   OdoJit &j = s_jit[thr_id];
   if ( !j.have ) return false;

   CUdeviceptr dp = 0; size_t sz = 0;
   if ( cuModuleGetGlobal( &dp, &sz, j.mod, "c_job" ) != CUDA_SUCCESS || sz < 20 * sizeof(uint32_t) )
   {
      odocrypt_jit_disable( thr_id, "c_job not found in the JIT module" );
      return false;
   }
   uint32_t job[20];
   memcpy( job, header19, 19 * sizeof(uint32_t) );
   job[19] = target_hi;
   if ( cuMemcpyHtoD( dp, job, sizeof job ) != CUDA_SUCCESS )
   {
      odocrypt_jit_disable( thr_id, "c_job upload failed" );
      return false;
   }
   return true;
}

static bool jit_launch( int thr_id, CUfunction fn, uint32_t threads, uint32_t startNonce,
                        const void *sbox1, const void *sbox2, void *out, uint32_t tpb )
{
   if ( !fn ) return false;
   CUdeviceptr ds1 = (CUdeviceptr)sbox1, ds2 = (CUdeviceptr)sbox2, dout = (CUdeviceptr)out;
   void *args[] = { &threads, &startNonce, &ds1, &ds2, &dout };
   const uint32_t grid = ( threads + tpb - 1 ) / tpb;
   const CUresult cr = cuLaunchKernel( fn, grid, 1, 1, tpb, 1, 1, 0, NULL, args, NULL );
   if ( cr != CUDA_SUCCESS )
   {
      const char *es = NULL; cuGetErrorString( cr, &es );
      gpulog( LOG_WARNING, thr_id, "odocrypt: cuLaunchKernel: %s", es ? es : "?" );
      return false;
   }
   return true;
}

extern "C" bool odocrypt_jit_launch_hash( int thr_id, uint32_t threads, uint32_t startNonce,
                                          const void *sbox1, const void *sbox2,
                                          void *resNonce, uint32_t tpb )
{
   return jit_launch( thr_id, s_jit[thr_id].fn_hash, threads, startNonce, sbox1, sbox2, resNonce, tpb );
}

extern "C" bool odocrypt_jit_launch_digest( int thr_id, uint32_t threads, uint32_t startNonce,
                                            const void *sbox1, const void *sbox2,
                                            void *out, uint32_t tpb )
{
   return jit_launch( thr_id, s_jit[thr_id].fn_dig, threads, startNonce, sbox1, sbox2, out, tpb );
}
