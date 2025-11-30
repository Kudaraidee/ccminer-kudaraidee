#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

// Argon2 constants
#define ARGON2_BLOCK_SIZE 1024
#define ARGON2_QWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 8)
#define ARGON2_OWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 16)
#define ARGON2_HWORDS_IN_BLOCK (ARGON2_BLOCK_SIZE / 32)
#define ARGON2_SYNC_POINTS 4
#define ARGON2_PREHASH_DIGEST_LENGTH 64
#define ARGON2_PREHASH_SEED_LENGTH 72

// Blake2b constants
#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES 64
#define BLAKE2B_KEYBYTES 64
#define BLAKE2B_SALTBYTES 16
#define BLAKE2B_PERSONALBYTES 16
#define BLAKE2B_ROUNDS 12

// Argon2 block structure
typedef struct block_ {
    uint64_t v[ARGON2_QWORDS_IN_BLOCK];
} block;

// Blake2b state structure
typedef struct blake2b_state_ {
    uint64_t h[8];
    uint64_t t[2];
    uint64_t f[2];
    uint8_t buf[BLAKE2B_BLOCKBYTES];
    size_t buflen;
} blake2b_state;

// Blake2b IV
__constant__ uint64_t blake2b_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// Blake2b sigma table
__constant__ uint8_t blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

// RinHash salt in constant memory for better performance
__constant__ char c_rinhash_salt[] = "RinCoinSalt";
__constant__ uint32_t c_rinhash_t_cost = 2;
__constant__ uint32_t c_rinhash_m_cost = 64;
__constant__ uint32_t c_rinhash_lanes = 1;

// CUDA block indexing function
__device__ uint32_t index_alpha(const uint32_t pass, const uint32_t slice, 
                               const uint32_t index, const uint32_t pseudo_rand, 
                               const uint32_t lanes, const uint32_t lane_length, 
                               const uint32_t segmentLength) {
    uint32_t reference_area_size;
    
    if (pass == 0) {
        // First pass
        if (slice == 0) {
            // First slice
            reference_area_size = index - 1; // all but the previous
        } else {
            // Other slices
            reference_area_size = slice * segmentLength + index - 1;
        }
    } else {
        // Other passes
        if (slice == 0) {
            // First slice
            reference_area_size = lane_length - segmentLength - 1;
        } else {
            // Other slices
            reference_area_size = lane_length - 1;
        }
    }

    // Start from the pseudo-random value
    uint64_t relative_position = pseudo_rand;
    // Add 1 to prevent reference to the current block
    relative_position = (relative_position * relative_position) >> 32;
    relative_position = reference_area_size - 1 - (reference_area_size * relative_position >> 32);

    uint32_t start_position;
    if (pass != 0) {
        start_position = 0;
    } else {
        if (slice == 0) {
            start_position = 0;
        } else {
            start_position = slice * segmentLength;
        }
    }

    return (start_position + relative_position) % lane_length;
}

// Rotate right function for 64-bit values
__host__ __device__ __forceinline__ uint64_t rotr64(uint64_t x, uint32_t n) {
    return (x >> n) | (x << (64 - n));
}

// Blake2b mixing function G
__host__ __device__ __forceinline__ void blake2b_G(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t x, uint64_t y) {
    a = a + b + x;
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + y;
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

__device__ void G1(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t x, uint64_t y) {
    a = a + b + x;
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + y;
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

// Blake2b compression function F
__device__ void blake2b_compress(blake2b_state* S, const uint8_t block[BLAKE2B_BLOCKBYTES]) {
    uint64_t m[16];
    uint64_t v[16];

    // Load message block into m[16]
    for (int i = 0; i < 16; i++) {
        const uint8_t* p = block + i * 8;
        m[i] = ((uint64_t)p[0])
             | ((uint64_t)p[1] << 8)
             | ((uint64_t)p[2] << 16)
             | ((uint64_t)p[3] << 24)
             | ((uint64_t)p[4] << 32)
             | ((uint64_t)p[5] << 40)
             | ((uint64_t)p[6] << 48)
             | ((uint64_t)p[7] << 56);
    }

    // Initialize v[0..15]
    for (int i = 0; i < 8; i++) {
        v[i] = S->h[i];
        v[i + 8] = blake2b_IV[i];
    }

    v[12] ^= S->t[0];
    v[13] ^= S->t[1];
    v[14] ^= S->f[0];
    v[15] ^= S->f[1];

    for (int r = 0; r < BLAKE2B_ROUNDS; r++) {
        const uint8_t* s = blake2b_sigma[r];

        // Column step
        G1(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
        G1(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
        G1(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
        G1(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);

        // Diagonal step
        G1(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
        G1(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        G1(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
        G1(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
    }

    // Finalization
    for (int i = 0; i < 8; i++) {
        S->h[i] ^= v[i] ^ v[i + 8];
    }
}

// Block XOR function
__device__ void xor_block(block* dst, const block* src) {
    for (uint32_t i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        dst->v[i] ^= src->v[i];
    }
}

// Block copy function
__device__ void copy_block(block* dst, const block* src) {
    for (uint32_t i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        dst->v[i] = src->v[i];
    }
}

// P-function (permutation) for Argon2
__device__ void P(uint64_t* a, uint64_t* b, uint64_t* c, uint64_t* d) {
    *a = rotr64(*a, 32);
    *c = rotr64(*c, 32);
    
    uint64_t temp = *a;
    *a = *b;
    *b = temp;
    
    temp = *c;
    *c = *d;
    *d = temp;
}

// Block mixing function G for Argon2
__device__ void G_argon2(uint64_t* a, uint64_t* b, uint64_t* c, uint64_t* d) {
    *a = *a + *b + 2 * (*a) * (*b);
    *d = rotr64(*d ^ *a, 32);
    
    *c = *c + *d + 2 * (*c) * (*d);
    *b = rotr64(*b ^ *c, 24);
    
    *a = *a + *b + 2 * (*a) * (*b);
    *d = rotr64(*d ^ *a, 16);
    
    *c = *c + *d + 2 * (*c) * (*d);
    *b = rotr64(*b ^ *c, 63);
}

// Argon2 block mixing operation
__device__ void mix_block(block* dst, const block* src) {
    block temp_block;
    copy_block(&temp_block, dst);
    xor_block(&temp_block, src);
    
    uint64_t a[4], b[4], c[4], d[4];
    
    // Apply the mixing function to each 16-byte subset of the block
    for (int i = 0; i < 8; i++) {
        uint32_t i16 = i * 16;
        
        for (int j = 0; j < 4; j++) {
            a[j] = temp_block.v[i16 + j];
            b[j] = temp_block.v[i16 + j + 4];
            c[j] = temp_block.v[i16 + j + 8];
            d[j] = temp_block.v[i16 + j + 12];
        }
        
        G_argon2(&a[0], &b[0], &c[0], &d[0]);
        G_argon2(&a[1], &b[1], &c[1], &d[1]);
        G_argon2(&a[2], &b[2], &c[2], &d[2]);
        G_argon2(&a[3], &b[3], &c[3], &d[3]);
        
        G_argon2(&a[0], &b[1], &c[2], &d[3]);
        G_argon2(&a[1], &b[2], &c[3], &d[0]);
        G_argon2(&a[2], &b[3], &c[0], &d[1]);
        G_argon2(&a[3], &b[0], &c[1], &d[2]);
        
        for (int j = 0; j < 4; j++) {
            dst->v[i16 + j] = a[j];
            dst->v[i16 + j + 4] = b[j];
            dst->v[i16 + j + 8] = c[j];
            dst->v[i16 + j + 12] = d[j];
        }
    }
}

// CUDA kernel for Argon2d filling first blocks
__global__ void argon2d_init_blocks_kernel(
    block* memory,
    uint32_t lanes,
    uint32_t segmentLength,
    uint32_t lane_length,
    uint64_t* initial_hash
) {
    uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lane >= lanes) {
        return;
    }
    
    // First block for this lane
    block* first_block = &memory[lane * lane_length];
    
    // Fill the first block with the initial hash
    for (int i = 0; i < 8; i++) {
        first_block->v[i] = initial_hash[i];
    }
    
    // Zero out the rest of the first block
    for (int i = 8; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        first_block->v[i] = 0;
    }
    
    // Generate second block from the first
    block* second_block = &memory[lane * lane_length + 1];
    
    // Fill the second block as a simple transform of the first
    for (int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        second_block->v[i] = 0xdeadbeefdeadbeefULL;
    }
    
    // Mix first block with second block
    mix_block(second_block, first_block);
}

// CUDA kernel for Argon2d memory filling
__global__ void argon2d_fill_memory_kernel(
    block* memory,
    uint32_t passes,
    uint32_t lanes,
    uint32_t segmentLength,
    uint32_t lane_length
) {
    uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lane >= lanes) {
        return;
    }
    
    // Iterate through passes
    for (uint32_t pass = 0; pass < passes; pass++) {
        // Iterate through segments
        for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; slice++) {
            // Iterate through the blocks in this segment
            for (uint32_t index = 0; index < segmentLength; index++) {
                // Skip first two blocks of the first slice of the first pass
                if (pass == 0 && slice == 0 && index < 2) {
                    continue;
                }
                
                // Current block position
                uint32_t curr_offset = lane * lane_length + slice * segmentLength + index;
                
                // Calculate reference block position
                uint32_t prev_offset = lane * lane_length + 
                                      ((slice * segmentLength + index - 1) % lane_length);
                
                // Get the previous block
                block* prev_block = &memory[prev_offset];
                
                // Calculate pseudo-random value from previous block
                uint32_t pseudo_rand = prev_block->v[0];
                
                // Calculate reference index
                uint32_t ref_lane = pseudo_rand % lanes;
                uint32_t ref_index = index_alpha(pass, slice, index, pseudo_rand >> 32, 
                                              lanes, lane_length, segmentLength);
                
                // Calculate reference position
                uint32_t ref_offset = ref_lane * lane_length + ref_index;
                
                // Get the reference block
                block* ref_block = &memory[ref_offset];
                
                // Get current block
                block* curr_block = &memory[curr_offset];
                
                // Mix reference block with previous block and store in current block
                copy_block(curr_block, prev_block);
                xor_block(curr_block, ref_block);
                mix_block(curr_block, ref_block);
            }
            
            // Synchronize after each segment
            __syncthreads();
        }
    }
}

// Host implementation of Blake2b initialization
void blake2b_init(blake2b_state* S, size_t outlen) {
    // Initialize state to default values
    memset(S, 0, sizeof(blake2b_state));
    
    // Set hash output length in bytes
    uint32_t param = (uint32_t)outlen;
    
    // Set up IV with modified parameters
    for (int i = 0; i < 8; i++) {
        S->h[i] = blake2b_IV[i];
    }
    
    // Mix output length with first h value
    S->h[0] ^= param | (0 << 8) | (1 << 16) | (0 << 24);
    
    S->buflen = 0;
    S->t[0] = 0;
    S->t[1] = 0;
    S->f[0] = 0;
    S->f[1] = 0;
}

// Host implementation of Blake2b update
void blake2b_update(blake2b_state* S, const uint8_t* in, size_t inlen) {
    size_t left = S->buflen;
    size_t fill = BLAKE2B_BLOCKBYTES - left;
    
    // Handle existing data in buffer
    if (left > 0 && inlen >= fill) {
        memcpy(S->buf + left, in, fill);
        S->buflen += fill;
        S->t[0] += BLAKE2B_BLOCKBYTES;
        S->t[1] += (S->t[0] < BLAKE2B_BLOCKBYTES);
        
        // Process buffer
        blake2b_compress(S, S->buf);
        S->buflen = 0;
        
        in += fill;
        inlen -= fill;
        left = 0;
    }
    
    // Process full blocks directly from input
    while (inlen >= BLAKE2B_BLOCKBYTES) {
        S->t[0] += BLAKE2B_BLOCKBYTES;
        S->t[1] += (S->t[0] < BLAKE2B_BLOCKBYTES);
        
        blake2b_compress(S, in);
        in += BLAKE2B_BLOCKBYTES;
        inlen -= BLAKE2B_BLOCKBYTES;
    }
    
    // Store remaining input in buffer
    if (inlen > 0) {
        memcpy(S->buf + left, in, inlen);
        S->buflen = left + inlen;
    }
}

// Host implementation of Blake2b finalization
void blake2b_final(blake2b_state* S, uint8_t* out, size_t outlen) {
    // Mark last block
    S->f[0] = ~0ULL;
    
    // Pad buffer with zeros
    memset(S->buf + S->buflen, 0, BLAKE2B_BLOCKBYTES - S->buflen);
    
    // Update counter for last block
    S->t[0] += S->buflen;
    S->t[1] += (S->t[0] < S->buflen);
    
    // Compress last block
    blake2b_compress(S, S->buf);
    
    // Copy output
    memcpy(out, S->h, outlen);
}

// One-shot Blake2b hash function
void blake2b(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen) {
    blake2b_state S;
    blake2b_init(&S, outlen);
    blake2b_update(&S, in, inlen);
    blake2b_final(&S, out, outlen);
}

// Main Argon2d hash function
extern "C" void argon2d_hash_cuda(
    uint8_t* output,
    const uint8_t* input, size_t input_len,
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes
) {
    // Calculate derived values
    uint32_t memory_blocks = m_cost;
    uint32_t lane_length = memory_blocks / lanes;
    uint32_t segmentLength = lane_length / ARGON2_SYNC_POINTS;
    
    // Ensure minimum values
    if (lanes < 1 || t_cost < 1 || memory_blocks < 8 * lanes) {
        printf("Invalid Argon2d parameters\n");
        return;
    }
    
    // Initial hashing
    uint8_t initial_hash[ARGON2_PREHASH_DIGEST_LENGTH];
    blake2b_state BlakeHash;
    blake2b_init(&BlakeHash, ARGON2_PREHASH_DIGEST_LENGTH);
    
    // Hash parameters
    uint32_t value;
    value = lanes; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    value = 32; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value)); // output length
    value = m_cost; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    value = t_cost; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    value = 0x13; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value)); // version
    value = 0; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value)); // type = 0 for Argon2d
    value = input_len; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    
    // Hash input
    blake2b_update(&BlakeHash, input, input_len);
    
    // Finalize initial hash
    blake2b_final(&BlakeHash, initial_hash, ARGON2_PREHASH_DIGEST_LENGTH);
    
    // Allocate memory for blocks on GPU
    block* d_memory;
    size_t memory_size = memory_blocks * sizeof(block);
    cudaMalloc(&d_memory, memory_size);
    
    // Convert initial hash to 64-bit words for kernel
    uint64_t h_initial_hash[8];
    for (int i = 0; i < 8; i++) {
        h_initial_hash[i] = 
            ((uint64_t)initial_hash[i * 8 + 0]) |
            ((uint64_t)initial_hash[i * 8 + 1] << 8) |
            ((uint64_t)initial_hash[i * 8 + 2] << 16) |
            ((uint64_t)initial_hash[i * 8 + 3] << 24) |
            ((uint64_t)initial_hash[i * 8 + 4] << 32) |
            ((uint64_t)initial_hash[i * 8 + 5] << 40) |
            ((uint64_t)initial_hash[i * 8 + 6] << 48) |
            ((uint64_t)initial_hash[i * 8 + 7] << 56);
    }
    
    // Copy initial hash to device
    uint64_t* d_initial_hash;
    cudaMalloc(&d_initial_hash, 8 * sizeof(uint64_t));
    cudaMemcpy(d_initial_hash, h_initial_hash, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Initialize first blocks in each lane
    int threads_per_block = 256;
    int blocks = (lanes + threads_per_block - 1) / threads_per_block;
    
    argon2d_init_blocks_kernel<<<blocks, threads_per_block>>>(
        d_memory, lanes, segmentLength, lane_length, d_initial_hash
    );
    
    // Fill memory
    argon2d_fill_memory_kernel<<<blocks, threads_per_block>>>(
        d_memory, t_cost, lanes, segmentLength, lane_length
    );
    
    // Allocate memory for the final block
    block h_final_block;
    
    // Copy the final block back to host (XOR of the last block in each lane)
    block* h_memory = (block*)malloc(memory_size);
    cudaMemcpy(h_memory, d_memory, memory_size, cudaMemcpyDeviceToHost);
    
    // Initialize final block to the last block of the first lane
    memcpy(&h_final_block, &h_memory[(0 * lane_length) + (lane_length - 1)], sizeof(block));
    
    // XOR with last block of each other lane
    for (uint32_t lane = 1; lane < lanes; lane++) {
        block* last_block = &h_memory[(lane * lane_length) + (lane_length - 1)];
        for (uint32_t i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
            h_final_block.v[i] ^= last_block->v[i];
        }
    }
    
    // Hash the final block to produce the output
    blake2b(output, 32, (uint8_t*)&h_final_block, sizeof(block));
    
    // Clean up
    cudaFree(d_memory);
    cudaFree(d_initial_hash);
    free(h_memory);
}

// Simple wrapper function for standard parameters
extern "C" void argon2d_hash(uint8_t* output, const uint8_t* input, size_t input_len) {
    // Use typical cryptocurrency mining parameters: t=1, m=4096, lanes=1
    argon2d_hash_cuda(output, input, input_len, 1, 4096, 1);
}

// Function optimized for RinHash parameters
extern "C" void argon2d_hash_rinhash(uint8_t* output, const uint8_t* input, size_t input_len) {
    uint32_t t_cost = 2;
    uint32_t m_cost = 64;
    uint32_t lanes  = 1;
    const char* salt = "RinCoinSalt";

    // __constant__ メモリにコピー
    cudaMemcpyToSymbol(c_rinhash_t_cost, &t_cost, sizeof(uint32_t));
    cudaMemcpyToSymbol(c_rinhash_m_cost, &m_cost, sizeof(uint32_t));
    cudaMemcpyToSymbol(c_rinhash_lanes,  &lanes, sizeof(uint32_t));
    cudaMemcpyToSymbol(c_rinhash_salt,   salt, strlen(salt) + 1);
    // Prepare initial hash with input and salt combined
    uint8_t initial_hash[ARGON2_PREHASH_DIGEST_LENGTH];
    blake2b_state BlakeHash;
    blake2b_init(&BlakeHash, ARGON2_PREHASH_DIGEST_LENGTH);
    
    // Hash parameters first (constant for RinHash)
    uint32_t value;
    value = c_rinhash_lanes; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    value = 32; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value)); // output length
    value = c_rinhash_m_cost; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    value = c_rinhash_t_cost; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    value = 0x13; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value)); // version
    value = 0; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value)); // type = 0 for Argon2d
    
    // Hash input data
    value = input_len; blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    blake2b_update(&BlakeHash, input, input_len);
    
    // Hash salt from constant memory
    value = strlen(c_rinhash_salt); blake2b_update(&BlakeHash, (uint8_t*)&value, sizeof(value));
    blake2b_update(&BlakeHash, (const uint8_t*)c_rinhash_salt, strlen(c_rinhash_salt));
    
    // Finalize initial hash
    blake2b_final(&BlakeHash, initial_hash, ARGON2_PREHASH_DIGEST_LENGTH);
    
    // Allocate memory for blocks on GPU
    block* d_memory;
    size_t memory_size = c_rinhash_m_cost * sizeof(block);
    cudaMalloc(&d_memory, memory_size);
    
    // Convert initial hash to 64-bit words for kernel
    uint64_t h_initial_hash[8];
    for (int i = 0; i < 8; i++) {
        h_initial_hash[i] = 
            ((uint64_t)initial_hash[i * 8 + 0]) |
            ((uint64_t)initial_hash[i * 8 + 1] << 8) |
            ((uint64_t)initial_hash[i * 8 + 2] << 16) |
            ((uint64_t)initial_hash[i * 8 + 3] << 24) |
            ((uint64_t)initial_hash[i * 8 + 4] << 32) |
            ((uint64_t)initial_hash[i * 8 + 5] << 40) |
            ((uint64_t)initial_hash[i * 8 + 6] << 48) |
            ((uint64_t)initial_hash[i * 8 + 7] << 56);
    }
    
    // Copy initial hash to device
    uint64_t* d_initial_hash;
    cudaMalloc(&d_initial_hash, 8 * sizeof(uint64_t));
    cudaMemcpy(d_initial_hash, h_initial_hash, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Calculate derived values
    uint32_t lane_length = m_cost / lanes;
    uint32_t segmentLength = lane_length / ARGON2_SYNC_POINTS;
    
    // Initialize first blocks in each lane
    int threads_per_block = 256;
    int blocks = (lanes + threads_per_block - 1) / threads_per_block;
    
    argon2d_init_blocks_kernel<<<blocks, threads_per_block>>>(d_memory, lanes, segmentLength, lane_length, d_initial_hash);
    argon2d_fill_memory_kernel<<<blocks, threads_per_block>>>(d_memory, t_cost, lanes, segmentLength, lane_length);
    
    // Allocate memory for the final block
    block h_final_block;
    
    // Copy the final block back to host
    block* h_memory = (block*)malloc(memory_size);
    cudaMemcpy(h_memory, d_memory, memory_size, cudaMemcpyDeviceToHost);
    
    // Initialize final block to the last block of the first lane
    memcpy(&h_final_block, &h_memory[(0 * lane_length) + (lane_length - 1)], sizeof(block));
    
    // XOR with last block of each other lane (for RinHash this loop won't be executed as lanes=1)
    for (uint32_t lane = 1; lane < c_rinhash_lanes; lane++) {
        block* last_block = &h_memory[(lane * lane_length) + (lane_length - 1)];
        for (uint32_t i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
            h_final_block.v[i] ^= last_block->v[i];
        }
    }
    
    // Hash the final block to produce the output
    blake2b(output, 32, (uint8_t*)&h_final_block, sizeof(block));
    
    // Clean up
    cudaFree(d_memory);
    cudaFree(d_initial_hash);
    free(h_memory);
}

