/*
 * Kernel "Wave Scheduling" (1 luồng / 128 nonces)
 *
 * 🚀 PHIÊN BẢN TỐI ƯU V3 (GPU-REDUCE):
 * - Thêm 2 kernel "reduction" (find_best_hash_part1/2)
 * để tìm hash tốt nhất hoàn toàn trên GPU.
 * - Loại bỏ vòng lặp for-loop trên CPU.
 * - Loại bỏ thao tác download d_outputs (hàng MB) về CPU.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdexcept>

// Include shared device functions
#include "rinhash_device.cuh"
#include "argon2d_device.cuh"
#include "sha3-256.cu"
#include "blake3_device.cuh"

// 🚀 THAM SỐ CÓ THỂ TINH CHỈNH (Xem Giai đoạn 1 trong hướng dẫn)
#define OPTIMAL_THREADS_PER_BLOCK 256 
#define NUM_WAVES 128 

#define MAX_BATCH_BLOCKS_PER_GPU (4 * 1024 * 1024) 
#define NUM_STREAMS 4

// 🚀 Định nghĩa cho kernel reduction
#define REDUCE_THREADS_PER_BLOCK 256
#define REDUCE_MAX_BLOCKS 1024 // Hỗ trợ lên đến 256*1024 = 262,144 luồng

// 🚀 Cấu trúc kết quả (dùng cho kernel reduction)
struct BestResult {
    uint8_t hash[32];
    uint32_t nonce;
};

// 🚀 Persistent Memory (Quản lý VRAM)
struct PersistentMemory {
    uint8_t *d_headers;
    uint8_t *d_outputs;
    block *d_memories;
    uint32_t *d_target;
    uint32_t *d_solution_found;
    uint32_t *d_solution_nonce;
    uint8_t *h_headers_pinned;
    uint32_t *h_target_pinned;
    uint32_t *h_solution_found_pinned;
    uint32_t *h_solution_nonce_pinned;
    cudaStream_t streams[NUM_STREAMS];
    uint32_t allocated_blocks; 
    bool initialized;
    
    // 🚀 Vùng nhớ cho "Best Hash" Reduction
    BestResult* d_partial_results; // Kết quả trung gian
    BestResult* d_final_result;    // Kết quả cuối cùng
    BestResult* h_final_result_pinned; // Pinned memory để download
};

// Thread-local persistent memory
thread_local PersistentMemory g_persistent_mem = {0};
thread_local uint32_t g_m_cost = 64; // 64KB cho Argon2d

// Helper: kiểm tra lỗi CUDA
inline void check_cuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        throw std::runtime_error("CUDA error");
    }
}

// 🚀 KERNEL BĂM CHÍNH (Wave Scheduling)
extern "C" __global__ void rinhash_cuda_kernel_optimized(
    const uint8_t* base_header, 
    size_t header_len,
    uint8_t* outputs,
    uint32_t num_threads,       // Tổng số luồng
    block* memories,            // Vùng nhớ cho tất cả luồng
    uint32_t m_cost,
    uint32_t* target,
    uint32_t* solution_found,
    uint32_t* solution_nonce,
    uint32_t start_nonce
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    block* my_memory = memories + tid * m_cost;
    uint32_t my_header[20]; 
    memcpy(my_header, base_header, 80);

    for (int wave = 0; wave < NUM_WAVES; wave++) {
        if (atomicAdd(solution_found, 0) > 0) return;

        uint32_t current_nonce = start_nonce + (tid * NUM_WAVES) + wave;
        my_header[19] = current_nonce; 

        uint8_t blake3_out[32];
        light_hash_device((const uint8_t*)my_header, header_len, blake3_out);

        uint8_t salt[11] = { 'R','i','n','C','o','i','n','S','a','l','t' };
        uint8_t argon2_out[32];
        device_argon2d_hash(argon2_out, blake3_out, 32, 2, m_cost, 1, my_memory, salt, sizeof(salt));

        uint8_t* output_hash = outputs + ((tid * NUM_WAVES) + wave) * 32;
        sha3_256_device(argon2_out, 32, output_hash);

        // Kiểm tra target
        uint32_t* hash_words = (uint32_t*)output_hash;
        bool meets_target = true;
        for (int i = 7; i >= 0; i--) {
            if (hash_words[i] > target[i]) {
                meets_target = false;
                break;
            } else if (hash_words[i] < target[i]) {
                break;
            }
        }

        if (meets_target) {
            if (atomicCAS(solution_found, 0, 1) == 0) {
                *solution_nonce = current_nonce;
            }
        }
    }
}

// 🚀 ==============================================================
// 🚀 KERNEL TÌM HASH TỐT NHẤT (REDUCTION)
// 🚀 ==============================================================

// Hàm device helper để so sánh 2 hash 32-byte
__device__ __forceinline__ bool is_better_hash(uint32_t* new_hash, uint32_t* old_hash) {
    for (int k = 7; k >= 0; k--) {
        if (new_hash[k] < old_hash[k]) return true;
        if (new_hash[k] > old_hash[k]) return false;
    }
    return false; // Chúng bằng nhau
}

// Kernel Giai đoạn 1: Mỗi block tìm hash tốt nhất của block mình
extern "C" __global__ void find_best_hash_part1(
    const uint8_t* d_outputs,
    uint32_t num_nonces,
    uint32_t start_nonce,
    BestResult* d_partial_results
) {
    __shared__ BestResult s_best_result;
    
    uint32_t tid = threadIdx.x;
    uint32_t global_idx = blockIdx.x * blockDim.x + tid;

    // Khởi tạo shared memory: luồng đầu tiên sao chép hash đầu tiên
    if (tid == 0) {
        if (global_idx < num_nonces) {
            memcpy(s_best_result.hash, d_outputs + global_idx * 32, 32);
            s_best_result.nonce = start_nonce + global_idx;
        } else {
            // Block này không có việc gì làm, set hash là FFFF...
            memset(s_best_result.hash, 0xFF, 32);
            s_best_result.nonce = 0;
        }
    }
    __syncthreads();

    // Mỗi luồng kiểm tra một phần (stride)
    for (uint32_t i = global_idx; i < num_nonces; i += blockDim.x * gridDim.x) {
        uint32_t* current_hash = (uint32_t*)(d_outputs + i * 32);
        
        if (is_better_hash(current_hash, (uint32_t*)s_best_result.hash)) {
            // Đây là một "critical section" đơn giản
            // Chúng ta dùng atomicCAS trên 32-bit (word) đầu tiên của hash
            // (Đây là một cách đơn giản hóa, đủ tốt cho mining)
            unsigned int* s_hash_word = (unsigned int*)s_best_result.hash;
            unsigned int* c_hash_word = (unsigned int*)current_hash;

            // Thực hiện 1 spin-lock đơn giản
            unsigned int old_val = *s_hash_word;
            while(is_better_hash(current_hash, (uint32_t*)s_best_result.hash)) {
                old_val = atomicCAS(s_hash_word, old_val, *c_hash_word);
                if (old_val == *c_hash_word) break; // Thắng lock
            }

            // Nếu chúng ta thắng lock, cập nhật phần còn lại
            if (*s_hash_word == *c_hash_word) {
                 memcpy(s_best_result.hash, current_hash, 32);
                 s_best_result.nonce = start_nonce + i;
            }
        }
    }
    __syncthreads();

    // Luồng đầu tiên ghi kết quả của block này ra global memory
    if (tid == 0) {
        memcpy(&d_partial_results[blockIdx.x], &s_best_result, sizeof(BestResult));
    }
}

// Kernel Giai đoạn 2: Chỉ 1 block chạy để tìm hash tốt nhất từ kết quả trung gian
extern "C" __global__ void find_best_hash_part2(
    const BestResult* d_partial_results,
    uint32_t num_partial_results, // = gridDim.x của Giai đoạn 1
    BestResult* d_final_result
) {
    __shared__ BestResult s_best_result;
    
    uint32_t tid = threadIdx.x;

    // Khởi tạo shared memory
    if (tid == 0) {
        memcpy(&s_best_result, &d_partial_results[0], sizeof(BestResult));
    }
    __syncthreads();

    // Mỗi luồng kiểm tra 1 phần
    for (uint32_t i = tid; i < num_partial_results; i += blockDim.x) {
        uint32_t* current_hash = (uint32_t*)d_partial_results[i].hash;
        
        if (is_better_hash(current_hash, (uint32_t*)s_best_result.hash)) {
            // Tương tự, dùng spin-lock
            unsigned int* s_hash_word = (unsigned int*)s_best_result.hash;
            unsigned int* c_hash_word = (unsigned int*)current_hash;
            unsigned int old_val = *s_hash_word;
            while(is_better_hash(current_hash, (uint32_t*)s_best_result.hash)) {
                old_val = atomicCAS(s_hash_word, old_val, *c_hash_word);
                if (old_val == *c_hash_word) break; 
            }
            
            if (*s_hash_word == *c_hash_word) {
                 memcpy(s_best_result.hash, current_hash, 32);
                 s_best_result.nonce = d_partial_results[i].nonce;
            }
        }
    }
    __syncthreads();

    // Luồng đầu tiên ghi kết quả cuối cùng
    if (tid == 0) {
        memcpy(d_final_result, &s_best_result, sizeof(BestResult));
    }
}

// Forward declaration
extern "C" void rinhash_persistent_cleanup();

// 🚀 Quản lý Persistent Memory (Đã cập nhật)
extern "C" void rinhash_persistent_init(uint32_t max_threads) {
    if (g_persistent_mem.initialized && g_persistent_mem.allocated_blocks >= max_threads) {
        return;
    }

    if (g_persistent_mem.initialized) {
        rinhash_persistent_cleanup();
    }

    size_t memories_size = max_threads * g_m_cost * sizeof(block); 
    size_t headers_size = 80; 
    size_t target_size = 8 * sizeof(uint32_t);
    size_t solution_size = sizeof(uint32_t);
    size_t outputs_size = max_threads * NUM_WAVES * 32;
    
    // VRAM cho Reduction
    size_t partial_results_size = REDUCE_MAX_BLOCKS * sizeof(BestResult);
    size_t final_result_size = sizeof(BestResult);

    cudaMalloc(&g_persistent_mem.d_headers, headers_size);
    cudaMalloc(&g_persistent_mem.d_outputs, outputs_size);
    cudaMalloc(&g_persistent_mem.d_memories, memories_size);
    cudaMalloc(&g_persistent_mem.d_target, target_size);
    cudaMalloc(&g_persistent_mem.d_solution_found, solution_size);
    cudaMalloc(&g_persistent_mem.d_solution_nonce, solution_size);
    cudaMalloc(&g_persistent_mem.d_partial_results, partial_results_size);
    cudaMalloc(&g_persistent_mem.d_final_result, final_result_size);


    cudaHostAlloc(&g_persistent_mem.h_headers_pinned, headers_size, cudaHostAllocDefault);
    cudaHostAlloc(&g_persistent_mem.h_target_pinned, target_size, cudaHostAllocDefault);
    cudaHostAlloc(&g_persistent_mem.h_solution_found_pinned, solution_size, cudaHostAllocDefault);
    cudaHostAlloc(&g_persistent_mem.h_solution_nonce_pinned, solution_size, cudaHostAllocDefault);
    cudaHostAlloc(&g_persistent_mem.h_final_result_pinned, final_result_size, cudaHostAllocDefault);


    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&g_persistent_mem.streams[i]);
    }

    g_persistent_mem.allocated_blocks = max_threads;
    g_persistent_mem.initialized = true;

    fprintf(stderr, "✅ Persistent memory (v3-GPU-Reduce) initialized: %u threads (%.2f MB VRAM)\n",
            max_threads, (outputs_size + memories_size + partial_results_size) / (1024.0 * 1024.0));
}

// 🚀 Dọn dẹp Persistent Memory (Đã cập nhật)
extern "C" void rinhash_persistent_cleanup() {
    if (!g_persistent_mem.initialized) return;

    cudaFree(g_persistent_mem.d_headers);
    cudaFree(g_persistent_mem.d_outputs);
    cudaFree(g_persistent_mem.d_memories);
    cudaFree(g_persistent_mem.d_target);
    cudaFree(g_persistent_mem.d_solution_found);
    cudaFree(g_persistent_mem.d_solution_nonce);
    cudaFree(g_persistent_mem.d_partial_results);
    cudaFree(g_persistent_mem.d_final_result);

    cudaFreeHost(g_persistent_mem.h_headers_pinned);
    cudaFreeHost(g_persistent_mem.h_target_pinned);
    cudaFreeHost(g_persistent_mem.h_solution_found_pinned);
    cudaFreeHost(g_persistent_mem.h_solution_nonce_pinned);
    cudaFreeHost(g_persistent_mem.h_final_result_pinned);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(g_persistent_mem.streams[i]);
    }

    memset(&g_persistent_mem, 0, sizeof(PersistentMemory));
    fprintf(stderr, "🧹 Persistent memory (v3-GPU-Reduce) cleaned up\n");
}

// 🚀 Hàm mine chính (Đã cập nhật)
extern "C" void RinHash_mine_persistent(
    const uint32_t* work_data,  
    uint32_t nonce_offset,      
    uint32_t start_nonce,
    uint32_t num_nonces,        
    uint32_t* target,           
    uint32_t* found_nonce,      
    uint8_t* target_hash,       
    uint8_t* best_hash,         
    uint32_t* solution_found    
) {
    uint32_t num_threads = (num_nonces + NUM_WAVES - 1) / NUM_WAVES;
    
    if (num_threads > MAX_BATCH_BLOCKS_PER_GPU) {
        fprintf(stderr, "Batch too large (max %u threads)\n", MAX_BATCH_BLOCKS_PER_GPU);
        num_threads = MAX_BATCH_BLOCKS_PER_GPU;
    }
    
    rinhash_persistent_init(num_threads);
    
    const int threads_per_block = OPTIMAL_THREADS_PER_BLOCK;
    int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    memcpy(g_persistent_mem.h_headers_pinned, work_data, 80);
    memcpy(g_persistent_mem.h_target_pinned, target, 8 * sizeof(uint32_t));

    cudaStream_t stream = g_persistent_mem.streams[0];

    cudaMemcpyAsync(g_persistent_mem.d_headers, g_persistent_mem.h_headers_pinned,
                    80, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_persistent_mem.d_target, g_persistent_mem.h_target_pinned,
                    8 * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(g_persistent_mem.d_solution_found, 0, sizeof(uint32_t), stream);

    // Chạy kernel băm
    rinhash_cuda_kernel_optimized<<<blocks, threads_per_block, 0, stream>>>(
        g_persistent_mem.d_headers, 80, g_persistent_mem.d_outputs,
        num_threads, g_persistent_mem.d_memories, g_m_cost,
        g_persistent_mem.d_target, g_persistent_mem.d_solution_found, 
        g_persistent_mem.d_solution_nonce, start_nonce
    );
    
    // Download cờ giải pháp
    cudaMemcpyAsync(g_persistent_mem.h_solution_found_pinned, g_persistent_mem.d_solution_found,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(g_persistent_mem.h_solution_nonce_pinned, g_persistent_mem.d_solution_nonce,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    check_cuda("rinhash_cuda_batch_persistent");

    *solution_found = *g_persistent_mem.h_solution_found_pinned;
    
    // Tìm hash tốt nhất
    if (*solution_found) {
        // 🚀 Đã tìm thấy share!
        *found_nonce = *g_persistent_mem.h_solution_nonce_pinned;
        
        uint32_t winner_index = *found_nonce - start_nonce;
        if (winner_index < num_nonces) {
            // Download chỉ 32 bytes của hash thắng cuộc
            cudaMemcpyAsync(best_hash, // Ghi trực tiếp vào con trỏ kết quả
                            g_persistent_mem.d_outputs + winner_index * 32, 
                            32, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    } else {
        // 🚀 KHÔNG tìm thấy share. Chạy kernel reduction.
        
        uint32_t reduce_blocks = (num_nonces + REDUCE_THREADS_PER_BLOCK - 1) / REDUCE_THREADS_PER_BLOCK;
        if (reduce_blocks > REDUCE_MAX_BLOCKS) {
            reduce_blocks = REDUCE_MAX_BLOCKS;
        }

        find_best_hash_part1<<<reduce_blocks, REDUCE_THREADS_PER_BLOCK, 0, stream>>>(
            g_persistent_mem.d_outputs,
            num_nonces,
            start_nonce,
            g_persistent_mem.d_partial_results
        );
        
        find_best_hash_part2<<<1, REDUCE_THREADS_PER_BLOCK, 0, stream>>>(
            g_persistent_mem.d_partial_results,
            reduce_blocks,
            g_persistent_mem.d_final_result
        );

        // Download KẾT QUẢ CUỐI CÙNG
        cudaMemcpyAsync(g_persistent_mem.h_final_result_pinned, 
                        g_persistent_mem.d_final_result,
                        sizeof(BestResult), cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream);
        check_cuda("find_best_hash_reduction");

        // Gán kết quả
        memcpy(best_hash, g_persistent_mem.h_final_result_pinned->hash, 32);
        *found_nonce = g_persistent_mem.h_final_result_pinned->nonce;
        
        // 🚀 Vòng lặp 'for' trên CPU đã bị xóa.
    }
}
