#include <stdio.h>
#include <cstdint>
#include <memory.h>
#include <miner.h>
#include "cuda_helper.h" //

using namespace std;

// External reference to RinHash CUDA functions
//
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
);

extern "C" void rinhash_persistent_init(uint32_t max_blocks);
extern "C" void rinhash_persistent_cleanup();

// Thread-local variables
thread_local uint32_t *d_hash = NULL;
thread_local uint8_t *d_rinhash_out = NULL;

// Initialization function for RinHash algorithm
extern "C" void rinhash_init(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);
    // Khởi tạo VRAM cho 32768 luồng CUDA (không phải nonces)
    // 32768 luồng * 128 sóng/luồng = 4,194,304 nonces/lô
    // VRAM = 32768 luồng * 64KB/luồng = 2 GB VRAM
    // 🚀 FIX: Chúng ta sẽ để rinhash_persistent_init tự động điều chỉnh trong RinHash_mine_persistent
    // rinhash_persistent_init(32768); //
}

// Cleanup function for RinHash algorithm
extern "C" void rinhash_free(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);

    rinhash_persistent_cleanup(); //
    
    cudaFree(d_hash);
    cudaFree(d_rinhash_out);

    d_hash = NULL;
    d_rinhash_out = NULL;
}

// Main scanning function that tries different nonces to find a valid hash
int scanhash_rinhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    uint32_t nonce = first_nonce;
    if (opt_benchmark)
        ptarget[7] = 0xff;

    // 🚀 --- ĐÃ SỬA LỖI LOGIC (Lần 5) --- 🚀

    // 1. TỔNG CÔNG VIỆC (Total Batch):
    // Đọc từ --batch-size. Nếu không có, mặc định là 2M (2097152).
    // Con số này đủ lớn để GPU không bị "đói".
    uint32_t total_batch_size = (opt_batch_size > 0) ? opt_batch_size : 2097152;

    // 2. LÔ CHO KERNEL (Kernel Chunk):
    // Đây là kích thước của MỖI LẦN gọi kernel.
    // 128K (131072) là giá trị an toàn, không gây treo TDR.
    const uint32_t kernel_chunk_size = 131072;
    
    // Giới hạn tổng số nonces cho vòng lặp này
    max_nonce = min(first_nonce + total_batch_size, max_nonce);
    
    // 🚀 --- KẾT THÚC SỬA LỖI LOGIC --- 🚀

    uint32_t found_nonce = 0;
    uint32_t solution_found = 0;
    uint8_t best_hash[32];
    uint8_t target_hash[32]; // (Không dùng, nhưng API yêu cầu)
    uint32_t target[8];

    // Convert target (đã là little-endian)
    for (int i = 0; i < 8; i++) {
        target[i] = ptarget[i];
    }

    work->valid_nonces = 0;
    cudaSetDevice(device_map[thr_id]);

    do {
        // 🚀 LOGIC ĐÚNG: Tính toán lô (chunk) cho lần gọi kernel này
        uint32_t current_chunk_size = min(kernel_chunk_size, max_nonce - nonce);
        
        if (current_chunk_size <= 0) {
            *hashes_done = nonce - first_nonce;
            return 0; // Hoàn thành total_batch_size
        }

        solution_found = 0;
        RinHash_mine_persistent(
            pdata,
            19, // nonce offset
            nonce,
            current_chunk_size, // 🚀 GỌI KERNEL VỚI LÔ 128K
            target,
            &found_nonce,
            target_hash, // (Không dùng)
            best_hash,
            &solution_found
        );

        *hashes_done = nonce - first_nonce + current_chunk_size;

        if (solution_found) {
            uint32_t _ALIGN(64) vhash[8];
            memcpy(vhash, best_hash, 32);
            
            const uint32_t Htarg = ptarget[7];
            if (vhash[7] <= Htarg) {    
                work->valid_nonces = 1;
                work_set_target_ratio(work, vhash);
                work->nonces[0] = found_nonce;
                pdata[19] = found_nonce;
                
                // Trả về 1 (đã tìm thấy)
                // Cập nhật nonce cuối cùng đã quét
                pdata[19] = nonce + current_chunk_size;
                *hashes_done = nonce - first_nonce + current_chunk_size;
                return 1;
            } else {
                gpu_increment_reject(thr_id);
            }
        }
        
        // 🚀 LOGIC ĐÚNG: Tăng nonce và lặp lại vòng 'do...while' 
        // để xử lý chunk tiếp theo
        nonce += current_chunk_size;

    } while (nonce < max_nonce && !work_restart[thr_id].restart);

    pdata[19] = nonce;
    *hashes_done = nonce - first_nonce;
    return 0; // Hoàn thành lô (batch) mà không tìm thấy gì
}

// Empty function to detect algorithm - needed by ccminer
// (Hàm này không được gọi trong vòng lặp đào chính)
extern "C" void rinhash_hash(const void *output, const void *input)
{
    // (Bỏ qua, không cần thiết cho hiệu suất)
    //
}
