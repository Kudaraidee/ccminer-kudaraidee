/*
 * Copyright (C) 2018-2019 Ehsan Dalvand <dalvand.ehsan@gmail.com>, Alireza Jahandideh <ar.jahandideh@gmail.com>
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation: either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* For IDE: */
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "argon2d_kernel.h"

#define INPUT_LEN 80
__constant__ uint32_t d_data[20];

#include "argon2d_blake2b_device.cuh"
#include "cuda/candidate_report.cuh"

__global__ void argon2_initialize(struct block* memory, uint32_t startNonce,
    uint32_t mcost, uint32_t lanes, uint32_t passes, uint32_t version,
    uint32_t type, uint32_t total_blocks)
{

    uint32_t buffer[32];
    const uint32_t nonce = (blockIdx.x*blockDim.y+threadIdx.y) + startNonce;

    computeInitialHash(d_data, buffer, nonce, mcost, lanes, passes, version, type);
    fillFirstBlock(memory, buffer, lanes, total_blocks);

}

/* digests: optional 32-byte-per-job output, NULL in normal mining; used by
 * argon2d_gpu_differential() to compare the GPU against the CPU reference. */
__global__ void argon2_finalize(
    block* memory, uint32_t startNonce,
    uint32_t target, uint32_t* resNonces,
    uint32_t total_blocks, uint32_t* digests)
{

    extern __shared__ uint32_t input_t[];
    uint32_t* input = &(input_t[threadIdx.y*258]);
    uint64_t* input_64=(uint64_t*)input;

    uint32_t idx = threadIdx.x;
    uint32_t jobId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t nonce = jobId + startNonce;

    uint32_t* memLane = (uint32_t*) ((memory + jobId * total_blocks));
    partialState state;

    load_block(&input[1], memLane, idx);

    input[0] = 32;

    state.a = blake2b_Init_928[idx];
    state.b = blake2b_Init_928[idx + 4];

    blake2b_compress_4w(&state, &input_64[0], 1, idx);
    blake2b_compress_4w(&state, &input_64[16], 2, idx);
    blake2b_compress_4w(&state, &input_64[32], 3, idx);
    blake2b_compress_4w(&state, &input_64[48], 4, idx);
    blake2b_compress_4w(&state, &input_64[64], 5, idx);
    blake2b_compress_4w(&state, &input_64[80], 6, idx);
    blake2b_compress_4w(&state, &input_64[96], 7, idx);
    blake2b_compress_4w(&state, &input_64[112], 8, idx);

    /* The four threads of a job share this buffer; independent thread
     * scheduling (sm_70+) needs explicit barriers between read and rewrite. */
    const uint32_t tail = input[256];
    __syncwarp();
    zero_buffer(&input[0], idx);
    input[0] = tail;
    __syncwarp();

    blake2b_compress_4w(&state, &input_64[0], 9, idx, true, 4);

    /* the compress above read the words this store overwrites */
    __syncwarp();
    input_64[idx] = state.a;

    if (digests != NULL)
        ((uint64_t*)digests)[jobId * 4 + idx] = state.a;

    __syncthreads();


    /* keeps the two lowest candidates; armed to UINT32_MAX per batch */
    if (idx == 0 && input[7] <= target)
        report_candidate_2(resNonces, nonce);

}

__host__ void set_data(const void* data) {

    cudaMemcpyToSymbol(d_data, data, INPUT_LEN);

}
