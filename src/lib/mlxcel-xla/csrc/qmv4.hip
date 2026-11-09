// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Native 4-bit affine GEMV for single-sequence decode projections and the LM
// head (issues #2889, #2891). Compiled to an HSACO at model load and dispatched
// through `hal.executable.source @custom_qmv`; see `src/native_qmv.rs` for the
// ABI. The packed layout is MLX's: eight 4-bit values per u32, little-endian
// nibbles, one f16 scale and bias per group of 64 inputs.
//
// One warp owns one output row and the block handles `blockDim.x / 32` rows, so
// the reduction is a warp shuffle with no shared memory or barrier. Each lane
// loads four consecutive u32 (16 bytes) per iteration; a group of 64 inputs is
// eight u32, so one 16-byte load always falls inside a single group and costs
// one scale/bias pair.
//
// Long rows (the down projection's K = 4 * hidden) do better with the whole
// block on one row, where eight warps cover the row in parallel:
// `qmv4_row_kernel` keeps that layout and the emitter picks it for K >= 4096.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void qmv4_kernel(const __half* __restrict__ x,
                            const uint32_t* __restrict__ w,
                            const __half* __restrict__ scales,
                            const __half* __restrict__ biases,
                            float* __restrict__ y,
                            int K, int N) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int row = blockIdx.x * warps + warp;
    if (row >= N) {
        return;
    }
    const int words = K / 8;   // u32 words per row
    const int groups = K / 64; // scale/bias pairs per row
    const uint32_t* wrow = w + (size_t)row * words;
    const __half* srow = scales + (size_t)row * groups;
    const __half* brow = biases + (size_t)row * groups;

    float acc = 0.f;
    // Word offset of this lane; the warp covers 128 words per iteration.
    for (int i = lane * 4; i + 4 <= words; i += 32 * 4) {
        const uint4 p = *reinterpret_cast<const uint4*>(wrow + i);
        const int g = i >> 3;
        const float s = __half2float(srow[g]);
        const float b = __half2float(brow[g]);
        const int k0 = i * 8;
        const uint32_t q[4] = {p.x, p.y, p.z, p.w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int qv = (q[u] >> (4 * j)) & 0xF;
                acc += (qv * s + b) * __half2float(x[k0 + u * 8 + j]);
            }
        }
    }
    // Warp reduction; lane 0 writes the row.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    if (lane == 0) {
        y[row] = acc;
    }
}

// One workgroup per output row: every thread strides the row's u32 words, then
// the block reduces with a warp shuffle plus shared memory.
extern "C" __global__ void qmv4_row_kernel(const __half* __restrict__ x,
                            const uint32_t* __restrict__ w,
                            const __half* __restrict__ scales,
                            const __half* __restrict__ biases,
                            float* __restrict__ y,
                            int K, int N) {
    const int row = blockIdx.x;
    if (row >= N) {
        return;
    }
    const int words = K / 8;
    const int groups = K / 64;
    float acc = 0.f;
    for (int i = threadIdx.x; i < words; i += blockDim.x) {
        const uint32_t p = w[(size_t)row * words + i];
        const int g = i >> 3;
        const float s = __half2float(scales[(size_t)row * groups + g]);
        const float b = __half2float(biases[(size_t)row * groups + g]);
        const int k0 = i * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int qv = (p >> (4 * j)) & 0xF;
            acc += (qv * s + b) * __half2float(x[k0 + j]);
        }
    }
    __shared__ float warp_sum[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        warp_sum[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        acc = (lane < warps) ? warp_sum[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down(acc, off);
        }
        if (lane == 0) {
            y[row] = acc;
        }
    }
}
