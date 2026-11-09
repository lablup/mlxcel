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
// Native 4-bit affine GEMV for single-sequence decode projections (issue
// #2889). Compiled to an HSACO at model load and dispatched through
// `hal.executable.source @custom_qmv`; see `src/native_qmv.rs` for the ABI.
// The packed layout is MLX's: eight 4-bit values per u32, little-endian
// nibbles, one f16 scale and bias per group of 64 inputs.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void qmv4_kernel(const __half* __restrict__ x,
                            const uint32_t* __restrict__ w,
                            const __half* __restrict__ scales,
                            const __half* __restrict__ biases,
                            float* __restrict__ y,
                            int K, int N) {
    const int n = blockIdx.x;
    const int words = K / 8;
    const int groups = K / 64;
    float acc = 0.f;
    for (int i = threadIdx.x; i < words; i += blockDim.x) {
        const uint32_t p = w[(size_t)n * words + i];
        const int g = (i * 8) >> 6;
        const float s = __half2float(scales[(size_t)n * groups + g]);
        const float b = __half2float(biases[(size_t)n * groups + g]);
        const int k0 = i * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int q = (p >> (4 * j)) & 0xF;
            acc += (q * s + b) * __half2float(x[k0 + j]);
        }
    }
    // Block reduction: warp shuffle, then shared memory across warps.
    __shared__ float warp_sum[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    if (lane == 0) warp_sum[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < warps) ? warp_sum[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
        if (lane == 0) y[n] = acc;
    }
}

