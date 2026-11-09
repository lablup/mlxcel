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
// head (issues #2889, #2891, #2895). Compiled to an HSACO at model load and
// dispatched through `hal.executable.source`; see `src/native_qmv.rs` for the
// ABI. The packed layout is MLX's: eight 4-bit values per u32, little-endian
// nibbles, one f16 scale and bias per group of 64 inputs.
//
// Two row layouts share the dot-product helpers below. The warp layout gives
// each output row one warp and the block eight rows (K < 4096); each lane loads
// four consecutive u32 (16 bytes), which always falls inside one 64-input group.
// The block layout puts the whole workgroup on one row and is faster for long
// rows (K >= 4096). The q/k/v and gate/up kernels reuse the same layouts but
// bind several weight sets and write several outputs in one launch, so the
// activations are read once per group instead of once per projection.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

struct QmvSet {
    const uint32_t* w;
    const __half* s;
    const __half* b;
    float* y;
    int n;
};

__device__ __forceinline__ float qmv4_dot_warp(const __half* __restrict__ x,
                                               const uint32_t* __restrict__ wrow,
                                               const __half* __restrict__ srow,
                                               const __half* __restrict__ brow,
                                               int words, int lane) {
    float acc = 0.f;
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
    return acc;
}

__device__ __forceinline__ float qmv4_dot_block(const __half* __restrict__ x,
                                                const uint32_t* __restrict__ wrow,
                                                const __half* __restrict__ srow,
                                                const __half* __restrict__ brow,
                                                int words) {
    float acc = 0.f;
    for (int i = threadIdx.x; i < words; i += blockDim.x) {
        const uint32_t p = wrow[i];
        const int g = i >> 3;
        const float s = __half2float(srow[g]);
        const float b = __half2float(brow[g]);
        const int k0 = i * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int qv = (p >> (4 * j)) & 0xF;
            acc += (qv * s + b) * __half2float(x[k0 + j]);
        }
    }
    return acc;
}

__device__ __forceinline__ float qmv4_warp_sum(float acc, int lane) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off);
    }
    return acc;
}

__device__ __forceinline__ void qmv4_block_sum(float acc, float* warp_sum, float* out) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    acc = qmv4_warp_sum(acc, lane);
    if (lane == 0) {
        warp_sum[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        acc = (lane < warps) ? warp_sum[lane] : 0.f;
        acc = qmv4_warp_sum(acc, lane);
        if (lane == 0) {
            *out = acc;
        }
    }
}

__device__ __forceinline__ void qmv4_select2(int row, int n0, int n1,
                                             const uint32_t* w0, const __half* s0,
                                             const __half* b0, float* y0,
                                             const uint32_t* w1, const __half* s1,
                                             const __half* b1, float* y1,
                                             QmvSet* set, int* local) {
    *set = {w0, s0, b0, y0, n0};
    *local = row;
    if (row >= n0) {
        *set = {w1, s1, b1, y1, n1};
        *local = row - n0;
    }
}

__device__ __forceinline__ void qmv4_select3(int row, int n0, int n1, int n2,
                                             const uint32_t* w0, const __half* s0,
                                             const __half* b0, float* y0,
                                             const uint32_t* w1, const __half* s1,
                                             const __half* b1, float* y1,
                                             const uint32_t* w2, const __half* s2,
                                             const __half* b2, float* y2,
                                             QmvSet* set, int* local) {
    *set = {w0, s0, b0, y0, n0};
    *local = row;
    if (row >= n0) {
        *set = {w1, s1, b1, y1, n1};
        *local = row - n0;
    }
    if (row >= n0 + n1) {
        *set = {w2, s2, b2, y2, n2};
        *local = row - n0 - n1;
    }
}

// One warp per output row, eight rows per workgroup.
extern "C" __global__ void qmv4_kernel(const __half* __restrict__ x,
                            const uint32_t* __restrict__ w,
                            const __half* __restrict__ scales,
                            const __half* __restrict__ biases,
                            float* __restrict__ y,
                            int K, int N) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) {
        return;
    }
    const int words = K / 8;
    const int groups = K / 64;
    float acc = qmv4_dot_warp(x, w + (size_t)row * words,
                              scales + (size_t)row * groups,
                              biases + (size_t)row * groups, words, lane);
    acc = qmv4_warp_sum(acc, lane);
    if (lane == 0) {
        y[row] = acc;
    }
}

// One workgroup per output row.
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
    __shared__ float warp_sum[32];
    const int words = K / 8;
    const int groups = K / 64;
    float acc = qmv4_dot_block(x, w + (size_t)row * words,
                               scales + (size_t)row * groups,
                               biases + (size_t)row * groups, words);
    qmv4_block_sum(acc, warp_sum, y + row);
}

// q/k/v in one launch: three weight sets, three outputs.
extern "C" __global__ void qmv4_qkv_kernel(const __half* __restrict__ x,
                            const uint32_t* __restrict__ w0,
                            const __half* __restrict__ s0,
                            const __half* __restrict__ b0,
                            const uint32_t* __restrict__ w1,
                            const __half* __restrict__ s1,
                            const __half* __restrict__ b1,
                            const uint32_t* __restrict__ w2,
                            const __half* __restrict__ s2,
                            const __half* __restrict__ b2,
                            float* __restrict__ y0,
                            float* __restrict__ y1,
                            float* __restrict__ y2,
                            int K, int N0, int N1, int N2) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N0 + N1 + N2) {
        return;
    }
    QmvSet set;
    int r;
    qmv4_select3(row, N0, N1, N2, w0, s0, b0, y0, w1, s1, b1, y1,
                 w2, s2, b2, y2, &set, &r);
    const int words = K / 8;
    const int groups = K / 64;
    float acc = qmv4_dot_warp(x, set.w + (size_t)r * words,
                              set.s + (size_t)r * groups,
                              set.b + (size_t)r * groups, words, lane);
    acc = qmv4_warp_sum(acc, lane);
    if (lane == 0) {
        set.y[r] = acc;
    }
}

extern "C" __global__ void qmv4_qkv_row_kernel(const __half* __restrict__ x,
                            const uint32_t* __restrict__ w0,
                            const __half* __restrict__ s0,
                            const __half* __restrict__ b0,
                            const uint32_t* __restrict__ w1,
                            const __half* __restrict__ s1,
                            const __half* __restrict__ b1,
                            const uint32_t* __restrict__ w2,
                            const __half* __restrict__ s2,
                            const __half* __restrict__ b2,
                            float* __restrict__ y0,
                            float* __restrict__ y1,
                            float* __restrict__ y2,
                            int K, int N0, int N1, int N2) {
    const int row = blockIdx.x;
    if (row >= N0 + N1 + N2) {
        return;
    }
    __shared__ float warp_sum[32];
    QmvSet set;
    int r;
    qmv4_select3(row, N0, N1, N2, w0, s0, b0, y0, w1, s1, b1, y1,
                 w2, s2, b2, y2, &set, &r);
    const int words = K / 8;
    const int groups = K / 64;
    float acc = qmv4_dot_block(x, set.w + (size_t)r * words,
                               set.s + (size_t)r * groups,
                               set.b + (size_t)r * groups, words);
    qmv4_block_sum(acc, warp_sum, set.y + r);
}

// gate/up in one launch: two weight sets, two outputs.
extern "C" __global__ void qmv4_gateup_kernel(const __half* __restrict__ x,
                            const uint32_t* __restrict__ w0,
                            const __half* __restrict__ s0,
                            const __half* __restrict__ b0,
                            const uint32_t* __restrict__ w1,
                            const __half* __restrict__ s1,
                            const __half* __restrict__ b1,
                            float* __restrict__ y0,
                            float* __restrict__ y1,
                            int K, int N0, int N1) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N0 + N1) {
        return;
    }
    QmvSet set;
    int r;
    qmv4_select2(row, N0, N1, w0, s0, b0, y0, w1, s1, b1, y1, &set, &r);
    const int words = K / 8;
    const int groups = K / 64;
    float acc = qmv4_dot_warp(x, set.w + (size_t)r * words,
                              set.s + (size_t)r * groups,
                              set.b + (size_t)r * groups, words, lane);
    acc = qmv4_warp_sum(acc, lane);
    if (lane == 0) {
        set.y[r] = acc;
    }
}

extern "C" __global__ void qmv4_gateup_row_kernel(const __half* __restrict__ x,
                            const uint32_t* __restrict__ w0,
                            const __half* __restrict__ s0,
                            const __half* __restrict__ b0,
                            const uint32_t* __restrict__ w1,
                            const __half* __restrict__ s1,
                            const __half* __restrict__ b1,
                            float* __restrict__ y0,
                            float* __restrict__ y1,
                            int K, int N0, int N1) {
    const int row = blockIdx.x;
    if (row >= N0 + N1) {
        return;
    }
    __shared__ float warp_sum[32];
    QmvSet set;
    int r;
    qmv4_select2(row, N0, N1, w0, s0, b0, y0, w1, s1, b1, y1, &set, &r);
    const int words = K / 8;
    const int groups = K / 64;
    float acc = qmv4_dot_block(x, set.w + (size_t)r * words,
                               set.s + (size_t)r * groups,
                               set.b + (size_t)r * groups, words);
    qmv4_block_sum(acc, warp_sum, set.y + r);
}
