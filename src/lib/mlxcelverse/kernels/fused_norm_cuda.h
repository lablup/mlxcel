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

#pragma once

// CUDA body and launch geometry of the fused residual-add + RMSNorm kernel.
//
// Kept apart from `fused_norm.cpp` only for size, as `fused_norm_hip.h` is:
// the launch, and the `fast::cuda_kernel` call that compiles this string, stay
// in `fused_norm.cpp`, so `make verify-kernel-dtype-keys` keeps its pinned
// scope. This header holds data and one pure function, not a launch.
//
// Byte-identical to the graph it replaces on CUDA, `add` then MLX's
// `rms_norm` (`mlx/backend/cuda/rms_norm.cu`), for every dtype pair and row
// width the fused path accepts (#3510). `cuda_byte_identity.rs` in the parity
// tests pins that, and the cuda-oxide artifact
// (`spike/cuda-oxide-fused-norm`) is a bitwise port of this body. Before
// #3510 the kernel swept the row with `fused_norm_threads(dim)` lanes of four
// elements, which reduced the fp32 sum of squares in another order than MLX
// and left a `normed` element one or two bf16 ulps away on a few rows in ten
// thousand (#3501).
//
// What makes it identical, each point taken from the MLX kernels:
//
// - Compute dtype `TC = promote_types(T, TW)`, the dtype `fast::rms_norm`
//   casts both inputs to and returns. With this tree's promotion table
//   (`overlays/*/mlx/dtype.cpp`) bf16 with f32 is bf16, either way round,
//   while f16 with f32 and f16 with bf16 are f32. A row is read as vectors of
//   `N_READS = 16 / sizeof(TC)` elements: 8 for f16 and bf16, 4 for f32.
// - Block width `fused_norm_cuda_block(dim, N_READS)` below, which is MLX's
//   `dispatch_num_chunks` plus the `eval_gpu` threshold for the looped kernel.
// - Summation order. Thread `t` adds the squares of vectors `t`, `t + block`,
//   `t + 2 * block`, ... in ascending order, element by element, into one fp32
//   register (`acc += s * s`, one statement, so NVRTC contracts it to an FMA
//   as nvcc does for MLX's `normalizer += t * t`). `rms_norm_small` unrolls
//   the same walk over `N_CHUNKS` vectors and `rms_norm` loops over it; the
//   elements a vector has past the end of the row are zero there and skipped
//   here, which adds `+0` to a non-negative sum either way. One loop therefore
//   covers both MLX kernels.
// - Reduction. A `__shfl_xor_sync` butterfly (16 down to 1) per warp, the
//   warp totals into `local_sums`, and a second butterfly over the 32 slots
//   with the unused ones zero. That is `BlockBroadcastReduce`: `cg::reduce` on
//   a float is the same xor butterfly, and its second stage reads zero for
//   lanes past the warp count.
// - `rsqrtf(total / dim + eps)`, then `w * (TC)(x * inv)` with both factors
//   in `TC`. For a 16-bit `TC` the fp32 product of two 16-bit values is
//   exact, so rounding it once is MLX's multiply in `TC`. For `TC = f32`
//   nothing is rounded in between.
// - Casts to `TC` where the graph has them: the residual sum before it is
//   squared and scaled when `TC` is not `T` (an f32 activation under a bf16
//   weight computes in bf16), and the weight when `TC` is not `TW` (a bf16
//   activation under an f32 weight rounds the weight to bf16). `RoundX` and
//   `RoundW` say which apply, so a same-dtype launch compiles neither.
//
// Both outputs are written in the activation dtype. `new_residual` is the
// graph's `add`, which is in `T` too. `normed` is the graph's when `TC` is
// `T`, and the graph's cast to `T` otherwise (f16 under an f32 or bf16
// weight, where the graph returns f32, and f32 under a bf16 weight, where it
// returns bf16); changing the output dtype is not this kernel's business.
//
// Same one-block-per-row mapping as the Metal kernel, with `__syncthreads()`
// for the threadgroup barrier. Grid mapping: MLX passes Metal-style total
// threads and ceil-divides by the threadgroup tuple (cf.
// `backend/cuda/custom_kernel.cpp`), so grid `(Threads, 1, rows)` over
// threadgroup `(Threads, 1, 1)` yields blocks `(1, 1, rows)` with
// `blockDim = (Threads, 1, 1)`. Every field divides exactly, so no padded
// threads are launched and the `__syncthreads()` calls are always reached by
// the whole block.
//
// Template constants:
//   T       - activation dtype (also the dtype of both outputs).
//   TW      - weight dtype.
//   TC      - compute dtype, `promote_types(T, TW)`.
//   Dim     - row length D (the normalized axis).
//   Threads - block width, `fused_norm_cuda_block(Dim, 16 / sizeof(TC))`.
//   RoundX  - 1 when `TC` is not `T`, else 0.
//   RoundW  - 1 when `TC` is not `TW`, else 0.

namespace mlxcel::turbo {

inline constexpr const char* FUSED_ADD_RMS_NORM_CUDA_SOURCE = R"(
    uint32_t row = blockIdx.z;
    uint32_t lid = threadIdx.x;
    uint32_t lane = threadIdx.x % 32u;
    uint32_t sg = threadIdx.x / 32u;

    const uint32_t dim = (uint32_t)Dim;
    const uint32_t block = (uint32_t)Threads;
    const uint32_t n_reads = 16u / (uint32_t)sizeof(TC);
    const uint32_t nvec = (dim + n_reads - 1u) / n_reads;
    const uint64_t base = (uint64_t)row * (uint64_t)dim;

    __shared__ float local_sums[32];
    __shared__ float local_inv_mean[1];

    // Stage 1: residual sum, store, sum of squares.
    float acc = 0.0f;
    for (uint32_t v = lid; v < nvec; v += block) {
        for (uint32_t j = 0; j < n_reads; j++) {
            uint32_t d = v * n_reads + j;
            if (d < dim) {
                T st = (T)((float)x[base + d] + (float)residual[base + d]);
                new_residual[base + d] = st;
                // The value the graph's rms_norm reads: the sum cast to `TC`.
                float s = (float)st;
                if (RoundX) {
                    s = (float)(TC)s;
                }
                acc += s * s;
            }
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    }
    if (sg == 0u) {
        local_sums[lane] = 0.0f;
    }
    __syncthreads();
    if (lane == 0u) {
        local_sums[sg] = acc;
    }
    __syncthreads();
    if (sg == 0u) {
        float total = local_sums[lane];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            total += __shfl_xor_sync(0xffffffffu, total, o);
        }
        if (lane == 0u) {
            local_inv_mean[0] = rsqrtf(total / dim + eps[0]);
        }
    }
    __syncthreads();

    // Stage 2: scale and apply the (weight_bias + weight) gain.
    float inv_mean = local_inv_mean[0];
    float wbias = weight_bias[0];
    for (uint32_t v = lid; v < nvec; v += block) {
        for (uint32_t j = 0; j < n_reads; j++) {
            uint32_t d = v * n_reads + j;
            if (d < dim) {
                float s = (float)new_residual[base + d];
                if (RoundX) {
                    s = (float)(TC)s;
                }
                float wv = (float)weight[d];
                // Fold the bias in the weight's own precision so `weight_bias
                // = 1` reproduces GemmaRMSNorm's precomputed `(1 + w)` tensor,
                // then cast to `TC` as rms_norm casts its weight.
                float gain = (wbias == 0.0f) ? wv : (float)(TW)(wbias + wv);
                if (RoundW) {
                    gain = (float)(TC)gain;
                }
                // Round `x * inv_mean` to the compute dtype before the gain
                // multiply, as MLX's `w * (T)y` does. The product is rounded
                // once: to `T` by the store, or to `TC` first when they
                // differ, which is the graph's multiply in `TC`.
                float scaled = (float)(TC)(s * inv_mean);
                float product = gain * scaled;
                if (RoundX) {
                    product = (float)(TC)product;
                }
                normed[base + d] = (T)product;
            }
        }
    }
)";

// MLX's `N_READS` for a compute dtype of `compute_bytes` bytes: the elements
// of one 16-byte vector (`16 / sizeof(T)` in `RMSNorm::eval_gpu`).
constexpr int fused_norm_cuda_n_reads(int compute_bytes) {
    return 16 / compute_bytes;
}

// Block width MLX's CUDA `rms_norm` launches for a row of `dim` elements read
// as `n_reads`-wide vectors (`RMSNorm::eval_gpu` and `dispatch_num_chunks` in
// `mlx/backend/cuda/rms_norm.cu`, at the pin in `CMakeLists.txt`; re-check on
// an MLX pin bump, `cuda_byte_identity.rs` fails if it moves). Rows wider than
// `n_reads * 1024` take the looped kernel at 1024 threads. Narrower ones take
// `rms_norm_small`: 64 threads for one chunk, 128 when the vector count is a
// multiple of 128, 64 when it is a multiple of 64 only and fits eight chunks,
// 128 otherwise.
constexpr int fused_norm_cuda_block(int dim, int n_reads) {
    const int nvec = (dim + n_reads - 1) / n_reads;
    if (dim > n_reads * 1024) {
        return 1024;
    }
    if (dim <= n_reads * 64) {
        return 64;
    }
    if (nvec % 128 == 0) {
        return 128;
    }
    if (nvec % 64 == 0 && nvec <= 512) {
        return 64;
    }
    return 128;
}

} // namespace mlxcel::turbo
