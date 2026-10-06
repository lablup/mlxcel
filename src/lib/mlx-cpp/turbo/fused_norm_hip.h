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

// HIP body of the fused residual-add + RMSNorm kernel (issue #2063).
//
// Kept apart from `fused_norm.cpp` only for size: the launch, and the
// `fast::hip_kernel` call that compiles this string, stay in `fused_norm.cpp`
// next to the Metal and CUDA launches, so `make verify-kernel-dtype-keys` keeps
// its pinned scope unchanged. This header holds data, not a launch.
//
// A line-for-line port of `FUSED_ADD_RMS_NORM_CUDA_SOURCE`: same one-block-per-
// row mapping, same `N_READS = 4` sweep, same template args (`T`, `TW`, `Dim`,
// `Threads`), same grid `(Threads, 1, rows)` over `(Threads, 1, 1)`, which the
// ROCm `CustomKernel::eval_gpu` ceil-divides into blocks `(1, 1, rows)` exactly
// as CUDA does. The differences, each chosen to land on the numbers of the
// graph this kernel replaces on ROCm (`add` then the overlay's
// `rms_norm_kernel` in `rms_norm.hip`):
//
// - The launch in `fused_norm.cpp` fixes `Threads` at 256 on ROCm, the
//   overlay's `BLOCK_DIM`, instead of sizing it from the row, so the
//   per-thread strided sweep, the 32-wide xor folds and the group sums follow
//   the graph's reduction tree.
//
// - The row length comes from `weight_shape[0]` at run time rather than from
//   the `Dim` constant, as the graph's kernel takes it. With the constant,
//   hipRTC compiled the 4096-wide sweep so that 44 of 4096 f32 rows in a
//   stress run (row scales spread over e^-8 to e^8) came out a normalizer ulp
//   from the graph's; with the runtime length, none did, at widths 128, 2048,
//   3584 and 4096 in f32, f16 and bf16.
//
// - `__shfl_xor(v, o, 32)` for `__shfl_xor_sync(mask, v, o)`. HIP's `_sync`
//   form is a compatibility shim that ignores its mask, and the native form
//   takes the width. The width of 32 is what keeps each butterfly inside one
//   32-lane group: `lane` and `sg` below are `threadIdx.x % 32` and
//   `threadIdx.x / 32`, so on a 64-lane wavefront (CDNA) the shuffle stays in
//   its half and both halves reduce separately into `local_sums`, which is the
//   intended result. Wave64 has not been run.
// - The gain multiply runs in `T` (`gain * scaled`, both `T`), as the
//   graph's `w * normalized` does, instead of in f32 followed by one rounding.
//   For f16, bf16 and f32 the two give the same value, but not the same zero:
//   hipRTC's code for the f32 form returned +0 where the product is -0 (an
//   underflowed negative element, or a zero weight times a negative one),
//   which the graph keeps, and that was enough to move Llama 3.1 logits at 5
//   of 128 decode positions with the fusion on. A weight dtype other than
//   `T` is rounded to `T` first.
// - `1.0f / sqrtf(...)` for `rsqrtf(...)`: the overlay's `rms_norm_row` uses
//   the correctly rounded division for the same reason Metal uses
//   `metal::precise::rsqrt`, and matching it keeps the normalizer identical up
//   to the order of the sum.
//
// The wavefront `#error` guard below is the one #1814 asks of every shuffle-
// based port. It is inert with HIP 7.15's AMD clang 23, which defines neither
// `__AMDGCN_WAVEFRONT_SIZE` spelling for gfx1151 or gfx942 (checked with
// `hipcc -E -dM` in #2065), so it does not protect anything on current
// toolchains; the explicit width of 32 is what keeps the reduction correct.
// With a compiler that does define the macro it would fail the hipRTC compile
// at the first launch on a wave64 device (an error at evaluation, not a
// fallback) rather than run an untested kernel.
//
// The bf16 reads keep the CUDA text's explicit `(float)`, which is required:
// `hip_bfloat16`, the type MLX substitutes for bfloat16 on ROCm, converts to
// float only through an `explicit` operator.

namespace mlxcel::turbo {

inline constexpr const char* FUSED_ADD_RMS_NORM_HIP_SOURCE = R"(
    #if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
    #error "mlxcel_fused_add_rms_norm (HIP) has only been run on a 32-lane wavefront"
    #endif
    #if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
    #error "mlxcel_fused_add_rms_norm (HIP) has only been run on a 32-lane wavefront"
    #endif
    uint32_t row = blockIdx.z;
    uint32_t lid = threadIdx.x;
    uint32_t lane = threadIdx.x % 32u;
    uint32_t sg = threadIdx.x / 32u;

    // The row length is read from the weight's shape at run time, not from
    // the `Dim` template constant (which stays in the cache key): with a
    // compile-time 4096, hipRTC compiled the sweep differently enough that
    // about 1% of f32 rows came out a normalizer ulp away from the graph,
    // whose `rms_norm_kernel` takes the length as a runtime argument.
    const uint32_t dim = (uint32_t)weight_shape[0];
    const uint32_t tg = (uint32_t)Threads;
    const uint64_t base = (uint64_t)row * (uint64_t)dim;

    __shared__ float local_sums[32];
    __shared__ float local_inv_mean[1];

    // Stage 1: residual sum, store, sum of squares.
    float acc = 0.0f;
    for (uint32_t r = 0; r < dim; r += tg * 4) {
        uint32_t d0 = r + lid * 4;
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t d = d0 + j;
            if (d < dim) {
                float s = (float)x[base + d] + (float)residual[base + d];
                T st = (T)s;
                new_residual[base + d] = st;
                float sr = (float)st;
                acc += sr * sr;
            }
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor(acc, o, 32);
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
            total += __shfl_xor(total, o, 32);
        }
        if (lane == 0u) {
            local_inv_mean[0] = 1.0f / sqrtf(total / dim + (float)eps[0]);
        }
    }
    __syncthreads();

    // Stage 2: scale and apply the (weight_bias + weight) gain.
    float inv_mean = local_inv_mean[0];
    float wbias = (float)weight_bias[0];
    for (uint32_t r = 0; r < dim; r += tg * 4) {
        uint32_t d0 = r + lid * 4;
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t d = d0 + j;
            if (d < dim) {
                float sr = (float)new_residual[base + d];
                float wv = (float)weight[d];
                T gain = (wbias == 0.0f) ? (T)wv : (T)(float)(TW)(wbias + wv);
                T scaled = (T)(sr * inv_mean);
                normed[base + d] = gain * scaled;
            }
        }
    }
)";

} // namespace mlxcel::turbo
