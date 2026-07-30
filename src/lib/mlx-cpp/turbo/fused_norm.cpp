// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

#include "fused_norm.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

// Backend availability probes for the Metal-vs-CUDA kernel gate. Both headers
// are backend-agnostic: an Apple build links the CUDA no_cuda stub, a CUDA build
// links the Metal no_metal stub, so `is_available()` always resolves and returns
// false for the absent backend. Same pattern as `paged_attention.cpp`.
#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/metal/metal.h>

#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Body of the fused residual-add + RMSNorm Metal kernel. The string is the
// kernel BODY only; `mlx::core::fast::metal_kernel` wraps it with the
// declaration, buffer arguments, and template-argument substitution.
//
// One threadgroup owns one row of `Dim` elements. `Threads` lanes sweep the row
// in `N_READS = 4`-wide contiguous chunks, matching MLX's own `rms_single_row`
// access pattern, so the coalescing behavior of the fused and unfused paths is
// the same and a measured difference is attributable to the fusion rather than
// to a worse memory schedule.
//
// Two passes over the row:
//
//   1. Read `x` and `residual`, sum in fp32, round the sum to the activation
//      dtype, store it as `new_residual`, and accumulate its square. Rounding
//      *before* squaring is what makes this agree with the unfused
//      `add -> fast::rms_norm` pair: the unfused RMSNorm reads the dtype-rounded
//      sum out of global memory, so squaring the un-rounded fp32 sum here would
//      introduce a systematic (if tiny) difference that grows with row length.
//   2. Re-read `new_residual` (still hot in cache, it was just written by this
//      same threadgroup) and write `normed`.
//
//   The alternative to the second read is holding the row in registers, which
//   at Dim = 8192 is 8 floats per lane at Threads = 1024 and spills. The re-read
//   is an L2 hit, the register file is not.
//
// The reduction is the standard two stage shape: `simd_sum` inside each SIMD
// group, one scalar per SIMD group into threadgroup memory, then SIMD group 0
// reduces those and publishes `rsqrt(mean + eps)`.
//
// Template constants:
//   T       - activation dtype (also the dtype of both outputs).
//   Dim     - row length D (the normalized axis).
//   Threads - threadgroup width; a multiple of 32, at most 1024.
//
// Buffers (order matches the launcher's input vector):
//   x            [..., Dim]  T
//   residual     [..., Dim]  T
//   weight       [Dim]       (any float dtype)
//   eps          [1]         f32
//   weight_bias  [1]         f32
//   normed       [..., Dim]  T   (output)
//   new_residual [..., Dim]  T   (output)
constexpr const char* FUSED_ADD_RMS_NORM_SOURCE = R"(
    uint row = threadgroup_position_in_grid.z;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    const uint dim = (uint)Dim;
    const uint tg = (uint)Threads;
    const ulong base = (ulong)row * (ulong)dim;

    threadgroup float local_sums[32];
    threadgroup float local_inv_mean[1];

    // Stage 1: residual sum, store, sum of squares.
    float acc = 0.0f;
    for (uint r = 0; r < dim; r += tg * 4) {
        uint d0 = r + lid * 4;
        for (uint j = 0; j < 4; j++) {
            uint d = d0 + j;
            if (d < dim) {
                float s = (float)x[base + d] + (float)residual[base + d];
                T st = (T)s;
                new_residual[base + d] = st;
                float sr = (float)st;
                acc += sr * sr;
            }
        }
    }

    acc = simd_sum(acc);
    if (sg == 0u) {
        local_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        local_sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            local_inv_mean[0] = metal::precise::rsqrt(total / dim + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: scale and apply the (weight_bias + weight) gain.
    float inv_mean = local_inv_mean[0];
    float wbias = weight_bias[0];
    for (uint r = 0; r < dim; r += tg * 4) {
        uint d0 = r + lid * 4;
        for (uint j = 0; j < 4; j++) {
            uint d = d0 + j;
            if (d < dim) {
                float sr = (float)new_residual[base + d];
                float wv = (float)weight[d];
                // Fold the bias in the weight's own precision so `weight_bias
                // = 1` reproduces GemmaRMSNorm's precomputed `(1 + w)` tensor
                // exactly rather than approximately.
                float gain = (wbias == 0.0f) ? wv : (float)(TW)(wbias + wv);
                // Round `x * inv_mean` to T before the gain multiply: this is
                // the store order MLX's rms_norm kernel uses.
                float scaled = (float)(T)(sr * inv_mean);
                normed[base + d] = (T)(gain * scaled);
            }
        }
    }
)";

// CUDA port of the fused residual-add + RMSNorm kernel.
//
// UNVALIDATED. This host has no CUDA device and no `nvcc`, so this string has
// never been compiled or executed; it is a structural transliteration of the
// Metal body above, kept deliberately line-for-line parallel so a reviewer with
// a CUDA box can diff the two rather than re-derive the algorithm. The first
// CUDA run should be checked against `fused_add_rms_norm_matches_graph_*` in
// `src/lib/mlxcel-core/src/fused_norm_parity_tests.rs` before the kill switch
// is left on by default there.
//
// Same one-block-per-row mapping, same `N_READS = 4` sweep, same two-stage
// reduction with `__shfl_xor_sync` standing in for `simd_sum` (a butterfly
// all-reduce, so every lane sees the total, matching Metal's semantics) and
// `__syncthreads()` for the threadgroup barrier.
//
// Grid mapping: MLX passes Metal-style total threads and ceil-divides by the
// threadgroup tuple (cf. `backend/cuda/custom_kernel.cpp`), so grid
// `(Threads, 1, rows)` over threadgroup `(Threads, 1, 1)` yields blocks
// `(1, 1, rows)` with `blockDim = (Threads, 1, 1)`. Every field divides exactly,
// so no padded threads are launched and the `__syncthreads()` calls are always
// reached by the whole block.
constexpr const char* FUSED_ADD_RMS_NORM_CUDA_SOURCE = R"(
    uint32_t row = blockIdx.z;
    uint32_t lid = threadIdx.x;
    uint32_t lane = threadIdx.x % 32u;
    uint32_t sg = threadIdx.x / 32u;

    const uint32_t dim = (uint32_t)Dim;
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
    for (uint32_t r = 0; r < dim; r += tg * 4) {
        uint32_t d0 = r + lid * 4;
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t d = d0 + j;
            if (d < dim) {
                float sr = (float)new_residual[base + d];
                float wv = (float)weight[d];
                float gain = (wbias == 0.0f) ? wv : (float)(TW)(wbias + wv);
                float scaled = (float)(T)(sr * inv_mean);
                normed[base + d] = (T)(gain * scaled);
            }
        }
    }
)";

// Elements each lane reads per sweep step. Matches MLX's `RMS_N_READS`.
constexpr int FUSED_NORM_N_READS = 4;
constexpr int FUSED_NORM_SIMD_WIDTH = 32;
constexpr int FUSED_NORM_MAX_THREADS = 1024;

// Threadgroup width for a row of `dim` elements: enough lanes to cover the row
// at `FUSED_NORM_N_READS` elements each, rounded up to a whole SIMD group and
// clamped to the hardware threadgroup limit. Rows longer than
// `FUSED_NORM_MAX_THREADS * FUSED_NORM_N_READS` loop.
int fused_norm_threads(int dim) {
    if (dim <= 0) {
        return FUSED_NORM_SIMD_WIDTH;
    }
    int threads = (dim + FUSED_NORM_N_READS - 1) / FUSED_NORM_N_READS;
    threads = ((threads + FUSED_NORM_SIMD_WIDTH - 1) / FUSED_NORM_SIMD_WIDTH) *
        FUSED_NORM_SIMD_WIDTH;
    if (threads > FUSED_NORM_MAX_THREADS) {
        threads = FUSED_NORM_MAX_THREADS;
    }
    if (threads < FUSED_NORM_SIMD_WIDTH) {
        threads = FUSED_NORM_SIMD_WIDTH;
    }
    return threads;
}

const std::vector<std::string>& fused_norm_input_names() {
    static const std::vector<std::string> names = {
        "x", "residual", "weight", "eps", "weight_bias"};
    return names;
}

const std::vector<std::string>& fused_norm_output_names() {
    static const std::vector<std::string> names = {"normed", "new_residual"};
    return names;
}

// Thread-safe lazy-initialised holder for the JIT-compiled kernel. Mirrors the
// `std::call_once` pattern in `paged_attention.cpp`: the server reaches first
// use concurrently from per-request blocking workers, and `call_once` re-runs
// the initializer if MLX device lookup throws.
struct FusedNormKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_fused_add_rms_norm",
                fused_norm_input_names(),
                fused_norm_output_names(),
                std::string(FUSED_ADD_RMS_NORM_SOURCE));
        });
        return *kernel;
    }
};

inline FusedNormKernelHolder& get_fused_norm_kernel() {
    static FusedNormKernelHolder holder;
    return holder;
}

struct FusedNormKernelHolderCuda {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::cuda_kernel(
                "mlxcel_fused_add_rms_norm",
                fused_norm_input_names(),
                fused_norm_output_names(),
                std::string(FUSED_ADD_RMS_NORM_CUDA_SOURCE));
        });
        return *kernel;
    }
};

inline FusedNormKernelHolderCuda& get_fused_norm_kernel_cuda() {
    static FusedNormKernelHolderCuda holder;
    return holder;
}

} // namespace

bool fused_add_rms_norm_available() {
    return mlx::core::metal::is_available() || mlx::core::cu::is_available();
}

std::vector<mlx::core::array> fused_add_rms_norm(
    const mlx::core::array& x,
    const mlx::core::array& residual,
    const mlx::core::array& weight,
    float eps,
    float weight_bias) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    if (x.shape() != residual.shape()) {
        std::ostringstream msg;
        msg << "[fused_add_rms_norm] x and residual must have the same shape.";
        throw std::invalid_argument(msg.str());
    }
    if (x.dtype() != residual.dtype()) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] x and residual must have the same dtype.");
    }
    if (x.ndim() < 1 || weight.ndim() != 1) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] x must be at least 1-D and weight exactly 1-D.");
    }
    const int dim = x.shape().back();
    if (weight.shape()[0] != dim) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] weight length must equal the last dim of x.");
    }
    if (x.size() % dim != 0) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] x size is not a multiple of its last dim.");
    }

    const int rows = static_cast<int>(x.size() / dim);
    const int threads = fused_norm_threads(dim);

    // Metal kernel on Apple, CUDA port elsewhere. `fast::metal_kernel` throws
    // "[metal_kernel] No Metal back-end" on the CUDA backend and vice versa;
    // `metal::is_available()` is false on a CUDA-only build.
    const bool use_cuda = !mlx::core::metal::is_available();
    auto& kernel =
        use_cuda ? get_fused_norm_kernel_cuda().get() : get_fused_norm_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"T", x.dtype()},
        {"TW", weight.dtype()},
        {"Dim", dim},
        {"Threads", threads},
    };

    // `eps` and `weight_bias` ride in as 1-element f32 arrays rather than 0-D
    // scalars: a 0-D input is declared `const constant float&` by the Metal
    // signature writer and `const float*` by the CUDA one, so the two kernel
    // bodies would have to differ purely on argument syntax. The `array({v})`
    // constructor sets the data directly, so neither of these costs a dispatch.
    auto eps_arr = mlx::core::array({eps});
    auto bias_arr = mlx::core::array({weight_bias});

    std::vector<mlx::core::array> inputs = {x, residual, weight, eps_arr, bias_arr};
    std::vector<Shape> output_shapes = {x.shape(), x.shape()};
    std::vector<Dtype> output_dtypes = {x.dtype(), x.dtype()};

    return kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(threads, 1, rows), // grid
        std::make_tuple(threads, 1, 1),    // threadgroup
        template_args,
        std::nullopt,
        false,
        {});
}

} // namespace mlxcel::turbo
