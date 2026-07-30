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

// Variable-length attention-state merge kernel (issue #898).
//
// Split out of `paged_attention_v2.cpp` because it is the reusable half: the
// cascade-attention issue #903 consumes it unchanged, driving it with a
// different `o_indptr` and nothing else. It knows nothing about pages, block
// tables, or GQA; it merges `(V, LSE)` states. See `paged_attention_v2.h` for
// the full contract, including the log2 LSE units.

#include "paged_attention_v2.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/metal/metal.h>

#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Body of the merge kernel (Metal).
//
// One thread owns one `(output row, head, dim)` element and walks that row's
// partial list once, keeping a running `(max, denominator, weighted sum)`. The
// LSE values are re-read per thread rather than staged in threadgroup memory:
// they are `H` floats per partial against `H * D` for the values, so the extra
// traffic is 1/D of the V traffic and lands in cache, and in exchange the
// kernel needs no threadgroup memory and no barrier, which is what keeps it
// usable for an arbitrary grouping.
//
// Grid `(Dim, H, M)` over threadgroup `(Dim, 1, 1)`: one threadgroup per
// `(output row, head)`.
//
// Buffers:
//   v_in     [N, H, Dim]  f32   partials, each already softmax-normalized
//   lse_in   [N, H]       f32   matching LSE, log2 units
//   o_indptr [M + 1]      i32   output row o merges [o_indptr[o], o_indptr[o+1])
//   out_v    [M, H, Dim]  f32
//   out_lse  [M, H]       f32
constexpr const char* PAGED_ATTENTION_MERGE_SOURCE = R"(
    uint d = thread_position_in_threadgroup.x;       // 0 .. Dim-1
    uint h = threadgroup_position_in_grid.y;         // head
    uint o = threadgroup_position_in_grid.z;         // output row

    uint dim = (uint)Dim;
    uint heads = (uint)v_in_shape[1];
    if (d >= dim || h >= heads) {
        return;
    }

    uint begin = (uint)o_indptr[o];
    uint end = (uint)o_indptr[o + 1];

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;
    for (uint i = begin; i < end; i++) {
        float s = lse_in[i * heads + h];
        // Skips -inf (an empty chunk) and any NaN, so an empty partial can
        // never poison the rescale with (-inf) - (-inf).
        if (!(s > -INFINITY)) {
            continue;
        }
        float m_new = fmax(m, s);
        float corr = exp2(m - m_new);
        float w = exp2(s - m_new);
        l = l * corr + w;
        acc = acc * corr + w * v_in[(i * heads + h) * dim + d];
        m = m_new;
    }

    out_v[(o * heads + h) * dim + d] = l > 0.0f ? (acc / l) : 0.0f;
    if (d == 0u) {
        out_lse[o * heads + h] = l > 0.0f ? (m + log2(l)) : -INFINITY;
    }
)";

// CUDA port of the merge kernel.
//
// **Unvalidated**: no CUDA hardware and no `nvcc` were available for issue
// #898. Structurally identical to the Metal body; `exp2`/`log2`/`fmax` become
// `exp2f`/`log2f`/`fmaxf` and the thread indices come from `threadIdx` /
// `blockIdx`. MLX ceil-divides the total-thread grid by the threadgroup tuple,
// so grid `(Dim, H, M)` over threadgroup `(Dim, 1, 1)` yields blocks
// `(1, H, M)` with `blockDim = (Dim, 1, 1)`. The kernel has no barrier, so the
// early `return` on the guard is safe.
constexpr const char* PAGED_ATTENTION_MERGE_CUDA_SOURCE = R"(
    uint32_t d = threadIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t o = blockIdx.z;

    uint32_t dim = (uint32_t)Dim;
    uint32_t heads = (uint32_t)v_in_shape[1];
    if (d >= dim || h >= heads) {
        return;
    }

    uint32_t begin = (uint32_t)o_indptr[o];
    uint32_t end = (uint32_t)o_indptr[o + 1];

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;
    for (uint32_t i = begin; i < end; i++) {
        float s = lse_in[i * heads + h];
        if (!(s > -INFINITY)) {
            continue;
        }
        float m_new = fmaxf(m, s);
        float corr = exp2f(m - m_new);
        float w = exp2f(s - m_new);
        l = l * corr + w;
        acc = acc * corr + w * v_in[(i * heads + h) * dim + d];
        m = m_new;
    }

    out_v[(o * heads + h) * dim + d] = l > 0.0f ? (acc / l) : 0.0f;
    if (d == 0u) {
        out_lse[o * heads + h] = l > 0.0f ? (m + log2f(l)) : -INFINITY;
    }
)";

const std::vector<std::string>& merge_input_names() {
    static const std::vector<std::string> names = {"v_in", "lse_in", "o_indptr"};
    return names;
}

const std::vector<std::string>& merge_output_names() {
    static const std::vector<std::string> names = {"out_v", "out_lse"};
    return names;
}

struct PagedMergeHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;
    bool cuda;

    explicit PagedMergeHolder(bool use_cuda) : cuda(use_cuda) {}

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = cuda
                ? mlx::core::fast::cuda_kernel(
                      "mlxcel_paged_attention_merge_states",
                      merge_input_names(),
                      merge_output_names(),
                      std::string(PAGED_ATTENTION_MERGE_CUDA_SOURCE))
                : mlx::core::fast::metal_kernel(
                      "mlxcel_paged_attention_merge_states",
                      merge_input_names(),
                      merge_output_names(),
                      std::string(PAGED_ATTENTION_MERGE_SOURCE));
        });
        return *kernel;
    }
};

inline PagedMergeHolder& get_merge_kernel(bool cuda) {
    static PagedMergeHolder metal_holder(false);
    static PagedMergeHolder cuda_holder(true);
    return cuda ? cuda_holder : metal_holder;
}

} // namespace

std::vector<mlx::core::array> paged_attention_merge_states(
    const mlx::core::array& v_in,
    const mlx::core::array& lse_in,
    const mlx::core::array& o_indptr) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    const auto& v_shape = v_in.shape(); // [N, H, Dim]
    int heads = v_shape[1];
    int dim = v_shape[2];
    int num_outputs = static_cast<int>(o_indptr.size()) - 1;
    if (num_outputs < 0) {
        num_outputs = 0;
    }

    const bool use_cuda = !mlx::core::metal::is_available();
    auto& kernel = get_merge_kernel(use_cuda).get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"Dim", dim},
    };

    std::vector<mlx::core::array> inputs = {v_in, lse_in, o_indptr};
    std::vector<Shape> output_shapes = {
        Shape{num_outputs, heads, dim},
        Shape{num_outputs, heads},
    };
    std::vector<Dtype> output_dtypes = {mlx::core::float32, mlx::core::float32};

    return kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(dim, heads, num_outputs), // grid
        std::make_tuple(dim, 1, 1),               // threadgroup
        template_args,
        std::nullopt,
        false,
        {});
}

} // namespace mlxcel::turbo
