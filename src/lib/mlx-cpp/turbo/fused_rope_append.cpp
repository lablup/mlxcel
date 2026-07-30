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

#include "fused_rope_append.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/metal/metal.h>

#include <cmath>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Body of the fused RoPE + append-layout Metal kernel. The string is the kernel
// BODY only; `mlx::core::fast::metal_kernel` wraps it with the declaration,
// buffer arguments, and template-argument substitution.
//
// Thread mapping. One thread owns one *pair* of trailing-dimension elements of
// one (batch, token, head) slot, which is the natural granularity because the
// non-traditional rotation couples element `p` with element `p + RopeDims/2`
// and the traditional rotation couples `2p` with `2p + 1`. Both are two
// elements, so the same thread count covers either convention and the untouched
// `[RopeDims, HeadDim)` tail as well.
//
//   grid.x - HeadDim / 2 element pairs
//   grid.y - token index within the window
//   grid.z - B * (Hq + 2 * Hkv) slots: q heads, then k heads, then v heads
//
// Reading the fused-QKV projection output whole (rather than three
// `slice_last_dim` views) is what keeps this a single dispatch: a trailing-axis
// slice is not row-contiguous, and `ensure_row_contiguous` would insert exactly
// the materializing copies this kernel exists to remove. The q/k/v split is
// therefore a column offset computed here (`col`), not an MLX op.
//
// The rotation math is transcribed from MLX's own `rope.metal` so the fused and
// unfused paths produce the same numbers: `inv_freq = exp2(-d * log2(base))`
// with `d = p / (RopeDims/2)`, `theta = (scale * position) * inv_freq`, and
// `metal::fast::cos` / `metal::fast::sin` (not the `precise` variants, which
// MLX does not use here either).
//
// V is not rotated. It rides through the same dispatch purely to get the
// relayout for free, which removes its own reshape/transpose/contiguous chain.
//
// Template constants:
//   T          - activation dtype (also the dtype of all three outputs).
//   HeadDim    - D.
//   RopeDims   - rotated prefix of each head; the rest is copied through.
//   NHeadsQ    - Hq.
//   NHeadsKV   - Hkv.
//   Traditional- interleaved (true) vs half-split (false) rotation.
//   DestLayout - 0: dense KVCache slab order [B, Hkv, L, D]
//                1: paged pool row order     [B, L, Hkv, D]
//
// Buffers (order matches the launcher's input vector):
//   qkv             [B, L, (Hq + 2*Hkv) * D]  T
//   rope_params     [2]                       f32  {log2(base), scale}
//   positions_base  [1]                       i32
//   q_out           [B, Hq, L, D]             T    (output)
//   k_out           layout-dependent          T    (output)
//   v_out           layout-dependent          T    (output)
constexpr const char* FUSED_ROPE_APPEND_SOURCE = R"(
    uint p = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint z = thread_position_in_grid.z;

    const uint dim = (uint)HeadDim;
    const uint half_dim = dim / 2u;
    const uint rdims = (uint)RopeDims;
    const uint rhalf = rdims / 2u;
    const uint hq = (uint)NHeadsQ;
    const uint hkv = (uint)NHeadsKV;
    const uint batch = (uint)qkv_shape[0];
    const uint seq = (uint)qkv_shape[1];
    const uint slots = hq + 2u * hkv;

    if (p >= half_dim || t >= seq || z >= batch * slots) {
        return;
    }

    uint b = z / slots;
    uint slot = z - b * slots;

    uint kind;   // 0 = q, 1 = k, 2 = v
    uint h;
    uint col;    // column offset of this projection block within the qkv row
    if (slot < hq) {
        kind = 0u;
        h = slot;
        col = 0u;
    } else if (slot < hq + hkv) {
        kind = 1u;
        h = slot - hq;
        col = hq * dim;
    } else {
        kind = 2u;
        h = slot - hq - hkv;
        col = (hq + hkv) * dim;
    }

    ulong in_base = ((ulong)b * seq + t) * (ulong)(slots * dim)
        + (ulong)col + (ulong)h * dim;

    ulong out_base;
    if (kind == 0u) {
        out_base = (((ulong)b * hq + h) * seq + t) * dim;
    } else if (DestLayout == 0) {
        out_base = (((ulong)b * hkv + h) * seq + t) * dim;
    } else {
        out_base = (((ulong)b * seq + t) * hkv + h) * dim;
    }

    if (kind == 2u) {
        uint j = 2u * p;
        v_out[out_base + j] = qkv[in_base + j];
        v_out[out_base + j + 1u] = qkv[in_base + j + 1u];
        return;
    }

    bool rotate = p < rhalf;
    uint i1;
    uint i2;
    if (rotate) {
        if (Traditional) {
            i1 = 2u * p;
            i2 = i1 + 1u;
        } else {
            i1 = p;
            i2 = p + rhalf;
        }
    } else {
        // Untouched tail [RopeDims, HeadDim): two contiguous elements per
        // leftover thread. Both counts are even, so this covers it exactly.
        i1 = rdims + 2u * (p - rhalf);
        i2 = i1 + 1u;
    }

    float x1 = (float)qkv[in_base + i1];
    float x2 = (float)qkv[in_base + i2];
    float r1 = x1;
    float r2 = x2;
    if (rotate) {
        float d = (float)p / (float)rhalf;
        float inv_freq = metal::exp2(-d * rope_params[0]);
        float pos = (float)((int)t + positions_base[0]);
        float theta = rope_params[1] * pos * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        r1 = x1 * costheta - x2 * sintheta;
        r2 = x1 * sintheta + x2 * costheta;
    }

    if (kind == 0u) {
        q_out[out_base + i1] = (T)r1;
        q_out[out_base + i2] = (T)r2;
    } else {
        k_out[out_base + i1] = (T)r1;
        k_out[out_base + i2] = (T)r2;
    }
)";

// CUDA port of the fused RoPE + append-layout kernel.
//
// UNVALIDATED. This host is Apple Silicon with no CUDA device and no `nvcc`, so
// this string has never been compiled or executed. It is a structural
// transliteration of the Metal body above, kept deliberately line-for-line
// parallel so a reviewer with a CUDA box can diff the two rather than re-derive
// the addressing. Check it against `fused_rope_append_matches_graph_rope` in
// `src/lib/mlxcel-core/src/fused_norm_parity_tests.rs` on first CUDA run,
// before `MLXCEL_FUSED_ROPE_APPEND` is left on by default there.
//
// The only substantive differences from the Metal body are the thread-index
// expressions (MLX's CUDA custom kernel ceil-divides the Metal-style grid by the
// threadgroup tuple, so the global index has to be reconstructed from
// `blockIdx * blockDim + threadIdx`; the bounds guards at the top already cover
// the padding that introduces) and `exp2f` / `__cosf` / `__sinf` standing in for
// `metal::exp2` / `metal::fast::cos` / `metal::fast::sin`.
constexpr const char* FUSED_ROPE_APPEND_CUDA_SOURCE = R"(
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    const uint32_t dim = (uint32_t)HeadDim;
    const uint32_t half_dim = dim / 2u;
    const uint32_t rdims = (uint32_t)RopeDims;
    const uint32_t rhalf = rdims / 2u;
    const uint32_t hq = (uint32_t)NHeadsQ;
    const uint32_t hkv = (uint32_t)NHeadsKV;
    const uint32_t batch = (uint32_t)qkv_shape[0];
    const uint32_t seq = (uint32_t)qkv_shape[1];
    const uint32_t slots = hq + 2u * hkv;

    if (p >= half_dim || t >= seq || z >= batch * slots) {
        return;
    }

    uint32_t b = z / slots;
    uint32_t slot = z - b * slots;

    uint32_t kind;   // 0 = q, 1 = k, 2 = v
    uint32_t h;
    uint32_t col;
    if (slot < hq) {
        kind = 0u;
        h = slot;
        col = 0u;
    } else if (slot < hq + hkv) {
        kind = 1u;
        h = slot - hq;
        col = hq * dim;
    } else {
        kind = 2u;
        h = slot - hq - hkv;
        col = (hq + hkv) * dim;
    }

    uint64_t in_base = ((uint64_t)b * seq + t) * (uint64_t)(slots * dim)
        + (uint64_t)col + (uint64_t)h * dim;

    uint64_t out_base;
    if (kind == 0u) {
        out_base = (((uint64_t)b * hq + h) * seq + t) * dim;
    } else if (DestLayout == 0) {
        out_base = (((uint64_t)b * hkv + h) * seq + t) * dim;
    } else {
        out_base = (((uint64_t)b * seq + t) * hkv + h) * dim;
    }

    if (kind == 2u) {
        uint32_t j = 2u * p;
        v_out[out_base + j] = qkv[in_base + j];
        v_out[out_base + j + 1u] = qkv[in_base + j + 1u];
        return;
    }

    bool rotate = p < rhalf;
    uint32_t i1;
    uint32_t i2;
    if (rotate) {
        if (Traditional) {
            i1 = 2u * p;
            i2 = i1 + 1u;
        } else {
            i1 = p;
            i2 = p + rhalf;
        }
    } else {
        i1 = rdims + 2u * (p - rhalf);
        i2 = i1 + 1u;
    }

    float x1 = (float)qkv[in_base + i1];
    float x2 = (float)qkv[in_base + i2];
    float r1 = x1;
    float r2 = x2;
    if (rotate) {
        float d = (float)p / (float)rhalf;
        float inv_freq = exp2f(-d * rope_params[0]);
        float pos = (float)((int)t + positions_base[0]);
        float theta = rope_params[1] * pos * inv_freq;
        float costheta = __cosf(theta);
        float sintheta = __sinf(theta);
        r1 = x1 * costheta - x2 * sintheta;
        r2 = x1 * sintheta + x2 * costheta;
    }

    if (kind == 0u) {
        q_out[out_base + i1] = (T)r1;
        q_out[out_base + i2] = (T)r2;
    } else {
        k_out[out_base + i1] = (T)r1;
        k_out[out_base + i2] = (T)r2;
    }
)";

// Threadgroup width along the element-pair axis. 64 covers a head_dim-128 head
// in one group without over-subscribing the smaller head dims, which fall back
// to `half_dim` threads.
constexpr int FUSED_ROPE_TG_X = 64;

const std::vector<std::string>& fused_rope_input_names() {
    static const std::vector<std::string> names = {
        "qkv", "rope_params", "positions_base"};
    return names;
}

const std::vector<std::string>& fused_rope_output_names() {
    static const std::vector<std::string> names = {"q_out", "k_out", "v_out"};
    return names;
}

// Thread-safe lazy-initialised holders for the JIT-compiled kernels. Same
// `std::call_once` contract as `paged_attention.cpp`: first use is reachable
// concurrently from per-request blocking workers, and `call_once` re-runs the
// initializer if MLX device lookup throws.
struct FusedRopeKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_fused_rope_qk_append",
                fused_rope_input_names(),
                fused_rope_output_names(),
                std::string(FUSED_ROPE_APPEND_SOURCE));
        });
        return *kernel;
    }
};

inline FusedRopeKernelHolder& get_fused_rope_kernel() {
    static FusedRopeKernelHolder holder;
    return holder;
}

struct FusedRopeKernelHolderCuda {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::cuda_kernel(
                "mlxcel_fused_rope_qk_append",
                fused_rope_input_names(),
                fused_rope_output_names(),
                std::string(FUSED_ROPE_APPEND_CUDA_SOURCE));
        });
        return *kernel;
    }
};

inline FusedRopeKernelHolderCuda& get_fused_rope_kernel_cuda() {
    static FusedRopeKernelHolderCuda holder;
    return holder;
}

} // namespace

bool fused_rope_qk_append_available() {
    return mlx::core::metal::is_available() || mlx::core::cu::is_available();
}

std::vector<mlx::core::array> fused_rope_qk_append(
    const mlx::core::array& qkv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rope_dims,
    float rope_base,
    float rope_scale,
    bool traditional,
    int positions_base,
    int dest_layout) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    if (qkv.ndim() != 3) {
        throw std::invalid_argument(
            "[fused_rope_qk_append] qkv must be [B, L, (Hq + 2*Hkv) * D].");
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument(
            "[fused_rope_qk_append] head counts and head_dim must be positive.");
    }
    if (head_dim % 2 != 0) {
        throw std::invalid_argument(
            "[fused_rope_qk_append] head_dim must be even.");
    }
    if (rope_dims <= 0 || rope_dims % 2 != 0 || rope_dims > head_dim) {
        throw std::invalid_argument(
            "[fused_rope_qk_append] rope_dims must be even, positive and <= head_dim.");
    }
    if (dest_layout != 0 && dest_layout != 1) {
        throw std::invalid_argument(
            "[fused_rope_qk_append] dest_layout must be 0 (dense slab) or 1 (paged pool).");
    }
    const int batch = qkv.shape()[0];
    const int seq = qkv.shape()[1];
    const int expected = (num_heads + 2 * num_kv_heads) * head_dim;
    if (qkv.shape()[2] != expected) {
        throw std::invalid_argument(
            "[fused_rope_qk_append] qkv trailing dim does not match the head geometry.");
    }

    const bool use_cuda = !mlx::core::metal::is_available();
    auto& kernel = use_cuda ? get_fused_rope_kernel_cuda().get()
                            : get_fused_rope_kernel().get();

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"T", qkv.dtype()},
        {"HeadDim", head_dim},
        {"RopeDims", rope_dims},
        {"NHeadsQ", num_heads},
        {"NHeadsKV", num_kv_heads},
        {"Traditional", traditional},
        {"DestLayout", dest_layout},
    };

    // MLX's own RoPE precomputes `log2(base)` on the host and reconstructs the
    // frequency with `exp2` in the kernel; matching that (rather than a `pow`)
    // is what keeps the fused rotation numerically identical to `fast::rope`.
    // `array({...})` sets the data directly, so neither of these small
    // parameter arrays costs a dispatch.
    auto rope_params = mlx::core::array(
        std::initializer_list<float>{std::log2(rope_base), rope_scale},
        mlx::core::float32);
    auto pos_arr = mlx::core::array(
        std::initializer_list<int>{positions_base}, mlx::core::int32);

    std::vector<mlx::core::array> inputs = {qkv, rope_params, pos_arr};

    Shape q_shape{batch, num_heads, seq, head_dim};
    Shape kv_shape = dest_layout == 0
        ? Shape{batch, num_kv_heads, seq, head_dim}
        : Shape{batch, seq, num_kv_heads, head_dim};
    std::vector<Shape> output_shapes = {q_shape, kv_shape, kv_shape};
    std::vector<Dtype> output_dtypes = {qkv.dtype(), qkv.dtype(), qkv.dtype()};

    const int half_dim = head_dim / 2;
    const int tg_x = half_dim < FUSED_ROPE_TG_X ? half_dim : FUSED_ROPE_TG_X;
    const int slots = num_heads + 2 * num_kv_heads;

    return kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(half_dim, seq, batch * slots), // grid
        std::make_tuple(tg_x, 1, 1),                   // threadgroup
        template_args,
        std::nullopt,
        false,
        {});
}

} // namespace mlxcel::turbo
