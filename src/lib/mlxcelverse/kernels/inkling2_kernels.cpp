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

// Inkling 2 Mini short convolution and attention head preparation kernels
// (issue #3359). See `inkling2_kernels.h` for the contracts.
//
// Derived from mlx-vlm `mlx_vlm/models/inkling2/kernels.py` (MIT, Copyright
// 2026 Neywa Labs Inc, Abdur Rahim), commits 02053f5c and de08fdf8. The head
// preparation body is adapted there from mlx-dspark's `fused_head_prep` (MIT,
// Copyright 2026 erahim3).
//
// The convolution differs from upstream in its second output. Upstream emits
// the `W - 1 + L` row window and keeps its last `W - 1` rows as the state;
// mlxcel keeps up to `margin` older rows so a rejected tail can be trimmed
// exactly (`CONV_HISTORY_MARGIN` in `src/models/inkling2/conv.rs`), so the
// kernel writes that history directly: the newest `min(W - 1 + margin, R + L)`
// rows of the old history and the inputs, all f32. The scales also stay f32
// here, as the model keeps them, where upstream reads its bf16 parameters.

#include "inkling2_kernels.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "kernel_port.h"

#include <cmath>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace mlxcel::turbo {
namespace {

// One thread per output element (b, t, c). Taps at negative positions read the
// history's newest rows; the rest read `x` through its strides, so a strided
// `x` is never copied. The convolution, its skip and the scale are f32, then
// rounded once to `T` before the optional residual, as the ops path does.
constexpr const char* SCONV_HEAD = R"(
    uint i = thread_position_in_grid.x;
    int B = x_shape[0], L = x_shape[1], C = x_shape[2];
    if (i >= uint(B * L * C)) return;
    int c = i % C, t = (i / C) % L, b = i / (C * L);
    int R = history_shape[1];
    float value = 0;
    for (int j = 0; j < W; ++j) {
        int p = t + j - W + 1;
        float xv = p < 0
            ? float(history[b * history_strides[0] + (R + p) * history_strides[1]
                            + c * history_strides[2]])
            : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]);
        float w = weight[c * weight_strides[0] + j * weight_strides[1]];
        value = value + xv * w;
    }
    float input = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
    float result = float(T((value + input) / divisor[0]));
)";

constexpr const char* SCONV_RESIDUAL = R"(
    result += float(residual[b * residual_strides[0] + t * residual_strides[1]
                             + c * residual_strides[2]]);
)";

constexpr const char* SCONV_TAIL = R"(
    out[i] = OT(result);
    int keep = min(W - 1 + MARGIN, R + L);
    size_t base = size_t(b) * keep * C + c;
    next[base + size_t(keep - L + t) * C] = input;
    if (t == 0) {
        for (int row = 0; row < keep - L; ++row) {
            next[base + size_t(row) * C] =
                history[b * history_strides[0] + (R + L - keep + row) * history_strides[1]
                        + c * history_strides[2]];
        }
    }
)";

// One SIMD group per (row, head slot). Slots are the H query heads, the KV key
// heads, the KV value heads and the H relative heads. The 128-wide reduction
// is four consecutive squares per lane then `simd_sum`; the 16-wide one is
// sequential. Both are MLX's summation orders, and a NaN norm reaches the
// whole head as `mx.minimum` propagates it.
constexpr const char* HEADS_SOURCE = R"(
    const uint lane = thread_index_in_simdgroup;
    const uint heads = q_shape[2], kv_heads = k_shape[2];
    const uint slot = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    if (slot >= heads + 2 * kv_heads) {
        if (lane >= 16) return;
        const uint head = slot - heads - 2 * kv_heads;
        const uint base = (row * heads + head) * 16;
        float sum = 0, own = 0;
        for (uint i = 0; i < 16; ++i) {
            float x = float(r[base + i]) * float(r_scale[head * 16 + i]);
            const float square = x * x;
            sum += square;
            if (i == lane) own = x;
        }
        float factor = params[1] / (metal::precise::sqrt(sum) + 1e-6f);
        ro[base + lane] = bfloat16_t(own * (factor > 1.0f ? 1.0f : factor));
        return;
    }
    const bool query = slot < heads;
    const bool value = slot >= heads + kv_heads;
    const uint head = query ? slot : slot - heads - (value ? kv_heads : 0);
    const uint offset = head * 128 + lane * 4;
    const uint base = row * (query ? heads : kv_heads) * 128 + offset;
    float xs[4], sum = 0;
    for (uint i = 0; i < 4; ++i) {
        xs[i] = float(query ? q[base + i] : (value ? v[base + i] : k[base + i]));
        if (query) xs[i] *= float(q_scale[offset + i]);
        const float square = xs[i] * xs[i];
        sum += square;
    }
    sum = simd_sum(sum);
    float factor;
    if (value) {
        factor = metal::precise::rsqrt(sum / 128.0f + params[2]);
    } else {
        float ratio = params[0] / (metal::precise::sqrt(sum) + 1e-6f);
        factor = ratio > 1.0f ? 1.0f : ratio;
    }
    for (uint i = 0; i < 4; ++i) {
        bfloat16_t y = bfloat16_t(xs[i] * factor);
        if (query) qo[base + i] = y;
        else if (value) vo[base + i] = y;
        else ko[base + i] = y;
    }
)";

struct Inkling2Kernels {
    std::optional<mlx::core::fast::CustomKernelFunction> sconv;
    std::optional<mlx::core::fast::CustomKernelFunction> sconv_residual;
    std::optional<mlx::core::fast::CustomKernelFunction> heads;
    std::once_flag once;

    void init() {
        std::call_once(once, [this] {
            using mlx::core::fast::metal_kernel;
            // Not row-contiguous: `x` and the history are read through their
            // strides, so a sliced history or a strided input is not copied.
            sconv = metal_kernel(
                "mlxcel_inkling2_short_convolution",
                {"history", "weight", "x", "divisor"},
                {"out", "next"},
                std::string(SCONV_HEAD) + SCONV_TAIL,
                "",
                /* ensure_row_contiguous = */ false);
            sconv_residual = metal_kernel(
                "mlxcel_inkling2_short_convolution_residual",
                {"history", "weight", "x", "divisor", "residual"},
                {"out", "next"},
                std::string(SCONV_HEAD) + SCONV_RESIDUAL + SCONV_TAIL,
                "",
                /* ensure_row_contiguous = */ false);
            // Row-contiguous: the body indexes the heads densely.
            heads = metal_kernel(
                "mlxcel_inkling2_prepare_attention_heads",
                {"q", "k", "v", "r", "q_scale", "r_scale", "params"},
                {"qo", "ko", "vo", "ro"},
                HEADS_SOURCE);
        });
    }
};

Inkling2Kernels& kernels() {
    static Inkling2Kernels holder;
    return holder;
}

// No CUDA or ROCm port: the ops path runs there.
const mlxcel::KernelPorts& sconv_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            kernels().init();
            return *kernels().sconv;
        },
        .cuda = nullptr,
        .rocm = nullptr,
    };
    return ports;
}

const mlxcel::KernelPorts& sconv_residual_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            kernels().init();
            return *kernels().sconv_residual;
        },
        .cuda = nullptr,
        .rocm = nullptr,
    };
    return ports;
}

const mlxcel::KernelPorts& heads_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            kernels().init();
            return *kernels().heads;
        },
        .cuda = nullptr,
        .rocm = nullptr,
    };
    return ports;
}

[[noreturn]] void fail(const char* entry, const std::string& what) {
    throw std::invalid_argument(std::string("[") + entry + "] " + what);
}

bool is_half(mlx::core::Dtype dt) {
    return dt == mlx::core::bfloat16 || dt == mlx::core::float16;
}

bool is_float(mlx::core::Dtype dt) {
    return is_half(dt) || dt == mlx::core::float32;
}

} // namespace

bool inkling2_kernels_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu &&
        mlxcel::has_kernel_port(sconv_ports());
}

Inkling2ShortConvolution inkling2_short_convolution(
    const mlx::core::array& history,
    const mlx::core::array& weight,
    const mlx::core::array& x,
    const std::optional<mlx::core::array>& residual,
    mlx::core::Dtype output_dtype,
    int margin) {
    using namespace mlx::core;
    constexpr const char* entry = "inkling2_short_convolution";
    auto& kernel = mlxcel::select_kernel_port(
        entry,
        "Inkling 2 conv1d fallback",
        residual ? sconv_residual_ports() : sconv_ports());
    if (x.ndim() != 3 || !is_half(x.dtype())) fail(entry, "x must be a [B, L, C] bf16 or f16 array");
    const int B = x.shape(0), L = x.shape(1), C = x.shape(2);
    if (weight.ndim() != 3 || weight.shape(0) != C || weight.shape(2) != 1 ||
        weight.dtype() != float32) {
        fail(entry, "weight must be an f32 [C, width, 1] array");
    }
    const int W = weight.shape(1);
    if (history.ndim() != 3 || history.shape(0) != B || history.shape(2) != C ||
        history.shape(1) < W - 1 || history.dtype() != float32) {
        fail(entry, "history must be an f32 [B, R >= width - 1, C] array");
    }
    if (residual && (residual->shape() != x.shape() || !is_float(residual->dtype()))) {
        fail(entry, "residual must be a float array shaped like x");
    }
    if (!is_float(output_dtype)) fail(entry, "the output dtype must be a float dtype");
    if (margin < 0) fail(entry, "margin must not be negative");
    const int R = history.shape(1);
    const int keep = std::min(W - 1 + margin, R + L);
    if (keep < L) fail(entry, "the kept history must hold every input row");

    // `sqrt(width + 1)` rounded once to f32, as the ops path's divisor.
    std::vector<array> inputs = {
        history, weight, x, array({std::sqrt(static_cast<float>(W + 1))}, Shape{1})};
    if (residual) inputs.push_back(*residual);
    const int total = B * L * C;
    auto outputs = kernel(
        inputs,
        {x.shape(), Shape{B, keep, C}},
        {output_dtype, float32},
        std::make_tuple(total, 1, 1),
        std::make_tuple(256, 1, 1),
        {{"T", x.dtype()}, {"OT", output_dtype}, {"W", W}, {"MARGIN", margin}},
        std::nullopt,
        false,
        {});
    return {std::move(outputs[0]), std::move(outputs[1])};
}

Inkling2AttentionHeads inkling2_prepare_attention_heads(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& r,
    const mlx::core::array& q_scale,
    const mlx::core::array& r_scale,
    float clip,
    float rel_clip,
    float eps) {
    using namespace mlx::core;
    constexpr const char* entry = "inkling2_prepare_attention_heads";
    auto& kernel =
        mlxcel::select_kernel_port(entry, "Inkling 2 l2_clip / rms_norm fallback", heads_ports());
    if (q.ndim() != 4 || q.shape(0) != 1 || q.shape(1) < 1 || q.shape(1) > 8 || q.shape(3) != 128) {
        fail(entry, "q must be [1, 1..=8, H, 128]");
    }
    const int L = q.shape(1), H = q.shape(2);
    if (k.ndim() != 4 || k.shape(0) != 1 || k.shape(1) != L || k.shape(3) != 128) {
        fail(entry, "k must be [1, L, KV, 128]");
    }
    const int KV = k.shape(2);
    if (v.shape() != k.shape()) fail(entry, "v must be shaped like k");
    if (r.shape() != Shape{1, L, H, 16}) fail(entry, "r must be [1, L, H, 16]");
    if (q_scale.shape() != Shape{H, 128} || r_scale.shape() != Shape{H, 16}) {
        fail(entry, "q_scale must be [H, 128] and r_scale [H, 16]");
    }
    for (const auto* a : {&q, &k, &v, &r}) {
        if (a->dtype() != bfloat16) fail(entry, "q, k, v and r must be bf16");
    }
    if (q_scale.dtype() != float32 || r_scale.dtype() != float32) {
        fail(entry, "q_scale and r_scale must be f32");
    }
    auto outputs = kernel(
        {q, k, v, r, q_scale, r_scale, array({clip, rel_clip, eps}, Shape{3})},
        {q.shape(), k.shape(), v.shape(), r.shape()},
        {bfloat16, bfloat16, bfloat16, bfloat16},
        std::make_tuple(32, 2 * (H + KV), L),
        std::make_tuple(32, 1, 1),
        {},
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs[0]), std::move(outputs[1]), std::move(outputs[2]),
        std::move(outputs[3])};
}

} // namespace mlxcel::turbo
