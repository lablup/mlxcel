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

// Native sliding-window SDPA bridge for Metal prefill (issue #2621).
//
// Upstream MLX's `fast::scaled_dot_product_attention` takes a `window_size`
// argument (ml-explore/mlx#4734). With `mask_mode = "causal"` and a positive
// window, query row `r` (counted from the end of the key axis, so the queries
// are the suffix of the keys) sees key `c` iff `c <= r` and `r - c < W`: the
// current position plus up to `W - 1` earlier ones. That is exactly the band
// `create_causal_mask_with_window_full(q_len, k_len - q_len, Some(W))` builds
// on the Rust side, so the two entries are interchangeable for a causal
// multi-query prefill whose only mask semantics are "causal plus window".
//
// Every other bridge SDPA entry keeps passing `window_size = -1`; the route
// selection (backend, softcap, sinks, extra masks, query length) lives in
// `src/window_sdpa.rs`, which is the only caller of these two symbols.

#include "mlx_cxx_internal.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"

namespace mlx_cxx {

std::unique_ptr<MlxArray> fast_scaled_dot_product_attention_causal_window(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    int32_t window_size) {
    if (window_size <= 0) {
        throw std::invalid_argument(
            "[fast_scaled_dot_product_attention_causal_window] window_size must be "
            "positive; the unwindowed causal entry is "
            "fast_scaled_dot_product_attention_causal");
    }
    return std::make_unique<MlxArray>(mlx::core::fast::scaled_dot_product_attention(
        q.inner, k.inner, v.inner, scale, window_size, "causal", std::nullopt));
}

// Whether upstream MLX would run a fused attention kernel (rather than its
// graph fallback) for an SDPA over these shapes. This is MLX's own verdict,
// `ScaledDotProductAttention::use_fallback`, asked over lazily built zero
// arrays of the given shapes, so nothing is allocated or evaluated. The three
// inputs are cast to `result_type(q, k, v)` first, because the frontend does
// that before it consults `use_fallback`.
//
// `has_arr_mask` with `do_causal` is the legacy explicit-mask arm; `do_causal`
// with a positive `window_size` is the native window arm. The answer depends
// on the device generation and NAX availability, which is why it is probed
// rather than tabulated in Rust.
bool sdpa_uses_fused_kernel(
    rust::Slice<const int32_t> q_shape,
    rust::Slice<const int32_t> k_shape,
    rust::Slice<const int32_t> v_shape,
    int32_t q_dtype,
    int32_t k_dtype,
    int32_t v_dtype,
    bool has_arr_mask,
    bool do_causal,
    int32_t window_size) {
#ifdef MLXCEL_BRIDGE_METAL_BACKEND
    if (!mlx::core::metal::is_available()) {
        return false;
    }
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        return false;
    }
    auto shape_of = [](rust::Slice<const int32_t> s) {
        return mlx::core::Shape(s.begin(), s.end());
    };
    auto q = mlx::core::zeros(shape_of(q_shape), to_dtype(q_dtype));
    auto k = mlx::core::zeros(shape_of(k_shape), to_dtype(k_dtype));
    auto v = mlx::core::zeros(shape_of(v_shape), to_dtype(v_dtype));
    auto final_type = mlx::core::result_type(q, k, v);
    if (!mlx::core::issubdtype(final_type, mlx::core::floating)) {
        return false;
    }
    q = mlx::core::astype(q, final_type);
    k = mlx::core::astype(k, final_type);
    v = mlx::core::astype(v, final_type);
    const bool has_mask = has_arr_mask || do_causal;
    auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
    return !mlx::core::fast::ScaledDotProductAttention::use_fallback(
        q,
        k,
        v,
        has_mask,
        has_arr_mask,
        do_causal,
        /* has_sinks = */ false,
        window_size,
        /* is_training = */ false,
        /* output_logsumexp = */ false,
        /* force_fused = */ false,
        stream);
#else
    (void)q_shape;
    (void)k_shape;
    (void)v_shape;
    (void)q_dtype;
    (void)k_dtype;
    (void)v_dtype;
    (void)has_arr_mask;
    (void)do_causal;
    (void)window_size;
    return false;
#endif
}

}  // namespace mlx_cxx
