// Copyright 2025 mlx-lm-rs authors
// Shared internals for the mlx_cxx bridge translation units.
//
// The cxx bridge implementation is split across several .cpp files
// (mlx_cxx_bridge.cpp, mlx_cxx_kernels.cpp, mlx_cxx_nemotron.cpp,
// mlx_cxx_ext.cpp). They all live in `namespace mlx_cxx`, share the common MLX
// / std includes below, and reuse a handful of dtype/shape helpers. Free
// functions declared in the cxx bridge (src/lib.rs) are visible across these
// TUs through the generated header, so only these file-local helpers need to be
// shared here; everything else stays private to its owning .cpp.
#pragma once

#include "mlx_cxx_bridge.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/compile.h"
#include "mlx/memory.h"
#include "mlx/linalg.h"
#include "mlx/fft.h"
#include "mlx/graph_utils.h"
#include "mlx/random.h"
#include "mlx/einsum.h"
#include "mlx/utils.h"
#ifdef __APPLE__
#include "mlx/backend/metal/metal.h"
#endif
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace mlx_cxx {

using namespace mlx::core;

// Helper to convert rust slice to std::vector<int32_t> (Shape)
inline Shape to_shape(rust::Slice<const int32_t> slice) {
    return Shape(slice.begin(), slice.end());
}

// Helper to convert Dtype int to mlx::core::Dtype
inline Dtype to_dtype(int32_t dtype) {
    switch (dtype) {
        case 0: return bool_;
        case 1: return uint8;
        case 2: return uint16;
        case 3: return uint32;
        case 4: return uint64;
        case 5: return int8;
        case 6: return int16;
        case 7: return int32;
        case 8: return int64;
        case 9: return float16;
        case 10: return float32;
        case 11: return float64;
        case 12: return bfloat16;
        case 13: return complex64;
        default: return float32;
    }
}

// Helper to convert mlx::core::Dtype to int
inline int32_t from_dtype(Dtype dtype) {
    switch (dtype.val()) {
        case bool_.val(): return 0;
        case uint8.val(): return 1;
        case uint16.val(): return 2;
        case uint32.val(): return 3;
        case uint64.val(): return 4;
        case int8.val(): return 5;
        case int16.val(): return 6;
        case int32.val(): return 7;
        case int64.val(): return 8;
        case float16.val(): return 9;
        case float32.val(): return 10;
        case float64.val(): return 11;
        case bfloat16.val(): return 12;
        case complex64.val(): return 13;
        default: return 10;  // float32
    }
}

// The dtype attention over q, k and v runs in.
//
// `fast::scaled_dot_product_attention` casts q, k and v to
// `result_type(q, k, v)` before it picks a kernel, so this is the dtype the
// fused-kernel eligibility checks and any materialized score matrix see, and
// the dtype of its output. It is read from the backend's promotion table, so
// the CUDA overlay's patched table decides on CUDA builds. A bf16 query over
// fp16 K/V gives float32 on every backend: a bf16 model over an fp16
// TurboQuant KV cache attends in float32 (issues #2336, #2422). The fold is
// pairwise and left to right, so on CUDA, where float32 + bf16 is patched to
// bf16, an fp16 query over bf16 K/V gives bf16. The softcap composite returns
// this dtype too, so both attention paths agree on it.
inline Dtype sdpa_attention_dtype_for_inputs(
    const array& q, const array& k, const array& v) {
    return mlx::core::result_type(q, k, v);
}

// Coerce an SDPA array mask (or attention sinks, which follow the same rule)
// to the dtype `fast::scaled_dot_product_attention` requires.
//
// SDPA attends in `result_type(q, k, v)` and rejects an array mask whose dtype
// does not promote into that type. Bool and integer masks keep their select
// semantics and pass through unchanged. An additive (floating) mask is cast to
// the attention dtype, not to the query's dtype: the two differ whenever K/V
// are held in another precision than the activations, which is every bf16
// model on an fp16 TurboQuant KV cache. bf16 + fp16 promotes to float32, and
// the CUDA overlay's patched table keeps bf16 + float32 at bf16, so a mask
// cast to the bf16 query does not promote into the float32 output and SDPA
// throws (issue #2336). SDPA casts an accepted mask to the attention dtype
// itself, so when q, k and v share a dtype this is the cast it always did, and
// otherwise it drops only an extra rounding through the query dtype.
inline array sdpa_mask_for_inputs(
    const array& mask, const array& q, const array& k, const array& v) {
    if (!mlx::core::issubdtype(mask.dtype(), mlx::core::floating)) {
        return mask;
    }
    auto attn_dtype = sdpa_attention_dtype_for_inputs(q, k, v);
    return mask.dtype() == attn_dtype ? mask : mlx::core::astype(mask, attn_dtype);
}

// MLX Python's nn.gelu_approx uses the tanh approximation:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
// Keep this helper shared so exact-GELU models can continue to use the
// erf-based compiled_geglu_activation path.
inline array gelu_tanh_approx(const array& x) {
    auto dtype = x.dtype();
    auto half = array(0.5f, dtype);
    auto one = array(1.0f, dtype);
    auto coeff = array(0.7978845608028654f, dtype);  // sqrt(2/pi)
    auto cubic_coeff = array(0.044715f, dtype);
    auto x2 = mlx::core::multiply(x, x);
    auto x3 = mlx::core::multiply(x2, x);
    auto inner = mlx::core::multiply(
        coeff,
        mlx::core::add(x, mlx::core::multiply(cubic_coeff, x3)));
    auto cdf = mlx::core::multiply(half, mlx::core::add(one, mlx::core::tanh(inner)));
    return mlx::core::multiply(x, cdf);
}

// Affine 1-bit quantization (issue #1370), defined in mlx_cxx_one_bit.cpp.
// MLX has no kernel for `bits == 1`, so every bridge primitive that would hand
// such a triple to `mlx::core::quantized_matmul` / `dequantize` routes here
// instead when `is_one_bit_affine` holds and a `.biases` plane is present.
bool is_one_bit_affine(int32_t bits, rust::Str mode);
array one_bit_dequantize_impl(
    const array& w, const array& scales, const array& biases, int group_size);
// `transpose == true` is `x @ W^T` (the linear layout); `false` is `x @ W`.
array one_bit_matmul_impl(
    const array& x, const array& w, const array& scales, const array& biases,
    int group_size, bool transpose, bool force_fallback);
array one_bit_embedding_impl(
    const array& w, const array& scales, const array& biases,
    const array& indices, int group_size);

}  // namespace mlx_cxx
