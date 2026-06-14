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

}  // namespace mlx_cxx
