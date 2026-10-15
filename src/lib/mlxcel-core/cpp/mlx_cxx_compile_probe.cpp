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

// Test-only compiled-vs-eager probe for fused kernel input casts (issue #1792).
//
// The Metal overlay of MLX's compiled-kernel generator
// (`mlxcelverse/overlays/metal/mlx/backend/metal/compiled.cpp`) casts an op's
// mismatched inputs so mixed bf16/f32 arithmetic deduces one template type.
// Keyed on the op's output type, that cast turned every comparison operand
// into `bool`, so a compiled `less(0.2, 0.7)` evaluated `false`. No production
// compile site contains a comparison, and the bridge exposes no generic compile
// call, so this entry runs a fixed set of small graphs eagerly, compiled, or
// compiled shapeless, and the Rust test compares the results.

#include "mlx_cxx_internal.h"

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace mlx_cxx {

namespace mx = mlx::core;

namespace {

using GraphFn =
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

// Graph ids mirror `compiled_dtype_cast_tests.rs`.
mx::array compile_probe_graph(
    int32_t graph, const mx::array& a, const mx::array& b) {
    switch (graph) {
        case 0:  // where(a < b, a, b): a comparison feeding a select.
            return mx::where(mx::less(a, b), a, b);
        case 1:  // A comparison as the kernel's bool output.
            return mx::greater(mx::multiply(a, mx::array(2.0f, a.dtype())), b);
        case 2: {  // The six comparisons, packed into one int32 code.
            auto bit = [](const mx::array& m, int32_t w) {
                return mx::multiply(
                    mx::astype(m, mx::int32), mx::array(w, mx::int32));
            };
            auto code = mx::add(
                bit(mx::less(a, b), 1), bit(mx::less_equal(a, b), 2));
            code = mx::add(code, bit(mx::greater(a, b), 4));
            code = mx::add(code, bit(mx::greater_equal(a, b), 8));
            code = mx::add(code, bit(mx::equal(a, b), 16));
            return mx::add(code, bit(mx::not_equal(a, b), 32));
        }
        case 3:  // isnan lowers to not_equal(a, a).
            return mx::where(mx::isnan(a), mx::array(7.0f, a.dtype()), a);
        case 4:  // isinf lowers to equal against +/-inf.
            return mx::where(mx::isinf(a), mx::array(9.0f, a.dtype()), a);
        case 5: {  // The min-p filter #1379 removed from the sampler chain.
            auto probs = mx::softmax(a, -1);
            auto max_prob = mx::max(probs, -1, /*keepdims=*/true);
            auto threshold = mx::multiply(max_prob, mx::array(0.3f));
            return mx::where(
                mx::greater_equal(probs, threshold),
                a,
                mx::array(std::numeric_limits<float>::lowest(), a.dtype()));
        }
        case 6: {  // Mixed-dtype arithmetic, the case the overlay cast is for.
            auto q = mx::divide(a, b);
            auto r = mx::subtract(mx::multiply(q, a), b);
            r = mx::add(r, mx::multiply(a, mx::array(0.5f)));
            return mx::add(mx::astype(r, a.dtype()), a);
        }
        default:
            throw std::invalid_argument(
                "compile_dtype_probe_for_test: unknown graph " +
                std::to_string(graph));
    }
}

}  // namespace

std::unique_ptr<MlxArray> compile_dtype_probe_for_test(
    int32_t graph,
    const MlxArray& a,
    const MlxArray& b,
    int32_t mode
) {
    GraphFn fn = [graph](const std::vector<mx::array>& inputs) {
        return std::vector<mx::array>{
            compile_probe_graph(graph, inputs[0], inputs[1])};
    };
    switch (mode) {
        case 0:
            break;
        case 1:
            fn = mx::compile(fn, /*shapeless=*/false);
            break;
        case 2:
            fn = mx::compile(fn, /*shapeless=*/true);
            break;
        default:
            throw std::invalid_argument(
                "compile_dtype_probe_for_test: mode must be 0, 1 or 2");
    }
    auto out = fn({a.inner, b.inner});
    mx::eval(out[0]);
    return std::make_unique<MlxArray>(std::move(out[0]));
}

}  // namespace mlx_cxx
