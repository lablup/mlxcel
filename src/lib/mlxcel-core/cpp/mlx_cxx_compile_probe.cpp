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

// Test-only compiled-vs-eager probe for fused kernels (issues #1792, #2333).
//
// mlxcel used to overlay MLX's Metal compiled-kernel generator
// (`mlx/backend/metal/compiled.cpp`) with a cast of every op input whose type
// differed from the op's output type. That cast turned comparison operands
// into `bool` (#1792: a compiled `less(0.2, 0.7)` evaluated `false`) and the
// complex64 input of `Imag` into `float` (#2333: a compiled `imag()` returned
// 0). The overlay is retired in #2333; MLX ops promote their own inputs, so
// upstream emission hands every fused op the operand types it expects. No
// production compile site contains these ops and the bridge exposes no
// generic compile call, so this entry runs a fixed set of small graphs
// eagerly, compiled, or compiled shapeless, and the Rust test compares the
// results.

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

// z = a + i*b as complex64. Built from two float inputs because an `astype`
// to complex64 alone has a zero imaginary part, under which a broken `imag()`
// returning 0 would still match eager.
mx::array complex_of(const mx::array& a, const mx::array& b) {
    return mx::add(
        mx::astype(a, mx::complex64),
        mx::multiply(
            mx::astype(b, mx::complex64),
            mx::array(mx::complex64_t{0.0f, 1.0f})));
}

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
        // Graphs 7-10 read a complex64 z = a + i*b. Graphs 7 and 8 build z
        // inside the compiled function, so the fused kernel computes it;
        // graphs 9 and 10 receive z as a complex64 input built eagerly by
        // `probe_inputs` (there `a` is z and `b` is the original `a`). Each
        // feeds real(z) or imag(z) into a multiply and an add so `Real` /
        // `Imag` lands inside a fused kernel. Doubling is exact, so a fused
        // multiply-add rounds like the eager pair.
        case 7:  // real(z) * 2 + a, z built in the graph.
            return mx::add(
                mx::multiply(mx::real(complex_of(a, b)), mx::array(2.0f)), a);
        case 8:  // imag(z) * 2 + a, z built in the graph.
            return mx::add(
                mx::multiply(mx::imag(complex_of(a, b)), mx::array(2.0f)), a);
        case 9:  // real(z) * 2 + a, z a complex64 input.
            return mx::add(mx::multiply(mx::real(a), mx::array(2.0f)), b);
        case 10:  // imag(z) * 2 + a, z a complex64 input.
            return mx::add(mx::multiply(mx::imag(a), mx::array(2.0f)), b);
        case 11:  // abs(z) * 2 + a, z built in the graph. MLX's `Abs` node
                  // keeps the complex64 dtype and an `AsType` to float32
                  // follows, so the fused kernel holds |z| in a complex64_t.
            return mx::add(
                mx::multiply(mx::abs(complex_of(a, b)), mx::array(2.0f)), a);
        // Graphs 12 and 13 cast between complex64 and bool. Both receive
        // their source as an eagerly built input (see `probe_inputs`), so the
        // eager mode runs a standalone `astype`, the eager cast kernel, and
        // the compiled modes run the same cast in a fused kernel.
        case 12:  // astype(z, bool), z a complex64 input.
            return mx::astype(a, mx::bool_);
        case 13: {  // real(c) + 10 * imag(c), c = astype(m, complex64) from
                    // a bool input m. One float packs both parts: true gives
                    // 1 for 1+0i and 11 for 1+1i.
            auto c = mx::astype(a, mx::complex64);
            return mx::add(
                mx::real(c), mx::multiply(mx::imag(c), mx::array(10.0f)));
        }
        default:
            throw std::invalid_argument(
                "compile_dtype_probe_for_test: unknown graph " +
                std::to_string(graph));
    }
}

// The inputs graph `graph` receives: (a, b) as given, except that graphs 9
// and 10 receive (z, a) with z = a + i*b evaluated eagerly up front, so the
// compiled function sees a genuine complex64 input in every mode.
std::vector<mx::array> probe_inputs(
    int32_t graph, const mx::array& a, const mx::array& b) {
    if (graph == 9 || graph == 10) {
        auto z = complex_of(a, b);
        mx::eval(z);
        return {z, a};
    }
    if (graph == 12) {
        auto z = complex_of(a, b);
        mx::eval(z);
        return {z, a};
    }
    if (graph == 13) {  // m = (a != 0) as bool, evaluated eagerly.
        auto m = mx::not_equal(a, mx::array(0.0f, a.dtype()));
        mx::eval(m);
        return {m, a};
    }
    return {a, b};
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
    auto out = fn(probe_inputs(graph, a.inner, b.inner));
    mx::eval(out[0]);
    return std::make_unique<MlxArray>(std::move(out[0]));
}

}  // namespace mlx_cxx
