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

#include "gelu_tanh_f32.h"

#include <mlx/fast.h>

#include "kernel_port.h"
#include <mlx/ops.h>

#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Fused tanh-GELU custom Metal kernel launcher, f32 arithmetic.
//
// The string is the kernel BODY only; `mlx::core::fast::metal_kernel` wraps it
// with the declaration, buffer arguments, and template-argument substitution.
//
// The kernel is 1-D elementwise: one thread per element. The element type is
// upcast to f32, the tanh-GELU is computed in f32, and the result is rounded
// back to the output dtype at the store:
//
//     x2 = x * x
//     x3 = x2 * x
//     inner = 0.7978845608028654 * (x + 0.044715 * x3)
//     cdf = 0.5 * (1 + tanh(inner))
//     out = x * cdf
//
// Same formula and op order as the eager `gelu_tanh_f32` chain
// (cpp/mlx_cxx_internal.h `gelu_tanh_approx`); the fused kernel may contract
// a [multiply-add] inside the compiled region, so lanes may differ from it by
// f32 rounding.
namespace mlxcel::turbo {

namespace {

constexpr const char* GELU_TANH_F32_SOURCE = R"(
    uint i = thread_position_in_grid.x;
    float xv = (float)x[i];
    float x2 = xv * xv;                                        // square(x)
    float x3 = x2 * xv;                                        // cube(x)
    float inner = 0.7978845608028654f * (xv + 0.044715f * x3); // mul-add
    float cdf = 0.5f * (1.0f + metal::precise::tanh(inner));   // tanh CDF
    out[i] = (T)(xv * cdf);                                    // scale + round
)";

struct GeluTanhF32KernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_fused_gelu_tanh_f32",
                {"x"},
                {"out"},
                std::string(GELU_TANH_F32_SOURCE));
        });
        return *kernel;
    }
};

inline GeluTanhF32KernelHolder& get_gelu_tanh_f32_kernel() {
    static GeluTanhF32KernelHolder holder;
    return holder;
}

// This kernel's ports, in one place (#2479). Metal only: the fused tanh-GELU
// is a Metal JIT kernel. A backend without a port is refused through
// `select_kernel_port`, and the Rust caller takes its eager routing chain
// instead of falling into the Metal arm.
const mlxcel::KernelPorts& gelu_tanh_f32_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_gelu_tanh_f32_kernel().get();
        },
        // No CUDA port (#1814): CUDA takes the eager `gelu_tanh_f32` chain.
        // Written out so that adding one is a line here rather than a
        // restructure at the call site.
        .cuda = nullptr,
        // No ROCm port. The eager chain covers ROCm the same way.
        .rocm = nullptr,
    };
    return ports;
}

} // namespace

// Whether the fused kernel runs on this backend rather than refusing. Read
// from the table the launcher reads, and false off the GPU stream: custom
// kernels run only on the GPU, so with MLXCEL_DEVICE=cpu the eager
// `gelu_tanh_f32` chain runs on the CPU instead of the launch throwing (the
// same check as fused_xielu_kernel_available).
bool fused_gelu_tanh_f32_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu &&
        mlxcel::has_kernel_port(gelu_tanh_f32_ports());
}

mlx::core::array fused_gelu_tanh_f32(const mlx::core::array& x) {
    using namespace mlx::core;
    auto T = x.dtype();
    auto xs = x.shape();

    int64_t n = 1;
    for (auto d : xs) n *= d;
    auto xflat = reshape(x, {(int)n});

    auto& kernel = mlxcel::select_kernel_port(
        "fused_gelu_tanh_f32", "eager gelu_tanh_f32 fallback", gelu_tanh_f32_ports());

    // The dtype is the only template argument: the body reads its index from
    // `thread_position_in_grid` and needs no size constant.
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> ta = {
        {"T", T},
    };
    std::vector<array> inputs = {xflat};
    const int tg = 256;
    // Metal dispatches exactly `grid` threads (`dispatchThreads`), with no
    // bounds guard generated around the body, so the grid is the element count
    // itself and not a rounded-up multiple of the threadgroup size. A rounded
    // grid would launch lanes past `n` and read unwritten output.
    const int grid = (int)n;
    auto results = kernel(
        inputs, {Shape{(int)n}}, {T},
        std::make_tuple(grid, 1, 1),   // grid
        std::make_tuple(tg, 1, 1),     // threadgroup
        ta,
        std::nullopt,
        false,
        {});

    return reshape(results[0], xs);
}

} // namespace mlxcel::turbo
