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

#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Fused tanh-GELU custom kernel launcher, f32 arithmetic: a Metal port (#2857)
// and a CUDA port (#2780) of one body, chosen through the port table.
//
// Each string is the kernel BODY only; `mlx::core::fast::metal_kernel` and
// `mlx::core::fast::cuda_kernel` wrap it with the declaration, buffer
// arguments, and template-argument substitution.
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
// (cpp/mlx_cxx_internal.h `gelu_tanh_approx`). The two ports differ in how
// they round it. The Metal body leaves `x + 0.044715 * x3` to the compiler,
// which may contract it into a fused multiply-add, so a Metal lane may differ
// from the eager chain by f32 rounding. The CUDA body rounds every operation
// on its own and returns the eager chain's bits; see its comment below.
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

// CUDA port of the same body (#2780). Three things differ from the Metal
// string.
//
// The index comes from the block and thread ids, and the lanes past `n` return
// before touching memory: MLX's CUDA launcher rounds the grid up to whole
// blocks (`mlx/backend/cuda/custom_kernel.cpp` launches
// `(grid + threadgroup - 1) / threadgroup` blocks of
// `min(threadgroup, grid)` threads) where Metal dispatches exactly `grid`
// threads. `x_shape[0]` is the flattened element count, passed as a launch
// argument rather than baked into the module, so the JIT cache key is the
// dtype alone (`T`): one module, and one PTX disk-cache entry (#2492), per
// dtype rather than per shape. The module name is
// `custom_kernel_mlxcel_fused_gelu_tanh_f32` plus two hashes, well inside the
// NAME_MAX window that once kept the compiled chain's entry from being written
// (#2423).
//
// `tanhf` stands in for `metal::precise::tanh`.
//
// Every multiply and add is written as `__fmul_rn` / `__fadd_rn`, which NVRTC
// never merges into a fused multiply-add. Written with plain operators the
// body compiles `x + 0.044715 * x3` to one FFMA, and then about four f32
// lanes in a thousand sit one rounding step of tanh away from the eager chain
// (1171 of 275456 on the tower-shaped input, GB10), because the eager chain
// rounds the multiply and the add in separate kernels. With each operation
// rounded on its own the kernel does what the eager chain does, in the same
// order, and returns its bits: f32, bf16 and f16, on every lane measured.
// That is what lets `GeluActivation::TanhF32`, whose call sites are pinned
// bit for bit, take this port too (`gelu_tanh_f32_eager_exact_ports`).
constexpr const char* GELU_TANH_F32_CUDA_SOURCE = R"(
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint32_t)x_shape[0]) {
        return;
    }
    float xv = (float)x[i];
    float x2 = __fmul_rn(xv, xv);                               // square(x)
    float x3 = __fmul_rn(x2, xv);                               // cube(x)
    float poly = __fadd_rn(xv, __fmul_rn(0.044715f, x3));       // never an FMA
    float inner = __fmul_rn(0.7978845608028654f, poly);
    float cdf = __fmul_rn(0.5f, __fadd_rn(1.0f, tanhf(inner))); // tanh CDF
    out[i] = (T)__fmul_rn(xv, cdf);                             // scale + round
)";

struct GeluTanhF32KernelHolderCuda {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::cuda_kernel(
                "mlxcel_fused_gelu_tanh_f32",
                {"x"},
                {"out"},
                std::string(GELU_TANH_F32_CUDA_SOURCE));
        });
        return *kernel;
    }
};

inline GeluTanhF32KernelHolderCuda& get_gelu_tanh_f32_kernel_cuda() {
    static GeluTanhF32KernelHolderCuda holder;
    return holder;
}

// This kernel's ports, in one place (#2479): Metal (#2857) and CUDA (#2780),
// each a JIT kernel of the same body. A backend without a port is refused
// through `select_kernel_port`, and the Rust caller takes
// `compiled_gelu_tanh_f32` instead of falling into another backend's arm.
const mlxcel::KernelPorts& gelu_tanh_f32_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_gelu_tanh_f32_kernel().get();
        },
        .cuda = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_gelu_tanh_f32_kernel_cuda().get();
        },
        // No ROCm port. `compiled_gelu_tanh_f32` covers ROCm the same way.
        .rocm = nullptr,
    };
    return ports;
}

// The ports whose body returns the eager chain's bits, as a table of their
// own so that "may `TanhF32` take the kernel here" is answered the way every
// other port question is (`has_kernel_port`) and not by comparing the backend.
// CUDA only: its body rounds each operation on its own (see
// `GELU_TANH_F32_CUDA_SOURCE`). Metal's leaves the multiply-add to the
// compiler and differs from the eager chain by f32 rounding on some lanes
// (#2858), so `TanhF32` keeps the eager helper there.
const mlxcel::KernelPorts& gelu_tanh_f32_eager_exact_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = nullptr,
        .cuda = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_gelu_tanh_f32_kernel_cuda().get();
        },
        .rocm = nullptr,
    };
    return ports;
}

} // namespace

// Whether the fused kernel runs on this backend rather than refusing. Read
// from the table the launcher reads, and false off the GPU stream: custom
// kernels run only on the GPU, so with MLXCEL_DEVICE=cpu
// `compiled_gelu_tanh_f32` runs on the CPU instead of the launch throwing (the
// same check as fused_xielu_kernel_available).
bool fused_gelu_tanh_f32_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu &&
        mlxcel::has_kernel_port(gelu_tanh_f32_ports());
}

// Whether the fused kernel on this backend returns the eager chain's bits, so
// that a caller pinned to the eager chain may launch it instead. The same two
// conditions as above, read from the narrower table.
bool fused_gelu_tanh_f32_matches_eager_bits() {
    return mlx::core::default_device() == mlx::core::Device::gpu &&
        mlxcel::has_kernel_port(gelu_tanh_f32_eager_exact_ports());
}

mlx::core::array fused_gelu_tanh_f32(const mlx::core::array& x) {
    using namespace mlx::core;
    auto T = x.dtype();
    auto xs = x.shape();

    auto& kernel = mlxcel::select_kernel_port(
        "fused_gelu_tanh_f32", "compiled_gelu_tanh_f32 fallback", gelu_tanh_f32_ports());

    int64_t n = 1;
    for (auto d : xs) n *= d;
    // MLX's Metal dispatch clamps the threadgroup to the grid, so an empty
    // input would dispatch with a zero-width threadgroup, and its CUDA
    // dispatch would launch zero blocks. There is nothing to compute: the
    // result is the empty input, same shape and dtype.
    if (n == 0) {
        return x;
    }
    // The flattened shape and the grid are `int`. The Rust dispatch routes a
    // larger array to the compiled chain; refuse here rather than truncate.
    if (n > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "[fused_gelu_tanh_f32] input has more elements than a kernel grid can address");
    }
    auto xflat = reshape(x, {(int)n});

    // The dtype is the only template argument: the Metal body reads its index
    // from `thread_position_in_grid` and the CUDA body its bound from
    // `x_shape[0]`, a launch argument, so neither needs a size constant and
    // one module per dtype serves every shape.
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> ta = {
        {"T", T},
    };
    std::vector<array> inputs = {xflat};
    const int tg = 256;
    // Metal dispatches exactly `grid` threads (`dispatchThreads`), with no
    // bounds guard generated around the body, so the grid is the element count
    // itself and not a rounded-up multiple of the threadgroup size. A rounded
    // grid would launch lanes past `n` and read unwritten output. CUDA rounds
    // the same grid up to whole blocks itself, and its body returns on the
    // lanes past `n`.
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
