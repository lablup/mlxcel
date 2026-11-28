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

#include "fused_norm.h"
#include "fused_norm_cuda.h"
#include "fused_norm_hip.h"
#include "gpu_backend.h"
#include "kernel_port.h"
#ifdef MLXCEL_CUDA_OXIDE_FUSED_NORM
#include "fused_norm_oxide.h"
#endif

#include <mlx/fast.h>
#include <mlx/ops.h>

// Backend availability probes for the Metal-vs-CUDA kernel gate. Both headers
// are backend-agnostic: an Apple build links the CUDA no_cuda stub, a CUDA build
// links the Metal no_metal stub, so `is_available()` always resolves and returns
// false for the absent backend. Same pattern as `paged_attention.cpp`.
#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/metal/metal.h>

#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlxcel::turbo {

namespace {

// Body of the fused residual-add + RMSNorm Metal kernel. The string is the
// kernel BODY only; `mlx::core::fast::metal_kernel` wraps it with the
// declaration, buffer arguments, and template-argument substitution.
//
// One threadgroup owns one row of `Dim` elements. `Threads` lanes sweep the row
// in `N_READS = 4`-wide contiguous chunks, matching MLX's own `rms_single_row`
// access pattern, so the coalescing behavior of the fused and unfused paths is
// the same and a measured difference is attributable to the fusion rather than
// to a worse memory schedule.
//
// Two passes over the row:
//
//   1. Read `x` and `residual`, sum in fp32, round the sum to the activation
//      dtype, store it as `new_residual`, and accumulate its square. Rounding
//      *before* squaring is what makes this agree with the unfused
//      `add -> fast::rms_norm` pair: the unfused RMSNorm reads the dtype-rounded
//      sum out of global memory, so squaring the un-rounded fp32 sum here would
//      introduce a systematic (if tiny) difference that grows with row length.
//   2. Re-read `new_residual` (still hot in cache, it was just written by this
//      same threadgroup) and write `normed`.
//
//   The alternative to the second read is holding the row in registers, which
//   at Dim = 8192 is 8 floats per lane at Threads = 1024 and spills. The re-read
//   is an L2 hit, the register file is not.
//
// The reduction is the standard two stage shape: `simd_sum` inside each SIMD
// group, one scalar per SIMD group into threadgroup memory, then SIMD group 0
// reduces those and publishes `rsqrt(mean + eps)`.
//
// Template constants:
//   T       - activation dtype (also the dtype of both outputs).
//   Dim     - row length D (the normalized axis).
//   Threads - threadgroup width; a multiple of 32, at most 1024.
//
// Buffers (order matches the launcher's input vector):
//   x            [..., Dim]  T
//   residual     [..., Dim]  T
//   weight       [Dim]       (any float dtype)
//   eps          [1]         f32
//   weight_bias  [1]         f32
//   normed       [..., Dim]  T   (output)
//   new_residual [..., Dim]  T   (output)
constexpr const char* FUSED_ADD_RMS_NORM_SOURCE = R"(
    uint row = threadgroup_position_in_grid.z;
    uint lid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    const uint dim = (uint)Dim;
    const uint tg = (uint)Threads;
    const ulong base = (ulong)row * (ulong)dim;

    threadgroup float local_sums[32];
    threadgroup float local_inv_mean[1];

    // Stage 1: residual sum, store, sum of squares.
    float acc = 0.0f;
    for (uint r = 0; r < dim; r += tg * 4) {
        uint d0 = r + lid * 4;
        for (uint j = 0; j < 4; j++) {
            uint d = d0 + j;
            if (d < dim) {
                float s = (float)x[base + d] + (float)residual[base + d];
                T st = (T)s;
                new_residual[base + d] = st;
                float sr = (float)st;
                acc += sr * sr;
            }
        }
    }

    acc = simd_sum(acc);
    if (sg == 0u) {
        local_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        local_sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float total = simd_sum(local_sums[lane]);
        if (lane == 0u) {
            local_inv_mean[0] = metal::precise::rsqrt(total / dim + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: scale and apply the (weight_bias + weight) gain.
    float inv_mean = local_inv_mean[0];
    float wbias = weight_bias[0];
    for (uint r = 0; r < dim; r += tg * 4) {
        uint d0 = r + lid * 4;
        for (uint j = 0; j < 4; j++) {
            uint d = d0 + j;
            if (d < dim) {
                float sr = (float)new_residual[base + d];
                float wv = (float)weight[d];
                // Fold the bias in the weight's own precision so `weight_bias
                // = 1` reproduces GemmaRMSNorm's precomputed `(1 + w)` tensor
                // exactly rather than approximately.
                float gain = (wbias == 0.0f) ? wv : (float)(TW)(wbias + wv);
                // Round `x * inv_mean` to T before the gain multiply: this is
                // the store order MLX's rms_norm kernel uses.
                float scaled = (float)(T)(sr * inv_mean);
                normed[base + d] = (T)(gain * scaled);
            }
        }
    }
)";

// The CUDA body, `FUSED_ADD_RMS_NORM_CUDA_SOURCE`, and its launch geometry,
// `fused_norm_cuda_block`, live in `fused_norm_cuda.h`. Unlike the Metal body
// above, that port follows MLX's CUDA `rms_norm` geometry and is
// byte-identical to the graph (#3510).

// Elements each Metal lane reads per sweep step. Matches MLX's `RMS_N_READS`.
constexpr int FUSED_NORM_N_READS = 4;
constexpr int FUSED_NORM_SIMD_WIDTH = 32;
constexpr int FUSED_NORM_MAX_THREADS = 1024;
#ifdef MLXCEL_BRIDGE_ROCM_BACKEND
// `BLOCK_DIM` of the overlay's `rms_norm_kernel` launch in `rms_norm.hip`.
constexpr int FUSED_NORM_ROCM_THREADS = 256;
#endif

// Metal threadgroup width for a row of `dim` elements: enough lanes to cover
// the row at `FUSED_NORM_N_READS` elements each, rounded up to a whole SIMD
// group and clamped to the hardware threadgroup limit. Rows longer than
// `FUSED_NORM_MAX_THREADS * FUSED_NORM_N_READS` loop. CUDA does not use this:
// see `fused_norm_cuda_block`.
[[maybe_unused]] int fused_norm_threads(int dim) {
    if (dim <= 0) {
        return FUSED_NORM_SIMD_WIDTH;
    }
    int threads = (dim + FUSED_NORM_N_READS - 1) / FUSED_NORM_N_READS;
    threads = ((threads + FUSED_NORM_SIMD_WIDTH - 1) / FUSED_NORM_SIMD_WIDTH) *
        FUSED_NORM_SIMD_WIDTH;
    if (threads > FUSED_NORM_MAX_THREADS) {
        threads = FUSED_NORM_MAX_THREADS;
    }
    if (threads < FUSED_NORM_SIMD_WIDTH) {
        threads = FUSED_NORM_SIMD_WIDTH;
    }
    return threads;
}

const std::vector<std::string>& fused_norm_input_names() {
    static const std::vector<std::string> names = {
        "x", "residual", "weight", "eps", "weight_bias"};
    return names;
}

const std::vector<std::string>& fused_norm_output_names() {
    static const std::vector<std::string> names = {"normed", "new_residual"};
    return names;
}

// Thread-safe lazy-initialised holder for the JIT-compiled kernel. Mirrors the
// `std::call_once` pattern in `paged_attention.cpp`: the server reaches first
// use concurrently from per-request blocking workers, and `call_once` re-runs
// the initializer if MLX device lookup throws.
struct FusedNormKernelHolder {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::metal_kernel(
                "mlxcel_fused_add_rms_norm",
                fused_norm_input_names(),
                fused_norm_output_names(),
                std::string(FUSED_ADD_RMS_NORM_SOURCE));
        });
        return *kernel;
    }
};

inline FusedNormKernelHolder& get_fused_norm_kernel() {
    static FusedNormKernelHolder holder;
    return holder;
}

struct FusedNormKernelHolderCuda {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
            kernel = mlx::core::fast::cuda_kernel(
                "mlxcel_fused_add_rms_norm",
                fused_norm_input_names(),
                fused_norm_output_names(),
                std::string(FUSED_ADD_RMS_NORM_CUDA_SOURCE));
        });
        return *kernel;
    }
};

inline FusedNormKernelHolderCuda& get_fused_norm_kernel_cuda() {
    static FusedNormKernelHolderCuda holder;
    return holder;
}

// HIP counterpart (issue #2063), reached only on a ROCm build. The body lives
// in `fused_norm_hip.h`; the launch below is shared with the Metal and CUDA ports. On a
// build without the ROCm backend `fast::hip_kernel` is not declared, and
// `fused_norm_ports()` never resolves this entry there (`select_kernel_port`
// reads the running backend), so the throw is unreachable and exists only to
// keep the table shape uniform.
struct FusedNormKernelHolderHip {
    std::optional<mlx::core::fast::CustomKernelFunction> kernel;
    std::once_flag init_flag;

    mlx::core::fast::CustomKernelFunction& get() {
        std::call_once(init_flag, [this] {
#ifdef MLXCEL_BRIDGE_ROCM_BACKEND
            kernel = mlx::core::fast::hip_kernel(
                "mlxcel_fused_add_rms_norm",
                fused_norm_input_names(),
                fused_norm_output_names(),
                std::string(FUSED_ADD_RMS_NORM_HIP_SOURCE));
#else
            throw std::runtime_error(
                "[fused_add_rms_norm] this build has no ROCm backend");
#endif
        });
        return *kernel;
    }
};

inline FusedNormKernelHolderHip& get_fused_norm_kernel_hip() {
    static FusedNormKernelHolderHip holder;
    return holder;
}


// This kernel's ports, in one place. `has_kernel_port` and
// `select_kernel_port` both read it, so a support predicate and the dispatch
// cannot answer differently (#1801).
const mlxcel::KernelPorts& fused_norm_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_fused_norm_kernel().get();
        },
        .cuda = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_fused_norm_kernel_cuda().get();
        },
        .rocm = +[]() -> mlx::core::fast::CustomKernelFunction& {
            return get_fused_norm_kernel_hip().get();
        },
    };
    return ports;
}

} // namespace
bool fused_add_rms_norm_available() {
    // Custom kernels run only on the GPU stream; on a CPU default device
    // (`MLXCEL_DEVICE=cpu` on a GPU build) their `eval_cpu` throws, so the
    // caller must take the graph path there (#2069).
    if (mlx::core::default_device() != mlx::core::Device::gpu) {
        return false;
    }
    return mlxcel::has_kernel_port(fused_norm_ports());
}

std::vector<mlx::core::array> fused_add_rms_norm(
    const mlx::core::array& x,
    const mlx::core::array& residual,
    const mlx::core::array& weight,
    float eps,
    float weight_bias) {
    using mlx::core::Dtype;
    using mlx::core::Shape;
    using mlx::core::fast::TemplateArg;

    if (x.shape() != residual.shape()) {
        std::ostringstream msg;
        msg << "[fused_add_rms_norm] x and residual must have the same shape.";
        throw std::invalid_argument(msg.str());
    }
    if (x.dtype() != residual.dtype()) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] x and residual must have the same dtype.");
    }
    if (x.ndim() < 1 || weight.ndim() != 1) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] x must be at least 1-D and weight exactly 1-D.");
    }
    const int dim = x.shape().back();
    if (dim <= 0 || x.size() == 0) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] x must be non-empty with a positive last dim.");
    }
    if (weight.shape()[0] != dim) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] weight length must equal the last dim of x.");
    }
    if (x.size() % dim != 0) {
        throw std::invalid_argument(
            "[fused_add_rms_norm] x size is not a multiple of its last dim.");
    }

    const int rows = static_cast<int>(x.size() / dim);
    // Threads per row. Metal: enough lanes to cover the row at `N_READS` each.
    // ROCm: the overlay's `rms_norm_kernel` block of 256 (`rms_norm.hip`),
    // whose per-thread strided sums and 32-wide folds the HIP body then
    // reproduces term for term, so the fused sum of squares is the graph's bit
    // for bit (#2063). CUDA: the block MLX's `rms_norm` launches for this
    // width in the compute dtype, the promotion `fast::rms_norm` applies to
    // its inputs, which the CUDA body sweeps in MLX's order, so it is the
    // graph's bit for bit too (#3510). A ROCm or CUDA build has no other GPU
    // backend, so the build flag is the backend here, as in
    // `fused_add3_layer_norm`. `Threads` is a template arg, so each width
    // compiles and caches apart.
#if defined(MLXCEL_BRIDGE_ROCM_BACKEND)
    const int threads = FUSED_NORM_ROCM_THREADS;
#elif defined(MLXCEL_BRIDGE_CUDA_BACKEND)
    const Dtype compute = mlx::core::promote_types(x.dtype(), weight.dtype());
    const int threads = fused_norm_cuda_block(
        dim, fused_norm_cuda_n_reads(mlx::core::size_of(compute)));
#else
    const int threads = fused_norm_threads(dim);
#endif

#ifdef MLXCEL_CUDA_OXIDE_FUSED_NORM
    // Experimental cuda-oxide route (#2833): only in a build that opted in,
    // and only once a validated artifact is installed. `nullopt` means the
    // port below builds the node, unchanged. See `fused_norm_oxide.h`.
    const auto oxide_rows = static_cast<std::size_t>(x.size() / dim);
    if (auto oxide = try_oxide_fused_add_rms_norm(
            x, residual, weight, eps, weight_bias, oxide_rows, dim, threads)) {
        return std::move(*oxide);
    }
#endif

    // Refuses when this backend has no port, naming the entry point and the
    // fallback from the table rather than whichever port was tried. Callers gate
    // on the same table, so a refusal here means a direct call; the bridge
    // declares this function `Result`, so it becomes an `Err` (issues #1885, #1801).
    auto& kernel = mlxcel::select_kernel_port(
        "fused_add_rms_norm", "graph fallback", fused_norm_ports());

    std::vector<std::pair<std::string, TemplateArg>> template_args = {
        {"T", x.dtype()},
        {"TW", weight.dtype()},
        {"Dim", dim},
        {"Threads", threads},
    };
#if defined(MLXCEL_BRIDGE_CUDA_BACKEND) && !defined(MLXCEL_BRIDGE_ROCM_BACKEND)
    // CUDA only: the compute dtype, and which of the two inputs is cast to it
    // (see `fused_norm_cuda.h`). The Metal and ROCm bodies take none of
    // these, so their template args and cache keys are unchanged.
    template_args.emplace_back("TC", compute);
    template_args.emplace_back("RoundX", compute != x.dtype() ? 1 : 0);
    template_args.emplace_back("RoundW", compute != weight.dtype() ? 1 : 0);
#endif

    // `eps` and `weight_bias` ride in as 1-element f32 arrays rather than 0-D
    // scalars: a 0-D input is declared `const constant float&` by the Metal
    // signature writer and `const float*` by the CUDA one, so the two kernel
    // bodies would have to differ purely on argument syntax. The `array({v})`
    // constructor sets the data directly, so neither of these costs a dispatch.
    auto eps_arr = mlx::core::array({eps});
    auto bias_arr = mlx::core::array({weight_bias});

    std::vector<mlx::core::array> inputs = {x, residual, weight, eps_arr, bias_arr};
    std::vector<Shape> output_shapes = {x.shape(), x.shape()};
    std::vector<Dtype> output_dtypes = {x.dtype(), x.dtype()};

    return kernel(
        inputs,
        output_shapes,
        output_dtypes,
        std::make_tuple(threads, 1, rows), // grid
        std::make_tuple(threads, 1, 1),    // threadgroup
        template_args,
        std::nullopt,
        false,
        {});
}

} // namespace mlxcel::turbo
