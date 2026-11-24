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

// Row-invariant dense matvec (issue #3280). See `row_invariant_matvec.h`
// for the contract.
//
// MLX multiplies one row with gemv and several rows with gemm, and the two
// accumulate in different orders, so a K-row speculative verify block cannot
// match K single-token decode steps through `matmul`. This kernel reads each
// weight row once for up to 16 input rows and gives every input row its own
// accumulator, walked in the same order whatever the row count, so the decode
// step (one row) and the verify block (several) produce the same bits when
// both go through it.

#include "row_invariant_matvec.h"

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "kernel_port.h"

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace mlxcel::turbo {
namespace {

// One SIMD group per NO consecutive output columns. Lane l reads, for
// k = 4 * l + 128 * j (j = 0, 1, ...), the four inputs of every row once and
// the four weights of every column, and adds the four products of each
// (row, column) pair to that pair's f32 accumulator in a fixed order;
// `simd_sum` then combines the lanes. The arithmetic of one output depends
// only on K: neither the row count nor NO changes it. Inputs are indexed,
// not read through a `device` pointer: MLX binds an array of fewer than 8
// elements in the `constant` address space (a tiny test model's x or w).
constexpr const char* ROWS_MATVEC_SOURCE = R"(
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = thread_position_in_grid.y * NO;
    const uint N = w_shape[0];
    const uint K = w_shape[1];
    if (n0 >= N) return;
    float acc[ROWS][NO];
    for (int m = 0; m < ROWS; ++m)
        for (int j = 0; j < NO; ++j) acc[m][j] = 0.0f;
    for (uint k = lane * 4; k < K; k += 128) {
        float xs[ROWS][4];
        for (int m = 0; m < ROWS; ++m) {
            const size_t xr = size_t(m) * K + k;
            for (int i = 0; i < 4; ++i) xs[m][i] = float(x[xr + i]);
        }
        for (int j = 0; j < NO; ++j) {
            const uint n = min(n0 + j, N - 1);
            const size_t wr = size_t(n) * K + k;
            const float w0 = float(w[wr]), w1 = float(w[wr + 1]);
            const float w2 = float(w[wr + 2]), w3 = float(w[wr + 3]);
            for (int m = 0; m < ROWS; ++m) {
                float part = w0 * xs[m][0];
                part = fma(w1, xs[m][1], part);
                part = fma(w2, xs[m][2], part);
                part = fma(w3, xs[m][3], part);
                acc[m][j] += part;
            }
        }
    }
    for (int m = 0; m < ROWS; ++m) {
        for (int j = 0; j < NO; ++j) {
            const float total = simd_sum(acc[m][j]);
            if (lane == 0 && n0 + j < N) out[size_t(m) * N + n0 + j] = OT(total);
        }
    }
)";

// The narrow-output variant (N < 512): SPLIT SIMD groups share one output
// column. Group s takes k = 128 * s + 4 * lane + 128 * SPLIT * j, reduces
// with `simd_sum`, and the first thread adds the SPLIT partials in group
// order. The variant is chosen from N alone, so the order still ignores the
// rows.
constexpr const char* ROWS_MATVEC_SPLIT_SOURCE = R"(
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;
    const uint n = threadgroup_position_in_grid.y;
    const uint N = w_shape[0];
    const uint K = w_shape[1];
    threadgroup float partial[SPLIT][ROWS];
    float acc[ROWS];
    for (int m = 0; m < ROWS; ++m) acc[m] = 0.0f;
    const size_t wr = size_t(n) * K;
    for (uint k = group * 128 + lane * 4; k < K; k += 128 * SPLIT) {
        const float w0 = float(w[wr + k]), w1 = float(w[wr + k + 1]);
        const float w2 = float(w[wr + k + 2]), w3 = float(w[wr + k + 3]);
        for (int m = 0; m < ROWS; ++m) {
            const size_t xr = size_t(m) * K + k;
            float part = w0 * float(x[xr]);
            part = fma(w1, float(x[xr + 1]), part);
            part = fma(w2, float(x[xr + 2]), part);
            part = fma(w3, float(x[xr + 3]), part);
            acc[m] += part;
        }
    }
    for (int m = 0; m < ROWS; ++m) {
        const float total = simd_sum(acc[m]);
        if (lane == 0) partial[group][m] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0 && lane < uint(ROWS)) {
        float sum = 0.0f;
        for (int g = 0; g < SPLIT; ++g) sum += partial[g][lane];
        out[size_t(lane) * N + n] = OT(sum);
    }
)";

struct RowsMatvecKernels {
    std::optional<mlx::core::fast::CustomKernelFunction> wide;
    std::optional<mlx::core::fast::CustomKernelFunction> split;
    std::once_flag once;

    void init() {
        std::call_once(once, [this] {
            wide = mlx::core::fast::metal_kernel(
                "mlxcel_row_invariant_matvec", {"x", "w"}, {"out"}, ROWS_MATVEC_SOURCE);
            split = mlx::core::fast::metal_kernel(
                "mlxcel_row_invariant_matvec_split",
                {"x", "w"},
                {"out"},
                ROWS_MATVEC_SPLIT_SOURCE);
        });
    }
};

RowsMatvecKernels& rows_kernels() {
    static RowsMatvecKernels holder;
    return holder;
}

const mlxcel::KernelPorts& rows_matvec_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            rows_kernels().init();
            return *rows_kernels().wide;
        },
        .cuda = nullptr,
        .rocm = nullptr,
    };
    return ports;
}

const mlxcel::KernelPorts& rows_matvec_split_ports() {
    static const mlxcel::KernelPorts ports{
        .metal = +[]() -> mlx::core::fast::CustomKernelFunction& {
            rows_kernels().init();
            return *rows_kernels().split;
        },
        .cuda = nullptr,
        .rocm = nullptr,
    };
    return ports;
}

// Columns below which a column gets several SIMD groups. A function of N
// only: the summation order must not depend on the row count.
constexpr int kSplitBelowColumns = 512;
constexpr int kSplitGroups = 8;

[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("[row_invariant_matvec] " + what);
}

bool supported(mlx::core::Dtype dt) {
    return dt == mlx::core::bfloat16 || dt == mlx::core::float16 || dt == mlx::core::float32;
}

} // namespace

bool row_invariant_matvec_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu &&
        mlxcel::has_kernel_port(rows_matvec_ports());
}

mlx::core::array row_invariant_matvec(
    const mlx::core::array& x,
    const mlx::core::array& w,
    mlx::core::Dtype output_dtype) {
    using namespace mlx::core;
    if (w.ndim() != 2 || !supported(w.dtype())) fail("w must be a float [N, K] array");
    const int N = w.shape(0), K = w.shape(1);
    if (x.ndim() != 2 || x.shape(1) != K || !supported(x.dtype())) {
        fail("x must be a float [M, K] array with the weight's K");
    }
    const int M = x.shape(0);
    if (M < 1 || M > kRowInvariantMatvecMaxRows) fail("x must hold 1 to 16 rows");
    if (K % 4 != 0) fail("K must be a multiple of 4");
    if (!supported(output_dtype)) fail("the output dtype must be a float dtype");
    if (N < kSplitBelowColumns) {
        auto& kernel = mlxcel::select_kernel_port(
            "row_invariant_matvec", "row-invariant matmul", rows_matvec_split_ports());
        auto outputs = kernel(
            {x, w},
            {Shape{M, N}},
            {output_dtype},
            std::make_tuple(32 * kSplitGroups, N, 1),
            std::make_tuple(32 * kSplitGroups, 1, 1),
            {{"T", x.dtype()},
             {"WT", w.dtype()},
             {"OT", output_dtype},
             {"ROWS", M},
             {"SPLIT", kSplitGroups}},
            std::nullopt,
            false,
            {});
        return std::move(outputs[0]);
    }
    auto& kernel =
        mlxcel::select_kernel_port("row_invariant_matvec", "row-invariant matmul", rows_matvec_ports());
    // One column per SIMD group measured fastest on M3 Ultra decode; the
    // template keeps NO so a wider tiling can be tried without touching the
    // summation order.
    constexpr int NO = 1;
    const int groups = (N + NO - 1) / NO;
    auto outputs = kernel(
        {x, w},
        {Shape{M, N}},
        {output_dtype},
        std::make_tuple(32, groups, 1),
        std::make_tuple(32, 8, 1),
        {{"T", x.dtype()}, {"WT", w.dtype()}, {"OT", output_dtype}, {"ROWS", M}, {"NO", NO}},
        std::nullopt,
        false,
        {});
    return std::move(outputs[0]);
}

} // namespace mlxcel::turbo
