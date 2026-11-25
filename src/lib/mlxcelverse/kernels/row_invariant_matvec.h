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

#pragma once

// Row-invariant dense matvec (issue #3280).
//
// MLX multiplies one row with gemv and several with gemm, which sum in
// different orders, so a K-row speculative verify block cannot reproduce K
// single-token decode steps through `matmul`. A model that routes every dense
// product of both forwards through this kernel gets the same bits from both.

#include <mlx/array.h>

namespace mlxcel::turbo {

// Most input rows `row_invariant_matvec` takes.
constexpr int kRowInvariantMatvecMaxRows = 16;

// Whether the kernel can run here (Metal default device).
bool row_invariant_matvec_available();

// `x [M, K] @ w [N, K]^T` as `[M, N]` in `output_dtype`, `1 <= M <= 16`,
// `K % 4 == 0`, x and w row-contiguous bf16, f16 or f32. Each output is an
// f32 sum whose order depends on K and N, never on M, so row m of an M-row
// product equals the one-row product of row m bit for bit. Throws
// `std::invalid_argument` outside the contract.
mlx::core::array row_invariant_matvec(
    const mlx::core::array& x,
    const mlx::core::array& w,
    mlx::core::Dtype output_dtype);

} // namespace mlxcel::turbo
