// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Shared shape and backend keys for the BF16 `addmm` rounding oracles.
//!
//! The vision layers that mirror upstream `mlx.nn.Linear` call
//! `mlxcel_core::addmm`, and their tests prove that routing with the
//! authenticated Sapiens2 BF16 case, where a fused bias epilogue (one rounding
//! of `x @ w.T + b`) and a separate bias add (round the product, then round the
//! sum) give different results.
//!
//! Which rounding MLX produces depends on the kernel its `addmm` selects, and
//! at the pinned MLX every kernel rounds once:
//!
//! - CPU and ROCm copy the bias into the output and run one GEMM with `beta`,
//!   so the product and the bias meet before the single rounding to BF16.
//!   CUDA's `AddMM::eval_gpu` also has no gemv route (one GEMM with the bias),
//!   but no CUDA run has confirmed it.
//! - Metal routes a single-row product (`min(M, N) == 1`) to `gemv_axbpy`, and
//!   two or more rows to the steel GEMM, `gemv_wide`, or mlxcel's batched gemv
//!   overlay. Upstream ml-explore/mlx#4769 ("Fix precision loss in addmm",
//!   commit `49366677`, in the pin since #2876) changed the `GEMVKernel` and
//!   `GEMVTKernel` epilogues in `mlx/backend/metal/kernels/gemv.h` to add the
//!   bias to the F32 accumulator before the cast, so the gemv route also
//!   rounds once.
//!
//! Before that fix the gemv epilogue cast the accumulator to BF16 first and
//! rounded twice, exactly like a separate add. Released `mlx` 0.32.4, the
//! source of the oracle values, still does: on an Apple M1 Ultra and an M3
//! Ultra its GPU `mx.addmm` gives the separate value at one row and the fused
//! value at two, while its CPU stream gives the fused value at both. The
//! oracles therefore assert both shapes: [`ROUTING_ROWS`] identical rows, the
//! shape that rounds once in every MLX version, and one row, which rounds once
//! only with the fix and fails if a pin change brings the double rounding
//! back.

/// Rows that select a single-rounding `addmm` kernel on every backend, with or
/// without ml-explore/mlx#4769.
pub(crate) const ROUTING_ROWS: usize = 2;

/// Repeat one row of values [`ROUTING_ROWS`] times.
pub(crate) fn routing_rows(row: &[f32]) -> Vec<f32> {
    row.repeat(ROUTING_ROWS)
}
