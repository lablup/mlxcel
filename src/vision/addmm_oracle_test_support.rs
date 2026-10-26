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
//! Which rounding MLX produces depends on the kernel its `addmm` selects:
//!
//! - CPU and ROCm round once for every row count: both copy the bias into the
//!   output and run one GEMM with `beta`, so the product and the bias meet
//!   before the single rounding to BF16. CUDA's `AddMM::eval_gpu` also has no
//!   gemv route (one GEMM with the bias), so it is keyed with them, but no
//!   CUDA run has confirmed it.
//! - Metal routes a single-row product (`min(M, N) == 1`) to `gemv_axbpy`.
//!   Its epilogue in
//!   <https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/gemv.h>
//!   (`GEMVKernel::run`) casts the F32 accumulator to the output type before
//!   adding the bias, so it rounds twice, exactly like a separate add. Two or
//!   more rows select the steel GEMM or `gemv_wide` kernels, which apply the
//!   bias to the F32 accumulator and round once.
//!
//! Upstream `mlx` 0.32.4 shows the same split on an Apple M1 Ultra: GPU
//! `mx.addmm` gives the separate value at one row and the fused value at two or
//! more rows, while the CPU stream gives the fused value at every row count.
//! So at one row on Metal, `addmm` and `matmul + add` cannot be told apart, and
//! an oracle at that shape cannot prove the routing. The oracles therefore run
//! on [`ROUTING_ROWS`] identical rows, where every backend rounds once, and pin
//! the single-row value per backend with [`single_row_addmm_rounds_twice`].

/// Rows that select a single-rounding `addmm` kernel on every backend.
pub(crate) const ROUTING_ROWS: usize = 2;

/// Repeat one row of values [`ROUTING_ROWS`] times.
pub(crate) fn routing_rows(row: &[f32]) -> Vec<f32> {
    row.repeat(ROUTING_ROWS)
}

/// True when a single-row BF16 `addmm` on the current default device rounds
/// the product before adding the bias (Metal's `gemv_axbpy` epilogue).
///
/// Metal takes the same route whenever `min(M, N) == 1`, so an oracle with a
/// one-column output also rounds twice there; keep oracle outputs at least two
/// columns wide.
///
/// The answer depends on the process-wide default device, so callers hold
/// [`mlxcel_core::streams::lock_default_device`] for as long as they rely on it.
pub(crate) fn single_row_addmm_rounds_twice() -> bool {
    mlxcel_core::default_device_is_gpu()
        && mlxcel_core::hardware::gpu_backend_kind() == mlxcel_core::hardware::GpuBackendKind::Metal
}
