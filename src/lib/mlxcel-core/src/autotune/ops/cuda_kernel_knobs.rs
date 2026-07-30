// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! CUDA kernel knobs wired into the autotuner (issue #906).
//!
//! # UNVALIDATED
//!
//! **Neither op in this file has been measured on CUDA hardware.** They were
//! written on an Apple-Silicon-only host, so the candidate sets, the defaults,
//! and the workload shapes are derived from the kernel sources and the existing
//! GB10 tuning reports, not from a run. Before anything here is trusted:
//!
//! - run `mlxcel tune --op qmm-tile --op qmv-multirow` on a GB10 (or other
//!   CUDA) host and compare against
//!   `docs/benchmark_results/qmm-sm121-tile-tuning-gb10-2026-07-10.md` and
//!   `docs/benchmark_results/qmv-multirow-gb10-2026-07-11.md`;
//! - confirm the tuner at minimum reproduces the current best-known manual
//!   configuration before letting it override anything.
//!
//! Both ops are inert until `MLXCEL_AUTOTUNE` is set, and both are
//! [`TunableOp::lazy_tunable`]`== false`, so a serving process never profiles
//! them.
//!
//! ## Why these two knobs are process-wide, not per-launch
//!
//! Unlike the paged-decode split count, these knobs are read by patched MLX
//! CUDA kernels through the environment
//! (`src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/`). mlxcel's C++
//! and MLX's C++ are separate link units in the "mlxcel calls MLX" direction
//! only, so there is no in-process channel from the tuner to those kernels
//! other than the environment they already read. That has two consequences
//! this module works within:
//!
//! 1. A tuned value is applied by [`apply_tuned_cuda_kernel_env`], called once
//!    at the top of `main` before any MLX work or worker thread exists, in
//!    exactly the pattern [`crate::hardware::apply_cuda_graph_cache_default`]
//!    established.
//! 2. Because the value is process-wide, the cache key's shape bucket is the
//!    *representative* shape the knob was tuned at, not the shape of every
//!    later launch. The canonical buckets are pinned below so a startup lookup
//!    and an offline tuning run agree on the key.
//!
//! An explicitly-set env var always wins: [`TunableOp::env_override`] returns
//! it, and the startup applier never overwrites a variable the operator set.

use std::sync::OnceLock;

use crate::autotune::bucket::ShapeBucket;
use crate::autotune::store::TacticStore;
use crate::autotune::tactic::{Tactic, TunableOp, TuneError};
use crate::autotune::{TuneKey, device_label, mode};
use crate::{MlxArray, UniquePtr, ffi};

// ── qmm CTA tile ─────────────────────────────────────────────────────────────

/// Logical op name for the Blackwell qmm CTA tile choice.
pub const OP_QMM_TILE: &str = "qmm_sm80_cta_tile_m";

/// Existing manual override, honored verbatim and unchanged by #906.
pub const QMM_TILE_M_ENV: &str = "MLXCEL_QMM_TILE_M";

/// Upstream MLX Ampere CTA `tile_m` cap, kept on sm_80/sm_90.
pub const TILE_M_CAP_DEFAULT: i64 = 64;

/// mlxcel's consumer-Blackwell (sm_120/121) CTA `tile_m` cap from issue #637.
pub const TILE_M_CAP_BLACKWELL: i64 = 128;

/// Smallest CTA `tile_m` `make_cta_tiler` will produce.
pub const TILE_M_MIN: i64 = 16;

/// The explicit `MLXCEL_QMM_TILE_M`, read once.
fn env_tile_m() -> Option<i64> {
    static VALUE: OnceLock<Option<i64>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var(QMM_TILE_M_ENV).ok()?;
        match raw.trim().parse::<i64>() {
            Ok(v) if v > 0 => Some(v),
            _ => None,
        }
    })
}

/// Round `m` up to the next power of two, floored at [`TILE_M_MIN`] and capped
/// at `cap`. Mirrors `make_cta_tiler` in
/// `patches/mlx/backend/cuda/quantized/qmm/qmm_sm80.cu` exactly, so the
/// autotuner's "default" is the value the kernel would have picked.
#[must_use]
pub fn default_tile_m(m: usize, cap: i64) -> i64 {
    let next = i64::from(crate::autotune::round_up_pow2(m));
    next.clamp(TILE_M_MIN, cap.max(TILE_M_MIN))
}

/// Candidate CTA `tile_m` values: powers of two from [`TILE_M_MIN`] to `cap`.
///
/// Only `tile_m` is tuned. `tile_n` and `tile_k` are fixed because the kernel's
/// MMA shared-memory layout is built for them; the #637 spike swept both and
/// every wider `tile_n` / deeper `tile_k` failed to JIT.
#[must_use]
pub fn tile_m_candidates(cap: i64) -> Vec<i64> {
    let cap = cap.max(TILE_M_MIN);
    let mut out = Vec::new();
    let mut v = TILE_M_MIN;
    while v <= cap {
        out.push(v);
        v *= 2;
    }
    out
}

/// Shape of the representative quantized GEMM a tile candidate is timed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QmmShape {
    /// Activation rows (prefill token count).
    pub m: usize,
    /// Output features.
    pub n: usize,
    /// Input features.
    pub k: usize,
    pub bits: i32,
    pub group_size: i32,
}

impl QmmShape {
    /// Canonical prefill shape the CTA tile is tuned at.
    ///
    /// M = 8192 is the prefill length the #637 / GB10 tile sweep used, and N/K
    /// are llama-3.1-8b's `down_proj` (the largest per-layer GEMM in that
    /// sweep). Pinned as a constant so a startup lookup and an offline tuning
    /// run produce the same cache key.
    #[must_use]
    pub fn canonical() -> Self {
        Self {
            m: 8192,
            n: 4096,
            k: 14336,
            bits: 4,
            group_size: 64,
        }
    }

    #[must_use]
    pub fn bucket(&self) -> ShapeBucket {
        ShapeBucket::from_dims(&[self.m, self.n, self.k])
    }

    #[must_use]
    pub fn dtype_tag(&self) -> String {
        format!("q{}g{}", self.bits, self.group_size)
    }
}

/// Tunes the CTA `tile_m` of the patched Blackwell qmm kernel. **Unvalidated**,
/// see the module docs.
pub struct QmmTileOp {
    shape: QmmShape,
    cap: i64,
    workload: Option<QmmWorkload>,
}

/// Pre-built arrays for the timed GEMM. Built once so the sweep measures the
/// kernel, not the quantization.
struct QmmWorkload {
    x: UniquePtr<MlxArray>,
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: Option<UniquePtr<MlxArray>>,
}

impl QmmTileOp {
    /// Build the op without a workload. Enough for key material, candidates,
    /// and cache lookups; [`Self::with_workload`] is required before profiling.
    #[must_use]
    pub fn new(shape: QmmShape, cap: i64) -> Self {
        Self {
            shape,
            cap,
            workload: None,
        }
    }

    /// Allocate and quantize the timed GEMM's operands, once, so the sweep
    /// measures the kernel rather than the quantization. Values are irrelevant
    /// (the tile choice is a launch-shape decision), so the operands are zeros.
    #[must_use]
    pub fn with_workload(mut self) -> Self {
        let s = self.shape;
        let x = ffi::zeros(&[s.m as i32, s.k as i32], crate::dtype::FLOAT16);
        let w_dense = ffi::zeros(&[s.n as i32, s.k as i32], crate::dtype::FLOAT16);
        let quantized = ffi::quantize_weights(&w_dense, s.group_size, s.bits);
        let w = ffi::quantized_weights_w(&quantized);
        let scales = ffi::quantized_weights_scales(&quantized);
        let biases = ffi::quantized_weights_has_biases(&quantized)
            .then(|| ffi::quantized_weights_biases(&quantized));
        for arr in [&x, &w, &scales] {
            crate::eval(arr);
        }
        if let Some(b) = biases.as_ref() {
            crate::eval(b);
        }
        crate::synchronize_default();
        self.workload = Some(QmmWorkload {
            x,
            w,
            scales,
            biases,
        });
        self
    }
}

impl TunableOp for QmmTileOp {
    fn op_name(&self) -> &str {
        OP_QMM_TILE
    }

    fn runner_id(&self) -> String {
        format!("cuda-tile-cap-{}", self.cap)
    }

    fn dtype_tag(&self) -> String {
        self.shape.dtype_tag()
    }

    fn bucket(&self) -> ShapeBucket {
        self.shape.bucket()
    }

    fn candidates(&self, _bucket: &ShapeBucket) -> Vec<Tactic> {
        tile_m_candidates(self.cap)
            .into_iter()
            .map(|v| Tactic::scalar("tile_m", v))
            .collect()
    }

    fn default_tactic(&self, _bucket: &ShapeBucket) -> Tactic {
        Tactic::scalar("tile_m", default_tile_m(self.shape.m, self.cap))
    }

    fn env_override(&self) -> Option<Tactic> {
        env_tile_m().map(|v| Tactic::scalar("tile_m", v))
    }

    fn lazy_tunable(&self) -> bool {
        // `run` mutates the process environment, which is only sound while the
        // process is effectively single-threaded. Offline `mlxcel tune` only.
        false
    }

    fn run(&self, tactic: &Tactic) -> Result<(), TuneError> {
        let Some(workload) = self.workload.as_ref() else {
            return Err(TuneError::infeasible(
                tactic,
                "no workload was built; call with_workload() before profiling",
            ));
        };
        let tile_m = tactic
            .param(0)
            .ok_or_else(|| TuneError::infeasible(tactic, "tactic carries no tile_m"))?;

        // Drain every outstanding MLX operation before touching the
        // environment. `make_cta_tiler` calls `getenv` from whichever thread is
        // encoding the GEMM, so the mutation below must not overlap a live
        // encode.
        crate::synchronize_default();
        // SAFETY: `set_var` mutates the process-global environment and is
        // unsound if another thread reads or writes the environment
        // concurrently. This function is reachable only from the offline
        // `mlxcel tune` path (`lazy_tunable() == false` keeps every serving
        // process out), which runs before any model, server, or request worker
        // exists; the `synchronize_default()` above additionally parks MLX's
        // own stream worker, so no thread is inside `getenv` here.
        unsafe { std::env::set_var(QMM_TILE_M_ENV, tile_m.to_string()) };

        let s = self.shape;
        let biases_ptr = workload
            .biases
            .as_deref()
            .map_or(std::ptr::null(), |r| r as *const MlxArray);
        // SAFETY: `quantized_matmul` takes a nullable `biases` pointer; the
        // pointer above is either null or borrowed from `workload`, which
        // outlives the call.
        let out = unsafe {
            ffi::quantized_matmul(
                &workload.x,
                &workload.w,
                &workload.scales,
                biases_ptr,
                true,
                s.group_size,
                s.bits,
                "affine",
            )
        };
        crate::eval(&out);
        Ok(())
    }
}

// ── multirow qmv row window ──────────────────────────────────────────────────

/// Logical op name for the multirow-qmv row-window ceiling.
pub const OP_QMV_MULTIROW: &str = "qmv_multirow_max_rows";

/// Existing kill switch (issue #725). `0` disables the multirow path outright
/// and wins over everything here.
pub const QMV_MULTIROW_ENV: &str = "MLXCEL_QMV_MULTIROW";

/// Explicit row-window ceiling override (issue #906). Read once by
/// `patches/mlx/backend/cuda/quantized/qmm/qmv.cu`.
pub const QMV_MULTIROW_MAX_ROWS_ENV: &str = "MLXCEL_QMV_MULTIROW_MAX_ROWS";

/// Compile-time ceiling in `qmv.cu` (`max_x_rows`). The tuned value can only
/// narrow the window, never widen it: the multirow kernel keeps its
/// accumulators in registers sized by this constant.
pub const QMV_MAX_ROWS_HARD: i64 = 8;

/// Whether the multirow path is disabled outright by the existing kill switch.
#[must_use]
pub fn multirow_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var(QMV_MULTIROW_ENV).is_ok_and(|v| v.trim() == "0"))
}

/// The explicit `MLXCEL_QMV_MULTIROW_MAX_ROWS`, read once and clamped to
/// `[1, QMV_MAX_ROWS_HARD]`.
fn env_max_rows() -> Option<i64> {
    static VALUE: OnceLock<Option<i64>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var(QMV_MULTIROW_MAX_ROWS_ENV).ok()?;
        match raw.trim().parse::<i64>() {
            Ok(v) if v >= 1 => Some(v.min(QMV_MAX_ROWS_HARD)),
            _ => None,
        }
    })
}

/// Candidate row-window ceilings, `1 ..= QMV_MAX_ROWS_HARD`.
///
/// `1` means "never take the multirow path" (the kernel gate is
/// `x_rows >= 2`), so the kill switch is representable as a tactic and the
/// crossover search covers the whole documented 2-7 row window plus its
/// endpoints.
#[must_use]
pub fn max_rows_candidates() -> Vec<i64> {
    (1..=QMV_MAX_ROWS_HARD).collect()
}

/// Shape of the representative batched decode matvec the window is tuned on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QmvShape {
    pub n: usize,
    pub k: usize,
    pub bits: i32,
    pub group_size: i32,
    /// Total decode rows the sweep issues per timed repetition, held constant
    /// across candidates so wall times are comparable.
    pub total_rows: usize,
}

impl QmvShape {
    /// Canonical shape: llama-3.1-8b `down_proj`, 8 decode rows per repetition
    /// (the documented `M*B` window the multirow path covers).
    #[must_use]
    pub fn canonical() -> Self {
        Self {
            n: 4096,
            k: 14336,
            bits: 4,
            group_size: 64,
            total_rows: QMV_MAX_ROWS_HARD as usize,
        }
    }

    #[must_use]
    pub fn bucket(&self) -> ShapeBucket {
        ShapeBucket::from_dims(&[self.total_rows, self.n, self.k])
    }
}

/// Tunes the multirow-qmv row-window ceiling. **Unvalidated**, see the module
/// docs.
///
/// ## How a candidate is measured without switching kernels
///
/// The kernel's window gate is read once per process, so a sweep cannot flip it
/// between candidates. It does not need to: a candidate window of `w` is
/// measured by issuing the *same* `total_rows` of decode work as
/// `ceil(total_rows / w)` launches of `w` rows each. A window of 8 is one
/// 8-row multirow launch; a window of 2 is four 2-row launches; a window of 1
/// is eight single-row launches, which take the stock per-row kernel because
/// the multirow gate requires `x_rows >= 2`. Total work is identical across
/// candidates, so the min-latency winner is exactly the row grouping the
/// hardware prefers, which is the crossover the hardcoded window encodes.
pub struct QmvMultirowOp {
    shape: QmvShape,
    workload: Option<QmvWorkload>,
}

struct QmvWorkload {
    /// One activation tensor per candidate row count, indexed by `rows - 1`.
    x_by_rows: Vec<UniquePtr<MlxArray>>,
    w: UniquePtr<MlxArray>,
    scales: UniquePtr<MlxArray>,
    biases: Option<UniquePtr<MlxArray>>,
}

impl QmvMultirowOp {
    #[must_use]
    pub fn new(shape: QmvShape) -> Self {
        Self {
            shape,
            workload: None,
        }
    }

    /// Allocate the per-row-count activations and the shared quantized weight.
    #[must_use]
    pub fn with_workload(mut self) -> Self {
        let s = self.shape;
        let x_by_rows: Vec<UniquePtr<MlxArray>> = (1..=QMV_MAX_ROWS_HARD)
            .map(|rows| ffi::zeros(&[rows as i32, s.k as i32], crate::dtype::FLOAT16))
            .collect();
        let w_dense = ffi::zeros(&[s.n as i32, s.k as i32], crate::dtype::FLOAT16);
        let quantized = ffi::quantize_weights(&w_dense, s.group_size, s.bits);
        let w = ffi::quantized_weights_w(&quantized);
        let scales = ffi::quantized_weights_scales(&quantized);
        let biases = ffi::quantized_weights_has_biases(&quantized)
            .then(|| ffi::quantized_weights_biases(&quantized));
        for arr in &x_by_rows {
            crate::eval(arr);
        }
        crate::eval(&w);
        crate::eval(&scales);
        if let Some(b) = biases.as_ref() {
            crate::eval(b);
        }
        crate::synchronize_default();
        self.workload = Some(QmvWorkload {
            x_by_rows,
            w,
            scales,
            biases,
        });
        self
    }
}

impl TunableOp for QmvMultirowOp {
    fn op_name(&self) -> &str {
        OP_QMV_MULTIROW
    }

    fn runner_id(&self) -> String {
        "cuda".to_string()
    }

    fn dtype_tag(&self) -> String {
        format!("q{}g{}", self.shape.bits, self.shape.group_size)
    }

    fn bucket(&self) -> ShapeBucket {
        self.shape.bucket()
    }

    fn candidates(&self, _bucket: &ShapeBucket) -> Vec<Tactic> {
        if multirow_disabled() {
            return Vec::new();
        }
        max_rows_candidates()
            .into_iter()
            .map(|v| Tactic::scalar("max_rows", v))
            .collect()
    }

    fn default_tactic(&self, _bucket: &ShapeBucket) -> Tactic {
        Tactic::scalar("max_rows", QMV_MAX_ROWS_HARD)
    }

    fn env_override(&self) -> Option<Tactic> {
        if multirow_disabled() {
            // The kill switch is the strongest signal there is: the operator
            // asked for the stock per-row dispatch, so report the window that
            // reproduces it and never consult the cache.
            return Some(Tactic::scalar("max_rows", 1));
        }
        env_max_rows().map(|v| Tactic::scalar("max_rows", v))
    }

    fn lazy_tunable(&self) -> bool {
        false
    }

    fn run(&self, tactic: &Tactic) -> Result<(), TuneError> {
        let Some(workload) = self.workload.as_ref() else {
            return Err(TuneError::infeasible(
                tactic,
                "no workload was built; call with_workload() before profiling",
            ));
        };
        let rows = tactic
            .param(0)
            .ok_or_else(|| TuneError::infeasible(tactic, "tactic carries no row window"))?;
        if !(1..=QMV_MAX_ROWS_HARD).contains(&rows) {
            return Err(TuneError::infeasible(
                tactic,
                format!("row window outside [1, {QMV_MAX_ROWS_HARD}]"),
            ));
        }
        let Some(x) = workload.x_by_rows.get((rows - 1) as usize) else {
            return Err(TuneError::infeasible(tactic, "missing activation tensor"));
        };
        let launches = self.shape.total_rows.div_ceil(rows as usize);
        let s = self.shape;
        let biases_ptr = workload
            .biases
            .as_deref()
            .map_or(std::ptr::null(), |r| r as *const MlxArray);
        for _ in 0..launches {
            // SAFETY: `quantized_matmul` takes a nullable `biases` pointer; the
            // pointer above is either null or borrowed from `workload`, which
            // outlives the call.
            let out = unsafe {
                ffi::quantized_matmul(
                    x,
                    &workload.w,
                    &workload.scales,
                    biases_ptr,
                    true,
                    s.group_size,
                    s.bits,
                    "affine",
                )
            };
            crate::eval(&out);
        }
        Ok(())
    }
}

// ── Startup application ──────────────────────────────────────────────────────

/// Apply cached CUDA kernel tactics to the process environment.
///
/// Call this once, early in `main()`, before any MLX op runs and before
/// spawning threads: the patched CUDA kernels read these variables when they
/// first encode work, and setting an environment variable is only sound while
/// the process is effectively single-threaded. This is the same contract as
/// [`crate::hardware::apply_cuda_graph_cache_default`].
///
/// A variable the operator already set is never overwritten. When
/// `MLXCEL_AUTOTUNE` is unset the function returns immediately without reading
/// the cache, so the default build behaves exactly as it did before #906.
///
/// Returns the `(variable, value)` pairs it applied, for logging.
pub fn apply_tuned_cuda_kernel_env(tile_m_cap: i64) -> Vec<(&'static str, String)> {
    if !mode().reads_cache() {
        return Vec::new();
    }
    let store = TacticStore::from_cache_root();
    let mut applied = Vec::new();

    let qmm = QmmTileOp::new(QmmShape::canonical(), tile_m_cap);
    if let Some(value) = cached_param(&store, &qmm) {
        applied.extend(apply_env_once(QMM_TILE_M_ENV, value));
    }

    let qmv = QmvMultirowOp::new(QmvShape::canonical());
    if let Some(value) = cached_param(&store, &qmv) {
        applied.extend(apply_env_once(QMV_MULTIROW_MAX_ROWS_ENV, value));
    }

    applied
}

/// Cached first parameter for `op`, or `None` when there is no usable entry or
/// an explicit env override already decides the knob.
fn cached_param(store: &TacticStore, op: &dyn TunableOp) -> Option<i64> {
    if op.env_override().is_some() {
        return None;
    }
    let bucket = op.bucket();
    let key = TuneKey::new(
        op.op_name().to_string(),
        op.runner_id(),
        device_label(),
        bucket,
        op.dtype_tag(),
    );
    store.load(&key)?.tactic.param(0)
}

/// Set `var` to `value` unless it is already set. Returns the applied pair.
fn apply_env_once(var: &'static str, value: i64) -> Option<(&'static str, String)> {
    if std::env::var_os(var).is_some() {
        return None;
    }
    let rendered = value.to_string();
    // SAFETY: `set_var` mutates the process-global environment and is unsound
    // only if another thread reads or writes the environment concurrently. Per
    // `apply_tuned_cuda_kernel_env`'s documented contract, callers invoke it
    // once at the top of `main`, before any model load, MLX op, or worker
    // thread touches the environment.
    unsafe { std::env::set_var(var, &rendered) };
    Some((var, rendered))
}
