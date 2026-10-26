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

//! Native sliding-window SDPA route for Metal prefill (issue #2621).
//!
//! A causal multi-query prefill over a sliding window of `W` keys used to be
//! expressed as an explicit additive `[q_len, k_len]` mask
//! (`utils::create_causal_mask_with_window_full`) handed to MLX's array-mask
//! SDPA. Upstream MLX now takes the window as an argument
//! (ml-explore/mlx#4734): `mask_mode = "causal"` plus `window_size = W` makes
//! query row `r` (the queries being the suffix of the keys) see key `c` iff
//! `c <= r && r - c < W`, the current position plus up to `W - 1` earlier
//! ones, which is the band the explicit mask encodes. On Metal that lets the
//! steel attention kernel skip the key blocks outside the band instead of
//! reading a dense mask over all of them.
//!
//! This module is the one place that decides between the two routes. The
//! decision is explicit about what the call carries: a softcap, an attention
//! sink, or any extra mask array (padding, segments, per-head, cache
//! validity) keeps the legacy route, because the native entry cannot take an
//! array mask next to causal mode and the band is all it can express. CPU,
//! CUDA and ROCm keep their existing routes unconditionally: a positive
//! window there selects MLX's graph fallback.
//!
//! Two switches, both read once per process:
//!
//! - `MLXCEL_NATIVE_WINDOW_SDPA`: `0`/`off` keeps the legacy mask route
//!   everywhere, `1`/`on` takes the native route wherever the static
//!   eligibility holds (even where MLX would run its graph fallback, which
//!   is what an A/B needs), `auto` takes it only where MLX also reports a
//!   fused kernel for the shape. Unset is [`DEFAULT_MODE`].
//! - `MLXCEL_LOG_WINDOW_SDPA`: `1` prints a sampled dispatch line per call to
//!   stderr, `all` prints every one. The line carries the route, the reason,
//!   MLX's fused-kernel verdict and the shapes, which is the evidence a
//!   benchmark report quotes to show the route ran.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ffi::{self, MlxArray};
use crate::hardware::{GpuBackendKind, gpu_backend_kind};
use crate::{UniquePtr, dtype, layers, utils};

/// Which SDPA entry a causal window prefill takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowSdpaRoute {
    /// Upstream `window_size` with `mask_mode = "causal"`, no mask array.
    Native,
    /// Explicit additive window mask through the array-mask SDPA.
    Legacy,
}

/// Process-wide selector from `MLXCEL_NATIVE_WINDOW_SDPA`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeWindowSdpaMode {
    /// Legacy route everywhere.
    Off,
    /// Native route wherever the static eligibility holds.
    On,
    /// Native route where the static eligibility holds and MLX reports a
    /// fused kernel for the shape.
    Auto,
}

/// The mode an unset `MLXCEL_NATIVE_WINDOW_SDPA` selects.
///
/// Measured on an M1 Ultra with the cached head-dim 256 sliding-window
/// checkpoints (`docs/benchmark_results/native-window-sdpa-m1ultra-2026-10-26.md`).
/// `Auto` engages the native entry only where MLX's own dispatch reports a
/// fused kernel, so on shapes MLX would route to its graph fallback the
/// legacy mask route runs unchanged.
pub const DEFAULT_MODE: NativeWindowSdpaMode = NativeWindowSdpaMode::Auto;

/// Query lengths at or below this keep the legacy route. MLX serves them with
/// its vector kernels, which is the decode and speculative-verify geometry;
/// the issue scopes the native route to prefill, and the verify block's
/// byte-identity contract with decode is measured, not assumed.
pub const MIN_NATIVE_QUERY_LEN: i32 = 9;

/// Parse a `MLXCEL_NATIVE_WINDOW_SDPA` value. Unset or unrecognised is
/// [`DEFAULT_MODE`].
pub fn parse_mode(raw: Option<&str>) -> NativeWindowSdpaMode {
    match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("0" | "false" | "no" | "off" | "legacy") => NativeWindowSdpaMode::Off,
        Some("1" | "true" | "yes" | "on" | "native" | "force") => NativeWindowSdpaMode::On,
        Some("auto") => NativeWindowSdpaMode::Auto,
        _ => DEFAULT_MODE,
    }
}

/// The process-wide mode, read once.
pub fn native_window_sdpa_mode() -> NativeWindowSdpaMode {
    static MODE: OnceLock<NativeWindowSdpaMode> = OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var("MLXCEL_NATIVE_WINDOW_SDPA").ok().as_deref()))
}

/// The shape and dtype facts the route decision reads, so a model can plan a
/// prefill before it has the projected arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSdpaShape {
    pub batch: i32,
    pub q_heads: i32,
    pub kv_heads: i32,
    pub q_len: i32,
    pub k_len: i32,
    pub head_dim: i32,
    pub value_dim: i32,
    pub q_dtype: i32,
    pub k_dtype: i32,
    pub v_dtype: i32,
}

impl WindowSdpaShape {
    /// Read the facts off `[B, H, L, D]` query and `[B, Hkv, S, D]` key and
    /// value arrays. Shape reads only, no sync.
    pub fn of(q: &MlxArray, k: &MlxArray, v: &MlxArray) -> Self {
        let q_shape = ffi::array_shape(q);
        let k_shape = ffi::array_shape(k);
        let v_shape = ffi::array_shape(v);
        Self {
            batch: q_shape[0],
            q_heads: q_shape[1],
            kv_heads: k_shape[1],
            q_len: q_shape[2],
            k_len: k_shape[2],
            head_dim: q_shape[3],
            value_dim: v_shape[3],
            q_dtype: ffi::array_dtype(q),
            k_dtype: ffi::array_dtype(k),
            v_dtype: ffi::array_dtype(v),
        }
    }

    /// The facts of a text prefill whose projections have not run yet: one
    /// activation dtype for q, k and v and one head dim for query and value.
    pub fn prefill(
        batch: i32,
        q_heads: i32,
        kv_heads: i32,
        q_len: i32,
        k_len: i32,
        head_dim: i32,
        activation_dtype: i32,
    ) -> Self {
        Self {
            batch,
            q_heads,
            kv_heads,
            q_len,
            k_len,
            head_dim,
            value_dim: head_dim,
            q_dtype: activation_dtype,
            k_dtype: activation_dtype,
            v_dtype: activation_dtype,
        }
    }

    fn q_shape(&self) -> [i32; 4] {
        [self.batch, self.q_heads, self.q_len, self.head_dim]
    }

    fn k_shape(&self) -> [i32; 4] {
        [self.batch, self.kv_heads, self.k_len, self.head_dim]
    }

    fn v_shape(&self) -> [i32; 4] {
        [self.batch, self.kv_heads, self.k_len, self.value_dim]
    }
}

/// Everything a call carries beyond "causal plus window". Any of these keeps
/// the legacy route, because the native entry cannot express it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WindowSdpaExtras {
    /// Attention logit softcap (Gemma 2); the composite needs the scores.
    pub softcap: f32,
    /// An explicit mask array (padding, segments, per-head, cache validity,
    /// or anything else whose meaning is not read off its shape).
    pub array_mask: bool,
    /// Attention sinks (gpt-oss); kept on the sinks entry.
    pub sinks: bool,
}

/// Head dims upstream MLX's full attention kernel handles for `q_len > 8`
/// (`has_fused_kernel` in
/// <https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp>).
/// Outside this set MLX takes its graph fallback whatever the mask mode, so
/// the native route would only re-spell the mask; the legacy route stays.
const FULL_KERNEL_HEAD_DIMS: [i32; 7] = [64, 72, 80, 96, 128, 192, 256];

/// Whether a causal window call is statically eligible for the native route,
/// or why it is not. Pure: no arrays, no device query beyond `backend`.
pub fn native_window_eligibility(
    shape: &WindowSdpaShape,
    window: i32,
    extras: WindowSdpaExtras,
    backend: GpuBackendKind,
) -> Result<(), &'static str> {
    if backend != GpuBackendKind::Metal {
        return Err("backend_not_metal");
    }
    if window <= 0 {
        return Err("window_disabled");
    }
    if extras.softcap > 0.0 {
        return Err("softcap");
    }
    if extras.array_mask {
        return Err("array_mask");
    }
    if extras.sinks {
        return Err("sinks");
    }
    if shape.q_len < MIN_NATIVE_QUERY_LEN {
        return Err("query_len_le_8");
    }
    if shape.q_len > shape.k_len {
        return Err("query_longer_than_keys");
    }
    if shape.k_len <= window {
        return Err("keys_within_window");
    }
    if shape.kv_heads <= 0 || shape.q_heads % shape.kv_heads != 0 {
        return Err("head_count_ratio");
    }
    if shape.head_dim != shape.value_dim || !FULL_KERNEL_HEAD_DIMS.contains(&shape.head_dim) {
        return Err("head_dim");
    }
    let attention_dtype = [shape.q_dtype, shape.k_dtype, shape.v_dtype];
    if !attention_dtype
        .iter()
        .all(|d| matches!(*d, dtype::FLOAT16 | dtype::BFLOAT16 | dtype::FLOAT32))
    {
        return Err("dtype");
    }
    Ok(())
}

/// A route decision with the facts that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSdpaPlan {
    pub route: WindowSdpaRoute,
    /// `"eligible"` for a native route; the exclusion otherwise, or
    /// `"mode_off"` / `"graph_fallback"` when the mode declined an eligible
    /// shape.
    pub reason: &'static str,
    /// MLX's fused-kernel verdict for the native arm, where it was asked.
    pub fused: Option<bool>,
}

impl WindowSdpaPlan {
    pub fn is_native(&self) -> bool {
        self.route == WindowSdpaRoute::Native
    }
}

/// MLX's own verdict on whether the native arm over `shape` runs a fused
/// kernel. Metal only; `false` elsewhere.
pub fn native_arm_is_fused(shape: &WindowSdpaShape, window: i32) -> bool {
    ffi::sdpa_uses_fused_kernel(
        &shape.q_shape(),
        &shape.k_shape(),
        &shape.v_shape(),
        shape.q_dtype,
        shape.k_dtype,
        shape.v_dtype,
        false,
        true,
        window,
    )
}

/// MLX's verdict for the legacy explicit-mask arm over `shape`, for reports.
pub fn legacy_arm_is_fused(shape: &WindowSdpaShape) -> bool {
    ffi::sdpa_uses_fused_kernel(
        &shape.q_shape(),
        &shape.k_shape(),
        &shape.v_shape(),
        shape.q_dtype,
        shape.k_dtype,
        shape.v_dtype,
        true,
        false,
        -1,
    )
}

/// Decide the route under an explicit mode, backend and fused probe, so the
/// decision can be tested without the process environment or a GPU.
pub fn plan_with(
    shape: &WindowSdpaShape,
    window: i32,
    extras: WindowSdpaExtras,
    mode: NativeWindowSdpaMode,
    backend: GpuBackendKind,
    fused_probe: impl Fn(&WindowSdpaShape, i32) -> bool,
) -> WindowSdpaPlan {
    if let Err(reason) = native_window_eligibility(shape, window, extras, backend) {
        return WindowSdpaPlan {
            route: WindowSdpaRoute::Legacy,
            reason,
            fused: None,
        };
    }
    match mode {
        NativeWindowSdpaMode::Off => WindowSdpaPlan {
            route: WindowSdpaRoute::Legacy,
            reason: "mode_off",
            fused: None,
        },
        NativeWindowSdpaMode::On => WindowSdpaPlan {
            route: WindowSdpaRoute::Native,
            reason: "eligible",
            fused: None,
        },
        NativeWindowSdpaMode::Auto => {
            let fused = fused_probe(shape, window);
            WindowSdpaPlan {
                route: if fused {
                    WindowSdpaRoute::Native
                } else {
                    WindowSdpaRoute::Legacy
                },
                reason: if fused { "eligible" } else { "graph_fallback" },
                fused: Some(fused),
            }
        }
    }
}

/// Decide the route for the running process: its mode, its backend and
/// MLX's dispatch verdict.
pub fn plan(shape: &WindowSdpaShape, window: i32, extras: WindowSdpaExtras) -> WindowSdpaPlan {
    plan_with(
        shape,
        window,
        extras,
        native_window_sdpa_mode(),
        gpu_backend_kind(),
        native_arm_is_fused,
    )
}

/// Keys a `RotatingKVCache` returns for a multi-token append of `seq_len`
/// rows after `live_len` tokens: at most `window - 1` retained prior keys
/// plus every new one (the #678 contract that
/// `utils::create_sliding_window_prefill_mask` sizes its key axis by).
pub fn rotating_prefill_key_len(seq_len: i32, live_len: i32, window: i32) -> i32 {
    let kept_prior = if live_len <= 0 {
        0
    } else {
        live_len.min((window - 1).max(0))
    };
    kept_prior + seq_len
}

/// Dispatch counts since process start, for route evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WindowSdpaCounts {
    pub native: u64,
    pub legacy: u64,
}

static NATIVE_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static LEGACY_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// How many causal window prefills took each route so far.
pub fn dispatch_counts() -> WindowSdpaCounts {
    WindowSdpaCounts {
        native: NATIVE_DISPATCHES.load(Ordering::Relaxed),
        legacy: LEGACY_DISPATCHES.load(Ordering::Relaxed),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogMode {
    Off,
    Sampled,
    All,
}

fn log_mode() -> LogMode {
    static MODE: OnceLock<LogMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("MLXCEL_LOG_WINDOW_SDPA")
            .map(|v| v.trim().to_ascii_lowercase())
            .ok()
            .as_deref()
        {
            Some("1" | "true" | "yes" | "on" | "sample" | "sampled") => LogMode::Sampled,
            Some("all" | "full") => LogMode::All,
            _ => LogMode::Off,
        }
    })
}

fn record_dispatch(shape: &WindowSdpaShape, window: i32, plan: &WindowSdpaPlan) {
    let counter = match plan.route {
        WindowSdpaRoute::Native => &NATIVE_DISPATCHES,
        WindowSdpaRoute::Legacy => &LEGACY_DISPATCHES,
    };
    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
    let should_log = match log_mode() {
        LogMode::Off => false,
        LogMode::Sampled => count <= 16 || count.is_multiple_of(256),
        LogMode::All => true,
    };
    if !should_log {
        return;
    }
    let fused = plan
        .fused
        .unwrap_or_else(|| native_arm_is_fused(shape, window));
    eprintln!(
        "[mlxcel][window-sdpa] route={} dispatch={} reason={} native_fused={} legacy_fused={} q=[{}, {}, {}, {}] k=[{}, {}, {}, {}] window={} dtype={}",
        match plan.route {
            WindowSdpaRoute::Native => "native",
            WindowSdpaRoute::Legacy => "legacy",
        },
        count,
        plan.reason,
        fused,
        legacy_arm_is_fused(shape),
        shape.batch,
        shape.q_heads,
        shape.q_len,
        shape.head_dim,
        shape.batch,
        shape.kv_heads,
        shape.k_len,
        shape.head_dim,
        window,
        shape.q_dtype,
    );
}

/// Causal attention over a sliding window for a multi-query prefill whose
/// keys outgrow the window (`q_len > 1`, `k_len > window`), with no softcap,
/// no sinks and no extra mask. The queries are aligned to the suffix of the
/// keys the cache returned, so `k_len - q_len` prior keys are visible to the
/// first row's window; this is the cache layout, not the generation offset.
///
/// Used by: `crate::causal_attention` (the over-window prefill branch, which
/// `RotatingKVCache::attend` for Gemma 3 and Gemma 4's unmasked text prefill
/// reach; Ministral 3 can reach it only when its own prefill mask is absent).
pub fn causal_window_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    window: i32,
) -> UniquePtr<MlxArray> {
    causal_window_attention_with_mode(q, k, v, scale, window, native_window_sdpa_mode())
}

/// [`causal_window_attention`] under an explicit mode, for A/B harnesses and
/// tests that compare the two routes inside one process.
pub fn causal_window_attention_with_mode(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    window: i32,
    mode: NativeWindowSdpaMode,
) -> UniquePtr<MlxArray> {
    let shape = WindowSdpaShape::of(q, k, v);
    let plan = plan_with(
        &shape,
        window,
        WindowSdpaExtras::default(),
        mode,
        gpu_backend_kind(),
        native_arm_is_fused,
    );
    record_dispatch(&shape, window, &plan);
    match plan.route {
        WindowSdpaRoute::Native => native_causal_window_attention(q, k, v, scale, window),
        WindowSdpaRoute::Legacy => legacy_causal_window_attention(q, k, v, scale, window),
    }
}

/// The native entry, unconditionally. Callers that want the route decision
/// go through [`causal_window_attention`]; this is for the microbenchmark
/// and the parity tests.
pub fn native_causal_window_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    window: i32,
) -> UniquePtr<MlxArray> {
    ffi::ffi_fast_scaled_dot_product_attention_causal_window(q, k, v, scale, window)
}

/// The legacy entry, unconditionally: the full-width windowed-causal mask
/// over every returned key, through the array-mask SDPA. This is the
/// pre-#2621 over-window prefill path of `crate::causal_attention`, kept
/// byte-for-byte so a legacy-mode process matches the previous release.
pub fn legacy_causal_window_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    window: i32,
) -> UniquePtr<MlxArray> {
    let mask = legacy_causal_window_mask(q, k, window);
    layers::attention(q, k, v, scale, Some(mask.as_ref().unwrap()), 0.0, window)
}

/// The explicit mask the legacy route builds: `[q_len, k_len]`, queries
/// aligned to the suffix of the keys.
pub fn legacy_causal_window_mask(q: &MlxArray, k: &MlxArray, window: i32) -> UniquePtr<MlxArray> {
    let q_len = ffi::array_shape(q)[2];
    let k_len = ffi::array_shape(k)[2];
    let offset = (k_len - q_len).max(0);
    utils::create_causal_mask_with_window_full(q_len, offset, Some(window))
}
