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

//! Gates for the Inkling 2 Mini Metal kernels (issue #3359).
//!
//! Derived from the gates of mlx-vlm `ShortConvolution.__call__`,
//! `Attention._prepare_heads`, `SparseMoE.__call__` and
//! `grouped_expert_projections` (`mlx_vlm/models/inkling2/`, MIT). Each gate
//! returns `None` to keep the ops path, which every kernel matches bitwise:
//!
//! - the short convolution for bf16 `x [B <= 4, L <= 8, C % 128 == 0]` with
//!   kernel size 4 and an f32 history;
//! - attention head preparation on a looped model (Mini) with the gamma-free
//!   value norm, for bf16 `[1, 1..=8, heads, 128]` heads and `d_rel == 16`;
//! - the grouped expert matvec on a looped model, only inside a speculative
//!   verify forward ([`mlxcel_core::cache::speculative_append_active`], set
//!   by `Engine::verify`), for bf16 `[1, 2..=8, hidden]` and native NVFP4 or
//!   MXFP8 planes without biases.
//!
//! mlxcel has no Inkling 2 training or sharded path, so upstream's
//! `not self.training` and `sharding_group is None` terms always hold here.
//!
//! Every dispatch that reaches a kernel is counted per thread (read by the
//! tests through `dispatch_counts`), and the first one of each kernel is
//! logged at debug level, as is each model's gate decision at load, so
//! `RUST_LOG=mlxcel::models::inkling2=debug` shows which kernels a run took
//! (#3373: Small takes the short convolution and neither loop kernel).

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::Inkling2TextConfig;
use crate::models::switch_layers::{SwitchGLU, SwitchLinear};

/// Kill switch: `0` (or `false`/`off`/`no`) keeps all three kernels off, so
/// every Inkling 2 forward runs the ops path. Read when a model is built.
pub const KERNELS_ENV: &str = "MLXCEL_INKLING2_KERNELS";

/// Pure decision behind [`enabled`]: unset or any other value is on.
pub(crate) fn enabled_from(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

/// Whether [`KERNELS_ENV`] leaves the kernels on.
pub(crate) fn enabled() -> bool {
    enabled_from(std::env::var(KERNELS_ENV).ok().as_deref())
}

/// The three kernels, as indices into [`dispatch_counts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    ShortConvolution = 0,
    AttentionHeads = 1,
    GroupedExperts = 2,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Self::ShortConvolution => "short convolution",
            Self::AttentionHeads => "attention head preparation",
            Self::GroupedExperts => "grouped expert matvec",
        }
    }
}

/// Whether each kernel has been logged as taken in this process.
static LOGGED: [AtomicBool; 3] = [
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];

thread_local! {
    /// Dispatches on this thread: a forward runs on one thread, so a test
    /// reads its own counts without racing tests on other threads.
    static DISPATCHES: Cell<[u64; 3]> = const { Cell::new([0; 3]) };
}

/// Count one forward that ran `kernel` instead of the ops path.
pub(super) fn record_dispatch(kernel: Kernel) {
    DISPATCHES.with(|counts| {
        let mut value = counts.get();
        value[kernel as usize] += 1;
        counts.set(value);
    });
    if !LOGGED[kernel as usize].swap(true, Ordering::Relaxed) {
        tracing::debug!("Inkling 2 {} kernel dispatched (first use)", kernel.name());
    }
}

/// Dispatch counts on the calling thread, indexed by [`Kernel`].
#[cfg(test)]
pub(crate) fn dispatch_counts() -> [u64; 3] {
    DISPATCHES.with(Cell::get)
}

/// Log which kernels a model built from `config` can take, once at load.
pub(crate) fn log_model_gates(config: &Inkling2TextConfig) {
    let enabled = enabled();
    let looped = loop_kernels_wanted(config, enabled);
    tracing::debug!(
        "Inkling 2 kernels: short convolution {}, attention head preparation {}, grouped expert matvec {}",
        if enabled { "on" } else { "off" },
        if looped && config.use_value_norm_no_gamma {
            "on"
        } else {
            "off"
        },
        if looped {
            "on (verify forwards)"
        } else {
            "off"
        },
    );
}

fn shape(x: &MlxArray) -> Vec<i32> {
    mlxcel_core::array_shape(x)
}

/// The short-convolution shape gate (upstream: bf16, kernel 4, `B <= 4`,
/// `L <= 8`, `C % 128 == 0`, Metal on the GPU).
pub(crate) fn short_convolution_applies(x: &MlxArray, history: &MlxArray, kernel: i32) -> bool {
    let s = shape(x);
    s.len() == 3
        && mlxcel_core::array_dtype(x) == dtype::BFLOAT16
        && mlxcel_core::array_dtype(history) == dtype::FLOAT32
        && kernel == 4
        && s[0] <= 4
        && s[1] <= 8
        && s[2] % 128 == 0
        && mlxcel_core::inkling2_kernels::available()
}

/// Whether a model built from `config` wants the looped-model kernels (head
/// preparation and grouped experts): Mini only, as upstream gates them on
/// `loop_config`, and only while [`KERNELS_ENV`] leaves them on.
pub(crate) fn loop_kernels_wanted(config: &Inkling2TextConfig, enabled: bool) -> bool {
    enabled && config.loop_config.is_some()
}

/// The head-preparation shape gate, after the model-level one (looped model,
/// value norm without gamma, kernels on) the caller holds.
pub(crate) fn attention_heads_apply(q: &MlxArray, r: &MlxArray) -> bool {
    let (qs, rs) = (shape(q), shape(r));
    qs.len() == 4
        && qs[0] == 1
        && (1..=8).contains(&qs[1])
        && qs[3] == 128
        && rs.len() == 4
        && rs[3] == 16
        && mlxcel_core::array_dtype(q) == dtype::BFLOAT16
        && mlxcel_core::inkling2_kernels::available()
}

/// l2-clipped `(q, k, r)` and RMS-normed `v` from the fused kernel, or `None`
/// where [`attention_heads_apply`] or the launcher's contract refuses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_attention_heads(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    r: &MlxArray,
    q_scale: &MlxArray,
    r_scale: &MlxArray,
    clip: f32,
    rel_clip: f32,
    eps: f32,
) -> Option<[UniquePtr<MlxArray>; 4]> {
    if !attention_heads_apply(q, r) {
        return None;
    }
    let out = mlxcel_core::inkling2_kernels::prepare_attention_heads(
        q, k, v, r, q_scale, r_scale, clip, rel_clip, eps,
    )?;
    record_dispatch(Kernel::AttentionHeads);
    Some(out)
}

/// `(bits, group_size)` of a plane the grouped kernel decodes: native NVFP4
/// (an optional MLX-native global scale) or MXFP8, no biases and no output
/// multiplier, input width a multiple of 32 and output width of 8.
fn grouped_plane(linear: &SwitchLinear) -> Option<(i32, i32)> {
    let SwitchLinear::Quantized {
        weight,
        scales,
        biases: None,
        group_size,
        bits,
        mode,
        inv_global_scale: None,
        native_global_scale,
    } = linear
    else {
        return None;
    };
    let supported = match (mode.as_str(), *bits, *group_size) {
        ("nvfp4", 4, 16) => true,
        ("mxfp8", 8, 32) => native_global_scale.is_none(),
        _ => false,
    };
    let (w, s) = (shape(weight), shape(scales));
    let ok = supported
        && w.len() == 3
        && s.len() == 3
        && mlxcel_core::array_dtype(weight) == dtype::UINT32
        && mlxcel_core::array_dtype(scales) == dtype::UINT8
        && (w[2] * 32 / bits) % 32 == 0
        && w[1] % 8 == 0
        && s[2] * group_size == w[2] * 32 / bits;
    ok.then_some((*bits, *group_size))
}

/// Whether every projection of `experts` is a plane the grouped kernel takes
/// (checked once at load; upstream checks per call).
pub(crate) fn grouped_experts_supported(experts: &SwitchGLU) -> bool {
    experts
        .projections()
        .iter()
        .all(|linear| grouped_plane(linear).is_some())
}

/// The per-forward grouped-expert gate: a speculative verify forward of
/// `x [1, 2..=8, hidden]` bf16 with fewer than 64 experts per token.
pub(crate) fn grouped_experts_apply(x: &MlxArray, top_k: i32) -> bool {
    let s = shape(x);
    s.len() == 3
        && s[0] == 1
        && (2..=8).contains(&s[1])
        && top_k < 64
        && mlxcel_core::array_dtype(x) == dtype::BFLOAT16
        && mlxcel_core::cache::speculative_append_active()
        && mlxcel_core::inkling2_kernels::available()
}

fn grouped_projection(
    x: &MlxArray,
    linear: &SwitchLinear,
    experts: &MlxArray,
    source: &MlxArray,
    destination: &MlxArray,
) -> Option<UniquePtr<MlxArray>> {
    let (bits, group_size) = grouped_plane(linear)?;
    let SwitchLinear::Quantized {
        weight,
        scales,
        native_global_scale,
        ..
    } = linear
    else {
        return None;
    };
    mlxcel_core::inkling2_kernels::grouped_expert_qmv(
        x,
        weight,
        scales,
        native_global_scale.as_deref(),
        experts,
        source,
        destination,
        bits,
        group_size,
    )
}

/// Routed expert outputs `[tokens, k, hidden]` for `x [tokens, hidden]` and
/// `indices [tokens, k]`, with each expert's weights decoded once for all its
/// routes: the routes are sorted by expert, up and gate come out in sorted
/// order, and down writes back in router order. `None` keeps the gather path.
pub(crate) fn grouped_expert_projections(
    experts: &SwitchGLU,
    x: &MlxArray,
    indices: &MlxArray,
) -> Option<UniquePtr<MlxArray>> {
    let idx = shape(indices);
    let (tokens, top_k) = (idx[0], idx[1]);
    let hidden = shape(x)[1];
    let routes = mlxcel_core::astype(
        &mlxcel_core::reshape(indices, &[tokens * top_k]),
        dtype::UINT32,
    );
    let order = mlxcel_core::astype(&mlxcel_core::argsort(&routes, 0), dtype::UINT32);
    let sorted = mlxcel_core::take(&routes, &order, 0);
    let source =
        mlxcel_core::floor_divide(&order, &mlxcel_core::from_slice_u32(&[top_k as u32], &[1]));
    let identity = mlxcel_core::arange_u32(tokens * top_k);
    let [gate_proj, up_proj, down_proj] = experts.projections();
    let up = grouped_projection(x, up_proj, &sorted, &source, &identity)?;
    let gate = grouped_projection(x, gate_proj, &sorted, &source, &identity)?;
    let activated = experts.activate(&gate, &up);
    let down = grouped_projection(&activated, down_proj, &sorted, &identity, &order)?;
    record_dispatch(Kernel::GroupedExperts);
    Some(mlxcel_core::reshape(&down, &[tokens, top_k, hidden]))
}

#[cfg(test)]
#[path = "kernel_grouped_tests.rs"]
mod grouped_tests;
#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
