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

//! Causal and sliding-window prefill attention for layers that may carry
//! attention sinks (issue #3098).
//!
//! #3051 and #3072 moved the decoders without sinks onto MLX's causal mode
//! and native `window_size`, leaving the explicit `[q_len, k_len]` f32 mask
//! only where MLX would take its graph fallback anyway. The sink models could
//! not follow, because the sink SDPA entry took only an array mask. Upstream
//! `fast::scaled_dot_product_attention` accepts `mask_mode = "causal"` and
//! `window_size` together with `sinks`, and
//! [`ffi::ffi_fast_scaled_dot_product_attention_with_sinks_causal`] exposes
//! exactly that. This module decides when a layer may use it.
//!
//! The rule is the one [`utils::prefill_causal_mask`] and
//! [`utils::prefill_sliding_window_mask`] apply: a single unpadded row, no
//! caller mask, no speculative verify block, `MLXCEL_PREFILL_CAUSAL_MODE` on,
//! and MLX's own verdict that the maskless call fuses at this shape (asked
//! with sinks when the layer has them). Anything else keeps the explicit band.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ffi::{self, MlxArray};
use crate::hardware::gpu_backend_kind;
use crate::window_sdpa::{self, WindowSdpaExtras, WindowSdpaShape};
use crate::{UniquePtr, layers, utils};

/// Whether MLX runs a causal SDPA over `shape` fused without a mask array:
/// causal mode while every key is inside `window` (or there is no window,
/// `window <= 0`), the native `window_size` entry once the keys outgrow it.
/// `sinks` asks about the call with attention sinks. Metal only; `false`
/// elsewhere. Every verdict is memoized per thread.
///
/// Used by: [`utils::prefill_causal_mask`], [`utils::prefill_sliding_window_mask`],
/// [`causal_prefill_attention`].
pub fn maskless_route_is_fused(shape: &WindowSdpaShape, window: i32, sinks: bool) -> bool {
    let within_window = window <= 0 || shape.k_len <= window;
    match (within_window, sinks) {
        (true, false) => window_sdpa::sdpa_mode_is_fused(shape, false, true),
        (true, true) => window_sdpa::sinks_causal_arm_is_fused(shape, -1),
        (false, false) => window_sdpa::plan(shape, window, WindowSdpaExtras::default()).is_native(),
        (false, true) => window_sdpa::plan_with(
            shape,
            window,
            WindowSdpaExtras::default(),
            window_sdpa::native_window_sdpa_mode(),
            gpu_backend_kind(),
            window_sdpa::sinks_causal_arm_is_fused,
        )
        .is_native(),
    }
}

/// The `window_size` MLX's causal entry takes for `k_len` keys: the window
/// once the keys outgrow it, `-1` (none) while they fit or with no window.
pub fn causal_window_arg(window: i32, k_len: i32) -> i32 {
    if window > 0 && k_len > window {
        window
    } else {
        -1
    }
}

/// Multi-token causal attention over the keys a cache returned, with the
/// queries aligned to the suffix of the keys, a sliding `window` when
/// positive, and optional per-head `sinks`.
///
/// When the forward may take MLX's own masking (see the module docs) this
/// runs MLX's causal mode, plus `window_size` once the keys outgrow the
/// window, with no mask array. Otherwise it builds the band the layer used to
/// build itself, `create_causal_mask_with_window_full(q_len, k_len - q_len,
/// window)`, and runs the array entry: the sink SDPA with `sinks`, else
/// [`layers::attention`]. Both routes attend the same keys.
///
/// A layer whose model left its shared prefill mask to MLX (a `None` from
/// [`utils::prefill_causal_mask`] or [`utils::prefill_sliding_window_mask`]
/// with sink geometry) calls this for its multi-token forward. A layer that
/// builds its own mask per forward calls it instead of building one.
///
/// Used by: GptOss (full and sliding layers), Laguna, FalconOcr, MiMoV2Flash
/// (sink layers), DeepSeekV4 (local layers), HyV4 (the maskless multi-token
/// step of its sink MLA).
pub fn causal_prefill_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    window: i32,
    sinks: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let shape = WindowSdpaShape::of(q, k, v);
    let maskless = utils::prefill_uses_causal_mode(shape.q_len, shape.batch, false)
        && maskless_route_is_fused(&shape, window, sinks.is_some());
    if maskless {
        match sinks {
            None => return crate::maskless_prefill_attention(q, k, v, scale, window),
            Some(sinks) => {
                let window_arg = causal_window_arg(window, shape.k_len);
                match ffi::ffi_fast_scaled_dot_product_attention_with_sinks_causal(
                    q, k, v, scale, window_arg, sinks,
                ) {
                    Ok(out) => return out,
                    Err(err) => warn_causal_sinks_fallback(&err.to_string()),
                }
            }
        }
    }
    let mask = utils::create_causal_mask_with_window_full(
        shape.q_len,
        (shape.k_len - shape.q_len).max(0),
        (window > 0).then_some(window),
    );
    match sinks {
        // SAFETY: `mask` and `sinks` are live arrays borrowed for the call.
        Some(sinks) => unsafe {
            ffi::fast_scaled_dot_product_attention_with_sinks(
                q,
                k,
                v,
                scale,
                mask.as_ref().expect("prefill mask") as *const MlxArray,
                sinks as *const MlxArray,
            )
        },
        None => layers::attention(q, k, v, scale, mask.as_ref(), 0.0, window),
    }
}

/// The sink causal entry rejected a call the route allowed. MLX validated
/// these shapes when it answered the fused probe, so this is not expected;
/// the explicit band keeps the forward correct, and the first occurrence is
/// reported once per process.
fn warn_causal_sinks_fallback(err: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[mlxcel] causal sink SDPA failed ({err}); using the explicit prefill mask instead"
        );
    }
}
