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

//! TurboQuant KV cache compression.
//!
//! This module is the entry point for all TurboQuant / PolarQuant components
//! used by the KV cache compression pipeline. Sub-modules are added
//! incrementally as the sub-issues land:
//!
//! | Sub-module    | Sub-issue   | Status   | Description                          |
//! |---------------|-------------|----------|--------------------------------------|
//! | `codebook`    | B1   | done     | Lloyd-Max centroid generator         |
//! | `quant`       | B2/B4 | done | V-side + K-side PolarQuant pipeline |
//! | `pack3`       | B5   | done     | 3-bit pack/unpack helpers (24-bit grouping) |
//! | `quant3`      | B5   | done     | 3-bit V-side PolarQuant pipeline (asymmetric only) |
//! | `allowlist`   | B4   | done     | Per-model symmetric Turbo4 gating    |
//! | `boundary`    | B6   | done     | First/last layer V protection policy |
//! | `sparse_v`    | B8   | done     | Attention-gated V-dequant scaffold   |
//! | (more to come)| B10–B12     | pending  | paged KV / docs / turbo2             |
//!
//! # Usage by downstream sub-issues
//!
//! ```rust,ignore
//! use mlxcel_core::cache::turbo::codebook::optimal_codebook;
//!
//! let cb = optimal_codebook(4, 128);
//! ```

use std::sync::OnceLock;

pub mod allowlist;
pub mod boundary;
pub mod codebook;
pub mod pack3;
pub mod quant;
pub mod quant3;
pub mod sparse_v;

pub use allowlist::{
    is_symmetric_turbo_allowed, symmetric_turbo_warning_message, ALLOWED_SYMMETRIC_TURBO_FAMILIES,
};
pub use boundary::{
    boundary_mode_for, boundary_v_layers_from_env, is_boundary_layer, parse_boundary_v_str,
    resolve_boundary_count, resolve_layer_mode, resolve_layer_modes, BOUNDARY_V_ENV,
    BOUNDARY_V_ENV_ALT, DEFAULT_BOUNDARY_V_LAYERS,
};

// Re-export the most commonly used entry points for convenience
pub use codebook::{
    compute_centroids, nearest_centroid_indices, nearest_centroid_indices_with_boundaries,
    optimal_centroids, optimal_codebook, Codebook,
};
pub use quant::{
    dequantize_k_turbo4, dequantize_k_turbo4_rotated, dequantize_v_turbo4, generate_signs,
    quantize_k_turbo4, quantize_v_turbo4, turbo4_k_rotate, turbo4_v_rotate, TurboQuantParams,
    BLOCK_SIZE, K_BIT_WIDTH, K_SEED_OFFSET, V_BIT_WIDTH,
};

/// Default hot-tail threshold for `KVCacheMode::Turbo4Delegated`.
///
/// When the FP16 hot tail exceeds this many tokens, the oldest
/// [`DELEGATED_FOLD_BLOCK`]-token block is folded into cold packed storage on
/// the background re-compression stream. The TurboQuant+ MLX port uses 256
/// (a multiple of [`BLOCK_SIZE`]=32) which gives ample slack between folds at
/// the typical 32–256 tokens-per-second decode rate without bloating the hot
/// buffer footprint past a few hundred KB even on dense models.
pub const DELEGATED_HOT_THRESHOLD: i32 = 256;

/// Block size (tokens) folded from hot to cold per re-compression step.
///
/// Must be a multiple of [`BLOCK_SIZE`] (32) so the resulting cold append is
/// block-aligned. 128 keeps the per-fold cost bounded (one quantize + one
/// slice-update for ~64 packed bytes) while limiting the number of folds in
/// flight per decode burst.
pub const DELEGATED_FOLD_BLOCK: i32 = 128;

/// Maximum hot-tail capacity (tokens) before the next fold is forced
/// synchronously. Acts as a safety net so a slow background stream cannot let
/// the hot buffer grow without bound. Set to 4× [`DELEGATED_HOT_THRESHOLD`]
/// so a healthy stream never trips it; under contention we synchronize to
/// preserve the speed gate's invariant that hot reads stay fast.
pub const DELEGATED_HOT_MAX: i32 = DELEGATED_HOT_THRESHOLD * 4;

/// Environment variable that opts `KVCacheMode::Turbo4Delegated` into the
/// TurboQuant+ delegated-style decode path.
///
/// When enabled, the cache still builds packed cold V sidecars, but it keeps a
/// unified FP16 V buffer and routes decode attention through native SDPA. This
/// trades the compressed-only memory target for a speed reference path that
/// mirrors `references/turboquant_plus`' delegated KVCache architecture.
pub const DELEGATED_FP16_FAST_PATH_ENV: &str = "MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH";

static DELEGATED_FP16_FAST_PATH_ENABLED: OnceLock<bool> = OnceLock::new();

/// Packed sidecar maintenance policy for the opt-in delegated FP16 fast path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelegatedFp16SidecarPolicy {
    /// Build the initial sidecars in the prefill->decode handoff and keep
    /// folding hot-tail blocks during decode. This is the conservative
    /// default because packed sidecars stay close to the visible offset.
    #[default]
    Predecode,
    /// Do not build packed sidecars on the foreground generation path. Missing
    /// sidecars are compacted only by explicit preservation paths such as
    /// detach / prompt-cache donation.
    Lazy,
}

/// Environment variable controlling packed sidecar maintenance for
/// `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH=1`.
///
/// Accepted values:
/// - `predecode` / `eager` (default): compact before decode and periodically
///   during decode.
/// - `lazy` / `on-demand`: skip foreground generation compaction and compact
///   only when a preservation path explicitly asks for sidecars.
pub const DELEGATED_FP16_SIDECAR_POLICY_ENV: &str = "MLXCEL_TURBO4_DELEGATED_FP16_SIDECARS";

static DELEGATED_FP16_SIDECAR_POLICY: OnceLock<DelegatedFp16SidecarPolicy> = OnceLock::new();

/// Returns `true` iff Turbo4Delegated caches should keep unified FP16 V for
/// native-SDPA decode while maintaining packed sidecars for measurement and
/// later memory recovery work.
pub fn delegated_fp16_fast_path_enabled() -> bool {
    *DELEGATED_FP16_FAST_PATH_ENABLED.get_or_init(|| {
        match std::env::var(DELEGATED_FP16_FAST_PATH_ENV) {
            Ok(s) => matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            ),
            Err(_) => false,
        }
    })
}

/// Parse the delegated FP16 sidecar policy env var value.
pub fn parse_delegated_fp16_sidecar_policy(value: &str) -> Option<DelegatedFp16SidecarPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "predecode" | "pre-decode" | "eager" => Some(DelegatedFp16SidecarPolicy::Predecode),
        "lazy" | "on-demand" | "ondemand" => Some(DelegatedFp16SidecarPolicy::Lazy),
        _ => None,
    }
}

/// Returns the packed sidecar maintenance policy for delegated FP16 fast-path
/// caches. Unknown env values fall back to the conservative predecode policy.
pub fn delegated_fp16_sidecar_policy() -> DelegatedFp16SidecarPolicy {
    *DELEGATED_FP16_SIDECAR_POLICY.get_or_init(|| {
        std::env::var(DELEGATED_FP16_SIDECAR_POLICY_ENV)
            .ok()
            .and_then(|s| parse_delegated_fp16_sidecar_policy(&s))
            .unwrap_or_default()
    })
}
