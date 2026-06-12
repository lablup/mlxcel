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
//
// Portions of this file are derived from turboquant_plus
// (https://github.com/TheTom/turboquant_plus), Copyright 2026 Tom Turney,
// licensed under the Apache License, Version 2.0. See the top-level NOTICE
// file for the attribution carried forward under Apache-2.0 Section 4(d).

//! Boundary-V layer protection (B6).
//!
//! TurboQuant+ measured that the first 2 and last 2 transformer layers'
//! V quantization error contributes disproportionately to perplexity
//! regression: keeping these 4 layers at higher precision recovers
//! 37–91% of the quality gap at zero speed cost. See
//! https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md
//! for the original measurements (LA-V7 policy: q8_0 boundary V, turbo
//! middle V, K unchanged).
//!
//! In mlxcel we approximate the LA-V7 policy by **upgrading the entire
//! per-layer cache mode** for boundary layers from a turbo V mode to
//! `KVCacheMode::Fp16` (uncompressed both K and V). This is conservative
//! — it spends a few extra bytes per boundary layer compared to the
//! paper's q8_0-V boundary — but matches the issue's design hint
//! ("by default, the first 2 and last 2 layers run V at Int8, or Fp16
//! if Int8 not in use") and keeps the K side at least as good as the
//! nominal mode requires (since FP16 is always at least as accurate as
//! any quantized K).
//!
//! The policy is **inert when no Turbo4 mode is configured**: caches
//! constructed with `KVCacheMode::Fp16` or `KVCacheMode::Int8` are
//! returned unchanged regardless of layer index, so existing call sites
//! see bit-identical behavior.
//!
//! # Configuration
//!
//! - `MLXCEL_KV_BOUNDARY_V_LAYERS` env var (also accepted as `MLXCEL_TURBO_BOUNDARY_V` for compatibility with the wording): integer count of boundary layers to protect on each
//!   end. Default `2`. Setting `0` disables the policy.
//! - The count is clamped to `min(boundary, n_layers / 2)` so a
//!   too-aggressive setting on a shallow model doesn't end up
//!   protecting *every* layer, which would defeat the compression
//!   entirely. Negative inputs are treated as `0`. Non-numeric inputs
//!   fall back to the default with no warning (the env var is an
//!   advanced knob; users who set bogus values get the safe default).

use super::super::KVCacheMode;

/// Default boundary count when no env var is set. Matches the LA-V7
/// policy from `layer-aware-v-compression.md` (first 2 + last 2).
pub const DEFAULT_BOUNDARY_V_LAYERS: i32 = 2;

/// Environment variable controlling the number of boundary layers
/// protected on each end of the transformer stack.
///
/// Accepts a non-negative integer. `0` disables the policy. Out-of-range
/// or non-numeric values fall back to [`DEFAULT_BOUNDARY_V_LAYERS`].
pub const BOUNDARY_V_ENV: &str = "MLXCEL_KV_BOUNDARY_V_LAYERS";

/// Alternative spelling of [`BOUNDARY_V_ENV`] for users who pattern-match
/// the issue title's "boundary V" wording. Both names resolve to the
/// same effective count; the first one set wins.
pub const BOUNDARY_V_ENV_ALT: &str = "MLXCEL_TURBO_BOUNDARY_V";

/// Read the configured boundary count from the process environment.
///
/// Returns the parsed value, clamped to `>= 0`. Non-numeric or unset
/// values resolve to [`DEFAULT_BOUNDARY_V_LAYERS`]. The actual policy
/// application also clamps against `n_layers / 2` (see
/// [`resolve_boundary_count`]).
///
/// Used by: `CxxGenerator::new_with_kv_mode` (cache construction) and
/// the boundary-policy unit tests.
pub fn boundary_v_layers_from_env() -> i32 {
    // Primary env var wins; fall back to the alt name; finally the
    // hardcoded default. This keeps the docs/turbo-kv-cache.md story
    // single-knobbed for users while remaining backward-compatible
    // with whichever wording lands in tests/CI scripts.
    for var in [BOUNDARY_V_ENV, BOUNDARY_V_ENV_ALT] {
        if let Ok(s) = std::env::var(var) {
            return parse_boundary_v_str(&s);
        }
    }
    DEFAULT_BOUNDARY_V_LAYERS
}

/// Parse a candidate boundary count string. Pure helper exposed for
/// tests so the env-var-free path stays deterministic.
///
/// Rules:
/// - Empty / whitespace-only → default.
/// - Non-numeric → default.
/// - Negative → `0` (disabled).
/// - Otherwise, the parsed value.
pub fn parse_boundary_v_str(s: &str) -> i32 {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return DEFAULT_BOUNDARY_V_LAYERS;
    }
    match trimmed.parse::<i32>() {
        Ok(v) if v < 0 => 0,
        Ok(v) => v,
        Err(_) => DEFAULT_BOUNDARY_V_LAYERS,
    }
}

/// Clamp the requested boundary count against the transformer depth.
///
/// We require `2 * boundary <= n_layers` so at least one middle layer
/// keeps the configured turbo mode; otherwise the boundary policy
/// degenerates into "every layer is FP16" and the caller is paying for
/// turbo allocation paths that never fire. The clamped value is
/// `boundary.min(n_layers / 2)`. Negative / zero inputs return `0`.
///
/// Used by: [`resolve_layer_mode`] and the per-model cache pool
/// constructors.
pub fn resolve_boundary_count(requested: i32, n_layers: usize) -> i32 {
    if requested <= 0 || n_layers == 0 {
        return 0;
    }
    let half = (n_layers / 2) as i32;
    requested.min(half)
}

/// Decide whether `layer_idx` falls in the protected boundary band
/// (first `boundary` layers or last `boundary` layers).
///
/// Pure function so the cache pool can pre-compute the per-layer mode
/// table at construction time without ever touching the env var inside
/// the hot decode loop.
#[inline]
pub fn is_boundary_layer(layer_idx: usize, n_layers: usize, boundary: i32) -> bool {
    if boundary <= 0 || n_layers == 0 {
        return false;
    }
    let b = boundary as usize;
    layer_idx < b || layer_idx + b >= n_layers
}

/// Pick the boundary protection mode for a given nominal `KVCacheMode`.
///
/// Boundary layers fall back to a "safer" representation:
/// - For any Turbo4* mode → `KVCacheMode::Fp16` (full FP16 K + V).
///   This is the cheapest implementation: no special cache storage
///   field is needed because `Fp16` already has its own update/fetch
///   path. The paper recommends q8_0-V at the boundary; FP16 spends a
///   few extra bytes per token in those 4 layers but recovers strictly
///   more quality, and the boundary-layer count is tiny relative to
///   the model depth.
/// - `Fp16` and `Int8` modes are passed through unchanged: there is
///   nothing for the boundary policy to upgrade to.
///
/// Used by: [`resolve_layer_mode`] — kept as a separate helper so the
/// "boundary precision choice" is a single edit point if we later
/// follow up with a true partial-V override (follow-up).
#[inline]
pub fn boundary_mode_for(nominal: KVCacheMode) -> KVCacheMode {
    match nominal {
        KVCacheMode::Turbo4Asym
        | KVCacheMode::Turbo4
        | KVCacheMode::Turbo4Delegated
        | KVCacheMode::Turbo3Asym => KVCacheMode::Fp16,
        // Non-turbo modes have no boundary upgrade path — return the
        // nominal mode unchanged.
        KVCacheMode::Fp16 | KVCacheMode::Int8 => nominal,
    }
}

/// Pre-compute the per-layer effective `KVCacheMode` for an entire model.
///
/// Returns a `Vec<KVCacheMode>` of length `n_layers` where boundary
/// layers carry [`boundary_mode_for(nominal)`](boundary_mode_for) and
/// the middle band carries `nominal` unchanged. The boundary count is
/// resolved via [`resolve_boundary_count`] so callers can pass the raw
/// env-var value directly.
///
/// This is the single entry point used by `CxxGenerator::new_with_kv_mode`
/// — it keeps the `update_and_fetch` hot path branchless because each
/// `KVCache` already holds its resolved mode.
///
/// Used by: [`crate::generate::CxxGenerator::new_with_kv_mode`].
pub fn resolve_layer_modes(
    nominal: KVCacheMode,
    n_layers: usize,
    requested_boundary: i32,
) -> Vec<KVCacheMode> {
    let boundary = resolve_boundary_count(requested_boundary, n_layers);
    let bmode = boundary_mode_for(nominal);
    (0..n_layers)
        .map(|layer_idx| {
            if bmode != nominal && is_boundary_layer(layer_idx, n_layers, boundary) {
                bmode
            } else {
                nominal
            }
        })
        .collect()
}

/// Resolve a single layer's effective mode. Convenience wrapper for
/// callers that already know the layer index (cache-pool detach/adopt,
/// per-layer test scaffolding).
///
/// Used by: tests and `KVCache::new_with_layer_position` constructor.
pub fn resolve_layer_mode(
    nominal: KVCacheMode,
    layer_idx: usize,
    n_layers: usize,
    requested_boundary: i32,
) -> KVCacheMode {
    let boundary = resolve_boundary_count(requested_boundary, n_layers);
    let bmode = boundary_mode_for(nominal);
    if bmode != nominal && is_boundary_layer(layer_idx, n_layers, boundary) {
        bmode
    } else {
        nominal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_default_when_unset_or_empty() {
        assert_eq!(parse_boundary_v_str(""), DEFAULT_BOUNDARY_V_LAYERS);
        assert_eq!(parse_boundary_v_str("   "), DEFAULT_BOUNDARY_V_LAYERS);
    }

    #[test]
    fn parse_explicit_zero_disables() {
        assert_eq!(parse_boundary_v_str("0"), 0);
    }

    #[test]
    fn parse_explicit_value() {
        assert_eq!(parse_boundary_v_str("4"), 4);
        assert_eq!(parse_boundary_v_str(" 8 "), 8);
    }

    #[test]
    fn parse_negative_clamps_to_zero() {
        assert_eq!(parse_boundary_v_str("-3"), 0);
    }

    #[test]
    fn parse_garbage_falls_back_to_default() {
        assert_eq!(parse_boundary_v_str("garbage"), DEFAULT_BOUNDARY_V_LAYERS);
        assert_eq!(parse_boundary_v_str("1.5"), DEFAULT_BOUNDARY_V_LAYERS);
    }

    #[test]
    fn resolve_boundary_count_clamps_to_half_layers() {
        // 2 boundary on 32 layers → 2 (no clamp).
        assert_eq!(resolve_boundary_count(2, 32), 2);
        // 4 boundary on 4 layers → 2 (n_layers/2).
        assert_eq!(resolve_boundary_count(4, 4), 2);
        // 8 boundary on 6 layers → 3 (n_layers/2).
        assert_eq!(resolve_boundary_count(8, 6), 3);
        // 0 boundary always → 0.
        assert_eq!(resolve_boundary_count(0, 100), 0);
        // n_layers == 0 → 0.
        assert_eq!(resolve_boundary_count(2, 0), 0);
    }

    #[test]
    fn resolve_boundary_count_handles_negative() {
        assert_eq!(resolve_boundary_count(-1, 32), 0);
    }

    #[test]
    fn is_boundary_layer_flags_first_and_last_n() {
        // 32-layer model with 2-layer boundary on each side.
        let n = 32;
        let b = 2;
        assert!(is_boundary_layer(0, n, b));
        assert!(is_boundary_layer(1, n, b));
        assert!(!is_boundary_layer(2, n, b));
        assert!(!is_boundary_layer(15, n, b));
        assert!(!is_boundary_layer(29, n, b));
        assert!(is_boundary_layer(30, n, b));
        assert!(is_boundary_layer(31, n, b));
    }

    #[test]
    fn is_boundary_layer_handles_zero_boundary() {
        // boundary=0 disables — no layer is ever boundary.
        let n = 32;
        for i in 0..n {
            assert!(!is_boundary_layer(i, n, 0));
        }
    }

    #[test]
    fn boundary_mode_upgrades_turbo_to_fp16() {
        assert_eq!(
            boundary_mode_for(KVCacheMode::Turbo4Asym),
            KVCacheMode::Fp16
        );
        assert_eq!(boundary_mode_for(KVCacheMode::Turbo4), KVCacheMode::Fp16);
        assert_eq!(
            boundary_mode_for(KVCacheMode::Turbo4Delegated),
            KVCacheMode::Fp16
        );
    }

    #[test]
    fn boundary_mode_passes_through_non_turbo() {
        assert_eq!(boundary_mode_for(KVCacheMode::Fp16), KVCacheMode::Fp16);
        assert_eq!(boundary_mode_for(KVCacheMode::Int8), KVCacheMode::Int8);
    }

    #[test]
    fn resolve_layer_modes_no_op_for_fp16() {
        // Fp16 nominal: every layer stays Fp16, regardless of boundary.
        let modes = resolve_layer_modes(KVCacheMode::Fp16, 8, 2);
        assert!(modes.iter().all(|m| *m == KVCacheMode::Fp16));
    }

    #[test]
    fn resolve_layer_modes_no_op_for_int8() {
        // Int8 nominal: every layer stays Int8, regardless of boundary.
        let modes = resolve_layer_modes(KVCacheMode::Int8, 8, 2);
        assert!(modes.iter().all(|m| *m == KVCacheMode::Int8));
    }

    #[test]
    fn resolve_layer_modes_upgrades_boundary_for_turbo4_asym() {
        // 8 layers, 2 boundary each side → layers 0,1,6,7 = Fp16,
        // 2..5 = Turbo4Asym.
        let modes = resolve_layer_modes(KVCacheMode::Turbo4Asym, 8, 2);
        assert_eq!(modes.len(), 8);
        assert_eq!(modes[0], KVCacheMode::Fp16);
        assert_eq!(modes[1], KVCacheMode::Fp16);
        assert_eq!(modes[2], KVCacheMode::Turbo4Asym);
        assert_eq!(modes[3], KVCacheMode::Turbo4Asym);
        assert_eq!(modes[4], KVCacheMode::Turbo4Asym);
        assert_eq!(modes[5], KVCacheMode::Turbo4Asym);
        assert_eq!(modes[6], KVCacheMode::Fp16);
        assert_eq!(modes[7], KVCacheMode::Fp16);
    }

    #[test]
    fn resolve_layer_modes_zero_boundary_keeps_all_layers_at_nominal() {
        let modes = resolve_layer_modes(KVCacheMode::Turbo4, 8, 0);
        assert!(modes.iter().all(|m| *m == KVCacheMode::Turbo4));
    }

    #[test]
    fn resolve_layer_modes_clamps_when_too_many_boundary() {
        // 4 layers requested with 4 boundary each side → clamped to 2 each
        // → layers 0,1 boundary, 2,3 boundary too (since 2 each side covers
        // the whole model).
        let modes = resolve_layer_modes(KVCacheMode::Turbo4Asym, 4, 4);
        assert_eq!(modes.len(), 4);
        assert!(modes.iter().all(|m| *m == KVCacheMode::Fp16));
    }

    #[test]
    fn resolve_layer_modes_empty_model() {
        let modes = resolve_layer_modes(KVCacheMode::Turbo4Asym, 0, 2);
        assert!(modes.is_empty());
    }

    #[test]
    fn resolve_layer_mode_matches_resolve_layer_modes() {
        // Cross-check the single-layer helper against the bulk helper.
        let n = 16;
        let bulk = resolve_layer_modes(KVCacheMode::Turbo4Asym, n, 2);
        for (i, &expected) in bulk.iter().enumerate().take(n) {
            let single = resolve_layer_mode(KVCacheMode::Turbo4Asym, i, n, 2);
            assert_eq!(expected, single, "layer {i} mismatch");
        }
    }
}
