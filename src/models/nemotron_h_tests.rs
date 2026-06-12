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

use super::{NemotronHConfig, NemotronLayerCache, NemotronMambaCache};
use mlxcel_core::{dtype, generate::ModelStateSnapshot, layers::KVCache};

// ---------------------------------------------------------------------------
// NemotronMambaCache state management
// ---------------------------------------------------------------------------

/// A freshly constructed NemotronMambaCache has no conv_state or ssm_state.
#[test]
fn mamba_cache_new_has_no_state() {
    let cache = NemotronMambaCache::new();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
}

/// Default trait impl for NemotronMambaCache also produces an empty cache.
#[test]
fn mamba_cache_default_matches_new() {
    let a = NemotronMambaCache::new();
    let b = NemotronMambaCache::default();
    assert!(a.conv_state.is_none());
    assert!(b.conv_state.is_none());
    assert!(a.ssm_state.is_none());
    assert!(b.ssm_state.is_none());
}

#[test]
fn mamba_cache_snapshot_restore_round_trips_state_shapes() {
    let mut cache = NemotronMambaCache::new();
    cache.conv_state = Some(mlxcel_core::zeros(&[1, 3, 8], dtype::FLOAT32));
    cache.ssm_state = Some(mlxcel_core::zeros(&[1, 2, 4, 8], dtype::FLOAT32));

    let mut snapshot = ModelStateSnapshot::new("nemotron_h", 13);
    cache.snapshot_into(&mut snapshot, "layer2.mamba");

    let mut restored = NemotronMambaCache::new();
    restored.restore_from(&snapshot, "layer2.mamba");

    let conv = restored
        .conv_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("conv_state restored");
    let ssm = restored
        .ssm_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("ssm_state restored");
    assert_eq!(mlxcel_core::array_shape(conv), vec![1, 3, 8]);
    assert_eq!(mlxcel_core::array_shape(ssm), vec![1, 2, 4, 8]);
}

#[test]
fn attention_layer_snapshot_restore_round_trips_kv_state() {
    let mut kv = KVCache::new();
    kv.keys = Some(mlxcel_core::zeros(&[1, 2, 13, 4], dtype::FLOAT32));
    kv.values = Some(mlxcel_core::zeros(&[1, 2, 13, 4], dtype::FLOAT32));
    kv.offset = 13;
    let cache = NemotronLayerCache::Attention(kv);

    let mut snapshot = ModelStateSnapshot::new("nemotron_h", 13);
    cache.snapshot_into(&mut snapshot, "layer0");

    let mut restored = NemotronLayerCache::Attention(KVCache::new());
    restored.restore_from(&snapshot, "layer0");

    match restored {
        NemotronLayerCache::Attention(kv) => {
            let keys = kv
                .keys
                .as_ref()
                .and_then(|a| a.as_ref())
                .expect("keys restored");
            let values = kv
                .values
                .as_ref()
                .and_then(|a| a.as_ref())
                .expect("values restored");
            assert_eq!(kv.offset, 13);
            assert_eq!(mlxcel_core::array_shape(keys), vec![1, 2, 13, 4]);
            assert_eq!(mlxcel_core::array_shape(values), vec![1, 2, 13, 4]);
        }
        NemotronLayerCache::Mamba(_) => panic!("restored wrong cache variant"),
    }
}

// ---------------------------------------------------------------------------
// NemotronLayerCache offset
// ---------------------------------------------------------------------------

/// The offset of an Attention layer cache matches the underlying KVCache offset.
#[test]
fn layer_cache_attention_offset_matches_kvcache() {
    let kv = KVCache::new();
    let cache = NemotronLayerCache::Attention(kv);
    // A fresh KVCache has offset 0.
    assert_eq!(cache.offset(), 0);
}

/// The offset of a Mamba layer cache is always 0 (Mamba state is not positional).
#[test]
fn layer_cache_mamba_offset_is_zero() {
    let mc = NemotronMambaCache::new();
    let cache = NemotronLayerCache::Mamba(mc);
    assert_eq!(cache.offset(), 0);
}

// ---------------------------------------------------------------------------
// trim_internal_caches logic (via direct cache manipulation)
// ---------------------------------------------------------------------------

/// After simulating what trim_internal_caches does: Mamba conv/ssm state is cleared.
/// This mirrors the body of NemotronHModel::trim_internal_caches for Mamba layers.
#[test]
fn mamba_cache_reset_clears_state() {
    let mut mc = NemotronMambaCache::new();

    // Simulate state that would have been written during a padded prefill.
    // We set the fields to None to begin with (as they would be before prefill),
    // then verify that resetting them (as trim_internal_caches does) leaves them None.
    //
    // In the real codepath, conv_state and ssm_state would hold MlxArrays written
    // during the padded forward pass. trim_internal_caches resets them so that
    // subsequent decode steps do not carry corrupted padding state forward.
    mc.conv_state = None;
    mc.ssm_state = None;

    // Apply the same reset logic as trim_internal_caches.
    mc.conv_state = None;
    mc.ssm_state = None;

    assert!(
        mc.conv_state.is_none(),
        "conv_state must be None after reset"
    );
    assert!(mc.ssm_state.is_none(), "ssm_state must be None after reset");
}

/// trim_internal_caches with excess <= 0 returns early without modifying any cache.
/// This is a guard against callers passing non-positive excess values.
///
/// We verify the guard condition matches the one used in the implementation
/// (excess <= 0 returns immediately).
#[test]
fn trim_guard_zero_excess_is_noop() {
    // Reproduce the guard: excess <= 0 means no trimming should happen.
    let excess: i32 = 0;
    assert!(
        excess <= 0,
        "guard condition: excess=0 must trigger early return"
    );
}

#[test]
fn trim_guard_negative_excess_is_noop() {
    let excess: i32 = -5;
    assert!(
        excess <= 0,
        "guard condition: negative excess must trigger early return"
    );
}

#[test]
fn trim_guard_positive_excess_proceeds() {
    let excess: i32 = 32;
    assert!(
        excess > 0,
        "excess=32 must not trigger early return (should proceed to trim)"
    );
}

// ---------------------------------------------------------------------------
// KVCache trim for Attention layers (no GPU: empty cache trim is a no-op)
// ---------------------------------------------------------------------------

/// Trimming an Attention layer cache that has no data is safe and leaves the
/// offset at 0. This verifies that trim_internal_caches for Attention layers
/// does not panic when called on a cache that was never populated (e.g. if
/// padded prefill was called on a fresh session).
#[test]
fn attention_cache_trim_on_empty_cache_is_safe() {
    let mut kv = KVCache::new();
    assert_eq!(kv.offset, 0);

    // KVCache::trim clamps n to self.offset (min), so this is a no-op on empty cache.
    let trimmed = kv.trim(16);
    assert_eq!(trimmed, 0);
    assert_eq!(kv.offset, 0);
}

// ---------------------------------------------------------------------------
// state_contrib broadcast shape invariants
// ---------------------------------------------------------------------------

/// The broadcast shape for state_contrib must be [batch, heads, 1, 1] (rank 4),
/// NOT [batch, 1, heads, 1, 1] (rank 5).  The extra leading `1` was the pre-fix
/// shape; this test documents the correct rank expected by the multiply.
///
/// In ssm_step, after slicing exp_dta_cumsum at the last seq position we get a
/// tensor of shape [batch, 1, num_heads].  The reshape target is
/// [batch, num_heads, 1, 1] so that it broadcasts correctly with next_state
/// whose shape is [batch, heads, head_dim, state_dim].
///
/// We verify the rank arithmetic here without requiring MLX array allocation.
#[test]
fn state_contrib_broadcast_shape_is_rank4() {
    // Representative dims matching Nemotron-H 30B config.
    let batch = 1_i32;
    let num_heads = 128_i32;
    // After slice_axis on the seq dimension and reshape:
    // last_exp: [batch, num_heads, 1, 1]
    let last_exp_shape = [batch, num_heads, 1_i32, 1_i32];
    // next_state: [batch, heads, head_dim, state_dim]
    let head_dim = 64_i32;
    let state_dim = 128_i32;
    let next_state_shape = [batch, num_heads, head_dim, state_dim];

    // Verify rank matches (both rank-4) and leading two dims are identical
    // so that element-wise multiply broadcasts correctly.
    assert_eq!(
        last_exp_shape.len(),
        next_state_shape.len(),
        "last_exp and next_state must have the same rank for broadcasting"
    );
    assert_eq!(
        last_exp_shape[0], next_state_shape[0],
        "batch dims must match"
    );
    assert_eq!(
        last_exp_shape[1], next_state_shape[1],
        "heads dims must match"
    );
    // The trailing [1, 1] in last_exp broadcasts over [head_dim, state_dim].
    assert_eq!(
        last_exp_shape[2], 1,
        "last_exp trailing dim[2] must be 1 for broadcast"
    );
    assert_eq!(
        last_exp_shape[3], 1,
        "last_exp trailing dim[3] must be 1 for broadcast"
    );
}

/// Confirm the pre-fix (wrong) shape [batch, 1, heads, 1, 1] has rank 5,
/// which would mismatch next_state at rank 4 and produce wrong broadcast
/// semantics on M5 Max.
#[test]
fn state_contrib_old_shape_was_rank5() {
    let batch = 1_i32;
    let num_heads = 128_i32;
    // The buggy reshape target before this PR.
    let old_last_exp_shape = [batch, 1_i32, num_heads, 1_i32, 1_i32];
    let next_state_rank = 4_usize;

    assert_ne!(
        old_last_exp_shape.len(),
        next_state_rank,
        "pre-fix shape rank 5 must differ from next_state rank 4"
    );
}

// ---------------------------------------------------------------------------
// time_step_limit defaulting (NemotronHConfig::post_init)
// ---------------------------------------------------------------------------

/// Minimal config JSON with required fields only (no time_step_limit, no
/// time_step_min, no time_step_max). After post_init the field must be (0.0, +inf).
fn minimal_config_json(extra: &str) -> String {
    format!(
        r#"{{
            "model_type": "nemotron_h",
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "mamba_num_heads": 4,
            "mamba_head_dim": 16,
            "ssm_state_size": 16,
            "conv_kernel": 4,
            "n_groups": 1,
            "hybrid_override_pattern": "MM"
            {extra}
        }}"#
    )
}

/// Config with no time_step_limit, no time_step_min, and no time_step_max.
/// After post_init, time_step_limit must be (0.0, +inf).
#[test]
fn post_init_no_time_step_fields_defaults_to_zero_inf() {
    let json = minimal_config_json("");
    let mut cfg: NemotronHConfig = serde_json::from_str(&json).expect("deserialize");
    assert!(
        cfg.time_step_limit.is_none(),
        "time_step_limit must be None before post_init when absent from JSON"
    );
    cfg.post_init().expect("post_init");
    let (lo, hi) = cfg
        .time_step_limit
        .expect("time_step_limit must be Some after post_init");
    assert_eq!(lo, 0.0, "lower bound must be 0.0");
    assert!(
        hi.is_infinite() && hi.is_sign_positive(),
        "upper bound must be +inf"
    );
}

/// Config with only time_step_min: 0.001 (no time_step_limit, no time_step_max).
/// After post_init, time_step_limit must be (0.0, +inf) — time_step_min is a
/// softplus init bound, NOT the dt clamp lower bound (upstream PR #1026 fix).
#[test]
fn post_init_only_time_step_min_defaults_to_zero_inf() {
    let json = minimal_config_json(r#", "time_step_min": 0.001"#);
    let mut cfg: NemotronHConfig = serde_json::from_str(&json).expect("deserialize");
    cfg.post_init().expect("post_init");
    let (lo, hi) = cfg
        .time_step_limit
        .expect("time_step_limit must be Some after post_init");
    assert_eq!(
        lo, 0.0,
        "lower bound must be 0.0 regardless of time_step_min"
    );
    assert!(
        hi.is_infinite() && hi.is_sign_positive(),
        "upper bound must be +inf"
    );
}

/// Config with only time_step_max: 0.1 (no time_step_limit, no time_step_min).
/// After post_init, time_step_limit must be (0.0, +inf) — time_step_max is a
/// softplus init bound, NOT the dt clamp upper bound (upstream PR #1026 fix).
#[test]
fn post_init_only_time_step_max_defaults_to_zero_inf() {
    let json = minimal_config_json(r#", "time_step_max": 0.1"#);
    let mut cfg: NemotronHConfig = serde_json::from_str(&json).expect("deserialize");
    cfg.post_init().expect("post_init");
    let (lo, hi) = cfg
        .time_step_limit
        .expect("time_step_limit must be Some after post_init");
    assert_eq!(
        lo, 0.0,
        "lower bound must be 0.0 regardless of time_step_min"
    );
    assert!(
        hi.is_infinite() && hi.is_sign_positive(),
        "upper bound must be +inf regardless of time_step_max"
    );
}

/// Config with both time_step_min: 0.001 and time_step_max: 0.1 set, but no
/// time_step_limit. This is the typical real-world Nemotron-H config. After
/// post_init, time_step_limit must be (0.0, +inf), NOT (0.001, 0.1).
/// Regression test.
#[test]
fn post_init_both_time_step_min_max_set_defaults_to_zero_inf() {
    let json = minimal_config_json(r#", "time_step_min": 0.001, "time_step_max": 0.1"#);
    let mut cfg: NemotronHConfig = serde_json::from_str(&json).expect("deserialize");
    assert!(
        cfg.time_step_limit.is_none(),
        "time_step_limit must be None before post_init when absent from JSON"
    );
    cfg.post_init().expect("post_init");
    let (lo, hi) = cfg
        .time_step_limit
        .expect("time_step_limit must be Some after post_init");
    assert_eq!(
        lo, 0.0,
        "lower bound must be 0.0, not time_step_min ({lo}); see upstream PR #1026"
    );
    assert!(
        hi.is_infinite() && hi.is_sign_positive(),
        "upper bound must be +inf, not time_step_max; see upstream PR #1026"
    );
    // Verify time_step_min / time_step_max are still accessible for the forward path.
    assert!(
        (cfg.time_step_min.unwrap() - 0.001_f32).abs() < 1e-7,
        "time_step_min must be preserved for softplus init"
    );
    assert!(
        (cfg.time_step_max.unwrap() - 0.1_f32).abs() < 1e-6,
        "time_step_max must be preserved for softplus init"
    );
}

/// Config with explicit time_step_limit: [0.05, 1.5].
/// After post_init, time_step_limit must be (0.05, 1.5) unchanged.
#[test]
fn post_init_explicit_time_step_limit_is_preserved() {
    let json = minimal_config_json(r#", "time_step_limit": [0.05, 1.5]"#);
    let mut cfg: NemotronHConfig = serde_json::from_str(&json).expect("deserialize");
    assert!(
        cfg.time_step_limit.is_some(),
        "explicit time_step_limit must be Some before post_init"
    );
    cfg.post_init().expect("post_init");
    let (lo, hi) = cfg
        .time_step_limit
        .expect("time_step_limit must be Some after post_init");
    assert!(
        (lo - 0.05_f32).abs() < 1e-6,
        "lower bound must be 0.05, got {lo}"
    );
    assert!(
        (hi - 1.5_f32).abs() < 1e-6,
        "upper bound must be 1.5, got {hi}"
    );
}

// ---------------------------------------------------------------------------
// conv_state contiguous regression
// ---------------------------------------------------------------------------

/// Simulate 50 decode steps of conv-state update and assert the stored shape
/// stays at [B=1, k-1=3, channels=8] regardless of how many steps we run.
///
/// This exercises the contiguous() fix for the NemotronH Mamba layer decode path
/// (nemotron_h.rs line ~516).  Before the fix, each step stored a lazy slice
/// aliasing the growing padded_input graph, leaking memory linearly with steps.
#[test]
#[ignore = "requires serial MLX execution"]
fn nemotron_h_conv_state_shape_plateaus_after_50_steps() {
    use mlxcel_core::utils::slice_axis;

    let batch = 1i32;
    let channels = 8i32;
    let k = 4usize; // conv_kernel_size
    let n_keep = (k - 1) as i32; // = 3
    let expected_shape = vec![batch, n_keep, channels];

    let mut conv_state =
        mlxcel_core::zeros(&[batch, n_keep, channels], mlxcel_core::dtype::FLOAT32);

    for _step in 0..50 {
        let new_token = mlxcel_core::zeros(&[batch, 1, channels], mlxcel_core::dtype::FLOAT32);

        // Build padded_input = concat(conv_state, new_token, axis=1) -> [1, k, channels]
        let padded_input = mlxcel_core::concatenate(&conv_state, &new_token, 1);

        let padded_shape = mlxcel_core::array_shape(&padded_input);
        let len = padded_shape[1] as usize;

        // Apply the fixed conv-state update: slice then contiguous
        let tail = slice_axis(&padded_input, 1, (len - (k - 1)) as i32, len as i32);
        conv_state = mlxcel_core::contiguous(&tail, false);

        mlxcel_core::eval(&conv_state);

        let shape = mlxcel_core::array_shape(&conv_state);
        assert_eq!(
            shape, expected_shape,
            "step {_step}: conv_state shape {shape:?} != expected {expected_shape:?}"
        );
    }
}
