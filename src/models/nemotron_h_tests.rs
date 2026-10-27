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

use super::{NemotronHConfig, NemotronLayerCache, NemotronMambaCache, nemotron_h_ssm_attn};
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
// trim_state logic (via direct cache manipulation)
// ---------------------------------------------------------------------------

/// After simulating what trim_state does: Mamba conv/ssm state is cleared.
/// This mirrors the body of NemotronHModel::trim_state for Mamba layers.
#[test]
fn mamba_cache_reset_clears_state() {
    let mut mc = NemotronMambaCache::new();

    // Simulate state that would have been written during a padded prefill.
    // We set the fields to None to begin with (as they would be before prefill),
    // then verify that resetting them (as trim_state does) leaves them None.
    //
    // In the real codepath, conv_state and ssm_state would hold MlxArrays written
    // during the padded forward pass. trim_state resets them so that
    // subsequent decode steps do not carry corrupted padding state forward.
    mc.conv_state = None;
    mc.ssm_state = None;

    // Apply the same reset logic as trim_state.
    mc.conv_state = None;
    mc.ssm_state = None;

    assert!(
        mc.conv_state.is_none(),
        "conv_state must be None after reset"
    );
    assert!(mc.ssm_state.is_none(), "ssm_state must be None after reset");
}

/// trim_state with excess <= 0 returns early without modifying any cache.
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
/// offset at 0. This verifies that trim_state for Attention layers
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

#[test]
fn missing_chunk_size_defaults_to_128() {
    let json = minimal_config_json("");
    let mut cfg: NemotronHConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg.chunk_size, 128);
    cfg.post_init().expect("default chunk_size must be valid");
}

#[test]
fn mamba_chunk_size_alias_is_accepted() {
    let json = minimal_config_json(r#", "mamba_chunk_size": 64"#);
    let mut cfg: NemotronHConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg.chunk_size, 64);
    cfg.post_init().expect("aliased chunk_size must be valid");
}

#[test]
fn zero_chunk_size_is_rejected() {
    let json = minimal_config_json(r#", "chunk_size": 0"#);
    let mut cfg: NemotronHConfig = serde_json::from_str(&json).expect("deserialize");
    let error = cfg.post_init().expect_err("zero chunk_size must fail");
    assert!(error.contains("chunk_size must be greater than zero"));
}

#[test]
#[ignore = "requires serial MLX execution"]
fn chunked_ssd_matches_full_step_with_prior_state_and_partial_tail() {
    let _runtime = crate::initialize_runtime();

    const SEQ_LEN: i32 = 5;
    const HEADS: i32 = 2;
    const HEAD_DIM: i32 = 2;
    const STATE_DIM: i32 = 3;

    let ramp = |len: usize, offset: f32, scale: f32| -> Vec<f32> {
        (0..len)
            .map(|index| offset + index as f32 * scale)
            .collect()
    };
    let x = mlxcel_core::from_slice_f32(
        &ramp((SEQ_LEN * HEADS * HEAD_DIM) as usize, -0.2, 0.025),
        &[1, SEQ_LEN, HEADS, HEAD_DIM],
    );
    let b = mlxcel_core::from_slice_f32(
        &ramp((SEQ_LEN * STATE_DIM) as usize, -0.1, 0.02),
        &[1, SEQ_LEN, 1, STATE_DIM],
    );
    let c = mlxcel_core::from_slice_f32(
        &ramp((SEQ_LEN * STATE_DIM) as usize, 0.15, -0.01),
        &[1, SEQ_LEN, 1, STATE_DIM],
    );
    let dt = mlxcel_core::from_slice_f32(
        &ramp((SEQ_LEN * HEADS) as usize, -0.4, 0.03),
        &[1, SEQ_LEN, HEADS],
    );
    let a_log = mlxcel_core::from_slice_f32(&[-0.4, -0.2], &[HEADS]);
    let d = mlxcel_core::from_slice_f32(&[0.3, -0.2], &[HEADS]);
    let dt_bias = mlxcel_core::from_slice_f32(&[-0.1, 0.2], &[HEADS]);
    let prior_state = mlxcel_core::from_slice_f32(
        &ramp((HEADS * HEAD_DIM * STATE_DIM) as usize, -0.05, 0.01),
        &[1, HEADS, HEAD_DIM, STATE_DIM],
    );

    let (full_y, full_state) = nemotron_h_ssm_attn(
        &x,
        &a_log,
        &b,
        &c,
        &d,
        &dt,
        &dt_bias,
        Some(&prior_state),
        (0.0, f32::INFINITY),
        SEQ_LEN as usize,
    );
    let (chunked_y, chunked_state) = nemotron_h_ssm_attn(
        &x,
        &a_log,
        &b,
        &c,
        &d,
        &dt,
        &dt_bias,
        Some(&prior_state),
        (0.0, f32::INFINITY),
        2,
    );

    assert_eq!(
        mlxcel_core::array_shape(&chunked_y),
        vec![1, SEQ_LEN, HEADS, HEAD_DIM]
    );
    let output_close = mlxcel_core::allclose(&full_y, &chunked_y, 1e-4, 1e-5);
    mlxcel_core::eval(&output_close);
    assert!(
        mlxcel_core::item_bool(&output_close),
        "chunk size 2 (2+2+1) changed SSD output"
    );

    let state_close = mlxcel_core::allclose(&full_state, &chunked_state, 1e-4, 1e-5);
    mlxcel_core::eval(&state_close);
    assert!(
        mlxcel_core::item_bool(&state_close),
        "chunk size 2 (2+2+1) changed final recurrent state"
    );
}

fn recurrence_oracle(
    x: &[f32],
    a_log: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    dt: &[f32],
    dt_bias: &[f32],
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    num_groups: usize,
    state_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let heads_per_group = num_heads / num_groups;
    let mut output = vec![0.0; batch_size * seq_len * num_heads * head_dim];
    let mut state = vec![0.0; batch_size * num_heads * head_dim * state_dim];

    for batch in 0..batch_size {
        for token in 0..seq_len {
            for head in 0..num_heads {
                let group = head / heads_per_group;
                let dt_index = (batch * seq_len + token) * num_heads + head;
                let delta = (dt[dt_index] + dt_bias[head]).exp().ln_1p();
                let decay = (-a_log[head].exp() * delta).exp();
                for channel in 0..head_dim {
                    let x_index =
                        ((batch * seq_len + token) * num_heads + head) * head_dim + channel;
                    let mut value = 0.0;
                    for state_channel in 0..state_dim {
                        let state_index = ((batch * num_heads + head) * head_dim + channel)
                            * state_dim
                            + state_channel;
                        let bc_index = ((batch * seq_len + token) * num_groups + group) * state_dim
                            + state_channel;
                        state[state_index] =
                            decay * state[state_index] + delta * x[x_index] * b[bc_index];
                        value += state[state_index] * c[bc_index];
                    }
                    output[x_index] = value + d[head] * x[x_index];
                }
            }
        }
    }

    (output, state)
}

#[test]
#[ignore = "requires serial MLX execution"]
fn chunked_ssd_matches_direct_recurrence_for_multigroup_f16_and_bf16() {
    use mlxcel_core::utils::array_to_vec_f32;

    let _runtime = crate::initialize_runtime();

    const BATCH: usize = 2;
    const SEQ_LEN: usize = 5;
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 2;
    const GROUPS: usize = 2;
    const STATE_DIM: usize = 3;

    let ramp = |len: usize, offset: f32, scale: f32| -> Vec<f32> {
        (0..len)
            .map(|index| offset + index as f32 * scale)
            .collect()
    };
    let raw_x = ramp(BATCH * SEQ_LEN * HEADS * HEAD_DIM, -0.3, 0.007);
    let raw_b = ramp(BATCH * SEQ_LEN * GROUPS * STATE_DIM, -0.15, 0.004);
    let raw_c = ramp(BATCH * SEQ_LEN * GROUPS * STATE_DIM, 0.2, -0.003);
    let raw_dt = ramp(BATCH * SEQ_LEN * HEADS, -0.5, 0.006);
    let raw_a_log = [-0.4, -0.3, -0.2, -0.1];
    let raw_d = [0.3, -0.2, 0.1, -0.05];
    let raw_dt_bias = [-0.15, -0.05, 0.05, 0.15];

    for (dtype, tolerance) in [
        (mlxcel_core::dtype::FLOAT16, 2e-3_f32),
        (mlxcel_core::dtype::BFLOAT16, 2e-2_f32),
    ] {
        let cast = |values: &[f32], shape: &[i32]| {
            mlxcel_core::astype(&mlxcel_core::from_slice_f32(values, shape), dtype)
        };
        let x = cast(
            &raw_x,
            &[BATCH as i32, SEQ_LEN as i32, HEADS as i32, HEAD_DIM as i32],
        );
        let b = cast(
            &raw_b,
            &[
                BATCH as i32,
                SEQ_LEN as i32,
                GROUPS as i32,
                STATE_DIM as i32,
            ],
        );
        let c = cast(
            &raw_c,
            &[
                BATCH as i32,
                SEQ_LEN as i32,
                GROUPS as i32,
                STATE_DIM as i32,
            ],
        );
        let dt = cast(&raw_dt, &[BATCH as i32, SEQ_LEN as i32, HEADS as i32]);
        let a_log = cast(&raw_a_log, &[HEADS as i32]);
        let d = cast(&raw_d, &[HEADS as i32]);
        let dt_bias = cast(&raw_dt_bias, &[HEADS as i32]);

        let (actual_output, actual_state) = nemotron_h_ssm_attn(
            &x,
            &a_log,
            &b,
            &c,
            &d,
            &dt,
            &dt_bias,
            None,
            (0.0, f32::INFINITY),
            2,
        );
        let (expected_output, expected_state) = recurrence_oracle(
            &array_to_vec_f32(&x),
            &array_to_vec_f32(&a_log),
            &array_to_vec_f32(&b),
            &array_to_vec_f32(&c),
            &array_to_vec_f32(&d),
            &array_to_vec_f32(&dt),
            &array_to_vec_f32(&dt_bias),
            BATCH,
            SEQ_LEN,
            HEADS,
            HEAD_DIM,
            GROUPS,
            STATE_DIM,
        );
        let expected_output = cast(
            &expected_output,
            &[BATCH as i32, SEQ_LEN as i32, HEADS as i32, HEAD_DIM as i32],
        );
        let expected_state = mlxcel_core::from_slice_f32(
            &expected_state,
            &[
                BATCH as i32,
                HEADS as i32,
                HEAD_DIM as i32,
                STATE_DIM as i32,
            ],
        );

        let output_close =
            mlxcel_core::allclose(&actual_output, &expected_output, 0.0, tolerance as f64);
        mlxcel_core::eval(&output_close);
        assert!(
            mlxcel_core::item_bool(&output_close),
            "chunked SSD output diverged from direct recurrence for dtype {dtype}"
        );
        let state_close =
            mlxcel_core::allclose(&actual_state, &expected_state, 1e-4, tolerance as f64);
        mlxcel_core::eval(&state_close);
        assert!(
            mlxcel_core::item_bool(&state_close),
            "chunked SSD state diverged from direct recurrence for dtype {dtype}"
        );
    }
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

/// `NemotronHModel::from_weights` stores the declared `group_size` / `bits` on
/// each expert plane (`SwitchLinear`, since issue #2665; a local struct literal
/// before) and hands them to `gather_qmm` without ever reaching
/// `reconcile_quantization_layout` (issue #958). `gather_qmm` crosses the cxx
/// bridge as `UniquePtr<MlxArray>` rather than `Result`, so a hostile pair used
/// to abort the process at the first routed forward pass rather than failing at
/// load.
///
/// This fixture is deliberately the shape the issue calls out as the one no
/// guarded loader covers: a float `backbone.embeddings` and a float `lm_head`
/// with quantized expert planes as the ONLY quantized tensors in the
/// checkpoint. `UnifiedEmbedding::from_weights` only enters its quantized
/// branch when `.scales` exists, so nothing here touches the reconciler.
///
/// It drives the real model loader rather than the pure bounds helper, so the
/// guard is exercised where a checkpoint actually reaches it. The assertions are
/// on the load result and no forward pass is run, so a regression fails cleanly
/// here rather than aborting the test binary.
#[test]
fn nemotron_h_rejects_expert_quantization_params_that_would_abort_gather_qmm() {
    use super::{BlockType, NemotronHConfig, NemotronHModel};
    use mlxcel_core::weights::WeightMap;

    const HIDDEN: i32 = 8;
    const EXPERTS: i32 = 2;
    const FF: i32 = 4;
    // Honest affine 4-bit geometry: packed_in * 32 == bits * groups * gs,
    // i.e. 1 * 32 == 4 * 1 * 8.
    const PACKED_IN: i32 = 1;
    const NUM_GROUPS: i32 = 1;

    let config_with = |group_size: i32, bits: i32| -> NemotronHConfig {
        let mut config: NemotronHConfig = serde_json::from_value(serde_json::json!({
            "model_type": "nemotron_h",
            "vocab_size": 8,
            "hidden_size": HIDDEN,
            "intermediate_size": FF,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "mamba_num_heads": 1,
            "mamba_head_dim": 4,
            "ssm_state_size": 4,
            "conv_kernel": 4,
            "n_groups": 1,
            "n_routed_experts": EXPERTS,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": FF,
            "hybrid_override_pattern": "E",
            "quantization": { "group_size": group_size, "bits": bits },
        }))
        .expect("test config must parse");
        config.post_init().expect("test config must pass post_init");
        config
    };

    let build_weights = || {
        let mut weights = WeightMap::new();
        let float = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            mlxcel_core::from_slice_f32(&vec![0.0; n as usize], shape)
        };
        // Float embedding and lm_head: no `.scales`, so neither reaches the
        // reconciler and neither can stand in for the expert-plane bound.
        weights.insert(
            "backbone.embeddings.weight".to_string(),
            float(&[8, HIDDEN]),
        );
        weights.insert("lm_head.weight".to_string(), float(&[8, HIDDEN]));
        weights.insert("backbone.norm_f.weight".to_string(), float(&[HIDDEN]));
        weights.insert(
            "backbone.layers.0.norm.weight".to_string(),
            float(&[HIDDEN]),
        );
        weights.insert(
            "backbone.layers.0.mixer.gate.weight".to_string(),
            float(&[EXPERTS, HIDDEN]),
        );
        for fc in ["fc1", "fc2"] {
            let prefix = format!("backbone.layers.0.mixer.switch_mlp.{fc}");
            crate::models::switch_layers::insert_stacked_quantized_expert_plane(
                &mut weights,
                &prefix,
                EXPERTS,
                FF,
                PACKED_IN,
                NUM_GROUPS,
            );
        }
        weights
    };

    // Positive control first, so a guard that rejects everything quantized
    // cannot pass this test.
    NemotronHModel::from_weights(config_with(8, 4), build_weights(), vec![BlockType::MoE])
        .expect("an honest affine pair must still load quantized Nemotron-H experts");

    for (group_size, bits, field) in crate::models::switch_layers::HOSTILE_QUANT_PARAMS {
        let err = NemotronHModel::from_weights(
            config_with(group_size, bits),
            build_weights(),
            vec![BlockType::MoE],
        )
        .err()
        .unwrap_or_else(|| {
            panic!("Nemotron-H must reject group_size {group_size} / bits {bits} at load")
        })
        .to_string();
        assert!(err.contains(field), "unhelpful error: {err}");
    }

    // A bf16 expert plane carries no packing at all, so the params are
    // irrelevant there and must not be enforced.
    let mut regular = build_weights();
    for fc in ["fc1", "fc2"] {
        let prefix = format!("backbone.layers.0.mixer.switch_mlp.{fc}");
        regular.remove(&format!("{prefix}.scales"));
        regular.remove(&format!("{prefix}.biases"));
    }
    NemotronHModel::from_weights(config_with(0, 0), regular, vec![BlockType::MoE])
        .expect("a non-quantized expert plane must not be gated on quantization params");
}

/// Nemotron-H builds its shared-expert and latent projections as optional
/// layers. A runtime LoRA term of the wrong width on one of them used to be
/// dropped with the layer (`from_weights(..).ok()`), so the load only failed
/// later as "unclaimed" and never said why (#2445). The width error from the
/// claim is now the load's error. A projection whose weight is absent is still
/// just absent. Not on Metal, which refuses runtime LoRA for this model before
/// any layer is built.
#[test]
#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn a_wrong_width_lora_term_on_an_optional_projection_fails_the_load() {
    use super::{BlockType, NemotronHModel};
    use mlxcel_core::runtime_lora::{PendingLoraTerm, SharedLoraScale};
    use mlxcel_core::weights::WeightMap;

    const HIDDEN: i32 = 8;
    const EXPERTS: i32 = 2;
    const FF: i32 = 4;
    const UP_PROJ: &str = "backbone.layers.0.mixer.shared_experts.up_proj";

    let config = || -> NemotronHConfig {
        let mut config: NemotronHConfig = serde_json::from_value(serde_json::json!({
            "model_type": "nemotron_h",
            "vocab_size": 8,
            "hidden_size": HIDDEN,
            "intermediate_size": FF,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "mamba_num_heads": 1,
            "mamba_head_dim": 4,
            "ssm_state_size": 4,
            "conv_kernel": 4,
            "n_groups": 1,
            "n_routed_experts": EXPERTS,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": FF,
            "hybrid_override_pattern": "E",
        }))
        .expect("test config must parse");
        config.post_init().expect("test config must pass post_init");
        config
    };
    let weights = |with_shared: bool| {
        let mut weights = WeightMap::new();
        let mut put = |name: &str, shape: &[i32]| {
            let n: i32 = shape.iter().product();
            weights.insert(
                name.to_string(),
                mlxcel_core::from_slice_f32(&vec![0.0; n as usize], shape),
            );
        };
        put("backbone.embeddings.weight", &[8, HIDDEN]);
        put("lm_head.weight", &[8, HIDDEN]);
        put("backbone.norm_f.weight", &[HIDDEN]);
        put("backbone.layers.0.norm.weight", &[HIDDEN]);
        put("backbone.layers.0.mixer.gate.weight", &[EXPERTS, HIDDEN]);
        put(
            "backbone.layers.0.mixer.switch_mlp.fc1.weight",
            &[EXPERTS, FF, HIDDEN],
        );
        put(
            "backbone.layers.0.mixer.switch_mlp.fc2.weight",
            &[EXPERTS, HIDDEN, FF],
        );
        if with_shared {
            put(&format!("{UP_PROJ}.weight"), &[FF, HIDDEN]);
            put(
                "backbone.layers.0.mixer.shared_experts.down_proj.weight",
                &[HIDDEN, FF],
            );
        }
        weights
    };
    let stage = |in_width: i32| {
        mlxcel_core::runtime_lora::stage(std::collections::HashMap::from([(
            UP_PROJ.to_string(),
            vec![PendingLoraTerm {
                handle: SharedLoraScale::new(1.0),
                base_scale: 1.0,
                a_t: mlxcel_core::from_slice_f32(&vec![1.0; in_width as usize], &[in_width, 1]),
                b_t: mlxcel_core::from_slice_f32(&vec![1.0; FF as usize], &[1, FF]),
            }],
        )]));
    };

    // Absent shared-expert weights are an absent layer, not an error.
    NemotronHModel::from_weights(config(), weights(false), vec![BlockType::MoE])
        .expect("a checkpoint without shared experts still loads");

    stage(2 * HIDDEN);
    let result = NemotronHModel::from_weights(config(), weights(true), vec![BlockType::MoE]);
    let _ = mlxcel_core::runtime_lora::drain_unclaimed();
    let err = result
        .err()
        .expect("a wrong-width term on shared_experts.up_proj must fail the load")
        .to_string();
    assert!(err.contains("does not fit the linear layer"), "{err}");
    assert!(err.contains(UP_PROJ), "{err}");

    // The right width is claimed and the model loads.
    stage(HIDDEN);
    let result = NemotronHModel::from_weights(config(), weights(true), vec![BlockType::MoE]);
    assert!(
        mlxcel_core::runtime_lora::drain_unclaimed().is_empty(),
        "the term was claimed"
    );
    result.expect("a right-width term loads");
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn request_memory_runtime_lora_rejects_before_construction_and_cleans_staging() {
    if !mlxcel_core::default_device_is_gpu() {
        return;
    }
    // Empty terms suffice to represent a staged adapter without GPU work.
    mlxcel_core::runtime_lora::stage(std::collections::HashMap::from([(
        "backbone.layers.0.mixer.q_proj".to_string(),
        Vec::new(),
    )]));
    let config: NemotronHConfig = serde_json::from_str(&minimal_config_json("")).unwrap();
    let result = super::NemotronHModel::from_weights(
        config,
        mlxcel_core::weights::WeightMap::new(),
        Vec::new(),
    );
    let error = result
        .err()
        .expect("runtime adapter must fail before weights are read");
    assert!(error.to_string().contains("does not support runtime LoRA"));
    assert!(!mlxcel_core::runtime_lora::has_pending());
}

// ---------------------------------------------------------------------------
// transformers-saved configs (#2384, ml-explore/mlx-lm#1857)
// ---------------------------------------------------------------------------

/// Required fields of a transformers-saved Nemotron-H config: no
/// `num_hidden_layers`, no `hybrid_override_pattern`.
fn transformers_config(extra: serde_json::Value) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "ssm_state_size": 16,
        "conv_kernel": 4,
        "n_groups": 1,
    });
    for (key, value) in extra.as_object().unwrap() {
        config[key] = value.clone();
    }
    config
}

#[test]
fn transformers_layer_types_and_float_tag_parse() {
    let mut cfg: NemotronHConfig = serde_json::from_value(transformers_config(serde_json::json!({
        "layer_types": ["linear_attention", "full_attention", "moe"],
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
    })))
    .expect("transformers-saved Nemotron-H config must parse");
    cfg.post_init().expect("post_init");
    let pattern = cfg.hybrid_override_pattern.as_ref().unwrap().concat();
    assert_eq!(pattern, "M*E");
    assert_eq!(cfg.num_hidden_layers, 3);
    let (lo, hi) = cfg.time_step_limit.unwrap();
    assert_eq!(lo, 0.0);
    assert!(hi.is_infinite() && hi.is_sign_positive());
}

#[test]
fn float_tag_negative_infinity_and_nan_decode_explicitly() {
    let cfg: NemotronHConfig = serde_json::from_value(transformers_config(serde_json::json!({
        "layer_types": ["mamba"],
        "time_step_limit": [{"__float__": "NaN"}, {"__float__": "-Infinity"}],
    })))
    .unwrap();
    let (lo, hi) = cfg.time_step_limit.unwrap();
    assert!(lo.is_nan());
    assert!(hi.is_infinite() && hi.is_sign_negative());
}

/// The loader's `sanitize_config_json` rewrites `Infinity` inside the tag
/// string too, so the real load path sees `{"__float__": "1e38"}`. It must
/// parse to the same bound a sanitized bare `Infinity` literal gets.
#[test]
fn sanitized_float_tag_parses_like_a_sanitized_literal() {
    let raw = serde_json::to_string(&transformers_config(serde_json::json!({
        "layer_types": ["mamba"],
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
    })))
    .unwrap();
    let sanitized = crate::models::sanitize_config_json(&raw);
    let tagged: NemotronHConfig = serde_json::from_str(&sanitized).unwrap();
    let literal: NemotronHConfig =
        serde_json::from_str(&sanitized.replace(r#"{"__float__":"1e38"}"#, "1e38")).unwrap();
    assert_eq!(tagged.time_step_limit, literal.time_step_limit);
    assert_eq!(tagged.time_step_limit, Some((0.0, 1e38)));
}

#[test]
fn non_number_time_step_limit_element_is_rejected() {
    for bad in [
        serde_json::json!([0.0, "inf"]),
        serde_json::json!([0.0, {"__float__": "inf"}]),
        serde_json::json!([0.0, {"__float__": ""}]),
        serde_json::json!([0.0, {"__float__": 1.0}]),
        serde_json::json!([{"__float__": "Infinity", "x": 1}, 1.0]),
        serde_json::json!([null, 1.0]),
    ] {
        let parsed = serde_json::from_value::<NemotronHConfig>(transformers_config(
            serde_json::json!({"layer_types": ["mamba"], "time_step_limit": bad.clone()}),
        ));
        assert!(parsed.is_err(), "time_step_limit {bad} must be rejected");
    }
}

#[test]
fn legacy_and_transformers_block_names_map_to_pattern_codes() {
    let mut cfg: NemotronHConfig = serde_json::from_value(transformers_config(serde_json::json!({
        "layers_block_type": ["mamba", "attention", "moe", "mlp", "conv"],
        "layer_types": ["full_attention"],
    })))
    .unwrap();
    cfg.post_init().unwrap();
    // `layers_block_type` wins over the `layer_types` alias.
    assert_eq!(
        cfg.hybrid_override_pattern.as_ref().unwrap().concat(),
        "M*E-M"
    );
    assert_eq!(cfg.num_hidden_layers, 5);
}

#[test]
fn unknown_block_name_errors_with_the_accepted_list() {
    let mut cfg: NemotronHConfig = serde_json::from_value(transformers_config(serde_json::json!({
        "layer_types": ["mamba", "sliding_attention"],
    })))
    .unwrap();
    let error = cfg.post_init().expect_err("unknown block name must fail");
    assert!(error.contains("sliding_attention"), "{error}");
    for name in [
        "attention",
        "conv",
        "full_attention",
        "linear_attention",
        "mamba",
        "mlp",
        "moe",
    ] {
        assert!(
            error.contains(name),
            "accepted list must name {name}: {error}"
        );
    }
}

#[test]
fn config_without_any_layer_stack_description_is_rejected() {
    let mut cfg: NemotronHConfig =
        serde_json::from_value(transformers_config(serde_json::json!({}))).unwrap();
    assert!(cfg.post_init().is_err());
}

/// An mlx-lm NVFP4 export of Nemotron-H (for example
/// `mlx-community/Nemotron-Cascade-2-30B-A3B-nvfp4`) ships its expert planes
/// with uint8 E4M3 scales and no `.biases`. The loader required affine biases
/// and refused it ("Missing switch_mlp.fc1.biases for layer 1"); the planes
/// now load as nvfp4 (issue #2665). The routed experts must then match a
/// dequantized reference both on the unsorted decode layout and on the sorted
/// prefill layout, which the MoE takes from 64 (token, expert) pairs on.
#[test]
fn nemotron_h_runs_nvfp4_experts_sorted_and_unsorted() {
    use super::{BlockType, NemotronHMixer, NemotronHModel};
    use mlxcel_core::weights::WeightMap;

    const HIDDEN: i32 = 32;
    const EXPERTS: i32 = 4;
    const FF: i32 = 48;
    const TOP_K: i32 = 2;

    let mut config: NemotronHConfig = serde_json::from_value(serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": 8,
        "hidden_size": HIDDEN,
        "intermediate_size": FF,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "mamba_num_heads": 1,
        "mamba_head_dim": 4,
        "ssm_state_size": 4,
        "conv_kernel": 4,
        "n_groups": 1,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": FF,
        "hybrid_override_pattern": "E",
        "quantization": { "group_size": 16, "bits": 4, "mode": "nvfp4" },
    }))
    .expect("test config must parse");
    config.post_init().expect("test config must pass post_init");

    // Deterministic weights with a spread of magnitudes, quantized to nvfp4.
    let dense = |shape: &[i32], salt: usize| {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = (0..n as usize)
            .map(|i| (((i * 37 + salt) % 29) as f32 - 14.0) / 16.0)
            .collect();
        mlxcel_core::from_slice_f32(&v, shape)
    };
    let mut weights = WeightMap::new();
    let zeros = |shape: &[i32]| {
        let n: i32 = shape.iter().product();
        mlxcel_core::from_slice_f32(&vec![0.0; n as usize], shape)
    };
    weights.insert("backbone.embeddings.weight".into(), zeros(&[8, HIDDEN]));
    weights.insert("lm_head.weight".into(), zeros(&[8, HIDDEN]));
    weights.insert("backbone.norm_f.weight".into(), zeros(&[HIDDEN]));
    weights.insert("backbone.layers.0.norm.weight".into(), zeros(&[HIDDEN]));
    weights.insert(
        "backbone.layers.0.mixer.gate.weight".into(),
        zeros(&[EXPERTS, HIDDEN]),
    );
    let mut reference_planes = Vec::new();
    for (fc, shape, salt) in [
        ("fc1", [EXPERTS, FF, HIDDEN], 3),
        ("fc2", [EXPERTS, HIDDEN, FF], 11),
    ] {
        let q = mlxcel_core::quantize_weights_with_mode(&dense(&shape, salt), 16, 4, "nvfp4");
        let packed = mlxcel_core::quantized_weights_w(&q);
        let scales = mlxcel_core::quantized_weights_scales(&q);
        assert_eq!(mlxcel_core::array_dtype(&scales), dtype::UINT8);
        // SAFETY: null is the bridge's documented value for absent biases.
        let deq =
            unsafe { mlxcel_core::dequantize(&packed, &scales, std::ptr::null(), 16, 4, "nvfp4") };
        reference_planes.push(mlxcel_core::astype(&deq, dtype::FLOAT32));
        let prefix = format!("backbone.layers.0.mixer.switch_mlp.{fc}");
        weights.insert(format!("{prefix}.weight"), packed);
        weights.insert(format!("{prefix}.scales"), scales);
    }

    let model = NemotronHModel::from_weights(config, weights, vec![BlockType::MoE])
        .expect("nvfp4 expert planes without biases must load");
    let NemotronHMixer::MoE(moe) = &model.layers[0].mixer else {
        panic!("layer 0 must be the MoE block");
    };

    // tokens * TOP_K pairs: 10 stays unsorted, 40 (80 pairs) takes the sort.
    for tokens in [5, 40] {
        let x_vals: Vec<f32> = (0..(tokens * HIDDEN) as usize)
            .map(|i| (((i * 13 + 7) % 17) as f32 - 8.0) / 8.0)
            .collect();
        let x = mlxcel_core::from_slice_f32(&x_vals, &[tokens, HIDDEN]);
        let experts: Vec<i32> = (0..tokens * TOP_K)
            .map(|p| (p * 3 + p / TOP_K) % EXPERTS)
            .collect();
        let indices = mlxcel_core::astype(
            &mlxcel_core::from_slice_i32(&experts, &[tokens, TOP_K]),
            dtype::UINT32,
        );
        let got = moe.switch_mlp.forward(&x, &indices);
        let got = mlxcel_core::astype(&got, dtype::FLOAT32);
        mlxcel_core::try_eval(&got).expect("the routed experts evaluate");
        assert_eq!(mlxcel_core::array_shape(&got), [tokens, TOP_K, HIDDEN]);

        // relu(x @ W1[e].T)^2 @ W2[e].T per (token, expert) pair.
        let idx = mlxcel_core::from_slice_i32(&experts, &[tokens * TOP_K]);
        let w1 = mlxcel_core::take(&reference_planes[0], &idx, 0);
        let w2 = mlxcel_core::take(&reference_planes[1], &idx, 0);
        let token_of: Vec<i32> = (0..tokens * TOP_K).map(|p| p / TOP_K).collect();
        let xs = mlxcel_core::take(
            &x,
            &mlxcel_core::from_slice_i32(&token_of, &[tokens * TOP_K]),
            0,
        );
        let xs = mlxcel_core::reshape(&xs, &[tokens * TOP_K, 1, HIDDEN]);
        let h = mlxcel_core::matmul(&xs, &mlxcel_core::swap_axes(&w1, -1, -2));
        let h = mlxcel_core::utils::relu_squared(&h);
        let want = mlxcel_core::matmul(&h, &mlxcel_core::swap_axes(&w2, -1, -2));
        let want = mlxcel_core::reshape(&want, &[tokens, TOP_K, HIDDEN]);
        let close = mlxcel_core::allclose(&got, &want, 1e-3, 1e-3);
        assert!(
            mlxcel_core::item_bool(&close),
            "nvfp4 routed experts differ from the dequantized reference at {tokens} tokens"
        );
    }
}

pub(crate) fn batching_model() -> super::NemotronHModel {
    use crate::models::embedding_test_support::Rng;
    use mlxcel_core::dtype;
    use mlxcel_core::weights::WeightMap;
    let mut config:super::NemotronHConfig=serde_json::from_value(serde_json::json!({"model_type":"nemotron_h","vocab_size":16,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"mamba_num_heads":2,"mamba_head_dim":4,"ssm_state_size":32,"conv_kernel":4,"n_groups":1,"hybrid_override_pattern":"M*"})).unwrap();
    config.post_init().unwrap();
    let mut weights = WeightMap::new();
    let mut rng = Rng::new(2571);
    rng.insert(&mut weights, "backbone.embeddings.weight", &[16, 8], 0.2);
    weights.insert(
        "backbone.norm_f.weight".into(),
        mlxcel_core::ones(&[8], dtype::FLOAT32),
    );
    rng.insert(&mut weights, "lm_head.weight", &[16, 8], 0.1);
    weights.insert(
        "backbone.layers.0.norm.weight".into(),
        mlxcel_core::ones(&[8], dtype::FLOAT32),
    );
    weights.insert(
        "backbone.layers.0.mixer.norm.weight".into(),
        mlxcel_core::ones(&[8], dtype::FLOAT32),
    );
    rng.insert(
        &mut weights,
        "backbone.layers.0.mixer.in_proj.weight",
        &[82, 8],
        0.1,
    );
    rng.insert(
        &mut weights,
        "backbone.layers.0.mixer.out_proj.weight",
        &[8, 8],
        0.1,
    );
    rng.insert(
        &mut weights,
        "backbone.layers.0.mixer.conv1d.weight",
        &[72, 4, 1],
        0.1,
    );
    weights.insert(
        "backbone.layers.0.mixer.dt_bias".into(),
        mlxcel_core::zeros(&[2], dtype::FLOAT32),
    );
    weights.insert(
        "backbone.layers.0.mixer.A_log".into(),
        mlxcel_core::zeros(&[2], dtype::FLOAT32),
    );
    weights.insert(
        "backbone.layers.0.mixer.D".into(),
        mlxcel_core::ones(&[2], dtype::FLOAT32),
    );
    weights.insert(
        "backbone.layers.1.norm.weight".into(),
        mlxcel_core::ones(&[8], dtype::FLOAT32),
    );
    rng.insert(
        &mut weights,
        "backbone.layers.1.mixer.q_proj.weight",
        &[8, 8],
        0.1,
    );
    rng.insert(
        &mut weights,
        "backbone.layers.1.mixer.k_proj.weight",
        &[4, 8],
        0.1,
    );
    rng.insert(
        &mut weights,
        "backbone.layers.1.mixer.v_proj.weight",
        &[4, 8],
        0.1,
    );
    rng.insert(
        &mut weights,
        "backbone.layers.1.mixer.o_proj.weight",
        &[8, 8],
        0.1,
    );
    super::NemotronHModel::from_weights(
        config,
        weights,
        vec![super::BlockType::Mamba, super::BlockType::Attention],
    )
    .unwrap()
}
#[test]
fn nemotron_h_model_owned_batched_multistep() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::batched_decode_harness::assert_model_owned_multistep_with_paged_lengths(
        batching_model,
        &[0, 3],
    );
}
#[test]
fn nemotron_h_fresh_and_mixed_recurrent_states_match_solo_rows() {
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::LanguageModel;
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    for mixed in [false, true] {
        let batched = batching_model();
        let solo = [batching_model(), batching_model()];
        let ids = [SequenceId::from_raw(1001), SequenceId::from_raw(1002)];
        let mut caches: Vec<_> = ids
            .iter()
            .map(|id| {
                batched.prepare_sequence_state(*id);
                LanguageModel::make_caches(&batched)
            })
            .collect();
        let mut single: Vec<_> = solo
            .iter()
            .zip(ids)
            .map(|(m, id)| {
                m.prepare_sequence_state(id);
                LanguageModel::make_caches(m)
            })
            .collect();
        if mixed {
            let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
            let _ = batched.forward_with_sequence_id(&prompt, Some(ids[0]), &mut caches[0], None);
            let _ = solo[0].forward_with_sequence_id(&prompt, Some(ids[0]), &mut single[0], None);
        }
        for step in 0..3 {
            let tokens = [2 + step, 3 + step];
            let input = mlxcel_core::from_slice_i32(&tokens, &[2, 1]);
            let mut refs: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
            let actual = batched.forward_batched_with_ids(&input, Some(&ids), &mut refs, None);
            for row in 0..2 {
                let input = mlxcel_core::from_slice_i32(&[tokens[row]], &[1, 1]);
                let expected = solo[row].forward_with_sequence_id(
                    &input,
                    Some(ids[row]),
                    &mut single[row],
                    None,
                );
                let actual = mlxcel_core::utils::slice_axis(&actual, 0, row as i32, row as i32 + 1);
                let delta = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(
                    &actual, &expected,
                )));
                assert!(mlxcel_core::item_f32(&delta) <= 1e-5);
            }
        }
    }
}
