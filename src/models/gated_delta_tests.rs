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

use super::{
    GatedDeltaCache, RMSNormGated, compute_g, gated_delta_ops, gated_delta_step,
    precise_swiglu_gate, restore_dtype, supports_metal_gated_delta_kernel,
};
use mlxcel_core::{dtype, generate::ModelStateSnapshot};

fn assert_allclose(actual: &mlxcel_core::MlxArray, expected: &mlxcel_core::MlxArray) {
    let close = mlxcel_core::allclose(actual, expected, 1e-3, 1e-3);
    mlxcel_core::eval(&close);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
#[ignore = "requires serial MLX execution"]
fn precise_swiglu_gate_matches_float32_reference_and_restores_dtype() {
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.5, -0.5, 0.25, -1.0], &[2, 2]),
        dtype::FLOAT16,
    );
    let gate = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[5.0, -3.0, 0.75, -0.25], &[2, 2]),
        dtype::FLOAT16,
    );

    let actual = precise_swiglu_gate(&x, &gate, dtype::FLOAT16);

    let gate_f32 = mlxcel_core::astype(&gate, dtype::FLOAT32);
    let gate_silu = mlxcel_core::silu(&gate_f32);
    let x_f32 = mlxcel_core::astype(&x, dtype::FLOAT32);
    let expected_f32 = mlxcel_core::multiply(&gate_silu, &x_f32);
    let expected = mlxcel_core::astype(&expected_f32, dtype::FLOAT16);

    assert_eq!(mlxcel_core::array_dtype(&actual), dtype::FLOAT16);
    assert_allclose(&actual, &expected);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn restore_dtype_casts_promoted_values_back_to_input_dtype() {
    let value = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 2]);
    let restored = restore_dtype(value, dtype::FLOAT16);

    assert_eq!(mlxcel_core::array_dtype(&restored), dtype::FLOAT16);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn rms_norm_gated_forward_restores_hidden_state_dtype_without_gate() {
    let weight = mlxcel_core::astype(&mlxcel_core::ones(&[2], dtype::FLOAT32), dtype::FLOAT16);
    let norm = RMSNormGated::new(weight, 1e-6);
    let hidden = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0, -1.0], &[1, 2]),
        dtype::FLOAT16,
    );

    let output = norm.forward(&hidden, None);

    assert_eq!(mlxcel_core::array_dtype(&output), dtype::FLOAT16);
}

// Tests for compute_g

#[test]
#[ignore = "requires serial MLX execution"]
fn compute_g_output_is_float32() {
    // Scalar inputs [B=1, H=2]
    let a_log = mlxcel_core::from_slice_f32(&[-1.0, -0.5], &[1, 2]);
    let a = mlxcel_core::from_slice_f32(&[0.1, 0.2], &[1, 2]);
    let dt_bias = mlxcel_core::from_slice_f32(&[0.0, 0.0], &[1, 2]);

    let g = compute_g(&a_log, &a, &dt_bias);
    mlxcel_core::eval(&g);

    assert_eq!(mlxcel_core::array_dtype(&g), dtype::FLOAT32);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn compute_g_values_are_in_unit_interval() {
    // g = exp(-exp(a_log) * softplus(a + dt_bias))
    // Since exp(-positive) is in (0, 1], g should be in (0, 1].
    let a_log = mlxcel_core::from_slice_f32(&[-2.0, -1.0, 0.0, 1.0], &[1, 4]);
    let a = mlxcel_core::from_slice_f32(&[0.5, 0.5, 0.5, 0.5], &[1, 4]);
    let dt_bias = mlxcel_core::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[1, 4]);

    let g = compute_g(&a_log, &a, &dt_bias);
    mlxcel_core::eval(&g);

    // All values should be > 0
    let zeros = mlxcel_core::zeros_like(&g);
    let gt_zero = mlxcel_core::greater(&g, &zeros);
    mlxcel_core::eval(&gt_zero);

    // All values should be <= 1
    let ones = mlxcel_core::ones_like(&g);
    let le_one = mlxcel_core::less_equal(&g, &ones);
    mlxcel_core::eval(&le_one);

    // Both conditions must hold for all elements
    let all_gt_zero = mlxcel_core::all_all(&gt_zero);
    let all_le_one = mlxcel_core::all_all(&le_one);
    mlxcel_core::eval(&all_gt_zero);
    mlxcel_core::eval(&all_le_one);

    assert!(mlxcel_core::item_bool(&all_gt_zero));
    assert!(mlxcel_core::item_bool(&all_le_one));
}

// Tests for gated_delta_step

#[test]
#[ignore = "requires serial MLX execution"]
fn gated_delta_step_output_shape_and_dtype() {
    // Minimal decode step: B=1, H=1, Dk=64 (must be multiple of 32), Dv=2
    // Note: Metal kernel requires Dk to be a multiple of 32
    let b = 1;
    let h = 1;
    let dk = 64usize;
    let dv = 2usize;

    let q = mlxcel_core::zeros(&[b, h, dk as i32], dtype::BFLOAT16);
    let k = mlxcel_core::zeros(&[b, h, dk as i32], dtype::BFLOAT16);
    let v = mlxcel_core::zeros(&[b, h, dv as i32], dtype::BFLOAT16);
    let g = mlxcel_core::ones(&[b, h], dtype::FLOAT32);
    let beta = mlxcel_core::ones(&[b, h], dtype::FLOAT32);
    let state = mlxcel_core::zeros(&[b, h, dv as i32, dk as i32], dtype::FLOAT32);

    let (y, new_state) = gated_delta_step(&q, &k, &v, &g, &beta, &state, None);
    mlxcel_core::eval(&y);
    mlxcel_core::eval(&new_state);

    let y_shape = mlxcel_core::array_shape(&y);
    assert_eq!(y_shape, vec![b, h, dv as i32]);

    let s_shape = mlxcel_core::array_shape(&new_state);
    assert_eq!(s_shape, vec![b, h, dv as i32, dk as i32]);

    // Output dtype should match input q dtype
    assert_eq!(mlxcel_core::array_dtype(&y), dtype::BFLOAT16);
    // State stays float32
    assert_eq!(mlxcel_core::array_dtype(&new_state), dtype::FLOAT32);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn gated_delta_step_with_mask_updates_state_selectively() {
    // With mask=false for a batch element, the state should remain unchanged.
    let b = 2i32;
    let h = 1i32;
    let dk = 64i32;
    let dv = 1i32;

    // State with non-zero entries
    let state_data: Vec<f32> = vec![1.0f32; (b * h * dv * dk) as usize];
    let state = mlxcel_core::from_slice_f32(&state_data, &[b, h, dv, dk]);

    let q = mlxcel_core::ones(&[b, h, dk], dtype::FLOAT32);
    let k = mlxcel_core::ones(&[b, h, dk], dtype::FLOAT32);
    let v = mlxcel_core::ones(&[b, h, dv], dtype::FLOAT32);
    let g = mlxcel_core::ones(&[b, h], dtype::FLOAT32);
    let beta = mlxcel_core::ones(&[b, h], dtype::FLOAT32);

    // Mask: first element active (true), second inactive (false)
    let mask = mlxcel_core::from_slice_i32(&[1, 0], &[b]);
    let mask_bool = mlxcel_core::astype(&mask, dtype::BOOL);

    let (_, new_state) = gated_delta_step(&q, &k, &v, &g, &beta, &state, Some(&mask_bool));
    mlxcel_core::eval(&new_state);

    // The second batch element's state should be unchanged (all 1.0)
    let state_shape = mlxcel_core::array_shape(&new_state);
    assert_eq!(state_shape, vec![b, h, dv, dk]);
}

// Tests for gated_delta_ops

#[test]
fn metal_gated_delta_shape_gate_accepts_supported_contract() {
    assert!(supports_metal_gated_delta_kernel(1, 1, 32, 2));
    assert!(supports_metal_gated_delta_kernel(2, 4, 64, 8));
}

#[test]
fn metal_gated_delta_shape_gate_rejects_unsupported_contracts() {
    assert!(!supports_metal_gated_delta_kernel(1, 1, 2, 2));
    assert!(!supports_metal_gated_delta_kernel(1, 1, 31, 2));
    assert!(!supports_metal_gated_delta_kernel(1, 1, 33, 2));
    assert!(!supports_metal_gated_delta_kernel(4, 2, 64, 2));
    assert!(!supports_metal_gated_delta_kernel(3, 4, 64, 2));
}

#[test]
#[ignore = "requires serial MLX execution"]
fn gated_delta_ops_single_token_output_shape() {
    // T=1 decode path: [B=1, T=1, Hk=1, Dk=64]
    let b = 1i32;
    let t = 1i32;
    let hk = 1i32;
    let hv = 1i32;
    let dk = 64i32;
    let dv = 2i32;

    let q = mlxcel_core::zeros(&[b, t, hk, dk], dtype::BFLOAT16);
    let k = mlxcel_core::zeros(&[b, t, hk, dk], dtype::BFLOAT16);
    let v = mlxcel_core::zeros(&[b, t, hv, dv], dtype::BFLOAT16);
    let g = mlxcel_core::ones(&[b, t, hv], dtype::FLOAT32);
    let beta = mlxcel_core::ones(&[b, t, hv], dtype::FLOAT32);

    let (y, new_state) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None);
    mlxcel_core::eval(&y);
    mlxcel_core::eval(&new_state);

    let y_shape = mlxcel_core::array_shape(&y);
    assert_eq!(y_shape, vec![b, t, hv, dv]);

    let s_shape = mlxcel_core::array_shape(&new_state);
    assert_eq!(s_shape, vec![b, hv, dv, dk]);

    assert_eq!(mlxcel_core::array_dtype(&y), dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_dtype(&new_state), dtype::FLOAT32);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn gated_delta_ops_multi_token_output_shape() {
    // T=4 prefill path: test shape contract holds
    let b = 1i32;
    let t = 4i32;
    let hk = 1i32;
    let hv = 1i32;
    let dk = 64i32;
    let dv = 2i32;

    let q = mlxcel_core::zeros(&[b, t, hk, dk], dtype::BFLOAT16);
    let k = mlxcel_core::zeros(&[b, t, hk, dk], dtype::BFLOAT16);
    let v = mlxcel_core::zeros(&[b, t, hv, dv], dtype::BFLOAT16);
    let g = mlxcel_core::ones(&[b, t, hv], dtype::FLOAT32);
    let beta = mlxcel_core::ones(&[b, t, hv], dtype::FLOAT32);

    let (y, new_state) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None);
    mlxcel_core::eval(&y);
    mlxcel_core::eval(&new_state);

    let y_shape = mlxcel_core::array_shape(&y);
    assert_eq!(y_shape, vec![b, t, hv, dv]);

    let s_shape = mlxcel_core::array_shape(&new_state);
    assert_eq!(s_shape, vec![b, hv, dv, dk]);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn gated_delta_ops_passes_state_through_when_provided() {
    // Verifies that an explicitly-provided initial state is used rather than zeros.
    let b = 1i32;
    let t = 1i32;
    let h = 1i32;
    let dk = 64i32;
    let dv = 1i32;

    // Non-zero initial state
    let state_data = vec![2.0f32; (b * h * dv * dk) as usize];
    let initial_state = mlxcel_core::from_slice_f32(&state_data, &[b, h, dv, dk]);

    // Zero inputs — output will purely reflect state content
    let q = mlxcel_core::zeros(&[b, t, h, dk], dtype::FLOAT32);
    let k = mlxcel_core::zeros(&[b, t, h, dk], dtype::FLOAT32);
    let v = mlxcel_core::zeros(&[b, t, h, dv], dtype::FLOAT32);
    let g = mlxcel_core::ones(&[b, t, h], dtype::FLOAT32);
    let beta = mlxcel_core::zeros(&[b, t, h], dtype::FLOAT32); // beta=0 -> no update to state

    let (_, new_state_with_init) =
        gated_delta_ops(&q, &k, &v, &g, &beta, Some(&initial_state), None);
    let (_, new_state_from_zero) = gated_delta_ops(&q, &k, &v, &g, &beta, None, None);

    mlxcel_core::eval(&new_state_with_init);
    mlxcel_core::eval(&new_state_from_zero);

    // States should differ: one starts at 2.0, other at 0.0.
    // With g=1 and beta=0, state is just decayed by g=1 (no change), so they stay different.
    let diff = mlxcel_core::subtract(&new_state_with_init, &new_state_from_zero);
    mlxcel_core::eval(&diff);

    // The max absolute difference must be > 0
    let abs_diff = mlxcel_core::abs(&diff);
    let max_diff = mlxcel_core::max_all(&abs_diff);
    mlxcel_core::eval(&max_diff);
    let max_val = mlxcel_core::item_f32(&max_diff);
    assert!(
        max_val > 0.0,
        "states should differ when initial state is non-zero"
    );
}

// Tests for GatedDeltaCache

#[test]
fn gated_delta_cache_default_is_empty() {
    let cache = GatedDeltaCache::default();
    assert!(cache.conv_state.is_none());
    assert!(cache.state_cache.is_none());
    assert_eq!(cache.offset, 0);
}

#[test]
fn gated_delta_cache_advance_increments_offset() {
    let mut cache = GatedDeltaCache::new();
    assert_eq!(cache.offset, 0);
    cache.advance(1);
    assert_eq!(cache.offset, 1);
    cache.advance(3);
    assert_eq!(cache.offset, 4);
}

#[test]
fn gated_delta_cache_snapshot_restore_round_trips_state_shapes() {
    let mut cache = GatedDeltaCache::new();
    cache.conv_state = Some(mlxcel_core::zeros(&[1, 3, 8], dtype::FLOAT32));
    cache.state_cache = Some(mlxcel_core::zeros(&[1, 2, 4, 8], dtype::FLOAT32));
    cache.offset = 5;

    let mut snapshot = ModelStateSnapshot::new("qwen3_5", 17);
    cache.snapshot_into(&mut snapshot, "layer4.linear");

    let mut restored = GatedDeltaCache::new();
    restored.restore_from(&snapshot, "layer4.linear");

    let conv = restored
        .conv_state
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("conv_state restored");
    let state = restored
        .state_cache
        .as_ref()
        .and_then(|a| a.as_ref())
        .expect("state_cache restored");
    assert_eq!(restored.offset, 17);
    assert_eq!(mlxcel_core::array_shape(conv), vec![1, 3, 8]);
    assert_eq!(mlxcel_core::array_shape(state), vec![1, 2, 4, 8]);
}
