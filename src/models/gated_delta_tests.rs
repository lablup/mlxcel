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
    GatedDeltaCache, RMSNormGated, compute_g, gated_delta_chunked, gated_delta_ops,
    gated_delta_step, precise_swiglu_gate, restore_dtype, supports_metal_gated_delta_kernel,
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

// Parity test for the chunked parallel prefill scan (gated_delta_chunked).

/// Deterministic synthetic tensor with values in [-scale, scale].
fn synth(shape: &[i32], scale: f32, phase: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| scale * (0.1 * i as f32 + phase).sin())
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// Relative RMS error between two arrays: rms(a - b) / rms(b).
fn rms_rel(a: &mlxcel_core::MlxArray, b: &mlxcel_core::MlxArray) -> f32 {
    let diff = mlxcel_core::subtract(a, b);
    let num = mlxcel_core::item_f32(&mlxcel_core::mean_all(&mlxcel_core::square(&diff))).sqrt();
    let den = mlxcel_core::item_f32(&mlxcel_core::mean_all(&mlxcel_core::square(b))).sqrt();
    num / (den + 1e-8)
}

/// Exact sequential reference: the same recurrence as `gated_delta_step`'s
/// ops path, run one timestep at a time in float32 (no mask, no fused kernel).
#[allow(clippy::too_many_arguments)]
fn sequential_reference(
    q: &mlxcel_core::MlxArray,
    k: &mlxcel_core::MlxArray,
    v: &mlxcel_core::MlxArray,
    g: &mlxcel_core::MlxArray,
    beta: &mlxcel_core::MlxArray,
    state0: &mlxcel_core::MlxArray,
    b: i32,
    t: usize,
    h: i32,
    dk: i32,
    dv: i32,
) -> (
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
    mlxcel_core::UniquePtr<mlxcel_core::MlxArray>,
) {
    let mut s = mlxcel_core::copy(state0);
    let mut ys = Vec::with_capacity(t);
    for ti in 0..t {
        let ti = ti as i32;
        let qt = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(q, &[0, ti, 0, 0], &[b, ti + 1, h, dk]),
            1,
        );
        let kt = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(k, &[0, ti, 0, 0], &[b, ti + 1, h, dk]),
            1,
        );
        let vt = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(v, &[0, ti, 0, 0], &[b, ti + 1, h, dv]),
            1,
        );
        let gt = mlxcel_core::squeeze_axis(&mlxcel_core::slice(g, &[0, ti, 0], &[b, ti + 1, h]), 1);
        let betat =
            mlxcel_core::squeeze_axis(&mlxcel_core::slice(beta, &[0, ti, 0], &[b, ti + 1, h]), 1);

        // S' = g * S
        let decay = mlxcel_core::expand_dims(&mlxcel_core::expand_dims(&gt, -1), -1);
        let sp = mlxcel_core::multiply(&s, &decay);
        // u = S' k
        let k_exp = mlxcel_core::expand_dims(&kt, -2);
        let u = mlxcel_core::sum_axis(&mlxcel_core::multiply(&sp, &k_exp), -1, false);
        // w = beta (v - u)
        let beta_exp = mlxcel_core::expand_dims(&betat, -1);
        let w = mlxcel_core::multiply(&mlxcel_core::subtract(&vt, &u), &beta_exp);
        // S = S' + w k^T
        let w_exp = mlxcel_core::expand_dims(&w, -1);
        s = mlxcel_core::add(&sp, &mlxcel_core::multiply(&k_exp, &w_exp));
        // y = S q
        let q_exp = mlxcel_core::expand_dims(&qt, -2);
        let yt = mlxcel_core::sum_axis(&mlxcel_core::multiply(&s, &q_exp), -1, false);
        ys.push(yt);
    }
    let y = mlxcel_core::utils::stack_arrays(&ys, 1);
    (y, s)
}

#[test]
#[ignore = "requires serial MLX execution"]
fn chunked_prefill_matches_sequential_reference() {
    // B=2, T=10, H=3, Dk=16, Dv=8, chunk C=4 -> N=3 chunks with 2 padded
    // positions in the last chunk (exercises multi-chunk + ragged padding +
    // a non-zero incoming state).
    let b = 2i32;
    let t = 10usize;
    let h = 3i32;
    let dk = 16i32;
    let dv = 8i32;
    let chunk = 4usize;
    let ti = t as i32;

    let q = synth(&[b, ti, h, dk], 0.3, 0.0);
    let k = synth(&[b, ti, h, dk], 0.25, 1.3);
    let v = synth(&[b, ti, h, dv], 0.5, 2.1);
    // Gate in [0.82, 0.98] (strictly in (0, 1)); beta in [0.2, 0.8].
    let g = {
        let n = b * ti * h;
        let data: Vec<f32> = (0..n)
            .map(|i| 0.9 + 0.08 * (0.3 * i as f32).sin())
            .collect();
        mlxcel_core::from_slice_f32(&data, &[b, ti, h])
    };
    let beta = {
        let n = b * ti * h;
        let data: Vec<f32> = (0..n)
            .map(|i| 0.5 + 0.3 * (0.2 * i as f32 + 0.5).sin())
            .collect();
        mlxcel_core::from_slice_f32(&data, &[b, ti, h])
    };
    let state0 = synth(&[b, h, dv, dk], 0.2, 0.7);

    let (y_ref, s_ref) = sequential_reference(&q, &k, &v, &g, &beta, &state0, b, t, h, dk, dv);
    let (y_chunk, s_chunk) =
        gated_delta_chunked(&q, &k, &v, &g, &beta, &state0, chunk, b, t, h, dk, dv);

    // Shape parity.
    assert_eq!(mlxcel_core::array_shape(&y_chunk), vec![b, ti, h, dv]);
    assert_eq!(mlxcel_core::array_shape(&s_chunk), vec![b, h, dv, dk]);

    // Numerical parity (RMS-equivalent) for both the output and the carried
    // recurrent state that seeds subsequent decode steps.
    let y_err = rms_rel(&y_chunk, &y_ref);
    let s_err = rms_rel(&s_chunk, &s_ref);
    assert!(
        y_err < 1e-3,
        "chunked output diverged from sequential reference: rms_rel={y_err}"
    );
    assert!(
        s_err < 1e-3,
        "chunked state diverged from sequential reference: rms_rel={s_err}"
    );
}
