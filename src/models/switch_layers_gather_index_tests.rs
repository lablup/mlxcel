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

//! Shared gather indices in `SwitchGLU` (issue #1713).
//!
//! `SwitchGLU` now builds the `uint32` expert ids and the default
//! `lhs_indices` once per block instead of letting every `gather_qmm` call
//! synthesize them. That only removes graph nodes if the arrays it passes are
//! the ones MLX would have built, so these tests pin both: the prepared
//! indices against MLX's `indices_or_default` + `broadcast_arrays` values,
//! and the whole block, bit for bit, against the previous per-call form in the
//! unsorted (decode, small prefill) and sorted (large prefill) shapes.

use mlxcel_core::streams::lock_default_device;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::{
    SwitchGLU, broadcast_shape, gather_sort, insert_honest_affine_swiglu_experts,
    prepare_gather_indices, scatter_unsort,
};

const EXPERTS: i32 = 4;
const HIDDEN: i32 = 64;
const DFF: i32 = 128;

fn honest_moe() -> SwitchGLU {
    let mut weights = WeightMap::new();
    insert_honest_affine_swiglu_experts(&mut weights, "moe.switch_mlp", EXPERTS, HIDDEN, DFF);
    SwitchGLU::from_weights(&weights, "moe.switch_mlp", 64, 4).expect("honest experts load")
}

fn activations(tokens: i32) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..tokens * HIDDEN)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
        .collect();
    let x = mlxcel_core::from_slice_f32(&data, &[tokens, HIDDEN]);
    mlxcel_core::astype(&x, dtype::BFLOAT16)
}

/// Distinct experts per row, spread over every expert.
fn routing(tokens: i32, top_k: i32) -> Vec<i32> {
    (0..tokens)
        .flat_map(|t| (0..top_k).map(move |k| (t * 3 + k) % EXPERTS))
        .collect()
}

/// `SwitchGLU::forward` as it was before issue #1713: two `expand_dims`, and
/// every projection lets `gather_qmm` derive its own indices.
fn per_call_reference(moe: &SwitchGLU, x: &MlxArray, indices: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(indices);
    let x_exp = mlxcel_core::expand_dims(x, -2);
    let x_exp = mlxcel_core::expand_dims(&x_exp, -3);
    if shape[0] * shape[1] >= 64 {
        let (sorted_x, sorted_idx, inv_order) = gather_sort(&x_exp, indices);
        let gate = moe.gate_proj.forward(&sorted_x, &sorted_idx, true);
        let up = moe.up_proj.forward(&sorted_x, &sorted_idx, true);
        let activated = moe.activate(&gate, &up);
        let out = moe.down_proj.forward(&activated, &sorted_idx, true);
        scatter_unsort(&out, &inv_order, &shape)
    } else {
        let gate = moe.gate_proj.forward(&x_exp, indices, false);
        let up = moe.up_proj.forward(&x_exp, indices, false);
        let activated = moe.activate(&gate, &up);
        let out = moe.down_proj.forward(&activated, indices, false);
        mlxcel_core::squeeze_axis(&out, -2)
    }
}

fn assert_bit_identical(actual: &MlxArray, expected: &MlxArray, what: &str) {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected),
        "{what}: shape"
    );
    assert_eq!(
        mlxcel_core::array_dtype(actual),
        mlxcel_core::array_dtype(expected),
        "{what}: dtype"
    );
    assert!(
        mlxcel_core::item_bool(&mlxcel_core::array_equal(actual, expected, false)),
        "{what}: values differ from the per-call gather form"
    );
}

#[test]
fn broadcast_shape_follows_numpy_rules() {
    assert_eq!(broadcast_shape(&[3, 1], &[3, 2]), Some(vec![3, 2]));
    assert_eq!(broadcast_shape(&[6], &[6]), Some(vec![6]));
    assert_eq!(broadcast_shape(&[1], &[2, 4]), Some(vec![2, 4]));
    assert_eq!(broadcast_shape(&[], &[5]), Some(vec![5]));
    assert_eq!(broadcast_shape(&[3, 1], &[4, 2]), None);
}

#[test]
fn prepared_indices_match_what_gather_qmm_would_build() {
    let _device = lock_default_device();
    // Unsorted form: x `[n, 1, 1, d]` gives a batch of `[n, 1]`, broadcast
    // against the `[n, k]` expert ids.
    let x_exp = mlxcel_core::zeros(&[3, 1, 1, HIDDEN], dtype::BFLOAT16);
    let ids = mlxcel_core::from_slice_i32(&[0, 1, 2, 3, 1, 0], &[3, 2]);
    let gi = prepare_gather_indices(&x_exp, &ids);
    let lhs = gi.lhs.as_ref().expect("broadcastable shapes build an lhs");
    let expected_lhs = mlxcel_core::from_slice_u32(&[0, 0, 1, 1, 2, 2], &[3, 2]);
    assert_bit_identical(lhs, &expected_lhs, "unsorted lhs");
    let expected_rhs = mlxcel_core::from_slice_u32(&[0, 1, 2, 3, 1, 0], &[3, 2]);
    assert_bit_identical(&gi.rhs, &expected_rhs, "unsorted rhs");

    // Sorted form: x `[n * k, 1, d]` and flat ids share batch `[n * k]`.
    let x_sorted = mlxcel_core::zeros(&[6, 1, HIDDEN], dtype::BFLOAT16);
    let flat = mlxcel_core::from_slice_i32(&[0, 0, 1, 1, 2, 3], &[6]);
    let gi = prepare_gather_indices(&x_sorted, &flat);
    let lhs = gi.lhs.as_ref().expect("equal shapes build an lhs");
    let expected_lhs = mlxcel_core::from_slice_u32(&[0, 1, 2, 3, 4, 5], &[6]);
    assert_bit_identical(lhs, &expected_lhs, "sorted lhs");

    // Shapes MLX would reject are left to MLX.
    let bad = mlxcel_core::from_slice_i32(&[0; 8], &[4, 2]);
    assert!(prepare_gather_indices(&x_exp, &bad).lhs.is_none());
}

#[test]
fn switch_glu_is_bit_identical_to_the_per_call_gather_form() {
    let _device = lock_default_device();
    let moe = honest_moe();
    // (tokens, top_k): single-token decode, unsorted small prefill, and a
    // sorted prefill (tokens * top_k >= 64).
    for (tokens, top_k) in [(1, 2), (3, 2), (40, 2)] {
        let x = activations(tokens);
        let ids = routing(tokens, top_k);
        let as_i32 = mlxcel_core::from_slice_i32(&ids, &[tokens, top_k]);
        let as_u32: Vec<u32> = ids.iter().map(|&e| e as u32).collect();
        let as_u32 = mlxcel_core::from_slice_u32(&as_u32, &[tokens, top_k]);
        for (label, indices) in [("int32", &as_i32), ("uint32", &as_u32)] {
            let actual = moe.forward(&x, indices);
            let expected = per_call_reference(&moe, &x, indices);
            assert_bit_identical(
                &actual,
                &expected,
                &format!("forward tokens={tokens} top_k={top_k} {label} ids"),
            );
        }
    }
}

#[test]
fn expert_scales_path_is_bit_identical_to_the_per_call_gather_form() {
    let _device = lock_default_device();
    let moe = honest_moe();
    let (tokens, top_k) = (3, 2);
    let x = activations(tokens);
    let indices = mlxcel_core::from_slice_i32(&routing(tokens, top_k), &[tokens, top_k]);
    let gate_scale = mlxcel_core::from_slice_f32(&[1.0, 1.5, 0.5, 2.0], &[EXPERTS]);
    let out_scale = mlxcel_core::from_slice_f32(&[0.75, 2.0, 1.25, 0.5], &[EXPERTS]);
    let actual = moe.forward_with_expert_scales(&x, &indices, Some(&gate_scale), Some(&out_scale));

    let scale_for = |scale: &MlxArray, like: &MlxArray| {
        let selected = mlxcel_core::take(scale, &indices, 0);
        let selected = mlxcel_core::expand_dims(&selected, -1);
        let selected = mlxcel_core::expand_dims(&selected, -1);
        mlxcel_core::astype(&selected, mlxcel_core::array_dtype(like))
    };
    let x_exp = mlxcel_core::expand_dims(&x, -2);
    let x_exp = mlxcel_core::expand_dims(&x_exp, -3);
    let gate = moe.gate_proj.forward(&x_exp, &indices, false);
    let gate = mlxcel_core::multiply(&gate, &scale_for(&gate_scale, &gate));
    let up = moe.up_proj.forward(&x_exp, &indices, false);
    let activated = moe.activate(&gate, &up);
    let out = moe.down_proj.forward(&activated, &indices, false);
    let out = mlxcel_core::multiply(&out, &scale_for(&out_scale, &out));
    let expected = mlxcel_core::squeeze_axis(&out, -2);
    assert_bit_identical(&actual, &expected, "forward_with_expert_scales");
}
