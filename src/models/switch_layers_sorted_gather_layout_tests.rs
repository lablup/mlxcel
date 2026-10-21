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

//! Issue #2462 A1: the layouts that made the sorted `gather_mm` branch
//! (`gather_mm_rhs`) read a size-1 axis stride as its leading dimension. See
//! `switch_layers_sorted_gather_history_tests.rs` for the mechanism and the
//! allocation-history tests.
//!
//! GPU-only; skipped on a CPU-only build.

use mlxcel_core::dtype;

use super::SwitchLinear;
use super::sorted_gather_test_support::{
    EXPERTS, HIDDEN, INTER, Lcg, assert_matches_reference, bf16_plane, kernel_activations,
    reference, report, sorted_expert_ids, to_vec,
};
use crate::models::embedding_test_support::mlx_test_guard;

/// Issue #2462 A1 mechanism check: the activations as `[rows, 1, K]` with a
/// stride of 1 on the size-1 axis. MLX still calls that row-contiguous, and
/// `gather_mm_rhs` read `strides(-2)` as the leading dimension.
#[test]
fn sorted_gather_mm_with_a_unit_stride_on_the_singleton_axis() {
    if !mlxcel_core::gpu_backend_available() {
        return;
    }
    let _guard = mlx_test_guard();
    let mut rng = Lcg(0x2462_0001);
    let (weight, exact_w) = bf16_plane(&mut rng, INTER, HIDDEN);
    let ids = sorted_expert_ids(&mut rng, 24);
    let rows = ids.len() as i32;
    let flat = rng.f32_array(&[rows, HIDDEN, 1], 1.0);
    // [rows, 1, HIDDEN] with strides [HIDDEN, 1, 1].
    let x = mlxcel_core::swap_axes(&flat, 1, 2);
    mlxcel_core::eval(&x);
    assert_eq!(
        mlxcel_core::array_strides(&x),
        vec![i64::from(HIDDEN), 1, 1],
        "the view must carry a unit stride on the singleton axis"
    );
    let x_host = kernel_activations(&x);
    let idx = mlxcel_core::from_slice_u32(&ids, &[rows]);
    let layer = SwitchLinear::Regular { weight };
    let want = reference(&x_host, &exact_w, &ids, INTER as usize, HIDDEN as usize);
    let sorted = to_vec(&layer.forward(&x, &idx, true));
    let unsorted = to_vec(&layer.forward(&x, &idx, false));
    report(
        "unit-stride sorted vs f64",
        &sorted,
        &want,
        &ids,
        INTER as usize,
    );
    report(
        "unit-stride unsorted vs f64",
        &unsorted,
        &want,
        &ids,
        INTER as usize,
    );
    assert_matches_reference("unit-stride unsorted", &unsorted, &want);
    assert_matches_reference("unit-stride sorted", &sorted, &want);
}

/// Issue #2462 A1: the expert plane stored in transposed layout, as a
/// materialized copy of a `swap_axes` view (what a loader's layout fix-up
/// leaves behind). `swap_axes` on it inside the forward is then
/// row-contiguous `[E, K, N]`, and the grouped GEMM takes its row-major B
/// instantiation.
#[test]
fn sorted_gather_mm_with_a_transposed_layout_plane() {
    if !mlxcel_core::gpu_backend_available() {
        return;
    }
    let _guard = mlx_test_guard();
    let mut rng = Lcg(0x2462_0002);
    // [E, K, N] natural, then viewed as [E, N, K] and copied in that layout.
    let kn = mlxcel_core::astype(
        &rng.f32_array(&[EXPERTS, HIDDEN, INTER], 0.05),
        dtype::BFLOAT16,
    );
    let plane = mlxcel_core::copy(&mlxcel_core::swap_axes(&kn, -1, -2));
    mlxcel_core::eval(&plane);
    assert_eq!(
        mlxcel_core::array_strides(&plane),
        vec![i64::from(HIDDEN) * i64::from(INTER), 1, i64::from(HIDDEN)],
        "the plane must be stored in transposed layout"
    );
    let exact_w = to_vec(&plane);
    let ids = sorted_expert_ids(&mut rng, 24);
    let rows = ids.len() as i32;
    let x = rng.f32_array(&[rows, 1, HIDDEN], 1.0);
    mlxcel_core::eval(&x);
    let x_host = kernel_activations(&x);
    let idx = mlxcel_core::from_slice_u32(&ids, &[rows]);
    let layer = SwitchLinear::Regular { weight: plane };
    let want = reference(&x_host, &exact_w, &ids, INTER as usize, HIDDEN as usize);
    let sorted = to_vec(&layer.forward(&x, &idx, true));
    let unsorted = to_vec(&layer.forward(&x, &idx, false));
    report(
        "transposed-plane sorted vs f64",
        &sorted,
        &want,
        &ids,
        INTER as usize,
    );
    report(
        "transposed-plane unsorted vs f64",
        &unsorted,
        &want,
        &ids,
        INTER as usize,
    );
    assert_matches_reference("transposed-plane unsorted", &unsorted, &want);
    assert_matches_reference("transposed-plane sorted", &sorted, &want);
}

/// Issue #2462 A1 through the Privacy Filter's own op chain: a gathered expert
/// output plus a per-expert bias, `out + expand_dims(take(bias, ids), -2)`,
/// evaluated while `out` is still referenced so the add can only inherit the
/// bias operand's layout (strides `[N, 1, 1]`), then widened to f32 and fed to
/// the sorted gather as its `[rows, 1, K]` activations.
#[test]
fn sorted_gather_mm_after_a_biased_expert_add() {
    if !mlxcel_core::gpu_backend_available() {
        return;
    }
    let _guard = mlx_test_guard();
    let mut rng = Lcg(0x2462_0003);
    let (first_plane, _) = bf16_plane(&mut rng, HIDDEN, HIDDEN);
    let bias = mlxcel_core::astype(&rng.f32_array(&[EXPERTS, HIDDEN], 0.05), dtype::BFLOAT16);
    let (down_plane, exact_down) = bf16_plane(&mut rng, INTER, HIDDEN);
    let ids = sorted_expert_ids(&mut rng, 24);
    let rows = ids.len() as i32;
    let x = rng.f32_array(&[rows, 1, HIDDEN], 1.0);
    let idx = mlxcel_core::from_slice_u32(&ids, &[rows]);

    // The first plane's biased output, as `ExpertLinear::forward` builds it.
    let first = SwitchLinear::Regular {
        weight: first_plane,
    };
    let out = first.forward(&x, &idx, false);
    // The gathered bias is held by the add alone, so the add can donate it;
    // `out` is still alive here, so the add cannot donate that operand and
    // takes the bias operand's layout. The test is only meaningful if it did.
    let biased = mlxcel_core::add(
        &out,
        &mlxcel_core::expand_dims(&mlxcel_core::take(&bias, &idx, 0), -2),
    );
    mlxcel_core::eval(&biased);
    let strides = mlxcel_core::array_strides(&biased);
    assert_eq!(
        strides,
        vec![i64::from(HIDDEN), 1, 1],
        "the biased output did not inherit the bias layout; the layout under test is not reached"
    );
    drop(out);

    let activations = mlxcel_core::astype(&biased, dtype::FLOAT32);
    mlxcel_core::eval(&activations);
    let act_host = kernel_activations(&activations);
    let want = reference(
        &act_host,
        &exact_down,
        &ids,
        INTER as usize,
        HIDDEN as usize,
    );
    let down = SwitchLinear::Regular { weight: down_plane };
    let sorted = to_vec(&down.forward(&activations, &idx, true));
    let unsorted = to_vec(&down.forward(&activations, &idx, false));
    report(
        "biased-add sorted vs f64",
        &sorted,
        &want,
        &ids,
        INTER as usize,
    );
    assert_matches_reference("biased-add unsorted", &unsorted, &want);
    assert_matches_reference("biased-add sorted", &sorted, &want);
}
