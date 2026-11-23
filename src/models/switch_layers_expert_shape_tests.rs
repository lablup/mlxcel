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

//! Mismatched per-expert planes fail the load instead of aborting the process
//! (issue #3297). `mlxcel_core::stack` is not fallible, so before the check a
//! mismatched expert reached MLX's `[stack]` exception and terminated the test
//! binary: on a tree without the fix these tests abort rather than fail.

use super::*;
use mlxcel_core::dtype;

const ROOT: &str = "model.layers.0.mlp";

fn plane(rows: i32, cols: i32, value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(&vec![value; (rows * cols) as usize], &[rows, cols])
}

/// Three experts under `{ROOT}.experts.{e}.gate_proj`, expert `odd` built
/// with `odd_cols` columns instead of 8.
fn experts_with_odd_weight(odd: usize, odd_cols: i32) -> WeightMap {
    let mut weights = WeightMap::new();
    for e in 0..3 {
        let cols = if e == odd { odd_cols } else { 8 };
        weights.insert(
            format!("{ROOT}.experts.{e}.gate_proj.weight"),
            plane(4, cols, e as f32),
        );
    }
    weights
}

#[test]
fn draining_refuses_an_expert_weight_of_another_shape() {
    let mut weights = experts_with_odd_weight(1, 6);
    let prefix = format!("{ROOT}.switch_mlp.gate_proj");
    let err = drain_numbered_expert_planes(
        &mut weights,
        &format!("{ROOT}.experts"),
        "gate_proj",
        &prefix,
    )
    .expect_err("a mismatched expert must fail the load");
    assert!(
        err.contains(&format!("{ROOT}.experts.1.gate_proj.weight")),
        "{err}"
    );
    assert!(
        err.contains("expert 1") && err.contains("expert 0"),
        "{err}"
    );
    assert!(err.contains("[4, 6]") && err.contains("[4, 8]"), "{err}");
    assert!(
        !weights.contains_key(&format!("{prefix}.weight")),
        "nothing is stacked on failure"
    );
}

#[test]
fn draining_refuses_an_expert_weight_of_another_dtype() {
    let mut weights = experts_with_odd_weight(usize::MAX, 8);
    let key = format!("{ROOT}.experts.2.gate_proj.weight");
    let f16 = mlxcel_core::astype(&weights[&key], dtype::FLOAT16);
    weights.insert(key.clone(), f16);
    let err = drain_numbered_expert_planes(
        &mut weights,
        &format!("{ROOT}.experts"),
        "gate_proj",
        &format!("{ROOT}.switch_mlp.gate_proj"),
    )
    .expect_err("a mixed-dtype expert stack must fail the load");
    assert!(err.contains(&key) && err.contains("dtype"), "{err}");
}

#[test]
fn draining_refuses_mismatched_scales() {
    let mut weights = experts_with_odd_weight(usize::MAX, 8);
    for e in 0..3 {
        let groups = if e == 2 { 2 } else { 1 };
        weights.insert(
            format!("{ROOT}.experts.{e}.gate_proj.scales"),
            plane(4, groups, 1.0),
        );
    }
    let err = drain_numbered_expert_planes(
        &mut weights,
        &format!("{ROOT}.experts"),
        "gate_proj",
        &format!("{ROOT}.switch_mlp.gate_proj"),
    )
    .expect_err("mismatched scales must fail the load");
    assert!(
        err.contains(&format!("{ROOT}.experts.2.gate_proj.scales")),
        "{err}"
    );
}

#[test]
fn draining_well_formed_experts_still_stacks() {
    let mut weights = experts_with_odd_weight(usize::MAX, 8);
    let prefix = format!("{ROOT}.switch_mlp.gate_proj");
    drain_numbered_expert_planes(
        &mut weights,
        &format!("{ROOT}.experts"),
        "gate_proj",
        &prefix,
    )
    .expect("matching experts stack");
    assert_eq!(
        mlxcel_core::array_shape(&weights[&format!("{prefix}.weight")]),
        vec![3, 4, 8]
    );
}

#[test]
fn switch_linear_refuses_unstacked_experts_of_another_shape() {
    let weights = experts_with_odd_weight(2, 4);
    let prefix = format!("{ROOT}.switch_mlp.gate_proj");
    for loaded in [
        SwitchLinear::from_weights_with_mode(&weights, &prefix, 64, 4, "affine"),
        SwitchLinear::from_weights(&weights, &prefix, 64, 4),
    ] {
        let err = match loaded {
            Ok(_) => panic!("a mismatched unstacked expert must fail the load"),
            Err(err) => err,
        };
        assert!(
            err.contains(&format!("{ROOT}.experts.2.gate_proj.weight")),
            "{err}"
        );
        assert!(err.contains("[4, 4]") && err.contains("[4, 8]"), "{err}");
    }
}

#[test]
fn switch_linear_refuses_an_unstacked_expert_missing_scales() {
    let mut weights = experts_with_odd_weight(usize::MAX, 8);
    for e in [0, 2] {
        weights.insert(
            format!("{ROOT}.experts.{e}.gate_proj.scales"),
            plane(4, 1, 1.0),
        );
    }
    let prefix = format!("{ROOT}.switch_mlp.gate_proj");
    let err = match SwitchLinear::from_weights_with_mode(&weights, &prefix, 64, 4, "affine") {
        Ok(_) => panic!("an expert without scales must fail the load"),
        Err(err) => err,
    };
    assert!(
        err.contains(&format!("{ROOT}.experts.1.gate_proj.scales is missing")),
        "{err}"
    );
}

#[test]
fn stack_individual_experts_reports_the_mismatch() {
    let weights = experts_with_odd_weight(1, 2);
    let err =
        match stack_individual_experts(&weights, &format!("{ROOT}.switch_mlp.gate_proj"), None) {
            Ok(_) => panic!("a mismatched expert must fail the load"),
            Err(err) => err,
        };
    assert!(
        err.contains(&format!("{ROOT}.experts.1.gate_proj.weight")),
        "{err}"
    );
}
