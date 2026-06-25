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

//! Apply-correctness unit tests for `AddOp`.
//!
//! These cover the numerical "happy path" — default alpha,
//! fractional alpha, negative alpha, alpha = 0 fast path, multi-key
//! fan-out, and donor dtype casting. The error-case tests
//! (zero-match, missing source, shape mismatch, quantized base) live
//! in `add_tests.rs`. Splitting keeps each file under the project's
//! 500-line per-file budget.
//!
//! Used by: `cargo test -p mlxcel-surgery`.

use mlxcel_core::dtype as mlx_dtype;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{array_dtype, astype};

use super::add::AddOp;
use super::add_test_helpers::{OwnedTensor, extract_f32, f32_tensor, mlx_f32, write_single_donor};
use crate::SurgeryOp;

#[test]
fn applies_correct_value_with_default_alpha() {
    // alpha defaults to 1.0 — base + 1.0 * delta.
    let dir = tempfile::tempdir().expect("tempdir");
    let donor_path = write_single_donor(
        dir.path(),
        "model.layers.0.mlp.down_proj.weight",
        f32_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]),
    );

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        mlx_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );

    let op =
        AddOp::new("model.layers.*.mlp.down_proj.weight", &donor_path, 1.0).expect("construct");

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply must succeed");

    let updated = weights
        .get("model.layers.0.mlp.down_proj.weight")
        .expect("base must be present");
    let actual = extract_f32(updated);
    assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn applies_correct_value_with_alpha_half() {
    let dir = tempfile::tempdir().expect("tempdir");
    let donor_path = write_single_donor(
        dir.path(),
        "model.layers.0.mlp.down_proj.weight",
        f32_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]),
    );

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        mlx_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );

    let op =
        AddOp::new("model.layers.*.mlp.down_proj.weight", &donor_path, 0.5).expect("construct");

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply must succeed");

    let updated = weights
        .get("model.layers.0.mlp.down_proj.weight")
        .expect("base must be present");
    let actual = extract_f32(updated);
    // 1 + 0.5*10 = 6, 2 + 0.5*20 = 12, 3 + 0.5*30 = 18, 4 + 0.5*40 = 24
    assert_eq!(actual, vec![6.0, 12.0, 18.0, 24.0]);
}

#[test]
fn applies_correct_value_with_negative_alpha() {
    // Negative alpha subtracts the task vector — also valid weight
    // arithmetic ("forget the task vector").
    let dir = tempfile::tempdir().expect("tempdir");
    let donor_path = write_single_donor(
        dir.path(),
        "model.layers.0.mlp.down_proj.weight",
        f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        mlx_f32(&[10.0, 10.0, 10.0, 10.0], &[2, 2]),
    );

    let op =
        AddOp::new("model.layers.*.mlp.down_proj.weight", &donor_path, -2.0).expect("construct");

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply must succeed");

    let updated = weights
        .get("model.layers.0.mlp.down_proj.weight")
        .expect("base must be present");
    let actual = extract_f32(updated);
    assert_eq!(actual, vec![8.0, 6.0, 4.0, 2.0]);
}

#[test]
fn alpha_zero_is_a_no_op_and_skips_donor_load() {
    // Important edge case: alpha = 0.0 should leave the base
    // untouched. We exercise this by pointing the AddOp at a
    // safetensors source that would *fail* to load (the file does
    // not exist) — and verifying we still succeed because the op
    // short-circuits before the I/O.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        mlx_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );

    let op = AddOp::new(
        "model.layers.*.mlp.down_proj.weight",
        "/this/path/does/not/exist.safetensors",
        0.0,
    )
    .expect("construct");

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("alpha=0 must be a no-op even when donor is missing");

    let updated = weights
        .get("model.layers.0.mlp.down_proj.weight")
        .expect("base must remain");
    let actual = extract_f32(updated);
    assert_eq!(actual, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn alpha_zero_still_diagnoses_zero_match_patterns() {
    // The alpha == 0 fast path must not mask the "pattern matches
    // nothing" diagnostic — that's typically a user typo and
    // they'd hate to silently get a no-op.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        mlx_f32(&[1.0; 4], &[2, 2]),
    );

    let op = AddOp::new("layer.does.not.exist.*", "/anywhere.safetensors", 0.0).expect("construct");

    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("zero match should still error");
    assert!(format!("{err}").contains("matched zero tensor keys"));
}

#[test]
fn applies_to_multiple_matched_keys() {
    // Verify the glob actually fans out: 3 layers, each with one
    // matching tensor; the AddOp must rewrite all three.
    let dir = tempfile::tempdir().expect("tempdir");

    let mut donor: std::collections::HashMap<String, OwnedTensor> =
        std::collections::HashMap::new();
    for layer in 0..3 {
        donor.insert(
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            f32_tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]),
        );
    }
    let donor_path = dir.path().join("donor.safetensors");
    safetensors::serialize_to_file(&donor, None, &donor_path)
        .expect("serialize multi-tensor donor");

    let mut weights = WeightMap::new();
    for layer in 0..3 {
        weights.insert(
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            mlx_f32(&[(layer * 10) as f32; 4], &[2, 2]),
        );
    }
    // Extra unmatched tensor — must not be touched.
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlx_f32(&[100.0; 4], &[2, 2]),
    );

    let op =
        AddOp::new("model.layers.*.mlp.down_proj.weight", &donor_path, 2.5).expect("construct");

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("apply must succeed across layers");

    for layer in 0..3 {
        let key = format!("model.layers.{layer}.mlp.down_proj.weight");
        let arr = weights.get(&key).expect("matched key present");
        let actual = extract_f32(arr);
        let expected = vec![(layer * 10) as f32 + 2.5; 4];
        assert_eq!(actual, expected, "layer {layer} value");
    }

    // Unmatched tensor untouched.
    let embed = weights.get("model.embed_tokens.weight").unwrap();
    assert_eq!(extract_f32(embed), vec![100.0; 4]);
}

#[test]
fn donor_dtype_is_cast_to_base_dtype() {
    // Donor is f32; base is f16. We verify the rewritten base
    // stays in f16 (no silent upcast) and the value is correct
    // within f16 precision.
    let dir = tempfile::tempdir().expect("tempdir");
    let donor_path = write_single_donor(
        dir.path(),
        "model.layers.0.mlp.down_proj.weight",
        f32_tensor(&[2.0, 4.0, 6.0, 8.0], &[2, 2]),
    );

    let mut weights = WeightMap::new();
    // Build a base f16 array via f32 → astype(f16).
    let base_f32 = mlx_f32(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let base_f16 = astype(&base_f32, mlx_dtype::FLOAT16);
    weights.insert("model.layers.0.mlp.down_proj.weight".to_string(), base_f16);

    let op =
        AddOp::new("model.layers.*.mlp.down_proj.weight", &donor_path, 1.0).expect("construct");

    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("dtype-cast donor must succeed");

    let updated = weights
        .get("model.layers.0.mlp.down_proj.weight")
        .expect("base must be present");
    assert_eq!(
        array_dtype(updated),
        mlx_dtype::FLOAT16,
        "base must keep its f16 dtype after the add"
    );

    // Convert back to f32 for content assertion.
    let as_f32 = astype(updated, mlx_dtype::FLOAT32);
    let actual = extract_f32(&as_f32);
    assert_eq!(actual, vec![3.0, 5.0, 7.0, 9.0]);
}
