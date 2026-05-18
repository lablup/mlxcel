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

//! Error-case unit tests for `AddOp`.
//!
//! Cover constructor validation, the zero-match diagnostic, missing
//! source files / keys, shape mismatch, and the quantized-base
//! diagnostic. The apply-correctness tests (default alpha, half
//! alpha, multi-match, dtype cast, alpha=0 fast path, …) live in
//! `add_apply_tests.rs` so each file stays under the project's
//! 500-line per-file budget.
//!
//! Used by: `cargo test -p mlxcel-surgery`.

use mlxcel_core::weights::WeightMap;

use super::add::AddOp;
use super::add_test_helpers::{f32_tensor, mlx_f32, write_single_donor};
use crate::{SurgeryError, SurgeryOp};

#[test]
fn new_rejects_non_finite_alpha() {
    let err = AddOp::new("model.*", "/tmp/anywhere.safetensors", f32::NAN)
        .expect_err("nan alpha must be rejected");
    assert!(format!("{err}").contains("finite"));
}

#[test]
fn new_rejects_bad_glob() {
    // `{0` is an unclosed brace alternation — globset rejects it.
    let err = AddOp::new("model.layers.{0", "/tmp/x.safetensors", 1.0)
        .expect_err("bad glob must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("malformed glob pattern") && msg.contains("{0"),
        "error must surface the offending pattern: {msg}"
    );
}

#[test]
fn zero_match_pattern_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let donor_path = write_single_donor(dir.path(), "unused.key", f32_tensor(&[1.0, 1.0], &[2]));

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        mlx_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );

    let op = AddOp::new("does.not.match.*", &donor_path, 1.0).expect("construct");
    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("zero matches must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("matched zero tensor keys") && msg.contains("does.not.match.*"),
        "zero-match error must quote the pattern: {msg}"
    );
}

#[test]
fn missing_source_file_errors() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        mlx_f32(&[1.0; 4], &[2, 2]),
    );

    let op = AddOp::new(
        "model.layers.*.self_attn.q_proj.weight",
        "/nonexistent/path/should/not/exist.safetensors",
        1.0,
    )
    .expect("construct");

    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("missing source must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("failed to load task-vector source"),
        "missing-source error must be explicit: {msg}"
    );
}

#[test]
fn missing_source_key_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let donor_path = write_single_donor(
        dir.path(),
        // Donor has a *different* key from what the base map has.
        "model.layers.0.self_attn.v_proj.weight",
        f32_tensor(&[1.0; 4], &[2, 2]),
    );

    let mut weights = WeightMap::new();
    // Base key matches the glob; donor file does *not* contain it.
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        mlx_f32(&[1.0; 4], &[2, 2]),
    );

    let op =
        AddOp::new("model.layers.*.self_attn.q_proj.weight", &donor_path, 1.0).expect("construct");

    match op.apply(&mut weights, &serde_json::Value::Null) {
        Err(SurgeryError::TensorNotFound(key)) => {
            assert_eq!(key, "model.layers.0.self_attn.q_proj.weight");
        }
        other => panic!("expected TensorNotFound, got {other:?}"),
    }

    // The base tensor must still be present (we restored it on error).
    assert!(
        weights.contains_key("model.layers.0.self_attn.q_proj.weight"),
        "base tensor must be restored on error so the map stays consistent",
    );
}

#[test]
fn shape_mismatch_returns_structured_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let donor_path = write_single_donor(
        dir.path(),
        "model.layers.0.self_attn.q_proj.weight",
        // Donor is [4] but base is [2, 2].
        f32_tensor(&[1.0, 1.0, 1.0, 1.0], &[4]),
    );

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        mlx_f32(&[1.0; 4], &[2, 2]),
    );

    let op =
        AddOp::new("model.layers.*.self_attn.q_proj.weight", &donor_path, 1.0).expect("construct");

    match op.apply(&mut weights, &serde_json::Value::Null) {
        Err(SurgeryError::ShapeMismatch {
            key,
            expected,
            actual,
        }) => {
            assert_eq!(key, "model.layers.0.self_attn.q_proj.weight");
            assert_eq!(expected, vec![2, 2]);
            assert_eq!(actual, vec![4]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn quantized_packed_base_returns_focused_error() {
    use mlxcel_core::from_slice_u32;

    // Build a base tensor whose dtype is integer (uint32) —
    // matching MLX's representation of packed quantized weights.
    let dir = tempfile::tempdir().expect("tempdir");
    let donor_path = write_single_donor(
        dir.path(),
        "model.layers.0.mlp.gate_proj.weight",
        f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );

    let mut weights = WeightMap::new();
    // u32 packed bits with the same logical shape.
    let packed_bits: Vec<u32> = vec![0, 0, 0, 0];
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        from_slice_u32(&packed_bits, &[2, 2]),
    );

    let op =
        AddOp::new("model.layers.*.mlp.gate_proj.weight", &donor_path, 1.0).expect("construct");

    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("integer base must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-floating dtype") && msg.contains("uint32"),
        "quantized error must name the dtype: {msg}"
    );
    assert!(
        msg.contains("dequantize") || msg.contains(".scales") || msg.contains("scales"),
        "quantized error must hint the workaround: {msg}"
    );
}

#[test]
fn name_is_add() {
    let op = AddOp::new("*", "/x.safetensors", 1.0).expect("construct");
    assert_eq!(op.name(), "add");
}
