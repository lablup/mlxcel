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

//! Unit tests for `ReplaceOp`.
//!
//! Covers acceptance criterion (a):
//! - construction-time validation (wildcard count, empty inputs);
//! - single-tensor replace;
//! - multi-tensor replace via glob;
//! - source_key wildcard substitution;
//! - shape mismatch error;
//! - dtype mismatch error;
//! - missing donor key (TensorNotFound);
//! - missing donor file (Io);
//! - zero-match pattern.
//!
//! Quantization-layout-specific paths live in
//! [`super::quant_tests`] to keep this file focused on the core
//! replace semantics.

use super::*;
use crate::SurgeryError;
use mlxcel_core::{MlxArray, UniquePtr};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(super) fn make_tensor(values: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, shape)
}

pub(super) fn make_tensor_u32(values: &[u32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_u32(values, shape)
}

/// Convert an MLX array's bytes into a `Vec<f32>` for inspection.
pub(super) fn array_as_f32(arr: &UniquePtr<MlxArray>) -> Vec<f32> {
    mlxcel_core::eval(arr);
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    let mut floats = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    floats
}

/// Write a donor `.safetensors` file containing the given f32
/// tensors. Returns the path to the file.
pub(super) fn write_donor_safetensors_f32(
    dir: &Path,
    entries: &[(&str, &[f32], &[usize])],
) -> PathBuf {
    use safetensors::tensor::{Dtype, View};
    use std::borrow::Cow;
    use std::collections::HashMap;
    struct Tensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }
    impl View for &Tensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<'_, [u8]> {
            self.data.as_slice().into()
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }
    let path = dir.join("donor.safetensors");
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for (name, values, shape) in entries {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in *values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        map.insert(
            (*name).to_string(),
            Tensor {
                dtype: Dtype::F32,
                shape: shape.to_vec(),
                data: bytes,
            },
        );
    }
    safetensors::serialize_to_file(&map, None, &path).expect("write donor");
    path
}

#[test]
fn replace_op_construction_rejects_wildcard_count_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let donor = dir.path().join("donor.safetensors");
    std::fs::File::create(&donor).unwrap().write_all(b"x").ok();
    let err = ReplaceOp::new("a.*.b", "a.b", donor).expect_err("mismatch must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("wildcard") || msg.contains("source_key") || msg.contains("`*`"),
        "error must explain wildcard mismatch: {msg}"
    );
}

#[test]
fn replace_op_rejects_empty_pattern() {
    let dir = tempfile::tempdir().unwrap();
    let donor = dir.path().join("donor.safetensors");
    std::fs::File::create(&donor).unwrap().write_all(b"x").ok();
    let err = ReplaceOp::new("", "x", donor.clone()).expect_err("empty pattern fails");
    assert!(format!("{err}").contains("pattern"));
    let err = ReplaceOp::new("x", "", donor).expect_err("empty source_key fails");
    assert!(format!("{err}").contains("source_key"));
}

#[test]
fn apply_single_tensor_replace_swaps_payload() {
    let dir = tempfile::tempdir().unwrap();
    let donor_path = write_donor_safetensors_f32(
        dir.path(),
        &[(
            "model.embed_tokens.weight",
            &[10.0, 20.0, 30.0, 40.0],
            &[2, 2],
        )],
    );
    let op = ReplaceOp::new(
        "model.embed_tokens.weight",
        "model.embed_tokens.weight",
        donor_path,
    )
    .unwrap();

    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );
    // An unrelated key that must remain untouched.
    weights.insert(
        "model.norm.weight".to_string(),
        make_tensor(&[7.0, 7.0], &[2]),
    );

    op.apply(&mut weights, &serde_json::Value::Null).unwrap();

    assert_eq!(
        array_as_f32(weights.get("model.embed_tokens.weight").unwrap()),
        vec![10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(
        array_as_f32(weights.get("model.norm.weight").unwrap()),
        vec![7.0, 7.0]
    );
}

#[test]
fn apply_multi_tensor_glob_replaces_every_match() {
    let dir = tempfile::tempdir().unwrap();
    let donor_path = write_donor_safetensors_f32(
        dir.path(),
        &[
            ("model.layers.0.attn.weight", &[100.0, 100.0], &[1, 2]),
            ("model.layers.1.attn.weight", &[200.0, 200.0], &[1, 2]),
        ],
    );
    let op = ReplaceOp::new(
        "model.layers.*.attn.weight",
        "model.layers.*.attn.weight",
        donor_path,
    )
    .unwrap();

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.attn.weight".to_string(),
        make_tensor(&[1.0, 1.0], &[1, 2]),
    );
    weights.insert(
        "model.layers.1.attn.weight".to_string(),
        make_tensor(&[2.0, 2.0], &[1, 2]),
    );

    op.apply(&mut weights, &serde_json::Value::Null).unwrap();

    assert_eq!(
        array_as_f32(weights.get("model.layers.0.attn.weight").unwrap()),
        vec![100.0, 100.0]
    );
    assert_eq!(
        array_as_f32(weights.get("model.layers.1.attn.weight").unwrap()),
        vec![200.0, 200.0]
    );
}

#[test]
fn apply_with_source_key_wildcard_substitution() {
    // Demonstrates renaming: base uses
    // `model.layers.X.attn.weight`, donor stores them under
    // `donor.h.X.attn.weight`.
    let dir = tempfile::tempdir().unwrap();
    let donor_path = write_donor_safetensors_f32(
        dir.path(),
        &[
            ("donor.h.0.attn.weight", &[100.0], &[1]),
            ("donor.h.7.attn.weight", &[700.0], &[1]),
        ],
    );
    let op = ReplaceOp::new(
        "model.layers.*.attn.weight",
        "donor.h.*.attn.weight",
        donor_path,
    )
    .unwrap();

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.attn.weight".to_string(),
        make_tensor(&[1.0], &[1]),
    );
    weights.insert(
        "model.layers.7.attn.weight".to_string(),
        make_tensor(&[1.0], &[1]),
    );

    op.apply(&mut weights, &serde_json::Value::Null).unwrap();

    assert_eq!(
        array_as_f32(weights.get("model.layers.0.attn.weight").unwrap()),
        vec![100.0]
    );
    assert_eq!(
        array_as_f32(weights.get("model.layers.7.attn.weight").unwrap()),
        vec![700.0]
    );
}

#[test]
fn apply_shape_mismatch_errors_and_leaves_weights_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let donor_path =
        write_donor_safetensors_f32(dir.path(), &[("model.x.weight", &[1.0, 2.0, 3.0], &[3])]);
    let op = ReplaceOp::new("model.x.weight", "model.x.weight", donor_path).unwrap();

    let mut weights = WeightMap::new();
    // Base has shape [2], donor has shape [3] -> mismatch.
    weights.insert(
        "model.x.weight".to_string(),
        make_tensor(&[10.0, 20.0], &[2]),
    );

    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("shape mismatch must fail");
    match err {
        SurgeryError::ShapeMismatch {
            key,
            expected,
            actual,
        } => {
            assert_eq!(key, "model.x.weight");
            assert_eq!(expected, vec![2]);
            assert_eq!(actual, vec![3]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }

    // Atomic-on-error: original tensor unchanged.
    assert_eq!(
        array_as_f32(weights.get("model.x.weight").unwrap()),
        vec![10.0, 20.0]
    );
}

#[test]
fn apply_dtype_mismatch_errors() {
    let dir = tempfile::tempdir().unwrap();
    // Donor is f32; base will be int32 -> mismatch.
    let donor_path =
        write_donor_safetensors_f32(dir.path(), &[("model.x.weight", &[1.0, 2.0], &[2])]);
    let op = ReplaceOp::new("model.x.weight", "model.x.weight", donor_path).unwrap();

    let mut weights = WeightMap::new();
    let i32_data: Vec<i32> = vec![10, 20];
    weights.insert(
        "model.x.weight".to_string(),
        mlxcel_core::from_slice_i32(&i32_data, &[2]),
    );

    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("dtype mismatch must fail");
    match err {
        SurgeryError::DtypeMismatch {
            key,
            expected,
            actual,
        } => {
            assert_eq!(key, "model.x.weight");
            assert_eq!(expected, "int32");
            assert_eq!(actual, "float32");
        }
        other => panic!("expected DtypeMismatch, got {other:?}"),
    }
}

#[test]
fn apply_missing_source_key_returns_tensor_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let donor_path = write_donor_safetensors_f32(dir.path(), &[("some.other.key", &[1.0], &[1])]);
    let op = ReplaceOp::new("model.x.weight", "model.x.weight", donor_path).unwrap();

    let mut weights = WeightMap::new();
    weights.insert("model.x.weight".to_string(), make_tensor(&[7.0], &[1]));

    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("missing donor key must fail");
    match err {
        SurgeryError::TensorNotFound(msg) => {
            assert!(
                msg.contains("model.x.weight"),
                "error must mention missing key: {msg}"
            );
        }
        other => panic!("expected TensorNotFound, got {other:?}"),
    }
}

#[test]
fn apply_missing_source_file_returns_io_error() {
    let op = ReplaceOp::new(
        "model.x.weight",
        "model.x.weight",
        PathBuf::from("/nonexistent/path/donor.safetensors"),
    )
    .unwrap();
    let mut weights = WeightMap::new();
    weights.insert("model.x.weight".to_string(), make_tensor(&[7.0], &[1]));
    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("missing file must fail");
    match err {
        SurgeryError::Io(_) => {}
        other => panic!("expected Io error, got {other:?}"),
    }
}

#[test]
fn apply_zero_match_pattern_errors() {
    let dir = tempfile::tempdir().unwrap();
    let donor_path = write_donor_safetensors_f32(dir.path(), &[("any.key", &[1.0], &[1])]);
    let op = ReplaceOp::new("nothing.matches.*", "any.key.*", donor_path).unwrap();
    let mut weights = WeightMap::new();
    weights.insert("model.real.weight".to_string(), make_tensor(&[1.0], &[1]));
    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("zero match must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("matched no keys"),
        "error must say nothing matched: {msg}"
    );
}

#[test]
fn name_is_replace() {
    let dir = tempfile::tempdir().unwrap();
    let donor = dir.path().join("donor.safetensors");
    std::fs::File::create(&donor).unwrap().write_all(b"x").ok();
    let op = ReplaceOp::new("x", "x", donor).unwrap();
    assert_eq!(op.name(), "replace");
}
