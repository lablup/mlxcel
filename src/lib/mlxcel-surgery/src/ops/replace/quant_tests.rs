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

//! Quantization-layout tests for `ReplaceOp`.
//!
//! Splits the heavier multi-tensor donor fixtures out of the main
//! [`super::tests`] module so each file stays focused and small.

// `tests` is a sibling test module under `replace/`. We reach its
// helpers via `super::tests::*` rather than re-implementing them.
use super::tests::{array_as_f32, make_tensor, make_tensor_u32};
use crate::{ReplaceOp, SurgeryError, SurgeryOp, WeightMap};
use mlxcel_core::dtype;
use safetensors::tensor::{Dtype, View};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Internal helper: a self-contained safetensors writer that
/// supports arbitrary dtypes per tensor.
struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
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

fn write_quant_donor(dir: &Path, tensors: HashMap<String, OwnedTensor>) -> PathBuf {
    let path = dir.join("donor.safetensors");
    safetensors::serialize_to_file(&tensors, None, &path).expect("write quant donor");
    path
}

fn tensor_u32(values: &[u32], shape: &[usize]) -> OwnedTensor {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    OwnedTensor {
        dtype: Dtype::U32,
        shape: shape.to_vec(),
        data: bytes,
    }
}

fn tensor_f32(values: &[f32], shape: &[usize]) -> OwnedTensor {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    OwnedTensor {
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data: bytes,
    }
}

#[test]
fn quantized_layout_replaces_weight_and_siblings_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let mut donor_tensors: HashMap<String, OwnedTensor> = HashMap::new();
    donor_tensors.insert(
        "q.weight".to_string(),
        tensor_u32(&[0xAAAA_BBBB, 0xCCCC_DDDD], &[1, 2]),
    );
    donor_tensors.insert(
        "q.weight.scales".to_string(),
        tensor_f32(&[9.0, 9.0], &[1, 2]),
    );
    donor_tensors.insert(
        "q.weight.biases".to_string(),
        tensor_f32(&[-1.0, -1.0], &[1, 2]),
    );
    let donor_path = write_quant_donor(dir.path(), donor_tensors);
    let op = ReplaceOp::new("q.weight", "q.weight", donor_path).unwrap();

    let mut weights = WeightMap::new();
    weights.insert(
        "q.weight".to_string(),
        make_tensor_u32(&[0u32, 0u32], &[1, 2]),
    );
    weights.insert(
        "q.weight.scales".to_string(),
        make_tensor(&[1.0, 1.0], &[1, 2]),
    );
    weights.insert(
        "q.weight.biases".to_string(),
        make_tensor(&[0.0, 0.0], &[1, 2]),
    );

    op.apply(&mut weights, &serde_json::Value::Null).unwrap();

    // All three siblings replaced.
    let w = weights.get("q.weight").unwrap();
    mlxcel_core::eval(w);
    assert_eq!(mlxcel_core::array_dtype(w), dtype::UINT32);

    assert_eq!(
        array_as_f32(weights.get("q.weight.scales").unwrap())[0],
        9.0
    );
    assert_eq!(
        array_as_f32(weights.get("q.weight.biases").unwrap())[0],
        -1.0
    );
}

#[test]
fn quantized_layout_errors_when_donor_missing_scales() {
    // Base has .weight + .scales but donor has only .weight.
    let dir = tempfile::tempdir().unwrap();
    let mut donor_tensors: HashMap<String, OwnedTensor> = HashMap::new();
    donor_tensors.insert("q.weight".to_string(), tensor_u32(&[0u32, 0u32], &[1, 2]));
    let donor_path = write_quant_donor(dir.path(), donor_tensors);
    let op = ReplaceOp::new("q.weight", "q.weight", donor_path).unwrap();
    let mut weights = WeightMap::new();
    weights.insert("q.weight".to_string(), make_tensor_u32(&[1, 1], &[1, 2]));
    weights.insert(
        "q.weight.scales".to_string(),
        make_tensor(&[1.0, 1.0], &[1, 2]),
    );
    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("missing .scales sibling must fail");
    match err {
        SurgeryError::TensorNotFound(msg) => {
            assert!(
                msg.contains("q.weight.scales"),
                "error must mention missing sibling: {msg}"
            );
        }
        other => panic!("expected TensorNotFound, got {other:?}"),
    }
    // Atomic-on-error: the original `.weight` payload is still
    // [1, 1], not [0, 0] from the donor.
    let still = weights.get("q.weight").unwrap();
    mlxcel_core::eval(still);
    let bytes = mlxcel_core::array_to_raw_bytes(still);
    let v = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(v, 1);
}

#[test]
fn quantized_layout_errors_on_scales_shape_mismatch() {
    // Base scales shape [1, 2], donor scales shape [1, 4].
    let dir = tempfile::tempdir().unwrap();
    let mut donor_tensors: HashMap<String, OwnedTensor> = HashMap::new();
    donor_tensors.insert("q.weight".to_string(), tensor_u32(&[0u32, 0u32], &[1, 2]));
    donor_tensors.insert(
        "q.weight.scales".to_string(),
        tensor_f32(&[1.0, 1.0, 1.0, 1.0], &[1, 4]),
    );
    let donor_path = write_quant_donor(dir.path(), donor_tensors);
    let op = ReplaceOp::new("q.weight", "q.weight", donor_path).unwrap();
    let mut weights = WeightMap::new();
    weights.insert("q.weight".to_string(), make_tensor_u32(&[1, 1], &[1, 2]));
    weights.insert(
        "q.weight.scales".to_string(),
        make_tensor(&[1.0, 1.0], &[1, 2]),
    );
    let err = op
        .apply(&mut weights, &serde_json::Value::Null)
        .expect_err("scales shape mismatch must fail");
    match err {
        SurgeryError::ShapeMismatch { key, .. } => {
            assert_eq!(key, "q.weight.scales");
        }
        other => panic!("expected ShapeMismatch on scales, got {other:?}"),
    }
}

#[test]
fn non_quantized_base_with_no_siblings_does_not_require_donor_siblings() {
    // Pure sanity check: a non-quantized base (just a `.weight`,
    // no `.scales` / `.biases`) should not try to fetch siblings
    // from the donor.
    let dir = tempfile::tempdir().unwrap();
    let mut donor_tensors: HashMap<String, OwnedTensor> = HashMap::new();
    donor_tensors.insert(
        "model.linear.weight".to_string(),
        tensor_f32(&[3.5, 2.5], &[2]),
    );
    // Note: NO `.scales` / `.biases` in the donor.
    let donor_path = write_quant_donor(dir.path(), donor_tensors);
    let op = ReplaceOp::new("model.linear.weight", "model.linear.weight", donor_path).unwrap();
    let mut weights = WeightMap::new();
    weights.insert(
        "model.linear.weight".to_string(),
        make_tensor(&[1.0, 1.0], &[2]),
    );
    op.apply(&mut weights, &serde_json::Value::Null)
        .expect("non-quantized base must not need donor siblings");
    assert_eq!(
        array_as_f32(weights.get("model.linear.weight").unwrap()),
        vec![3.5, 2.5]
    );
}
