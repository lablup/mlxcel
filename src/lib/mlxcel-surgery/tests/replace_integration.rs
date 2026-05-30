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

//! Integration tests for `ReplaceOp` driven through the YAML config
//! parser and the `SurgeryPipeline` `WeightTransform` hook.
//!
//! These tests exercise the public API surface a downstream caller
//! (`mlxcel::models::load_text_weights` once the `--surgery` CLI flag lands) would use:
//!
//! ```text
//! YAML -> parse_config_file -> SurgeryPipeline
//!      -> WeightTransform::apply(&mut WeightMap, &cfg)
//!      -> mutates only the matched tensors
//! ```
//!
//! Acceptance criterion (b) — when integrated via the
//! `SurgeryPipeline` builder, `ReplaceOp` produces a `WeightMap`
//! with the targeted tensor replaced by the donor's tensor.

use mlxcel_surgery::{parse_config_file, WeightMap, WeightTransform};
use std::collections::HashMap;
use std::path::Path;

fn write_donor(dir: &Path) -> std::path::PathBuf {
    use safetensors::tensor::{Dtype, View};
    use std::borrow::Cow;
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
    let donor_path = dir.join("donor.safetensors");
    let mut bytes = Vec::new();
    for v in [42.0f32, 43.0, 44.0, 45.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor {
            dtype: Dtype::F32,
            shape: vec![2, 2],
            data: bytes,
        },
    );
    safetensors::serialize_to_file(&map, None, &donor_path).expect("write donor");
    donor_path
}

#[test]
fn replace_yaml_replaces_targeted_tensor_through_pipeline() {
    let dir = tempfile::tempdir().expect("tempdir");
    let _donor = write_donor(dir.path());

    let yaml = r#"version: 1
operations:
  - op: replace
    pattern: "model.embed_tokens.weight"
    source: "./donor.safetensors"
    source_key: "model.embed_tokens.weight"
"#;
    let yaml_path = dir.path().join("surgery.yaml");
    std::fs::write(&yaml_path, yaml).expect("write yaml");

    let pipeline = parse_config_file(&yaml_path).expect("parse replace yaml");
    assert_eq!(pipeline.len(), 1);

    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::from_slice_f32(&[7.0, 7.0], &[2]),
    );

    // Invoke through the `WeightTransform` trait — same call A1's
    // consolidated loader makes.
    WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect("apply through pipeline");

    // Targeted tensor replaced with donor values.
    let post = weights.get("model.embed_tokens.weight").unwrap();
    mlxcel_core::eval(post);
    let bytes = mlxcel_core::array_to_raw_bytes(post);
    let mut floats = Vec::with_capacity(4);
    for chunk in bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(floats, vec![42.0, 43.0, 44.0, 45.0]);

    // Untouched tensor preserved exactly.
    let untouched = weights.get("model.norm.weight").unwrap();
    mlxcel_core::eval(untouched);
    let bytes = mlxcel_core::array_to_raw_bytes(untouched);
    let mut floats = Vec::with_capacity(2);
    for chunk in bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(floats, vec![7.0, 7.0]);
}

#[test]
fn replace_yaml_missing_tensor_in_base_errors_through_pipeline() {
    let dir = tempfile::tempdir().expect("tempdir");
    let _donor = write_donor(dir.path());

    // Pattern that won't match anything in the base map.
    let yaml = r#"version: 1
operations:
  - op: replace
    pattern: "model.nonexistent.weight"
    source: "./donor.safetensors"
    source_key: "model.embed_tokens.weight"
"#;
    let yaml_path = dir.path().join("surgery.yaml");
    std::fs::write(&yaml_path, yaml).expect("write yaml");
    let pipeline = parse_config_file(&yaml_path).expect("parse");

    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
    );

    let err = WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect_err("zero-match pattern must error end-to-end through the pipeline");
    assert!(
        err.contains("matched no keys") || err.contains("replace"),
        "error must surface through SurgeryPipeline -> WeightTransform: {err}"
    );
}
