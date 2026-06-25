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

//! End-to-end integration test for `AddOp` (— A6).
//!
//! Drives the full public surface — `SurgeryPipeline` constructed
//! via `parse_config_file` and also directly via `push` — over a
//! synthetic base + delta safetensors pair. The expected post-load
//! state is `base += alpha * delta` on every matched key.
//!
//! A4 (CLI flag `--surgery`) is in parallel with this issue; until
//! it lands, this integration test stands in for the "the pipeline
//! integrates with the load path" acceptance criterion by exercising
//! the same `WeightTransform::apply` entry point that the
//! consolidated text/VLM loaders use (A1).
//!
//! Used by: `cargo test -p mlxcel-surgery`.

use std::collections::HashMap;
use std::path::Path;

use mlxcel_core::dtype as mlx_dtype;
use mlxcel_core::weights::{WeightMap, WeightTransform};
use mlxcel_core::{MlxArray, UniquePtr, array_to_raw_bytes, eval, from_bytes};
use mlxcel_surgery::{SurgeryPipeline, parse_config_file};
use safetensors::View;
use safetensors::tensor::Dtype as SafeTensorDtype;

/// `safetensors::View` over owned bytes — same shape as the helper
/// in `src/lib/mlxcel-surgery/src/ops/add.rs`. Duplicated here
/// because integration tests cannot reach `#[cfg(test)]` items in
/// the crate.
#[derive(Clone)]
struct OwnedTensor {
    dtype: SafeTensorDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> SafeTensorDtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        self.data.as_slice().into()
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn f32_tensor(values: &[f32], shape: &[usize]) -> OwnedTensor {
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    OwnedTensor {
        dtype: SafeTensorDtype::F32,
        shape: shape.to_vec(),
        data,
    }
}

fn write_donor(dir: &Path, tensors: &HashMap<String, OwnedTensor>) -> std::path::PathBuf {
    let path = dir.join("task_vector.safetensors");
    safetensors::serialize_to_file(tensors, None, &path).expect("serialize donor safetensors");
    path
}

fn mlx_f32(values: &[f32], shape: &[usize]) -> UniquePtr<MlxArray> {
    let shape_i32: Vec<i32> = shape.iter().map(|d| *d as i32).collect();
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    from_bytes(&bytes, &shape_i32, mlx_dtype::FLOAT32)
}

fn extract_f32(arr: &UniquePtr<MlxArray>) -> Vec<f32> {
    eval(arr);
    let bytes = array_to_raw_bytes(arr);
    assert_eq!(bytes.len() % 4, 0);
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(buf));
    }
    out
}

/// End-to-end: write a YAML config + donor safetensors to a tempdir,
/// parse the YAML, run the pipeline through `WeightTransform::apply`
/// (the exact entry point the consolidated weight loaders use), and
/// verify the on-map tensors are now `base + alpha * delta`.
///
/// This covers acceptance criterion (b) — the operation is wired
/// through the same trait the load path consumes, so when A4 lands
/// and exposes `--surgery <yaml>` on the CLI, this same flow runs
/// against real models.
#[test]
fn yaml_driven_pipeline_applies_add_op_end_to_end() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Synthesize a controlled delta: small random-scaled values for
    // three layers' down_proj weights, plus an extra unrelated key
    // (`embed_tokens.weight`) that the glob must skip.
    let mut donor: HashMap<String, OwnedTensor> = HashMap::new();
    for layer in 0..3 {
        donor.insert(
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            f32_tensor(&[0.1 * (layer + 1) as f32; 8], &[2, 4]),
        );
    }
    // Extra key that exists in the donor but should be irrelevant
    // because no base tensor matches the pattern *and* this key.
    donor.insert(
        "model.unrelated.weight".to_string(),
        f32_tensor(&[999.0; 4], &[2, 2]),
    );
    let donor_path = write_donor(dir.path(), &donor);

    let yaml = format!(
        "version: 1\n\
         operations:\n\
         \x20\x20- op: add\n\
         \x20\x20\x20\x20pattern: \"model.layers.*.mlp.down_proj.weight\"\n\
         \x20\x20\x20\x20source: \"{}\"\n\
         \x20\x20\x20\x20alpha: 0.5\n",
        donor_path.display(),
    );
    let yaml_path = dir.path().join("surgery.yaml");
    std::fs::write(&yaml_path, yaml).expect("write yaml");

    let pipeline = parse_config_file(&yaml_path).expect("yaml parses with real AddOp factory");
    assert_eq!(pipeline.len(), 1);

    // Build a base WeightMap with 3 matching layers plus an
    // unrelated key.  Base values are 1.0 everywhere for an easy
    // post-condition.
    let mut weights = WeightMap::new();
    for layer in 0..3 {
        weights.insert(
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            mlx_f32(&[1.0; 8], &[2, 4]),
        );
    }
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlx_f32(&[42.0; 16], &[4, 4]),
    );

    // Apply via the WeightTransform trait — exact entry point of
    // the consolidated loaders.
    pipeline
        .apply(&mut weights, &serde_json::Value::Null)
        .expect("pipeline must apply cleanly");

    // base + 0.5 * delta with delta = 0.1 * (layer + 1)
    for layer in 0..3 {
        let key = format!("model.layers.{layer}.mlp.down_proj.weight");
        let arr = weights.get(&key).expect("matched key present");
        let actual = extract_f32(arr);
        let expected = 1.0_f32 + 0.5 * 0.1 * (layer + 1) as f32;
        for v in &actual {
            assert!(
                (v - expected).abs() < 1e-6,
                "layer {layer}: got {v}, expected {expected}"
            );
        }
    }

    // Unrelated tensor must be untouched.
    let embed = weights.get("model.embed_tokens.weight").unwrap();
    assert_eq!(extract_f32(embed), vec![42.0; 16]);
}

/// Builder-style pipeline (no YAML) is also valid integration
/// surface, especially for callers that build pipelines
/// programmatically. Verifies a non-trivial delta + alpha = 1.0 (the
/// implicit default) lands in the base tensor exactly.
///
/// This also documents how downstream callers can integrate without
/// waiting for A4's CLI flag.
#[test]
fn programmatic_pipeline_applies_add_op_end_to_end() {
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("tempdir");

    let mut donor: HashMap<String, OwnedTensor> = HashMap::new();
    donor.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
    );
    let donor_path = write_donor(dir.path(), &donor);

    let mut pipeline = SurgeryPipeline::new();
    pipeline.push(Arc::new(
        mlxcel_surgery::AddOp::new("model.layers.*.self_attn.q_proj.weight", &donor_path, 1.0)
            .expect("construct AddOp"),
    ));

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        mlx_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], &[2, 4]),
    );

    pipeline
        .apply(&mut weights, &serde_json::Value::Null)
        .expect("programmatic pipeline must apply");

    let arr = weights
        .get("model.layers.0.self_attn.q_proj.weight")
        .expect("present");
    let actual = extract_f32(arr);
    assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]);
}

/// Acceptance criterion (e): without the op in the pipeline, an
/// empty `SurgeryPipeline` is a bit-exact no-op.
///
/// Sanity check that just exercising the pipeline does not introduce
/// any spurious mutation of weights. This is inherited from A4 but
/// validating it here keeps the A6 PR self-contained.
#[test]
fn empty_pipeline_is_bit_exact_no_op() {
    let pipeline = SurgeryPipeline::new();
    let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();

    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlx_f32(&original, &[4, 4]),
    );

    pipeline
        .apply(&mut weights, &serde_json::Value::Null)
        .expect("empty pipeline must succeed");

    let arr = weights.get("model.embed_tokens.weight").unwrap();
    let bits = extract_f32(arr);
    assert_eq!(bits, original);
}
