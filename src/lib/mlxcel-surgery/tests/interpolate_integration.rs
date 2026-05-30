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

//! Integration test for the Axis A `InterpolateOp`.
//!
//! Verifies the end-to-end flow that the consolidated weight loader
//! will exercise once A4 wires the `--surgery` CLI flag: parse a
//! YAML config, materialize an `InterpolateOp`, load two donor
//! `.safetensors` files from disk, and produce a fully-blended
//! [`mlxcel_core::weights::WeightMap`] without any synthetic injection
//! shortcuts. The donor files are constructed in-test using a
//! hand-rolled minimal safetensors writer (see [`write_safetensors`])
//! to avoid depending on an external Python tool just to bring a
//! pair of donor files into existence.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use mlxcel_surgery::{
    parse_config_file, InterpolateMethod, InterpolateOp, SurgeryError, SurgeryPipeline, WeightMap,
    WeightTransform,
};

/// Minimal safetensors writer producing a single-file checkpoint
/// containing the given f32 tensors. The format is intentionally
/// simple — the safetensors spec only requires:
///
/// 1. 8 little-endian bytes: header length `H` in bytes.
/// 2. `H` bytes of UTF-8 JSON header. The header is an object whose
///    keys are tensor names; each value is `{ "dtype", "shape",
///    "data_offsets": [start, end] }`.
/// 3. Tensor data blob — concatenated little-endian f32 buffers
///    listed in `data_offsets` order.
///
/// We deliberately keep the implementation minimal so it can live
/// inside an integration test crate without adding `safetensors`
/// as a dev-dependency.
fn write_safetensors(path: &Path, tensors: &BTreeMap<String, Vec<f32>>) {
    // Build the JSON header. `data_offsets` are byte offsets into
    // the data blob (NOT into the file).
    let mut entries: Vec<String> = Vec::new();
    let mut data_blob: Vec<u8> = Vec::new();
    for (name, values) in tensors {
        let start = data_blob.len();
        for v in values {
            data_blob.extend_from_slice(&v.to_le_bytes());
        }
        let end = data_blob.len();
        // Single-axis tensor — sufficient for the smoke test below.
        entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[{start},{end}]}}"#,
            shape = values.len(),
        ));
    }
    let header_json = format!("{{{}}}", entries.join(","));
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut file = fs::File::create(path).expect("create safetensors file");
    file.write_all(&header_len.to_le_bytes())
        .expect("write header length");
    file.write_all(header_bytes).expect("write header");
    file.write_all(&data_blob).expect("write data blob");
}

/// Helper: build the four-tensor synthetic donor used by the smoke
/// test below. Two layers, two tensors per layer. The values are
/// chosen so that LERP at `t=0.5` produces a sentinel pattern we
/// can verify after loading.
fn synth_donor_a() -> BTreeMap<String, Vec<f32>> {
    let mut m = BTreeMap::new();
    m.insert("model.layers.0.w".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    m.insert("model.layers.1.w".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
    m
}

fn synth_donor_b() -> BTreeMap<String, Vec<f32>> {
    let mut m = BTreeMap::new();
    m.insert("model.layers.0.w".to_string(), vec![11.0, 22.0, 33.0, 44.0]);
    m.insert("model.layers.1.w".to_string(), vec![55.0, 66.0, 77.0, 88.0]);
    m
}

/// Compose a base weight map that mirrors the donor structure but
/// with placeholder zeros — surgery will overwrite every value.
fn synth_base_zeros() -> WeightMap {
    let mut base = WeightMap::new();
    base.insert(
        "model.layers.0.w".to_string(),
        mlxcel_core::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[4]),
    );
    base.insert(
        "model.layers.1.w".to_string(),
        mlxcel_core::from_slice_f32(&[0.0, 0.0, 0.0, 0.0], &[4]),
    );
    base
}

/// Helper: read back an MLX f32 tensor as `Vec<f32>`.
fn read_f32(t: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::eval(t);
    let bytes = mlxcel_core::array_to_raw_bytes(t);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn parse_yaml_and_apply_interpolate_via_pipeline() {
    // 1. Materialize donor files on disk.
    let tmp = tempfile::tempdir().expect("tempdir");
    let donor_a_path = tmp.path().join("model_a.safetensors");
    let donor_b_path = tmp.path().join("model_b.safetensors");
    write_safetensors(&donor_a_path, &synth_donor_a());
    write_safetensors(&donor_b_path, &synth_donor_b());

    // 2. Write a YAML config next to the donors — relative paths
    //    are resolved against the YAML's parent directory.
    let yaml_path = tmp.path().join("interpolate.yaml");
    let yaml = r#"version: 1
operations:
  - op: interpolate
    pattern: "model.layers.*.w"
    source_a: "./model_a.safetensors"
    source_b: "./model_b.safetensors"
    ratio: 0.5
    method: lerp
"#;
    fs::write(&yaml_path, yaml).expect("write yaml");

    // 3. Parse the YAML — this is the exact entry point the CLI
    //    flag (A4) will invoke.
    let pipeline: SurgeryPipeline = parse_config_file(&yaml_path).expect("parse_config_file");
    assert_eq!(pipeline.len(), 1);

    // 4. Apply the pipeline through the WeightTransform trait —
    //    this is the same interface the consolidated text/VLM
    //    loader will use to call into surgery.
    let mut weights = synth_base_zeros();
    WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect("pipeline apply must succeed");

    // 5. Verify the blend produced the expected LERP@0.5 values.
    let l0 = read_f32(weights.get("model.layers.0.w").unwrap());
    let l1 = read_f32(weights.get("model.layers.1.w").unwrap());

    // LERP at t=0.5 of [1,2,3,4] and [11,22,33,44] is [6,12,18,24].
    assert_close(&l0, &[6.0, 12.0, 18.0, 24.0], 1e-5);
    // LERP at t=0.5 of [5,6,7,8] and [55,66,77,88] is [30,36,42,48].
    assert_close(&l1, &[30.0, 36.0, 42.0, 48.0], 1e-5);
    // Output must contain no NaN / Inf — the basic sanity gate.
    for v in l0.iter().chain(l1.iter()) {
        assert!(v.is_finite(), "non-finite output value: {v}");
    }
}

#[test]
fn direct_interpolate_op_via_pipeline_loads_safetensors_from_disk() {
    // Same flow as above but constructs the `InterpolateOp` directly
    // (mirroring how a programmatic caller might wire surgery)
    // rather than going through YAML parsing.
    let tmp = tempfile::tempdir().expect("tempdir");
    let donor_a_path = tmp.path().join("model_a.safetensors");
    let donor_b_path = tmp.path().join("model_b.safetensors");
    write_safetensors(&donor_a_path, &synth_donor_a());
    write_safetensors(&donor_b_path, &synth_donor_b());

    let op = InterpolateOp::new(
        "model.layers.*.w",
        donor_a_path,
        donor_b_path,
        0.0, // ratio=0 ⇒ output equals donor A
        InterpolateMethod::Slerp,
    )
    .expect("constructor");

    let mut pipeline = SurgeryPipeline::new();
    pipeline.push(std::sync::Arc::new(op));

    let mut weights = synth_base_zeros();
    WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect("pipeline apply with disk donors");

    // ratio=0.0 ⇒ exact donor A values.
    let l0 = read_f32(weights.get("model.layers.0.w").unwrap());
    let l1 = read_f32(weights.get("model.layers.1.w").unwrap());
    assert_close(&l0, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    assert_close(&l1, &[5.0, 6.0, 7.0, 8.0], 1e-5);
}

#[test]
fn pipeline_without_interpolate_op_leaves_weights_untouched() {
    // Acceptance (e): without the op in the pipeline, bit-exact to
    // baseline (inherited from A4). An empty pipeline is the
    // strongest version of this guarantee.
    let pipeline = SurgeryPipeline::new();
    let mut weights = synth_base_zeros();
    let l0_before = read_f32(weights.get("model.layers.0.w").unwrap());
    let l1_before = read_f32(weights.get("model.layers.1.w").unwrap());

    WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect("empty pipeline apply");

    let l0_after = read_f32(weights.get("model.layers.0.w").unwrap());
    let l1_after = read_f32(weights.get("model.layers.1.w").unwrap());
    assert_eq!(l0_before, l0_after, "empty pipeline must be bit-exact");
    assert_eq!(l1_before, l1_after, "empty pipeline must be bit-exact");
}

#[test]
fn interpolate_op_surfaces_missing_donor_file_error() {
    // The donor path passed in does not exist on disk; the op
    // must fail with a readable error rather than panicking.
    let op = InterpolateOp::new(
        "model.layers.*.w",
        Path::new("/nonexistent/path/donor_a.safetensors").to_path_buf(),
        Path::new("/nonexistent/path/donor_b.safetensors").to_path_buf(),
        0.5,
        InterpolateMethod::Lerp,
    )
    .expect("constructor — paths are not validated until apply");

    let mut pipeline = SurgeryPipeline::new();
    pipeline.push(std::sync::Arc::new(op));

    let mut weights = synth_base_zeros();
    let err = WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect_err("missing file must surface as pipeline error");
    assert!(
        err.contains("source_a") || err.contains("source_b") || err.contains("nonexistent"),
        "error must mention the failing donor source: {err}",
    );
}

#[test]
fn yaml_parser_rejects_nonexistent_donor_at_parse_time() {
    // The config layer refuses to construct a pipeline when
    // any donor file does not exist. This is the first line of
    // defense and means surgery never gets as far as touching
    // weights for a clearly-broken config.
    let tmp = tempfile::tempdir().expect("tempdir");
    let yaml_path = tmp.path().join("bad.yaml");
    let yaml = r#"version: 1
operations:
  - op: interpolate
    pattern: "*"
    source_a: "./missing_a.safetensors"
    source_b: "./missing_b.safetensors"
    ratio: 0.5
    method: lerp
"#;
    fs::write(&yaml_path, yaml).expect("write yaml");
    let err: SurgeryError =
        parse_config_file(&yaml_path).expect_err("missing donor must fail at parse time");
    let msg = format!("{err}");
    assert!(
        msg.contains("does not exist"),
        "error must mention missing donor: {msg}"
    );
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {actual:?} vs {expected:?}",
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {i}: got {a}, expected {e} (tol {tol})",
        );
    }
}
