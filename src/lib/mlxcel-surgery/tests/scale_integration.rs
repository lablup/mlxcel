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

//! Integration tests for the A5 `ScaleOp` surgery — exercise the full
//! YAML → `SurgeryPipeline` → `WeightTransform::apply` chain against a
//! synthetic `WeightMap` populated through the public mlxcel-core
//! FFI. This is the harness that mirrors what the consolidated text
//! loader will do once A4's `--surgery` CLI flag lands: parse a YAML
//! config, build a pipeline, and apply it to an in-memory map.
//!
//! NOTE: This file does not load a real model on disk. A real-model
//! end-to-end smoke test (with `cargo run --release -- generate`)
//! lives in the PR description and the issue's acceptance criterion
//! (b). The unit + integration coverage here covers the deterministic
//! cases (zero-match error, dtype/shape preservation, quantized
//! routing, multi-layer wildcard) without depending on a model
//! download.

use mlxcel_core::weights::{WeightMap, WeightTransform};
use mlxcel_core::{MlxArray, UniquePtr};
use mlxcel_surgery::{SurgeryPipeline, parse_config_str};

/// Build a tiny synthetic 1-D f32 tensor from a literal slice.
fn f32_tensor(values: &[f32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, &[values.len() as i32])
}

/// Read a 1-D f32 tensor back to a `Vec<f32>`.
fn read_f32(tensor: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(tensor);
    let bytes = mlxcel_core::array_to_raw_bytes(tensor);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect()
}

#[test]
fn yaml_scale_config_mutates_matched_tensor_through_pipeline() {
    // Parse a YAML scale config and apply the resulting pipeline to a
    // synthetic WeightMap that contains the matched tensor plus two
    // unrelated tensors. The matched tensor must be multiplied by
    // the configured factor; the others must remain bit-identical.

    let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "model.layers.*.self_attn.o_proj.weight"
    factor: 1.5
"#;
    let pipeline: SurgeryPipeline = parse_config_str(yaml, None).expect("yaml parses");
    assert_eq!(pipeline.len(), 1);

    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        f32_tensor(&[2.0, 4.0, -6.0, 0.0]),
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        f32_tensor(&[100.0, 200.0]),
    );
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        f32_tensor(&[1.0, 2.0]),
    );

    WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect("pipeline apply succeeds");

    let scaled = weights
        .get("model.layers.0.self_attn.o_proj.weight")
        .unwrap();
    let values = read_f32(scaled);
    assert_eq!(values.len(), 4);
    for (got, want) in values.iter().zip([3.0_f32, 6.0, -9.0, 0.0].iter()) {
        assert!((got - want).abs() < 1e-5, "expected {want}, got {got}",);
    }

    // Untouched siblings — bit-identical.
    assert_eq!(
        read_f32(
            weights
                .get("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
        ),
        vec![100.0, 200.0],
    );
    assert_eq!(
        read_f32(weights.get("model.embed_tokens.weight").unwrap()),
        vec![1.0, 2.0],
    );
}

#[test]
fn yaml_multi_layer_glob_scales_every_matched_layer_once() {
    let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "model.layers.*.self_attn.o_proj.weight"
    factor: 0.25
"#;
    let pipeline = parse_config_str(yaml, None).expect("yaml parses");

    let mut weights = WeightMap::new();
    for layer in 0..4 {
        weights.insert(
            format!("model.layers.{layer}.self_attn.o_proj.weight"),
            f32_tensor(&[4.0, 8.0, 16.0]),
        );
    }
    // Unrelated key — must remain untouched.
    weights.insert("lm_head.weight".to_string(), f32_tensor(&[1.0]));

    WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect("apply succeeds");

    for layer in 0..4 {
        let arr = weights
            .get(&format!("model.layers.{layer}.self_attn.o_proj.weight"))
            .unwrap();
        let values = read_f32(arr);
        assert_eq!(values.len(), 3);
        for (got, want) in values.iter().zip([1.0_f32, 2.0, 4.0].iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "layer {layer}: expected {want}, got {got}",
            );
        }
    }
    assert_eq!(read_f32(weights.get("lm_head.weight").unwrap()), vec![1.0]);
}

#[test]
fn yaml_zero_match_surfaces_actionable_error() {
    // Pattern matches no tensors → loader-style error. This
    // verifies acceptance criterion (a)'s zero-match contract
    // through the YAML path.
    let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "nonexistent.*.weight"
    factor: 2.0
"#;
    let pipeline = parse_config_str(yaml, None).expect("yaml parses");

    let mut weights = WeightMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        f32_tensor(&[1.0, 2.0]),
    );

    let result = WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null);
    match result {
        Ok(_) => panic!("zero-match scale must surface an error"),
        Err(msg) => {
            assert!(
                msg.contains("matched zero tensors") || msg.contains("zero"),
                "expected zero-match error: {msg}"
            );
        }
    }

    // Weights must not have been touched on the error path.
    assert_eq!(
        read_f32(weights.get("model.embed_tokens.weight").unwrap()),
        vec![1.0, 2.0],
    );
}

#[test]
fn yaml_pipeline_with_real_scale_preserves_dtype_and_shape() {
    // A5 acceptance criterion (a) end-to-end: dtype and shape
    // preserved through the YAML pipeline, with multi-dimensional
    // inputs.
    let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "model.layer.weight"
    factor: -1.0
"#;
    let pipeline = parse_config_str(yaml, None).expect("yaml parses");

    let mut weights = WeightMap::new();
    let arr = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    weights.insert("model.layer.weight".to_string(), arr);

    WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
        .expect("apply succeeds");

    let after = weights.get("model.layer.weight").unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(after),
        mlxcel_core::dtype::FLOAT32,
        "dtype preserved"
    );
    assert_eq!(
        mlxcel_core::array_shape(after),
        vec![2, 3],
        "shape preserved"
    );
    mlxcel_core::eval(after);
    let bytes = mlxcel_core::array_to_raw_bytes(after);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
}
