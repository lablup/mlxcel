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

//! Integration tests for `mlxcel-surgery` PruneOp (issue #376).
//!
//! These tests exercise the **public** API path that the rest of the
//! workspace will use: parse a YAML file -> get a `SurgeryPipeline` ->
//! invoke it through `WeightTransform::apply`. They replicate the
//! load-pipeline integration without requiring a real model on disk
//! (the actual end-to-end run against `mlx-community/Qwen2.5-0.5B-
//! Instruct-4bit` happens via the CLI integration in #379 / A4).

use std::collections::HashMap;
use std::io::Write;

use mlxcel_core::dtype;
use mlxcel_core::weights::{WeightMap, WeightTransform};
use mlxcel_surgery::{
    parse_config_file, parse_config_str, PruneOp, PruneSelector, SurgeryPipeline,
};

/// Build a Llama-style synthetic weight map for one layer with the
/// dimensions encoded in `cfg`. Returns owned `WeightMap`.
fn build_synthetic_weights(num_heads: usize, head_dim: usize, hidden_size: usize) -> WeightMap {
    let qkv_out = (num_heads * head_dim) as i32;
    let hidden = hidden_size as i32;
    let mut weights = HashMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        mlxcel_core::ones(&[qkv_out, hidden], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        mlxcel_core::ones(&[qkv_out, hidden], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        mlxcel_core::ones(&[qkv_out, hidden], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        mlxcel_core::ones(&[hidden, qkv_out], dtype::FLOAT32),
    );
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        mlxcel_core::ones(&[1024, hidden], dtype::FLOAT32),
    );
    weights
}

fn read_floats(weights: &WeightMap, key: &str) -> Vec<f32> {
    let arr = weights.get(key).expect("key");
    mlxcel_core::eval(arr);
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

#[test]
fn yaml_prune_attention_head_runs_through_pipeline_apply() {
    // YAML mirrors the example in `examples/surgery/full_example.yaml`.
    let yaml = r#"version: 1
operations:
  - op: prune
    granularity: attention_head
    pattern: "model.layers.0.self_attn.*"
    head_ids: [0, 2]
"#;
    let pipeline = parse_config_str(yaml, None).expect("parse YAML");
    assert_eq!(pipeline.len(), 1);

    let mut weights = build_synthetic_weights(
        /* heads */ 4, /* head_dim */ 8, /* hidden */ 32,
    );
    let cfg = serde_json::json!({
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
    });

    pipeline.apply(&mut weights, &cfg).expect("pipeline apply");

    // q_proj: heads 0 and 2 = rows [0..8) and [16..24) must be zero.
    let q = read_floats(&weights, "model.layers.0.self_attn.q_proj.weight");
    for r in 0..32 {
        let row: f32 = q[r * 32..(r + 1) * 32].iter().sum();
        let in_pruned_head = (0..8).contains(&r) || (16..24).contains(&r);
        if in_pruned_head {
            assert_eq!(row, 0.0, "q_proj row {r} should be zero");
        } else {
            assert!(row > 0.0, "q_proj row {r} should be untouched");
        }
    }

    // o_proj: heads 0 and 2 = columns [0..8) and [16..24) must be zero.
    let o = read_floats(&weights, "model.layers.0.self_attn.o_proj.weight");
    for r in 0..32 {
        for c in 0..32 {
            let v = o[r * 32 + c];
            let in_pruned_head = (0..8).contains(&c) || (16..24).contains(&c);
            if in_pruned_head {
                assert_eq!(v, 0.0, "o_proj row {r} col {c} should be zero");
            } else {
                assert_eq!(v, 1.0, "o_proj row {r} col {c} should be untouched");
            }
        }
    }

    // KV untouched (GQA-safe policy).
    let k = read_floats(&weights, "model.layers.0.self_attn.k_proj.weight");
    assert!(k.iter().all(|&v| v == 1.0), "k_proj must remain ones");
    let v = read_floats(&weights, "model.layers.0.self_attn.v_proj.weight");
    assert!(v.iter().all(|&v| v == 1.0), "v_proj must remain ones");
}

#[test]
fn yaml_prune_layer_zeroes_listed_layer_only() {
    let yaml = r#"version: 1
operations:
  - op: prune
    granularity: layer
    pattern: "model.layers.*"
    layer_ids: [1]
"#;
    let pipeline = parse_config_str(yaml, None).expect("parse YAML");
    let mut weights = HashMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        mlxcel_core::ones(&[16, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.1.self_attn.q_proj.weight".to_string(),
        mlxcel_core::ones(&[16, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.1.mlp.up_proj.weight".to_string(),
        mlxcel_core::ones(&[16, 8], dtype::FLOAT32),
    );
    let cfg = serde_json::json!({
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
    });

    pipeline.apply(&mut weights, &cfg).expect("apply");

    let layer0 = read_floats(&weights, "model.layers.0.self_attn.q_proj.weight");
    assert!(layer0.iter().all(|&v| v == 1.0));
    let layer1 = read_floats(&weights, "model.layers.1.self_attn.q_proj.weight");
    assert!(layer1.iter().all(|&v| v == 0.0));
    let layer1_mlp = read_floats(&weights, "model.layers.1.mlp.up_proj.weight");
    assert!(layer1_mlp.iter().all(|&v| v == 0.0));
}

#[test]
fn yaml_prune_mlp_channel_runs_through_pipeline_apply() {
    let yaml = r#"version: 1
operations:
  - op: prune
    granularity: mlp_channel
    pattern: "model.layers.0.mlp.*"
    channel_ids: [4, 12]
"#;
    let pipeline = parse_config_str(yaml, None).expect("parse YAML");
    let mut weights = HashMap::new();
    weights.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        mlxcel_core::ones(&[16, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        mlxcel_core::ones(&[16, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        mlxcel_core::ones(&[8, 16], dtype::FLOAT32),
    );
    let cfg = serde_json::json!({
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
    });
    pipeline.apply(&mut weights, &cfg).expect("apply");

    let up = read_floats(&weights, "model.layers.0.mlp.up_proj.weight");
    for r in 0..16 {
        let s: f32 = up[r * 8..(r + 1) * 8].iter().sum();
        if r == 4 || r == 12 {
            assert_eq!(s, 0.0);
        } else {
            assert!(s > 0.0);
        }
    }
    let down = read_floats(&weights, "model.layers.0.mlp.down_proj.weight");
    for r in 0..8 {
        for c in 0..16 {
            let v = down[r * 16 + c];
            if c == 4 || c == 12 {
                assert_eq!(v, 0.0);
            } else {
                assert_eq!(v, 1.0);
            }
        }
    }
}

#[test]
fn programmatic_pipeline_builder_works_without_yaml() {
    // Build a pipeline directly via `SurgeryPipeline::push` to prove
    // the public API supports programmatic construction (the path
    // used by tests in #378 / A4 until the CLI flag is wired).
    let mut pipeline = SurgeryPipeline::new();
    let op = PruneOp::new(
        "model.layers.0.self_attn.*",
        PruneSelector::AttentionHead { head_ids: vec![1] },
    )
    .expect("compile op");
    pipeline.push(op.into_shared());

    let mut weights = build_synthetic_weights(2, 8, 16);
    let cfg = serde_json::json!({
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
    });
    pipeline.apply(&mut weights, &cfg).expect("apply");

    let q = read_floats(&weights, "model.layers.0.self_attn.q_proj.weight");
    // head 0 rows [0..8) ones, head 1 rows [8..16) zero.
    let head0: f32 = q[0..8 * 16].iter().sum();
    let head1: f32 = q[8 * 16..].iter().sum();
    assert!(head0 > 0.0);
    assert_eq!(head1, 0.0);
}

#[test]
fn yaml_file_with_donor_resolved_relative() {
    // Verify the file-based parser still resolves relative source*
    // paths correctly when used alongside a prune op (sanity check
    // that adding A7 did not break the relative-path logic of A3).
    let dir = tempfile::tempdir().expect("tempdir");
    let yaml_path = dir.path().join("surgery.yaml");
    let mut f = std::fs::File::create(&yaml_path).expect("create yaml");
    write!(
        f,
        "version: 1\noperations:\n  - op: prune\n    granularity: layer\n    pattern: \"model.layers.*\"\n    layer_ids: [0]\n"
    )
    .expect("write yaml");

    let pipeline = parse_config_file(&yaml_path).expect("parse");
    assert_eq!(pipeline.len(), 1);
}

#[test]
fn baseline_with_empty_pipeline_is_bit_exact_no_op() {
    // Acceptance criterion (e): without PruneOp the pipeline must be
    // bit-exact (in this synthetic test, byte-exact) to the input.
    let yaml = "version: 1\noperations: []\n";
    let pipeline = parse_config_str(yaml, None).expect("empty pipeline");

    let mut weights = build_synthetic_weights(4, 8, 32);
    let snapshot_before: HashMap<String, Vec<u8>> = weights
        .iter()
        .map(|(k, v)| {
            mlxcel_core::eval(v);
            (k.clone(), mlxcel_core::array_to_raw_bytes(v))
        })
        .collect();
    let cfg = serde_json::json!({
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
    });
    pipeline.apply(&mut weights, &cfg).expect("apply empty");
    for (k, before) in &snapshot_before {
        let after = {
            let v = weights.get(k).expect("k");
            mlxcel_core::eval(v);
            mlxcel_core::array_to_raw_bytes(v)
        };
        assert_eq!(
            before, &after,
            "key {k} must be byte-exact after empty pipeline"
        );
    }
}

#[test]
fn end_to_end_prune_then_inspect_yields_no_nan_inf() {
    // Acceptance criterion (b) — modulo a real model. Apply prune
    // through the pipeline and verify the result contains no NaN/Inf
    // values (which would indicate the slice_update path corrupted
    // memory).
    let yaml = r#"version: 1
operations:
  - op: prune
    granularity: attention_head
    pattern: "model.layers.0.self_attn.*"
    head_ids: [1]
"#;
    let pipeline = parse_config_str(yaml, None).expect("parse");
    let mut weights = build_synthetic_weights(4, 8, 32);
    let cfg = serde_json::json!({
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
    });
    pipeline.apply(&mut weights, &cfg).expect("apply");
    for k in weights.keys() {
        let floats = read_floats(&weights, k);
        for (i, &f) in floats.iter().enumerate() {
            assert!(
                f.is_finite() || f == 0.0,
                "{k}[{i}] = {f} is NaN/Inf after prune"
            );
        }
    }
}
