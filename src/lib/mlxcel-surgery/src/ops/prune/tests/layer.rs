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

//! `granularity: layer` unit tests for `PruneOp`.

use super::super::{PruneOp, PruneSelector};
use super::{make_cfg, read_f32_2d};
use crate::{SurgeryOp, WeightMap};
use mlxcel_core as ffi;
use mlxcel_core::dtype;

#[test]
fn layer_prune_zeros_only_requested_layer() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[8, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        ffi::ones(&[8, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.1.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[8, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        ffi::ones(&[16, 8], dtype::FLOAT32),
    );

    let op = PruneOp::new(
        "model.layers.*",
        PruneSelector::Layer { layer_ids: vec![0] },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 8, 16, 2);
    op.apply(&mut weights, &cfg).expect("apply");

    // Layer 0 q_proj is zero.
    let (_, floats) = read_f32_2d(&weights, "model.layers.0.self_attn.q_proj.weight");
    assert!(
        floats.iter().all(|&x| x == 0.0),
        "layer 0 q_proj must be zero"
    );

    // Layer 0 up_proj is zero.
    let (_, floats) = read_f32_2d(&weights, "model.layers.0.mlp.up_proj.weight");
    assert!(
        floats.iter().all(|&x| x == 0.0),
        "layer 0 up_proj must be zero"
    );

    // Layer 1 q_proj is unchanged (still ones).
    let (_, floats) = read_f32_2d(&weights, "model.layers.1.self_attn.q_proj.weight");
    assert!(
        floats.iter().all(|&x| x == 1.0),
        "layer 1 q_proj must be untouched"
    );

    // Non-layer weight is untouched.
    let (_, floats) = read_f32_2d(&weights, "model.embed_tokens.weight");
    assert!(
        floats.iter().all(|&x| x == 1.0),
        "non-layer weight must be untouched"
    );
}

#[test]
fn layer_prune_errors_on_out_of_range_id() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[8, 8], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.*",
        PruneSelector::Layer {
            layer_ids: vec![99],
        },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 8, 16, 2);
    let err = op.apply(&mut weights, &cfg).expect_err("oor");
    assert!(format!("{err}").contains("out of range"));
}

#[test]
fn layer_prune_errors_on_zero_match() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[4, 4], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.does_not_exist.*",
        PruneSelector::Layer { layer_ids: vec![0] },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 4, 8, 1);
    let err = op.apply(&mut weights, &cfg).expect_err("no match");
    assert!(format!("{err}").contains("zero tensors"));
}

#[test]
fn config_missing_num_heads_is_reported() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        ffi::ones(&[16, 8], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.self_attn.*",
        PruneSelector::AttentionHead { head_ids: vec![0] },
    )
    .expect("compile");
    let cfg = serde_json::json!({"hidden_size": 16});
    let err = op
        .apply(&mut weights, &cfg)
        .expect_err("missing num_heads must fail");
    assert!(
        format!("{err}").contains("num_attention_heads"),
        "error should mention missing field"
    );
}
