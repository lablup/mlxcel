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

//! Unit tests for `PruneOp`.
//!
//! Split into multiple files to keep each below the workspace's
//! 500-line soft target:
//!
//! - `tests/mod.rs` (this file): shared helpers and classifier /
//!   internal-helper tests.
//! - `tests/layer.rs`: layer-granularity tests.
//! - `tests/attention_head.rs`: attention-head-granularity tests
//!   (including the GQA-safe policy and the quantized layout).
//! - `tests/mlp_channel.rs`: MLP-channel-granularity tests.

mod attention_head;
mod layer;
mod mlp_channel;

use crate::WeightMap;
use mlxcel_core as ffi;
use mlxcel_core::dtype;

/// Build a `config.json` value with the structural fields the op
/// needs.
pub(super) fn make_cfg(
    num_heads: usize,
    num_kv_heads: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
) -> serde_json::Value {
    serde_json::json!({
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
    })
}

/// Read a float weight tensor into a flat Vec<f32> for comparison.
/// Returns `(shape, floats)`.
pub(super) fn read_f32_2d(weights: &WeightMap, key: &str) -> (Vec<i32>, Vec<f32>) {
    let arr = weights.get(key).expect("key must exist");
    mlxcel_core::eval(arr);
    let shape = ffi::array_shape(arr);
    let bytes = ffi::array_to_raw_bytes(arr);
    assert_eq!(ffi::array_dtype(arr), dtype::FLOAT32);
    let mut floats = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    (shape, floats)
}

// ============================================================
// Classifier / internal-helper tests (kept here in mod.rs because
// they exercise the file-private helpers in `super::granularity`).
// ============================================================

use super::granularity::{
    AttentionRole, MlpRole, classify_attention_key, classify_mlp_key, extract_layer_index,
    key_has_dotted_segment,
};

#[test]
fn extract_layer_index_parses_dotted_keys() {
    assert_eq!(
        extract_layer_index("model.layers.7.self_attn.q_proj.weight"),
        Some(7)
    );
    assert_eq!(
        extract_layer_index("language_model.model.layers.0.mlp.up_proj.weight"),
        Some(0)
    );
    assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
    assert_eq!(extract_layer_index("model.layers.notanumber.x"), None);
}

#[test]
fn dotted_segment_matcher_avoids_substring_collisions() {
    // `q_projection` must not collide with `q_proj` (and vice
    // versa). This ensures `key_has_dotted_segment` is segment-
    // exact, not substring.
    assert!(key_has_dotted_segment(
        "model.layers.0.self_attn.q_proj.weight",
        "q_proj"
    ));
    assert!(!key_has_dotted_segment(
        "model.layers.0.self_attn.q_projection.weight",
        "q_proj"
    ));
}

#[test]
fn classify_attention_key_recognizes_each_role() {
    use AttentionRole::*;
    assert!(matches!(
        classify_attention_key("model.layers.0.self_attn.q_proj.weight"),
        QProj
    ));
    assert!(matches!(
        classify_attention_key("model.layers.0.self_attn.o_proj.bias"),
        OProj
    ));
    assert!(matches!(
        classify_attention_key("model.layers.0.self_attn.k_proj.weight"),
        KvProj
    ));
    assert!(matches!(
        classify_attention_key("model.layers.0.self_attn.v_proj.scales"),
        KvProj
    ));
    assert!(matches!(
        classify_attention_key("model.layers.0.self_attn.q_norm.weight"),
        Norm
    ));
    assert!(matches!(
        classify_attention_key("model.layers.0.self_attn.something_else.weight"),
        Unknown
    ));
}

#[test]
fn classify_mlp_key_recognizes_each_role() {
    use MlpRole::*;
    assert!(matches!(
        classify_mlp_key("model.layers.0.mlp.up_proj.weight"),
        UpOrGate
    ));
    assert!(matches!(
        classify_mlp_key("model.layers.0.mlp.gate_proj.weight"),
        UpOrGate
    ));
    assert!(matches!(
        classify_mlp_key("model.layers.0.mlp.down_proj.scales"),
        Down
    ));
    assert!(matches!(
        classify_mlp_key("model.layers.0.mlp.gate_up_proj.weight"),
        CombinedGateUp
    ));
    assert!(matches!(
        classify_mlp_key("model.layers.0.mlp.other.weight"),
        Unknown
    ));
}
