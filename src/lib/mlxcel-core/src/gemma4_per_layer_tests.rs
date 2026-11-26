// Copyright 2025-2026 Lablup Inc.
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

//! Ports of `TestGemma4PerLayerConfig` in Blaizzy/mlx-vlm#2719
//! (`mlx_vlm/tests/test_models.py`), against the shared function rather than
//! a config class.

use super::gemma4_global_attention_from_per_layer_config as resolve;
use serde_json::{Map, Value, json};

/// Upstream's `_text` fixture: `layers` layers in a sliding pattern of 6, so
/// every sixth layer (5, 11, ...) is full attention.
fn layer_types(layers: usize) -> Vec<&'static str> {
    (0..layers)
        .map(|layer| {
            if (layer + 1) % 6 == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect()
}

fn block(value: Value) -> Map<String, Value> {
    value
        .as_object()
        .expect("per_layer_config is an object")
        .clone()
}

#[test]
fn absent_block_resolves_nothing() {
    assert_eq!(resolve(None, &layer_types(12), 256, 16), Ok(None));
}

#[test]
fn reads_full_attention_shapes() {
    let per_layer = block(json!({
        "05": {"head_dim": 512, "num_key_value_heads": 4},
        "11": {"head_dim": 512, "num_key_value_heads": 4},
    }));
    assert_eq!(
        resolve(Some(&per_layer), &layer_types(12), 256, 16),
        Ok(Some((512, 4)))
    );
    // The unpadded spelling names the same layer.
    let per_layer = block(json!({"5": {"head_dim": 384, "num_key_value_heads": 2}}));
    assert_eq!(
        resolve(Some(&per_layer), &layer_types(6), 256, 16),
        Ok(Some((384, 2)))
    );
}

#[test]
fn empty_map_inherits_base_shapes() {
    let per_layer = block(json!({}));
    assert_eq!(
        resolve(Some(&per_layer), &layer_types(6), 64, 4),
        Ok(Some((64, 4)))
    );
    // A KV-head-only entry inherits the base head size.
    let per_layer = block(json!({"05": {"num_key_value_heads": 2}}));
    assert_eq!(
        resolve(Some(&per_layer), &layer_types(6), 64, 4),
        Ok(Some((64, 2)))
    );
    // No full-attention layer at all: nothing to report.
    assert_eq!(
        resolve(Some(&block(json!({}))), &layer_types(5), 64, 4),
        Ok(None)
    );
}

#[test]
fn head_dim_only_inherits_kv_heads() {
    let per_layer = block(json!({"05": {"head_dim": 512}}));
    assert_eq!(
        resolve(Some(&per_layer), &layer_types(6), 256, 16),
        Ok(Some((512, 16)))
    );
}

#[test]
fn rejects_different_shapes() {
    let differing = block(json!({
        "05": {"head_dim": 512, "num_key_value_heads": 4},
        "11": {"head_dim": 512, "num_key_value_heads": 2},
    }));
    let error = resolve(Some(&differing), &layer_types(12), 256, 16).unwrap_err();
    assert_eq!(
        error,
        "per_layer_config gives full-attention layers different shapes; this is not supported"
    );
    // One entry while layer 11 inherits a different base shape.
    for overrides in [json!({"head_dim": 512}), json!({"num_key_value_heads": 4})] {
        let single = block(json!({"05": overrides}));
        let error = resolve(Some(&single), &layer_types(12), 256, 16).unwrap_err();
        assert!(error.contains("different shapes"), "{error}");
    }
}

#[test]
fn rejects_sliding_layer_override() {
    let sliding = block(json!({"00": {"head_dim": 512}}));
    let error = resolve(Some(&sliding), &layer_types(12), 256, 16).unwrap_err();
    assert_eq!(
        error,
        "per_layer_config overrides layer 0, which is not a full-attention layer; \
         this is not supported"
    );
    // Out of range in either direction is the same error.
    for index in ["12", "17", "-1"] {
        let out_of_range = block(json!({index: {"head_dim": 512}}));
        let error = resolve(Some(&out_of_range), &layer_types(12), 256, 16).unwrap_err();
        assert!(error.contains("not a full-attention layer"), "{error}");
    }
}

#[test]
fn rejects_unknown_key() {
    let unknown = block(json!({"05": {"head_dim": 512, "window": 8, "bias": true}}));
    let error = resolve(Some(&unknown), &layer_types(6), 256, 16).unwrap_err();
    assert_eq!(
        error,
        "per_layer_config has unsupported keys ['bias', 'window']"
    );
}

#[test]
fn rejects_entries_upstream_would_fail_on_later() {
    let cases = [
        (json!({"five": {"head_dim": 512}}), "not an integer"),
        (json!({"05": 512}), "is not an object"),
        (
            json!({"05": {"head_dim": "512"}}),
            "`head_dim` must be a positive integer",
        ),
        (
            json!({"05": {"num_key_value_heads": 0}}),
            "`num_key_value_heads` must be a positive integer",
        ),
        (
            json!({"5": {"head_dim": 512}, "05": {"head_dim": 512}}),
            "names layer 5 more than once",
        ),
    ];
    for (per_layer, expected) in cases {
        let error = resolve(Some(&block(per_layer)), &layer_types(6), 256, 16).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}
