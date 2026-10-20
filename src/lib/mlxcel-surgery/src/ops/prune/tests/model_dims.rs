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

//! Direct unit tests for `ModelDims::from_config` and `validate_ids`.
//!
//! Both helpers are pure `serde_json::Value -> Result` functions, so
//! these tests use table-driven JSON fixtures and never touch MLX
//! arrays. They pin every fallback branch and every error message.

use super::super::PruneSelector;
use super::super::model_dims::{ModelDims, validate_ids};
use serde_json::{Value, json};

fn dims(cfg: Value) -> ModelDims {
    ModelDims::from_config(&cfg).expect("config should parse")
}

fn dims_err(cfg: Value) -> String {
    ModelDims::from_config(&cfg)
        .expect_err("config should be rejected")
        .to_string()
}

fn model(num_heads: usize, intermediate_size: usize, num_hidden_layers: usize) -> ModelDims {
    ModelDims {
        num_heads,
        num_kv_heads: num_heads,
        head_dim: 8,
        intermediate_size,
        num_hidden_layers,
    }
}

// ---------- ModelDims::from_config: success paths ----------

#[test]
fn from_config_reads_all_top_level_fields() {
    let d = dims(json!({
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
    }));
    assert_eq!(d.num_heads, 8);
    assert_eq!(d.num_kv_heads, 2);
    // Explicit head_dim wins over hidden_size / num_heads (= 8).
    assert_eq!(d.head_dim, 16);
    assert_eq!(d.intermediate_size, 256);
    assert_eq!(d.num_hidden_layers, 4);
    assert!(d.is_gqa());
}

#[test]
fn from_config_falls_back_to_text_config_for_vlm() {
    let d = dims(json!({
        "model_type": "vlm",
        "text_config": {
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 128,
            "num_hidden_layers": 3,
        },
    }));
    assert_eq!(d.num_heads, 4);
    assert_eq!(d.head_dim, 8);
    assert_eq!(d.intermediate_size, 128);
    assert_eq!(d.num_hidden_layers, 3);
}

#[test]
fn from_config_prefers_top_level_over_text_config() {
    let d = dims(json!({
        "num_attention_heads": 4,
        "hidden_size": 32,
        "text_config": { "num_attention_heads": 16, "hidden_size": 64 },
    }));
    assert_eq!(d.num_heads, 4);
    assert_eq!(d.head_dim, 8);
}

#[test]
fn from_config_kv_heads_default_to_num_heads() {
    let d = dims(json!({ "num_attention_heads": 6, "hidden_size": 48 }));
    assert_eq!(d.num_kv_heads, 6);
    assert!(!d.is_gqa());
}

#[test]
fn from_config_derives_head_dim_from_hidden_size() {
    let d = dims(json!({ "num_attention_heads": 4, "hidden_size": 64 }));
    assert_eq!(d.head_dim, 16);
}

#[test]
fn from_config_explicit_head_dim_needs_no_hidden_size() {
    let d = dims(json!({ "num_attention_heads": 4, "head_dim": 24 }));
    assert_eq!(d.head_dim, 24);
}

#[test]
fn from_config_optional_sizes_use_sentinels() {
    let d = dims(json!({ "num_attention_heads": 2, "hidden_size": 16 }));
    // Absent intermediate_size => 0 (rejected later by validate_ids
    // for mlp_channel); absent num_hidden_layers => unbounded.
    assert_eq!(d.intermediate_size, 0);
    assert_eq!(d.num_hidden_layers, usize::MAX);
}

// ---------- ModelDims::from_config: error messages ----------

#[test]
fn from_config_rejects_missing_num_attention_heads() {
    let msg = dims_err(json!({ "hidden_size": 32 }));
    assert_eq!(
        msg,
        "prune: config.json missing required field `num_attention_heads`"
    );
}

#[test]
fn from_config_rejects_non_integer_field() {
    for (value, shown) in [
        (json!("8"), "\"8\""),
        (json!(-1), "-1"),
        (json!(1.5), "1.5"),
    ] {
        let msg = dims_err(json!({ "num_attention_heads": value, "hidden_size": 32 }));
        assert_eq!(
            msg,
            format!(
                "prune: config.json `num_attention_heads` must be an unsigned integer; got {shown}"
            )
        );
    }
}

#[test]
fn from_config_rejects_non_integer_optional_field() {
    let msg = dims_err(json!({
        "num_attention_heads": 4,
        "num_key_value_heads": "two",
        "hidden_size": 32,
    }));
    assert_eq!(
        msg,
        "prune: config.json `num_key_value_heads` must be an unsigned integer; got \"two\""
    );
}

#[test]
fn from_config_rejects_missing_head_dim_and_hidden_size() {
    let msg = dims_err(json!({ "num_attention_heads": 4 }));
    assert_eq!(
        msg,
        "prune: config.json must provide head_dim or hidden_size + num_attention_heads"
    );
}

#[test]
fn from_config_rejects_zero_heads_during_head_dim_derivation() {
    // No head_dim, so the guard inside the derivation fires first.
    let msg = dims_err(json!({ "num_attention_heads": 0, "hidden_size": 32 }));
    assert_eq!(msg, "prune: num_attention_heads must be > 0");
}

#[test]
fn from_config_rejects_zero_heads_with_explicit_head_dim() {
    // Explicit head_dim skips the derivation, so the unconditional
    // trailing guard is the one that fires.
    let msg = dims_err(json!({ "num_attention_heads": 0, "head_dim": 8 }));
    assert_eq!(msg, "prune: num_attention_heads must be > 0");
}

#[test]
fn from_config_rejects_indivisible_hidden_size() {
    let msg = dims_err(json!({ "num_attention_heads": 5, "hidden_size": 32 }));
    assert_eq!(
        msg,
        "prune: hidden_size (32) not divisible by num_attention_heads (5)"
    );
}

#[test]
fn from_config_rejects_zero_kv_heads() {
    let msg = dims_err(json!({
        "num_attention_heads": 4,
        "num_key_value_heads": 0,
        "hidden_size": 32,
    }));
    assert_eq!(msg, "prune: num_key_value_heads must be > 0");
}

#[test]
fn from_config_rejects_zero_head_dim() {
    let msg = dims_err(json!({ "num_attention_heads": 4, "head_dim": 0 }));
    assert_eq!(msg, "prune: head_dim must be > 0");
}

#[test]
fn from_config_rejects_zero_derived_head_dim() {
    // hidden_size 0 is divisible by any head count and derives head_dim 0.
    let msg = dims_err(json!({ "num_attention_heads": 4, "hidden_size": 0 }));
    assert_eq!(msg, "prune: head_dim must be > 0");
}

// ---------- validate_ids ----------

fn validation_err(selector: &PruneSelector, m: &ModelDims) -> String {
    validate_ids(selector, m)
        .expect_err("ids should be rejected")
        .to_string()
}

#[test]
fn validate_ids_accepts_in_range_ids_and_empty_lists() {
    let m = model(4, 16, 3);
    let ok = [
        PruneSelector::Layer {
            layer_ids: vec![0, 2],
        },
        PruneSelector::Layer { layer_ids: vec![] },
        PruneSelector::AttentionHead {
            head_ids: vec![0, 3],
        },
        PruneSelector::AttentionHead { head_ids: vec![] },
        PruneSelector::MlpChannel {
            channel_ids: vec![0, 15],
        },
        PruneSelector::MlpChannel {
            channel_ids: vec![],
        },
    ];
    for selector in &ok {
        assert!(validate_ids(selector, &m).is_ok(), "{selector:?}");
    }
}

#[test]
fn validate_ids_rejects_out_of_range_layer() {
    let m = model(4, 16, 3);
    // The upper bound is exclusive; the first offender is reported.
    let selector = PruneSelector::Layer {
        layer_ids: vec![1, 3, 9],
    };
    assert_eq!(
        validation_err(&selector, &m),
        "prune: layer_id 3 out of range (num_hidden_layers=3)"
    );
}

#[test]
fn validate_ids_rejects_out_of_range_head() {
    let m = model(4, 16, 3);
    let selector = PruneSelector::AttentionHead {
        head_ids: vec![0, 4],
    };
    assert_eq!(
        validation_err(&selector, &m),
        "prune: head_id 4 out of range (num_attention_heads=4)"
    );
}

#[test]
fn validate_ids_rejects_out_of_range_channel() {
    let m = model(4, 16, 3);
    let selector = PruneSelector::MlpChannel {
        channel_ids: vec![16],
    };
    assert_eq!(
        validation_err(&selector, &m),
        "prune: channel_id 16 out of range (intermediate_size=16)"
    );
}

#[test]
fn validate_ids_requires_intermediate_size_for_mlp_channel() {
    let m = model(4, 0, 3);
    // Fires even when the id list is empty.
    for channel_ids in [vec![], vec![0]] {
        let selector = PruneSelector::MlpChannel { channel_ids };
        assert_eq!(
            validation_err(&selector, &m),
            "prune: mlp_channel granularity requires intermediate_size in config.json"
        );
    }
}

#[test]
fn validate_ids_ignores_unrelated_dimensions() {
    // Layer / head selectors do not depend on intermediate_size.
    let m = model(4, 0, 3);
    assert!(validate_ids(&PruneSelector::Layer { layer_ids: vec![2] }, &m).is_ok());
    assert!(validate_ids(&PruneSelector::AttentionHead { head_ids: vec![3] }, &m).is_ok());
}
