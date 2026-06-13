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

//! Unit tests for Granite (dense) config parsing and derived scalar factors.
//!
//! These cover the checkpoint-free surface: that `config.json` deserializes
//! into `ModelArgs` with the four Granite multipliers and that the derived
//! `head_dim`/quantization defaults match the reference. End-to-end generation
//! is validated against a real `mlx-community` Granite checkpoint and is not
//! exercised here (no Metal device is assumed in unit tests).

use super::granite::ModelArgs;

/// A trimmed `granite-3.3-2b-instruct` config, with the fields the loader reads.
const GRANITE_33_2B_CONFIG: &str = r#"{
    "model_type": "granite",
    "attention_bias": false,
    "attention_multiplier": 0.015625,
    "embedding_multiplier": 12.0,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "logits_scaling": 8.0,
    "max_position_embeddings": 131072,
    "mlp_bias": false,
    "num_attention_heads": 32,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "residual_multiplier": 0.22,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000000.0,
    "tie_word_embeddings": true,
    "vocab_size": 49159,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn granite_config_parses_scalar_multipliers() {
    let args: ModelArgs = serde_json::from_str(GRANITE_33_2B_CONFIG).expect("parse granite config");
    assert_eq!(args.model_type, "granite");
    assert_eq!(args.attention_multiplier, 0.015625);
    assert_eq!(args.embedding_multiplier, 12.0);
    assert_eq!(args.residual_multiplier, 0.22);
    assert_eq!(args.logits_scaling, 8.0);
    assert!(args.tie_word_embeddings);
}

#[test]
fn granite_head_dim_defaults_from_hidden_and_heads() {
    let args: ModelArgs = serde_json::from_str(GRANITE_33_2B_CONFIG).expect("parse granite config");
    // head_dim is omitted from the config, so it derives from hidden/heads.
    assert_eq!(args.head_dim(), 2048 / 32);
    assert_eq!(args.head_dim(), 64);
}

#[test]
fn granite_quantization_is_read_from_config() {
    let args: ModelArgs = serde_json::from_str(GRANITE_33_2B_CONFIG).expect("parse granite config");
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn granite_unquantized_config_uses_defaults() {
    // Drop the quantization block (a bf16 checkpoint) and confirm the defaults.
    let no_quant = GRANITE_33_2B_CONFIG.replace(
        ",\n    \"quantization\": { \"group_size\": 64, \"bits\": 4 }",
        "",
    );
    let args: ModelArgs =
        serde_json::from_str(&no_quant).expect("parse unquantized granite config");
    assert!(args.quantization.is_none());
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn granite_tie_word_embeddings_defaults_true_when_absent() {
    // Granite ships tied embeddings by default; the loader must honor that when
    // the field is missing entirely.
    let minimal = r#"{
        "model_type": "granite",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-05,
        "vocab_size": 100,
        "logits_scaling": 8.0,
        "attention_multiplier": 0.1,
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22
    }"#;
    let args: ModelArgs = serde_json::from_str(minimal).expect("parse minimal granite config");
    assert!(args.tie_word_embeddings);
    assert_eq!(args.rope_theta, 10_000.0); // default when omitted
}
