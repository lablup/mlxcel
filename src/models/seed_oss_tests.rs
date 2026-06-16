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

//! Unit tests for ByteDance Seed-OSS (`seed_oss`) config parsing.
//!
//! These cover the checkpoint-free surface: that `config.json` deserializes
//! into `ModelArgs` with the Seed-OSS deltas (explicit `head_dim`, the split
//! `attention_bias` / `attention_out_bias`, untied embeddings, the `"default"`
//! rope_scaling, and 4-bit quantization defaults). The MLX op path and
//! end-to-end generation are validated against a real `mlx-community` Seed-OSS
//! checkpoint, which needs a Metal device and is not exercised here.

use super::seed_oss::ModelArgs;

/// A trimmed `Seed-OSS-36B-Instruct-4bit` config, with the fields the loader
/// reads. Mirrors the real checkpoint's `config.json`.
const SEED_OSS_36B_CONFIG: &str = r#"{
    "model_type": "seed_oss",
    "attention_bias": true,
    "attention_out_bias": false,
    "head_dim": 128,
    "hidden_size": 5120,
    "intermediate_size": 27648,
    "max_position_embeddings": 524288,
    "mlp_bias": false,
    "num_attention_heads": 80,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "rope_type": "default"
    },
    "rope_theta": 10000000.0,
    "tie_word_embeddings": false,
    "vocab_size": 155136,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn seed_oss_config_parses_core_fields() {
    let args: ModelArgs = serde_json::from_str(SEED_OSS_36B_CONFIG).expect("parse seed_oss config");
    assert_eq!(args.model_type, "seed_oss");
    assert_eq!(args.hidden_size, 5120);
    assert_eq!(args.num_hidden_layers, 64);
    assert_eq!(args.intermediate_size, 27648);
    assert_eq!(args.num_attention_heads, 80);
    assert_eq!(args.num_key_value_heads, 8);
    assert_eq!(args.vocab_size, 155136);
    assert_eq!(args.rms_norm_eps, 1e-06);
    // rope_theta is 1e7 for Seed-OSS, not the 1e4 default.
    assert_eq!(args.rope_theta, 10_000_000.0);
}

#[test]
fn seed_oss_head_dim_is_explicit_not_derived() {
    let args: ModelArgs = serde_json::from_str(SEED_OSS_36B_CONFIG).expect("parse seed_oss config");
    // head_dim is given explicitly (128) and must NOT be hidden / heads
    // (5120 / 80 = 64).
    assert_eq!(args.head_dim, Some(128));
    assert_eq!(args.head_dim(), 128);
    assert_ne!(args.head_dim(), args.hidden_size / args.num_attention_heads);
}

#[test]
fn seed_oss_split_attention_bias_flags_parse() {
    let args: ModelArgs = serde_json::from_str(SEED_OSS_36B_CONFIG).expect("parse seed_oss config");
    // q/k/v carry a bias; o_proj does not. The two flags are independent.
    assert!(args.attention_bias);
    assert!(!args.attention_out_bias);
    assert!(!args.mlp_bias);
}

#[test]
fn seed_oss_embeddings_are_untied() {
    let args: ModelArgs = serde_json::from_str(SEED_OSS_36B_CONFIG).expect("parse seed_oss config");
    // Seed-OSS-36B ships a separate lm_head.
    assert!(!args.tie_word_embeddings);
}

#[test]
fn seed_oss_rope_scaling_is_default_no_scaling() {
    let args: ModelArgs = serde_json::from_str(SEED_OSS_36B_CONFIG).expect("parse seed_oss config");
    let scaling = args
        .rope_scaling
        .as_ref()
        .expect("seed_oss ships rope_scaling");
    // "default" means no llama3/linear scaling is applied.
    assert_eq!(
        scaling.get("rope_type").and_then(|v| v.as_str()),
        Some("default")
    );
}

#[test]
fn seed_oss_quantization_is_read_from_config() {
    let args: ModelArgs = serde_json::from_str(SEED_OSS_36B_CONFIG).expect("parse seed_oss config");
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn seed_oss_optional_flags_default_when_absent() {
    // A minimal config drops the bias flags, rope fields, and quantization; the
    // loader must fall back to safe defaults (no bias, tied default, 4-bit/64,
    // 1e4 rope_theta) and derive head_dim from hidden / heads.
    let minimal = r#"{
        "model_type": "seed_oss",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "vocab_size": 100
    }"#;
    let args: ModelArgs = serde_json::from_str(minimal).expect("parse minimal seed_oss config");
    assert!(!args.attention_bias);
    assert!(!args.attention_out_bias);
    assert!(!args.mlp_bias);
    // tie_word_embeddings defaults to true (the reference dataclass default).
    assert!(args.tie_word_embeddings);
    assert!(args.quantization.is_none());
    assert!(args.rope_scaling.is_none());
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    assert_eq!(args.rope_theta, 10_000.0); // default when omitted
    // head_dim derives from hidden / heads when not given.
    assert_eq!(args.head_dim, None);
    assert_eq!(args.head_dim(), 64 / 4);
    assert_eq!(args.head_dim(), 16);
}
