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

//! Unit tests for PhiMoE config parsing.
//!
//! These cover the checkpoint-free surface: config deserialization and the
//! config-derived helper methods. Numeric correctness of the MoE forward pass
//! is validated end-to-end against a real `mlx-community` checkpoint and is
//! not exercised here (no Metal device is assumed in unit tests).

use super::phimoe::ModelArgs;

/// Trimmed config matching the `phi-3.5-moe-instruct` HuggingFace checkpoint
/// at `mlx-community/Phi-3.5-MoE-instruct-4bit`.
const PHI35_MOE_4BIT_CONFIG: &str = r#"{
    "model_type": "phimoe",
    "vocab_size": 32064,
    "hidden_size": 4096,
    "intermediate_size": 6400,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 16,
    "num_experts_per_tok": 2,
    "layer_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

/// Unquantized (bf16) config to confirm the quantization defaults.
const PHI35_MOE_BF16_CONFIG: &str = r#"{
    "model_type": "phimoe",
    "vocab_size": 32064,
    "hidden_size": 4096,
    "intermediate_size": 6400,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 16,
    "num_experts_per_tok": 2,
    "layer_norm_eps": 1e-5,
    "rope_theta": 10000.0
}"#;

#[test]
fn phimoe_4bit_config_parses() {
    let args: ModelArgs =
        serde_json::from_str(PHI35_MOE_4BIT_CONFIG).expect("parse 4-bit phimoe config");
    assert_eq!(args.model_type, "phimoe");
    assert_eq!(args.vocab_size, 32064);
    assert_eq!(args.hidden_size, 4096);
    assert_eq!(args.intermediate_size, 6400);
    assert_eq!(args.num_hidden_layers, 32);
    assert_eq!(args.num_attention_heads, 32);
    assert_eq!(args.num_key_value_heads, 8);
    assert_eq!(args.num_local_experts, 16);
    assert_eq!(args.num_experts_per_tok, 2);
    assert_eq!(args.layer_norm_eps, 1e-5);
    assert_eq!(args.rope_theta, 10000.0);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn phimoe_head_dim_derives_correctly() {
    let args: ModelArgs =
        serde_json::from_str(PHI35_MOE_4BIT_CONFIG).expect("parse 4-bit phimoe config");
    // 4096 hidden / 32 attention heads = 128 head_dim
    assert_eq!(args.head_dim(), 128);
}

#[test]
fn phimoe_num_kv_heads_accessor() {
    let args: ModelArgs =
        serde_json::from_str(PHI35_MOE_4BIT_CONFIG).expect("parse 4-bit phimoe config");
    assert_eq!(args.num_kv_heads(), 8);
}

#[test]
fn phimoe_unquantized_config_uses_quantization_defaults() {
    let args: ModelArgs =
        serde_json::from_str(PHI35_MOE_BF16_CONFIG).expect("parse bf16 phimoe config");
    assert!(args.quantization.is_none());
    // Defaults: group_size=64, bits=4
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}
