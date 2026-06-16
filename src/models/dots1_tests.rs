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

//! Unit tests for rednote dots.llm1 (`dots1`) config parsing and derivations.
//!
//! These cover the checkpoint-free surface: that `config.json` deserializes
//! into `ModelArgs`, that the dense-vs-MoE layer split honours
//! `first_k_dense_replace`, the shared-MLP intermediate derivation, the head_dim
//! fallback, nullable grouped-routing fields, and EOS resolution. The MLX op
//! path and end-to-end generation are validated against a real `mlx-community`
//! dots.llm1 checkpoint, which needs a Metal device and is not exercised here.

use super::dots1::{EosTokenId, ModelArgs};

/// A trimmed `dots.llm1.inst-mixed-4-6bit` config carrying the fields the loader
/// reads. `n_group` / `topk_group` are intentionally absent (null in the real
/// checkpoint), and `head_dim` is omitted (derived from hidden/heads).
const DOTS1_CONFIG: &str = r#"{
    "model_type": "dots1",
    "hidden_size": 4096,
    "num_hidden_layers": 62,
    "intermediate_size": 10944,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "rms_norm_eps": 1e-05,
    "vocab_size": 152064,
    "first_k_dense_replace": 1,
    "moe_intermediate_size": 1408,
    "n_routed_experts": 128,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "norm_topk_prob": true,
    "routed_scaling_factor": 2.5,
    "max_position_embeddings": 32768,
    "rope_theta": 10000000,
    "scoring_func": "sigmoid",
    "attention_bias": false,
    "tie_word_embeddings": false,
    "eos_token_id": 151649,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn parses_moe_topology_fields() {
    let args: ModelArgs = serde_json::from_str(DOTS1_CONFIG).expect("parse dots1 config");

    assert_eq!(args.model_type, "dots1");
    assert_eq!(args.num_hidden_layers, 62);
    assert_eq!(args.n_routed_experts, 128);
    assert_eq!(args.n_shared_experts, 2);
    assert_eq!(args.num_experts_per_tok, 6);
    assert_eq!(args.first_k_dense_replace, 1);
    assert!(args.norm_topk_prob);
    assert_eq!(args.routed_scaling_factor, 2.5);
    assert_eq!(args.scoring_func, "sigmoid");
    assert!(!args.tie_word_embeddings);
}

#[test]
fn quantization_defaults_resolve() {
    let args: ModelArgs = serde_json::from_str(DOTS1_CONFIG).expect("parse dots1 config");
    // Top-level quant block: group_size 64, bits 4. The per-tensor 6-bit
    // tensors (v_proj / down_proj) are detected from shape by the loaders, so
    // the config-level default is all the model needs to pass through.
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn dense_prefix_then_moe_layers() {
    let args: ModelArgs = serde_json::from_str(DOTS1_CONFIG).expect("parse dots1 config");
    // first_k_dense_replace == 1: layer 0 dense, layers 1.. are MoE.
    assert!(!args.is_moe_layer(0));
    assert!(args.is_moe_layer(1));
    assert!(args.is_moe_layer(61));
}

#[test]
fn shared_mlp_intermediate_derivation() {
    let args: ModelArgs = serde_json::from_str(DOTS1_CONFIG).expect("parse dots1 config");
    // The fused shared expert spans moe_intermediate_size * n_shared_experts.
    assert_eq!(args.moe_intermediate_size * args.n_shared_experts, 2816);
}

#[test]
fn head_dim_falls_back_to_hidden_over_heads() {
    let args: ModelArgs = serde_json::from_str(DOTS1_CONFIG).expect("parse dots1 config");
    // No explicit head_dim in config -> hidden_size / num_attention_heads.
    assert_eq!(args.head_dim(), 128);
}

#[test]
fn grouped_routing_defaults_to_one_when_absent() {
    let args: ModelArgs = serde_json::from_str(DOTS1_CONFIG).expect("parse dots1 config");
    // Null n_group / topk_group in the checkpoint -> 1 / 1, which degenerates
    // the group-limited selection step to a no-op (skipped at runtime).
    assert_eq!(args.n_group(), 1);
    assert_eq!(args.topk_group(), 1);
}

#[test]
fn grouped_routing_parses_when_present() {
    let json = r#"{
        "model_type": "dots1",
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "intermediate_size": 10944,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "rms_norm_eps": 1e-05,
        "vocab_size": 152064,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 1408,
        "n_routed_experts": 128,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "n_group": 8,
        "topk_group": 4
    }"#;
    let args: ModelArgs = serde_json::from_str(json).expect("parse dots1 config with groups");
    assert_eq!(args.n_group(), 8);
    assert_eq!(args.topk_group(), 4);
}

#[test]
fn eos_single_id_merges_known_stop_tokens() {
    let args: ModelArgs = serde_json::from_str(DOTS1_CONFIG).expect("parse dots1 config");
    assert!(matches!(
        args.eos_token_id,
        Some(EosTokenId::Single(151649))
    ));
    // <|endoftext|> (151643) and <|endofresponse|> (151649) both present.
    let eos = args.resolved_eos();
    assert!(eos.contains(&151643));
    assert!(eos.contains(&151649));
}

#[test]
fn eos_array_form_parses() {
    let json = r#"{
        "model_type": "dots1",
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "intermediate_size": 10944,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "rms_norm_eps": 1e-05,
        "vocab_size": 152064,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 1408,
        "n_routed_experts": 128,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "eos_token_id": [151643, 151649]
    }"#;
    let args: ModelArgs = serde_json::from_str(json).expect("parse dots1 config with eos array");
    let eos = args.resolved_eos();
    assert!(eos.contains(&151643));
    assert!(eos.contains(&151649));
    assert_eq!(eos.len(), 2);
}
