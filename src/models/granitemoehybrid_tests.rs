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

//! Unit tests for Granite 4.x (`granitemoehybrid`) config parsing and the
//! config-derived geometry.
//!
//! These cover the checkpoint-free surface: that the `granitemoehybrid`
//! `config.json` deserializes into `ModelArgs` with the four Granite multipliers,
//! that the `layer_types` interleave classifies Mamba vs attention layers, that
//! `use_moe` follows `num_local_experts`, that the NoPE flag disables RoPE, that
//! the Mamba2 inner dimensions and the `conv_dim` / `projection_size`
//! derivations match the reference, and that the EOS default is the Granite
//! `<|end_of_text|>` id (100257). The numeric correctness of the interleaved
//! SSM + attention hybrid is validated end-to-end against the real
//! `mlx-community/granite-4.0-h-350m-4bit` checkpoint and is not exercised here
//! (no Metal device is assumed in unit tests).

use super::granitemoehybrid::ModelArgs;

/// A trimmed `granite-4.0-h-350m` config (the dense-MLP hybrid). Values mirror
/// the real checkpoint, including the full 32-entry `layer_types` interleave.
const GRANITE_4_350M_CONFIG: &str = r#"{
    "model_type": "granitemoehybrid",
    "attention_bias": false,
    "attention_multiplier": 0.015625,
    "embedding_multiplier": 12,
    "eos_token_id": 100257,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "logits_scaling": 3,
    "layer_types": [
        "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
        "mamba", "mamba", "attention", "mamba", "mamba", "attention", "mamba",
        "mamba", "mamba", "attention", "mamba", "mamba", "mamba", "mamba",
        "mamba", "mamba", "mamba", "mamba", "mamba", "attention", "mamba",
        "mamba", "mamba", "mamba"
    ],
    "mamba_conv_bias": true,
    "mamba_d_conv": 4,
    "mamba_d_head": 32,
    "mamba_d_state": 128,
    "mamba_n_groups": 1,
    "mamba_n_heads": 48,
    "mamba_proj_bias": false,
    "max_position_embeddings": 32768,
    "num_attention_heads": 12,
    "num_experts_per_tok": 0,
    "num_hidden_layers": 32,
    "num_key_value_heads": 4,
    "num_local_experts": 0,
    "position_embedding_type": "nope",
    "residual_multiplier": 0.246,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "shared_intermediate_size": 2048,
    "tie_word_embeddings": true,
    "vocab_size": 100352,
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn granite_hybrid_config_parses_scalar_multipliers() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(args.model_type, "granitemoehybrid");
    assert_eq!(args.embedding_multiplier, 12.0);
    assert_eq!(args.attention_multiplier, 0.015625);
    assert_eq!(args.residual_multiplier, 0.246);
    assert_eq!(args.logits_scaling, 3.0);
    assert!(args.tie_word_embeddings);
}

#[test]
fn granite_hybrid_layer_types_interleave() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");

    // The 350m checkpoint places attention at layers 10, 13, 17, 27; the rest
    // are Mamba (28 Mamba + 4 attention).
    let attention_layers: Vec<usize> = (0..args.num_hidden_layers)
        .filter(|&i| !args.is_mamba_layer(i))
        .collect();
    assert_eq!(attention_layers, vec![10, 13, 17, 27]);

    assert!(args.is_mamba_layer(0));
    assert!(!args.is_mamba_layer(10));
    assert!(!args.is_mamba_layer(27));
    assert!(args.is_mamba_layer(31));
}

#[test]
fn granite_hybrid_use_moe_follows_num_local_experts() {
    // num_local_experts == 0 -> dense MLP mode.
    let dense: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert!(!dense.use_moe());

    // A positive expert count switches on the MoE feed-forward.
    let moe_cfg = GRANITE_4_350M_CONFIG
        .replace("\"num_experts_per_tok\": 0,", "\"num_experts_per_tok\": 8,")
        .replace("\"num_local_experts\": 0,", "\"num_local_experts\": 64,");
    let moe: ModelArgs = serde_json::from_str(&moe_cfg).expect("parse moe config");
    assert!(moe.use_moe());
    assert_eq!(moe.num_local_experts, 64);
    assert_eq!(moe.num_experts_per_tok, 8);
}

#[test]
fn granite_hybrid_nope_disables_rope() {
    // position_embedding_type == "nope" -> no RoPE.
    let nope: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(nope.position_embedding_type, "nope");
    assert!(!nope.use_rope());

    // An explicit "rope" re-enables it.
    let rope_cfg = GRANITE_4_350M_CONFIG.replace(
        "\"position_embedding_type\": \"nope\",",
        "\"position_embedding_type\": \"rope\",",
    );
    let rope: ModelArgs = serde_json::from_str(&rope_cfg).expect("parse rope config");
    assert!(rope.use_rope());
}

#[test]
fn granite_hybrid_head_dim_and_mamba_dims() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");

    // Attention head_dim = hidden_size / num_attention_heads = 768 / 12 = 64.
    assert_eq!(args.head_dim(), 64);

    // Mamba inner width = mamba_n_heads * mamba_d_head = 48 * 32 = 1536.
    assert_eq!(args.mamba_intermediate(), 1536);
    assert_eq!(args.mamba_n_heads, 48);
    assert_eq!(args.mamba_d_head, 32);
    assert_eq!(args.mamba_d_state, 128);
    assert_eq!(args.mamba_d_conv, 4);
    assert_eq!(args.mamba_n_groups, 1);
    assert!(args.mamba_conv_bias);
}

#[test]
fn granite_hybrid_conv_and_projection_derivations() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");

    // conv_dim = mamba_intermediate + 2 * n_groups * d_state = 1536 + 2*1*128.
    assert_eq!(args.conv_dim(), 1792);
    assert_eq!(
        args.conv_dim(),
        args.mamba_intermediate() + 2 * args.mamba_n_groups * args.mamba_d_state
    );

    // projection_size = mamba_intermediate + conv_dim + mamba_n_heads.
    assert_eq!(args.projection_size(), 1536 + 1792 + 48);
    assert_eq!(args.projection_size(), 3376);
}

#[test]
fn granite_hybrid_time_step_limit_defaults_to_ssm_update_default() {
    // Granite passes no explicit limit, so it falls back to (0.001, 100.0).
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(args.time_step_limit, (0.001, 100.0));
}

#[test]
fn granite_hybrid_eos_token_id_scalar_and_missing() {
    // Scalar form (the 350m checkpoint ships eos_token_id == 100257).
    let scalar: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(scalar.eos_token_ids(), vec![100257]);

    // Missing -> default to the Granite `<|end_of_text|>` id (100257).
    let no_eos = GRANITE_4_350M_CONFIG.replace("\"eos_token_id\": 100257,", "");
    let missing: ModelArgs = serde_json::from_str(&no_eos).expect("parse no-eos config");
    assert_eq!(missing.eos_token_ids(), vec![100257]);
}

#[test]
fn granite_hybrid_quantization_read_from_config() {
    let args: ModelArgs = serde_json::from_str(GRANITE_4_350M_CONFIG).expect("parse config");
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);

    // Dropping the quantization block falls back to the defaults.
    let no_quant = GRANITE_4_350M_CONFIG.replace(
        ",\n    \"quantization\": { \"group_size\": 64, \"bits\": 4 }",
        "",
    );
    let bf16: ModelArgs = serde_json::from_str(&no_quant).expect("parse unquantized config");
    assert!(bf16.quantization.is_none());
    assert_eq!(bf16.group_size(), 64);
    assert_eq!(bf16.bits(), 4);
}
