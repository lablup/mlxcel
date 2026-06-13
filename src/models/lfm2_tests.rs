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

//! Unit tests for LFM2 / LFM2-MoE config parsing and the config-derived layer
//! typing and routing wiring.
//!
//! These cover the checkpoint-free surface: that the dense `lfm2` and sparse
//! `lfm2_moe` `config.json` files deserialize into `ModelArgs`, that the
//! full-attention layer indices derive correctly (explicit list for `lfm2`,
//! `layer_types` for `lfm2_moe`), that dense-vs-MoE feed-forward selection
//! follows `num_dense_layers`, and that the sigmoid-gating fields parse. The
//! numeric correctness of the short-conv and the sigmoid-gated routing is
//! validated end-to-end against real `mlx-community` checkpoints
//! (`LFM2-350M-8bit`, `LFM2-8B-A1B-4bit`) and is not exercised here (no Metal
//! device is assumed in unit tests).

use super::lfm2::ModelArgs;

/// A trimmed `LFM2-350M` (dense) config with the fields the loader reads.
const LFM2_350M_CONFIG: &str = r#"{
    "model_type": "lfm2",
    "vocab_size": 65536,
    "hidden_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "max_position_embeddings": 128000,
    "norm_eps": 1e-05,
    "conv_bias": false,
    "conv_L_cache": 3,
    "rope_theta": 1000000.0,
    "full_attn_idxs": [2, 5, 8, 10, 12, 14],
    "eos_token_id": 7,
    "block_dim": 1024,
    "block_ff_dim": 6656,
    "quantization": { "group_size": 64, "bits": 8 }
}"#;

/// A trimmed `LFM2-8B-A1B` (MoE) config. `layer_types` drives the attention
/// layer derivation; the MoE block fields drive sigmoid-gated routing.
const LFM2_8B_MOE_CONFIG: &str = r#"{
    "model_type": "lfm2_moe",
    "vocab_size": 65536,
    "hidden_size": 2048,
    "intermediate_size": 7168,
    "moe_intermediate_size": 1792,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 128000,
    "norm_eps": 1e-05,
    "conv_bias": false,
    "conv_L_cache": 3,
    "rope_theta": 1000000.0,
    "num_dense_layers": 2,
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "norm_topk_prob": true,
    "use_expert_bias": true,
    "routed_scaling_factor": 1.0,
    "eos_token_id": 7,
    "layer_types": [
        "conv", "conv", "full_attention", "conv", "conv", "conv",
        "full_attention", "conv", "conv", "conv", "full_attention", "conv",
        "conv", "conv", "full_attention", "conv", "conv", "conv",
        "full_attention", "conv", "conv", "full_attention", "conv", "conv"
    ],
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn lfm2_dense_config_parses() {
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert_eq!(args.model_type, "lfm2");
    assert_eq!(args.hidden_size, 1024);
    assert_eq!(args.num_attention_heads, 16);
    assert_eq!(args.num_key_value_heads, 8);
    assert_eq!(args.num_hidden_layers, 16);
    assert_eq!(args.conv_l_cache, 3);
    assert!(!args.conv_bias);
    assert_eq!(args.rope_theta, 1_000_000.0);
    // head_dim derives from hidden/heads; the q/k layernorm weights are [64].
    assert_eq!(args.head_dim(), 64);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 8);
    assert_eq!(args.eos_token_ids(), vec![7]);
    // No MoE fields → a pure dense checkpoint.
    assert!(!args.is_moe());
}

#[test]
fn lfm2_dense_full_attn_idxs_are_explicit() {
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert_eq!(args.full_attn_idxs(), vec![2, 5, 8, 10, 12, 14]);
    for idx in 0..args.num_hidden_layers {
        let expect_attention = [2, 5, 8, 10, 12, 14].contains(&idx);
        assert_eq!(
            args.is_attention_layer(idx),
            expect_attention,
            "layer {idx}"
        );
    }
}

#[test]
fn lfm2_dense_every_layer_is_dense_ffn() {
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    // The dense checkpoint has no experts, so no layer routes to MoE and the
    // dense-layer boundary defaults to the full layer count.
    assert_eq!(args.num_dense_layers(), args.num_hidden_layers);
    for idx in 0..args.num_hidden_layers {
        assert!(!args.layer_is_moe(idx), "layer {idx} must use a dense MLP");
    }
}

#[test]
fn lfm2_moe_config_parses_sigmoid_gating_fields() {
    let args: ModelArgs = serde_json::from_str(LFM2_8B_MOE_CONFIG).expect("parse MoE config");
    assert_eq!(args.model_type, "lfm2_moe");
    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_attention_heads, 32);
    assert_eq!(args.head_dim(), 64);
    assert_eq!(args.bits(), 4);
    assert!(args.is_moe());
    assert_eq!(args.num_experts, Some(32));
    assert_eq!(args.num_experts_per_tok, Some(4));
    assert_eq!(args.moe_intermediate_size, Some(1792));
    assert_eq!(args.num_dense_layers, Some(2));
    // The load-bearing sigmoid-gating switches.
    assert_eq!(args.norm_topk_prob, Some(true));
    assert_eq!(args.use_expert_bias, Some(true));
    assert_eq!(args.routed_scaling_factor, 1.0);
    assert_eq!(args.eos_token_ids(), vec![7]);
}

#[test]
fn lfm2_moe_full_attn_idxs_derive_from_layer_types() {
    let args: ModelArgs = serde_json::from_str(LFM2_8B_MOE_CONFIG).expect("parse MoE config");
    // `full_attn_idxs` is absent, so the indices come from layer_types entries
    // equal to "full_attention".
    assert!(args.full_attn_idxs.is_none());
    assert_eq!(args.full_attn_idxs(), vec![2, 6, 10, 14, 18, 21]);
    assert!(args.is_attention_layer(2));
    assert!(args.is_attention_layer(21));
    assert!(!args.is_attention_layer(0));
    assert!(!args.is_attention_layer(23));
}

#[test]
fn lfm2_moe_feed_forward_selection_follows_num_dense_layers() {
    let args: ModelArgs = serde_json::from_str(LFM2_8B_MOE_CONFIG).expect("parse MoE config");
    assert_eq!(args.num_dense_layers(), 2);
    // Layers 0 and 1 use a dense MLP; every later layer routes to the sparse
    // MoE block, including the attention layer at index 2.
    assert!(!args.layer_is_moe(0));
    assert!(!args.layer_is_moe(1));
    assert!(args.layer_is_moe(2));
    assert!(args.is_attention_layer(2) && args.layer_is_moe(2));
    assert!(args.layer_is_moe(23));
}

#[test]
fn lfm2_conv_state_keep_length_invariant() {
    // The per-layer conv cache holds the last `L_cache - 1` time steps of `Bx`
    // (shape `[batch, L_cache - 1, hidden]`), which is what the depthwise
    // kernel-size-`L_cache` causal conv needs prepended on decode.
    let args: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert_eq!(args.conv_l_cache, 3);
    assert_eq!(args.conv_l_cache - 1, 2);
}

#[test]
fn lfm2_eos_token_id_handles_scalar_array_and_missing() {
    // Scalar (the shape both shipped checkpoints use).
    let scalar: ModelArgs = serde_json::from_str(LFM2_350M_CONFIG).expect("parse dense config");
    assert_eq!(scalar.eos_token_ids(), vec![7]);

    // Array form.
    let array_cfg = LFM2_350M_CONFIG.replace("\"eos_token_id\": 7", "\"eos_token_id\": [7, 1]");
    let array: ModelArgs = serde_json::from_str(&array_cfg).expect("parse array eos config");
    assert_eq!(array.eos_token_ids(), vec![7, 1]);

    // Missing → default `<|im_end|>` (id 7).
    let no_eos = LFM2_350M_CONFIG.replace("\"eos_token_id\": 7,", "");
    let missing: ModelArgs = serde_json::from_str(&no_eos).expect("parse no-eos config");
    assert_eq!(missing.eos_token_ids(), vec![7]);
}

#[test]
fn lfm2_unquantized_config_uses_quantization_defaults() {
    // Drop the quantization block (a bf16 checkpoint) and confirm the defaults.
    let no_quant = LFM2_350M_CONFIG.replace(
        ",\n    \"quantization\": { \"group_size\": 64, \"bits\": 8 }",
        "",
    );
    let args: ModelArgs = serde_json::from_str(&no_quant).expect("parse unquantized config");
    assert!(args.quantization.is_none());
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}
