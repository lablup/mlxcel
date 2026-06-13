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

//! Unit tests for Falcon-H1 config parsing and the config-derived Mamba2
//! geometry, the pre-folded-multiplier tied-head factor, and gate selection.
//!
//! These cover the checkpoint-free surface: that the `falcon_h1` `config.json`
//! deserializes into `ModelArgs`, that the conv/projection geometry derives
//! correctly, that the tied-head runtime factor is `lm_head / embedding`
//! (the only multiplier applied at runtime), that `mamba_rms_norm` gate
//! selection follows the config, and that the multiplier and EOS defaults are
//! safe. The numeric correctness of the parallel SSM + attention hybrid is
//! validated end-to-end against the real `mlx-community`
//! `Falcon-H1-Tiny-90M-Instruct-4bit` checkpoint and is not exercised here
//! (no Metal device is assumed in unit tests).

use super::falcon_h1::ModelArgs;

/// A trimmed `Falcon-H1-Tiny-90M-Instruct-4bit` config with the fields the
/// loader reads. Values mirror the real checkpoint.
const FALCON_H1_TINY_CONFIG: &str = r#"{
    "model_type": "falcon_h1",
    "vocab_size": 32768,
    "hidden_size": 512,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rms_norm_eps": 1e-05,
    "rope_theta": 100000000000.0,
    "mamba_d_conv": 4,
    "mamba_d_ssm": 768,
    "mamba_d_state": 64,
    "mamba_d_head": 32,
    "mamba_n_heads": 24,
    "mamba_n_groups": 1,
    "mamba_conv_bias": true,
    "mamba_proj_bias": false,
    "mamba_rms_norm": false,
    "mamba_norm_before_gate": false,
    "embedding_multiplier": 0.11083984375,
    "lm_head_multiplier": 0.078125,
    "attention_in_multiplier": 1.0,
    "attention_out_multiplier": 1.0,
    "key_multiplier": 1.0,
    "ssm_in_multiplier": 1.0,
    "ssm_out_multiplier": 1.0,
    "tie_word_embeddings": true,
    "attention_bias": false,
    "projectors_bias": false,
    "eos_token_id": [228, 11],
    "quantization": { "group_size": 64, "bits": 4 }
}"#;

#[test]
fn falcon_h1_config_parses() {
    let args: ModelArgs = serde_json::from_str(FALCON_H1_TINY_CONFIG).expect("parse config");
    assert_eq!(args.model_type, "falcon_h1");
    assert_eq!(args.vocab_size, 32768);
    assert_eq!(args.hidden_size, 512);
    assert_eq!(args.num_hidden_layers, 24);
    assert_eq!(args.num_attention_heads, 8);
    assert_eq!(args.num_key_value_heads, 2);
    assert_eq!(args.head_dim, 64);
    assert_eq!(args.rope_theta, 100_000_000_000.0);
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    assert!(args.tie_word_embeddings);
}

#[test]
fn falcon_h1_mamba_dims_parse() {
    let args: ModelArgs = serde_json::from_str(FALCON_H1_TINY_CONFIG).expect("parse config");
    assert_eq!(args.mamba_d_conv, 4);
    assert_eq!(args.mamba_d_ssm, 768);
    assert_eq!(args.mamba_d_state, 64);
    assert_eq!(args.mamba_n_heads, 24);
    assert_eq!(args.mamba_n_groups, 1);
    // mamba_d_head is explicit (32) and must tile d_ssm exactly.
    assert_eq!(args.mamba_head_dim(), 32);
    assert_eq!(args.mamba_head_dim() * args.mamba_n_heads, args.mamba_d_ssm);
}

#[test]
fn falcon_h1_mamba_head_dim_falls_back_to_dssm_over_heads() {
    // When `mamba_d_head` is absent, head_dim derives from d_ssm / n_heads.
    let no_head = FALCON_H1_TINY_CONFIG.replace("\"mamba_d_head\": 32,", "");
    let args: ModelArgs =
        serde_json::from_str(&no_head).expect("parse config without mamba_d_head");
    assert!(args.mamba_d_head.is_none());
    assert_eq!(args.mamba_head_dim(), 768 / 24);
    assert_eq!(args.mamba_head_dim(), 32);
}

#[test]
fn falcon_h1_conv_and_projection_geometry() {
    let args: ModelArgs = serde_json::from_str(FALCON_H1_TINY_CONFIG).expect("parse config");
    // conv_dim = d_ssm + 2 * n_groups * d_state = 768 + 2*1*64 = 896.
    assert_eq!(args.conv_dim(), 896);
    // projection_size = d_ssm + conv_dim + n_heads = 768 + 896 + 24 = 1688.
    assert_eq!(args.projection_size(), 1688);
    assert_eq!(
        args.projection_size(),
        args.mamba_d_ssm + args.conv_dim() + args.mamba_n_heads
    );
}

#[test]
fn falcon_h1_conv_state_keep_length() {
    // The per-layer conv cache holds the last `d_conv - 1` time steps, which is
    // what the depthwise kernel-size-`d_conv` causal conv prepends on decode.
    let args: ModelArgs = serde_json::from_str(FALCON_H1_TINY_CONFIG).expect("parse config");
    assert_eq!(args.mamba_d_conv, 4);
    assert_eq!(args.conv_state_len(), 3);
}

#[test]
fn falcon_h1_tied_head_factor_is_lm_over_embedding() {
    // The reference applies `logits *= lm_head_multiplier / embedding_multiplier`
    // for the tied head. 0.078125 / 0.11083984375 == 160/227 ≈ 0.7048458.
    let args: ModelArgs = serde_json::from_str(FALCON_H1_TINY_CONFIG).expect("parse config");
    let factor = args.tied_head_factor();
    assert!(
        (factor - 0.704_845_8_f32).abs() < 1e-4,
        "expected ~0.7048458, got {factor}"
    );
    // Catch an inverted ratio: embedding / lm_head would be ~1.4187.
    assert!(
        factor < 1.0,
        "tied-head factor must be lm/embedding, not the inverse"
    );
}

#[test]
fn falcon_h1_multipliers_default_to_one_when_absent() {
    // A config without the MUP multipliers (e.g. an upstream variant) must
    // default both to 1.0 so the tied head applies no spurious scale.
    let bare = FALCON_H1_TINY_CONFIG
        .replace("\"embedding_multiplier\": 0.11083984375,", "")
        .replace("\"lm_head_multiplier\": 0.078125,", "");
    let args: ModelArgs = serde_json::from_str(&bare).expect("parse config without multipliers");
    assert_eq!(args.embedding_multiplier, 1.0);
    assert_eq!(args.lm_head_multiplier, 1.0);
    assert_eq!(args.tied_head_factor(), 1.0);
}

#[test]
fn falcon_h1_mamba_rms_norm_defaults_false_and_parses_true() {
    // Default (and the Tiny checkpoint): SwiGLU gating, no mamba norm weight.
    let args: ModelArgs = serde_json::from_str(FALCON_H1_TINY_CONFIG).expect("parse config");
    assert!(!args.mamba_rms_norm);

    // When absent entirely it must still default to false.
    let no_flag = FALCON_H1_TINY_CONFIG.replace("\"mamba_rms_norm\": false,", "");
    let args2: ModelArgs = serde_json::from_str(&no_flag).expect("parse config without rms flag");
    assert!(!args2.mamba_rms_norm);

    // The gated-RMSNorm path is opt-in via the flag.
    let on =
        FALCON_H1_TINY_CONFIG.replace("\"mamba_rms_norm\": false,", "\"mamba_rms_norm\": true,");
    let args3: ModelArgs = serde_json::from_str(&on).expect("parse config with rms flag");
    assert!(args3.mamba_rms_norm);
}

#[test]
fn falcon_h1_eos_token_id_handles_array_scalar_and_missing() {
    // Array form (the Tiny checkpoint ships `[228, 11]`).
    let array: ModelArgs = serde_json::from_str(FALCON_H1_TINY_CONFIG).expect("parse config");
    assert_eq!(array.eos_token_ids(), vec![228, 11]);

    // Scalar form.
    let scalar_cfg =
        FALCON_H1_TINY_CONFIG.replace("\"eos_token_id\": [228, 11]", "\"eos_token_id\": 11");
    let scalar: ModelArgs = serde_json::from_str(&scalar_cfg).expect("parse scalar eos config");
    assert_eq!(scalar.eos_token_ids(), vec![11]);

    // Missing → default 11.
    let no_eos = FALCON_H1_TINY_CONFIG.replace("\"eos_token_id\": [228, 11],", "");
    let missing: ModelArgs = serde_json::from_str(&no_eos).expect("parse no-eos config");
    assert_eq!(missing.eos_token_ids(), vec![11]);
}

#[test]
fn falcon_h1_unquantized_config_uses_quantization_defaults() {
    // Drop the quantization block (a bf16 checkpoint) and confirm the defaults.
    let no_quant = FALCON_H1_TINY_CONFIG.replace(
        ",\n    \"quantization\": { \"group_size\": 64, \"bits\": 4 }",
        "",
    );
    let args: ModelArgs = serde_json::from_str(&no_quant).expect("parse unquantized config");
    assert!(args.quantization.is_none());
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}
