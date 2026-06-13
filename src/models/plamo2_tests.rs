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

//! Unit tests for PLaMo 2 config parsing and the config-derived geometry.
//!
//! These cover the checkpoint-free surface: that the `plamo2` `config.json`
//! deserializes into `ModelArgs`, that the Mamba inner dimensions and the
//! `dt_dim = max(64, hidden / 16)` derivation are correct, that `tie_word_embeddings`
//! defaults to true when absent, that the `is_mamba(i)` interleave pattern
//! matches the reference, that the EOS default is the `<|plamo:eos|>` id (2),
//! and that the normformer-style per-norm offsets keep their exact PLaMo 2
//! values. The numeric correctness of the interleaved SSM + attention hybrid is
//! validated end-to-end against the real `plamo-2-1b` checkpoint and is not
//! exercised here (no Metal device is assumed in unit tests).

use super::plamo2::{
    self, FINAL_NORM_OFFSET, ModelArgs, PRE_MIXER_NORM_OFFSET, PRE_MLP_NORM_OFFSET,
};

/// A trimmed `plamo-2-1b` config with the fields the loader reads. Values
/// mirror the real checkpoint.
const PLAMO2_1B_CONFIG: &str = r#"{
    "model_type": "plamo2",
    "hidden_size": 2048,
    "hidden_size_per_head": 128,
    "intermediate_size": 8192,
    "mamba_d_conv": 4,
    "mamba_d_state": 64,
    "mamba_enabled": true,
    "mamba_num_heads": 32,
    "mamba_step": 2,
    "num_attention_heads": 16,
    "num_hidden_layers": 16,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-06,
    "vocab_size": 100000,
    "eos_token_id": 2
}"#;

#[test]
fn plamo2_config_parses() {
    let args: ModelArgs = serde_json::from_str(PLAMO2_1B_CONFIG).expect("parse config");
    assert_eq!(args.model_type, "plamo2");
    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_hidden_layers, 16);
    assert_eq!(args.num_attention_heads, 16);
    assert_eq!(args.num_key_value_heads, 1);
    assert_eq!(args.vocab_size, 100000);
    assert_eq!(args.intermediate_size, 8192);
    assert_eq!(args.rms_norm_eps, 1e-6);
    // Unquantized checkpoint: quantization helpers fall back to defaults.
    assert!(args.quantization.is_none());
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
}

#[test]
fn plamo2_mamba_dims_parse() {
    let args: ModelArgs = serde_json::from_str(PLAMO2_1B_CONFIG).expect("parse config");
    assert_eq!(args.mamba_d_conv, 4);
    assert_eq!(args.mamba_d_state, 64);
    assert_eq!(args.mamba_num_heads, 32);
    assert_eq!(args.mamba_step, 2);
    assert!(args.mamba_enabled);
    // head_dim == hidden_size_per_head, shared by Mamba and attention.
    assert_eq!(args.head_dim(), 128);
    // Mamba inner width = mamba_num_heads * hidden_size_per_head = 32 * 128.
    assert_eq!(args.mamba_intermediate(), 4096);
    assert_eq!(
        args.mamba_intermediate(),
        args.mamba_num_heads * args.hidden_size_per_head
    );
}

#[test]
fn plamo2_dt_dim_is_max_64_hidden_over_16() {
    // dt_dim = max(64, hidden_size / 16). For hidden 2048 -> 128.
    let args: ModelArgs = serde_json::from_str(PLAMO2_1B_CONFIG).expect("parse config");
    assert_eq!(args.dt_dim(), 128);
    assert_eq!(args.dt_dim(), std::cmp::max(64, args.hidden_size / 16));

    // A small hidden size floors at 64 (e.g. hidden 512 -> 32 -> clamped to 64).
    let small = PLAMO2_1B_CONFIG.replace("\"hidden_size\": 2048,", "\"hidden_size\": 512,");
    let small_args: ModelArgs = serde_json::from_str(&small).expect("parse small config");
    assert_eq!(small_args.dt_dim(), 64);
}

#[test]
fn plamo2_tie_word_embeddings_defaults_true() {
    // The `plamo-2-1b` config has no `tie_word_embeddings`; it must default true.
    let args: ModelArgs = serde_json::from_str(PLAMO2_1B_CONFIG).expect("parse config");
    assert!(args.tie_word_embeddings);

    // An explicit `false` is honored.
    let untied = PLAMO2_1B_CONFIG.replace(
        "\"eos_token_id\": 2",
        "\"eos_token_id\": 2,\n    \"tie_word_embeddings\": false",
    );
    let untied_args: ModelArgs = serde_json::from_str(&untied).expect("parse untied config");
    assert!(!untied_args.tie_word_embeddings);
}

#[test]
fn plamo2_is_mamba_interleave_pattern() {
    // mamba_step == 2, mamba_step / 2 == 1, num_hidden_layers (16) > 1, so
    // is_mamba(i) = (i % 2) != 1, i.e. even layers are Mamba, odd are attention.
    let args: ModelArgs = serde_json::from_str(PLAMO2_1B_CONFIG).expect("parse config");
    for i in 0..args.num_hidden_layers {
        let expected_mamba = i % 2 == 0;
        assert_eq!(
            args.is_mamba(i),
            expected_mamba,
            "layer {i} mamba classification mismatch"
        );
    }
    // Layer 0 is Mamba (carries in_proj/bcdt), layer 1 is attention (qkv).
    assert!(args.is_mamba(0));
    assert!(!args.is_mamba(1));
}

#[test]
fn plamo2_is_mamba_disabled_uses_attention_everywhere() {
    let disabled =
        PLAMO2_1B_CONFIG.replace("\"mamba_enabled\": true,", "\"mamba_enabled\": false,");
    let args: ModelArgs = serde_json::from_str(&disabled).expect("parse mamba-disabled config");
    for i in 0..args.num_hidden_layers {
        assert!(
            !args.is_mamba(i),
            "layer {i} must be attention when mamba is disabled"
        );
    }
}

#[test]
fn plamo2_eos_token_id_handles_scalar_and_missing() {
    // Scalar form (the `plamo-2-1b` checkpoint ships eos_token_id == 2).
    let scalar: ModelArgs = serde_json::from_str(PLAMO2_1B_CONFIG).expect("parse config");
    assert_eq!(scalar.eos_token_ids(), vec![2]);

    // Missing -> default to the `<|plamo:eos|>` id (2).
    let no_eos = PLAMO2_1B_CONFIG.replace(",\n    \"eos_token_id\": 2", "");
    let missing: ModelArgs = serde_json::from_str(&no_eos).expect("parse no-eos config");
    assert_eq!(missing.eos_token_ids(), vec![2]);

    // Array form is flattened.
    let array_cfg = PLAMO2_1B_CONFIG.replace("\"eos_token_id\": 2", "\"eos_token_id\": [2, 4]");
    let array: ModelArgs = serde_json::from_str(&array_cfg).expect("parse array eos config");
    assert_eq!(array.eos_token_ids(), vec![2, 4]);
}

#[test]
fn plamo2_norm_offsets_match_reference() {
    // The PLaMo 2 normformer offsets, folded into each stored norm weight.
    assert_eq!(PRE_MIXER_NORM_OFFSET, 1.0);
    assert_eq!(PRE_MLP_NORM_OFFSET, 1.0);
    assert_eq!(FINAL_NORM_OFFSET, 1.0);

    // post_mixer_norm offset == 1/5 == 0.2.
    assert!((plamo2::post_mixer_norm_offset() - 0.2).abs() < 1e-7);

    // post_mlp_norm offset == 1/(5**1.5) ~= 0.0894427191.
    let expected_post_mlp = 1.0 / 5.0_f32.powf(1.5);
    assert!((plamo2::post_mlp_norm_offset() - expected_post_mlp).abs() < 1e-9);
    assert!((plamo2::post_mlp_norm_offset() - 0.089_442_72).abs() < 1e-6);
}
