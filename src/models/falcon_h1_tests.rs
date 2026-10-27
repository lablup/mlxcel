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
//! `Falcon-H1-Tiny-90M-Instruct-4bit` checkpoint. Tiny deterministic fixtures also exercise native batched decoding.

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

fn batching_model() -> super::falcon_h1::FalconH1Model {
    use crate::models::embedding_test_support::Rng;
    use mlxcel_core::{dtype, weights::WeightMap};
    let config = serde_json::from_value(
        serde_json::json!({ "model_type":"falcon_h1", "vocab_size":32, "hidden_size":8,
        "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":1,
        "head_dim":4, "mamba_d_ssm":8, "mamba_n_heads":2, "mamba_d_head":4,
        "mamba_d_state":32, "mamba_n_groups":1, "mamba_d_conv":3,
        "mamba_rms_norm":true, "tie_word_embeddings":true,
        "embedding_multiplier":2.0, "lm_head_multiplier":3.0 }),
    )
    .unwrap();
    let mut weights = WeightMap::new();
    let mut rng = Rng::new(2571);
    rng.insert(&mut weights, "model.embed_tokens.weight", &[32, 8], 0.2);
    weights.insert(
        "model.final_layernorm.weight".into(),
        mlxcel_core::ones(&[8], dtype::FLOAT32),
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        for key in [
            "input_layernorm.weight",
            "pre_ff_layernorm.weight",
            "mamba.norm.weight",
        ] {
            weights.insert(
                format!("{prefix}.{key}"),
                mlxcel_core::ones(&[8], dtype::FLOAT32),
            );
        }
        for (key, shape) in [
            ("mamba.in_proj.weight", vec![82, 8]),
            ("mamba.out_proj.weight", vec![8, 8]),
            ("mamba.conv1d.weight", vec![72, 3, 1]),
            ("self_attn.q_proj.weight", vec![8, 8]),
            ("self_attn.k_proj.weight", vec![4, 8]),
            ("self_attn.v_proj.weight", vec![4, 8]),
            ("self_attn.o_proj.weight", vec![8, 8]),
            ("feed_forward.gate_proj.weight", vec![16, 8]),
            ("feed_forward.up_proj.weight", vec![16, 8]),
            ("feed_forward.down_proj.weight", vec![8, 16]),
        ] {
            rng.insert(&mut weights, &format!("{prefix}.{key}"), &shape, 0.1);
        }
        for key in ["dt_bias", "A_log"] {
            weights.insert(
                format!("{prefix}.mamba.{key}"),
                mlxcel_core::zeros(&[2], dtype::FLOAT32),
            );
        }
        weights.insert(
            format!("{prefix}.mamba.D"),
            mlxcel_core::ones(&[2], dtype::FLOAT32),
        );
    }
    super::falcon_h1::FalconH1Model::from_weights(config, weights).unwrap()
}

#[test]
fn falcon_h1_model_owned_batched_multistep() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::batched_decode_harness::assert_model_owned_multistep_with_paged_lengths(
        batching_model,
        &[3, 3],
    );
}

#[test]
fn falcon_h1_fresh_and_mixed_recurrent_states_match_solo_rows() {
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::LanguageModel;
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    for mixed in [false, true] {
        let batched = batching_model();
        let solo = [batching_model(), batching_model()];
        let ids = [SequenceId::from_raw(1001), SequenceId::from_raw(1002)];
        let mut caches: Vec<_> = ids
            .iter()
            .map(|id| {
                batched.prepare_sequence_state(*id);
                LanguageModel::make_caches(&batched)
            })
            .collect();
        let mut single: Vec<_> = solo
            .iter()
            .zip(ids)
            .map(|(m, id)| {
                m.prepare_sequence_state(id);
                LanguageModel::make_caches(m)
            })
            .collect();
        if mixed {
            let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
            let _ = batched.forward_with_sequence_id(&prompt, Some(ids[0]), &mut caches[0], None);
            let _ = solo[0].forward_with_sequence_id(&prompt, Some(ids[0]), &mut single[0], None);
        }
        for step in 0..3 {
            let tokens = [2 + step, 3 + step];
            let input = mlxcel_core::from_slice_i32(&tokens, &[2, 1]);
            let mut refs: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
            let actual = batched.forward_batched_with_ids(&input, Some(&ids), &mut refs, None);
            for row in 0..2 {
                let input = mlxcel_core::from_slice_i32(&[tokens[row]], &[1, 1]);
                let expected = solo[row].forward_with_sequence_id(
                    &input,
                    Some(ids[row]),
                    &mut single[row],
                    None,
                );
                let actual = mlxcel_core::utils::slice_axis(&actual, 0, row as i32, row as i32 + 1);
                let delta = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(
                    &actual, &expected,
                )));
                assert!(mlxcel_core::item_f32(&delta) <= 1e-5);
            }
        }
    }
}
