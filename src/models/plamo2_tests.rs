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

//! Unit tests for PLaMo 2 config parsing and the config-derived geometry.
//!
//! These cover the checkpoint-free surface: that the `plamo2` `config.json`
//! deserializes into `ModelArgs`, that the Mamba inner dimensions and the
//! `dt_dim = max(64, hidden / 16)` derivation are correct, that `tie_word_embeddings`
//! defaults to true when absent, that the `is_mamba(i)` interleave pattern
//! matches the reference, that the EOS default is the `<|plamo:eos|>` id (2),
//! and that the normformer-style per-norm offsets keep their exact PLaMo 2
//! values. The numeric correctness of the interleaved SSM + attention hybrid is
//! validated end-to-end against the real `plamo-2-1b` checkpoint. Tiny deterministic fixtures also exercise native batched decoding.

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

fn batching_model() -> super::plamo2::Plamo2Model {
    use crate::models::embedding_test_support::Rng;
    use mlxcel_core::{dtype, weights::WeightMap};
    let config = serde_json::from_value(
        serde_json::json!({ "model_type":"plamo2", "vocab_size":32, "hidden_size":8,
        "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":1,
        "hidden_size_per_head":4, "intermediate_size":16, "mamba_num_heads":2,
        "mamba_d_state":32, "mamba_d_conv":3, "mamba_step":2,
        "tie_word_embeddings":true }),
    )
    .unwrap();
    let mut weights = WeightMap::new();
    let mut rng = Rng::new(2571);
    rng.insert(&mut weights, "model.embed_tokens.weight", &[32, 8], 0.2);
    weights.insert(
        "model.norm.weight".into(),
        mlxcel_core::zeros(&[8], dtype::FLOAT32),
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.layers.{layer}");
        for key in [
            "pre_mixer_norm",
            "post_mixer_norm",
            "pre_mlp_norm",
            "post_mlp_norm",
        ] {
            weights.insert(
                format!("{prefix}.{key}.weight"),
                mlxcel_core::zeros(&[8], dtype::FLOAT32),
            );
        }
        rng.insert(
            &mut weights,
            &format!("{prefix}.mlp.gate_up_proj.weight"),
            &[32, 8],
            0.1,
        );
        rng.insert(
            &mut weights,
            &format!("{prefix}.mlp.down_proj.weight"),
            &[8, 16],
            0.1,
        );
        if layer == 0 {
            for (key, shape) in [
                ("in_proj.weight", vec![16, 8]),
                ("out_proj.weight", vec![8, 8]),
                ("conv1d.weight", vec![8, 3, 1]),
                ("bcdt_proj.weight", vec![128, 8]),
                ("dt_proj.weight", vec![2, 64]),
            ] {
                rng.insert(&mut weights, &format!("{prefix}.mixer.{key}"), &shape, 0.1);
            }
            for (key, size) in [
                ("dt_norm_weight", 64),
                ("B_norm_weight", 32),
                ("C_norm_weight", 32),
                ("D", 2),
            ] {
                weights.insert(
                    format!("{prefix}.mixer.{key}"),
                    mlxcel_core::ones(&[size], dtype::FLOAT32),
                );
            }
            for key in ["dt_bias", "A_log"] {
                weights.insert(
                    format!("{prefix}.mixer.{key}"),
                    mlxcel_core::zeros(&[2], dtype::FLOAT32),
                );
            }
        } else {
            rng.insert(
                &mut weights,
                &format!("{prefix}.mixer.qkv_proj.weight"),
                &[16, 8],
                0.1,
            );
            rng.insert(
                &mut weights,
                &format!("{prefix}.mixer.o_proj.weight"),
                &[8, 8],
                0.1,
            );
            rng.insert(
                &mut weights,
                &format!("{prefix}.mixer.q_weight"),
                &[2, 4],
                0.5,
            );
            rng.insert(
                &mut weights,
                &format!("{prefix}.mixer.k_weight"),
                &[1, 4],
                0.5,
            );
        }
    }
    super::plamo2::Plamo2Model::from_weights(config, weights).unwrap()
}

#[test]
fn plamo2_model_owned_batched_multistep() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::batched_decode_harness::assert_model_owned_multistep_with_paged_lengths(
        batching_model,
        &[0, 3],
    );
}

#[test]
fn plamo2_fresh_and_mixed_recurrent_states_match_solo_rows() {
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
