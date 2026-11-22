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

//! Arcee dense config screening, the squared-ReLU MLP and the shared model
//! contracts. Checkpoint-free: the config fixture is the published
//! `arcee-ai/AFM-4.5B` `config.json`.

use serde_json::{Value, json};

use super::{ArceeConfig, ArceeMLP, ArceeModel, parse_config, transformers_yarn_attention_factor};
use crate::models::rope_utils::RopeScalingKind;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;

/// `arcee-ai/AFM-4.5B` `config.json`.
fn afm_4_5b() -> Value {
    json!({
        "architectures": ["ArceeForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128003,
        "head_dim": 128,
        "hidden_act": "relu2",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 18432,
        "max_position_embeddings": 65536,
        "mlp_bias": false,
        "model_type": "arcee",
        "num_attention_heads": 20,
        "num_hidden_layers": 36,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 20.0,
            "mscale": 1.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "type": "yarn"
        },
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "vocab_size": 128005
    })
}

fn parse(value: &Value) -> Result<ArceeConfig, String> {
    parse_config(&value.to_string())
}

fn refusal(value: &Value) -> String {
    parse(value).expect_err("config should be refused")
}

#[test]
fn afm_4_5b_config_translates_to_llama3_args() {
    let config = parse(&afm_4_5b()).unwrap();
    let args = config.to_llama3_args();
    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.num_kv_heads(), 4);
    assert_eq!(args.intermediate_size, 18432);
    assert!(!args.rope_traditional);
    assert_eq!(args.eos_token_ids(), vec![128003]);
    let spec = args.rope_scaling.as_ref().unwrap();
    assert_eq!(spec.rope_type(), "yarn");
    assert_eq!(spec.factor, Some(20.0));
    assert_eq!(spec.original_max_position_embeddings, Some(4096.0));
}

#[test]
fn afm_4_5b_yarn_matches_transformers_attention_factor() {
    // transformers: mscale set, mscale_all_dim absent -> get_mscale(20) =
    // 0.1 * ln(20) + 1 = 1.29957; the shared path must apply the same.
    let config = parse(&afm_4_5b()).unwrap();
    let RopeScalingKind::Yarn { freqs, mscale } = config.to_llama3_args().rope_scaling_kind()
    else {
        panic!("AFM-4.5B must resolve to the YaRN table");
    };
    let expected = 0.1 * 20f32.ln() + 1.0;
    assert!((mscale - expected).abs() < 1e-6, "mscale {mscale}");
    assert!(
        (transformers_yarn_attention_factor(20.0, None, Some(1.0), None) - expected).abs() < 1e-6
    );

    // transformers' inv_freq for this config at the first (extrapolated) and
    // last (interpolated) pairs: 1 and 1 / (20 * 10000^(126/128)). The table
    // holds the divisors `fast_rope_with_freqs` takes.
    let table = array_to_vec_f32(&freqs);
    assert_eq!(table.len(), 64);
    assert!((table[0] - 1.0).abs() < 1e-6, "first {}", table[0]);
    let last = 20.0 * 10000f32.powf(126.0 / 128.0);
    assert!((table[63] / last - 1.0).abs() < 1e-5, "last {}", table[63]);
}

#[test]
fn transformers_attention_factor_resolution() {
    // A declared attention_factor wins.
    assert_eq!(
        transformers_yarn_attention_factor(16.0, Some(1.5), Some(1.0), Some(1.0)),
        1.5
    );
    // Both mscale and mscale_all_dim non-zero: their ratio.
    let ratio = transformers_yarn_attention_factor(40.0, None, Some(0.707), Some(0.707));
    assert!((ratio - 1.0).abs() < 1e-6);
    // mscale alone is ignored by transformers.
    let plain = transformers_yarn_attention_factor(40.0, None, Some(2.0), None);
    assert!((plain - (0.1 * 40f32.ln() + 1.0)).abs() < 1e-6);
}

#[test]
fn rope_parameters_block_is_lifted() {
    // transformers 5 writes RoPE under `rope_parameters`.
    let mut value = afm_4_5b();
    let block = value["rope_scaling"].clone();
    let map = value.as_object_mut().unwrap();
    map.remove("rope_scaling");
    map.remove("rope_theta");
    let mut block = block;
    block["rope_theta"] = json!(10000.0);
    map.insert("rope_parameters".into(), block);
    let config = parse(&value).unwrap();
    assert_eq!(config.rope_theta, 10000.0);
    assert_eq!(config.rope_scaling.as_ref().unwrap().factor, Some(20.0));
}

#[test]
fn non_relu2_activation_is_refused() {
    let mut value = afm_4_5b();
    value["hidden_act"] = json!("silu");
    assert!(refusal(&value).contains("hidden_act"));
}

#[test]
fn unimplemented_rope_type_is_refused() {
    for rope_type in ["dynamic", "longrope"] {
        let mut value = afm_4_5b();
        value["rope_scaling"] = json!({"rope_type": rope_type, "factor": 4.0});
        assert!(refusal(&value).contains(rope_type), "{rope_type}");
    }
}

#[test]
fn yarn_without_factor_is_refused() {
    let mut value = afm_4_5b();
    value["rope_scaling"]
        .as_object_mut()
        .unwrap()
        .remove("factor");
    assert!(refusal(&value).contains("factor"));
}

#[test]
fn yarn_truncate_false_is_refused() {
    let mut value = afm_4_5b();
    value["rope_scaling"]["truncate"] = json!(false);
    assert!(refusal(&value).contains("truncate"));
    value["rope_scaling"]["truncate"] = json!(true);
    parse(&value).unwrap();
}

#[test]
fn adapter_route_screens_rope_keys_too() {
    // The adapter path builds the config with plain serde (no parse_config),
    // so `validate` alone must still see `truncate`.
    let mut value = afm_4_5b();
    value["rope_scaling"]["truncate"] = json!(false);
    let config: ArceeConfig = serde_json::from_value(value).unwrap();
    assert!(config.validate().unwrap_err().contains("truncate"));
}

#[test]
fn other_transformers_rope_keys_are_screened() {
    let mut value = afm_4_5b();
    value["partial_rotary_factor"] = json!(0.5);
    assert!(refusal(&value).contains("partial_rotary_factor"));

    let mut value = afm_4_5b();
    value["original_max_position_embeddings"] = json!(8192);
    assert!(refusal(&value).contains("original_max_position_embeddings"));
    value["original_max_position_embeddings"] = json!(4096);
    parse(&value).unwrap();
    // Top level only: the shared path would use max_position_embeddings.
    value["rope_scaling"]
        .as_object_mut()
        .unwrap()
        .remove("original_max_position_embeddings");
    assert!(refusal(&value).contains("original_max_position_embeddings"));

    let mut value = afm_4_5b();
    value["rope_scaling"]["type"] = json!("linear");
    assert!(refusal(&value).contains("rope_type"));

    let mut value = afm_4_5b();
    value["rope_scaling"]["beta_fast"] = json!(0.5);
    assert!(refusal(&value).contains("beta_fast"));

    let mut value = afm_4_5b();
    value["rope_scaling"] = json!({"rope_type": "linear"});
    assert!(refusal(&value).contains("factor"));
}

#[test]
fn yarn_attention_factor_mismatch_is_refused() {
    // mscale 2 without mscale_all_dim: transformers ignores it, the shared
    // path would scale by get_mscale(20, 2).
    let mut value = afm_4_5b();
    value["rope_scaling"]["mscale"] = json!(2.0);
    assert!(refusal(&value).contains("attention factor"));

    let mut value = afm_4_5b();
    value["rope_scaling"]["attention_factor"] = json!(1.0);
    assert!(refusal(&value).contains("attention factor"));
}

#[test]
fn bad_geometry_is_refused() {
    for (key, bad) in [
        ("num_attention_heads", json!(0)),
        ("num_key_value_heads", json!(3)),
        ("head_dim", json!(7)),
        ("num_hidden_layers", json!(1_000_000)),
        ("rms_norm_eps", json!(0.0)),
        ("rope_theta", json!(-1.0)),
    ] {
        let mut value = afm_4_5b();
        value[key] = bad;
        assert!(refusal(&value).contains(key), "{key}");
    }
}

fn put(w: &mut WeightMap, key: &str, shape: &[i32], seed: f32) {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| seed + ((i * 7) % 13) as f32 * 0.01 - 0.06)
        .collect();
    w.insert(key.into(), mlxcel_core::from_slice_f32(&data, shape));
}

#[test]
fn mlp_is_gate_free_squared_relu() {
    // up: [3, 2], down: [2, 3]. x = [1, -2].
    let mut w = WeightMap::new();
    let up = [1.0f32, 0.5, -1.0, 0.25, 2.0, 1.0];
    let down = [1.0f32, 0.0, 2.0, 0.0, 1.0, -1.0];
    w.insert(
        "mlp.up_proj.weight".into(),
        mlxcel_core::from_slice_f32(&up, &[3, 2]),
    );
    w.insert(
        "mlp.down_proj.weight".into(),
        mlxcel_core::from_slice_f32(&down, &[2, 3]),
    );
    let args = parse(&afm_4_5b()).unwrap().to_llama3_args();
    let mlp = ArceeMLP::from_weights(&w, &args, "mlp").unwrap();
    let x = mlxcel_core::from_slice_f32(&[1.0, -2.0], &[1, 1, 2]);
    let out = array_to_vec_f32(&mlp.forward(&x));

    // up(x) = [0, -1.5, 0]; relu^2 = [0, 0, 0] would hide the square, so
    // use a second input with mixed signs too.
    assert_eq!(out, vec![0.0, 0.0]);
    let x = mlxcel_core::from_slice_f32(&[2.0, 1.0], &[1, 1, 2]);
    let out = array_to_vec_f32(&mlp.forward(&x));
    // up(x) = [2.5, -1.75, 5]; relu^2 = [6.25, 0, 25].
    // down rows: [1, 0, 2] -> 56.25; [0, 1, -1] -> -25. Plain ReLU would give
    // [12.5, -5].
    assert_eq!(out, vec![56.25, -25.0]);
}

#[test]
fn gate_proj_is_refused() {
    let mut w = WeightMap::new();
    put(&mut w, "mlp.up_proj.weight", &[3, 2], 0.1);
    put(&mut w, "mlp.down_proj.weight", &[2, 3], 0.1);
    put(&mut w, "mlp.gate_proj.weight", &[3, 2], 0.1);
    let args = parse(&afm_4_5b()).unwrap().to_llama3_args();
    let err = ArceeMLP::from_weights(&w, &args, "mlp").err().unwrap();
    assert!(err.contains("gate_proj"));
}

/// Two layers, hidden 8, 2 heads over 1 KV head of width 4, YaRN RoPE.
fn tiny_model() -> ArceeModel {
    let config = parse(&json!({
        "model_type": "arcee",
        "hidden_act": "relu2",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "mscale": 1.0,
                         "original_max_position_embeddings": 16},
    }))
    .unwrap();
    let mut w = WeightMap::new();
    put(&mut w, "model.embed_tokens.weight", &[32, 8], 0.0);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        put(
            &mut w,
            &format!("{p}.self_attn.q_proj.weight"),
            &[8, 8],
            0.01,
        );
        put(
            &mut w,
            &format!("{p}.self_attn.k_proj.weight"),
            &[4, 8],
            0.02,
        );
        put(
            &mut w,
            &format!("{p}.self_attn.v_proj.weight"),
            &[4, 8],
            -0.01,
        );
        put(
            &mut w,
            &format!("{p}.self_attn.o_proj.weight"),
            &[8, 8],
            0.01,
        );
        put(&mut w, &format!("{p}.mlp.up_proj.weight"), &[12, 8], 0.03);
        put(
            &mut w,
            &format!("{p}.mlp.down_proj.weight"),
            &[8, 12],
            -0.01,
        );
        put(&mut w, &format!("{p}.input_layernorm.weight"), &[8], 1.0);
        put(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[8],
            1.0,
        );
    }
    put(&mut w, "model.norm.weight", &[8], 1.0);
    put(&mut w, "lm_head.weight", &[32, 8], 0.02);
    ArceeModel::from_weights(&w, &config).unwrap()
}

crate::models::last_row_logits_harness::last_row_case!(arcee_last_row_logits, tiny_model);

#[test]
fn native_batched_decode_matches_three_unequal_rows() {
    let model = tiny_model();
    let prompts: &[&[i32]] = &[&[1], &[2, 3], &[4, 5, 6]];
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        prompts,
        &[7, 8, 9],
        1e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        &model,
        &[1, 2],
        3,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}
