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

//! Unit tests for the Jina VLM text config and decoder.

use super::{JinaVlmTextConfig, JinaVlmTextModel, Quantization};
use mlxcel_core::MlxArray;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

/// The `text_config` block exactly as `jinaai/jina-vlm-mlx` ships it, trimmed
/// to the keys the parser reads.
fn released_text_config() -> serde_json::Value {
    json!({
        "additional_vocab_size": 128,
        "block_config": {
            "attn_config": {
                "head_dim": 128,
                "k_lnorm": true,
                "lnorm_config": { "bias": false, "eps": 1e-06, "type": "rms", "with_affine": true },
                "n_heads": 16,
                "n_kv_heads": 8,
                "q_lnorm": true,
                "qkv_lnorm_on_heads": true,
                "sliding_window": -1
            },
            "ffn_config": {
                "activation_type": "silu",
                "gated_activation": true,
                "ratio": 4,
                "size": 6144
            },
            "lnorm_config": { "bias": false, "eps": 1e-06, "type": "rms", "with_affine": true }
        },
        "embedding_size": 151936,
        "hidden_size": 2048,
        "max_sequence_length": 40960,
        "model_type": "jvlm",
        "n_layers": 28,
        "num_hidden_layers": 28,
        "partial_rotary_factor": 1.0,
        "rope": true,
        "rope_scaling": null,
        "rope_theta": 1000000,
        "tie_word_embeddings": false,
        "vocab_size": 151936
    })
}

#[test]
fn the_nested_text_config_flattens_to_the_released_shapes() {
    let config = JinaVlmTextConfig::from_json(&released_text_config());

    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 151936);
    assert_eq!(config.additional_vocab_size, 128);
    assert_eq!(config.intermediate_size, 6144);
    assert_eq!(config.rope_theta, 1_000_000.0);
    assert_eq!(config.rms_norm_eps, 1e-6);
    assert!(config.use_qk_norm);
    assert!(!config.tie_word_embeddings);

    // The fused QKV width the loader must be able to split.
    assert_eq!(
        (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim,
        4096
    );
    // The fused gate/up width.
    assert_eq!(2 * config.intermediate_size, 12288);
}

#[test]
fn head_counts_come_from_the_attn_sub_block_not_the_top_level() {
    // A config that puts plausible-looking values at the top level must not
    // shadow the nested ones; reading the wrong level silently reshapes QKV.
    let mut value = released_text_config();
    value["n_heads"] = json!(99);
    value["head_dim"] = json!(99);
    let config = JinaVlmTextConfig::from_json(&value);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.head_dim, 128);
}

#[test]
fn an_empty_config_falls_back_to_the_released_defaults() {
    let config = JinaVlmTextConfig::from_json(&json!({}));
    let default = JinaVlmTextConfig::default();
    assert_eq!(config.hidden_size, default.hidden_size);
    assert_eq!(config.num_hidden_layers, default.num_hidden_layers);
    assert_eq!(config.num_attention_heads, default.num_attention_heads);
    assert_eq!(config.num_key_value_heads, default.num_key_value_heads);
    assert_eq!(config.head_dim, default.head_dim);
    assert_eq!(config.intermediate_size, default.intermediate_size);
    assert_eq!(config.vocab_size, default.vocab_size);
    assert_eq!(config.additional_vocab_size, default.additional_vocab_size);
    assert_eq!(config.rope_theta, default.rope_theta);
    assert_eq!(config.use_qk_norm, default.use_qk_norm);
}

#[test]
fn quantization_defaults_to_the_released_four_bit_pair() {
    let mut config = JinaVlmTextConfig::from_json(&released_text_config());
    assert_eq!((config.group_size(), config.bits()), (64, 4));
    config.quantization = Some(Quantization {
        group_size: 32,
        bits: 8,
    });
    assert_eq!((config.group_size(), config.bits()), (32, 8));
}

// Synthetic decoder.
//
// Small enough to run anywhere, wide enough that a transposed QKV split or a
// swapped SwiGLU half changes the output.
fn tiny_config() -> JinaVlmTextConfig {
    JinaVlmTextConfig {
        hidden_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 12,
        additional_vocab_size: 4,
        intermediate_size: 6,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        use_qk_norm: true,
        tie_word_embeddings: false,
        quantization: None,
    }
}

fn deterministic(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.2)
        .collect()
}

fn insert(wm: &mut WeightMap, key: &str, data: Vec<f32>, shape: &[i32]) {
    wm.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

pub(crate) fn tiny_text_weights(config: &JinaVlmTextConfig, prefix: &str) -> WeightMap {
    let mut wm = WeightMap::new();
    let hidden = config.hidden_size as i32;
    let head_dim = config.head_dim as i32;
    let qkv = (config.num_attention_heads + 2 * config.num_key_value_heads) as i32 * head_dim;
    let inter = config.intermediate_size as i32;

    insert(
        &mut wm,
        &format!("{prefix}.embedding.embedding"),
        deterministic(config.vocab_size * config.hidden_size, 0.1),
        &[config.vocab_size as i32, hidden],
    );
    insert(
        &mut wm,
        &format!("{prefix}.embedding.new_embedding"),
        deterministic(config.additional_vocab_size * config.hidden_size, 0.2),
        &[config.additional_vocab_size as i32, hidden],
    );

    for layer in 0..config.num_hidden_layers {
        let p = format!("{prefix}.layers.{layer}");
        insert(
            &mut wm,
            &format!("{p}.attn.qkv.weight"),
            deterministic((qkv * hidden) as usize, layer as f32 + 0.3),
            &[qkv, hidden],
        );
        insert(
            &mut wm,
            &format!("{p}.attn.out.weight"),
            deterministic((hidden * hidden) as usize, layer as f32 + 0.4),
            &[hidden, hidden],
        );
        insert(
            &mut wm,
            &format!("{p}.attn.q_norm.weight"),
            vec![1.0; head_dim as usize],
            &[head_dim],
        );
        insert(
            &mut wm,
            &format!("{p}.attn.k_norm.weight"),
            vec![1.0; head_dim as usize],
            &[head_dim],
        );
        insert(
            &mut wm,
            &format!("{p}.attn_norm.weight"),
            vec![1.0; hidden as usize],
            &[hidden],
        );
        insert(
            &mut wm,
            &format!("{p}.ffn.gate_up.weight"),
            deterministic((2 * inter * hidden) as usize, layer as f32 + 0.5),
            &[2 * inter, hidden],
        );
        insert(
            &mut wm,
            &format!("{p}.ffn.down.weight"),
            deterministic((hidden * inter) as usize, layer as f32 + 0.6),
            &[hidden, inter],
        );
        insert(
            &mut wm,
            &format!("{p}.ffn_norm.weight"),
            vec![1.0; hidden as usize],
            &[hidden],
        );
    }

    insert(
        &mut wm,
        &format!("{prefix}.ln_f.weight"),
        vec![1.0; hidden as usize],
        &[hidden],
    );
    insert(
        &mut wm,
        &format!("{prefix}.lm_head.weight"),
        deterministic(config.vocab_size * config.hidden_size, 0.7),
        &[config.vocab_size as i32, hidden],
    );
    wm
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn row(logits: &MlxArray, position: i32, width: i32) -> Vec<f32> {
    let flat = mlxcel_core::reshape(logits, &[-1, width]);
    let slice = mlxcel_core::slice(&flat, &[position, 0], &[position + 1, width]);
    to_vec_f32(&slice)
}

#[test]
fn the_extension_table_is_addressable_past_the_base_vocabulary() {
    let config = tiny_config();
    let weights = tiny_text_weights(&config, "language_model");
    let model =
        JinaVlmTextModel::from_weights(&weights, &config, "language_model", vec![0]).unwrap();

    // vocab_size = 12, additional = 4: id 13 lives in the extension table and
    // must resolve to `new_embedding[1]`.
    let ids = mlxcel_core::from_slice_i32(&[13], &[1, 1]);
    let embedded = model.embedding.forward(&ids);
    let got = to_vec_f32(&embedded);

    let expected: Vec<f32> = deterministic(config.additional_vocab_size * config.hidden_size, 0.2)
        [config.hidden_size..2 * config.hidden_size]
        .to_vec();
    for (a, b) in got.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "got {got:?} expected {expected:?}");
    }
}

#[test]
fn incremental_decode_matches_a_single_full_sequence_pass() {
    // Catches a wrong RoPE offset or a QK-norm applied on the wrong axis: both
    // leave prefill self-consistent while breaking the cached decode step.
    let config = tiny_config();
    let weights = tiny_text_weights(&config, "language_model");
    let model =
        JinaVlmTextModel::from_weights(&weights, &config, "language_model", vec![0]).unwrap();
    let width = config.vocab_size as i32;

    let tokens = [3i32, 7, 1, 9, 13];

    let full_ids = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);
    let mut full_caches: Vec<KVCache> = model.make_caches();
    let full_logits = model.forward(&full_ids, &mut full_caches, None);
    let expected = row(&full_logits, tokens.len() as i32 - 1, width);

    let mut caches: Vec<KVCache> = model.make_caches();
    let prefill_ids = mlxcel_core::from_slice_i32(&tokens[..4], &[1, 4]);
    let _ = model.forward(&prefill_ids, &mut caches, None);
    let step_ids = mlxcel_core::from_slice_i32(&tokens[4..], &[1, 1]);
    let step_logits = model.forward(&step_ids, &mut caches, None);
    let got = row(&step_logits, 0, width);

    for (a, b) in got.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 2e-3,
            "decode diverged from prefill: got {got:?} expected {expected:?}"
        );
    }
}

#[test]
fn precomputed_embeddings_take_the_place_of_the_token_lookup() {
    // The VLM merge path feeds `forward_with_embeddings`; if the decoder ever
    // re-embedded `input_ids` instead, the image features would be dropped and
    // generation would still look fluent.
    let config = tiny_config();
    let weights = tiny_text_weights(&config, "language_model");
    let model =
        JinaVlmTextModel::from_weights(&weights, &config, "language_model", vec![0]).unwrap();
    let width = config.vocab_size as i32;

    let tokens = [3i32, 7, 1];
    let ids = mlxcel_core::from_slice_i32(&tokens, &[1, 3]);
    let other = mlxcel_core::from_slice_i32(&[5i32, 5, 5], &[1, 3]);

    let embeds = model.embedding.forward(&ids);
    let mut caches = model.make_caches();
    let via_embeds = model.forward_with_embeddings(&other, Some(&embeds), &mut caches, None);
    let mut caches = model.make_caches();
    let via_ids = model.forward(&ids, &mut caches, None);

    let a = row(&via_embeds, 2, width);
    let b = row(&via_ids, 2, width);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < 1e-4,
            "embeddings path diverged: {a:?} vs {b:?}"
        );
    }
}
