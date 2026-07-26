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

//! Unit tests for `youtu_vl_lm`.
//!
//! Avoid coupling to a checkpoint on disk so the tests can run anywhere
//! `mlxcel_core` does (Linux/CUDA CI included).

use super::*;

fn minimal_config() -> YoutuTextConfig {
    YoutuTextConfig {
        model_type: "youtu_vl".to_string(),
        vocab_size: 32,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(4),
        kv_lora_rank: 16,
        q_lora_rank: 32,
        qk_rope_head_dim: 8,
        v_head_dim: 16,
        qk_nope_head_dim: 16,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 500_000.0,
        rope_scaling: None,
        rope_traditional: true,
        rope_interleave: true,
        tie_word_embeddings: true,
        attention_bias: false,
        mlp_bias: false,
        n_shared_experts: None,
        n_routed_experts: None,
        moe_intermediate_size: None,
        num_experts_per_tok: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
        norm_topk_prob: true,
        moe_layer_freq: 1,
        first_k_dense_replace: 0,
        quantization: None,
    }
}

fn tiny_diagnostic_config() -> YoutuTextConfig {
    YoutuTextConfig {
        hidden_size: 16,
        intermediate_size: 32,
        num_attention_heads: 2,
        num_key_value_heads: Some(2),
        kv_lora_rank: 4,
        q_lora_rank: 8,
        qk_rope_head_dim: 2,
        v_head_dim: 2,
        qk_nope_head_dim: 2,
        ..minimal_config()
    }
}

fn put_small(weights: &mut WeightMap, key: impl Into<String>, shape: &[i32], seed: i32) {
    let total = shape
        .iter()
        .try_fold(1usize, |total, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|dimension| total.checked_mul(dimension))
        })
        .unwrap();
    let values = (0..total)
        .map(|index| ((index as i32 * 7 + seed) % 29 - 14) as f32 * 0.002)
        .collect::<Vec<_>>();
    let array = mlxcel_core::from_slice_f32(&values, shape);
    weights.insert(key.into(), mlxcel_core::copy(&array));
}

fn build_tiny_diagnostic_weights(config: &YoutuTextConfig) -> WeightMap {
    let mut weights = WeightMap::new();
    put_small(
        &mut weights,
        "model.embed_tokens.weight",
        &[config.vocab_size as i32, config.hidden_size as i32],
        1,
    );
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        for (name, dimension) in [
            ("input_layernorm.weight", config.hidden_size),
            ("post_attention_layernorm.weight", config.hidden_size),
        ] {
            let values = mlxcel_core::ones(&[dimension as i32], mlxcel_core::dtype::FLOAT32);
            weights.insert(format!("{prefix}.{name}"), mlxcel_core::copy(&values));
        }
        let attention = format!("{prefix}.self_attn");
        put_small(
            &mut weights,
            format!("{attention}.q_a_proj.weight"),
            &[config.q_lora_rank as i32, config.hidden_size as i32],
            10 + layer as i32,
        );
        let norm = mlxcel_core::ones(&[config.q_lora_rank as i32], mlxcel_core::dtype::FLOAT32);
        weights.insert(
            format!("{attention}.q_a_layernorm.weight"),
            mlxcel_core::copy(&norm),
        );
        put_small(
            &mut weights,
            format!("{attention}.q_b_proj.weight"),
            &[
                (config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim))
                    as i32,
                config.q_lora_rank as i32,
            ],
            20 + layer as i32,
        );
        put_small(
            &mut weights,
            format!("{attention}.kv_a_proj_with_mqa.weight"),
            &[
                (config.kv_lora_rank + config.qk_rope_head_dim) as i32,
                config.hidden_size as i32,
            ],
            30 + layer as i32,
        );
        let norm = mlxcel_core::ones(&[config.kv_lora_rank as i32], mlxcel_core::dtype::FLOAT32);
        weights.insert(
            format!("{attention}.kv_a_layernorm.weight"),
            mlxcel_core::copy(&norm),
        );
        put_small(
            &mut weights,
            format!("{attention}.embed_q.weight"),
            &[
                config.num_attention_heads as i32,
                config.kv_lora_rank as i32,
                config.qk_nope_head_dim as i32,
            ],
            40 + layer as i32,
        );
        put_small(
            &mut weights,
            format!("{attention}.unembed_out.weight"),
            &[
                config.num_attention_heads as i32,
                config.v_head_dim as i32,
                config.kv_lora_rank as i32,
            ],
            50 + layer as i32,
        );
        put_small(
            &mut weights,
            format!("{attention}.o_proj.weight"),
            &[
                config.hidden_size as i32,
                (config.num_attention_heads * config.v_head_dim) as i32,
            ],
            60 + layer as i32,
        );
        let mlp = format!("{prefix}.mlp");
        put_small(
            &mut weights,
            format!("{mlp}.gate_proj.weight"),
            &[config.intermediate_size as i32, config.hidden_size as i32],
            70 + layer as i32,
        );
        put_small(
            &mut weights,
            format!("{mlp}.up_proj.weight"),
            &[config.intermediate_size as i32, config.hidden_size as i32],
            80 + layer as i32,
        );
        put_small(
            &mut weights,
            format!("{mlp}.down_proj.weight"),
            &[config.hidden_size as i32, config.intermediate_size as i32],
            90 + layer as i32,
        );
    }
    let norm = mlxcel_core::ones(&[config.hidden_size as i32], mlxcel_core::dtype::FLOAT32);
    weights.insert("model.norm.weight".to_string(), mlxcel_core::copy(&norm));
    weights
}

#[test]
fn config_defaults_match_upstream() {
    // Round-trip through serde_json with only the required fields set —
    // mirrors what the loader sees on a real `config.json`.
    let raw = serde_json::json!({
        "model_type": "youtu_vl",
        "vocab_size": 283386,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "qk_nope_head_dim": 128,
    });
    let config: YoutuTextConfig = serde_json::from_value(raw).unwrap();

    assert!(config.tie_word_embeddings);
    assert!(config.rope_traditional);
    assert!(config.rope_interleave);
    assert_eq!(config.rope_theta, 500_000.0);
    assert_eq!(config.max_position_embeddings, 32_768);
    assert!(config.n_routed_experts.is_none());
}

#[test]
fn sanitize_decomposes_kv_b_proj_per_head() {
    let config = minimal_config();
    let mut weights = WeightMap::new();

    // Build a fake non-quantized kv_b_proj weight per layer.
    let num_heads = config.num_attention_heads;
    let head_dim = config.qk_nope_head_dim + config.v_head_dim;
    let kv_lora_rank = config.kv_lora_rank;
    let total = num_heads * head_dim * kv_lora_rank;

    for layer_idx in 0..config.num_hidden_layers {
        let key = format!("model.layers.{}.self_attn.kv_b_proj.weight", layer_idx);
        let arange = mlxcel_core::arange_f32(0.0, total as f32, 1.0);
        let reshaped = mlxcel_core::reshape(
            &arange,
            &[(num_heads * head_dim) as i32, kv_lora_rank as i32],
        );
        weights.insert(key, mlxcel_core::copy(&reshaped));
    }

    // Tied lm_head should be dropped after sanitization.
    weights.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::zeros(&[1, 1], mlxcel_core::dtype::FLOAT32),
    );

    let sanitized = sanitize_text_weights(weights, &config).unwrap();

    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);
        assert!(!sanitized.contains_key(&format!("{}.kv_b_proj.weight", prefix)));

        let embed_q_key = format!("{}.embed_q.weight", prefix);
        let unembed_out_key = format!("{}.unembed_out.weight", prefix);
        assert!(sanitized.contains_key(&embed_q_key));
        assert!(sanitized.contains_key(&unembed_out_key));

        // Shapes: embed_q = [H, kv_rank, qk_nope], unembed_out = [H, v_head, kv_rank]
        let eq_shape = mlxcel_core::array_shape(sanitized.get(&embed_q_key).unwrap());
        assert_eq!(
            eq_shape,
            vec![
                num_heads as i32,
                kv_lora_rank as i32,
                config.qk_nope_head_dim as i32
            ]
        );
        let uo_shape = mlxcel_core::array_shape(sanitized.get(&unembed_out_key).unwrap());
        assert_eq!(
            uo_shape,
            vec![
                num_heads as i32,
                config.v_head_dim as i32,
                kv_lora_rank as i32
            ]
        );
    }

    assert!(!sanitized.contains_key("lm_head.weight"));
}

#[test]
fn kv_b_proj_decompose_returns_error_when_biases_missing() {
    // A quantized kv_b_proj (scales present) with no biases tensor must
    // produce a clear error rather than panic (M1 hardening).
    let config = minimal_config();
    let mut weights = WeightMap::new();

    // Insert a plausible scales tensor but deliberately omit biases.
    let layer_idx = 0;
    let key = format!("model.layers.{layer_idx}.self_attn.kv_b_proj.weight");
    let scales_key = format!("model.layers.{layer_idx}.self_attn.kv_b_proj.scales");

    // Minimal weight tensor: shape doesn't matter for the biases-missing path.
    weights.insert(
        key,
        mlxcel_core::zeros(&[1, 1], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        scales_key,
        mlxcel_core::zeros(&[1, 1], mlxcel_core::dtype::FLOAT32),
    );
    // biases are intentionally absent

    let result = sanitize_text_weights(weights, &config);
    assert!(
        result.is_err(),
        "expected Err when biases are missing for a quantized kv_b_proj"
    );
    let msg = result.err().unwrap();
    assert!(
        msg.contains("biases"),
        "error message should mention 'biases'; got: {msg}"
    );
    assert!(
        msg.contains(&format!("layer {layer_idx}")),
        "error message should identify the layer; got: {msg}"
    );
}

#[test]
fn tiny_dense_mla_diagnostics_cover_zero_nonzero_and_replaced_embeddings() {
    let config = tiny_diagnostic_config();
    let weights = build_tiny_diagnostic_weights(&config);
    let model = YoutuLanguageModel::from_weights(&weights, &config).unwrap();

    let token0 = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
    let position0 = model.capture_mla_diagnostics(&token0, None).unwrap();
    assert_eq!(position0.position, 0);
    assert_eq!(position0.layers, 2);
    assert_eq!(position0.logits.len(), config.vocab_size);
    assert_eq!(
        position0.latent_kv.len(),
        config.num_hidden_layers * config.kv_lora_rank
    );
    assert_eq!(
        position0.rotary_kv.len(),
        config.num_hidden_layers * config.qk_rope_head_dim
    );

    let tokens = mlxcel_core::from_slice_i32(&[1, 3], &[1, 2]);
    let position1 = model.capture_mla_diagnostics(&tokens, None).unwrap();
    assert_eq!(position1.position, 1);
    assert_ne!(position0.rotary_kv, position1.rotary_kv);

    let text_embeddings = model.get_embed_tokens(&tokens);
    let replacements = mlxcel_core::from_slice_f32(
        &(0..config.hidden_size)
            .map(|index| 0.25 + index as f32 * 0.01)
            .collect::<Vec<_>>(),
        &[1, 1, config.hidden_size as i32],
    );
    let prefix = mlxcel_core::slice(
        &text_embeddings,
        &[0, 0, 0],
        &[1, 1, config.hidden_size as i32],
    );
    let replaced = mlxcel_core::concatenate(&prefix, &replacements, 1);
    let image_position = model
        .capture_mla_diagnostics(&tokens, Some(&replaced))
        .unwrap();
    assert_eq!(image_position.position, 1);
    assert_ne!(position1.logits, image_position.logits);
    assert_ne!(position1.latent_kv, image_position.latent_kv);
}
