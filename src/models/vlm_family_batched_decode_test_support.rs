// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Nontrivial one-layer fixtures and runtime contracts for issue #2563.

use mlxcel_core::weights::WeightMap;
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::models::embedding_test_support::Rng;

const HIDDEN: i32 = 16;
const INTERMEDIATE: i32 = 32;
const HEAD_DIM: i32 = 8;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const Q_OUT: i32 = HEADS * HEAD_DIM;
const KV_OUT: i32 = KV_HEADS * HEAD_DIM;
const VOCAB: i32 = 32;

fn config<T: DeserializeOwned>(value: serde_json::Value) -> T {
    serde_json::from_value(value).expect("tiny VLM text config parses")
}

fn linear(rng: &mut Rng, weights: &mut WeightMap, key: &str, out: i32, input: i32) {
    rng.insert(weights, key, &[out, input], 0.18);
}

fn norm(rng: &mut Rng, weights: &mut WeightMap, key: &str, width: i32) {
    let values: Vec<f32> = rng
        .values(width as usize, 0.18)
        .into_iter()
        .map(|value| 0.9 + value)
        .collect();
    weights.insert(
        key.to_string(),
        mlxcel_core::from_slice_f32(&values, &[width]),
    );
}

fn embedding(rng: &mut Rng, weights: &mut WeightMap) {
    linear(rng, weights, "model.embed_tokens.weight", VOCAB, HIDDEN);
}

fn separate_attention(rng: &mut Rng, weights: &mut WeightMap, prefix: &str) {
    linear(
        rng,
        weights,
        &format!("{prefix}.q_proj.weight"),
        Q_OUT,
        HIDDEN,
    );
    linear(
        rng,
        weights,
        &format!("{prefix}.k_proj.weight"),
        KV_OUT,
        HIDDEN,
    );
    linear(
        rng,
        weights,
        &format!("{prefix}.v_proj.weight"),
        KV_OUT,
        HIDDEN,
    );
    linear(
        rng,
        weights,
        &format!("{prefix}.o_proj.weight"),
        HIDDEN,
        Q_OUT,
    );
}

fn gated_mlp(rng: &mut Rng, weights: &mut WeightMap, prefix: &str) {
    linear(
        rng,
        weights,
        &format!("{prefix}.gate_proj.weight"),
        INTERMEDIATE,
        HIDDEN,
    );
    linear(
        rng,
        weights,
        &format!("{prefix}.up_proj.weight"),
        INTERMEDIATE,
        HIDDEN,
    );
    linear(
        rng,
        weights,
        &format!("{prefix}.down_proj.weight"),
        HIDDEN,
        INTERMEDIATE,
    );
}

pub(crate) fn tiny_glm4v_model() -> crate::models::glm4v::Glm4vTextModel {
    let cfg = config(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
        "attention_bias": false,
        "tie_word_embeddings": false,
        "rope_scaling": { "mrope_section": [2, 1, 1] }
    }));
    let mut rng = Rng::new(0x2563_0001);
    let mut w = WeightMap::new();
    embedding(&mut rng, &mut w);
    linear(&mut rng, &mut w, "lm_head.weight", VOCAB, HIDDEN);
    norm(&mut rng, &mut w, "model.norm.weight", HIDDEN);
    let p = "model.layers.0";
    separate_attention(&mut rng, &mut w, &format!("{p}.self_attn"));
    linear(
        &mut rng,
        &mut w,
        &format!("{p}.mlp.gate_up_proj.weight"),
        2 * INTERMEDIATE,
        HIDDEN,
    );
    linear(
        &mut rng,
        &mut w,
        &format!("{p}.mlp.down_proj.weight"),
        HIDDEN,
        INTERMEDIATE,
    );
    for name in [
        "input_layernorm",
        "post_self_attn_layernorm",
        "post_attention_layernorm",
        "post_mlp_layernorm",
    ] {
        norm(&mut rng, &mut w, &format!("{p}.{name}.weight"), HIDDEN);
    }
    crate::models::glm4v::Glm4vTextModel::from_weights(&w, &cfg).expect("tiny GLM-4V loads")
}

pub(crate) fn tiny_glm4v_moe_model() -> crate::models::glm4v_moe::Glm4vMoeTextModel {
    let args = config(json!({
        "model_type": "glm4v_moe",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": 128,
        "moe_intermediate_size": 4,
        "num_attention_heads": HEADS,
        "num_hidden_layers": 1,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "norm_topk_prob": false,
        "first_k_dense_replace": 0,
        "attention_bias": false,
        "tie_word_embeddings": false
    }));
    let mut rng = Rng::new(0x2563_0002);
    let mut w = WeightMap::new();
    embedding(&mut rng, &mut w);
    linear(&mut rng, &mut w, "lm_head.weight", VOCAB, HIDDEN);
    norm(&mut rng, &mut w, "model.norm.weight", HIDDEN);
    let p = "model.layers.0";
    separate_attention(&mut rng, &mut w, &format!("{p}.self_attn"));
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.input_layernorm.weight"),
        HIDDEN,
    );
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.post_attention_layernorm.weight"),
        HIDDEN,
    );
    rng.insert(&mut w, &format!("{p}.mlp.gate.weight"), &[2, HIDDEN], 0.2);
    rng.insert(
        &mut w,
        &format!("{p}.mlp.gate.e_score_correction_bias"),
        &[2],
        0.04,
    );
    rng.insert(
        &mut w,
        &format!("{p}.mlp.switch_mlp.gate_up_proj.weight"),
        &[2, 8, HIDDEN],
        0.18,
    );
    rng.insert(
        &mut w,
        &format!("{p}.mlp.switch_mlp.down_proj.weight"),
        &[2, HIDDEN, 4],
        0.18,
    );
    crate::models::glm4v_moe::Glm4vMoeTextModel::from_weights(&w, &args, &[2, 1, 1], vec![2])
        .expect("tiny GLM-4V-MoE loads")
}

pub(crate) fn tiny_hunyuan_vl_model() -> crate::models::hunyuan_vl::HunyuanVlTextModel {
    let cfg = config(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "use_qk_norm": true,
        "attention_bias": false,
        "tie_word_embeddings": false,
        "rope_scaling": { "type": "xdrope", "alpha": 1.0, "xdrope_section": [1, 1, 1, 1] }
    }));
    let mut rng = Rng::new(0x2563_0003);
    let mut w = WeightMap::new();
    embedding(&mut rng, &mut w);
    linear(&mut rng, &mut w, "lm_head.weight", VOCAB, HIDDEN);
    norm(&mut rng, &mut w, "model.norm.weight", HIDDEN);
    let p = "model.layers.0";
    separate_attention(&mut rng, &mut w, &format!("{p}.self_attn"));
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.self_attn.query_layernorm.weight"),
        HEAD_DIM,
    );
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.self_attn.key_layernorm.weight"),
        HEAD_DIM,
    );
    gated_mlp(&mut rng, &mut w, &format!("{p}.mlp"));
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.input_layernorm.weight"),
        HIDDEN,
    );
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.post_attention_layernorm.weight"),
        HIDDEN,
    );
    crate::models::hunyuan_vl::HunyuanVlTextModel::from_weights(&w, &cfg)
        .expect("tiny Hunyuan-VL loads")
}

pub(crate) fn tiny_ernie45_moe_vl_model() -> crate::models::ernie4_5_moe_vl::Ernie45MoeVlTextModel {
    let cfg = config(json!({
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 1,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "moe_num_experts": [2, 2],
        "moe_intermediate_size": [4, 4],
        "moe_k": 1,
        "moe_layer_start_index": [0, 0],
        "moe_layer_end_index": [0, 0],
        "moe_layer_interval": 1,
        "moe_num_shared_experts": 0,
        "rope_scaling": { "mrope_section": [1, 1, 2] },
        "im_patch_id": 29
    }));
    let mut rng = Rng::new(0x2563_0004);
    let mut w = WeightMap::new();
    embedding(&mut rng, &mut w);
    linear(&mut rng, &mut w, "lm_head.weight", VOCAB, HIDDEN);
    norm(&mut rng, &mut w, "model.norm.weight", HIDDEN);
    let p = "model.layers.0";
    separate_attention(&mut rng, &mut w, &format!("{p}.self_attn"));
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.input_layernorm.weight"),
        HIDDEN,
    );
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.post_attention_layernorm.weight"),
        HIDDEN,
    );
    let mlp = format!("{p}.mlp");
    for suffix in ["", "_1"] {
        linear(
            &mut rng,
            &mut w,
            &format!("{mlp}.gate{suffix}.weight"),
            2,
            HIDDEN,
        );
        rng.insert(
            &mut w,
            &format!("{mlp}.e_score_correction_bias{suffix}"),
            &[2],
            0.04,
        );
        rng.insert(
            &mut w,
            &format!("{mlp}.switch_mlp{suffix}.gate_proj.weight"),
            &[2, 4, HIDDEN],
            0.18,
        );
        rng.insert(
            &mut w,
            &format!("{mlp}.switch_mlp{suffix}.up_proj.weight"),
            &[2, 4, HIDDEN],
            0.18,
        );
        rng.insert(
            &mut w,
            &format!("{mlp}.switch_mlp{suffix}.down_proj.weight"),
            &[2, HIDDEN, 4],
            0.18,
        );
    }
    crate::models::ernie4_5_moe_vl::Ernie45MoeVlTextModel::from_weights(&w, &cfg)
        .expect("tiny ERNIE-4.5-MoE-VL loads")
}

pub(crate) fn tiny_paddleocr_model() -> crate::models::PaddleOcrTextModel {
    let cfg = config(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "use_bias": false,
        "tie_word_embeddings": false,
        "rope_scaling": { "mrope_section": [2, 1, 1] }
    }));
    let mut rng = Rng::new(0x2563_0005);
    let mut w = WeightMap::new();
    embedding(&mut rng, &mut w);
    linear(&mut rng, &mut w, "lm_head.weight", VOCAB, HIDDEN);
    norm(&mut rng, &mut w, "model.norm.weight", HIDDEN);
    let p = "model.layers.0";
    separate_attention(&mut rng, &mut w, &format!("{p}.self_attn"));
    gated_mlp(&mut rng, &mut w, &format!("{p}.mlp"));
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.input_layernorm.weight"),
        HIDDEN,
    );
    norm(
        &mut rng,
        &mut w,
        &format!("{p}.post_attention_layernorm.weight"),
        HIDDEN,
    );
    crate::models::PaddleOcrTextModel::from_weights(&w, &cfg).expect("tiny PaddleOCR-VL loads")
}

fn tiny_cohere_compass_model_with_rope(
    rope_parameters: serde_json::Value,
) -> crate::models::CohereCompassTextModel {
    let cfg = config(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "norm_type": "rms_norm",
        "transformer_block_type": "parallel",
        "logit_scale": 0.25,
        "sliding_window": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": rope_parameters,
        "tie_word_embeddings": false,
        "attention_bias": false,
        "eos_token_id": 2
    }));
    let mut rng = Rng::new(0x2563_0006);
    let mut w = WeightMap::new();
    embedding(&mut rng, &mut w);
    linear(&mut rng, &mut w, "lm_head.weight", VOCAB, HIDDEN);
    norm(&mut rng, &mut w, "model.norm.weight", HIDDEN);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        separate_attention(&mut rng, &mut w, &format!("{p}.self_attn"));
        gated_mlp(&mut rng, &mut w, &format!("{p}.mlp"));
        norm(
            &mut rng,
            &mut w,
            &format!("{p}.input_layernorm.weight"),
            HIDDEN,
        );
    }
    crate::models::CohereCompassTextModel::from_weights(&w, &cfg)
        .expect("tiny Cohere Compass loads")
}

pub(crate) fn tiny_cohere_compass_model() -> crate::models::CohereCompassTextModel {
    tiny_cohere_compass_model_with_rope(json!({
        "sliding_attention": {
            "mrope_section": [1, 1, 2],
            "rope_type": "default",
            "rope_theta": 50000.0
        },
        "full_attention": null
    }))
}

pub(crate) fn tiny_cohere_compass_sliding_nope_model() -> crate::models::CohereCompassTextModel {
    tiny_cohere_compass_model_with_rope(json!({
        "sliding_attention": null,
        "full_attention": null
    }))
}

pub(crate) fn tiny_cohere_compass_full_mrope_model() -> crate::models::CohereCompassTextModel {
    tiny_cohere_compass_model_with_rope(json!({
        "sliding_attention": null,
        "full_attention": {
            "mrope_section": [1, 1, 2],
            "rope_type": "default",
            "rope_theta": 50000.0
        }
    }))
}
