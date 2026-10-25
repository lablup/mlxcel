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

//! Deterministic decoder fixtures shared by the issue #2564 tower and wrapper
//! runtime gates.
//!
//! Used by: `crate::models::wrapper_batched_decode_tests` and
//! `crate::vision::wrapper_native_batched_decode_tests`.

use mlxcel_core::weights::WeightMap;

use crate::models::embedding_test_support::Rng;

const HIDDEN: i32 = 8;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const HEAD_DIM: i32 = 4;
const INTERMEDIATE: i32 = 12;
pub(crate) const VOCAB: i32 = 24;
const EXTRA_VOCAB: i32 = 4;

fn linear(rng: &mut Rng, weights: &mut WeightMap, prefix: &str, output: i32, input: i32) {
    rng.insert(weights, &format!("{prefix}.weight"), &[output, input], 0.2);
}

fn norm(rng: &mut Rng, weights: &mut WeightMap, prefix: &str, width: i32, bias: bool) {
    let values: Vec<f32> = (0..width).map(|index| 0.7 + index as f32 * 0.08).collect();
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::from_slice_f32(&values, &[width]),
    );
    if bias {
        rng.insert(weights, &format!("{prefix}.bias"), &[width], 0.03);
    }
}

fn molmo2_weights(prefix: &str, seed: u64) -> WeightMap {
    let mut rng = Rng::new(seed);
    let mut weights = WeightMap::new();
    rng.insert(
        &mut weights,
        &format!("{prefix}.wte.embedding"),
        &[VOCAB, HIDDEN],
        0.4,
    );
    rng.insert(
        &mut weights,
        &format!("{prefix}.wte.new_embedding"),
        &[EXTRA_VOCAB, HIDDEN],
        0.4,
    );
    let block = format!("{prefix}.blocks.0");
    linear(
        &mut rng,
        &mut weights,
        &format!("{block}.self_attn.att_proj"),
        HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM,
        HIDDEN,
    );
    linear(
        &mut rng,
        &mut weights,
        &format!("{block}.self_attn.attn_out"),
        HIDDEN,
        HIDDEN,
    );
    norm(
        &mut rng,
        &mut weights,
        &format!("{block}.self_attn.q_norm"),
        HEAD_DIM,
        false,
    );
    norm(
        &mut rng,
        &mut weights,
        &format!("{block}.self_attn.k_norm"),
        HEAD_DIM,
        false,
    );
    linear(
        &mut rng,
        &mut weights,
        &format!("{block}.mlp.ff_proj"),
        2 * INTERMEDIATE,
        HIDDEN,
    );
    linear(
        &mut rng,
        &mut weights,
        &format!("{block}.mlp.ff_out"),
        HIDDEN,
        INTERMEDIATE,
    );
    norm(
        &mut rng,
        &mut weights,
        &format!("{block}.attn_norm"),
        HIDDEN,
        false,
    );
    norm(
        &mut rng,
        &mut weights,
        &format!("{block}.ff_norm"),
        HIDDEN,
        false,
    );
    norm(
        &mut rng,
        &mut weights,
        &format!("{prefix}.ln_f"),
        HIDDEN,
        false,
    );
    weights
}

fn molmo2_config(model_type: &str) -> crate::models::molmo2::Molmo2TextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": model_type,
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "additional_vocab_size": EXTRA_VOCAB,
        "rope_theta": 5000000.0,
        "layer_norm_eps": 1e-6
    }))
    .expect("tiny Molmo2 config parses")
}

pub(crate) fn tiny_molmo_model() -> crate::models::molmo::MolmoModel {
    let prefix = "language_model.model";
    let mut rng = Rng::new(0x2564_0101);
    let mut weights = molmo2_weights(prefix, 0x2564_0102);
    let block = format!("{prefix}.blocks.0");
    let old_prefix = format!("{block}.self_attn.");
    for suffix in ["att_proj.weight", "attn_out.weight"] {
        let value = weights
            .remove(&format!("{old_prefix}{suffix}"))
            .expect("Molmo2 attention weight exists");
        weights.insert(format!("{block}.{suffix}"), value);
    }
    for suffix in ["ff_proj.weight", "ff_out.weight"] {
        let value = weights
            .remove(&format!("{block}.mlp.{suffix}"))
            .expect("Molmo2 MLP weight exists");
        weights.insert(format!("{block}.{suffix}"), value);
    }
    for key in [
        format!("{block}.self_attn.q_norm.weight"),
        format!("{block}.self_attn.k_norm.weight"),
    ] {
        weights.remove(&key);
    }
    linear(
        &mut rng,
        &mut weights,
        &format!("{prefix}.ff_out"),
        VOCAB,
        HIDDEN,
    );
    let config: crate::models::molmo::MolmoTextConfig = serde_json::from_value(serde_json::json!({
        "model_type": "molmo",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "embedding_size": VOCAB,
        "additional_vocab_size": EXTRA_VOCAB,
        "layer_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "rope_impl": "llama"
    }))
    .expect("tiny Molmo config parses");
    crate::models::molmo::MolmoModel::from_weights(&weights, &config, prefix)
        .expect("tiny Molmo model builds")
}

pub(crate) fn tiny_molmo2_model() -> crate::models::molmo2::Molmo2Model {
    let prefix = "language_model.model";
    let mut rng = Rng::new(0x2564_0201);
    let mut weights = molmo2_weights(prefix, 0x2564_0202);
    linear(
        &mut rng,
        &mut weights,
        "language_model.lm_head",
        VOCAB,
        HIDDEN,
    );
    crate::models::molmo2::Molmo2Model::from_weights(&weights, &molmo2_config("molmo2"), prefix)
        .expect("tiny Molmo2 model builds")
}

pub(crate) fn tiny_molmo_point_language_model()
-> crate::models::molmo_point::MolmoPointLanguageModel {
    let prefix = "language_model";
    let mut rng = Rng::new(0x2564_0301);
    let mut weights = molmo2_weights(&format!("{prefix}.model"), 0x2564_0302);
    rng.insert(
        &mut weights,
        &format!("{prefix}.lm_head.output_embeddings"),
        &[VOCAB, HIDDEN],
        0.25,
    );
    rng.insert(
        &mut weights,
        &format!("{prefix}.lm_head.new_output_embeddings"),
        &[EXTRA_VOCAB, HIDDEN],
        0.25,
    );
    crate::models::molmo_point::MolmoPointLanguageModel::from_weights(
        &weights,
        &molmo2_config("molmo_point"),
        prefix,
    )
    .expect("tiny Molmo-Point language model builds")
}

fn moondream_weights(model3: bool, disjoint_router: bool) -> (WeightMap, serde_json::Value) {
    let mut rng = Rng::new(if model3 { 0x2564_0403 } else { 0x2564_0402 });
    let mut weights = WeightMap::new();
    if model3 && disjoint_router {
        let mut embeddings = rng.values((VOCAB * HIDDEN) as usize, 0.4);
        for (token, first) in [(13_i32, 3.0_f32), (14, -3.0), (15, 3.0)] {
            let start = (token * HIDDEN) as usize;
            embeddings[start..start + HIDDEN as usize].fill(0.0);
            embeddings[start] = first;
        }
        weights.insert(
            "text.wte.weight".into(),
            mlxcel_core::from_slice_f32(&embeddings, &[VOCAB, HIDDEN]),
        );
    } else {
        rng.insert(&mut weights, "text.wte.weight", &[VOCAB, HIDDEN], 0.4);
    }
    norm(&mut rng, &mut weights, "text.blocks.0.ln", HIDDEN, true);
    linear(
        &mut rng,
        &mut weights,
        "text.blocks.0.attn.qkv",
        3 * HIDDEN,
        HIDDEN,
    );
    linear(
        &mut rng,
        &mut weights,
        "text.blocks.0.attn.proj",
        HIDDEN,
        HIDDEN,
    );
    if model3 {
        rng.insert(
            &mut weights,
            "text.blocks.0.attn.tau.wq",
            &[HEADS, 3 * HIDDEN],
            0.1,
        );
        rng.insert(
            &mut weights,
            "text.blocks.0.attn.tau.wv",
            &[HEADS, 3 * HIDDEN],
            0.1,
        );
        rng.insert(&mut weights, "text.blocks.0.attn.tau.alpha", &[HEADS], 0.1);
        let router = if disjoint_router {
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        } else {
            vec![0.0; (2 * HIDDEN) as usize]
        };
        weights.insert(
            "text.blocks.0.mlp.router.weight".into(),
            mlxcel_core::from_slice_f32(&router, &[2, HIDDEN]),
        );
        rng.insert(
            &mut weights,
            "text.blocks.0.mlp.fc1.weight",
            &[2, 2 * 6, HIDDEN],
            0.2,
        );
        rng.insert(
            &mut weights,
            "text.blocks.0.mlp.fc2.weight",
            &[2, HIDDEN, 6],
            0.2,
        );
    } else {
        linear(
            &mut rng,
            &mut weights,
            "text.blocks.0.mlp.fc1",
            INTERMEDIATE,
            HIDDEN,
        );
        linear(
            &mut rng,
            &mut weights,
            "text.blocks.0.mlp.fc2",
            HIDDEN,
            INTERMEDIATE,
        );
    }
    norm(&mut rng, &mut weights, "text.post_ln", HIDDEN, true);
    linear(&mut rng, &mut weights, "text.lm_head", VOCAB, HIDDEN);
    let config = if model3 {
        serde_json::json!({
            "model_type": "moondream3", "dim": HIDDEN, "ff_dim": INTERMEDIATE,
            "n_layers": 1, "vocab_size": VOCAB, "n_heads": HEADS,
            "n_kv_heads": HEADS, "prefix_attn": 0, "group_size": 64,
            "bits": 4, "eos_token_id": 2,
            "moe": {"num_experts": 2, "start_layer": 0, "experts_per_token": 1,
                    "expert_inner_dim": 6, "expert_group_size": 64}
        })
    } else {
        serde_json::json!({
            "model_type": "moondream2", "dim": HIDDEN, "ff_dim": INTERMEDIATE,
            "n_layers": 1, "vocab_size": VOCAB, "n_heads": HEADS,
            "n_kv_heads": KV_HEADS, "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5, "layer_norm_eps": 1e-5,
            "group_size": 64, "bits": 4, "eos_token_id": 2, "bos_token_id": 1
        })
    };
    (weights, config)
}

pub(crate) fn tiny_moondream2_model() -> crate::models::moondream2::Moondream2Model {
    let (weights, value) = moondream_weights(false, false);
    let config = serde_json::from_value(value).expect("tiny Moondream2 config parses");
    crate::models::moondream2::Moondream2Model::from_weights(&weights, &config)
        .expect("tiny Moondream2 model builds")
}

pub(crate) fn tiny_moondream3_model() -> crate::models::moondream3::Moondream3Model {
    tiny_moondream3_model_with_router(false)
}

pub(crate) fn tiny_moondream3_model_with_router(
    disjoint_router: bool,
) -> crate::models::moondream3::Moondream3Model {
    let (weights, value) = moondream_weights(true, disjoint_router);
    let config = serde_json::from_value(value).expect("tiny Moondream3 config parses");
    crate::models::moondream3::Moondream3Model::from_weights(&weights, &config)
        .expect("tiny Moondream3 model builds")
}

pub(crate) fn tiny_jina_vlm_text_model() -> crate::models::jina_vlm::JinaVlmTextModel {
    let config = crate::models::jina_vlm::JinaVlmTextConfig {
        hidden_size: HIDDEN as usize,
        num_hidden_layers: 1,
        num_attention_heads: HEADS as usize,
        num_key_value_heads: KV_HEADS as usize,
        head_dim: HEAD_DIM as usize,
        vocab_size: VOCAB as usize,
        additional_vocab_size: EXTRA_VOCAB as usize,
        intermediate_size: INTERMEDIATE as usize,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        use_qk_norm: true,
        tie_word_embeddings: false,
        quantization: None,
    };
    let mut weights = crate::models::jina_vlm::tests::tiny_text_weights(&config, "language_model");
    let qk = mlxcel_core::from_slice_f32(&[0.7, 0.9, 1.1, 1.3], &[HEAD_DIM]);
    weights.insert(
        "language_model.layers.0.attn.q_norm.weight".into(),
        mlxcel_core::copy(&qk),
    );
    weights.insert("language_model.layers.0.attn.k_norm.weight".into(), qk);
    crate::models::jina_vlm::JinaVlmTextModel::from_weights(
        &weights,
        &config,
        "language_model",
        vec![2],
    )
    .expect("tiny Jina VLM text model builds")
}

pub(crate) fn tiny_qwen2_model() -> crate::models::Qwen2Model {
    let mut rng = Rng::new(0x2564_0702);
    let weights = crate::vision::llmjp_vl_test_support::tiny_text_weights(&mut rng, false, false);
    crate::models::Qwen2Model::from_weights(
        &weights,
        &crate::vision::llmjp_vl_test_support::tiny_llama_args(),
    )
    .expect("tiny Qwen2 model builds")
}

pub(crate) fn tiny_qwen3_model() -> crate::models::Qwen3Model {
    let mut rng = Rng::new(0x2564_0703);
    let weights = crate::vision::llmjp_vl_test_support::tiny_text_weights(&mut rng, true, true);
    crate::models::Qwen3Model::from_weights(
        &weights,
        &crate::vision::llmjp_vl_test_support::tiny_qwen3_args(),
    )
    .expect("tiny Qwen3 model builds")
}
