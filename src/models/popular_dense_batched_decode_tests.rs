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

//! Runtime contracts for dense decoders which do not otherwise have a tiny
//! model fixture. Every fixture uses real attention and MLP projections plus
//! non-uniform learned norm gains; a zero-layer model would only prove routing.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::de::DeserializeOwned;
use serde_json::json;
use std::sync::{Mutex, OnceLock};

const HIDDEN: i32 = 8;
const HEAD_DIM: i32 = 4;
const Q_OUT: i32 = 8;
const KV_OUT: i32 = 4;
const INTERMEDIATE: i32 = 16;
const VOCAB: i32 = 32;

fn test_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

fn config<T: DeserializeOwned>(value: serde_json::Value) -> T {
    serde_json::from_value(value).expect("tiny dense config parses")
}

struct SyntheticWeights {
    weights: WeightMap,
    phase: usize,
}

impl SyntheticWeights {
    fn new() -> Self {
        Self {
            weights: WeightMap::new(),
            phase: 1,
        }
    }

    fn values(&mut self, shape: &[i32], base: f32, amplitude: f32) -> UniquePtr<MlxArray> {
        self.phase += 7;
        let phase = self.phase;
        let len: i32 = shape.iter().product();
        let data: Vec<f32> = (0..len)
            .map(|i| base + ((i as usize + phase) as f32).sin() * amplitude)
            .collect();
        mlxcel_core::from_slice_f32(&data, shape)
    }

    fn linear(&mut self, name: impl Into<String>, out: i32, input: i32) {
        let value = self.values(&[out, input], 0.0, 0.18);
        self.weights.insert(name.into(), value);
    }

    fn norm(&mut self, name: impl Into<String>, width: i32) {
        // Centering near one keeps activations bounded while still ensuring
        // that swapping or skipping a learned norm is observable.
        let value = self.values(&[width], 0.85, 0.2);
        self.weights.insert(name.into(), value);
    }

    fn bias(&mut self, name: impl Into<String>, width: i32) {
        let value = self.values(&[width], 0.0, 0.07);
        self.weights.insert(name.into(), value);
    }

    fn scalar(&mut self, name: impl Into<String>, value: f32) {
        self.weights
            .insert(name.into(), mlxcel_core::from_slice_f32(&[value], &[1]));
    }

    fn embedding(&mut self, name: impl Into<String>) {
        self.linear(name, VOCAB, HIDDEN);
    }

    fn separate_attention(&mut self, prefix: &str) {
        self.linear(format!("{prefix}.q_proj.weight"), Q_OUT, HIDDEN);
        self.linear(format!("{prefix}.k_proj.weight"), KV_OUT, HIDDEN);
        self.linear(format!("{prefix}.v_proj.weight"), KV_OUT, HIDDEN);
        self.linear(format!("{prefix}.o_proj.weight"), HIDDEN, Q_OUT);
    }

    fn gated_mlp(&mut self, prefix: &str) {
        self.linear(format!("{prefix}.gate_proj.weight"), INTERMEDIATE, HIDDEN);
        self.linear(format!("{prefix}.up_proj.weight"), INTERMEDIATE, HIDDEN);
        self.linear(format!("{prefix}.down_proj.weight"), HIDDEN, INTERMEDIATE);
    }

    fn standard_layer(&mut self, prefix: &str) {
        self.separate_attention(&format!("{prefix}.self_attn"));
        self.gated_mlp(&format!("{prefix}.mlp"));
        self.norm(format!("{prefix}.input_layernorm.weight"), HIDDEN);
        self.norm(format!("{prefix}.post_attention_layernorm.weight"), HIDDEN);
    }
}

fn assert_contract<M: LanguageModel>(model: &M) {
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        model,
        &[&[1, 2], &[3, 4, 5, 6], &[7, 8, 9, 10, 11, 12]],
        &[13, 14, 15],
        1e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        model,
        &[2, 4, 6],
        8,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(model);
}

fn tiny_apertus_model() -> crate::models::apertus::ApertusModel {
    let args = config(json!({
        "model_type": "apertus",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "qk_norm": true,
        "tie_word_embeddings": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    let p = "model.layers.0";
    w.separate_attention(&format!("{p}.self_attn"));
    w.norm(format!("{p}.self_attn.q_norm.weight"), HEAD_DIM);
    w.norm(format!("{p}.self_attn.k_norm.weight"), HEAD_DIM);
    w.linear(format!("{p}.mlp.up_proj.weight"), INTERMEDIATE, HIDDEN);
    w.linear(format!("{p}.mlp.down_proj.weight"), HIDDEN, INTERMEDIATE);
    w.scalar(format!("{p}.mlp.act_fn.alpha_p"), 0.25);
    w.scalar(format!("{p}.mlp.act_fn.alpha_n"), -0.35);
    w.scalar(format!("{p}.mlp.act_fn.beta"), 0.5);
    w.scalar(format!("{p}.mlp.act_fn.eps"), -1e-6);
    w.norm(format!("{p}.attention_layernorm.weight"), HIDDEN);
    w.norm(format!("{p}.feedforward_layernorm.weight"), HIDDEN);
    crate::models::apertus::ApertusModel::from_weights(&w.weights, &args)
        .expect("tiny Apertus builds")
}

fn tiny_cohere_model() -> crate::models::cohere::CohereModel {
    let args = config(json!({
        "model_type": "cohere",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
        "layer_norm_eps": 1e-5,
        "logit_scale": 0.7,
        "use_qk_norm": true
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    let p = "model.layers.0";
    w.separate_attention(&format!("{p}.self_attn"));
    w.norm(format!("{p}.self_attn.q_norm.weight"), HEAD_DIM);
    w.norm(format!("{p}.self_attn.k_norm.weight"), HEAD_DIM);
    w.gated_mlp(&format!("{p}.mlp"));
    w.norm(format!("{p}.input_layernorm.weight"), HIDDEN);
    crate::models::cohere::CohereModel::from_weights(&w.weights, &args).expect("tiny Cohere builds")
}

fn tiny_ernie45_model() -> crate::models::ernie4_5::Ernie45Model {
    let args = config(json!({
        "model_type": "ernie4_5",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false,
        "use_bias": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    w.standard_layer("model.layers.0");
    crate::models::ernie4_5::Ernie45Model::from_weights(&w.weights, &args)
        .expect("tiny ERNIE 4.5 builds")
}

fn tiny_exaone_model() -> crate::models::exaone::ExaOneModel {
    let args = config(json!({
        "model_type": "exaone",
        "hidden_size": HIDDEN,
        "num_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "layer_norm_epsilon": 1e-5,
        "tie_word_embeddings": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("transformer.wte.weight");
    w.norm("transformer.ln_f.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    let p = "transformer.h.0";
    let a = format!("{p}.attn.attention");
    w.linear(format!("{a}.q_proj.weight"), Q_OUT, HIDDEN);
    w.linear(format!("{a}.k_proj.weight"), KV_OUT, HIDDEN);
    w.linear(format!("{a}.v_proj.weight"), KV_OUT, HIDDEN);
    w.linear(format!("{a}.out_proj.weight"), HIDDEN, Q_OUT);
    w.linear(format!("{p}.mlp.c_fc_0.weight"), INTERMEDIATE, HIDDEN);
    w.linear(format!("{p}.mlp.c_fc_1.weight"), INTERMEDIATE, HIDDEN);
    w.linear(format!("{p}.mlp.c_proj.weight"), HIDDEN, INTERMEDIATE);
    w.norm(format!("{p}.ln_1.weight"), HIDDEN);
    w.norm(format!("{p}.ln_2.weight"), HIDDEN);
    crate::models::exaone::ExaOneModel::from_weights(&w.weights, &args).expect("tiny ExaOne builds")
}

fn tiny_gemma_model() -> crate::models::gemma::GemmaModel {
    let args = config(json!({
        "model_type": "gemma",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.standard_layer("model.layers.0");
    crate::models::gemma::GemmaModel::from_weights(&w.weights, &args).expect("tiny Gemma builds")
}

fn tiny_glm4_model() -> crate::models::glm4::Glm4Model {
    let args = config(json!({
        "model_type": "glm4",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
        "attention_bias": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    let p = "model.layers.0";
    w.separate_attention(&format!("{p}.self_attn"));
    w.linear(
        format!("{p}.mlp.gate_up_proj.weight"),
        2 * INTERMEDIATE,
        HIDDEN,
    );
    w.linear(format!("{p}.mlp.down_proj.weight"), HIDDEN, INTERMEDIATE);
    for name in [
        "input_layernorm",
        "post_self_attn_layernorm",
        "post_attention_layernorm",
        "post_mlp_layernorm",
    ] {
        w.norm(format!("{p}.{name}.weight"), HIDDEN);
    }
    crate::models::glm4::Glm4Model::from_weights(&w.weights, &args).expect("tiny GLM4 builds")
}

pub(crate) fn tiny_granite_model() -> crate::models::granite::GraniteModel {
    let args = config(json!({
        "model_type": "granite",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "logits_scaling": 1.7,
        "attention_multiplier": 0.8,
        "embedding_multiplier": 1.2,
        "residual_multiplier": 0.75,
        "tie_word_embeddings": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    w.standard_layer("model.layers.0");
    crate::models::granite::GraniteModel::from_weights(&w.weights, &args)
        .expect("tiny Granite builds")
}

fn tiny_hunyuan_model() -> crate::models::hunyuan_v1_dense::HunyuanV1DenseModel {
    let args = config(json!({
        "model_type": "hunyuan_v1_dense",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "use_qk_norm": true,
        "tie_word_embeddings": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    w.standard_layer("model.layers.0");
    // These deliberately non-uniform per-head gammas distinguish the model's
    // required post-RoPE Q/K norm from the tempting pre-RoPE implementation.
    w.norm("model.layers.0.self_attn.query_layernorm.weight", HEAD_DIM);
    w.norm("model.layers.0.self_attn.key_layernorm.weight", HEAD_DIM);
    crate::models::hunyuan_v1_dense::HunyuanV1DenseModel::from_weights(&w.weights, &args)
        .expect("tiny Hunyuan V1 Dense builds")
}

fn tiny_internlm2_model() -> crate::models::internlm2::InternLM2Model {
    let args = config(json!({
        "model_type": "internlm2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.tok_embeddings.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("output.weight", VOCAB, HIDDEN);
    let p = "model.layers.0";
    // InternLM2 interleaves Q groups, K and V within this fused projection.
    w.linear(format!("{p}.attention.wqkv.weight"), 16, HIDDEN);
    w.linear(format!("{p}.attention.wo.weight"), HIDDEN, Q_OUT);
    w.linear(format!("{p}.feed_forward.w1.weight"), INTERMEDIATE, HIDDEN);
    w.linear(format!("{p}.feed_forward.w2.weight"), HIDDEN, INTERMEDIATE);
    w.linear(format!("{p}.feed_forward.w3.weight"), INTERMEDIATE, HIDDEN);
    w.norm(format!("{p}.attention_norm.weight"), HIDDEN);
    w.norm(format!("{p}.ffn_norm.weight"), HIDDEN);
    crate::models::internlm2::InternLM2Model::from_weights(&w.weights, &args)
        .expect("tiny InternLM2 builds")
}

fn tiny_internlm3_model() -> crate::models::internlm3::InternLM3Model {
    let args = config(json!({
        "model_type": "internlm3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    w.standard_layer("model.layers.0");
    crate::models::internlm3::InternLM3Model::from_weights(&w.weights, &args)
        .expect("tiny InternLM3 builds")
}

fn tiny_mimo_model() -> crate::models::mimo::MiMoModel {
    let args = config(json!({
        "model_type": "mimo",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false,
        "num_nextn_predict_layers": 0
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    w.standard_layer("model.layers.0");
    crate::models::mimo::MiMoModel::from_weights(&w.weights, &args).expect("tiny MiMo builds")
}

fn tiny_olmo2_model() -> crate::models::olmo2::OLMo2Model {
    let args = config(json!({
        "model_type": "olmo2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    let p = "model.layers.0";
    w.separate_attention(&format!("{p}.self_attn"));
    // OLMo2 normalizes the full projected dimensions before head reshape.
    w.norm(format!("{p}.self_attn.q_norm.weight"), Q_OUT);
    w.norm(format!("{p}.self_attn.k_norm.weight"), KV_OUT);
    w.gated_mlp(&format!("{p}.mlp"));
    w.norm(format!("{p}.post_attention_layernorm.weight"), HIDDEN);
    w.norm(format!("{p}.post_feedforward_layernorm.weight"), HIDDEN);
    crate::models::olmo2::OLMo2Model::from_weights(&w.weights, &args).expect("tiny OLMo2 builds")
}

fn tiny_seed_oss_model() -> crate::models::seed_oss::SeedOssModel {
    let args = config(json!({
        "model_type": "seed_oss",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false,
        "attention_bias": false,
        "attention_out_bias": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    w.standard_layer("model.layers.0");
    crate::models::seed_oss::SeedOssModel::from_weights(&w.weights, &args)
        .expect("tiny Seed-OSS builds")
}

fn tiny_smollm3_model() -> crate::models::smollm3::SmolLM3Model {
    let args = config(json!({
        "model_type": "smollm3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false,
        "no_rope_layers": [1, 0]
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    w.standard_layer("model.layers.0");
    w.standard_layer("model.layers.1");
    crate::models::smollm3::SmolLM3Model::from_weights(&w.weights, &args)
        .expect("tiny SmolLM3 builds")
}

fn tiny_starcoder2_model() -> crate::models::starcoder2::StarCoder2Model {
    let args = config(json!({
        "model_type": "starcoder2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "norm_epsilon": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": false
    }));
    let mut w = SyntheticWeights::new();
    w.embedding("model.embed_tokens.weight");
    w.norm("model.norm.weight", HIDDEN);
    w.bias("model.norm.bias", HIDDEN);
    w.linear("lm_head.weight", VOCAB, HIDDEN);
    let p = "model.layers.0";
    w.separate_attention(&format!("{p}.self_attn"));
    w.bias(format!("{p}.self_attn.q_proj.bias"), Q_OUT);
    w.bias(format!("{p}.self_attn.k_proj.bias"), KV_OUT);
    w.bias(format!("{p}.self_attn.v_proj.bias"), KV_OUT);
    w.bias(format!("{p}.self_attn.o_proj.bias"), HIDDEN);
    w.linear(format!("{p}.mlp.c_fc.weight"), INTERMEDIATE, HIDDEN);
    w.bias(format!("{p}.mlp.c_fc.bias"), INTERMEDIATE);
    w.linear(format!("{p}.mlp.c_proj.weight"), HIDDEN, INTERMEDIATE);
    w.bias(format!("{p}.mlp.c_proj.bias"), HIDDEN);
    w.norm(format!("{p}.input_layernorm.weight"), HIDDEN);
    w.bias(format!("{p}.input_layernorm.bias"), HIDDEN);
    w.norm(format!("{p}.post_attention_layernorm.weight"), HIDDEN);
    w.bias(format!("{p}.post_attention_layernorm.bias"), HIDDEN);
    crate::models::starcoder2::StarCoder2Model::from_weights(&w.weights, &args)
        .expect("tiny StarCoder2 builds")
}

macro_rules! dense_contract {
    ($name:ident, $fixture:ident) => {
        #[test]
        fn $name() {
            let _guard = test_guard().lock().expect("dense test lock");
            assert_contract(&$fixture());
        }
    };
}

dense_contract!(apertus_native_batched_decode_contract, tiny_apertus_model);
dense_contract!(cohere_native_batched_decode_contract, tiny_cohere_model);
dense_contract!(ernie45_native_batched_decode_contract, tiny_ernie45_model);
dense_contract!(exaone_native_batched_decode_contract, tiny_exaone_model);
dense_contract!(gemma_native_batched_decode_contract, tiny_gemma_model);
dense_contract!(glm4_native_batched_decode_contract, tiny_glm4_model);
dense_contract!(granite_native_batched_decode_contract, tiny_granite_model);
dense_contract!(
    hunyuan_v1_dense_native_batched_decode_contract,
    tiny_hunyuan_model
);
dense_contract!(
    internlm2_native_batched_decode_contract,
    tiny_internlm2_model
);
dense_contract!(
    internlm3_native_batched_decode_contract,
    tiny_internlm3_model
);
dense_contract!(mimo_native_batched_decode_contract, tiny_mimo_model);
dense_contract!(olmo2_native_batched_decode_contract, tiny_olmo2_model);
dense_contract!(seed_oss_native_batched_decode_contract, tiny_seed_oss_model);
dense_contract!(smollm3_native_batched_decode_contract, tiny_smollm3_model);
dense_contract!(
    starcoder2_native_batched_decode_contract,
    tiny_starcoder2_model
);
