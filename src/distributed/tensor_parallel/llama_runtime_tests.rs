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

use std::collections::HashMap;
use std::fs;

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;

use super::{
    TensorParallelErnie45Model, TensorParallelGemma3Model, TensorParallelGemma4Model,
    TensorParallelHunyuanV1DenseModel, TensorParallelLlamaModel, TensorParallelQwen3Model,
    TensorParallelQwen35Model, local_llama_args, logical_weight_name, validate_supported_runtime,
};
use crate::distributed::tensor_parallel::{ShardConfig, generate_shard_plan};
use crate::models::ernie4_5::{Ernie45Model, ModelArgs as Ernie45ModelArgs};
use crate::models::gemma3::{Gemma3Wrapper, ModelArgs as Gemma3ModelArgs};
use crate::models::gemma4::{
    Gemma4Wrapper, ModelArgs as Gemma4ModelArgs, RopeParameters as Gemma4RopeParameters,
};
use crate::models::hunyuan_v1_dense::{HunyuanV1DenseModel, ModelArgs as HunyuanV1DenseModelArgs};
use crate::models::llama3::{Llama3Model, ModelArgs as LlamaModelArgs};
use crate::models::qwen2::Qwen2Model;
use crate::models::qwen3::{ModelArgs as Qwen3ModelArgs, Qwen3Model};
use crate::models::qwen3_5::{Qwen35Config, Qwen35Model};

fn make_test_model_args() -> LlamaModelArgs {
    LlamaModelArgs {
        model_type: "llama".to_string(),
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 8,
        head_dim: Some(2),
        num_key_value_heads: Some(2),
        attention_bias: false,
        mlp_bias: false,
        rope_theta: 10_000.0,
        rope_scaling: None,
        quantization: None,
        tie_word_embeddings: false,
    }
}

fn make_test_qwen3_args() -> Qwen3ModelArgs {
    Qwen3ModelArgs {
        model_type: "qwen3".to_string(),
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 8,
        num_key_value_heads: 2,
        head_dim: 2,
        max_position_embeddings: None,
        rope_theta: 10_000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        quantization: None,
    }
}

fn make_test_qwen35_args() -> Qwen35Config {
    Qwen35Config {
        model_type: "qwen3_5".to_string(),
        hidden_size: 8,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: Some(4),
        linear_num_value_heads: 2,
        linear_num_key_heads: 2,
        linear_key_head_dim: 2,
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 4,
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        rope_parameters: None,
        full_attention_interval: 2,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: false,
        attention_bias: false,
        vocab_size: 8,
        quantization: None,
        mlp_only_layers: Vec::new(),
        norm_topk_prob: true,
    }
}

fn make_test_ernie45_args() -> Ernie45ModelArgs {
    Ernie45ModelArgs {
        model_type: "ernie4_5".to_string(),
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        rms_norm_eps: 1e-5,
        vocab_size: 8,
        head_dim: Some(2),
        num_key_value_heads: Some(2),
        use_bias: false,
        rope_theta: 10_000.0,
        max_position_embeddings: 4096,
        tie_word_embeddings: false,
        quantization: None,
    }
}

fn make_test_hunyuan_v1_dense_args() -> HunyuanV1DenseModelArgs {
    HunyuanV1DenseModelArgs {
        model_type: "hunyuan_v1_dense".to_string(),
        vocab_size: 8,
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 4096,
        attention_bias: false,
        use_qk_norm: true,
        rope_scaling: None,
        tie_word_embeddings: false,
        head_dim: Some(2),
        quantization: None,
    }
}

fn make_test_gemma3_args() -> Gemma3ModelArgs {
    Gemma3ModelArgs {
        model_type: "gemma3_text".to_string(),
        hidden_size: 4,
        num_hidden_layers: 1,
        intermediate_size: 8,
        num_attention_heads: 2,
        head_dim: 2,
        rms_norm_eps: 1e-6,
        vocab_size: 8,
        num_key_value_heads: 1,
        rope_theta: 10_000.0,
        rope_local_base_freq: 10_000.0,
        query_pre_attn_scalar: 2.0,
        sliding_window: 8,
        sliding_window_pattern: 1,
        max_position_embeddings: 4096,
        rope_scaling: None,
        quantization: None,
    }
}

fn make_test_gemma4_args() -> Gemma4ModelArgs {
    let mut rope_parameters = HashMap::new();
    rope_parameters.insert(
        "sliding_attention".to_string(),
        Gemma4RopeParameters {
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rope_type: "default".to_string(),
        },
    );
    rope_parameters.insert(
        "full_attention".to_string(),
        Gemma4RopeParameters {
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rope_type: "default".to_string(),
        },
    );

    Gemma4ModelArgs {
        model_type: "gemma4".to_string(),
        text_config: serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 8,
            "vocab_size_per_layer_input": 0,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": null,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "rope_traditional": false,
            "rope_parameters": rope_parameters,
            "sliding_window": 8,
            "sliding_window_pattern": 1,
            "max_position_embeddings": 4096,
            "attention_k_eq_v": false,
            "final_logit_softcapping": null,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "num_experts": null,
            "top_k_experts": null,
            "moe_intermediate_size": null,
            "layer_types": ["sliding_attention"],
            "quantization": null
        }),
        eos_token_id: Some(serde_json::json!([1])),
        quantization: None,
    }
}

fn tensor(values: &[f32], shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_f32(values, shape)
}

fn insert_tensor(weights: &mut WeightMap, name: &str, values: &[f32], shape: &[i32]) {
    weights.insert(name.to_string(), tensor(values, shape));
}

fn seq_values(len: usize, start: f32, step: f32) -> Vec<f32> {
    (0..len).map(|idx| start + idx as f32 * step).collect()
}

fn concat_axis(
    parts: Vec<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>>,
    axis: i32,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut iter = parts.into_iter();
    let first = iter.next().expect("concat_axis requires at least one part");
    iter.fold(first, |acc, part| {
        mlxcel_core::concatenate(&acc, &part, axis)
    })
}

fn slice_last_dim(
    tensor: &mlxcel_core::MlxArray,
    start: i32,
    end: i32,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let shape = mlxcel_core::array_shape(tensor);
    let mut starts = vec![0; shape.len()];
    let mut stops = shape.clone();
    let last = shape.len() - 1;
    starts[last] = start;
    stops[last] = end;
    mlxcel_core::slice(tensor, &starts, &stops)
}

fn make_test_weight_map() -> WeightMap {
    let mut weights = HashMap::new();
    insert_tensor(
        &mut weights,
        "model.embed_tokens.weight",
        &[
            0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30,
            1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70,
            2.80, 2.90, 3.00, 3.10,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_proj.weight",
        &[
            0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_proj.weight",
        &[
            0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 0.1,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.v_proj.weight",
        &[
            0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.20, 0.25, 0.15, 0.20, 0.25, 0.30, 0.20, 0.25,
            0.30, 0.35,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.o_proj.weight",
        &[
            0.20, 0.10, 0.30, 0.40, 0.10, 0.30, 0.20, 0.40, 0.40, 0.30, 0.10, 0.20, 0.30, 0.40,
            0.20, 0.10,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.gate_proj.weight",
        &[
            0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60, 0.40, 0.50,
            0.60, 0.70, 0.50, 0.60, 0.70, 0.80, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 1.00,
            0.80, 0.90, 1.00, 1.10,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.up_proj.weight",
        &[
            0.15, 0.25, 0.35, 0.45, 0.25, 0.35, 0.45, 0.55, 0.35, 0.45, 0.55, 0.65, 0.45, 0.55,
            0.65, 0.75, 0.55, 0.65, 0.75, 0.85, 0.65, 0.75, 0.85, 0.95, 0.75, 0.85, 0.95, 1.05,
            0.85, 0.95, 1.05, 1.15,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.down_proj.weight",
        &[
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55,
        ],
        &[4, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.input_layernorm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.post_attention_layernorm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "model.norm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    insert_tensor(
        &mut weights,
        "lm_head.weight",
        &[
            0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60, 0.40, 0.50,
            0.60, 0.70, 0.50, 0.60, 0.70, 0.80, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 1.00,
            0.80, 0.90, 1.00, 1.10,
        ],
        &[8, 4],
    );
    weights
}

fn make_test_qwen3_weight_map() -> WeightMap {
    let mut weights = make_test_weight_map();
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    weights
}

fn make_test_qwen35_weight_map() -> WeightMap {
    let mut weights = HashMap::new();
    insert_tensor(
        &mut weights,
        "model.embed_tokens.weight",
        &seq_values(64, 0.01, 0.01),
        &[8, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        &seq_values(96, 0.01, 0.005),
        &[12, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.in_proj_z.weight",
        &seq_values(32, 0.02, 0.005),
        &[4, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.in_proj_b.weight",
        &seq_values(16, 0.03, 0.005),
        &[2, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.in_proj_a.weight",
        &seq_values(16, 0.04, 0.005),
        &[2, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.conv1d.weight",
        &seq_values(48, 0.05, 0.0025),
        &[12, 4, 1],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.dt_bias",
        &[0.1, 0.2],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.A_log",
        &[0.3, 0.4],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.linear_attn.out_proj.weight",
        &seq_values(32, 0.06, 0.005),
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.gate_proj.weight",
        &seq_values(64, 0.07, 0.005),
        &[8, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.up_proj.weight",
        &seq_values(64, 0.08, 0.005),
        &[8, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.down_proj.weight",
        &seq_values(64, 0.09, 0.005),
        &[8, 8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.input_layernorm.weight",
        &[1.0; 8],
        &[8],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.post_attention_layernorm.weight",
        &[1.0; 8],
        &[8],
    );
    insert_tensor(&mut weights, "model.norm.weight", &[1.0; 8], &[8]);
    insert_tensor(
        &mut weights,
        "lm_head.weight",
        &seq_values(64, 0.1, 0.01),
        &[8, 8],
    );
    weights
}

fn make_test_hunyuan_v1_dense_weight_map() -> WeightMap {
    let mut weights = make_test_weight_map();
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.query_layernorm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.key_layernorm.weight",
        &[1.0, 1.0],
        &[2],
    );
    weights
}

fn make_test_gemma3_weight_map() -> WeightMap {
    let mut weights = HashMap::new();
    insert_tensor(
        &mut weights,
        "model.embed_tokens.weight",
        &[
            0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30,
            1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70,
            2.80, 2.90, 3.00, 3.10,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_proj.weight",
        &[
            0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_proj.weight",
        &[0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.3],
        &[2, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.v_proj.weight",
        &[0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.20, 0.25],
        &[2, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.o_proj.weight",
        &[
            0.20, 0.10, 0.30, 0.40, 0.10, 0.30, 0.20, 0.40, 0.40, 0.30, 0.10, 0.20, 0.30, 0.40,
            0.20, 0.10,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.q_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.self_attn.k_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.gate_proj.weight",
        &[
            0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60, 0.40, 0.50,
            0.60, 0.70, 0.50, 0.60, 0.70, 0.80, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 1.00,
            0.80, 0.90, 1.00, 1.10,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.up_proj.weight",
        &[
            0.15, 0.25, 0.35, 0.45, 0.25, 0.35, 0.45, 0.55, 0.35, 0.45, 0.55, 0.65, 0.45, 0.55,
            0.65, 0.75, 0.55, 0.65, 0.75, 0.85, 0.65, 0.75, 0.85, 0.95, 0.75, 0.85, 0.95, 1.05,
            0.85, 0.95, 1.05, 1.15,
        ],
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "model.layers.0.mlp.down_proj.weight",
        &[
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55,
        ],
        &[4, 8],
    );
    for norm_name in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.pre_feedforward_layernorm.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
        "model.norm.weight",
    ] {
        insert_tensor(&mut weights, norm_name, &[1.0, 1.0, 1.0, 1.0], &[4]);
    }
    insert_tensor(
        &mut weights,
        "lm_head.weight",
        &[
            0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.40, 0.50, 0.30, 0.40, 0.50, 0.60, 0.40, 0.50,
            0.60, 0.70, 0.50, 0.60, 0.70, 0.80, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 1.00,
            0.80, 0.90, 1.00, 1.10,
        ],
        &[8, 4],
    );
    weights
}

fn make_test_gemma4_weight_map() -> WeightMap {
    let mut weights = HashMap::new();
    insert_tensor(
        &mut weights,
        "language_model.model.embed_tokens.weight",
        &seq_values(32, 0.0, 0.1),
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.self_attn.q_proj.weight",
        &[
            0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.self_attn.k_proj.weight",
        &[0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.3],
        &[2, 4],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.self_attn.v_proj.weight",
        &[0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.20, 0.25],
        &[2, 4],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.self_attn.o_proj.weight",
        &[
            0.20, 0.10, 0.30, 0.40, 0.10, 0.30, 0.20, 0.40, 0.40, 0.30, 0.10, 0.20, 0.30, 0.40,
            0.20, 0.10,
        ],
        &[4, 4],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.self_attn.q_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.self_attn.k_norm.weight",
        &[1.0, 1.0],
        &[2],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.mlp.gate_proj.weight",
        &seq_values(32, 0.01, 0.01),
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.mlp.up_proj.weight",
        &seq_values(32, 0.02, 0.01),
        &[8, 4],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.mlp.down_proj.weight",
        &seq_values(32, 0.03, 0.01),
        &[4, 8],
    );
    for norm in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ] {
        insert_tensor(
            &mut weights,
            &format!("language_model.model.layers.0.{norm}.weight"),
            &[1.0, 1.0, 1.0, 1.0],
            &[4],
        );
    }
    insert_tensor(
        &mut weights,
        "language_model.model.layers.0.layer_scalar",
        &[1.0],
        &[1],
    );
    insert_tensor(
        &mut weights,
        "language_model.model.norm.weight",
        &[1.0, 1.0, 1.0, 1.0],
        &[4],
    );
    weights
}

#[test]
fn logical_weight_name_maps_auxiliary_quantization_keys_to_weight_name() {
    assert_eq!(
        logical_weight_name("model.layers.0.self_attn.q_proj.scales"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        logical_weight_name("model.layers.0.self_attn.q_proj.biases"),
        "model.layers.0.self_attn.q_proj.weight"
    );
}

#[test]
fn validate_supported_runtime_accepts_llama_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-llama-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 1
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(!support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_qwen3_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-qwen3-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3",
            "num_hidden_layers": 1
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(!support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_qwen35_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-qwen35-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3_5",
            "num_hidden_layers": 24
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(!support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_qwen2_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-qwen2-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen2",
            "num_hidden_layers": 24
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(!support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_ernie45_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-ernie45-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "ernie4_5",
            "num_hidden_layers": 18
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(!support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_hunyuan_v1_dense_replicated_path() {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel-tp-hunyuan-v1-dense-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "hunyuan_v1_dense",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(!support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_gemma3_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-gemma3-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma3_text",
            "num_hidden_layers": 26
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(!support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_supported_runtime_accepts_gemma4_replicated_path() {
    let dir = std::env::temp_dir().join(format!("mlxcel-tp-gemma4-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 26
            }
        }"#,
    )
    .unwrap();

    let support = validate_supported_runtime(&dir, ShardConfig::with_tp_size(2), None).unwrap();
    assert!(!support.force_no_batch);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tensor_parallel_llama_matches_full_model_logits() {
    let args = make_test_model_args();
    let weights = make_test_weight_map();
    let full = Llama3Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "llama",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelLlamaModel::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn local_llama_args_preserves_computed_head_dim_when_config_omits_it() {
    let args = LlamaModelArgs {
        model_type: "llama".to_string(),
        hidden_size: 896,
        num_hidden_layers: 1,
        intermediate_size: 4864,
        num_attention_heads: 14,
        rms_norm_eps: 1e-5,
        vocab_size: 1024,
        head_dim: None,
        num_key_value_heads: Some(2),
        attention_bias: false,
        mlp_bias: false,
        rope_theta: 1_000_000.0,
        rope_scaling: None,
        quantization: None,
        tie_word_embeddings: true,
    };
    let plan = generate_shard_plan("llama", 1, &ShardConfig::with_tp_size(2)).unwrap();

    let local = local_llama_args(&args, &plan).unwrap();

    assert_eq!(local.head_dim, Some(64));
    assert_eq!(local.num_attention_heads, 7);
}

#[test]
fn tensor_parallel_qwen2_matches_full_model_logits() {
    let args = make_test_model_args();
    let weights = make_test_weight_map();
    let full = Qwen2Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "qwen2",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelLlamaModel::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn tensor_parallel_qwen3_matches_full_model_logits() {
    let args = make_test_qwen3_args();
    let weights = make_test_qwen3_weight_map();
    let full = Qwen3Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "qwen3",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelQwen3Model::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn tensor_parallel_qwen35_matches_full_model_logits() {
    let args = make_test_qwen35_args();
    let weights = make_test_qwen35_weight_map();
    let full = Qwen35Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "qwen3_5",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelQwen35Model::from_full_weights(&args, &weights, &plan, None).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
#[ignore = "requires local qwen3.5 large weights and CUDA runtime"]
fn tensor_parallel_qwen35_text_only_9b_debug_linear_attn_stages() {
    let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("qwen3.5-9b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let config_path = model_dir.join("config.json");
    let config_str = fs::read_to_string(&config_path).unwrap();
    let config_str = crate::models::sanitize_config_json(&config_str);
    let config_json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let mut text_config = config_json
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| config_json.clone());
    if text_config.get("quantization").is_none()
        && let Some(quantization) = config_json.get("quantization")
    {
        text_config
            .as_object_mut()
            .unwrap()
            .insert("quantization".to_string(), quantization.clone());
    }
    let args: Qwen35Config = serde_json::from_value(text_config).unwrap();
    let mut weights = super::load_qwen35_tp_text_weights(&model_dir, &config_json, &args).unwrap();
    super::ensure_qwen35_lm_head_weights(&mut weights);

    let full = Qwen35Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "qwen3_5",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelQwen35Model::from_full_weights(&args, &weights, &plan, None).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[11, 22, 33, 44], &[1, 4]);
    let hidden = full.embed_tokens.forward(&input_ids);
    let attn_norm = full.layers[0].input_layernorm.forward(&hidden);
    let mask = mlxcel_core::astype(
        &mlxcel_core::from_slice_i32(&[1, 1, 1, 1], &[1, 4]),
        mlxcel_core::dtype::BOOL,
    );

    let full_attn = match &full.layers[0].attention {
        crate::models::qwen3_5::Qwen35AttentionVariant::Linear(attn) => attn,
        _ => panic!("expected linear attention in layer 0"),
    };
    let full_dbg = full_attn.debug_prefill_no_cache(&attn_norm, Some(&mask));
    let full_runtime_attn = full_attn.forward(&attn_norm, None, None);

    let local_k_dim =
        ((args.linear_num_key_heads * args.linear_key_head_dim) / plan.tp_size) as i32;
    let local_v_dim =
        ((args.linear_num_value_heads * args.linear_value_head_dim) / plan.tp_size) as i32;

    let mut qkv_q_parts = Vec::new();
    let mut qkv_k_parts = Vec::new();
    let mut qkv_v_parts = Vec::new();
    let mut z_parts = Vec::new();
    let mut b_parts = Vec::new();
    let mut a_parts = Vec::new();
    let mut conv_q_parts = Vec::new();
    let mut conv_k_parts = Vec::new();
    let mut conv_v_parts = Vec::new();
    let mut q_parts = Vec::new();
    let mut k_parts = Vec::new();
    let mut v_parts = Vec::new();
    let mut beta_parts = Vec::new();
    let mut g_parts = Vec::new();
    let mut gated_parts = Vec::new();
    let mut normed_parts = Vec::new();
    let mut projected_parts = Vec::new();

    for rank in &tp.ranks {
        let rank_attn = match &rank.layers[0].attention {
            crate::models::qwen3_5::Qwen35AttentionVariant::Linear(attn) => attn,
            _ => panic!("expected linear attention in rank layer 0"),
        };
        let dbg = rank_attn.debug_prefill_no_cache(&attn_norm, Some(&mask));
        qkv_q_parts.push(slice_last_dim(&dbg.qkv, 0, local_k_dim));
        qkv_k_parts.push(slice_last_dim(&dbg.qkv, local_k_dim, local_k_dim * 2));
        qkv_v_parts.push(slice_last_dim(
            &dbg.qkv,
            local_k_dim * 2,
            local_k_dim * 2 + local_v_dim,
        ));
        z_parts.push(mlxcel_core::reshape(&dbg.z, &[1, 4, -1]));
        b_parts.push(dbg.b_proj);
        a_parts.push(dbg.a);
        conv_q_parts.push(slice_last_dim(&dbg.conv_out, 0, local_k_dim));
        conv_k_parts.push(slice_last_dim(&dbg.conv_out, local_k_dim, local_k_dim * 2));
        conv_v_parts.push(slice_last_dim(
            &dbg.conv_out,
            local_k_dim * 2,
            local_k_dim * 2 + local_v_dim,
        ));
        q_parts.push(mlxcel_core::reshape(&dbg.q, &[1, 4, -1]));
        k_parts.push(mlxcel_core::reshape(&dbg.k, &[1, 4, -1]));
        v_parts.push(mlxcel_core::reshape(&dbg.v, &[1, 4, -1]));
        beta_parts.push(dbg.beta);
        g_parts.push(dbg.g);
        gated_parts.push(mlxcel_core::reshape(&dbg.gated_out, &[1, 4, -1]));
        normed_parts.push(dbg.normed_out);
        projected_parts.push(dbg.projected);
    }

    let recon_qkv = {
        let q = concat_axis(qkv_q_parts, -1);
        let k = concat_axis(qkv_k_parts, -1);
        let qk = mlxcel_core::concatenate(&q, &k, -1);
        let v = concat_axis(qkv_v_parts, -1);
        mlxcel_core::concatenate(&qk, &v, -1)
    };
    let recon_z = concat_axis(z_parts, -1);
    let recon_b = concat_axis(b_parts, -1);
    let recon_a = concat_axis(a_parts, -1);
    let recon_conv = {
        let q = concat_axis(conv_q_parts, -1);
        let k = concat_axis(conv_k_parts, -1);
        let qk = mlxcel_core::concatenate(&q, &k, -1);
        let v = concat_axis(conv_v_parts, -1);
        mlxcel_core::concatenate(&qk, &v, -1)
    };
    let recon_q = concat_axis(q_parts, -1);
    let recon_k = concat_axis(k_parts, -1);
    let recon_v = concat_axis(v_parts, -1);
    let recon_beta = concat_axis(beta_parts, -1);
    let recon_g = concat_axis(g_parts, -1);
    let recon_gated = concat_axis(gated_parts, -1);
    let recon_normed = concat_axis(normed_parts, -1);
    let recon_projected = super::reduce_sum_f32(projected_parts);
    let tp_runtime_hidden_parts: Vec<_> = tp
        .ranks
        .iter()
        .map(|rank| match &rank.layers[0].attention {
            crate::models::qwen3_5::Qwen35AttentionVariant::Linear(attn) => {
                attn.forward_hidden_tp(&attn_norm, None, None)
            }
            _ => panic!("expected linear attention in rank layer 0"),
        })
        .collect();
    let tp_runtime_attn = tp.full_linear_out_projs[0]
        .as_ref()
        .unwrap()
        .forward(&super::concat_last_dim(tp_runtime_hidden_parts));

    let full_z = mlxcel_core::reshape(&full_dbg.z, &[1, 4, -1]);
    let full_q = mlxcel_core::reshape(&full_dbg.q, &[1, 4, -1]);
    let full_k = mlxcel_core::reshape(&full_dbg.k, &[1, 4, -1]);
    let full_v = mlxcel_core::reshape(&full_dbg.v, &[1, 4, -1]);
    let full_gated = mlxcel_core::reshape(&full_dbg.gated_out, &[1, 4, -1]);

    for (stage, lhs, rhs) in [
        ("qkv", &full_dbg.qkv, &recon_qkv),
        ("z", &full_z, &recon_z),
        ("b", &full_dbg.b_proj, &recon_b),
        ("a", &full_dbg.a, &recon_a),
        ("conv", &full_dbg.conv_out, &recon_conv),
        ("q", &full_q, &recon_q),
        ("k", &full_k, &recon_k),
        ("v", &full_v, &recon_v),
        ("beta", &full_dbg.beta, &recon_beta),
        ("g", &full_dbg.g, &recon_g),
        ("gated", &full_gated, &recon_gated),
        ("normed", &full_dbg.normed_out, &recon_normed),
        ("projected", &full_dbg.projected, &recon_projected),
    ] {
        let close = mlxcel_core::allclose(lhs, rhs, 1e-4, 1e-4);
        eprintln!("{stage}_close={}", mlxcel_core::item_bool(&close));
    }

    let runtime_attn_close =
        mlxcel_core::allclose(&full_runtime_attn, &tp_runtime_attn, 1e-4, 1e-4);
    eprintln!(
        "runtime_attn_close={}",
        mlxcel_core::item_bool(&runtime_attn_close)
    );

    let projected_close = mlxcel_core::allclose(&full_dbg.projected, &recon_projected, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&projected_close));
}

#[test]
#[ignore = "requires local qwen3.5 large weights and CUDA runtime"]
fn tensor_parallel_qwen35_text_only_9b_prefill_logits_match_full_model() {
    let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("qwen3.5-9b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let config_path = model_dir.join("config.json");
    let config_str = fs::read_to_string(&config_path).unwrap();
    let config_str = crate::models::sanitize_config_json(&config_str);
    let config_json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let mut text_config = config_json
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| config_json.clone());
    if text_config.get("quantization").is_none()
        && let Some(quantization) = config_json.get("quantization")
    {
        text_config
            .as_object_mut()
            .unwrap()
            .insert("quantization".to_string(), quantization.clone());
    }
    let args: Qwen35Config = serde_json::from_value(text_config).unwrap();
    let mut weights = super::load_qwen35_tp_text_weights(&model_dir, &config_json, &args).unwrap();
    super::ensure_qwen35_lm_head_weights(&mut weights);
    let mrope = super::qwen35_mrope_params(&args);

    let mut full = Qwen35Model::from_weights(&weights, &args).unwrap();
    if let Some((mrope_section, rope_theta, rope_dims)) = mrope.clone() {
        full.set_mrope(mrope_section, rope_theta, rope_dims);
    }
    let plan = generate_shard_plan(
        "qwen3_5",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelQwen35Model::from_full_weights(&args, &weights, &plan, mrope).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[11, 22, 33, 44], &[1, 4]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
#[ignore = "requires local qwen3.5 large weights and CUDA runtime"]
fn tensor_parallel_qwen35_text_only_9b_debug_layerwise_runtime_divergence() {
    run_qwen35_text_only_layerwise_runtime_divergence("qwen3.5-9b-4bit", 2);
}

#[test]
#[ignore = "requires local qwen3.5 large weights and CUDA runtime"]
fn tensor_parallel_qwen35_text_only_27b_debug_layerwise_runtime_divergence() {
    run_qwen35_text_only_layerwise_runtime_divergence("qwen3.5-27b-4bit", 4);
}

fn run_qwen35_text_only_layerwise_runtime_divergence(model_name: &str, tp_size: usize) {
    let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(model_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let config_path = model_dir.join("config.json");
    let config_str = fs::read_to_string(&config_path).unwrap();
    let config_str = crate::models::sanitize_config_json(&config_str);
    let config_json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let mut text_config = config_json
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| config_json.clone());
    if text_config.get("quantization").is_none()
        && let Some(quantization) = config_json.get("quantization")
    {
        text_config
            .as_object_mut()
            .unwrap()
            .insert("quantization".to_string(), quantization.clone());
    }
    let args: Qwen35Config = serde_json::from_value(text_config).unwrap();
    let mut weights = super::load_qwen35_tp_text_weights(&model_dir, &config_json, &args).unwrap();
    super::ensure_qwen35_lm_head_weights(&mut weights);
    let mrope = super::qwen35_mrope_params(&args);

    let mut full = Qwen35Model::from_weights(&weights, &args).unwrap();
    if let Some((mrope_section, rope_theta, rope_dims)) = mrope.clone() {
        full.set_mrope(mrope_section, rope_theta, rope_dims);
    }
    let plan = generate_shard_plan(
        "qwen3_5",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(tp_size),
    )
    .unwrap();
    let tp = TensorParallelQwen35Model::from_full_weights(&args, &weights, &plan, mrope).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[11, 22, 33, 44], &[1, 4]);
    let mut full_h = full.embed_tokens.forward(&input_ids);
    let mut tp_h = tp.ranks[0].embed_tokens.forward(&input_ids);
    let mut full_caches: Vec<crate::models::qwen3_next::Qwen3NextCache> = full
        .layers
        .iter()
        .map(|layer| {
            if layer.is_linear {
                crate::models::qwen3_next::Qwen3NextCache::Linear(
                    crate::models::gated_delta::GatedDeltaCache::new(),
                )
            } else {
                crate::models::qwen3_next::Qwen3NextCache::Attention(
                    mlxcel_core::layers::KVCache::new(),
                )
            }
        })
        .collect();
    let mut tp_caches = tp.fresh_rank_caches();
    let seq_len = 4;
    let fa_mask = Some(mlxcel_core::utils::create_causal_mask(seq_len, 0));

    for layer_idx in 0..args.num_hidden_layers {
        let full_attn_norm = full.layers[layer_idx].input_layernorm.forward(&full_h);
        let full_attn_out = match (
            &full.layers[layer_idx].attention,
            &mut full_caches[layer_idx],
        ) {
            (
                crate::models::qwen3_5::Qwen35AttentionVariant::Linear(attn),
                crate::models::qwen3_next::Qwen3NextCache::Linear(cache),
            ) => attn.forward(&full_attn_norm, None, Some(&mut *cache)),
            (
                crate::models::qwen3_5::Qwen35AttentionVariant::FullAttention(attn),
                crate::models::qwen3_next::Qwen3NextCache::Attention(cache),
            ) => attn.forward_with_position_ids(
                &full_attn_norm,
                &mut *cache,
                fa_mask.as_deref(),
                None,
            ),
            _ => unreachable!(),
        };
        full_h = mlxcel_core::add(&full_h, &full_attn_out);

        let tp_attn_norm = tp.ranks[0].layers[layer_idx].input_layernorm.forward(&tp_h);
        let tp_attn_parts: Vec<_> = tp
            .ranks
            .iter()
            .zip(tp_caches.iter_mut())
            .map(|(rank, caches)| {
                match (&rank.layers[layer_idx].attention, &mut caches[layer_idx]) {
                    (
                        crate::models::qwen3_5::Qwen35AttentionVariant::Linear(attn),
                        crate::models::qwen3_next::Qwen3NextCache::Linear(cache),
                    ) => attn.forward_hidden_tp(&tp_attn_norm, None, Some(cache)),
                    (
                        crate::models::qwen3_5::Qwen35AttentionVariant::FullAttention(attn),
                        crate::models::qwen3_next::Qwen3NextCache::Attention(cache),
                    ) => attn.forward_hidden_with_position_ids(
                        &tp_attn_norm,
                        cache,
                        fa_mask.as_deref(),
                        None,
                    ),
                    _ => unreachable!(),
                }
            })
            .collect();
        let tp_attn_out = if tp.ranks[0].layers[layer_idx].is_linear {
            tp.full_linear_out_projs[layer_idx]
                .as_ref()
                .unwrap()
                .forward(&super::concat_last_dim(tp_attn_parts))
        } else {
            tp.full_attention_out_projs[layer_idx]
                .as_ref()
                .unwrap()
                .forward(&super::concat_last_dim(tp_attn_parts))
        };
        tp_h = mlxcel_core::add(&tp_h, &tp_attn_out);

        let attn_close = mlxcel_core::allclose(&full_h, &tp_h, 1e-4, 1e-4);
        eprintln!(
            "layer={layer_idx} after_attn={}",
            mlxcel_core::item_bool(&attn_close)
        );
        if !mlxcel_core::item_bool(&attn_close) {
            panic!("diverged after attention at layer {layer_idx}");
        }

        let full_ffn_norm = full.layers[layer_idx]
            .post_attention_layernorm
            .forward(&full_h);
        let full_ffn = match &full.layers[layer_idx].mlp {
            crate::models::qwen3_5::Qwen35MLPVariant::Dense(mlp) => mlp.forward(&full_ffn_norm),
            crate::models::qwen3_5::Qwen35MLPVariant::MoE(_) => unreachable!(),
        };
        let full_ff_hidden = match &full.layers[layer_idx].mlp {
            crate::models::qwen3_5::Qwen35MLPVariant::Dense(mlp) => {
                mlp.forward_hidden(&full_ffn_norm)
            }
            crate::models::qwen3_5::Qwen35MLPVariant::MoE(_) => unreachable!(),
        };
        full_h = mlxcel_core::add(&full_h, &full_ffn);

        let tp_ffn_norm = tp.ranks[0].layers[layer_idx]
            .post_attention_layernorm
            .forward(&tp_h);
        let tp_ffn_parts: Vec<_> = tp
            .ranks
            .iter()
            .map(|rank| match &rank.layers[layer_idx].mlp {
                crate::models::qwen3_5::Qwen35MLPVariant::Dense(mlp) => {
                    mlp.forward_hidden(&tp_ffn_norm)
                }
                crate::models::qwen3_5::Qwen35MLPVariant::MoE(_) => unreachable!(),
            })
            .collect();
        let tp_ff_hidden = super::concat_last_dim(tp_ffn_parts);
        let hidden_close = mlxcel_core::allclose(&full_ff_hidden, &tp_ff_hidden, 1e-4, 1e-4);
        eprintln!(
            "layer={layer_idx} ffn_hidden={}",
            mlxcel_core::item_bool(&hidden_close)
        );
        let tp_ffn = tp.full_mlp_down_projs[layer_idx].forward(&tp_ff_hidden);
        let ffn_close = mlxcel_core::allclose(&full_ffn, &tp_ffn, 1e-4, 1e-4);
        eprintln!(
            "layer={layer_idx} ffn_out={}",
            mlxcel_core::item_bool(&ffn_close)
        );
        tp_h = mlxcel_core::add(&tp_h, &tp_ffn);

        let layer_close = mlxcel_core::allclose(&full_h, &tp_h, 1e-4, 1e-4);
        eprintln!(
            "layer={layer_idx} after_ffn={}",
            mlxcel_core::item_bool(&layer_close)
        );
        if !mlxcel_core::item_bool(&layer_close) {
            panic!("diverged after ffn at layer {layer_idx}");
        }
    }

    let full_norm = full.norm.forward(&full_h);
    let tp_norm = tp.ranks[0].norm.forward(&tp_h);
    let norm_close = mlxcel_core::allclose(&full_norm, &tp_norm, 1e-4, 1e-4);
    eprintln!("final_norm={}", mlxcel_core::item_bool(&norm_close));

    let full_logits = if let Some(ref lm_head) = full.lm_head {
        lm_head.forward(&full_norm)
    } else {
        full.embed_tokens.as_linear(&full_norm)
    };
    let tp_logits = tp.final_logits(&tp_norm);
    let logits_close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    eprintln!("final_logits={}", mlxcel_core::item_bool(&logits_close));
    if !mlxcel_core::item_bool(&logits_close) {
        let diff = mlxcel_core::subtract(&full_logits, &tp_logits);
        let abs_diff = mlxcel_core::abs(&diff);
        let max_diff = mlxcel_core::max_all(&abs_diff);
        mlxcel_core::eval(&max_diff);
        eprintln!("final_logits_max_abs={}", mlxcel_core::item_f32(&max_diff));
        panic!("diverged at final logits");
    }
}

#[test]
fn tensor_parallel_ernie45_matches_full_model_logits() {
    let args = make_test_ernie45_args();
    let weights = make_test_weight_map();
    let full = Ernie45Model::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "ernie4_5",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelErnie45Model::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn tensor_parallel_hunyuan_v1_dense_matches_full_model_logits() {
    let args = make_test_hunyuan_v1_dense_args();
    let weights = make_test_hunyuan_v1_dense_weight_map();
    let full = HunyuanV1DenseModel::from_weights(&weights, &args).unwrap();
    let plan = generate_shard_plan(
        "hunyuan_v1_dense",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelHunyuanV1DenseModel::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn tensor_parallel_gemma3_matches_full_model_logits() {
    let args = make_test_gemma3_args();
    let weights = make_test_gemma3_weight_map();
    let full =
        Gemma3Wrapper::new(crate::models::Gemma3Model::from_weights(&weights, &args).unwrap());
    let plan = generate_shard_plan(
        "gemma3_text",
        args.num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelGemma3Model::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);
    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-4, 1e-4);
    assert!(mlxcel_core::item_bool(&close));
}

#[test]
fn tensor_parallel_gemma4_matches_full_model_logits() {
    let args = make_test_gemma4_args();
    let weights = make_test_gemma4_weight_map();
    let full =
        Gemma4Wrapper::new(crate::models::Gemma4Model::from_weights(&weights, &args).unwrap());
    let plan = generate_shard_plan(
        "gemma4",
        args.text_args().num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelGemma4Model::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mut full_caches = full.make_caches();
    let mut tp_caches = tp.make_caches();
    let full_logits = full.forward(&input_ids, &mut full_caches, None);
    let tp_logits = tp.forward(&input_ids, &mut tp_caches, None);

    let close = mlxcel_core::allclose(&full_logits, &tp_logits, 1e-5, 1e-5);
    assert!(mlxcel_core::item_bool(&close));
}

/// Regression test for TP Gemma4 must not double-apply embed_scale
/// when `input_embeddings` is supplied.
///
/// The non-TP path was fixed. This test verifies that
/// `TensorParallelGemma4Model::forward_impl` applies the same conditional:
/// `sqrt(hidden_size)` scale only for the `input_ids` path, not for the
/// pre-merged `input_embeddings` path.
///
/// Strategy: run both non-TP (already fixed) and TP with identical synthetic
/// embeddings. If TP still double-scales, the two outputs diverge.
/// Additionally, verify the text-only (input_ids) path still scales correctly
/// by comparing it to the full model.
#[test]
fn tensor_parallel_gemma4_embed_scale_not_doubled_for_input_embeddings() {
    let args = make_test_gemma4_args();
    let weights = make_test_gemma4_weight_map();
    // hidden_size = 4, so sqrt(hidden_size) = 2.0
    let hidden_size = args.text_args().hidden_size;
    assert_eq!(hidden_size, 4, "test relies on hidden_size == 4");

    let full =
        Gemma4Wrapper::new(crate::models::Gemma4Model::from_weights(&weights, &args).unwrap());
    let plan = generate_shard_plan(
        "gemma4",
        args.text_args().num_hidden_layers,
        &ShardConfig::with_tp_size(2),
    )
    .unwrap();
    let tp = TensorParallelGemma4Model::from_full_weights(&args, &weights, &plan).unwrap();

    let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);

    // Build synthetic embeddings: shape [1, 2, 4], all elements 1.0.
    // With hidden_size=4, sqrt(4)=2.0.  A double-scale would yield 2.0; a
    // correct single-skip yields the original 1.0 feeding through the layers.
    let embeddings = mlxcel_core::from_slice_f32(&[1.0_f32; 8], &[1, 2, 4]);

    // --- input_embeddings path: TP must match non-TP (no double-scale) ---
    let mut full_caches_emb = full.make_caches();
    let mut tp_caches_emb = tp.make_caches();
    let full_logits_emb = LanguageModel::forward_with_embeddings(
        &full,
        &input_ids,
        Some(&embeddings),
        &mut full_caches_emb,
        None,
    );
    let tp_logits_emb = LanguageModel::forward_with_embeddings(
        &tp,
        &input_ids,
        Some(&embeddings),
        &mut tp_caches_emb,
        None,
    );
    let close_emb = mlxcel_core::allclose(&full_logits_emb, &tp_logits_emb, 1e-5, 1e-5);
    assert!(
        mlxcel_core::item_bool(&close_emb),
        "TP Gemma4 logits with input_embeddings must match non-TP (embed_scale must not be applied twice)"
    );

    // --- input_ids path: TP must still match non-TP (text-only scale intact) ---
    let mut full_caches_ids = full.make_caches();
    let mut tp_caches_ids = tp.make_caches();
    let full_logits_ids = full.forward(&input_ids, &mut full_caches_ids, None);
    let tp_logits_ids = tp.forward(&input_ids, &mut tp_caches_ids, None);
    let close_ids = mlxcel_core::allclose(&full_logits_ids, &tp_logits_ids, 1e-5, 1e-5);
    assert!(
        mlxcel_core::item_bool(&close_ids),
        "TP Gemma4 logits with input_ids must match non-TP (text-only embed_scale must still be applied)"
    );
}
