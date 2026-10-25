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

//! Shared native-batch contracts for the Mixtral-shaped MoE families.
//!
//! Each fixture has real attention and routed-expert layers.
//! The unequal prompt lengths make the two decode rows use different RoPE and
//! cache offsets; the shared harness compares them against isolated decodes and
//! separately pins the exact B=1 route and the empty scheduler guard.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn filled(shape: &[i32], seed: u32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state = seed;
    let values: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.1
        })
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn insert(weights: &mut WeightMap, name: impl Into<String>, shape: &[i32], seed: u32) {
    weights.insert(name.into(), filled(shape, seed));
}

fn assert_contract<M: LanguageModel>(model: &M) {
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        model,
        &[&[1, 2, 3], &[4, 5], &[6]],
        &[7, 8, 9],
        1e-5,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        model,
        &[1, 3, 5],
        7,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(model);
}

fn insert_attention(
    weights: &mut WeightMap,
    prefix: &str,
    hidden: i32,
    kv_width: i32,
    with_bias: bool,
) {
    for (offset, (name, output)) in [
        ("q_proj", hidden),
        ("k_proj", kv_width),
        ("v_proj", kv_width),
        ("o_proj", hidden),
    ]
    .into_iter()
    .enumerate()
    {
        insert(
            weights,
            format!("{prefix}.{name}.weight"),
            &[output, hidden],
            100 + offset as u32,
        );
        if with_bias && name != "o_proj" {
            insert(
                weights,
                format!("{prefix}.{name}.bias"),
                &[output],
                110 + offset as u32,
            );
        }
    }
}

fn insert_switch_glu(
    weights: &mut WeightMap,
    prefix: &str,
    experts: i32,
    hidden: i32,
    intermediate: i32,
) {
    insert(
        weights,
        format!("{prefix}.gate_proj.weight"),
        &[experts, intermediate, hidden],
        201,
    );
    insert(
        weights,
        format!("{prefix}.up_proj.weight"),
        &[experts, intermediate, hidden],
        202,
    );
    insert(
        weights,
        format!("{prefix}.down_proj.weight"),
        &[experts, hidden, intermediate],
        203,
    );
}

#[test]
fn routed_expert_grid_handles_shared_and_disjoint_expert_sets() {
    use super::phimoe::sparsemixer;
    use super::switch_layers::SwitchGLU;

    let mut weights = WeightMap::new();
    insert_switch_glu(&mut weights, "moe.switch_mlp", 4, 4, 6);
    let experts = SwitchGLU::from_weights(&weights, "moe.switch_mlp", 64, 4).unwrap();
    let x = filled(&[3, 4], 900);

    let gates = mlxcel_core::from_slice_f32(
        &[
            4.0, 3.0, 0.0, -1.0, // {0,1}
            5.0, 2.0, 0.0, -1.0, // {0,1} again
            -1.0, 0.0, 4.0, 3.0, // disjoint {2,3}
        ],
        &[3, 4],
    );
    let (routed, _) = sparsemixer(&gates, 2, 0.01);
    let routed = mlxcel_core::astype(&routed, mlxcel_core::dtype::INT32);
    mlxcel_core::eval(&routed);
    let raw = mlxcel_core::array_to_raw_bytes(&routed);
    let routed: Vec<i32> = raw
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    for (row, expected) in routed.chunks_exact(2).zip([[0, 1], [0, 1], [2, 3]]) {
        let mut row = row.to_vec();
        row.sort_unstable();
        assert_eq!(row, expected);
    }

    // Three top-2 rows make six routed slots, more than the four experts. The
    // first grid sends every row to the same experts; the second includes a
    // disjoint {2,3} row and a mixed row. This is the exact post-router shape
    // each family's unchanged SwitchGLU path receives at B > 1.
    let same = mlxcel_core::from_slice_i32(&[0, 1, 0, 1, 0, 1], &[3, 2]);
    let disjoint = mlxcel_core::from_slice_i32(&[0, 1, 2, 3, 0, 3], &[3, 2]);
    let same_out = experts.forward(&x, &same);
    let disjoint_out = experts.forward(&x, &disjoint);
    mlxcel_core::eval(&same_out);
    mlxcel_core::eval(&disjoint_out);
    assert_eq!(mlxcel_core::array_shape(&same_out), vec![3, 2, 4]);
    assert_eq!(mlxcel_core::array_shape(&disjoint_out), vec![3, 2, 4]);
    let diff = mlxcel_core::sum_all(&mlxcel_core::abs(&mlxcel_core::subtract(
        &same_out,
        &disjoint_out,
    )));
    assert!(
        mlxcel_core::item_f32(&diff) > 0.0,
        "disjoint expert routing must execute different expert planes"
    );
}

#[test]
fn mixtral_native_batched_decode_contract() {
    use super::mixtral::{MixtralModel, ModelArgs};

    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "mixtral",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts_per_tok": 2,
        "num_local_experts": 3,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let mut weights = WeightMap::new();
    insert(&mut weights, "model.embed_tokens.weight", &[16, 8], 1);
    insert_attention(&mut weights, "model.layers.0.self_attn", 8, 4, false);
    weights.insert("model.layers.0.input_layernorm.weight".into(), ones(&[8]));
    weights.insert(
        "model.layers.0.post_attention_layernorm.weight".into(),
        ones(&[8]),
    );
    insert(
        &mut weights,
        "model.layers.0.block_sparse_moe.gate.weight",
        &[3, 8],
        2,
    );
    for expert in 0..3 {
        let prefix = format!("model.layers.0.block_sparse_moe.experts.{expert}");
        insert(
            &mut weights,
            format!("{prefix}.w1.weight"),
            &[12, 8],
            10 + expert,
        );
        insert(
            &mut weights,
            format!("{prefix}.w3.weight"),
            &[12, 8],
            20 + expert,
        );
        insert(
            &mut weights,
            format!("{prefix}.w2.weight"),
            &[8, 12],
            30 + expert,
        );
    }
    weights.insert("model.norm.weight".into(), ones(&[8]));
    insert(&mut weights, "lm_head.weight", &[16, 8], 3);

    let model = MixtralModel::from_weights(&weights, &args).unwrap();
    assert_contract(&model);
}

#[test]
fn phimoe_native_batched_decode_contract() {
    use super::phimoe::{ModelArgs, PhiMoeModel};

    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "phimoe",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_local_experts": 3,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "original_max_position_embeddings": 2,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0, 1.0],
            "long_factor": [2.0, 4.0],
            "short_mscale": 1.0,
            "long_mscale": 1.0
        }
    }))
    .unwrap();
    let mut weights = WeightMap::new();
    insert(&mut weights, "model.embed_tokens.weight", &[16, 8], 1);
    insert_attention(&mut weights, "model.layers.0.self_attn", 8, 4, true);
    weights.insert("model.layers.0.input_layernorm.weight".into(), ones(&[8]));
    weights.insert(
        "model.layers.0.post_attention_layernorm.weight".into(),
        ones(&[8]),
    );
    insert(
        &mut weights,
        "model.layers.0.block_sparse_moe.gate.weight",
        &[3, 8],
        2,
    );
    insert_switch_glu(
        &mut weights,
        "model.layers.0.block_sparse_moe.switch_mlp",
        3,
        8,
        12,
    );
    weights.insert("model.norm.weight".into(), ones(&[8]));
    insert(&mut weights, "lm_head.weight", &[16, 8], 3);

    let model = PhiMoeModel::from_weights(&weights, &args).unwrap();
    assert_eq!(LanguageModel::rope_table_regime(&model, 2), Some(0));
    assert_eq!(LanguageModel::rope_table_regime(&model, 3), Some(1));
    // At decode, the length-2 row crosses onto the long table while the
    // length-1 row remains on the short table. The third row gives the MoE a
    // six-slot (B * top-k) grid, larger than its three-expert population.
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1, 2], &[3], &[4, 5, 6]],
        &[7, 8, 9],
        1e-5,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        &model,
        &[1, 3, 5],
        7,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}

#[test]
fn qwen2_moe_native_batched_decode_contract() {
    use super::qwen2_moe::{ModelArgs, Qwen2MoeModel};

    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "qwen2_moe",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts_per_tok": 2,
        "num_experts": 3,
        "moe_intermediate_size": 12,
        "shared_expert_intermediate_size": 10,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": false
    }))
    .unwrap();
    let mut weights = WeightMap::new();
    insert(&mut weights, "model.embed_tokens.weight", &[16, 8], 1);
    insert_attention(&mut weights, "model.layers.0.self_attn", 8, 4, true);
    weights.insert("model.layers.0.input_layernorm.weight".into(), ones(&[8]));
    weights.insert(
        "model.layers.0.post_attention_layernorm.weight".into(),
        ones(&[8]),
    );
    insert(&mut weights, "model.layers.0.mlp.gate.weight", &[3, 8], 2);
    insert_switch_glu(&mut weights, "model.layers.0.mlp.switch_mlp", 3, 8, 12);
    for (offset, (name, shape)) in [
        ("gate_proj", vec![10, 8]),
        ("up_proj", vec![10, 8]),
        ("down_proj", vec![8, 10]),
    ]
    .into_iter()
    .enumerate()
    {
        insert(
            &mut weights,
            format!("model.layers.0.mlp.shared_expert.{name}.weight"),
            &shape,
            220 + offset as u32,
        );
    }
    insert(
        &mut weights,
        "model.layers.0.mlp.shared_expert_gate.weight",
        &[1, 8],
        230,
    );
    weights.insert("model.norm.weight".into(), ones(&[8]));
    insert(&mut weights, "lm_head.weight", &[16, 8], 3);

    let model = Qwen2MoeModel::from_weights(&weights, &args).unwrap();
    assert_contract(&model);
}

#[test]
fn olmoe_native_batched_decode_contract() {
    use super::olmoe::{ModelArgs, OlmoeModel};

    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "olmoe",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "norm_topk_prob": false,
        "tie_word_embeddings": true
    }))
    .unwrap();
    let mut weights = WeightMap::new();
    insert(&mut weights, "model.embed_tokens.weight", &[16, 8], 1);
    insert_attention(&mut weights, "model.layers.0.self_attn", 8, 8, false);
    weights.insert("model.layers.0.self_attn.q_norm.weight".into(), ones(&[8]));
    weights.insert("model.layers.0.self_attn.k_norm.weight".into(), ones(&[8]));
    weights.insert("model.layers.0.input_layernorm.weight".into(), ones(&[8]));
    weights.insert(
        "model.layers.0.post_attention_layernorm.weight".into(),
        ones(&[8]),
    );
    insert(&mut weights, "model.layers.0.mlp.gate.weight", &[3, 8], 2);
    insert_switch_glu(&mut weights, "model.layers.0.mlp.switch_mlp", 3, 8, 12);
    weights.insert("model.norm.weight".into(), ones(&[8]));

    let model = OlmoeModel::from_weights(&weights, &args).unwrap();
    assert_contract(&model);
}

#[test]
fn deepseek_native_batched_decode_contract() {
    use super::deepseek::{DeepSeekModel, FeedForward, ModelArgs};

    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "deepseek",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "moe_intermediate_size": 10,
        "n_shared_experts": 1,
        "n_routed_experts": 3,
        "num_experts_per_tok": 2,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": 1.0
    }))
    .unwrap();
    let mut weights = WeightMap::new();
    insert(&mut weights, "model.embed_tokens.weight", &[16, 8], 1);
    for layer in 0..2 {
        insert_attention(
            &mut weights,
            &format!("model.layers.{layer}.self_attn"),
            8,
            4,
            false,
        );
        weights.insert(
            format!("model.layers.{layer}.input_layernorm.weight"),
            ones(&[8]),
        );
        weights.insert(
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            ones(&[8]),
        );
    }
    for (offset, (name, shape)) in [
        ("gate_proj", vec![12, 8]),
        ("up_proj", vec![12, 8]),
        ("down_proj", vec![8, 12]),
    ]
    .into_iter()
    .enumerate()
    {
        insert(
            &mut weights,
            format!("model.layers.0.mlp.{name}.weight"),
            &shape,
            235 + offset as u32,
        );
    }
    insert(&mut weights, "model.layers.1.mlp.gate.weight", &[3, 8], 2);
    insert_switch_glu(&mut weights, "model.layers.1.mlp.switch_mlp", 3, 8, 10);
    for (offset, (name, shape)) in [
        ("gate_proj", vec![10, 8]),
        ("up_proj", vec![10, 8]),
        ("down_proj", vec![8, 10]),
    ]
    .into_iter()
    .enumerate()
    {
        insert(
            &mut weights,
            format!("model.layers.1.mlp.shared_experts.{name}.weight"),
            &shape,
            240 + offset as u32,
        );
    }
    weights.insert("model.norm.weight".into(), ones(&[8]));
    insert(&mut weights, "lm_head.weight", &[16, 8], 3);

    let model = DeepSeekModel::from_weights(&weights, &args).unwrap();
    assert!(matches!(
        &model.layers[0].feed_forward,
        FeedForward::Dense(_)
    ));
    assert!(matches!(&model.layers[1].feed_forward, FeedForward::MoE(_)));
    assert_contract(&model);
}
