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

//! Regression test for the bound on declared quantization params that the
//! Qwen3-Next `SwitchLinear` loader applies before it stores them on a
//! quantized expert plane (issue #958).

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;

use super::{Qwen3NextCache, Qwen3NextConfig, Qwen3NextModel, SwitchLinear};
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};

/// Honest 4-bit expert geometry: `packed_in * 32 == bits * num_groups *
/// group_size` (8 * 32 == 4 * 1 * 64), so the positive control below is a plane
/// MLX can actually describe.
const EXPERTS: i32 = 3;
const OUT: i32 = 4;
const PACKED_IN: i32 = 8;
const NUM_GROUPS: i32 = 1;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

/// Smallest Qwen3-Next config that parses, varying only the declared
/// quantization pair. Qwen3-Next carries it in a nested `quantization` block.
/// Built through serde rather than a struct literal so every
/// `#[serde(default)]` field fills itself in.
fn config_with(group_size: i32, bits: i32) -> Qwen3NextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "num_experts": 3,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "vocab_size": 32,
        "quantization": { "group_size": group_size, "bits": bits },
    }))
    .expect("test config must parse")
}

fn recurrent_config() -> Qwen3NextConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 8,
        "vocab_size": 32,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "num_experts": 0,
        "num_experts_per_tok": 1,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "tie_word_embeddings": true,
    }))
    .expect("test config must parse")
}

fn deterministic_values(seed: u32, count: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            let value = (unit * 2.0 - 1.0) * scale;
            if value.abs() < 1e-5 {
                scale / 17.0
            } else {
                value
            }
        })
        .collect()
}

fn recurrent_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    let mut seed = 2285u32;
    let mut tensor = |name: &str, shape: &[i32], scale: f32| {
        seed += 1;
        let count = shape.iter().product::<i32>() as usize;
        weights.insert(
            name.to_string(),
            mlxcel_core::from_slice_f32(&deterministic_values(seed, count, scale), shape),
        );
    };

    tensor("model.embed_tokens.weight", &[32, 8], 0.2);
    tensor("model.layers.0.linear_attn.conv1d.weight", &[16, 4, 1], 0.2);
    tensor(
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        &[24, 8],
        0.2,
    );
    tensor("model.layers.0.linear_attn.in_proj_ba.weight", &[4, 8], 0.2);
    tensor("model.layers.0.linear_attn.dt_bias", &[2], 0.1);
    tensor("model.layers.0.linear_attn.A_log", &[2], 0.1);
    tensor("model.layers.0.linear_attn.norm.weight", &[4], 0.2);
    tensor("model.layers.0.linear_attn.out_proj.weight", &[8, 8], 0.2);
    tensor("model.layers.0.mlp.gate_proj.weight", &[8, 8], 0.2);
    tensor("model.layers.0.mlp.up_proj.weight", &[8, 8], 0.2);
    tensor("model.layers.0.mlp.down_proj.weight", &[8, 8], 0.2);
    tensor("model.layers.0.input_layernorm.weight", &[8], 0.2);
    tensor("model.layers.0.post_attention_layernorm.weight", &[8], 0.2);
    tensor("model.norm.weight", &[8], 0.2);

    weights
}

fn tokens(values: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_i32(values, &[1, values.len() as i32])
}

fn linear_offset(caches: &[Qwen3NextCache]) -> i32 {
    match caches.first().expect("one-layer fixture has one cache") {
        Qwen3NextCache::Linear(cache) => cache.offset,
        Qwen3NextCache::Attention(_) => panic!("layer 0 must be GatedDeltaNet"),
    }
}

/// The pair this loader stores is handed straight to `gather_qmm`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
///
/// Three config sources reach this one loader (Qwen3-Next, Qwen3.5 through
/// `to_qwen3next_config`, and the synthesized bridge config in the Qwen3-Omni
/// talker), which is why the bound lives at the storage point rather than at
/// any single producer.
#[test]
fn qwen3_next_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
    let mut weights = WeightMap::new();
    insert_stacked_quantized_expert_plane(
        &mut weights,
        PREFIX,
        EXPERTS,
        OUT,
        PACKED_IN,
        NUM_GROUPS,
    );

    // Positive control first, so a guard that rejected every quantized plane
    // could not pass this test.
    SwitchLinear::from_weights(&weights, &config_with(GROUP_SIZE, BITS), PREFIX)
        .expect("honest 4-bit expert plane must load");

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(&weights, &config_with(group_size, bits), PREFIX)
        {
            Ok(_) => panic!(
                "(group_size {group_size}, bits {bits}) must be refused at load, \
                 not stored for gather_qmm"
            ),
            Err(e) => e,
        };
        assert!(
            err.contains(field),
            "(group_size {group_size}, bits {bits}) must be blamed on {field}, got: {err}"
        );
    }

    // A bf16 expert plane carries no packing at all, so the declared pair is
    // irrelevant there and must not gate the non-quantized fallback.
    let mut regular = WeightMap::new();
    regular.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::ones(&[EXPERTS, OUT, PACKED_IN], mlxcel_core::dtype::BFLOAT16),
    );
    SwitchLinear::from_weights(&regular, &config_with(0, 0), PREFIX)
        .expect("a bf16 expert plane must not be gated on quantization params");
}

#[test]
fn qwen3_next_loaded_model_chunking_contract_is_safe() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = Qwen3NextModel::from_weights(&recurrent_weights(), &recurrent_config())
        .expect("one-layer GatedDeltaNet model must load");
    let language_model: &dyn LanguageModel = &model;
    assert!(
        !language_model.supports_chunked_prefill(),
        "no-ID prompt chunks cannot retain the state built by earlier chunks"
    );

    let mut compatibility_caches = LanguageModel::make_caches(&model);
    let first_prompt =
        LanguageModel::forward(&model, &tokens(&[1, 2]), &mut compatibility_caches, None);
    mlxcel_core::eval(&first_prompt);
    let second_prompt =
        LanguageModel::forward(&model, &tokens(&[3, 4, 5]), &mut compatibility_caches, None);
    mlxcel_core::eval(&second_prompt);
    assert_eq!(
        model
            .sequence_state
            .with_sequence_state(None, |caches| linear_offset(caches)),
        3,
        "a second no-ID prompt must start from fresh model-owned state"
    );

    let seq_id = SequenceId::from_raw(2285);
    LanguageModel::prepare_sequence_state(&model, seq_id);
    let first_chunk = LanguageModel::forward_with_sequence_id(
        &model,
        &tokens(&[6, 7]),
        Some(seq_id),
        &mut compatibility_caches,
        None,
    );
    mlxcel_core::eval(&first_chunk);
    let second_chunk = LanguageModel::forward_with_sequence_id(
        &model,
        &tokens(&[8, 9, 10]),
        Some(seq_id),
        &mut compatibility_caches,
        None,
    );
    mlxcel_core::eval(&second_chunk);
    assert_eq!(
        model
            .sequence_state
            .with_sequence_state_ref(seq_id, linear_offset),
        Some(5),
        "stable sequence IDs must retain state across prefill chunks"
    );
}
