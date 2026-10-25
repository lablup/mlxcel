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

//! Regression test for the bound on declared quantization params that
//! `glm4_moe_lite::SwitchLinear::from_weights` applies before it stores them on
//! a quantized expert plane (issue #958).

use super::{Glm4MoeLiteModel, ModelArgs, SwitchLinear};
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};
use mlxcel_core::weights::WeightMap;

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

/// The pair this loader stores is handed straight to `gather_qmm`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
#[test]
fn glm4_moe_lite_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
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
    match SwitchLinear::from_weights(&weights, PREFIX, GROUP_SIZE, BITS) {
        Ok(_) => {}
        Err(e) => panic!("honest 4-bit expert plane must load: {e}"),
    }

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(&weights, PREFIX, group_size, bits) {
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
        assert!(
            err.contains(PREFIX),
            "the load error must name the offending tensor {PREFIX}, got: {err}"
        );
    }

    // A bf16 expert plane carries no packing and no `.scales`, so the declared
    // pair is inert on the `gather_mm` path and must not gate the load.
    let mut dense = WeightMap::new();
    let n = (EXPERTS * OUT * PACKED_IN) as usize;
    dense.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::from_slice_f32(&vec![0.0f32; n], &[EXPERTS, OUT, PACKED_IN]),
    );
    match SwitchLinear::from_weights(&dense, PREFIX, 0, 0) {
        Ok(_) => {}
        Err(e) => panic!("a non-quantized expert plane must load with an unset pair: {e}"),
    }
}

fn put(weights: &mut WeightMap, key: &str, shape: &[i32]) {
    let n = shape.iter().product::<i32>() as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 17 + 7) % 41) as f32 * 0.002 - 0.04)
        .collect();
    weights.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

fn tiny_batched_model() -> Glm4MoeLiteModel {
    let args = ModelArgs {
        model_type: "glm4_moe_lite".to_string(),
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 8,
        moe_intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        kv_lora_rank: 4,
        q_lora_rank: None,
        qk_rope_head_dim: 2,
        qk_nope_head_dim: 2,
        v_head_dim: 4,
        n_routed_experts: None,
        num_experts_per_tok: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
        norm_topk_prob: true,
        first_k_dense_replace: 0,
        moe_layer_freq: 1,
        n_shared_experts: None,
        max_position_embeddings: 64,
        partial_rotary_factor: 1.0,
        attention_bias: false,
        tie_word_embeddings: false,
        group_size: None,
        bits: None,
        num_nextn_predict_layers: 0,
    };
    let mut weights = WeightMap::new();
    put(&mut weights, "model.embed_tokens.weight", &[16, 8]);
    put(&mut weights, "model.norm.weight", &[8]);
    put(&mut weights, "lm_head.weight", &[16, 8]);
    let p = "model.layers.0";
    put(&mut weights, &format!("{p}.input_layernorm.weight"), &[8]);
    put(
        &mut weights,
        &format!("{p}.post_attention_layernorm.weight"),
        &[8],
    );
    let a = format!("{p}.self_attn");
    put(&mut weights, &format!("{a}.q_proj.weight"), &[8, 8]);
    put(
        &mut weights,
        &format!("{a}.kv_a_proj_with_mqa.weight"),
        &[6, 8],
    );
    put(&mut weights, &format!("{a}.kv_a_layernorm.weight"), &[4]);
    put(&mut weights, &format!("{a}.embed_q.weight"), &[2, 4, 2]);
    put(&mut weights, &format!("{a}.unembed_out.weight"), &[2, 4, 4]);
    put(&mut weights, &format!("{a}.o_proj.weight"), &[8, 8]);
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        put(&mut weights, &format!("{p}.mlp.{proj}.weight"), &[8, 8]);
    }
    Glm4MoeLiteModel::from_weights(&weights, &args).expect("tiny GLM4 MoE Lite loads")
}

#[test]
fn glm4_moe_lite_batched_decode_harness_covers_ragged_single_and_empty() {
    let _runtime = crate::initialize_runtime();
    let model = tiny_batched_model();
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1], &[2, 3], &[4, 5, 6]],
        &[7, 8, 9],
        1e-5,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        &model,
        &[1, 2],
        3,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}
