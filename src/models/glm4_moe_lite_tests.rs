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

/// Full-model logits with the absorbed/materializing choice pinned.
///
/// Mirrors `Glm4MoeLiteModel::forward_with_hidden`'s block glue (embed, mask,
/// residual + MLP per layer, final norm, lm_head) but calls the attention's
/// gate-pinnable entry, so one warm cache can be replayed through both
/// branches. Production `forward` is unchanged: it passes `None`.
fn logits_with_gate(
    model: &Glm4MoeLiteModel,
    input_ids: &mlxcel_core::MlxArray,
    caches: &mut [mlxcel_core::layers::KVCache],
    forced_absorbed: Option<bool>,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut h = model.embed_tokens.forward(input_ids);
    let seq_len = mlxcel_core::array_shape(&h)[1];
    let mask = if seq_len > 1 {
        let offset = caches.first().map(|c| c.seq_len()).unwrap_or(0);
        Some(mlxcel_core::utils::create_causal_mask(seq_len, offset))
    } else {
        None
    };
    for (i, layer) in model.layers.iter().enumerate() {
        let normed = layer.input_layernorm.forward(&h);
        let attn = layer.self_attn.forward_with_gate(
            &normed,
            &mut caches[i],
            mask.as_deref(),
            forced_absorbed,
        );
        let after_attn = mlxcel_core::add(&h, &attn);
        let normed = layer.post_attention_layernorm.forward(&after_attn);
        let mlp = layer.mlp.forward(&normed);
        h = mlxcel_core::add(&after_attn, &mlp);
    }
    let normed = model.norm.forward(&h);
    model.lm_head.forward(&normed)
}

#[test]
fn four_token_step_on_a_warm_cache_matches_across_both_attention_paths() {
    let _runtime = crate::initialize_runtime();
    let model = tiny_batched_model();

    // Warm both arms identically: 64 tokens on a cold cache, where the cost
    // model picks the materializing side at every layer.
    let warm: Vec<u32> = (0..64).map(|i| (i % 16) as u32).collect();
    let warm_ids = mlxcel_core::from_slice_u32(&warm, &[1, 64]);
    let step = mlxcel_core::from_slice_u32(&[3, 7, 11, 5], &[1, 4]);
    let mut absorbed_caches = model.make_caches();
    let mut materialized_caches = model.make_caches();
    for caches in [&mut absorbed_caches, &mut materialized_caches] {
        let out = logits_with_gate(&model, &warm_ids, caches, None);
        mlxcel_core::eval(&out);
    }

    // Tiny geometry: r 4, nope 2, v 4. Behind 64 warm tokens plus this step's
    // four, the bound is well above four, so the helper selects absorbed.
    assert!(
        mlxcel_core::mla::use_absorbed_mla(4, 68, 4, 2, 4),
        "the 4-token step on a 64-token cache must select the absorbed path"
    );

    let absorbed = logits_with_gate(&model, &step, &mut absorbed_caches, None);
    let materialized = logits_with_gate(&model, &step, &mut materialized_caches, Some(false));
    mlxcel_core::eval(&absorbed);
    mlxcel_core::eval(&materialized);
    let diff = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(
        &absorbed,
        &materialized,
    )));
    mlxcel_core::eval(&diff);
    let diff = mlxcel_core::item_f32(&diff);
    assert!(
        diff < 2e-2,
        "absorbed and materialized logits must agree, max |diff| = {diff}"
    );
}
