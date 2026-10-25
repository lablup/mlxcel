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
//! `hunyuan_moe::SwitchLinear::from_weights` applies before it stores them on a
//! quantized expert plane (issue #958).

use super::{HunyuanMoeModel, ModelArgs, SwitchLinear};
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
fn hunyuan_moe_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
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

fn tiny_cla_args() -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "hunyuan_moe",
        "vocab_size": 24,
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "intermediate_size": 10,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "num_experts": 3,
        "moe_topk": [2, 2, 2, 2],
        "moe_intermediate_size": [6, 6, 6, 6],
        "num_shared_expert": [1, 1, 1, 1],
        "use_qk_norm": true,
        "use_mixed_mlp_moe": true,
        "use_cla": true,
        "cla_share_factor": 2,
        "tie_word_embeddings": false,
        "head_dim": 4
    }))
    .unwrap()
}

fn tensor(shape: &[i32], phase: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|i| ((i as usize + phase * 3) as f32).cos() * 0.07)
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn tiny_cla_weights(args: &ModelArgs) -> WeightMap {
    let h = args.hidden_size as i32;
    let hd = args.head_dim() as i32;
    let q = args.num_attention_heads as i32 * hd;
    let kv = args.num_key_value_heads as i32 * hd;
    let experts = args.num_experts as i32;
    let mut weights = WeightMap::new();
    let mut phase = 1;
    let mut put = |key: String, shape: &[i32]| {
        weights.insert(key, tensor(shape, phase));
        phase += 1;
    };
    put(
        "model.embed_tokens.weight".into(),
        &[args.vocab_size as i32, h],
    );
    put("model.norm.weight".into(), &[h]);
    put("lm_head.weight".into(), &[args.vocab_size as i32, h]);
    for layer in 0..args.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        let a = format!("{p}.self_attn");
        put(format!("{a}.q_proj.weight"), &[q, h]);
        if args.has_kv_proj(layer) {
            put(format!("{a}.k_proj.weight"), &[kv, h]);
            put(format!("{a}.v_proj.weight"), &[kv, h]);
        }
        put(format!("{a}.o_proj.weight"), &[h, q]);
        put(format!("{a}.query_layernorm.weight"), &[hd]);
        put(format!("{a}.key_layernorm.weight"), &[hd]);
        put(format!("{p}.input_layernorm.weight"), &[h]);
        put(format!("{p}.post_attention_layernorm.weight"), &[h]);
        let m = format!("{p}.mlp");
        let d = args.moe_intermediate_size.get(layer) as i32;
        put(format!("{m}.shared_mlp.gate_proj.weight"), &[d, h]);
        put(format!("{m}.shared_mlp.up_proj.weight"), &[d, h]);
        put(format!("{m}.shared_mlp.down_proj.weight"), &[h, d]);
        put(format!("{m}.gate.wg.weight"), &[experts, h]);
        put(format!("{m}.switch_mlp.gate_proj.weight"), &[experts, d, h]);
        put(format!("{m}.switch_mlp.up_proj.weight"), &[experts, d, h]);
        put(format!("{m}.switch_mlp.down_proj.weight"), &[experts, h, d]);
    }
    weights
}

#[test]
fn hunyuan_cla_native_batch_uses_both_owner_groups_and_isolates_every_cache() {
    let args = tiny_cla_args();
    let model = HunyuanMoeModel::from_weights(&tiny_cla_weights(&args), &args).unwrap();
    assert!(model.layers[0].has_kv_proj);
    assert!(!model.layers[1].has_kv_proj);
    assert!(model.layers[2].has_kv_proj);
    assert!(!model.layers[3].has_kv_proj);
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1, 2], &[3, 4, 5], &[6, 7, 8, 9]],
        &[10, 11, 12],
        3e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        &model,
        &[2, 4, 6],
        8,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}

#[test]
fn hunyuan_cla_missing_owner_projection_names_the_layer() {
    let args = tiny_cla_args();
    let mut model = HunyuanMoeModel::from_weights(&tiny_cla_weights(&args), &args).unwrap();
    model.layers[2].self_attn.k_proj = None;
    let input = mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]);
    let mut row0 = model.make_caches();
    let mut row1 = model.make_caches();
    let mut rows = [row0.as_mut_slice(), row1.as_mut_slice()];
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.forward_batched_impl(&input, &mut rows, None);
    }))
    .expect_err("a CLA owner without K projection must fail before attention");
    let message = if let Some(message) = panic.downcast_ref::<String>() {
        message.as_str()
    } else if let Some(message) = panic.downcast_ref::<&str>() {
        message
    } else {
        "non-string panic"
    };
    assert!(message.contains("HunyuanMoeModel"), "{message}");
    assert!(message.contains("owner layer 2"), "{message}");
    assert!(message.contains("K projection"), "{message}");
}
