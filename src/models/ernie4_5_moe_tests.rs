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
//! ERNIE-4.5 MoE `SwitchLinear` loader applies before it stores them on a
//! quantized expert plane (issue #958).

use mlxcel_core::weights::WeightMap;

use super::{Ernie45MoeModel, ModelArgs, SwitchLinear};
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

/// Smallest ERNIE-4.5 MoE config that parses, varying only the declared
/// quantization pair. ERNIE spells it as flat top-level `group_size` / `bits`
/// keys rather than a nested `quantization` block. Built through serde rather
/// than a struct literal so every `#[serde(default)]` field fills itself in.
fn args_with(group_size: i32, bits: i32) -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "ernie4_5_moe",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "moe_num_experts": 3,
        "group_size": group_size,
        "bits": bits,
    }))
    .expect("test config must parse")
}

/// The pair this loader stores is handed straight to `gather_qmm`, which
/// crosses the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`. A C++
/// throw there is an uncatchable `std::terminate`,
/// so losing the bound turns a rejected load into an uncatchable abort at the
/// first routed forward pass in production. This test asserts on the load
/// result rather than running a forward pass, so a regression fails cleanly
/// here instead of aborting the test binary.
#[test]
fn ernie4_5_moe_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
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
    SwitchLinear::from_weights(&weights, &args_with(GROUP_SIZE, BITS), PREFIX)
        .expect("honest 4-bit expert plane must load");

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match SwitchLinear::from_weights(&weights, &args_with(group_size, bits), PREFIX) {
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
    SwitchLinear::from_weights(&regular, &args_with(0, 0), PREFIX)
        .expect("a bf16 expert plane must not be gated on quantization params");
}

fn tiny_routed_args() -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "ernie4_5_moe",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": 24,
        "moe_num_experts": 3,
        "moe_layer_start_index": 1,
        "moe_intermediate_size": 6,
        "moe_k": 2,
        "moe_num_shared_experts": 1,
        "tie_word_embeddings": false
    }))
    .unwrap()
}

fn tensor(shape: &[i32], phase: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|i| ((i as usize + phase) as f32).sin() * 0.08)
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn tiny_routed_weights(args: &ModelArgs) -> WeightMap {
    let h = args.hidden_size as i32;
    let hd = args.head_dim() as i32;
    let q = args.num_attention_heads as i32 * hd;
    let kv = args.num_key_value_heads as i32 * hd;
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
        put(format!("{a}.k_proj.weight"), &[kv, h]);
        put(format!("{a}.v_proj.weight"), &[kv, h]);
        put(format!("{a}.o_proj.weight"), &[h, q]);
        put(format!("{p}.input_layernorm.weight"), &[h]);
        put(format!("{p}.post_attention_layernorm.weight"), &[h]);
        let m = format!("{p}.mlp");
        if args.is_moe_layer(layer) {
            let e = args.moe_num_experts as i32;
            let d = args.moe_intermediate_size() as i32;
            put(format!("{m}.gate.weight"), &[e, h]);
            put(format!("{m}.switch_mlp.gate_proj.weight"), &[e, d, h]);
            put(format!("{m}.switch_mlp.up_proj.weight"), &[e, d, h]);
            put(format!("{m}.switch_mlp.down_proj.weight"), &[e, h, d]);
            for shared in 0..args.moe_num_shared_experts {
                let s = format!("{m}.shared_experts.{shared}");
                put(format!("{s}.gate_proj.weight"), &[d, h]);
                put(format!("{s}.up_proj.weight"), &[d, h]);
                put(format!("{s}.down_proj.weight"), &[h, d]);
            }
        } else {
            let d = args.intermediate_size as i32;
            put(format!("{m}.gate_proj.weight"), &[d, h]);
            put(format!("{m}.up_proj.weight"), &[d, h]);
            put(format!("{m}.down_proj.weight"), &[h, d]);
        }
    }
    weights
}

#[test]
fn ernie_native_batched_decode_matches_three_routed_rows() {
    let args = tiny_routed_args();
    let model = Ernie45MoeModel::from_weights(&tiny_routed_weights(&args), &args).unwrap();
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1, 2], &[3, 4, 5], &[6, 7, 8, 9]],
        &[10, 11, 12],
        2e-4,
    );
    crate::models::batched_decode_harness::assert_single_row_batch_is_single_sequence_forward(
        &model,
        &[2, 4, 6],
        8,
    );
    crate::models::batched_decode_harness::assert_empty_batch_returns_empty_logits(&model);
}
