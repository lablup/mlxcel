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

use super::{Ernie45MoeModel, MoEBlock, ModelArgs, SwitchLinear};
use crate::models::switch_layers::{HOSTILE_QUANT_PARAMS, insert_stacked_quantized_expert_plane};
use mlxcel_core::{dtype, utils::array_to_vec_f32};

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

fn tiny_routed_config() -> serde_json::Value {
    serde_json::json!({
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
    })
}

fn tiny_routed_args() -> ModelArgs {
    serde_json::from_value(tiny_routed_config()).unwrap()
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
fn correction_bias_changes_selection_not_weights() {
    let mut args = tiny_routed_args();
    args.hidden_size = 4;
    let mut weights = tiny_routed_weights(&args);
    weights.insert(
        "model.layers.1.mlp.gate.weight".into(),
        mlxcel_core::from_slice_f32(&[1., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.], &[3, 4]),
    );
    weights.insert(
        "model.layers.1.mlp.e_score_correction_bias".into(),
        mlxcel_core::from_slice_f32(&[0., 0., 10.], &[3]),
    );
    let x = mlxcel_core::from_slice_f32(&[2., 0., 0., 0.], &[1, 4]);

    for softmax in [true, false] {
        let mut block = MoEBlock::from_weights(&weights, &args, "model.layers.1").unwrap();
        block.use_softmax = softmax;
        let (indices, scores) = block.route(&x);
        let mut selected: Vec<_> = array_to_vec_f32(&indices)
            .into_iter()
            .map(|i| i as usize)
            .zip(array_to_vec_f32(&scores))
            .collect();
        selected.sort_by_key(|&(expert, _)| expert);
        assert_eq!([selected[0].0, selected[1].0], [0, 2]);

        // The raw logits of the selected experts are 2 and 0.
        let (g0, g2) = if softmax {
            (2_f32.exp(), 1.)
        } else {
            (1. / (1. + (-2_f32).exp()), 0.5)
        };
        assert!((selected[0].1 - g0 / (g0 + g2)).abs() < 1e-6);
        assert!((selected[1].1 - g2 / (g0 + g2)).abs() < 1e-6);
    }
}

#[test]
fn correction_bias_loads_from_moe_statics_and_squeezes() {
    let mut args = tiny_routed_args();
    args.moe_num_experts = 4;
    let mut weights = tiny_routed_weights(&args);
    let prefix = "model.layers.1";
    let raw = format!("{prefix}.mlp.moe_statics.e_score_correction_bias");
    let converted = format!("{prefix}.mlp.e_score_correction_bias");
    let check = |weights: &WeightMap, expected: &[f32]| {
        let block = MoEBlock::from_weights(weights, &args, prefix).unwrap();
        let bias = &block.e_score_correction_bias;
        assert_eq!(mlxcel_core::array_shape(bias), [4]);
        assert_eq!(mlxcel_core::array_dtype(bias), dtype::FLOAT32);
        assert_eq!(array_to_vec_f32(bias), expected);
    };

    check(&weights, &[0.; 4]);
    let values = [0.25, -0.5, 1.0, 2.0];
    weights.insert(
        raw.clone(),
        mlxcel_core::astype(
            &mlxcel_core::from_slice_f32(&values, &[1, 4]),
            dtype::BFLOAT16,
        ),
    );
    check(&weights, &values);
    // The converted key takes precedence when both exist.
    weights.insert(converted.clone(), mlxcel_core::ones(&[4], dtype::FLOAT16));
    check(&weights, &[1.; 4]);
    weights.remove(&raw);
    check(&weights, &[1.; 4]);

    for shape in [&[2, 4][..], &[3], &[1, 3]] {
        weights.insert(converted.clone(), mlxcel_core::zeros(shape, dtype::FLOAT32));
        let err = MoEBlock::from_weights(&weights, &args, prefix)
            .err()
            .expect("malformed bias must fail to load");
        assert!(err.contains(&converted), "{err}");
    }
}

#[test]
fn router_runs_in_f32() {
    let mut args = tiny_routed_args();
    args.hidden_size = 2;
    args.moe_num_experts = 2;
    args.moe_k = 1;
    let mut weights = tiny_routed_weights(&args);
    // The f32 dot products are 1 and 1 + 2^-12. Both round to 1 in f16,
    // even though each input and gate weight is exactly representable in f16.
    let gate = mlxcel_core::from_slice_f32(&[1., 0., 1., 1. / 4096.], &[2, 2]);
    for gate_dtype in [dtype::FLOAT16, dtype::BFLOAT16] {
        let weight = mlxcel_core::astype(&gate, gate_dtype);
        let x = mlxcel_core::ones(&[1, 2], gate_dtype);
        let low_precision = mlxcel_core::matmul(&x, &mlxcel_core::transpose(&weight));
        assert_eq!(array_to_vec_f32(&low_precision), [1., 1.]);
        weights.insert("model.layers.1.mlp.gate.weight".into(), weight);
        let block = MoEBlock::from_weights(&weights, &args, "model.layers.1").unwrap();
        let (indices, scores) = block.route(&x);
        assert_eq!(mlxcel_core::item_i32(&indices), 1);
        assert_eq!(mlxcel_core::array_dtype(&scores), dtype::FLOAT32);
    }
}

#[test]
fn quantized_router_runs_in_f32_with_bf16_scales() {
    let mut args = tiny_routed_args();
    args.hidden_size = 64;
    args.moe_num_experts = 2;
    args.moe_k = 1;
    let mut weights = tiny_routed_weights(&args);
    let gate = "model.layers.1.mlp.gate";
    // Affine 4-bit rows [1, 0, ...] and [1, 1, 0, ...].
    let mut packed = [0; 16];
    packed[0] = 0x01;
    packed[8] = 0x11;
    weights.insert(
        format!("{gate}.weight"),
        mlxcel_core::from_slice_u32(&packed, &[2, 8]),
    );
    weights.insert(
        format!("{gate}.scales"),
        mlxcel_core::ones(&[2, 1], dtype::BFLOAT16),
    );
    weights.insert(
        format!("{gate}.biases"),
        mlxcel_core::zeros(&[2, 1], dtype::BFLOAT16),
    );
    let block = MoEBlock::from_weights(&weights, &args, "model.layers.1").unwrap();
    let mut values = [0.; 64];
    values[0] = 1.;
    values[1] = 1. / 4096.;
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&values, &[1, 64]),
        dtype::BFLOAT16,
    );
    let (indices, scores) = block.route(&x);
    assert_eq!(mlxcel_core::item_i32(&indices), 1);
    assert_eq!(mlxcel_core::array_dtype(&scores), dtype::FLOAT32);
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

#[test]
fn ernie45_moe_last_row_logits_match_the_full_forward() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::last_row_logits_harness::assert_last_row_logits(
        || {
            let args = tiny_routed_args();
            Ernie45MoeModel::from_weights(&tiny_routed_weights(&args), &args).unwrap()
        },
        crate::models::last_row_logits_harness::TOKENS,
    );
}

#[test]
fn checkpoint_layout_loads_fused_shared_glu_at_stored_width() {
    let mut args = tiny_routed_args();
    args.moe_num_shared_experts = 0;
    let mut weights = tiny_routed_weights(&args);
    args.moe_num_shared_experts = 2;
    let h = args.hidden_size as i32;
    let width = 2 * args.moe_intermediate_size() as i32;
    for (proj, shape) in [
        ("gate_proj", [width, h]),
        ("up_proj", [width, h]),
        ("down_proj", [h, width]),
    ] {
        weights.insert(
            format!("model.layers.1.mlp.shared_experts.{proj}.weight"),
            tensor(&shape, 1),
        );
    }
    let model = Ernie45MoeModel::from_weights(&weights, &args).unwrap();
    let super::MLPVariant::MoE(block) = &model.layers[1].mlp else {
        panic!("expected routed layer")
    };
    let shared = &block.shared_experts.as_ref().unwrap().experts;
    assert_eq!(shared.len(), 1);
    let input = mlxcel_core::ones(&[1, h], dtype::FLOAT32);
    assert_eq!(
        mlxcel_core::array_shape(&shared[0].gate_proj.forward(&input)),
        [1, width]
    );
    assert_eq!(mlxcel_core::array_shape(&shared[0].forward(&input)), [1, h]);
}

#[test]
fn checkpoint_layout_stacks_numbered_experts_in_order() {
    check_numbered_expert_planes(false);
}

#[test]
fn checkpoint_layout_stacks_affine_expert_planes_in_order() {
    check_numbered_expert_planes(true);
}

fn check_numbered_expert_planes(quantized: bool) {
    let mut args = tiny_routed_args();
    args.hidden_size = 64;
    args.moe_intermediate_size = Some(64);
    let mut weights = tiny_routed_weights(&args);
    let root = "model.layers.1.mlp";
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        weights.remove(&format!("{root}.switch_mlp.{proj}.weight"));
        for e in (0..args.moe_num_experts).rev() {
            let key = format!("{root}.experts.{e}.{proj}");
            let plane = if quantized {
                weights.insert(
                    format!("{key}.scales"),
                    mlxcel_core::full_f32(&[64, 1], e as f32 + 1., dtype::FLOAT32),
                );
                weights.insert(
                    format!("{key}.biases"),
                    mlxcel_core::full_f32(&[64, 1], -(e as f32), dtype::FLOAT32),
                );
                mlxcel_core::from_slice_u32(&[e as u32; 64 * 8], &[64, 8])
            } else {
                mlxcel_core::full_f32(&[64, 64], e as f32, dtype::FLOAT32)
            };
            weights.insert(format!("{key}.weight"), plane);
        }
    }
    let model = Ernie45MoeModel::from_weights(&weights, &args).unwrap();
    let super::MLPVariant::MoE(block) = &model.layers[1].mlp else {
        panic!("expected routed layer")
    };
    let experts = &block.switch_mlp;
    for proj in [&experts.gate_proj, &experts.up_proj, &experts.down_proj] {
        let check = |array: &mlxcel_core::MlxArray, cols, expected: [f32; 3]| {
            assert_eq!(mlxcel_core::array_shape(array), [3, 64, cols]);
            let first = mlxcel_core::slice(array, &[0, 0, 0], &[3, 1, 1]);
            assert_eq!(array_to_vec_f32(&first), expected);
        };
        match proj {
            SwitchLinear::Regular { weight, .. } => {
                assert!(!quantized);
                check(weight, 64, [0., 1., 2.]);
            }
            SwitchLinear::Quantized {
                weight,
                scales,
                biases,
                ..
            } => {
                assert!(quantized);
                assert_eq!(mlxcel_core::array_dtype(weight), dtype::UINT32);
                check(weight, 8, [0., 1., 2.]);
                check(scales, 1, [1., 2., 3.]);
                check(biases, 1, [0., -1., -2.]);
            }
        }
    }
}

#[test]
fn checkpoint_layout_drops_mtp_and_preserves_numbered_shared_and_stacked_routed() {
    let mut config = tiny_routed_config();
    config["moe_num_shared_experts"] = 2.into();
    let mut weights = tiny_routed_weights(&serde_json::from_value(config.clone()).unwrap());
    let mtp = [
        "model.mtp_block.0.mlp.gate_proj.weight",
        "model.mtp_linear_proj.0.weight",
        "model.mtp_hidden_norm.0.weight",
        "model.mtp_emb_norm.0.weight",
    ];
    for key in mtp {
        weights.insert(key.into(), tensor(&[1], 1));
    }
    let model = crate::loading::config_backed::try_load_config_backed_model_from_weights(
        crate::models::ModelType::Ernie45Moe,
        &config.to_string(),
        &mut weights,
    )
    .unwrap();
    let Some(crate::LoadedModel::Ernie45Moe(model)) = model else {
        panic!("expected ERNIE text MoE")
    };
    let super::MLPVariant::MoE(block) = &model.layers[1].mlp else {
        panic!("expected routed layer")
    };
    assert_eq!(block.shared_experts.as_ref().unwrap().experts.len(), 2);
    assert_eq!(block.switch_mlp.gate_proj.num_experts(), 3);
    for key in mtp {
        assert!(!weights.contains_key(key), "retained {key}");
    }
}

#[test]
fn checkpoint_layout_refuses_malformed_numbered_experts() {
    let args = tiny_routed_args();
    let root = "model.layers.1.mlp";
    let numbered = |dtype_of: &dyn Fn(usize) -> i32, count: usize| {
        let mut weights = tiny_routed_weights(&args);
        weights.remove(&format!("{root}.switch_mlp.gate_proj.weight"));
        for e in 0..count {
            let plane = mlxcel_core::full_f32(&[6, 8], e as f32, dtype::FLOAT32);
            weights.insert(
                format!("{root}.experts.{e}.gate_proj.weight"),
                mlxcel_core::astype(&plane, dtype_of(e)),
            );
        }
        weights
    };
    let load_err = |weights: &WeightMap| {
        Ernie45MoeModel::from_weights(weights, &args)
            .err()
            .expect("malformed numbered experts must fail to load")
    };

    // A plane in another dtype reaches the shared stacking helper, whose error must surface.
    let mixed = numbered(
        &|e| {
            if e == 2 {
                dtype::FLOAT16
            } else {
                dtype::FLOAT32
            }
        },
        3,
    );
    let err = load_err(&mixed);
    assert!(
        err.contains(&format!("{root}.experts.2.gate_proj.weight")) && err.contains("dtype"),
        "{err}"
    );

    let short = numbered(&|_| dtype::FLOAT32, 2);
    let err = load_err(&short);
    assert!(
        err.contains(&format!("{root}.experts.2.gate_proj.weight")),
        "{err}"
    );

    let long = numbered(&|_| dtype::FLOAT32, 4);
    let err = load_err(&long);
    assert!(err.contains("expected 3 experts"), "{err}");
}
