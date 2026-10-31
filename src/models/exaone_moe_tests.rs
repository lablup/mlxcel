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
//! `exaone_moe::load_switch_linear` applies before it stores them on a
//! quantized expert plane (issue #958).

use super::{AnyKVCache, ExaoneMoeModel, KVCacheTrait, ModelArgs, load_switch_linear};
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
fn exaone_moe_switch_linear_rejects_quantization_params_that_would_abort_gather_qmm() {
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
    match load_switch_linear(&weights, PREFIX, GROUP_SIZE, BITS) {
        Ok(_) => {}
        Err(e) => panic!("honest 4-bit expert plane must load: {e}"),
    }

    for (group_size, bits, field) in HOSTILE_QUANT_PARAMS {
        let err = match load_switch_linear(&weights, PREFIX, group_size, bits) {
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
    match load_switch_linear(&dense, PREFIX, 0, 0) {
        Ok(_) => {}
        Err(e) => panic!("a non-quantized expert plane must load with an unset pair: {e}"),
    }
}

fn tiny_args() -> ModelArgs {
    serde_json::from_value(serde_json::json!({
        "model_type": "exaone_moe",
        "vocab_size": 24,
        "hidden_size": 8,
        "intermediate_size": 10,
        "moe_intermediate_size": 6,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 64,
        "sliding_window": 4,
        "layer_types": ["full_attention", "sliding_attention"],
        "is_moe_layer": [false, true],
        "tie_word_embeddings": false
    }))
    .unwrap()
}

fn tensor(shape: &[i32], phase: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as usize + phase * 11) as f32).sin() * 0.05)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let h = args.hidden_size as i32;
    let q = args.num_attention_heads as i32 * args.head_dim as i32;
    let kv = args.num_key_value_heads as i32 * args.head_dim as i32;
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
        put(format!("{a}.q_norm.weight"), &[args.head_dim as i32]);
        put(format!("{a}.k_norm.weight"), &[args.head_dim as i32]);
        put(format!("{p}.input_layernorm.weight"), &[h]);
        put(format!("{p}.post_attention_layernorm.weight"), &[h]);
        let m = format!("{p}.mlp");
        if args.is_moe_layer[layer] {
            let e = args.num_experts as i32;
            let d = args.moe_intermediate_size as i32;
            put(format!("{m}.gate.weight.weight"), &[e, h]);
            put(format!("{m}.gate.e_score_correction_bias"), &[e]);
            put(format!("{m}.switch_mlp.gate_proj.weight"), &[e, d, h]);
            put(format!("{m}.switch_mlp.up_proj.weight"), &[e, d, h]);
            put(format!("{m}.switch_mlp.down_proj.weight"), &[e, h, d]);
            let shared = d * args.num_shared_experts as i32;
            put(format!("{m}.shared_experts.gate_proj.weight"), &[shared, h]);
            put(format!("{m}.shared_experts.up_proj.weight"), &[shared, h]);
            put(format!("{m}.shared_experts.down_proj.weight"), &[h, shared]);
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
fn exaone_native_batch_matches_three_rows_past_the_sliding_window() {
    let args = tiny_args();
    let model = ExaoneMoeModel::from_weights(&tiny_weights(&args), &args).unwrap();
    crate::models::batched_decode_harness::assert_batched_decode_matches_sequential(
        &model,
        &[&[1, 2], &[3, 4, 5, 6, 7], &[2, 3, 4, 5, 6, 7, 8]],
        &[3, 8, 9],
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
fn exaone_language_model_cache_matches_persistent_any_cache_past_the_window() {
    let args = tiny_args();
    let model = ExaoneMoeModel::from_weights(&tiny_weights(&args), &args).unwrap();
    let mut standard =
        <ExaoneMoeModel as mlxcel_core::generate::LanguageModel>::make_caches(&model);
    let mut model_owned = model.make_caches();
    let sequence = [1, 2, 3, 4, 5, 6, 7];
    let mut persistent = None;

    for (step, token) in sequence.into_iter().enumerate() {
        let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        let standard_logits = <ExaoneMoeModel as mlxcel_core::generate::LanguageModel>::forward(
            &model,
            &input,
            &mut standard,
            None,
        );
        let model_owned_logits = model.forward(&input, &mut model_owned, None);
        let close = mlxcel_core::allclose(&standard_logits, &model_owned_logits, 3e-4, 3e-4);
        mlxcel_core::eval(&close);
        assert!(
            mlxcel_core::item_bool(&close),
            "standard cache diverged from persistent AnyKVCache at step {step}"
        );
        for cache in &standard {
            assert_eq!(cache.offset, step as i32 + 1);
        }
        for cache in &model_owned {
            assert_eq!(cache.offset(), step as i32 + 1);
        }
        persistent = Some(standard_logits);
    }

    let last = persistent.unwrap();
    let input = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
    let mut fresh = <ExaoneMoeModel as mlxcel_core::generate::LanguageModel>::make_caches(&model);
    let without_history = <ExaoneMoeModel as mlxcel_core::generate::LanguageModel>::forward(
        &model, &input, &mut fresh, None,
    );
    let same = mlxcel_core::allclose(&last, &without_history, 1e-6, 1e-6);
    mlxcel_core::eval(&same);
    assert!(
        !mlxcel_core::item_bool(&same),
        "fixture must prove that retained sliding-window history changes logits"
    );

    assert!(
        model_owned
            .iter()
            .any(|cache| matches!(cache, AnyKVCache::Rotating(_))),
        "fixture must exercise the rotating sliding-window reference"
    );
}

// Issue #2798: the sliding window is validated before any rotating cache.
#[test]
fn rejects_a_sliding_window_a_rotating_cache_cannot_take() {
    for window in [0usize, i32::MAX as usize + 1] {
        let mut args = tiny_args();
        args.sliding_window = window;
        let err = ExaoneMoeModel::from_weights(&WeightMap::new(), &args)
            .err()
            .expect("an unusable sliding window must fail the load");
        assert!(err.contains("ExaOne MoE"), "{err}");
        assert!(err.contains(&format!("got {window}")), "{err}");
    }
}
