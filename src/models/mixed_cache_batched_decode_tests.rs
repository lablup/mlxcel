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

//! Native mixed-cache decode contracts use nonzero weights and wrapped windows.
use super::batched_decode_harness::{
    assert_batched_decode_matches_sequential_with_ids, assert_empty_batch_returns_empty_logits,
    assert_single_row_batch_is_single_sequence_forward_with_ids,
};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn patterned(shape: &[i32], seed: usize) -> UniquePtr<MlxArray> {
    let len: i32 = shape.iter().product();
    let data: Vec<f32> = (0..len as usize)
        .map(|i| (((i * 37 + seed * 17) % 101) as f32 - 50.0) * 0.003)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}
fn dense_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert("model.embed_tokens.weight".into(), patterned(&[16, 8], 1));
    weights.insert("lm_head.weight".into(), patterned(&[16, 8], 2));
    weights.insert(
        "model.norm.weight".into(),
        mlxcel_core::ones(&[8], mlxcel_core::dtype::FLOAT32),
    );
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        for (name, shape) in [
            ("self_attn.q_proj", vec![8, 8]),
            ("self_attn.k_proj", vec![4, 8]),
            ("self_attn.v_proj", vec![4, 8]),
            ("self_attn.o_proj", vec![8, 8]),
            ("self_attn.g_proj", vec![2, 8]),
            ("mlp.gate_proj", vec![16, 8]),
            ("mlp.up_proj", vec![16, 8]),
            ("mlp.down_proj", vec![8, 16]),
        ] {
            weights.insert(
                format!("{p}.{name}.weight"),
                patterned(&shape, weights.len()),
            );
        }
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "post_feedforward_layernorm",
        ] {
            weights.insert(
                format!("{p}.{name}.weight"),
                mlxcel_core::ones(&[8], mlxcel_core::dtype::FLOAT32),
            );
        }
        for name in ["q_norm", "k_norm"] {
            weights.insert(
                format!("{p}.self_attn.{name}.weight"),
                mlxcel_core::ones(&[4], mlxcel_core::dtype::FLOAT32),
            );
        }
    }
    weights
}
fn dense_config() -> serde_json::Value {
    serde_json::json!({"model_type":"test", "hidden_size":8, "num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"num_attention_groups":1,
        "head_dim":4,"intermediate_size":16,"vocab_size":16,"rms_norm_eps":1e-6,
        "sliding_window":4,"layer_types":["full_attention","sliding_attention"],
        "max_position_embeddings":64,"rope_theta":10000.0,"tie_word_embeddings":false})
}
fn ministral() -> super::ministral3::Ministral3Wrapper {
    let mut value = dense_config();
    value["rope_parameters"] = serde_json::json!({"rope_theta":10000.0,"llama_4_scaling_beta":0.2,"original_max_position_embeddings":2});
    let args = serde_json::from_value(value).unwrap();
    super::ministral3::Ministral3Wrapper::new(
        super::ministral3::Ministral3Model::from_weights(&dense_weights(), &args).unwrap(),
    )
}
fn exaone() -> super::exaone4::ExaOne4Wrapper {
    let mut value = dense_config();
    value["sliding_window_pattern"] = serde_json::json!("LG");
    let args = serde_json::from_value(value).unwrap();
    super::exaone4::ExaOne4Wrapper::new(
        super::exaone4::ExaOne4Model::from_weights(&dense_weights(), &args).unwrap(),
    )
}
pub(crate) fn step() -> super::step3p5::Step3p5Model {
    let mut value = dense_config();
    value["moe_layers_enum"] = serde_json::json!([]);
    value["rope_theta"] = serde_json::json!([10000.0, 5000.0]);
    value["partial_rotary_factors"] = serde_json::json!([0.5, 1.0]);
    let args = serde_json::from_value(value).unwrap();
    super::step3p5::Step3p5Model::from_weights(&dense_weights(), &args).unwrap()
}
fn mellum() -> super::mellum::MellumWrapper {
    let mut args = super::mellum_tests::tiny_config(false);
    args.sliding_window = 4;
    let mut weights = super::mellum_tests::tiny_weights(&args);
    let mut keys: Vec<_> = weights.keys().cloned().collect();
    keys.sort();
    for (seed, name) in keys.into_iter().enumerate() {
        let shape = mlxcel_core::array_shape(&weights[&name]);
        let tensor = if name.contains("norm") {
            mlxcel_core::ones(&shape, mlxcel_core::dtype::FLOAT32)
        } else {
            patterned(&shape, seed)
        };
        weights.insert(name, tensor);
    }
    super::mellum::MellumWrapper::new(
        super::mellum::MellumModel::from_weights(&weights, &args).unwrap(),
    )
}
fn laguna() -> super::laguna::LagunaWrapper {
    super::laguna::LagunaWrapper::new(super::laguna_tests::tiny_model().0)
}
fn assert_close(actual: &MlxArray, expected: &MlxArray, tolerance: f32) {
    let difference = mlxcel_core::abs(&mlxcel_core::subtract(actual, expected));
    let maximum = mlxcel_core::max_all(&difference);
    mlxcel_core::eval(&maximum);
    let error = mlxcel_core::item_f32(&maximum);
    assert!(error <= tolerance, "max logit error {error}");
}
fn contracts<M: LanguageModel>(make: impl Fn() -> M, offsets: impl Fn(&M, SequenceId) -> Vec<i32>) {
    let model = make();
    assert!(model.supports_batching());
    assert!(!model.supports_padded_prefill());
    assert_eq!(
        model.sequence_state_layout(),
        mlxcel_core::cache::SequenceStateLayout::model_owned(model.num_layers())
    );
    assert_eq!(model.batched_decode_kind(), BatchedDecodeKind::Native);
    let prompts: &[&[i32]] = &[&[1, 2], &[3, 2, 4, 1], &[2, 5, 1, 3, 4, 6, 2, 1]];
    assert_empty_batch_returns_empty_logits(&model);
    assert_single_row_batch_is_single_sequence_forward_with_ids(&model, prompts[2], 3);
    assert_batched_decode_matches_sequential_with_ids(&model, prompts, &[2, 3, 4], 1e-5);

    // Cache creation and unrelated admission must not reset any live request.
    let ids: Vec<_> = (0..3).map(|row| SequenceId::from_raw(100 + row)).collect();
    let solos: Vec<_> = (0..3).map(|_| make()).collect();
    let mut solo_caches: Vec<_> = solos.iter().map(|m| m.make_caches()).collect();
    let mut caches: Vec<_> = (0..3).map(|_| model.make_caches()).collect();
    for row in 0..3 {
        model.prepare_sequence_state(ids[row]);
        let input = mlxcel_core::from_slice_i32(prompts[row], &[1, prompts[row].len() as i32]);
        let actual = model.forward_with_sequence_id(&input, Some(ids[row]), &mut caches[row], None);
        let expected = solos[row].forward(&input, &mut solo_caches[row], None);
        assert_close(&actual, &expected, 1e-5);
        assert_eq!(
            offsets(&model, ids[row]),
            vec![prompts[row].len() as i32; model.num_layers()]
        );
        let _ = model.make_caches();
    }
    for tick in 0..6 {
        for row in 0..3 {
            let input = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
            let got =
                model.forward_with_sequence_id(&input, Some(ids[row]), &mut caches[row], None);
            let expected = solos[row].forward(&input, &mut solo_caches[row], None);
            assert_close(&got, &expected, 1e-5);
        }
        let tokens: Vec<_> = (0..3).map(|row| (row + tick + 1) % 7).collect();
        let input = mlxcel_core::from_slice_i32(&tokens, &[3, 1]);
        let mut refs: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
        let actual = model.forward_batched_with_ids(&input, Some(&ids), &mut refs, None);
        for row in 0..3 {
            let input = mlxcel_core::from_slice_i32(&[tokens[row]], &[1, 1]);
            let expected = solos[row].forward(&input, &mut solo_caches[row], None);
            let got =
                mlxcel_core::slice(&actual, &[row as i32, 0, 0], &[row as i32 + 1, 1, i32::MAX]);
            assert_close(&got, &expected, 1e-5);
            assert_eq!(
                offsets(&model, ids[row]),
                vec![prompts[row].len() as i32 + 2 * (tick + 1); model.num_layers()]
            );
        }
        let _ = model.make_caches();
        let unrelated = SequenceId::from_raw(200 + tick as u64);
        model.prepare_sequence_state(unrelated);
        let input = mlxcel_core::from_slice_i32(&[3, 1, 2], &[1, 3]);
        let out = model.forward_with_sequence_id(&input, Some(unrelated), &mut [], None);
        mlxcel_core::eval(&out);
        model.release_sequence_state_by_id(unrelated);
    }
    for id in ids {
        model.release_sequence_state_by_id(id);
    }
}
#[test]
fn ministral3_native_isolation() {
    contracts(ministral, |m, id| m.sequence_offsets(id));
}
#[test]
fn exaone4_native_isolation() {
    contracts(exaone, |m, id| m.sequence_offsets(id));
}
#[test]
fn mellum_native_isolation() {
    contracts(mellum, |m, id| m.sequence_offsets(id));
}
#[test]
fn laguna_native_isolation() {
    contracts(laguna, |m, id| m.sequence_offsets(id));
}
#[test]
fn step3p5_native_isolation() {
    contracts(step, |m, id| m.sequence_offsets(id));
}

#[test]
fn laguna_sinks_and_gate_modes_preserve_batched_rows() {
    for gate in [
        serde_json::json!(false),
        serde_json::json!("per-head"),
        serde_json::json!("per-element"),
    ] {
        contracts(
            || {
                let mut value: serde_json::Value =
                    serde_json::from_str(super::laguna_tests::TINY_CONFIG).unwrap();
                value["gating"] = gate.clone();
                value["swa_attention_sink_enabled"] = serde_json::json!(true);
                let args: super::laguna::ModelArgs = serde_json::from_value(value).unwrap();
                let mut weights = super::laguna_tests::tiny_weights(&args);
                for layer in 0..args.num_hidden_layers {
                    let heads = args.num_heads_for_layer(layer).unwrap() as i32;
                    let prefix = format!("model.layers.{layer}.self_attn");
                    if gate == serde_json::json!("per-element") {
                        weights.insert(
                            format!("{prefix}.g_proj.weight"),
                            patterned(
                                &[heads * args.head_dim() as i32, args.hidden_size as i32],
                                layer,
                            ),
                        );
                    }
                    if args.is_sliding(layer) {
                        weights.insert(format!("{prefix}.sink"), patterned(&[heads], layer));
                    }
                }
                super::laguna::LagunaWrapper::new(
                    super::laguna::LagunaModel::from_weights(&weights, &args).unwrap(),
                )
            },
            |m, id| m.sequence_offsets(id),
        );
    }
}

#[test]
#[should_panic(expected = "requires sequence ids")]
fn mixed_cache_multirow_without_ids_is_rejected() {
    let model = ministral();
    let mut caches = [model.make_caches(), model.make_caches()];
    let mut rows: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
    model.forward_batched(
        &mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]),
        &mut rows,
        None,
    );
}

#[test]
#[should_panic(expected = "missing model-owned sequence state for sequence")]
fn mixed_cache_missing_id_names_the_sequence() {
    let model = ministral();
    let mut caches = [model.make_caches(), model.make_caches()];
    let mut rows: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
    let ids = [SequenceId::from_raw(800), SequenceId::from_raw(801)];
    model.prepare_sequence_state(ids[0]);
    model.forward_batched_with_ids(
        &mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]),
        Some(&ids),
        &mut rows,
        None,
    );
}
