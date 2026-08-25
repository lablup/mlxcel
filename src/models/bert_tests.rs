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

//! Unit tests for the BERT / XLM-RoBERTa encoder: config resolution, weight
//! sanitization, position-id construction and the forward pass over a tiny
//! deterministic random-weight checkpoint.

use std::sync::{Mutex, MutexGuard, OnceLock};

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use serde_json::{Value, json};

use super::{BertArgs, BertEncoder, BertVariant, sanitize, xlm_roberta_position_ids};

/// Serializes every BERT test that drives MLX.
///
/// `cargo test` runs test functions on parallel threads, and concurrent MLX
/// forward passes in one process perturb each other: two byte-identical rows
/// of one batch came back at cosine 0.999912 instead of 1.0, and a classifier
/// logit moved by 0.05, only while other real-checkpoint tests were running.
/// `EmbeddingModel` is documented as single-thread and the server honors that
/// through the embedding worker, so the hazard is test-side only, but a gate
/// measured under it is meaningless. A poisoned lock is recovered rather than
/// propagated, so one failing test does not turn every later one into a
/// confusing poisoning panic.
pub(crate) fn mlx_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Geometry of the in-memory fixture. Small enough that a forward pass is
// instant and large enough to exercise multi-head reshaping.
pub(crate) const VOCAB: usize = 32;
pub(crate) const HIDDEN: usize = 8;
pub(crate) const HEADS: usize = 2;
pub(crate) const LAYERS: usize = 2;
pub(crate) const INTERMEDIATE: usize = 16;
pub(crate) const MAX_POSITIONS: usize = 24;

/// Deterministic pseudo-random weights in `[-0.1, 0.1)`: a 64-bit LCG keyed
/// on the tensor name, so a fixture is reproducible across runs and machines
/// without a random-number dependency.
fn pseudo_random(name: &str, len: usize) -> Vec<f32> {
    let mut state = name
        .bytes()
        .fold(0x2545_F491_4F6C_DD1Du64, |acc, byte| {
            acc.wrapping_mul(0x0100_0000_01B3).wrapping_add(byte as u64)
        })
        .wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as f32) / (u32::MAX >> 1) as f32;
            (unit - 0.5) * 0.2
        })
        .collect()
}

fn put(weights: &mut WeightMap, name: &str, shape: &[i32]) {
    let len = shape.iter().product::<i32>() as usize;
    weights.insert(
        name.to_string(),
        mlxcel_core::from_slice_f32(&pseudo_random(name, len), shape),
    );
}

/// `config.json` of the fixture checkpoint.
pub(crate) fn tiny_config(variant: BertVariant) -> Value {
    let model_type = match variant {
        BertVariant::Bert => "bert",
        BertVariant::XlmRoberta => "xlm-roberta",
    };
    json!({
        "model_type": model_type,
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": MAX_POSITIONS,
        "type_vocab_size": match variant { BertVariant::Bert => 2, BertVariant::XlmRoberta => 1 },
        "layer_norm_eps": match variant { BertVariant::Bert => 1e-12, BertVariant::XlmRoberta => 1e-5 },
        "pad_token_id": match variant { BertVariant::Bert => 0, BertVariant::XlmRoberta => 1 },
        "hidden_act": "gelu",
    })
}

/// Resolved args of the fixture checkpoint.
pub(crate) fn tiny_args(variant: BertVariant) -> BertArgs {
    BertArgs::from_config(&tiny_config(variant), variant).expect("fixture config parses")
}

/// Every tensor the encoder reads, in the sanitized (prefix-free) layout.
/// `with_classifier` adds the `pooler.` and `classifier.` head tensors.
pub(crate) fn tiny_weights(args: &BertArgs, with_classifier: bool) -> WeightMap {
    let d = args.hidden_size as i32;
    let i = args.intermediate_size as i32;
    let mut weights = WeightMap::new();

    put(
        &mut weights,
        "embeddings.word_embeddings.weight",
        &[args.vocab_size as i32, d],
    );
    put(
        &mut weights,
        "embeddings.position_embeddings.weight",
        &[args.max_position_embeddings as i32, d],
    );
    put(
        &mut weights,
        "embeddings.token_type_embeddings.weight",
        &[args.type_vocab_size as i32, d],
    );
    put(&mut weights, "embeddings.LayerNorm.weight", &[d]);
    put(&mut weights, "embeddings.LayerNorm.bias", &[d]);

    for layer in 0..args.num_hidden_layers {
        let base = format!("encoder.layer.{layer}");
        for projection in ["query", "key", "value"] {
            put(
                &mut weights,
                &format!("{base}.attention.self.{projection}.weight"),
                &[d, d],
            );
            put(
                &mut weights,
                &format!("{base}.attention.self.{projection}.bias"),
                &[d],
            );
        }
        put(
            &mut weights,
            &format!("{base}.attention.output.dense.weight"),
            &[d, d],
        );
        put(
            &mut weights,
            &format!("{base}.attention.output.dense.bias"),
            &[d],
        );
        put(
            &mut weights,
            &format!("{base}.attention.output.LayerNorm.weight"),
            &[d],
        );
        put(
            &mut weights,
            &format!("{base}.attention.output.LayerNorm.bias"),
            &[d],
        );
        put(
            &mut weights,
            &format!("{base}.intermediate.dense.weight"),
            &[i, d],
        );
        put(
            &mut weights,
            &format!("{base}.intermediate.dense.bias"),
            &[i],
        );
        put(
            &mut weights,
            &format!("{base}.output.dense.weight"),
            &[d, i],
        );
        put(&mut weights, &format!("{base}.output.dense.bias"), &[d]);
        put(
            &mut weights,
            &format!("{base}.output.LayerNorm.weight"),
            &[d],
        );
        put(&mut weights, &format!("{base}.output.LayerNorm.bias"), &[d]);
    }

    if with_classifier {
        let labels = args.num_labels as i32;
        match args.variant {
            BertVariant::Bert => {
                put(&mut weights, "pooler.dense.weight", &[d, d]);
                put(&mut weights, "pooler.dense.bias", &[d]);
                put(&mut weights, "classifier.weight", &[labels, d]);
                put(&mut weights, "classifier.bias", &[labels]);
            }
            BertVariant::XlmRoberta => {
                put(&mut weights, "classifier.dense.weight", &[d, d]);
                put(&mut weights, "classifier.dense.bias", &[d]);
                put(&mut weights, "classifier.out_proj.weight", &[labels, d]);
                put(&mut weights, "classifier.out_proj.bias", &[labels]);
            }
        }
    }
    weights
}

/// Encoder over the fixture weights.
pub(crate) fn tiny_encoder(variant: BertVariant) -> BertEncoder {
    let args = tiny_args(variant);
    let weights = tiny_weights(&args, false);
    BertEncoder::from_weights(&weights, args, 64, 4).expect("fixture encoder builds")
}

/// Cosine similarity of two equal-length vectors.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b)).max(1e-9)
}

/// Render an `Err` as a string without requiring `Debug` on the `Ok` type
/// (`MlxArray` has none).
fn err_string<T>(result: anyhow::Result<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(err) => err.to_string(),
    }
}

fn keys(weights: &WeightMap) -> Vec<String> {
    let mut out: Vec<String> = weights.keys().cloned().collect();
    out.sort();
    out
}

fn named(names: &[&str]) -> WeightMap {
    names
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                mlxcel_core::from_slice_f32(&[0.0], &[1]),
            )
        })
        .collect()
}

#[test]
fn sanitize_strips_prefix_and_drops_mlm_and_buffers() {
    let _guard = mlx_test_guard();
    for prefix in ["bert.", "roberta."] {
        let raw = named(&[
            &format!("{prefix}embeddings.word_embeddings.weight"),
            &format!("{prefix}embeddings.position_ids"),
            &format!("{prefix}embeddings.LayerNorm.weight"),
            &format!("{prefix}encoder.layer.0.attention.self.query.weight"),
            "cls.predictions.transform.dense.weight",
            "lm_head.dense.weight",
            "pooler.dense.weight",
            "classifier.out_proj.weight",
        ]);
        assert_eq!(
            keys(&sanitize(raw, false)),
            vec![
                "classifier.out_proj.weight".to_string(),
                "embeddings.LayerNorm.weight".to_string(),
                "embeddings.word_embeddings.weight".to_string(),
                "encoder.layer.0.attention.self.query.weight".to_string(),
            ],
            "prefix {prefix}"
        );
    }
}

#[test]
fn sanitize_keeps_pooler_only_for_the_classifier_head() {
    let _guard = mlx_test_guard();
    // `WeightMap` holds `UniquePtr`s and is not `Clone`, so each pass gets a
    // freshly built map.
    let raw = || named(&["bert.pooler.dense.weight", "pooler.dense.bias"]);
    assert!(sanitize(raw(), false).is_empty());
    assert_eq!(
        keys(&sanitize(raw(), true)),
        vec![
            "pooler.dense.bias".to_string(),
            "pooler.dense.weight".to_string()
        ]
    );
}

#[test]
fn sanitize_is_idempotent() {
    let _guard = mlx_test_guard();
    let raw = || {
        named(&[
            "roberta.embeddings.word_embeddings.weight",
            "roberta.embeddings.position_ids",
            "roberta.encoder.layer.3.output.LayerNorm.bias",
            "classifier.dense.weight",
        ])
    };
    for keep_pooler in [false, true] {
        let once = sanitize(raw(), keep_pooler);
        let expected = keys(&once);
        assert_eq!(keys(&sanitize(once, keep_pooler)), expected);
    }
}

#[test]
fn bert_args_fill_the_variant_defaults() {
    let bare = |model_type: &str| {
        json!({
            "model_type": model_type,
            "vocab_size": 100,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "max_position_embeddings": 514,
        })
    };

    let bert = BertArgs::from_config(&bare("bert"), BertVariant::Bert).unwrap();
    assert_eq!(bert.type_vocab_size, 2);
    assert_eq!(bert.pad_token_id, 0);
    assert!((bert.layer_norm_eps - 1e-12).abs() < 1e-18);
    assert_eq!(bert.hidden_act, "gelu");
    assert_eq!(bert.num_labels, 1);
    assert_eq!(bert.head_dim(), 4);
    // BERT indexes the position table from 0, so every row is usable.
    assert_eq!(bert.max_sequence_length(), 514);

    let xlmr = BertArgs::from_config(&bare("xlm-roberta"), BertVariant::XlmRoberta).unwrap();
    assert_eq!(xlmr.type_vocab_size, 1);
    assert_eq!(xlmr.pad_token_id, 1);
    assert!((xlmr.layer_norm_eps - 1e-5).abs() < 1e-10);
    // Positions start at pad_token_id + 1, so 514 rows hold 512 real tokens.
    assert_eq!(xlmr.max_sequence_length(), 512);
}

#[test]
fn bert_args_read_num_labels_from_id2label_when_absent() {
    let mut config = tiny_config(BertVariant::XlmRoberta);
    config["id2label"] = json!({"0": "yes", "1": "no", "2": "maybe"});
    assert_eq!(
        BertArgs::from_config(&config, BertVariant::XlmRoberta)
            .unwrap()
            .num_labels,
        3
    );
    config["num_labels"] = json!(1);
    assert_eq!(
        BertArgs::from_config(&config, BertVariant::XlmRoberta)
            .unwrap()
            .num_labels,
        1
    );
}

#[test]
fn bert_args_reject_a_head_count_that_does_not_divide_the_width() {
    let mut config = tiny_config(BertVariant::Bert);
    config["num_attention_heads"] = json!(3);
    let err = BertArgs::from_config(&config, BertVariant::Bert)
        .expect_err("8 is not divisible by 3")
        .to_string();
    assert!(err.contains("divisible"), "{err}");
}

#[test]
fn bert_variant_maps_from_model_type_and_config() {
    assert_eq!(
        BertVariant::from_model_type(crate::models::ModelType::Bert),
        Some(BertVariant::Bert)
    );
    assert_eq!(
        BertVariant::from_model_type(crate::models::ModelType::XlmRoberta),
        Some(BertVariant::XlmRoberta)
    );
    assert_eq!(
        BertVariant::from_model_type(crate::models::ModelType::ModernBert),
        None
    );
    assert_eq!(
        BertVariant::from_config(&json!({"model_type": "xlm_roberta"})),
        Some(BertVariant::XlmRoberta)
    );
    assert_eq!(BertVariant::from_config(&json!({})), None);
    assert_eq!(BertVariant::Bert.weight_prefix(), "bert.");
    assert_eq!(BertVariant::XlmRoberta.weight_prefix(), "roberta.");
}

#[test]
fn xlm_roberta_position_ids_start_after_padding_idx() {
    let _guard = mlx_test_guard();
    let input_ids = mlxcel_core::from_slice_i32(&[5, 6, 1, 1], &[1, 4]);
    let positions = xlm_roberta_position_ids(&input_ids, 1);
    mlxcel_core::try_eval(&positions).unwrap();
    assert_eq!(mlxcel_core::array_shape(&positions), vec![1, 4]);
    let values = array_to_vec_f32(&mlxcel_core::astype(
        &positions,
        mlxcel_core::dtype::FLOAT32,
    ));
    assert_eq!(values, vec![2.0, 3.0, 1.0, 1.0]);
}

#[test]
fn xlm_roberta_position_ids_handle_a_batch_and_a_zero_pad_id() {
    let _guard = mlx_test_guard();
    let input_ids = mlxcel_core::from_slice_i32(&[7, 8, 9, 4, 5, 0], &[2, 3]);
    let positions = xlm_roberta_position_ids(&input_ids, 0);
    mlxcel_core::try_eval(&positions).unwrap();
    let values = array_to_vec_f32(&mlxcel_core::astype(
        &positions,
        mlxcel_core::dtype::FLOAT32,
    ));
    assert_eq!(values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 0.0]);
}

#[test]
fn encoder_forward_produces_finite_hidden_states_of_the_right_shape() {
    let _guard = mlx_test_guard();
    for variant in [BertVariant::Bert, BertVariant::XlmRoberta] {
        let encoder = tiny_encoder(variant);
        let pad = encoder.args().pad_token_id;
        let input_ids =
            mlxcel_core::from_slice_i32(&[5, 6, 7, 8, 9, 10, 11, 12, pad, pad], &[2, 5]);
        let attention_mask = mlxcel_core::from_slice_i32(&[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], &[2, 5]);
        let hidden = encoder.encode(&input_ids, &attention_mask, None).unwrap();
        mlxcel_core::try_eval(&hidden).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(&hidden),
            vec![2, 5, HIDDEN as i32],
            "{variant:?}"
        );
        assert!(
            array_to_vec_f32(&hidden).iter().all(|v| v.is_finite()),
            "{variant:?}: hidden states must be finite"
        );
    }
}

#[test]
fn encoder_rejects_more_tokens_than_the_position_table_addresses() {
    let _guard = mlx_test_guard();
    let encoder = tiny_encoder(BertVariant::XlmRoberta);
    let usable = encoder.args().max_sequence_length();
    assert_eq!(usable, MAX_POSITIONS - 2);
    let length = usable + 1;
    let input_ids = mlxcel_core::from_slice_i32(&vec![5; length], &[1, length as i32]);
    let attention_mask = mlxcel_core::from_slice_i32(&vec![1; length], &[1, length as i32]);
    let err = err_string(encoder.encode(&input_ids, &attention_mask, None));
    assert!(err.contains("exceed"), "{err}");
    assert!(err.contains("max_position_embeddings"), "{err}");
}

#[test]
fn encoder_rejects_a_rank_one_input() {
    let _guard = mlx_test_guard();
    let encoder = tiny_encoder(BertVariant::Bert);
    let input_ids = mlxcel_core::from_slice_i32(&[5, 6, 7], &[3]);
    let attention_mask = mlxcel_core::from_slice_i32(&[1, 1, 1], &[3]);
    let err = err_string(encoder.encode(&input_ids, &attention_mask, None));
    assert!(err.contains("[B, L]"), "{err}");
}

#[test]
fn token_type_ids_change_the_bert_hidden_states() {
    let _guard = mlx_test_guard();
    // The segment table is a real learned parameter: passing segment 1 must
    // not be silently equivalent to passing zeros.
    let encoder = tiny_encoder(BertVariant::Bert);
    let input_ids = mlxcel_core::from_slice_i32(&[5, 6, 7], &[1, 3]);
    let attention_mask = mlxcel_core::from_slice_i32(&[1, 1, 1], &[1, 3]);
    let segments = mlxcel_core::from_slice_i32(&[0, 1, 1], &[1, 3]);

    let without = encoder.encode(&input_ids, &attention_mask, None).unwrap();
    let with = encoder
        .encode(&input_ids, &attention_mask, Some(&segments))
        .unwrap();
    mlxcel_core::try_eval(&without).unwrap();
    mlxcel_core::try_eval(&with).unwrap();
    let (a, b) = (array_to_vec_f32(&without), array_to_vec_f32(&with));
    assert!(
        a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6),
        "segment ids must reach the embedding sum"
    );
}
