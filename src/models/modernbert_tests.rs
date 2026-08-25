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

//! Unit tests for the ModernBERT port: config parsing, layer parity, the
//! sliding-window mask, the first-layer `attn_norm` exception, the GeGLU
//! half-split, padding invariance and the classifier head.
//!
//! Every forward-pass test runs a deliberately tiny encoder built from
//! deterministic synthetic weights, so it needs no checkpoint. The
//! real-checkpoint gates live in `modernbert_real_checkpoint_tests.rs`.

use std::sync::{Mutex, MutexGuard, OnceLock};

use mlxcel_core::utils::{array_to_vec_f32, create_bidirectional_window_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use crate::embeddings::pooling::{PoolingMode, pool};
use crate::models::modernbert::{
    ModernBertArgs, ModernBertEncoder, geglu, sanitize_modernbert_weights,
};
use crate::models::modernbert_heads::ModernBertSequenceClassifier;
use crate::models::{ModelType, get_model_type};

/// Serializes every test that touches MLX, in this module and in
/// [`super::modernbert_real_checkpoint_tests`], onto one thread at a time.
///
/// [`crate::embeddings::model::EmbeddingModel`] is documented as living on
/// exactly one thread, and the server backs that with a dedicated MLX-owning
/// worker per model. `cargo test` runs these in parallel by default, which
/// breaks that contract in two observed ways on this CUDA host: silently wrong
/// numbers (one parallel run in three scored two byte-identical rows of a single
/// batch at cosine 0.99991 instead of 1.0 and moved a reranker logit by 0.05),
/// and an outright abort inside MLX's CUDA graph capture
/// (`cudaStreamEndCapture ... previous error during capture`, SIGABRT).
/// Serializing makes the suite obey the same contract the product does.
///
/// Tests that only parse config or touch the filesystem do not need the guard;
/// every test that builds an encoder, a classifier, or any `MlxArray` does.
///
/// A poisoned lock is recovered rather than propagated so one failing test does
/// not cascade into false failures in the rest.
pub(super) fn mlx_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Synthetic checkpoint.

const HIDDEN: usize = 8;
const HEADS: usize = 2;
const INTERMEDIATE: usize = 6;
const LAYERS: usize = 4;
const VOCAB: usize = 20;
const LABELS: usize = 2;

/// Deterministic pseudo-random values in `[-0.5, 0.5)` from a fixed seed, so
/// every run of the suite sees byte-identical weights.
fn pseudo_random(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / 16_777_216.0 - 0.5
        })
        .collect()
}

fn put(weights: &mut WeightMap, name: &str, shape: &[i32], seed: u64) {
    let count: usize = shape.iter().map(|&d| d as usize).product();
    weights.insert(
        name.to_string(),
        mlxcel_core::from_slice_f32(&pseudo_random(seed, count), shape),
    );
}

/// `config.json` of the synthetic encoder. `local_attention` is 8, so the
/// window bound is 5 and every token of a 3- or 5-token input is inside it.
fn tiny_config() -> Value {
    json!({
        "model_type": "modernbert",
        "architectures": ["ModernBertModel"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": 64,
        "norm_eps": 1e-5,
        "layer_norm_eps": 1e-5,
        "norm_bias": false,
        "global_rope_theta": 160000.0,
        "local_rope_theta": 10000.0,
        "global_attn_every_n_layers": 3,
        "local_attention": 8,
        "attention_bias": false,
        "mlp_bias": false,
        "hidden_activation": "gelu",
        "classifier_pooling": "mean",
        "classifier_activation": "gelu",
        "classifier_bias": false,
        "pad_token_id": 0
    })
}

fn tiny_args() -> ModernBertArgs {
    ModernBertArgs::from_config(&tiny_config()).expect("tiny config parses")
}

/// Backbone tensors of the synthetic checkpoint, in the published key layout.
fn tiny_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    put(
        &mut weights,
        "embeddings.tok_embeddings.weight",
        &[VOCAB as i32, HIDDEN as i32],
        1,
    );
    put(&mut weights, "embeddings.norm.weight", &[HIDDEN as i32], 2);
    put(&mut weights, "final_norm.weight", &[HIDDEN as i32], 3);
    for layer in 0..LAYERS {
        let seed = 10 + layer as u64 * 7;
        put(
            &mut weights,
            &format!("layers.{layer}.attn.Wqkv.weight"),
            &[3 * HIDDEN as i32, HIDDEN as i32],
            seed,
        );
        put(
            &mut weights,
            &format!("layers.{layer}.attn.Wo.weight"),
            &[HIDDEN as i32, HIDDEN as i32],
            seed + 1,
        );
        put(
            &mut weights,
            &format!("layers.{layer}.mlp.Wi.weight"),
            &[2 * INTERMEDIATE as i32, HIDDEN as i32],
            seed + 2,
        );
        put(
            &mut weights,
            &format!("layers.{layer}.mlp.Wo.weight"),
            &[HIDDEN as i32, INTERMEDIATE as i32],
            seed + 3,
        );
        put(
            &mut weights,
            &format!("layers.{layer}.mlp_norm.weight"),
            &[HIDDEN as i32],
            seed + 4,
        );
        // Layer 0 deliberately has none: upstream makes it nn.Identity().
        if layer > 0 {
            put(
                &mut weights,
                &format!("layers.{layer}.attn_norm.weight"),
                &[HIDDEN as i32],
                seed + 5,
            );
        }
    }
    weights
}

/// Backbone plus the sequence-classification head.
fn tiny_classifier_weights() -> WeightMap {
    let mut weights = tiny_weights();
    put(
        &mut weights,
        "head.dense.weight",
        &[HIDDEN as i32, HIDDEN as i32],
        90,
    );
    put(&mut weights, "head.norm.weight", &[HIDDEN as i32], 91);
    put(
        &mut weights,
        "classifier.weight",
        &[LABELS as i32, HIDDEN as i32],
        92,
    );
    put(&mut weights, "classifier.bias", &[LABELS as i32], 93);
    weights
}

fn tiny_encoder() -> ModernBertEncoder {
    ModernBertEncoder::from_weights(&tiny_weights(), &tiny_args(), None).expect("encoder builds")
}

/// `[B, L]` int32 ids and mask from per-row token lists, right-padded with 0.
fn padded_batch(rows: &[&[i32]]) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let width = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut ids: Vec<i32> = Vec::with_capacity(rows.len() * width);
    let mut mask: Vec<i32> = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let pad = width - row.len();
        ids.extend_from_slice(row);
        ids.extend(vec![0_i32; pad]);
        mask.extend(vec![1_i32; row.len()]);
        mask.extend(vec![0_i32; pad]);
    }
    let shape = [rows.len() as i32, width as i32];
    (
        mlxcel_core::from_slice_i32(&ids, &shape),
        mlxcel_core::from_slice_i32(&mask, &shape),
    )
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "{what}: component {index} differs: {a} vs {e} (tolerance {tolerance})"
        );
    }
}

// Config and layer parity.

#[test]
fn layer_parity_selects_local_and_global_rope_base() {
    let args = tiny_args();
    assert_eq!(args.global_attn_every_n_layers, 3);
    for layer in 0..9 {
        let expect_global = layer % 3 == 0;
        assert_eq!(
            args.is_local_layer(layer),
            !expect_global,
            "layer {layer} parity"
        );
        let expected_base = if expect_global { 160_000.0 } else { 10_000.0 };
        assert_eq!(args.rope_base(layer), expected_base, "layer {layer} base");
    }
    // A config without `local_rope_theta` falls back to the global base on
    // every layer, matching upstream's `if config.local_rope_theta is not None`.
    let mut config = tiny_config();
    config["local_rope_theta"] = Value::Null;
    let args = ModernBertArgs::from_config(&config).unwrap();
    assert_eq!(args.rope_base(1), 160_000.0);
    assert_eq!(args.rope_base(3), 160_000.0);
}

#[test]
fn norm_eps_accepts_either_spelling_and_both_together() {
    // Published checkpoints carry `norm_eps` and `layer_norm_eps` at once, so
    // the two must be independent fields rather than one serde alias (an alias
    // reports a duplicate-field error when both keys are present).
    let both = ModernBertArgs::from_config(&tiny_config()).unwrap();
    assert_eq!(both.norm_eps(), 1e-5);

    let mut legacy_only = tiny_config();
    legacy_only.as_object_mut().unwrap().remove("norm_eps");
    legacy_only["layer_norm_eps"] = json!(1e-6);
    assert_eq!(
        ModernBertArgs::from_config(&legacy_only)
            .unwrap()
            .norm_eps(),
        1e-6
    );

    let mut modern_only = tiny_config();
    modern_only
        .as_object_mut()
        .unwrap()
        .remove("layer_norm_eps");
    modern_only["norm_eps"] = json!(1e-7);
    assert_eq!(
        ModernBertArgs::from_config(&modern_only)
            .unwrap()
            .norm_eps(),
        1e-7
    );

    let mut neither = tiny_config();
    neither.as_object_mut().unwrap().remove("norm_eps");
    neither.as_object_mut().unwrap().remove("layer_norm_eps");
    assert_eq!(
        ModernBertArgs::from_config(&neither).unwrap().norm_eps(),
        1e-5
    );
}

#[test]
fn config_validation_rejects_unsafe_scalars() {
    let cases: &[(&str, Value, &str)] = &[
        ("hidden_size", json!(0), "hidden_size"),
        ("num_attention_heads", json!(3), "divisible"),
        (
            "global_attn_every_n_layers",
            json!(0),
            "global_attn_every_n",
        ),
        ("num_hidden_layers", json!(4096), "layer cap"),
        ("local_attention", json!(1), "local_attention"),
        ("global_rope_theta", json!(0.0), "global_rope_theta"),
        ("hidden_activation", json!("silu"), "hidden_activation"),
        ("classifier_pooling", json!("max"), "classifier_pooling"),
    ];
    for (key, value, fragment) in cases {
        let mut config = tiny_config();
        config[*key] = value.clone();
        let err = ModernBertArgs::from_config(&config)
            .expect_err(&format!("{key} = {value} must be rejected"))
            .to_string();
        assert!(err.contains(fragment), "{key}: {err}");
    }
}

#[test]
fn local_window_bound_attends_half_the_declared_window() {
    // The published default: local_attention 128 means 64 keys on each side,
    // so the exclusive bound handed to the mask builder is 65.
    let mut config = tiny_config();
    config["local_attention"] = json!(128);
    let args = ModernBertArgs::from_config(&config).unwrap();
    assert_eq!(args.local_window(), 65);
    assert_eq!(tiny_args().local_window(), 5);
}

// Weight sanitize.

#[test]
fn sanitize_strips_model_prefix_and_drops_the_right_heads() {
    let _mlx = mlx_guard();
    let build = || {
        let mut weights = WeightMap::new();
        for key in [
            "model.embeddings.tok_embeddings.weight",
            "model.layers.0.attn.Wqkv.weight",
            "model.final_norm.weight",
            "head.dense.weight",
            "head.norm.weight",
            "classifier.weight",
            "classifier.bias",
            "decoder.weight",
            "decoder.bias",
            "pooler.dense.weight",
        ] {
            put(&mut weights, key, &[2, 2], 5);
        }
        weights
    };

    let embedder = sanitize_modernbert_weights(build(), false);
    assert!(embedder.contains_key("embeddings.tok_embeddings.weight"));
    assert!(embedder.contains_key("layers.0.attn.Wqkv.weight"));
    assert!(embedder.contains_key("final_norm.weight"));
    assert!(embedder.keys().all(|k| !k.starts_with("model.")));
    for dropped in ["head.dense.weight", "classifier.weight", "decoder.weight"] {
        assert!(!embedder.contains_key(dropped), "{dropped} must be dropped");
    }
    assert!(embedder.keys().all(|k| !k.starts_with("pooler.")));

    let classifier = sanitize_modernbert_weights(build(), true);
    assert!(classifier.contains_key("head.dense.weight"));
    assert!(classifier.contains_key("head.norm.weight"));
    assert!(classifier.contains_key("classifier.weight"));
    assert!(classifier.contains_key("classifier.bias"));
    // decoder.* is the MLM output projection and is never loaded.
    assert!(classifier.keys().all(|k| !k.starts_with("decoder.")));
}

// Structure.

#[test]
fn first_layer_has_no_attn_norm() {
    let _mlx = mlx_guard();
    let args = tiny_args();
    // The published layout: layers.0.attn_norm is absent and the stack loads.
    let weights = tiny_weights();
    assert!(!weights.contains_key("layers.0.attn_norm.weight"));
    assert!(ModernBertEncoder::from_weights(&weights, &args, None).is_ok());

    // Any other layer missing its attn_norm is a truncated checkpoint.
    let mut broken = tiny_weights();
    broken.remove("layers.1.attn_norm.weight");
    // `expect_err` would need `ModernBertEncoder: Debug`, which no MLX-owning
    // type implements, so match the error out instead.
    let Err(err) = ModernBertEncoder::from_weights(&broken, &args, None) else {
        panic!("a missing layers.1.attn_norm must fail the load");
    };
    assert!(err.contains("layers.1.attn_norm.weight"), "{err}");
    assert!(err.contains("only layer 0"), "{err}");
}

#[test]
fn sliding_mask_attends_within_64_and_blocks_beyond() {
    let _mlx = mlx_guard();
    // Real ModernBERT geometry: local_attention 128, so |q - k| <= 64 attends.
    const LEN: usize = 200;
    const REAL: usize = 150;
    let mut config = tiny_config();
    config["local_attention"] = json!(128);
    let window = ModernBertArgs::from_config(&config).unwrap().local_window();
    assert_eq!(window, 65);

    // Row 0 is fully real; row 1 has REAL tokens followed by padding.
    let mut mask = vec![1_i32; 2 * LEN];
    mask[LEN + REAL..].fill(0);
    let mask = mlxcel_core::from_slice_i32(&mask, &[2, LEN as i32]);

    let additive = create_bidirectional_window_mask(&mask, window);
    assert_eq!(
        mlxcel_core::array_shape(&additive),
        vec![2, 1, LEN as i32, LEN as i32]
    );
    let values = array_to_vec_f32(&additive);
    let at = |b: usize, q: usize, k: usize| values[(b * LEN + q) * LEN + k];

    // Fully real row: the window is the only constraint.
    assert_eq!(at(0, 100, 100), 0.0, "self-attention is always allowed");
    assert_eq!(at(0, 100, 36), 0.0, "|q - k| == 64 attends");
    assert_eq!(at(0, 100, 164), 0.0, "|q - k| == 64 attends on the right");
    assert!(at(0, 100, 35).is_infinite(), "|q - k| == 65 is blocked");
    assert!(at(0, 100, 165).is_infinite(), "|q - k| == 65 is blocked");
    assert!(at(0, 0, 65).is_infinite());
    assert_eq!(at(0, 0, 64), 0.0);

    // Padded row: padding keys are blocked even inside the window.
    assert_eq!(at(1, 100, 149), 0.0, "last real key inside the window");
    assert!(
        at(1, 100, 150).is_infinite(),
        "the first padding key is blocked although |q - k| == 50"
    );
    assert!(at(1, 10, 149).is_infinite(), "far real key still blocked");
}

#[test]
fn geglu_splits_input_then_gate() {
    let _mlx = mlx_guard();
    // Direct: y = [inputs | gate], output = gelu(input) * gate.
    let y = mlxcel_core::from_slice_f32(&[1.0, -2.0, 3.0, 0.5, 2.0, -1.0], &[1, 1, 6]);
    let out = geglu(&y, 3);
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 1, 3]);
    let got = array_to_vec_f32(&out);
    let gelu_exact = |x: f32| x * 0.5 * (1.0 + erf_approx(x / std::f32::consts::SQRT_2));
    let expected = [
        gelu_exact(1.0) * 0.5,
        gelu_exact(-2.0) * 2.0,
        -gelu_exact(3.0),
    ];
    assert_close(&got, &expected, 1e-5, "geglu");

    // The gate is the SECOND half: zeroing it silences the MLP entirely, so
    // the block output stops depending on mlp.Wo.
    let args = tiny_args();
    let zero_gate = |weights: &mut WeightMap| {
        for layer in 0..LAYERS {
            let key = format!("layers.{layer}.mlp.Wi.weight");
            let mut rows = array_to_vec_f32(&weights[&key]);
            rows[INTERMEDIATE * HIDDEN..].fill(0.0);
            weights.insert(
                key,
                mlxcel_core::from_slice_f32(&rows, &[2 * INTERMEDIATE as i32, HIDDEN as i32]),
            );
        }
    };
    let (ids, mask) = padded_batch(&[&[3, 4, 5, 6, 7]]);

    let mut a = tiny_weights();
    zero_gate(&mut a);
    let hidden_a = ModernBertEncoder::from_weights(&a, &args, None)
        .unwrap()
        .encode(&ids, &mask)
        .unwrap();

    let mut b = tiny_weights();
    zero_gate(&mut b);
    for layer in 0..LAYERS {
        put(
            &mut b,
            &format!("layers.{layer}.mlp.Wo.weight"),
            &[HIDDEN as i32, INTERMEDIATE as i32],
            500 + layer as u64,
        );
    }
    let hidden_b = ModernBertEncoder::from_weights(&b, &args, None)
        .unwrap()
        .encode(&ids, &mask)
        .unwrap();

    assert_close(
        &array_to_vec_f32(&hidden_a),
        &array_to_vec_f32(&hidden_b),
        1e-6,
        "a zero gate half makes mlp.Wo irrelevant",
    );
}

/// `erf` for the reference GeGLU value, without pulling in a math crate.
/// Abramowitz and Stegun 7.1.26; accurate to ~1.5e-7, well inside the 1e-5
/// tolerance the assertion uses.
fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

#[test]
fn padding_invariance() {
    let _mlx = mlx_guard();
    let encoder = tiny_encoder();

    let short: &[i32] = &[11, 4, 9];
    let (solo_ids, solo_mask) = padded_batch(&[short]);
    let solo = pool(
        &encoder.encode(&solo_ids, &solo_mask).unwrap(),
        &solo_mask,
        PoolingMode::Mean,
    );
    let solo = array_to_vec_f32(&solo);

    // The same text as the shorter member of a padded two-row batch.
    let (batch_ids, batch_mask) = padded_batch(&[&[3, 4, 5, 6, 7, 8], short]);
    let batched = pool(
        &encoder.encode(&batch_ids, &batch_mask).unwrap(),
        &batch_mask,
        PoolingMode::Mean,
    );
    let batched = array_to_vec_f32(&batched);
    assert_eq!(batched.len(), 2 * HIDDEN);

    assert_close(
        &batched[HIDDEN..],
        &solo,
        1e-4,
        "padding must not change the pooled vector",
    );
}

#[test]
fn classifier_mean_pooling_ignores_padding() {
    let _mlx = mlx_guard();
    let mut config = tiny_config();
    config["architectures"] = json!(["ModernBertForSequenceClassification"]);
    config["id2label"] = json!({"0": "LABEL_0", "1": "LABEL_1"});
    let args = ModernBertArgs::from_config(&config).unwrap();
    assert!(args.is_sequence_classifier());
    assert_eq!(args.num_labels(), LABELS);

    let classifier =
        ModernBertSequenceClassifier::from_weights(&tiny_classifier_weights(), &args, None)
            .expect("classifier builds");
    assert_eq!(classifier.num_labels(), LABELS);
    assert_eq!(classifier.classifier_pooling(), PoolingMode::Mean);

    let short: &[i32] = &[11, 4, 9];
    let (solo_ids, solo_mask) = padded_batch(&[short]);
    let solo = array_to_vec_f32(&classifier.logits(&solo_ids, &solo_mask).unwrap());
    assert_eq!(solo.len(), LABELS);
    assert!(solo.iter().all(|v| v.is_finite()), "logits are finite");

    let (batch_ids, batch_mask) = padded_batch(&[&[3, 4, 5, 6, 7, 8], short]);
    let batched = array_to_vec_f32(&classifier.logits(&batch_ids, &batch_mask).unwrap());
    assert_eq!(batched.len(), 2 * LABELS);
    assert_close(
        &batched[LABELS..],
        &solo,
        1e-4,
        "padding must not change the classifier logits",
    );
}

#[test]
fn classifier_cls_pooling_reads_the_first_real_token() {
    let _mlx = mlx_guard();
    let mut config = tiny_config();
    config["architectures"] = json!(["ModernBertForSequenceClassification"]);
    config["classifier_pooling"] = json!("cls");
    let args = ModernBertArgs::from_config(&config).unwrap();
    // The config declares no `num_labels` and no `id2label`, so its own
    // default is one label ...
    assert_eq!(
        args.num_labels(),
        1,
        "no num_labels and no id2label means one"
    );
    // ... but classifier.weight is the tensor that actually produces the
    // logits, so the head must advertise its row count instead.
    let classifier =
        ModernBertSequenceClassifier::from_weights(&tiny_classifier_weights(), &args, None)
            .unwrap();
    assert_eq!(classifier.classifier_pooling(), PoolingMode::Cls);
    assert_eq!(
        classifier.num_labels(),
        LABELS,
        "classifier.weight overrules a config that understates the label count"
    );

    let (ids, mask) = padded_batch(&[&[11, 4, 9]]);
    let logits = array_to_vec_f32(&classifier.logits(&ids, &mask).unwrap());
    assert_eq!(
        logits.len(),
        classifier.num_labels(),
        "num_labels() must describe the real logits width"
    );
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn classifier_rejects_a_weight_that_does_not_match_the_hidden_size() {
    let _mlx = mlx_guard();
    let mut config = tiny_config();
    config["architectures"] = json!(["ModernBertForSequenceClassification"]);
    let args = ModernBertArgs::from_config(&config).unwrap();

    let mut weights = tiny_classifier_weights();
    put(&mut weights, "classifier.weight", &[LABELS as i32, 3], 94);
    let Err(err) = ModernBertSequenceClassifier::from_weights(&weights, &args, None) else {
        panic!("a classifier.weight of the wrong width must fail the load");
    };
    assert!(err.contains("classifier.weight"), "{err}");
    assert!(err.contains(&HIDDEN.to_string()), "{err}");
}

#[test]
fn encoder_rejects_a_mask_that_does_not_match_the_ids() {
    let _mlx = mlx_guard();
    let encoder = tiny_encoder();
    let (ids, _) = padded_batch(&[&[1, 2, 3]]);
    let mask = mlxcel_core::from_slice_i32(&[1, 1], &[1, 2]);
    let Err(err) = encoder.encode(&ids, &mask) else {
        panic!("a mismatched mask must be rejected");
    };
    assert!(err.contains("attention_mask"), "{err}");
}

// Detection.

#[test]
fn modernbert_config_detects_as_the_embedding_family() {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel_modernbert_detect_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();

    for arch in ["ModernBertModel", "ModernBertForMaskedLM"] {
        let mut config = tiny_config();
        config["architectures"] = json!([arch]);
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        assert_eq!(
            get_model_type(&dir).unwrap(),
            ModelType::ModernBert,
            "{arch} must detect as the ModernBERT embedding family"
        );
    }

    // A reranker is never an embedder. Since #1356 it detects as the reranker
    // family instead, which is what lets `-m <cross-encoder>` serve
    // `/v1/rerank` without `--reranker-model`.
    let mut config = tiny_config();
    config["architectures"] = json!(["ModernBertForSequenceClassification"]);
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    assert_eq!(
        get_model_type(&dir).expect("a classifier detects as the reranker family"),
        ModelType::SequenceClassifier,
        "a classifier must not detect as an embedder"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}
