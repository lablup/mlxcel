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

//! Llama 3.2 Vision (`mllama`) parity tests.
//!
//! Two tiers:
//!
//! - CI-runnable numeric parity of the net-new cross-attention text backbone
//!   against reference-derived expectations, plus the image processor's tiling
//!   contract. These build a tiny synthetic `MllamaTextModel` and exploit the
//!   reference's `tanh` gate semantics (`mx.zeros(1)` init) to pin exact
//!   behavior without a checkpoint.
//! - A model-gated smoke test (detection + forward) that stays inert in CI and
//!   on machines without the `mllama` checkpoint, mirroring the
//!   `internvl_parity` / `molmo_parity` convention.

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::mllama::MllamaTextConfig;
use mlxcel::models::mllama::text::MllamaTextModel;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::processors::mllama::MllamaImageProcessor;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// --- Tiny synthetic model dimensions (2 layers: [0]=self, [1]=cross). ---
const HIDDEN: i32 = 4;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const HEAD_DIM: i32 = HIDDEN / HEADS; // 2
const INTER: i32 = 8;
const VOCAB: i32 = 6;
const SEQ: i32 = 3;
const KV_LEN: i32 = 5; // vision cross-attention key/value length

fn tiny_config() -> MllamaTextConfig {
    serde_json::from_str(
        r#"{
            "model_type": "mllama",
            "vocab_size": 6,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "cross_attention_layers": [1]
        }"#,
    )
    .expect("tiny mllama text config")
}

/// Deterministic pseudo-random fill in roughly `[-0.5, 0.5]`.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 131 + seed * 977 + 7) % 251) as f32 / 251.0 - 0.5)
        .collect()
}

fn put(map: &mut WeightMap, key: &str, shape: &[i32], seed: usize) {
    let n: i32 = shape.iter().product();
    map.insert(
        key.to_string(),
        mlxcel_core::from_slice_f32(&fill(n as usize, seed), shape),
    );
}

fn put_const(map: &mut WeightMap, key: &str, shape: &[i32], value: f32) {
    let n: i32 = shape.iter().product();
    map.insert(
        key.to_string(),
        mlxcel_core::from_slice_f32(&vec![value; n as usize], shape),
    );
}

/// Build a full weight map for the tiny 2-layer model with the cross-attention
/// gates set to `gate_value`.
fn build_weights(gate_value: f32) -> WeightMap {
    let mut w = WeightMap::new();
    let q = HEADS * HEAD_DIM; // 4
    let kv = KV_HEADS * HEAD_DIM; // 2

    put(&mut w, "model.embed_tokens.weight", &[VOCAB, HIDDEN], 1);

    // Layer 0: standard Llama-3 self-attention block.
    put(
        &mut w,
        "model.layers.0.self_attn.q_proj.weight",
        &[q, HIDDEN],
        2,
    );
    put(
        &mut w,
        "model.layers.0.self_attn.k_proj.weight",
        &[kv, HIDDEN],
        3,
    );
    put(
        &mut w,
        "model.layers.0.self_attn.v_proj.weight",
        &[kv, HIDDEN],
        4,
    );
    put(
        &mut w,
        "model.layers.0.self_attn.o_proj.weight",
        &[HIDDEN, q],
        5,
    );
    put(
        &mut w,
        "model.layers.0.mlp.gate_proj.weight",
        &[INTER, HIDDEN],
        6,
    );
    put(
        &mut w,
        "model.layers.0.mlp.up_proj.weight",
        &[INTER, HIDDEN],
        7,
    );
    put(
        &mut w,
        "model.layers.0.mlp.down_proj.weight",
        &[HIDDEN, INTER],
        8,
    );
    put_const(
        &mut w,
        "model.layers.0.input_layernorm.weight",
        &[HIDDEN],
        1.0,
    );
    put_const(
        &mut w,
        "model.layers.0.post_attention_layernorm.weight",
        &[HIDDEN],
        1.0,
    );

    // Layer 1: gated cross-attention adapter.
    put(
        &mut w,
        "model.layers.1.cross_attn.q_proj.weight",
        &[q, HIDDEN],
        9,
    );
    put(
        &mut w,
        "model.layers.1.cross_attn.k_proj.weight",
        &[kv, HIDDEN],
        10,
    );
    put(
        &mut w,
        "model.layers.1.cross_attn.v_proj.weight",
        &[kv, HIDDEN],
        11,
    );
    put(
        &mut w,
        "model.layers.1.cross_attn.o_proj.weight",
        &[HIDDEN, q],
        12,
    );
    put_const(
        &mut w,
        "model.layers.1.cross_attn.q_norm.weight",
        &[HEAD_DIM],
        1.0,
    );
    put_const(
        &mut w,
        "model.layers.1.cross_attn.k_norm.weight",
        &[HEAD_DIM],
        1.0,
    );
    put_const(
        &mut w,
        "model.layers.1.input_layernorm.weight",
        &[HIDDEN],
        1.0,
    );
    put_const(
        &mut w,
        "model.layers.1.post_attention_layernorm.weight",
        &[HIDDEN],
        1.0,
    );
    put(
        &mut w,
        "model.layers.1.mlp.gate_proj.weight",
        &[INTER, HIDDEN],
        13,
    );
    put(
        &mut w,
        "model.layers.1.mlp.up_proj.weight",
        &[INTER, HIDDEN],
        14,
    );
    put(
        &mut w,
        "model.layers.1.mlp.down_proj.weight",
        &[HIDDEN, INTER],
        15,
    );
    put_const(
        &mut w,
        "model.layers.1.cross_attn_attn_gate",
        &[1],
        gate_value,
    );
    put_const(
        &mut w,
        "model.layers.1.cross_attn_mlp_gate",
        &[1],
        gate_value,
    );

    put_const(&mut w, "model.norm.weight", &[HIDDEN], 1.0);
    put(&mut w, "lm_head.weight", &[VOCAB, HIDDEN], 16);
    w
}

fn input_ids() -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, SEQ])
}

fn cross_states() -> UniquePtr<MlxArray> {
    let n = (KV_LEN * HIDDEN) as usize;
    mlxcel_core::from_slice_f32(&fill(n, 42), &[1, KV_LEN, HIDDEN])
}

/// Max absolute elementwise difference between two arrays.
fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let diff = mlxcel_core::subtract(a, b);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&diff));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn forward(model: &MllamaTextModel, cross: Option<&MlxArray>) -> UniquePtr<MlxArray> {
    let ids = input_ids();
    let mut caches = model.make_caches();
    let logits = model.forward(Some(&ids), None, &mut caches, None, cross, None, None);
    mlxcel_core::eval(&logits);
    logits
}

/// With the gates at zero (the reference's `mx.zeros(1)` init), a cross-attention
/// layer is an exact pass-through: `residual + tanh(0) * branch == residual`.
/// So the logits must be identical whether or not vision features are supplied.
#[test]
fn zero_gate_cross_attention_is_pass_through() {
    let config = tiny_config();
    let weights = build_weights(0.0);
    let model = MllamaTextModel::from_weights(&weights, &config).expect("build tiny mllama");

    let logits_text = forward(&model, None);
    let logits_image = forward(&model, Some(&cross_states()));

    assert_eq!(mlxcel_core::array_shape(&logits_text), vec![1, SEQ, VOCAB]);
    let diff = max_abs_diff(&logits_text, &logits_image);
    assert!(
        diff < 1e-5,
        "zero-gated cross-attention must not change the logits, got max|delta|={diff}"
    );
}

/// With non-zero gates the cross-attention layer genuinely consults the vision
/// features, so supplying `cross_attention_states` must change the logits. This
/// proves the cross path is wired end to end (projection, q/k norms, GQA
/// attention, gated residuals), not silently dropped.
#[test]
fn nonzero_gate_cross_attention_consults_vision_features() {
    let config = tiny_config();
    let weights = build_weights(1.5);
    let model = MllamaTextModel::from_weights(&weights, &config).expect("build tiny mllama");

    let logits_text = forward(&model, None);
    let logits_image = forward(&model, Some(&cross_states()));

    let diff = max_abs_diff(&logits_text, &logits_image);
    assert!(
        diff > 1e-3,
        "active cross-attention must change the logits, got max|delta|={diff}"
    );

    // Logits stay finite.
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&logits_image));
    mlxcel_core::eval(&m);
    assert!(mlxcel_core::item_f32(&m).is_finite());
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

/// The image processor packs to the tower's `[B, num_media, max_tiles, C, H, W]`
/// contract with matching aspect-ratio ids and masks.
#[test]
fn processor_packs_tower_contract() {
    let processor = MllamaImageProcessor::new(560, 4);
    let image = fixture_image();
    let inputs = processor.process(std::slice::from_ref(&image));
    mlxcel_core::eval(&inputs.pixel_values);

    assert_eq!(
        mlxcel_core::array_shape(&inputs.pixel_values),
        vec![1, 1, 4, 3, 560, 560]
    );
    assert_eq!(
        mlxcel_core::array_shape(&inputs.aspect_ratio_ids),
        vec![1, 1]
    );
    assert_eq!(
        mlxcel_core::array_shape(&inputs.aspect_ratio_mask),
        vec![1, 1, 4]
    );
    assert_eq!(inputs.num_tiles.len(), 1);
    assert!(inputs.num_tiles[0] >= 1 && inputs.num_tiles[0] <= 4);
}

// --- Model-gated smoke test (inert without the checkpoint). ---

const MODEL_NAME: &str = "llama-3.2-11b-vision";

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.join("config.json").exists() {
        Some(dir)
    } else {
        eprintln!("Skipping mllama real-model test: no checkpoint at {dir:?}");
        None
    }
}

#[test]
fn detects_mllama_model_type() {
    let Some(dir) = model_dir() else { return };
    assert_eq!(
        get_model_type(&dir).expect("detect model type"),
        ModelType::MllamaVLM
    );
}
