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
use mlxcel::models::mllama::text::MllamaTextModel;
use mlxcel::models::mllama::{MllamaConfig, MllamaTextConfig};
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::MllamaVLModel;
use mlxcel::vision::encoders::mllama::MllamaVisionModel;
use mlxcel::vision::processors::mllama::{MllamaImageInputs, MllamaImageProcessor};
use mlxcel_core::generate::LanguageModel;
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

// --- Real-tile cross-attention states (issue #527 perf follow-up). ---
//
// The processor pads every image's tile axis to `max_num_tiles` with zero
// tiles. The tower must keep processing all of those lanes (its aspect-ratio
// mask deliberately leaves real->padding attention open, see the encoder unit
// tests), but the reference then masks every padding-tile position out of the
// TEXT cross-attention with an additive -1e9, which zeroes their softmax
// weights exactly. `MllamaVLModel` reproduces that tile-level masking by
// building `cross_attention_states` from the real-tile rows only. These tests
// pin the states-level contract on a tiny but fully forward-capable VL model.

const V_OUT: i32 = 16; // tower hidden (8) * (1 + |intermediate_layers_indices|)
const V_PATCHES: i32 = 5; // (4 / 2)^2 patches + class token

fn tiny_vl_config() -> MllamaConfig {
    serde_json::from_str(
        r#"{
            "text_config": {
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
            },
            "vision_config": {
                "image_size": 4,
                "patch_size": 2,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_global_layers": 1,
                "num_attention_heads": 2,
                "max_num_tiles": 4,
                "intermediate_layers_indices": [0]
            }
        }"#,
    )
    .expect("tiny mllama VL config")
}

/// Forward-capable tiny tower weights (mirrors the encoder unit-test harness:
/// 4x4 image, patch 2, hidden 8, 2 local + 1 gated global layer).
fn build_tiny_vision_weights(prefix: &str) -> WeightMap {
    let (h, inter, np, tiles, ar_rows) = (8, 16, 5, 4, 9);
    let mut w = WeightMap::new();

    put(&mut w, &format!("{prefix}.class_embedding"), &[h], 1);
    put(
        &mut w,
        &format!("{prefix}.patch_embedding.weight"),
        &[h, 3, 2, 2],
        2,
    );
    for (name, seed) in [("layernorm_pre", 3), ("layernorm_post", 5)] {
        put(&mut w, &format!("{prefix}.{name}.weight"), &[h], seed);
        put(&mut w, &format!("{prefix}.{name}.bias"), &[h], seed + 1);
    }
    put(
        &mut w,
        &format!("{prefix}.gated_positional_embedding.embedding"),
        &[np, h],
        7,
    );
    put(
        &mut w,
        &format!("{prefix}.gated_positional_embedding.gate"),
        &[1],
        8,
    );
    put(
        &mut w,
        &format!("{prefix}.gated_positional_embedding.tile_embedding.weight"),
        &[ar_rows, tiles * np * h],
        9,
    );
    for (name, seed) in [
        ("pre_tile_positional_embedding", 10),
        ("post_tile_positional_embedding", 12),
    ] {
        put(
            &mut w,
            &format!("{prefix}.{name}.embedding.weight"),
            &[ar_rows, tiles * h],
            seed,
        );
        put(&mut w, &format!("{prefix}.{name}.gate"), &[1], seed + 1);
    }
    let mut add_layers = |stack: &str, count: usize, gated: bool, base: usize| {
        for i in 0..count {
            let p = format!("{prefix}.{stack}.layers.{i}");
            let s = base + i * 20;
            put(&mut w, &format!("{p}.input_layernorm.weight"), &[h], s);
            put(&mut w, &format!("{p}.input_layernorm.bias"), &[h], s + 1);
            put(
                &mut w,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
                s + 2,
            );
            put(
                &mut w,
                &format!("{p}.post_attention_layernorm.bias"),
                &[h],
                s + 3,
            );
            for (j, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
                put(
                    &mut w,
                    &format!("{p}.self_attn.{proj}.weight"),
                    &[h, h],
                    s + 4 + j,
                );
            }
            put(&mut w, &format!("{p}.mlp.fc1.weight"), &[inter, h], s + 8);
            put(&mut w, &format!("{p}.mlp.fc1.bias"), &[inter], s + 9);
            put(&mut w, &format!("{p}.mlp.fc2.weight"), &[h, inter], s + 10);
            put(&mut w, &format!("{p}.mlp.fc2.bias"), &[h], s + 11);
            if gated {
                put(&mut w, &format!("{p}.gate_attn"), &[1], s + 12);
                put(&mut w, &format!("{p}.gate_ffn"), &[1], s + 13);
            }
        }
    };
    add_layers("transformer", 2, false, 100);
    add_layers("global_transformer", 1, true, 200);
    w
}

fn tiny_vl_model() -> MllamaVLModel {
    let config = tiny_vl_config();
    let text_model = MllamaTextModel::from_weights(&build_weights(0.5), &config.text_config)
        .expect("tiny mllama text model");
    let tower = MllamaVisionModel::from_weights(
        &build_tiny_vision_weights("vision_tower"),
        &config.vision_config,
        "vision_tower",
    )
    .expect("tiny mllama vision tower");

    let mut pw = WeightMap::new();
    put(
        &mut pw,
        "multi_modal_projector.weight",
        &[HIDDEN, V_OUT],
        300,
    );
    put(&mut pw, "multi_modal_projector.bias", &[HIDDEN], 301);
    let projector = MllamaVLModel::load_projector(&pw, 64, 4).expect("tiny dense projector");

    let processor = MllamaImageProcessor::new(4, 4);
    MllamaVLModel::from_parts(text_model, tower, projector, processor, config, vec![5])
}

/// `[1, media, 4, 3, 4, 4]` pixel values; `Some(seed)` tiles carry content,
/// `None` tiles are the processor's zero padding.
fn tile_pixels(media_fills: &[[Option<usize>; 4]]) -> UniquePtr<MlxArray> {
    let per_tile = 3 * 4 * 4;
    let mut pixels = Vec::with_capacity(media_fills.len() * 4 * per_tile);
    for tiles in media_fills {
        for seed in tiles {
            match seed {
                Some(seed) => pixels.extend(fill(per_tile, *seed)),
                None => pixels.extend(std::iter::repeat_n(0.0f32, per_tile)),
            }
        }
    }
    mlxcel_core::from_slice_f32(&pixels, &[1, media_fills.len() as i32, 4, 3, 4, 4])
}

fn states_rows(states: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(states, &[0, start, 0], &[1, end, HIDDEN])
}

/// (a) Sub-max real tiles: the real-tile states are byte-identical to the
/// corresponding rows of the legacy all-tiles states (slicing before the
/// per-position projector changes nothing), and only the padding-tile rows
/// are dropped.
#[test]
fn sub_max_real_tiles_keep_the_legacy_real_rows_byte_identical() {
    let model = tiny_vl_model();
    let pv = tile_pixels(&[[Some(400), None, None, None]]);
    let ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
    let mask = mlxcel_core::from_slice_i32(&[1, 0, 0, 0], &[1, 1, 4]);

    let full = model.compute_cross_attention_states(&pv, &ids, &mask);
    let sub = model.compute_cross_attention_states_for_tiles(&pv, &ids, &mask, &[1]);

    assert_eq!(
        mlxcel_core::array_shape(&full),
        vec![1, 4 * V_PATCHES, HIDDEN]
    );
    assert_eq!(mlxcel_core::array_shape(&sub), vec![1, V_PATCHES, HIDDEN]);

    let expected = states_rows(&full, 0, V_PATCHES);
    assert_eq!(
        max_abs_diff(&sub, &expected),
        0.0,
        "real-tile states must be byte-identical to the legacy states' real rows"
    );
}

/// (b) Full tile count: selection is a no-op and the states must be
/// byte-identical to the legacy all-tiles path (zero behavior change).
#[test]
fn full_tile_count_states_are_byte_identical_to_legacy() {
    let model = tiny_vl_model();
    let pv = tile_pixels(&[[Some(410), Some(411), Some(412), Some(413)]]);
    let ids = mlxcel_core::from_slice_i32(&[6], &[1, 1]); // (2, 2) tiling
    let mask = mlxcel_core::from_slice_i32(&[1, 1, 1, 1], &[1, 1, 4]);

    let full = model.compute_cross_attention_states(&pv, &ids, &mask);
    let via_tiles = model.compute_cross_attention_states_for_tiles(&pv, &ids, &mask, &[4]);

    assert_eq!(
        max_abs_diff(&full, &via_tiles),
        0.0,
        "num_tiles == max_num_tiles must take the identical legacy path"
    );
}

/// Ragged multi-image: each image contributes exactly its real-tile rows,
/// media-major, byte-identical to the legacy states' corresponding rows.
#[test]
fn ragged_multi_image_states_concatenate_real_rows() {
    let model = tiny_vl_model();
    let pv = tile_pixels(&[
        [Some(420), None, None, None],
        [Some(430), Some(431), None, None],
    ]);
    let ids = mlxcel_core::from_slice_i32(&[1, 2], &[1, 2]);
    let mask = mlxcel_core::from_slice_i32(&[1, 0, 0, 0, 1, 1, 0, 0], &[1, 2, 4]);

    let full = model.compute_cross_attention_states(&pv, &ids, &mask);
    let sub = model.compute_cross_attention_states_for_tiles(&pv, &ids, &mask, &[1, 2]);

    assert_eq!(
        mlxcel_core::array_shape(&full),
        vec![1, 2 * 4 * V_PATCHES, HIDDEN]
    );
    assert_eq!(
        mlxcel_core::array_shape(&sub),
        vec![1, 3 * V_PATCHES, HIDDEN]
    );

    // Media 0 tile 0 = rows 0..5; media 1 tiles 0..2 = rows 20..30.
    let media0 = states_rows(&full, 0, V_PATCHES);
    let media1 = states_rows(&full, 4 * V_PATCHES, 6 * V_PATCHES);
    let expected = mlxcel_core::concatenate(&media0, &media1, 1);
    assert_eq!(
        max_abs_diff(&sub, &expected),
        0.0,
        "ragged selection must keep each image's real rows in media-major order"
    );
}

/// `prepare_cross_attention_states` threads the processor's real tile counts:
/// the logits match a manual stash of the real-tile states, and genuinely
/// differ from stashing the legacy all-tiles states (whose garbage padding
/// rows the old unmasked cross-attention consulted).
#[test]
fn prepare_stashes_real_tile_states() {
    let model = tiny_vl_model();
    let pv = tile_pixels(&[[Some(440), None, None, None]]);
    let ids = mlxcel_core::from_slice_i32(&[1], &[1, 1]);
    let mask = mlxcel_core::from_slice_i32(&[1, 0, 0, 0], &[1, 1, 4]);

    let vl_forward = |m: &MllamaVLModel| {
        let ids = input_ids();
        let mut caches = LanguageModel::make_caches(m);
        let logits = LanguageModel::forward(m, &ids, &mut caches, None);
        mlxcel_core::eval(&logits);
        logits
    };

    let inputs = MllamaImageInputs {
        pixel_values: mlxcel_core::copy(&pv),
        aspect_ratio_ids: mlxcel_core::copy(&ids),
        aspect_ratio_mask: mlxcel_core::copy(&mask),
        num_tiles: vec![1],
    };
    model.prepare_cross_attention_states(&inputs);
    assert!(model.has_cross_attention_states());
    let logits_prepare = vl_forward(&model);

    let sub = model.compute_cross_attention_states_for_tiles(&pv, &ids, &mask, &[1]);
    model.set_cross_attention_states(sub);
    let logits_sub = vl_forward(&model);
    assert_eq!(
        max_abs_diff(&logits_prepare, &logits_sub),
        0.0,
        "prepare must stash exactly the real-tile states"
    );

    let full = model.compute_cross_attention_states(&pv, &ids, &mask);
    model.set_cross_attention_states(full);
    let logits_full = vl_forward(&model);
    assert!(
        max_abs_diff(&logits_prepare, &logits_full) > 1e-6,
        "the legacy all-tiles states let the unmasked cross-attention consult \
         padding-lane rows; dropping them must actually change the logits"
    );
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
