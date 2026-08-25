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

//! ColQwen2.5 tests.
//!
//! The synthetic half runs the text path on a bare `Qwen2VLModel` built
//! from deterministic weights, through the same
//! [`super::text_input_embeddings`] and [`super::token_vectors`] the model
//! uses, so the mask, the head-free forward and the per-token
//! normalization are exercised as product code. The vision half is covered
//! by the real-checkpoint gates below rather than by a hand-built windowed
//! ViT, because a 32-block synthetic tower would restate the encoder
//! instead of testing it.
//!
//! Every test that drives MLX takes
//! [`crate::models::embedding_test_support::mlx_test_guard`].

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use super::{
    ColQwen25Model, IMAGE_DOCUMENT_PROMPT, normalize_patch_embed_layout, rewrite_colqwen25_key,
    sanitize_colqwen25_weights, text_input_embeddings, token_vectors,
};
use crate::embeddings::model::EmbeddingModel;
use crate::models::col_late_interaction::QUERY_AUGMENTATION_TOKENS;
use crate::models::embedding_test_support::{
    Rng, local_checkpoint, max_abs_diff, mlx_test_guard, to_vec,
};
use crate::models::qwen2_vl::{Qwen2VLConfig, Qwen2VLModel};

const HIDDEN: i32 = 16;
const INTERMEDIATE: i32 = 32;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const HEAD_DIM: i32 = 8;
const VOCAB: i32 = 64;
const LAYERS: usize = 2;
const EMBED_DIM: i32 = 8;

const COLQWEN25_BASE: &str = "vidore/colqwen2.5-base";

fn tiny_config() -> Qwen2VLConfig {
    serde_json::from_value(json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        // The three M-RoPE sections must sum to head_dim / 2.
        "rope_scaling": {"mrope_section": [2, 1, 1], "type": "default"},
        "head_dim": HEAD_DIM,
        "tie_word_embeddings": true
    }))
    .expect("the tiny Qwen2-VL config parses")
}

/// Deterministic backbone weights plus the retrieval projection.
fn tiny_weights() -> WeightMap {
    let mut rng = Rng::new(0xC0_1B_E2_5D);
    let mut weights = WeightMap::new();
    let q_out = HEADS * HEAD_DIM;
    let kv_out = KV_HEADS * HEAD_DIM;

    rng.insert(
        &mut weights,
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        0.5,
    );
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}");
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![q_out, HIDDEN]),
            ("self_attn.k_proj.weight", vec![kv_out, HIDDEN]),
            ("self_attn.v_proj.weight", vec![kv_out, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, q_out]),
            ("mlp.gate_proj.weight", vec![INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", vec![INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, INTERMEDIATE]),
        ] {
            rng.insert(&mut weights, &format!("{p}.{name}"), &shape, 0.2);
        }
        for bias in ["q_proj", "k_proj", "v_proj"] {
            let width = if bias == "q_proj" { q_out } else { kv_out };
            rng.insert(
                &mut weights,
                &format!("{p}.self_attn.{bias}.bias"),
                &[width],
                0.1,
            );
        }
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            rng.insert(&mut weights, &format!("{p}.{norm}.weight"), &[HIDDEN], 0.1);
        }
    }
    rng.insert(&mut weights, "model.norm.weight", &[HIDDEN], 0.1);
    rng.insert(
        &mut weights,
        "custom_text_proj.weight",
        &[EMBED_DIM, HIDDEN],
        0.4,
    );
    rng.insert(&mut weights, "custom_text_proj.bias", &[EMBED_DIM], 0.2);
    weights
}

/// `[B, L]` ids and mask for a right-padded batch of rows.
fn padded_batch(rows: &[Vec<i32>]) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>, usize) {
    let width = rows.iter().map(Vec::len).max().unwrap();
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for row in rows {
        ids.extend(row.iter().copied());
        ids.extend(std::iter::repeat_n(0, width - row.len()));
        mask.extend(std::iter::repeat_n(1, row.len()));
        mask.extend(std::iter::repeat_n(0, width - row.len()));
    }
    let shape = [rows.len() as i32, width as i32];
    (
        mlxcel_core::from_slice_i32(&ids, &shape),
        mlxcel_core::from_slice_i32(&mask, &shape),
        width,
    )
}

/// The family's text path over a padded batch, read back row-major.
fn text_vectors(model: &Qwen2VLModel, projection: &UnifiedLinear, rows: &[Vec<i32>]) -> Vec<f32> {
    let (ids, mask, width) = padded_batch(rows);
    let embeddings = text_input_embeddings(model, &ids);
    let out = token_vectors(model, projection, &ids, &embeddings, &mask);
    mlxcel_core::eval(&out);
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![rows.len() as i32, width as i32, EMBED_DIM]
    );
    to_vec(&out)
}

#[test]
fn native_layout_keys_remap_to_base_layout() {
    // The five rules the native `ColQwen2ForRetrieval` layout needs.
    assert_eq!(
        rewrite_colqwen25_key("embedding_proj_layer.weight"),
        "custom_text_proj.weight"
    );
    assert_eq!(
        rewrite_colqwen25_key("vlm.model.language_model.layers.0.self_attn.q_proj.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        rewrite_colqwen25_key("vlm.model.visual.blocks.0.attn.qkv.weight"),
        "vision_tower.blocks.0.attn.qkv.weight"
    );
    assert_eq!(
        rewrite_colqwen25_key("vlm.model.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(
        rewrite_colqwen25_key("vlm.embedding_proj_layer.bias"),
        "custom_text_proj.bias"
    );

    // The `vidore/colqwen2.5-base` layout: the raw HuggingFace tower name
    // becomes the one the Qwen2.5-VL encoder loads from.
    assert_eq!(
        rewrite_colqwen25_key("visual.patch_embed.proj.weight"),
        "vision_tower.patch_embed.proj.weight"
    );
    assert_eq!(
        rewrite_colqwen25_key("model.norm.weight"),
        "model.norm.weight"
    );
    assert_eq!(
        rewrite_colqwen25_key("custom_text_proj.weight"),
        "custom_text_proj.weight"
    );
    // An mlx conversion is already in the target layout and is untouched.
    assert_eq!(
        rewrite_colqwen25_key("vision_tower.merger.mlp.0.weight"),
        "vision_tower.merger.mlp.0.weight"
    );
    // A `language_model.` wrapper without the `model.` root is stripped.
    assert_eq!(
        rewrite_colqwen25_key("language_model.model.norm.weight"),
        "model.norm.weight"
    );
}

#[test]
fn sanitize_rewrites_every_key_of_a_native_map() {
    let _guard = mlx_test_guard();
    let mut weights = WeightMap::new();
    for key in [
        "vlm.model.language_model.norm.weight",
        "vlm.model.visual.merger.ln_q.weight",
        "vlm.embedding_proj_layer.weight",
    ] {
        weights.insert(
            key.to_string(),
            mlxcel_core::from_slice_f32(&[1.0, 2.0], &[2]),
        );
    }
    let sanitized = sanitize_colqwen25_weights(weights);
    let mut keys: Vec<&String> = sanitized.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "custom_text_proj.weight",
            "model.norm.weight",
            "vision_tower.merger.ln_q.weight"
        ]
    );
}

#[test]
fn patch_embed_layout_is_converted_from_the_pytorch_conv3d_form() {
    let _guard = mlx_test_guard();
    // Raw HuggingFace: [out, in, kT, kH, kW]. The encoder wants the mlx
    // conversion's [out, kT, kH, kW, in], so this must be permuted.
    let (out, channels, kt, k) = (4i32, 3i32, 2i32, 2i32);
    let count = (out * channels * kt * k * k) as usize;
    let values: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let mut weights = WeightMap::new();
    weights.insert(
        "vision_tower.patch_embed.proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&values, &[out, channels, kt, k, k]),
    );
    assert!(normalize_patch_embed_layout(&mut weights, 3));
    let converted = &weights["vision_tower.patch_embed.proj.weight"];
    assert_eq!(
        mlxcel_core::array_shape(converted),
        vec![out, kt, k, k, channels]
    );
    // Element count is preserved and the permutation is the transpose, not a
    // reinterpretation: element [0, 0, 0, 0, 1] of the output is element
    // [0, 1, 0, 0, 0] of the input, which is `kt * k * k` = 8.
    mlxcel_core::eval(converted);
    let flat = to_vec(converted);
    assert_eq!(flat.len(), count);
    assert_eq!(flat[1], 8.0);

    // An mlx conversion is already channels-last and is left untouched.
    let mut already = WeightMap::new();
    already.insert(
        "vision_tower.patch_embed.proj.weight".to_string(),
        mlxcel_core::from_slice_f32(&values, &[out, kt, k, k, channels]),
    );
    assert!(!normalize_patch_embed_layout(&mut already, 3));
    assert_eq!(
        mlxcel_core::array_shape(&already["vision_tower.patch_embed.proj.weight"]),
        vec![out, kt, k, k, channels]
    );

    // A checkpoint without the tower is not an error.
    let mut empty = WeightMap::new();
    assert!(!normalize_patch_embed_layout(&mut empty, 3));
}

#[test]
fn forward_hidden_then_head_matches_forward() {
    let _guard = mlx_test_guard();
    let config = tiny_config();
    let weights = tiny_weights();
    let model = Qwen2VLModel::from_weights(&weights, &config).expect("the tiny backbone loads");
    // With `tie_word_embeddings: true` the generation head is the embedding
    // table used as a linear, so the same tensor reproduces it exactly.
    let head = UnifiedLinear::from_weights(&weights, "model.embed_tokens", 0, 0).unwrap();

    let rows = vec![vec![3, 9, 17, 42, 5, 1]];
    let (ids, mask, width) = padded_batch(&rows);
    let mask4 = mlxcel_core::utils::create_causal_padding_mask(&mask, 0);

    model.clear_mrope_state();
    let mut caches = model.make_caches();
    let hidden = model.forward_hidden(&ids, None, &mut caches, Some(&mask4), None);
    let via_split = head.forward(&hidden);
    mlxcel_core::eval(&via_split);
    assert_eq!(
        mlxcel_core::array_shape(&hidden),
        vec![1, width as i32, HIDDEN],
        "forward_hidden stops at the final norm"
    );

    model.clear_mrope_state();
    let mut caches = model.make_caches();
    let logits = model.forward_impl(&ids, None, &mut caches, Some(&mask4));
    mlxcel_core::eval(&logits);

    let split = to_vec(&via_split);
    let whole = to_vec(&logits);
    assert_eq!(split.len(), whole.len());
    assert_eq!(
        max_abs_diff(&split, &whole),
        0.0,
        "splitting the head out of the forward pass must be token-exact"
    );
}

#[test]
fn batched_text_rows_match_single_rows() {
    let _guard = mlx_test_guard();
    let config = tiny_config();
    let weights = tiny_weights();
    let model = Qwen2VLModel::from_weights(&weights, &config).unwrap();
    let projection = UnifiedLinear::from_weights(&weights, "custom_text_proj", 0, 0).unwrap();

    let a = vec![5, 12, 33, 7, 2, 61, 18];
    let b = vec![9, 21, 4];
    let batched = text_vectors(&model, &projection, &[a.clone(), b.clone()]);
    let dim = EMBED_DIM as usize;
    let width = a.len();

    let alone_a = text_vectors(&model, &projection, std::slice::from_ref(&a));
    let alone_b = text_vectors(&model, &projection, std::slice::from_ref(&b));

    let inside_a = &batched[..a.len() * dim];
    assert!(
        max_abs_diff(&alone_a, inside_a) < 1e-3,
        "the long row drifted by {} inside the batch",
        max_abs_diff(&alone_a, inside_a)
    );

    let second = &batched[width * dim..];
    let inside_b = &second[..b.len() * dim];
    assert!(
        max_abs_diff(&alone_b, inside_b) < 1e-3,
        "the padded row drifted by {} inside the batch",
        max_abs_diff(&alone_b, inside_b)
    );
    for row in b.len()..width {
        let slice = &second[row * dim..(row + 1) * dim];
        assert!(
            slice.iter().all(|&v| v == 0.0),
            "padding row {row} is not zeroed: {slice:?}"
        );
    }
    for row in 0..b.len() {
        let slice = &second[row * dim..(row + 1) * dim];
        let norm = slice.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "real row {row} has norm {norm}");
    }

    // Determinism: the same call twice gives the same bytes.
    assert_eq!(alone_a, text_vectors(&model, &projection, &[a]));
}

#[test]
fn stale_image_positions_do_not_leak_into_a_text_batch() {
    let _guard = mlx_test_guard();
    let config = tiny_config();
    let weights = tiny_weights();
    let model = Qwen2VLModel::from_weights(&weights, &config).unwrap();
    let projection = UnifiedLinear::from_weights(&weights, "custom_text_proj", 0, 0).unwrap();

    let rows = vec![vec![5, 12, 33, 7]];
    let clean = text_vectors(&model, &projection, &rows);

    // Install the kind of `[3, 1, L]` grid an image request leaves behind:
    // spatial positions that repeat instead of increasing.
    let grid: Vec<i32> = vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1];
    model.set_mrope_state(mlxcel_core::from_slice_i32(&grid, &[3, 1, 4]), 0);
    let after = text_vectors(&model, &projection, &rows);
    assert_eq!(
        clean, after,
        "a stale M-RoPE grid must be cleared before a text batch"
    );
}

// Real-checkpoint gates. Each soft-skips when the checkpoint is absent.

#[test]
fn real_colqwen25_base_loads_and_projects_to_128() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(COLQWEN25_BASE) else {
        return;
    };
    let config = crate::embeddings::loader::read_embedding_config(&dir).unwrap();
    let model = ColQwen25Model::load(&dir, &config).expect("the base checkpoint loads");
    assert_eq!(model.embedding_dim(), 128);
    assert!(model.multi_vector());
    assert!(model.supports_images());
    assert_eq!(model.vlm.image_token_id, 151_655);
    assert_eq!(model.vlm.spatial_merge_size, 2);
    // `preprocessor_config.json` caps the tower at 768 * 28 * 28 pixels.
    assert_eq!(model.vlm.processor.max_pixels, 768 * 28 * 28);

    assert_eq!(model.format_text("", None), IMAGE_DOCUMENT_PROMPT);
    let query = model.format_text("What was the total revenue in 2023?", None);
    assert!(query.starts_with("Query: "));
    assert_eq!(
        query.matches("<|endoftext|>").count(),
        QUERY_AUGMENTATION_TOKENS
    );
}
