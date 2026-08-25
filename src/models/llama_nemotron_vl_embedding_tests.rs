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

//! Llama-Nemotron-VL-Embed gates.
//!
//! Without a checkpoint: the weight-key normalization, the tiling budget and
//! the `pixel_shuffle` plus `mlp1` shape contract on random weights, and the
//! bidirectionality of the text stack (a late token must move an early hidden
//! state, which a causal stack cannot do).
//!
//! With `nvidia/llama-nemotron-embed-vl-1b-v2` present: the epic's
//! self-consistency contract and the issue's retrieval and cross-modal
//! margins. Every gate soft-skips when the checkpoint is absent.

use mlxcel_core::weights::WeightMap;

use super::{
    DEFAULT_NUM_IMAGE_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
    LlamaNemotronVLEmbeddingModel, sanitize_nemotron_vl_weights,
};
use crate::embeddings::model::EmbeddingModel;
use crate::embeddings::{EmbedOptions, EmbeddingEngine, ImageInput, load_embedding_model};
use crate::models::ModelType;
use crate::models::embedding_test_support::{
    Rng, cosine, local_checkpoint, mlx_test_guard, to_vec,
};
use crate::models::llama_nemotron_vl_tiling::NemotronTiling;
use crate::models::vl_embedding_test_images::{bar_chart, beach};
use crate::vision::internvl::InternVLConnector;

const CHECKPOINT: &str = "nvidia/llama-nemotron-embed-vl-1b-v2";
/// `llm_config.hidden_size` of the published checkpoint.
const DIM: usize = 2048;
/// `vision_config.hidden_size`.
const VISION_HIDDEN: i32 = 1152;
/// Patches one `512x512` tile produces at `patch_size: 16`.
const PATCHES_PER_TILE: i32 = 1024;
/// `<IMG_CONTEXT>` in the checkpoint's vocabulary (`img_context_token_id`).
const IMG_CONTEXT_TOKEN_ID: i32 = 128_258;
/// `<|begin_of_text|>`, which the tokenizer's post-processor prepends.
const BEGIN_OF_TEXT_TOKEN_ID: i32 = 128_000;

const TEXT_MARGIN: f32 = 0.15;
/// The epic's contract for two byte-identical rows of one batch.
///
/// It holds for the inputs below. It is not a property of every batch shape on
/// every backend: on the CUDA validation host, three identical rows of some
/// token lengths differ by roughly `1e-5` to `1e-4` in cosine, one bf16 ulp
/// appearing at a middle decoder layer and compounding. The same sweep
/// reproduces the same drift on the already-merged `Qwen3Embedding` family, so
/// it is a property of the shared bf16 batched decode path rather than of this
/// port. If this assertion ever fires, re-run the sweep across batch sizes and
/// token lengths before suspecting the forward pass.
const IDENTICAL_TOLERANCE: f32 = 1e-6;
const PADDED_TOLERANCE: f32 = 1e-3;
const UNRELATED_CEILING: f32 = 0.5;

// Weight-key normalization.

#[test]
fn sanitize_drops_the_vision_head_and_maps_the_language_model_prefix() {
    let mut rng = Rng::new(0x1345_0001);
    let mut weights = WeightMap::new();
    for key in [
        "language_model.embed_tokens.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.norm.weight",
        "vision_model.vision_model.post_layernorm.weight",
        "vision_model.vision_model.head.probe",
        "vision_model.vision_model.head.attention.in_proj_weight",
        "mlp1.0.weight",
        "mlp1.1.weight",
        "mlp1.3.weight",
        "lm_head.weight",
        "language_model.layers.0.self_attn.rotary_emb.inv_freq",
        "language_model.embed_positions.position_ids",
    ] {
        rng.insert(&mut weights, key, &[2, 2], 0.1);
    }

    sanitize_nemotron_vl_weights(&mut weights);

    for key in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
        "vision_model.vision_model.post_layernorm.weight",
        "mlp1.0.weight",
        "mlp1.1.weight",
        "mlp1.3.weight",
    ] {
        assert!(weights.contains_key(key), "{key} should survive sanitize");
    }
    assert!(
        weights.keys().all(|key| !key.contains(".head.")),
        "the SigLIP attention-pooling head is still present"
    );
    assert!(
        weights
            .keys()
            .all(|key| !key.starts_with("language_model.")),
        "a language_model. prefix survived"
    );
    assert!(!weights.contains_key("lm_head.weight"));
    assert!(
        weights
            .keys()
            .all(|key| !key.contains("inv_freq") && !key.ends_with("position_ids")),
        "a non-parameter buffer survived"
    );
}

// Tiling.

#[test]
fn a_square_image_uses_one_tile_and_a_wide_image_uses_the_budget() {
    let tiling = NemotronTiling::default();
    assert_eq!(tiling.image_size, 512);
    assert_eq!(tiling.max_tiles, 6);

    // 512x512 covers exactly one tile: no split, so no thumbnail either.
    assert_eq!(tiling.tiles(&bar_chart(512, 512)).len(), 1);

    // A 6:1 strip lands on the 6x1 grid, plus the thumbnail.
    let wide = tiling.closest_aspect_ratio(3072, 512);
    assert_eq!(wide, (6, 1));
    assert_eq!(tiling.tiles(&beach(3072, 512)).len(), 7);

    // A 2:3 page lands on a 2x3 grid, plus the thumbnail.
    assert_eq!(tiling.closest_aspect_ratio(1024, 1536), (2, 3));
    assert_eq!(tiling.tiles(&beach(1024, 1536)).len(), 7);
}

#[test]
fn preprocess_emits_channels_last_tiles_in_the_siglip_range() {
    let tiling = NemotronTiling {
        image_size: 64,
        ..NemotronTiling::default()
    };
    let (pixels, counts) = tiling.preprocess(&[bar_chart(64, 64)]);
    assert_eq!(counts, vec![1]);
    assert_eq!(mlxcel_core::array_shape(&pixels), vec![1, 64, 64, 3]);
    let values = to_vec(&pixels);
    assert!(
        values.iter().all(|v| (-1.0..=1.0).contains(v)),
        "SigLIP normalization maps [0, 255] onto [-1, 1]"
    );
    // The chart is mostly white, which normalizes to +1.
    let white = values.iter().filter(|v| (**v - 1.0).abs() < 1e-6).count();
    assert!(white > values.len() / 2, "the chart canvas should be white");
}

#[test]
fn image_block_expands_to_num_image_token_per_tile() {
    // The prompt the family emits carries one placeholder; the forward pass
    // expands it to `num_image_token * tiles`. This checks the arithmetic the
    // expansion is given, against the reference's 256 tokens per tile.
    let tiling = NemotronTiling::default();
    for (width, height, expected_tiles) in
        [(512u32, 512u32, 1usize), (3072, 512, 7), (1024, 1536, 7)]
    {
        let tiles = tiling.tiles(&beach(width, height)).len();
        assert_eq!(tiles, expected_tiles, "{width}x{height}");
        assert_eq!(
            tiles * DEFAULT_NUM_IMAGE_TOKEN,
            expected_tiles * 256,
            "{width}x{height}"
        );
    }
}

// Connector shapes.

#[test]
fn pixel_shuffle_then_mlp1_maps_one_tile_to_256_language_tokens() {
    let _guard = mlx_test_guard();
    let mut rng = Rng::new(0x1345_0002);
    let mut weights = WeightMap::new();
    let shuffled = VISION_HIDDEN * 4;
    rng.insert(&mut weights, "mlp1.0.weight", &[shuffled], 0.1);
    rng.insert(&mut weights, "mlp1.0.bias", &[shuffled], 0.1);
    rng.insert(&mut weights, "mlp1.1.weight", &[DIM as i32, shuffled], 0.05);
    rng.insert(&mut weights, "mlp1.1.bias", &[DIM as i32], 0.05);
    rng.insert(
        &mut weights,
        "mlp1.3.weight",
        &[DIM as i32, DIM as i32],
        0.05,
    );
    rng.insert(&mut weights, "mlp1.3.bias", &[DIM as i32], 0.05);

    let connector =
        InternVLConnector::from_weights(&weights, "mlp1", 1e-5, 0.5, 0, 0).expect("mlp1 loads");
    let features = rng.tensor(&[1, PATCHES_PER_TILE, VISION_HIDDEN], 0.5);
    let projected = connector.forward(&features);
    assert_eq!(
        mlxcel_core::array_shape(&projected),
        vec![1, DEFAULT_NUM_IMAGE_TOKEN as i32, DIM as i32]
    );
    assert!(to_vec(&projected).iter().all(|v| v.is_finite()));
}

// Real checkpoint.

fn model_and_engine() -> Option<EmbeddingEngine> {
    let dir = local_checkpoint(CHECKPOINT)?;
    let _runtime = crate::initialize_runtime();
    let loaded = load_embedding_model(&dir).expect("Llama-Nemotron-VL-Embed loads");
    assert_eq!(loaded.model_type, ModelType::LlamaNemotronVLEmbedding);
    assert_eq!(loaded.limits.dim, DIM);
    assert!(loaded.model.supports_images());
    Some(EmbeddingEngine::new(loaded, 16))
}

#[test]
fn format_text_is_identity_for_text_and_the_document_form_for_an_image() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(CHECKPOINT) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let config = crate::embeddings::loader::read_embedding_config(&dir).expect("config parses");
    let model = LlamaNemotronVLEmbeddingModel::load(&dir, &config).expect("loads");

    assert_eq!(
        model.format_text("query: what is the revenue", None),
        "query: what is the revenue"
    );
    assert_eq!(
        model.format_text("passage: Revenue grew 12%.", Some("ignored")),
        "passage: Revenue grew 12%."
    );
    assert_eq!(
        model.format_text("", None),
        format!("passage: {IMG_START_TOKEN}{IMG_CONTEXT_TOKEN}{IMG_END_TOKEN} ")
    );
}

#[test]
fn the_document_prompt_expands_to_256_tokens_per_tile() {
    // End-to-end on the checkpoint's own tokenizer: the prompt the family
    // emits carries exactly one `<IMG_CONTEXT>`, and the production expansion
    // grows it to `tiles * 256` without disturbing anything else. A 6:1 image
    // uses six tiles plus a thumbnail, a square image one tile.
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(CHECKPOINT) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let config = crate::embeddings::loader::read_embedding_config(&dir).expect("config parses");
    let model = LlamaNemotronVLEmbeddingModel::load(&dir, &config).expect("loads");
    let tokenizer = crate::embeddings::tokenize::strip_padding_and_truncation(
        crate::tokenizer::load_tokenizer(&dir).expect("tokenizer loads"),
    );

    let prompt = model.format_text("", None);
    let row = crate::embeddings::tokenize::encode_row(
        &tokenizer,
        &prompt,
        crate::embeddings::tokenize::EncodeOptions {
            add_special_tokens: true,
            max_length: 4096,
            with_token_type_ids: false,
        },
    )
    .expect("encodes");
    let ids: Vec<i32> = row.ids.iter().map(|&id| id as i32).collect();
    assert_eq!(
        ids.iter().filter(|&&id| id == IMG_CONTEXT_TOKEN_ID).count(),
        1,
        "the document prompt should carry exactly one placeholder: {ids:?}"
    );
    assert_eq!(ids[0], BEGIN_OF_TEXT_TOKEN_ID, "the tokenizer prepends BOS");

    let mask = vec![1i32; ids.len()];
    let tiling = NemotronTiling::default();
    for (width, height, tiles) in [(512u32, 512u32, 1usize), (3072, 512, 7)] {
        assert_eq!(tiling.tiles(&bar_chart(width, height)).len(), tiles);
        let (expanded, expanded_mask) =
            crate::models::qwen3_vl_embedding::expand_image_placeholders(
                &ids,
                &mask,
                IMG_CONTEXT_TOKEN_ID,
                &[tiles * DEFAULT_NUM_IMAGE_TOKEN],
            )
            .expect("expands");
        assert_eq!(
            expanded.len(),
            ids.len() - 1 + tiles * DEFAULT_NUM_IMAGE_TOKEN,
            "{width}x{height}"
        );
        assert_eq!(expanded_mask.len(), expanded.len());
        assert_eq!(
            expanded
                .iter()
                .filter(|&&id| id == IMG_CONTEXT_TOKEN_ID)
                .count(),
            tiles * DEFAULT_NUM_IMAGE_TOKEN
        );
    }
}

#[test]
fn bidirectional_prefill_lets_an_early_token_see_a_later_one() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(CHECKPOINT) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let config = crate::embeddings::loader::read_embedding_config(&dir).expect("config parses");
    let model = LlamaNemotronVLEmbeddingModel::load(&dir, &config).expect("loads");

    // A 96-token ramp, then the same ramp with only the last token changed.
    // Under a causal mask every earlier hidden state would be untouched.
    let length = 96i32;
    let mut ids: Vec<i32> = (0..length).map(|i| 1000 + i * 7).collect();
    let mask = vec![1i32; length as usize];
    let baseline = pooled_hidden(&model, &ids, &mask);
    ids[(length - 1) as usize] += 13;
    let perturbed = pooled_hidden(&model, &ids, &mask);

    let moved = cosine(&baseline, &perturbed);
    assert!(
        moved < 0.9999,
        "changing the last token left the pooled vector at cosine {moved}; \
         the stack is not bidirectional"
    );
}

/// Pooled `[DIM]` vector for one unpadded row, read through the trait.
fn pooled_hidden(model: &LlamaNemotronVLEmbeddingModel, ids: &[i32], mask: &[i32]) -> Vec<f32> {
    let length = ids.len() as i32;
    let input_ids = mlxcel_core::from_slice_i32(ids, &[1, length]);
    let attention_mask = mlxcel_core::from_slice_i32(mask, &[1, length]);
    let output = model
        .embed(&crate::embeddings::model::EmbeddingBatch {
            input_ids: &input_ids,
            attention_mask: &attention_mask,
            token_type_ids: None,
            images: None,
        })
        .expect("embeds");
    to_vec(&output.embeddings)
}

#[test]
fn llama_nemotron_vl_text_gates_hold_on_the_real_checkpoint() {
    let _guard = mlx_test_guard();
    let Some(engine) = model_and_engine() else {
        return;
    };

    // Index 3 repeats index 0 verbatim.
    let texts: Vec<String> = [
        "query: what does the chart say about 2023 revenue",
        "passage: Revenue grew 12% in 2023 driven by subscriptions.",
        "passage: The museum opens at nine in the morning.",
        "query: what does the chart say about 2023 revenue",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let reply = engine
        .embed_texts(&texts, &EmbedOptions::default())
        .expect("embeds");
    assert_eq!(reply.vectors.len(), 4);
    for vector in &reply.vectors {
        assert_eq!(vector.shape, vec![DIM]);
        assert!(vector.values.iter().all(|v| v.is_finite()));
        let norm: f32 = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "vector is not unit length: {norm}"
        );
    }

    let identical = cosine(&reply.vectors[0].values, &reply.vectors[3].values);
    assert!(
        (identical - 1.0).abs() < IDENTICAL_TOLERANCE,
        "identical inputs scored {identical}"
    );

    let relevant = cosine(&reply.vectors[0].values, &reply.vectors[1].values);
    let irrelevant = cosine(&reply.vectors[0].values, &reply.vectors[2].values);
    assert!(
        irrelevant < UNRELATED_CEILING,
        "unrelated passages scored {irrelevant}, above {UNRELATED_CEILING}"
    );
    assert!(
        relevant - irrelevant >= TEXT_MARGIN,
        "the revenue passage beats the museum passage by only {}",
        relevant - irrelevant
    );

    let single = engine
        .embed_texts(&texts[..1], &EmbedOptions::default())
        .expect("embeds one");
    let drift = cosine(&single.vectors[0].values, &reply.vectors[0].values);
    assert!(
        (drift - 1.0).abs() < PADDED_TOLERANCE,
        "a padded batch drifted from the single-input vector: cosine {drift}"
    );
}

#[test]
fn llama_nemotron_vl_image_gates_hold_on_the_real_checkpoint() {
    let _guard = mlx_test_guard();
    let Some(engine) = model_and_engine() else {
        return;
    };
    let options = EmbedOptions::default();

    // A 6-tile landscape plus its thumbnail: 7 * 256 image tokens must embed
    // without error and without truncating the block.
    let wide = engine
        .embed_image(
            ImageInput {
                image: bar_chart(3072, 512),
            },
            &options,
        )
        .expect("embeds a 7-tile chart");
    assert_eq!(wide.vectors[0].shape, vec![DIM]);
    assert!(wide.vectors[0].values.iter().all(|v| v.is_finite()));

    let chart = engine
        .embed_image(
            ImageInput {
                image: bar_chart(768, 512),
            },
            &options,
        )
        .expect("embeds the chart");
    let shore = engine
        .embed_image(
            ImageInput {
                image: beach(768, 512),
            },
            &options,
        )
        .expect("embeds the beach");
    assert!(
        chart.vectors[0].values.iter().all(|v| v.is_finite())
            && shore.vectors[0].values.iter().all(|v| v.is_finite())
    );

    let queries: Vec<String> = [
        "query: a bar chart of quarterly revenue growth",
        "query: a sunny beach with the sea and the sky",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let text = engine.embed_texts(&queries, &options).expect("embeds");

    let chart_match = cosine(&text.vectors[0].values, &chart.vectors[0].values);
    let chart_miss = cosine(&text.vectors[0].values, &shore.vectors[0].values);
    assert!(
        chart_match > chart_miss,
        "the chart query prefers the beach image (match {chart_match}, miss {chart_miss})"
    );

    let beach_match = cosine(&text.vectors[1].values, &shore.vectors[0].values);
    let beach_miss = cosine(&text.vectors[1].values, &chart.vectors[0].values);
    assert!(
        beach_match > beach_miss,
        "the beach query prefers the chart image (match {beach_match}, miss {beach_miss})"
    );

    let repeat = engine
        .embed_image(
            ImageInput {
                image: bar_chart(768, 512),
            },
            &options,
        )
        .expect("embeds the chart again");
    let stability = cosine(&chart.vectors[0].values, &repeat.vectors[0].values);
    assert!(
        (stability - 1.0).abs() < IDENTICAL_TOLERANCE,
        "the same image scored {stability} against itself across two calls"
    );
}
