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

//! Hunyuan-VL (`hunyuan_vl`) real-model parity / smoke tests.
//!
//! Gated on the presence of `models/HunyuanOCR-mlx-4bit`; the tests
//! `eprintln!` + return when the model is absent, so they are inert in CI and
//! on machines without the checkpoint. With the model present they exercise:
//!
//! - model-type detection of `hunyuan_vl`
//! - the smart-resize processor's raster patch rows and placeholder count
//! - a text-only forward pass producing finite logits (degenerate XD-RoPE)
//! - the image path (count-based placeholder expansion + merger framing rows
//!   + 4D XD-RoPE prefill) producing finite logits
//!
//! Run with the model present via:
//! ```text
//! cargo test --release --test hunyuan_vl_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::processors::hunyuan_vl::HunyuanVlProcessor;

const MODEL_NAME: &str = "HunyuanOCR-mlx-4bit";
const VOCAB: i32 = 120_818;

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping Hunyuan-VL test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

#[test]
fn detects_hunyuan_vl_model_type() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::HunyuanVLM);
}

#[test]
fn processor_emits_raster_rows_and_count() {
    let proc = HunyuanVlProcessor::default();
    let img = fixture_image();
    let (pixels, grid) = proc.preprocess_with_grid(std::slice::from_ref(&img));
    let (t, gh, gw) = grid[0];
    assert_eq!(t, 1);
    assert_eq!(gh % 2, 0);
    assert_eq!(gw % 2, 0);
    assert_eq!(mlxcel_core::array_shape(&pixels), vec![gh * gw, 768]);
    assert_eq!(proc.placeholder_count(gh, gw), (gh / 2) * (gw / 2 + 1) + 2);
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load HunyuanOCR-mlx-4bit");
    assert!(model.is_vlm(), "Hunyuan-VL must register as a VLM");

    let tokens: Vec<i32> = tokenizer
        .encode("Hello, world.", true)
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();
    let input_ids = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);

    let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
    let logits =
        mlxcel_core::generate::LanguageModel::forward(&model, &input_ids, &mut caches, None);
    mlxcel_core::eval(&logits);

    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(*shape.last().unwrap(), VOCAB);

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, VOCAB]);
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "text-only logits must be finite"
    );
}

#[test]
fn image_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load HunyuanOCR-mlx-4bit");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode("Read the text in this image.", true)
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "Read the text in this image.",
        &images,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect()
        },
    )
    .expect("prepare VLM embeddings")
    .expect("Hunyuan-VL should produce embeddings for an image request");

    // The spliced run count is mh * (mw + 1) + 2, matching the merger rows.
    let placeholders = prompt_tokens.iter().filter(|&&t| t == 120_120).count();
    assert!(placeholders > 0, "placeholder run must be spliced");

    let input_ids = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
    let logits = mlxcel_core::generate::LanguageModel::forward_with_embeddings(
        &model,
        &input_ids,
        prepared.embeddings.inputs_embeds.as_ref(),
        &mut caches,
        None,
    );
    mlxcel_core::eval(&logits);

    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(*shape.last().unwrap(), VOCAB);

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, VOCAB]);
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "image-conditioned logits must be finite"
    );
}
