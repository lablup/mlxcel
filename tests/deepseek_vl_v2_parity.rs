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

//! DeepSeek-VL2 (`deepseek_vl_v2`) real-model parity / smoke tests.
//!
//! Gated on the presence of `models/deepseek-vl2-small-4bit`; the tests
//! `eprintln!` + return when the model is absent, so they are inert in CI and
//! on machines without the checkpoint. With the model present they exercise:
//!
//! - model-type detection of `deepseek_vl_v2`
//! - the candidate-resolution processor's global / tile shapes and the flat
//!   placeholder count (421 for a square fixture at patch 14, ds 2)
//! - a text-only forward pass producing finite logits of the right vocab dim
//! - the image path (token expansion + merged embeddings + a forward pass)
//!
//! Run with the model present via:
//! ```text
//! cargo test --release --test deepseek_vl_v2_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::processors::deepseek_vl2::DeepSeekVl2Processor;

const MODEL_NAME: &str = "deepseek-vl2-small-4bit";
const VOCAB: i32 = 102_400;

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping DeepSeek-VL2 test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

fn processor() -> DeepSeekVl2Processor {
    DeepSeekVl2Processor::new(
        vec![
            (384, 384),
            (384, 768),
            (768, 384),
            (768, 768),
            (1152, 384),
            (384, 1152),
            (1152, 1152),
        ],
        14,
        2,
    )
}

#[test]
fn detects_deepseek_vl_v2_model_type() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::DeepSeekVL2);
}

#[test]
fn processor_square_fixture_is_one_global_one_tile_421_tokens() {
    // A square fixture best-fits the (384, 384) candidate -> a (1, 1) grid: one
    // global thumbnail plus one local tile, 210 + 1 + 210 = 421 placeholders.
    let proc = processor();
    let img = fixture_image();
    let pre = proc.preprocess(std::slice::from_ref(&img));

    assert_eq!(mlxcel_core::array_shape(&pre.global), vec![1, 384, 384, 3]);
    assert_eq!(mlxcel_core::array_shape(&pre.tiles), vec![1, 384, 384, 3]);
    assert_eq!(pre.crops, vec![(1, 1)]);
    assert_eq!(pre.placeholder_counts, vec![421]);
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load deepseek-vl2-small-4bit");
    assert!(model.is_vlm(), "DeepSeek-VL2 must register as a VLM");

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
    assert_eq!(
        *shape.last().unwrap(),
        VOCAB,
        "logits last dim must equal the DeepSeek-V2 vocab size"
    );

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

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load deepseek-vl2-small-4bit");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode(
            "<|User|>: <image>\nDescribe the image.\n\n<|Assistant|>:",
            true,
        )
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "Describe the image.",
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
    .expect("DeepSeek-VL2 should produce embeddings for an image request");

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
