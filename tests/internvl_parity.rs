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

//! InternVL (`internvl_chat`) real-model parity / smoke tests.
//!
//! These tests are gated on the presence of `models/internvl3-1b`; they
//! `eprintln!` + return when the model is absent, so they are inert in CI and
//! on machines without the checkpoint. With the model present they exercise:
//!
//! - model-type detection of `internvl_chat`
//! - the dynamic-tiling image processor's tile/pixel shapes
//! - a text-only forward pass producing finite logits of the right vocab dim
//! - the image path (token expansion + merged embeddings + a forward pass)
//!
//! Run with the model present via:
//! ```text
//! cargo test --release --test internvl_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::processors::ImageProcessor;
use mlxcel::vision::processors::internvl::InternVLProcessor;

const MODEL_NAME: &str = "internvl3-1b";
const QWEN2_VOCAB: i32 = 151674;

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping InternVL test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

#[test]
fn detects_internvl_chat_model_type() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::InternVLChatVLM);
}

#[test]
fn processor_tiles_square_fixture_into_one_tile() {
    // The repo fixture is 224x224 (square) -> closest to the 1x1 aspect ratio
    // -> exactly one 448x448 tile, with no thumbnail (blocks == 1).
    let proc = InternVLProcessor::new(448, 1, 12, true);
    let img = fixture_image();
    let pixels = proc.preprocess(std::slice::from_ref(&img));
    let shape = mlxcel_core::array_shape(&pixels);
    assert_eq!(shape, vec![1, 3, 448, 448]);
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load internvl3-1b");
    assert!(model.is_vlm(), "InternVL must register as a VLM");

    // A short text-only prompt (no images).
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
        QWEN2_VOCAB,
        "logits last dim must equal the Qwen2 vocab size"
    );

    // Last-position logits must be finite.
    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, QWEN2_VOCAB]);
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    let max_val = mlxcel_core::item_f32(&max);
    assert!(max_val.is_finite(), "text-only logits must be finite");
}

#[test]
fn image_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load internvl3-1b");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode(
            "<|im_start|>user\nWhat is in this image?<|im_end|>\n<|im_start|>assistant\n",
            true,
        )
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "What is in this image?",
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
    .expect("InternVL should produce embeddings for an image request");

    // 224x224 fixture -> 1 tile -> 256 image-context tokens were inserted.
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
    assert_eq!(*shape.last().unwrap(), QWEN2_VOCAB);

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, QWEN2_VOCAB]);
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "image-conditioned logits must be finite"
    );
}
