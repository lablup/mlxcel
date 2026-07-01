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

//! SmolVLM / SmolVLM2 (`smolvlm`) parity / smoke tests.
//!
//! Two layers of coverage:
//!
//! - **Checkpoint-free reference parity** (always runs): the image processor's
//!   tile/pixel shapes and SigLIP normalization, and the `<image>` prompt
//!   expansion, checked against values derived directly from the upstream
//!   SmolVLM algorithm. These exercise the net-new port logic without a real
//!   checkpoint. (The pixel-shuffle connector math has its own reference oracle
//!   in `src/vision/smolvlm.rs`'s inline tests.)
//! - **Checkpoint-gated smoke** (`eprintln!` + return when the model is
//!   absent, so CI stays green): model-type detection, a text-only forward
//!   producing finite logits, and the image path (token expansion + merged
//!   embeddings + a forward pass).
//!
//! Run the gated tests with the model present via:
//! ```text
//! cargo test --release --test smolvlm_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::smolvlm_prompt::insert_smolvlm_image_tokens;
use mlxcel::vision::processors::ImageProcessor;
use mlxcel::vision::processors::smolvlm::SmolVLMProcessor;

const MODEL_NAME: &str = "SmolVLM-Instruct";

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping SmolVLM test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

// ----- Checkpoint-free reference parity -----

#[test]
fn processor_single_global_tile_shape() {
    // Splitting disabled -> one global tile resized to image_size x image_size.
    let proc = SmolVLMProcessor::new(64, false, 4);
    let img = fixture_image();
    let pixels = proc.preprocess(std::slice::from_ref(&img));
    let shape = mlxcel_core::array_shape(&pixels);
    assert_eq!(shape, vec![1, 3, 64, 64]);
}

#[test]
fn processor_siglip_normalization_range() {
    // SigLIP normalization maps [0, 255] -> [-1, 1]; a solid-white image must
    // land at +1.0 (255/255 = 1.0 -> (1.0 - 0.5) / 0.5 = 1.0).
    let proc = SmolVLMProcessor::new(8, false, 4);
    let white = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        8,
        8,
        image::Rgb([255, 255, 255]),
    ));
    let pixels = proc.preprocess(std::slice::from_ref(&white));
    let corner = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
    mlxcel_core::eval(&corner);
    assert!((mlxcel_core::item_f32(&corner) - 1.0).abs() < 1e-6);
}

#[test]
fn prompt_expansion_matches_reference_token_stream() {
    // Reference: one <image> placeholder expands to
    // <fake> <global-img> <image>*(num_image_token * tiles) <fake>.
    const IMAGE: i32 = 49153;
    const FAKE: i32 = 49152;
    const GLOBAL: i32 = 49155;
    let num_image_token = 4;
    let tiles = 1;

    let mut prompt = vec![1, IMAGE, 2];
    let stats =
        insert_smolvlm_image_tokens(&mut prompt, &[tiles], num_image_token, IMAGE, FAKE, GLOBAL)
            .expect("expansion happens for a prompt with an <image> placeholder");

    let mut expected = vec![1, FAKE, GLOBAL];
    expected.extend(std::iter::repeat_n(IMAGE, num_image_token * tiles));
    expected.push(FAKE);
    expected.push(2);

    assert_eq!(prompt, expected);
    assert_eq!(stats.total_image_tokens, num_image_token * tiles);
    // The number of surviving <image> tokens must equal the feature-row count
    // the merge will scatter, or the merge would panic on a length mismatch.
    assert_eq!(
        prompt.iter().filter(|&&t| t == IMAGE).count(),
        num_image_token * tiles
    );
}

// ----- Checkpoint-gated smoke -----

#[test]
fn detects_smolvlm_model_type() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::SmolVLM);
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load SmolVLM");
    assert!(model.is_vlm(), "SmolVLM must register as a VLM");

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
    let vocab = *shape.last().unwrap();
    assert!(vocab > 0, "logits must carry a vocab dimension");

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, vocab]);
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

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load SmolVLM");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode(
            "<|im_start|>User:<image>What is in this image?<end_of_utterance>\nAssistant:",
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
    .expect("SmolVLM should produce embeddings for an image request");

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
    let vocab = *shape.last().unwrap();
    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, vocab]);
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "image-conditioned logits must be finite"
    );
}
