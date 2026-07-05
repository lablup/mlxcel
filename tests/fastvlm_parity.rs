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

//! FastVLM (`llava_qwen2`) real-model parity / smoke tests.
//!
//! Gated on the presence of `models/FastVLM-0.5B-bf16`; the tests `eprintln!` +
//! return when the model is absent, so they are inert in CI and on machines
//! without the checkpoint. With the model present they exercise:
//!
//! - model-type detection of `llava_qwen2`
//! - the pad-to-square processor's `(B, 3, 1024, 1024)` output
//! - a text-only forward pass producing finite logits of the Qwen2 vocab dim
//! - the image path (`-200` splice + 256-token expansion + LLaVA merge + forward)
//!
//! Run with the model present via:
//! ```text
//! cargo test --release --test fastvlm_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::processors::ImageProcessor;
use mlxcel::vision::processors::fastvlm::FastvlmProcessor;

const MODEL_NAME: &str = "FastVLM-0.5B-bf16";
const VOCAB: i32 = 151_936;

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping FastVLM test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

#[test]
fn detects_fastvlm_model_type() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::FastVLM);
}

#[test]
fn processor_emits_1024_square_channels_first() {
    let proc = FastvlmProcessor::default();
    let img = fixture_image();
    let out = proc.preprocess(std::slice::from_ref(&img));
    assert_eq!(mlxcel_core::array_shape(&out), vec![1, 3, 1024, 1024]);
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load FastVLM-0.5B-bf16");
    assert!(model.is_vlm(), "FastVLM must register as a VLM");

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
        "logits last dim must equal the Qwen2 vocab"
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

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load FastVLM-0.5B-bf16");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode(
            "<|im_start|>user\n<image>\nWhat is in this image?<|im_end|>\n<|im_start|>assistant\n",
            true,
        )
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "<|im_start|>user\n<image>\nWhat is in this image?<|im_end|>\n<|im_start|>assistant\n",
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
    .expect("FastVLM should produce embeddings for an image request");

    // One image expands to 256 sentinel positions.
    let sentinels = prompt_tokens.iter().filter(|&&t| t == -200).count();
    assert_eq!(sentinels, 256, "one image must expand to 256 -200 tokens");

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
