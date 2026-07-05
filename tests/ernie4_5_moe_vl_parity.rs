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

//! ERNIE-4.5 MoE VL (`ernie4_5_moe_vl`) real-model parity / smoke tests.
//!
//! Gated on the presence of `models/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`; the
//! tests `eprintln!` + return when the model is absent, so they are inert in CI
//! and on machines without the checkpoint. With the model present they
//! exercise:
//!
//! - model-type detection of `ernie4_5_moe_vl`
//! - the smart-resize processor's 588-wide merge-window patch rows
//! - a text-only forward pass producing finite logits of the right vocab dim
//!   (exercising the text expert bank + degenerate MRoPE)
//! - the image path (placeholder expansion + resampler + merge + dual-bank
//!   dispatch + 3D MRoPE) producing finite logits
//!
//! Run with the model present via:
//! ```text
//! cargo test --release --test ernie4_5_moe_vl_parity -- --test-threads=1
//! ```
//!
//! Serial execution matters: two tests each load the ~16 GB 28B checkpoint, so
//! the default parallel runner doubles the resident footprint and can produce
//! non-finite logits under memory pressure.

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::processors::ernie4_5_vl::Ernie45VlProcessor;

const MODEL_NAME: &str = "ERNIE-4.5-VL-28B-A3B-Thinking-4bit";
const VOCAB: i32 = 103_424;

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping ERNIE-4.5-VL test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

#[test]
fn detects_ernie4_5_moe_vl_model_type() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::Ernie45MoeVLM);
}

#[test]
fn processor_emits_merge_window_rows() {
    // The 224x224 fixture smart-resizes to a multiple of 28; rows are
    // (patches, 588) with grid (1, gh, gw) and gh/gw even.
    let proc = Ernie45VlProcessor::default();
    let img = fixture_image();
    let (pixels, grid) = proc.preprocess_with_grid(std::slice::from_ref(&img));
    let (t, gh, gw) = grid[0];
    assert_eq!(t, 1);
    assert_eq!(gh % 2, 0);
    assert_eq!(gw % 2, 0);
    assert_eq!(mlxcel_core::array_shape(&pixels), vec![gh * gw, 588]);
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load ERNIE-4.5-VL");
    assert!(model.is_vlm(), "ERNIE-4.5-VL must register as a VLM");

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
        "logits last dim must equal the ERNIE vocab size"
    );

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, VOCAB]);
    let row_flat = mlxcel_core::reshape(&row, &[VOCAB]);
    let max = mlxcel_core::max_all(&row_flat);
    let argmax = mlxcel_core::argmax(&row_flat, -1, false);
    mlxcel_core::eval(&max);
    mlxcel_core::eval(&argmax);
    let max_v = mlxcel_core::item_f32(&max);
    let arg_v = mlxcel_core::item_i32(&argmax);
    assert!(
        max_v.is_finite(),
        "text-only logits must be finite (max {max_v}, argmax id {arg_v})"
    );
}

/// Diagnostic: greedy-decode a few steps after an image prefill and dump the
/// raw ids plus their decode (with and without special tokens) so a
/// special-token-only degeneration is visible. Ignored by default; run with
/// `--ignored --nocapture` when debugging.
#[test]
#[ignore]
fn debug_image_greedy_ids() {
    let Some(dir) = model_dir() else { return };
    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load ERNIE-4.5-VL");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode("What color dominates this image?", true)
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();
    eprintln!("prompt ids (pre-splice): {prompt_tokens:?}");

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "What color dominates this image?",
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
    .expect("prepare")
    .expect("embeddings");
    eprintln!(
        "prompt len post-splice: {} (placeholders: {})",
        prompt_tokens.len(),
        prompt_tokens.iter().filter(|&&t| t == 100_295).count()
    );

    let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
    let input_ids = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut logits = mlxcel_core::generate::LanguageModel::forward_with_embeddings(
        &model,
        &input_ids,
        prepared.embeddings.inputs_embeds.as_ref(),
        &mut caches,
        None,
    );

    let mut generated: Vec<i32> = Vec::new();
    for step in 0..16 {
        let shape = mlxcel_core::array_shape(&logits);
        let last = shape[1] - 1;
        let row = mlxcel_core::slice(&logits, &[0, last, 0], &[1, last + 1, shape[2]]);
        let row = mlxcel_core::reshape(&row, &[shape[2]]);
        let next = mlxcel_core::argmax(&row, -1, false);
        mlxcel_core::eval(&next);
        let id = mlxcel_core::item_i32(&next);
        generated.push(id);
        let piece_raw = tokenizer.decode(&[id as u32], false).unwrap_or_default();
        let piece_skip = tokenizer.decode(&[id as u32], true).unwrap_or_default();
        eprintln!("step {step}: id {id} raw {piece_raw:?} skip {piece_skip:?}");

        let next_ids = mlxcel_core::from_slice_i32(&[id], &[1, 1]);
        logits = mlxcel_core::generate::LanguageModel::forward_with_sequence_id(
            &model,
            &next_ids,
            None,
            &mut caches,
            None,
        );
    }
    eprintln!("generated: {generated:?}");

    // Replicate the CLI's decode_generated_text to find where the display
    // pipeline loses the text.
    let all: Vec<u32> = prompt_tokens
        .iter()
        .map(|&x| x as u32)
        .chain(generated.iter().map(|&x| x as u32))
        .collect();
    let prompt_u32: Vec<u32> = prompt_tokens.iter().map(|&x| x as u32).collect();
    let gen_u32: Vec<u32> = generated.iter().map(|&x| x as u32).collect();
    let full_text = tokenizer.decode(&all, false);
    let prompt_text = tokenizer.decode(&prompt_u32, false);
    let gen_text = tokenizer.decode(&gen_u32, false);
    eprintln!("decode(prompt+gen, keep-special): {full_text:?}");
    eprintln!("decode(prompt, keep-special):     {prompt_text:?}");
    eprintln!("decode(gen only):                 {gen_text:?}");
}

#[test]
fn image_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load ERNIE-4.5-VL");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode("User: What is in this image?\nAssistant:", true)
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "User: What is in this image?\nAssistant:",
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
    .expect("ERNIE-4.5-VL should produce embeddings for an image request");

    // The spliced run is framed by <|IMAGE_START|> / <|IMAGE_END|> and carries
    // (gh/2)*(gw/2) placeholders, matching the resampler output rows.
    let placeholders = prompt_tokens.iter().filter(|&&t| t == 100_295).count();
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
