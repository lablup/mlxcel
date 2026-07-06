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

//! Qwen3-Omni MoE (`qwen3_omni_moe`) real-model parity / smoke tests.
//!
//! Gated on `models/qwen3-omni-30b-a3b-instruct-4bit`; inert when absent.
//! Run serially (two tests load the ~18 GB checkpoint):
//! ```text
//! cargo test --release --test qwen3_omni_moe_parity -- --test-threads=1
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};

const MODEL_NAME: &str = "qwen3-omni-30b-a3b-instruct-4bit";
const VOCAB: i32 = 152_064;

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping Qwen3-Omni test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

#[test]
fn detects_qwen3_omni_moe_model_type() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::Qwen3OmniMoe);
}

#[test]
fn audio_out_len_formula() {
    use mlxcel::audio::qwen3_omni_moe::audio_out_len;
    assert_eq!(audio_out_len(100), 13);
    assert_eq!(audio_out_len(500), 65);
    assert_eq!(audio_out_len(163), 21);
}

/// Diagnostic: greedy-decode a few steps after a TEMPLATED image prefill and
/// dump raw ids. Ignored by default.
#[test]
#[ignore]
fn debug_templated_image_greedy_ids() {
    let Some(dir) = model_dir() else { return };
    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load qwen3-omni");

    let images = vec![fixture_image()];
    // Variant matrix: isolate what differs between the working manual prompt
    // and the CLI-rendered one (image marker in user content, no system turn).
    let variants: Vec<(&str, String)> = vec![
        (
            "cli-render (no system, image-before-text)",
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What colors are in this image?<|im_end|>\n<|im_start|>assistant\n".to_string(),
        ),
        (
            "system + image-before-text",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What colors are in this image?<|im_end|>\n<|im_start|>assistant\n".to_string(),
        ),
        (
            "system + splice fallback (image-after-text)",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat colors are in this image?<|im_end|>\n<|im_start|>assistant\n".to_string(),
        ),
    ];

    for (label, prompt) in variants {
        eprintln!("==== variant: {label} ====");
        let mut prompt_tokens: Vec<i32> = tokenizer
            .encode(&prompt, false)
            .expect("tokenize")
            .iter()
            .map(|&t| t as i32)
            .collect();

        let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
            &model,
            &mut prompt_tokens,
            &prompt,
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
            "post-splice len {}, image tokens {}",
            prompt_tokens.len(),
            prompt_tokens.iter().filter(|&&t| t == 151_655).count()
        );

        let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
        let input_ids =
            mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let mut logits = mlxcel_core::generate::LanguageModel::forward_with_embeddings(
            &model,
            &input_ids,
            prepared.embeddings.inputs_embeds.as_ref(),
            &mut caches,
            None,
        );

        let mut pieces = String::new();
        for _step in 0..12 {
            let shape = mlxcel_core::array_shape(&logits);
            let last = shape[1] - 1;
            let row = mlxcel_core::slice(&logits, &[0, last, 0], &[1, last + 1, shape[2]]);
            let row = mlxcel_core::reshape(&row, &[shape[2]]);
            let next = mlxcel_core::argmax(&row, -1, false);
            mlxcel_core::eval(&next);
            let id = mlxcel_core::item_i32(&next);
            pieces.push_str(&format!(
                "[{}]{}",
                id,
                tokenizer.decode(&[id as u32], false).unwrap_or_default()
            ));
            let next_ids = mlxcel_core::from_slice_i32(&[id], &[1, 1]);
            logits = mlxcel_core::generate::LanguageModel::forward_with_sequence_id(
                &model,
                &next_ids,
                None,
                &mut caches,
                None,
            );
        }
        eprintln!("greedy: {pieces}");
    }
}

#[test]
fn image_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load qwen3-omni");
    assert!(model.is_vlm());

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode("What colors are in this image?", true)
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "What colors are in this image?",
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
    .expect("Qwen3-Omni should produce embeddings for an image request");

    let placeholders = prompt_tokens.iter().filter(|&&t| t == 151_655).count();
    assert!(placeholders > 0);

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
    assert!(mlxcel_core::item_f32(&max).is_finite());
}
