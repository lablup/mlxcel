// Copyright 2025-2026 Lablup Inc.
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

//! The prompt templates, `drop_idx` and the tokenized prompt against
//! upstream's `QwenImageTextEncoder` (its values are in
//! `tests/fixtures/imagegen/qwen_image_parity.safetensors`), and (ignored,
//! GPU and the real checkpoint) the conditioner output against the oracle's.

use std::path::PathBuf;

use super::*;
use crate::imagegen::qwen_image::parity_tests::{GEN_PROMPT, model_dir, oracle, parity};
use crate::imagegen::testing::{rel_rms, values};

#[test]
fn templates_match_upstream() {
    assert_eq!(
        generation_text("a cat"),
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n\
         <|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        edit_text("make it blue", 2),
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n\
         <|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> \
         <image2><|vision_start|><|image_pad|><|vision_end|>make it blue<|im_end|>\n\
         <|im_start|>assistant\n"
    );
}

fn tokenizer() -> MlxcelTokenizer {
    let root = model_dir().expect("models/mlx/qwen-image-2.1 is present");
    crate::tokenizer::load_tokenizer(&root.join("processor")).expect("tokenizer loads")
}

fn ints(key: &str) -> Vec<i32> {
    let p = parity();
    let a = mlxcel_core::astype(&p[key], mlxcel_core::dtype::FLOAT32);
    mlxcel_core::utils::array_to_vec_f32(&a)
        .into_iter()
        .map(|v| v as i32)
        .collect()
}

#[test]
#[ignore = "needs the real Qwen-Image-2.1 processor tokenizer (models/mlx/qwen-image-2.1)"]
fn drop_idx_and_prompt_ids_match_the_processor_tokenizer() {
    let tokenizer = tokenizer();
    let drop = drop_idx(&tokenizer).unwrap();
    assert_eq!(drop, ints("gen.drop_idx")[0] as usize);
    assert_eq!(drop, 14);
    let ids = generation_ids(&tokenizer, GEN_PROMPT, drop).unwrap();
    assert_eq!(ids, ints("gen.input_ids"));
    // `max_length = 1024 + drop_idx` truncation.
    let long = "word ".repeat(1500);
    assert_eq!(
        generation_ids(&tokenizer, &long, drop).unwrap().len(),
        MAX_PROMPT_TOKENS + drop
    );
}

#[test]
#[ignore = "needs the real Qwen-Image-2.1 processor tokenizer (models/mlx/qwen-image-2.1)"]
fn edit_prompts_with_vision_markers_or_too_many_tokens_are_bad_requests() {
    let tokenizer = tokenizer();
    for prompt in ["<|image_pad|> again".to_string(), "word ".repeat(1500)] {
        let err = check_edit_prompt(&tokenizer, &prompt).unwrap_err();
        assert!(
            err.downcast_ref::<InvalidImageRequest>().is_some(),
            "{err:#}"
        );
    }
    check_edit_prompt(&tokenizer, "make the teapot blue").unwrap();
}

/// Bound on the conditioner's text-position hidden states against
/// upstream's: both run the 36-layer Qwen3-VL in bf16, and the measured
/// drift is 3.7e-2 (generate and edit alike), spread evenly over the
/// positions; the pipeline still lands at 52 dB from upstream's PNG.
const COND_TEXT_REL_RMS: f64 = 6e-2;

#[test]
#[ignore = "loads the real Qwen-Image-2.1 text encoder (GPU, ~17 GB)"]
fn conditioner_matches_upstream() {
    let root = model_dir().expect("models/mlx/qwen-image-2.1 is present");
    let encoder = TextEncoder::load(&root).unwrap();
    let encoded = encoder.encode(GEN_PROMPT).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&encoded.hidden),
        [1, ints("gen.input_ids").len() as i32 - 14, 4096]
    );
    if let Some(out) = std::env::var_os("MLXCEL_QWEN_IMAGE_OUT") {
        let mut w = mlxcel_core::weights::WeightMap::new();
        w.insert("gen".into(), mlxcel_core::copy(&encoded.hidden));
        crate::imagegen::testing::write_bf16_safetensors(
            &PathBuf::from(out).join("cond_gen.safetensors"),
            &w,
        );
    }
    if let Some(full) = oracle("gen_full") {
        let rel = rel_rms(&values(&encoded.hidden), &values(&full["cond"]));
        eprintln!("qwen_image conditioner (generate) rel-RMS vs upstream: {rel:.3e}");
        assert!(rel < COND_TEXT_REL_RMS, "conditioner rel-RMS {rel}");
    }
    let reference = image::open(
        std::env::var_os("MLXCEL_QWEN_IMAGE_ORACLE_DIR")
            .map(PathBuf::from)
            .map(|d| d.join("edit_ref_rgba.png"))
            .filter(|p| p.is_file())
            .unwrap_or_else(|| root.join("assets").join("missing.png")),
    );
    let Ok(reference) = reference else {
        return;
    };
    let edited = encoder
        .encode_edit(
            crate::imagegen::qwen_image::parity_tests::EDIT_PROMPT,
            &[reference.to_rgba8()],
        )
        .unwrap();
    let mask = edited.image_pad_mask.as_ref().unwrap();
    assert_eq!(
        mask.iter().filter(|m| **m).count() as i32,
        ints("edit.pad_count")[0]
    );
    assert_eq!(mask.len() as i32, ints("edit.cond_len")[0]);
    if let Some(out) = std::env::var_os("MLXCEL_QWEN_IMAGE_OUT") {
        let mut w = mlxcel_core::weights::WeightMap::new();
        w.insert("edit".into(), mlxcel_core::copy(&edited.hidden));
        crate::imagegen::testing::write_bf16_safetensors(
            &PathBuf::from(out).join("cond_edit.safetensors"),
            &w,
        );
    }
    if let Some(edit) = oracle("edit_full") {
        // Only the text positions reach the transformer: every
        // `<|image_pad|>` row is replaced there by the reference's VAE tokens.
        let rows = |a: &MlxArray, pads: bool| -> Vec<f32> {
            values(a)
                .chunks(4096)
                .zip(mask)
                .filter(|(_, m)| **m == pads)
                .flat_map(|(row, _)| row.to_vec())
                .collect()
        };
        let text = rel_rms(&rows(&edited.hidden, false), &rows(&edit["cond"], false));
        let pads = rel_rms(&rows(&edited.hidden, true), &rows(&edit["cond"], true));
        eprintln!(
            "qwen_image conditioner (edit) rel-RMS vs upstream: text positions {text:.3e}, \
             image-pad positions {pads:.3e} (dropped by the transformer)"
        );
        assert!(
            text < COND_TEXT_REL_RMS,
            "edit conditioner text rel-RMS {text}"
        );
    }
}
