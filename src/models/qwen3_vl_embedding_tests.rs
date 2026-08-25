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

//! Qwen3-VL-Embedding gates.
//!
//! Three properties are checked without a checkpoint: the instruction
//! punctuation rule, the placeholder expansion that turns the single
//! `<|image_pad|>` the chat template emits into the processor's patch count,
//! and the exact prompt the checkpoint's own `chat_template.jinja` renders
//! (which needs the template file but no weights).
//!
//! The remaining gates load `Qwen/Qwen3-VL-Embedding-2B` and soft-skip when it
//! is not downloaded. They cover the epic's self-consistency contract
//! (identical inputs, a padded batch against single inputs, unrelated
//! sentences) and the issue's cross-modal margin.

use super::{
    DEFAULT_INSTRUCTION, expand_image_placeholders, render_prompt, with_trailing_punctuation,
};
use crate::embeddings::{EmbedOptions, EmbeddingEngine, ImageInput, load_embedding_model};
use crate::models::ModelType;
use crate::models::embedding_test_support::{cosine, local_checkpoint, mlx_test_guard};
use crate::models::vl_embedding_test_images::{bar_chart, beach};
use crate::server::chat_template::ChatTemplateProcessor;

const CHECKPOINT: &str = "Qwen/Qwen3-VL-Embedding-2B";
const IMAGE_PAD: i32 = 151_655;
/// `text_config.hidden_size` of the 2B checkpoint.
const DIM: usize = 2048;

/// Margins the issue sets for the real-checkpoint gates.
const TEXT_MARGIN: f32 = 0.15;
const IMAGE_MARGIN: f32 = 0.1;
/// The epic's shared self-consistency tolerances.
const IDENTICAL_TOLERANCE: f32 = 1e-6;
const PADDED_TOLERANCE: f32 = 1e-3;
const UNRELATED_CEILING: f32 = 0.5;

// Instruction formatting.

#[test]
fn instruction_gets_trailing_period_only_when_it_ends_mid_sentence() {
    assert_eq!(
        with_trailing_punctuation("Represent the user's input"),
        "Represent the user's input."
    );
    assert_eq!(
        with_trailing_punctuation(DEFAULT_INSTRUCTION),
        DEFAULT_INSTRUCTION
    );
    assert_eq!(
        with_trailing_punctuation("Find the matching image?"),
        "Find the matching image?"
    );
    // A non-ASCII terminator counts as punctuation, so nothing is appended.
    assert_eq!(with_trailing_punctuation("画像を表す。"), "画像を表す。");
}

// Placeholder expansion.

#[test]
fn expand_image_placeholders_repeats_each_placeholder_in_order() {
    let ids = [1, IMAGE_PAD, 2, IMAGE_PAD, 3];
    let mask = [1, 1, 1, 1, 1];
    let (out_ids, out_mask) =
        expand_image_placeholders(&ids, &mask, IMAGE_PAD, &[3, 2]).expect("expands");
    assert_eq!(
        out_ids,
        vec![
            1, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 2, IMAGE_PAD, IMAGE_PAD, 3
        ]
    );
    assert_eq!(out_mask, vec![1; 8]);
}

#[test]
fn expand_image_placeholders_keeps_padding_flags() {
    let ids = [1, IMAGE_PAD, 9, 9];
    let mask = [1, 1, 0, 0];
    let (out_ids, out_mask) =
        expand_image_placeholders(&ids, &mask, IMAGE_PAD, &[2]).expect("expands");
    assert_eq!(out_ids, vec![1, IMAGE_PAD, IMAGE_PAD, 9, 9]);
    assert_eq!(out_mask, vec![1, 1, 1, 0, 0]);
}

#[test]
fn expand_image_placeholders_rejects_a_count_mismatch() {
    let ids = [1, IMAGE_PAD, 2];
    let mask = [1, 1, 1];
    let err = expand_image_placeholders(&ids, &mask, IMAGE_PAD, &[4, 4])
        .expect_err("two counts for one placeholder is a mismatch");
    assert!(format!("{err:#}").contains("placeholder"), "{err:#}");

    let err = expand_image_placeholders(&ids, &mask, IMAGE_PAD, &[0])
        .expect_err("a zero-token image is a mismatch");
    assert!(format!("{err:#}").contains("zero visual tokens"), "{err:#}");
}

// Chat template rendering (template file only, no weights).

fn checkpoint_template() -> Option<ChatTemplateProcessor> {
    let dir = local_checkpoint(CHECKPOINT)?;
    ChatTemplateProcessor::from_model_path(&dir).expect("the chat template parses")
}

#[test]
fn format_text_renders_system_instruction_and_generation_prompt() {
    let Some(template) = checkpoint_template() else {
        return;
    };
    let rendered =
        render_prompt(&template, DEFAULT_INSTRUCTION, "a photo of a dog", false).expect("renders");
    assert_eq!(
        rendered,
        "<|im_start|>system\nRepresent the user's input.<|im_end|>\n\
         <|im_start|>user\na photo of a dog<|im_end|>\n\
         <|im_start|>assistant\n"
    );
}

#[test]
fn format_text_inserts_one_image_pad_for_an_image_row() {
    let Some(template) = checkpoint_template() else {
        return;
    };
    let rendered = render_prompt(&template, "Represent the image.", "", true).expect("renders");
    assert_eq!(
        rendered,
        "<|im_start|>system\nRepresent the image.<|im_end|>\n\
         <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n\
         <|im_start|>assistant\n"
    );
}

// Real checkpoint.

/// Load the checkpoint through the production embedding loader.
fn engine() -> Option<EmbeddingEngine> {
    let dir = local_checkpoint(CHECKPOINT)?;
    let _runtime = crate::initialize_runtime();
    let loaded = load_embedding_model(&dir).expect("Qwen3-VL-Embedding loads");
    assert_eq!(loaded.model_type, ModelType::Qwen3VLEmbedding);
    assert_eq!(loaded.limits.dim, DIM);
    assert!(loaded.model.supports_images());
    Some(EmbeddingEngine::new(loaded, 16))
}

#[test]
fn qwen3_vl_embedding_text_gates_hold_on_the_real_checkpoint() {
    let _guard = mlx_test_guard();
    let Some(engine) = engine() else {
        return;
    };

    // Index 3 repeats index 0 verbatim: two byte-identical rows of one
    // right-padded batch must score cosine 1.0.
    let texts: Vec<String> = [
        "A woman playing with her dog on a beach",
        "A person and a pet by the sea",
        "Quarterly revenue report for a software company",
        "A woman playing with her dog on a beach",
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
        assert!(
            vector.values.iter().all(|v| v.is_finite()),
            "a non-finite component reached the caller"
        );
        let norm: f32 = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "vector is not unit length: {norm}"
        );
    }

    let identical = cosine(&reply.vectors[0].values, &reply.vectors[3].values);
    assert!(
        (identical - 1.0).abs() < IDENTICAL_TOLERANCE,
        "identical inputs scored {identical}, not 1.0 within {IDENTICAL_TOLERANCE}"
    );

    let paraphrase = cosine(&reply.vectors[0].values, &reply.vectors[1].values);
    let unrelated = cosine(&reply.vectors[0].values, &reply.vectors[2].values);
    assert!(
        unrelated < UNRELATED_CEILING,
        "unrelated sentences scored {unrelated}, above {UNRELATED_CEILING}"
    );
    assert!(
        paraphrase - unrelated >= TEXT_MARGIN,
        "paraphrase {paraphrase} beats unrelated {unrelated} by less than {TEXT_MARGIN}"
    );

    // A padded batch must reproduce the single-input vectors.
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
fn qwen3_vl_embedding_image_gates_hold_on_the_real_checkpoint() {
    let _guard = mlx_test_guard();
    let Some(engine) = engine() else {
        return;
    };
    let options = EmbedOptions::default();

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
    assert_eq!(chart.vectors[0].shape, vec![DIM]);
    assert_eq!(shore.vectors[0].shape, vec![DIM]);
    assert!(
        chart.vectors[0].values.iter().all(|v| v.is_finite())
            && shore.vectors[0].values.iter().all(|v| v.is_finite()),
        "a non-finite component reached the caller"
    );
    assert!(
        chart.prompt_tokens > 0 && shore.prompt_tokens > 0,
        "an image row reported no prompt tokens"
    );

    let captions: Vec<String> = [
        "a bar chart of quarterly revenue growth",
        "a sunny beach with the sea and the sky",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let text = engine.embed_texts(&captions, &options).expect("embeds");

    let chart_match = cosine(&text.vectors[0].values, &chart.vectors[0].values);
    let chart_miss = cosine(&text.vectors[0].values, &shore.vectors[0].values);
    assert!(
        chart_match - chart_miss >= IMAGE_MARGIN,
        "the chart caption prefers its image by only {}, below {IMAGE_MARGIN} \
         (match {chart_match}, miss {chart_miss})",
        chart_match - chart_miss
    );

    let beach_match = cosine(&text.vectors[1].values, &shore.vectors[0].values);
    let beach_miss = cosine(&text.vectors[1].values, &chart.vectors[0].values);
    assert!(
        beach_match - beach_miss >= IMAGE_MARGIN,
        "the beach caption prefers its image by only {}, below {IMAGE_MARGIN} \
         (match {beach_match}, miss {beach_miss})",
        beach_match - beach_miss
    );

    // Re-embedding the same image must be deterministic.
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
