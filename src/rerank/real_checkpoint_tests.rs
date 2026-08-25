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

//! Real-checkpoint gates for `/v1/rerank`.
//!
//! Each test soft-skips when its checkpoint is absent, following the
//! convention of `src/embeddings/real_checkpoint_tests.rs`. Fetch with:
//!
//! ```sh
//! mlxcel download BAAI/bge-reranker-v2-m3
//! mlxcel download cross-encoder/ms-marco-MiniLM-L6-v2
//! mlxcel download Alibaba-NLP/gte-reranker-modernbert-base
//! mlxcel download mlx-community/Qwen3-Reranker-0.6B-4bit
//! mlxcel download Qwen/Qwen3-VL-Reranker-2B
//! ```
//!
//! Where a checkpoint's model card publishes reference scores they are the
//! gate (`bge-reranker-v2-m3`). Where it does not, the gate is the ordering
//! plus a margin, because no PyTorch or `transformers` install exists on the
//! validation host to produce a reference of our own.

use std::path::PathBuf;

use crate::models::embedding_test_support::{local_checkpoint, mlx_test_guard};
use crate::models::get_model_type;

use super::*;

const BGE_RERANKER: &str = "BAAI/bge-reranker-v2-m3";
const MS_MARCO: &str = "cross-encoder/ms-marco-MiniLM-L6-v2";
const GTE_MODERNBERT: &str = "Alibaba-NLP/gte-reranker-modernbert-base";
const QWEN3_RERANKER: &str = "mlx-community/Qwen3-Reranker-0.6B-4bit";
const QWEN3_VL_RERANKER: &str = "Qwen/Qwen3-VL-Reranker-2B";

/// The Beijing request the issue uses as the shared ordering gate.
const BEIJING_QUERY: &str = "What is the capital of China?";
const BEIJING_DOCUMENTS: [&str; 3] = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other.",
    "Berlin is in Germany.",
];

fn load(dir: &std::path::Path, batch_size: Option<usize>) -> LoadedReranker {
    load_reranker_with_options(
        dir,
        RerankLoadOptions {
            batch_size,
            max_length: None,
        },
    )
    .expect("reranker loads")
}

fn score(loaded: &LoadedReranker, query: &str, documents: &[&str]) -> Vec<f32> {
    let items: Vec<RerankItem> = documents.iter().map(|d| RerankItem::text(*d)).collect();
    loaded
        .reranker
        .score(&RerankItem::text(query), &items, None)
        .expect("scoring succeeds")
        .scores
}

/// Index of the highest score.
fn argmax(scores: &[f32]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Run the Beijing ordering gate against one loaded reranker.
fn assert_ranks_beijing_first(loaded: &LoadedReranker, label: &str) -> Vec<f32> {
    let scores = score(loaded, BEIJING_QUERY, &BEIJING_DOCUMENTS);
    assert_eq!(scores.len(), 3);
    assert!(
        scores
            .iter()
            .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
        "{label}: scores must be probabilities, got {scores:?}"
    );
    assert_eq!(
        argmax(&scores),
        0,
        "{label}: the Beijing document must rank first, got {scores:?}"
    );
    eprintln!("[gate] {label} beijing scores: {scores:?}");
    scores
}

#[test]
fn bge_reranker_v2_m3_matches_the_model_card_scores() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(BGE_RERANKER) else {
        eprintln!("skipping: {BGE_RERANKER} is not downloaded");
        return;
    };
    // Detection must route the cross-encoder to the reranker family now that
    // `-m` can serve it directly.
    assert_eq!(
        get_model_type(&dir).expect("detection succeeds"),
        crate::models::ModelType::SequenceClassifier
    );

    let loaded = load(&dir, None);
    assert_eq!(loaded.kind, RerankerKind::SequenceClassifier);
    assert_eq!(loaded.model_type, "xlm-roberta");
    // 8194 position rows minus the `pad_token_id + 1` offset.
    assert_eq!(loaded.reranker.max_length(), 8192);
    assert!(!loaded.reranker.supports_images());

    let panda = "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or \
                 simply panda, is a bear species endemic to China.";
    let scores = score(&loaded, "what is panda?", &["hi", panda]);
    eprintln!("[gate] bge-reranker-v2-m3 scores: {scores:?}");
    // The model card publishes the normalized (sigmoid) pair as
    // [0.00027803096387751553, 0.9948403768236574] for exactly this input.
    assert!(
        (scores[0] - 0.0003).abs() < 2e-2,
        "unrelated pair scored {}, expected ~0.0003",
        scores[0]
    );
    assert!(
        (scores[1] - 0.9948).abs() < 2e-2,
        "relevant pair scored {}, expected ~0.9948",
        scores[1]
    );

    assert_ranks_beijing_first(&loaded, "bge-reranker-v2-m3");
}

#[test]
fn ms_marco_minilm_cross_encoder_ranks_the_beijing_document_first() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(MS_MARCO) else {
        eprintln!("skipping: {MS_MARCO} is not downloaded");
        return;
    };
    assert_eq!(
        get_model_type(&dir).expect("detection succeeds"),
        crate::models::ModelType::SequenceClassifier
    );
    let loaded = load(&dir, None);
    assert_eq!(loaded.kind, RerankerKind::SequenceClassifier);
    assert_eq!(loaded.model_type, "bert");
    // `max_position_embeddings` is 512 on this checkpoint and the tokenizer
    // agrees, so the derived cap is 512 rather than the 8192 ceiling.
    assert_eq!(loaded.reranker.max_length(), 512);
    assert_ranks_beijing_first(&loaded, "ms-marco-MiniLM-L6-v2");
}

#[test]
fn gte_reranker_modernbert_ranks_the_beijing_document_first() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(GTE_MODERNBERT) else {
        eprintln!("skipping: {GTE_MODERNBERT} is not downloaded");
        return;
    };
    assert_eq!(
        get_model_type(&dir).expect("detection succeeds"),
        crate::models::ModelType::SequenceClassifier
    );
    let loaded = load(&dir, None);
    assert_eq!(loaded.kind, RerankerKind::SequenceClassifier);
    assert_eq!(loaded.model_type, "modernbert");
    assert_ranks_beijing_first(&loaded, "gte-reranker-modernbert-base");
}

#[test]
fn qwen3_generative_reranker_ranks_the_beijing_document_first() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(QWEN3_RERANKER) else {
        eprintln!("skipping: {QWEN3_RERANKER} is not downloaded");
        return;
    };
    // The checkpoint is an ordinary causal export, so `-m` must keep seeing a
    // chat model; only `--reranker-model` reaches the reranker path.
    assert_eq!(
        get_model_type(&dir).expect("detection succeeds"),
        crate::models::ModelType::Qwen3
    );
    let loaded = load(&dir, None);
    assert_eq!(loaded.kind, RerankerKind::GenerativeText);
    assert_eq!(loaded.reranker.max_length(), 8192);
    assert!(!loaded.reranker.supports_images());

    let scores = assert_ranks_beijing_first(&loaded, "Qwen3-Reranker-0.6B-4bit");
    // The issue's acceptance gate for this checkpoint.
    assert!(
        scores[0] > 0.9,
        "the Beijing document scored {}, expected > 0.9",
        scores[0]
    );
    assert!(
        scores[1] < 0.2 && scores[2] < 0.2,
        "the irrelevant documents scored {} and {}, expected < 0.2",
        scores[1],
        scores[2]
    );
}

#[test]
fn qwen3_vl_reranker_scores_text_and_image_documents() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(QWEN3_VL_RERANKER) else {
        eprintln!("skipping: {QWEN3_VL_RERANKER} is not downloaded");
        return;
    };
    // `modules.json` carries a LogitScore module, not a Pooling one, so
    // detection must keep this on the generative Qwen3-VL path.
    assert_eq!(
        get_model_type(&dir).expect("detection succeeds"),
        crate::models::ModelType::Qwen3VL
    );
    let loaded = load(&dir, None);
    assert_eq!(loaded.kind, RerankerKind::GenerativeVl);
    assert!(loaded.reranker.supports_images());
    assert_eq!(
        loaded.reranker.batch_size(),
        DEFAULT_RERANK_VL_BATCH_SIZE,
        "the multimodal default is the smaller batch"
    );

    // Text-only pairs go through the same prompt as the text reranker.
    assert_ranks_beijing_first(&loaded, "Qwen3-VL-Reranker-2B (text)");

    // Two synthetic image documents: a bar-chart-like figure and a flat
    // texture. The gate is that both produce finite probabilities, which is
    // what exercises the image merge and the per-row image forward.
    let images = vec![
        RerankItem::image(ImageInput {
            image: synthetic_chart(),
        }),
        RerankItem::image(ImageInput {
            image: synthetic_noise(),
        }),
    ];
    let scored = loaded
        .reranker
        .score(
            &RerankItem::text("a chart of quarterly revenue"),
            &images,
            None,
        )
        .expect("image scoring succeeds");
    eprintln!("[gate] Qwen3-VL-Reranker image scores: {:?}", scored.scores);
    assert_eq!(scored.scores.len(), 2);
    assert!(
        scored
            .scores
            .iter()
            .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
        "image scores must be probabilities, got {:?}",
        scored.scores
    );
    assert!(
        scored.prompt_tokens > 100,
        "an image row should carry visual tokens, got {}",
        scored.prompt_tokens
    );
}

/// A 224x224 figure with white background and dark vertical bars of
/// increasing height, so it reads as a bar chart without shipping a binary
/// fixture into the repository.
fn synthetic_chart() -> image::DynamicImage {
    let mut buffer = image::RgbImage::from_pixel(224, 224, image::Rgb([255, 255, 255]));
    for (index, x0) in (20..200).step_by(30).enumerate() {
        let height = 20 + index as u32 * 30;
        for x in x0..x0 + 20 {
            for y in (200 - height)..200 {
                buffer.put_pixel(x, y, image::Rgb([20, 40, 160]));
            }
        }
    }
    for x in 10..214 {
        buffer.put_pixel(x, 200, image::Rgb([0, 0, 0]));
    }
    image::DynamicImage::ImageRgb8(buffer)
}

/// A 224x224 deterministic texture with no chart structure.
fn synthetic_noise() -> image::DynamicImage {
    let mut buffer = image::RgbImage::new(224, 224);
    let mut state: u32 = 0x1356_ABCD;
    for pixel in buffer.pixels_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let value = (state >> 16) as u8;
        *pixel = image::Rgb([value, value.wrapping_add(64), value.wrapping_add(128)]);
    }
    image::DynamicImage::ImageRgb8(buffer)
}

/// Every reranker checkpoint on this host must reach the kind the issue's
/// detection table assigns it.
#[test]
fn detection_table_matches_the_local_checkpoints() {
    let expected: &[(&str, RerankerKind)] = &[
        (BGE_RERANKER, RerankerKind::SequenceClassifier),
        (MS_MARCO, RerankerKind::SequenceClassifier),
        (GTE_MODERNBERT, RerankerKind::SequenceClassifier),
        (QWEN3_RERANKER, RerankerKind::GenerativeText),
        (QWEN3_VL_RERANKER, RerankerKind::GenerativeVl),
    ];
    let mut checked = 0;
    for (repo, kind) in expected {
        let Some(dir): Option<PathBuf> = local_checkpoint(repo) else {
            continue;
        };
        checked += 1;
        let config = crate::embeddings::loader::read_embedding_config(&dir)
            .unwrap_or_else(|err| panic!("{repo}: config.json reads: {err}"));
        assert_eq!(
            detect_reranker_kind(&config).unwrap_or_else(|err| panic!("{repo}: {err}")),
            *kind,
            "{repo}"
        );
    }
    eprintln!("checked {checked} local reranker checkpoints");
}
