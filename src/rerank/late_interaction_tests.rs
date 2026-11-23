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

//! Tests for the shared late-interaction scorer and the `mlxcel rerank`
//! loader fallback (#3300).

use std::cell::RefCell;
use std::path::Path;

use serde_json::json;

use super::{
    IMAGES_UNSUPPORTED, LateInteractionError, LateInteractionReranker, MultiVectorEmbedder,
    score_late_interaction,
};
use crate::embeddings::stub::stub_loaded_model;
use crate::embeddings::{
    DEFAULT_EMBEDDING_BATCH_SIZE, EmbedOptions, EmbedReply, EmbeddingEngine, EmbeddingVector,
    ImageInput, maxsim,
};
use crate::rerank::loader::late_interaction_from_embedder;
use crate::rerank::{
    RerankItem, RerankLoadOptions, Reranker, RerankerKind, load_reranker_or_late_interaction,
};

/// Records every call and answers from fixed `[n, 2]` rows.
struct Recorder {
    images: bool,
    text_calls: RefCell<Vec<Vec<String>>>,
    image_calls: RefCell<usize>,
    /// Drop one vector from the text reply.
    short_reply: bool,
}

impl Recorder {
    fn new(images: bool) -> Self {
        Self {
            images,
            text_calls: RefCell::new(Vec::new()),
            image_calls: RefCell::new(0),
            short_reply: false,
        }
    }
}

fn rows(rows: &[[f32; 2]]) -> EmbeddingVector {
    EmbeddingVector {
        values: rows.iter().flatten().copied().collect(),
        shape: vec![rows.len(), 2],
    }
}

fn text_rows(text: &str) -> EmbeddingVector {
    match text {
        "query" => rows(&[[1.0, 0.0], [0.0, 1.0]]),
        "relevant" => rows(&[[0.6, 0.8], [1.0, 0.0]]),
        "unrelated" => rows(&[[0.0, 0.3]]),
        _ => rows(&[]),
    }
}

impl MultiVectorEmbedder for Recorder {
    type Error = String;

    fn supports_images(&self) -> bool {
        self.images
    }

    fn embed_texts(
        &self,
        texts: Vec<String>,
        _options: &EmbedOptions,
    ) -> Result<EmbedReply, String> {
        self.text_calls.borrow_mut().push(texts.clone());
        let mut vectors: Vec<EmbeddingVector> = texts.iter().map(|t| text_rows(t)).collect();
        if self.short_reply {
            vectors.pop();
        }
        Ok(EmbedReply {
            prompt_tokens: texts.len() * 2,
            vectors,
        })
    }

    fn embed_image(
        &self,
        _image: ImageInput,
        _options: &EmbedOptions,
    ) -> Result<EmbedReply, String> {
        *self.image_calls.borrow_mut() += 1;
        Ok(EmbedReply {
            vectors: vec![rows(&[[0.5, 0.5], [0.9, 0.1]])],
            prompt_tokens: 5,
        })
    }
}

fn tiny_image() -> RerankItem {
    RerankItem::image(ImageInput {
        image: image::DynamicImage::new_rgb8(1, 1),
    })
}

#[test]
fn texts_go_in_one_call_and_images_one_each() {
    let embedder = Recorder::new(true);
    let documents = [
        RerankItem::text("relevant"),
        tiny_image(),
        RerankItem::text("unrelated"),
        tiny_image(),
    ];
    let scored = score_late_interaction(
        &embedder,
        &RerankItem::text("query"),
        &documents,
        &EmbedOptions::default(),
    )
    .expect("scores");

    assert_eq!(
        *embedder.text_calls.borrow(),
        vec![vec![
            "query".to_string(),
            "relevant".to_string(),
            "unrelated".to_string()
        ]]
    );
    assert_eq!(*embedder.image_calls.borrow(), 2);
    assert_eq!(scored.prompt_tokens, 3 * 2 + 2 * 5);

    let query = [[1.0f32, 0.0], [0.0, 1.0]];
    let expected = [
        maxsim(&query, &[[0.6f32, 0.8], [1.0, 0.0]]),
        maxsim(&query, &[[0.5f32, 0.5], [0.9, 0.1]]),
        maxsim(&query, &[[0.0f32, 0.3]]),
        maxsim(&query, &[[0.5f32, 0.5], [0.9, 0.1]]),
    ];
    assert_eq!(scored.scores, expected);
}

#[test]
fn an_image_on_a_text_only_embedder_is_refused_before_any_call() {
    let embedder = Recorder::new(false);
    let err = score_late_interaction(
        &embedder,
        &RerankItem::text("query"),
        &[tiny_image()],
        &EmbedOptions::default(),
    )
    .expect_err("text-only");
    assert!(matches!(err, LateInteractionError::ImagesUnsupported));
    assert_eq!(err.to_string(), IMAGES_UNSUPPORTED);
    assert!(embedder.text_calls.borrow().is_empty());
}

#[test]
fn a_short_text_reply_is_an_internal_error() {
    let mut embedder = Recorder::new(false);
    embedder.short_reply = true;
    let err = score_late_interaction(
        &embedder,
        &RerankItem::text("query"),
        &[RerankItem::text("relevant")],
        &EmbedOptions::default(),
    )
    .expect_err("short reply");
    assert_eq!(
        err.to_string(),
        "the embedder returned 1 vectors for 2 texts"
    );
}

#[test]
fn a_document_without_rows_scores_zero() {
    let embedder = Recorder::new(false);
    let scored = score_late_interaction(
        &embedder,
        &RerankItem::text("query"),
        &[RerankItem::text("nothing")],
        &EmbedOptions::default(),
    )
    .expect("scores");
    assert_eq!(scored.scores, vec![0.0]);
}

#[test]
fn the_engine_reranker_scores_the_engine_rows() {
    let reranker = LateInteractionReranker::new(
        EmbeddingEngine::new(stub_loaded_model(true), 4),
        EmbedOptions::default(),
    );
    assert_eq!(reranker.kind(), RerankerKind::LateInteraction);
    assert_eq!(reranker.batch_size(), 4);
    assert!(!reranker.supports_images());

    let scored = reranker
        .score(
            &RerankItem::text("hello world"),
            &[RerankItem::text("world"), RerankItem::text("hello")],
            None,
        )
        .expect("scores");

    let engine = EmbeddingEngine::new(stub_loaded_model(true), 4);
    let reply = engine
        .embed_texts(
            &["hello world", "world", "hello"].map(str::to_string),
            &EmbedOptions::default(),
        )
        .expect("embeds");
    let rows = |i: usize| -> Vec<&[f32]> { reply.vectors[i].rows().collect() };
    assert_eq!(
        scored.scores,
        vec![maxsim(&rows(0), &rows(1)), maxsim(&rows(0), &rows(2))]
    );
    assert_eq!(scored.prompt_tokens, reply.prompt_tokens);
}

#[test]
fn the_engine_reranker_refuses_images_on_a_text_only_engine() {
    let reranker = LateInteractionReranker::new(
        EmbeddingEngine::new(stub_loaded_model(true), 4),
        EmbedOptions::default(),
    );
    let err = reranker
        .score(&RerankItem::text("hello"), &[tiny_image()], None)
        .expect_err("text-only");
    assert_eq!(err.to_string(), IMAGES_UNSUPPORTED);
}

fn detection_error() -> anyhow::Error {
    anyhow::anyhow!("Unsupported reranker model type `stub`")
        .context("/stub is not a reranker checkpoint")
}

#[test]
fn a_multi_vector_embedder_becomes_a_late_interaction_reranker() {
    let config = json!({"model_type": "lfm2"});
    let loaded = late_interaction_from_embedder(
        Path::new("/stub"),
        &config,
        RerankLoadOptions::default(),
        detection_error(),
        |_, _| Ok(stub_loaded_model(true)),
    )
    .expect("multi-vector loads");
    assert_eq!(loaded.kind, RerankerKind::LateInteraction);
    assert_eq!(loaded.model_type, "lfm2");
    assert_eq!(loaded.reranker.batch_size(), DEFAULT_EMBEDDING_BATCH_SIZE);

    let sized = late_interaction_from_embedder(
        Path::new("/stub"),
        &config,
        RerankLoadOptions {
            batch_size: Some(3),
            max_length: Some(7),
        },
        detection_error(),
        |_, options| {
            assert_eq!(options.max_length, Some(7));
            Ok(stub_loaded_model(true))
        },
    )
    .expect("multi-vector loads");
    assert_eq!(sized.reranker.batch_size(), 3);
}

#[test]
fn a_single_vector_embedder_keeps_the_detection_error() {
    let err = late_interaction_from_embedder(
        Path::new("/stub"),
        &json!({"model_type": "bert"}),
        RerankLoadOptions::default(),
        detection_error(),
        |_, _| Ok(stub_loaded_model(false)),
    )
    .err()
    .expect("single-vector is refused");
    assert_eq!(
        format!("{err:#}"),
        "/stub is not a reranker checkpoint: Unsupported reranker model type `stub`"
    );
}

#[test]
fn a_chat_checkpoint_keeps_the_detection_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "vocab_size": 32
        })
        .to_string(),
    )
    .expect("config writes");
    let err = load_reranker_or_late_interaction(dir.path(), RerankLoadOptions::default())
        .err()
        .expect("a chat checkpoint is not a reranker");
    let message = format!("{err:#}");
    assert!(
        message.contains("is not a reranker checkpoint")
            && message.contains("Unsupported reranker model type `llama`"),
        "{message}"
    );
}
