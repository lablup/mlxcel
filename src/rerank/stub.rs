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

//! Test-only reranker: word-overlap scoring, no MLX and no checkpoint.
//!
//! The stub lets the worker and the `/v1/rerank` route run end to end. A
//! document's score is the fraction of the query's whitespace-separated words
//! it repeats, which makes the ordering assertions in the route tests
//! meaningful without loading weights. Its `prompt_tokens` is the word count
//! of every scored pair, so the `usage` field is exercised too.

use anyhow::{Result, bail};

use super::loader::LoadedReranker;
use super::{RerankItem, RerankScores, Reranker, RerankerKind};

/// Token cap the stub reports.
pub(crate) const STUB_MAX_LENGTH: usize = 64;

/// Pairs per forward pass the stub reports.
pub(crate) const STUB_BATCH_SIZE: usize = 4;

pub(crate) struct StubReranker {
    kind: RerankerKind,
    supports_images: bool,
}

impl StubReranker {
    pub(crate) fn new(kind: RerankerKind, supports_images: bool) -> Self {
        Self {
            kind,
            supports_images,
        }
    }
}

/// Fraction of `query`'s words that appear in `document`.
fn overlap(query: &str, document: &str) -> f32 {
    let query_words: Vec<&str> = query.split_whitespace().collect();
    if query_words.is_empty() {
        return 0.0;
    }
    let hits = query_words
        .iter()
        .filter(|word| {
            document
                .split_whitespace()
                .any(|other| other.eq_ignore_ascii_case(word))
        })
        .count();
    hits as f32 / query_words.len() as f32
}

impl Reranker for StubReranker {
    fn kind(&self) -> RerankerKind {
        self.kind
    }

    fn score(
        &self,
        query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
    ) -> Result<RerankScores> {
        if !self.supports_images
            && (query.has_image() || documents.iter().any(RerankItem::has_image))
        {
            bail!("stub reranker does not accept images");
        }
        let query_text = query.text_or_empty();
        let mut prompt_tokens = 0usize;
        let scores = documents
            .iter()
            .map(|document| {
                let text = document.text_or_empty();
                prompt_tokens +=
                    query_text.split_whitespace().count() + text.split_whitespace().count();
                // An image-only document has no words to overlap; give it a
                // fixed mid score so image requests still produce a ranking.
                if text.is_empty() && document.has_image() {
                    0.5
                } else {
                    overlap(query_text, text)
                }
            })
            .collect();
        Ok(RerankScores {
            scores,
            prompt_tokens,
        })
    }

    fn supports_images(&self) -> bool {
        self.supports_images
    }

    fn max_length(&self) -> usize {
        STUB_MAX_LENGTH
    }

    fn batch_size(&self) -> usize {
        STUB_BATCH_SIZE
    }
}

/// A fully assembled stub reranker for the worker and route tests.
pub(crate) fn stub_loaded_reranker(kind: RerankerKind, supports_images: bool) -> LoadedReranker {
    LoadedReranker {
        reranker: Box::new(StubReranker::new(kind, supports_images)),
        kind,
        model_type: "stub".to_string(),
    }
}
