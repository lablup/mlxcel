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

//! Late-interaction (ColBERT MaxSim) reranking over a multi-vector embedder
//! (#1426).
//!
//! A multi-vector embedder (ColIdefics3, ColQwen2.5, LFM2.5-ColBERT) returns
//! one `[num_tokens, D]` matrix per input, and the score that ranks those
//! matrices is MaxSim, `sum_i max_j dot(q_i, d_j)`. [`MaxSimRerankProvider`]
//! puts that score behind [`RerankModelProvider`] so `POST /v1/rerank` serves
//! it with its existing request and response schema: the query and each
//! document are embedded through the already-running embedding worker and
//! scored on the route side with [`crate::embeddings::maxsim`], the same
//! function `mlxcel embed` prints its similarity matrix with.
//!
//! The provider owns no worker of its own. It shares the embedding worker, so
//! it is never registered as a separate side worker for shutdown.

use std::sync::Arc;

use crate::embeddings::{EmbdNormalize, EmbedOptions, EmbedReply, maxsim};
use crate::rerank::{RerankItem, RerankScores, RerankerKind};
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};
use crate::server::rerank_model::{RerankError, RerankModelProvider};

/// `/v1/rerank` provider that scores documents by MaxSim over a served
/// multi-vector embedder.
pub struct MaxSimRerankProvider {
    embedding: Arc<dyn EmbeddingModelProvider>,
    /// The server's `--embd-normalize`, so the rows scored here are the rows
    /// `/v1/embeddings` returns for the same input.
    normalize: Option<EmbdNormalize>,
}

impl MaxSimRerankProvider {
    /// Wrap a multi-vector embedder.
    #[must_use]
    pub fn new(
        embedding: Arc<dyn EmbeddingModelProvider>,
        normalize: Option<EmbdNormalize>,
    ) -> Self {
        Self {
            embedding,
            normalize,
        }
    }

    fn options(&self) -> EmbedOptions {
        EmbedOptions {
            normalize: self.normalize,
            ..EmbedOptions::default()
        }
    }

    /// Embed one item: its image when it carries one, otherwise its text.
    fn embed_item(&self, item: &RerankItem) -> Result<EmbedReply, RerankError> {
        let reply = match &item.image {
            Some(_) if !self.embedding.supports_images() => {
                return Err(RerankError::InvalidInput(
                    "the loaded late-interaction reranker does not accept images".to_string(),
                ));
            }
            Some(image) => self.embedding.embed_image(image.clone(), self.options()),
            None => self
                .embedding
                .embed_texts(vec![item.text_or_empty().to_string()], self.options()),
        };
        reply.map_err(embedding_error_to_rerank)
    }
}

/// Map an embedding-worker failure onto the rerank route's error envelope.
fn embedding_error_to_rerank(err: EmbeddingError) -> RerankError {
    match err {
        EmbeddingError::QueueFull => RerankError::QueueFull,
        EmbeddingError::Timeout => RerankError::Timeout,
        EmbeddingError::InvalidInput(message) => RerankError::InvalidInput(message),
        EmbeddingError::Internal(message) => RerankError::Internal(message),
    }
}

/// The rows of a one-input reply.
fn single_rows<'a>(reply: &'a EmbedReply, label: &str) -> Result<Vec<&'a [f32]>, RerankError> {
    match reply.vectors.as_slice() {
        [vector] => Ok(vector.rows().collect()),
        vectors => Err(RerankError::Internal(format!(
            "the embedder returned {} vectors for {label}, expected 1",
            vectors.len()
        ))),
    }
}

impl RerankModelProvider for MaxSimRerankProvider {
    fn rerank(
        &self,
        query: RerankItem,
        documents: Vec<RerankItem>,
        _instruction: Option<String>,
    ) -> Result<RerankScores, RerankError> {
        // The route rejects an `instruction` for this kind before dispatch,
        // because `accepts_instruction()` is false, so it is never seen here.
        let query_reply = self.embed_item(&query)?;
        let query_rows = single_rows(&query_reply, "the query")?;
        let mut prompt_tokens = query_reply.prompt_tokens;
        let mut scores = Vec::with_capacity(documents.len());
        for (index, document) in documents.iter().enumerate() {
            let reply = self.embed_item(document)?;
            prompt_tokens += reply.prompt_tokens;
            let rows = single_rows(&reply, &format!("documents[{index}]"))?;
            // A document with no rows scores 0.0 inside `maxsim`, matching
            // `mlxcel embed`'s similarity matrix.
            scores.push(maxsim(&query_rows, &rows));
        }
        Ok(RerankScores {
            scores,
            prompt_tokens,
        })
    }

    fn model_id(&self) -> &str {
        self.embedding.model_id()
    }

    fn created_at(&self) -> i64 {
        self.embedding.created_at()
    }

    fn kind(&self) -> RerankerKind {
        RerankerKind::LateInteraction
    }

    fn supports_images(&self) -> bool {
        self.embedding.supports_images()
    }

    fn max_length(&self) -> usize {
        self.embedding.max_length()
    }

    /// A no-op: the embedding worker this provider shares is joined through
    /// its own side-worker entry.
    fn shutdown_and_wait(&self, _timeout: std::time::Duration) -> anyhow::Result<()> {
        Ok(())
    }
}

/// What startup does with a served embedder on `/v1/rerank`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LateInteractionDecision {
    /// No embedder, or a single-vector one: MaxSim would degenerate into a
    /// dot product, so nothing is installed.
    NotApplicable,
    /// A reranker was configured as well; it keeps `/v1/rerank`.
    ShadowedByReranker,
    /// Install [`MaxSimRerankProvider`] on `/v1/rerank`.
    Install,
}

/// Decide whether a served embedder backs `/v1/rerank`.
#[must_use]
pub(crate) fn late_interaction_decision(
    embedding: Option<&dyn EmbeddingModelProvider>,
    reranker_configured: bool,
) -> LateInteractionDecision {
    match embedding {
        Some(embedding) if embedding.multi_vector() => {
            if reranker_configured {
                LateInteractionDecision::ShadowedByReranker
            } else {
                LateInteractionDecision::Install
            }
        }
        _ => LateInteractionDecision::NotApplicable,
    }
}

/// Startup wiring: keep a configured reranker, otherwise install the MaxSim
/// provider over a multi-vector embedder, logging either outcome.
///
/// `reranker_configured` is true when `--reranker-model` was given or `-m`
/// resolved a reranker, even if that load failed, so a broken reranker is
/// never silently replaced by a different scoring rule.
#[must_use]
pub(crate) fn resolve_late_interaction_reranker(
    rerank_model: Option<Arc<dyn RerankModelProvider>>,
    reranker_configured: bool,
    embedding: Option<&Arc<dyn EmbeddingModelProvider>>,
    normalize: Option<EmbdNormalize>,
) -> Option<Arc<dyn RerankModelProvider>> {
    let decision = late_interaction_decision(
        embedding.map(|provider| provider.as_ref()),
        reranker_configured || rerank_model.is_some(),
    );
    match (decision, embedding) {
        (LateInteractionDecision::Install, Some(embedding)) => {
            tracing::info!(
                model_id = %embedding.model_id(),
                kind = RerankerKind::LateInteraction.as_str(),
                supports_images = embedding.supports_images(),
                "Late-interaction (MaxSim) reranker ready on /v1/rerank over the multi-vector \
                 embedder"
            );
            Some(Arc::new(MaxSimRerankProvider::new(
                Arc::clone(embedding),
                normalize,
            )))
        }
        (LateInteractionDecision::ShadowedByReranker, Some(embedding)) => {
            tracing::info!(
                model_id = %embedding.model_id(),
                "The multi-vector embedder is not exposed on /v1/rerank: the configured \
                 reranker serves it"
            );
            rerank_model
        }
        _ => rerank_model,
    }
}

#[cfg(test)]
#[path = "rerank_maxsim_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "rerank_maxsim_checkpoint_tests.rs"]
mod checkpoint_tests;
