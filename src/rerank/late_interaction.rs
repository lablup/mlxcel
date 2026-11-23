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

//! Late-interaction (ColBERT MaxSim) scoring over a multi-vector embedder,
//! shared by `/v1/rerank` and `mlxcel rerank` (#1426, #3300).
//!
//! [`score_late_interaction`] is the whole scoring rule: embed the query and
//! every document through one [`MultiVectorEmbedder`], then score each
//! document with [`crate::embeddings::maxsim`]. The server's
//! `MaxSimRerankProvider` runs it over the embedding worker; the CLI runs it
//! over an [`EmbeddingEngine`] on the calling thread through
//! [`LateInteractionReranker`]. Both therefore embed the same items in the
//! same calls and score the same rows.

use std::fmt;

use anyhow::Result;

use crate::embeddings::{
    EmbedOptions, EmbedReply, EmbeddingEngine, EmbeddingEngineError, EmbeddingVector, ImageInput,
    maxsim,
};

use super::{RerankItem, RerankScores, Reranker, RerankerKind};

/// What [`score_late_interaction`] embeds through.
pub trait MultiVectorEmbedder {
    /// The backend's own failure type, mapped by the caller.
    type Error;

    /// Whether image items can be embedded.
    fn supports_images(&self) -> bool;

    /// Embed every text in one call (special tokens added, per-family
    /// formatting applied).
    fn embed_texts(
        &self,
        texts: Vec<String>,
        options: &EmbedOptions,
    ) -> Result<EmbedReply, Self::Error>;

    /// Embed one image.
    fn embed_image(
        &self,
        image: ImageInput,
        options: &EmbedOptions,
    ) -> Result<EmbedReply, Self::Error>;
}

impl MultiVectorEmbedder for EmbeddingEngine {
    type Error = EmbeddingEngineError;

    fn supports_images(&self) -> bool {
        EmbeddingEngine::supports_images(self)
    }

    fn embed_texts(
        &self,
        texts: Vec<String>,
        options: &EmbedOptions,
    ) -> Result<EmbedReply, Self::Error> {
        EmbeddingEngine::embed_texts(self, &texts, options)
    }

    fn embed_image(
        &self,
        image: ImageInput,
        options: &EmbedOptions,
    ) -> Result<EmbedReply, Self::Error> {
        EmbeddingEngine::embed_image(self, image, options)
    }
}

/// Failure of [`score_late_interaction`].
#[derive(Debug)]
pub enum LateInteractionError<E> {
    /// The embedder failed.
    Embed(E),
    /// An item carries an image and the embedder is text-only.
    ImagesUnsupported,
    /// The embedder replied with an unexpected number of vectors.
    Internal(String),
}

impl<E: fmt::Display> fmt::Display for LateInteractionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embed(err) => err.fmt(f),
            Self::ImagesUnsupported => f.write_str(IMAGES_UNSUPPORTED),
            Self::Internal(message) => f.write_str(message),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for LateInteractionError<E> {}

/// Message for an image item on a text-only embedder.
pub const IMAGES_UNSUPPORTED: &str = "the loaded late-interaction reranker does not accept images";

/// Embed the query and every document, then score each document by MaxSim
/// against the query.
///
/// Every text item goes to the embedder in ONE `embed_texts` call, the way
/// `/v1/embeddings` dispatches a text batch: on the server that is one queue
/// slot, one reply timeout and length-sorted micro-batching for the whole
/// request. One call per document would let a single request holding
/// thousands of short documents occupy the worker for thousands of reply
/// timeouts back to back, and hit `QueueFull` halfway through after spending
/// the earlier forwards. Image items stay one call each, as the embedder
/// takes one image per call; the server caps their count with the route's
/// per-request image limit.
///
/// A document with no rows scores 0.0 inside `maxsim`, matching
/// `mlxcel embed`'s similarity matrix.
pub fn score_late_interaction<B>(
    embedder: &B,
    query: &RerankItem,
    documents: &[RerankItem],
    options: &EmbedOptions,
) -> Result<RerankScores, LateInteractionError<B::Error>>
where
    B: MultiVectorEmbedder + ?Sized,
{
    let items: Vec<&RerankItem> = std::iter::once(query).chain(documents.iter()).collect();
    if items.iter().any(|item| item.image.is_some()) && !embedder.supports_images() {
        return Err(LateInteractionError::ImagesUnsupported);
    }
    let mut slots: Vec<Option<EmbeddingVector>> = (0..items.len()).map(|_| None).collect();
    let mut prompt_tokens = 0usize;

    let text_indices: Vec<usize> = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| item.image.is_none().then_some(index))
        .collect();
    if !text_indices.is_empty() {
        let texts = text_indices
            .iter()
            .map(|&index| items[index].text_or_empty().to_string())
            .collect();
        let reply = embedder
            .embed_texts(texts, options)
            .map_err(LateInteractionError::Embed)?;
        if reply.vectors.len() != text_indices.len() {
            return Err(LateInteractionError::Internal(format!(
                "the embedder returned {} vectors for {} texts",
                reply.vectors.len(),
                text_indices.len()
            )));
        }
        prompt_tokens += reply.prompt_tokens;
        for (&index, vector) in text_indices.iter().zip(reply.vectors) {
            slots[index] = Some(vector);
        }
    }

    for (index, item) in items.iter().enumerate() {
        let Some(image) = &item.image else {
            continue;
        };
        let reply = embedder
            .embed_image(image.clone(), options)
            .map_err(LateInteractionError::Embed)?;
        prompt_tokens += reply.prompt_tokens;
        slots[index] = Some(single_vector(reply, index)?);
    }

    let vectors = slots
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| LateInteractionError::Internal("an item produced no vector".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let Some((query_vector, document_vectors)) = vectors.split_first() else {
        return Err(LateInteractionError::Internal(
            "the query produced no vector".to_string(),
        ));
    };
    let query_rows: Vec<&[f32]> = query_vector.rows().collect();
    let scores = document_vectors
        .iter()
        .map(|vector| maxsim(&query_rows, &vector.rows().collect::<Vec<_>>()))
        .collect();
    Ok(RerankScores {
        scores,
        prompt_tokens,
    })
}

/// The one vector of a one-image reply; `index` 0 is the query.
fn single_vector<E>(
    reply: EmbedReply,
    index: usize,
) -> Result<EmbeddingVector, LateInteractionError<E>> {
    let count = reply.vectors.len();
    match <[EmbeddingVector; 1]>::try_from(reply.vectors) {
        Ok([vector]) => Ok(vector),
        Err(_) => {
            let label = match index {
                0 => "the query".to_string(),
                n => format!("documents[{}]", n - 1),
            };
            Err(LateInteractionError::Internal(format!(
                "the embedder returned {count} vectors for {label}, expected 1"
            )))
        }
    }
}

/// [`Reranker`] over a multi-vector embedder, for `mlxcel rerank`.
///
/// Holds the engine on the calling thread; the server does not use it,
/// because its embedder already runs on the embedding worker.
pub struct LateInteractionReranker {
    engine: EmbeddingEngine,
    options: EmbedOptions,
}

impl LateInteractionReranker {
    /// Wrap a multi-vector engine. `options` is what every embed call gets;
    /// the server's equivalent is `--embd-normalize` alone.
    #[must_use]
    pub fn new(engine: EmbeddingEngine, options: EmbedOptions) -> Self {
        Self { engine, options }
    }
}

impl Reranker for LateInteractionReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::LateInteraction
    }

    /// `instruction` is ignored: [`RerankerKind::accepts_instruction`] is
    /// false for this kind, so callers refuse one before scoring.
    fn score(
        &self,
        query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
    ) -> Result<RerankScores> {
        score_late_interaction(&self.engine, query, documents, &self.options)
            .map_err(|err| anyhow::anyhow!("{err}"))
    }

    fn supports_images(&self) -> bool {
        self.engine.supports_images()
    }

    fn max_length(&self) -> usize {
        self.engine.max_length()
    }

    fn batch_size(&self) -> usize {
        self.engine.batch_size()
    }
}

#[cfg(test)]
#[path = "late_interaction_tests.rs"]
mod late_interaction_tests;
