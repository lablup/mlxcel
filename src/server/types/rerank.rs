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

//! `POST /v1/rerank` request and response types.
//!
//! The shape is the one Cohere and Jina both publish: a `query`, a
//! `documents` list, an optional `top_n` cut and a `return_documents` echo,
//! answered with `results` sorted by `relevance_score`. Items accept the plain
//! string form of both APIs and an object form that adds an image, which is
//! what the Qwen3-VL reranker scores.

use serde::{Deserialize, Serialize};

/// `POST /v1/rerank` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct RerankRequest {
    /// Optional; must equal the served reranker id when given.
    #[serde(default)]
    pub model: Option<String>,
    /// The query every document is scored against.
    pub query: RerankInput,
    /// The documents to score; at least one.
    ///
    /// b10621 accepts two spellings for this list and answers a **different
    /// response shape** for each: `documents` is the Jina/Cohere form and
    /// `texts` is the Text Embeddings Inference form. `texts` is a serde alias
    /// here, and [`RerankRequest::is_tei_form`] recovers which one arrived so
    /// the response can follow it (#1452).
    #[serde(alias = "texts")]
    pub documents: Vec<RerankInput>,
    /// Keep only the `top_n` highest-scoring results.
    #[serde(default)]
    pub top_n: Option<usize>,
    /// Echo each scored item back in its result entry (Cohere spelling).
    #[serde(default)]
    pub return_documents: bool,
    /// Echo each scored item back (b10621 / TEI spelling). Emits `text` rather
    /// than `document`.
    #[serde(default)]
    pub return_text: bool,
    /// Task description for the generative rerankers. Rejected for a
    /// sequence-classifier reranker, which has nowhere to put it.
    #[serde(default)]
    pub instruction: Option<String>,
    /// Accepted for Cohere compatibility and ignored.
    #[serde(default)]
    pub user: Option<String>,
}

/// One query or document: a bare string, or an object that may carry an image.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum RerankInput {
    /// `"a passage of text"`.
    Text(String),
    /// `{"text": ..., "image": ...}` or `{"text": ..., "image_url": ...}`.
    Object(RerankObject),
}

/// The object form of a rerank item.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct RerankObject {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Jina spells the image field `image`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<RerankImage>,
    /// OpenAI-style clients spell it `image_url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<RerankImage>,
}

/// An image reference: a bare URL string, or `{"url": "..."}`.
///
/// Both spellings appear in the wild for the same field, and accepting the
/// object form keeps a client that already builds OpenAI content parts from
/// having to special-case this endpoint.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum RerankImage {
    Url(String),
    Object { url: String },
}

impl RerankImage {
    /// The URL, whichever spelling carried it.
    #[must_use]
    pub fn url(&self) -> &str {
        match self {
            Self::Url(url) => url,
            Self::Object { url } => url,
        }
    }
}

impl RerankInput {
    /// Text content, or `None` for an image-only item.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Object(object) => object.text.as_deref(),
        }
    }

    /// Image URL when the item carries one. `image` wins over `image_url`
    /// when a client sends both, which is the field the Jina schema defines.
    #[must_use]
    pub fn image_url(&self) -> Option<&str> {
        match self {
            Self::Text(_) => None,
            Self::Object(object) => object
                .image
                .as_ref()
                .or(object.image_url.as_ref())
                .map(RerankImage::url),
        }
    }

    /// Whether the item carries neither usable text nor an image.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.image_url().is_none() && self.text().unwrap_or("").trim().is_empty()
    }
}

/// One scored document, Jina/Cohere shape.
#[derive(Debug, Clone, Serialize)]
pub struct RerankResult {
    /// Position of this document in the request's `documents` list.
    pub index: usize,
    /// Relevance in `[0, 1]`; higher is more relevant.
    pub relevance_score: f32,
    /// The request item, echoed only when `return_documents` was set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankInput>,
}

/// One scored document, b10621's TEI shape.
///
/// A different envelope, not a different computation: the score is the same
/// number `relevance_score` carries, and the echo is spelled `text` rather than
/// `document`. Which one a request gets is decided by the spelling it used for
/// the document list (#1452).
#[derive(Debug, Clone, Serialize)]
pub struct RerankTeiResult {
    pub index: usize,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<RerankInput>,
}

/// Token accounting of one request.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RerankUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// `POST /v1/rerank` response body, Jina/Cohere shape.
///
/// `object` is emitted because b10621 emits it; it was the recorded divergence
/// on the four rerank routes before #1452.
#[derive(Debug, Clone, Serialize)]
pub struct RerankResponse {
    pub model: String,
    pub object: &'static str,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
}

impl RerankResponse {
    /// The `object` value b10621 emits on this envelope.
    pub const OBJECT: &'static str = "list";
}

impl RerankRequest {
    /// Whether this request used b10621's TEI spelling for the document list.
    ///
    /// serde cannot report which alias matched, so the raw body is consulted:
    /// `texts` present and `documents` absent is the TEI form. A body carrying
    /// both is the Jina form, which is also what serde deserialized.
    #[must_use]
    pub fn is_tei_form(body: &serde_json::Value) -> bool {
        let Some(object) = body.as_object() else {
            return false;
        };
        object.contains_key("texts") && !object.contains_key("documents")
    }

    /// Whether the scored items should be echoed, under either spelling.
    #[must_use]
    pub fn echoes_items(&self) -> bool {
        self.return_documents || self.return_text
    }
}

/// Turn Jina-shaped results into b10621's TEI array.
///
/// Ordering and truncation have already been applied, so this is a pure
/// renaming of the two keys.
#[must_use]
pub fn to_tei_results(results: Vec<RerankResult>) -> Vec<RerankTeiResult> {
    results
        .into_iter()
        .map(|r| RerankTeiResult {
            index: r.index,
            score: r.relevance_score,
            text: r.document,
        })
        .collect()
}

/// Sort scored documents the way the endpoint promises: descending by score,
/// ties broken by ascending index, then cut to `top_n`.
///
/// The tie rule matters for reproducibility: a stable sort alone would leave
/// the order of equal scores dependent on the order the worker happened to
/// finish them in, and equal scores are common once a batch saturates the
/// sigmoid.
pub fn sort_and_truncate(
    mut results: Vec<RerankResult>,
    top_n: Option<usize>,
) -> Vec<RerankResult> {
    results.sort_by(|a, b| {
        b.relevance_score
            .total_cmp(&a.relevance_score)
            .then(a.index.cmp(&b.index))
    });
    if let Some(n) = top_n {
        results.truncate(n);
    }
    results
}
