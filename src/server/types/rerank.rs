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
    pub documents: Vec<RerankInput>,
    /// Keep only the `top_n` highest-scoring results.
    #[serde(default)]
    pub top_n: Option<usize>,
    /// Echo each scored item back in its result entry.
    #[serde(default)]
    pub return_documents: bool,
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

/// One scored document.
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

/// Token accounting of one request.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RerankUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// `POST /v1/rerank` response body.
#[derive(Debug, Clone, Serialize)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
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
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    });
    if let Some(n) = top_n {
        results.truncate(n);
    }
    results
}
