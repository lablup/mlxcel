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

//! OpenAI-compatible `POST /v1/embeddings` request and response types.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::embeddings::EmbeddingVector;

/// `POST /v1/embeddings` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsRequest {
    /// Optional; must equal the served embedding model id when given.
    #[serde(default)]
    pub model: Option<String>,
    /// The inputs to embed.
    pub input: EmbeddingInput,
    /// `float` (default) or `base64`.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Keep only the first `dimensions` components of each vector.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Forwarded to the family's text formatting (instruction prefix).
    #[serde(default)]
    pub instruction: Option<String>,
    /// Accepted for OpenAI compatibility and ignored.
    #[serde(default)]
    pub user: Option<String>,
}

/// The `input` field: a string, a list of strings, a token-id array, a list
/// of token-id arrays, or a list of typed parts.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Tokens(Vec<u32>),
    Texts(Vec<String>),
    TokenLists(Vec<Vec<u32>>),
    Parts(Vec<EmbeddingInputPart>),
}

/// One typed part of a multimodal `input` list.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingInputPart {
    Text { text: String },
    ImageUrl { image_url: EmbeddingImageUrl },
}

/// `{"url": "..."}` (data URI, `file://`, `http(s)://`, or a local path).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct EmbeddingImageUrl {
    pub url: String,
}

/// One normalized input item in request order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedItem {
    Text(String),
    Tokens(Vec<u32>),
    ImageUrl(String),
}

impl EmbeddingInput {
    /// Flatten the accepted shapes into ordered items.
    pub fn into_items(self) -> Vec<EmbedItem> {
        match self {
            EmbeddingInput::Text(text) => vec![EmbedItem::Text(text)],
            EmbeddingInput::Tokens(ids) => vec![EmbedItem::Tokens(ids)],
            EmbeddingInput::Texts(texts) => texts.into_iter().map(EmbedItem::Text).collect(),
            EmbeddingInput::TokenLists(rows) => rows.into_iter().map(EmbedItem::Tokens).collect(),
            EmbeddingInput::Parts(parts) => parts
                .into_iter()
                .map(|part| match part {
                    EmbeddingInputPart::Text { text } => EmbedItem::Text(text),
                    EmbeddingInputPart::ImageUrl { image_url } => {
                        EmbedItem::ImageUrl(image_url.url)
                    }
                })
                .collect(),
        }
    }
}

/// `encoding_format` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingEncoding {
    Float,
    Base64,
}

impl EmbeddingEncoding {
    /// Parse the request field; `None` and `""` mean `float`.
    pub fn parse(value: Option<&str>) -> Option<Self> {
        match value.map(str::trim).unwrap_or("") {
            "" | "float" => Some(EmbeddingEncoding::Float),
            "base64" => Some(EmbeddingEncoding::Base64),
            _ => None,
        }
    }
}

/// One entry of the `data` list.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    /// `[f32]` (float), `[[f32]]` (float, multi-vector), or a base64 string.
    pub embedding: Value,
    /// `[num_real_tokens, D]`, present only for base64 multi-vector output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
}

/// Token accounting of one request.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// `POST /v1/embeddings` response body.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Little-endian f32 bytes, standard base64 with padding.
pub fn base64_f32(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Decode the base64 form back into f32 values (tests and clients).
pub fn decode_base64_f32(encoded: &str) -> Option<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

impl EmbeddingData {
    /// Encode one engine vector at `index` in the requested format.
    pub fn from_vector(
        index: usize,
        vector: &EmbeddingVector,
        encoding: EmbeddingEncoding,
    ) -> Self {
        let multi = vector.is_multi_vector();
        let (embedding, shape) = match encoding {
            EmbeddingEncoding::Float if multi => {
                let rows: Vec<Value> = vector.rows().map(|row| Value::from(row.to_vec())).collect();
                (Value::Array(rows), None)
            }
            EmbeddingEncoding::Float => (Value::from(vector.values.clone()), None),
            EmbeddingEncoding::Base64 => (
                Value::String(base64_f32(&vector.values)),
                multi.then(|| vector.shape.clone()),
            ),
        };
        Self {
            object: "embedding".to_string(),
            index,
            embedding,
            shape,
        }
    }
}
