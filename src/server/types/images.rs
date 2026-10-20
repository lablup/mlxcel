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

//! OpenAI-compatible `POST /v1/images/generations` and `POST /v1/images/edits`
//! request and response types.
//!
//! Field names and defaults follow the upstream schemas in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/server/schemas.py>.
//! Fields the server deliberately does not honor (`response_format` other than
//! `b64_json`, `output_path`, `output_dir`) are captured here so the route can
//! reject them with a precise message instead of ignoring them.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// `POST /v1/images/generations` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationsRequest {
    pub prompt: String,
    /// Optional; must equal the served image model id when given and non-empty.
    #[serde(default)]
    pub model: Option<String>,
    /// Number of images, `1..=10`; default 1.
    #[serde(default)]
    pub n: Option<u32>,
    /// `WIDTHxHEIGHT`. `width` and `height`, when both given, override it.
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    /// Base seed; output `i` uses `seed + i`.
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub guidance: Option<f32>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Only `b64_json` is served.
    #[serde(default)]
    pub response_format: Option<String>,
    /// Only `png` is produced.
    #[serde(default)]
    pub output_format: Option<String>,
    /// Rejected: the server never writes files on behalf of a client.
    #[serde(default)]
    pub output_path: Option<String>,
    /// Rejected: the server never writes files on behalf of a client.
    #[serde(default)]
    pub output_dir: Option<String>,
    /// Accepted for OpenAI compatibility and ignored.
    #[serde(default)]
    pub user: Option<String>,
    /// Every other field, passed to the model as `extra`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `image` field of an edit request: one string or a list.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ImageEditInput {
    One(String),
    Many(Vec<String>),
}

impl ImageEditInput {
    /// The references in request order.
    #[must_use]
    pub fn into_vec(self) -> Vec<String> {
        match self {
            ImageEditInput::One(s) => vec![s],
            ImageEditInput::Many(v) => v,
        }
    }
}

/// `POST /v1/images/edits` request body (JSON; multipart is not served).
#[derive(Debug, Clone, Deserialize)]
pub struct ImageEditsRequest {
    pub prompt: String,
    /// One data URL or URL, or a list of them.
    pub image: ImageEditInput,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub n: Option<u32>,
    /// `WIDTHxHEIGHT`; defaults to the first reference image's size.
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub guidance: Option<f32>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub output_format: Option<String>,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default)]
    pub output_dir: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One generated image in a response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ImageResponseData {
    pub b64_json: String,
    pub mime_type: &'static str,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub revised_prompt: Option<String>,
}

/// Response body of both image routes.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ImagesResponse {
    pub created: i64,
    pub data: Vec<ImageResponseData>,
    pub output_format: &'static str,
    /// `WIDTHxHEIGHT` of the request.
    pub size: String,
}
