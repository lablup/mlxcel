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

//! `/v1/embeddings` with a refused remote `image_url` (issue #2644): the 400
//! must not say which internal names exist or what they resolve to.

use std::sync::Arc;

use axum::http::StatusCode;
use serde_json::json;

use super::{app_with, post};
use crate::embeddings::{EmbedOptions, EmbedReply, EmbeddingVector, ImageInput};
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};
use crate::server::media_net::{ALLOW_PRIVATE_MEDIA_URLS_ENV, deny_private_media_urls_in_tests};

const GENERIC: &str = "media URL host is not allowed or could not be resolved";

/// An image-capable provider; the refusal happens before it is ever called.
struct ImageEmbeddingProvider;

impl ImageEmbeddingProvider {
    fn reply() -> EmbedReply {
        EmbedReply {
            vectors: vec![EmbeddingVector {
                values: vec![1.0],
                shape: vec![1],
            }],
            prompt_tokens: 1,
        }
    }
}

impl EmbeddingModelProvider for ImageEmbeddingProvider {
    fn embed_texts(&self, _: Vec<String>, _: EmbedOptions) -> Result<EmbedReply, EmbeddingError> {
        Ok(Self::reply())
    }

    fn embed_tokens(
        &self,
        _: Vec<Vec<u32>>,
        _: EmbedOptions,
    ) -> Result<EmbedReply, EmbeddingError> {
        Ok(Self::reply())
    }

    fn embed_image(&self, _: ImageInput, _: EmbedOptions) -> Result<EmbedReply, EmbeddingError> {
        Ok(Self::reply())
    }

    fn model_id(&self) -> &str {
        "image-embedding"
    }

    fn created_at(&self) -> i64 {
        0
    }

    fn dim(&self) -> usize {
        1
    }

    fn multi_vector(&self) -> bool {
        false
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn vocab_size(&self) -> usize {
        128
    }

    fn max_length(&self) -> usize {
        32
    }
}

#[tokio::test]
async fn a_refused_image_url_returns_the_generic_text_without_host_or_address() {
    let _private = deny_private_media_urls_in_tests();
    // `localhost` exercises the resolved-address leak (the response used to
    // carry `::1` / `127.0.0.1`); the literal exercises the plain host echo.
    for (url, host) in [
        ("http://localhost:18391/x.png", "localhost"),
        ("http://127.0.0.1:9090/admin", "127.0.0.1"),
    ] {
        let app = app_with(Some(Arc::new(ImageEmbeddingProvider)));
        let (status, body) = post(
            app,
            "/v1/embeddings",
            json!({"input": [{"type": "image_url", "image_url": {"url": url}}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        let message = body["error"]["message"].as_str().expect("message");
        assert_eq!(
            message,
            format!("input[0]: failed to read image: image URL rejected: {GENERIC}")
        );
        let raw = body.to_string();
        for leaked in [host, "127.0.0.1", "::1", ALLOW_PRIVATE_MEDIA_URLS_ENV] {
            assert!(!raw.contains(leaked), "{leaked:?} leaked for {url}: {raw}");
        }
    }
}
