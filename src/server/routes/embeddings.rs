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

//! `POST /v1/embeddings` (and the `/embeddings` alias).
//!
//! Validation happens here, in request order, before any work reaches the
//! worker: the parsed body, the served model id, `encoding_format`,
//! `dimensions`, every item (empty strings, empty token lists, token ids
//! against `vocab_size`, images against `supports_images`). Text items go
//! to the worker as one call (it sorts and micro-batches them), token items
//! as one call, images one at a time; the results are written back in the
//! caller's order.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::Value;

use crate::embeddings::{EmbdNormalize, EmbedOptions, EmbeddingVector, ImageInput};
use crate::server::AppState;
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};
use crate::server::media::{
    ImageInputLimits, current_image_input_limits, try_read_image_url_with_limits,
    validate_image_count,
};
use crate::server::model_provider::model_worker::decode_request_images_with_limits;
use crate::server::types::ErrorResponse;
use crate::server::types::embeddings::{
    EmbedItem, EmbeddingData, EmbeddingEncoding, EmbeddingUsage, EmbeddingsRequest,
    EmbeddingsResponse,
};

/// Message of the `501` returned while no embedding model is loaded.
///
/// Opens with b10621's own sentence, verbatim, so a client that matches on the
/// upstream string keeps working, and then names the mlxcel spellings that also
/// select an embedding worker (#1452). The b10621 flag really is accepted here:
/// `--embeddings` is a startup error when it resolves no embedder, so a server
/// that reaches this message was started without asking for embeddings at all.
pub const NO_EMBEDDING_MODEL_MESSAGE: &str = "This server does not support embeddings. Start it with `--embeddings`, with an embedding \
     checkpoint in -m, or with --embedding-model <path>.";

fn invalid_request(message: impl Into<String>) -> ErrorResponse {
    ErrorResponse::new(message, "invalid_request_error")
}

/// Convert a provider error into the matching HTTP error, mirroring the
/// audio routes' mapping.
pub(crate) fn embedding_error_response(err: EmbeddingError) -> ErrorResponse {
    match err {
        EmbeddingError::QueueFull => {
            ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
        }
        EmbeddingError::Timeout => {
            ErrorResponse::gateway_timeout("Embedding request timed out. Please try again later.")
        }
        EmbeddingError::InvalidInput(message) => invalid_request(message),
        EmbeddingError::Internal(message) => {
            let mut response = ErrorResponse::new(
                format!("embedding inference failed: {message}"),
                "server_error",
            );
            response.status = StatusCode::INTERNAL_SERVER_ERROR;
            response
        }
    }
}

/// Validate every item against the provider before dispatch.
fn validate_items(
    items: &[EmbedItem],
    provider: &dyn EmbeddingModelProvider,
    limits: ImageInputLimits,
) -> Result<(), ErrorResponse> {
    if items.is_empty() {
        return Err(invalid_request("`input` must not be empty"));
    }
    let image_count = items
        .iter()
        .filter(|item| matches!(item, EmbedItem::ImageUrl(_)))
        .count();
    validate_image_count(image_count, limits).map_err(|err| invalid_request(err.to_string()))?;

    let vocab_size = provider.vocab_size();
    for (index, item) in items.iter().enumerate() {
        match item {
            EmbedItem::Text(text) if text.is_empty() => {
                return Err(invalid_request(format!(
                    "input[{index}] is an empty string"
                )));
            }
            EmbedItem::Text(_) => {}
            EmbedItem::Tokens(ids) if ids.is_empty() => {
                return Err(invalid_request(format!(
                    "input[{index}] is an empty token list"
                )));
            }
            EmbedItem::Tokens(ids) => {
                if vocab_size > 0
                    && let Some(bad) = ids.iter().find(|&&id| id as usize >= vocab_size)
                {
                    return Err(invalid_request(format!(
                        "input[{index}] contains token id {bad}, which is >= vocab_size {vocab_size}"
                    )));
                }
            }
            EmbedItem::ImageUrl(_) if !provider.supports_images() => {
                return Err(invalid_request(format!(
                    "input[{index}] is an image, but the loaded embedding model does not accept images"
                )));
            }
            EmbedItem::ImageUrl(_) => {}
        }
    }
    Ok(())
}

/// Resolve every `image_url` item into a decoded image, in item order.
async fn fetch_images(
    items: &[EmbedItem],
    limits: ImageInputLimits,
) -> Result<Vec<(usize, ImageInput)>, ErrorResponse> {
    let mut images = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let EmbedItem::ImageUrl(url) = item else {
            continue;
        };
        let bytes = match try_read_image_url_with_limits(url, limits).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                return Err(invalid_request(format!(
                    "input[{index}]: unsupported image URL scheme"
                )));
            }
            Err(err) => {
                return Err(invalid_request(format!(
                    "input[{index}]: failed to read image: {err:#}"
                )));
            }
        };
        let mut decoded = decode_request_images_with_limits(&[bytes], limits)
            .map_err(|err| invalid_request(format!("input[{index}]: {err:#}")))?;
        let Some(image) = decoded.pop() else {
            return Err(invalid_request(format!(
                "input[{index}]: image decoded to nothing"
            )));
        };
        images.push((index, ImageInput { image }));
    }
    Ok(images)
}

/// Run the provider calls for one request on the calling (blocking) thread
/// and return the vectors in item order plus the total prompt tokens.
fn embed_items(
    provider: &dyn EmbeddingModelProvider,
    items: &[EmbedItem],
    images: Vec<(usize, ImageInput)>,
    opts: &EmbedOptions,
) -> Result<(Vec<EmbeddingVector>, usize), EmbeddingError> {
    let mut slots: Vec<Option<EmbeddingVector>> = (0..items.len()).map(|_| None).collect();
    let mut prompt_tokens = 0usize;

    let text_indices: Vec<usize> = items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| matches!(item, EmbedItem::Text(_)).then_some(i))
        .collect();
    if !text_indices.is_empty() {
        let texts: Vec<String> = text_indices
            .iter()
            .filter_map(|&i| match &items[i] {
                EmbedItem::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect();
        let reply = provider.embed_texts(texts, opts.clone())?;
        prompt_tokens += reply.prompt_tokens;
        for (&i, vector) in text_indices.iter().zip(reply.vectors) {
            slots[i] = Some(vector);
        }
    }

    let token_indices: Vec<usize> = items
        .iter()
        .enumerate()
        .filter_map(|(i, item)| matches!(item, EmbedItem::Tokens(_)).then_some(i))
        .collect();
    if !token_indices.is_empty() {
        let rows: Vec<Vec<u32>> = token_indices
            .iter()
            .filter_map(|&i| match &items[i] {
                EmbedItem::Tokens(ids) => Some(ids.clone()),
                _ => None,
            })
            .collect();
        let reply = provider.embed_tokens(rows, opts.clone())?;
        prompt_tokens += reply.prompt_tokens;
        for (&i, vector) in token_indices.iter().zip(reply.vectors) {
            slots[i] = Some(vector);
        }
    }

    for (index, image) in images {
        let reply = provider.embed_image(image, opts.clone())?;
        prompt_tokens += reply.prompt_tokens;
        if let Some(vector) = reply.vectors.into_iter().next() {
            slots[index] = Some(vector);
        }
    }

    let vectors = slots
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| EmbeddingError::Internal("an input produced no vector".to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((vectors, prompt_tokens))
}

/// Which wire shape an embedding response is rendered in.
///
/// b10621 routes `/embeddings` and `/embedding` to its native handler and
/// `/v1/embeddings` to the OpenAI one, and the two shapes genuinely differ:
/// the native one is a bare JSON array of `{index, embedding}` whose
/// `embedding` is an array OF arrays, while the OpenAI one is an object with
/// `object` / `data` / `model` / `usage` and a flat `embedding`. mlxcel used
/// to answer the OpenAI shape on all three paths (#1441).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingShape {
    /// `POST /v1/embeddings`.
    OpenAi,
    /// `POST /embeddings` and `POST /embedding`.
    Native,
}

/// POST /v1/embeddings
pub async fn create_embeddings(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    embeddings_impl(state, body, EmbeddingShape::OpenAi).await
}

/// POST /embeddings and POST /embedding (the b10621 native routes).
///
/// Accepts the native `content` spelling as well as `input`, because upstream's
/// legacy `/embedding` takes `{"content": ...}`.
pub async fn native_embeddings(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    let mut body = body;
    if let Some(object) = body.as_object_mut()
        && !object.contains_key("input")
        && let Some(content) = object.remove("content")
    {
        object.insert("input".to_string(), content);
    }
    embeddings_impl(state, body, EmbeddingShape::Native).await
}

async fn embeddings_impl(state: AppState, body: Value, shape: EmbeddingShape) -> Response {
    // b10621 answers a body with neither key with this exact sentence. serde
    // would answer "missing field `input`", which names only one of the two
    // spellings the server accepts and is not the string a b10621 client
    // matches on (#1452).
    if body
        .as_object()
        .is_some_and(|object| !object.contains_key("input") && !object.contains_key("content"))
    {
        return invalid_request("\"input\" or \"content\" must be provided").into_response();
    }
    let request: EmbeddingsRequest = match serde_json::from_value(body) {
        Ok(request) => request,
        Err(err) => return invalid_request(format!("invalid request body: {err}")).into_response(),
    };

    let Some(provider) = state.embedding_model.clone() else {
        return ErrorResponse::not_implemented(NO_EMBEDDING_MODEL_MESSAGE).into_response();
    };

    if let Some(model) = request.model.as_deref()
        && model != provider.model_id()
    {
        return invalid_request(format!(
            "model `{model}` is not the served embedding model `{}`",
            provider.model_id()
        ))
        .into_response();
    }

    let Some(encoding) = EmbeddingEncoding::parse(request.encoding_format.as_deref()) else {
        return invalid_request(format!(
            "unsupported encoding_format `{}`; expected `float` or `base64`",
            request.encoding_format.unwrap_or_default()
        ))
        .into_response();
    };

    if let Some(n) = request.dimensions
        && (n == 0 || n > provider.dim())
    {
        return invalid_request(format!(
            "`dimensions` must be between 1 and {} for this model, got {n}",
            provider.dim()
        ))
        .into_response();
    }

    let items = request.input.into_items();
    let image_limits = current_image_input_limits();
    if let Err(err) = validate_items(&items, provider.as_ref(), image_limits) {
        return err.into_response();
    }
    let images = match fetch_images(&items, image_limits).await {
        Ok(images) => images,
        Err(err) => return err.into_response(),
    };

    // b10621 reads `embd_normalize` per request, defaulting to the server's
    // `--embd-normalize` and, when that is unset too, to the checkpoint's own
    // `normalize` flag (#1452).
    let normalize = match request.embd_normalize {
        Some(raw) => match EmbdNormalize::new(raw) {
            Ok(kind) => Some(kind),
            Err(message) => {
                return ErrorResponse::new(
                    message.replace("--embd-normalize", "embd_normalize"),
                    "invalid_request_error",
                )
                .into_response();
            }
        },
        None => state.config.embd_normalize,
    };
    let opts = EmbedOptions {
        instruction: request.instruction,
        dimensions: request.dimensions,
        normalize,
    };
    let started = std::time::Instant::now();
    // The provider blocks on the worker's reply channel, so keep it off the
    // async executor.
    let worker_provider: Arc<dyn EmbeddingModelProvider> = provider.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        embed_items(worker_provider.as_ref(), &items, images, &opts)
    })
    .await;
    let (vectors, prompt_tokens) = match outcome {
        Ok(Ok(result)) => result,
        Ok(Err(err)) => return embedding_error_response(err).into_response(),
        Err(join_err) => {
            tracing::error!("embedding task panicked: {join_err}");
            let mut response = ErrorResponse::new("embedding request failed", "server_error");
            response.status = StatusCode::INTERNAL_SERVER_ERROR;
            return response.into_response();
        }
    };

    if let Some((vector_index, value_index)) = vectors.iter().enumerate().find_map(|(i, vector)| {
        vector
            .values
            .iter()
            .position(|value| !value.is_finite())
            .map(|j| (i, j))
    }) {
        tracing::error!(
            vector_index,
            value_index,
            "embedding inference returned a non-finite value"
        );
        let mut response = ErrorResponse::new(
            "embedding inference returned an invalid numeric result",
            "server_error",
        );
        response.status = StatusCode::INTERNAL_SERVER_ERROR;
        return response.into_response();
    }

    state
        .metrics
        .record_request(prompt_tokens, 0, started.elapsed().as_millis() as u64);

    match shape {
        EmbeddingShape::Native => {
            // A bare array of `{index, embedding}`, with `embedding` nested one
            // level deeper than the OpenAI shape: upstream reports one row per
            // pooled sequence, and mlxcel pools to exactly one.
            let rows: Vec<Value> = vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| {
                    serde_json::json!({
                        "index": index,
                        "embedding": [vector.values],
                    })
                })
                .collect();
            Json(rows).into_response()
        }
        EmbeddingShape::OpenAi => {
            let data = vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| EmbeddingData::from_vector(index, vector, encoding))
                .collect();
            Json(EmbeddingsResponse {
                object: "list".to_string(),
                data,
                model: provider.model_id().to_string(),
                usage: EmbeddingUsage {
                    prompt_tokens,
                    total_tokens: prompt_tokens,
                },
            })
            .into_response()
        }
    }
}

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
