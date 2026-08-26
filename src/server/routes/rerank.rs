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

//! `POST /v1/rerank` (and the `/rerank` alias).
//!
//! Validation happens here, before any work reaches the worker: the parsed
//! body, the served model id, `top_n`, the query, every document (empty text,
//! images against the loaded kind) and `instruction` against a reranker that
//! can use one. The worker then scores the whole document list in one call and
//! the results are sorted, cut to `top_n` and optionally echoed back.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::Value;

use crate::rerank::{ImageInput, RerankItem};
use crate::server::AppState;
use crate::server::media::{
    ImageInputLimits, current_image_input_limits, try_read_image_url_with_limits,
    validate_image_count,
};
use crate::server::model_provider::model_worker::decode_request_images_with_limits;
use crate::server::rerank_model::{RerankError, RerankModelProvider};
use crate::server::types::ErrorResponse;
use crate::server::types::rerank::{
    RerankInput, RerankRequest, RerankResponse, RerankResult, RerankUsage, sort_and_truncate,
};

/// Message of the `501` returned while no reranker is loaded.
pub const NO_RERANKER_MODEL_MESSAGE: &str =
    "No reranker loaded; start with -m <sequence-classifier checkpoint> or --reranker-model <path>";

fn invalid_request(message: impl Into<String>) -> ErrorResponse {
    ErrorResponse::new(message, "invalid_request_error")
}

/// Convert a provider error into the matching HTTP error, mirroring the
/// embedding route's mapping.
pub(crate) fn rerank_error_response(err: RerankError) -> ErrorResponse {
    match err {
        RerankError::QueueFull => {
            ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
        }
        RerankError::Timeout => {
            ErrorResponse::gateway_timeout("Rerank request timed out. Please try again later.")
        }
        RerankError::InvalidInput(message) => invalid_request(message),
        RerankError::Internal(message) => {
            let mut response = ErrorResponse::new(
                format!("rerank inference failed: {message}"),
                "server_error",
            );
            response.status = StatusCode::INTERNAL_SERVER_ERROR;
            response
        }
    }
}

/// Validate the query and every document against the loaded reranker.
fn validate_items(
    query: &RerankInput,
    documents: &[RerankInput],
    provider: &dyn RerankModelProvider,
    limits: ImageInputLimits,
) -> Result<(), ErrorResponse> {
    if documents.is_empty() {
        return Err(invalid_request("`documents` must not be empty"));
    }
    let image_count = usize::from(query.image_url().is_some())
        + documents
            .iter()
            .filter(|document| document.image_url().is_some())
            .count();
    validate_image_count(image_count, limits).map_err(|err| invalid_request(err.to_string()))?;

    let supports_images = provider.supports_images();
    let check = |item: &RerankInput, label: String| -> Result<(), ErrorResponse> {
        if item.is_empty() {
            return Err(invalid_request(format!(
                "{label} carries neither text nor an image"
            )));
        }
        if item.image_url().is_some() && !supports_images {
            return Err(invalid_request(format!(
                "{label} is an image, but the loaded reranker does not accept images"
            )));
        }
        Ok(())
    };
    check(query, "`query`".to_string())?;
    for (index, document) in documents.iter().enumerate() {
        check(document, format!("documents[{index}]"))?;
    }
    Ok(())
}

/// Resolve an item's `image_url` into a decoded image.
async fn fetch_image(
    item: &RerankInput,
    label: &str,
    limits: ImageInputLimits,
) -> Result<Option<ImageInput>, ErrorResponse> {
    let Some(url) = item.image_url() else {
        return Ok(None);
    };
    let bytes = match try_read_image_url_with_limits(url, limits).await {
        Ok(Some(bytes)) => bytes,
        Ok(None) => {
            return Err(invalid_request(format!(
                "{label}: unsupported image URL scheme"
            )));
        }
        Err(err) => {
            return Err(invalid_request(format!(
                "{label}: failed to read image: {err:#}"
            )));
        }
    };
    let mut decoded = decode_request_images_with_limits(&[bytes], limits)
        .map_err(|err| invalid_request(format!("{label}: {err:#}")))?;
    let Some(image) = decoded.pop() else {
        return Err(invalid_request(format!(
            "{label}: image decoded to nothing"
        )));
    };
    Ok(Some(ImageInput { image }))
}

/// Turn one request item into the worker's item type, fetching its image.
async fn to_rerank_item(
    item: &RerankInput,
    label: &str,
    limits: ImageInputLimits,
) -> Result<RerankItem, ErrorResponse> {
    Ok(RerankItem {
        text: item.text().map(str::to_string),
        image: fetch_image(item, label, limits).await?,
    })
}

/// POST /v1/rerank
pub async fn create_rerank(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    let request: RerankRequest = match serde_json::from_value(body) {
        Ok(request) => request,
        Err(err) => return invalid_request(format!("invalid request body: {err}")).into_response(),
    };

    let Some(provider) = state.rerank_model.clone() else {
        return ErrorResponse::not_implemented(NO_RERANKER_MODEL_MESSAGE).into_response();
    };

    if let Some(model) = request.model.as_deref()
        && model != provider.model_id()
    {
        return invalid_request(format!(
            "model `{model}` is not the served reranker `{}`",
            provider.model_id()
        ))
        .into_response();
    }

    if request.top_n == Some(0) {
        return invalid_request("`top_n` must be at least 1").into_response();
    }

    let instruction = request
        .instruction
        .as_deref()
        .map(str::trim)
        .filter(|task| !task.is_empty())
        .map(str::to_string);
    if instruction.is_some() && !provider.kind().accepts_instruction() {
        return invalid_request(
            "`instruction` is only supported by the generative rerankers; the loaded \
             sequence-classifier reranker scores the query/document pair directly",
        )
        .into_response();
    }

    let image_limits = current_image_input_limits();
    if let Err(err) = validate_items(
        &request.query,
        &request.documents,
        provider.as_ref(),
        image_limits,
    ) {
        return err.into_response();
    }

    let query = match to_rerank_item(&request.query, "`query`", image_limits).await {
        Ok(item) => item,
        Err(err) => return err.into_response(),
    };
    let mut documents = Vec::with_capacity(request.documents.len());
    for (index, document) in request.documents.iter().enumerate() {
        match to_rerank_item(document, &format!("documents[{index}]"), image_limits).await {
            Ok(item) => documents.push(item),
            Err(err) => return err.into_response(),
        }
    }

    let started = std::time::Instant::now();
    // The provider blocks on the worker's reply channel, so keep it off the
    // async executor.
    let worker_provider: Arc<dyn RerankModelProvider> = provider.clone();
    let outcome =
        tokio::task::spawn_blocking(move || worker_provider.rerank(query, documents, instruction))
            .await;
    let scored = match outcome {
        Ok(Ok(result)) => result,
        Ok(Err(err)) => return rerank_error_response(err).into_response(),
        Err(join_err) => {
            tracing::error!("rerank task panicked: {join_err}");
            let mut response = ErrorResponse::new("rerank request failed", "server_error");
            response.status = StatusCode::INTERNAL_SERVER_ERROR;
            return response.into_response();
        }
    };
    if scored.scores.len() != request.documents.len() {
        let mut response = ErrorResponse::new(
            format!(
                "reranker returned {} scores for {} documents",
                scored.scores.len(),
                request.documents.len()
            ),
            "server_error",
        );
        response.status = StatusCode::INTERNAL_SERVER_ERROR;
        return response.into_response();
    }

    if let Some((index, score)) = scored
        .scores
        .iter()
        .copied()
        .enumerate()
        .find(|(_, score)| !score.is_finite() || !(0.0..=1.0).contains(score))
    {
        tracing::error!(index, score, "reranker returned an invalid relevance score");
        let mut response = ErrorResponse::new(
            "rerank inference returned an invalid numeric result",
            "server_error",
        );
        response.status = StatusCode::INTERNAL_SERVER_ERROR;
        return response.into_response();
    }
    state.metrics.record_request(
        scored.prompt_tokens,
        0,
        started.elapsed().as_millis() as u64,
    );

    let results = scored
        .scores
        .iter()
        .enumerate()
        .map(|(index, &relevance_score)| RerankResult {
            index,
            relevance_score,
            document: request
                .return_documents
                .then(|| request.documents[index].clone()),
        })
        .collect();
    Json(RerankResponse {
        model: provider.model_id().to_string(),
        results: sort_and_truncate(results, request.top_n),
        usage: RerankUsage {
            prompt_tokens: scored.prompt_tokens,
            total_tokens: scored.prompt_tokens,
        },
    })
    .into_response()
}

#[cfg(test)]
#[path = "rerank_tests.rs"]
mod tests;
