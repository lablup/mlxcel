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

//! `POST /v1/decisions` (and the `/decisions` alias).
//!
//! The request flow: parse the body, refuse with `501` when no decision model
//! is loaded, check `model`, run the shared validation (rules 1 to 8) against
//! the worker's declared types and media, and only then acquire media, so a
//! refused medium is never fetched. Images and audio go through the same
//! fetchers and limits as chat requests; videos go through the allowlist-
//! guarded resolver. The worker answers in one call inside `spawn_blocking`.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use serde_json::{Map, Value};

use crate::decision::{MediaPresence, validate_questions};
use crate::multimodal::video::PrivateTempDir;
use crate::server::AppState;
use crate::server::decision_model::{DecisionMedia, DecisionModelProvider, DecisionProviderError};
use crate::server::media::{
    current_image_input_limits, fetch_labeled_image, read_decision_audio, resolve_video_url,
    validate_image_count,
};
use crate::server::types::ErrorResponse;
use crate::server::types::request::VideoUrl;

/// Message of the `501` returned while no decision model is loaded.
pub const NO_DECISION_MODEL_MESSAGE: &str = "This server does not serve typed decisions. Start it with --decision-model \
     <path> (or MLXCEL_DECISION_MODEL) to serve POST /v1/decisions.";

/// Returned for `videos` when `ffmpeg` is not on `PATH`.
pub(crate) const VIDEO_NEEDS_FFMPEG_MESSAGE: &str = "Video input requires `ffmpeg` on PATH. Install \
     ffmpeg (e.g. `brew install ffmpeg` on macOS or `apt install ffmpeg` on Linux) and retry.";

/// Videos one request may carry. Each one is resolved (possibly downloaded)
/// before the model runs, and the model's token gate refuses an overflow only
/// after that, so the count is capped up front (issue #3296).
pub(crate) const MAX_DECISION_VIDEOS: usize = 8;

/// Refuse more than [`MAX_DECISION_VIDEOS`] videos, and `videos` on a host
/// without `ffmpeg`. Both checks run before any clip is resolved.
fn check_video_support(video_count: usize, ffmpeg_available: bool) -> Result<(), ErrorResponse> {
    if video_count > MAX_DECISION_VIDEOS {
        return Err(invalid_request(format!(
            "`videos` holds {video_count} entries; at most {MAX_DECISION_VIDEOS} are allowed per \
             request"
        )));
    }
    if video_count > 0 && !ffmpeg_available {
        return Err(invalid_request(VIDEO_NEEDS_FFMPEG_MESSAGE));
    }
    Ok(())
}

fn invalid_request(message: impl Into<String>) -> ErrorResponse {
    ErrorResponse::new(message, "invalid_request_error")
}

/// Convert a provider error into the matching HTTP error.
pub(crate) fn decision_error_response(err: DecisionProviderError) -> ErrorResponse {
    match err {
        DecisionProviderError::QueueFull => {
            ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
        }
        DecisionProviderError::Timeout => {
            ErrorResponse::gateway_timeout("Decision request timed out. Please try again later.")
        }
        DecisionProviderError::InvalidInput(message) => invalid_request(message),
        DecisionProviderError::Internal(message) => {
            tracing::error!(target: "mlxcel::decision", "decision prediction failed: {message}");
            ErrorResponse::internal_server_error("Decision prediction failed")
        }
    }
}

/// The parsed body of one request.
struct DecisionBody {
    model: Option<String>,
    state: Option<Value>,
    questions: Value,
    images: Vec<String>,
    videos: Vec<String>,
    audio: Option<String>,
}

/// A list of strings, or `null` / absent.
fn string_list(body: &Map<String, Value>, field: &str) -> Result<Vec<String>, ErrorResponse> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid_request(format!("`{field}` must be a list of strings")))
            })
            .collect(),
        Some(_) => Err(invalid_request(format!(
            "`{field}` must be a list of strings"
        ))),
    }
}

/// A string, or `null` / absent.
fn optional_string(
    body: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, ErrorResponse> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(invalid_request(format!("`{field}` must be a string"))),
    }
}

fn parse_body(body: Value) -> Result<DecisionBody, ErrorResponse> {
    let Value::Object(mut body) = body else {
        return Err(invalid_request(
            "invalid request body: expected a JSON object",
        ));
    };
    let model = optional_string(&body, "model")?.filter(|model| !model.is_empty());
    let images = string_list(&body, "images")?;
    let videos = string_list(&body, "videos")?;
    let audio = optional_string(&body, "audio")?.filter(|audio| !audio.is_empty());
    let state = body.remove("state").filter(|state| !state.is_null());
    let questions = body.remove("questions").unwrap_or(Value::Null);
    Ok(DecisionBody {
        model,
        state,
        questions,
        images,
        videos,
        audio,
    })
}

/// Fetch and decode every medium the request carries.
///
/// Returned video guards keep server-owned temp files alive; the provider
/// drops them once the prediction has finished.
async fn acquire_media(
    state: &AppState,
    body: &DecisionBody,
) -> Result<(DecisionMedia, Vec<PrivateTempDir>), ErrorResponse> {
    // Refuse too many clips, then tell the caller ffmpeg is missing, before
    // any medium is fetched. The probe is memoized at startup.
    check_video_support(
        body.videos.len(),
        crate::multimodal::video::ffmpeg_available(),
    )?;
    let limits = current_image_input_limits();
    validate_image_count(body.images.len(), limits)
        .map_err(|err| invalid_request(err.to_string()))?;
    let mut images = Vec::with_capacity(body.images.len());
    for (index, url) in body.images.iter().enumerate() {
        images.push(fetch_labeled_image(url, &format!("images[{index}]"), limits).await?);
    }

    let mut videos = Vec::with_capacity(body.videos.len());
    let mut guards = Vec::new();
    for (index, url) in body.videos.iter().enumerate() {
        let video = VideoUrl {
            url: url.clone(),
            fps: None,
        };
        let resolved = resolve_video_url(&video, &state.video_dir_allowlist)
            .await
            .ok_or_else(|| {
                invalid_request(format!(
                    "videos[{index}]: could not read the video. Local paths must sit inside a \
                     directory listed in {}.",
                    crate::server::media::VIDEO_DIR_ALLOWLIST_ENV
                ))
            })?;
        videos.push(resolved.source);
        if let Some(guard) = resolved.temp_guard {
            guards.push(guard);
        }
    }

    let audio = match body.audio.as_deref() {
        None => None,
        Some(data) => {
            let bytes = read_decision_audio(data)
                .await
                .map_err(|err| invalid_request(format!("audio: {err}")))?;
            let samples = tokio::task::spawn_blocking(move || {
                crate::decision::media::decode_audio_16k(&bytes)
            })
            .await
            .map_err(|err| {
                tracing::error!("decision audio decode task panicked: {err}");
                ErrorResponse::internal_server_error("Decision prediction failed")
            })?
            .map_err(|err| invalid_request(format!("audio: {err}")))?;
            Some(samples)
        }
    };
    Ok((
        DecisionMedia {
            images,
            videos,
            audio,
        },
        guards,
    ))
}

/// POST /v1/decisions
pub async fn create_decisions(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    let body = match parse_body(body) {
        Ok(body) => body,
        Err(err) => return err.into_response(),
    };
    let Some(provider) = state.decision_model.clone() else {
        return ErrorResponse::not_implemented(NO_DECISION_MODEL_MESSAGE).into_response();
    };
    if let Some(model) = body.model.as_deref()
        && model != provider.model_id()
    {
        return invalid_request(format!(
            "model `{model}` is not the served decision model `{}`",
            provider.model_id()
        ))
        .into_response();
    }

    let presence = MediaPresence {
        images: !body.images.is_empty(),
        videos: !body.videos.is_empty(),
        audio: body.audio.is_some(),
    };
    let questions = match validate_questions(
        provider.decision_types(),
        provider.decision_media(),
        body.state.as_ref(),
        &body.questions,
        presence,
    ) {
        Ok(questions) => questions,
        Err(err) => return invalid_request(err.to_string()).into_response(),
    };

    let (media, guards) = match acquire_media(&state, &body).await {
        Ok(acquired) => acquired,
        Err(err) => return err.into_response(),
    };

    let started = std::time::Instant::now();
    let worker: Arc<dyn DecisionModelProvider> = provider.clone();
    let request_state = body.state;
    // The guards move with the prediction: on a timeout the worker may still
    // be decoding these files, so it, not this handler, removes them.
    let outcome = tokio::task::spawn_blocking(move || {
        worker.predict(request_state, questions, media, guards)
    })
    .await;
    let result = match outcome {
        Ok(Ok(result)) => result,
        Ok(Err(err)) => return decision_error_response(err).into_response(),
        Err(join_err) => {
            tracing::error!("decision task panicked: {join_err}");
            return ErrorResponse::internal_server_error("Decision prediction failed")
                .into_response();
        }
    };
    state.metrics.record_request(
        result.usage.input_tokens,
        0,
        started.elapsed().as_millis() as u64,
    );
    Json(result.to_response_json(provider.model_id())).into_response()
}

#[cfg(test)]
#[path = "decisions_tests.rs"]
mod tests;
