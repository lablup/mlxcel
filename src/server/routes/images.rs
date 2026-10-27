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

//! `POST /v1/images/generations` and `POST /v1/images/edits` (and the
//! unprefixed `/images/...` aliases).
//!
//! Validation happens here, in request order, before any work reaches the
//! worker: the parsed body, the served model id, `response_format`,
//! `output_format`, the file-writing options, `n`, the size, the task and the
//! size multiple, and (for edits) each reference image. Generation runs on the
//! blocking pool through the provider; the worker serializes requests.
//!
//! The server never writes files on behalf of a client, so `response_format`
//! `url` and `path`, `output_path` and `output_dir` are `400`. Edit images are
//! data URLs or URLs read through the same limits as chat images; a raw
//! filesystem path is `400`.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use image::{DynamicImage, GenericImageView};
use serde_json::{Map, Value};

use crate::imagegen::run::{parse_size, random_seed};
use crate::imagegen::{ImageRequest, ImageTask, to_b64_png};
use crate::server::AppState;
use crate::server::image_model::{ImageError, ImageModelProvider};
use crate::server::media::{
    ImageInputLimits, current_image_input_limits, fetch_labeled_image, validate_image_count,
};
use crate::server::types::ErrorResponse;
use crate::server::types::images::{
    ImageEditsRequest, ImageGenerationsRequest, ImageResponseData, ImagesResponse,
};

/// Message of the `501` returned while no image model is loaded.
pub const NO_IMAGE_MODEL_MESSAGE: &str =
    "This server does not serve image generation. Start it with `--image-model <path>`.";

/// Size used when a generation request names none.
const DEFAULT_SIZE: (u32, u32) = crate::imagegen::pipeline::DEFAULT_IMAGE_SIZE;

fn invalid_request(message: impl Into<String>) -> ErrorResponse {
    ErrorResponse::new(message, "invalid_request_error")
}

/// Convert a provider error into the matching HTTP error.
pub(crate) fn image_error_response(err: ImageError) -> ErrorResponse {
    match err {
        ImageError::QueueFull => {
            ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
        }
        ImageError::Timeout => {
            ErrorResponse::gateway_timeout("Image request timed out. Please try again later.")
        }
        ImageError::InvalidInput(message) => invalid_request(message),
        ImageError::Internal(message) => {
            ErrorResponse::internal_server_error(format!("image generation failed: {message}"))
        }
    }
}

/// The fields both bodies share, moved out of whichever one was parsed.
struct Common {
    model: Option<String>,
    n: Option<u32>,
    size: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    seed: Option<u64>,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
    response_format: Option<String>,
    output_format: Option<String>,
    output_path: Option<String>,
    output_dir: Option<String>,
    extra: Map<String, Value>,
}

impl Common {
    /// Reject what the server does not serve, before looking at sizes.
    fn check_wire_options(&self) -> Result<(), ErrorResponse> {
        match self.response_format.as_deref() {
            None | Some("b64_json") => {}
            Some(other @ ("url" | "path")) => {
                return Err(invalid_request(format!(
                    "response_format `{other}` is not supported: the server does not write \
                     files on behalf of a client; use `b64_json`"
                )));
            }
            Some(other) => {
                return Err(invalid_request(format!(
                    "unsupported response_format `{other}`; expected `b64_json`"
                )));
            }
        }
        if let Some(format) = self.output_format.as_deref()
            && !format.eq_ignore_ascii_case("png")
        {
            return Err(invalid_request(format!(
                "unsupported output_format `{format}`; only `png` is produced"
            )));
        }
        for (name, value) in [
            ("output_path", &self.output_path),
            ("output_dir", &self.output_dir),
        ] {
            if value.is_some() {
                return Err(invalid_request(format!(
                    "`{name}` is not supported: the server does not write files on behalf of a client"
                )));
            }
        }
        Ok(())
    }

    /// `width` and `height` must come together; they override `size`.
    ///
    /// `None` means the caller named no size and the route supplies a default.
    fn requested_size(&self) -> Result<Option<(u32, u32)>, ErrorResponse> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => return Ok(Some((w, h))),
            (Some(_), None) | (None, Some(_)) => {
                return Err(invalid_request(
                    "`width` and `height` must be given together, or neither",
                ));
            }
            (None, None) => {}
        }
        match self.size.as_deref() {
            Some(size) if !size.trim().is_empty() => parse_size(size)
                .map(Some)
                .map_err(|e| invalid_request(e.to_string())),
            _ => Ok(None),
        }
    }
}

/// POST /v1/images/generations
pub async fn create_image_generation(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Response {
    let request: ImageGenerationsRequest = match serde_json::from_value(body) {
        Ok(request) => request,
        Err(err) => return invalid_request(format!("invalid request body: {err}")).into_response(),
    };
    let Some(provider) = state.image_model.clone() else {
        return ErrorResponse::not_implemented(NO_IMAGE_MODEL_MESSAGE).into_response();
    };

    let prompt = request.prompt;
    let common = Common {
        model: request.model,
        n: request.n,
        size: request.size,
        width: request.width,
        height: request.height,
        steps: request.steps,
        seed: request.seed,
        guidance: request.guidance,
        negative_prompt: request.negative_prompt,
        response_format: request.response_format,
        output_format: request.output_format,
        output_path: request.output_path,
        output_dir: request.output_dir,
        extra: request.extra,
    };
    let size = match common.requested_size() {
        Ok(size) => size.unwrap_or(DEFAULT_SIZE),
        Err(err) => return err.into_response(),
    };
    let image_request = match build_request(
        provider.as_ref(),
        &common,
        prompt,
        ImageTask::Generate,
        Vec::new(),
        size,
    ) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    respond(provider, image_request, size).await
}

/// POST /v1/images/edits
pub async fn create_image_edit(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    let request: ImageEditsRequest = match serde_json::from_value(body) {
        Ok(request) => request,
        Err(err) => return invalid_request(format!("invalid request body: {err}")).into_response(),
    };
    let Some(provider) = state.image_model.clone() else {
        return ErrorResponse::not_implemented(NO_IMAGE_MODEL_MESSAGE).into_response();
    };

    let prompt = request.prompt;
    let sources = request.image.into_vec();
    let common = Common {
        model: request.model,
        n: request.n,
        size: request.size,
        width: request.width,
        height: request.height,
        steps: request.steps,
        seed: request.seed,
        guidance: request.guidance,
        negative_prompt: request.negative_prompt,
        response_format: request.response_format,
        output_format: request.output_format,
        output_path: request.output_path,
        output_dir: request.output_dir,
        extra: request.extra,
    };

    // Cheap checks first so a bad body never costs a download.
    if let Err(err) = check_model(provider.as_ref(), common.model.as_deref())
        .and_then(|()| common.check_wire_options())
    {
        return err.into_response();
    }
    if !provider.supports(ImageTask::Edit) {
        return unsupported_task(provider.as_ref(), ImageTask::Edit).into_response();
    }
    let requested = match common.requested_size() {
        Ok(size) => size,
        Err(err) => return err.into_response(),
    };

    let limits = current_image_input_limits();
    let references = match fetch_references(&sources, limits).await {
        Ok(images) => images,
        Err(err) => return err.into_response(),
    };
    let size = requested.unwrap_or_else(|| {
        let (width, height) = references[0].dimensions();
        provider.default_edit_size(width, height)
    });

    let image_request = match build_request(
        provider.as_ref(),
        &common,
        prompt,
        ImageTask::Edit,
        references,
        size,
    ) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    respond(provider, image_request, size).await
}

fn check_model(
    provider: &dyn ImageModelProvider,
    model: Option<&str>,
) -> Result<(), ErrorResponse> {
    match model {
        Some(model) if !model.is_empty() && model != provider.model_id() => {
            Err(invalid_request(format!(
                "model `{model}` is not the served image model `{}`",
                provider.model_id()
            )))
        }
        _ => Ok(()),
    }
}

fn unsupported_task(provider: &dyn ImageModelProvider, task: ImageTask) -> ErrorResponse {
    invalid_request(format!(
        "the loaded image model family `{}` does not support task `{}`",
        provider.family(),
        task.as_str()
    ))
}

/// Assemble and validate the model-facing request.
fn build_request(
    provider: &dyn ImageModelProvider,
    common: &Common,
    prompt: String,
    task: ImageTask,
    reference_images: Vec<DynamicImage>,
    size: (u32, u32),
) -> Result<ImageRequest, ErrorResponse> {
    check_model(provider, common.model.as_deref())?;
    common.check_wire_options()?;
    if !provider.supports(task) {
        return Err(unsupported_task(provider, task));
    }
    let request = ImageRequest {
        prompt,
        negative_prompt: common.negative_prompt.clone(),
        task,
        reference_images,
        width: size.0,
        height: size.1,
        steps: common.steps,
        guidance: common.guidance,
        seed: common.seed.unwrap_or_else(random_seed),
        n: common.n.unwrap_or(1),
        extra: common.extra.clone(),
    };
    request
        .validate(provider.size_multiple())
        .map_err(|e| invalid_request(e.to_string()))?;
    Ok(request)
}

/// Resolve each reference into a decoded image, in request order.
async fn fetch_references(
    sources: &[String],
    limits: ImageInputLimits,
) -> Result<Vec<DynamicImage>, ErrorResponse> {
    if sources.is_empty() {
        return Err(invalid_request("`image` must contain at least one image"));
    }
    validate_image_count(sources.len(), limits).map_err(|err| invalid_request(err.to_string()))?;
    let mut images = Vec::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        if source.is_empty() {
            return Err(invalid_request(format!("image[{index}] is empty")));
        }
        // A raw filesystem path would let a client name any file the server can
        // read; only data URLs and URLs are accepted.
        if !source.starts_with("data:") && !source.contains("://") {
            return Err(invalid_request(format!(
                "image[{index}]: raw file paths are not accepted; send a data URL or a URL"
            )));
        }
        let image = fetch_labeled_image(source, &format!("image[{index}]"), limits).await?;
        images.push(image);
    }
    Ok(images)
}

/// Run the request on the blocking pool and render the response.
async fn respond(
    provider: Arc<dyn ImageModelProvider>,
    request: ImageRequest,
    size: (u32, u32),
) -> Response {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let outcome =
        tokio::task::spawn_blocking(move || -> Result<Vec<ImageResponseData>, ImageError> {
            let images = provider.generate(request)?;
            images
                .iter()
                .map(|image| {
                    Ok(ImageResponseData {
                        b64_json: to_b64_png(image)
                            .map_err(|e| ImageError::Internal(format!("{e:#}")))?,
                        mime_type: "image/png",
                        width: image.width,
                        height: image.height,
                        seed: image.seed,
                        revised_prompt: image.revised_prompt.clone(),
                    })
                })
                .collect()
        })
        .await;
    match outcome {
        Ok(Ok(data)) => Json(ImagesResponse {
            created,
            data,
            output_format: "png",
            size: format!("{}x{}", size.0, size.1),
        })
        .into_response(),
        Ok(Err(err)) => image_error_response(err).into_response(),
        Err(join) => {
            ErrorResponse::internal_server_error(format!("image generation task failed: {join}"))
                .into_response()
        }
    }
}

#[cfg(test)]
#[path = "images_tests.rs"]
mod tests;
