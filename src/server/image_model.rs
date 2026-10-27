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

//! Image-model provider interface.
//!
//! The transport-agnostic seam between `POST /v1/images/generations`,
//! `POST /v1/images/edits` and the image worker, mirroring
//! [`super::embedding_model`]. Routes see only [`ImageRequest`] in and PNG-ready
//! [`GeneratedImage`] out; weight loading and the diffusion loop stay behind
//! this trait on the worker thread. Until a provider is registered the
//! [`AppState`](crate::server::AppState) slot stays `None` and both routes
//! return a structured `501 Not Implemented` naming `--image-model`.

use thiserror::Error;

use crate::imagegen::{GeneratedImage, ImageRequest, ImageSamplingDefaults, ImageTask};

/// Failure modes of one image request.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ImageError {
    /// The bounded worker queue was full at admission time (`503`).
    #[error("image worker queue is full")]
    QueueFull,
    /// The worker did not reply within the per-request timeout (`504`). The
    /// in-flight MLX work is not cancelled.
    #[error("image request timed out")]
    Timeout,
    /// The request content is invalid for the loaded family (`400`).
    #[error("{0}")]
    InvalidInput(String),
    /// Generation failed (`500`). The worker stays alive for the next request.
    #[error("image generation failed: {0}")]
    Internal(String),
}

/// Provider-facing interface implemented by the image worker.
///
/// Synchronous and blocking (it waits for the worker's reply), so routes call
/// it inside `spawn_blocking`.
pub trait ImageModelProvider: Send + Sync {
    /// Run `req` and return `req.n` images; image `i` uses seed `req.seed + i`.
    fn generate(&self, req: ImageRequest) -> Result<Vec<GeneratedImage>, ImageError>;

    /// Served model id, reported in responses.
    fn model_id(&self) -> &str;

    /// Unix timestamp the model was loaded at.
    fn created_at(&self) -> i64;

    /// Probe family name of the loaded checkpoint (`flux2`, ...).
    fn family(&self) -> &str;

    /// Whether the family implements `task`.
    fn supports(&self, task: ImageTask) -> bool;

    /// Default steps and guidance for `task`.
    fn defaults(&self, task: ImageTask) -> ImageSamplingDefaults;

    /// Size multiple that width and height must satisfy.
    fn size_multiple(&self) -> u32;

    /// Output size of an edit that names none, from the first reference's
    /// width and height. The default is the reference's own size.
    ///
    /// Used by: the `/v1/images/edits` route; implemented by
    /// `ImageWorkerProvider` from the model's `default_edit_size`.
    fn default_edit_size(&self, width: u32, height: u32) -> (u32, u32) {
        (width, height)
    }
}
