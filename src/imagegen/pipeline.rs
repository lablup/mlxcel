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

//! Request, result and model-trait types shared by every image family.
//!
//! Mirrors `ImageGenerationRequest` / `ImageGenerationResult` and the
//! `ImageGenerationModel` protocol in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/generate/image.py>
//! and `ImageSamplingDefaults` in `image_defaults.py`.

use anyhow::{Result, bail};
use image::DynamicImage;
use serde_json::{Map, Value};

/// Which operation a request asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageTask {
    /// Text-to-image.
    Generate,
    /// Reference-image editing.
    Edit,
}

impl ImageTask {
    /// Lowercase name used on the CLI and in error messages.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ImageTask::Generate => "generate",
            ImageTask::Edit => "edit",
        }
    }
}

/// Per-model default step count and guidance scale.
///
/// Resolved from checkpoint metadata only, never from weights, so a request
/// can be validated before any weight is loaded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageSamplingDefaults {
    pub steps: u32,
    pub guidance: f32,
}

impl ImageSamplingDefaults {
    /// Build defaults, rejecting a zero step count and a negative or
    /// non-finite guidance (the same checks as upstream's `__post_init__`).
    pub fn new(steps: u32, guidance: f32) -> Result<Self> {
        if steps < 1 {
            bail!("Default image steps must be a positive integer");
        }
        if !guidance.is_finite() || guidance < 0.0 {
            bail!("Default image guidance must be finite and non-negative");
        }
        Ok(Self { steps, guidance })
    }
}

/// One image generation or edit request.
///
/// `steps` and `guidance` are `None` when the caller did not choose a value;
/// the model's [`ImageSamplingDefaults`] fill them in. `n` outputs are
/// produced by running the model `n` times, image `i` with seed `seed + i`.
/// A family may instead batch from one noise tensor and report `seed` for
/// every image (FLUX.2 generation; see `docs/image-generation.md`, "Batched seeds").
#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub task: ImageTask,
    /// Reference images for [`ImageTask::Edit`]; empty for generation.
    pub reference_images: Vec<DynamicImage>,
    pub width: u32,
    pub height: u32,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub seed: u64,
    /// Number of images, `1..=10`.
    pub n: u32,
    /// Model-specific knobs (`--gen-kwargs`), passed through untouched.
    pub extra: Map<String, Value>,
}

/// Largest `n` a request may ask for.
pub const MAX_IMAGES_PER_REQUEST: u32 = 10;

/// Default output size for [`ImageTask::Generate`].
pub const DEFAULT_IMAGE_SIZE: (u32, u32) = (512, 512);

/// Shared allocation ceiling: at most 16 megapixels per output image.
pub const MAX_IMAGE_PIXELS: u64 = 4096 * 4096;
/// Bound explicit diffusion work before allocating a schedule or entering a worker.
pub const MAX_IMAGE_STEPS: u32 = 1000;

impl ImageRequest {
    /// A text-to-image request with the default size and one image.
    #[must_use]
    pub fn generate(prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            task: ImageTask::Generate,
            reference_images: Vec::new(),
            width: DEFAULT_IMAGE_SIZE.0,
            height: DEFAULT_IMAGE_SIZE.1,
            steps: None,
            guidance: None,
            seed,
            n: 1,
            extra: Map::new(),
        }
    }

    /// Steps to run: the caller's choice or `defaults.steps`.
    #[must_use]
    pub fn resolve_steps(&self, defaults: ImageSamplingDefaults) -> u32 {
        self.steps.unwrap_or(defaults.steps)
    }

    /// Guidance scale: the caller's choice or `defaults.guidance`.
    #[must_use]
    pub fn resolve_guidance(&self, defaults: ImageSamplingDefaults) -> f32 {
        self.guidance.unwrap_or(defaults.guidance)
    }

    /// Seed of output `index` (0-based): `seed + index`, as upstream does.
    #[must_use]
    pub fn seed_for(&self, index: u32) -> u64 {
        self.seed.wrapping_add(u64::from(index))
    }

    /// Check the fields that do not depend on the model.
    ///
    /// `grid` is the family's size multiple (16 for most families); width and
    /// height must be positive multiples of it.
    pub fn validate(&self, grid: u32) -> Result<()> {
        if self.prompt.trim().is_empty() {
            bail!("prompt must not be empty");
        }
        if self.n < 1 || self.n > MAX_IMAGES_PER_REQUEST {
            bail!(
                "n must be between 1 and {MAX_IMAGES_PER_REQUEST}, got {}",
                self.n
            );
        }
        if self.width == 0 || self.height == 0 {
            bail!(
                "width and height must be positive, got {}x{}",
                self.width,
                self.height
            );
        }
        if grid > 1 && !(self.width.is_multiple_of(grid) && self.height.is_multiple_of(grid)) {
            bail!(
                "width and height must be multiples of {grid}, got {}x{}",
                self.width,
                self.height
            );
        }
        if u64::from(self.width) * u64::from(self.height) > MAX_IMAGE_PIXELS {
            bail!("image size must not exceed {MAX_IMAGE_PIXELS} pixels");
        }
        if self
            .steps
            .is_some_and(|steps| steps == 0 || steps > MAX_IMAGE_STEPS)
        {
            bail!("steps must be between 1 and {MAX_IMAGE_STEPS}");
        }
        if let Some(g) = self.guidance
            && (!g.is_finite() || g < 0.0)
        {
            bail!("guidance must be finite and non-negative, got {g}");
        }
        match self.task {
            ImageTask::Edit if self.reference_images.is_empty() => {
                bail!("--task edit requires at least one reference image")
            }
            ImageTask::Generate if !self.reference_images.is_empty() => {
                bail!("reference images are only accepted with --task edit")
            }
            _ => {}
        }
        Ok(())
    }
}

/// One generated image: tightly packed HWC `uint8` pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedImage {
    /// `height * width * channels` bytes, row-major, channels interleaved.
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// 3 (RGB) or 4 (RGBA).
    pub channels: u8,
    pub seed: u64,
    pub revised_prompt: Option<String>,
}

impl GeneratedImage {
    /// Number of bytes `width * height * channels` that `pixels` must hold.
    #[must_use]
    pub fn expected_len(&self) -> usize {
        self.width as usize * self.height as usize * usize::from(self.channels)
    }
}

/// An error a model raises for the request's content (an option out of range,
/// the wrong number of reference images), as opposed to a failure while
/// running it. The server answers `400` for it instead of `500`; anyhow
/// context added on top keeps it detectable with `downcast_ref`.
///
/// Used by: ZImage.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct InvalidImageRequest(pub String);

/// The default [`ImageGenerationModel::default_edit_size`]: the first
/// reference's own size.
#[must_use]
pub fn reference_edit_size(width: u32, height: u32) -> (u32, u32) {
    (width, height)
}

/// What a family implements; see `docs/image-generation.md`.
///
/// Models are built and driven on one thread (MLX work is thread-affine), so
/// the trait carries no `Send` bound; the server constructs the model on its
/// worker thread.
pub trait ImageGenerationModel {
    /// Probe family name (`flux2`, `z_image`, ...).
    fn family(&self) -> &'static str;

    /// Whether this family implements `task`.
    fn supports(&self, task: ImageTask) -> bool;

    /// The checkpoint variant, for a family whose checkpoints differ in the
    /// tasks they implement; error messages name it. `None` by default.
    ///
    /// Used by: [`crate::imagegen::run::validate_request`] and the server's
    /// image routes; implemented by Mage-Flow.
    fn variant(&self) -> Option<&'static str> {
        None
    }

    /// Default steps and guidance for `task`.
    fn defaults(&self, task: ImageTask) -> ImageSamplingDefaults;

    /// Size multiple that width and height must satisfy.
    fn size_multiple(&self) -> u32 {
        16
    }

    /// Family-specific request checks (size bounds, `extra` options) that
    /// run before any GPU work. An error here is the caller's fault: the
    /// server answers `400` with its message. The default accepts everything.
    ///
    /// Used by: [`crate::imagegen::run::validate_request`]; implemented by
    /// FLUX.2.
    fn check_request(&self, _req: &ImageRequest) -> Result<()> {
        Ok(())
    }

    /// Output size an edit uses when the request names none, given the first
    /// reference image's width and height. The default is the reference's own
    /// size. A plain function pointer, so the server can keep it in the worker
    /// snapshot.
    ///
    /// Used by: `commands::generate_image::run_generate_image_with` and
    /// [`crate::server::image_worker`] (the `/v1/images/edits` route);
    /// implemented by FLUX.2.
    fn default_edit_size(&self) -> fn(u32, u32) -> (u32, u32) {
        reference_edit_size
    }

    /// Run `req` and return `req.n` images; image `i` uses seed `req.seed + i`.
    /// A family may batch from one noise tensor and report `req.seed` for every
    /// image (FLUX.2 generation; see `docs/image-generation.md`, "Batched seeds").
    fn generate(&self, req: &ImageRequest) -> Result<Vec<GeneratedImage>>;
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
