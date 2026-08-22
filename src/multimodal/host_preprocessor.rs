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

//! Host-side multimodal preprocessing for compiler backends.
//!
//! The producer in this module may use MLX to run the existing vision tower,
//! projector, and embedding lookup. Its output is the fully-owned
//! [`PreparedPrefill`] contract from `mlxcel-core`, so no MLX value crosses the
//! session/worker boundary.

#[cfg(feature = "xla-iree")]
use std::cell::RefCell;
use std::fmt;
use std::path::Path;

use image::DynamicImage;
use mlxcel_core::layers::UnifiedEmbedding;
use mlxcel_core::session::{
    OwnedTensor, PreparedPrefill, PreparedPrefillError, PreparedTensorDType,
};
use thiserror::Error;

use crate::vision::connectors::MultiModalConnector;
use crate::vision::encoders::VisionEncoder;
use crate::vision::merge::{InputEmbeddings, merge_llava};
use crate::vision::processors::ImageProcessor;

use super::vlm_prompt::{ImageTokenBlockError, ImageTokenBlockInfo, apply_image_token_blocks};

#[path = "host_preprocessor_export.rs"]
mod export;
#[cfg(feature = "xla-iree")]
use super::qwen_vl::insert_qwen_vl_image_tokens;
#[cfg(feature = "xla-iree")]
#[path = "molmo2_xla_preprocessor.rs"]
mod molmo2_xla;
use export::export_mlx_tensor;
use export::{
    build_prepared_prefill, export_llava_prefill, usize_to_i32, validate_embedding_shape,
    validate_processor_shape, validate_projected_shape, validate_sequence_capacity,
};
// Only the `xla-iree` Qwen2-VL prefill path calls this, so importing it
// unconditionally leaves it unused in a default-feature build and `-D warnings`
// turns that into a hard error. The gate has to match the single call site.
#[cfg(any(feature = "xla-iree", test))]
use export::export_qwen2_vl_prefill;
#[cfg(feature = "xla-iree")]
pub use molmo2_xla::Molmo2IreeHostPreprocessor;

/// Vision implementation selected for OpenXLA multimodal preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XlaVisionBackend {
    /// MLX executes the vision tower and multimodal projector.
    Host,
    /// IREE executes the vision tower and multimodal projector.
    Iree,
}

impl XlaVisionBackend {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Iree => "iree",
        }
    }
}

impl fmt::Display for XlaVisionBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XlaVisionBackendPolicy {
    Auto,
    Host,
    Iree,
}

impl XlaVisionBackendPolicy {
    fn from_value(value: Option<&str>) -> Result<Self, HostPreprocessorError> {
        match value.map(str::trim).filter(|value| !value.is_empty()) {
            None | Some("auto") => Ok(Self::Auto),
            Some("host") => Ok(Self::Host),
            Some("iree") => Ok(Self::Iree),
            Some(other) => Err(HostPreprocessorError::InvalidConfig(format!(
                "MLXCEL_XLA_VISION_BACKEND must be auto, host, or iree; got {other:?}"
            ))),
        }
    }

    fn from_env() -> Result<Self, HostPreprocessorError> {
        let value = std::env::var("MLXCEL_XLA_VISION_BACKEND").ok();
        Self::from_value(value.as_deref())
    }
}

/// A host preprocessor consumes tokenized text plus already-decoded images.
///
/// URL fetching, data-URI limits, image decoding, and other request security
/// policy remain at the server boundary. Audio and video are intentionally not
/// admitted by this first image-only contract.
pub trait HostMultimodalPreprocessor {
    /// Runtime selected for the vision tower and projector.
    ///
    /// This is deliberately queryable at the common CLI/server boundary so an
    /// automatic host fallback cannot be mistaken for native IREE execution.
    fn backend(&self) -> XlaVisionBackend {
        XlaVisionBackend::Host
    }

    /// Prepare an owned prefill payload for an engine worker.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, tensor-export, or family error before any
    /// compiler runtime is invoked.
    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError>;
}

/// Load the image preprocessor supported by the OpenXLA host-first path.
///
/// Three outcomes are possible, one per axis this loader decides on:
///
/// - `Ok(None)` is the conservative result for text-only checkpoints and VLM
///   families whose processor/position contract has not been qualified for XLA
///   yet.
/// - `Ok(Some(preprocessor))` is returned for a qualified family that can run
///   its vision path in this build. LLaVA reaches this even without the
///   `xla-iree` feature, because it has a complete MLX host vision tower and
///   projector to fall back to; only an explicit
///   `MLXCEL_XLA_VISION_BACKEND=iree` makes feature absence fatal there.
/// - `Err(..)` is returned for a qualified family this build cannot serve
///   images for. That covers missing or malformed processor/projector weights
///   on an identified LLaVA checkpoint, and Qwen2-VL in a build without
///   `xla-iree`.
///
/// The Qwen2-VL asymmetry with LLaVA is deliberate, not an oversight. Qwen2-VL
/// has no MLX vision fallback, which is the same reason
/// `MLXCEL_XLA_VISION_BACKEND=host` is rejected for it below, so without
/// `xla-iree` no code path can embed its images. `Ok(None)` there would be
/// indistinguishable from a text-only checkpoint and would start a session that
/// silently ignores every image. Failing takes away nothing that worked:
/// `xla-iree` also enables `mlxcel-xla/iree`, and without that the session's
/// `prefill` and `decode_step` return `mlxcel_xla::NOT_WIRED`, so an
/// `xla-backend`-only build cannot generate from any checkpoint at all. Both
/// callers (`backend::xla::create_session` and the server's image-preprocess
/// stage) turn this error into a startup failure, so the message carries the
/// rebuild instruction the operator needs.
///
/// # Errors
///
/// Returns a typed configuration or weight-loading error for a qualified
/// checkpoint that cannot construct a complete image path in this build.
pub fn load_xla_image_preprocessor(
    model_path: &Path,
) -> Result<Option<Box<dyn HostMultimodalPreprocessor>>, HostPreprocessorError> {
    let model_type = crate::models::get_model_type(model_path).map_err(|error| {
        HostPreprocessorError::InvalidConfig(format!(
            "failed to identify model family from {}: {error}",
            model_path.display()
        ))
    })?;
    if model_type == crate::models::ModelType::MuseGlimmerVLM {
        return Err(HostPreprocessorError::InvalidConfig(
            "Muse Glimmer VLM does not support OpenXLA image execution yet; use the MLX backend"
                .to_string(),
        ));
    }
    if model_type != crate::models::ModelType::LlavaVLM
        && model_type != crate::models::ModelType::Molmo2VLM
        && model_type != crate::models::ModelType::Qwen2VL
    {
        return Ok(None);
    }

    let policy = XlaVisionBackendPolicy::from_env()?;
    match model_type {
        crate::models::ModelType::Qwen2VL => {
            if policy == XlaVisionBackendPolicy::Host {
                return Err(HostPreprocessorError::InvalidConfig(
                    "Qwen2-VL XLA vision has no MLX fallback; MLXCEL_XLA_VISION_BACKEND=host is unsupported"
                        .to_string(),
                ));
            }
            #[cfg(feature = "xla-iree")]
            {
                let device = std::env::var("MLXCEL_XLA_DEVICE")
                    .unwrap_or_else(|_| mlxcel_xla::default_device().to_string());
                let preprocessor = Qwen2VlIreeHostPreprocessor::load(model_path, &device)?;
                tracing::info!(
                    vision_backend = "iree",
                    vision_device = %device,
                    family = "qwen2_vl",
                    "OpenXLA multimodal vision backend selected"
                );
                Ok(Some(Box::new(preprocessor)))
            }
            // No MLX fallback exists for this family, so an image-capable session
            // cannot be built here. Report it as a build-configuration error with
            // the remedy attached: both callers surface this string verbatim, so
            // the rebuild instruction is the only actionable part the operator
            // gets.
            #[cfg(not(feature = "xla-iree"))]
            {
                Err(HostPreprocessorError::InvalidConfig(
                    "Qwen2-VL XLA image execution requires the xla-iree feature; rebuild mlxcel \
                     with `--features xla-iree` (this family has no MLX vision fallback)"
                        .to_string(),
                ))
            }
        }
        crate::models::ModelType::LlavaVLM => load_llava_image_preprocessor(model_path, policy),
        crate::models::ModelType::Molmo2VLM => {
            #[cfg(feature = "xla-iree")]
            {
                if policy == XlaVisionBackendPolicy::Host {
                    return Ok(None);
                }
                let device = std::env::var("MLXCEL_XLA_DEVICE")
                    .unwrap_or_else(|_| mlxcel_xla::default_device().to_string());
                let preprocessor = Molmo2IreeHostPreprocessor::load(model_path, &device)?;
                tracing::info!(
                    vision_backend = "iree",
                    vision_device = %device,
                    "OpenXLA Molmo2 image preprocessing stage ready"
                );
                Ok(Some(Box::new(preprocessor)))
            }
            #[cfg(not(feature = "xla-iree"))]
            {
                if policy == XlaVisionBackendPolicy::Iree {
                    return Err(HostPreprocessorError::InvalidConfig(
                        "MLXCEL_XLA_VISION_BACKEND=iree requires the xla-iree feature".to_string(),
                    ));
                }
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// Worst-case logical prompt length one image contributes, from config alone.
///
/// The static StableHLO context shape is fixed when the engine is built, which
/// on the server path happens before any preprocessor exists, so this answers
/// the question from `config.json` and `preprocessor_config.json` without
/// loading weights, a processor, or a vision tower.
///
/// `None` means "not derived for this checkpoint", not "no images": a family
/// without a formula here, or a config this cannot read, falls through without
/// a capacity guard rather than inventing a number.
///
/// # Molmo2
///
/// The prompt carries one pooled token per pooled patch, plus one column token
/// per pooled row, for a low-resolution crop and a high-resolution tiling:
///
/// ```text
/// tokens = lo_h * (lo_w + 1) + hi_h * (hi_w + 1) + 4
/// ```
///
/// The low-resolution crop is always one tile. The high-resolution tiling is
/// whichever `(rows, cols)` with `rows * cols <= max_crops` the processor picks
/// for the image, so the worst case is the admissible tiling that maximizes the
/// token count. That is the tallest one, `(max_crops, 1)`, because every extra
/// row also adds a column token. On the pinned 4B checkpoint (`max_crops = 8`,
/// 378px crops, 14px patches, 2x2 pooling) a square image expands to 424 tokens
/// while a tall one reaches 1834, so sizing on the observed square case would
/// still reject ordinary photographs.
#[must_use]
pub fn xla_image_context_floor(model_path: &Path) -> Option<usize> {
    match crate::models::get_model_type(model_path) {
        Ok(crate::models::ModelType::Molmo2VLM) => molmo2_image_context_floor(model_path),
        Ok(crate::models::ModelType::LlavaVLM) => llava_image_context_floor(model_path),
        Ok(crate::models::ModelType::Qwen2VL) => qwen2_vl_image_context_floor(model_path),
        // A family without a formula here reports `None` rather than being
        // guarded with another family's shape.
        _ => None,
    }
}

/// Read `config.json` without loading anything else.
fn read_config_json(model_path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(model_path.join("config.json")).ok()?).ok()
}

/// Worst-case LLaVA image expansion.
///
/// LLaVA's block carries no begin- or end-of-image markers, no separator and no
/// per-row token (`llava_token_block_info` sets `use_boi_eoi: false` with empty
/// prefix and suffix lists), so one image contributes exactly the projector's
/// token count and there is no shape to maximize over.
///
/// That count is what `load_llava_host_preprocessor` computes: an explicit
/// `mm_tokens_per_image` when the checkpoint states one, otherwise
/// `(image_size / patch_size)^2` from the vision config. This mirrors that
/// expression rather than the upstream HF behavior, because the guard has to
/// predict what this codebase's preprocessor will actually emit. In particular
/// no anyres or multi-patch grid is applied here, since this loader applies
/// none either.
fn llava_image_context_floor(model_path: &Path) -> Option<usize> {
    let config = read_config_json(model_path)?;
    if let Some(explicit) = config.get("mm_tokens_per_image").and_then(|v| v.as_u64()) {
        return usize::try_from(explicit).ok().filter(|count| *count > 0);
    }
    let vision = config.get("vision_config")?;
    let image_size = usize::try_from(vision.get("image_size")?.as_u64()?).ok()?;
    let patch_size = usize::try_from(vision.get("patch_size")?.as_u64()?).ok()?;
    if patch_size == 0 {
        return None;
    }
    let per_side = image_size / patch_size;
    per_side.checked_mul(per_side).filter(|count| *count > 0)
}

/// Worst-case Qwen2-VL image expansion.
///
/// Unlike Molmo2 and LLaVA the bound does not come from a fixed grid. The
/// processor's `smart_resize` rounds both edges to `patch_size *
/// spatial_merge_size` and then caps the area, and
/// `insert_qwen_vl_image_tokens` emits `(h / merge) * (w / merge)` tokens for a
/// single-frame image, so the count is `pixels / (patch_size * merge)^2` and
/// its maximum is that quotient at the pixel cap.
///
/// The cap is deliberately read from
/// [`crate::vision::processors::qwen2_vl::DEFAULT_MAX_PIXELS`] rather than from
/// `preprocessor_config.json`. `qwen_vl_processor` builds the processor with
/// `Qwen2VLProcessor::new`, which never consults that file, so a `max_pixels`
/// key in the checkpoint does not affect what this path admits and reading one
/// would produce a floor the runtime does not honor.
fn qwen2_vl_image_context_floor(model_path: &Path) -> Option<usize> {
    let config = read_config_json(model_path)?;
    let vision = config.get("vision_config")?;
    let patch_size = usize::try_from(vision.get("patch_size")?.as_u64()?).ok()?;
    let merge_size = usize::try_from(vision.get("spatial_merge_size")?.as_u64()?).ok()?;
    if patch_size == 0 || merge_size == 0 {
        return None;
    }
    crate::vision::processors::qwen2_vl::max_image_tokens(
        patch_size,
        merge_size,
        crate::vision::processors::qwen2_vl::DEFAULT_MAX_PIXELS,
    )
    .filter(|count| *count > 0)
}

/// Name of the variable that selects the static OpenXLA context shape.
///
/// Duplicated from `mlxcel_xla` so this guard, and its tests, still compile in
/// a build without the `xla-backend` crate. The two are kept in step by
/// `the_capacity_variable_name_matches_the_xla_crate`.
pub const CONTEXT_CAPACITY_ENV: &str = "MLXCEL_XLA_CONTEXT_CAPACITY";

/// Reject a graph too small to ever admit one image from this checkpoint.
///
/// Returns `Ok(())` when no floor is derived, when the graph already fits, or
/// when the operator pinned the capacity themselves. Pinning is treated as a
/// decision, not a mistake: text-only serving from a VLM checkpoint is a real
/// workload, and the capacity is also what every decode step attends over, so
/// a larger graph is a throughput cost the operator may be declining on
/// purpose. What this stops is the silent case, where an unset default admits
/// no image at all and every image request runs its vision tower and only then
/// fails at admission.
///
/// # Errors
///
/// Returns a message naming the derived requirement and the variable to set.
pub fn ensure_xla_image_context_capacity(
    model_path: &Path,
    context_capacity: usize,
    operator_pinned: bool,
) -> Result<(), HostPreprocessorError> {
    if operator_pinned {
        return Ok(());
    }
    let Some(floor) = xla_image_context_floor(model_path) else {
        return Ok(());
    };
    if context_capacity >= floor {
        return Ok(());
    }
    Err(HostPreprocessorError::InvalidConfig(format!(
        "this checkpoint can expand a single image into up to {floor} tokens, and the default \
         OpenXLA context capacity of {context_capacity} admits none of them. Set {env} to the \
         largest image expansion you intend to serve: {floor} accepts any image, and a smaller \
         value accepts every image that fits it while rejecting the rest at admission, after \
         their vision tower has already run. The capacity is also the sequence length every \
         decode step attends over, so a larger graph costs throughput on every request, \
         including text-only ones. To keep this graph exactly as it is, set \
         {env}={context_capacity} explicitly.",
        env = CONTEXT_CAPACITY_ENV,
    )))
}

/// Worst-case Molmo2 image expansion, read from `preprocessor_config.json`.
fn molmo2_image_context_floor(model_path: &Path) -> Option<usize> {
    let raw = std::fs::read_to_string(model_path.join("preprocessor_config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&raw).ok()?;

    let max_crops = usize::try_from(config.get("max_crops")?.as_u64()?).ok()?;
    let patch_size = usize::try_from(config.get("patch_size")?.as_u64()?).ok()?;
    let size = config.get("size")?;
    let crop_height = usize::try_from(size.get("height")?.as_u64()?).ok()?;
    let crop_width = usize::try_from(size.get("width")?.as_u64()?).ok()?;
    let pooling = config.get("pooling_size")?.as_array()?;
    let pool_h = usize::try_from(pooling.first()?.as_u64()?).ok()?;
    let pool_w = usize::try_from(pooling.get(1)?.as_u64()?).ok()?;
    if max_crops == 0 || patch_size == 0 || pool_h == 0 || pool_w == 0 {
        return None;
    }

    let patch_rows = crop_height.checked_div(patch_size)?;
    let patch_cols = crop_width.checked_div(patch_size)?;
    let pooled = |patches: usize, pool: usize| patches.div_ceil(pool);

    let lo_h = pooled(patch_rows, pool_h);
    let lo_w = pooled(patch_cols, pool_w);
    let low_res = lo_h.checked_mul(lo_w.checked_add(1)?)?;

    // Every admissible tiling, because the maximum is not always the squarest
    // or the largest-area one: a column token per row makes tall tilings cost
    // more than wide ones of the same area.
    let mut high_res = 0usize;
    for rows in 1..=max_crops {
        for cols in 1..=max_crops {
            if rows.checked_mul(cols)? > max_crops {
                continue;
            }
            let hi_h = pooled(patch_rows.checked_mul(rows)?, pool_h);
            let hi_w = pooled(patch_cols.checked_mul(cols)?, pool_w);
            high_res = high_res.max(hi_h.checked_mul(hi_w.checked_add(1)?)?);
        }
    }

    // The four framing tokens are the low-res start/end and high-res start/end.
    low_res.checked_add(high_res)?.checked_add(4)
}

fn load_llava_host_preprocessor_boxed(
    model_path: &Path,
) -> Result<Option<Box<dyn HostMultimodalPreprocessor>>, HostPreprocessorError> {
    match LlavaHostPreprocessor::load(model_path) {
        Ok(preprocessor) => Ok(Some(Box::new(preprocessor))),
        // A LLaVA-shaped checkpoint can still carry an unqualified text or
        // vision backend. Keep capability false for that combination.
        Err(HostPreprocessorError::FamilyMismatch { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

fn load_llava_image_preprocessor(
    model_path: &Path,
    policy: XlaVisionBackendPolicy,
) -> Result<Option<Box<dyn HostMultimodalPreprocessor>>, HostPreprocessorError> {
    #[cfg(feature = "xla-iree")]
    if policy != XlaVisionBackendPolicy::Host {
        let device = std::env::var("MLXCEL_XLA_DEVICE")
            .unwrap_or_else(|_| mlxcel_xla::default_device().to_string());
        match LlavaIreeHostPreprocessor::load(model_path, &device) {
            Ok(preprocessor) => {
                tracing::info!(
                    vision_backend = "iree",
                    vision_device = %device,
                    vision_backend_policy = ?policy,
                    "OpenXLA multimodal vision backend selected"
                );
                return Ok(Some(Box::new(preprocessor)));
            }
            Err(error) if policy == XlaVisionBackendPolicy::Auto => {
                tracing::warn!(
                    vision_backend = "host",
                    vision_device = %device,
                    vision_backend_policy = "auto",
                    fallback_reason = %error,
                    "IREE vision backend unavailable; using observable host fallback"
                );
            }
            Err(HostPreprocessorError::FamilyMismatch { .. }) => return Ok(None),
            Err(error) => return Err(error),
        }
    }

    #[cfg(not(feature = "xla-iree"))]
    if policy == XlaVisionBackendPolicy::Iree {
        return Err(HostPreprocessorError::InvalidConfig(
            "MLXCEL_XLA_VISION_BACKEND=iree requires the xla-iree feature".to_string(),
        ));
    }

    let loaded = load_llava_host_preprocessor_boxed(model_path)?;
    if loaded.is_some() {
        tracing::info!(
            vision_backend = "host",
            vision_backend_policy = ?policy,
            "OpenXLA multimodal vision backend selected"
        );
    }
    Ok(loaded)
}

/// LLaVA host preprocessor retaining only the components needed before XLA.
pub struct LlavaHostPreprocessor {
    processor: Box<dyn ImageProcessor>,
    encoder: Box<dyn VisionEncoder>,
    connector: Box<dyn MultiModalConnector>,
    /// Canonical checkpoint embedding lookup. This owns only
    /// `model.embed_tokens.{weight,scales,biases}` handles loaded through the
    /// filtered loader; no decoder layer or LM head is constructed or retained.
    text_embeddings: UnifiedEmbedding,
    image_token_id: i32,
    tokens_per_image: usize,
    hidden_size: usize,
    image_size: usize,
    max_sequence_len: usize,
}

/// LLaVA producer that retains host image processing/text lookup while moving
/// the complete vision tower and projector to a resident IREE module.
#[cfg(feature = "xla-iree")]
pub struct LlavaIreeHostPreprocessor {
    processor: Box<dyn ImageProcessor>,
    text_embeddings: UnifiedEmbedding,
    projector: RefCell<mlxcel_xla::IreeVisionProjector>,
    image_token_id: i32,
    tokens_per_image: usize,
    hidden_size: usize,
    image_size: usize,
    max_sequence_len: usize,
    device: String,
}

/// Qwen2-VL producer with shared host smart-resize/patch extraction, a
/// filtered text embedding table, and the complete vision tower in IREE.
#[cfg(feature = "xla-iree")]
pub struct Qwen2VlIreeHostPreprocessor {
    processor: crate::vision::processors::qwen2_vl::Qwen2VLProcessor,
    text_embeddings: UnifiedEmbedding,
    projector: RefCell<mlxcel_xla::IreeQwen2VlProjector>,
    image_token_id: i32,
    video_token_id: i32,
    vision_start_token_id: i32,
    spatial_merge_size: usize,
    hidden_size: usize,
    max_sequence_len: usize,
    device: String,
}

#[cfg(feature = "xla-iree")]
impl Qwen2VlIreeHostPreprocessor {
    pub fn load(model_path: &Path, device: &str) -> Result<Self, HostPreprocessorError> {
        crate::loading::load_qwen2_vl_iree_host_preprocessor(model_path, device)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        processor: crate::vision::processors::qwen2_vl::Qwen2VLProcessor,
        text_embeddings: UnifiedEmbedding,
        projector: mlxcel_xla::IreeQwen2VlProjector,
        image_token_id: i32,
        video_token_id: i32,
        vision_start_token_id: i32,
        spatial_merge_size: usize,
        hidden_size: usize,
        max_sequence_len: usize,
        device: String,
    ) -> Result<Self, HostPreprocessorError> {
        if hidden_size == 0 || max_sequence_len == 0 || spatial_merge_size == 0 {
            return Err(HostPreprocessorError::InvalidConfig(
                "Qwen2-VL hidden size, sequence capacity, and spatial merge size must be positive"
                    .to_string(),
            ));
        }
        if projector.text_hidden() != hidden_size {
            return Err(HostPreprocessorError::InvalidConfig(format!(
                "Qwen2-VL IREE projector hidden size {} disagrees with text hidden size {hidden_size}",
                projector.text_hidden()
            )));
        }
        Ok(Self {
            processor,
            text_embeddings,
            projector: RefCell::new(projector),
            image_token_id,
            video_token_id,
            vision_start_token_id,
            spatial_merge_size,
            hidden_size,
            max_sequence_len,
            device,
        })
    }

    fn prepare_iree(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        if images.is_empty() {
            return Err(HostPreprocessorError::InvalidConfig(
                "Qwen2-VL image preprocessing requires at least one decoded image; text-only requests use ordinary XLA token prefill"
                    .to_string(),
            ));
        }
        let (patch_values, grids) = self.processor.preprocess_values_with_grid(images);
        let mut logical_tokens = token_ids.to_vec();
        let inserted = insert_qwen_vl_image_tokens(
            &mut logical_tokens,
            &grids,
            self.spatial_merge_size,
            self.vision_start_token_id,
            self.image_token_id,
        )
        .ok_or_else(|| {
            HostPreprocessorError::InvalidConfig(format!(
                "Qwen2-VL image placeholder count does not match {} decoded image(s), or the prompt is already expanded",
                images.len()
            ))
        })?;
        if inserted.image_blocks != images.len() {
            return Err(HostPreprocessorError::InvalidConfig(format!(
                "Qwen2-VL expanded {} image blocks for {} decoded images",
                inserted.image_blocks,
                images.len()
            )));
        }
        validate_sequence_capacity(logical_tokens.len(), self.max_sequence_len)?;
        let input_ids = mlxcel_core::from_slice_i32(
            &logical_tokens,
            &[1, usize_to_i32(logical_tokens.len(), "sequence length")?],
        );
        let text_embeddings = mlxcel_core::astype(
            &self.text_embeddings.forward(&input_ids),
            mlxcel_core::dtype::FLOAT32,
        );
        validate_embedding_shape(
            &mlxcel_core::array_shape(&text_embeddings),
            logical_tokens.len(),
            self.hidden_size,
            "Qwen2-VL text embedding table",
        )?;
        let mut projector = self.projector.try_borrow_mut().map_err(|_| {
            HostPreprocessorError::Iree(
                "concurrent/re-entrant Qwen2-VL IREE vision invocation is unsupported".to_string(),
            )
        })?;
        let projection = projector
            .project(&patch_values, &grids)
            .map_err(HostPreprocessorError::Iree)?;
        let expected_projected_tokens = usize::try_from(inserted.total_image_tokens)
            .map_err(|_| HostPreprocessorError::ShapeOverflow)?;
        if projection.shape != [expected_projected_tokens, self.hidden_size] {
            return Err(HostPreprocessorError::InvalidConfig(format!(
                "Qwen2-VL IREE projection shape {:?} disagrees with expanded image-token shape [{}, {}]",
                projection.shape, inserted.total_image_tokens, self.hidden_size
            )));
        }
        tracing::info!(
            vision_backend = "iree",
            vision_device = %self.device,
            image_count = images.len(),
            patch_upload_bytes = projection.metrics.patch_upload_bytes,
            metadata_upload_bytes = projection.metrics.metadata_upload_bytes,
            projected_transfer_bytes = projection.metrics.projected_transfer_bytes,
            iree_vision_seconds = projection.metrics.elapsed_seconds,
            packed_cu_seqlens = ?projection.packed_cu_seqlens,
            "Qwen2-VL OpenXLA vision projection completed"
        );
        let projected = mlxcel_core::from_slice_f32(
            &projection.values,
            &[
                usize_to_i32(projection.shape[0], "Qwen2-VL projected token count")?,
                usize_to_i32(self.hidden_size, "hidden size")?,
            ],
        );
        let merged = merge_llava(
            self.image_token_id,
            &projected,
            &text_embeddings,
            &input_ids,
        );
        export_qwen2_vl_prefill(
            logical_tokens,
            InputEmbeddings {
                inputs_embeds: mlxcel_core::astype(
                    &merged.inputs_embeds,
                    mlxcel_core::dtype::FLOAT32,
                ),
                attention_mask_4d: merged.attention_mask_4d,
            },
            &grids,
            self.image_token_id,
            self.video_token_id,
            self.spatial_merge_size,
            self.hidden_size,
        )
    }
}

#[cfg(feature = "xla-iree")]
impl HostMultimodalPreprocessor for Qwen2VlIreeHostPreprocessor {
    fn backend(&self) -> XlaVisionBackend {
        XlaVisionBackend::Iree
    }

    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        self.prepare_iree(token_ids, images)
    }
}

#[cfg(feature = "xla-iree")]
impl LlavaIreeHostPreprocessor {
    /// Load the host processor/text embedding table and resident IREE vision
    /// module from one immutable checkpoint directory.
    pub fn load(model_path: &Path, device: &str) -> Result<Self, HostPreprocessorError> {
        crate::loading::load_llava_iree_host_preprocessor(model_path, device)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        processor: Box<dyn ImageProcessor>,
        text_embeddings: UnifiedEmbedding,
        projector: mlxcel_xla::IreeVisionProjector,
        image_token_id: i32,
        tokens_per_image: usize,
        hidden_size: usize,
        image_size: usize,
        max_sequence_len: usize,
        device: String,
    ) -> Result<Self, HostPreprocessorError> {
        validate_llava_preprocessor_dimensions(
            tokens_per_image,
            hidden_size,
            image_size,
            max_sequence_len,
        )?;
        let expected_input = [1, 3, image_size, image_size];
        if projector.input_shape() != expected_input {
            return Err(HostPreprocessorError::InvalidConfig(format!(
                "IREE vision input shape {:?} does not match host processor shape {expected_input:?}",
                projector.input_shape()
            )));
        }
        let expected_output = [tokens_per_image, hidden_size];
        if projector.output_shape() != expected_output {
            return Err(HostPreprocessorError::InvalidConfig(format!(
                "IREE vision output shape {:?} does not match prepared-prefill shape {expected_output:?}",
                projector.output_shape()
            )));
        }
        Ok(Self {
            processor,
            text_embeddings,
            projector: RefCell::new(projector),
            image_token_id,
            tokens_per_image,
            hidden_size,
            image_size,
            max_sequence_len,
            device,
        })
    }

    fn token_block_info(&self) -> ImageTokenBlockInfo {
        llava_token_block_info(self.image_token_id, self.tokens_per_image)
    }

    fn prepare_iree(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        let mut logical_tokens = token_ids.to_vec();
        apply_image_token_blocks(&mut logical_tokens, self.token_block_info(), images.len())?;
        validate_sequence_capacity(logical_tokens.len(), self.max_sequence_len)?;

        let input_ids = mlxcel_core::from_slice_i32(
            &logical_tokens,
            &[1, usize_to_i32(logical_tokens.len(), "sequence length")?],
        );
        // The resident vision graph returns F32. Widen the checkpoint's
        // BF16/F16 embedding lookup before the merge; otherwise MLX selects
        // the narrower text dtype for the combined tensor and silently rounds
        // every native projected row before the later ownership-boundary cast.
        let text_embeddings = mlxcel_core::astype(
            &self.text_embeddings.forward(&input_ids),
            mlxcel_core::dtype::FLOAT32,
        );
        validate_embedding_shape(
            &mlxcel_core::array_shape(&text_embeddings),
            logical_tokens.len(),
            self.hidden_size,
            "text embedding table",
        )?;

        let merged = if images.is_empty() {
            InputEmbeddings {
                inputs_embeds: text_embeddings,
                attention_mask_4d: None,
            }
        } else {
            let pixels = self.processor.preprocess(images);
            validate_processor_shape(
                &mlxcel_core::array_shape(&pixels),
                images.len(),
                self.image_size,
            )?;
            let pixels = export_mlx_tensor(&pixels, "processor pixel_values")?;
            if pixels.dtype != PreparedTensorDType::Float32 {
                return Err(HostPreprocessorError::InvalidConfig(format!(
                    "IREE vision requires float32 processor output, got {:?}",
                    pixels.dtype
                )));
            }
            let pixel_values = pixels
                .bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|bytes| {
                    f32::from_ne_bytes(
                        bytes
                            .try_into()
                            .expect("chunks_exact yields one native f32"),
                    )
                })
                .collect::<Vec<_>>();
            let pixels_per_image = 3usize
                .checked_mul(self.image_size)
                .and_then(|count| count.checked_mul(self.image_size))
                .ok_or(HostPreprocessorError::ShapeOverflow)?;
            let projected_per_image = self
                .tokens_per_image
                .checked_mul(self.hidden_size)
                .ok_or(HostPreprocessorError::ShapeOverflow)?;
            let mut projected_values = Vec::with_capacity(
                images
                    .len()
                    .checked_mul(projected_per_image)
                    .ok_or(HostPreprocessorError::ShapeOverflow)?,
            );
            let mut elapsed_seconds = 0.0;
            let mut upload_bytes = 0usize;
            let mut transfer_bytes = 0usize;
            let mut projector = self.projector.try_borrow_mut().map_err(|_| {
                HostPreprocessorError::Iree(
                    "concurrent/re-entrant IREE vision invocation is unsupported".to_string(),
                )
            })?;
            for image_pixels in pixel_values.chunks_exact(pixels_per_image) {
                let projection = projector
                    .project(image_pixels)
                    .map_err(HostPreprocessorError::Iree)?;
                if projection.shape != [self.tokens_per_image, self.hidden_size] {
                    return Err(HostPreprocessorError::ProjectedShape {
                        actual: projection
                            .shape
                            .into_iter()
                            .map(|dimension| i32::try_from(dimension).unwrap_or(i32::MAX))
                            .collect(),
                        image_count: 1,
                        tokens_per_image: self.tokens_per_image,
                        hidden_size: self.hidden_size,
                    });
                }
                elapsed_seconds += projection.metrics.elapsed_seconds;
                upload_bytes = upload_bytes
                    .checked_add(projection.metrics.pixel_upload_bytes)
                    .ok_or(HostPreprocessorError::ShapeOverflow)?;
                transfer_bytes = transfer_bytes
                    .checked_add(projection.metrics.projected_transfer_bytes)
                    .ok_or(HostPreprocessorError::ShapeOverflow)?;
                projected_values.extend(projection.values);
            }
            if projected_values.len()
                != images
                    .len()
                    .checked_mul(projected_per_image)
                    .ok_or(HostPreprocessorError::ShapeOverflow)?
            {
                return Err(HostPreprocessorError::Iree(
                    "IREE vision produced an incomplete image batch".to_string(),
                ));
            }
            tracing::info!(
                vision_backend = "iree",
                vision_device = %self.device,
                image_count = images.len(),
                pixel_upload_bytes = upload_bytes,
                projected_transfer_bytes = transfer_bytes,
                iree_vision_seconds = elapsed_seconds,
                "OpenXLA multimodal vision projection completed"
            );
            let projected = mlxcel_core::from_slice_f32(
                &projected_values,
                &[
                    usize_to_i32(images.len(), "image count")?,
                    usize_to_i32(self.tokens_per_image, "tokens per image")?,
                    usize_to_i32(self.hidden_size, "hidden size")?,
                ],
            );
            validate_projected_shape(
                &mlxcel_core::array_shape(&projected),
                images.len(),
                self.tokens_per_image,
                self.hidden_size,
            )?;
            merge_llava(
                self.image_token_id,
                &projected,
                &text_embeddings,
                &input_ids,
            )
        };

        let merged = InputEmbeddings {
            inputs_embeds: mlxcel_core::astype(&merged.inputs_embeds, mlxcel_core::dtype::FLOAT32),
            attention_mask_4d: merged.attention_mask_4d,
        };
        export_llava_prefill(
            logical_tokens,
            merged,
            self.image_token_id,
            images.len(),
            self.tokens_per_image,
            self.hidden_size,
        )
    }
}

#[cfg(feature = "xla-iree")]
impl HostMultimodalPreprocessor for LlavaIreeHostPreprocessor {
    fn backend(&self) -> XlaVisionBackend {
        XlaVisionBackend::Iree
    }

    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        self.prepare_iree(token_ids, images)
    }
}

/// Diagnostics-only capture from the exact host preprocessing path used by
/// OpenXLA image requests.
#[cfg(feature = "xla-diagnostics")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlavaHostReferenceCapture {
    pub prepared: PreparedPrefill,
    /// Canonical processor output in `[images, 3, height, width]` layout.
    pub pixel_values: Option<OwnedTensor>,
    /// Selected vision-tower output before the multimodal projector in
    /// `[images, image_tokens, vision_hidden]` layout.
    pub selected_vision_features: Option<OwnedTensor>,
    /// Encoder embedding output followed by each selected vision layer output.
    pub vision_hidden_states: Vec<OwnedTensor>,
    /// First encoder block's normalization, attention, and MLP sub-stages.
    pub vision_block0_states: Vec<OwnedTensor>,
    /// Projected vision features in `[images, image_tokens, hidden]` layout.
    pub projected_image_features: Option<OwnedTensor>,
}

impl LlavaHostPreprocessor {
    /// Load the LLaVA processor, vision tower, projector, and embedding lookup
    /// from a checkpoint without constructing a text decoder.
    ///
    /// The canonical filtered host loader and IREE both read the same immutable
    /// model directory. It explicitly bypasses process-global weight surgery,
    /// which IREE does not apply, so the host representation cannot select a
    /// different embedding revision or transformed value from the uploaded
    /// buffers. Changing either requires changing the shared model path.
    ///
    /// # Errors
    ///
    /// Returns an actionable typed error for incompatible families, malformed
    /// config, missing required weights, or an invalid embedding layout.
    pub fn load(model_path: &Path) -> Result<Self, HostPreprocessorError> {
        crate::loading::load_llava_host_preprocessor(model_path)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        processor: Box<dyn ImageProcessor>,
        encoder: Box<dyn VisionEncoder>,
        connector: Box<dyn MultiModalConnector>,
        text_embeddings: UnifiedEmbedding,
        image_token_id: i32,
        tokens_per_image: usize,
        hidden_size: usize,
        image_size: usize,
        max_sequence_len: usize,
    ) -> Result<Self, HostPreprocessorError> {
        validate_llava_preprocessor_dimensions(
            tokens_per_image,
            hidden_size,
            image_size,
            max_sequence_len,
        )?;
        Ok(Self {
            processor,
            encoder,
            connector,
            text_embeddings,
            image_token_id,
            tokens_per_image,
            hidden_size,
            image_size,
            max_sequence_len,
        })
    }

    fn token_block_info(&self) -> ImageTokenBlockInfo {
        llava_token_block_info(self.image_token_id, self.tokens_per_image)
    }

    fn prepare_internal(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
        capture_diagnostics: bool,
    ) -> Result<
        (
            PreparedPrefill,
            Option<OwnedTensor>,
            Option<OwnedTensor>,
            Option<OwnedTensor>,
            Vec<OwnedTensor>,
            Vec<OwnedTensor>,
        ),
        HostPreprocessorError,
    > {
        let mut logical_tokens = token_ids.to_vec();
        apply_image_token_blocks(&mut logical_tokens, self.token_block_info(), images.len())?;
        validate_sequence_capacity(logical_tokens.len(), self.max_sequence_len)?;

        let input_ids = mlxcel_core::from_slice_i32(
            &logical_tokens,
            &[1, usize_to_i32(logical_tokens.len(), "sequence length")?],
        );
        let text_embeddings = self.text_embeddings.forward(&input_ids);
        validate_embedding_shape(
            &mlxcel_core::array_shape(&text_embeddings),
            logical_tokens.len(),
            self.hidden_size,
            "text embedding table",
        )?;

        let mut pixel_values = None;
        let mut selected_vision_features = None;
        let mut projected_image_features = None;
        #[cfg(feature = "xla-diagnostics")]
        let mut vision_hidden_states = Vec::new();
        #[cfg(not(feature = "xla-diagnostics"))]
        let vision_hidden_states = Vec::new();
        #[cfg(feature = "xla-diagnostics")]
        let mut vision_block0_states = Vec::new();
        #[cfg(not(feature = "xla-diagnostics"))]
        let vision_block0_states = Vec::new();
        let merged = if images.is_empty() {
            InputEmbeddings {
                inputs_embeds: text_embeddings,
                attention_mask_4d: None,
            }
        } else {
            let pixels = self.processor.preprocess(images);
            validate_processor_shape(
                &mlxcel_core::array_shape(&pixels),
                images.len(),
                self.image_size,
            )?;
            if capture_diagnostics {
                pixel_values = Some(export_mlx_tensor(&pixels, "processor pixel_values")?);
            }

            // The standard processor emits channels-first f32. Normalize layout
            // for the shared CLIP/SigLIP encoder. Keep the qualified LLaVA
            // vision path in f32: layer-by-layer BF16 reduction differences
            // compound sharply in this 26-layer SigLIP tower even when the
            // checkpoint itself stores BF16 weights.
            let pixels = mlxcel_core::transpose_axes(&pixels, &[0, 2, 3, 1]);
            let pixels = mlxcel_core::astype(&pixels, mlxcel_core::dtype::FLOAT32);
            #[cfg(feature = "xla-diagnostics")]
            let encoded = if capture_diagnostics {
                let (encoded, hidden_states, block0_states) =
                    self.encoder.forward_with_hidden_state_diagnostics(&pixels);
                vision_hidden_states = hidden_states
                    .iter()
                    .map(|state| export_mlx_tensor(state, "vision hidden state"))
                    .collect::<Result<Vec<_>, _>>()?;
                vision_block0_states = block0_states
                    .iter()
                    .map(|state| export_mlx_tensor(state, "vision block 0 state"))
                    .collect::<Result<Vec<_>, _>>()?;
                encoded
            } else {
                self.encoder.forward(&pixels)
            };
            #[cfg(not(feature = "xla-diagnostics"))]
            let encoded = self.encoder.forward(&pixels);
            if capture_diagnostics {
                selected_vision_features = Some(export_mlx_tensor(
                    &encoded.hidden_states,
                    "selected vision features",
                )?);
            }
            let projected = self.connector.forward(&encoded.hidden_states);
            validate_projected_shape(
                &mlxcel_core::array_shape(&projected),
                images.len(),
                self.tokens_per_image,
                self.hidden_size,
            )?;
            if capture_diagnostics {
                projected_image_features =
                    Some(export_mlx_tensor(&projected, "projected image features")?);
            }

            merge_llava(
                self.image_token_id,
                &projected,
                &text_embeddings,
                &input_ids,
            )
        };

        // The qualified IREE embeddings entry is intentionally F32-only. MLX
        // checkpoints may store the language embedding table and vision tower
        // in BF16/F16, so widen the fully merged result exactly once at the
        // ownership boundary instead of making the runtime accept a dtype it
        // cannot execute.
        let merged = InputEmbeddings {
            inputs_embeds: mlxcel_core::astype(&merged.inputs_embeds, mlxcel_core::dtype::FLOAT32),
            attention_mask_4d: merged.attention_mask_4d,
        };

        let prepared = export_llava_prefill(
            logical_tokens,
            merged,
            self.image_token_id,
            images.len(),
            self.tokens_per_image,
            self.hidden_size,
        )?;
        Ok((
            prepared,
            pixel_values,
            selected_vision_features,
            projected_image_features,
            vision_hidden_states,
            vision_block0_states,
        ))
    }

    /// Capture processor/projector/prepared values without changing the
    /// production preprocessing sequence.
    #[cfg(feature = "xla-diagnostics")]
    pub fn prepare_with_reference_diagnostics(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<LlavaHostReferenceCapture, HostPreprocessorError> {
        let (
            prepared,
            pixel_values,
            selected_vision_features,
            projected_image_features,
            vision_hidden_states,
            vision_block0_states,
        ) = self.prepare_internal(token_ids, images, true)?;
        Ok(LlavaHostReferenceCapture {
            prepared,
            pixel_values,
            selected_vision_features,
            vision_hidden_states,
            vision_block0_states,
            projected_image_features,
        })
    }
}

impl HostMultimodalPreprocessor for LlavaHostPreprocessor {
    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        self.prepare_internal(token_ids, images, false)
            .map(|(prepared, _, _, _, _, _)| prepared)
    }
}

fn validate_llava_preprocessor_dimensions(
    tokens_per_image: usize,
    hidden_size: usize,
    image_size: usize,
    max_sequence_len: usize,
) -> Result<(), HostPreprocessorError> {
    if tokens_per_image == 0 {
        return Err(HostPreprocessorError::InvalidConfig(
            "LLaVA mm_tokens_per_image must be greater than zero".to_string(),
        ));
    }
    if hidden_size == 0 {
        return Err(HostPreprocessorError::InvalidConfig(
            "LLaVA text hidden_size must be greater than zero".to_string(),
        ));
    }
    if image_size == 0 {
        return Err(HostPreprocessorError::InvalidConfig(
            "LLaVA processor image_size must be greater than zero".to_string(),
        ));
    }
    if max_sequence_len == 0 {
        return Err(HostPreprocessorError::InvalidConfig(
            "LLaVA max sequence length must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn llava_token_block_info(image_token_id: i32, tokens_per_image: usize) -> ImageTokenBlockInfo {
    ImageTokenBlockInfo {
        use_boi_eoi: false,
        image_token_id,
        mm_tokens_per_image: tokens_per_image,
        boi_token_id: 0,
        eoi_token_id: 0,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    }
}

/// Deterministic checkpoint-free producer for engine and server contract tests.
#[derive(Debug, Clone)]
pub struct FakeHostMultimodalPreprocessor {
    pub image_token_id: i32,
    pub tokens_per_image: usize,
    pub hidden_size: usize,
    pub max_sequence_len: usize,
}

impl Default for FakeHostMultimodalPreprocessor {
    fn default() -> Self {
        Self {
            image_token_id: -200,
            tokens_per_image: 4,
            hidden_size: 8,
            max_sequence_len: 4096,
        }
    }
}

impl HostMultimodalPreprocessor for FakeHostMultimodalPreprocessor {
    fn prepare(
        &self,
        token_ids: &[i32],
        images: &[DynamicImage],
    ) -> Result<PreparedPrefill, HostPreprocessorError> {
        let mut logical_tokens = token_ids.to_vec();
        let info = ImageTokenBlockInfo {
            use_boi_eoi: false,
            image_token_id: self.image_token_id,
            mm_tokens_per_image: self.tokens_per_image,
            boi_token_id: 0,
            eoi_token_id: 0,
            has_bos: true,
            separator_token_id: None,
            suffix_tokens: Vec::new(),
            block_prefix_tokens: Vec::new(),
            block_suffix_tokens: Vec::new(),
        };
        apply_image_token_blocks(&mut logical_tokens, info, images.len())?;
        validate_sequence_capacity(logical_tokens.len(), self.max_sequence_len)?;

        let element_count = logical_tokens
            .len()
            .checked_mul(self.hidden_size)
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        let byte_count = element_count
            .checked_mul(PreparedTensorDType::Float32.size_bytes())
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        let mut bytes = Vec::with_capacity(byte_count);
        for (position, &token) in logical_tokens.iter().enumerate() {
            for lane in 0..self.hidden_size {
                // Stable across platforms and independent of image contents.
                let value = token as f32 * 0.001 + position as f32 * 0.01 + lane as f32;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let embeddings = OwnedTensor::new(
            bytes,
            PreparedTensorDType::Float32,
            vec![1, logical_tokens.len(), self.hidden_size],
        )?;
        let image_token_count = images
            .len()
            .checked_mul(self.tokens_per_image)
            .ok_or(HostPreprocessorError::ShapeOverflow)?;
        build_prepared_prefill(
            logical_tokens,
            embeddings,
            images.len(),
            image_token_count,
            "fake-llava",
        )
    }
}

/// Typed failures surfaced before an XLA runtime invocation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HostPreprocessorError {
    #[error(transparent)]
    Placeholder(#[from] ImageTokenBlockError),
    #[error("incompatible multimodal family: expected LLaVA, got {actual}")]
    FamilyMismatch { actual: String },
    #[error("invalid host-preprocessor config: {0}")]
    InvalidConfig(String),
    #[error("failed to load host-preprocessor weights: {0}")]
    WeightLoad(String),
    #[error("IREE vision backend failed: {0}")]
    Iree(String),
    #[error(
        "processor output shape {actual:?} does not match decoded RGB batch [{image_count}, 3, {image_size}, {image_size}]"
    )]
    ProcessorShape {
        actual: Vec<i32>,
        image_count: usize,
        image_size: usize,
    },
    #[error("{source_name} shape {actual:?} does not match [1, {sequence_len}, {hidden_size}]")]
    EmbeddingShape {
        source_name: &'static str,
        actual: Vec<i32>,
        sequence_len: usize,
        hidden_size: usize,
    },
    #[error(
        "projected image shape {actual:?} does not match [{image_count}, {tokens_per_image}, {hidden_size}]"
    )]
    ProjectedShape {
        actual: Vec<i32>,
        image_count: usize,
        tokens_per_image: usize,
        hidden_size: usize,
    },
    #[error("expanded sequence length {actual} exceeds model capacity {maximum}")]
    SequenceCapacity { actual: usize, maximum: usize },
    #[error(
        "expanded prompt contains {actual} image token(s), expected {expected} from decoded media"
    )]
    ExpandedLength { actual: usize, expected: usize },
    #[error("unsupported prepared tensor dtype code {0}")]
    UnsupportedDType(i32),
    #[error("failed to export evaluated contiguous {tensor} tensor: {message}")]
    TensorExport {
        tensor: &'static str,
        message: String,
    },
    #[error("{label} cannot be represented by the MLX i32 shape ABI: {value}")]
    DimensionOverflow { label: &'static str, value: usize },
    #[error("prepared tensor shape calculation overflowed")]
    ShapeOverflow,
    #[error(transparent)]
    Prepared(#[from] PreparedPrefillError),
}

#[cfg(test)]
#[path = "host_preprocessor_tests.rs"]
mod tests;
