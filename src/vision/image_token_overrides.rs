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

//! Operator-supplied image-token budgets, applied to a dynamic-resolution
//! vision processor's own pixel bounds before an image is resized.
//!
//! This is the mlxcel side of llama-server b10621's `--image-min-tokens` and
//! `--image-max-tokens` (issue #1451). Upstream stores them on
//! `clip_hparams::custom_image_min_tokens` / `custom_image_max_tokens` and
//! converts them into pixel bounds in
//! [`set_limit_image_tokens`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/mtmd/clip-model.h):
//!
//! ```text
//! patch_area       = patch_size^2 * n_merge^2
//! image_min_pixels = image_min_tokens * patch_area
//! image_max_pixels = image_max_tokens * patch_area
//! ```
//!
//! mlxcel's dynamic-resolution processors express exactly the same two bounds
//! as `min_pixels` / `max_pixels`, so the translation is the identity: the same
//! multiplication against the same patch area. That is why these two options
//! are honored rather than refused. `--mtmd-batch-max-tokens`, which names an
//! encode *batch* width rather than a per-image budget, has no such counterpart
//! and is refused in [`crate::cli::multimodal_compat_args`].
//!
//! # Why the applications counter exists
//!
//! Only the processors that resize dynamically have pixel bounds for a token
//! budget to move. A fixed-tile encoder (LLaVA-style square resize, SigLIP at a
//! fixed `image_size`, the SAM/CLIP channel-concat towers) crops or pads to a
//! geometry its weights were trained at, and emits the same token count for
//! every input; a minimum or maximum token budget has nothing to act on there.
//! Accepting the flag on such a checkpoint and encoding at the checkpoint's own
//! geometry is the silent-acceptance failure epic #1431 exists to remove.
//!
//! Rather than maintain a hand-written table of which architecture honors the
//! budget (which goes stale the moment a family is ported, the lesson
//! [`crate::models::rope_overrides`] already paid for), every processor that
//! consumes the override increments [`applications`]. Server startup compares
//! that count against zero after the model is loaded and refuses to serve when
//! a budget was requested and never applied. A new family that routes its
//! resize through a bound-consuming processor starts being accepted with no
//! list to update, and one that does not is named in the error.
//!
//! Used by: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Qwen3-VL-MoE,
//! Qwen3-Omni, GLM-4V, GLM-4V-MoE, GLM-OCR, ColQwen2.5 (every family whose
//! preprocessing goes through [`crate::vision::processors::qwen2_vl`]).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A requested per-image token budget, in b10621's units.
///
/// At least one half is always `Some`: [`ImageTokenOverride::from_bounds`]
/// returns `None` when neither was requested, so an installed override always
/// means the operator asked for something.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTokenOverride {
    min_tokens: Option<u32>,
    max_tokens: Option<u32>,
}

impl ImageTokenOverride {
    /// Build an override from the two CLI halves, or `None` when neither was
    /// requested.
    #[must_use]
    pub const fn from_bounds(min_tokens: Option<u32>, max_tokens: Option<u32>) -> Option<Self> {
        if min_tokens.is_none() && max_tokens.is_none() {
            return None;
        }
        Some(Self {
            min_tokens,
            max_tokens,
        })
    }

    /// Render the request the way the operator wrote it, for diagnostics.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut parts = Vec::with_capacity(2);
        if let Some(min) = self.min_tokens {
            parts.push(format!("--image-min-tokens {min}"));
        }
        if let Some(max) = self.max_tokens {
            parts.push(format!("--image-max-tokens {max}"));
        }
        parts.join(" ")
    }

    /// Apply this override to a processor's declared pixel bounds.
    ///
    /// `patch_size` and `merge_size` are the processor's own geometry; the
    /// product `(patch_size * merge_size)^2` is upstream's `patch_area`. A half
    /// the operator did not request keeps the processor's declared bound, which
    /// is what upstream's `custom_image_*_tokens > 0` guard does.
    ///
    /// A degenerate geometry (either factor zero, or an overflowing product)
    /// leaves both declared bounds alone rather than producing a zero-pixel
    /// budget that would resize every image to one patch.
    #[must_use]
    pub fn apply(
        &self,
        patch_size: usize,
        merge_size: usize,
        declared_min_pixels: usize,
        declared_max_pixels: usize,
    ) -> (usize, usize) {
        let Some(patch_area) = patch_area(patch_size, merge_size) else {
            return (declared_min_pixels, declared_max_pixels);
        };
        let min_pixels = self
            .min_tokens
            .and_then(|tokens| (tokens as usize).checked_mul(patch_area))
            .unwrap_or(declared_min_pixels);
        let max_pixels = self
            .max_tokens
            .and_then(|tokens| (tokens as usize).checked_mul(patch_area))
            .unwrap_or(declared_max_pixels);
        (min_pixels, max_pixels)
    }
}

/// Upstream's `patch_area`: `patch_size^2 * n_merge^2`.
///
/// `None` for a zero factor or an overflowing product.
#[must_use]
fn patch_area(patch_size: usize, merge_size: usize) -> Option<usize> {
    let factor = patch_size.checked_mul(merge_size)?;
    if factor == 0 {
        return None;
    }
    factor.checked_mul(factor)
}

/// The process-wide override, installed once before the first model load.
static INSTALLED: OnceLock<Option<ImageTokenOverride>> = OnceLock::new();

/// How many processors have consumed the installed override.
static APPLICATIONS: AtomicUsize = AtomicUsize::new(0);

/// Install the process-wide override before any model is loaded.
///
/// Installing the same value twice is accepted, so the server's model-switch
/// path (which re-enters startup) is not an error.
///
/// # Errors
///
/// Returns `Err` when an override was already installed with a different value,
/// which can only mean two startup paths raced or a caller installed after the
/// first load.
pub fn install(override_value: Option<ImageTokenOverride>) -> Result<(), String> {
    match INSTALLED.set(override_value) {
        Ok(()) => Ok(()),
        Err(_) if INSTALLED.get() == Some(&override_value) => Ok(()),
        Err(_) => Err(format!(
            "an image-token budget is already installed ({:?}); it must be set once, before the \
             first model load",
            INSTALLED.get()
        )),
    }
}

/// The installed override, if the process has one.
#[must_use]
pub fn installed() -> Option<&'static ImageTokenOverride> {
    INSTALLED.get().and_then(|slot| slot.as_ref())
}

/// How many processors have consumed the installed override.
#[must_use]
pub fn applications() -> usize {
    APPLICATIONS.load(Ordering::Relaxed)
}

/// Record that a processor consumed the override.
pub(crate) fn note_application() {
    APPLICATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Record that a processor able to consume the override was constructed.
///
/// [`resolve_pixel_bounds`] only runs when an image is actually preprocessed,
/// which is per request, long after startup has to decide whether the budget
/// reached anything. A processor that *has* the two bounds registers itself at
/// construction instead, so [`verify_applied`] can refuse to serve immediately
/// after the checkpoint loads rather than after the first image arrives.
pub(crate) fn note_dynamic_resolution_processor() {
    if installed().is_some() {
        note_application();
    }
}

/// Resolve the pixel bounds a processor should resize against.
///
/// With no override installed this returns the declared bounds and counts
/// nothing, so the hot path for every ordinary preprocess is one relaxed atomic
/// load.
#[must_use]
pub(crate) fn resolve_pixel_bounds(
    patch_size: usize,
    merge_size: usize,
    declared_min_pixels: usize,
    declared_max_pixels: usize,
) -> (usize, usize) {
    let Some(over) = installed() else {
        return (declared_min_pixels, declared_max_pixels);
    };
    note_application();
    over.apply(
        patch_size,
        merge_size,
        declared_min_pixels,
        declared_max_pixels,
    )
}

/// The diagnostic returned when a requested budget reached no processor.
///
/// Split out so the message can be asserted without installing a process-wide
/// override in a unit test.
#[must_use]
pub fn unapplied_diagnostic(model_label: &str, over: &ImageTokenOverride) -> String {
    format!(
        "{model_label}: {} was accepted on the command line but reached no image preprocessing \
         path, so images would be encoded at the geometry the checkpoint's own preprocessor \
         config declares. b10621 applies an image-token budget only to vision models with \
         dynamic resolution, and this checkpoint's preprocessor has no per-image pixel bound for \
         it to move: it resizes to a fixed geometry and emits the same token count for every \
         input. Drop the flag to serve this checkpoint at its own image resolution.",
        over.describe()
    )
}

/// Confirm that the budget an operator asked for actually reached the model.
///
/// Called once, on the worker thread, immediately after the checkpoint loads
/// and a first image would be preprocessed. Three outcomes:
///
/// - No override installed: `Ok(())`, and nothing was ever consulted.
/// - Override installed and a processor applied it: `Ok(())`.
/// - Override installed and no processor saw it: `Err`, naming the checkpoint
///   and what was asked for.
///
/// # Errors
///
/// Returns the diagnostic above when an installed budget reached no processor.
pub fn verify_applied(model_label: &str) -> Result<(), String> {
    let Some(over) = installed() else {
        return Ok(());
    };
    if applications() == 0 {
        return Err(unapplied_diagnostic(model_label, over));
    }
    Ok(())
}

#[cfg(test)]
#[path = "image_token_overrides_tests.rs"]
mod tests;
