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

//! The one OpenXLA image sizing rule, shared by the CLI and the serve worker.
//!
//! An image request is sized and admitted on the prompt length the family's
//! preprocessor actually produced: `PreparedPrefill::sequence_len`, the token
//! count after every image placeholder has been expanded for the image that was
//! decoded. The serve worker routes that length to the smallest compiled bucket
//! that admits it; the CLI, which compiles exactly one shape per run, compiles
//! the shape for it (issue #3361). Neither path adds a per-family worst case on
//! top, so a 224x224 image on Qwen2-VL needs its 64 merged tokens rather than
//! the 16384 a maximum-pixel image could expand to, and families without a
//! config-derived bound (Gemma3, Molmo v1, Youtu-VL) are sized like the rest.
//!
//! The worst-case floor ([`crate::xla_image_context_floor`]) still exists for
//! the one decision that has to be made before any image is seen: which bucket
//! set the server compiles at startup.
//!
//! # Bounds that stay fail-closed
//!
//! Sizing from the processed media never widens what a family accepts. Each
//! preprocessor still enforces its own limits before returning, and admission
//! still checks `effective_prompt_len + max_new_tokens <= context_capacity`
//! against the shape that was compiled (or pinned with
//! `MLXCEL_XLA_CONTEXT_CAPACITY`). In particular the Qwen3-VL IREE vision
//! tower is compiled for packed patch buckets of 16, 64, 256 and 512 patches,
//! so an image (or a set of images) whose resized grid exceeds 512 patches,
//! that is more than 128 merged image tokens with the 2x2 merge, is refused by
//! the preprocessor with "exceeding qualified capacity 512" before any shape is
//! compiled. Raising that bound needs a larger vision bucket, not a larger
//! context shape. Youtu-VL's vision buckets top out at 256 patches instead,
//! and its processor shrinks a larger image until it fits, so one image never
//! expands past 64 tokens there. The Qwen2-VL and Qwen2.5-VL vision towers are
//! compiled for packed patch buckets of 16, 64 and 256 patches
//! (`QWEN2_VL_PATCH_BUCKETS`), so their images (summed across the request) are
//! refused past 256 patches, 64 merged tokens, with "exceeding qualified
//! capacity 256". A 224x224 image sits exactly at that cap.

use image::DynamicImage;
use mlxcel_core::session::PreparedPrefill;
use mlxcel_xla::DeepStackPreparedPrefill;

use super::host_preprocessor::{HostMultimodalPreprocessor, HostPreprocessorError};

/// An image request after host preprocessing, in the form its prefill entry
/// takes.
pub enum PreparedImagePrefill {
    /// Ordinary merged-embedding prefill.
    Standard(PreparedPrefill),
    /// Qwen3-VL prefill with its DeepStack side inputs.
    DeepStack(DeepStackPreparedPrefill),
}

impl PreparedImagePrefill {
    /// The prompt length after placeholder expansion, which is what sizing and
    /// admission use.
    #[must_use]
    pub fn sequence_len(&self) -> usize {
        self.prepared().sequence_len
    }

    /// The merged-embedding prefill, for either variant.
    #[must_use]
    pub fn prepared(&self) -> &PreparedPrefill {
        match self {
            Self::Standard(prepared) => prepared,
            Self::DeepStack(request) => request.prepared(),
        }
    }
}

/// Prepare decoded images through `preprocessor`, taking the DeepStack entry
/// first so a Qwen3-VL request is never downgraded to main-merger-only
/// embeddings.
///
/// # Errors
///
/// Returns the preprocessor's own validation or execution error, including the
/// family vision bounds described in the module documentation.
pub fn prepare_image_prefill(
    preprocessor: &dyn HostMultimodalPreprocessor,
    token_ids: &[i32],
    images: &[DynamicImage],
) -> Result<PreparedImagePrefill, HostPreprocessorError> {
    if let Some(request) = preprocessor.prepare_deepstack(token_ids, images)? {
        return Ok(PreparedImagePrefill::DeepStack(request));
    }
    preprocessor
        .prepare(token_ids, images)
        .map(PreparedImagePrefill::Standard)
}

/// The positions a single-shape OpenXLA request needs: the expanded prompt
/// plus the generation budget it is sized with.
#[must_use]
pub fn xla_request_positions(effective_prompt_len: usize, sizing_budget: usize) -> usize {
    effective_prompt_len.saturating_add(sizing_budget)
}

#[cfg(test)]
#[path = "xla_image_prefill_tests.rs"]
mod tests;
