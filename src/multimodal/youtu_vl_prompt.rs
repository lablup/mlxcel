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

//! Youtu-VL prompt token insertion.
//!
//! Each image consumes `(h_patches / merge) * (w_patches / merge)`
//! `image_token_id` tokens, one per merged vision feature. The checkpoint's
//! chat template renders a single `<|vision_start|><|image_pad|><|vision_end|>`
//! per image, and the checkpoint's own processor (`YoutuVLProcessor.__call__`
//! in `processing_youtu_vl.py`) then replaces each `<|image_pad|>`, in order,
//! with that image's full run. This module performs the same rewrite on token
//! ids:
//!
//! - one placeholder per image: each is expanded in place to its image's run,
//!   keeping the template's own start/end framing;
//! - already expanded (one token per feature): left as is;
//! - no placeholder at all (a caller that passed only an image): one framed run
//!   per image is spliced in after the BOS token, mirroring the Qwen-VL
//!   fallback;
//! - any other placeholder count is an error.
//!
//! Issue #1618: the first case used to be read as "already expanded", so the
//! prompt kept one image token while the tower produced one feature per merged
//! patch. The embedding scatter then placed only the first (top-left) feature,
//! and every multi-object image was described from its corner.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedYoutuVlmTokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// The prompt's image placeholders cannot be matched to the images supplied.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error(
    "Youtu-VL prompt carries {found} image placeholder token(s), but {images} image(s) need either one placeholder each or {expected} expanded tokens"
)]
pub struct YoutuVlPlaceholderMismatch {
    pub found: usize,
    pub images: usize,
    pub expected: usize,
}

/// Rewrite `prompt_tokens` so it carries exactly one `image_token_id` per
/// merged vision feature, given each image's `(h_patches, w_patches)`.
///
/// Returns `Ok(Some(stats))` when the prompt was rewritten, `Ok(None)` when
/// there was nothing to do (empty inputs, or the prompt was already expanded),
/// and an error when the placeholder count matches neither the image count nor
/// the expanded token count.
pub fn insert_youtu_vl_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    spatial_shapes: &[(i32, i32)],
    spatial_merge_size: usize,
    vision_start_token_id: i32,
    vision_end_token_id: i32,
    image_token_id: i32,
) -> Result<Option<InsertedYoutuVlmTokens>, YoutuVlPlaceholderMismatch> {
    if prompt_tokens.is_empty() || spatial_shapes.is_empty() || spatial_merge_size == 0 {
        return Ok(None);
    }

    let merge = spatial_merge_size as i32;
    let per_image: Vec<i32> = spatial_shapes
        .iter()
        .map(|&(h, w)| (h / merge) * (w / merge))
        .collect();
    let total_image_tokens: i32 = per_image.iter().sum();
    let expected = usize::try_from(total_image_tokens).unwrap_or(0);
    let stats = InsertedYoutuVlmTokens {
        image_blocks: spatial_shapes.len(),
        total_image_tokens,
    };

    let found = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();

    // Already one token per feature. Checked before the per-image case so a
    // grid whose runs are a single token each is not expanded twice.
    if found == expected {
        return Ok(None);
    }

    if found == spatial_shapes.len() {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + expected);
        let mut runs = per_image.iter();
        for &token in prompt_tokens.iter() {
            // `found == runs.len()`, so every placeholder has a run.
            match (token == image_token_id).then(|| runs.next()).flatten() {
                Some(&count) => {
                    expanded.extend(std::iter::repeat_n(image_token_id, count.max(0) as usize))
                }
                None => expanded.push(token),
            }
        }
        *prompt_tokens = expanded;
        return Ok(Some(stats));
    }

    if found > 0 {
        return Err(YoutuVlPlaceholderMismatch {
            found,
            images: spatial_shapes.len(),
            expected,
        });
    }

    let mut image_tokens: Vec<i32> = Vec::with_capacity(expected + 2 * per_image.len());
    for &count in &per_image {
        image_tokens.push(vision_start_token_id);
        image_tokens.extend(std::iter::repeat_n(image_token_id, count.max(0) as usize));
        image_tokens.push(vision_end_token_id);
    }

    let bos = prompt_tokens[0];
    let rest: Vec<i32> = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![bos];
    prompt_tokens.extend(image_tokens);
    prompt_tokens.extend(rest);

    Ok(Some(stats))
}

#[cfg(test)]
#[path = "youtu_vl_prompt_tests.rs"]
mod tests;
