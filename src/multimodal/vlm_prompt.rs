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

//! Generic image-token block expansion helpers for non-Qwen VLMs.
//!
//! Families such as Gemma3n, LLaVA-style VLMs, and related wrappers all need
//! predictable insertion or expansion of image-token blocks. This module keeps
//! that policy separate from model loading and request parsing.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageTokenBlockInfo {
    pub use_boi_eoi: bool,
    pub image_token_id: i32,
    pub mm_tokens_per_image: usize,
    pub boi_token_id: i32,
    pub eoi_token_id: i32,
    /// Whether the tokenizer adds a BOS token that should be preserved before image tokens.
    /// When false (e.g., PaliGemma), image tokens are simply prepended.
    /// Used by: PaliGemma (false), Gemma3/LLaVA/InternVL (true)
    pub has_bos: bool,
    /// Token to insert between image tokens and text when `has_bos` is false.
    /// PaliGemma: BOS(2) goes between images and text despite add_bos_token=false.
    pub separator_token_id: Option<i32>,
    /// Tokens to append after text.
    /// PaliGemma: newline(108) appended after text prompt.
    pub suffix_tokens: Vec<i32>,
    /// Tokens to insert before each image block during expansion.
    /// Gemma3 VLM: `[108, 108]` (\n\n) to match Python processor behavior.
    /// Used by: Gemma3 VLM
    pub block_prefix_tokens: Vec<i32>,
    /// Tokens to insert after each image block during expansion.
    /// Gemma3 VLM: `[108, 108]` (\n\n) to match Python processor behavior.
    /// Used by: Gemma3 VLM
    pub block_suffix_tokens: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageTokenBlockAction {
    Expanded { existing_image_count: usize },
    Inserted { image_blocks: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTokenBlockStats {
    pub action: ImageTokenBlockAction,
    pub tokens_per_image: usize,
}

/// Invalid family-neutral image-placeholder layouts.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ImageTokenBlockError {
    #[error("cannot place {image_count} image(s) into an empty tokenized prompt")]
    EmptyPrompt { image_count: usize },
    #[error("image-token block size must be greater than zero")]
    EmptyImageBlock,
    #[error(
        "prompt contains {placeholder_count} image placeholder(s), but {image_count} decoded image(s) were provided"
    )]
    MediaCardinality {
        placeholder_count: usize,
        image_count: usize,
    },
    #[error("image-token expansion capacity overflowed")]
    CapacityOverflow,
}

pub fn apply_image_token_blocks(
    prompt_tokens: &mut Vec<i32>,
    info: ImageTokenBlockInfo,
    num_images: usize,
) -> Result<Option<ImageTokenBlockStats>, ImageTokenBlockError> {
    if info.mm_tokens_per_image == 0 {
        return Err(ImageTokenBlockError::EmptyImageBlock);
    }

    // Count existing image tokens or BOI tokens (from chat template) that
    // should be expanded into full image-token blocks.
    // Gemma3 VLM chat templates emit <start_of_image> (= boi_token_id) which
    // must be expanded the same way as a bare image_token_id placeholder.
    let existing_image_count = prompt_tokens
        .iter()
        .filter(|&&token| token == info.image_token_id)
        .count();
    let existing_boi_count = if info.use_boi_eoi && info.boi_token_id != info.image_token_id {
        prompt_tokens
            .iter()
            .filter(|&&token| token == info.boi_token_id)
            .count()
    } else {
        0
    };
    let total_existing = existing_image_count + existing_boi_count;

    if total_existing != 0 && total_existing != num_images {
        return Err(ImageTokenBlockError::MediaCardinality {
            placeholder_count: total_existing,
            image_count: num_images,
        });
    }
    if num_images == 0 {
        return Ok(None);
    }
    if prompt_tokens.is_empty() {
        return Err(ImageTokenBlockError::EmptyPrompt {
            image_count: num_images,
        });
    }

    if total_existing > 0 {
        let wrapper_tokens = if info.use_boi_eoi { 2 } else { 0 };
        let extra_tokens = info
            .mm_tokens_per_image
            .checked_sub(1)
            .and_then(|tokens| tokens.checked_add(wrapper_tokens))
            .and_then(|tokens| tokens.checked_add(info.block_prefix_tokens.len()))
            .and_then(|tokens| tokens.checked_add(info.block_suffix_tokens.len()))
            .and_then(|tokens| tokens.checked_mul(total_existing))
            .ok_or(ImageTokenBlockError::CapacityOverflow)?;
        let capacity = prompt_tokens
            .len()
            .checked_add(extra_tokens)
            .ok_or(ImageTokenBlockError::CapacityOverflow)?;
        let mut expanded = Vec::with_capacity(capacity);
        for &token in prompt_tokens.iter() {
            if token == info.image_token_id || (info.use_boi_eoi && token == info.boi_token_id) {
                // Insert block prefix tokens (e.g., \n\n for Gemma3 VLM)
                expanded.extend_from_slice(&info.block_prefix_tokens);
                if info.use_boi_eoi {
                    expanded.push(info.boi_token_id);
                }
                for _ in 0..info.mm_tokens_per_image {
                    expanded.push(info.image_token_id);
                }
                if info.use_boi_eoi {
                    expanded.push(info.eoi_token_id);
                }
                // Insert block suffix tokens (e.g., \n\n for Gemma3 VLM)
                expanded.extend_from_slice(&info.block_suffix_tokens);
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;

        return Ok(Some(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Expanded {
                existing_image_count: total_existing,
            },
            tokens_per_image: info.mm_tokens_per_image,
        }));
    }

    let wrapper_tokens = if info.use_boi_eoi { 2 } else { 0 };
    let block_len = info
        .mm_tokens_per_image
        .checked_add(wrapper_tokens)
        .ok_or(ImageTokenBlockError::CapacityOverflow)?;
    let image_token_capacity = block_len
        .checked_mul(num_images)
        .ok_or(ImageTokenBlockError::CapacityOverflow)?;
    let separator_len = usize::from(!info.has_bos && info.separator_token_id.is_some());
    prompt_tokens
        .len()
        .checked_add(image_token_capacity)
        .and_then(|len| len.checked_add(separator_len))
        .and_then(|len| len.checked_add(info.suffix_tokens.len()))
        .ok_or(ImageTokenBlockError::CapacityOverflow)?;

    let mut image_tokens = Vec::with_capacity(image_token_capacity);
    for _ in 0..num_images {
        if info.use_boi_eoi {
            image_tokens.push(info.boi_token_id);
        }
        for _ in 0..info.mm_tokens_per_image {
            image_tokens.push(info.image_token_id);
        }
        if info.use_boi_eoi {
            image_tokens.push(info.eoi_token_id);
        }
    }

    if info.has_bos {
        // [BOS, img_tokens..., text_tokens..., suffix...]
        let bos = prompt_tokens[0];
        let rest = prompt_tokens[1..].to_vec();
        *prompt_tokens = vec![bos];
        prompt_tokens.extend(image_tokens);
        prompt_tokens.extend(rest);
    } else {
        // [img_tokens..., separator?, text_tokens..., suffix...]
        let text = std::mem::take(prompt_tokens);
        *prompt_tokens = image_tokens;
        if let Some(sep) = info.separator_token_id {
            prompt_tokens.push(sep);
        }
        prompt_tokens.extend(text);
    }

    prompt_tokens.extend_from_slice(&info.suffix_tokens);

    Ok(Some(ImageTokenBlockStats {
        action: ImageTokenBlockAction::Inserted {
            image_blocks: num_images,
        },
        tokens_per_image: info.mm_tokens_per_image,
    }))
}

#[cfg(test)]
#[path = "vlm_prompt_tests.rs"]
mod tests;
