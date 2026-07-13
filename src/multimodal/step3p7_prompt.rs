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

//! Step-3.7 image-placeholder expansion.
//!
//! Each image's single `<im_patch>` placeholder expands to, in order:
//!
//! ```text
//! for each patch i:  <patch_start>  81 x <im_patch>  <patch_end>  [<patch_newline> at row end]
//! then:              <im_start>    169 x <im_patch>  <im_end>
//! ```
//!
//! The patch-then-base placeholder order matches the projected-feature order in
//! `vision::Step3p7VlModel::input_embeddings` (patches first, then base, per
//! image), so the LLaVA-style scatter aligns row-for-row. The counts come from
//! the processor's per-image layout, keeping prompt and processor in lockstep.

use crate::vision::processors::step3p7::{
    BASE_FEATURE_TOKENS, PATCH_FEATURE_TOKENS, Step3p7ImageLayout,
};
use crate::vision::step3p7::Step3p7TokenIds;

/// Summary of the Step-3.7 placeholder expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedStep3p7Tokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// Token sequence one image's placeholder expands to.
fn build_image_block(layout: &Step3p7ImageLayout, ids: &Step3p7TokenIds) -> Vec<i32> {
    let placeholder = ids.image_token_index;
    let mut block = Vec::new();
    let per_row = layout.patches_per_row;
    for i in 0..layout.num_patches {
        block.push(ids.patch_start);
        block.extend(std::iter::repeat_n(placeholder, PATCH_FEATURE_TOKENS));
        block.push(ids.patch_end);
        let is_row_end = per_row > 0 && (i + 1) % per_row == 0;
        let is_last = i + 1 == layout.num_patches;
        if is_row_end && !is_last {
            block.push(ids.patch_newline);
        }
    }
    block.push(ids.im_start);
    block.extend(std::iter::repeat_n(placeholder, BASE_FEATURE_TOKENS));
    block.push(ids.im_end);
    block
}

/// Number of `<im_patch>` scatter targets in one image's block
/// (`169 + 81 * num_patches`).
fn image_patch_count(layout: &Step3p7ImageLayout) -> i32 {
    layout.feature_tokens() as i32
}

/// Expand Step-3.7 image placeholders in `prompt_tokens`.
///
/// Case 1 (canonical templated prompt): exactly one `<im_patch>` per image is
/// replaced by that image's block. Case 2 (no placeholder present): the blocks
/// are inserted right after the first (BOS) token. A non-zero placeholder count
/// that does not match the image count means the prompt is already expanded (or
/// malformed) and is left untouched.
pub fn insert_step3p7_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    layouts: &[Step3p7ImageLayout],
    ids: &Step3p7TokenIds,
) -> Option<InsertedStep3p7Tokens> {
    if prompt_tokens.is_empty() || layouts.is_empty() {
        return None;
    }

    let placeholder = ids.image_token_index;
    let total_image_tokens: i32 = layouts.iter().map(image_patch_count).sum();
    let blocks: Vec<Vec<i32>> = layouts.iter().map(|l| build_image_block(l, ids)).collect();

    let placeholder_count = prompt_tokens.iter().filter(|&&t| t == placeholder).count();

    // Case 1: one placeholder per image -> replace each with its block.
    if placeholder_count == layouts.len() {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total_image_tokens as usize);
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == placeholder {
                expanded.extend_from_slice(&blocks[image_idx]);
                image_idx += 1;
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedStep3p7Tokens {
            image_blocks: layouts.len(),
            total_image_tokens,
        });
    }

    // A mismatched non-zero count means the prompt was already expanded.
    if placeholder_count != 0 {
        return None;
    }

    // Case 2: no placeholder -> insert blocks after the first (BOS) token.
    let bos = prompt_tokens[0];
    let rest = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![bos];
    for block in &blocks {
        prompt_tokens.extend_from_slice(block);
    }
    prompt_tokens.extend(rest);

    Some(InsertedStep3p7Tokens {
        image_blocks: layouts.len(),
        total_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids() -> Step3p7TokenIds {
        Step3p7TokenIds {
            image_token_index: 900,
            im_start: 901,
            im_end: 902,
            patch_start: 903,
            patch_end: 904,
            patch_newline: 905,
        }
    }

    #[test]
    fn unwindowed_image_expands_to_base_block_only() {
        let ids = ids();
        let layout = Step3p7ImageLayout {
            num_patches: 0,
            patches_per_row: 0,
        };
        let mut tokens = vec![1, 900, 2];
        let stats = insert_step3p7_image_tokens(&mut tokens, &[layout], &ids).unwrap();

        assert_eq!(stats.total_image_tokens, 169);
        // 1, <im_start>, 169x<im_patch>, <im_end>, 2.
        assert_eq!(tokens.len(), 1 + 1 + 169 + 1 + 1);
        assert_eq!(tokens[0], 1);
        assert_eq!(tokens[1], ids.im_start);
        assert_eq!(tokens[2], ids.image_token_index);
        assert_eq!(tokens[2 + 169], ids.im_end);
        assert_eq!(*tokens.last().unwrap(), 2);
        let patch_count = tokens
            .iter()
            .filter(|&&t| t == ids.image_token_index)
            .count();
        assert_eq!(patch_count, 169);
    }

    #[test]
    fn windowed_image_places_newlines_at_row_ends_except_last() {
        let ids = ids();
        // 4 patches, 2 per row -> newline only after patch index 1 (first row).
        let layout = Step3p7ImageLayout {
            num_patches: 4,
            patches_per_row: 2,
        };
        let mut tokens = vec![1, 900];
        let stats = insert_step3p7_image_tokens(&mut tokens, &[layout], &ids).unwrap();

        assert_eq!(stats.total_image_tokens, 169 + 81 * 4);
        // Exactly one newline marker (after the first row of two patches).
        let newlines = tokens.iter().filter(|&&t| t == ids.patch_newline).count();
        assert_eq!(newlines, 1);
        // Patch/base framing counts.
        assert_eq!(tokens.iter().filter(|&&t| t == ids.patch_start).count(), 4);
        assert_eq!(tokens.iter().filter(|&&t| t == ids.patch_end).count(), 4);
        assert_eq!(tokens.iter().filter(|&&t| t == ids.im_start).count(), 1);
        assert_eq!(tokens.iter().filter(|&&t| t == ids.im_end).count(), 1);
        // Total scatter targets equals 169 + 81*4.
        let patch_count = tokens
            .iter()
            .filter(|&&t| t == ids.image_token_index)
            .count();
        assert_eq!(patch_count as i32, stats.total_image_tokens);
    }

    #[test]
    fn already_expanded_prompt_is_left_untouched() {
        let ids = ids();
        let layout = Step3p7ImageLayout {
            num_patches: 0,
            patches_per_row: 0,
        };
        // Two placeholders but one image -> mismatch -> no-op.
        let mut tokens = vec![1, 900, 900, 2];
        let before = tokens.clone();
        assert!(insert_step3p7_image_tokens(&mut tokens, &[layout], &ids).is_none());
        assert_eq!(tokens, before);
    }
}
