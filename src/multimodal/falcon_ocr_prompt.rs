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

//! Falcon-OCR prompt assembly.
//!
//! The reference builds `f"<|image|>{instruction}\n<|OCR_PLAIN|>"` and then
//! substitutes each `<|image|>` with a full image block
//! `[<|image_cls|>, <|image_reg_1..4|>, <|image|> * patches, <|end_of_image|>]`
//! (`processing_falcon_ocr.py::tokenize_inputs`).
//!
//! mlxcel receives an already-tokenized user prompt, so the same two edits are
//! applied at the token level: expand (or prepend) the image blocks, and make
//! sure the sequence ends with the OCR task token. The task token is not
//! optional decoration. Dropping it makes the model narrate the page instead of
//! transcribing it, and it is the one part of the reference prompt a plain
//! `-p "..."` cannot carry.

use crate::models::falcon_ocr_rope::FalconOcrTokenIds;

/// What the expansion did, for the CLI summary line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedFalconOcrTokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
    /// True when the OCR task token had to be appended.
    pub appended_task_token: bool,
}

/// Expand image placeholders into Falcon-OCR image blocks and append the OCR
/// task token.
///
/// `grids` lists `(rows, cols)` of 16x16 patches per image, in prompt order.
///
/// Two shapes are accepted, mirroring `insert_qwen_vl_image_tokens`:
///
/// - one `<|image|>` placeholder per image: each is replaced in place by its
///   block, so a caller that templated the prompt itself keeps its layout;
/// - no placeholder at all: the blocks are prepended, which is what a plain
///   `mlxcel generate -p "..." --image page.png` produces.
///
/// Any other placeholder count means the prompt was already expanded (or is
/// malformed) and is left untouched.
pub fn insert_falcon_ocr_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    grids: &[(i32, i32)],
    ids: &FalconOcrTokenIds,
    ocr_task_token_id: Option<i32>,
) -> Option<InsertedFalconOcrTokens> {
    if grids.is_empty() {
        return None;
    }

    let per_image: Vec<i32> = grids
        .iter()
        .map(|&(rows, cols)| rows.max(0) * cols.max(0))
        .collect();
    let total_image_tokens: i32 = per_image.iter().sum();

    let block_for = |patches: i32| {
        let mut block = Vec::with_capacity(patches as usize + 6);
        block.extend_from_slice(&ids.block_prefix());
        block.extend(std::iter::repeat_n(ids.img_id, patches.max(0) as usize));
        block.push(ids.img_end_id);
        block
    };

    let placeholders = prompt_tokens.iter().filter(|&&t| t == ids.img_id).count();
    if placeholders == grids.len() {
        let mut expanded =
            Vec::with_capacity(prompt_tokens.len() + total_image_tokens as usize + 6);
        let mut idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == ids.img_id {
                expanded.extend(block_for(per_image[idx]));
                idx += 1;
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
    } else if placeholders == 0 {
        let mut expanded =
            Vec::with_capacity(prompt_tokens.len() + total_image_tokens as usize + 6);
        for &patches in &per_image {
            expanded.extend(block_for(patches));
        }
        expanded.extend_from_slice(prompt_tokens);
        *prompt_tokens = expanded;
    } else {
        return None;
    }

    let appended_task_token = match ocr_task_token_id {
        Some(task) if prompt_tokens.last() != Some(&task) => {
            prompt_tokens.push(task);
            true
        }
        _ => false,
    };

    Some(InsertedFalconOcrTokens {
        image_blocks: grids.len(),
        total_image_tokens,
        appended_task_token,
    })
}

#[cfg(test)]
#[path = "falcon_ocr_prompt_tests.rs"]
mod tests;
