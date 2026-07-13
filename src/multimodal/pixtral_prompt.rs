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

//! Pixtral / Mistral3 row-structured image-token expansion.
//!
//! Port of the token layout in
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/pixtral/processing_pixtral.py
//! (`PixtralProcessor.__call__`). Each `[IMG]` placeholder expands to a grid of
//! `[IMG]` tokens laid out row by row: `[IMG_BREAK]` after every row except the
//! last, `[IMG_END]` after the whole image. Only the `[IMG]` tokens are replaced
//! by vision features on merge; `[IMG_BREAK]` and `[IMG_END]` keep their learned
//! text embeddings and carry the spatial structure the model was trained on.
//!
//! The `[IMG]` count per image is exactly `tokens_h * tokens_w`, matching the
//! post-merge feature count from the processor and connector. That equality is
//! the core correctness invariant: a mismatch feeds the model garbage.

/// Summary of a Pixtral/Mistral3 image-token expansion.
pub struct InsertedPixtralTokens {
    /// Number of image blocks emitted (one per image).
    pub image_blocks: usize,
    /// Total `[IMG]` tokens across all images (== total vision features).
    pub total_image_tokens: i32,
}

/// Append one image's row-structured block: `cols` `[IMG]` per row for `rows`
/// rows, `[IMG_BREAK]` between rows, `[IMG_END]` at the end.
fn push_image_block(
    out: &mut Vec<i32>,
    rows: usize,
    cols: usize,
    image_token_id: i32,
    image_break_token_id: i32,
    image_end_token_id: i32,
) {
    for row in 0..rows {
        for _ in 0..cols {
            out.push(image_token_id);
        }
        if row + 1 < rows {
            out.push(image_break_token_id);
        }
    }
    out.push(image_end_token_id);
}

/// Total tokens (including break/end structure) one block occupies.
fn block_len(rows: usize, cols: usize) -> usize {
    rows * cols + rows.saturating_sub(1) + 1
}

/// Expand each `[IMG]` placeholder in `prompt_tokens` into its row-structured
/// block using the per-image `(tokens_h, tokens_w)` grids (in order).
///
/// Two placeholder shapes are supported, mirroring
/// [`crate::deepseekocr_prompt::insert_deepseekocr_image_tokens`]:
/// - one `[IMG]` per image (the chat template's rendered form): each is
///   expanded in place with its own grid;
/// - no `[IMG]` at all: one block per image is spliced right after the leading
///   BOS token.
///
/// Returns `None` when there are no images, the prompt is empty, or the
/// placeholder count is neither `grids.len()` nor zero.
pub fn insert_pixtral_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    grids: &[(usize, usize)],
    image_token_id: i32,
    image_break_token_id: i32,
    image_end_token_id: i32,
) -> Option<InsertedPixtralTokens> {
    if prompt_tokens.is_empty() || grids.is_empty() {
        return None;
    }

    let placeholders = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();

    let total_image_tokens: i32 = grids.iter().map(|&(h, w)| (h * w) as i32).sum();
    let block_budget: usize = grids.iter().map(|&(h, w)| block_len(h, w)).sum();

    // One placeholder per image: expand each in place with its own grid.
    if placeholders == grids.len() {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + block_budget);
        let mut idx = 0usize;
        for &tok in prompt_tokens.iter() {
            if tok == image_token_id {
                let (rows, cols) = grids[idx];
                push_image_block(
                    &mut expanded,
                    rows,
                    cols,
                    image_token_id,
                    image_break_token_id,
                    image_end_token_id,
                );
                idx += 1;
            } else {
                expanded.push(tok);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedPixtralTokens {
            image_blocks: grids.len(),
            total_image_tokens,
        });
    }

    // No placeholder: splice one block per image after the leading BOS token.
    if placeholders == 0 {
        let bos = prompt_tokens[0];
        let rest = prompt_tokens[1..].to_vec();
        let mut out = Vec::with_capacity(prompt_tokens.len() + block_budget);
        out.push(bos);
        for &(rows, cols) in grids {
            push_image_block(
                &mut out,
                rows,
                cols,
                image_token_id,
                image_break_token_id,
                image_end_token_id,
            );
        }
        out.extend(rest);
        *prompt_tokens = out;
        return Some(InsertedPixtralTokens {
            image_blocks: grids.len(),
            total_image_tokens,
        });
    }

    None
}

#[cfg(test)]
#[path = "pixtral_prompt_tests.rs"]
mod tests;
