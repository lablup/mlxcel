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

//! LocateAnything (`locateanything`) prompt token insertion.
//!
//! Mirrors the text rewrite in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/locateanything/processing_locateanything.py:
//! each `<image-N>` marker emitted by the chat template becomes
//! `<img> + <IMG_CONTEXT> * (grid_h * grid_w / merge_length) + </img>`, where
//! `merge_length = merge_kernel_size[0] * merge_kernel_size[1]` (4 for the
//! released 2x2 checkpoint).
//!
//! Unlike InternVL, whose per-tile count is a constant, LocateAnything is
//! native-resolution: every image contributes a different count derived from
//! its own patch grid. The counts are therefore passed in per image rather than
//! multiplied out from a single `num_image_token`.
//!
//! `<image-N>` is not a vocabulary token, so a rendered prompt never carries it
//! as an id. Two cases are handled:
//!
//! 1. The prompt already contains bare `<IMG_CONTEXT>` ids (one per image, e.g.
//!    a caller that pre-expanded the template): each is replaced in place by the
//!    full framed block.
//! 2. The prompt carries no placeholder (the common CLI / OpenAI-server path):
//!    one framed block per image is spliced in after the first token, which is
//!    the ChatML `<|im_start|>` that opens the turn. This mirrors the
//!    InternVL / Youtu-VL / Kimi-VL fallback.
//!
//! Case 1 is the normal path, and it happens at the **text** level rather than
//! the token level: `<image-N>` is not a vocabulary token, so the rendered
//! prompt has to be rewritten and re-encoded, exactly as upstream's
//! `LocateAnythingProcessor.__call__` does with its `re.sub(r"<image-\d+>")`.

/// Statistics describing the LocateAnything image-token insertion/expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedLocateAnythingTokens {
    pub image_blocks: usize,
    pub total_image_tokens: usize,
}

/// The literal marker the chat template emits per image, without its index.
const IMAGE_MARKER_PREFIX: &str = "<image-";
/// Token strings hardcoded by upstream `LocateAnythingProcessor.__init__`.
pub const IMAGE_START_TOKEN: &str = "<img>";
pub const IMAGE_TOKEN: &str = "<IMG_CONTEXT>";
pub const IMAGE_END_TOKEN: &str = "</img>";

/// Rewrite the `<image-N>` markers a rendered chat prompt carries into framed
/// `<img> + <IMG_CONTEXT> * count + </img>` runs.
///
/// This is the text-level half of the contract, matching upstream
/// `LocateAnythingProcessor.__call__`. The caller re-encodes the returned
/// string; the markers are plain text and never single tokens, so they cannot
/// be expanded after tokenization.
///
/// Returns `Ok(None)` when the prompt carries no marker at all (a
/// `--no-chat-template` run, or a template that does not render image parts),
/// which is the signal to fall back to
/// [`insert_locateanything_image_tokens`]. Returns `Err` when the marker count
/// and the image count disagree, because silently dropping or reusing an image
/// would desync the placeholder run from the vision features.
pub fn expand_locateanything_image_markers(
    prompt: &str,
    per_image_counts: &[usize],
) -> Result<Option<(String, InsertedLocateAnythingTokens)>, String> {
    let mut out = String::with_capacity(prompt.len());
    let mut cursor = 0usize;
    let mut image_idx = 0usize;
    let bytes = prompt.as_bytes();

    while let Some(offset) = prompt[cursor..].find(IMAGE_MARKER_PREFIX) {
        let start = cursor + offset;
        let digits_start = start + IMAGE_MARKER_PREFIX.len();
        let mut end = digits_start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        // Not a real marker (no digits, or no closing `>`): copy the literal
        // prefix through and keep scanning after it.
        if end == digits_start || end >= bytes.len() || bytes[end] != b'>' {
            out.push_str(&prompt[cursor..digits_start]);
            cursor = digits_start;
            continue;
        }

        let count = *per_image_counts.get(image_idx).ok_or_else(|| {
            format!(
                "prompt carries more <image-N> markers than the {} image(s) provided",
                per_image_counts.len()
            )
        })?;
        out.push_str(&prompt[cursor..start]);
        out.push_str(IMAGE_START_TOKEN);
        for _ in 0..count {
            out.push_str(IMAGE_TOKEN);
        }
        out.push_str(IMAGE_END_TOKEN);
        image_idx += 1;
        cursor = end + 1;
    }

    if image_idx == 0 {
        return Ok(None);
    }
    if image_idx != per_image_counts.len() {
        return Err(format!(
            "prompt carries {} <image-N> marker(s) but {} image(s) were provided",
            image_idx,
            per_image_counts.len()
        ));
    }
    out.push_str(&prompt[cursor..]);

    Ok(Some((
        out,
        InsertedLocateAnythingTokens {
            image_blocks: image_idx,
            total_image_tokens: per_image_counts.iter().sum(),
        },
    )))
}

/// Merged-token count for one `(grid_h, grid_w)` patch grid under a
/// `merge_h x merge_w` kernel.
#[inline]
pub fn merged_token_count(grid: (i32, i32), merge_kernel_size: [usize; 2]) -> usize {
    let merge = (merge_kernel_size[0] * merge_kernel_size[1]).max(1) as i32;
    ((grid.0 * grid.1) / merge).max(0) as usize
}

/// Build the `<img> + <IMG_CONTEXT> * count + </img>` run for one image.
fn build_block(
    img_start_token_id: i32,
    img_context_token_id: i32,
    img_end_token_id: i32,
    count: usize,
) -> Vec<i32> {
    let mut block = Vec::with_capacity(count + 2);
    block.push(img_start_token_id);
    block.extend(std::iter::repeat_n(img_context_token_id, count));
    block.push(img_end_token_id);
    block
}

/// Insert (or expand) LocateAnything image-token runs into `prompt_tokens`.
///
/// `per_image_counts[i]` is the number of `<IMG_CONTEXT>` tokens image `i`
/// contributes, which must equal the number of merged feature rows the MoonViT
/// tower plus connector emit for that image.
///
/// Returns `None` when there is nothing to do (empty prompt or no images).
pub fn insert_locateanything_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    per_image_counts: &[usize],
    img_start_token_id: i32,
    img_context_token_id: i32,
    img_end_token_id: i32,
) -> Option<InsertedLocateAnythingTokens> {
    if prompt_tokens.is_empty() || per_image_counts.is_empty() {
        return None;
    }

    let total_image_tokens: usize = per_image_counts.iter().sum();
    let image_blocks = per_image_counts.len();

    // Case 1: bare <IMG_CONTEXT> placeholders already present, one per image.
    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == img_context_token_id)
        .count();
    if placeholder_count > 0 {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total_image_tokens);
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == img_context_token_id && image_idx < per_image_counts.len() {
                expanded.extend(build_block(
                    img_start_token_id,
                    img_context_token_id,
                    img_end_token_id,
                    per_image_counts[image_idx],
                ));
                image_idx += 1;
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedLocateAnythingTokens {
            image_blocks,
            total_image_tokens,
        });
    }

    // Case 2: no placeholder — splice one block per image after the first token.
    let mut blocks: Vec<i32> = Vec::with_capacity(total_image_tokens + 2 * image_blocks);
    for &count in per_image_counts {
        blocks.extend(build_block(
            img_start_token_id,
            img_context_token_id,
            img_end_token_id,
            count,
        ));
    }

    let head = prompt_tokens[0];
    let rest: Vec<i32> = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![head];
    prompt_tokens.extend(blocks);
    prompt_tokens.extend(rest);

    Some(InsertedLocateAnythingTokens {
        image_blocks,
        total_image_tokens,
    })
}

#[cfg(test)]
#[path = "locateanything_prompt_tests.rs"]
mod tests;
