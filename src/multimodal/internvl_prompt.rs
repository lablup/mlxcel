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

//! InternVL (`internvl_chat`) prompt token insertion.
//!
//! Mirrors the upstream processor's text rewrite: each image expands into
//! `<img> + <IMG_CONTEXT> * (num_image_token * tiles) + </img>`, where `tiles`
//! is the number of dynamic tiles the processor produced for that image
//! (1 tile for a near-square image; more for wide/tall images, plus an extra
//! thumbnail tile when split).
//!
//! Placement follows the references, which agree on it: InternVL's own
//! `model.chat` prepends `'<image>\n'` to the question, the transformers
//! InternVL chat template renders `<IMG_CONTEXT>\n` ahead of the text, and
//! mlx-vlm renders `<image>\n` there too. Every image block therefore opens the
//! user turn's content and is followed by a newline (issue #2547).
//!
//! 1. When the prompt already carries bare `<IMG_CONTEXT>` placeholders (a
//!    transformers-style template that emits typed image content), each is
//!    expanded in place; the template supplies the newline.
//! 2. The released `internvl3-1b` checkpoint ships a plain Qwen2 ChatML
//!    template with no image-content handling, so the rendered prompt carries
//!    no placeholder. Each block, plus a newline, is spliced right after the
//!    last `<|im_start|>user\n` opener. The split point is located in the
//!    rendered text and verified against the already-tokenized prompt, so the
//!    request keeps its own tokenization. When the question starts with a
//!    newline, Qwen2's pre-tokenizer fuses it with the opener's `\n` (for
//!    example `\n\n\n` as one token), so the opener is no token boundary; the
//!    text after the opener is then re-encoded behind the last block as
//!    `"\n" + rest`, the text the references tokenize there.
//! 3. With no user-turn opener (`--no-chat-template` or a raw prompt) the
//!    blocks and their newlines are prepended, which is `model.chat`'s
//!    `'<image>\n' + question` without a template.
//!
//! Before #2547 the fallback spliced the blocks after the first token, which
//! put the image between `<|im_start|>` and `system` of the Qwen system turn.

/// The ChatML user-turn opener the image blocks follow.
pub const INTERNVL_USER_TURN_MARKER: &str = "<|im_start|>user\n";

/// Statistics describing the InternVL image-token insertion/expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedInternVlTokens {
    pub image_blocks: usize,
    pub total_image_tokens: usize,
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

/// Expand bare `<IMG_CONTEXT>` placeholders in place, one framed block per
/// image, in placeholder order.
///
/// Returns `None` when the prompt carries no placeholder or there is nothing
/// to expand. Shared with LLM-jp-VL, whose `<|image_pad|>` placeholder follows
/// the same rewrite.
pub fn expand_internvl_image_placeholders(
    prompt_tokens: &mut Vec<i32>,
    tiles_per_image: &[usize],
    num_image_token: usize,
    img_start_token_id: i32,
    img_context_token_id: i32,
    img_end_token_id: i32,
) -> Option<InsertedInternVlTokens> {
    if prompt_tokens.is_empty() || tiles_per_image.is_empty() || num_image_token == 0 {
        return None;
    }
    if !prompt_tokens.contains(&img_context_token_id) {
        return None;
    }
    let per_image_counts: Vec<usize> = tiles_per_image
        .iter()
        .map(|&tiles| num_image_token * tiles)
        .collect();
    let total_image_tokens: usize = per_image_counts.iter().sum();
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
    Some(InsertedInternVlTokens {
        image_blocks: tiles_per_image.len(),
        total_image_tokens,
    })
}

/// Insert (or expand) InternVL image-token runs into `prompt_tokens`.
///
/// `prompt` is the rendered text `prompt_tokens` was encoded from, and
/// `encode` the tokenizer it was encoded with; both locate the user turn.
/// `tiles_per_image[i]` is the number of tiles for image `i`; the per-image
/// token count is `num_image_token * tiles_per_image[i]`.
///
/// Returns `None` when there is nothing to do (empty prompt or no images).
#[allow(clippy::too_many_arguments)]
pub fn insert_internvl_image_tokens<E>(
    prompt: &str,
    prompt_tokens: &mut Vec<i32>,
    tiles_per_image: &[usize],
    num_image_token: usize,
    img_start_token_id: i32,
    img_context_token_id: i32,
    img_end_token_id: i32,
    encode: &mut E,
) -> Option<InsertedInternVlTokens>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    if prompt_tokens.is_empty() || tiles_per_image.is_empty() || num_image_token == 0 {
        return None;
    }

    // Case 1: bare placeholders from the template.
    if let Some(stats) = expand_internvl_image_placeholders(
        prompt_tokens,
        tiles_per_image,
        num_image_token,
        img_start_token_id,
        img_context_token_id,
        img_end_token_id,
    ) {
        return Some(stats);
    }

    // Cases 2 and 3: one `block + "\n"` per image at the head of the user turn
    // (or of the prompt when there is no user turn).
    let newline = encode("\n", false);
    let mut blocks: Vec<i32> = Vec::new();
    let mut total_image_tokens = 0usize;
    for &tiles in tiles_per_image {
        let count = num_image_token * tiles;
        total_image_tokens += count;
        blocks.extend(build_block(
            img_start_token_id,
            img_context_token_id,
            img_end_token_id,
            count,
        ));
        blocks.extend_from_slice(&newline);
    }
    match user_turn_split(prompt, prompt_tokens, encode) {
        Some(UserTurnSplit::Aligned(insert_at)) => {
            prompt_tokens.splice(insert_at..insert_at, blocks);
        }
        Some(UserTurnSplit::Fused { head_tokens, rest }) => {
            // The last block's newline is encoded together with the rest of
            // the turn, so a leading newline of the question fuses with it.
            blocks.truncate(blocks.len() - newline.len());
            let mut rebuilt = head_tokens;
            rebuilt.extend(blocks);
            rebuilt.extend(encode(&format!("\n{rest}"), false));
            *prompt_tokens = rebuilt;
        }
        None => {
            prompt_tokens.splice(0..0, blocks);
        }
    }

    Some(InsertedInternVlTokens {
        image_blocks: tiles_per_image.len(),
        total_image_tokens,
    })
}

/// Where the user turn's content starts in the caller's tokens.
enum UserTurnSplit<'a> {
    /// The rendered head up to and including `<|im_start|>user\n` encodes to
    /// a token prefix of the prompt; content starts at this offset.
    Aligned(usize),
    /// Every head token but the opener's final `\n` is a prefix of the
    /// prompt: that `\n` fused with the start of the content. `head_tokens` is
    /// the head's own encoding and `rest` the rendered text after it.
    Fused {
        head_tokens: Vec<i32>,
        rest: &'a str,
    },
}

/// Locate the last `<|im_start|>user\n` opener in the caller's tokens.
///
/// `None` when the rendered prompt has no user turn or the re-tokenized head
/// does not line up with the caller's tokens (a tokenizer disagreement then
/// degrades to prepending instead of shifting the block mid-token).
fn user_turn_split<'a, E>(
    prompt: &'a str,
    prompt_tokens: &[i32],
    encode: &mut E,
) -> Option<UserTurnSplit<'a>>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let marker_start = prompt.rfind(INTERNVL_USER_TURN_MARKER)?;
    let head_end = marker_start + INTERNVL_USER_TURN_MARKER.len();
    let head = &prompt[..head_end];
    // Which `add_special_tokens` the caller used is not visible here; accept
    // whichever encoding really is a prefix of the caller's tokens.
    let encodings: Vec<Vec<i32>> = [false, true]
        .into_iter()
        .map(|add_special| encode(head, add_special))
        .filter(|tokens| !tokens.is_empty() && tokens.len() <= prompt_tokens.len())
        .collect();
    if let Some(aligned) = encodings
        .iter()
        .find(|head_tokens| prompt_tokens.starts_with(head_tokens))
    {
        return Some(UserTurnSplit::Aligned(aligned.len()));
    }
    let fused = encodings.into_iter().find(|head_tokens| {
        head_tokens.len() >= 2 && prompt_tokens.starts_with(&head_tokens[..head_tokens.len() - 1])
    })?;
    Some(UserTurnSplit::Fused {
        head_tokens: fused,
        rest: &prompt[head_end..],
    })
}

#[cfg(test)]
#[path = "internvl_prompt_tests.rs"]
mod tests;
