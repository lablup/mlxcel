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

//! LLM-jp-VL (`llmjpvl`) prompt token insertion.
//!
//! Upstream's `LLMjpVLProcessor.__call__` rewrites the templated text by
//! replacing each `<image>` placeholder with
//! `<|image_start|> + <|image_pad|> * (image_seq_length * tiles) +
//! <|image_end|>`. The placeholder sits inside the user turn, because
//! `apply_chat_template` joins the structured content parts before the template
//! renders `<|start|>user<|message|>{content}<|end|>`.
//!
//! mlxcel renders the chat template first and tokenizes it, so this module
//! reproduces that placement on the token stream:
//!
//! 1. When the prompt already carries bare `<|image_pad|>` placeholders (a
//!    template that emits typed image content, or an operator writing the
//!    special token by hand), each is expanded in place. This is the shared
//!    InternVL path, [`insert_internvl_image_tokens`].
//! 2. Otherwise the block is spliced immediately after the Harmony user-turn
//!    opener `<|start|>user<|message|>`, which is where upstream's `<image>`
//!    would have been. The split point is located in the rendered *text* and
//!    then verified against the already-tokenized prompt, so the request keeps
//!    its original tokenization and a mismatch degrades instead of corrupting.
//! 3. With no user-turn opener at all (`--no-chat-template`, or a raw prompt),
//!    the block is prepended, keeping the image ahead of the question the way
//!    upstream's content ordering does.
//!
//! Used by: `multimodal::vlm_runtime` (LLM-jp-VL arm).

use crate::multimodal::internvl_prompt::{InsertedInternVlTokens, insert_internvl_image_tokens};

/// The Harmony user-turn opener both checkpoints' `chat_template.jinja`
/// renders. Upstream's `<image>` placeholder lands directly after it.
pub const LLMJP_USER_TURN_MARKER: &str = "<|start|>user<|message|>";

/// The generation-prompt suffix upstream appends after `apply_chat_template`.
///
/// The shipped template stops at `<|start|>assistant`; `LLMjpVLProcessor.
/// apply_chat_template` then does `text += "<|channel|>final<|message|>"`, so a
/// bare template render is an incomplete prompt and the model is left to guess
/// the channel.
pub const LLMJP_GENERATION_PROMPT_SUFFIX: &str = "<|channel|>final<|message|>";

/// The bare assistant opener the shipped template ends a generation prompt
/// with.
pub const LLMJP_ASSISTANT_OPENER: &str = "<|start|>assistant";

/// Build the `<|image_start|> + <|image_pad|> * count + <|image_end|>` run for
/// one image.
fn build_block(img_start: i32, img_pad: i32, img_end: i32, count: usize) -> Vec<i32> {
    let mut block = Vec::with_capacity(count + 2);
    block.push(img_start);
    block.extend(std::iter::repeat_n(img_pad, count));
    block.push(img_end);
    block
}

/// Token count of the prompt with the image placeholders removed, upstream's
/// `text_tokens` input to the tile budget.
pub fn text_token_count(prompt_tokens: &[i32], img_pad: i32) -> usize {
    prompt_tokens.iter().filter(|&&t| t != img_pad).count()
}

/// Insert (or expand) LLM-jp-VL image-token runs into `prompt_tokens`.
///
/// `tiles_per_image[i]` is the tile count for image `i`; the per-image token
/// count is `num_image_token * tiles_per_image[i]`.
///
/// Returns `None` when there is nothing to do (empty prompt, no images, or a
/// zero per-tile token count).
#[allow(clippy::too_many_arguments)]
pub fn insert_llmjp_image_tokens<E>(
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

    // Case 1: bare placeholders already present. The InternVL helper expands
    // each in place into a framed block, which is exactly upstream's rewrite.
    if prompt_tokens.contains(&img_context_token_id) {
        return insert_internvl_image_tokens(
            prompt_tokens,
            tiles_per_image,
            num_image_token,
            img_start_token_id,
            img_context_token_id,
            img_end_token_id,
        );
    }

    let per_image_counts: Vec<usize> = tiles_per_image
        .iter()
        .map(|&tiles| num_image_token * tiles)
        .collect();
    let total_image_tokens: usize = per_image_counts.iter().sum();
    let image_blocks = tiles_per_image.len();

    let mut blocks: Vec<i32> = Vec::with_capacity(total_image_tokens + 2 * image_blocks);
    for &count in &per_image_counts {
        blocks.extend(build_block(
            img_start_token_id,
            img_context_token_id,
            img_end_token_id,
            count,
        ));
    }

    let insert_at = user_turn_token_offset(prompt, prompt_tokens, encode).unwrap_or(0);
    prompt_tokens.splice(insert_at..insert_at, blocks);

    Some(InsertedInternVlTokens {
        image_blocks,
        total_image_tokens,
    })
}

/// Token offset just past the last `<|start|>user<|message|>` opener, or `None`
/// when the rendered prompt has no user turn or the re-tokenized prefix does
/// not match the prompt the caller already tokenized.
///
/// The verification matters: it keeps the request on its original token stream
/// (nothing is re-tokenized into the prompt) and turns any tokenizer
/// disagreement into a fallback rather than a silently shifted image block.
fn user_turn_token_offset<E>(prompt: &str, prompt_tokens: &[i32], encode: &mut E) -> Option<usize>
where
    E: FnMut(&str, bool) -> Vec<i32>,
{
    let marker_start = prompt.rfind(LLMJP_USER_TURN_MARKER)?;
    let head_end = marker_start + LLMJP_USER_TURN_MARKER.len();
    let head = &prompt[..head_end];
    // The caller tokenized the whole prompt; which `add_special_tokens` it used
    // is not visible here, so try both and accept the one whose tokens really
    // are a prefix of what the caller produced. Neither checkpoint prepends BOS
    // (`add_bos_token: false`), so in practice the first attempt matches.
    for add_special in [false, true] {
        let head_tokens = encode(head, add_special);
        if head_tokens.is_empty() || head_tokens.len() > prompt_tokens.len() {
            continue;
        }
        if prompt_tokens[..head_tokens.len()] == head_tokens[..] {
            return Some(head_tokens.len());
        }
    }
    None
}

#[cfg(test)]
#[path = "llmjp_vl_prompt_tests.rs"]
mod tests;
