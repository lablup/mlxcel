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

//! InternVL (`internvl_chat`) prompt token insertion.
//!
//! Mirrors the upstream processor's text rewrite: each image expands into
//! `<img> + <IMG_CONTEXT> * (num_image_token * tiles) + </img>`, where `tiles`
//! is the number of dynamic tiles the processor produced for that image
//! (1 tile for a near-square image; more for wide/tall images, plus an extra
//! thumbnail tile when split).
//!
//! The released `internvl3-1b` checkpoint ships a standard Qwen2 ChatML
//! template with no image-content handling, so the rendered prompt carries no
//! `<IMG_CONTEXT>` placeholder. In that common case we splice one image block
//! per image into the prompt right after the first token (the `<|im_start|>`
//! that opens the user turn), mirroring the Qwen-VL / Youtu-VL fallback. If a
//! future template *does* emit `<IMG_CONTEXT>` placeholders, each placeholder
//! is expanded in place instead.

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

/// Insert (or expand) InternVL image-token runs into `prompt_tokens`.
///
/// `tiles_per_image[i]` is the number of tiles for image `i`; the per-image
/// token count is `num_image_token * tiles_per_image[i]`.
///
/// Returns `None` when there is nothing to do (empty prompt or no images).
pub fn insert_internvl_image_tokens(
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

    let per_image_counts: Vec<usize> = tiles_per_image
        .iter()
        .map(|&tiles| num_image_token * tiles)
        .collect();
    let total_image_tokens: usize = per_image_counts.iter().sum();
    let image_blocks = tiles_per_image.len();

    // Case 1: the prompt already carries bare <IMG_CONTEXT> placeholders
    // (one per image). Expand each in place into a full framed block.
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
        return Some(InsertedInternVlTokens {
            image_blocks,
            total_image_tokens,
        });
    }

    // Case 2: no placeholder — splice one block per image after the first
    // token (the ChatML `<|im_start|>` opening the user turn).
    let mut blocks: Vec<i32> = Vec::with_capacity(total_image_tokens + 2 * image_blocks);
    for &count in &per_image_counts {
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

    Some(InsertedInternVlTokens {
        image_blocks,
        total_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG_START: i32 = 151_665;
    const IMG_CONTEXT: i32 = 151_667;
    const IMG_END: i32 = 151_666;

    #[test]
    fn splices_single_image_block_after_first_token() {
        // ChatML opener token + body. Single image, 1 tile, 4 tokens/tile.
        let mut prompt = vec![151_644, 200, 300];
        let stats =
            insert_internvl_image_tokens(&mut prompt, &[1], 4, IMG_START, IMG_CONTEXT, IMG_END)
                .unwrap();

        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 4);
        // [opener, <img>, ctx×4, </img>, body...]
        assert_eq!(prompt[0], 151_644);
        assert_eq!(prompt[1], IMG_START);
        assert_eq!(&prompt[2..6], &[IMG_CONTEXT; 4]);
        assert_eq!(prompt[6], IMG_END);
        assert_eq!(&prompt[7..], &[200, 300]);
    }

    #[test]
    fn expands_existing_placeholder_in_place() {
        // A prompt that already has one <IMG_CONTEXT> placeholder.
        let mut prompt = vec![151_644, IMG_CONTEXT, 300];
        let stats =
            insert_internvl_image_tokens(&mut prompt, &[2], 4, IMG_START, IMG_CONTEXT, IMG_END)
                .unwrap();

        // 2 tiles × 4 tokens = 8 context tokens.
        assert_eq!(stats.total_image_tokens, 8);
        assert_eq!(prompt[0], 151_644);
        assert_eq!(prompt[1], IMG_START);
        assert_eq!(&prompt[2..10], &[IMG_CONTEXT; 8]);
        assert_eq!(prompt[10], IMG_END);
        assert_eq!(prompt[11], 300);
    }

    #[test]
    fn per_image_tile_counts_drive_block_sizes() {
        let mut prompt = vec![151_644, 200];
        let stats =
            insert_internvl_image_tokens(&mut prompt, &[1, 3], 4, IMG_START, IMG_CONTEXT, IMG_END)
                .unwrap();
        // image 0: 1×4 = 4, image 1: 3×4 = 12, total 16.
        assert_eq!(stats.image_blocks, 2);
        assert_eq!(stats.total_image_tokens, 16);
    }

    #[test]
    fn returns_none_for_empty_inputs() {
        let mut prompt: Vec<i32> = vec![];
        assert!(
            insert_internvl_image_tokens(&mut prompt, &[1], 4, IMG_START, IMG_CONTEXT, IMG_END)
                .is_none()
        );
        let mut prompt = vec![1, 2, 3];
        assert!(
            insert_internvl_image_tokens(&mut prompt, &[], 4, IMG_START, IMG_CONTEXT, IMG_END)
                .is_none()
        );
    }
}
