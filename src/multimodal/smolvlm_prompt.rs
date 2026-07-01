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

//! SmolVLM (`smolvlm`) prompt token insertion.
//!
//! Mirrors the upstream processor's text rewrite: each image expands into
//! `<fake_token_around_image> <global-img> <image> * (num_image_token * tiles)
//! <fake_token_around_image>`, where `tiles` is the number of tiles the
//! processor produced for that image.
//!
//! The `<fake_token_around_image>` and `<global-img>` framing tokens are
//! ordinary text tokens (the merge only replaces the `<image>` positions with
//! projected vision features), so they are emitted only when the tokenizer
//! exposes their ids; the invariant that matters for the merge is that the
//! number of `<image>` placeholders equals the number of image-feature rows.
//! The exact `<row_i_col_j>` per-tile framing that upstream emits for a split
//! image is a real-checkpoint-validation concern (deferred to the
//! orchestrator); the token accounting here stays consistent because the same
//! per-image tile counts drive both the pixel tensor and this expansion.
//!
//! When the rendered prompt already carries bare `<image>` placeholders (the
//! common case for a chat template that emits one `<image>` per image), each is
//! expanded in place. Otherwise one image block per image is spliced in after
//! the first prompt token.

/// Statistics describing the SmolVLM image-token insertion/expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedSmolVlmTokens {
    pub image_blocks: usize,
    pub total_image_tokens: usize,
}

/// Build the `<fake> <global-img> <image> * count <fake>` run for one image.
///
/// Framing tokens with id `0` (unknown) are skipped.
fn build_block(
    image_token_id: i32,
    fake_image_token_id: i32,
    global_image_token_id: i32,
    count: usize,
) -> Vec<i32> {
    let mut block = Vec::with_capacity(count + 3);
    if fake_image_token_id != 0 {
        block.push(fake_image_token_id);
    }
    if global_image_token_id != 0 {
        block.push(global_image_token_id);
    }
    block.extend(std::iter::repeat_n(image_token_id, count));
    if fake_image_token_id != 0 {
        block.push(fake_image_token_id);
    }
    block
}

/// Insert (or expand) SmolVLM image-token runs into `prompt_tokens`.
///
/// `tiles_per_image[i]` is the number of tiles for image `i`; the per-image
/// `<image>` token count is `num_image_token * tiles_per_image[i]`.
///
/// Returns `None` when there is nothing to do (empty prompt or no images).
pub fn insert_smolvlm_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    tiles_per_image: &[usize],
    num_image_token: usize,
    image_token_id: i32,
    fake_image_token_id: i32,
    global_image_token_id: i32,
) -> Option<InsertedSmolVlmTokens> {
    if prompt_tokens.is_empty() || tiles_per_image.is_empty() || num_image_token == 0 {
        return None;
    }

    let per_image_counts: Vec<usize> = tiles_per_image
        .iter()
        .map(|&tiles| num_image_token * tiles)
        .collect();
    let total_image_tokens: usize = per_image_counts.iter().sum();
    let image_blocks = tiles_per_image.len();

    // Case 1: the prompt already carries bare <image> placeholders (one per
    // image). Expand each in place into a full framed block.
    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();
    if placeholder_count > 0 {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total_image_tokens);
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == image_token_id && image_idx < per_image_counts.len() {
                expanded.extend(build_block(
                    image_token_id,
                    fake_image_token_id,
                    global_image_token_id,
                    per_image_counts[image_idx],
                ));
                image_idx += 1;
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedSmolVlmTokens {
            image_blocks,
            total_image_tokens,
        });
    }

    // Case 2: no placeholder, so splice one block per image after the first
    // token (which typically opens the user turn).
    let mut blocks: Vec<i32> = Vec::with_capacity(total_image_tokens + 3 * image_blocks);
    for &count in &per_image_counts {
        blocks.extend(build_block(
            image_token_id,
            fake_image_token_id,
            global_image_token_id,
            count,
        ));
    }

    let head = prompt_tokens[0];
    let rest: Vec<i32> = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![head];
    prompt_tokens.extend(blocks);
    prompt_tokens.extend(rest);

    Some(InsertedSmolVlmTokens {
        image_blocks,
        total_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: i32 = 49153;
    const FAKE: i32 = 49152;
    const GLOBAL: i32 = 49155;

    #[test]
    fn expands_single_image_placeholder_in_place() {
        // one <image> placeholder, 1 tile, 4 tokens/tile.
        let mut prompt = vec![1, IMAGE, 2, 3];
        let stats = insert_smolvlm_image_tokens(&mut prompt, &[1], 4, IMAGE, FAKE, GLOBAL).unwrap();

        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 4);
        // [1, fake, global, image*4, fake, 2, 3]
        assert_eq!(prompt[0], 1);
        assert_eq!(prompt[1], FAKE);
        assert_eq!(prompt[2], GLOBAL);
        assert_eq!(&prompt[3..7], &[IMAGE; 4]);
        assert_eq!(prompt[7], FAKE);
        assert_eq!(&prompt[8..], &[2, 3]);
        // Exactly `count` image tokens survive for the merge.
        assert_eq!(prompt.iter().filter(|&&t| t == IMAGE).count(), 4);
    }

    #[test]
    fn splices_block_after_first_token_when_no_placeholder() {
        let mut prompt = vec![100, 200, 300];
        let stats = insert_smolvlm_image_tokens(&mut prompt, &[2], 3, IMAGE, FAKE, GLOBAL).unwrap();
        // 2 tiles * 3 = 6 image tokens.
        assert_eq!(stats.total_image_tokens, 6);
        assert_eq!(prompt[0], 100);
        assert_eq!(prompt[1], FAKE);
        assert_eq!(prompt[2], GLOBAL);
        assert_eq!(&prompt[3..9], &[IMAGE; 6]);
        assert_eq!(prompt[9], FAKE);
        assert_eq!(&prompt[10..], &[200, 300]);
    }

    #[test]
    fn omits_framing_tokens_when_unknown() {
        let mut prompt = vec![1, IMAGE, 2];
        insert_smolvlm_image_tokens(&mut prompt, &[1], 3, IMAGE, 0, 0).unwrap();
        // No fake/global framing: [1, image*3, 2].
        assert_eq!(prompt, vec![1, IMAGE, IMAGE, IMAGE, 2]);
    }

    #[test]
    fn per_image_tile_counts_drive_block_sizes() {
        let mut prompt = vec![1, 2];
        let stats =
            insert_smolvlm_image_tokens(&mut prompt, &[1, 3], 4, IMAGE, FAKE, GLOBAL).unwrap();
        // image 0: 1*4 = 4, image 1: 3*4 = 12, total 16.
        assert_eq!(stats.image_blocks, 2);
        assert_eq!(stats.total_image_tokens, 16);
        assert_eq!(prompt.iter().filter(|&&t| t == IMAGE).count(), 16);
    }

    #[test]
    fn returns_none_for_empty_inputs() {
        let mut prompt: Vec<i32> = vec![];
        assert!(insert_smolvlm_image_tokens(&mut prompt, &[1], 4, IMAGE, FAKE, GLOBAL).is_none());
        let mut prompt = vec![1, 2, 3];
        assert!(insert_smolvlm_image_tokens(&mut prompt, &[], 4, IMAGE, FAKE, GLOBAL).is_none());
    }
}
