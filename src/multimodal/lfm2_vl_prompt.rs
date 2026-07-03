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

//! LFM2-VL prompt token expansion.
//!
//! Each `<image>` (396) sentinel expands, in prompt order, into
//! `[<|image_start|>?, <image> * T_i, <|image_end|>?]` where
//! `T_i = ceil(h_i / f) * ceil(w_i / f)` is image `i`'s post-downsample token
//! count. The framing tokens are emitted only when `use_image_special_tokens`
//! is set and the tokenizer exposes them; the merge only replaces the `<image>`
//! rows, so the invariant that matters is that the count of 396-tokens equals
//! the number of image-feature rows.

/// Statistics describing the LFM2-VL image-token expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedLfm2VlTokens {
    pub image_blocks: usize,
    pub total_image_tokens: usize,
}

fn build_block(
    image_token_id: i32,
    image_start_id: i32,
    image_end_id: i32,
    use_special: bool,
    count: usize,
) -> Vec<i32> {
    let mut block = Vec::with_capacity(count + 2);
    if use_special && image_start_id != 0 {
        block.push(image_start_id);
    }
    block.extend(std::iter::repeat_n(image_token_id, count));
    if use_special && image_end_id != 0 {
        block.push(image_end_id);
    }
    block
}

/// Expand (or splice) LFM2-VL image-token runs into `prompt_tokens`. Per-image
/// counts come from `grids` (`(h_i, w_i)` patch grids) and `downsample_factor`.
pub fn insert_lfm2_vl_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    grids: &[(i32, i32)],
    downsample_factor: i32,
    image_token_id: i32,
    image_start_id: i32,
    image_end_id: i32,
    use_special: bool,
) -> Option<InsertedLfm2VlTokens> {
    if prompt_tokens.is_empty() || grids.is_empty() {
        return None;
    }
    let f = downsample_factor.max(1);
    let per_image_counts: Vec<usize> = grids
        .iter()
        .map(|&(h, w)| (((h + f - 1) / f) * ((w + f - 1) / f)).max(0) as usize)
        .collect();
    let total_image_tokens: usize = per_image_counts.iter().sum();
    let image_blocks = grids.len();

    // Case 1: expand each bare <image> placeholder in place (one per image).
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
                    image_start_id,
                    image_end_id,
                    use_special,
                    per_image_counts[image_idx],
                ));
                image_idx += 1;
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedLfm2VlTokens {
            image_blocks,
            total_image_tokens,
        });
    }

    // Case 2: no placeholder; splice one block per image after the first token.
    let mut blocks: Vec<i32> = Vec::with_capacity(total_image_tokens + 2 * image_blocks);
    for &count in &per_image_counts {
        blocks.extend(build_block(
            image_token_id,
            image_start_id,
            image_end_id,
            use_special,
            count,
        ));
    }
    let head = prompt_tokens[0];
    let rest: Vec<i32> = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![head];
    prompt_tokens.extend(blocks);
    prompt_tokens.extend(rest);

    Some(InsertedLfm2VlTokens {
        image_blocks,
        total_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const IMAGE: i32 = 396;
    const START: i32 = 498;
    const END: i32 = 499;

    #[test]
    fn expands_placeholder_with_framing() {
        // grid (4, 6), f=2 -> T = 2*3 = 6.
        let mut prompt = vec![1, IMAGE, 2];
        let stats = insert_lfm2_vl_image_tokens(&mut prompt, &[(4, 6)], 2, IMAGE, START, END, true)
            .unwrap();
        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 6);
        assert_eq!(prompt[0], 1);
        assert_eq!(prompt[1], START);
        assert_eq!(&prompt[2..8], &[IMAGE; 6]);
        assert_eq!(prompt[8], END);
        assert_eq!(prompt[9], 2);
        assert_eq!(prompt.iter().filter(|&&t| t == IMAGE).count(), 6);
    }

    #[test]
    fn no_framing_when_disabled() {
        let mut prompt = vec![1, IMAGE, 2];
        insert_lfm2_vl_image_tokens(&mut prompt, &[(2, 2)], 2, IMAGE, START, END, false).unwrap();
        // grid (2,2), f=2 -> T=1. No start/end.
        assert_eq!(prompt, vec![1, IMAGE, 2]);
    }

    #[test]
    fn per_image_counts_and_odd_grid_ceil() {
        // grid (3,5), f=2 -> ceil(3/2)*ceil(5/2) = 2*3 = 6; grid (2,2) -> 1.
        let mut prompt = vec![1, IMAGE, IMAGE, 2];
        let stats =
            insert_lfm2_vl_image_tokens(&mut prompt, &[(3, 5), (2, 2)], 2, IMAGE, 0, 0, true)
                .unwrap();
        assert_eq!(stats.total_image_tokens, 7);
        assert_eq!(prompt.iter().filter(|&&t| t == IMAGE).count(), 7);
    }
}
