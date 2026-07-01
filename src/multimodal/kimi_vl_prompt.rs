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

//! Kimi-VL / Kimi-VL 2.5 prompt token insertion.
//!
//! Mirrors the upstream processor's text rewrite: each image's single
//! `<|media_pad|>` placeholder expands into
//! `(h_patches / merge) * (w_patches / merge)` `media_placeholder_token_id`
//! tokens (the number of merged patch tokens the MoonViT tower emits). Unlike
//! Youtu-VL / InternVL there is no extra `<vision_start>` / `<img>` framing to
//! add here: the Kimi chat template already wraps the media span in its
//! `<|media_start|>` / `<|media_end|>` markers, so the placeholder simply grows
//! in place.
//!
//! When the rendered prompt carries no `<|media_pad|>` placeholder (a bare CLI
//! image), one expanded run per image is spliced in after the first token,
//! mirroring the Youtu-VL / InternVL fallback.

/// Statistics describing the Kimi-VL media-token insertion/expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedKimiVlTokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// Insert (or expand) Kimi-VL media-placeholder runs into `prompt_tokens`.
///
/// `grid_shapes[i]` is the `(h_patches, w_patches)` grid for image `i`; the
/// per-image token count is `(h / merge) * (w / merge)`.
///
/// Returns `None` when there is nothing to do (empty prompt, no images, or a
/// zero merge size).
pub fn insert_kimi_vl_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    grid_shapes: &[(i32, i32)],
    spatial_merge_size: i32,
    media_placeholder_token_id: i32,
) -> Option<InsertedKimiVlTokens> {
    if prompt_tokens.is_empty() || grid_shapes.is_empty() || spatial_merge_size <= 0 {
        return None;
    }

    let per_image_counts: Vec<i32> = grid_shapes
        .iter()
        .map(|&(h, w)| (h / spatial_merge_size) * (w / spatial_merge_size))
        .collect();
    let total_image_tokens: i32 = per_image_counts.iter().sum();
    let image_blocks = grid_shapes.len();

    // Case 1: the prompt already carries one <|media_pad|> per image. Expand
    // each in place into its full run of placeholder tokens.
    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == media_placeholder_token_id)
        .count();
    if placeholder_count > 0 {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total_image_tokens as usize);
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == media_placeholder_token_id && image_idx < per_image_counts.len() {
                let count = per_image_counts[image_idx].max(0) as usize;
                expanded.extend(std::iter::repeat_n(media_placeholder_token_id, count));
                image_idx += 1;
            } else {
                expanded.push(token);
            }
        }
        *prompt_tokens = expanded;
        return Some(InsertedKimiVlTokens {
            image_blocks,
            total_image_tokens,
        });
    }

    // Case 2: no placeholder — splice one run per image after the first token.
    let mut runs: Vec<i32> = Vec::with_capacity(total_image_tokens as usize);
    for &count in &per_image_counts {
        runs.extend(std::iter::repeat_n(
            media_placeholder_token_id,
            count.max(0) as usize,
        ));
    }
    let head = prompt_tokens[0];
    let rest: Vec<i32> = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![head];
    prompt_tokens.extend(runs);
    prompt_tokens.extend(rest);

    Some(InsertedKimiVlTokens {
        image_blocks,
        total_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEDIA_PAD: i32 = 163_606;

    #[test]
    fn expands_existing_placeholder_in_place() {
        // One image, (4,4) grid, merge 2 -> (4/2)*(4/2) = 4 media tokens.
        let mut prompt = vec![1i32, MEDIA_PAD, 300];
        let stats = insert_kimi_vl_image_tokens(&mut prompt, &[(4, 4)], 2, MEDIA_PAD).unwrap();
        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 4);
        assert_eq!(prompt[0], 1);
        assert_eq!(&prompt[1..5], &[MEDIA_PAD; 4]);
        assert_eq!(prompt[5], 300);
    }

    #[test]
    fn splices_after_first_token_when_absent() {
        let mut prompt = vec![1i32, 200, 300];
        let stats = insert_kimi_vl_image_tokens(&mut prompt, &[(4, 4)], 2, MEDIA_PAD).unwrap();
        assert_eq!(stats.total_image_tokens, 4);
        assert_eq!(prompt[0], 1);
        assert_eq!(&prompt[1..5], &[MEDIA_PAD; 4]);
        assert_eq!(&prompt[5..], &[200, 300]);
    }

    #[test]
    fn per_image_grid_drives_run_sizes() {
        let mut prompt = vec![1i32, MEDIA_PAD, MEDIA_PAD, 9];
        let stats =
            insert_kimi_vl_image_tokens(&mut prompt, &[(4, 4), (2, 2)], 2, MEDIA_PAD).unwrap();
        // image 0: (4/2)*(4/2)=4, image 1: (2/2)*(2/2)=1, total 5.
        assert_eq!(stats.image_blocks, 2);
        assert_eq!(stats.total_image_tokens, 5);
    }

    #[test]
    fn returns_none_for_empty_inputs() {
        let mut empty: Vec<i32> = vec![];
        assert!(insert_kimi_vl_image_tokens(&mut empty, &[(4, 4)], 2, MEDIA_PAD).is_none());
        let mut prompt = vec![1, 2];
        assert!(insert_kimi_vl_image_tokens(&mut prompt, &[], 2, MEDIA_PAD).is_none());
    }
}
