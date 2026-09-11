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

//! Youtu-VL prompt token insertion.
//!
//! Mirrors upstream's prompt-format expectation: each image consumes
//! `(h_patches / merge) * (w_patches / merge)` `image_token_id` tokens,
//! framed by `vision_start_token_id` and `vision_end_token_id`. When the
//! prompt does not already carry an `image_token_id`, the run is spliced in
//! after the BOS token (mirroring the Qwen-VL fallback behaviour for callers
//! that pass only an image without an explicit placeholder).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedYoutuVlmTokens {
    pub image_blocks: usize,
    pub total_image_tokens: i32,
}

/// Insert image-placeholder runs into `prompt_tokens` based on each image's
/// `(h_patches, w_patches)`. Returns `None` when the inputs are empty or
/// the prompt already contains expanded image tokens (in which case the
/// caller should not re-expand).
pub fn insert_youtu_vl_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    spatial_shapes: &[(i32, i32)],
    spatial_merge_size: usize,
    vision_start_token_id: i32,
    vision_end_token_id: i32,
    image_token_id: i32,
) -> Option<InsertedYoutuVlmTokens> {
    if prompt_tokens.is_empty()
        || spatial_shapes.is_empty()
        || prompt_tokens.contains(&image_token_id)
        || spatial_merge_size == 0
    {
        return None;
    }

    let merge = spatial_merge_size as i32;
    let mut image_tokens: Vec<i32> = Vec::new();
    let mut total_image_tokens = 0i32;

    for &(h, w) in spatial_shapes {
        let tokens_per_image = (h / merge) * (w / merge);
        total_image_tokens += tokens_per_image;
        image_tokens.push(vision_start_token_id);
        for _ in 0..tokens_per_image {
            image_tokens.push(image_token_id);
        }
        image_tokens.push(vision_end_token_id);
    }

    let bos = prompt_tokens[0];
    let rest: Vec<i32> = prompt_tokens[1..].to_vec();
    *prompt_tokens = vec![bos];
    prompt_tokens.extend(image_tokens);
    prompt_tokens.extend(rest);

    Some(InsertedYoutuVlmTokens {
        image_blocks: spatial_shapes.len(),
        total_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_when_prompt_already_has_image_tokens() {
        let mut prompt = vec![1, 100, 100, 200];
        let result = insert_youtu_vl_image_tokens(&mut prompt, &[(4, 4)], 2, 128_262, 128_263, 100);
        assert!(result.is_none());
        // Prompt unchanged.
        assert_eq!(prompt, vec![1, 100, 100, 200]);
    }

    #[test]
    fn splices_image_tokens_after_bos() {
        let mut prompt = vec![1i32, 200, 300];
        let result =
            insert_youtu_vl_image_tokens(&mut prompt, &[(4, 4)], 2, 128_262, 128_263, 128_264)
                .unwrap();

        // 4 patches at merge=2 → (4/2)*(4/2) = 4 image tokens.
        assert_eq!(result.image_blocks, 1);
        assert_eq!(result.total_image_tokens, 4);

        // Layout: [BOS, vstart, img × 4, vend, original tail...]
        assert_eq!(prompt[0], 1);
        assert_eq!(prompt[1], 128_262);
        assert_eq!(&prompt[2..6], &[128_264; 4]);
        assert_eq!(prompt[6], 128_263);
        assert_eq!(&prompt[7..], &[200, 300]);
    }

    #[test]
    fn handles_multiple_images() {
        let mut prompt = vec![1i32, 200];
        let result = insert_youtu_vl_image_tokens(
            &mut prompt,
            &[(4, 4), (2, 2)],
            2,
            128_262,
            128_263,
            128_264,
        )
        .unwrap();
        // First image: 4 tokens, second image: 1 token, total = 5.
        assert_eq!(result.image_blocks, 2);
        assert_eq!(result.total_image_tokens, 5);
    }
}
