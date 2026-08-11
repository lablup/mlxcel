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

//! Muse Glimmer prompt placeholder expansion.
//!
//! The pinned checkpoint template renders one `<|patch|>` (`200092`) marker per
//! image. Legacy/direct callers may instead provide `<|image|>` (`200090`). The
//! processor accepts exactly one unambiguous marker spelling per image and
//! expands it to `<|image_start|> <|patch|>*N <|image_end|>`, where
//! `N = t * (grid_h / merge) * (grid_w / merge)`.

use crate::models::{
    DEFAULT_IMAGE_END_TOKEN_ID, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, DEFAULT_IMAGE_START_TOKEN_ID,
    DEFAULT_IMAGE_TOKEN_ID,
};
use crate::vision::processors::muse_glimmer::merged_visual_token_count;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MuseGlimmerPromptTokens {
    pub image_placeholder_token_id: i32,
    pub image_start_token_id: i32,
    pub image_token_id: i32,
    pub image_end_token_id: i32,
}

impl Default for MuseGlimmerPromptTokens {
    fn default() -> Self {
        Self {
            image_placeholder_token_id: DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID,
            image_start_token_id: DEFAULT_IMAGE_START_TOKEN_ID,
            image_token_id: DEFAULT_IMAGE_TOKEN_ID,
            image_end_token_id: DEFAULT_IMAGE_END_TOKEN_ID,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MuseGlimmerPromptStats {
    pub image_blocks: usize,
    pub image_tokens: usize,
    pub total_tokens: usize,
}

pub fn image_placeholder_tokens(grid: (i32, i32, i32), merge_size: usize) -> Result<usize, String> {
    merged_visual_token_count(grid, merge_size)
}

pub fn expand_muse_glimmer_image_placeholders(
    prompt_tokens: &mut Vec<i32>,
    image_grid_thw: &[(i32, i32, i32)],
    merge_size: usize,
    tokens: MuseGlimmerPromptTokens,
) -> Result<MuseGlimmerPromptStats, String> {
    let image_marker_count = prompt_tokens
        .iter()
        .filter(|&&id| id == tokens.image_placeholder_token_id)
        .count();
    let patch_marker_count = prompt_tokens
        .iter()
        .filter(|&&id| id == tokens.image_token_id)
        .count();
    let marker_id = match (image_marker_count, patch_marker_count) {
        (count, 0) if count == image_grid_thw.len() => tokens.image_placeholder_token_id,
        (0, count) if count == image_grid_thw.len() => tokens.image_token_id,
        _ => {
            return Err(format!(
                "Muse Glimmer prompt contains {image_marker_count} image placeholders and \
                 {patch_marker_count} template patch placeholders but {} images were processed",
                image_grid_thw.len()
            ));
        }
    };

    let per_image = image_grid_thw
        .iter()
        .map(|&grid| image_placeholder_tokens(grid, merge_size))
        .collect::<Result<Vec<_>, _>>()?;

    let mut expanded =
        Vec::with_capacity(prompt_tokens.len() + per_image.iter().map(|n| n + 1).sum::<usize>());
    let mut image_idx = 0;
    for &id in prompt_tokens.iter() {
        if id == marker_id {
            let count = per_image[image_idx];
            expanded.push(tokens.image_start_token_id);
            expanded.extend(std::iter::repeat_n(tokens.image_token_id, count));
            expanded.push(tokens.image_end_token_id);
            image_idx += 1;
        } else {
            expanded.push(id);
        }
    }
    *prompt_tokens = expanded;
    Ok(MuseGlimmerPromptStats {
        image_blocks: image_grid_thw.len(),
        image_tokens: per_image.iter().sum(),
        total_tokens: prompt_tokens.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_arithmetic_uses_merged_grid_tokens() {
        assert_eq!(image_placeholder_tokens((1, 32, 32), 2).unwrap(), 256);
        assert_eq!(image_placeholder_tokens((1, 4, 8), 2).unwrap(), 8);
        assert!(image_placeholder_tokens((1, 3, 8), 2).is_err());
    }

    #[test]
    fn expands_each_image_marker_in_order() {
        let tokens = MuseGlimmerPromptTokens::default();
        let mut prompt = vec![
            10,
            tokens.image_placeholder_token_id,
            11,
            tokens.image_placeholder_token_id,
        ];
        let stats =
            expand_muse_glimmer_image_placeholders(&mut prompt, &[(1, 4, 4), (1, 2, 2)], 2, tokens)
                .unwrap();
        assert_eq!(
            stats,
            MuseGlimmerPromptStats {
                image_blocks: 2,
                image_tokens: 5,
                total_tokens: 11,
            }
        );
        assert_eq!(
            prompt,
            vec![
                10,
                tokens.image_start_token_id,
                tokens.image_token_id,
                tokens.image_token_id,
                tokens.image_token_id,
                tokens.image_token_id,
                tokens.image_end_token_id,
                11,
                tokens.image_start_token_id,
                tokens.image_token_id,
                tokens.image_end_token_id,
            ]
        );
    }

    #[test]
    fn expands_pinned_template_patch_markers() {
        let tokens = MuseGlimmerPromptTokens::default();
        let mut prompt = vec![10, tokens.image_token_id, 11, tokens.image_token_id, 12];
        let stats =
            expand_muse_glimmer_image_placeholders(&mut prompt, &[(1, 2, 2), (1, 2, 4)], 2, tokens)
                .expect("pinned Muse template patch markers should expand");

        assert_eq!(stats.image_blocks, 2);
        assert_eq!(stats.image_tokens, 3);
        assert_eq!(
            prompt,
            vec![
                10,
                tokens.image_start_token_id,
                tokens.image_token_id,
                tokens.image_end_token_id,
                11,
                tokens.image_start_token_id,
                tokens.image_token_id,
                tokens.image_token_id,
                tokens.image_end_token_id,
                12,
            ]
        );
    }

    #[test]
    fn rejects_mixed_image_marker_spellings() {
        let tokens = MuseGlimmerPromptTokens::default();
        let mut prompt = vec![tokens.image_placeholder_token_id, tokens.image_token_id];
        let error =
            expand_muse_glimmer_image_placeholders(&mut prompt, &[(1, 2, 2), (1, 2, 2)], 2, tokens)
                .expect_err("mixed reserved marker spellings must be rejected");
        assert!(error.contains("1 image placeholders"));
        assert!(error.contains("1 template patch placeholders"));
    }

    #[test]
    fn rejects_placeholder_cardinality_mismatch() {
        let tokens = MuseGlimmerPromptTokens::default();
        let mut prompt = vec![tokens.image_placeholder_token_id];
        let err = expand_muse_glimmer_image_placeholders(&mut prompt, &[], 2, tokens).unwrap_err();
        assert!(err.contains("1 image placeholders"));
    }
}
