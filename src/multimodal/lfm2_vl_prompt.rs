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
//! `[<|image_start|>?, tile_marker?, <image> * T_view..., thumbnail_marker?, <|image_end|>?]` where
//! `T_view = ceil(h_view / f) * ceil(w_view / f)` is each view's post-downsample
//! token count. Tiled images use `<|img_row_r_col_c|>` markers before each
//! row-major tile and `<|img_thumbnail|>` before the optional thumbnail. The
//! framing tokens are emitted only when `use_image_special_tokens` is set and
//! the tokenizer exposes them; the merge only replaces the `<image>` rows, so
//! the invariant that matters is that the count of 396-tokens equals the number
//! of image-feature rows.

use crate::vision::processors::lfm2_vl::Lfm2VlImageLayout;

/// Statistics describing the LFM2-VL image-token expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedLfm2VlTokens {
    pub image_blocks: usize,
    pub total_image_tokens: usize,
}

fn view_token_count((h, w): (i32, i32), downsample_factor: i32) -> usize {
    let f = downsample_factor.max(1);
    (((h + f - 1) / f) * ((w + f - 1) / f)).max(0) as usize
}

fn append_image_run(block: &mut Vec<i32>, image_token_id: i32, count: usize) {
    block.extend(std::iter::repeat_n(image_token_id, count));
}

fn build_block(
    image_token_id: i32,
    image_start_id: i32,
    image_end_id: i32,
    img_row_col_ids: &[[i32; 10]; 10],
    thumbnail_id: i32,
    use_special: bool,
    layout: &Lfm2VlImageLayout,
    downsample_factor: i32,
) -> Vec<i32> {
    let total_count: usize = layout
        .views
        .iter()
        .map(|&grid| view_token_count(grid, downsample_factor))
        .sum();
    let mut block = Vec::with_capacity(total_count + 2 + layout.views.len());
    if use_special && image_start_id != 0 {
        block.push(image_start_id);
    }

    if layout.is_tiled() {
        let tile_count = layout.tile_count().min(layout.views.len());
        for tile_idx in 0..tile_count {
            let row = tile_idx / layout.cols;
            let col = tile_idx % layout.cols;
            if use_special {
                let marker = img_row_col_ids
                    .get(row)
                    .and_then(|ids| ids.get(col))
                    .copied()
                    .unwrap_or(0);
                if marker != 0 {
                    block.push(marker);
                }
            }
            append_image_run(
                &mut block,
                image_token_id,
                view_token_count(layout.views[tile_idx], downsample_factor),
            );
        }
        if layout.has_thumbnail() {
            if use_special && thumbnail_id != 0 {
                block.push(thumbnail_id);
            }
            append_image_run(
                &mut block,
                image_token_id,
                view_token_count(layout.views[tile_count], downsample_factor),
            );
        }
    } else if let Some(&grid) = layout.views.first() {
        append_image_run(
            &mut block,
            image_token_id,
            view_token_count(grid, downsample_factor),
        );
    }

    if use_special && image_end_id != 0 {
        block.push(image_end_id);
    }
    block
}

/// Expand (or splice) LFM2-VL image-token runs into `prompt_tokens`. Per-image
/// layouts come from the processor; each layout may contain one view or a
/// row-major tile grid plus an optional thumbnail view.
pub fn insert_lfm2_vl_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    layouts: &[Lfm2VlImageLayout],
    downsample_factor: i32,
    image_token_id: i32,
    image_start_id: i32,
    image_end_id: i32,
    img_row_col_ids: &[[i32; 10]; 10],
    thumbnail_id: i32,
    use_special: bool,
) -> Option<InsertedLfm2VlTokens> {
    if prompt_tokens.is_empty() || layouts.is_empty() {
        return None;
    }
    let total_image_tokens: usize = layouts
        .iter()
        .flat_map(|layout| layout.views.iter())
        .map(|&grid| view_token_count(grid, downsample_factor))
        .sum();
    let image_blocks = layouts.len();

    // Case 1: expand each bare <image> placeholder in place (one per logical image).
    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();
    if placeholder_count > 0 {
        let mut expanded = Vec::with_capacity(prompt_tokens.len() + total_image_tokens);
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == image_token_id && image_idx < layouts.len() {
                expanded.extend(build_block(
                    image_token_id,
                    image_start_id,
                    image_end_id,
                    img_row_col_ids,
                    thumbnail_id,
                    use_special,
                    &layouts[image_idx],
                    downsample_factor,
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
    for layout in layouts {
        blocks.extend(build_block(
            image_token_id,
            image_start_id,
            image_end_id,
            img_row_col_ids,
            thumbnail_id,
            use_special,
            layout,
            downsample_factor,
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
    const THUMB: i32 = 497;

    fn marker_ids() -> [[i32; 10]; 10] {
        let mut ids = [[0; 10]; 10];
        for (row_idx, row) in ids.iter_mut().enumerate() {
            for (col_idx, id) in row.iter_mut().enumerate() {
                *id = 397 + (row_idx as i32) * 10 + col_idx as i32;
            }
        }
        ids
    }

    fn layout(views: Vec<(i32, i32)>, rows: usize, cols: usize) -> Lfm2VlImageLayout {
        Lfm2VlImageLayout { views, rows, cols }
    }

    #[test]
    fn expands_placeholder_with_framing() {
        // grid (4, 6), f=2 -> T = 2*3 = 6.
        let mut prompt = vec![1, IMAGE, 2];
        let stats = insert_lfm2_vl_image_tokens(
            &mut prompt,
            &[layout(vec![(4, 6)], 1, 1)],
            2,
            IMAGE,
            START,
            END,
            &marker_ids(),
            THUMB,
            true,
        )
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
        insert_lfm2_vl_image_tokens(
            &mut prompt,
            &[layout(vec![(2, 2)], 1, 1)],
            2,
            IMAGE,
            START,
            END,
            &marker_ids(),
            THUMB,
            false,
        )
        .unwrap();
        // grid (2,2), f=2 -> T=1. No start/end.
        assert_eq!(prompt, vec![1, IMAGE, 2]);
    }

    #[test]
    fn per_image_counts_and_odd_grid_ceil() {
        // grid (3,5), f=2 -> ceil(3/2)*ceil(5/2) = 2*3 = 6; grid (2,2) -> 1.
        let mut prompt = vec![1, IMAGE, IMAGE, 2];
        let stats = insert_lfm2_vl_image_tokens(
            &mut prompt,
            &[layout(vec![(3, 5)], 1, 1), layout(vec![(2, 2)], 1, 1)],
            2,
            IMAGE,
            0,
            0,
            &marker_ids(),
            THUMB,
            true,
        )
        .unwrap();
        assert_eq!(stats.total_image_tokens, 7);
        assert_eq!(prompt.iter().filter(|&&t| t == IMAGE).count(), 7);
    }

    #[test]
    fn prompt_emits_row_col_markers() {
        let mut prompt = vec![1, IMAGE, 2];
        let layouts = [layout(vec![(32, 32); 6], 2, 3)];
        let stats = insert_lfm2_vl_image_tokens(
            &mut prompt,
            &layouts,
            2,
            IMAGE,
            START,
            END,
            &marker_ids(),
            THUMB,
            true,
        )
        .unwrap();
        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 6 * 256);
        assert_eq!(prompt[0], 1);
        assert_eq!(prompt[1], START);
        assert_eq!(prompt[2], 397);
        assert_eq!(&prompt[3..259], &[IMAGE; 256]);
        assert_eq!(prompt[259], 398);
        assert_eq!(&prompt[260..516], &[IMAGE; 256]);
        assert_eq!(prompt[516], 399);
        assert_eq!(prompt[773], 407);
        assert_eq!(prompt[1030], 408);
        assert_eq!(prompt[1287], 409);
        assert_eq!(prompt[prompt.len() - 2], END);
        assert_eq!(prompt[prompt.len() - 1], 2);
    }

    #[test]
    fn prompt_emits_thumbnail_after_tiles() {
        let mut prompt = vec![1, IMAGE, 2];
        let layouts = [layout(vec![(32, 32), (32, 32), (8, 8)], 1, 2)];
        insert_lfm2_vl_image_tokens(
            &mut prompt,
            &layouts,
            2,
            IMAGE,
            START,
            END,
            &marker_ids(),
            THUMB,
            true,
        )
        .unwrap();
        assert!(prompt.windows(1).any(|w| w[0] == THUMB));
        let thumb_pos = prompt.iter().position(|&id| id == THUMB).unwrap();
        assert_eq!(prompt[thumb_pos + 1..thumb_pos + 17], [IMAGE; 16]);
    }
}
