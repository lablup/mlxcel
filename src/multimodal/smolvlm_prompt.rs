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
//! Mirrors the upstream Idefics3 / SmolVLM processor's text rewrite. Split
//! images expand into row-major tile runs framed as
//! `<fake_token_around_image><row_r_col_c><image>*N`, one row per line, followed
//! by a blank line and the final global thumbnail run
//! `<fake_token_around_image><global-img><image>*N<fake_token_around_image>`.
//! Single-tile images keep the legacy global block. Marker text is tokenized
//! without adding special tokens so checkpoints whose row/global strings are not
//! added tokens follow the reference BPE path.
//!
//! When the rendered prompt already carries bare `<image>` placeholders (the
//! common case for a chat template that emits one `<image>` per image), each is
//! expanded in place. Otherwise one image block per image is spliced in after
//! the first prompt token.

use crate::vision::processors::smolvlm::TileLayout;

const FAKE_IMAGE_MARKER: &str = "<fake_token_around_image>";
const GLOBAL_IMAGE_MARKER: &str = "<global-img>";

/// Statistics describing the SmolVLM image-token insertion/expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertedSmolVlmTokens {
    pub image_blocks: usize,
    pub total_image_tokens: usize,
}

fn append_repeated_image_tokens(block: &mut Vec<i32>, image_token_id: i32, count: usize) {
    block.extend(std::iter::repeat_n(image_token_id, count));
}

fn build_single_smolvlm_block<E>(
    image_token_id: i32,
    num_image_token: usize,
    encode_marker: &mut E,
) -> Option<Vec<i32>>
where
    E: FnMut(&str) -> Vec<i32>,
{
    let mut block = Vec::with_capacity(num_image_token.checked_add(4)?);
    block.extend(encode_marker(&format!(
        "{FAKE_IMAGE_MARKER}{GLOBAL_IMAGE_MARKER}"
    )));
    append_repeated_image_tokens(&mut block, image_token_id, num_image_token);
    block.extend(encode_marker(FAKE_IMAGE_MARKER));
    Some(block)
}

fn build_split_smolvlm_block<E>(
    image_token_id: i32,
    num_image_token: usize,
    layout: TileLayout,
    encode_marker: &mut E,
) -> Option<Vec<i32>>
where
    E: FnMut(&str) -> Vec<i32>,
{
    let split_tiles = layout.rows.checked_mul(layout.cols)?;
    let total_image_tokens = layout.checked_total_tiles()?.checked_mul(num_image_token)?;
    let marker_capacity = split_tiles
        .checked_mul(3)?
        .checked_add(layout.rows)?
        .checked_add(4)?;
    let mut block = Vec::with_capacity(total_image_tokens.checked_add(marker_capacity)?);
    for row in 1..=layout.rows {
        for col in 1..=layout.cols {
            block.extend(encode_marker(&format!(
                "{FAKE_IMAGE_MARKER}<row_{row}_col_{col}>"
            )));
            append_repeated_image_tokens(&mut block, image_token_id, num_image_token);
        }
        block.extend(encode_marker("\n"));
    }
    block.extend(encode_marker(&format!(
        "\n{FAKE_IMAGE_MARKER}{GLOBAL_IMAGE_MARKER}"
    )));
    append_repeated_image_tokens(&mut block, image_token_id, num_image_token);
    block.extend(encode_marker(FAKE_IMAGE_MARKER));
    Some(block)
}

fn build_smolvlm_block<E>(
    image_token_id: i32,
    num_image_token: usize,
    layout: TileLayout,
    encode_marker: &mut E,
) -> Option<Vec<i32>>
where
    E: FnMut(&str) -> Vec<i32>,
{
    if layout.is_split() {
        build_split_smolvlm_block(image_token_id, num_image_token, layout, encode_marker)
    } else {
        build_single_smolvlm_block(image_token_id, num_image_token, encode_marker)
    }
}

fn image_tokens_for_layout(layout: TileLayout, num_image_token: usize) -> Option<usize> {
    layout.checked_total_tiles()?.checked_mul(num_image_token)
}

/// Insert (or expand) SmolVLM image-token runs into `prompt_tokens`.
///
/// `layouts[i]` is the tile layout for image `i`; split layouts emit
/// `rows * cols + 1` image-token runs and single layouts emit one run.
///
/// `encode_marker` must tokenize marker/newline text with `add_special_tokens = false`.
///
/// Returns `None` when there is nothing to do (empty prompt or no images).
pub fn insert_smolvlm_image_tokens<E>(
    prompt_tokens: &mut Vec<i32>,
    layouts: &[TileLayout],
    num_image_token: usize,
    image_token_id: i32,
    mut encode_marker: E,
) -> Option<InsertedSmolVlmTokens>
where
    E: FnMut(&str) -> Vec<i32>,
{
    if prompt_tokens.is_empty() || layouts.is_empty() || num_image_token == 0 {
        return None;
    }

    let per_image_counts: Vec<usize> = layouts
        .iter()
        .copied()
        .map(|layout| image_tokens_for_layout(layout, num_image_token))
        .collect::<Option<Vec<_>>>()?;
    let total_image_tokens: usize = per_image_counts
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))?;
    let image_blocks = layouts.len();

    // Case 1: the prompt already carries bare <image> placeholders (one per
    // image). Expand each in place into a full framed block.
    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();
    if placeholder_count == layouts.len() {
        let mut expanded = Vec::with_capacity(prompt_tokens.len().checked_add(total_image_tokens)?);
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == image_token_id {
                expanded.extend(build_smolvlm_block(
                    image_token_id,
                    num_image_token,
                    layouts[image_idx],
                    &mut encode_marker,
                )?);
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
    if placeholder_count != 0 {
        return None;
    }

    // Case 2: no placeholder, so splice one block per image after the first
    // token (which typically opens the user turn).
    let mut blocks: Vec<i32> =
        Vec::with_capacity(total_image_tokens.checked_add(4usize.checked_mul(image_blocks)?)?);
    for &layout in layouts {
        blocks.extend(build_smolvlm_block(
            image_token_id,
            num_image_token,
            layout,
            &mut encode_marker,
        )?);
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

fn build_idefics2_block(image_token_id: i32, fake_image_token_id: i32, count: usize) -> Vec<i32> {
    let mut block = Vec::with_capacity(count + 2);
    if fake_image_token_id != 0 {
        block.push(fake_image_token_id);
    }
    append_repeated_image_tokens(&mut block, image_token_id, count);
    if fake_image_token_id != 0 {
        block.push(fake_image_token_id);
    }
    block
}

/// Insert Idefics2 single-tile image runs without SmolVLM row/global framing.
pub fn insert_idefics2_image_tokens(
    prompt_tokens: &mut Vec<i32>,
    tiles_per_image: &[usize],
    num_image_token: usize,
    image_token_id: i32,
    fake_image_token_id: i32,
) -> Option<InsertedSmolVlmTokens> {
    if prompt_tokens.is_empty() || tiles_per_image.is_empty() || num_image_token == 0 {
        return None;
    }

    let per_image_counts: Vec<usize> = tiles_per_image
        .iter()
        .map(|&tiles| num_image_token.checked_mul(tiles))
        .collect::<Option<Vec<_>>>()?;
    let total_image_tokens: usize = per_image_counts
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))?;
    let image_blocks = tiles_per_image.len();

    let placeholder_count = prompt_tokens
        .iter()
        .filter(|&&t| t == image_token_id)
        .count();
    if placeholder_count == tiles_per_image.len() {
        let mut expanded = Vec::with_capacity(prompt_tokens.len().checked_add(total_image_tokens)?);
        let mut image_idx = 0usize;
        for &token in prompt_tokens.iter() {
            if token == image_token_id {
                expanded.extend(build_idefics2_block(
                    image_token_id,
                    fake_image_token_id,
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
    if placeholder_count != 0 {
        return None;
    }

    let mut blocks: Vec<i32> =
        Vec::with_capacity(total_image_tokens.checked_add(2usize.checked_mul(image_blocks)?)?);
    for &count in &per_image_counts {
        blocks.extend(build_idefics2_block(
            image_token_id,
            fake_image_token_id,
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

    fn fake_encode(text: &str) -> Vec<i32> {
        match text {
            "<fake_token_around_image><global-img>" => vec![1, 2],
            "<fake_token_around_image>" => vec![1],
            "<fake_token_around_image><row_1_col_1>" => vec![1, 11],
            "<fake_token_around_image><row_1_col_2>" => vec![1, 12],
            "<fake_token_around_image><row_2_col_1>" => vec![1, 21],
            "<fake_token_around_image><row_2_col_2>" => vec![1, 22],
            "\n" => vec![10],
            "\n<fake_token_around_image><global-img>" => vec![10, 1, 2],
            other => panic!("unexpected marker text: {other:?}"),
        }
    }

    #[test]
    fn split_layout_emits_row_col_markers_and_newlines() {
        let mut prompt = vec![100, IMAGE, 200];
        let stats = insert_smolvlm_image_tokens(
            &mut prompt,
            &[TileLayout::split(2, 2)],
            3,
            IMAGE,
            fake_encode,
        )
        .unwrap();

        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 15);
        assert_eq!(
            prompt,
            vec![
                100, 1, 11, IMAGE, IMAGE, IMAGE, 1, 12, IMAGE, IMAGE, IMAGE, 10, 1, 21, IMAGE,
                IMAGE, IMAGE, 1, 22, IMAGE, IMAGE, IMAGE, 10, 10, 1, 2, IMAGE, IMAGE, IMAGE, 1,
                200,
            ]
        );
    }

    #[test]
    fn single_layout_keeps_global_block() {
        let mut prompt = vec![1, IMAGE, 2, 3];
        let stats = insert_smolvlm_image_tokens(
            &mut prompt,
            &[TileLayout::single()],
            4,
            IMAGE,
            fake_encode,
        )
        .unwrap();

        assert_eq!(stats.image_blocks, 1);
        assert_eq!(stats.total_image_tokens, 4);
        assert_eq!(prompt, vec![1, 1, 2, IMAGE, IMAGE, IMAGE, IMAGE, 1, 2, 3]);
    }

    #[test]
    fn splices_block_after_first_token_when_no_placeholder() {
        let mut prompt = vec![100, 200, 300];
        let stats = insert_smolvlm_image_tokens(
            &mut prompt,
            &[TileLayout::single()],
            3,
            IMAGE,
            fake_encode,
        )
        .unwrap();
        assert_eq!(stats.total_image_tokens, 3);
        assert_eq!(prompt, vec![100, 1, 2, IMAGE, IMAGE, IMAGE, 1, 200, 300]);
    }

    #[test]
    fn image_token_count_equals_tiles_times_n() {
        let mut prompt = vec![1, 2];
        let stats = insert_smolvlm_image_tokens(
            &mut prompt,
            &[TileLayout::single(), TileLayout::split(2, 2)],
            4,
            IMAGE,
            fake_encode,
        )
        .unwrap();
        assert_eq!(stats.image_blocks, 2);
        assert_eq!(stats.total_image_tokens, 24);
        assert_eq!(prompt.iter().filter(|&&t| t == IMAGE).count(), 24);
    }

    #[test]
    fn smolvlm_mismatched_placeholder_count_returns_none() {
        let mut prompt = vec![1, IMAGE, 2];
        let original = prompt.clone();
        assert!(
            insert_smolvlm_image_tokens(
                &mut prompt,
                &[TileLayout::single(), TileLayout::single()],
                4,
                IMAGE,
                fake_encode,
            )
            .is_none()
        );
        assert_eq!(prompt, original);
    }

    #[test]
    fn idefics2_keeps_fake_only_single_tile_framing() {
        let mut prompt = vec![1, IMAGE, 2];
        let stats = insert_idefics2_image_tokens(&mut prompt, &[1], 3, IMAGE, FAKE).unwrap();
        assert_eq!(stats.total_image_tokens, 3);
        assert_eq!(prompt, vec![1, FAKE, IMAGE, IMAGE, IMAGE, FAKE, 2]);
    }

    #[test]
    fn idefics2_mismatched_placeholder_count_returns_none() {
        let mut prompt = vec![1, IMAGE, 2, IMAGE, 3];
        let original = prompt.clone();
        assert!(insert_idefics2_image_tokens(&mut prompt, &[1], 3, IMAGE, FAKE).is_none());
        assert_eq!(prompt, original);
    }

    #[test]
    fn returns_none_for_empty_inputs() {
        let mut prompt: Vec<i32> = vec![];
        assert!(
            insert_smolvlm_image_tokens(
                &mut prompt,
                &[TileLayout::single()],
                4,
                IMAGE,
                fake_encode,
            )
            .is_none()
        );
        let mut prompt = vec![1, 2, 3];
        assert!(insert_smolvlm_image_tokens(&mut prompt, &[], 4, IMAGE, fake_encode).is_none());
    }
}
