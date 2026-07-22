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

use super::{
    ImageTokenBlockAction, ImageTokenBlockInfo, ImageTokenBlockStats, apply_image_token_blocks,
};

#[test]
fn apply_image_token_blocks_expands_existing_tokens_with_boi_eoi() {
    let info = ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: 99,
        mm_tokens_per_image: 3,
        boi_token_id: 10,
        eoi_token_id: 11,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };
    let mut prompt_tokens = vec![1, 99, 2];

    let stats = apply_image_token_blocks(&mut prompt_tokens, info, 1).unwrap();

    assert_eq!(
        stats,
        Some(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Expanded {
                existing_image_count: 1,
            },
            tokens_per_image: 3,
        })
    );
    assert_eq!(prompt_tokens, vec![1, 10, 99, 99, 99, 11, 2]);
}

#[test]
fn apply_image_token_blocks_inserts_blocks_after_bos() {
    let info = ImageTokenBlockInfo {
        use_boi_eoi: false,
        image_token_id: 77,
        mm_tokens_per_image: 2,
        boi_token_id: 0,
        eoi_token_id: 0,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };
    let mut prompt_tokens = vec![1, 2, 3];

    let stats = apply_image_token_blocks(&mut prompt_tokens, info, 2).unwrap();

    assert_eq!(
        stats,
        Some(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Inserted { image_blocks: 2 },
            tokens_per_image: 2,
        })
    );
    assert_eq!(prompt_tokens, vec![1, 77, 77, 77, 77, 2, 3]);
}

#[test]
fn apply_image_token_blocks_paligemma_format() {
    // PaliGemma: [img*N, BOS(2), text, \n(108)]
    let info = ImageTokenBlockInfo {
        use_boi_eoi: false,
        image_token_id: 257152,
        mm_tokens_per_image: 3,
        boi_token_id: 0,
        eoi_token_id: 0,
        has_bos: false,
        separator_token_id: Some(2), // BOS between images and text
        suffix_tokens: vec![108],    // newline after text
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };
    let mut prompt_tokens = vec![100, 200, 300]; // text tokens only, no BOS

    let stats = apply_image_token_blocks(&mut prompt_tokens, info, 1).unwrap();

    assert_eq!(
        stats,
        Some(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Inserted { image_blocks: 1 },
            tokens_per_image: 3,
        })
    );
    // [img*3, BOS(2), text, \n(108)]
    assert_eq!(
        prompt_tokens,
        vec![257152, 257152, 257152, 2, 100, 200, 300, 108]
    );
}

#[test]
fn apply_image_token_blocks_is_noop_without_prompt_or_images() {
    let info = ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: 5,
        mm_tokens_per_image: 4,
        boi_token_id: 6,
        eoi_token_id: 7,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };

    let mut empty_prompt = Vec::new();
    assert_eq!(
        apply_image_token_blocks(&mut empty_prompt, info.clone(), 1),
        Err(super::ImageTokenBlockError::EmptyPrompt { image_count: 1 })
    );

    let mut prompt_tokens = vec![1, 2, 3];
    assert_eq!(
        apply_image_token_blocks(&mut prompt_tokens, info, 0),
        Ok(None)
    );
    assert_eq!(prompt_tokens, vec![1, 2, 3]);
}

#[test]
fn apply_image_token_blocks_expands_with_block_prefix_and_suffix() {
    // Gemma3 VLM: each image block is wrapped with \n\n (token 108) prefix and suffix.
    // Input:  [BOS(1), <start_of_image>(10), text(2)]
    // Expect: [BOS(1), \n(108), BOI(10), img*3, EOI(11), \n(108), text(2)]
    let info = ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: 99,
        mm_tokens_per_image: 3,
        boi_token_id: 10,
        eoi_token_id: 11,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: vec![108],
        block_suffix_tokens: vec![108],
    };
    // Prompt has a BOI token (10) that should be expanded
    let mut prompt_tokens = vec![1, 10, 2];

    let stats = apply_image_token_blocks(&mut prompt_tokens, info, 1).unwrap();

    assert_eq!(
        stats,
        Some(ImageTokenBlockStats {
            action: ImageTokenBlockAction::Expanded {
                existing_image_count: 1,
            },
            tokens_per_image: 3,
        })
    );
    // prefix(108) + BOI(10) + img*3(99,99,99) + EOI(11) + suffix(108)
    assert_eq!(prompt_tokens, vec![1, 108, 10, 99, 99, 99, 11, 108, 2]);
}

#[test]
fn apply_image_token_blocks_rejects_media_cardinality_mismatch() {
    let info = ImageTokenBlockInfo {
        use_boi_eoi: false,
        image_token_id: 99,
        mm_tokens_per_image: 2,
        boi_token_id: 0,
        eoi_token_id: 0,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };
    let mut prompt_tokens = vec![1, 99, 2];

    let error = apply_image_token_blocks(&mut prompt_tokens, info, 2).unwrap_err();

    assert_eq!(
        error,
        super::ImageTokenBlockError::MediaCardinality {
            placeholder_count: 1,
            image_count: 2,
        }
    );
    assert_eq!(prompt_tokens, vec![1, 99, 2]);
}

#[test]
fn apply_image_token_blocks_rejects_insert_capacity_overflow() {
    let info = ImageTokenBlockInfo {
        use_boi_eoi: true,
        image_token_id: 99,
        mm_tokens_per_image: 2,
        boi_token_id: 10,
        eoi_token_id: 11,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
    };
    let mut prompt_tokens = vec![1, 2];

    let error = apply_image_token_blocks(&mut prompt_tokens, info, usize::MAX).unwrap_err();

    assert_eq!(error, super::ImageTokenBlockError::CapacityOverflow);
    assert_eq!(prompt_tokens, vec![1, 2]);
}
