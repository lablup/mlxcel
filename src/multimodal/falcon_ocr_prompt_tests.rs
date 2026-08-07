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

//! Falcon-OCR prompt-expansion tests.

use super::*;

const OCR_PLAIN: i32 = 257;

fn ids() -> FalconOcrTokenIds {
    FalconOcrTokenIds {
        img_id: 227,
        image_cls_token_id: 244,
        img_end_id: 230,
        image_reg_token_ids: [245, 246, 247, 248],
    }
}

#[test]
fn a_plain_prompt_gets_the_block_prepended_and_the_task_token_appended() {
    let mut tokens = vec![900, 901];
    let stats = insert_falcon_ocr_image_tokens(&mut tokens, &[(2, 2)], &ids(), Some(OCR_PLAIN))
        .expect("expansion runs");
    assert_eq!(
        tokens,
        vec![
            244, 245, 246, 247, 248, 227, 227, 227, 227, 230, 900, 901, 257
        ]
    );
    assert_eq!(stats.image_blocks, 1);
    assert_eq!(stats.total_image_tokens, 4);
    assert!(stats.appended_task_token);
}

/// The reference substitutes `<|image|>` in place, so a caller that templated
/// the prompt itself keeps its ordering.
#[test]
fn an_existing_placeholder_is_expanded_in_place() {
    let mut tokens = vec![800, 227, 900];
    let stats = insert_falcon_ocr_image_tokens(&mut tokens, &[(1, 2)], &ids(), Some(OCR_PLAIN))
        .expect("expansion runs");
    assert_eq!(
        tokens,
        vec![800, 244, 245, 246, 247, 248, 227, 227, 230, 900, 257]
    );
    assert_eq!(stats.total_image_tokens, 2);
}

#[test]
fn two_images_get_two_blocks_sized_independently() {
    let mut tokens = vec![900];
    let stats =
        insert_falcon_ocr_image_tokens(&mut tokens, &[(1, 1), (2, 1)], &ids(), Some(OCR_PLAIN))
            .expect("expansion runs");
    assert_eq!(
        tokens,
        vec![
            244, 245, 246, 247, 248, 227, 230, 244, 245, 246, 247, 248, 227, 227, 230, 900, 257
        ]
    );
    assert_eq!(stats.image_blocks, 2);
    assert_eq!(stats.total_image_tokens, 3);
}

/// Appending a second task token would change the prompt the model was trained
/// on, so an already-terminated prompt is left alone.
#[test]
fn a_prompt_that_already_ends_with_the_task_token_is_not_extended() {
    let mut tokens = vec![900, OCR_PLAIN];
    let stats = insert_falcon_ocr_image_tokens(&mut tokens, &[(1, 1)], &ids(), Some(OCR_PLAIN))
        .expect("expansion runs");
    assert_eq!(tokens.iter().filter(|&&t| t == OCR_PLAIN).count(), 1);
    assert_eq!(tokens.last(), Some(&OCR_PLAIN));
    assert!(!stats.appended_task_token);
}

#[test]
fn a_tokenizer_without_the_task_token_still_expands_the_image() {
    let mut tokens = vec![900];
    let stats =
        insert_falcon_ocr_image_tokens(&mut tokens, &[(1, 1)], &ids(), None).expect("expansion");
    assert_eq!(tokens, vec![244, 245, 246, 247, 248, 227, 230, 900]);
    assert!(!stats.appended_task_token);
}

/// A placeholder count that matches neither "one per image" nor zero means the
/// prompt was already expanded; touching it would double the blocks.
#[test]
fn a_mismatched_placeholder_count_is_left_untouched() {
    let mut tokens = vec![227, 227, 227, 900];
    let before = tokens.clone();
    assert!(
        insert_falcon_ocr_image_tokens(&mut tokens, &[(1, 1)], &ids(), Some(OCR_PLAIN)).is_none()
    );
    assert_eq!(tokens, before);
}

#[test]
fn no_images_means_no_edit() {
    let mut tokens = vec![900, 901];
    let before = tokens.clone();
    assert!(insert_falcon_ocr_image_tokens(&mut tokens, &[], &ids(), Some(OCR_PLAIN)).is_none());
    assert_eq!(tokens, before);
}
