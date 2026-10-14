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

use super::*;

const VSTART: i32 = 128_262;
const VEND: i32 = 128_263;
const IMG: i32 = 128_264;

/// The prompt `tencent/Youtu-VL-4B-Instruct`'s chat template renders for one
/// image and a question, as token ids: system turn, then a user turn that
/// opens with `<|vision_start|><|image_pad|><|vision_end|>`.
fn templated_prompt() -> Vec<i32> {
    vec![
        128_000, 12_864, 198, 2_836, 128_001, 198, 128_000, 1_178, 198, VSTART, IMG, VEND, 3_753,
        21_314, 13, 128_001, 198, 128_000, 122_596, 198,
    ]
}

#[test]
fn leaves_an_already_expanded_prompt_unchanged() {
    let mut prompt = vec![1, VSTART, IMG, IMG, IMG, IMG, VEND, 200];
    let before = prompt.clone();
    let result = insert_youtu_vl_image_tokens(&mut prompt, &[(4, 4)], 2, VSTART, VEND, IMG);
    assert_eq!(result, Ok(None));
    assert_eq!(prompt, before);
}

#[test]
fn splices_image_tokens_after_bos_when_the_prompt_has_no_placeholder() {
    let mut prompt = vec![1i32, 200, 300];
    let result = insert_youtu_vl_image_tokens(&mut prompt, &[(4, 4)], 2, VSTART, VEND, IMG)
        .unwrap()
        .unwrap();

    // 4 patches at merge=2 → (4/2)*(4/2) = 4 image tokens.
    assert_eq!(result.image_blocks, 1);
    assert_eq!(result.total_image_tokens, 4);
    assert_eq!(prompt, vec![1, VSTART, IMG, IMG, IMG, IMG, VEND, 200, 300]);
}

#[test]
fn handles_multiple_images_without_placeholders() {
    let mut prompt = vec![1i32, 200];
    let result = insert_youtu_vl_image_tokens(&mut prompt, &[(4, 4), (2, 2)], 2, VSTART, VEND, IMG)
        .unwrap()
        .unwrap();
    // First image: 4 tokens, second image: 1 token, total = 5.
    assert_eq!(result.image_blocks, 2);
    assert_eq!(result.total_image_tokens, 5);
    assert_eq!(prompt.iter().filter(|&&t| t == IMG).count(), 5);
}

/// Issue #1618. The chat template leaves one `<|image_pad|>` per image; the
/// checkpoint's processor replaces it with one token per merged feature. A
/// 28x28 patch grid is the `test_image_shapes.png` fixture: a 14x14 merged
/// grid that spans 2x2 attention windows of 8x8, so 196 features must each
/// get their own slot, not just the first.
#[test]
fn expands_the_template_placeholder_to_one_token_per_merged_feature() {
    let mut prompt = templated_prompt();
    let pad_at = prompt.iter().position(|&t| t == IMG).unwrap();
    let head = prompt[..pad_at].to_vec();
    let tail = prompt[pad_at + 1..].to_vec();

    let result = insert_youtu_vl_image_tokens(&mut prompt, &[(28, 28)], 2, VSTART, VEND, IMG)
        .unwrap()
        .unwrap();

    assert_eq!(result.image_blocks, 1);
    assert_eq!(result.total_image_tokens, 196);
    assert_eq!(prompt.len(), head.len() + 196 + tail.len());
    // The run sits exactly where the placeholder was, inside the template's
    // own framing; no second start/end pair and no text moved.
    assert_eq!(&prompt[..head.len()], head.as_slice());
    assert!(
        prompt[head.len()..head.len() + 196]
            .iter()
            .all(|&t| t == IMG)
    );
    assert_eq!(&prompt[head.len() + 196..], tail.as_slice());
    assert_eq!(prompt.iter().filter(|&&t| t == VSTART).count(), 1);
    assert_eq!(prompt.iter().filter(|&&t| t == VEND).count(), 1);
}

#[test]
fn expands_each_placeholder_with_its_own_image_run_in_order() {
    let mut prompt = vec![1, VSTART, IMG, VEND, 50, VSTART, IMG, VEND, 60];
    let result =
        insert_youtu_vl_image_tokens(&mut prompt, &[(22, 22), (4, 2)], 2, VSTART, VEND, IMG)
            .unwrap()
            .unwrap();
    assert_eq!(result.total_image_tokens, 121 + 2);

    let mut expected = vec![1, VSTART];
    expected.extend(std::iter::repeat_n(IMG, 121));
    expected.extend([VEND, 50, VSTART]);
    expected.extend(std::iter::repeat_n(IMG, 2));
    expected.extend([VEND, 60]);
    assert_eq!(prompt, expected);
}

#[test]
fn rejects_a_placeholder_count_that_matches_neither_images_nor_features() {
    let mut prompt = vec![1, IMG, IMG, 200];
    let before = prompt.clone();
    let result = insert_youtu_vl_image_tokens(&mut prompt, &[(28, 28)], 2, VSTART, VEND, IMG);
    assert_eq!(
        result,
        Err(YoutuVlPlaceholderMismatch {
            found: 2,
            images: 1,
            expected: 196,
        })
    );
    assert_eq!(prompt, before);
}
