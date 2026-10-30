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

const IMAGE_START: i32 = 248_078;
const IMAGE_END: i32 = 248_079;
const IMAGE_PAD: i32 = 248_056;

fn block(index: usize, size: usize) -> String {
    format!(
        "<image_id>{index}</image_id><image>{}</image>",
        "<|image_pad|>".repeat(size)
    )
}

#[test]
fn template_marker_expands_in_place() {
    // The checkpoint template's render of `[image, text]`.
    let prompt = "<|im_start|>user\n<|image_pad|>\nQ<|im_end|>\n<|im_start|>assistant\n";
    let text = expand_minicpmv46_image_placeholders(prompt, &[3]).unwrap();
    assert_eq!(
        text,
        format!(
            "<|im_start|>user\n{}\nQ<|im_end|>\n<|im_start|>assistant\n",
            block(0, 3)
        )
    );
}

#[test]
fn inline_markers_keep_order_and_number_images_from_zero() {
    let prompt = "<|im_start|>user\n(<image>./</image>)\nA <|image_pad|> B<|im_end|>\n";
    let text = expand_minicpmv46_image_placeholders(prompt, &[2, 1]).unwrap();
    assert_eq!(
        text,
        format!(
            "<|im_start|>user\n({})\nA {} B<|im_end|>\n",
            block(0, 2),
            block(1, 1)
        )
    );
}

#[test]
fn user_turn_without_marker_gets_blocks_and_newlines_at_its_head() {
    let prompt =
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n";
    let text = expand_minicpmv46_image_placeholders(prompt, &[2, 2]).unwrap();
    assert_eq!(
        text,
        format!(
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n{}\n{}\nQ<|im_end|>\n<|im_start|>assistant\n",
            block(0, 2),
            block(1, 2)
        )
    );
}

#[test]
fn marker_count_must_match_images() {
    let err =
        expand_minicpmv46_image_placeholders("<|image_pad|> and <|image_pad|>", &[4]).unwrap_err();
    assert!(err.contains("2 image marker(s) but 1 image(s)"), "{err}");
    assert_eq!(
        expand_minicpmv46_image_placeholders("text only", &[]).unwrap(),
        "text only"
    );
}

/// Issue #2547 regression pin, with the real `minicpm-v-4.6` token ids.
///
/// The local checkpoint copy has no chat template, so the prompt reaching the
/// processor is the bare question. The finished prompt is the checkpoint
/// template's one-image user turn (ChatML, image item then question joined by
/// `\n`) with the transformers block for the 448x448 image: 82 tokens. The
/// MiniCPM-o rule this family used before produced 78, without the
/// `<image_id>0</image_id>` prefix and the newline, and with `<unk>` filler.
#[test]
fn minicpmv46_untemplated_prompt_matches_reference_ids() {
    let expected_text = format!(
        "<|im_start|>user\n{}\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n",
        block(0, 64)
    );
    let mut expected_ids = vec![248_045, 846, 198, 248_090, 15, 248_091, IMAGE_START];
    expected_ids.extend(std::iter::repeat_n(IMAGE_PAD, 64));
    expected_ids.extend([
        IMAGE_END, 198, 72240, 279, 2099, 13, 248_046, 198, 248_045, 74455, 198,
    ]);
    let ids_for_text = expected_ids.clone();
    let mut encode = |text: &str, _add_special: bool| -> Vec<i32> {
        match text {
            "<image>" => vec![IMAGE_START],
            "</image>" => vec![IMAGE_END],
            t if t == expected_text => ids_for_text.clone(),
            other => panic!("unexpected encode of {other:?}"),
        }
    };

    let prepared =
        prepare_minicpmv46_prompt_tokens("Describe the image.", &[64], &mut encode).unwrap();

    assert_eq!(prepared.tokens.len(), 82);
    assert_eq!(prepared.tokens, expected_ids);
    assert_eq!(prepared.image_slots, 1);
    // Feature rows land on the 64 pads between `<image>` (index 6) and `</image>`.
    assert_eq!(prepared.image_bounds, vec![(7, 71)]);
}

#[test]
fn span_size_mismatch_is_an_error() {
    // A tokenizer that splits the filler would shift every feature row.
    let mut encode = |text: &str, _add_special: bool| -> Vec<i32> {
        match text {
            "<image>" => vec![IMAGE_START],
            "</image>" => vec![IMAGE_END],
            _ => vec![IMAGE_START, IMAGE_PAD, IMAGE_END],
        }
    };
    let err = prepare_minicpmv46_prompt_tokens("Q", &[2], &mut encode).unwrap_err();
    assert!(
        err.contains("holds 1 token(s) but the image has 2"),
        "{err}"
    );
}
