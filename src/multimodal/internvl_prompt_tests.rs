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

const IMG_START: i32 = 151_665;
const IMG_CONTEXT: i32 = 151_667;
const IMG_END: i32 = 151_666;
const NEWLINE: i32 = 198;

/// A deterministic stand-in tokenizer: each ChatML special token is one id and
/// every other character is its own id, so any prefix of a string encodes to a
/// prefix of the string's tokens, as a real BPE does at these boundaries.
fn toy_encode(text: &str, _add_special: bool) -> Vec<i32> {
    const SPECIALS: [(&str, i32); 2] = [("<|im_start|>", 151_644), ("<|im_end|>", 151_645)];
    let mut out = Vec::new();
    let mut rest = text;
    'outer: while !rest.is_empty() {
        for (special, id) in SPECIALS {
            if let Some(tail) = rest.strip_prefix(special) {
                out.push(id);
                rest = tail;
                continue 'outer;
            }
        }
        let ch = rest.chars().next().unwrap_or_default();
        out.push(if ch == '\n' {
            NEWLINE
        } else {
            1_000 + ch as i32
        });
        rest = &rest[ch.len_utf8()..];
    }
    out
}

fn block(count: usize) -> Vec<i32> {
    let mut block = vec![IMG_START];
    block.extend(std::iter::repeat_n(IMG_CONTEXT, count));
    block.push(IMG_END);
    block
}

#[test]
fn chatml_prompt_places_block_and_newline_after_last_user_opener() {
    let prompt =
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n";
    let head = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n";
    let mut tokens = toy_encode(prompt, true);
    let stats = insert_internvl_image_tokens(
        prompt,
        &mut tokens,
        &[1],
        4,
        IMG_START,
        IMG_CONTEXT,
        IMG_END,
        &mut toy_encode,
    )
    .unwrap();

    assert_eq!(stats.image_blocks, 1);
    assert_eq!(stats.total_image_tokens, 4);
    let mut expected = toy_encode(head, false);
    expected.extend(block(4));
    expected.push(NEWLINE);
    expected.extend(toy_encode(&prompt[head.len()..], false));
    assert_eq!(tokens, expected);
}

#[test]
fn raw_prompt_prepends_block_and_newline() {
    let prompt = "Describe the image.";
    let mut tokens = toy_encode(prompt, true);
    insert_internvl_image_tokens(
        prompt,
        &mut tokens,
        &[1],
        3,
        IMG_START,
        IMG_CONTEXT,
        IMG_END,
        &mut toy_encode,
    )
    .unwrap();

    let mut expected = block(3);
    expected.push(NEWLINE);
    expected.extend(toy_encode(prompt, false));
    assert_eq!(tokens, expected);
}

#[test]
fn each_image_block_gets_its_own_newline() {
    let prompt = "<|im_start|>user\nQ<|im_end|>\n";
    let mut tokens = toy_encode(prompt, true);
    let stats = insert_internvl_image_tokens(
        prompt,
        &mut tokens,
        &[1, 3],
        4,
        IMG_START,
        IMG_CONTEXT,
        IMG_END,
        &mut toy_encode,
    )
    .unwrap();

    // image 0: 1x4 = 4, image 1: 3x4 = 12, total 16.
    assert_eq!(stats.image_blocks, 2);
    assert_eq!(stats.total_image_tokens, 16);
    let mut expected = toy_encode("<|im_start|>user\n", false);
    expected.extend(block(4));
    expected.push(NEWLINE);
    expected.extend(block(12));
    expected.push(NEWLINE);
    expected.extend(toy_encode("Q<|im_end|>\n", false));
    assert_eq!(tokens, expected);
}

#[test]
fn existing_placeholder_expands_in_place_without_extra_newline() {
    // A transformers-style template renders `<IMG_CONTEXT>\n` itself.
    let mut prompt = vec![151_644, IMG_CONTEXT, NEWLINE, 300];
    let stats = insert_internvl_image_tokens(
        "",
        &mut prompt,
        &[2],
        4,
        IMG_START,
        IMG_CONTEXT,
        IMG_END,
        &mut toy_encode,
    )
    .unwrap();

    // 2 tiles x 4 tokens = 8 context tokens.
    assert_eq!(stats.total_image_tokens, 8);
    let mut expected = vec![151_644];
    expected.extend(block(8));
    expected.extend([NEWLINE, 300]);
    assert_eq!(prompt, expected);
}

#[test]
fn tokenizer_disagreement_degrades_to_prepending() {
    // The rendered text has a user turn, but the caller's tokens do not start
    // with its encoding, so the block must not land at a guessed offset.
    let prompt = "<|im_start|>user\nQ";
    let mut tokens = vec![7, 8, 9];
    insert_internvl_image_tokens(
        prompt,
        &mut tokens,
        &[1],
        2,
        IMG_START,
        IMG_CONTEXT,
        IMG_END,
        &mut toy_encode,
    )
    .unwrap();
    let mut expected = block(2);
    expected.push(NEWLINE);
    expected.extend([7, 8, 9]);
    assert_eq!(tokens, expected);
}

#[test]
fn returns_none_for_empty_inputs() {
    let mut prompt: Vec<i32> = vec![];
    assert!(
        insert_internvl_image_tokens(
            "",
            &mut prompt,
            &[1],
            4,
            IMG_START,
            IMG_CONTEXT,
            IMG_END,
            &mut toy_encode,
        )
        .is_none()
    );
    let mut prompt = vec![1, 2, 3];
    assert!(
        insert_internvl_image_tokens(
            "",
            &mut prompt,
            &[],
            4,
            IMG_START,
            IMG_CONTEXT,
            IMG_END,
            &mut toy_encode,
        )
        .is_none()
    );
    assert!(
        expand_internvl_image_placeholders(&mut prompt, &[1], 4, IMG_START, IMG_CONTEXT, IMG_END)
            .is_none(),
        "no placeholder means nothing to expand"
    );
}

/// Issue #2547 regression pin, with the real `internvl3-1b` token ids.
///
/// The checkpoint's Qwen2 ChatML template renders this text-only turn (the
/// template has no image branch), and the transformers tokenizer encodes it to
/// `TEXT_IDS`. The finished prompt must be the template's tokens with the
/// 448x448 image's single-tile block and a newline at the head of the user
/// turn: 292 tokens, the same ids `tokenizer.apply_chat_template` plus
/// InternVL `model.chat`'s `'<image>\n' + question` placement produce. The
/// pre-#2547 fallback spliced the block after `<|im_start|>`, inside the
/// system turn.
#[test]
fn internvl3_1b_one_tile_prompt_matches_reference_ids() {
    const PROMPT: &str = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n";
    const HEAD: [i32; 24] = [
        151_644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525,
        264, 10950, 17847, 13, 151_645, 198, 151_644, 872, 198,
    ];
    const TAIL: [i32; 9] = [74785, 279, 2168, 13, 151_645, 198, 151_644, 77091, 198];
    let text_ids: Vec<i32> = HEAD.iter().chain(TAIL.iter()).copied().collect();
    let head_text = &PROMPT[..PROMPT.rfind("<|im_start|>user\n").unwrap() + 17];
    let mut encode = |text: &str, _add_special: bool| -> Vec<i32> {
        match text {
            "\n" => vec![NEWLINE],
            t if t == PROMPT => text_ids.clone(),
            t if t == head_text => HEAD.to_vec(),
            other => panic!("unexpected encode of {other:?}"),
        }
    };

    let mut tokens = text_ids.clone();
    insert_internvl_image_tokens(
        PROMPT,
        &mut tokens,
        &[1],
        256,
        IMG_START,
        IMG_CONTEXT,
        IMG_END,
        &mut encode,
    )
    .unwrap();

    let mut expected = HEAD.to_vec();
    expected.extend(block(256));
    expected.push(NEWLINE);
    expected.extend(TAIL);
    assert_eq!(tokens.len(), 292);
    assert_eq!(tokens, expected);
}
