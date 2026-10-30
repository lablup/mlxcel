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

/// Issue #2547 follow-up, with the real `internvl3-1b` token ids: a question
/// that starts with a newline.
///
/// Qwen2's pre-tokenizer fuses the opener's `\n` with the question's leading
/// `\n\n` into one `\n\n\n` token (1406), so the rendered head
/// `...<|im_start|>user\n` is not a token prefix of the prompt. The block must
/// still open the user turn, after the system turn, and the newline that
/// follows it fuses with the question's newlines exactly as in the reference
/// render (`<img>...</img>` + `"\n"` + question through the checkpoint's
/// template and the transformers tokenizer): 291 tokens. Before the fix the
/// block was prepended at token 0, ahead of `<|im_start|>system`.
#[test]
fn internvl3_1b_question_with_leading_newline_keeps_block_in_user_turn() {
    const PROMPT: &str = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n\nLeading newlines<|im_end|>\n<|im_start|>assistant\n";
    const HEAD: [i32; 24] = [
        151_644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525,
        264, 10950, 17847, 13, 151_645, 198, 151_644, 872, 198,
    ];
    // The caller's encoding of `PROMPT`: `user` (872) is followed by the fused
    // `\n\n\n` (1406), not by `\n` (198).
    const TEXT_TAIL: [i32; 9] = [1406, 69750, 501, 7969, 151_645, 198, 151_644, 77091, 198];
    // `"\n" + tail` re-encoded, the reference's ids after `</img>`.
    const NL_TAIL: [i32; 9] = [1406, 69750, 501, 7969, 151_645, 198, 151_644, 77091, 198];
    let text_ids: Vec<i32> = HEAD[..23].iter().chain(TEXT_TAIL.iter()).copied().collect();
    let head_len = PROMPT.rfind("<|im_start|>user\n").unwrap() + 17;
    let head_text = &PROMPT[..head_len];
    let nl_tail = format!("\n{}", &PROMPT[head_len..]);
    let mut encode = |text: &str, _add_special: bool| -> Vec<i32> {
        match text {
            "\n" => vec![NEWLINE],
            t if t == head_text => HEAD.to_vec(),
            t if t == nl_tail => NL_TAIL.to_vec(),
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
    expected.extend(NL_TAIL);
    assert_eq!(tokens.len(), 291);
    assert_eq!(&tokens[..24], &HEAD[..], "the system turn must come first");
    assert_eq!(tokens, expected);
}

/// Like [`toy_encode`], but a run of `n` newlines is one token (`NEWLINE` for
/// one, `500 + n` for more), the way Qwen2's pre-tokenizer fuses them.
fn fusing_encode(text: &str, add_special: bool) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    let mut run = 0i32;
    for id in toy_encode(text, add_special) {
        if id == NEWLINE {
            run += 1;
            continue;
        }
        if run > 0 {
            out.push(if run == 1 { NEWLINE } else { 500 + run });
            run = 0;
        }
        out.push(id);
    }
    if run > 0 {
        out.push(if run == 1 { NEWLINE } else { 500 + run });
    }
    out
}

#[test]
fn fused_opener_newline_with_two_images_keeps_blocks_in_user_turn() {
    let prompt = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n\nQ<|im_end|>\n";
    let head = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n";
    let mut tokens = fusing_encode(prompt, true);
    let stats = insert_internvl_image_tokens(
        prompt,
        &mut tokens,
        &[1, 3],
        4,
        IMG_START,
        IMG_CONTEXT,
        IMG_END,
        &mut fusing_encode,
    )
    .unwrap();

    assert_eq!(stats.image_blocks, 2);
    assert_eq!(stats.total_image_tokens, 16);
    // Each block is followed by a newline; the last one fuses with the
    // question's own leading newline, as `"\n" + question` would encode.
    let mut expected = fusing_encode(head, false);
    expected.extend(block(4));
    expected.push(NEWLINE);
    expected.extend(block(12));
    expected.extend(fusing_encode("\n\nQ<|im_end|>\n", false));
    assert_eq!(tokens, expected);
    assert_eq!(expected[expected.len() - 4], 502);
}

/// `toy_encode` with a leading BOS id when special tokens are requested.
fn bos_encode(text: &str, add_special: bool) -> Vec<i32> {
    let mut out = if add_special { vec![7] } else { Vec::new() };
    out.extend(toy_encode(text, false));
    out
}

#[test]
fn caller_tokens_encoded_with_special_tokens_still_align() {
    let prompt = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n";
    let head = "<|im_start|>user\n";
    let mut tokens = bos_encode(prompt, true);
    insert_internvl_image_tokens(
        prompt,
        &mut tokens,
        &[1],
        2,
        IMG_START,
        IMG_CONTEXT,
        IMG_END,
        &mut bos_encode,
    )
    .unwrap();

    let mut expected = bos_encode(head, true);
    expected.extend(block(2));
    expected.push(NEWLINE);
    expected.extend(toy_encode(&prompt[head.len()..], false));
    assert_eq!(tokens, expected);
}
