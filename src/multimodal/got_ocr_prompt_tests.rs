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

//! Tests for [`super`], the GOT-OCR 2.0 fixed-conversation prompt builder.

use super::*;

const IMAGE_TOKEN_LEN: usize = 256;

/// The exact string `modeling_GOT.py::chat` produces for `ocr_type != 'format'`
/// with a 256-token block, reproduced here as the pinned oracle. The 256
/// `<imgpad>` repeats are elided; the assertions below check the prefix, the
/// suffix and the placeholder count separately.
const UPSTREAM_PREFIX: &str = "<|im_start|>system\n        You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n<img><imgpad>";
const UPSTREAM_SUFFIX: &str = "<imgpad></img>\nOCR: <|im_end|><|im_start|>assistant\n";

fn build(text: &str) -> (String, GotPromptStats) {
    build_got_prompt(text, 1, IMAGE_TOKEN_LEN).expect("build")
}

/// A bare instruction is wrapped in the fixed conversation, and the result is
/// byte-identical to what upstream's `chat()` assembles.
#[test]
fn plain_instruction_is_wrapped_with_system_and_image_block() {
    let (prompt, stats) = build("OCR: ");
    assert!(
        prompt.starts_with(UPSTREAM_PREFIX),
        "prefix mismatch: {:?}",
        &prompt[..UPSTREAM_PREFIX.len().min(prompt.len())]
    );
    assert!(
        prompt.ends_with(UPSTREAM_SUFFIX),
        "suffix mismatch: {:?}",
        &prompt[prompt.len().saturating_sub(UPSTREAM_SUFFIX.len())..]
    );
    assert_eq!(stats.image_blocks, 1);
    assert_eq!(stats.total_image_tokens, IMAGE_TOKEN_LEN);
    assert!(!stats.pre_templated);
}

/// The system turn carries the eight spaces of Python source indentation and
/// no newline after its `<|im_end|>`. Losing either changes the tokenization of
/// a prefix the model saw on every training example.
#[test]
fn system_turn_keeps_the_literal_indentation_and_has_no_trailing_newline() {
    assert_eq!(
        GOT_SYSTEM,
        "<|im_start|>system\n        You should follow the instructions carefully and explain your answers in detail."
    );
    let (prompt, _) = build("OCR: ");
    assert!(prompt.contains("detail.<|im_end|><|im_start|>user\n"));
    assert!(!prompt.contains("detail.<|im_end|>\n"));
}

/// The block holds exactly `image_token_len` placeholders between one `<img>`
/// and one `</img>`. `merge_llava` writes one feature row per `<imgpad>`, so
/// this count is the contract with the tower's 256 output rows.
#[test]
fn block_has_256_imgpad_tokens() {
    let (prompt, stats) = build("OCR: ");
    assert_eq!(prompt.matches(GOT_IM_PATCH_TAG).count(), IMAGE_TOKEN_LEN);
    assert_eq!(prompt.matches(GOT_IM_START_TAG).count(), 1);
    assert_eq!(prompt.matches(GOT_IM_END_TAG).count(), 1);
    assert_eq!(stats.total_image_tokens, IMAGE_TOKEN_LEN);
}

/// Text that already carries the framing is not wrapped a second time, and the
/// block lands right after the user opener rather than ahead of the system turn.
#[test]
fn pre_templated_instruction_passes_through() {
    let rendered =
        format!("{GOT_SYSTEM}{GOT_SEP}{GOT_USER_ROLE}OCR: {GOT_SEP}{GOT_ASSISTANT_ROLE}");
    let (prompt, stats) = build(&rendered);
    assert!(stats.pre_templated);
    assert_eq!(prompt.matches("<|im_start|>system").count(), 1);
    assert!(prompt.starts_with(UPSTREAM_PREFIX));
    assert!(prompt.ends_with(UPSTREAM_SUFFIX));
}

/// Rendering the builtin template and then building must land on the same
/// bytes as building from the bare instruction, or the CLI and the server
/// prompt the model differently.
#[test]
fn templated_and_bare_paths_agree_byte_for_byte() {
    let (bare, _) = build("OCR: ");
    let rendered =
        format!("{GOT_SYSTEM}{GOT_SEP}{GOT_USER_ROLE}OCR: {GOT_SEP}{GOT_ASSISTANT_ROLE}");
    let (templated, _) = build(&rendered);
    assert_eq!(bare, templated);
}

/// An explicit `<image>` marker is replaced where it sits.
#[test]
fn image_placeholder_is_replaced_in_place() {
    let (prompt, stats) = build("describe this\n<image>\nOCR: ");
    assert!(!prompt.contains("<image>"));
    assert!(prompt.contains("describe this\n<img><imgpad>"));
    assert!(prompt.contains("</img>\nOCR: "));
    assert_eq!(prompt.matches(GOT_IM_PATCH_TAG).count(), IMAGE_TOKEN_LEN);
    assert!(!stats.pre_templated);
}

/// A marker inside already-framed text is still replaced in place, and no
/// second block is spliced after the user opener.
#[test]
fn image_placeholder_wins_over_the_user_turn_splice() {
    let rendered =
        format!("{GOT_SYSTEM}{GOT_SEP}{GOT_USER_ROLE}<image>\nOCR: {GOT_SEP}{GOT_ASSISTANT_ROLE}");
    let (prompt, _) = build(&rendered);
    assert_eq!(prompt.matches(GOT_IM_START_TAG).count(), 1);
    assert_eq!(prompt.matches(GOT_IM_PATCH_TAG).count(), IMAGE_TOKEN_LEN);
    assert!(prompt.ends_with(UPSTREAM_SUFFIX));
}

/// Framed text with no user opener (a prefill continuation) still leads with
/// the block instead of dropping it.
#[test]
fn separator_only_framing_prepends_the_block() {
    let (prompt, stats) = build("previous turn<|im_end|>");
    assert!(stats.pre_templated);
    assert!(prompt.starts_with(GOT_IM_START_TAG));
    assert!(prompt.ends_with("previous turn<|im_end|>"));
}

/// With no image the conversation is still assembled, but no block and no
/// stray marker survive.
#[test]
fn text_only_request_emits_no_block() {
    let (prompt, stats) = build_got_prompt("hello", 0, IMAGE_TOKEN_LEN).expect("build");
    assert_eq!(stats.image_blocks, 0);
    assert_eq!(stats.total_image_tokens, 0);
    assert!(!prompt.contains(GOT_IM_PATCH_TAG));
    assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));

    let (prompt, _) = build_got_prompt("a <image> b", 0, IMAGE_TOKEN_LEN).expect("build");
    assert!(!prompt.contains("<image>"));
    assert!(prompt.contains("a  b"));
}

/// More than one image is refused rather than silently producing a prompt
/// whose `<img>` / `</img>` counts upstream's `forward` would reject.
#[test]
fn multiple_images_are_rejected() {
    let err = build_got_prompt("OCR: ", 2, IMAGE_TOKEN_LEN).expect_err("must reject");
    assert!(err.contains("one image"), "{err}");
}

/// A second `<image>` marker is dropped rather than expanded, keeping the
/// placeholder count equal to the single block the tower produces.
#[test]
fn extra_image_markers_are_dropped() {
    let (prompt, _) = build("<image> and <image>");
    assert_eq!(prompt.matches(GOT_IM_START_TAG).count(), 1);
    assert_eq!(prompt.matches(GOT_IM_PATCH_TAG).count(), IMAGE_TOKEN_LEN);
    assert!(!prompt.contains("<image>"));
}

/// The tokenizing wrapper adds no BOS and passes the assembled string through
/// unchanged.
#[test]
fn prepare_tokens_encodes_without_special_prefix() {
    let mut seen: Vec<(String, bool)> = Vec::new();
    let mut encode = |text: &str, add_special: bool| {
        seen.push((text.to_string(), add_special));
        vec![1, 2, 3]
    };
    let (tokens, stats) =
        prepare_got_prompt_tokens("OCR: ", 1, IMAGE_TOKEN_LEN, &mut encode).expect("prepare");
    assert_eq!(tokens, vec![1, 2, 3]);
    assert_eq!(stats.total_image_tokens, IMAGE_TOKEN_LEN);
    assert_eq!(seen.len(), 1);
    assert!(!seen[0].1, "GOT adds no BOS");
    assert!(seen[0].0.starts_with(UPSTREAM_PREFIX));
}
