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

use super::{MOONDREAM2_BOS_ID, Moondream2PromptMode, prepare_moondream2_prompt_tokens};

/// Byte-length encoder: deterministic and injective enough to assert structure.
fn encode(text: &str, _add_special: bool) -> Vec<i32> {
    text.bytes().map(|b| b as i32).collect()
}

#[test]
fn bos_id_is_the_endoftext_token() {
    // The moondream2 GPT-2/CodeGen tokenizer uses `<|endoftext|>` (id 50256) as
    // its begin-of-text token, not Moondream3's id 0.
    assert_eq!(MOONDREAM2_BOS_ID, 50256);
}

#[test]
fn text_only_prompt_leads_with_bos_and_frames_question() {
    let prepared = prepare_moondream2_prompt_tokens("What is this?", 0, encode).unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Query);
    assert_eq!(prepared.tokens[0], MOONDREAM2_BOS_ID);
    let text: String = prepared.tokens[1..]
        .iter()
        .map(|&b| b as u8 as char)
        .collect();
    assert!(text.contains("Question: What is this?"));
    assert!(text.trim_end().ends_with("Answer:"));
}

#[test]
fn image_prompt_omits_bos_for_wrapper_prefix() {
    let prepared = prepare_moondream2_prompt_tokens("Count the cats.", 1, encode).unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Query);
    // The wrapper prepends BOS + image tokens, so the first token here is text.
    assert_ne!(prepared.tokens.first(), Some(&MOONDREAM2_BOS_ID));
    let text: String = prepared.tokens.iter().map(|&b| b as u8 as char).collect();
    assert!(text.contains("Question: Count the cats."));
}

#[test]
fn empty_image_prompt_falls_back_to_caption() {
    let prepared = prepare_moondream2_prompt_tokens("   ", 1, encode).unwrap();
    assert_eq!(prepared.mode, Moondream2PromptMode::Caption);
    assert!(!prepared.tokens.is_empty());
}

#[test]
fn text_only_empty_prompt_is_rejected() {
    assert!(prepare_moondream2_prompt_tokens("", 0, encode).is_err());
}

#[test]
fn multiple_images_are_rejected() {
    assert!(prepare_moondream2_prompt_tokens("hi", 2, encode).is_err());
}
