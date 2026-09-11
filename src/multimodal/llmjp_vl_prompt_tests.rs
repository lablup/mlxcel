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

//! LLM-jp-VL prompt-insertion gates.
//!
//! The placement is the point. Upstream puts the framed image block where its
//! `<image>` placeholder sat, which is immediately after the Harmony user-turn
//! opener, and the InternVL fallback (splice after the prompt's first token)
//! would instead drop it inside the *system* turn for this template, because
//! the first token of a rendered LLM-jp-VL prompt is the system `<|start|>`.

use super::{
    LLMJP_ASSISTANT_OPENER, LLMJP_GENERATION_PROMPT_SUFFIX, LLMJP_USER_TURN_MARKER,
    insert_llmjp_image_tokens, text_token_count,
};

const START: i32 = 10;
const END: i32 = 11;
const MESSAGE: i32 = 12;
const CHANNEL: i32 = 9;
const IMG_PAD: i32 = 14;
const IMG_START: i32 = 15;
const IMG_END: i32 = 16;
const SYSTEM: i32 = 100;
const USER: i32 = 101;
const ASSISTANT: i32 = 102;
const FINAL: i32 = 103;
const TEXT: i32 = 200;

/// A stand-in tokenizer over the Harmony scaffold: every marker is one token,
/// every other run of characters is one `TEXT` token. That is enough for the
/// prefix-verification the inserter performs, and it keeps the gate free of a
/// real tokenizer download.
fn encode(text: &str, _add_special: bool) -> Vec<i32> {
    const MARKERS: [(&str, i32); 8] = [
        ("<|start|>", START),
        ("<|end|>", END),
        ("<|message|>", MESSAGE),
        ("<|channel|>", CHANNEL),
        ("<|image_pad|>", IMG_PAD),
        ("<|image_start|>", IMG_START),
        ("<|image_end|>", IMG_END),
        ("final", FINAL),
    ];
    let mut out = Vec::new();
    let mut rest = text;
    let mut pending = String::new();
    'outer: while !rest.is_empty() {
        for (marker, id) in MARKERS {
            if let Some(tail) = rest.strip_prefix(marker) {
                if !pending.is_empty() {
                    out.push(word_token(&pending));
                    pending.clear();
                }
                out.push(id);
                rest = tail;
                continue 'outer;
            }
        }
        let mut chars = rest.chars();
        let c = chars.next().expect("non-empty");
        pending.push(c);
        rest = chars.as_str();
    }
    if !pending.is_empty() {
        out.push(word_token(&pending));
    }
    out
}

fn word_token(word: &str) -> i32 {
    match word {
        "system" => SYSTEM,
        "user" => USER,
        "assistant" => ASSISTANT,
        _ => TEXT,
    }
}

/// A rendered generation prompt, exactly as the shipped `chat_template.jinja`
/// plus the processor's suffix produce it.
fn rendered_prompt() -> String {
    format!(
        "<|start|>system<|message|>You are LLM-jp-VL.<|end|>\
         {LLMJP_USER_TURN_MARKER}What is this?<|end|>\
         {LLMJP_ASSISTANT_OPENER}{LLMJP_GENERATION_PROMPT_SUFFIX}"
    )
}

#[test]
fn the_block_lands_inside_the_user_turn_not_the_system_turn() {
    let prompt = rendered_prompt();
    let mut tokens = encode(&prompt, false);
    let stats = insert_llmjp_image_tokens(
        &prompt,
        &mut tokens,
        &[1],
        4,
        IMG_START,
        IMG_PAD,
        IMG_END,
        &mut encode,
    )
    .expect("insertion reports stats");

    assert_eq!(stats.image_blocks, 1);
    assert_eq!(stats.total_image_tokens, 4);

    // The user turn opens at `<|start|> user <|message|>`; the block starts on
    // the very next token and the question text follows it.
    let start = tokens
        .windows(3)
        .position(|w| w == [START, USER, MESSAGE])
        .expect("user turn present");
    let block = start + 3;
    assert_eq!(tokens[block], IMG_START);
    assert_eq!(&tokens[block + 1..block + 5], &[IMG_PAD; 4]);
    assert_eq!(tokens[block + 5], IMG_END);
    assert_eq!(tokens[block + 6], TEXT, "the question follows the block");

    // Nothing was spliced into the system turn.
    let system_message = tokens
        .windows(3)
        .position(|w| w == [START, SYSTEM, MESSAGE])
        .expect("system turn present");
    assert_ne!(tokens[system_message + 3], IMG_START);

    // The generation prompt is still the tail.
    assert_eq!(
        &tokens[tokens.len() - 5..],
        &[START, ASSISTANT, CHANNEL, FINAL, MESSAGE]
    );
}

#[test]
fn per_image_tile_counts_size_each_block() {
    let prompt = rendered_prompt();
    let mut tokens = encode(&prompt, false);
    let stats = insert_llmjp_image_tokens(
        &prompt,
        &mut tokens,
        &[1, 3],
        4,
        IMG_START,
        IMG_PAD,
        IMG_END,
        &mut encode,
    )
    .expect("insertion reports stats");
    assert_eq!(stats.image_blocks, 2);
    assert_eq!(stats.total_image_tokens, 16);
    assert_eq!(tokens.iter().filter(|&&t| t == IMG_PAD).count(), 16);
    assert_eq!(tokens.iter().filter(|&&t| t == IMG_START).count(), 2);
    assert_eq!(tokens.iter().filter(|&&t| t == IMG_END).count(), 2);
}

#[test]
fn an_existing_placeholder_is_expanded_in_place() {
    // A template that renders typed image content emits the bare special token
    // itself; upstream's rewrite then frames and repeats it.
    let prompt = format!("{LLMJP_USER_TURN_MARKER}<|image_pad|>What is this?<|end|>");
    let mut tokens = encode(&prompt, false);
    let placeholder_at = tokens
        .iter()
        .position(|&t| t == IMG_PAD)
        .expect("placeholder present");

    let stats = insert_llmjp_image_tokens(
        &prompt,
        &mut tokens,
        &[2],
        4,
        IMG_START,
        IMG_PAD,
        IMG_END,
        &mut encode,
    )
    .expect("insertion reports stats");

    assert_eq!(stats.total_image_tokens, 8);
    assert_eq!(tokens[placeholder_at], IMG_START);
    assert_eq!(
        &tokens[placeholder_at + 1..placeholder_at + 9],
        &[IMG_PAD; 8]
    );
    assert_eq!(tokens[placeholder_at + 9], IMG_END);
    assert_eq!(tokens.iter().filter(|&&t| t == IMG_START).count(), 1);
}

#[test]
fn a_raw_prompt_without_a_user_turn_gets_the_block_prepended() {
    // `--no-chat-template`: the image still precedes the question, which is the
    // content ordering upstream's `apply_chat_template` builds.
    let prompt = "describe".to_string();
    let mut tokens = encode(&prompt, false);
    let stats = insert_llmjp_image_tokens(
        &prompt,
        &mut tokens,
        &[1],
        4,
        IMG_START,
        IMG_PAD,
        IMG_END,
        &mut encode,
    )
    .expect("insertion reports stats");
    assert_eq!(stats.total_image_tokens, 4);
    assert_eq!(tokens[0], IMG_START);
    assert_eq!(&tokens[1..5], &[IMG_PAD; 4]);
    assert_eq!(tokens[5], IMG_END);
    assert_eq!(tokens[6], TEXT);
}

#[test]
fn a_tokenizer_that_disagrees_with_the_prompt_degrades_to_prepending() {
    // If the prompt the caller tokenized is not the prompt handed here, the
    // verified prefix will not match and the inserter must fall back rather
    // than splice at a wrong offset.
    let prompt = rendered_prompt();
    let mut tokens = vec![777, 778, 779];
    let stats = insert_llmjp_image_tokens(
        &prompt,
        &mut tokens,
        &[1],
        2,
        IMG_START,
        IMG_PAD,
        IMG_END,
        &mut encode,
    )
    .expect("insertion reports stats");
    assert_eq!(stats.total_image_tokens, 2);
    assert_eq!(
        tokens,
        vec![IMG_START, IMG_PAD, IMG_PAD, IMG_END, 777, 778, 779]
    );
}

#[test]
fn nothing_to_do_returns_none() {
    let prompt = rendered_prompt();
    let mut empty: Vec<i32> = Vec::new();
    assert!(
        insert_llmjp_image_tokens(
            &prompt,
            &mut empty,
            &[1],
            4,
            IMG_START,
            IMG_PAD,
            IMG_END,
            &mut encode
        )
        .is_none()
    );

    let mut tokens = encode(&prompt, false);
    assert!(
        insert_llmjp_image_tokens(
            &prompt,
            &mut tokens,
            &[],
            4,
            IMG_START,
            IMG_PAD,
            IMG_END,
            &mut encode
        )
        .is_none()
    );
}

#[test]
fn text_token_count_excludes_the_placeholders() {
    // Upstream measures the tile budget against the prompt with `<image>`
    // removed.
    let tokens = vec![START, USER, MESSAGE, IMG_PAD, IMG_PAD, TEXT, END];
    assert_eq!(text_token_count(&tokens, IMG_PAD), 5);
    assert_eq!(text_token_count(&tokens, 9999), tokens.len());
}
