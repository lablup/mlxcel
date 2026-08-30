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

//! `--prefill-assistant` resolution tests (#1470).

use super::*;

fn request(body: &str) -> ChatCompletionRequest {
    serde_json::from_str(body).expect("fixture request parses")
}

const USER_THEN_ASSISTANT: &str = r#"{
    "model": "m",
    "messages": [
        {"role": "user", "content": "Finish this: the capital of"},
        {"role": "assistant", "content": " France is"}
    ]
}"#;

#[test]
fn a_trailing_assistant_message_is_a_prefill_when_enabled() {
    let prefill = resolve(&request(USER_THEN_ASSISTANT), true)
        .expect("valid")
        .expect("a trailing assistant message prefills");
    assert_eq!(prefill.text, " France is");
    assert!(!prefill.is_reasoning);
}

#[test]
fn the_flag_off_leaves_the_trailing_assistant_message_alone() {
    assert_eq!(resolve(&request(USER_THEN_ASSISTANT), false), Ok(None));
}

#[test]
fn a_trailing_user_message_is_never_a_prefill() {
    let body = r#"{"model":"m","messages":[{"role":"assistant","content":"hi"},{"role":"user","content":"hello"}]}"#;
    assert_eq!(resolve(&request(body), true), Ok(None));
}

#[test]
fn two_trailing_assistant_messages_are_refused_with_upstreams_wording() {
    let body = r#"{"model":"m","messages":[
        {"role":"user","content":"go"},
        {"role":"assistant","content":"one"},
        {"role":"assistant","content":"two"}
    ]}"#;
    assert_eq!(
        resolve(&request(body), true),
        Err(TWO_TRAILING_ASSISTANTS.to_string())
    );
}

#[test]
fn a_trailing_tool_call_message_is_a_completed_turn_not_a_prefix() {
    let body = r#"{"model":"m","messages":[
        {"role":"user","content":"weather?"},
        {"role":"assistant","content":null,"tool_calls":[
            {"id":"c1","type":"function","function":{"name":"get","arguments":"{}"}}
        ]}
    ]}"#;
    assert_eq!(resolve(&request(body), true), Ok(None));
}

#[test]
fn a_reasoning_only_trailing_message_is_a_reasoning_continuation() {
    let body = r#"{"model":"m","messages":[
        {"role":"user","content":"think"},
        {"role":"assistant","content":"","reasoning":"Let me consider"}
    ]}"#;
    let prefill = resolve(&request(body), true)
        .expect("valid")
        .expect("prefill");
    assert!(prefill.is_reasoning);
    assert_eq!(prefill.text, "Let me consider");
}

#[test]
fn an_empty_trailing_assistant_message_prefills_nothing_but_still_continues() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":"go"},{"role":"assistant","content":""}]}"#;
    let prefill = resolve(&request(body), true)
        .expect("valid")
        .expect("prefill");
    assert!(prefill.text.is_empty());
    assert_eq!(
        append_to_prompt("PROMPT<|im_start|>assistant\n", &prefill, None).unwrap(),
        "PROMPT<|im_start|>assistant\n"
    );
}

#[test]
fn a_content_continuation_is_appended_with_no_closing_tag() {
    let prefill = AssistantPrefill {
        text: " France is".to_string(),
        is_reasoning: false,
    };
    assert_eq!(
        append_to_prompt("...<|im_start|>assistant\n", &prefill, None).unwrap(),
        "...<|im_start|>assistant\n France is"
    );
}

#[test]
fn a_content_continuation_closes_a_primed_thinking_block_first() {
    // b10621 emits `<think>\n` + reasoning + `\n</think>\n\n` before the content
    // when the family supports reasoning; the prompt already carries the open
    // half, so the continuation must not land inside the block.
    let prefill = AssistantPrefill {
        text: " France is".to_string(),
        is_reasoning: false,
    };
    assert_eq!(
        append_to_prompt(
            "...<|im_start|>assistant\n<think>\n",
            &prefill,
            Some("</think>")
        )
        .unwrap(),
        "...<|im_start|>assistant\n<think>\n\n</think>\n\n France is"
    );
}

#[test]
fn a_reasoning_continuation_needs_a_primed_thinking_block() {
    let prefill = AssistantPrefill {
        text: "Let me consider".to_string(),
        is_reasoning: true,
    };
    assert_eq!(
        append_to_prompt(
            "...<|im_start|>assistant\n<think>\n",
            &prefill,
            Some("</think>")
        )
        .unwrap(),
        "...<|im_start|>assistant\n<think>\nLet me consider"
    );
    assert_eq!(
        append_to_prompt("...<|im_start|>assistant\n", &prefill, None),
        Err(REASONING_ONLY_UNSUPPORTED.to_string())
    );
}
