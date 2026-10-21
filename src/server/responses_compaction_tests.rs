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

//! Capsule, replay, selection and compaction-core tests (issue #2397),
//! against a model-free backend that counts one token per JSON byte.

use std::sync::Mutex;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde_json::json;

use crate::server::responses_compaction::{
    CompactionBackend, CompactionError, CompactionLimits, SummaryPass, compact_history,
    decode_capsule, default_keep_tokens, encode_capsule, input_item_to_json, resolve_compaction,
    safe_boundaries,
};
use crate::server::types::responses_request::{
    ResponseInputContent, ResponseInputItem, ResponseInputRole, ResponseToolOutput,
};

const LIMIT: usize = 1 << 20;

fn message(role: ResponseInputRole, text: &str) -> ResponseInputItem {
    ResponseInputItem::Message {
        role,
        content: ResponseInputContent::Text(text.to_string()),
        name: None,
    }
}

fn user(text: &str) -> ResponseInputItem {
    message(ResponseInputRole::User, text)
}

fn assistant(text: &str) -> ResponseInputItem {
    message(ResponseInputRole::Assistant, text)
}

fn system(text: &str) -> ResponseInputItem {
    message(ResponseInputRole::System, text)
}

fn call(id: &str) -> ResponseInputItem {
    ResponseInputItem::FunctionCall {
        call_id: id.to_string(),
        name: "lookup".to_string(),
        arguments: "{}".to_string(),
    }
}

fn output(id: &str) -> ResponseInputItem {
    ResponseInputItem::FunctionCallOutput {
        call_id: id.to_string(),
        output: ResponseToolOutput::Text(format!("result {id}")),
    }
}

fn texts(items: &[ResponseInputItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| input_item_to_json(item).to_string())
        .collect()
}

fn capsule_item(summary: &str, retained: &[ResponseInputItem]) -> ResponseInputItem {
    ResponseInputItem::Compaction {
        encrypted_content: encode_capsule(summary, retained),
    }
}

#[test]
fn capsule_round_trips_summary_and_retained_items() {
    let retained = vec![user("latest"), call("c1"), output("c1"), assistant("done")];
    let encoded = encode_capsule("the summary", &retained);
    assert!(
        encoded
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "base64url without padding: {encoded}"
    );
    let decoded = decode_capsule(&encoded, LIMIT).expect("capsule decodes");
    assert_eq!(decoded.summary, "the summary");
    assert_eq!(texts(&decoded.retained), texts(&retained));
}

#[test]
fn corrupted_capsule_is_refused() {
    let encoded = encode_capsule("s", &[user("u")]);
    let corrupted = format!("{}!!", &encoded[..encoded.len() / 2]);
    assert!(decode_capsule(&corrupted, LIMIT).is_err());
    let not_json = URL_SAFE_NO_PAD.encode("not json");
    assert!(decode_capsule(&not_json, LIMIT).is_err());
}

#[test]
fn unknown_capsule_version_is_refused() {
    let v2 = URL_SAFE_NO_PAD.encode(json!({"v": 2, "summary": "s", "retained": []}).to_string());
    let err = decode_capsule(&v2, LIMIT).expect_err("v2 is refused");
    assert!(err.contains("version"), "{err}");
}

#[test]
fn capsule_above_the_body_limit_or_nesting_a_capsule_is_refused() {
    let encoded = encode_capsule("s", &[user("u")]);
    assert!(decode_capsule(&encoded, encoded.len() - 1).is_err());
    let nested = encode_capsule("outer", &[capsule_item("inner", &[])]);
    assert!(decode_capsule(&nested, LIMIT).is_err());
}

#[test]
fn replay_drops_history_before_the_newest_capsule_but_keeps_instructions() {
    let items = vec![
        system("sys"),
        user("old question"),
        capsule_item("first summary", &[user("first retained")]),
        assistant("between"),
        message(ResponseInputRole::Developer, "dev"),
        capsule_item("second summary", &[user("kept turn")]),
        user("new question"),
    ];
    let resolved = resolve_compaction(items).expect("capsule resolves");
    let rendered = texts(&resolved);
    assert_eq!(rendered.len(), 5, "{rendered:?}");
    assert!(rendered[0].contains("\"system\"") && rendered[0].contains("sys"));
    assert!(rendered[1].contains("\"developer\""));
    assert!(rendered[2].contains("\"assistant\"") && rendered[2].contains("second summary"));
    assert!(rendered[3].contains("kept turn"));
    assert!(rendered[4].contains("new question"));
    let joined = rendered.join("\n");
    for gone in ["old question", "first summary", "between"] {
        assert!(!joined.contains(gone), "{gone} survived replay");
    }
}

#[test]
fn replay_refuses_a_corrupted_capsule() {
    let items = vec![ResponseInputItem::Compaction {
        encrypted_content: "%%%".to_string(),
    }];
    assert!(resolve_compaction(items).is_err());
}

#[test]
fn compaction_item_parses_from_the_wire_form() {
    let item: ResponseInputItem = serde_json::from_value(json!({
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": "abc",
    }))
    .expect("compaction input item parses");
    assert!(
        matches!(item, ResponseInputItem::Compaction { encrypted_content } if encrypted_content == "abc")
    );
}

#[test]
fn boundaries_never_separate_a_call_from_its_output() {
    // u0 a1 u2 c3(x) c4(y) o5(x) o6(y) a7 u8
    let conversation = vec![
        user("u0"),
        assistant("a1"),
        user("u2"),
        call("x"),
        call("y"),
        output("x"),
        output("y"),
        assistant("a7"),
        user("u8"),
    ];
    // Splits: before u2, after the group closes (7), before u8. Never 4..=6.
    assert_eq!(safe_boundaries(&conversation), vec![2, 7, 8]);
}

#[test]
fn an_unfinished_tool_group_stays_whole() {
    // Two parallel calls, only one output arrived: nothing after the group's
    // start can be split off.
    let conversation = vec![
        user("u0"),
        assistant("a1"),
        user("u2"),
        call("x"),
        call("y"),
        output("x"),
    ];
    assert_eq!(safe_boundaries(&conversation), vec![2]);
}

/// One token per JSON byte; the summary pass returns a scripted text.
struct FakeBackend {
    summary: String,
    heads: Mutex<Vec<Vec<String>>>,
}

impl FakeBackend {
    fn new(summary: &str) -> Self {
        Self {
            summary: summary.to_string(),
            heads: Mutex::new(Vec::new()),
        }
    }
}

impl CompactionBackend for FakeBackend {
    async fn count(&self, items: &[ResponseInputItem]) -> Result<usize, CompactionError> {
        Ok(texts(items).iter().map(String::len).sum())
    }

    async fn summarize(&self, head: &[ResponseInputItem]) -> Result<SummaryPass, CompactionError> {
        self.heads.lock().expect("heads lock").push(texts(head));
        Ok(SummaryPass {
            text: self.summary.clone(),
            prompt_tokens: 10,
            completion_tokens: 3,
            cached_tokens: 0,
        })
    }
}

fn long_history() -> Vec<ResponseInputItem> {
    let filler = "x".repeat(400);
    vec![
        system("be brief"),
        user(&format!("first question {filler}")),
        assistant(&format!("first answer {filler}")),
        user(&format!("second question {filler}")),
        call("c1"),
        output("c1"),
        assistant("second answer"),
        user("third question"),
    ]
}

#[tokio::test]
async fn compaction_keeps_instructions_and_the_latest_user_message() {
    let backend = FakeBackend::new("SUMMARY");
    let outcome = compact_history(
        &backend,
        long_history(),
        CompactionLimits {
            context_tokens: 100_000,
            keep_tokens: 64,
        },
    )
    .await
    .expect("compaction succeeds");
    let instructions = texts(&outcome.instructions);
    assert_eq!(instructions.len(), 1);
    assert!(instructions[0].contains("\"system\""), "{instructions:?}");
    let retained = texts(&outcome.retained);
    assert_eq!(retained.len(), 1, "{retained:?}");
    assert!(retained[0].contains("third question"));
    assert!(outcome.tokens_after < outcome.tokens_before);
    let heads = backend.heads.lock().expect("heads lock");
    assert_eq!(heads.len(), 1, "exactly one summary pass");
    assert!(
        heads[0][0].contains("\"system\""),
        "head keeps instructions"
    );
    assert!(heads[0].iter().any(|item| item.contains("first question")));
}

#[tokio::test]
async fn latest_user_message_is_retained_even_when_it_precedes_the_cut() {
    let filler = "y".repeat(400);
    let items = vec![
        user("old"),
        assistant(&filler),
        user("latest request"),
        call("c1"),
        output("c1"),
        assistant(&filler),
    ];
    let backend = FakeBackend::new("S");
    let outcome = compact_history(
        &backend,
        items,
        CompactionLimits {
            context_tokens: 100_000,
            keep_tokens: 1,
        },
    )
    .await
    .expect("compaction succeeds");
    // The newest boundary is after the tool group (5); the latest user turn
    // at 2 is carried in front of the retained tail.
    let retained = texts(&outcome.retained);
    assert_eq!(retained.len(), 2, "{retained:?}");
    assert!(retained[0].contains("latest request"));
    assert!(retained[1].contains(&filler));
}

#[tokio::test]
async fn larger_keep_tokens_retains_more_whole_turns() {
    let backend = FakeBackend::new("SUMMARY");
    let outcome = compact_history(
        &backend,
        long_history(),
        CompactionLimits {
            context_tokens: 100_000,
            keep_tokens: 900,
        },
    )
    .await
    .expect("compaction succeeds");
    let retained = texts(&outcome.retained);
    // From "second question" on: the call stays with its output.
    assert_eq!(retained.len(), 5, "{retained:?}");
    assert!(retained[0].contains("second question"));
}

#[tokio::test]
async fn a_summary_that_does_not_shorten_the_conversation_fails() {
    let backend = FakeBackend::new(&"z".repeat(10_000));
    let err = compact_history(
        &backend,
        long_history(),
        CompactionLimits {
            context_tokens: 100_000,
            keep_tokens: 64,
        },
    )
    .await
    .expect_err("a longer summary is refused");
    let response = err.into_error_response();
    assert_eq!(
        response.error.code.as_deref(),
        Some("context_length_exceeded")
    );
    assert_eq!(response.status, axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn history_over_one_window_fails_before_summarizing() {
    let backend = FakeBackend::new("S");
    let err = compact_history(
        &backend,
        long_history(),
        CompactionLimits {
            context_tokens: 100,
            keep_tokens: 64,
        },
    )
    .await
    .expect_err("oversized history is refused");
    assert!(matches!(err, CompactionError::ContextLengthExceeded(_)));
    assert!(backend.heads.lock().expect("heads lock").is_empty());
}

#[tokio::test]
async fn nothing_to_compact_and_empty_summary_are_refused() {
    let backend = FakeBackend::new("S");
    let err = compact_history(
        &backend,
        vec![system("s"), user("only")],
        CompactionLimits {
            context_tokens: 100_000,
            keep_tokens: 64,
        },
    )
    .await
    .expect_err("single turn has nothing to compact");
    assert!(matches!(err, CompactionError::InvalidRequest(_)));

    let empty = FakeBackend::new("   ");
    let err = compact_history(
        &empty,
        long_history(),
        CompactionLimits {
            context_tokens: 100_000,
            keep_tokens: 64,
        },
    )
    .await
    .expect_err("empty summary is refused");
    assert_eq!(
        err.into_error_response().status,
        axum::http::StatusCode::BAD_GATEWAY
    );
}

#[test]
fn default_keep_tokens_is_a_sixteenth_capped_at_8192() {
    assert_eq!(default_keep_tokens(4096), 256);
    assert_eq!(default_keep_tokens(262_144), 8192);
    assert_eq!(default_keep_tokens(8), 1);
}
