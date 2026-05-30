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

use serde::{Serialize, Serializer};
use serde_json::Value;

use super::{
    CancellationToken, DONE_MARKER, SSE_KEEPALIVE_INTERVAL_SECS, payload_channel,
    serialize_json_data,
};
use crate::server::types::{ChatCompletionChunk, CompletionChunk};

#[derive(Serialize)]
struct TestPayload<'a> {
    token: &'a str,
}

struct FailingPayload;

impl Serialize for FailingPayload {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(serde::ser::Error::custom("boom"))
    }
}

#[test]
fn blocking_sse_sender_sends_json_text_and_done_in_order() {
    let (sender, mut rx) = payload_channel(4, None);

    sender.json(&TestPayload { token: "hello" }).unwrap();
    sender.text("plain-text");
    sender.done();

    assert_eq!(
        rx.blocking_recv().unwrap().unwrap(),
        r#"{"token":"hello"}"#.to_string()
    );
    assert_eq!(rx.blocking_recv().unwrap().unwrap(), "plain-text");
    assert_eq!(rx.blocking_recv().unwrap().unwrap(), DONE_MARKER);
}

#[test]
fn done_marker_matches_openai_stream_terminator() {
    assert_eq!(DONE_MARKER, "[DONE]");
}

#[test]
fn serialize_json_data_returns_errors_instead_of_panicking() {
    let err = serialize_json_data(&FailingPayload)
        .unwrap_err()
        .to_string();
    assert!(err.contains("boom"));
}

// ── OpenAI streaming format compliance tests ──────────────────────────────────

/// `finish_reason` must always be serialized in content chunks as `null` (not
/// omitted). Strict clients such as opencode, Continue, and Cursor reject
/// chunks that lack the key entirely.
#[test]
fn content_chunk_serializes_finish_reason_as_null() {
    let chunk = ChatCompletionChunk::content(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        "hello".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    let choice = &json["choices"][0];
    assert!(
        choice.get("finish_reason").is_some(),
        "finish_reason key must be present in content chunks"
    );
    assert_eq!(
        choice["finish_reason"],
        Value::Null,
        "finish_reason must be null in content chunks"
    );
}

/// The `system_fingerprint` field must always be present in every chunk (even
/// if null) to satisfy clients that rely on it for state tracking.
#[test]
fn content_chunk_has_system_fingerprint_field() {
    let chunk = ChatCompletionChunk::content(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        "hello".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    assert!(
        json.get("system_fingerprint").is_some(),
        "system_fingerprint key must be present"
    );
    assert_eq!(json["system_fingerprint"], Value::Null);
}

/// The `logprobs` field must always be present in each choice (even if null).
#[test]
fn content_chunk_choice_has_logprobs_field() {
    let chunk = ChatCompletionChunk::content(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        "hello".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    let choice = &json["choices"][0];
    assert!(
        choice.get("logprobs").is_some(),
        "logprobs key must be present in each choice"
    );
    assert_eq!(choice["logprobs"], Value::Null);
}

/// Usage chunks must carry an empty `choices` array and a populated `usage`
/// object, per the OpenAI streaming spec for `include_usage`.
#[test]
fn usage_chunk_has_empty_choices_and_populated_usage() {
    let chunk =
        ChatCompletionChunk::usage("chatcmpl-test".to_string(), "model".to_string(), 10, 20);
    let json = serde_json::to_value(&chunk).unwrap();

    let choices = json["choices"].as_array().unwrap();
    assert!(
        choices.is_empty(),
        "usage chunk must have empty choices array"
    );

    assert_eq!(json["usage"]["prompt_tokens"], 10);
    assert_eq!(json["usage"]["completion_tokens"], 20);
    assert_eq!(json["usage"]["total_tokens"], 30);
}

/// Initial chunk must have role="assistant" in delta and null finish_reason.
#[test]
fn initial_chunk_has_role_and_null_finish_reason() {
    let chunk = ChatCompletionChunk::initial("chatcmpl-test".to_string(), "model".to_string());
    let json = serde_json::to_value(&chunk).unwrap();
    let choice = &json["choices"][0];
    assert_eq!(choice["delta"]["role"], "assistant");
    assert_eq!(choice["finish_reason"], Value::Null);
}

/// Finish chunk must carry the finish reason string (not null).
#[test]
fn finish_chunk_has_stop_finish_reason() {
    let chunk = ChatCompletionChunk::finish(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        "stop".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

/// Same compliance checks for text-completion streaming chunks.
#[test]
fn completion_content_chunk_serializes_finish_reason_as_null() {
    let chunk = CompletionChunk::content(
        "cmpl-test".to_string(),
        "model".to_string(),
        "hello".to_string(),
    );
    let json = serde_json::to_value(&chunk).unwrap();
    let choice = &json["choices"][0];
    assert!(
        choice.get("finish_reason").is_some(),
        "finish_reason must be present in content chunks"
    );
    assert_eq!(choice["finish_reason"], Value::Null);
}

#[test]
fn completion_usage_chunk_has_empty_choices_and_populated_usage() {
    let chunk = CompletionChunk::usage("cmpl-test".to_string(), "model".to_string(), 5, 15);
    let json = serde_json::to_value(&chunk).unwrap();
    let choices = json["choices"].as_array().unwrap();
    assert!(choices.is_empty());
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    assert_eq!(json["usage"]["completion_tokens"], 15);
    assert_eq!(json["usage"]["total_tokens"], 20);
}

// ── Cache-aware usage chunk tests ───────────────────────────────

/// `usage_with_cache` with `cache_enabled=true` must include
/// `prompt_tokens_details.cached_tokens` in the final chunk.
#[test]
fn usage_with_cache_chunk_includes_cached_tokens_when_enabled() {
    let chunk = ChatCompletionChunk::usage_with_cache(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        100, // prompt_tokens
        20,  // completion_tokens
        64,  // cached_tokens
        true,
    );
    let json = serde_json::to_value(&chunk).unwrap();
    assert!(json["choices"].as_array().unwrap().is_empty());
    assert_eq!(json["usage"]["prompt_tokens"], 100);
    assert_eq!(json["usage"]["completion_tokens"], 20);
    assert_eq!(json["usage"]["total_tokens"], 120);
    assert_eq!(json["usage"]["prompt_tokens_details"]["cached_tokens"], 64);
}

/// `usage_with_cache` with `cache_enabled=false` must omit
/// `prompt_tokens_details` entirely (wire compat for disabled-cache clients).
#[test]
fn usage_with_cache_chunk_omits_cached_tokens_when_disabled() {
    let chunk = ChatCompletionChunk::usage_with_cache(
        "chatcmpl-test".to_string(),
        "model".to_string(),
        50,
        10,
        99,
        false,
    );
    let json = serde_json::to_value(&chunk).unwrap();
    assert!(
        !json["usage"]
            .as_object()
            .unwrap()
            .contains_key("prompt_tokens_details"),
        "prompt_tokens_details must be absent when cache is disabled"
    );
}

/// Plain `usage()` must not include `prompt_tokens_details` (backward compat).
#[test]
fn plain_usage_chunk_has_no_prompt_tokens_details() {
    let chunk = ChatCompletionChunk::usage("chatcmpl-test".to_string(), "model".to_string(), 10, 5);
    let json = serde_json::to_value(&chunk).unwrap();
    assert!(
        !json["usage"]
            .as_object()
            .unwrap()
            .contains_key("prompt_tokens_details"),
        "plain usage chunk must not include prompt_tokens_details"
    );
}

// ── Cancellation token tests ────────────────────────────────────────────────

#[test]
fn cancellation_token_set_when_receiver_dropped() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let token: CancellationToken = Arc::new(AtomicBool::new(false));
    let (sender, rx) = payload_channel(4, Some(token.clone()));

    // Drop the receiver to simulate client disconnect
    drop(rx);

    // Sending after receiver drop should set the cancellation flag
    sender.text("hello");
    assert!(
        token.load(Ordering::Relaxed),
        "cancellation token must be set when receiver is dropped"
    );
}

#[test]
fn cancellation_token_not_set_when_send_succeeds() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let token: CancellationToken = Arc::new(AtomicBool::new(false));
    let (sender, _rx) = payload_channel(4, Some(token.clone()));

    sender.text("hello");
    assert!(
        !token.load(Ordering::Relaxed),
        "cancellation token must not be set when send succeeds"
    );
}

#[test]
fn sender_without_cancellation_token_does_not_panic_on_dropped_receiver() {
    let (sender, rx) = payload_channel(4, None);
    drop(rx);
    // Should not panic even without a cancellation token
    sender.text("hello");
}

// ── Long-prefill keepalive regression tests ────────────────────

/// The SSE keepalive interval must be less than typical proxy idle timeouts
/// (nginx 60 s, HAProxy 60 s, AWS ALB 60 s) so that long-prefill requests
/// keep the connection alive.
///
/// This is enforced as a `const` assertion so a regression is caught at
/// compile time rather than test time. (Using `assert!` on a constant
/// expression triggers `clippy::assertions_on_constants`.)
const _: () = assert!(
    SSE_KEEPALIVE_INTERVAL_SECS < 60,
    "SSE keepalive interval must be less than the 60s default used by most reverse proxies"
);

/// During a long prefill phase (no events from the model worker), the SSE
/// channel must remain open and operational. This test exercises the token
/// queue flow without a real model: the sender holds the channel open while
/// a background thread simulates the prefill delay before sending the first
/// token. The receiver must not error out during the wait.
#[test]
fn long_prefill_channel_stays_open_before_first_token() {
    use std::thread;
    use std::time::Duration;

    // Buffer of 4 is enough for our synthetic token sequence.
    let (sender, mut rx) = payload_channel(4, None);

    // Simulate the model worker: pause for 50 ms (representing a long prefill),
    // then emit a token event and a DONE marker. The channel must survive the
    // delay without being closed or timing out.
    let sender_clone = sender.clone();
    let worker = thread::spawn(move || {
        // Simulate prefill latency — no events emitted during this window.
        thread::sleep(Duration::from_millis(50));
        // First generated token arrives after prefill completes.
        sender_clone.text("hello");
        sender_clone.done();
    });

    // The receiver should successfully collect both the token and the DONE
    // marker without error, even though there was a delay before the first
    // event arrived.
    let first = rx
        .blocking_recv()
        .expect("channel must stay open during prefill");
    assert_eq!(first.unwrap(), "hello", "first token must be correct");

    let second = rx
        .blocking_recv()
        .expect("channel must deliver DONE marker");
    assert_eq!(
        second.unwrap(),
        DONE_MARKER,
        "DONE marker must follow the token"
    );

    worker.join().expect("worker thread must not panic");
}

// TODO: add a test that verifies the SSE keepalive comment frame is emitted
// before the first event arrives (LOW #2). This requires an axum
// integration test harness to drive `Sse::new(stream).keep_alive(...)` end-to-end
// and read raw SSE frames from the response body. The existing unit-level
// `payload_channel` tests cannot reach the axum `KeepAlive` layer. A suitable
// approach would be `axum_test::TestClient` or `tower::ServiceExt::oneshot`
// with `hyper::body::to_bytes`; skipped here to avoid the additional test infra.
