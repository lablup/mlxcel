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

//! Unit tests for [`super::BackgroundEventLog`] and its readers (#2900).

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::Value;

use super::*;
use crate::server::responses_translator::{OutboundContext, build_response_object};
use crate::server::types::responses_request::CreateResponseRequest;

const WAIT: Duration = Duration::from_secs(10);

fn response(status: ResponseStatus) -> ResponseObject {
    let request: CreateResponseRequest = serde_json::from_value(serde_json::json!({
        "model": "log-test-model",
        "input": "hi",
        "instructions": "x".repeat(700),
    }))
    .expect("request parses");
    build_response_object(OutboundContext {
        response_id: "resp_log".to_string(),
        model_id: "log-test-model".to_string(),
        created_at: 0.0,
        completed_at: 0.0,
        status,
        prompt_tokens: 0,
        completion_tokens: 0,
        cached_tokens: 0,
        reasoning_tokens: 0,
        text: "done".to_string(),
        reasoning_text: None,
        parsed_tool_calls: None,
        max_tool_calls: None,
        request: &request,
        error: None,
        incomplete_reason: None,
        finish_reason: "stop".to_string(),
    })
}

fn delta(sequence_number: u64, text: &str) -> ResponseStreamEvent {
    ResponseStreamEvent::OutputTextDelta {
        sequence_number,
        item_id: "msg_1".to_string(),
        output_index: 0,
        content_index: 0,
        delta: text.to_string(),
    }
}

/// `(name, sequence_number)` of every retained event from `next` on.
fn retained(log: &BackgroundEventLog, next: u64) -> Vec<(&'static str, u64)> {
    match log.read_from(next, usize::MAX) {
        LogRead::Events(events) => events
            .into_iter()
            .map(|(sequence, name, json)| {
                let value: Value = serde_json::from_str(&json).expect("event is JSON");
                assert_eq!(value["sequence_number"], sequence, "{value}");
                assert_eq!(value["type"], name, "{value}");
                (name, sequence)
            })
            .collect(),
        other => panic!("expected events, got {other:?}"),
    }
}

/// Drain a reader to its end, as `(name, payload)`.
async fn drain(log: &Arc<BackgroundEventLog>, next: u64) -> Vec<(&'static str, Value)> {
    let frames = tail_frames(Arc::clone(log), "resp_log".to_string(), next);
    tokio::time::timeout(WAIT, frames.collect::<Vec<_>>())
        .await
        .expect("the reader ends")
        .into_iter()
        .map(|(name, json)| (name, serde_json::from_str(&json).expect("frame is JSON")))
        .collect()
}

#[test]
fn the_log_numbers_events_without_gaps_whatever_number_they_carry() {
    let log = BackgroundEventLog::new(1 << 20);
    assert_eq!(log.next_sequence_number(), 0);
    // Numbers that agree, repeat and jump: the log's own count wins.
    assert_eq!(log.append(&delta(0, "a")), Some(0));
    assert_eq!(log.append(&delta(0, "b")), Some(1));
    assert_eq!(log.append(&delta(40, "c")), Some(2));
    assert!(log.append_terminal(&response(ResponseStatus::Completed)));
    assert!(log.is_terminal());

    assert_eq!(
        retained(&log, 0),
        vec![
            ("response.output_text.delta", 0),
            ("response.output_text.delta", 1),
            ("response.output_text.delta", 2),
            ("response.completed", 3),
        ]
    );
    // A closed log takes nothing more.
    assert_eq!(log.append(&delta(4, "late")), None);
    assert!(!log.append_terminal(&response(ResponseStatus::Failed)));
    assert_eq!(log.next_sequence_number(), 4);
}

#[test]
fn the_terminal_event_follows_the_recorded_status() {
    for (status, name) in [
        (ResponseStatus::Completed, "response.completed"),
        (ResponseStatus::Cancelled, "response.completed"),
        (ResponseStatus::Incomplete, "response.incomplete"),
        (ResponseStatus::Failed, "response.failed"),
    ] {
        let log = BackgroundEventLog::new(1 << 20);
        log.append_terminal(&response(status));
        let LogRead::Events(events) = log.read_from(0, usize::MAX) else {
            panic!("the terminal event is retained");
        };
        assert_eq!(events[0].1, name);
        let value: Value = serde_json::from_str(&events[0].2).expect("event is JSON");
        assert_eq!(value["response"]["status"], status.as_str(), "{value}");
    }
}

#[test]
fn a_tiny_budget_bounds_the_bytes_and_keeps_the_terminal_event() {
    let budget = 600;
    let log = BackgroundEventLog::new(budget);
    let mut largest = 0;
    for i in 0..200 {
        log.append(&delta(i, &format!("token number {i}")));
        let LogRead::Events(events) = log.read_from(i, 1) else {
            panic!("the newest event is always retained");
        };
        largest = largest.max(event_bytes(events[0].1, events[0].2.len()));
        assert!(
            log.retained_bytes() <= budget,
            "{} bytes retained at event {i}",
            log.retained_bytes()
        );
    }
    assert!(largest < budget, "a delta fits the budget");
    // The oldest events were dropped, and the cursor says which is oldest.
    let oldest = log.resolve_cursor(None).expect_err("events were dropped");
    assert!(oldest > 0 && oldest < 200, "oldest retained: {oldest}");
    assert_eq!(log.resolve_cursor(Some(oldest - 1)), Ok(oldest));
    assert_eq!(log.resolve_cursor(Some(oldest - 2)), Err(oldest));
    assert!(matches!(
        log.read_from(oldest - 1, 1),
        LogRead::Evicted { oldest: reported } if reported == oldest
    ));

    // The terminal event is larger than the whole budget: it stays, alone.
    let terminal = response(ResponseStatus::Completed);
    let terminal_len = serde_json::to_string(&terminal).expect("serializes").len();
    assert!(
        terminal_len > budget,
        "the terminal object exceeds the budget"
    );
    log.append_terminal(&terminal);
    assert_eq!(retained(&log, 200), vec![("response.completed", 200)]);
    assert_eq!(log.resolve_cursor(None), Err(200));
    // Bounded by the budget plus the one event that is never dropped.
    assert!(log.retained_bytes() <= budget + terminal_len + 256);
    assert!(matches!(log.read_from(201, 1), LogRead::Closed));
}

#[tokio::test]
async fn readers_replay_then_tail_and_end_after_the_terminal_event() {
    let log = Arc::new(BackgroundEventLog::new(1 << 20));
    log.append(&delta(0, "a"));
    log.append(&delta(1, "b"));

    // One reader replays from the start, one resumes after event 0; both
    // are parked on the live tail when the rest arrives.
    let from_start = tokio::spawn({
        let log = Arc::clone(&log);
        async move { drain(&log, 0).await }
    });
    let resumed = tokio::spawn({
        let log = Arc::clone(&log);
        async move { drain(&log, 1).await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    log.append(&delta(2, "c"));
    tokio::time::sleep(Duration::from_millis(20)).await;
    log.append_terminal(&response(ResponseStatus::Completed));

    let from_start = from_start.await.expect("reader task");
    let resumed = resumed.await.expect("reader task");
    let sequences: Vec<u64> = from_start
        .iter()
        .map(|(_, value)| value["sequence_number"].as_u64().expect("numbered"))
        .collect();
    assert_eq!(sequences, vec![0, 1, 2, 3]);
    assert_eq!(from_start[3].0, "response.completed");
    assert_eq!(resumed, from_start[1..]);

    // After the terminal event a reader replays and ends at once, and a
    // cursor at or past the end gets nothing.
    assert_eq!(drain(&log, 0).await, from_start);
    assert!(drain(&log, 4).await.is_empty());
    assert!(drain(&log, 99).await.is_empty());
}

#[tokio::test]
async fn a_reader_that_falls_behind_gets_the_error_event_and_ends() {
    let log = Arc::new(BackgroundEventLog::new(600));
    log.append(&delta(0, "first"));
    let mut frames = Box::pin(tail_frames(Arc::clone(&log), "resp_log".to_string(), 0));
    let (name, _) = frames.next().await.expect("the first event");
    assert_eq!(name, "response.output_text.delta");

    // The reader stalls at sequence 1 while the log moves on.
    for i in 1..200 {
        log.append(&delta(i, &format!("token number {i}")));
    }
    let oldest = log
        .resolve_cursor(Some(0))
        .expect_err("event 1 was dropped");

    let (name, json) = frames.next().await.expect("the error event");
    assert_eq!(name, "response.error");
    let error: Value = serde_json::from_str(&json).expect("error is JSON");
    assert_eq!(error["code"], CODE_EVENTS_EVICTED, "{error}");
    assert_eq!(error["message"], evicted_cursor_message("resp_log", oldest));
    assert!(
        error["message"]
            .as_str()
            .expect("message")
            .contains(&format!("sequence_number {oldest}")),
        "{error}"
    );
    assert!(frames.next().await.is_none(), "the reader ends");
}
