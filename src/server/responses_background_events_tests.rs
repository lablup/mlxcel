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

/// The oldest retained sequence number a cursor that lost events reports.
fn evicted_oldest(log: &BackgroundEventLog, starting_after: Option<u64>) -> u64 {
    match log.resolve_cursor(starting_after) {
        Err(CursorError::Evicted { oldest }) => oldest,
        other => panic!("expected an evicted cursor, got {other:?}"),
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
    let oldest = evicted_oldest(&log, None);
    assert!(oldest > 0 && oldest < 200, "oldest retained: {oldest}");
    assert_eq!(log.resolve_cursor(Some(oldest - 1)), Ok(oldest));
    assert_eq!(
        log.resolve_cursor(Some(oldest - 2)),
        Err(CursorError::Evicted { oldest })
    );
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
    assert_eq!(
        log.resolve_cursor(None),
        Err(CursorError::Evicted { oldest: 200 })
    );
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

#[test]
fn a_cursor_past_the_newest_event_is_refused() {
    let log = BackgroundEventLog::new(1 << 20);
    assert_eq!(
        log.resolve_cursor(Some(0)),
        Err(CursorError::Ahead { newest: None })
    );
    log.append(&delta(0, "a"));
    log.append(&delta(1, "b"));
    // The newest event is a valid cursor (a live tail); one past it is not.
    assert_eq!(log.resolve_cursor(Some(1)), Ok(2));
    for after in [2, 3, u64::MAX] {
        assert_eq!(
            log.resolve_cursor(Some(after)),
            Err(CursorError::Ahead { newest: Some(1) }),
            "{after}"
        );
    }
    log.append_terminal(&response(ResponseStatus::Completed));
    assert_eq!(log.resolve_cursor(Some(2)), Ok(3));
    assert_eq!(
        log.resolve_cursor(Some(3)),
        Err(CursorError::Ahead { newest: Some(2) })
    );
}

#[test]
fn a_prepared_terminal_event_is_used_only_under_its_own_number() {
    let terminal = response(ResponseStatus::Completed);
    let log = BackgroundEventLog::new(1 << 20);
    log.append(&delta(0, "a"));
    let prepared = log.prepare_terminal(&terminal).expect("serializes");
    assert!(log.append_terminal_prepared(&terminal, Some(prepared)));
    assert_eq!(
        retained(&log, 0),
        vec![("response.output_text.delta", 0), ("response.completed", 1)]
    );

    // An event appended after the terminal one was prepared moves it on.
    let log = BackgroundEventLog::new(1 << 20);
    let prepared = log.prepare_terminal(&terminal).expect("serializes");
    log.append(&delta(0, "a"));
    let failed = response(ResponseStatus::Failed);
    assert!(log.append_terminal_prepared(&failed, Some(prepared)));
    assert_eq!(
        retained(&log, 0),
        vec![("response.output_text.delta", 0), ("response.failed", 1)]
    );
}

#[tokio::test]
async fn a_reader_does_not_keep_a_dropped_log_alive() {
    let log = Arc::new(BackgroundEventLog::new(1 << 20));
    for i in 0..3 {
        log.append(&delta(i, "x"));
    }
    log.append_terminal(&response(ResponseStatus::Completed));
    let weak = Arc::downgrade(&log);
    let mut frames = Box::pin(tail_frames(Arc::clone(&log), "resp_log".to_string(), 0));
    let (name, _) = frames.next().await.expect("the first event");
    assert_eq!(name, "response.output_text.delta");

    // The store drops the log while the reader is mid-replay: the reader's
    // buffered batch is all it still holds.
    drop(log);
    assert!(weak.upgrade().is_none(), "the reader holds the log weakly");
    let rest: Vec<_> = frames.collect().await;
    let names: Vec<_> = rest.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        vec![
            "response.output_text.delta",
            "response.output_text.delta",
            "response.completed"
        ]
    );

    // A reader that had not read the batch yet gets the not-retained error.
    let log = Arc::new(BackgroundEventLog::new(1 << 20));
    log.append(&delta(0, "x"));
    let frames = tail_frames(Arc::clone(&log), "resp_log".to_string(), 0);
    drop(log);
    let frames: Vec<_> = frames.collect().await;
    assert_eq!(frames.len(), 1, "{frames:?}");
    assert_eq!(frames[0].0, "response.error");
    let error: Value = serde_json::from_str(&frames[0].1).expect("error is JSON");
    assert_eq!(error["code"], CODE_EVENTS_EVICTED, "{error}");
    assert_eq!(error["message"], log_not_retained_message("resp_log"));
}

#[test]
fn a_read_batch_is_capped_by_bytes_but_always_carries_one_event() {
    let log = BackgroundEventLog::new(1 << 20);
    for i in 0..10 {
        log.append(&delta(i, &"y".repeat(1000)));
    }
    let LogRead::Events(events) = log.read_bounded(0, usize::MAX, 2500) else {
        panic!("events are retained");
    };
    assert_eq!(events.len(), 2);
    let LogRead::Events(events) = log.read_bounded(0, usize::MAX, 1) else {
        panic!("events are retained");
    };
    assert_eq!(events.len(), 1);
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
    let oldest = evicted_oldest(&log, Some(0));

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
