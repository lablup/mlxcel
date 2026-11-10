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

//! The edges of streaming and resuming a background response (issue
//! #2900): dropped events, failed and cut-off jobs, responses without an
//! event log, and bad query values.

use super::super::*;
use super::{SseReader, assert_contiguous_from, harness_with_log_bytes, names, streamed_body};
use crate::server::responses_background_events::CODE_EVENTS_EVICTED;

#[tokio::test]
async fn an_evicted_cursor_returns_400_naming_the_oldest_sequence_number() {
    let h = harness_with_log_bytes(4096);
    let (reader, id, _) = h.create_streamed(streamed_body()).await;
    drop(reader);
    for i in 0..60 {
        h.handle.token(&format!("token{i} "));
    }
    h.handle.finish();
    assert_eq!(h.terminal(&id).await["status"], "completed");

    let log = h
        .store
        .background_event_log(&id)
        .expect("the log is retained");
    let oldest = log.resolve_cursor(None).expect_err("events were dropped");
    assert!(oldest > 1, "oldest retained: {oldest}");
    // With no cursor, and with one below the retained window.
    for cursor in [None, Some(0), Some(oldest - 2)] {
        let (status, error) = h.error_of(h.resume(&id, cursor).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{cursor:?}: {error}");
        assert_eq!(error["error"]["type"], "invalid_request_error", "{error}");
        let message = error["error"]["message"].as_str().expect("message");
        assert!(
            message.contains(&format!("sequence_number {oldest}")),
            "{message}"
        );
        assert!(
            message.contains(&format!("GET /v1/responses/{id}")),
            "{message}"
        );
    }
    // The oldest retained event minus one is still a valid cursor.
    let resumed = h.replay(&id, Some(oldest - 1)).await;
    assert_contiguous_from(&resumed, oldest);
    assert_eq!(names(&resumed).last(), Some(&"response.completed"));
}

#[tokio::test]
async fn a_slow_reader_that_is_overrun_gets_the_error_event() {
    let h = harness_with_log_bytes(4096);
    let (mut reader, id, _) = h.create_streamed(streamed_body()).await;
    let queued = reader.event().await.expect("response.queued");
    assert_eq!(queued.0, "response.queued");

    // The reader stalls while the job runs to its end.
    for i in 0..60 {
        h.handle.token(&format!("token{i} "));
    }
    h.handle.finish();
    let done = h.terminal(&id).await;
    assert_eq!(done["status"], "completed", "the job is unaffected: {done}");

    let rest = reader.rest().await;
    assert_eq!(names(&rest), vec!["response.error"], "{rest:?}");
    let error = &rest[0].1;
    assert_eq!(error["code"], CODE_EVENTS_EVICTED, "{error}");
    let message = error["message"].as_str().expect("message");
    assert!(message.contains("oldest retained event"), "{message}");
    assert!(
        message.contains(&format!("GET /v1/responses/{id}")),
        "{message}"
    );
}

#[tokio::test]
async fn a_failure_ends_with_response_failed_and_a_cut_off_reply_with_response_incomplete() {
    let h = harness();
    let (mut reader, id, created) = h.create_streamed(streamed_body()).await;
    h.handle.fail("boom");
    let mut events = vec![created];
    events.extend(reader.rest().await);
    assert_contiguous_from(&events, 0);
    // The foreground's error event never appears on a background stream.
    assert_eq!(
        names(&events),
        vec!["response.created", "response.queued", "response.failed"]
    );
    let failed = &events[2].1["response"];
    assert_eq!(failed["status"], "failed", "{failed}");
    assert_eq!(failed["error"]["code"], "server_error", "{failed}");
    assert!(failed["error"]["message"].to_string().contains("boom"));
    assert_eq!(&h.get(&id).await.1, failed);

    // A failure outside the driver (a panic) still closes the stream.
    let job = h.job(streamed_body());
    let queued = spawn_background_job_with(job, |_run, _cancelled, _sink| {
        panic!("driver panic under test")
    })
    .expect("job is admitted");
    let events = h.replay(&queued.id, None).await;
    assert_eq!(names(&events).last(), Some(&"response.failed"));
    assert_contiguous_from(&events, 0);

    let h = harness_full(
        Some(ResponsesStoreConfig::default()),
        ServerConfig {
            context_size: 512,
            ..ServerConfig::default()
        },
        MlxcelTokenizer::stub_all_byte_fallback(),
    );
    let mut body = streamed_body();
    body["context_management"] = json!([{"type": "compaction", "compact_threshold": 10_000}]);
    let (mut reader, _, _) = h.create_streamed(body).await;
    h.handle.token("Hel");
    h.handle.finish_length();
    let events = reader.rest().await;
    let (name, terminal) = events.last().expect("a terminal event");
    assert_eq!(name, "response.incomplete");
    assert_eq!(terminal["response"]["status"], "incomplete", "{terminal}");
}

#[tokio::test]
async fn get_without_stream_is_unchanged_and_a_response_without_a_log_returns_400() {
    let h = harness();
    let (reader, streamed, _) = h.create_streamed(streamed_body()).await;
    drop(reader);
    let polled = h.create_background(background_body()).await;
    for _ in 0..2 {
        h.handle.token("Hello.");
        h.handle.finish();
    }
    h.terminal(&streamed).await;
    h.terminal(&polled).await;
    h.handle.token("Hello.");
    h.handle.finish();
    let (_, foreground) = h
        .create(json!({"model": "background-test-model", "input": "hi"}))
        .await;
    let foreground = foreground["id"].as_str().expect("id").to_string();

    // Without `stream=true` the body is the JSON object, whatever else the
    // query carries.
    for id in [&streamed, &polled, &foreground] {
        let (status, plain) = h.get(id).await;
        assert_eq!(status, StatusCode::OK, "{plain}");
        assert_eq!(plain["id"], id.as_str(), "{plain}");
        for query in [
            "stream=false",
            "starting_after=3",
            "stream=false&starting_after=x",
        ] {
            let path = format!("/v1/responses/{id}?{query}");
            let (status, body) = h.send(Method::GET, &path, None).await;
            assert_eq!(status, StatusCode::OK, "{query}: {body}");
            assert_eq!(body, plain, "{query}");
        }
    }
    // The alias route shares the handler.
    let alias = format!("/responses/{streamed}?stream=true");
    let events = SseReader::new(h.raw(Method::GET, &alias, None).await)
        .rest()
        .await;
    assert_eq!(names(&events).last(), Some(&"response.completed"));

    // No log: a foreground response, and a background one without `stream`.
    for id in [&polled, &foreground] {
        let (status, error) = h.error_of(h.resume(id, None).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert_eq!(error["error"]["type"], "invalid_request_error", "{error}");
        let message = error["error"]["message"].as_str().expect("message");
        assert!(message.contains("stream: true"), "{message}");
    }
    let (status, error) = h.error_of(h.resume("resp_missing", None).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{error}");
    // A cursor that is not a sequence number, and a `stream` that is not a
    // boolean.
    for query in [
        "stream=true&starting_after=abc",
        "stream=true&starting_after=-1",
        "stream=maybe",
    ] {
        let path = format!("/v1/responses/{streamed}?{query}");
        let (status, error) = h.error_of(h.raw(Method::GET, &path, None).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{query}: {error}");
        assert_eq!(error["error"]["type"], "invalid_request_error", "{error}");
    }
}

#[tokio::test]
async fn a_zero_event_log_budget_refuses_a_streamed_background_create() {
    let h = harness_with_log_bytes(0);
    let (status, reply) = h.create(streamed_body()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert_eq!(reply["error"]["type"], "invalid_request_error", "{reply}");
    assert!(
        reply["error"]["message"]
            .to_string()
            .contains("--responses-background-event-log-bytes"),
        "{reply}"
    );
    assert_eq!(h.store.active_background_jobs(), 0);
    // Polling mode is unaffected.
    let id = h.create_background(background_body()).await;
    h.handle.finish();
    assert_eq!(h.terminal(&id).await["status"], "completed");
}
