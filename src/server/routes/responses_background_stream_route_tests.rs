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

//! Streaming and resuming a background response through the real router
//! against a scripted model (issue #2900): `stream: true` on a background
//! create, and `GET /v1/responses/{id}?stream=true&starting_after=N`.
//!
//! A child of the background route tests, for their harness.

use http_body_util::BodyExt;

use super::*;

/// One parsed SSE event: its `event:` name and its JSON payload.
type SseEvent = (String, Value);

fn streamed_body() -> Value {
    let mut body = background_body();
    body["stream"] = json!(true);
    body
}

/// A harness whose event logs retain about `bytes`.
fn harness_with_log_bytes(bytes: usize) -> Harness {
    harness_with(ResponsesStoreConfig {
        background_event_log_bytes: bytes,
        ..ResponsesStoreConfig::default()
    })
}

/// Incremental reader of one SSE response body.
struct SseReader {
    body: Body,
    buffer: String,
}

impl SseReader {
    fn new(response: Response) -> Self {
        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response.headers()[header::CONTENT_TYPE]
            .to_str()
            .expect("header");
        assert!(
            content_type.starts_with("text/event-stream"),
            "{content_type}"
        );
        Self {
            body: response.into_body(),
            buffer: String::new(),
        }
    }

    /// The next frame: `Some(Some(event))`, `Some(None)` for a keepalive
    /// comment, `None` at the end of the stream.
    async fn frame(&mut self) -> Option<Option<SseEvent>> {
        loop {
            if let Some(end) = self.buffer.find("\n\n") {
                let raw: String = self.buffer.drain(..end + 2).collect();
                let mut name = None;
                let mut data = None;
                for line in raw.lines() {
                    if let Some(value) = line.strip_prefix("event: ") {
                        name = Some(value.to_string());
                    } else if let Some(value) = line.strip_prefix("data: ") {
                        data = Some(serde_json::from_str(value).expect("data is JSON"));
                    }
                }
                return Some(name.zip(data));
            }
            let frame = tokio::time::timeout(WAIT, self.body.frame())
                .await
                .expect("a frame arrives in time")?
                .expect("frame ok");
            if let Some(data) = frame.data_ref() {
                self.buffer.push_str(&String::from_utf8_lossy(data));
            }
        }
    }

    /// The next event, skipping keepalive comments.
    async fn event(&mut self) -> Option<SseEvent> {
        loop {
            if let Some(event) = self.frame().await? {
                return Some(event);
            }
        }
    }

    /// Read events up to and including the first one named `name`.
    async fn through(&mut self, name: &str) -> Vec<SseEvent> {
        let mut events = Vec::new();
        loop {
            let event = self.event().await.expect("the stream is still open");
            let found = event.0 == name;
            events.push(event);
            if found {
                return events;
            }
        }
    }

    /// Every remaining event, to the end of the stream.
    async fn rest(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        while let Some(event) = self.event().await {
            events.push(event);
        }
        events
    }
}

impl Harness {
    /// Create a streamed background response; returns its reader, having
    /// read `response.created`, and its id.
    async fn create_streamed(&self, body: Value) -> (SseReader, String, SseEvent) {
        let response = self.raw(Method::POST, "/v1/responses", Some(body)).await;
        let mut reader = SseReader::new(response);
        let created = reader.event().await.expect("response.created");
        assert_eq!(created.0, "response.created", "{created:?}");
        let id = created.1["response"]["id"]
            .as_str()
            .expect("id")
            .to_string();
        (reader, id, created)
    }

    /// `GET /v1/responses/{id}?stream=true`, plus `starting_after`.
    async fn resume(&self, id: &str, starting_after: Option<u64>) -> Response {
        let cursor = starting_after.map_or(String::new(), |n| format!("&starting_after={n}"));
        let path = format!("/v1/responses/{id}?stream=true{cursor}");
        self.raw(Method::GET, &path, None).await
    }

    /// Every event a resume at `starting_after` delivers.
    async fn replay(&self, id: &str, starting_after: Option<u64>) -> Vec<SseEvent> {
        SseReader::new(self.resume(id, starting_after).await)
            .rest()
            .await
    }

    async fn error_of(&self, response: Response) -> (StatusCode, Value) {
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body collects");
        (
            status,
            serde_json::from_slice(&bytes).expect("error is JSON"),
        )
    }
}

fn names(events: &[SseEvent]) -> Vec<&str> {
    events.iter().map(|(name, _)| name.as_str()).collect()
}

fn sequences(events: &[SseEvent]) -> Vec<u64> {
    events
        .iter()
        .map(|(name, data)| {
            assert_eq!(data["type"], name.as_str(), "{data}");
            data["sequence_number"].as_u64().expect("sequence_number")
        })
        .collect()
}

/// Assert `events` are numbered `first, first + 1, ...` without a gap.
fn assert_contiguous_from(events: &[SseEvent], first: u64) {
    let expected: Vec<u64> = (first..first + events.len() as u64).collect();
    assert_eq!(sequences(events), expected);
}

#[tokio::test]
async fn a_streamed_create_yields_the_ordered_events_with_increasing_sequence_numbers() {
    let h = harness();
    // Sent although background jobs ignore it: no resumable session appears.
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-conversation-id", "conv-ignored")
        .body(Body::from(streamed_body().to_string()))
        .expect("request builds");
    let response = h
        .app
        .clone()
        .oneshot(request)
        .await
        .expect("router responds");
    let mut reader = SseReader::new(response);
    // The two opening events arrive before anything is generated.
    let mut events = vec![
        reader.event().await.expect("response.created"),
        reader.event().await.expect("response.queued"),
    ];
    let id = events[0].1["response"]["id"]
        .as_str()
        .expect("id")
        .to_string();
    for opening in &events {
        assert_eq!(opening.1["response"]["status"], "queued", "{opening:?}");
        assert_eq!(opening.1["response"]["background"], true, "{opening:?}");
    }
    assert_eq!(h.get(&id).await.1["status"], "queued");

    h.handle.token("Hello.");
    h.handle.finish();
    events.extend(reader.rest().await);

    assert_eq!(
        names(&events),
        vec![
            "response.created",
            "response.queued",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_contiguous_from(&events, 0);
    assert_eq!(events[2].1["response"]["status"], "in_progress");
    let terminal = &events.last().expect("terminal").1["response"];
    assert_eq!(terminal["status"], "completed", "{terminal}");
    assert_eq!(terminal["output_text"], "Hello.", "{terminal}");
    // The terminal event carries the object a poll reads.
    assert_eq!(&h.get(&id).await.1, terminal);
    assert_eq!(h.store.active_background_jobs(), 0);
    assert!(h.state.stream_sessions.get("conv-ignored", &None).is_none());
}

#[tokio::test]
async fn a_disconnect_leaves_the_job_running_and_a_resume_continues_after_the_cursor() {
    let h = harness();
    let (mut reader, id, created) = h.create_streamed(streamed_body()).await;
    h.handle.token("Hel");
    let mut seen = vec![created];
    seen.extend(reader.through("response.output_text.delta").await);
    assert_contiguous_from(&seen, 0);
    let last_seen = *sequences(&seen).last().expect("events were read");
    // The client goes away mid-stream.
    drop(reader);

    h.handle.token("lo.");
    h.handle.finish();
    let done = h.terminal(&id).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(done["output_text"], "Hello.", "{done}");
    let flag = h.handle.cancellation_flag(0).expect("one generation ran");
    assert!(
        !flag.load(Ordering::Relaxed),
        "the disconnect cancelled nothing"
    );

    // Resume at the cursor: exactly the events after it, to the terminal.
    let resumed = h.replay(&id, Some(last_seen)).await;
    assert_contiguous_from(&resumed, last_seen + 1);
    assert_eq!(names(&resumed).last(), Some(&"response.completed"));
    // Joined, the two parts are the whole stream: no gap and no duplicate.
    let whole = h.replay(&id, None).await;
    seen.extend(resumed);
    assert_eq!(seen, whole);
    // After the terminal event a replay ends at once, wherever it starts.
    let terminal_sequence = *sequences(&whole).last().expect("terminal");
    assert_eq!(h.replay(&id, Some(terminal_sequence - 1)).await.len(), 1);
    assert!(h.replay(&id, Some(terminal_sequence)).await.is_empty());
}

#[tokio::test]
async fn two_concurrent_readers_receive_identical_sequences() {
    let h = harness();
    let (mut tailing, id, created) = h.create_streamed(streamed_body()).await;
    h.handle.token("Hel");
    let mut live = vec![created];
    live.extend(tailing.through("response.output_text.delta").await);

    // A second reader joins mid-stream and replays from the start.
    let mut replaying = SseReader::new(h.resume(&id, None).await);
    let mut replayed = replaying.through("response.output_text.delta").await;
    assert_eq!(replayed, live);

    h.handle.token("lo.");
    h.handle.finish();
    live.extend(tailing.rest().await);
    replayed.extend(replaying.rest().await);
    assert_eq!(replayed, live);
    assert_contiguous_from(&live, 0);
    assert_eq!(names(&live).last(), Some(&"response.completed"));
}

#[tokio::test]
async fn a_cancelled_stream_ends_with_response_completed_and_status_cancelled() {
    let h = harness();
    // Cancelled mid-generation.
    let (mut reader, id, _) = h.create_streamed(streamed_body()).await;
    h.handle.token("Hel");
    reader.through("response.output_text.delta").await;
    assert!(h.store.cancel_in_flight(&id));
    h.handle.finish();
    let rest = reader.rest().await;
    let (name, terminal) = rest.last().expect("a terminal event");
    assert_eq!(name, "response.completed");
    assert_eq!(terminal["response"]["status"], "cancelled", "{terminal}");
    assert_eq!(h.get(&id).await.1["status"], "cancelled");

    // Cancelled while queued: the job task has not run yet on this
    // single-threaded runtime, so the terminal event follows the opening.
    let queued = spawn_background_job(h.job(streamed_body())).expect("job is admitted");
    assert!(h.store.cancel_in_flight(&queued.id));
    let events = h.replay(&queued.id, None).await;
    assert_eq!(
        names(&events),
        vec!["response.created", "response.queued", "response.completed"]
    );
    assert_contiguous_from(&events, 0);
    assert_eq!(events[2].1["response"]["status"], "cancelled");
    assert_eq!(events[2].1["response"]["output"], json!([]));
}

#[tokio::test]
async fn keepalive_frames_arrive_while_the_job_is_paused() {
    let h = harness_full(
        Some(ResponsesStoreConfig::default()),
        ServerConfig {
            sse_ping_interval: Some(Duration::from_millis(20)),
            ..ServerConfig::default()
        },
        MlxcelTokenizer::stub(),
    );
    let (mut reader, id, _) = h.create_streamed(streamed_body()).await;
    assert_eq!(
        reader.event().await.expect("response.queued").0,
        "response.queued"
    );
    // Nothing is scripted: the stream stays open and only pings arrive.
    for _ in 0..3 {
        assert_eq!(reader.frame().await, Some(None), "a keepalive comment");
    }
    // A resumed reader gets the same keepalive.
    let mut resumed = SseReader::new(h.resume(&id, Some(1)).await);
    assert_eq!(resumed.frame().await, Some(None), "a keepalive comment");

    h.handle.token("Hello.");
    h.handle.finish();
    assert_eq!(
        names(&reader.rest().await).last(),
        Some(&"response.completed")
    );
    assert_eq!(
        names(&resumed.rest().await).last(),
        Some(&"response.completed")
    );
}

#[path = "responses_background_stream_errors_route_tests.rs"]
mod errors;
