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

//! SSE encoder for the OpenAI Responses API.
//!
//! Unlike `/v1/chat/completions` — which sends a single chat-completion-chunk
//! shape on every `data:` line — the Responses API uses typed SSE events:
//! each frame has both an `event:` header (the event name) and a `data:`
//! payload (the typed event object). This module owns the encoder.
//!
//! The channel pattern mirrors [`crate::server::streaming::sse_channel`]:
//! generation runs on a blocking task that calls `send_event` for each
//! event; the receiver maps each carrier into an `axum::sse::Event` with
//! both `event(name)` and `data(json)` set.
//!
//! ## Envelope ordering
//!
//! For a typical text-only generation, the encoder emits:
//!
//! 1. `response.created` (status `in_progress`)
//! 2. `response.output_item.added` (message in progress)
//! 3. `response.content_part.added` (output_text part empty)
//! 4. N × `response.output_text.delta` (one per emitted text token batch)
//! 5. `response.output_text.done`
//! 6. `response.content_part.done`
//! 7. `response.output_item.done`
//! 8. `response.completed` (final response with usage)

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive};
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::server::types::responses_stream::{ResponseStreamEvent, SequenceCounter};

const KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// Cancellation token shared with the scheduler for client-disconnect detection.
pub(crate) type CancellationToken = Arc<AtomicBool>;

/// One frame on the wire: the SSE `event:` header value plus the
/// JSON-serialised payload.
struct ResponseFrame {
    name: &'static str,
    data: String,
}

/// Blocking sender used by the generation task.
#[derive(Clone)]
pub struct ResponseStreamSender {
    tx: mpsc::Sender<Result<ResponseFrame, Infallible>>,
    cancelled: CancellationToken,
}

impl ResponseStreamSender {
    /// Send a typed Responses-API event. Silently drops the event when
    /// the receiver has been dropped (client disconnected), in which
    /// case the cancellation token is also flipped so the scheduler
    /// can abort the underlying sequence.
    pub fn send_event(&self, event: &ResponseStreamEvent) -> Result<(), serde_json::Error> {
        let payload = serde_json::to_string(event)?;
        let frame = ResponseFrame {
            name: event.event_name(),
            data: payload,
        };
        if self.tx.blocking_send(Ok(frame)).is_err() {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Newtype wrapping the keepalive configuration so it ships out of
/// the channel constructor and into the `Sse` response handler.
pub struct ResponseSseKeepAlive(KeepAlive);

impl ResponseSseKeepAlive {
    fn default_for_long_prefill() -> Self {
        Self(KeepAlive::new().interval(Duration::from_secs(KEEPALIVE_INTERVAL_SECS)))
    }

    pub fn into_inner(self) -> KeepAlive {
        self.0
    }
}

/// Construct a Responses-API SSE channel.
///
/// Returns `(sender, stream, cancelled, keepalive)`. The stream is fed
/// into `Sse::new(stream).keep_alive(keepalive.into_inner())`. The
/// sender accepts [`ResponseStreamEvent`] values from a blocking
/// generation task. The cancellation token flips when the receiver is
/// dropped so the scheduler can abort orphaned sequences.
pub fn responses_sse_channel(
    buffer: usize,
) -> (
    ResponseStreamSender,
    impl Stream<Item = Result<Event, Infallible>>,
    CancellationToken,
    ResponseSseKeepAlive,
) {
    let cancelled: CancellationToken = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Result<ResponseFrame, Infallible>>(buffer);
    let stream = ReceiverStream::new(rx).map(|frame| {
        frame.map(|f| {
            // `Event::default().event(name).data(data)` produces an SSE
            // frame of the form:
            //     event: <name>
            //     data: <data>
            //     <blank line>
            Event::default().event(f.name).data(f.data)
        })
    });
    let sender = ResponseStreamSender {
        tx,
        cancelled: cancelled.clone(),
    };
    (
        sender,
        stream,
        cancelled,
        ResponseSseKeepAlive::default_for_long_prefill(),
    )
}

/// Stateful event emitter used by the streaming route to keep
/// `sequence_number`, `output_index`, and `content_index` in lockstep
/// with the on-wire stream.
pub struct ResponseStreamEmitter {
    seq: SequenceCounter,
    output_index: usize,
    /// Item id of the active assistant message (when one is open).
    pub active_message_id: Option<String>,
    /// Item id of the active reasoning item (when one is open).
    pub active_reasoning_id: Option<String>,
    /// Accumulated output_text content for the active message item.
    pub message_text_acc: String,
    /// Accumulated reasoning content for the active reasoning item.
    pub reasoning_text_acc: String,
}

impl Default for ResponseStreamEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseStreamEmitter {
    pub fn new() -> Self {
        Self {
            seq: SequenceCounter::new(),
            output_index: 0,
            active_message_id: None,
            active_reasoning_id: None,
            message_text_acc: String::new(),
            reasoning_text_acc: String::new(),
        }
    }

    pub fn next_seq(&mut self) -> u64 {
        self.seq.next()
    }

    pub fn output_index(&self) -> usize {
        self.output_index
    }

    pub fn advance_output_index(&mut self) {
        self.output_index += 1;
    }

    pub fn open_message(&mut self, id: String) {
        self.active_message_id = Some(id);
        self.message_text_acc.clear();
    }

    pub fn close_message(&mut self) -> Option<(String, String)> {
        let id = self.active_message_id.take()?;
        let text = std::mem::take(&mut self.message_text_acc);
        Some((id, text))
    }

    pub fn open_reasoning(&mut self, id: String) {
        self.active_reasoning_id = Some(id);
        self.reasoning_text_acc.clear();
    }

    pub fn close_reasoning(&mut self) -> Option<(String, String)> {
        let id = self.active_reasoning_id.take()?;
        let text = std::mem::take(&mut self.reasoning_text_acc);
        Some((id, text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitter_next_seq_is_monotonic() {
        let mut emitter = ResponseStreamEmitter::new();
        assert_eq!(emitter.next_seq(), 0);
        assert_eq!(emitter.next_seq(), 1);
        assert_eq!(emitter.next_seq(), 2);
    }

    #[test]
    fn emitter_open_close_message_round_trips_text() {
        let mut emitter = ResponseStreamEmitter::new();
        emitter.open_message("msg_1".to_string());
        emitter.message_text_acc.push_str("hello");
        let (id, text) = emitter.close_message().expect("active");
        assert_eq!(id, "msg_1");
        assert_eq!(text, "hello");
        assert!(emitter.active_message_id.is_none());
        assert!(emitter.message_text_acc.is_empty());
    }

    #[test]
    fn emitter_advances_output_index() {
        let mut emitter = ResponseStreamEmitter::new();
        assert_eq!(emitter.output_index(), 0);
        emitter.advance_output_index();
        assert_eq!(emitter.output_index(), 1);
    }

    #[tokio::test]
    async fn channel_round_trips_event_with_header() {
        // `ResponseStreamSender::send_event` uses `blocking_send` because
        // generation runs on a `spawn_blocking` worker in production. We
        // mirror that here by sending from a `spawn_blocking` task while
        // the async receiver collects on the runtime worker.
        let (sender, stream, _cancelled, _keepalive) = responses_sse_channel(4);
        let send_task = tokio::task::spawn_blocking(move || {
            sender
                .send_event(&ResponseStreamEvent::OutputTextDelta {
                    sequence_number: 0,
                    item_id: "msg_1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    delta: "hi".to_string(),
                })
                .expect("send");
            // Dropping `sender` here closes the channel so the stream
            // terminates after the single event.
            drop(sender);
        });
        send_task.await.expect("blocking task");
        let collected: Vec<_> = stream.collect().await;
        assert_eq!(collected.len(), 1);
        let _ = collected.into_iter().next().unwrap().unwrap();
        // We can't introspect the produced `Event` directly (axum's
        // type is opaque), but the absence of a panic and the single
        // delivered item confirm the wiring.
    }
}
