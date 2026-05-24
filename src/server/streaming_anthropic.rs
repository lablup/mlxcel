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

//! SSE encoder for the Anthropic Messages API (`POST /v1/messages`).
//!
//! Mirrors [`crate::server::streaming_responses`]: each on-wire frame carries
//! an `event:` header (the event name) plus a typed `data:` payload. The
//! channel pattern matches [`crate::server::streaming::sse_channel`] —
//! generation runs on a blocking task that calls
//! [`AnthropicStreamSender::send_event`]; the receiver maps each frame into
//! an [`axum::response::sse::Event`].
//!
//! [`AnthropicBlockEmitter`] owns the Anthropic content-block index state
//! machine ported from the upstream `open_block` / `close_open_block`
//! closures in `server/anthropic.py`: there is at most one open block at a
//! time, and `index` increments each time a block closes.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive};
use futures::Stream;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::server::types::anthropic_response::AnthropicResponseBlock;
use crate::server::types::anthropic_stream::{AnthropicBlockDelta, AnthropicStreamEvent};

const KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// Cancellation token shared with the scheduler for client-disconnect detection.
pub(crate) type CancellationToken = Arc<AtomicBool>;

/// One frame on the wire: the SSE `event:` header value plus the
/// JSON-serialised payload.
struct AnthropicFrame {
    name: &'static str,
    data: String,
}

/// Blocking sender used by the generation task.
#[derive(Clone)]
pub struct AnthropicStreamSender {
    tx: mpsc::Sender<Result<AnthropicFrame, Infallible>>,
    cancelled: CancellationToken,
}

impl AnthropicStreamSender {
    /// Send a typed Anthropic SSE event. Silently drops the event when the
    /// receiver has been dropped (client disconnected); the cancellation
    /// token is flipped so the scheduler can abort the underlying sequence.
    pub fn send_event(&self, event: &AnthropicStreamEvent) -> Result<(), serde_json::Error> {
        let payload = serde_json::to_string(event)?;
        let frame = AnthropicFrame {
            name: event.event_name(),
            data: payload,
        };
        if self.tx.blocking_send(Ok(frame)).is_err() {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Newtype wrapping the keepalive configuration.
pub struct AnthropicSseKeepAlive(KeepAlive);

impl AnthropicSseKeepAlive {
    fn default_for_long_prefill() -> Self {
        // Anthropic clients accept `ping` events; the axum keepalive comment
        // frame is also tolerated. We use the standard keepalive comment to
        // match the Responses-API encoder.
        Self(KeepAlive::new().interval(Duration::from_secs(KEEPALIVE_INTERVAL_SECS)))
    }

    pub fn into_inner(self) -> KeepAlive {
        self.0
    }
}

/// Construct an Anthropic-API SSE channel.
///
/// Returns `(sender, stream, cancelled, keepalive)`. The stream is fed into
/// `Sse::new(stream).keep_alive(keepalive.into_inner())`. The cancellation
/// token flips when the receiver is dropped so the scheduler can abort
/// orphaned sequences.
pub fn anthropic_sse_channel(
    buffer: usize,
) -> (
    AnthropicStreamSender,
    impl Stream<Item = Result<Event, Infallible>>,
    CancellationToken,
    AnthropicSseKeepAlive,
) {
    let cancelled: CancellationToken = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Result<AnthropicFrame, Infallible>>(buffer);
    let stream = ReceiverStream::new(rx)
        .map(|frame| frame.map(|f| Event::default().event(f.name).data(f.data)));
    let sender = AnthropicStreamSender {
        tx,
        cancelled: cancelled.clone(),
    };
    (
        sender,
        stream,
        cancelled,
        AnthropicSseKeepAlive::default_for_long_prefill(),
    )
}

/// The kind of content block currently open in the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenBlock {
    Text,
    Thinking,
}

/// Stateful emitter that drives the Anthropic content-block index machine.
///
/// At most one block is open at a time. `block_index` is the index assigned
/// to the *next* block to open and is bumped on each close. Tool-use blocks
/// are emitted directly via [`AnthropicBlockEmitter::emit_tool_use`] (they
/// open and close in one shot) after any text/thinking block has closed.
pub struct AnthropicBlockEmitter {
    block_index: usize,
    open: Option<OpenBlock>,
}

impl Default for AnthropicBlockEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicBlockEmitter {
    pub fn new() -> Self {
        Self {
            block_index: 0,
            open: None,
        }
    }

    /// The index of the currently open (or next) block.
    pub fn current_index(&self) -> usize {
        self.block_index
    }

    /// Close the open block (if any), emitting `content_block_stop` and
    /// advancing the index. No-op when no block is open.
    pub fn close_open_block(&mut self, sender: &AnthropicStreamSender) {
        if self.open.is_none() {
            return;
        }
        let _ = sender.send_event(&AnthropicStreamEvent::ContentBlockStop {
            index: self.block_index,
        });
        self.open = None;
        self.block_index += 1;
    }

    /// Ensure a text block is open, closing any differently-typed block
    /// first. Emits `content_block_start` for a freshly opened block.
    pub fn open_text(&mut self, sender: &AnthropicStreamSender) {
        self.open_kind(sender, OpenBlock::Text);
    }

    /// Ensure a thinking block is open, closing any differently-typed block
    /// first. Emits `content_block_start` for a freshly opened block.
    pub fn open_thinking(&mut self, sender: &AnthropicStreamSender) {
        self.open_kind(sender, OpenBlock::Thinking);
    }

    fn open_kind(&mut self, sender: &AnthropicStreamSender, kind: OpenBlock) {
        if self.open == Some(kind) {
            return;
        }
        self.close_open_block(sender);
        let content_block = match kind {
            OpenBlock::Text => AnthropicResponseBlock::Text {
                text: String::new(),
            },
            OpenBlock::Thinking => AnthropicResponseBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
        };
        let _ = sender.send_event(&AnthropicStreamEvent::ContentBlockStart {
            index: self.block_index,
            content_block,
        });
        self.open = Some(kind);
    }

    /// Emit a text delta on the currently open text block.
    pub fn emit_text_delta(&self, sender: &AnthropicStreamSender, text: String) {
        let _ = sender.send_event(&AnthropicStreamEvent::ContentBlockDelta {
            index: self.block_index,
            delta: AnthropicBlockDelta::TextDelta { text },
        });
    }

    /// Emit a thinking delta on the currently open thinking block.
    pub fn emit_thinking_delta(&self, sender: &AnthropicStreamSender, thinking: String) {
        let _ = sender.send_event(&AnthropicStreamEvent::ContentBlockDelta {
            index: self.block_index,
            delta: AnthropicBlockDelta::ThinkingDelta { thinking },
        });
    }

    /// Emit a complete `tool_use` block: `content_block_start`, a single
    /// `input_json_delta` (when the JSON is non-empty), then
    /// `content_block_stop`. Advances the index. Any open text/thinking
    /// block must be closed first by the caller.
    pub fn emit_tool_use(
        &mut self,
        sender: &AnthropicStreamSender,
        id: String,
        name: String,
        input_json: &str,
    ) {
        let _ = sender.send_event(&AnthropicStreamEvent::ContentBlockStart {
            index: self.block_index,
            content_block: AnthropicResponseBlock::ToolUse {
                id,
                name,
                // The block starts with an empty input; the partial JSON is
                // streamed via input_json_delta to match the Anthropic
                // protocol.
                input: serde_json::json!({}),
            },
        });
        if !input_json.is_empty() {
            let _ = sender.send_event(&AnthropicStreamEvent::ContentBlockDelta {
                index: self.block_index,
                delta: AnthropicBlockDelta::InputJsonDelta {
                    partial_json: input_json.to_string(),
                },
            });
        }
        let _ = sender.send_event(&AnthropicStreamEvent::ContentBlockStop {
            index: self.block_index,
        });
        self.block_index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn channel_round_trips_event_with_header() {
        let (sender, stream, _cancelled, _keepalive) = anthropic_sse_channel(8);
        let send_task = tokio::task::spawn_blocking(move || {
            sender
                .send_event(&AnthropicStreamEvent::MessageStop)
                .expect("send");
            drop(sender);
        });
        send_task.await.expect("blocking task");
        let collected: Vec<_> = stream.collect().await;
        assert_eq!(collected.len(), 1);
        let _ = collected.into_iter().next().unwrap().unwrap();
    }

    #[tokio::test]
    async fn emitter_block_index_machine() {
        // Drive the emitter and collect the produced frames to confirm the
        // open/close/advance index machine matches the Anthropic protocol.
        let (sender, stream, _cancelled, _keepalive) = anthropic_sse_channel(64);
        let task = tokio::task::spawn_blocking(move || {
            let mut em = AnthropicBlockEmitter::new();
            // Open text, emit a delta.
            em.open_text(&sender);
            em.emit_text_delta(&sender, "hi".to_string());
            assert_eq!(em.current_index(), 0);
            // Switch to thinking (closes text → index 1).
            em.open_thinking(&sender);
            assert_eq!(em.current_index(), 1);
            em.emit_thinking_delta(&sender, "r".to_string());
            // Close the thinking block (→ index 2).
            em.close_open_block(&sender);
            assert_eq!(em.current_index(), 2);
            // Tool use opens+closes in one shot (→ index 3).
            em.emit_tool_use(&sender, "toolu_1".to_string(), "f".to_string(), "{\"a\":1}");
            assert_eq!(em.current_index(), 3);
            drop(sender);
        });
        task.await.expect("task");
        let frames: Vec<_> = stream.collect().await;
        // Expected on-wire frames:
        //  start(0,text), delta(0), stop(0),
        //  start(1,thinking), delta(1), stop(1),
        //  start(2,tool_use), delta(2,input_json), stop(2)
        assert_eq!(frames.len(), 9);
        for f in frames {
            let _ = f.unwrap();
        }
    }

    #[tokio::test]
    async fn emitter_idempotent_open_same_kind() {
        let (sender, stream, _cancelled, _keepalive) = anthropic_sse_channel(16);
        let task = tokio::task::spawn_blocking(move || {
            let mut em = AnthropicBlockEmitter::new();
            em.open_text(&sender);
            em.open_text(&sender); // no-op, same kind
            em.emit_text_delta(&sender, "x".to_string());
            assert_eq!(em.current_index(), 0);
            drop(sender);
        });
        task.await.expect("task");
        let frames: Vec<_> = stream.collect().await;
        // Only one start + one delta — the second open_text is a no-op.
        assert_eq!(frames.len(), 2);
    }
}
