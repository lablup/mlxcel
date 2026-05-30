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

//! Shared SSE helpers for server routes.
//!
//! Chat, completion, and llama-server-compatible routes all stream over the
//! same blocking channel pattern even though their payload shapes differ.
//!
//! ## Long-prefill keepalive
//!
//! When a prompt is large (e.g. 32k+ tokens), the batch scheduler may spend
//! tens of seconds running the prefill forward pass before emitting the first
//! generated token. During that window the SSE stream is open but silent.
//! Reverse proxies and HTTP clients that apply per-stream idle timeouts (nginx
//! `proxy_read_timeout`, HAProxy `timeout tunnel`, AWS ALB 60 s default, etc.)
//! will drop the connection before the first token arrives.
//!
//! `sse_channel` returns a keepalive configuration via `SseKeepAlive` that
//! route handlers must attach to the `Sse` response with `.keep_alive()`. The
//! keepalive interval is 15 seconds — shorter than typical proxy idle timeouts
//! and long enough not to spam comment events for short responses.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive};
use futures::{Stream, StreamExt};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub(crate) const DONE_MARKER: &str = "[DONE]";

/// Default interval at which SSE keepalive comment events are sent during
/// long prefills so that proxies and HTTP clients do not time out the
/// connection before the first token arrives.
///
/// 15 seconds is shorter than virtually all proxy idle timeouts (nginx
/// default 60 s, HAProxy 60 s, AWS ALB 60 s) while being long enough to
/// avoid noticeable overhead for ordinary short responses.
pub(crate) const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 15;

/// Keepalive configuration attached to an `Sse` response.
///
/// Constructed by `sse_channel` and passed through to route handlers so they
/// can call `Sse::new(stream).keep_alive(keepalive.into_inner())`. Using a
/// newtype keeps the keepalive wired to the same channel creation point and
/// makes it impossible to forget to attach it. The inner `KeepAlive` is private
/// to prevent callers from constructing a mismatched keepalive independently.
///
/// Used by: chat.rs, completions.rs, native_completion.rs
pub(crate) struct SseKeepAlive(KeepAlive);

impl SseKeepAlive {
    /// Build a keepalive that sends an empty comment every
    /// [`SSE_KEEPALIVE_INTERVAL_SECS`] seconds to prevent proxy timeouts
    /// during long prefill phases.
    ///
    /// `KeepAlive::new()` already emits an empty SSE comment by default, so
    /// only the interval needs to be customised.
    pub(crate) fn default_for_long_prefill() -> Self {
        Self(KeepAlive::new().interval(Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)))
    }

    /// Unwrap the inner `KeepAlive` for attaching to an `Sse` response.
    ///
    /// Consuming `self` ensures each `SseKeepAlive` is applied exactly once.
    pub(crate) fn into_inner(self) -> KeepAlive {
        self.0
    }
}

type SsePayload = Result<String, Infallible>;

/// A cancellation token shared between the SSE sender and the scheduler.
///
/// Set to `true` when the SSE channel detects that the client has disconnected
/// (i.e. `blocking_send` returns `Err`). The `BatchScheduler` polls this flag
/// to abort orphaned sequences promptly.
pub(crate) type CancellationToken = Arc<AtomicBool>;

#[derive(Clone)]
pub(crate) struct BlockingSseSender {
    tx: mpsc::Sender<SsePayload>,
    /// Shared flag set to `true` when the client disconnects (SSE receiver is
    /// dropped). Checked by the `BatchScheduler` to cancel orphaned sequences.
    cancelled: Option<CancellationToken>,
}

/// Create an SSE channel with a cancellation token.
///
/// Returns `(sender, stream, cancellation_token, keepalive)`. The cancellation
/// token is an `Arc<AtomicBool>` that is set to `true` when
/// `BlockingSseSender::text()` detects the client has disconnected (SSE
/// receiver dropped). Pass the token to `ModelRequest::Generate` so the
/// `BatchScheduler` can abort orphaned sequences.
///
/// The `keepalive` value must be attached to the `Sse` response via
/// `Sse::new(stream).keep_alive(keepalive.0)` in the route handler. This
/// ensures proxy idle timeouts do not close the connection during long prefill
/// phases before the first generated token arrives.
///
/// Used by: chat.rs, completions.rs, native_completion.rs
pub(crate) fn sse_channel(
    buffer: usize,
) -> (
    BlockingSseSender,
    impl Stream<Item = Result<Event, Infallible>>,
    CancellationToken,
    SseKeepAlive,
) {
    let cancelled: CancellationToken = Arc::new(AtomicBool::new(false));
    let (sender, rx) = payload_channel(buffer, Some(cancelled.clone()));
    let stream = ReceiverStream::new(rx).map(|payload| payload.map(sse_event));
    let keepalive = SseKeepAlive::default_for_long_prefill();
    (sender, stream, cancelled, keepalive)
}

impl BlockingSseSender {
    pub(crate) fn json<T: Serialize>(&self, value: &T) -> Result<(), serde_json::Error> {
        self.text(serialize_json_data(value)?);
        Ok(())
    }

    pub(crate) fn text(&self, data: impl Into<String>) {
        if self.tx.blocking_send(Ok(data.into())).is_err() {
            // The SSE receiver has been dropped, meaning the client
            // disconnected. Signal cancellation so the BatchScheduler can
            // abort the orphaned sequence.
            if let Some(ref flag) = self.cancelled {
                flag.store(true, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn done(&self) {
        self.text(DONE_MARKER);
    }
}

fn payload_channel(
    buffer: usize,
    cancelled: Option<CancellationToken>,
) -> (BlockingSseSender, mpsc::Receiver<SsePayload>) {
    let (tx, rx) = mpsc::channel(buffer);
    (BlockingSseSender { tx, cancelled }, rx)
}

fn serialize_json_data<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_string(value)
}

fn sse_event(data: String) -> Event {
    Event::default().data(data)
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
