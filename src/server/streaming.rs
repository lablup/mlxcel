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
//! `sse_channel` returns a keepalive configuration via `SseKeepAlive`, which a
//! route handler passes with its stream to [`sse_response`]. That constructor
//! is the single place the keepalive is attached, so a route cannot omit it
//! (#1107). Streams that do not come from `sse_channel` (`router_front.rs`)
//! build the same newtype directly rather than reaching for
//! `KeepAlive::default()`. The interval is
//! [`SSE_KEEPALIVE_INTERVAL_SECS`], short enough to beat typical
//! proxy idle timeouts and long enough not to spam comment events for short
//! responses. That constant is the single definition for every SSE surface in
//! the server, including the Responses and Anthropic-compatible ones, and the
//! `< 60` invariant is asserted next to it so it cannot be raised past a proxy
//! timeout on any of them.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub(crate) const DONE_MARKER: &str = "[DONE]";

/// Default interval at which SSE keepalive comment events are sent during
/// long prefills so that proxies and HTTP clients do not time out the
/// connection before the first token arrives.
///
/// 30 seconds is the `llama-server` b10621 `--sse-ping-interval` default
/// (#1432), and is shorter than virtually all proxy idle timeouts (nginx
/// default 60 s, HAProxy 60 s, AWS ALB 60 s) while being long enough to
/// avoid noticeable overhead for ordinary short responses. An operator can
/// override it per server with `--sse-ping-interval`, or per request with the
/// native `sse_ping_interval` field; `-1` / `None` disables the pings.
///
/// This is the single definition for every SSE surface. `streaming_responses`
/// and `streaming_anthropic` keep their own newtypes, so a route cannot attach
/// another surface's keepalive, but they read the interval from here (#1105).
pub(crate) const SSE_KEEPALIVE_INTERVAL_SECS: u64 = 30;

/// The SSE keepalive interval must stay under typical reverse-proxy idle
/// timeouts (nginx 60 s, HAProxy 60 s, AWS ALB 60 s) or a long prefill will
/// have its connection dropped before the first token arrives.
///
/// This lives next to the definition rather than in the test module so it
/// covers every consumer by construction (#1105). It was previously in
/// `streaming_tests.rs` and guarded only this constant, while
/// `streaming_responses.rs` and `streaming_anthropic.rs` carried their own
/// file-private copies that nothing checked.
///
/// Enforced as a `const` assertion so a regression fails the build rather than
/// a test run. (Using `assert!` on a constant expression in a runtime test
/// triggers `clippy::assertions_on_constants`.)
const _: () = assert!(
    SSE_KEEPALIVE_INTERVAL_SECS < 60,
    "SSE keepalive interval must be less than the 60s default used by most reverse proxies"
);

/// Keepalive configuration attached to an `Sse` response.
///
/// Usually constructed by `sse_channel` and handed, with the stream, to
/// [`sse_response`], which is what attaches it (#1107). Using a newtype keeps
/// the keepalive wired to the same channel creation point. The inner
/// `KeepAlive` is private to prevent callers from constructing a mismatched
/// keepalive independently.
///
/// `router_front.rs` builds its own with `from_interval` because its streams
/// do not come from `sse_channel`, but it goes through this type rather than
/// `KeepAlive::default()` so it tracks the configured interval (#1105, #1432).
///
/// `None` means the pings are switched off (`--sse-ping-interval -1`), which
/// is why the inner value is an `Option`: axum's `Sse` has no "no keepalive"
/// `KeepAlive`, so disabling has to be expressed by not attaching one.
///
/// Used by: chat.rs, completions.rs, native_completion.rs, router_front.rs
pub(crate) struct SseKeepAlive(Option<KeepAlive>);

impl SseKeepAlive {
    /// Build a keepalive that sends an empty comment every `interval`, or no
    /// keepalive at all when `interval` is `None`.
    ///
    /// `KeepAlive::new()` already emits an empty SSE comment by default, so
    /// only the interval needs to be customised.
    pub(crate) fn from_interval(interval: Option<Duration>) -> Self {
        Self(interval.map(|d| KeepAlive::new().interval(d)))
    }

    /// The compiled-in default interval, for streams with no configured
    /// server (tests and the disaggregated router front-end).
    pub(crate) fn default_for_long_prefill() -> Self {
        Self::from_interval(Some(Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)))
    }
}

impl IntoKeepAlive for SseKeepAlive {
    fn into_keep_alive(self) -> Option<KeepAlive> {
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
/// Hand the returned `stream` and `keepalive` to [`sse_response`], which builds
/// the HTTP response and attaches the keepalive. Do not assemble
/// `Sse::new(..).keep_alive(..)` in the route handler: that duplication is what
/// #1107 removed, and `sse_response` is the only path that cannot forget the
/// keepalive. Attaching it is what stops proxy idle timeouts from closing the
/// connection during long prefill phases, before the first generated token
/// arrives.
///
/// Used by: chat.rs, completions.rs, native_completion.rs
pub(crate) fn sse_channel(
    buffer: usize,
    ping_interval: Option<Duration>,
) -> (
    BlockingSseSender,
    impl Stream<Item = Result<Event, Infallible>>,
    CancellationToken,
    SseKeepAlive,
) {
    let cancelled: CancellationToken = Arc::new(AtomicBool::new(false));
    let (sender, rx) = payload_channel(buffer, Some(cancelled.clone()));
    let stream = ReceiverStream::new(rx).map(|payload| payload.map(sse_event));
    let keepalive = SseKeepAlive::from_interval(ping_interval);
    (sender, stream, cancelled, keepalive)
}

/// Unwrap a per-surface keepalive newtype into the axum [`KeepAlive`] that
/// [`sse_response`] attaches.
///
/// The three newtypes stay distinct so a route cannot attach another surface's
/// keepalive (#1105). This trait is the only thing they share, and it exists so
/// all three can reach one response constructor instead of each route
/// assembling `Sse::new(..).keep_alive(..)` by hand (#1107).
///
/// Consuming `self` ensures each keepalive is applied exactly once. This is the
/// only way to get at the inner `KeepAlive`, so there is no route from one of
/// these newtypes to a hand-assembled `Sse`.
pub(crate) trait IntoKeepAlive {
    fn into_keep_alive(self) -> Option<KeepAlive>;
}

/// Build the streaming HTTP response, attaching the keepalive that the matching
/// `*_sse_channel` produced.
///
/// Route handlers call this instead of assembling
/// `Sse::new(stream).keep_alive(..).into_response()` by
/// hand. Five routes carried byte-identical copies of that tail, and a sixth
/// that forgot the `.keep_alive(..)` would have compiled and passed the whole
/// suite (#1107). Routing every surface through one constructor removes the
/// hazard structurally rather than testing for it per route: this is the only
/// way to turn one of these streams into a `Response`, and it takes the
/// keepalive by value, so forgetting to attach it stops being expressible.
///
/// Deleting the `.keep_alive(..)` below is the one remaining way to break the
/// invariant, and it breaks it for every route at once, which is exactly the
/// property a per-route test could not give.
pub(crate) fn sse_response<S>(stream: S, keepalive: impl IntoKeepAlive) -> Response
where
    S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
{
    let sse = Sse::new(stream);
    match keepalive.into_keep_alive() {
        Some(keep_alive) => sse.keep_alive(keep_alive).into_response(),
        // `--sse-ping-interval -1`: the operator asked for no comment pings.
        None => sse.into_response(),
    }
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
