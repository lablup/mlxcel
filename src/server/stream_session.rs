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

//! Resumable stream sessions aligned with `llama-server` b10621 (#1444).
//!
//! b10621 keeps a per-conversation ring buffer of the exact SSE bytes a
//! streaming completion produced, surviving the HTTP connection that started
//! it (upstream
//! <https://github.com/ggml-org/llama.cpp/blob/main/tools/server/server-stream.cpp>).
//! A client that sends `X-Conversation-Id` on a streaming request can later
//! replay the stream from any byte offset with `GET /v1/stream`, discover
//! live sessions it already knows the ids of with `POST /v1/streams/lookup`,
//! and stop one with `DELETE /v1/stream`.
//!
//! This module owns the session store; the route handlers live in
//! [`crate::server::routes::stream`], and the producer side is attached by
//! the streaming senders in [`crate::server::streaming`],
//! [`crate::server::streaming_responses`] and
//! [`crate::server::streaming_anthropic`] through [`ResumeTee`].
//!
//! ## Semantics carried over from b10621
//!
//! - One conversation id maps to at most one live session; a new streaming
//!   request with the same id cancels and replaces the previous session.
//! - The buffer keeps at most [`STREAM_SESSION_MAX_BYTES`]; older bytes are
//!   dropped from the front and a replay from an offset that fell below the
//!   dropped prefix is refused ("Stream offset lost").
//! - A finished session is retained for [`STREAM_SESSION_TTL_SECONDS`] after
//!   completion, then evicted by the GC.
//! - `DELETE` cancels the producer (mlxcel: the generation itself, through
//!   the shared [`CancellationToken`]) and evicts the buffer, idempotently.
//!
//! ## mlxcel-side hardening (documented in the manifest entry notes)
//!
//! - Sessions are keyed by `(owner, conversation_id)`, where the owner is the
//!   API key that created the stream. With `--api-key` configured, one key
//!   can neither read, discover, replace nor delete another key's session;
//!   the responses are indistinguishable from "no such stream" so the
//!   endpoints cannot be used as an existence oracle. With authentication
//!   disabled every request shares one owner, which is b10621's behavior.
//! - The number of retained completed sessions is capped at
//!   [`STREAM_SESSION_MAX_RETAINED`] (oldest-completed evicted first), so
//!   retained replay buffers stay bounded even under a churn of distinct
//!   conversation ids faster than the TTL expires them.
//!
//! ## Concurrency
//!
//! The producer appends from the blocking generation task; readers drain from
//! async route handlers. State lives behind a `std::sync::Mutex` (never held
//! across an await) and readers park on a `tokio::sync::Notify`, re-checking
//! state after every wakeup, so a lost wakeup only costs one poll interval.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::Notify;

use super::streaming::CancellationToken;

/// Seconds a completed session stays discoverable and replayable before the
/// GC evicts it. b10621 `STREAM_SESSION_TTL_SECONDS`.
pub(crate) const STREAM_SESSION_TTL_SECONDS: i64 = 300;

/// Per-session buffer cap in bytes. b10621 `STREAM_SESSION_MAX_BYTES`.
/// Older bytes are dropped from the front when the cap is exceeded.
pub(crate) const STREAM_SESSION_MAX_BYTES: usize = 4 * 1024 * 1024;

/// GC sweep interval. b10621 `STREAM_SESSION_GC_INTERVAL_SECONDS`.
pub(crate) const STREAM_SESSION_GC_INTERVAL_SECONDS: u64 = 60;

/// mlxcel-side cap on retained *completed* sessions (b10621 has none).
/// Live sessions are bounded by queue admission; completed ones are bounded
/// here so a burst of distinct conversation ids cannot hold an unbounded
/// number of replay buffers for the TTL window.
pub(crate) const STREAM_SESSION_MAX_RETAINED: usize = 256;

/// Interval at which a blocked replay reader re-checks for client disconnect
/// even when no new bytes arrive. Mirrors b10621's 200 ms wake, relaxed to
/// one second because disconnect is also detected on the next send.
const READER_WAKE_MILLIS: u64 = 1000;

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The identity that owns a session: the presented API key when
/// authentication is configured, `None` when it is disabled. Compared for
/// equality on every lookup, replay, replace and delete.
pub(crate) type StreamOwner = Option<String>;

#[derive(Debug)]
struct SessionBuf {
    buffer: Vec<u8>,
    /// Bytes evicted from the front of `buffer` due to the cap. The logical
    /// stream offset of `buffer[0]` is exactly this value.
    prefix_dropped: usize,
    done: bool,
    /// Unix seconds at finalize; `0` while the producer is live.
    completed_ts: i64,
}

/// Outcome of one non-blocking read attempt at a byte offset.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ReadChunk {
    /// Bytes available at the offset (a copy; the lock is not held after).
    Data(Vec<u8>),
    /// No bytes at the offset yet and the producer is still live.
    Pending,
    /// The session is finalized and the offset is at (or past) the end.
    Eof,
    /// The offset fell below the dropped prefix; replay cannot resume there.
    OffsetLost,
}

/// One resumable stream: the SSE bytes of a single generation, surviving the
/// HTTP connection that started it.
#[derive(Debug)]
pub(crate) struct StreamSession {
    conversation_id: String,
    started_ts: i64,
    cap_bytes: usize,
    state: Mutex<SessionBuf>,
    notify: Notify,
    /// Shared with the generation request as its scheduler cancellation
    /// token: `DELETE /v1/stream` and session replacement flip it, which
    /// aborts the underlying sequence exactly as a client disconnect does on
    /// a non-resumable stream.
    cancelled: CancellationToken,
}

impl StreamSession {
    fn new(conversation_id: String, cap_bytes: usize) -> Self {
        Self {
            conversation_id,
            started_ts: now_seconds(),
            cap_bytes,
            state: Mutex::new(SessionBuf {
                buffer: Vec::with_capacity(64 * 1024),
                prefix_dropped: 0,
                done: false,
                completed_ts: 0,
            }),
            notify: Notify::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    pub(crate) fn started_at(&self) -> i64 {
        self.started_ts
    }

    /// The cancellation token generation shares with this session. Handed to
    /// the scheduler in place of the per-connection token, so client
    /// disconnect no longer aborts the sequence but `DELETE /v1/stream` does.
    pub(crate) fn cancellation_token(&self) -> CancellationToken {
        self.cancelled.clone()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    /// Append produced SSE bytes. Returns `false` once the session is
    /// finalized (late writes after a replace/delete are dropped).
    pub(crate) fn append(&self, data: &[u8]) -> bool {
        if data.is_empty() {
            return true;
        }
        {
            let Ok(mut st) = self.state.lock() else {
                return false;
            };
            if st.done {
                return false;
            }
            if data.len() >= self.cap_bytes {
                // A single chunk larger than the cap: keep only the tail.
                let skip = data.len() - self.cap_bytes;
                st.prefix_dropped += st.buffer.len() + skip;
                st.buffer.clear();
                st.buffer.extend_from_slice(&data[skip..]);
            } else {
                let needed = st.buffer.len() + data.len();
                if needed > self.cap_bytes {
                    let to_drop = needed - self.cap_bytes;
                    st.buffer.drain(..to_drop);
                    st.prefix_dropped += to_drop;
                }
                st.buffer.extend_from_slice(data);
            }
        }
        self.notify.notify_waiters();
        true
    }

    /// Mark the stream complete. Idempotent; wakes every blocked reader.
    pub(crate) fn finalize(&self) {
        {
            let Ok(mut st) = self.state.lock() else {
                return;
            };
            if st.done {
                return;
            }
            st.done = true;
            st.completed_ts = now_seconds();
        }
        self.notify.notify_waiters();
    }

    pub(crate) fn is_done(&self) -> bool {
        self.state.lock().map(|st| st.done).unwrap_or(true)
    }

    /// Total bytes that ever entered the session (dropped prefix included).
    pub(crate) fn total_size(&self) -> usize {
        self.state
            .lock()
            .map(|st| st.prefix_dropped + st.buffer.len())
            .unwrap_or(0)
    }

    pub(crate) fn dropped_prefix(&self) -> usize {
        self.state.lock().map(|st| st.prefix_dropped).unwrap_or(0)
    }

    /// `0` while live, unix seconds after finalize.
    pub(crate) fn completed_at(&self) -> i64 {
        self.state.lock().map(|st| st.completed_ts).unwrap_or(0)
    }

    /// One non-blocking read attempt at `offset`.
    pub(crate) fn read_chunk(&self, offset: usize) -> ReadChunk {
        let Ok(st) = self.state.lock() else {
            return ReadChunk::Eof;
        };
        if offset < st.prefix_dropped {
            return ReadChunk::OffsetLost;
        }
        let logical_end = st.prefix_dropped + st.buffer.len();
        if offset < logical_end {
            let local = offset - st.prefix_dropped;
            return ReadChunk::Data(st.buffer[local..].to_vec());
        }
        if st.done {
            return ReadChunk::Eof;
        }
        ReadChunk::Pending
    }

    /// Await new bytes, finalize, or the periodic re-check wake. The caller
    /// re-runs [`Self::read_chunk`] after this returns; a spurious wake only
    /// costs another check.
    pub(crate) async fn wait_for_change(&self) {
        let notified = self.notify.notified();
        let _ = tokio::time::timeout(
            std::time::Duration::from_millis(READER_WAKE_MILLIS),
            notified,
        )
        .await;
    }
}

/// Producer end of a session: appends framed SSE bytes and finalizes the
/// session when dropped, whatever path the generation task exited through.
///
/// Held as `Option<Arc<ResumeTee>>` on the streaming senders; clones share
/// one tee, so the finalize runs when the last sender clone drops (the
/// blocking generation task's, in every route).
#[derive(Debug)]
pub(crate) struct ResumeTee {
    session: Arc<StreamSession>,
}

impl ResumeTee {
    pub(crate) fn new(session: Arc<StreamSession>) -> Self {
        Self { session }
    }

    /// Tee one data-only SSE payload, framed exactly as axum's
    /// `Event::default().data(payload)` serializes on the wire.
    pub(crate) fn write_data(&self, payload: &str) {
        self.session.append(&frame_sse(None, payload));
    }

    /// Tee one named-event SSE payload, framed exactly as axum's
    /// `Event::default().event(name).data(payload)` serializes on the wire.
    pub(crate) fn write_event(&self, name: &str, payload: &str) {
        self.session.append(&frame_sse(Some(name), payload));
    }
}

impl Drop for ResumeTee {
    fn drop(&mut self) {
        self.session.finalize();
    }
}

/// Serialize one SSE frame the way axum's `Event` does: an optional
/// `event:` line, one `data:` line per newline-separated payload line, and a
/// terminating blank line. Replayed bytes must parse identically to the live
/// stream for an `EventSource` client.
fn frame_sse(event: Option<&str>, payload: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 16);
    if let Some(name) = event {
        out.extend_from_slice(b"event: ");
        out.extend_from_slice(name.as_bytes());
        out.push(b'\n');
    }
    for line in payload.split('\n') {
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(line.as_bytes());
        out.push(b'\n');
    }
    out.push(b'\n');
    out
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct SessionKey {
    owner: StreamOwner,
    conversation_id: String,
}

/// Owns every live session, keyed by `(owner, conversation_id)`.
///
/// One instance per [`crate::server::AppState`]. A periodic GC task (spawned
/// once, lazily, from the first router construction inside a tokio runtime)
/// evicts sessions whose completion is older than the TTL; every mutating
/// operation also sweeps opportunistically so tests and runtimes without the
/// task stay bounded.
#[derive(Debug, Default)]
pub(crate) struct StreamSessionManager {
    sessions: Mutex<HashMap<SessionKey, Arc<StreamSession>>>,
    gc_spawned: AtomicBool,
    /// Per-session buffer cap; [`STREAM_SESSION_MAX_BYTES`] outside tests.
    cap_bytes: Option<usize>,
}

impl StreamSessionManager {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn cap(&self) -> usize {
        self.cap_bytes.unwrap_or(STREAM_SESSION_MAX_BYTES)
    }

    /// Install a new session for `(owner, conversation_id)`, cancelling and
    /// finalizing any previous one under the same key ("one conv = at most
    /// one live session"). Another owner's session under the same
    /// conversation id is a different key and is untouched: replacement is
    /// not a cross-key control channel.
    pub(crate) fn create_or_replace(
        &self,
        conversation_id: &str,
        owner: StreamOwner,
    ) -> Arc<StreamSession> {
        self.sweep(now_seconds());
        let fresh = Arc::new(StreamSession::new(conversation_id.to_string(), self.cap()));
        let previous = {
            let Ok(mut map) = self.sessions.lock() else {
                return fresh;
            };
            map.insert(
                SessionKey {
                    owner,
                    conversation_id: conversation_id.to_string(),
                },
                fresh.clone(),
            )
        };
        if let Some(previous) = previous {
            // Cancel first so the producer stops, then finalize to wake any
            // reader still replaying the replaced stream.
            previous.cancel();
            previous.finalize();
        }
        fresh
    }

    /// The session for `(owner, conversation_id)`, if the caller owns one.
    pub(crate) fn get(
        &self,
        conversation_id: &str,
        owner: &StreamOwner,
    ) -> Option<Arc<StreamSession>> {
        let map = self.sessions.lock().ok()?;
        map.get(&SessionKey {
            owner: owner.clone(),
            conversation_id: conversation_id.to_string(),
        })
        .cloned()
    }

    /// Sessions matching the requested ids for this owner. Each id matches
    /// its exact conversation id and any `"<id>::<suffix>"` per-model
    /// variant, as b10621's lookup does. Ids the owner does not hold are
    /// silently absent; the server never lists sessions it was not asked
    /// about, and never another owner's.
    pub(crate) fn lookup(
        &self,
        requested: &[String],
        owner: &StreamOwner,
    ) -> Vec<Arc<StreamSession>> {
        self.sweep(now_seconds());
        let Ok(map) = self.sessions.lock() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for rid in requested {
            if rid.is_empty() {
                continue;
            }
            let with_sep = format!("{rid}::");
            for (key, session) in map.iter() {
                if key.owner != *owner {
                    continue;
                }
                if key.conversation_id == *rid || key.conversation_id.starts_with(&with_sep) {
                    out.push(session.clone());
                }
            }
        }
        out
    }

    /// Cancel and evict the owner's session for `conversation_id`.
    /// Idempotent: unknown ids, and ids owned by someone else, are no-ops.
    pub(crate) fn evict_and_cancel(&self, conversation_id: &str, owner: &StreamOwner) {
        let removed = {
            let Ok(mut map) = self.sessions.lock() else {
                return;
            };
            map.remove(&SessionKey {
                owner: owner.clone(),
                conversation_id: conversation_id.to_string(),
            })
        };
        if let Some(session) = removed {
            session.cancel();
            session.finalize();
        }
    }

    /// Evict completed sessions past the TTL, and enforce the retained-
    /// completed cap (oldest completed evicted first). Live sessions are
    /// never evicted here.
    pub(crate) fn sweep(&self, now: i64) {
        let mut to_finalize: Vec<Arc<StreamSession>> = Vec::new();
        {
            let Ok(mut map) = self.sessions.lock() else {
                return;
            };
            let cutoff = now - STREAM_SESSION_TTL_SECONDS;
            map.retain(|_, s| {
                let completed = s.completed_at();
                let expired = completed != 0 && completed <= cutoff;
                if expired {
                    to_finalize.push(s.clone());
                }
                !expired
            });

            // Retained-completed cap: evict oldest completed beyond the cap.
            let mut completed: Vec<(i64, SessionKey)> = map
                .iter()
                .filter(|(_, s)| s.completed_at() != 0)
                .map(|(k, s)| (s.completed_at(), k.clone()))
                .collect();
            if completed.len() > STREAM_SESSION_MAX_RETAINED {
                completed.sort_by_key(|(ts, _)| *ts);
                let excess = completed.len() - STREAM_SESSION_MAX_RETAINED;
                for (_, key) in completed.into_iter().take(excess) {
                    if let Some(s) = map.remove(&key) {
                        to_finalize.push(s);
                    }
                }
            }
        }
        for s in to_finalize {
            s.finalize();
        }
    }

    /// Spawn the periodic GC task once, if a tokio runtime is available.
    /// Called from router construction; a runtime is always present there in
    /// the real server, and tests without one fall back to the opportunistic
    /// sweeps in the mutating operations.
    pub(crate) fn ensure_gc_spawned(self: &Arc<Self>) {
        if self.gc_spawned.swap(true, Ordering::AcqRel) {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            self.gc_spawned.store(false, Ordering::Release);
            return;
        };
        let manager = Arc::downgrade(self);
        handle.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                STREAM_SESSION_GC_INTERVAL_SECONDS,
            ));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let Some(manager) = manager.upgrade() else {
                    // The owning AppState is gone; stop sweeping.
                    return;
                };
                manager.sweep(now_seconds());
            }
        });
    }
}

#[cfg(test)]
#[path = "stream_session_tests.rs"]
mod tests;
