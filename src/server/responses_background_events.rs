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

//! Replayable event log of one streamed background response (issue #2900).
//!
//! A `background: true` response created with `stream: true` owns one
//! [`BackgroundEventLog`]. The job appends every event to it; each reader (the
//! create call, and any `GET /v1/responses/{id}?stream=true`) is an
//! independent cursor over it built by [`tail_events`].
//!
//! ## Sequence numbers
//!
//! The log numbers the events itself, from 0 and without gaps, whoever
//! appends them: the admission path (`response.created`, `response.queued`),
//! the generation driver, and the response store, which appends the terminal
//! event when it records the job's terminal object. An event that arrives
//! with a different number is restamped.
//!
//! ## Byte budget
//!
//! The log keeps at most `max_bytes` of events and drops the oldest when a
//! new one does not fit. The newest event always stays, so the terminal
//! event is never dropped; the log can therefore exceed its budget by at most
//! one event. A reader whose cursor fell out of the retained window gets an
//! `response.error` event and its stream ends.
//!
//! A reader holds the log only weakly between reads, so the store's
//! eviction of the entry frees the log even while a client keeps a stalled
//! connection open; that reader then gets the `response.error` event. What a
//! reader buffers itself is capped by `READ_BATCH` events and
//! `READ_BATCH_BYTES`.
//!
//! ## Concurrency
//!
//! The pattern of [`crate::server::stream_session`] (#1444): a
//! `std::sync::Mutex` that is never held across an await, and readers parked
//! on a `tokio::sync::Notify` that re-check the log after every wake. An
//! append takes the lock only for the push, so no reader can stall the
//! generation, and a reader that goes away is just a dropped cursor: nothing
//! here can reach the job's cancellation token.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use axum::response::sse::Event;
use futures::{Stream, StreamExt};
use tokio::sync::Notify;

use crate::server::types::responses_response::{ResponseObject, ResponseStatus};
use crate::server::types::responses_stream::ResponseStreamEvent;

/// One logged event: sequence number, SSE `event:` name, serialized JSON.
pub(crate) type LoggedEvent = (u64, &'static str, Arc<str>);

/// Bytes accounted for one event beyond its name and payload: the tuple and
/// the `Arc<str>` header.
const EVENT_OVERHEAD_BYTES: usize =
    std::mem::size_of::<LoggedEvent>() + 2 * std::mem::size_of::<usize>();

/// Most events one reader copies out of the log per lock acquisition.
const READ_BATCH: usize = 256;

/// Most payload bytes one reader copies out of the log per lock acquisition
/// (at least one event is always copied), so a stalled reader pins little of
/// a log the store already evicted.
const READ_BATCH_BYTES: usize = 64 * 1024;

/// Periodic re-check while a reader waits, so a missed wake only delays it.
const READER_WAKE: Duration = Duration::from_secs(1);

/// `code` of the `response.error` event a reader gets when its cursor fell
/// out of the retained window.
pub(crate) const CODE_EVENTS_EVICTED: &str = "stream_events_evicted";

#[derive(Debug)]
struct LogState {
    events: VecDeque<LoggedEvent>,
    /// Accounted bytes of `events`.
    bytes: usize,
    /// Number the next appended event gets; also the count ever appended.
    next_sequence: u64,
    /// The terminal event was appended; later appends are ignored.
    terminal: bool,
}

impl LogState {
    fn push(&mut self, name: &'static str, json: String, max_bytes: usize) -> u64 {
        let sequence = self.next_sequence;
        self.bytes = self.bytes.saturating_add(event_bytes(name, json.len()));
        self.events.push_back((sequence, name, Arc::from(json)));
        self.next_sequence = sequence.saturating_add(1);
        // The newest event always stays: it may be the terminal one.
        while self.bytes > max_bytes && self.events.len() > 1 {
            if let Some((_, name, json)) = self.events.pop_front() {
                self.bytes = self.bytes.saturating_sub(event_bytes(name, json.len()));
            }
        }
        sequence
    }
}

fn event_bytes(name: &str, json_len: usize) -> usize {
    name.len()
        .saturating_add(json_len)
        .saturating_add(EVENT_OVERHEAD_BYTES)
}

/// What one read attempt at a cursor found.
#[derive(Debug)]
pub(crate) enum LogRead {
    /// Events at and after the cursor, oldest first.
    Events(Vec<LoggedEvent>),
    /// Nothing new yet; the job is still running.
    Pending,
    /// The terminal event is behind the cursor: the stream is over.
    Closed,
    /// The cursor is older than the oldest retained event.
    Evicted { oldest: u64 },
}

/// Why a resume cursor cannot be served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CursorError {
    /// An event the reader asked for was already dropped; `oldest` is the
    /// oldest retained sequence number.
    Evicted { oldest: u64 },
    /// The cursor is past the newest sequence number assigned so far
    /// (`None`: no event was appended yet).
    Ahead { newest: Option<u64> },
}

/// A terminal event serialized before the store takes its locks.
///
/// Used by: `ResponsesStore::finish_background`.
#[derive(Debug)]
pub(crate) struct PreparedTerminal {
    sequence_number: u64,
    name: &'static str,
    json: String,
}

/// The terminal event for `response`: `response.failed` for a failed
/// response, `response.incomplete` for an incomplete one, and
/// `response.completed` otherwise, which includes a cancelled response
/// (`status: "cancelled"`).
fn terminal_event(sequence_number: u64, response: ResponseObject) -> ResponseStreamEvent {
    match response.status {
        ResponseStatus::Failed => ResponseStreamEvent::Failed {
            sequence_number,
            response,
        },
        ResponseStatus::Incomplete => ResponseStreamEvent::Incomplete {
            sequence_number,
            response,
        },
        _ => ResponseStreamEvent::Completed {
            sequence_number,
            response,
        },
    }
}

/// The bounded, replayable event log of one streamed background response.
#[derive(Debug)]
pub struct BackgroundEventLog {
    state: Mutex<LogState>,
    notify: Notify,
    max_bytes: usize,
}

impl BackgroundEventLog {
    /// An empty log that retains about `max_bytes` of events
    /// (`--responses-background-event-log-bytes`).
    pub fn new(max_bytes: usize) -> Self {
        Self {
            state: Mutex::new(LogState {
                events: VecDeque::new(),
                bytes: 0,
                next_sequence: 0,
                terminal: false,
            }),
            notify: Notify::new(),
            max_bytes,
        }
    }

    fn lock(&self) -> MutexGuard<'_, LogState> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Append one non-terminal event under the next sequence number, which
    /// replaces the number the event came with when the two differ. Returns
    /// the number used, or `None` once the terminal event was appended.
    ///
    /// Never waits for a reader: the lock is held for the push only.
    ///
    /// Used by: the background job's admission path and its event sink.
    pub(crate) fn append(&self, event: &ResponseStreamEvent) -> Option<u64> {
        let sequence = {
            let mut state = self.lock();
            if state.terminal {
                return None;
            }
            let json = serialize_numbered(event, state.next_sequence)?;
            state.push(event.event_name(), json, self.max_bytes)
        };
        self.notify.notify_waiters();
        Some(sequence)
    }

    /// Clone and serialize the terminal event for `response` under the
    /// number the next event would get, holding the log lock only to read
    /// that number. [`Self::append_terminal_prepared`] uses the result when
    /// no other event was appended in between.
    pub(crate) fn prepare_terminal(&self, response: &ResponseObject) -> Option<PreparedTerminal> {
        let sequence_number = self.next_sequence_number();
        let event = terminal_event(sequence_number, response.clone());
        let json = serialize_numbered(&event, sequence_number)?;
        Some(PreparedTerminal {
            sequence_number,
            name: event.event_name(),
            json,
        })
    }

    /// Append the terminal event for `response` and close the log (see
    /// [`terminal_event`] for which event). Returns `false` when the log was
    /// already closed.
    #[cfg(test)]
    pub(crate) fn append_terminal(&self, response: &ResponseObject) -> bool {
        self.append_terminal_prepared(response, None)
    }

    /// Append the terminal event for `response` and close the log (see
    /// [`terminal_event`] for which event), pushing `prepared` as is when it
    /// was built under the current next sequence number and otherwise
    /// serializing the event under the log lock. Returns `false` when the log
    /// was already closed.
    ///
    /// Used by: `ResponsesStore::finish_background`, so the terminal event
    /// always carries the object the store recorded.
    pub(crate) fn append_terminal_prepared(
        &self,
        response: &ResponseObject,
        prepared: Option<PreparedTerminal>,
    ) -> bool {
        {
            let mut state = self.lock();
            if state.terminal {
                return false;
            }
            let sequence_number = state.next_sequence;
            let frame = match prepared {
                Some(prepared) if prepared.sequence_number == sequence_number => {
                    Some((prepared.name, prepared.json))
                }
                _ => {
                    let event = terminal_event(sequence_number, response.clone());
                    serialize_numbered(&event, sequence_number)
                        .map(|json| (event.event_name(), json))
                }
            };
            // Closed even if the event could not be serialized, so readers end.
            state.terminal = true;
            if let Some((name, json)) = frame {
                state.push(name, json, self.max_bytes);
            }
        }
        self.notify.notify_waiters();
        true
    }

    /// The sequence number the next appended event gets.
    pub(crate) fn next_sequence_number(&self) -> u64 {
        self.lock().next_sequence
    }

    /// Whether the event numbered `sequence` is the terminal one.
    fn is_terminal_event(&self, sequence: u64) -> bool {
        let state = self.lock();
        state.terminal && sequence.saturating_add(1) == state.next_sequence
    }

    /// Whether the terminal event was appended.
    #[cfg(test)]
    pub(crate) fn is_terminal(&self) -> bool {
        self.lock().terminal
    }

    /// Accounted bytes of the retained events.
    pub fn retained_bytes(&self) -> usize {
        self.lock().bytes
    }

    /// The first sequence number to deliver for a resume at `starting_after`
    /// (`None`: from the first event). Fails when an event the reader asked
    /// for was already dropped, or when `starting_after` is past the newest
    /// sequence number assigned so far.
    pub(crate) fn resolve_cursor(&self, starting_after: Option<u64>) -> Result<u64, CursorError> {
        let state = self.lock();
        if let Some(after) = starting_after
            && after >= state.next_sequence
        {
            return Err(CursorError::Ahead {
                newest: state.next_sequence.checked_sub(1),
            });
        }
        let next = starting_after.map_or(0, |after| after.saturating_add(1));
        match state.events.front() {
            Some(&(oldest, ..)) if next < oldest => Err(CursorError::Evicted { oldest }),
            _ => Ok(next),
        }
    }

    /// One non-blocking read of up to `limit` events starting at sequence
    /// number `next`.
    #[cfg(test)]
    pub(crate) fn read_from(&self, next: u64, limit: usize) -> LogRead {
        self.read_bounded(next, limit, usize::MAX)
    }

    /// One non-blocking read of up to `limit` events starting at sequence
    /// number `next`, stopping once `max_bytes` of payload were copied; the
    /// first event is copied whatever its size.
    fn read_bounded(&self, next: u64, limit: usize, max_bytes: usize) -> LogRead {
        let state = self.lock();
        let Some(&(oldest, ..)) = state.events.front() else {
            return if state.terminal {
                LogRead::Closed
            } else {
                LogRead::Pending
            };
        };
        if next < oldest {
            return LogRead::Evicted { oldest };
        }
        // Retained sequence numbers are contiguous.
        let skip = usize::try_from(next - oldest).unwrap_or(usize::MAX);
        if skip < state.events.len() {
            let mut copied = 0usize;
            let events = state
                .events
                .range(skip..)
                .take(limit)
                .take_while(|(_, _, json)| {
                    let first = copied == 0;
                    copied = copied.saturating_add(json.len().max(1));
                    first || copied <= max_bytes
                })
                .cloned()
                .collect();
            LogRead::Events(events)
        } else if state.terminal {
            LogRead::Closed
        } else {
            LogRead::Pending
        }
    }
}

/// Serialize `event` with `sequence_number` as its number.
fn serialize_numbered(event: &ResponseStreamEvent, sequence_number: u64) -> Option<String> {
    let serialized = if event.sequence_number() == sequence_number {
        serde_json::to_string(event)
    } else {
        let mut event = event.clone();
        *event.sequence_number_mut() = sequence_number;
        serde_json::to_string(&event)
    };
    match serialized {
        Ok(json) => Some(json),
        Err(err) => {
            tracing::error!(
                event = event.event_name(),
                "background response event could not be serialized: {err}"
            );
            None
        }
    }
}

/// The message of the evicted-cursor error, as the 400 body of a resume and
/// as the `response.error` event of a reader that fell behind.
pub(crate) fn evicted_cursor_message(response_id: &str, oldest: u64) -> String {
    format!(
        "the requested events of response '{response_id}' are no longer retained: the oldest \
         retained event has sequence_number {oldest} (--responses-background-event-log-bytes). \
         Retrieve the response with GET /v1/responses/{response_id} instead"
    )
}

/// The message of the `response.error` event a reader gets when the store
/// dropped the log (its entry was evicted) before the reader reached the
/// terminal event, and of the resume 400 for a log that was not retained.
pub(crate) fn log_not_retained_message(response_id: &str) -> String {
    format!(
        "the event stream of response '{response_id}' is no longer retained by the response \
         store (--responses-store-max-bytes). Retrieve the response with \
         GET /v1/responses/{response_id} instead"
    )
}

/// The frames a reader of `log` gets from sequence number `next` on, as
/// `(SSE event name, JSON payload)`: the retained events from there, then
/// the live ones, ending after the terminal event.
///
/// A reader that falls behind the retained window gets one `response.error`
/// frame carrying [`evicted_cursor_message`] and the stream ends; one whose
/// log the store dropped gets [`log_not_retained_message`] instead.
///
/// The reader keeps only a [`Weak`] handle while it is not reading, so a
/// client that stops reading cannot keep an evicted log alive. While it
/// waits for the next event it holds the log, which the job then holds too.
fn tail_frames(
    log: Arc<BackgroundEventLog>,
    response_id: String,
    next: u64,
) -> impl Stream<Item = (&'static str, Arc<str>)> + Send + 'static {
    struct Reader {
        log: Weak<BackgroundEventLog>,
        response_id: String,
        next: u64,
        pending: VecDeque<LoggedEvent>,
        finished: bool,
    }
    let reader = Reader {
        log: Arc::downgrade(&log),
        response_id,
        next,
        pending: VecDeque::new(),
        finished: false,
    };
    drop(log);
    futures::stream::unfold(reader, |mut reader| async move {
        if let Some((sequence, name, data)) = reader.pending.pop_front() {
            reader.next = sequence.saturating_add(1);
            return Some(((name, data), reader));
        }
        if reader.finished {
            return None;
        }
        let Some(log) = reader.log.upgrade() else {
            reader.finished = true;
            let message = log_not_retained_message(&reader.response_id);
            return Some((error_frame(reader.next, message), reader));
        };
        loop {
            // Created before the read: a `Notified` observes every
            // `notify_waiters` call made after its creation, so an append
            // between the read and the await is not missed.
            let notified = log.notify.notified();
            match log.read_bounded(reader.next, READ_BATCH, READ_BATCH_BYTES) {
                LogRead::Events(events) => {
                    // The reader must not need the log again after the
                    // terminal event, which it may no longer reach.
                    if let Some(&(last, ..)) = events.last() {
                        reader.finished = log.is_terminal_event(last);
                    }
                    reader.pending.extend(events);
                    break;
                }
                LogRead::Closed => return None,
                LogRead::Evicted { oldest } => {
                    reader.finished = true;
                    let message = evicted_cursor_message(&reader.response_id, oldest);
                    return Some((error_frame(reader.next, message), reader));
                }
                LogRead::Pending => {
                    let _ = tokio::time::timeout(READER_WAKE, notified).await;
                }
            }
        }
        drop(log);
        let (sequence, name, data) = reader.pending.pop_front()?;
        reader.next = sequence.saturating_add(1);
        Some(((name, data), reader))
    })
}

/// One `response.error` frame with [`CODE_EVENTS_EVICTED`].
fn error_frame(sequence_number: u64, message: String) -> (&'static str, Arc<str>) {
    let error = ResponseStreamEvent::Error {
        sequence_number,
        code: CODE_EVENTS_EVICTED.to_string(),
        message,
    };
    let data = serde_json::to_string(&error).unwrap_or_default();
    (error.event_name(), Arc::from(data))
}

/// [`tail_frames`] as an SSE stream. Dropping it (a client disconnect)
/// affects nothing but this reader.
///
/// Used by: the streamed background create and
/// `GET /v1/responses/{id}?stream=true`.
pub(crate) fn tail_events(
    log: Arc<BackgroundEventLog>,
    response_id: String,
    next: u64,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    tail_frames(log, response_id, next)
        .map(|(name, data)| Ok(Event::default().event(name).data(&*data)))
}

#[cfg(test)]
#[path = "responses_background_events_tests.rs"]
mod responses_background_events_tests;
