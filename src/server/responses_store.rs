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

//! In-memory store for OpenAI Responses API objects.
//!
//! Phase 1 keeps the store entirely in-process: a synchronous
//! [`std::sync::RwLock`] guards a `HashMap<id, Entry>` plus an access-ordered
//! LRU index and a TTL sweep on every insert/lookup. The store is wired into
//! [`crate::server::state::AppState`] when `store=true` requests are
//! allowed; persistence across restarts is reserved for Phase 3.
//!
//! ## Lifecycle
//!
//! - Insert on response create when `store=true`.
//! - Lookup on `GET /v1/responses/:id` and on chained creates that
//!   reference `previous_response_id`.
//! - Delete on `DELETE /v1/responses/:id` and by the LRU/TTL sweep.
//!
//! ## Background jobs
//!
//! `background: true` responses need a record from the moment they are
//! created, not only after generation ends. Those live in a separate `jobs`
//! map under its own lock, outside `entries`, so an active job is exempt from
//! TTL and LRU eviction by construction. Its approximate bytes are tracked
//! separately (`active_bytes`), counted by
//! [`ResponsesStore::approximate_total_bytes`], and never drive eviction of
//! stored entries. [`ResponsesStore::finish_background`] removes the job and
//! inserts its terminal object as a normal entry in one critical section.
//! The lock order is always `jobs`, then `inner`.
//!
//! A job created with `store: false` keeps its terminal object as an
//! *ephemeral* entry: [`ResponsesStore::background_snapshot`] can read it for
//! the ephemeral TTL, but [`ResponsesStore::get`] (and therefore
//! `previous_response_id` chaining) never returns it.
//!
//! A job created with `stream: true` also owns a
//! [`BackgroundEventLog`] (issue #2900). The job record holds it while the job
//! is active; `finish_background` appends the terminal event and moves the log
//! onto the stored entry, whose `size_bytes` then includes the log's bytes, so
//! evicting the entry frees it.
//!
//! Jobs are in memory only; a restart loses them.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::server::responses_background_events::{BackgroundEventLog, PreparedTerminal};
use crate::server::store_budget::{
    INITIAL_HASH_CAPACITY_LIMIT, LruKey, serialized_json_len_saturating,
};
use crate::server::types::responses_request::{
    ResponseInputContent, ResponseInputItem, ResponseInputPart, ResponseInputRole,
    ResponseToolOutput,
};
use crate::server::types::responses_response::{ResponseErrorBody, ResponseObject, ResponseStatus};
use tokio::sync::Notify;

/// Default approximate retained-byte budget for the in-memory response store.
///
/// This keeps the default bounded below the historical "1024 arbitrary
/// multimodal entries" footprint while leaving room for several max-size image
/// requests inside the one-hour TTL window.
pub const DEFAULT_RESPONSES_STORE_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Default cap on concurrently active `background: true` responses.
///
/// Each active job holds one blocking generation thread, so the default stays
/// well below Tokio's default of 512 blocking threads.
pub const DEFAULT_RESPONSES_BACKGROUND_MAX_JOBS: usize = 64;

/// Default retention, in seconds, of the terminal object of a `store: false`
/// background response. Only background polling can read it.
pub const DEFAULT_RESPONSES_BACKGROUND_EPHEMERAL_TTL_SECS: u64 = 600;

/// Default byte budget of the event log of one streamed background response
/// (`--responses-background-event-log-bytes`): the per-session cap of the
/// resumable streams of #1444.
pub const DEFAULT_RESPONSES_BACKGROUND_EVENT_LOG_BYTES: usize = 4 * 1024 * 1024;
const _: () = assert!(
    DEFAULT_RESPONSES_BACKGROUND_EVENT_LOG_BYTES
        == crate::server::stream_session::STREAM_SESSION_MAX_BYTES
);

/// Persisted entry. Inputs and outputs are kept separately so the
/// chain-resolution path can reconstruct the original conversation
/// without re-serialising the response.
#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub response: ResponseObject,
    pub input_items: Vec<ResponseInputItem>,
}

#[derive(Debug)]
struct Entry {
    payload: StoredResponse,
    inserted_at: Instant,
    last_accessed: Instant,
    size_bytes: usize,
    lru_key: LruKey,
    /// Set for the terminal object of a `store: false` background job. The
    /// entry expires after this long (or after the normal TTL, whichever
    /// comes first) and is hidden from [`ResponsesStore::get`].
    ephemeral_ttl: Option<Duration>,
    /// The event log of a streamed background response (issue #2900). Its
    /// bytes at the time the job finished are part of `size_bytes`.
    event_log: Option<Arc<BackgroundEventLog>>,
    /// The job had an event log that was not retained because it did not
    /// fit the store budget beside the response.
    event_log_dropped: bool,
}

/// What a background job's terminal entry carries beyond its payload.
#[derive(Debug, Default)]
struct BackgroundRetention {
    ephemeral_ttl: Option<Duration>,
    event_log: Option<Arc<BackgroundEventLog>>,
    event_log_dropped: bool,
}

impl Entry {
    fn is_expired(&self, ttl: Duration, now: Instant) -> bool {
        let age = now.saturating_duration_since(self.inserted_at);
        age >= ttl || self.ephemeral_ttl.is_some_and(|ephemeral| age >= ephemeral)
    }
}

#[derive(Debug)]
struct StoreState {
    entries: HashMap<String, Entry>,
    lru: BTreeSet<LruKey>,
    total_bytes: usize,
    next_sequence: u64,
}

impl StoreState {
    fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries.min(INITIAL_HASH_CAPACITY_LIMIT)),
            lru: BTreeSet::new(),
            total_bytes: 0,
            next_sequence: 0,
        }
    }
}

/// Configuration for [`ResponsesStore`].
#[derive(Debug, Clone)]
pub struct ResponsesStoreConfig {
    pub max_entries: usize,
    pub max_bytes: usize,
    pub ttl: Duration,
    /// Maximum number of concurrently active background jobs
    /// (`--responses-background-max-jobs`). `0` disables background mode.
    pub background_max_jobs: usize,
    /// Retention of the terminal object of a `store: false` background job
    /// (`--responses-background-ephemeral-ttl-secs`).
    pub background_ephemeral_ttl: Duration,
    /// Byte budget of the event log of one streamed background response
    /// (`--responses-background-event-log-bytes`). `0` refuses `stream: true`
    /// on a background response.
    pub background_event_log_bytes: usize,
}

impl Default for ResponsesStoreConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_bytes: DEFAULT_RESPONSES_STORE_MAX_BYTES,
            ttl: Duration::from_secs(3600),
            background_max_jobs: DEFAULT_RESPONSES_BACKGROUND_MAX_JOBS,
            background_ephemeral_ttl: Duration::from_secs(
                DEFAULT_RESPONSES_BACKGROUND_EPHEMERAL_TTL_SECS,
            ),
            background_event_log_bytes: DEFAULT_RESPONSES_BACKGROUND_EVENT_LOG_BYTES,
        }
    }
}

/// Why [`ResponsesStore::admit_background`] refused a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundAdmitError {
    /// Background mode is off: the job limit, `max_entries`, or `max_bytes`
    /// is `0`.
    Disabled,
    /// `limit` jobs are already active.
    LimitReached { limit: usize },
    /// The active jobs plus this one would exceed the store's byte budget.
    OverBudget,
}

/// Outcome of [`ResponsesStore::request_background_cancel`].
#[derive(Debug, Clone)]
pub enum BackgroundCancel {
    /// No active job and no retained terminal object has this id.
    NotFound,
    /// The job's cancellation token was flipped. The job stays active until
    /// its task calls [`ResponsesStore::finish_background`].
    Requested,
    /// The response already reached a terminal state; this is the recorded
    /// object.
    Terminal(ResponseObject),
}

/// Terminal hand-off shared between [`ResponsesStore::finish_background`]
/// and [`ResponsesStore::wait_background_terminal`]. The recorded object is
/// set before waiters are woken, so a waiter gets it even if the stored entry
/// was evicted in the meantime.
#[derive(Debug, Default)]
struct BackgroundSignal {
    notify: Notify,
    terminal: OnceLock<ResponseObject>,
}

#[derive(Debug)]
struct BackgroundJob {
    snapshot: ResponseObject,
    input_items: Vec<ResponseInputItem>,
    /// Approximate bytes of `input_items`, computed once at admission.
    input_bytes: usize,
    effective_store: bool,
    token: InFlightToken,
    /// A cancel was requested through [`ResponsesStore::request_background_cancel`]
    /// or [`ResponsesStore::cancel_background`]. Unlike `token`, which a
    /// refused admission also sets, this is only ever a client's cancel.
    cancel_requested: bool,
    size_bytes: usize,
    signal: Arc<BackgroundSignal>,
    /// Set for a job created with `stream: true` (issue #2900). Its bytes
    /// are not part of `size_bytes`: they change with every event.
    event_log: Option<Arc<BackgroundEventLog>>,
}

#[derive(Debug, Default)]
struct BackgroundState {
    jobs: HashMap<String, BackgroundJob>,
    /// Sum of `size_bytes` over `jobs`.
    active_bytes: usize,
}

/// Cancellation handle for in-flight streaming responses (review H2).
///
/// One token per active streaming response. Held by both the streaming
/// task (so it can poll for an external cancel) and by the in-flight
/// registry below so that
/// [`ResponsesStore::cancel_in_flight`] can flip the bool from a
/// different request thread.
pub type InFlightToken = Arc<AtomicBool>;

/// Thread-safe response store with TTL and LRU eviction.
pub struct ResponsesStore {
    inner: RwLock<StoreState>,
    config: ResponsesStoreConfig,
    /// Map of `response_id → cancellation token` for streaming responses
    /// that have not yet completed. The streaming route inserts on
    /// stream start and removes on stream completion; the cancel route
    /// looks up and flips the token.
    in_flight: RwLock<HashMap<String, InFlightToken>>,
    /// Ids whose cancel call was answered `cancelled`. Written under the
    /// `in_flight` read guard and drained under its write guard, so the
    /// streaming task learns exactly which cancels the client was told about.
    cancel_acknowledged: Mutex<HashSet<String>>,
    /// Active `background: true` jobs. Lock order: `jobs`, then `inner`;
    /// a cancel also takes `in_flight` (read) under `jobs`. Nothing takes
    /// `jobs` while holding `in_flight` or `inner`.
    jobs: Mutex<BackgroundState>,
}

impl ResponsesStore {
    pub fn new(config: ResponsesStoreConfig) -> Self {
        Self {
            inner: RwLock::new(StoreState::with_capacity(config.max_entries)),
            config,
            in_flight: RwLock::new(HashMap::new()),
            cancel_acknowledged: Mutex::new(HashSet::new()),
            jobs: Mutex::new(BackgroundState::default()),
        }
    }

    /// Register a streaming response so an external cancel call can
    /// abort it. The returned [`InFlightToken`] is shared with the
    /// generation task — once the task observes `true`, it stops
    /// emitting deltas and the scheduler aborts the underlying
    /// sequence.
    pub fn register_in_flight(&self, id: String) -> InFlightToken {
        let token: InFlightToken = Arc::new(AtomicBool::new(false));
        match self.in_flight.write() {
            Ok(mut g) => {
                g.insert(id, token.clone());
            }
            Err(poisoned) => {
                poisoned.into_inner().insert(id, token.clone());
            }
        }
        token
    }

    /// Remove a streaming response from the in-flight registry. Called
    /// by the streaming task after the final event has been emitted.
    pub fn unregister_in_flight(&self, id: &str) {
        self.finish_in_flight(id);
    }

    /// Retire a streaming response from the in-flight registry and report
    /// whether a cancel call was acknowledged for it. A cancel cannot be
    /// answered `cancelled` after this returns, so a stream that decides its
    /// final status from the result never contradicts the cancel reply.
    ///
    /// Used by: the streaming Responses task.
    pub fn finish_in_flight(&self, id: &str) -> bool {
        let mut g = match self.in_flight.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        g.remove(id);
        drop(g);
        match self.cancel_acknowledged.lock() {
            Ok(mut acknowledged) => acknowledged.remove(id),
            Err(poisoned) => poisoned.into_inner().remove(id),
        }
    }

    /// Flip the cancellation token for an in-flight streaming response.
    /// Returns `true` when a matching token was found; `false` when the
    /// response is unknown or has already completed.
    pub fn cancel_in_flight(&self, id: &str) -> bool {
        let g = match self.in_flight.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(token) = g.get(id) {
            token.store(true, Ordering::Relaxed);
            match self.cancel_acknowledged.lock() {
                Ok(mut acknowledged) => acknowledged.insert(id.to_string()),
                Err(poisoned) => poisoned.into_inner().insert(id.to_string()),
            };
            true
        } else {
            false
        }
    }

    /// Acquire a write guard on the in-flight registry. Used by the
    /// streaming route to install a caller-provided cancellation token
    /// (so the registry shares the same `Arc<AtomicBool>` that the SSE
    /// channel already uses for client-disconnect detection).
    pub fn in_flight_write(
        &self,
    ) -> Result<
        std::sync::RwLockWriteGuard<'_, HashMap<String, InFlightToken>>,
        std::sync::PoisonError<std::sync::RwLockWriteGuard<'_, HashMap<String, InFlightToken>>>,
    > {
        self.in_flight.write()
    }

    /// Insert a response. Evicts expired and LRU entries to keep both
    /// entry count and approximate retained bytes under budget. Returns the count of
    /// remaining entries after the insert for tests/telemetry.
    pub fn insert(&self, id: String, payload: StoredResponse) -> usize {
        let size_bytes = Self::entry_size_bytes(&id, &payload);
        let now = Instant::now();
        let mut state = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Self::insert_locked(
            &mut state,
            &self.config,
            id,
            payload,
            size_bytes,
            BackgroundRetention::default(),
            now,
        );
        state.entries.len()
    }

    fn insert_locked(
        state: &mut StoreState,
        config: &ResponsesStoreConfig,
        id: String,
        payload: StoredResponse,
        size_bytes: usize,
        retention: BackgroundRetention,
        now: Instant,
    ) {
        Self::sweep_expired(state, config, now);
        Self::remove_entry(state, &id);

        let lru_key = Self::next_lru_key(state, &id, now);
        state.total_bytes = state.total_bytes.saturating_add(size_bytes);
        state.lru.insert(lru_key.clone());
        state.entries.insert(
            id,
            Entry {
                payload,
                inserted_at: now,
                last_accessed: now,
                size_bytes,
                lru_key,
                ephemeral_ttl: retention.ephemeral_ttl,
                event_log: retention.event_log,
                event_log_dropped: retention.event_log_dropped,
            },
        );
        Self::evict_to_limits(state, config);
    }

    /// Look up a stored response. Refreshes the entry's LRU stamp.
    /// Returns `None` for missing or expired entries, and for the ephemeral
    /// terminal object of a `store: false` background job.
    pub fn get(&self, id: &str) -> Option<StoredResponse> {
        let now = Instant::now();
        let mut state = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Self::sweep_expired(&mut state, &self.config, now);
        let existing = state.entries.get(id)?;
        if existing.ephemeral_ttl.is_some() {
            return None;
        }
        let old_lru_key = existing.lru_key.clone();
        state.lru.remove(&old_lru_key);
        let lru_key = Self::next_lru_key(&mut state, id, now);
        let entry = state.entries.get_mut(id)?;
        entry.last_accessed = now;
        entry.lru_key = lru_key.clone();
        let payload = entry.payload.clone();
        state.lru.insert(lru_key);
        Some(payload)
    }

    /// Remove an entry. Returns the previous value when present.
    pub fn remove(&self, id: &str) -> Option<StoredResponse> {
        let mut state = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Self::remove_entry(&mut state, id).map(|e| e.payload)
    }

    /// Current number of live entries (snapshot).
    pub fn len(&self) -> usize {
        match self.inner.read() {
            Ok(g) => g.entries.len(),
            Err(poisoned) => poisoned.into_inner().entries.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Configuration snapshot.
    pub fn config(&self) -> &ResponsesStoreConfig {
        &self.config
    }

    /// Approximate retained bytes currently accounted by the store: stored
    /// entries (with the event logs they carry) plus active background jobs
    /// and the current size of their event logs.
    pub fn approximate_total_bytes(&self) -> usize {
        let jobs = self.lock_jobs();
        let stored = match self.inner.read() {
            Ok(g) => g.total_bytes,
            Err(poisoned) => poisoned.into_inner().total_bytes,
        };
        let live_logs = jobs
            .jobs
            .values()
            .filter_map(|job| job.event_log.as_ref())
            .fold(0usize, |sum, log| sum.saturating_add(log.retained_bytes()));
        stored
            .saturating_add(jobs.active_bytes)
            .saturating_add(live_logs)
    }

    // -- Background jobs -----------------------------------------------------

    /// Admit a `background: true` response and start tracking it as an active
    /// job with status `queued`. The snapshot's status is forced to
    /// [`ResponseStatus::Queued`]. The returned token is the job's
    /// cancellation flag; [`Self::request_background_cancel`] flips it.
    ///
    /// Refuses with [`BackgroundAdmitError::Disabled`] when the job limit,
    /// `max_entries`, or `max_bytes` is `0`,
    /// [`BackgroundAdmitError::LimitReached`] when the job limit is reached,
    /// and [`BackgroundAdmitError::OverBudget`] when the active jobs plus this
    /// record would exceed `max_bytes`.
    pub fn admit_background(
        &self,
        snapshot: ResponseObject,
        input_items: Vec<ResponseInputItem>,
        effective_store: bool,
    ) -> Result<InFlightToken, BackgroundAdmitError> {
        let limit = self.config.background_max_jobs;
        if limit == 0 || self.config.max_entries == 0 || self.config.max_bytes == 0 {
            return Err(BackgroundAdmitError::Disabled);
        }
        let mut snapshot = snapshot;
        snapshot.status = ResponseStatus::Queued;
        let id = snapshot.id.clone();
        let input_bytes = input_items_size_bytes(&input_items);
        let size_bytes = id
            .len()
            .saturating_add(serialized_json_len_saturating(&snapshot))
            .saturating_add(input_bytes);

        let mut jobs = self.lock_jobs();
        // A colliding id replaces the old job (ids are random, so this is
        // defensive); its bytes must not be counted twice.
        let replaced_bytes = jobs.jobs.get(&id).map_or(0, |job| job.size_bytes);
        let active_without_replaced = jobs.active_bytes.saturating_sub(replaced_bytes);
        if replaced_bytes == 0 && jobs.jobs.len() >= limit {
            return Err(BackgroundAdmitError::LimitReached { limit });
        }
        if active_without_replaced.saturating_add(size_bytes) > self.config.max_bytes {
            return Err(BackgroundAdmitError::OverBudget);
        }
        let token: InFlightToken = Arc::new(AtomicBool::new(false));
        let job = BackgroundJob {
            snapshot,
            input_items,
            input_bytes,
            effective_store,
            token: token.clone(),
            cancel_requested: false,
            size_bytes,
            signal: Arc::new(BackgroundSignal::default()),
            event_log: None,
        };
        if let Some(old) = jobs.jobs.insert(id, job) {
            old.token.store(true, Ordering::Relaxed);
        }
        jobs.active_bytes = active_without_replaced.saturating_add(size_bytes);
        Ok(token)
    }

    /// Move an active job from `queued` to `in_progress`. Returns `false`
    /// when the job is unknown, already terminal, or not `queued`.
    pub fn mark_background_in_progress(&self, id: &str) -> bool {
        let mut jobs = self.lock_jobs();
        match jobs.jobs.get_mut(id) {
            Some(job) if job.snapshot.status == ResponseStatus::Queued => {
                job.snapshot.status = ResponseStatus::InProgress;
                true
            }
            _ => false,
        }
    }

    /// Give an active job its event log (issue #2900): the job was created
    /// with `stream: true`. Returns `false` when no active job has this id.
    ///
    /// Call it right after [`Self::admit_background`], before the id is
    /// handed to anyone who could look the log up.
    pub fn attach_background_event_log(&self, id: &str, log: Arc<BackgroundEventLog>) -> bool {
        match self.lock_jobs().jobs.get_mut(id) {
            Some(job) => {
                job.event_log = Some(log);
                true
            }
            None => false,
        }
    }

    /// The event log of a streamed background response: the active job's,
    /// or the one retained with its terminal object. `None` for an unknown
    /// id and for a response that was not created with `background: true`
    /// and `stream: true`. Does not refresh the entry's LRU stamp.
    pub fn background_event_log(&self, id: &str) -> Option<Arc<BackgroundEventLog>> {
        let now = Instant::now();
        let jobs = self.lock_jobs();
        if let Some(job) = jobs.jobs.get(id) {
            return job.event_log.clone();
        }
        let mut state = self.write_inner();
        Self::sweep_expired(&mut state, &self.config, now);
        state
            .entries
            .get(id)
            .and_then(|entry| entry.event_log.clone())
    }

    /// Whether `id` is a retained streamed background response whose event
    /// log was not kept because it did not fit the store budget beside the
    /// response. Lets a resume tell that apart from a response that never
    /// had a log.
    pub fn background_event_log_dropped(&self, id: &str) -> bool {
        let state = match self.inner.read() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        state
            .entries
            .get(id)
            .is_some_and(|entry| entry.event_log_dropped)
    }

    /// Replace the input items an active job will be stored with. Automatic
    /// compaction rewrites them after admission: the stored response then
    /// carries the capsule in its output, so a chained request must not also
    /// replay the pre-compaction input. Returns `false` when no active job
    /// has this id.
    ///
    /// The byte budget is not re-checked: compaction only shrinks the items,
    /// and [`Self::finish_background`] fits the terminal object anyway.
    pub fn set_background_input_items(
        &self,
        id: &str,
        input_items: Vec<ResponseInputItem>,
    ) -> bool {
        let input_bytes = input_items_size_bytes(&input_items);
        let mut jobs = self.lock_jobs();
        let Some(job) = jobs.jobs.get_mut(id) else {
            return false;
        };
        let old_size = job.size_bytes;
        let new_size = old_size
            .saturating_sub(job.input_bytes)
            .saturating_add(input_bytes);
        job.input_items = input_items;
        job.input_bytes = input_bytes;
        job.size_bytes = new_size;
        jobs.active_bytes = jobs
            .active_bytes
            .saturating_sub(old_size)
            .saturating_add(new_size);
        true
    }

    /// Record a job's terminal object. The first call for an active job
    /// removes the job and inserts the terminal object as a stored entry in
    /// one critical section (ephemeral when the job was admitted with
    /// `effective_store: false`), wakes [`Self::wait_background_terminal`]
    /// callers, and returns what was recorded. A terminal object larger than
    /// `max_bytes` is recorded as a `failed` object instead.
    ///
    /// A streamed job's event log gets its terminal event here, built from
    /// the recorded object (`response.failed`, `response.incomplete`, or
    /// `response.completed`), and moves onto the stored entry with its bytes.
    /// The event is serialized before the job lock is taken. A log that does
    /// not fit `max_bytes` beside the object is not retained: readers waiting
    /// on the live tail still receive the terminal event, a reader that is
    /// still replaying gets the not-retained error, and a resume gets a 400
    /// saying so.
    ///
    /// Later calls (or a call for an unknown id) return the recorded object
    /// when it is still retained, otherwise the `terminal` argument.
    pub fn finish_background(&self, id: &str, terminal: ResponseObject) -> ResponseObject {
        // Serialize the (possibly large) terminal object once, before taking
        // any lock, so admits and polls are not blocked behind it.
        let terminal_bytes = serialized_json_len_saturating(&terminal);
        // Likewise the terminal event of a streamed job (issue #2900); it is
        // used when the object is recorded as given.
        let streamed = self
            .lock_jobs()
            .jobs
            .get(id)
            .and_then(|job| job.event_log.clone());
        let prepared = streamed.as_ref().and_then(|log| {
            log.prepare_terminal(&terminal)
                .map(|p| (Arc::clone(log), p))
        });
        let now = Instant::now();
        let mut jobs = self.lock_jobs();
        let Some(job) = jobs.jobs.remove(id) else {
            let mut state = self.write_inner();
            return Self::retained_response_locked(&mut state, &self.config, id, now)
                .unwrap_or(terminal);
        };
        self.record_terminal_locked(jobs, id, job, terminal, terminal_bytes, prepared, now)
    }

    /// Store `terminal` as the terminal object of `job`, which the caller
    /// already removed from `jobs`, and wake the waiters. Consumes the `jobs`
    /// guard: the entry is inserted before the guard is released, so no
    /// reader sees the job neither active nor stored. `prepared` is the
    /// terminal stream event serialized outside the lock for `terminal`; it
    /// is used only when `terminal` is recorded as given, otherwise the event
    /// is serialized here from the recorded object.
    ///
    /// Used by: [`Self::finish_background`] and the cancel paths in
    /// `responses_store_cancel.rs`.
    fn record_terminal_locked(
        &self,
        mut jobs: MutexGuard<'_, BackgroundState>,
        id: &str,
        job: BackgroundJob,
        terminal: ResponseObject,
        terminal_bytes: usize,
        prepared: Option<(Arc<BackgroundEventLog>, PreparedTerminal)>,
        now: Instant,
    ) -> ResponseObject {
        jobs.active_bytes = jobs.active_bytes.saturating_sub(job.size_bytes);

        let (payload, size_bytes, as_given) = self.fit_terminal(
            id,
            terminal,
            terminal_bytes,
            job.input_items,
            job.input_bytes,
        );
        let recorded = payload.response.clone();
        let ephemeral_ttl = (!job.effective_store).then_some(self.config.background_ephemeral_ttl);
        // The terminal event goes in before the entry is sized, so the
        // entry accounts for the whole log.
        let mut size_bytes = size_bytes;
        let mut event_log = None;
        let mut event_log_dropped = false;
        if let Some(log) = job.event_log {
            let prepared = prepared
                .filter(|(prepared_for, _)| as_given && Arc::ptr_eq(prepared_for, &log))
                .map(|(_, prepared)| prepared);
            log.append_terminal_prepared(&recorded, prepared);
            let total = size_bytes.saturating_add(log.retained_bytes());
            if total <= self.config.max_bytes {
                size_bytes = total;
                event_log = Some(log);
            } else {
                event_log_dropped = true;
            }
        }
        {
            let mut state = self.write_inner();
            Self::insert_locked(
                &mut state,
                &self.config,
                id.to_string(),
                payload,
                size_bytes,
                BackgroundRetention {
                    ephemeral_ttl,
                    event_log,
                    event_log_dropped,
                },
                now,
            );
        }
        let _ = job.signal.terminal.set(recorded.clone());
        drop(jobs);
        job.signal.notify.notify_waiters();
        recorded
    }

    /// Current object for `id`: the live snapshot of an active job, or the
    /// retained terminal object, including the ephemeral one of a
    /// `store: false` job. Does not refresh the entry's LRU stamp.
    pub fn background_snapshot(&self, id: &str) -> Option<ResponseObject> {
        let now = Instant::now();
        let jobs = self.lock_jobs();
        if let Some(job) = jobs.jobs.get(id) {
            return Some(job.snapshot.clone());
        }
        let mut state = self.write_inner();
        Self::retained_response_locked(&mut state, &self.config, id, now)
    }

    /// Status of an active job; `None` once the job is terminal or unknown.
    pub fn background_status(&self, id: &str) -> Option<ResponseStatus> {
        self.lock_jobs().jobs.get(id).map(|job| job.snapshot.status)
    }

    /// Request cancellation of a job. For an active job this flips its token,
    /// marks the cancel as requested
    /// ([`Self::background_cancel_requested`]) and keeps the job active; the
    /// job's task records the terminal object through
    /// [`Self::finish_background`].
    pub fn request_background_cancel(&self, id: &str) -> BackgroundCancel {
        let now = Instant::now();
        let mut jobs = self.lock_jobs();
        if let Some(job) = jobs.jobs.get_mut(id) {
            job.cancel_requested = true;
            job.token.store(true, Ordering::Release);
            return BackgroundCancel::Requested;
        }
        let mut state = self.write_inner();
        match Self::retained_response_locked(&mut state, &self.config, id, now) {
            Some(response) => BackgroundCancel::Terminal(response),
            None => BackgroundCancel::NotFound,
        }
    }

    /// Wait up to `timeout` for a job to reach a terminal state. Returns the
    /// recorded terminal object, immediately when the job is already terminal
    /// and retained. Returns `None` on timeout or when no job or retained
    /// object has this id.
    pub async fn wait_background_terminal(
        &self,
        id: &str,
        timeout: Duration,
    ) -> Option<ResponseObject> {
        let signal = {
            let jobs = self.lock_jobs();
            match jobs.jobs.get(id) {
                Some(job) => job.signal.clone(),
                None => {
                    let now = Instant::now();
                    let mut state = self.write_inner();
                    return Self::retained_response_locked(&mut state, &self.config, id, now);
                }
            }
        };
        // `finish_background` sets `terminal` before `notify_waiters`. A
        // `Notified` future observes every `notify_waiters` call made after it
        // is created, so creating it first and then checking `terminal` cannot
        // miss the transition.
        let notified = signal.notify.notified();
        if let Some(recorded) = signal.terminal.get() {
            return Some(recorded.clone());
        }
        let _ = tokio::time::timeout(timeout, notified).await;
        signal.terminal.get().cloned()
    }

    /// Number of active background jobs.
    pub fn active_background_jobs(&self) -> usize {
        self.lock_jobs().jobs.len()
    }

    /// Payload stored for a job's terminal object. When it exceeds
    /// `max_bytes` it is replaced by a small `failed` object so a poll
    /// gets an explanation instead of a surprise 404.
    /// `terminal_bytes` and `input_bytes` are the precomputed sizes of the
    /// two parts; the returned size is the stored entry's accounted bytes,
    /// and the flag says whether the object was stored as given.
    fn fit_terminal(
        &self,
        id: &str,
        terminal: ResponseObject,
        terminal_bytes: usize,
        input_items: Vec<ResponseInputItem>,
        input_bytes: usize,
    ) -> (StoredResponse, usize, bool) {
        let max_bytes = self.config.max_bytes;
        let size_bytes = id
            .len()
            .saturating_add(terminal_bytes)
            .saturating_add(input_bytes);
        let payload = StoredResponse {
            response: terminal,
            input_items,
        };
        if size_bytes <= max_bytes {
            return (payload, size_bytes, true);
        }
        let mut failed = payload.response;
        failed.status = ResponseStatus::Failed;
        failed.output = Vec::new();
        failed.output_text = String::new();
        failed.incomplete_details = None;
        failed.error = Some(ResponseErrorBody {
            code: "server_error".to_string(),
            message: format!(
                "The response exceeded the response store budget of {max_bytes} bytes \
                 (--responses-store-max-bytes) and was not retained."
            ),
        });
        let mut payload = StoredResponse {
            response: failed,
            input_items: Vec::new(),
        };
        let mut size_bytes = Self::entry_size_bytes(id, &payload);
        if size_bytes > max_bytes {
            // Large echoed request fields can still overflow the budget.
            payload.response.instructions = None;
            payload.response.tools = None;
            payload.response.metadata = None;
            size_bytes = Self::entry_size_bytes(id, &payload);
        }
        (payload, size_bytes, false)
    }

    /// Retained (non-expired) stored object for `id`, ephemeral entries
    /// included. Runs the lazy expiry sweep; does not refresh LRU.
    fn retained_response_locked(
        state: &mut StoreState,
        config: &ResponsesStoreConfig,
        id: &str,
        now: Instant,
    ) -> Option<ResponseObject> {
        Self::sweep_expired(state, config, now);
        state
            .entries
            .get(id)
            .map(|entry| entry.payload.response.clone())
    }

    fn lock_jobs(&self) -> MutexGuard<'_, BackgroundState> {
        match self.jobs.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn write_inner(&self) -> std::sync::RwLockWriteGuard<'_, StoreState> {
        match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn sweep_expired(state: &mut StoreState, config: &ResponsesStoreConfig, now: Instant) {
        let ttl = config.ttl;
        let expired: Vec<String> = state
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_expired(ttl, now))
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            Self::remove_entry(state, &id);
        }
    }

    fn evict_to_limits(state: &mut StoreState, config: &ResponsesStoreConfig) {
        while state.entries.len() > config.max_entries || state.total_bytes > config.max_bytes {
            let Some(victim) = state.lru.iter().next().cloned().map(|key| key.id) else {
                break;
            };
            Self::remove_entry(state, &victim);
        }
    }

    fn remove_entry(state: &mut StoreState, id: &str) -> Option<Entry> {
        let entry = state.entries.remove(id)?;
        state.lru.remove(&entry.lru_key);
        state.total_bytes = state.total_bytes.saturating_sub(entry.size_bytes);
        Some(entry)
    }

    fn next_lru_key(state: &mut StoreState, id: &str, now: Instant) -> LruKey {
        let sequence = state.next_sequence;
        state.next_sequence = state.next_sequence.wrapping_add(1);
        LruKey {
            last_accessed: now,
            sequence,
            id: id.to_string(),
        }
    }

    fn entry_size_bytes(id: &str, payload: &StoredResponse) -> usize {
        id.len()
            .saturating_add(serialized_json_len_saturating(&payload.response))
            .saturating_add(input_items_size_bytes(&payload.input_items))
    }
}

fn input_items_size_bytes(items: &[ResponseInputItem]) -> usize {
    items.iter().fold(2usize, |bytes, item| {
        bytes
            .saturating_add(response_input_item_size_bytes(item))
            .saturating_add(1)
    })
}

pub(crate) fn response_input_item_size_bytes(item: &ResponseInputItem) -> usize {
    match item {
        ResponseInputItem::Message {
            role,
            content,
            name,
        } => 48usize
            .saturating_add(response_role_size_bytes(*role))
            .saturating_add(response_input_content_size_bytes(content))
            .saturating_add(name.as_deref().map(str_size_bytes).unwrap_or(0)),
        ResponseInputItem::FunctionCall {
            call_id,
            name,
            arguments,
        } => 56usize
            .saturating_add(str_size_bytes(call_id))
            .saturating_add(str_size_bytes(name))
            .saturating_add(str_size_bytes(arguments)),
        ResponseInputItem::FunctionCallOutput { call_id, output } => 56usize
            .saturating_add(str_size_bytes(call_id))
            .saturating_add(response_tool_output_size_bytes(output)),
        ResponseInputItem::Reasoning { content } => content.iter().fold(32usize, |bytes, part| {
            bytes
                .saturating_add(str_size_bytes(&part.part_type))
                .saturating_add(str_size_bytes(&part.text))
                .saturating_add(16)
        }),
        ResponseInputItem::Compaction { encrypted_content } => {
            32usize.saturating_add(str_size_bytes(encrypted_content))
        }
    }
}

fn response_role_size_bytes(role: ResponseInputRole) -> usize {
    match role {
        ResponseInputRole::User => 4,
        ResponseInputRole::Assistant => 9,
        ResponseInputRole::System => 6,
        ResponseInputRole::Developer => 9,
    }
}

fn response_input_content_size_bytes(content: &ResponseInputContent) -> usize {
    match content {
        ResponseInputContent::Text(text) => str_size_bytes(text),
        ResponseInputContent::Parts(parts) => parts.iter().fold(2usize, |bytes, part| {
            bytes
                .saturating_add(response_input_part_size_bytes(part))
                .saturating_add(1)
        }),
    }
}

fn response_input_part_size_bytes(part: &ResponseInputPart) -> usize {
    match part {
        ResponseInputPart::InputText { text }
        | ResponseInputPart::Text { text }
        | ResponseInputPart::OutputText { text } => 32usize.saturating_add(str_size_bytes(text)),
        ResponseInputPart::Refusal { refusal } => 32usize.saturating_add(str_size_bytes(refusal)),
        ResponseInputPart::InputImage {
            image_url,
            detail,
            file_id,
        } => 48usize
            .saturating_add(image_url.as_deref().map(str_size_bytes).unwrap_or(0))
            .saturating_add(detail.as_deref().map(str_size_bytes).unwrap_or(0))
            .saturating_add(file_id.as_deref().map(str_size_bytes).unwrap_or(0)),
        ResponseInputPart::InputFile { raw }
        | ResponseInputPart::VideoUrl { raw }
        | ResponseInputPart::InputAudio { raw } => {
            32usize.saturating_add(serialized_json_len_saturating(raw))
        }
        ResponseInputPart::ImageUrl { image_url } => 40usize
            .saturating_add(str_size_bytes(&image_url.url))
            .saturating_add(image_url.detail.as_deref().map(str_size_bytes).unwrap_or(0))
            .saturating_add(
                image_url
                    .max_soft_tokens
                    .map(|value| serialized_json_len_saturating(&value))
                    .unwrap_or(0),
            ),
        ResponseInputPart::Unknown { part_type, raw } => 32usize
            .saturating_add(str_size_bytes(part_type))
            .saturating_add(serialized_json_len_saturating(raw)),
    }
}

fn response_tool_output_size_bytes(output: &ResponseToolOutput) -> usize {
    match output {
        ResponseToolOutput::Text(text) => str_size_bytes(text),
        ResponseToolOutput::Parts(parts) => parts.iter().fold(2usize, |bytes, part| {
            bytes
                .saturating_add(response_input_part_size_bytes(part))
                .saturating_add(1)
        }),
    }
}

fn str_size_bytes(value: &str) -> usize {
    serialized_json_len_saturating(value)
}

#[path = "responses_store_cancel.rs"]
mod background_cancel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::responses_request::{ResponseInputContent, ResponseInputRole};
    use crate::server::types::responses_response::{ResponseStatus, ResponseUsage};

    pub(super) fn make_response(id: &str) -> StoredResponse {
        StoredResponse {
            response: ResponseObject {
                id: id.to_string(),
                object: "response".to_string(),
                created_at: 0.0,
                completed_at: None,
                status: ResponseStatus::Completed,
                model: "m".to_string(),
                output: vec![],
                output_text: String::new(),
                usage: ResponseUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    total_tokens: 0,
                    input_tokens_details: None,
                    output_tokens_details: None,
                },
                error: None,
                incomplete_details: None,
                instructions: None,
                tools: None,
                tool_choice: None,
                text: None,
                reasoning: None,
                metadata: None,
                temperature: None,
                top_p: None,
                parallel_tool_calls: None,
                truncation: None,
                max_output_tokens: None,
                max_tool_calls: None,
                top_logprobs: None,
                previous_response_id: None,
                conversation: None,
                prompt_cache_key: None,
                service_tier: None,
                user: None,
                store: Some(true),
                background: None,
            },
            input_items: vec![],
        }
    }

    pub(super) fn make_response_with_input(id: &str, text: &str) -> StoredResponse {
        let mut response = make_response(id);
        response.input_items.push(ResponseInputItem::Message {
            role: ResponseInputRole::User,
            content: ResponseInputContent::Text(text.to_string()),
            name: None,
        });
        response
    }

    pub(super) fn config(max_entries: usize, max_bytes: usize) -> ResponsesStoreConfig {
        ResponsesStoreConfig {
            max_entries,
            max_bytes,
            ttl: Duration::from_secs(3600),
            ..ResponsesStoreConfig::default()
        }
    }

    fn entry_bytes(id: &str, payload: &StoredResponse) -> usize {
        ResponsesStore::entry_size_bytes(id, payload)
    }

    pub(super) fn assert_indexes_consistent(store: &ResponsesStore) {
        let state = store.inner.read().expect("store state lock");
        let summed = state.entries.values().fold(0usize, |bytes, entry| {
            assert!(
                state.lru.contains(&entry.lru_key),
                "entry missing from LRU index"
            );
            bytes.saturating_add(entry.size_bytes)
        });
        assert_eq!(state.lru.len(), state.entries.len());
        assert_eq!(state.total_bytes, summed);
    }

    #[test]
    fn insert_then_get_returns_payload() {
        let store = ResponsesStore::new(ResponsesStoreConfig::default());
        store.insert("resp_1".to_string(), make_response("resp_1"));
        let fetched = store.get("resp_1").expect("entry present");
        assert_eq!(fetched.response.id, "resp_1");
    }

    #[test]
    fn remove_returns_previous_and_drops_entry() {
        let store = ResponsesStore::new(ResponsesStoreConfig::default());
        store.insert("resp_2".to_string(), make_response("resp_2"));
        let removed = store.remove("resp_2").expect("entry present");
        assert_eq!(removed.response.id, "resp_2");
        assert!(store.get("resp_2").is_none());
    }

    #[test]
    fn lru_eviction_runs_when_capacity_exceeded() {
        let store = ResponsesStore::new(ResponsesStoreConfig {
            max_entries: 2,
            ttl: Duration::from_secs(3600),
            ..ResponsesStoreConfig::default()
        });
        store.insert("a".to_string(), make_response("a"));
        store.insert("b".to_string(), make_response("b"));
        // Touch a to update its LRU stamp so b is the LRU victim.
        std::thread::sleep(Duration::from_millis(2));
        let _ = store.get("a");
        store.insert("c".to_string(), make_response("c"));
        assert!(store.get("a").is_some(), "a must remain after eviction");
        assert!(store.get("c").is_some(), "c must remain after eviction");
        assert!(store.get("b").is_none(), "b should have been evicted");
    }

    #[test]
    fn ttl_sweep_drops_expired_entries() {
        let store = ResponsesStore::new(ResponsesStoreConfig {
            max_entries: 10,
            ttl: Duration::from_millis(10),
            ..ResponsesStoreConfig::default()
        });
        store.insert("a".to_string(), make_response("a"));
        std::thread::sleep(Duration::from_millis(20));
        // A subsequent operation triggers the sweep.
        store.insert("b".to_string(), make_response("b"));
        assert!(store.get("a").is_none(), "expired entry must be swept");
        assert!(store.get("b").is_some());
    }

    #[test]
    fn missing_entry_returns_none() {
        let store = ResponsesStore::new(ResponsesStoreConfig::default());
        assert!(store.get("nope").is_none());
    }

    #[test]
    fn cancel_in_flight_flips_token_when_registered() {
        let store = ResponsesStore::new(ResponsesStoreConfig::default());
        let token = Arc::new(AtomicBool::new(false));
        store
            .in_flight_write()
            .unwrap()
            .insert("resp_stream".to_string(), token.clone());
        assert!(!token.load(Ordering::Relaxed));
        let cancelled = store.cancel_in_flight("resp_stream");
        assert!(cancelled);
        assert!(token.load(Ordering::Relaxed));
    }

    #[test]
    fn cancel_in_flight_returns_false_when_unknown() {
        let store = ResponsesStore::new(ResponsesStoreConfig::default());
        assert!(!store.cancel_in_flight("never_registered"));
    }

    #[test]
    fn unregister_in_flight_drops_entry() {
        let store = ResponsesStore::new(ResponsesStoreConfig::default());
        let token = Arc::new(AtomicBool::new(false));
        store
            .in_flight_write()
            .unwrap()
            .insert("resp_stream".to_string(), token);
        store.unregister_in_flight("resp_stream");
        assert!(!store.cancel_in_flight("resp_stream"));
    }

    #[test]
    fn byte_only_eviction_runs_under_entry_cap() {
        let a = make_response_with_input("a", &"a".repeat(64));
        let b = make_response_with_input("b", &"b".repeat(64));
        let c = make_response_with_input("c", &"c".repeat(64));
        let budget = entry_bytes("a", &a).saturating_add(entry_bytes("b", &b));
        let store = ResponsesStore::new(config(10, budget));

        store.insert("a".to_string(), a);
        store.insert("b".to_string(), b);
        std::thread::sleep(Duration::from_millis(2));
        let _ = store.get("a");
        store.insert("c".to_string(), c);

        assert!(store.get("a").is_some(), "recently accessed entry remains");
        assert!(store.get("c").is_some(), "new entry remains");
        assert!(
            store.get("b").is_none(),
            "byte pressure evicts the LRU entry"
        );
        assert!(store.approximate_total_bytes() <= budget);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn simultaneous_count_and_byte_pressure_evicts_to_both_limits() {
        let a = make_response_with_input("a", &"a".repeat(32));
        let b = make_response_with_input("b", &"b".repeat(32));
        let c = make_response_with_input("c", &"c".repeat(32));
        let budget = entry_bytes("b", &b).saturating_add(entry_bytes("c", &c));
        let store = ResponsesStore::new(config(2, budget));

        store.insert("a".to_string(), a);
        store.insert("b".to_string(), b);
        store.insert("c".to_string(), c);

        assert_eq!(store.len(), 2);
        assert!(store.get("a").is_none(), "oldest entry is evicted");
        assert!(store.get("b").is_some());
        assert!(store.get("c").is_some());
        assert!(store.approximate_total_bytes() <= budget);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn oversized_single_response_is_not_retained() {
        let payload = make_response_with_input("large", &"x".repeat(256));
        let size = entry_bytes("large", &payload);
        let store = ResponsesStore::new(config(10, size.saturating_sub(1)));

        assert_eq!(store.insert("large".to_string(), payload), 0);

        assert!(store.get("large").is_none());
        assert_eq!(store.approximate_total_bytes(), 0);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn exact_byte_boundary_is_retained() {
        let payload = make_response_with_input("edge", &"x".repeat(64));
        let size = entry_bytes("edge", &payload);
        let store = ResponsesStore::new(config(1, size));

        assert_eq!(store.insert("edge".to_string(), payload), 1);

        assert!(store.get("edge").is_some());
        assert_eq!(store.len(), 1);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn replacement_updates_running_total_and_lru_index() {
        let small = make_response_with_input("same", "small");
        let large = make_response_with_input("same", &"x".repeat(128));
        let large_size = entry_bytes("same", &large);
        let store = ResponsesStore::new(config(4, large_size));

        store.insert("same".to_string(), small);
        store.insert("same".to_string(), large);

        assert_eq!(store.len(), 1);
        assert_eq!(store.approximate_total_bytes(), large_size);
        assert!(store.remove("same").is_some());
        assert_eq!(store.approximate_total_bytes(), 0);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn zero_byte_budget_keeps_store_enabled_but_retains_nothing() {
        let store = ResponsesStore::new(config(usize::MAX, 0));
        store.insert("a".to_string(), make_response("a"));

        assert_eq!(store.len(), 0);
        assert!(store.get("a").is_none());
        assert_eq!(store.approximate_total_bytes(), 0);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn extreme_budgets_do_not_overflow_or_preallocate_unbounded_capacity() {
        let store = ResponsesStore::new(config(usize::MAX, usize::MAX));
        store.insert("a".to_string(), make_response_with_input("a", "small"));

        assert_eq!(store.len(), 1);
        assert!(store.approximate_total_bytes() > 0);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn running_total_stays_consistent_across_remove_and_ttl_sweep() {
        let store = ResponsesStore::new(ResponsesStoreConfig {
            max_entries: 8,
            max_bytes: DEFAULT_RESPONSES_STORE_MAX_BYTES,
            ttl: Duration::from_millis(10),
            ..ResponsesStoreConfig::default()
        });
        store.insert("a".to_string(), make_response_with_input("a", "first"));
        store.insert("b".to_string(), make_response_with_input("b", "second"));
        assert!(store.remove("a").is_some());
        assert_indexes_consistent(&store);

        std::thread::sleep(Duration::from_millis(20));
        store.insert("c".to_string(), make_response_with_input("c", "third"));

        assert!(store.get("b").is_none());
        assert!(store.get("c").is_some());
        assert_indexes_consistent(&store);
    }
}

#[cfg(test)]
#[path = "responses_store_background_tests.rs"]
mod background_tests;
