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

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::server::store_budget::{LruKey, serialized_json_len_saturating};
use crate::server::types::responses_request::{
    ResponseInputContent, ResponseInputItem, ResponseInputPart, ResponseInputRole,
    ResponseToolOutput,
};
use crate::server::types::responses_response::ResponseObject;

/// Default approximate retained-byte budget for the in-memory response store.
///
/// This keeps the default bounded below the historical "1024 arbitrary
/// multimodal entries" footprint while leaving room for several max-size image
/// requests inside the one-hour TTL window.
pub const DEFAULT_RESPONSES_STORE_MAX_BYTES: usize = 256 * 1024 * 1024;

const INITIAL_HASH_CAPACITY_LIMIT: usize = 4096;

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
}

impl Default for ResponsesStoreConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_bytes: DEFAULT_RESPONSES_STORE_MAX_BYTES,
            ttl: Duration::from_secs(3600),
        }
    }
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
}

impl ResponsesStore {
    pub fn new(config: ResponsesStoreConfig) -> Self {
        Self {
            inner: RwLock::new(StoreState::with_capacity(config.max_entries)),
            config,
            in_flight: RwLock::new(HashMap::new()),
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
        let mut g = match self.in_flight.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        g.remove(id);
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
        let now = Instant::now();
        let mut state = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Self::sweep_expired(&mut state, &self.config, now);
        Self::remove_entry(&mut state, &id);

        let size_bytes = Self::entry_size_bytes(&id, &payload);
        let lru_key = Self::next_lru_key(&mut state, &id, now);
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
            },
        );
        Self::evict_to_limits(&mut state, &self.config);
        state.entries.len()
    }

    /// Look up a stored response. Refreshes the entry's LRU stamp.
    /// Returns `None` for missing or expired entries.
    pub fn get(&self, id: &str) -> Option<StoredResponse> {
        let now = Instant::now();
        let mut state = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Self::sweep_expired(&mut state, &self.config, now);
        let old_lru_key = state.entries.get(id)?.lru_key.clone();
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

    /// Approximate retained bytes currently accounted by the store.
    pub fn approximate_total_bytes(&self) -> usize {
        match self.inner.read() {
            Ok(g) => g.total_bytes,
            Err(poisoned) => poisoned.into_inner().total_bytes,
        }
    }

    fn sweep_expired(state: &mut StoreState, config: &ResponsesStoreConfig, now: Instant) {
        let ttl = config.ttl;
        let expired: Vec<String> = state
            .entries
            .iter()
            .filter(|(_, entry)| now.saturating_duration_since(entry.inserted_at) >= ttl)
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
        ResponseInputPart::InputText { text } | ResponseInputPart::Text { text } => {
            32usize.saturating_add(str_size_bytes(text))
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::responses_request::{ResponseInputContent, ResponseInputRole};
    use crate::server::types::responses_response::{ResponseStatus, ResponseUsage};

    fn make_response(id: &str) -> StoredResponse {
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
            },
            input_items: vec![],
        }
    }

    fn make_response_with_input(id: &str, text: &str) -> StoredResponse {
        let mut response = make_response(id);
        response.input_items.push(ResponseInputItem::Message {
            role: ResponseInputRole::User,
            content: ResponseInputContent::Text(text.to_string()),
            name: None,
        });
        response
    }

    fn config(max_entries: usize, max_bytes: usize) -> ResponsesStoreConfig {
        ResponsesStoreConfig {
            max_entries,
            max_bytes,
            ttl: Duration::from_secs(3600),
        }
    }

    fn entry_bytes(id: &str, payload: &StoredResponse) -> usize {
        ResponsesStore::entry_size_bytes(id, payload)
    }

    fn assert_indexes_consistent(store: &ResponsesStore) {
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
