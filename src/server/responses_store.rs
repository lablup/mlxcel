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
//! [`std::sync::RwLock`] guards a `HashMap<id, Entry>` with an LRU eviction
//! list and a TTL sweep on every insert/lookup. The store is wired into
//! [`crate::server::state::AppState`] when `store=true` requests are
//! allowed; persistence across restarts is reserved for Phase 3.
//!
//! ## Lifecycle
//!
//! - Insert on response create when `store=true`.
//! - Lookup on `GET /v1/responses/:id` and on chained creates that
//!   reference `previous_response_id`.
//! - Delete on `DELETE /v1/responses/:id` and by the LRU/TTL sweep.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::server::types::responses_request::ResponseInputItem;
use crate::server::types::responses_response::ResponseObject;

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
}

/// Configuration for [`ResponsesStore`].
#[derive(Debug, Clone)]
pub struct ResponsesStoreConfig {
    pub max_entries: usize,
    pub ttl: Duration,
}

impl Default for ResponsesStoreConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
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
    inner: RwLock<HashMap<String, Entry>>,
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
            inner: RwLock::new(HashMap::with_capacity(config.max_entries)),
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

    /// Insert a response. Evicts expired and LRU entries first to keep
    /// the map size at-or-below `max_entries`. Returns the count of
    /// remaining entries after the insert for tests/telemetry.
    pub fn insert(&self, id: String, payload: StoredResponse) -> usize {
        let now = Instant::now();
        let mut map = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Self::sweep_expired(&mut map, &self.config, now);
        Self::evict_to_capacity(&mut map, self.config.max_entries.saturating_sub(1));
        map.insert(
            id,
            Entry {
                payload,
                inserted_at: now,
                last_accessed: now,
            },
        );
        map.len()
    }

    /// Look up a stored response. Refreshes the entry's LRU stamp.
    /// Returns `None` for missing or expired entries.
    pub fn get(&self, id: &str) -> Option<StoredResponse> {
        let now = Instant::now();
        // First take a read lock to check existence/freshness without
        // taking the more expensive write lock on every lookup.
        if let Ok(guard) = self.inner.read()
            && let Some(entry) = guard.get(id)
            && now.saturating_duration_since(entry.inserted_at) >= self.config.ttl
        {
            // Fall through to the write-lock branch to evict.
        }

        let mut map = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Self::sweep_expired(&mut map, &self.config, now);
        let entry = map.get_mut(id)?;
        entry.last_accessed = now;
        Some(entry.payload.clone())
    }

    /// Remove an entry. Returns the previous value when present.
    pub fn remove(&self, id: &str) -> Option<StoredResponse> {
        let mut map = match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        map.remove(id).map(|e| e.payload)
    }

    /// Current number of live entries (snapshot).
    pub fn len(&self) -> usize {
        match self.inner.read() {
            Ok(g) => g.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Configuration snapshot.
    pub fn config(&self) -> &ResponsesStoreConfig {
        &self.config
    }

    fn sweep_expired(
        map: &mut HashMap<String, Entry>,
        config: &ResponsesStoreConfig,
        now: Instant,
    ) {
        let ttl = config.ttl;
        map.retain(|_, entry| now.saturating_duration_since(entry.inserted_at) < ttl);
    }

    fn evict_to_capacity(map: &mut HashMap<String, Entry>, target_size: usize) {
        while map.len() > target_size {
            // Pick the least-recently-accessed entry. O(n) per eviction;
            // acceptable for Phase 1's expected store sizes (≤1024).
            let Some(victim) = map
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            map.remove(&victim);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
