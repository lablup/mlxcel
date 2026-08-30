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

//! In-memory conversation store for the OpenAI Responses API.
//!
//! Phase 1 keeps a single ordered transcript per `conversation` id, holding
//! both prior inputs and prior outputs. Each `POST /v1/responses` that
//! references a `conversation` appends the current request's inputs and
//! the generated outputs after completion. Configuration mirrors
//! [`crate::server::responses_store`] — same TTL/LRU semantics, separate
//! capacity knobs.

use std::collections::{BTreeSet, HashMap};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::server::responses_store::response_input_item_size_bytes;
use crate::server::store_budget::{LruKey, serialized_json_len_saturating};
use crate::server::types::responses_request::ResponseInputItem;
use crate::server::types::responses_response::ResponseOutputItem;

/// Default approximate retained-byte budget for conversation transcripts.
pub const DEFAULT_CONVERSATION_STORE_MAX_BYTES: usize = 64 * 1024 * 1024;

const INITIAL_HASH_CAPACITY_LIMIT: usize = 4096;

/// One entry in a conversation transcript.
#[derive(Debug, Clone)]
pub enum ConversationItem {
    Input(ResponseInputItem),
    Output(ResponseOutputItem),
}

/// Ordered transcript for a single conversation id.
#[derive(Debug, Clone, Default)]
pub struct ConversationTranscript {
    pub items: Vec<ConversationItem>,
}

#[derive(Debug)]
struct Entry {
    transcript: ConversationTranscript,
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

/// Configuration for [`ConversationStore`].
#[derive(Debug, Clone)]
pub struct ConversationStoreConfig {
    pub max_entries: usize,
    pub max_bytes: usize,
    pub ttl: Duration,
}

impl Default for ConversationStoreConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_bytes: DEFAULT_CONVERSATION_STORE_MAX_BYTES,
            ttl: Duration::from_secs(3600),
        }
    }
}

/// Thread-safe conversation store with TTL and LRU eviction.
pub struct ConversationStore {
    inner: RwLock<StoreState>,
    config: ConversationStoreConfig,
}

impl ConversationStore {
    pub fn new(config: ConversationStoreConfig) -> Self {
        Self {
            inner: RwLock::new(StoreState::with_capacity(config.max_entries)),
            config,
        }
    }

    /// Snapshot of the transcript. Refreshes the LRU stamp.
    pub fn get(&self, id: &str) -> Option<ConversationTranscript> {
        let now = Instant::now();
        let mut state = self.write_guard();
        Self::sweep_expired(&mut state, &self.config, now);
        let old_lru_key = state.entries.get(id)?.lru_key.clone();
        state.lru.remove(&old_lru_key);
        let lru_key = Self::next_lru_key(&mut state, id, now);
        let entry = state.entries.get_mut(id)?;
        entry.last_accessed = now;
        entry.lru_key = lru_key.clone();
        let transcript = entry.transcript.clone();
        state.lru.insert(lru_key);
        Some(transcript)
    }

    /// Append items to a conversation transcript, creating it if needed.
    pub fn append(&self, id: &str, items: Vec<ConversationItem>) {
        if items.is_empty() {
            return;
        }
        let now = Instant::now();
        let mut state = self.write_guard();
        Self::sweep_expired(&mut state, &self.config, now);
        let mut entry = Self::remove_entry(&mut state, id).unwrap_or_else(|| Entry {
            transcript: ConversationTranscript::default(),
            inserted_at: now,
            last_accessed: now,
            size_bytes: 0,
            lru_key: LruKey {
                last_accessed: now,
                sequence: 0,
                id: id.to_string(),
            },
        });
        entry.transcript.items.extend(items);
        entry.last_accessed = now;
        entry.size_bytes = Self::transcript_size_bytes(id, &entry.transcript);
        entry.lru_key = Self::next_lru_key(&mut state, id, now);
        state.total_bytes = state.total_bytes.saturating_add(entry.size_bytes);
        state.lru.insert(entry.lru_key.clone());
        state.entries.insert(id.to_string(), entry);
        Self::evict_to_limits(&mut state, &self.config);
    }

    pub fn remove(&self, id: &str) -> Option<ConversationTranscript> {
        let mut state = self.write_guard();
        Self::remove_entry(&mut state, id).map(|e| e.transcript)
    }

    pub fn len(&self) -> usize {
        match self.inner.read() {
            Ok(g) => g.entries.len(),
            Err(poisoned) => poisoned.into_inner().entries.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate retained bytes currently accounted by the store.
    pub fn approximate_total_bytes(&self) -> usize {
        match self.inner.read() {
            Ok(g) => g.total_bytes,
            Err(poisoned) => poisoned.into_inner().total_bytes,
        }
    }

    fn write_guard(&self) -> std::sync::RwLockWriteGuard<'_, StoreState> {
        match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn sweep_expired(state: &mut StoreState, config: &ConversationStoreConfig, now: Instant) {
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

    fn evict_to_limits(state: &mut StoreState, config: &ConversationStoreConfig) {
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

    fn transcript_size_bytes(id: &str, transcript: &ConversationTranscript) -> usize {
        id.len()
            .saturating_add(transcript.items.iter().fold(2usize, |bytes, item| {
                bytes
                    .saturating_add(conversation_item_size_bytes(item))
                    .saturating_add(1)
            }))
    }
}

fn conversation_item_size_bytes(item: &ConversationItem) -> usize {
    match item {
        ConversationItem::Input(input) => response_input_item_size_bytes(input),
        ConversationItem::Output(output) => serialized_json_len_saturating(output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::types::responses_request::{ResponseInputContent, ResponseInputRole};

    fn user_input(text: &str) -> ResponseInputItem {
        ResponseInputItem::Message {
            role: ResponseInputRole::User,
            content: ResponseInputContent::Text(text.to_string()),
            name: None,
        }
    }

    fn config(max_entries: usize, max_bytes: usize) -> ConversationStoreConfig {
        ConversationStoreConfig {
            max_entries,
            max_bytes,
            ttl: Duration::from_secs(3600),
        }
    }

    fn transcript_with(text: &str) -> ConversationTranscript {
        ConversationTranscript {
            items: vec![ConversationItem::Input(user_input(text))],
        }
    }

    fn transcript_bytes(id: &str, text: &str) -> usize {
        ConversationStore::transcript_size_bytes(id, &transcript_with(text))
    }

    fn assert_indexes_consistent(store: &ConversationStore) {
        let state = store.inner.read().expect("conversation store state lock");
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
    fn append_creates_new_transcript() {
        let store = ConversationStore::new(ConversationStoreConfig::default());
        store.append("conv_1", vec![ConversationItem::Input(user_input("hi"))]);
        let transcript = store.get("conv_1").unwrap();
        assert_eq!(transcript.items.len(), 1);
    }

    #[test]
    fn append_extends_existing_transcript() {
        let store = ConversationStore::new(ConversationStoreConfig::default());
        store.append(
            "conv_1",
            vec![ConversationItem::Input(user_input("turn 1"))],
        );
        store.append(
            "conv_1",
            vec![ConversationItem::Input(user_input("turn 2"))],
        );
        let transcript = store.get("conv_1").unwrap();
        assert_eq!(transcript.items.len(), 2);
    }

    #[test]
    fn ttl_sweep_drops_expired_conversation() {
        let store = ConversationStore::new(ConversationStoreConfig {
            max_entries: 8,
            ttl: Duration::from_millis(10),
            ..ConversationStoreConfig::default()
        });
        store.append("conv_a", vec![ConversationItem::Input(user_input("hi"))]);
        std::thread::sleep(Duration::from_millis(20));
        store.append("conv_b", vec![ConversationItem::Input(user_input("hi"))]);
        assert!(store.get("conv_a").is_none());
        assert!(store.get("conv_b").is_some());
    }

    #[test]
    fn lru_eviction_picks_least_recent() {
        let store = ConversationStore::new(ConversationStoreConfig {
            max_entries: 2,
            ttl: Duration::from_secs(3600),
            ..ConversationStoreConfig::default()
        });
        store.append("a", vec![ConversationItem::Input(user_input("hi"))]);
        store.append("b", vec![ConversationItem::Input(user_input("hi"))]);
        std::thread::sleep(Duration::from_millis(2));
        let _ = store.get("a");
        store.append("c", vec![ConversationItem::Input(user_input("hi"))]);
        assert!(store.get("a").is_some());
        assert!(store.get("c").is_some());
        assert!(store.get("b").is_none());
    }

    #[test]
    fn empty_append_is_noop() {
        let store = ConversationStore::new(ConversationStoreConfig::default());
        store.append("conv", vec![]);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn byte_only_eviction_runs_under_entry_cap() {
        let budget = transcript_bytes("a", &"a".repeat(64))
            .saturating_add(transcript_bytes("b", &"b".repeat(64)));
        let store = ConversationStore::new(config(10, budget));

        store.append(
            "a",
            vec![ConversationItem::Input(user_input(&"a".repeat(64)))],
        );
        store.append(
            "b",
            vec![ConversationItem::Input(user_input(&"b".repeat(64)))],
        );
        std::thread::sleep(Duration::from_millis(2));
        let _ = store.get("a");
        store.append(
            "c",
            vec![ConversationItem::Input(user_input(&"c".repeat(64)))],
        );

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
        let budget = transcript_bytes("b", &"b".repeat(32))
            .saturating_add(transcript_bytes("c", &"c".repeat(32)));
        let store = ConversationStore::new(config(2, budget));

        store.append(
            "a",
            vec![ConversationItem::Input(user_input(&"a".repeat(32)))],
        );
        store.append(
            "b",
            vec![ConversationItem::Input(user_input(&"b".repeat(32)))],
        );
        store.append(
            "c",
            vec![ConversationItem::Input(user_input(&"c".repeat(32)))],
        );

        assert_eq!(store.len(), 2);
        assert!(store.get("a").is_none(), "oldest entry is evicted");
        assert!(store.get("b").is_some());
        assert!(store.get("c").is_some());
        assert!(store.approximate_total_bytes() <= budget);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn oversized_single_conversation_is_not_retained() {
        let size = transcript_bytes("large", &"x".repeat(256));
        let store = ConversationStore::new(config(10, size.saturating_sub(1)));

        store.append(
            "large",
            vec![ConversationItem::Input(user_input(&"x".repeat(256)))],
        );

        assert!(store.get("large").is_none());
        assert_eq!(store.approximate_total_bytes(), 0);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn exact_byte_boundary_is_retained() {
        let text = "x".repeat(64);
        let size = transcript_bytes("edge", &text);
        let store = ConversationStore::new(config(1, size));

        store.append("edge", vec![ConversationItem::Input(user_input(&text))]);

        assert!(store.get("edge").is_some());
        assert_eq!(store.len(), 1);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn appending_existing_transcript_updates_running_total_and_lru_index() {
        let first = "first";
        let second = "second".repeat(32);
        let combined = ConversationTranscript {
            items: vec![
                ConversationItem::Input(user_input(first)),
                ConversationItem::Input(user_input(&second)),
            ],
        };
        let budget = ConversationStore::transcript_size_bytes("same", &combined);
        let store = ConversationStore::new(config(4, budget));

        store.append("same", vec![ConversationItem::Input(user_input(first))]);
        store.append("same", vec![ConversationItem::Input(user_input(&second))]);

        assert_eq!(store.len(), 1);
        assert_eq!(store.approximate_total_bytes(), budget);
        assert_eq!(store.get("same").unwrap().items.len(), 2);
        assert!(store.remove("same").is_some());
        assert_eq!(store.approximate_total_bytes(), 0);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn zero_byte_budget_keeps_store_enabled_but_retains_nothing() {
        let store = ConversationStore::new(config(usize::MAX, 0));
        store.append("a", vec![ConversationItem::Input(user_input("hi"))]);

        assert_eq!(store.len(), 0);
        assert!(store.get("a").is_none());
        assert_eq!(store.approximate_total_bytes(), 0);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn extreme_budgets_do_not_overflow_or_preallocate_unbounded_capacity() {
        let store = ConversationStore::new(config(usize::MAX, usize::MAX));
        store.append("a", vec![ConversationItem::Input(user_input("small"))]);

        assert_eq!(store.len(), 1);
        assert!(store.approximate_total_bytes() > 0);
        assert_indexes_consistent(&store);
    }

    #[test]
    fn running_total_stays_consistent_across_remove_and_ttl_sweep() {
        let store = ConversationStore::new(ConversationStoreConfig {
            max_entries: 8,
            max_bytes: DEFAULT_CONVERSATION_STORE_MAX_BYTES,
            ttl: Duration::from_millis(10),
        });
        store.append("a", vec![ConversationItem::Input(user_input("first"))]);
        store.append("b", vec![ConversationItem::Input(user_input("second"))]);
        assert!(store.remove("a").is_some());
        assert_indexes_consistent(&store);

        std::thread::sleep(Duration::from_millis(20));
        store.append("c", vec![ConversationItem::Input(user_input("third"))]);

        assert!(store.get("b").is_none());
        assert!(store.get("c").is_some());
        assert_indexes_consistent(&store);
    }
}
