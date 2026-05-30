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

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::server::types::responses_request::ResponseInputItem;
use crate::server::types::responses_response::ResponseOutputItem;

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
}

/// Configuration for [`ConversationStore`].
#[derive(Debug, Clone)]
pub struct ConversationStoreConfig {
    pub max_entries: usize,
    pub ttl: Duration,
}

impl Default for ConversationStoreConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            ttl: Duration::from_secs(3600),
        }
    }
}

/// Thread-safe conversation store with TTL and LRU eviction.
pub struct ConversationStore {
    inner: RwLock<HashMap<String, Entry>>,
    config: ConversationStoreConfig,
}

impl ConversationStore {
    pub fn new(config: ConversationStoreConfig) -> Self {
        Self {
            inner: RwLock::new(HashMap::with_capacity(config.max_entries)),
            config,
        }
    }

    /// Snapshot of the transcript. Refreshes the LRU stamp.
    pub fn get(&self, id: &str) -> Option<ConversationTranscript> {
        let now = Instant::now();
        let mut map = self.write_guard();
        Self::sweep_expired(&mut map, &self.config, now);
        let entry = map.get_mut(id)?;
        entry.last_accessed = now;
        Some(entry.transcript.clone())
    }

    /// Append items to a conversation transcript, creating it if needed.
    pub fn append(&self, id: &str, items: Vec<ConversationItem>) {
        if items.is_empty() {
            return;
        }
        let now = Instant::now();
        let mut map = self.write_guard();
        Self::sweep_expired(&mut map, &self.config, now);
        if !map.contains_key(id) {
            Self::evict_to_capacity(&mut map, self.config.max_entries.saturating_sub(1));
            map.insert(
                id.to_string(),
                Entry {
                    transcript: ConversationTranscript::default(),
                    inserted_at: now,
                    last_accessed: now,
                },
            );
        }
        let entry = map.get_mut(id).expect("inserted above");
        entry.transcript.items.extend(items);
        entry.last_accessed = now;
    }

    pub fn remove(&self, id: &str) -> Option<ConversationTranscript> {
        let mut map = self.write_guard();
        map.remove(id).map(|e| e.transcript)
    }

    pub fn len(&self) -> usize {
        match self.inner.read() {
            Ok(g) => g.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn write_guard(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Entry>> {
        match self.inner.write() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn sweep_expired(
        map: &mut HashMap<String, Entry>,
        config: &ConversationStoreConfig,
        now: Instant,
    ) {
        let ttl = config.ttl;
        map.retain(|_, entry| now.saturating_duration_since(entry.inserted_at) < ttl);
    }

    fn evict_to_capacity(map: &mut HashMap<String, Entry>, target_size: usize) {
        while map.len() > target_size {
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
    use crate::server::types::responses_request::{ResponseInputContent, ResponseInputRole};

    fn user_input(text: &str) -> ResponseInputItem {
        ResponseInputItem::Message {
            role: ResponseInputRole::User,
            content: ResponseInputContent::Text(text.to_string()),
            name: None,
        }
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
}
