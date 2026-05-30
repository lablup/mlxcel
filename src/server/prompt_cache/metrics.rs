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

//! Pluggable metrics hooks for the prompt prefix cache store.
//!
//! is responsible for wiring these hooks into
//! [`super::super::state::BatchMetrics`] / Prometheus counters. Until then a
//! no-op default is supplied so the store can be constructed without any
//! explicit metrics dependency (tests, unit benchmarks, etc.).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Callback surface that the store invokes on relevant events.
///
/// Implementors must be cheap and `Send + Sync`; the store holds its
/// metrics behind an `Arc<dyn PromptCacheMetrics>` so a single instance can
/// be shared across threads without additional locking.
///
/// Each hook gets enough information to record a counter increment or
/// histogram observation on the consumer side. No method returns a value —
/// these calls are pure side effects.
pub trait PromptCacheMetrics: Send + Sync {
    /// Called when an entry is successfully inserted. `bytes` is the
    /// entry's `size_bytes` at insert time.
    fn record_insert(&self, bytes: usize) {
        let _ = bytes;
    }
    /// Called when an insert is rejected because the single entry would
    /// exceed the configured byte budget.
    fn record_reject_oversized(&self, bytes: usize) {
        let _ = bytes;
    }
    /// Called on every lookup. `hit` is `true` when the lookup returned
    /// `Some(..)`, `matched_len` is the number of tokens that matched for
    /// a hit and `0` otherwise.
    fn record_lookup(&self, hit: bool, matched_len: usize) {
        let _ = (hit, matched_len);
    }
    /// Called when an entry is evicted under LRU pressure (entry-cap or
    /// byte-cap). `bytes` is the freed byte count for that entry.
    fn record_evict_lru(&self, bytes: usize) {
        let _ = bytes;
    }
    /// Called when an entry is evicted because its TTL elapsed.
    fn record_evict_ttl(&self, bytes: usize) {
        let _ = bytes;
    }
}

/// No-op metrics implementation. The default for stores constructed without
/// an explicit metrics dependency.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopPromptCacheMetrics;

impl PromptCacheMetrics for NoopPromptCacheMetrics {}

/// Lightweight atomic-counter implementation. Intended for tests and for
/// ad-hoc diagnostics will add a real Prometheus-backed implementor.
#[derive(Debug, Default)]
pub struct AtomicPromptCacheMetrics {
    pub inserts: AtomicU64,
    pub insert_bytes: AtomicU64,
    pub rejects_oversized: AtomicU64,
    pub lookups: AtomicU64,
    pub hits: AtomicU64,
    pub hit_tokens_total: AtomicU64,
    pub evicts_lru: AtomicU64,
    pub evict_lru_bytes: AtomicU64,
    pub evicts_ttl: AtomicU64,
    pub evict_ttl_bytes: AtomicU64,
}

impl AtomicPromptCacheMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience constructor returning the metrics wrapped in an `Arc` so
    /// callers can plug it directly into `PromptCacheStore::with_metrics`.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

impl PromptCacheMetrics for AtomicPromptCacheMetrics {
    fn record_insert(&self, bytes: usize) {
        self.inserts.fetch_add(1, Ordering::Relaxed);
        self.insert_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    fn record_reject_oversized(&self, _bytes: usize) {
        self.rejects_oversized.fetch_add(1, Ordering::Relaxed);
    }
    fn record_lookup(&self, hit: bool, matched_len: usize) {
        self.lookups.fetch_add(1, Ordering::Relaxed);
        if hit {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.hit_tokens_total
                .fetch_add(matched_len as u64, Ordering::Relaxed);
        }
    }
    fn record_evict_lru(&self, bytes: usize) {
        self.evicts_lru.fetch_add(1, Ordering::Relaxed);
        self.evict_lru_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
    fn record_evict_ttl(&self, bytes: usize) {
        self.evicts_ttl.fetch_add(1, Ordering::Relaxed);
        self.evict_ttl_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
}
