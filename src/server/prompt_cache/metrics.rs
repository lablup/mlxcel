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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::InsertError;

/// Fixed set of reasons a prompt-cache store or adopt operation can decline
/// (issue #774). Each variant is grounded in a real decline site in either
/// `PromptCacheStore::insert`/`insert_snapshot` (this crate) or the
/// scheduler's `donate_finished_sequence_cache` / `try_adopt_cached_prefix`
/// (`server::batch::scheduler`) -- see the call sites for exact provenance.
///
/// This is a closed enum rather than an open string key so the counters that
/// track it stay fixed-size: no unbounded map, no per-request label
/// cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheRejectReason {
    /// `InsertError::OversizedEntry`: a single entry exceeds the configured
    /// byte budget on its own.
    Oversized,
    /// `InsertError::Disabled`: the store (or the prompt-cache feature as a
    /// whole) is disabled.
    Disabled,
    /// `InsertError::PrefixTooShort`, or a scheduler-side pre-check against
    /// `min_prefix_tokens` performed before a costly detach/snapshot
    /// capture so the store never even sees the attempt.
    PrefixTooShort,
    /// Cross-backend adoption declined: a dense entry under a paged decode
    /// backend (or vice versa), or a partial match declined under
    /// `require_whole_entry` (multimodal whole-entry policy).
    ModeMismatch,
    /// The detached, cloned, or snapshot KV set carried nothing to store or
    /// adopt (aborted before any prefill completed, or the model produced no
    /// capturable state).
    EmptySet,
    /// A pool-level operation failed: paged clone, dense truncate, paged
    /// trim, snapshot restore, or the final `adopt`/`adopt_paged` call.
    LayoutConstraints,
    /// The paged pool's block-size floor pushed the adoptable/donatable
    /// length below `min_prefix_tokens`.
    BlockBoundaryFloor,
}

impl PromptCacheRejectReason {
    /// All variants, in the stable order used by Prometheus/`/v1/cache/stats`
    /// exposition helpers.
    pub const ALL: [Self; 7] = [
        Self::Oversized,
        Self::Disabled,
        Self::PrefixTooShort,
        Self::ModeMismatch,
        Self::EmptySet,
        Self::LayoutConstraints,
        Self::BlockBoundaryFloor,
    ];

    /// Stable lowercase snake_case label used as the Prometheus `reason`
    /// label value and the `/v1/cache/stats` JSON field suffix.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Oversized => "oversized",
            Self::Disabled => "disabled",
            Self::PrefixTooShort => "prefix_too_short",
            Self::ModeMismatch => "mode_mismatch",
            Self::EmptySet => "empty_set",
            Self::LayoutConstraints => "layout_constraints",
            Self::BlockBoundaryFloor => "block_boundary_floor",
        }
    }
}

impl From<&InsertError> for PromptCacheRejectReason {
    fn from(err: &InsertError) -> Self {
        match err {
            InsertError::Disabled => Self::Disabled,
            InsertError::PrefixTooShort { .. } => Self::PrefixTooShort,
            InsertError::OversizedEntry { .. } => Self::Oversized,
        }
    }
}

/// Snapshot of the most recent prompt-cache reject/decline event.
///
/// `seq_id` is `None` for adopt-path declines that happen before a sequence
/// id has been allocated (the common case), and `Some` for donate-path
/// declines and the handful of adopt declines that occur after allocation.
/// `context_len` carries whatever length context the call site has
/// available: the matched-prefix length for adopt declines, or the donated
/// token count for donate declines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCacheLastReject {
    pub reason: &'static str,
    pub seq_id: Option<u64>,
    pub context_len: usize,
    pub at_unix_ms: u64,
}

/// Fixed-size per-reason reject counters plus a `last_reject` snapshot,
/// behind a light [`Mutex`] (issue #774).
///
/// Shared between [`AtomicPromptCacheMetrics`] (store-level: the three
/// reasons the store itself can detect) and
/// [`crate::server::batch::BatchObservability`] (scheduler-level: the full
/// reason set, including adopt-path declines the store never sees) so the
/// counter/last-reject bookkeeping is implemented exactly once.
#[derive(Debug, Default)]
pub struct PromptCacheRejectCounters {
    oversized: AtomicU64,
    disabled: AtomicU64,
    prefix_too_short: AtomicU64,
    mode_mismatch: AtomicU64,
    empty_set: AtomicU64,
    layout_constraints: AtomicU64,
    block_boundary_floor: AtomicU64,
    last: Mutex<Option<PromptCacheLastReject>>,
}

impl PromptCacheRejectCounters {
    pub fn new() -> Self {
        Self::default()
    }

    fn counter(&self, reason: PromptCacheRejectReason) -> &AtomicU64 {
        match reason {
            PromptCacheRejectReason::Oversized => &self.oversized,
            PromptCacheRejectReason::Disabled => &self.disabled,
            PromptCacheRejectReason::PrefixTooShort => &self.prefix_too_short,
            PromptCacheRejectReason::ModeMismatch => &self.mode_mismatch,
            PromptCacheRejectReason::EmptySet => &self.empty_set,
            PromptCacheRejectReason::LayoutConstraints => &self.layout_constraints,
            PromptCacheRejectReason::BlockBoundaryFloor => &self.block_boundary_floor,
        }
    }

    /// Increment `reason`'s counter and overwrite the `last_reject`
    /// snapshot. Cheap: one relaxed atomic increment plus a short-lived
    /// mutex hold to swap the small `Copy`-ish snapshot value.
    pub fn record(&self, reason: PromptCacheRejectReason, seq_id: Option<u64>, context_len: usize) {
        self.counter(reason).fetch_add(1, Ordering::Relaxed);
        let at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if let Ok(mut guard) = self.last.lock() {
            *guard = Some(PromptCacheLastReject {
                reason: reason.as_str(),
                seq_id,
                context_len,
                at_unix_ms,
            });
        }
    }

    /// Current count for one reason.
    pub fn count(&self, reason: PromptCacheRejectReason) -> u64 {
        self.counter(reason).load(Ordering::Relaxed)
    }

    /// Clone of the most recent reject event, if any has happened yet.
    pub fn last(&self) -> Option<PromptCacheLastReject> {
        self.last.lock().ok().and_then(|g| g.clone())
    }
}

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
    /// Called when a store-level insert/snapshot-insert is declined, with
    /// the specific [`PromptCacheRejectReason`] (issue #774). Supplements
    /// [`record_reject_oversized`](Self::record_reject_oversized) -- which
    /// only covers the byte-oversize decline -- with the two other
    /// store-detectable reasons (`Disabled`, `PrefixTooShort`). Adopt-path
    /// declines (mode mismatch, empty set, layout constraints, block-
    /// boundary flooring) never reach the store, so the scheduler records
    /// those directly against `BatchObservability` instead of through this
    /// trait. The production adapter (`BatchMetricsCacheAdapter`) leaves this
    /// hook at its no-op default, so `/v1/cache/stats` and Prometheus
    /// reject-reason exposition is fed solely by that scheduler path, not by
    /// this hook; only the `AtomicPromptCacheMetrics` test implementor uses it.
    fn record_reject(&self, reason: PromptCacheRejectReason, bytes: usize) {
        let _ = (reason, bytes);
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
    /// Called when an exact-prefix recurrent/model-owned snapshot is inserted.
    fn record_snapshot_insert(&self, bytes: usize) {
        let _ = bytes;
    }
    /// Called on every snapshot lookup.
    fn record_snapshot_lookup(&self, hit: bool, matched_len: usize) {
        let _ = (hit, matched_len);
    }
    /// Called when a snapshot is evicted under LRU pressure.
    fn record_snapshot_evict_lru(&self, bytes: usize) {
        let _ = bytes;
    }
    /// Called when a snapshot expires by TTL.
    fn record_snapshot_evict_ttl(&self, bytes: usize) {
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
    pub snapshot_inserts: AtomicU64,
    pub snapshot_insert_bytes: AtomicU64,
    pub snapshot_lookups: AtomicU64,
    pub snapshot_hits: AtomicU64,
    pub snapshot_hit_tokens_total: AtomicU64,
    pub snapshot_evicts_lru: AtomicU64,
    pub snapshot_evict_lru_bytes: AtomicU64,
    pub snapshot_evicts_ttl: AtomicU64,
    pub snapshot_evict_ttl_bytes: AtomicU64,
    /// Per-reason breakdown of store-level insert declines (issue #774).
    /// Only the three reasons the store itself can detect
    /// (`Disabled`, `PrefixTooShort`, `Oversized`) are ever recorded here;
    /// see [`PromptCacheRejectCounters`] for the shared implementation.
    pub reject_reasons: PromptCacheRejectCounters,
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
    fn record_reject(&self, reason: PromptCacheRejectReason, bytes: usize) {
        self.reject_reasons.record(reason, None, bytes);
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
    fn record_snapshot_insert(&self, bytes: usize) {
        self.snapshot_inserts.fetch_add(1, Ordering::Relaxed);
        self.snapshot_insert_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
    fn record_snapshot_lookup(&self, hit: bool, matched_len: usize) {
        self.snapshot_lookups.fetch_add(1, Ordering::Relaxed);
        if hit {
            self.snapshot_hits.fetch_add(1, Ordering::Relaxed);
            self.snapshot_hit_tokens_total
                .fetch_add(matched_len as u64, Ordering::Relaxed);
        }
    }
    fn record_snapshot_evict_lru(&self, bytes: usize) {
        self.snapshot_evicts_lru.fetch_add(1, Ordering::Relaxed);
        self.snapshot_evict_lru_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
    fn record_snapshot_evict_ttl(&self, bytes: usize) {
        self.snapshot_evicts_ttl.fetch_add(1, Ordering::Relaxed);
        self.snapshot_evict_ttl_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
}
