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

//! Turn-boundary state checkpoints in the prompt-cache store (issue #3047).
//!
//! A hybrid model's whole snapshot is mostly attention KV, and at agent-chat
//! lengths it is too large to keep one per turn: the session-chain supersede
//! rule keeps only a conversation's newest history boundary, so a request
//! that edits an earlier turn found nothing closer than the system-prompt
//! root. A checkpoint keeps only the recurrent and convolution state at each
//! history boundary, tens of megabytes instead of a gigabyte. Its attention
//! rows are the leading rows of the conversation's newest whole snapshot
//! (the donor), which the model cuts back at restore.
//!
//! Checkpoints live in their own map with their own byte budget
//! ([`PromptCacheConfig::state_checkpoint_capacity_bytes`]) and LRU, so they
//! never push whole snapshots out and whole snapshots never push them out. A
//! checkpoint is only useful while a donor exists: when a snapshot leaves the
//! store, or a checkpoint is inserted, the checkpoints of that conversation
//! that no remaining snapshot can rebuild are dropped instead of waiting for
//! the LRU (issue #3058). Only that conversation's checkpoints are checked,
//! since a donor never crosses a session bucket.
//!
//! Their bytes count toward the store's resident total
//! ([`PromptCacheStore::bytes`]), and the scheduler gates their growth on the
//! same host-memory guardrail as whole snapshots.
//!
//! A donor qualifies when it is in the same session bucket, its tokens extend
//! the checkpoint's, and its [`SnapshotLineage`] agrees with the checkpoint's
//! over the checkpoint's length: the rows were computed by the same
//! sequences, so cutting them back gives exactly the rows the checkpoint's
//! own sequence held.
//!
//! [`PromptCacheConfig::state_checkpoint_capacity_bytes`]: super::PromptCacheConfig::state_checkpoint_capacity_bytes
//! [`SnapshotLineage`]: super::super::lineage::SnapshotLineage

use std::sync::Arc;
use std::time::Instant;

use super::{
    BucketKey, Inner, InsertError, ModelSnapshotEntry, PromptCacheKey, PromptCacheKeyDigest,
    PromptCacheRejectReason, PromptCacheStore, SessionlessBucketKey, SnapshotSlot,
};

/// Whether a stored entry's session matches the request's, under the rule
/// the snapshot lookup uses: equal keys, or both absent.
fn same_session(stored: Option<&str>, requested: Option<&str>) -> bool {
    match (stored, requested) {
        (Some(a), Some(b)) => a == b,
        (None, None) => true,
        _ => false,
    }
}

impl Inner {
    fn remove_state_checkpoint(&mut self, digest: &PromptCacheKeyDigest) -> Option<usize> {
        let slot = self.state_checkpoints.remove(digest)?;
        self.state_checkpoint_bytes = self
            .state_checkpoint_bytes
            .saturating_sub(slot.entry.size_bytes);
        Some(slot.entry.size_bytes)
    }

    /// The newest whole snapshot that can rebuild `checkpoint`, if any.
    fn donor_for(
        &self,
        checkpoint: &ModelSnapshotEntry,
        sessionless: &SessionlessBucketKey,
        session_key: Option<&str>,
        strictly_longer: bool,
    ) -> Option<PromptCacheKeyDigest> {
        let len = checkpoint.tokens.len();
        self.snapshots
            .iter()
            .filter(|(_, slot)| {
                let donor_len = slot.entry.tokens.len();
                &slot.sessionless == sessionless
                    && same_session(slot.bucket.session_key.as_deref(), session_key)
                    && (donor_len > len || (!strictly_longer && donor_len == len))
                    && slot.entry.tokens.starts_with(&checkpoint.tokens)
                    && slot.entry.lineage.shares_prefix(&checkpoint.lineage, len)
            })
            .max_by_key(|(digest, slot)| (slot.entry.last_used(), *digest.as_bytes()))
            .map(|(digest, _)| *digest)
    }

    /// The longest checkpoint that is a prefix of `tokens`, reaches past
    /// `reach`, and has a donor: `(checkpoint, donor, length)`.
    ///
    /// The checkpoint must stop short of the whole request, because the
    /// prefill re-runs at least the last prompt token for its logits and a
    /// recurrent state cannot be rewound by one.
    pub(super) fn best_state_checkpoint(
        &self,
        sessionless: &SessionlessBucketKey,
        session_key: Option<&str>,
        tokens: &[i32],
        reach: usize,
    ) -> Option<(PromptCacheKeyDigest, PromptCacheKeyDigest, usize)> {
        if self.config.state_checkpoint_capacity_bytes == 0 || self.state_checkpoints.is_empty() {
            return None;
        }
        let min_len = self.config.min_prefix_tokens.max(1);
        let mut candidates: Vec<(&PromptCacheKeyDigest, &SnapshotSlot)> = self
            .state_checkpoints
            .iter()
            .filter(|(_, slot)| {
                let len = slot.entry.tokens.len();
                &slot.sessionless == sessionless
                    && same_session(slot.bucket.session_key.as_deref(), session_key)
                    && len > reach
                    && len >= min_len
                    && len < tokens.len()
                    && tokens.starts_with(&slot.entry.tokens)
            })
            .collect();
        candidates.sort_by_key(|(digest, slot)| {
            std::cmp::Reverse((
                slot.entry.tokens.len(),
                slot.entry.last_used(),
                *digest.as_bytes(),
            ))
        });
        candidates.into_iter().find_map(|(digest, slot)| {
            self.donor_for(&slot.entry, sessionless, session_key, true)
                .map(|donor| (*digest, donor, slot.entry.tokens.len()))
        })
    }

    /// Drop the checkpoints of one conversation that no stored snapshot can
    /// rebuild any more: those in `sessionless` whose session matches
    /// `session_key` under the donor rule. Checkpoints elsewhere are not
    /// probed, because a snapshot in another bucket or session can neither
    /// donate to them nor have been their donor (issue #3058). A snapshot of
    /// exactly the checkpoint's tokens counts: it is the checkpoint's own
    /// turn, and its successor will donate.
    pub(super) fn drop_orphan_state_checkpoints(
        &mut self,
        sessionless: &SessionlessBucketKey,
        session_key: Option<&str>,
    ) -> usize {
        if self.state_checkpoints.is_empty() {
            return 0;
        }
        let scoped: Vec<PromptCacheKeyDigest> = self
            .state_checkpoints
            .iter()
            .filter(|(_, slot)| {
                &slot.sessionless == sessionless
                    && same_session(slot.bucket.session_key.as_deref(), session_key)
            })
            .map(|(digest, _)| *digest)
            .collect();
        #[cfg(test)]
        self.orphan_probes.extend(scoped.iter().copied());
        let orphans: Vec<PromptCacheKeyDigest> = scoped
            .into_iter()
            .filter(|digest| {
                self.state_checkpoints.get(digest).is_some_and(|slot| {
                    self.donor_for(
                        &slot.entry,
                        &slot.sessionless,
                        slot.bucket.session_key.as_deref(),
                        false,
                    )
                    .is_none()
                })
            })
            .collect();
        for digest in &orphans {
            self.remove_state_checkpoint(digest);
        }
        self.state_checkpoint_evictions = self
            .state_checkpoint_evictions
            .saturating_add(orphans.len() as u64);
        orphans.len()
    }

    /// Expire checkpoints idle for longer than the snapshot TTL. Returns the
    /// bytes freed.
    pub(super) fn sweep_state_checkpoint_ttl(&mut self, now: Instant) -> usize {
        if self.config.snapshot_ttl.is_zero() || self.state_checkpoints.is_empty() {
            return 0;
        }
        let ttl = self.config.snapshot_ttl;
        let stale: Vec<PromptCacheKeyDigest> = self
            .state_checkpoints
            .iter()
            .filter(|(_, slot)| now.duration_since(slot.entry.last_used()) >= ttl)
            .map(|(digest, _)| *digest)
            .collect();
        let freed = stale
            .iter()
            .filter_map(|digest| self.remove_state_checkpoint(digest))
            .sum();
        self.state_checkpoint_evictions = self
            .state_checkpoint_evictions
            .saturating_add(stale.len() as u64);
        freed
    }

    /// Evict least-recently-used checkpoints until they fit their budget.
    fn enforce_state_checkpoint_caps(&mut self) {
        while self.state_checkpoint_bytes > self.config.state_checkpoint_capacity_bytes {
            let Some(oldest) = self
                .state_checkpoints
                .iter()
                .min_by_key(|(digest, slot)| (slot.entry.last_used(), *digest.as_bytes()))
                .map(|(digest, _)| *digest)
            else {
                break;
            };
            self.remove_state_checkpoint(&oldest);
            self.state_checkpoint_evictions = self.state_checkpoint_evictions.saturating_add(1);
        }
    }
}

impl PromptCacheStore {
    /// How many bytes inserting a state checkpoint of `entry_bytes` would add
    /// to the store's resident total: the part the checkpoint LRU cannot pay
    /// for because its budget has room to grow into. The scheduler checks it
    /// against the host-memory guardrail like a whole snapshot's growth
    /// (issue #3058).
    pub fn state_checkpoint_growth_bytes(&self, entry_bytes: usize) -> usize {
        self.inner.read().map_or(0, |g| {
            entry_bytes.min(
                g.config
                    .state_checkpoint_capacity_bytes
                    .saturating_sub(g.state_checkpoint_bytes),
            )
        })
    }

    /// Count a [`SnapshotLookupOutcome::CheckpointHit`] whose rebuild
    /// succeeded as one snapshot lookup and hit of `matched_len` tokens. The
    /// lookup leaves it uncounted, so a failed rebuild followed by a lookup
    /// without checkpoints counts once (issue #3058).
    ///
    /// [`SnapshotLookupOutcome::CheckpointHit`]: super::SnapshotLookupOutcome::CheckpointHit
    pub fn record_state_checkpoint_hit(&self, matched_len: usize) {
        {
            let mut guard = self.inner.write().expect("prompt cache inner lock");
            guard.snapshot_lookups = guard.snapshot_lookups.saturating_add(1);
            guard.snapshot_hits = guard.snapshot_hits.saturating_add(1);
            guard.state_checkpoint_hits = guard.state_checkpoint_hits.saturating_add(1);
        }
        self.metrics.record_snapshot_lookup(true, matched_len);
    }

    /// Drop `checkpoint` after the model failed to rebuild a state from it,
    /// so later requests are not offered it again (issue #3058). Matched by
    /// identity, so a checkpoint stored under the same key since then stays.
    /// Returns whether it was still stored.
    pub fn discard_state_checkpoint(&self, checkpoint: &Arc<ModelSnapshotEntry>) -> bool {
        let mut guard = self.inner.write().expect("prompt cache inner lock");
        let Some(digest) = guard
            .state_checkpoints
            .iter()
            .find(|(_, slot)| Arc::ptr_eq(&slot.entry, checkpoint))
            .map(|(digest, _)| *digest)
        else {
            return false;
        };
        guard.remove_state_checkpoint(&digest);
        guard.state_checkpoint_evictions = guard.state_checkpoint_evictions.saturating_add(1);
        true
    }

    /// Byte budget for turn-boundary state checkpoints; `0` when they are off.
    pub fn state_checkpoint_capacity_bytes(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.config.state_checkpoint_capacity_bytes)
            .unwrap_or(0)
    }

    /// Insert a turn-boundary state checkpoint (issue #3047).
    ///
    /// Keyed like a snapshot. The entry's lineage must name the sequences that
    /// computed its rows, since only a donor with the same lineage can rebuild
    /// it; an entry without one is refused as [`InsertError::Disabled`]. An
    /// insert replaces a checkpoint stored under the same key, drops the
    /// checkpoints of the same conversation that no stored snapshot can
    /// rebuild, and then evicts the least recently used ones until the budget
    /// holds.
    pub fn insert_state_checkpoint(
        &self,
        key: &PromptCacheKey<'_>,
        entry: ModelSnapshotEntry,
    ) -> Result<(), InsertError> {
        let digest = key.digest();
        let entry_bytes = entry.size_bytes;
        let mut guard = self.inner.write().expect("prompt cache inner lock");
        let capacity_bytes = guard.config.state_checkpoint_capacity_bytes;
        if !guard.config.is_enabled() || capacity_bytes == 0 || entry.lineage.is_empty() {
            return Err(InsertError::Disabled);
        }
        if key.effective_prefix_len() < guard.config.min_prefix_tokens {
            let got = key.effective_prefix_len();
            let min_required = guard.config.min_prefix_tokens;
            drop(guard);
            self.metrics
                .record_reject(PromptCacheRejectReason::PrefixTooShort, entry_bytes);
            return Err(InsertError::PrefixTooShort { got, min_required });
        }
        if entry_bytes > capacity_bytes {
            drop(guard);
            self.metrics
                .record_reject(PromptCacheRejectReason::Oversized, entry_bytes);
            return Err(InsertError::OversizedEntry {
                entry_bytes,
                capacity_bytes,
            });
        }
        guard.remove_state_checkpoint(&digest);
        guard.state_checkpoint_bytes = guard.state_checkpoint_bytes.saturating_add(entry_bytes);
        guard.state_checkpoints.insert(
            digest,
            SnapshotSlot {
                entry: Arc::new(entry),
                bucket: BucketKey::from_key(key),
                sessionless: SessionlessBucketKey::from_key(key),
            },
        );
        guard.state_checkpoint_inserts = guard.state_checkpoint_inserts.saturating_add(1);
        guard.drop_orphan_state_checkpoints(&SessionlessBucketKey::from_key(key), key.session_key);
        guard.enforce_state_checkpoint_caps();
        Ok(())
    }
}

#[cfg(test)]
#[path = "state_checkpoints_tests.rs"]
mod tests;
