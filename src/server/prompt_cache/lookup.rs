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

//! Two-tier longest-prefix matcher.
//!
//! This module hosts the selection logic invoked by
//! [`super::store::PromptCacheStore::lookup_longest_prefix`]. The actual
//! lock management, metrics, and LRU bookkeeping live on the store — this
//! module only implements the *candidate scoring* step:
//!
//! 1. Ask the radix trie for every candidate sharing at least the
//!    configured minimum prefix length with the request.
//! 2. Split those candidates into two tiers: **same session** (entries
//!    whose `session_key` equals the caller's) and **other session**.
//! 3. Apply tie-breaks: prefer the entry with the longer matched prefix;
//!    on equal length, prefer same-session; still on equal length, prefer
//!    most-recently-used.
//! 4. Return the winning digest and its matched length, or `None`.
//!
//! The tie-break is deliberately ordered so that an exact-session hit is
//! only replaced by a cross-session hit when the latter is *strictly
//! longer* — matching the epic's "same session preferred" rule. A
//! longer-prefix cross-session match is still useful because the shared
//! system prompt / tool preamble is identical regardless of session.

use std::time::Instant;

use super::key::{PromptCacheKey, PromptCacheKeyDigest};
use super::store::EntrySlot;
use super::trie::RadixTrie;

/// A single candidate under consideration during the two-tier selection.
#[derive(Clone, Copy)]
pub(super) struct BestCandidate {
    pub digest: PromptCacheKeyDigest,
    pub matched: usize,
    pub last_used: Instant,
}

impl BestCandidate {
    /// Strict tie-break ordering.
    ///
    /// Longer match always wins. On equal length the MRU wins. Ties on
    /// both axes are broken arbitrarily (first-seen wins) — tests that
    /// depend on this should avoid equal `last_used` values.
    fn beats(&self, other: &BestCandidate) -> bool {
        if self.matched != other.matched {
            return self.matched > other.matched;
        }
        self.last_used > other.last_used
    }
}

/// Resolve a two-tier longest-prefix match against an open radix trie.
///
/// Caller passes:
///
/// * `trie` — the per-`(model, lora, template)` trie indexed by
///   [`SessionlessBucketKey`].
/// * `entries_by_digest` — a lookup from digest to the store's entry
///   slot, used for tier classification (`session_key`) and MRU
///   bookkeeping (`last_used`).
/// * `key` — the incoming request's cache key (for session filtering).
/// * `tokens` — the incoming request's token prefix.
/// * `min_len` — the minimum number of matching tokens required before
///   the candidate is even considered.
/// * `apc_partial_allowed` — when `true`, accept candidates
///   whose stored token prefix is **not** fully contained in the request
///   (i.e. the request and the entry diverge somewhere inside the entry's
///   prefix). The reported `matched` field then carries the actual
///   request-side match depth, which the store's caller is expected to
///   clamp to a block boundary via
///   [`super::apc_lookup::apc_consistent_prefix_len`] before adopting.
///   When `false`, the legacy whole-prefix-contained safety check
///   applies: a candidate is rejected unless `matched == dl.token_len`,
///   matching the pre-APC behaviour exactly.
///
/// Returns the winning [`BestCandidate`], or `None` when no candidate
/// reaches the minimum-length threshold.
pub(super) fn select_best<'a, F>(
    trie: &RadixTrie,
    key: &PromptCacheKey<'_>,
    tokens: &[i32],
    min_len: usize,
    apc_partial_allowed: bool,
    mut slot_for_digest: F,
) -> Option<BestCandidate>
where
    F: FnMut(&PromptCacheKeyDigest) -> Option<&'a EntrySlot>,
{
    let caller_session = key.session_key;
    let m = trie.find_longest_prefix(tokens, min_len)?;
    let match_depth = m.depth();

    let mut best_same_session: Option<BestCandidate> = None;
    let mut best_other_session: Option<BestCandidate> = None;

    m.for_each_candidate(|dl| {
        let slot = match slot_for_digest(&dl.digest) {
            Some(s) => s,
            None => return,
        };
        if !slot.entry.has_detached() {
            return;
        }
        // A detached KV set can only be adopted safely when the whole stored
        // token prefix is contained in the incoming prompt. If the request
        // diverged before the stored entry ended, adopting that entry would
        // import KV state for tokens the caller did not send — UNLESS APC
        // partial adoption is enabled, in which case the
        // store's caller will clamp the matched length to the longest
        // block-aligned consistent prefix and pass that to the model
        // worker's truncate path.
        if dl.token_len > tokens.len() && !apc_partial_allowed {
            return;
        }
        // Cap at the lesser of the trie depth and the entry's stored
        // prefix length. With APC off this expression equals
        // `dl.token_len` whenever the candidate would survive the legacy
        // gate; with APC on it may be strictly less, which the caller
        // clamps to a block boundary via the block-hash discriminator.
        let matched = match_depth.min(dl.token_len);
        if !apc_partial_allowed && matched != dl.token_len {
            return;
        }
        if matched < min_len {
            return;
        }

        // Tier classification by session_key equality. `session_key` is
        // `Option<String>` on the bucket side and `Option<&str>` on the
        // key side; compare by value rather than allocating.
        let same_session = match (&slot.bucket.session_key, caller_session) {
            (Some(a), Some(b)) => a.as_str() == b,
            (None, None) => true,
            _ => false,
        };

        let candidate = BestCandidate {
            digest: dl.digest,
            matched,
            last_used: slot.entry.last_used(),
        };
        let bucket = if same_session {
            &mut best_same_session
        } else {
            &mut best_other_session
        };
        match bucket {
            None => *bucket = Some(candidate),
            Some(existing) => {
                if candidate.beats(existing) {
                    *existing = candidate;
                }
            }
        }
    });

    // Tie-break: same-session wins equal or shorter matches; cross-session
    // only overturns when strictly longer.
    match (best_same_session, best_other_session) {
        (Some(s), Some(o)) if o.matched > s.matched => Some(o),
        (Some(s), _) => Some(s),
        (None, Some(o)) => Some(o),
        (None, None) => None,
    }
}
