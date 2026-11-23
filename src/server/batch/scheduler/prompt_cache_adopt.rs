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

use super::prompt_cache::whole_prompt_reuse_cap;
use super::*;

/// What [`BatchScheduler::try_adopt_cached_prefix`] did for one request.
///
/// Admission records a dense prompt-cache miss only for [`Self::DenseMiss`]
/// (issue #3294). A bare `Option` could not tell "the dense lookup ran and
/// nothing was adopted" from "no dense lookup ran at all", so every cold
/// request on a snapshot or ModelOwned family used to count as a dense miss
/// while `/v1/cache/stats` `lookups` stayed at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefixAdoptOutcome {
    /// A KV prefix, snapshot, or state checkpoint was adopted.
    Adopted {
        seq_id: SequenceId,
        adopted_len: usize,
    },
    /// `store.lookup_longest_prefix` ran and nothing was adopted: a store
    /// miss, or a store hit the scheduler declined.
    DenseMiss,
    /// No dense KV lookup ran: the cache is inactive, the snapshot branch
    /// declined or failed, or the family never uses the dense path.
    NoDenseLookup,
}

impl PrefixAdoptOutcome {
    /// The adopted sequence and the length it covers, if anything was adopted.
    pub(super) fn adopted(self) -> Option<(SequenceId, usize)> {
        match self {
            Self::Adopted {
                seq_id,
                adopted_len,
            } => Some((seq_id, adopted_len)),
            Self::DenseMiss | Self::NoDenseLookup => None,
        }
    }

    /// Classify the result of [`BatchScheduler::adopt_dense_prefix`].
    ///
    /// Valid only after `lookup_longest_prefix` ran: a `None` is then a dense
    /// miss. Calling it for a request that never reached the dense lookup
    /// would count a miss for a lookup that did not happen (issue #3294).
    pub(super) fn from_dense_lookup(adopted: Option<(SequenceId, usize)>) -> Self {
        match adopted {
            Some((seq_id, adopted_len)) => Self::Adopted {
                seq_id,
                adopted_len,
            },
            None => Self::DenseMiss,
        }
    }
}

impl BatchScheduler {
    /// The dense KV half of [`Self::try_adopt_cached_prefix`]: look the
    /// request up in the store and adopt the longest usable prefix. It starts
    /// with the store lookup, so every `None` it returns is a dense miss
    /// (a store miss or a declined store hit) for the metrics (issue #3294).
    ///
    /// Used by: `try_adopt_cached_prefix`.
    pub(super) fn adopt_dense_prefix(
        &mut self,
        store: &Arc<PromptCacheStore>,
        key: &PromptCacheKey<'_>,
        tokens: &[i32],
        require_whole_entry: bool,
    ) -> Option<(SequenceId, usize)> {
        let (entry, store_matched_len) = store.lookup_longest_prefix(key, tokens)?;
        // #124 step c: multimodal sharing requires the matched prefix to cover
        // the ENTIRE stored entry. A partial (e.g. APC block-clamped) match
        // could leave image/audio placeholder tokens in the suffix, which the
        // token-path suffix prefill would mis-handle. Decline here (falling
        // back to a cold prefill) before consuming anything; the entry stays
        // available for a later exact match.
        if require_whole_entry && store_matched_len < entry.tokens.len() {
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::ModeMismatch,
                None,
                store_matched_len,
            );
            return None;
        }
        // Whole-prompt hit (#1760), same rule as the snapshot branch: adopt
        // at most `len - 1` tokens so the prefill's forward over the last
        // prompt token lands on a cache that does not already hold it. The
        // dense truncate and the paged block floors below then install
        // exactly this many (or fewer) tokens.
        let matched_len = whole_prompt_reuse_cap(store_matched_len, tokens.len());
        if matched_len < store.min_prefix_tokens().max(1) {
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::PrefixTooShort,
                None,
                matched_len,
            );
            return None;
        }
        if matched_len < store_matched_len {
            tracing::debug!(
                matched = store_matched_len,
                adopted = matched_len,
                "prompt-cache hit covered the entire prompt; adopting all but the last token \
                 so prefill re-runs it to produce a sampling logit"
            );
        }
        // Length the adopted cache actually covers. The dense path truncates
        // to exactly `matched_len`; the paged paths floor to the pool block
        // boundary (#225), so they report their own value.
        let mut adopted_len = matched_len;

        // #227: pool-backed paged entries adopt by CLONE, leaving the stored
        // entry intact for concurrent same-prefix siblings and deeper future
        // matches. The one-shot take below destroyed the entry on first use;
        // combined with the #225 trim, a short partial match (even the
        // chat-template preamble) could gut a multi-thousand-token entry.
        enum PagedCloneOutcome {
            /// Adoptable clone built; the source entry stays in the store.
            Cloned(Box<DetachedPagedCacheSet>, usize),
            /// Cold prefill, entry untouched (below minimum, pin failure,
            /// or cross-backend). Carries the classified reject reason
            /// (issue #774) alongside the human-readable message.
            Decline(PromptCacheRejectReason, String),
            /// Dense entry, or a clone-ineligible paged shape (dense-compat
            /// handles, Turbo4, sliding-window): the consuming take path
            /// below can still adopt those.
            TakePath,
        }
        let backend_is_paged = matches!(self.decode_storage_backend, DecodeStorageBackend::Paged);
        let min_prefix = store.min_prefix_tokens().max(1);
        let cache_pool = self.engine.pool_mut();
        // `with_detached` itself returns `None` for a drained shell; the take
        // below then also yields `None` (cold prefill).
        let clone_attempt: Option<PagedCloneOutcome> = entry.with_detached(|set| match set {
            DetachedKvSet::Paged(paged) if backend_is_paged && paged.clone_eligible() => {
                let block_size = paged.layout().block_size.max(1);
                // Floor BOTH the partial and the whole-entry match to the
                // pool block boundary: a donated entry's length
                // (prompt + generated tokens) is almost never block-aligned,
                // and the clone shares whole blocks only. The caller
                // re-prefills everything past the adopted length, which
                // re-covers the dropped partial tail.
                let adoptable = (matched_len.min(paged.seq_len()) / block_size) * block_size;
                if adoptable < min_prefix {
                    return PagedCloneOutcome::Decline(
                        PromptCacheRejectReason::BlockBoundaryFloor,
                        format!(
                            "block-floored match {adoptable} below the minimum prefix {min_prefix}"
                        ),
                    );
                }
                match cache_pool.clone_detached_paged_prefix(paged, adoptable) {
                    Ok(clone) => PagedCloneOutcome::Cloned(Box::new(clone), adoptable),
                    Err(err) => {
                        PagedCloneOutcome::Decline(PromptCacheRejectReason::LayoutConstraints, err)
                    }
                }
            }
            // Paged entry under a dense decode backend: cross-backend
            // adoption is invalid; decline without touching the entry
            // (the old path took the set just to release it).
            DetachedKvSet::Paged(_) if !backend_is_paged => PagedCloneOutcome::Decline(
                PromptCacheRejectReason::ModeMismatch,
                "paged entry under a dense decode backend".into(),
            ),
            // Clone-ineligible paged shapes and dense entries.
            DetachedKvSet::Paged(_) | DetachedKvSet::Dense(_) => PagedCloneOutcome::TakePath,
        });
        match clone_attempt {
            Some(PagedCloneOutcome::Cloned(clone, adoptable)) => {
                adopted_len = adoptable;
                let adopt_result = self
                    .engine
                    .adopt_paged(*clone)
                    .map_err(|err| err.to_string());
                return self.finish_prompt_cache_adopt(adopt_result, adopted_len, tokens.len());
            }
            Some(PagedCloneOutcome::Decline(reject_reason, reason)) => {
                tracing::debug!(
                    "prompt-cache adopt: paged clone declined ({reason}); falling back to cold prefill (entry preserved)"
                );
                self.batch_observability.record_prompt_cache_reject(
                    reject_reason,
                    None,
                    matched_len,
                );
                return None;
            }
            Some(PagedCloneOutcome::TakePath) | None => {}
        }

        // Dense entries (and clone-ineligible paged shapes) keep the legacy
        // one-shot consume: their adoption genuinely moves buffers.
        // `take_detached` returns `None` if a racing lookup already consumed
        // this entry; the miss path is safe (fresh prefill).
        let detached = entry.take_detached()?;
        if detached.is_empty() {
            // A paged set drained on this path would leak its block pins via
            // `Drop`; release them explicitly so the pool budget stays honest.
            self.release_unused_detached(detached);
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::EmptySet,
                None,
                matched_len,
            );
            return None;
        }

        // Reject cross-backend adoption: the active decode backend determines
        // the KV shape the model worker can install. Adopting the wrong variant
        // would corrupt the sequence, so fall through to a cold prefill (and
        // release any paged pins we took).
        let backend_mismatch = matches!(
            (&detached, self.decode_storage_backend),
            (DetachedKvSet::Dense(_), DecodeStorageBackend::Paged)
                | (DetachedKvSet::Paged(_), DecodeStorageBackend::Dense)
        );
        if backend_mismatch {
            self.release_unused_detached(detached);
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::ModeMismatch,
                None,
                matched_len,
            );
            return None;
        }

        let adopt_result = match detached {
            DetachedKvSet::Dense(mut dense) => {
                // APC block-level partial adoption. When APC clamps
                // `matched_len` to a block boundary shorter than the cached
                // entry's full token length, the request diverged from the
                // cached prefix at the next block. Truncate the detached KV
                // state to exactly `matched_len` so the adopted cache covers
                // only the consistent prefix; the prefill loop then re-prefills
                // the divergent tail. When `matched_len == seq_len` this branch
                // is skipped, preserving the bit-exact full-prefix path.
                let detached_seq_len = dense.seq_len();
                if matched_len < detached_seq_len as usize {
                    let target = matched_len as i32;
                    if let Err(err) = dense.truncate_to(target) {
                        tracing::warn!(
                            "prompt-cache adopt: APC partial truncate to {target} failed ({err}); falling back to cold prefill"
                        );
                        self.batch_observability.record_prompt_cache_reject(
                            PromptCacheRejectReason::LayoutConstraints,
                            None,
                            matched_len,
                        );
                        return None;
                    }
                    tracing::debug!(
                        from = detached_seq_len,
                        to = target,
                        "prompt-cache adopt: APC partial adoption truncated detached cache to block boundary"
                    );
                }
                if let Err(reason) = self.validate_dense_detached_kv_modes(&dense) {
                    tracing::debug!(
                        "prompt-cache adopt: dense KV mode mismatch ({reason}); falling back to cold prefill"
                    );
                    self.batch_observability.record_prompt_cache_reject(
                        PromptCacheRejectReason::KvModeMismatch,
                        None,
                        matched_len,
                    );
                    return None;
                }
                self.engine.adopt(dense).map_err(|err| err.to_string())
            }
            DetachedKvSet::Paged(mut paged) => {
                // Paged partial prefix adoption (#225). An APC block-clamped
                // lookup (or a request that diverges inside the stored entry)
                // matches only `matched_len` of the set. Floor that to the
                // POOL block boundary: no partially filled tail block survives
                // the trim, so the suffix re-prefill starts on a fresh block
                // and never needs copy-on-write against a shared tail. A
                // whole-entry match skips the trim and stays bit-exact with
                // the pre-#225 path.
                let paged_seq_len = paged.seq_len();
                let block_size = paged.layout().block_size.max(1);
                let adoptable = if matched_len < paged_seq_len {
                    (matched_len / block_size) * block_size
                } else {
                    paged_seq_len
                };
                let min_prefix = store.min_prefix_tokens().max(1);
                if adoptable < min_prefix {
                    tracing::debug!(
                        from = paged_seq_len,
                        to = adoptable,
                        "prompt-cache adopt: block-floored paged match below the minimum prefix; releasing and falling back to cold prefill"
                    );
                    self.engine.pool_mut().release_detached_paged(paged);
                    self.batch_observability.record_prompt_cache_reject(
                        PromptCacheRejectReason::BlockBoundaryFloor,
                        None,
                        matched_len,
                    );
                    return None;
                }
                if adoptable < paged_seq_len {
                    if let Err(err) = self
                        .engine
                        .pool_mut()
                        .trim_detached_paged_to(&mut paged, adoptable)
                    {
                        tracing::warn!(
                            "prompt-cache adopt: paged partial trim to {adoptable} failed ({err}); falling back to cold prefill"
                        );
                        self.engine.pool_mut().release_detached_paged(paged);
                        self.batch_observability.record_prompt_cache_reject(
                            PromptCacheRejectReason::LayoutConstraints,
                            None,
                            matched_len,
                        );
                        return None;
                    }
                    tracing::debug!(
                        from = paged_seq_len,
                        to = adoptable,
                        "prompt-cache adopt: paged partial adoption trimmed detached set to the pool block boundary"
                    );
                    adopted_len = adoptable;
                }
                self.engine
                    .adopt_paged(paged)
                    .map_err(|err| err.to_string())
            }
        };

        self.finish_prompt_cache_adopt(adopt_result, adopted_len, tokens.len())
    }
}
