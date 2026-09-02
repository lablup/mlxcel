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

use super::*;

impl BatchScheduler {
    /// Maximum queued background warm-ups (issue #1144).
    ///
    /// Small on purpose. A warm-up is only useful until its conversation's next
    /// turn arrives, so a deep queue would mostly hold jobs whose turn already
    /// came and went, spending idle time on snapshots nobody will look up.
    const MAX_PENDING_WARMUPS: usize = 8;

    /// Whether the installed prompt-cache store is currently accepting
    /// lookups and inserts (scheduler-level gate).
    #[inline]
    pub(super) fn prompt_cache_active(&self) -> bool {
        self.prompt_cache
            .as_ref()
            .map(|s| s.is_enabled())
            .unwrap_or(false)
    }

    /// Build a [`PromptCacheKey`] bound to the per-request metadata the
    /// scheduler captured at enqueue time. Returns `None` when the request
    /// carried no [`PromptCacheRequestContext`] (e.g. non-chat endpoints).
    /// The regime tag for a sequence the model was told is `encoded_span`
    /// positions long, or `None` when this model's rotation does not depend on
    /// the sequence length.
    ///
    /// `encoded_span` is the value the prefill announced through
    /// [`mlxcel_core::prefill_span`], not the length of the prefix being stored
    /// or looked up: a 3000-token history-boundary snapshot cut out of a
    /// 5136-token prompt was rotated with that prompt's table, not with the
    /// table 3000 tokens would have selected on their own (#1358).
    pub(super) fn rope_regime_for(&self, encoded_span: usize) -> Option<u8> {
        self.model.rope_table_regime(encoded_span)
    }

    pub(super) fn compose_prompt_cache_key<'a>(
        ctx: &'a PromptCacheRequestContext,
        tokens: &'a [i32],
        rope_regime: Option<u8>,
    ) -> PromptCacheKey<'a> {
        // The digest is computed over the request's resolved image/audio bytes
        // in the route layer (#124 step b). For text-only requests it is
        // `MultimodalDigest::empty()`, so the key is byte-identical to the
        // pre-#124 path. Today multimodal requests still bypass adopt/donate at
        // the `is_multimodal` gate, so a non-empty digest only starts mattering
        // once that gate is lifted (#124 step c); folding it in now keeps the
        // bucket safe from a text↔image collision the moment sharing turns on.
        PromptCacheKey::new_full(
            ctx.model_id.as_str(),
            ctx.lora_id.as_deref(),
            ctx.template_sig.as_str(),
            Some(ctx.session_key.as_str()),
            ctx.mm_digest,
            tokens,
        )
        .with_rope_regime(rope_regime)
    }

    /// Attempt to adopt a cached prefix for a freshly tokenized request,
    /// returning the adopted `SequenceId` together with the matched-prefix
    /// length on success.
    ///
    /// The caller invokes this **before** [`Self::allocate_sequence_state`]
    /// so the adopted id becomes the sequence's canonical id and no
    /// seq_id rebinding dance is required. On any miss path the caller
    /// proceeds with a fresh allocation under a brand-new id.
    ///
    /// Gating (all of these yield `None`, which maps to a cold prefill):
    /// * feature disabled at config time,
    /// * request carried no [`PromptCacheRequestContext`] (non-chat endpoint),
    /// * store miss / match shorter than `min_prefix_tokens`,
    /// * race with another worker that already consumed the entry,
    /// * empty detached set (e.g. stored against an aborted seq),
    /// * backend mismatch (a `Dense` entry under the paged decode backend, or
    ///   a `Paged` entry under the dense backend — the entry's KV shape cannot
    ///   be installed into the active pool),
    /// * `require_whole_entry` is set (multimodal sharing) and the match is
    ///   shorter than the full stored entry (see below),
    /// * [`CachePool::adopt`] / [`CachePool::adopt_paged`] error (capacity,
    ///   layout mismatch, …).
    ///
    /// `require_whole_entry` is set for multimodal requests (#124 step c). It
    /// forces a whole-entry match so the prefilled suffix is guaranteed to be
    /// the newly-appended text turn: every image/audio placeholder token sits
    /// inside the matched prefix (a different media payload lands in a
    /// different digest bucket), so the suffix can safely run through the
    /// token path. Text-only requests pass `false` and keep accepting partial
    /// (APC block-aligned) matches.
    ///
    /// Both dense and paged entries are adopted in-place: dense via
    /// [`CachePool::adopt`], paged via [`CachePool::adopt_paged`] (which shares
    /// the cached prefix's refcounted pool blocks so the prefix is never
    /// re-prefilled — #121 sub-step b).
    pub(super) fn try_adopt_cached_prefix(
        &mut self,
        ctx: &PromptCacheRequestContext,
        tokens: &[i32],
        require_whole_entry: bool,
    ) -> Option<(SequenceId, usize)> {
        if !self.prompt_cache_active() {
            return None;
        }
        // Return to the pool any paged pins a prior store op queued (an insert
        // eviction, or a previous lookup's TTL / drained-shell sweep). The
        // lookup below may sweep more; those drain on the next store touch.
        self.drain_store_paged_releases();

        let store = self.prompt_cache.as_ref()?.clone();
        // The request's own prompt length picks its table, so it may only match
        // entries stored by a request on the same side of the boundary (#1358).
        let key = Self::compose_prompt_cache_key(ctx, tokens, self.rope_regime_for(tokens.len()));
        let snapshot_outcome = if self.model.supports_snapshot_reuse() {
            // The truncation capability is the model's own answer (#1145):
            // rotating-attention families can restore to the longest common
            // prefix while their sliding layers are unwrapped, and every
            // family whose recurrent state cannot be rewound returns false
            // here and keeps exact-prefix semantics.
            store.lookup_snapshot_outcome(&key, tokens, |snapshot, target_len| {
                self.model.snapshot_truncatable_to(snapshot, target_len)
            })
        } else {
            SnapshotLookupOutcome::NoCandidate
        };
        // A snapshot candidate that sits in this request's own session bucket
        // but is not a prefix of it is a structural miss, not an empty store
        // (issue #1147). Classify it so `/v1/cache/stats` can tell the two
        // apart; the multi-turn miss class in epic #1148 is otherwise
        // invisible. `NoCandidate` deliberately records nothing, so the
        // counter never fires for a cold store or a foreign session bucket.
        if let SnapshotLookupOutcome::Diverged(divergence) = &snapshot_outcome {
            tracing::debug!(
                common_prefix = divergence.common_prefix_len,
                stored_len = divergence.stored_len,
                request_len = tokens.len(),
                "prompt-cache snapshot candidate diverged from request"
            );
            self.batch_observability
                .record_prompt_cache_reject_detailed(
                    PromptCacheRejectReason::SnapshotDiverged,
                    None,
                    divergence.common_prefix_len,
                    Some(divergence.stored_len),
                );
        }
        if let SnapshotLookupOutcome::Hit {
            entry: snapshot_entry,
            matched_len,
        } = snapshot_outcome
        {
            let seq_id = match self.allocate_sequence_state() {
                Ok(id) => id,
                Err(err) => {
                    // Intentionally not a `record_prompt_cache_reject` site: slot
                    // exhaustion is a scheduler-capacity failure, not a cache-specific decline.
                    tracing::warn!("Cache pool allocation failed during snapshot restore: {err}");
                    return None;
                }
            };
            // A `matched_len` shorter than the stored entry means the store
            // adopted at the longest common prefix, which only happens when
            // the model agreed it could truncate there (#1145).
            let partial = matched_len < snapshot_entry.tokens.len();
            let restore = snapshot_entry.with_snapshot(|snapshot| {
                if partial {
                    self.model
                        .restore_sequence_state_truncated(seq_id, snapshot, matched_len)
                } else {
                    self.model.restore_sequence_state(seq_id, snapshot)
                }
            });
            match restore {
                Ok(()) => {
                    tracing::debug!(
                        seq_id = %seq_id,
                        matched = matched_len,
                        stored = snapshot_entry.tokens.len(),
                        total = tokens.len(),
                        partial,
                        "prompt-cache snapshot hit: restored {matched_len}/{} tokens",
                        tokens.len()
                    );
                    self.batch_observability
                        .record_prompt_cache_hit(matched_len);
                    self.batch_metrics
                        .record_prompt_cache_snapshot_hit(matched_len);
                    if let Some(ref store) = self.prompt_cache {
                        self.batch_metrics
                            .update_prompt_cache_gauges(store.bytes(), store.len());
                    }
                    if apc_trace_enabled() {
                        tracing::info!(
                            apc_event = "adopt",
                            seq_id = %seq_id,
                            matched_len = matched_len,
                            total_tokens = tokens.len(),
                            "prompt-cache adopt (snapshot)"
                        );
                    }
                    return Some((seq_id, matched_len));
                }
                Err(err) => {
                    tracing::warn!(
                        seq_id = %seq_id,
                        "prompt-cache snapshot restore failed ({err}); falling back to cold prefill"
                    );
                    self.batch_observability.record_prompt_cache_reject(
                        PromptCacheRejectReason::LayoutConstraints,
                        Some(seq_id.as_u64()),
                        matched_len,
                    );
                    self.release_sequence_caches(seq_id);
                    return None;
                }
            }
        }
        // Issue #1346, the adopt half of the same gate, keyed on the same
        // natural backend for the same reason. `donate_finished_sequence_cache`
        // never produces a KV entry for a model-owned family, so the lookup
        // below is a guaranteed miss whose only effect is to advance the KV
        // `lookups` counter and depress `hit_rate` for a family that structurally
        // cannot use that path. Bail first: this is a never-applies, not a miss,
        // and it mirrors `SnapshotLookupOutcome::NoCandidate`, which likewise
        // records nothing. It is also the guard that refuses any pre-existing
        // shadow entry a store carried in from before this fix. The snapshot
        // block above has already run, so `supports_snapshot_reuse()` families
        // keep their snapshot hits and their `snapshot_lookups`.
        if self.model.sequence_state_layout().backend == SequenceStateBackend::ModelOwned {
            return None;
        }
        let (entry, matched_len) = store.lookup_longest_prefix(&key, tokens)?;
        // #124 step c: multimodal sharing requires the matched prefix to cover
        // the ENTIRE stored entry. A partial (e.g. APC block-clamped) match
        // could leave image/audio placeholder tokens in the suffix, which the
        // token-path suffix prefill would mis-handle. Decline here (falling
        // back to a cold prefill) before consuming anything; the entry stays
        // available for a later exact match.
        if require_whole_entry && matched_len < entry.tokens.len() {
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::ModeMismatch,
                None,
                matched_len,
            );
            return None;
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
        let cache_pool = &mut self.cache_pool;
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
                    .cache_pool
                    .adopt_paged(&self.model as &dyn LanguageModel, *clone);
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
                self.cache_pool
                    .adopt(&self.model as &dyn LanguageModel, dense)
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
                    self.cache_pool.release_detached_paged(paged);
                    self.batch_observability.record_prompt_cache_reject(
                        PromptCacheRejectReason::BlockBoundaryFloor,
                        None,
                        matched_len,
                    );
                    return None;
                }
                if adoptable < paged_seq_len {
                    if let Err(err) = self
                        .cache_pool
                        .trim_detached_paged_to(&mut paged, adoptable)
                    {
                        tracing::warn!(
                            "prompt-cache adopt: paged partial trim to {adoptable} failed ({err}); falling back to cold prefill"
                        );
                        self.cache_pool.release_detached_paged(paged);
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
                self.cache_pool
                    .adopt_paged(&self.model as &dyn LanguageModel, paged)
            }
        };

        self.finish_prompt_cache_adopt(adopt_result, adopted_len, tokens.len())
    }

    /// Shared tail of [`Self::try_adopt_cached_prefix`]: record hit metrics
    /// and gauges on success, log and fall back to a cold prefill on failure.
    pub(super) fn finish_prompt_cache_adopt(
        &mut self,
        adopt_result: Result<SequenceId, String>,
        adopted_len: usize,
        total_tokens: usize,
    ) -> Option<(SequenceId, usize)> {
        match adopt_result {
            Ok(adopted_id) => {
                tracing::debug!(
                    seq_id = %adopted_id,
                    matched = adopted_len,
                    total = total_tokens,
                    "prompt-cache hit: adopted {adopted_len}/{total_tokens} tokens"
                );
                self.batch_observability
                    .record_prompt_cache_hit(adopted_len);
                // also increment BatchMetrics Prometheus counters.
                self.batch_metrics.record_prompt_cache_hit(adopted_len);
                // Update byte/entry gauges so /metrics reflects current state.
                if let Some(ref store) = self.prompt_cache {
                    self.batch_metrics
                        .update_prompt_cache_gauges(store.bytes(), store.len());
                }
                if apc_trace_enabled() {
                    tracing::info!(
                        apc_event = "adopt",
                        seq_id = %adopted_id,
                        matched_len = adopted_len,
                        total_tokens = total_tokens,
                        "prompt-cache adopt"
                    );
                }
                Some((adopted_id, adopted_len))
            }
            Err(err) => {
                // `adopt_paged` already releases paged pins on its error path;
                // `adopt` (dense) simply drops the buffers. Nothing to reclaim.
                tracing::debug!("prompt-cache adopt failed ({err}); falling back to cold prefill");
                self.batch_observability.record_prompt_cache_reject(
                    PromptCacheRejectReason::LayoutConstraints,
                    None,
                    adopted_len,
                );
                None
            }
        }
    }

    /// Release a detached set the adopt path decided not to use.
    ///
    /// A dense set just drops its MLX buffers. A paged set additionally owns
    /// refcount pins on physical pool blocks, which [`Drop`] cannot release on
    /// its own (it has no pool handle) — so route it through
    /// [`CachePool::release_detached_paged`] to return the pins and keep the
    /// block budget accurate.
    pub(super) fn release_unused_detached(&mut self, detached: DetachedKvSet) {
        match detached {
            DetachedKvSet::Dense(_) => {}
            DetachedKvSet::Paged(paged) => {
                self.cache_pool.release_detached_paged(paged);
            }
        }
    }

    /// Return to the pool any paged block pins the prompt-cache store queued
    /// for release. The store evicts (LRU / TTL) and declines (oversized)
    /// paged entries but cannot return their pool pins — it holds no
    /// `CachePool` handle — so it stashes them. The scheduler owns the pool,
    /// so it drains the queue here and routes each set through
    /// [`CachePool::release_detached_paged`]. Called from the store-touching
    /// paths so pins are reclaimed promptly during serving; a cheap no-op when
    /// the queue is empty (#122 sub-step a).
    pub(super) fn drain_store_paged_releases(&mut self) {
        let store = match self.prompt_cache.as_ref() {
            Some(s) if s.has_pending_paged_releases() => s.clone(),
            _ => return,
        };
        for paged in store.drain_pending_paged_releases() {
            self.cache_pool.release_detached_paged(paged);
        }
    }

    /// Turn the route's history-boundary render into the exact token vector
    /// the boundary snapshot will be keyed by, or clear it (issue #1143).
    ///
    /// Runs once per request at enqueue time, while `prompt_tokens` is final
    /// (post multimodal placeholder expansion). Three things happen here:
    ///
    /// * The history render is tokenized when the dispatch thread had no
    ///   tokenizer to do it (mirrors the `prompt_token_ids` fallback).
    /// * The result is clipped to the longest common prefix it shares with
    ///   `prompt_tokens`. This is what makes the vector a genuine prefix of the
    ///   state the boundary snapshot will describe. It also drops the tail
    ///   tokens that a BPE merge across the history/generation-prompt seam
    ///   would otherwise make unstable: a token that merged with the scaffold
    ///   is not a token the next turn will reproduce either.
    /// * Requests that cannot use a boundary snapshot are cleared to `None`
    ///   so the prefill path pays nothing for them: dense-KV families (which
    ///   already reuse through the longest-prefix trie), multimodal requests,
    ///   and prefixes below the store's `min_prefix_tokens`.
    pub(super) fn resolve_history_boundary(
        &self,
        ctx: &mut PromptCacheRequestContext,
        prompt_tokens: &[i32],
        is_multimodal: bool,
    ) {
        let history_prompt = ctx.history_prompt.take();
        if !self.model.supports_snapshot_reuse()
            || is_multimodal
            || crate::server::prompt_cache::boundary_snapshot_disabled()
        {
            ctx.history_prefix_tokens = None;
            return;
        }
        let history_tokens = match ctx.history_prefix_tokens.take() {
            Some(ids) => ids,
            None => match history_prompt {
                // No pre-tokenizer on the dispatch thread; encode here with the
                // same convention so the two vectors stay comparable.
                Some(text) => {
                    match crate::server::model_provider::tokenize_prompt_for_generation(
                        &self.tokenizer,
                        &text,
                    ) {
                        Ok(ids) => ids,
                        Err(err) => {
                            tracing::debug!("history-boundary tokenization failed: {err}");
                            return;
                        }
                    }
                }
                None => return,
            },
        };

        let min_prefix = self
            .prompt_cache
            .as_ref()
            .map(|s| s.min_prefix_tokens())
            .unwrap_or(0);
        ctx.history_prefix_tokens =
            history_boundary_len(&history_tokens, prompt_tokens, min_prefix)
                .map(|boundary| prompt_tokens[..boundary].to_vec());
    }

    /// Copy the model-owned state of `seq_id` and insert it into the
    /// prompt-cache snapshot bucket keyed by `tokens`.
    ///
    /// The caller guarantees `tokens` is exactly the token prefix the current
    /// model state describes: the snapshot is restored later by matching that
    /// vector against a future request's prompt, so a vector that does not
    /// match the state would install a wrong recurrent history.
    ///
    /// `origin` records which producer this snapshot came from. It labels the
    /// log and trace lines, and it scopes the store's session-chain supersede
    /// rule so the two producers do not delete each other's entries.
    ///
    /// Used by: `donate_finished_sequence_cache`,
    /// `capture_history_boundary_snapshot`
    /// `encoded_span` is the total sequence length the model was told about when
    /// this state was produced, which is what selects a position-dependent RoPE
    /// table. It is the whole prompt for a history-boundary snapshot even though
    /// `tokens` holds only its prefix, and prompt plus generated for a
    /// completion snapshot (#1358).
    pub(super) fn insert_model_state_snapshot(
        &mut self,
        seq_id: SequenceId,
        ctx: &PromptCacheRequestContext,
        tokens: Vec<i32>,
        origin: SnapshotOrigin,
        encoded_span: usize,
    ) {
        let store = match self.prompt_cache.as_ref() {
            Some(s) => s.clone(),
            None => return,
        };
        if tokens.len() < store.min_prefix_tokens() {
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::PrefixTooShort,
                Some(seq_id.as_u64()),
                tokens.len(),
            );
            return;
        }
        let snapshot = match self.model.snapshot_sequence_state(seq_id, tokens.len()) {
            Some(s) if !s.is_empty() => s,
            Some(_) => {
                tracing::debug!(
                    seq_id = %seq_id,
                    token_len = tokens.len(),
                    origin = ?origin,
                    "prompt-cache snapshot skipped: captured snapshot was empty"
                );
                self.batch_observability.record_prompt_cache_reject(
                    PromptCacheRejectReason::EmptySet,
                    Some(seq_id.as_u64()),
                    tokens.len(),
                );
                return;
            }
            None => {
                tracing::debug!(
                    seq_id = %seq_id,
                    token_len = tokens.len(),
                    origin = ?origin,
                    "prompt-cache snapshot skipped: no model-owned state for sequence"
                );
                self.batch_observability.record_prompt_cache_reject(
                    PromptCacheRejectReason::EmptySet,
                    Some(seq_id.as_u64()),
                    tokens.len(),
                );
                return;
            }
        };
        let entry = ModelSnapshotEntry::new(tokens, snapshot).with_origin(origin);
        let key_tokens = entry.tokens.clone();
        let key =
            Self::compose_prompt_cache_key(ctx, &key_tokens, self.rope_regime_for(encoded_span));
        match store.insert_snapshot(&key, entry) {
            Ok(()) => {
                tracing::debug!(
                    seq_id = %seq_id,
                    token_len = key_tokens.len(),
                    bytes = store.stats().snapshot_bytes,
                    origin = ?origin,
                    "prompt-cache snapshot inserted"
                );
                self.batch_observability.record_prompt_cache_insert();
                self.batch_metrics
                    .update_prompt_cache_gauges(store.bytes(), store.len());
                if apc_trace_enabled() {
                    tracing::info!(
                        apc_event = "store",
                        seq_id = %seq_id,
                        matched_len = key_tokens.len(),
                        origin = ?origin,
                        "prompt-cache store (snapshot)"
                    );
                }
            }
            Err(err) => {
                tracing::debug!(?origin, "prompt-cache snapshot insert skipped: {err}");
                self.batch_observability.record_prompt_cache_insert_reject();
                self.batch_observability.record_prompt_cache_reject(
                    PromptCacheRejectReason::from(&err),
                    Some(seq_id.as_u64()),
                    key_tokens.len(),
                );
            }
        }
    }

    /// Queue a background warm-up for the next turn's history prefix.
    ///
    /// Newest-wins: when the queue is full the oldest pending job is dropped,
    /// because it is the one most likely to have been overtaken by its own next
    /// turn. Dropping is always safe; the affected conversation falls back to
    /// the #1143 boundary snapshot, which is still a hit.
    pub(super) fn enqueue_prompt_cache_warmup(
        &mut self,
        tokens: Vec<i32>,
        ctx: PromptCacheRequestContext,
    ) {
        if !self.prompt_cache_active()
            || !self.model.supports_snapshot_reuse()
            || crate::server::prompt_cache::cache_warmup_disabled()
        {
            return;
        }
        while self.prompt_cache_warmups.len() >= Self::MAX_PENDING_WARMUPS {
            self.prompt_cache_warmups.pop_front();
            self.batch_observability.record_prompt_cache_warmup_skip();
        }
        self.prompt_cache_warmups.push_back((tokens, ctx));
    }

    /// Whether a queued warm-up may run on this tick.
    ///
    /// The bar is "the scheduler has nothing else to do at all": no live decode
    /// batch, no queued prefill, and no chunked prefill parked mid-prompt.
    /// Acceptance for issue #1144 is that warm-ups never delay foreground work,
    /// and the only way to guarantee that for an uninterruptible forward pass is
    /// to not start one while foreground work exists.
    pub(super) fn can_run_prompt_cache_warmup(&self) -> bool {
        !self.prompt_cache_warmups.is_empty()
            && self.active_batch.is_empty()
            && self.prefill_queue.is_empty()
            && self.chunked_prefill_seq.is_none()
            && self.prompt_cache_active()
    }

    /// Run one queued warm-up: restore the longest stored snapshot for the
    /// conversation, prefill only the delta up to the next turn's history
    /// prefix, and store the result (issue #1144).
    ///
    /// The delta is normally the assistant reply as the template re-renders it
    /// in history form, which is tens of tokens. Everything before it is
    /// restored, not recomputed, so the cost stays bounded no matter how long
    /// the conversation has grown.
    ///
    /// Every failure path is a silent return: the conversation keeps whatever
    /// snapshot it already had, and its next turn still hits the #1143 boundary
    /// entry. A warm-up must never turn into a client-visible error.
    pub(super) fn run_next_prompt_cache_warmup(&mut self) {
        let Some((tokens, ctx)) = self.prompt_cache_warmups.pop_front() else {
            return;
        };
        // Warm-ups compute KV for future requests, which snapshot the server
        // default; run them under it (#1439).
        self.ensure_lora_applied(None);
        let Some(store) = self.prompt_cache.clone() else {
            return;
        };
        if tokens.len() < store.min_prefix_tokens() {
            self.batch_observability.record_prompt_cache_warmup_skip();
            return;
        }

        // Start from the longest snapshot this conversation already has. With
        // no ancestor to restore there is nothing incremental to do: warming
        // would mean prefilling the whole history from cold, which is exactly
        // the foreground work this is supposed to avoid, spent speculatively.
        let key = Self::compose_prompt_cache_key(&ctx, &tokens, self.rope_regime_for(tokens.len()));
        let Some((entry, matched_len)) = store.lookup_snapshot_prefix(&key, &tokens) else {
            self.batch_observability.record_prompt_cache_warmup_skip();
            return;
        };
        if matched_len >= tokens.len() {
            // Already warm: the stored vector covers the whole target.
            self.batch_observability.record_prompt_cache_warmup_skip();
            return;
        }

        let seq_id = match self.allocate_sequence_state() {
            Ok(id) => id,
            Err(_) => {
                self.batch_observability.record_prompt_cache_warmup_skip();
                return;
            }
        };
        if entry
            .with_snapshot(|snapshot| self.model.restore_sequence_state(seq_id, snapshot))
            .is_err()
        {
            self.release_sequence_caches(seq_id);
            self.batch_observability.record_prompt_cache_warmup_skip();
            return;
        }

        // The forward below covers only `tokens[matched_len..]`, and for a
        // non-batching family the restored prefix is not reflected in the
        // scheduler's `KVCache::offset` either, so the pass geometry alone
        // understates the sequence. Announce the warm-up target's full length.
        let _span = mlxcel_core::prefill_span::announce(tokens.len() as i32);
        let delta: Vec<i32> = tokens[matched_len..].to_vec();
        let delta_len = delta.len() as i32;
        let input = mlxcel_core::from_slice_i32(&delta, &[1, delta_len]);
        let eval = {
            let Some(caches) = self.cache_pool.get_caches_mut(seq_id) else {
                self.release_sequence_caches(seq_id);
                self.batch_observability.record_prompt_cache_warmup_skip();
                return;
            };
            let last = self.model.forward_last_logits_with_sequence_id(
                &input,
                Some(seq_id),
                caches,
                None,
                delta.len().saturating_sub(1),
            );
            mlxcel_core::try_eval(&last).map_err(|e| e.to_string())
        };
        // A throw here is recorded against the backend health counter exactly
        // like a foreground eval, but it fails nothing: there is no client.
        if self.record_eval_outcome(eval).is_err() {
            self.release_sequence_caches(seq_id);
            self.batch_observability.record_prompt_cache_warmup_skip();
            return;
        }
        self.sync_sequence_storage(seq_id);

        // Stored as a Boundary snapshot because that is exactly what it is: the
        // next turn's history prefix. It therefore supersedes this turn's
        // boundary entry through the same per-producer chain, keeping the
        // conversation at one boundary snapshot while making that one cover the
        // previous reply as well.
        let warmed_len = tokens.len();
        self.insert_model_state_snapshot(
            seq_id,
            &ctx,
            tokens,
            SnapshotOrigin::Boundary,
            warmed_len,
        );
        self.release_sequence_caches(seq_id);
        mlxcel_core::clear_memory_cache();

        self.batch_observability.record_prompt_cache_warmup_run();
        tracing::debug!(
            restored = matched_len,
            delta = warmed_len - matched_len,
            warmed = warmed_len,
            "prompt-cache warm-up completed"
        );
    }

    /// Split this sequence's prefill at the history boundary and donate a
    /// snapshot of the state at that point (issue #1143).
    ///
    /// ## Why a second snapshot exists at all
    ///
    /// For snapshot-only families the only reuse path is an exact-prefix match
    /// against a stored token vector. The end-of-generation donate stores
    /// `prompt + generated`, and epic #1148 measured that vector failing to
    /// prefix the next turn on every family tested, for three independent
    /// reasons: templates append generation-prompt-only scaffolds, templates
    /// drop `<think>` blocks when re-rendering an assistant turn as history,
    /// and a sampled token sequence is not the canonical tokenization of its
    /// own text. All three live *after* the history boundary. The prefix
    /// rendered with `add_generation_prompt = false` carries none of them, so a
    /// snapshot keyed by the tokenization of that render is a prefix of every
    /// follow-up turn by construction.
    ///
    /// ## What this does
    ///
    /// Runs one extra forward over `prompt_tokens[prefill_start_offset..
    /// boundary]`, snapshots the model state there, inserts it, and advances
    /// `prefill_start_offset` so the caller's normal prefill continues with the
    /// remaining suffix. Total tokens forwarded are unchanged; the cost is one
    /// additional graph launch plus the state copy.
    ///
    /// Returns `Err(msg)` only when the extra forward's eval threw, in which
    /// case the caller must abort the sequence exactly as it does for its own
    /// prefill eval failures. Every other decline (feature off, dense-KV
    /// family, no boundary, boundary already covered by an adopted prefix)
    /// returns `Ok(())` and leaves the sequence untouched, so the request
    /// simply prefills the way it did before this issue.
    ///
    /// ## Where this does not run
    ///
    /// Only [`Self::execute_full_prefill`] and [`Self::start_chunked_prefill`]
    /// call this. A `BatchedCold` cohort of two or more rows goes through
    /// [`Self::run_padded_batched_prefill`], which forwards every row in one
    /// pass and has no per-row split point, so those rows take no boundary
    /// snapshot and their next turn misses. The same is true of the MTP
    /// speculative burst.
    ///
    /// Left as a gap rather than fixed by marking boundary-eligible rows
    /// non-cold in [`plan_prefill_cohorts`]: that would route essentially every
    /// snapshot-family chat row to sequential prefill and give up batched
    /// prefill for the whole family, which costs more than the missed reuse.
    /// In practice the overlap is narrow, since a cohort only forms from rows
    /// whose prompts are exactly equal in length on the families that report
    /// `supports_padded_prefill() == false`. Every single-row and error
    /// fallback in that function routes back through `execute_full_prefill`, so
    /// the capture still happens there.
    ///
    /// ## Not bit-exact, deliberately
    ///
    /// Two forwards do not reduce in the same order as one, so a split prefill
    /// can flip an early near-tie token relative to an unsplit one. Measured on
    /// qwen3.5-0.8b-4bit: greedy output matched for 168 characters and then
    /// differed by one word, staying coherent; each configuration is itself
    /// deterministic across repeats. This is the documented #203 / #325 / #326
    /// jitter class, and it is already the status quo on this path for two
    /// other reasons: `--prefill-chunk-size` splits prefills the same way, and
    /// any prompt-cache hit forwards only the suffix. An operator who needs the
    /// unsplit shape sets `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1` (see
    /// [`boundary_snapshot_disabled`]).
    pub(super) fn capture_history_boundary_snapshot(
        &mut self,
        seq: &mut SequenceInfo,
    ) -> Result<(), String> {
        // The segment forward below covers `prompt_tokens[start..boundary]`, a
        // strict prefix of the prompt, so it must resolve a whole-prompt RoPE
        // table from the prompt and not from its own span. Both callers already
        // announce; announcing here as well keeps the function correct on its
        // own if a third caller ever appears.
        let _span = self.announce_prefill_span(seq);
        if !self.model.supports_snapshot_reuse() || !self.prompt_cache_active() {
            return Ok(());
        }
        // Embedding-bearing rows feed `forward_with_embeddings` over the whole
        // prompt at once and cannot be split at a token index.
        if seq.vlm_embeddings.is_some() {
            return Ok(());
        }
        // `take` rather than clone: this is the only consumer of the vector, so
        // moving it out both avoids copying it and releases the map's copy for
        // the rest of the request. The context itself is cloned afterwards for
        // the key, which is cheap once the vector is gone.
        let Some(boundary_tokens) = self
            .prompt_cache_seq_ctx
            .get_mut(&seq.seq_id)
            .and_then(|c| c.history_prefix_tokens.take())
        else {
            return Ok(());
        };
        let boundary = boundary_tokens.len();
        // Nothing to capture when an adopted prefix already reaches the
        // boundary, and nothing to split when the boundary is the whole prompt
        // (the suffix forward must stay non-empty so the sampler still sees
        // fresh logits).
        if !boundary_capture_applies(boundary, seq.prefill_start_offset, seq.prompt_tokens.len()) {
            return Ok(());
        }
        // The vector was clipped to a common prefix at enqueue time; re-checking
        // here keeps the snapshot/state correspondence a local invariant of this
        // function rather than a cross-function assumption.
        if seq.prompt_tokens[..boundary] != boundary_tokens[..] {
            return Ok(());
        }

        let start = seq.prefill_start_offset;
        let _span = tracing::info_span!(
            "history_boundary_prefill",
            seq_id = %seq.seq_id,
            start,
            boundary,
            prompt_len = seq.prompt_tokens.len(),
        )
        .entered();

        // No NA-tile padding here: several snapshot-only families report
        // `supports_padded_prefill() == false` because padding tokens corrupt
        // their conv / SSM recurrent state, and a padded segment would make the
        // captured state describe more tokens than the key claims.
        let segment: Vec<i32> = seq.prompt_tokens[start..boundary].to_vec();
        let segment_len = segment.len() as i32;
        let input = mlxcel_core::from_slice_i32(&segment, &[1, segment_len]);
        let eval = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                // Treated as a decline rather than an error: the caller's own
                // prefill hits the same missing-cache condition immediately
                // after and reports it with its own message.
                None => return Ok(()),
            };
            // Evaluate only the final position, not the whole vocabulary
            // projection for every history token. Large-vocabulary models can
            // slice hidden states before their LM head through this hook.
            let last = self.model.forward_last_logits_with_sequence_id(
                &input,
                Some(seq.seq_id),
                caches,
                None,
                segment.len().saturating_sub(1),
            );
            mlxcel_core::try_eval(&last).map_err(|e| e.to_string())
        };
        self.record_eval_outcome(eval)?;
        self.sync_sequence_storage(seq.seq_id);

        let ctx = match self.prompt_cache_seq_ctx.get(&seq.seq_id) {
            Some(c) => c.clone(),
            None => return Ok(()),
        };
        let encoded_span = seq.prompt_tokens.len();
        self.insert_model_state_snapshot(
            seq.seq_id,
            &ctx,
            boundary_tokens,
            SnapshotOrigin::Boundary,
            encoded_span,
        );

        // Return the segment's intermediates to the allocator before the
        // caller's suffix forward runs, the same way `execute_full_prefill`
        // clears after its own forward. Without this the segment's buffers stay
        // resident through the rest of the prefill.
        mlxcel_core::clear_memory_cache();

        // Count the segment as prefill work so `total_prefill_tokens` still
        // sums to the prompt: the caller records only the suffix it runs.
        // Deliberately NOT `record_prefill_chunk()`: that counter is the
        // dispatch proof for the issue #908 / #1011 mixed-step work, and a
        // boundary segment is not a chunk the chunked-prefill loop scheduled.
        self.batch_observability
            .record_prefill_tokens(boundary - start);
        seq.prefill_start_offset = boundary;
        Ok(())
    }

    /// Donate a finished sequence's KV cache back to the store so future
    /// requests sharing a prefix can adopt it.
    ///
    /// The caller must invoke this **before** calling
    /// [`Self::release_sequence_caches`] — once release runs the underlying
    /// tensors are gone. Safe to call unconditionally; all the gating checks
    /// (feature enabled, healthy finish, context present, detachable backend)
    /// live inside this method so the caller can keep its hot-path code
    /// simple.
    ///
    /// Both dense and paged sequences are donated: dense via
    /// [`CachePool::detach`] (→ [`DetachedKvSet::Dense`]) and paged via
    /// [`CachePool::detach_paged`] (→ [`DetachedKvSet::Paged`], which pins the
    /// prefix's physical pool blocks so a later `adopt_paged` can share them).
    /// `ModelOwned` sequences carry no detachable cross-request KV; families
    /// that opt into snapshot reuse donate a copied model-owned snapshot, while
    /// the rest are skipped.
    pub(super) fn donate_finished_sequence_cache(
        &mut self,
        seq_id: SequenceId,
        prompt_tokens: &[i32],
        generated_tokens: &[i32],
        healthy_finish: bool,
    ) {
        if !healthy_finish {
            return;
        }
        if !self.prompt_cache_active() {
            return;
        }
        // Remove the context regardless of whether the donate-back succeeds
        // so the map doesn't grow unbounded across sequences that never
        // qualified for a donate-back.
        let ctx = match self.prompt_cache_seq_ctx.remove(&seq_id) {
            Some(c) => c,
            None => return,
        };

        // Tokens stored against both KV entries and recurrent snapshots are
        // the full prompt + generated tail, so the next turn can restore the
        // exact previous conversation prefix and prefill only the appended
        // user turn.
        let mut tokens = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
        tokens.extend_from_slice(prompt_tokens);
        tokens.extend_from_slice(generated_tokens);

        // A generation that crossed a position-selected RoPE table boundary
        // mid-decode left this cache holding keys from both tables: the prompt
        // was rotated under the table its own length selected, and the tail
        // under the other one. There is no single regime that describes it, so
        // it can never be donated (#1358). Only Phi-3 / Phi-4 LongRoPE answers
        // this hook at all; every other family reports `None` on both sides and
        // takes the same path it always did.
        let prompt_regime = self.model.rope_table_regime(prompt_tokens.len());
        if prompt_regime.is_some() && self.model.rope_table_regime(tokens.len()) != prompt_regime {
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::RopeRegimeMismatch,
                Some(seq_id.as_u64()),
                tokens.len(),
            );
            return;
        }

        // Families with model-owned recurrent or linear-attention state opt
        // into exact-prefix snapshots explicitly. Check this capability before
        // consulting the allocated storage backend: under the paged decode
        // override these families may still carry a shadow `PagedKvCache`
        // placeholder even though the real state lives in
        // `ModelOwnedSequenceState` and cannot be detached as KV blocks.
        if self.model.supports_snapshot_reuse() {
            let encoded_span = tokens.len();
            self.insert_model_state_snapshot(
                seq_id,
                &ctx,
                tokens,
                SnapshotOrigin::Completion,
                encoded_span,
            );
            return;
        }

        // Issue #1346. Gate on the model's NATURAL sequence-state backend, not
        // the one this sequence was allocated on. Under
        // `--decode-storage-backend paged` (the `auto` default on a batching
        // server) `sequence_state_layout_override` allocates EVERY family on
        // `PagedKvCache`, model-owned ones included, purely so
        // `sync_paged_state_with_lengths` can keep a shadow block table for
        // accounting. Their real K/V never leaves `ModelOwnedSequenceState`,
        // so the shadow table indexes pool pages nothing ever wrote. The
        // allocated-backend check below reads `PagedKvCache` for exactly those
        // sequences and lets them through; donating one and adopting it next
        // turn skips prefill for tokens whose K/V does not exist, and the model
        // answers turn 2 from the appended suffix alone.
        //
        // `self.model.sequence_state_layout()` is the model's own answer and is
        // never rewritten by the override (the override is built from server
        // config alone, and `CachePool::allocate_with_layout` consults the
        // model's layout separately as `natural_backend` for the same reason).
        // `handoff_supported` already gates the disaggregated KV handoff on the
        // same predicate for the same underlying fact (#708).
        if self.model.sequence_state_layout().backend == SequenceStateBackend::ModelOwned {
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::ModelOwnedState,
                Some(seq_id.as_u64()),
                tokens.len(),
            );
            return;
        }

        let backend = self
            .cache_pool
            .get_mut(seq_id)
            .map(|s| s.backend)
            .unwrap_or(SequenceStateBackend::ModelOwned);

        // Other `ModelOwned` families carry no detachable cross-request KV.
        if backend == SequenceStateBackend::ModelOwned {
            return;
        }

        let store = match self.prompt_cache.as_ref() {
            Some(s) => s.clone(),
            None => return,
        };

        // Detach into the backend-appropriate variant.
        let kv_set: DetachedKvSet = match backend {
            SequenceStateBackend::DenseKvCache => match self.cache_pool.detach(seq_id) {
                Some(d) => DetachedKvSet::Dense(d),
                None => return,
            },
            SequenceStateBackend::PagedKvCache => {
                // `detach_paged` pins every physical prefix block, and those
                // pins can only be returned through `release_detached_paged`
                // (the set's `Drop` cannot). If the store would reject the
                // entry for being shorter than `min_prefix_tokens`, screen the
                // length BEFORE detaching so we never take pins we'd have to
                // immediately release. The dense path needs no such screen — a
                // rejected dense entry just drops its buffers.
                if tokens.len() < store.min_prefix_tokens() {
                    self.batch_observability.record_prompt_cache_reject(
                        PromptCacheRejectReason::PrefixTooShort,
                        Some(seq_id.as_u64()),
                        tokens.len(),
                    );
                    return;
                }
                match self.cache_pool.detach_paged(seq_id) {
                    Some(p) => DetachedKvSet::Paged(p),
                    None => return,
                }
            }
            SequenceStateBackend::ModelOwned => return,
        };

        if kv_set.is_empty() {
            // Nothing to cache: aborted before any prefill completed, or the
            // model never populated the KV state. Release any paged pins we
            // took so the pool budget stays honest.
            self.release_unused_detached(kv_set);
            self.batch_observability.record_prompt_cache_reject(
                PromptCacheRejectReason::EmptySet,
                Some(seq_id.as_u64()),
                tokens.len(),
            );
            return;
        }

        // The `CacheEntry` takes ownership of `tokens` and the key borrows
        // from the same buffer. Build the entry first, then form the key
        // against `entry.tokens` so both reference the same contiguous
        // allocation without copying the vector.
        let entry = CacheEntry::new(tokens, kv_set);
        let key_tokens = entry.tokens.clone();
        let key = Self::compose_prompt_cache_key(
            &ctx,
            &key_tokens,
            self.rope_regime_for(key_tokens.len()),
        );
        match store.insert(&key, entry) {
            Ok(()) => {
                self.batch_observability.record_prompt_cache_insert();
                // refresh byte/entry gauges after a successful insert.
                self.batch_metrics
                    .update_prompt_cache_gauges(store.bytes(), store.len());
                if apc_trace_enabled() {
                    tracing::info!(
                        apc_event = "store",
                        seq_id = %seq_id,
                        matched_len = key_tokens.len(),
                        "prompt-cache store"
                    );
                }
            }
            Err(err) => {
                // Oversized / disabled / prefix-too-short — `insert` declines
                // the entry. For dense that frees the buffers; for a paged entry
                // the store stashes its block pins on its pending-release queue
                // (it has no `CachePool` handle), which the
                // `drain_store_paged_releases()` below returns to the pool
                // (#122 sub-step a).
                tracing::debug!(
                    seq_id = %seq_id,
                    "prompt-cache donate-back skipped: {err:?}"
                );
                self.batch_observability.record_prompt_cache_insert_reject();
                self.batch_observability.record_prompt_cache_reject(
                    PromptCacheRejectReason::from(&err),
                    Some(seq_id.as_u64()),
                    key_tokens.len(),
                );
            }
        }
        // Return to the pool any paged pins this insert's eviction / rejection
        // paths queued: byte/entry-budget `enforce_caps` (LRU), idempotent
        // replacement removal, or an oversized / disabled decline.
        self.drain_store_paged_releases();
    }
}
