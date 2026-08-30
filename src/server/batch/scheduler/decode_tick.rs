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
    /// Whether a bounded sequence must stop now because its next token would
    /// not fit the KV window with context shifting disabled (#1472).
    ///
    /// Token-count based (`prompt + generated + 1 >= bound`), mirroring
    /// upstream's `slot.prompt.n_tokens() + 1 >= slot.n_ctx`: with shifting
    /// disabled nothing ever trims, so the token count IS the live KV window,
    /// including for the paged and Turbo cache modes whose trim operation is
    /// a recorded no-op. VLM sequences are exempt, as upstream exempts
    /// multimodal from the context machinery (their KV length is the
    /// embedding count, not the text token count).
    pub(super) fn context_bound_stop_due(
        seq: &SequenceInfo,
        max_kv_size: Option<usize>,
        context_shift: bool,
    ) -> bool {
        !context_shift
            && seq.vlm_embeddings.is_none()
            && max_kv_size
                .is_some_and(|max| seq.prompt_tokens.len() + seq.generated_tokens.len() + 1 >= max)
    }

    // ------------------------------------------------------------------
    // Preemptive eviction
    // ------------------------------------------------------------------

    /// Attempt to evict one sequence from the active batch to make room
    /// for a higher-priority queued request.
    ///
    /// Returns `true` if eviction succeeded (a slot is now free).
    ///
    /// **Streaming caveat:** Tokens already streamed to the client via
    /// `GenerateEvent::Token` are not recalled. When the evicted sequence
    /// is re-prefilled, duplicate tokens may be streamed. This is
    /// acceptable for preemptive scheduling (the client sees a retry)
    /// and is consistent with vLLM's eviction semantics.
    pub(super) fn try_evict_for_preemption(&mut self) -> bool {
        let victim_id = match self.select_eviction_victim() {
            Some(id) => id,
            None => return false,
        };

        if let Some(mut victim) = self.active_batch.remove(victim_id) {
            tracing::info!(
                "Preempting sequence {} (priority={:?}, {} tokens generated)",
                victim.seq_id,
                victim.priority,
                victim.generated_tokens.len()
            );

            // follow-up: when the victim is a VL request the
            // text model holds a per-sequence MRoPE entry under the old
            // seq id. `release_sequence_caches` below would drop it, but
            // `prepare_request_vlm_embeddings` does NOT re-run on
            // re-prefill so the entry would never be rebuilt under the
            // new id. Take the entry out *before* the release so we can
            // rebind it under the freshly allocated id below.
            //
            // For non-Qwen-VL models / text-only requests this returns
            // an empty snapshot and the rebind is a no-op.
            let mrope_snapshot = self.model.take_qwen_vl_mrope_entry(victim.seq_id);

            // same lifecycle invariant for Gemma 4 E2B/E4B
            // `per_layer_inputs`. The tensor is projected exactly once
            // by `prepare_request_vlm_embeddings` at enqueue time and
            // is consumed at prefill time; preemption-and-reallocate
            // would otherwise drop it and the re-prefill would observe
            // `per_layer_inputs == None` for an E2B/E4B request. Take
            // it out before `release_sequence_caches` drains the map.
            let pli_snapshot = self.model.take_gemma4_per_layer_inputs_entry(victim.seq_id);

            // Issue #85: same for Gemma 3n VLM `per_layer_inputs`.
            // Without this round trip the re-prefill would panic in
            // `Gemma3nVLModel::forward_with_embeddings_and_sequence_id`
            // (per_layer_inputs missing for this sequence).
            let pli3n_snapshot = self
                .model
                .take_gemma3n_per_layer_inputs_entry(victim.seq_id);

            // Drop the victim's prompt-cache context. Preemption reallocates
            // the sequence under a fresh `SequenceId` below, and the context
            // map is keyed by the old one — so an entry left here is
            // unreachable for the rest of the process's life. The leak is
            // pre-#1143 (every preemption of a chat request leaked one entry
            // of strings), but that issue put a token vector in the context
            // and made each leaked entry proportional to the conversation, so
            // it is cleared here now.
            //
            // Deliberately a drop and not a re-key: preemption already
            // discards the adopted prefix and re-prefills the victim cold (see
            // the reset below), so declining its donate-back keeps this path
            // exactly as consistent as it was.
            self.prompt_cache_seq_ctx.remove(&victim.seq_id);

            // Release its KV cache
            self.release_sequence_caches(victim.seq_id);

            // Reset the sequence for re-prefill: clear generated tokens,
            // reset decode state, and re-allocate a cache slot.
            //
            // Preemption discards the adopted prefix cache as well — the
            // victim must re-prefill from scratch to stay consistent with
            // the fresh `allocate_sequence_state` that follows.
            victim.generated_tokens.clear();
            victim.generated_text.clear();
            victim.prefill_offset = 0;
            victim.prefill_start_offset = 0;
            victim.already_cached_tokens = 0;
            victim.decode_state = StreamingDecodeState::new(&self.tokenizer, &victim.prompt_tokens);
            // The matcher's held tail and emitted-byte count describe the decode
            // being discarded, so they reset with it (issue #1466). The
            // request's stop strings survive: re-prefill must still honor them.
            victim.stop_matcher.reset();
            victim.token_history.clear();
            victim.merged_eos.clear();

            // Allocate a fresh cache slot
            match self.allocate_sequence_state() {
                Ok(new_id) => {
                    victim.seq_id = new_id;
                    // Re-install the previously-saved MRoPE entry under
                    // the new seq id so re-prefill resolves the same
                    // per-row delta the original prefill computed
                    // (follow-up).
                    self.model
                        .install_qwen_vl_mrope_entry(new_id, mrope_snapshot);
                    // same for Gemma 4 `per_layer_inputs`.
                    // The tensor is reused unchanged across re-prefill
                    // because both depend only on the request's
                    // input_ids (no decode-time updates).
                    self.model
                        .install_gemma4_per_layer_inputs_entry(new_id, pli_snapshot);
                    // Issue #85: same for Gemma 3n `per_layer_inputs`.
                    self.model
                        .install_gemma3n_per_layer_inputs_entry(new_id, pli3n_snapshot);
                    if let Err(err) = victim.state.transition_to(SequenceState::Queued) {
                        tracing::error!("Eviction state transition error: {err}");
                        self.release_sequence_caches(new_id);
                        let _ = victim
                            .response_tx
                            .send(GenerateEvent::Error(format!("Eviction state error: {err}")));
                        return true; // Slot is still freed
                    }
                }
                Err(err) => {
                    tracing::warn!("Re-allocation failed for evicted sequence: {err}");
                    let _ = victim.response_tx.send(GenerateEvent::Error(format!(
                        "Preemption re-queue failed: {err}"
                    )));
                    // The snapshots are dropped here; the request is
                    // about to error out so the entries have no further
                    // consumer.
                    drop(mrope_snapshot);
                    drop(pli_snapshot);
                    return true; // Slot is still freed
                }
            }

            // Re-queue the evicted sequence (it will re-prefill when admitted)
            if let Err(rejected) = self.prefill_queue.enqueue(victim) {
                self.release_sequence_caches(rejected.seq_id);
                let _ = rejected.response_tx.send(GenerateEvent::Error(
                    "Preemption re-queue failed: prefill queue full".to_string(),
                ));
            }

            self.batch_metrics.record_preemption();
            true
        } else {
            false
        }
    }

    /// Select the eviction victim based on the configured policy.
    ///
    /// The policy itself lives in [`select_eviction_victim_from`], which is its
    /// only implementation and the entry point the unit tests use; this method
    /// just supplies the batch and the configured policy.
    ///
    /// follow-up: sequences with an attached structured-output
    /// constraint are excluded from the candidate set. Preemption resets
    /// `generated_tokens`, the streaming decoder, and the KV cache, but the
    /// `llguidance` matcher carries grammar progress that cannot be safely
    /// rewound — re-prefill would advance the matcher from a state that
    /// reflects the discarded tokens, producing either an empty mask error
    /// or silent grammar mis-advance. Skipping these sequences trades a
    /// rare scheduling stall for correctness; if no other candidate is
    /// available, `try_evict_for_preemption` falls through to its existing
    /// "no candidate" path and the new request stays queued.
    pub(super) fn select_eviction_victim(&self) -> Option<SequenceId> {
        select_eviction_victim_from(self.active_batch.iter_sequences(), self.preemption_policy)
    }

    // ------------------------------------------------------------------
    // Decode execution (batched when B > 1, sequential fallback otherwise)
    // ------------------------------------------------------------------

    /// Run one decode step for the active sequences.
    pub(super) fn execute_decode_step(&mut self, seq_ids: &[SequenceId]) {
        // Filter-to-empty guard: a zero-sized decode step is a no-op, not a
        // failure. The observability counter already reflects length 0 for
        // caller-side traceability, so we still record it, then skip the
        // dispatch entirely. This matches the null-guard pattern upstream
        // `mlx-lm` added to `BatchKVCache.filter` when the filtered index
        // list is empty.
        if seq_ids.is_empty() {
            self.batch_observability.record_decode_step(0);
            return;
        }

        let _span = tracing::info_span!("decode_step", batch_size = seq_ids.len(),).entered();
        self.batch_observability.record_decode_step(seq_ids.len());

        // Lookahead async_eval pipeline (issue #632). Eligible batches overlap
        // the next forward with the current tick's host bookkeeping; anything
        // outside the narrow eligibility window (see `lookahead_params`) runs
        // the untouched synchronous path. `run_decode_tick` owns the state
        // machine and always leaves the caches in the synchronous-decode
        // invariant on any teardown, so the fallback is bit-exact.
        self.run_decode_tick(seq_ids);
    }

    /// Raw synchronous decode dispatch for `seq_ids` (no observability
    /// recording; the caller already counted the step). B=1 and non-batching
    /// models take the per-sequence path; larger batches take the batched
    /// forward. This is the exact pre-#632 behavior and the pipeline's
    /// guaranteed fallback.
    pub(super) fn dispatch_sync_decode(&mut self, seq_ids: &[SequenceId]) {
        if seq_ids.len() <= 1 || !self.model.supports_batching() {
            for &seq_id in seq_ids {
                self.decode_single_step(seq_id);
            }
            return;
        }
        self.execute_batched_decode(seq_ids);
    }

    /// Drive one decode tick through the lookahead pipeline state machine.
    ///
    /// States:
    /// - A prebuilt lookahead for the identical id set and still-safe
    ///   conditions -> steady pipelined commit + re-prime.
    /// - A prebuilt lookahead that is stale (id set changed) or now unsafe ->
    ///   discard (trim + drop), run synchronously, then re-prime if eligible.
    /// - No prebuilt lookahead -> run synchronously, then prime if eligible
    ///   (bootstrap).
    pub(super) fn run_decode_tick(&mut self, seq_ids: &[SequenceId]) {
        let params = self.lookahead_params(seq_ids);

        match self.decode_lookahead.take() {
            Some(la) if la.ids == seq_ids && params.is_some() && self.lookahead_safe() => {
                self.pipelined_steady_decode(la, seq_ids, &params.unwrap());
            }
            Some(la) => {
                // Stale id set or no longer eligible/safe: no step n+1 prime was
                // issued, so undo the one speculative KV position and fall back
                // to a clean sync step.
                self.apply_lookahead_trim(&la.ids, lookahead_teardown_positions(false));
                drop(la);
                self.dispatch_sync_decode(seq_ids);
                self.maybe_prime_lookahead(seq_ids);
            }
            None => {
                self.dispatch_sync_decode(seq_ids);
                self.maybe_prime_lookahead(seq_ids);
            }
        }
    }

    /// Batched decode-storage context for the active backend. Shared by the
    /// synchronous batched decode and the lookahead prime so both drive the
    /// identical dense / native-paged execution path.
    pub(super) fn decode_batch_context(&self) -> DecodeBatchContext {
        match self.decode_storage_backend {
            DecodeStorageBackend::Auto | DecodeStorageBackend::Dense => {
                debug_assert_ne!(
                    self.decode_storage_backend,
                    DecodeStorageBackend::Auto,
                    "scheduler should normalize decode storage backend before decode dispatch"
                );
                DecodeBatchContext::dense()
            }
            DecodeStorageBackend::Paged => DecodeBatchContext {
                storage_backend: CoreDecodeStorageBackend::Paged,
                paged_block_size: DEFAULT_PAGED_BLOCK_SIZE as i32,
                use_native_paged_kernel: true,
            },
        }
    }

    /// Whether the active decode batch is eligible for the lookahead pipeline
    /// this tick, returning the shared fused sampling params on success. A
    /// `None` return routes the tick to the synchronous path. The gate is
    /// deliberately narrow: it reuses the batched-fused predicate (which
    /// already rejects penalties/token-history, token bias, structured-output
    /// masks, thinking-budget overrides, and per-token logprobs) and further
    /// requires a trimmable KV tail (dense or pool-backed paged; model-owned
    /// SSM / hybrid / mixed-cache backends stay synchronous), no
    /// `--max-kv-size`, no speculative dispatch, and `MLXCEL_FORCE_SYNC` unset.
    pub(super) fn lookahead_params(&self, seq_ids: &[SequenceId]) -> Option<FusedSampleParams> {
        if self.lookahead_force_sync {
            return None;
        }
        // Speculative decoding drives its own decode loop.
        if self.should_dispatch_speculative() {
            return None;
        }
        // --max-kv-size trims the live window mid-decode (trim_front); keep
        // those runs synchronous so the speculative +1 accounting stays simple.
        if self.max_kv_size.is_some() {
            return None;
        }
        // The batched fused gate rejects every per-row feature the device
        // feedback path cannot honor; reusing it keeps pipeline sampling
        // bit-identical to the fast path it accelerates.
        let params = self.batched_decode_fused_params(seq_ids)?;
        for &seq_id in seq_ids {
            let seq = self.active_batch.get(seq_id)?;
            // Loop detection needs a post-commit host scan the steady path
            // skips (off by default, so no common-case cost).
            if seq.sampling.loop_detection.is_enabled() {
                return None;
            }
            // Dense and pool-backed paged sequences both have a trimmable KV
            // tail (dense via KVCache::trim, paged via the pool rewind API in
            // apply_lookahead_trim). Model-owned families (SSM / hybrid /
            // mixed-cache) carry no such tail, so they stay synchronous.
            match self.cache_pool.get(seq_id) {
                Some(set)
                    if matches!(
                        set.backend,
                        SequenceStateBackend::DenseKvCache | SequenceStateBackend::PagedKvCache
                    ) => {}
                _ => return None,
            }
        }
        Some(params)
    }

    /// Conditions under which priming the next forward is safe: the next tick
    /// will decode this identical id set. False on a pending admission
    /// (queue non-empty), a chunked-prefill interleave, or a pending
    /// preemption, each of which mutates batch membership next tick.
    pub(super) fn lookahead_safe(&self) -> bool {
        lookahead_pipeline_safe(
            self.prefill_queue.is_empty(),
            self.chunked_prefill_seq.is_some(),
            self.should_preempt(),
        )
    }

    /// Trim the one speculative KV position the prime forward appended from
    /// each sequence, restoring the synchronous-decode invariant (the last
    /// committed token is not yet in the KV cache). Called on every pipeline
    /// teardown before the synchronous path, a completion, or a prompt-cache
    /// donation runs, so slot reuse and detach always see clean caches.
    ///
    /// Pool-backed paged sequences rewind through the pool block table (one
    /// token per layer, releasing any tail block); dense (and dense-natural
    /// paged mirror) sequences trim the dense KV tail and re-mirror the shorter
    /// length into the paged bookkeeping state.
    ///
    /// `positions` is the number of speculative appends to unwind: `1` for a
    /// teardown before the step-n+1 prime forward has run (admission,
    /// preemption, stale id set, cancellation seen in `finalize_completed`),
    /// `2` for the steady-tick teardown that already issued the step-n+1 prime
    /// (both the step-n and step-n+1 appends).
    pub(super) fn apply_lookahead_trim(&mut self, ids: &[SequenceId], positions: usize) {
        if positions == 0 {
            return;
        }
        let num_layers = self.model.num_layers();
        for &seq_id in ids {
            let paged_backed = self
                .cache_pool
                .get(seq_id)
                .map(|s| s.caches.iter().any(|c| c.is_paged_backed()))
                .unwrap_or(false);
            if paged_backed {
                for layer in 0..num_layers {
                    // A failed rewind silently leaks the speculative KV
                    // position(s), which would corrupt a later donation of this
                    // sequence's cache; surface it so the leak is diagnosable.
                    if let Err(err) = self
                        .cache_pool
                        .rewind_paged_tokens(seq_id, layer, positions)
                    {
                        tracing::warn!(
                            seq_id = %seq_id,
                            layer,
                            positions,
                            "lookahead teardown: paged rewind failed, speculative KV \
                             position may leak: {err}"
                        );
                    }
                }
            } else if let Some(caches) = self.cache_pool.get_caches_mut(seq_id) {
                let want = positions as i32;
                for (layer, cache) in caches.iter_mut().enumerate() {
                    // KVCache::trim clamps to the live window and returns the
                    // count actually removed; a short trim means a speculative
                    // position was not unwound (e.g. an unexpectedly short cache),
                    // which would desync the KV against generated_tokens.
                    let trimmed = cache.trim(want);
                    if trimmed != want {
                        tracing::warn!(
                            seq_id = %seq_id,
                            layer,
                            requested = want,
                            trimmed,
                            "lookahead teardown: dense trim removed fewer positions \
                             than requested, KV may be out of sync"
                        );
                    }
                }
                // Re-mirror the shorter dense length into any paged bookkeeping
                // (no-op for a pure dense pool).
                self.sync_sequence_storage(seq_id);
            }
        }
    }

    /// Tear down any live lookahead: trim the speculative KV position from each
    /// of its sequences and drop the prebuilt tokens. Idempotent no-op when the
    /// pipeline is idle. Invoked before admission / preemption (`run`) and
    /// before completion / cancellation donation (`finalize_completed`).
    pub(super) fn discard_lookahead(&mut self) {
        if let Some(la) = self.decode_lookahead.take() {
            // A stored lookahead carries exactly one speculative append per
            // sequence (the prime forward that produced its tokens); no step
            // n+1 prime has been issued on this teardown path.
            //
            // Unlike the steady finishing path, no pre-trim eval is needed
            // here even if that prime is still in flight: KVCache::trim only
            // adjusts a host-tracked offset, and all decode work runs on the
            // single generation stream, so the lazy slice the trim enqueues is
            // dependency-ordered after the append. The finishing path's eval
            // is defensive, not required for safety.
            self.apply_lookahead_trim(&la.ids, lookahead_teardown_positions(false));
        }
    }

    /// Prime the next forward for `seq_ids` after a synchronous step (pipeline
    /// bootstrap). Reads each sequence's last committed token from host state,
    /// builds the `[B, 1]` input, and schedules the forward + fused sample.
    /// No-op unless eligible, safe, and every sequence is still live.
    pub(super) fn maybe_prime_lookahead(&mut self, seq_ids: &[SequenceId]) {
        let Some(params) = self.lookahead_params(seq_ids) else {
            return;
        };
        if !self.lookahead_safe() {
            return;
        }
        // A sequence that just finished (EOS / length) leaves the batch next
        // tick; do not prime across a membership change.
        let mut last_tokens: Vec<i32> = Vec::with_capacity(seq_ids.len());
        for &seq_id in seq_ids {
            match self.active_batch.get(seq_id) {
                Some(seq) if !seq.state.is_finished() => {
                    last_tokens.push(*seq.generated_tokens.last().unwrap_or(&0));
                }
                _ => return,
            }
        }
        let input = mlxcel_core::from_slice_i32(&last_tokens, &[seq_ids.len() as i32, 1]);
        self.decode_lookahead = self.prime_lookahead_with_input(seq_ids, &input, &params);
    }

    /// Run one forward for `seq_ids` on `input` (`[B, 1]`), fused-sample the
    /// next tokens on-device, schedule them with `async_eval`, and return the
    /// prebuilt step. The forward appends one speculative KV position per
    /// sequence (undone by [`Self::apply_lookahead_trim`]). Returns `None` if a
    /// sequence's caches vanished. The caller decides whether to keep the step
    /// (store it in `decode_lookahead`) or unwind it.
    pub(super) fn prime_lookahead_with_input(
        &mut self,
        seq_ids: &[SequenceId],
        input: &mlxcel_core::MlxArray,
        params: &FusedSampleParams,
    ) -> Option<DecodeLookahead> {
        let logits = self.lookahead_forward(seq_ids, input)?;
        let last_logits = mlxcel_core::slice_last_logits(&logits);
        // Same pre-fused row filters (top-n-sigma, typical_p) as `batched_fused_sample`,
        // so the pipelined lookahead samples from the identical distribution
        // as the synchronous fused path it accelerates. A no-op adding no
        // graph nodes while every filter is disabled.
        let last_logits = apply_row_filters(last_logits, params);
        let tokens = mlxcel_core::fused_sample(
            &last_logits,
            params.temperature,
            params.top_k,
            params.top_p,
            params.min_p,
        );
        // Announce a newly-seen sampling dispatch outcome at INFO (#901).
        mlxcel_core::report_sampling_dispatch();
        // Schedule the sampled tokens (and thus the whole forward graph) without
        // reading them to host, so the GPU runs ahead while the caller returns
        // to the scheduler loop and reads the PREVIOUS step's tokens.
        //
        // #822: go through the fallible async boundary. An MLX throw at graph
        // capture (e.g. a graph-cache abort) is recorded and priming is skipped;
        // the caller then takes the synchronous decode path, which re-evaluates
        // through the same guard and fails the affected request(s) cleanly. This
        // speculative helper never aborts sequences itself, so the failure is
        // handled once, in the sync path.
        if self
            .record_eval_outcome(mlxcel_core::try_async_eval(&tokens).map_err(|e| e.to_string()))
            .is_err()
        {
            // #822: `lookahead_forward` already appended one speculative KV
            // position per sequence before this async schedule. The async eval
            // threw and was caught, so unwind that append before bailing;
            // otherwise the untrimmed position desyncs the KV against
            // `generated_tokens` and the fallback synchronous decode runs on a
            // corrupted cache. Mirrors the one-position teardown the
            // stale/bootstrap fallbacks use.
            self.apply_lookahead_trim(seq_ids, lookahead_teardown_positions(false));
            return None;
        }
        Some(DecodeLookahead {
            ids: seq_ids.to_vec(),
            tokens,
        })
    }

    /// Forward pass for the lookahead pipeline, mirroring the synchronous decode
    /// forward exactly: the B=1 per-sequence path
    /// ([`Self::decode_single_step`]) or the batched path
    /// ([`Self::execute_batched_decode`]) with the same decode-storage context.
    /// Returns `None` if a sequence's caches vanished (the caller then skips
    /// priming).
    pub(super) fn lookahead_forward(
        &mut self,
        seq_ids: &[SequenceId],
        input: &mlxcel_core::MlxArray,
    ) -> Option<UniquePtr<mlxcel_core::MlxArray>> {
        let logits = if seq_ids.len() == 1 {
            let seq_id = seq_ids[0];
            let caches = self.cache_pool.get_caches_mut(seq_id)?;
            self.model
                .forward_with_sequence_id(input, Some(seq_id), caches, None)
        } else {
            let decode_context = self.decode_batch_context();
            let mut batch_caches = self.cache_pool.get_batch_caches_mut(seq_ids).ok()?;
            let logits = self.model.forward_batched_with_context_and_ids(
                input,
                Some(seq_ids),
                &mut batch_caches,
                None,
                Some(&decode_context),
            );
            drop(batch_caches);
            logits
        };
        for &seq_id in seq_ids {
            self.sync_sequence_storage(seq_id);
        }
        Some(logits)
    }

    /// Steady pipelined decode, ordered exactly like the CLI generation loop
    /// (`generate.rs`) so the GPU never idles on the host read:
    ///
    /// 1. FIRST build and `async_eval` step n+1 from `la.tokens` fed back
    ///    device-side (no host knowledge needed). The GPU starts the next
    ///    forward immediately.
    /// 2. THEN read step n's tokens to host (this blocks on the PREVIOUS tick's
    ///    prime forward, which by now has finished, while step n+1 runs on the
    ///    GPU) and run the finish pre-check.
    /// 3. If a row finishes (EOS / length / cancel) or the shape is off, unwind
    ///    BOTH speculative appends (step n and the just-issued step n+1) and
    ///    re-run the tick synchronously, so completion / donation flows through
    ///    the untouched sync path from a clean cache state.
    /// 4. Otherwise commit step n and keep the step n+1 prebuilt step.
    pub(super) fn pipelined_steady_decode(
        &mut self,
        la: DecodeLookahead,
        seq_ids: &[SequenceId],
        params: &FusedSampleParams,
    ) {
        // Step 1: speculatively build + schedule step n+1 FIRST. Feed la.tokens
        // ([B]) back as the next [B, 1] input device-side (reshape + int32 cast
        // to match the synchronous from_slice_i32 dtype), keeping the GPU busy
        // through the host read below. This appends a second speculative KV
        // position per sequence (the overshoot the issue accepts).
        let col = mlxcel_core::reshape_token_for_forward(&la.tokens);
        let next_input = mlxcel_core::astype(&col, mlxcel_core::dtype::INT32);
        let next = self.prime_lookahead_with_input(seq_ids, &next_input, params);

        // Step 2: read step n's tokens to host (the sync point) and finish-check.
        let toks = lookahead_tokens_to_host(&la.tokens);
        let mut finishing = toks.len() != seq_ids.len();
        if !finishing {
            for (i, &seq_id) in seq_ids.iter().enumerate() {
                let Some(seq) = self.active_batch.get(seq_id) else {
                    finishing = true;
                    break;
                };
                if lookahead_token_finishes(
                    toks[i],
                    seq.generated_tokens.len(),
                    seq.max_tokens,
                    &seq.merged_eos,
                    seq.cancelled.load(Ordering::Relaxed),
                ) {
                    finishing = true;
                    break;
                }
            }
        }

        if finishing {
            // Step 3: tear down. Sync the in-flight step n+1 forward so its
            // kernels are not still writing KV when we rewind, then unwind both
            // speculative appends (step n from the previous prime plus step n+1
            // when it was actually issued) back to the synchronous-decode
            // invariant and re-run the tick synchronously.
            if let Some(nla) = &next {
                // #822: sync the in-flight step n+1 forward through the fallible
                // boundary before rewinding. A throw here is recorded but does
                // not abort inline: the teardown + synchronous re-dispatch below
                // re-runs the tick and fails the affected request(s) through the
                // guarded synchronous decode path.
                let _ = self.record_eval_outcome(
                    mlxcel_core::try_eval(&nla.tokens).map_err(|e| e.to_string()),
                );
            }
            let positions = lookahead_teardown_positions(next.is_some());
            self.apply_lookahead_trim(&la.ids, positions);
            drop(next);
            drop(la);
            // The sync re-dispatch below re-samples step n's token. fused_sample
            // draws from MLX's global RNG (random::categorical without an
            // explicit key), so at temperature > 0 the re-drawn token can
            // differ from the discarded lookahead sample that triggered this
            // finish pre-check; greedy (temp 0) is unaffected, matching the
            // byte-equivalence gate. Bounded to one token per completing
            // request, and stochastic runs carry no cross-mode determinism
            // guarantee.
            self.dispatch_sync_decode(seq_ids);
            self.maybe_prime_lookahead(seq_ids);
            return;
        }

        // Step 4: every row continues. Commit step n (reusing the batched
        // fast-path bookkeeping; no finish can fire after the pre-check) and
        // keep the already-primed step n+1.
        drop(la);
        self.apply_fused_decode_tokens(seq_ids, &toks);
        self.decode_lookahead = next;
        self.batch_observability.record_lookahead_step();
    }

    /// Batched decode: one forward_batched() call for all active sequences.
    ///
    /// # Null/empty-cache safety
    ///
    /// Early-exits on `seq_ids.is_empty()`. Though the scheduler's current
    /// [`Self::decide_action`] never produces a `Decode(ids)` action with an
    /// empty list (it returns [`BatchSchedulerAction::Idle`] first), this
    /// guard makes the method robust against future policy changes and any
    /// direct caller. Dispatching a zero-batch forward pass would otherwise
    /// materialize an empty `[0, 1]` input tensor and invoke the model
    /// kernel with no work to do, which is both wasteful and potentially
    /// undefined behavior in downstream MLX kernels.
    ///
    /// This mirrors the upstream `mlx-lm` `BatchKVCache.filter` / `extend`
    /// null-guards that prevent cache operations from crashing when all
    /// sequences have been filtered out of the batch.
    pub(super) fn execute_batched_decode(&mut self, seq_ids: &[SequenceId]) {
        if seq_ids.is_empty() {
            // Filter-to-empty case: nothing to do. Bookkeeping is handled by
            // the caller (`execute_decode_step`) via its own length guard.
            return;
        }

        let b = seq_ids.len();

        // trim per-sequence plain KVCache layers before the batched
        // forward pass so all sequences stay within the --max-kv-size bound.
        // Sliding-window (model-internal RotatingKVCache) and Turbo-quantized
        // caches are unaffected (trim_front returns 0 for Turbo modes).
        for &seq_id in seq_ids {
            let retention = self
                .active_batch
                .get(seq_id)
                .map(|seq| seq.retention)
                .unwrap_or_default();
            self.enforce_max_kv_size_for(seq_id, retention);
        }

        let mut last_tokens: Vec<i32> = Vec::with_capacity(b);

        for &seq_id in seq_ids {
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => {
                    self.execute_decode_step_sequential_remaining(seq_ids, last_tokens.len());
                    return;
                }
            };
            last_tokens.push(*seq.generated_tokens.last().unwrap_or(&0));
        }

        let input = mlxcel_core::from_slice_i32(&last_tokens, &[b as i32, 1]);

        debug_assert!(
            {
                let unique: HashSet<_> = seq_ids.iter().collect();
                unique.len() == seq_ids.len()
            },
            "execute_batched_decode: duplicate SequenceId in seq_ids"
        );

        let decode_context = self.decode_batch_context();
        let mut batch_caches = match self.cache_pool.get_batch_caches_mut(seq_ids) {
            Ok(caches) => caches,
            Err(err) => {
                tracing::error!("{err} during batched decode");
                return;
            }
        };

        let logits = self.model.forward_batched_with_context_and_ids(
            &input,
            Some(seq_ids),
            &mut batch_caches,
            None,
            Some(&decode_context),
        );
        drop(batch_caches);

        for &seq_id in seq_ids {
            self.sync_sequence_storage(seq_id);
        }

        // Fast path: when every active row shares a fused-compatible sampling
        // config and none needs a structured-output mask, a thinking-budget
        // override, or a per-token logprobs payload, sample all B rows in ONE
        // fused `[B, vocab] -> [B]` dispatch + eval instead of B per-row
        // slice/sample/eval/extract round trips. The per-row loop below stays
        // the exact fallback for every other case (structured output,
        // row-specific logprobs, token-bias observability, thinking budgets,
        // mixed sampling configs).
        if let Some(params) = self.batched_decode_fused_params(seq_ids) {
            let tokens = batched_fused_sample(&logits, &params);
            // #822: a completed fused decode is a successful eval (the graph is
            // evaluated inside `batched_fused_sample`'s host readback), so clear
            // the consecutive-failure run. This keeps isolated earlier failures
            // from accumulating toward the shutdown threshold across a long run
            // when the hot path stays on the fused branch. The fused readback
            // itself still uses the infallible host-copy path; routing that
            // through the fallible readback is tracked as a follow-up.
            self.note_eval_success();
            self.apply_fused_decode_tokens(seq_ids, &tokens);
            return;
        }

        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let seq_logits =
                mlxcel_core::slice(&logits, &[i as i32, 0, 0], &[i as i32 + 1, 1, i32::MAX]);

            // when the sequence has a structured-output
            // constraint, apply the schema mask to the per-sequence logits
            // before sampling. Failures here surface as a clean
            // FinishReason::Error rather than silent non-conforming output.
            let constraint_clone = self
                .active_batch
                .get_mut(seq_id)
                .and_then(|s| s.structured.clone());
            let logits_for_sampling = if let Some(constraint) = constraint_clone.as_ref() {
                let shape = mlxcel_core::array_shape(&seq_logits);
                let vocab = *shape.last().unwrap_or(&0) as usize;
                match Self::apply_structured_mask(constraint, mlxcel_core::copy(&seq_logits), vocab)
                {
                    Ok(masked) => masked,
                    Err(msg) => {
                        Self::abort_sequence_with_error(
                            self.active_batch.get_mut(seq_id),
                            "structured output",
                            &msg,
                        );
                        continue;
                    }
                }
            } else {
                mlxcel_core::copy(&seq_logits)
            };

            // Use cached token_history (incrementally maintained) instead of
            // rebuilding per step. Use cached merged_eos computed once at prefill.
            //
            // follow-up: we capture `sampled` separately from
            // `final_id` so the structured-output matcher (below) can be
            // advanced by the *pre-override* token. The matcher's mask
            // describes which token ids are grammatically legal at this
            // step; feeding it the post-override forced `</think>` would
            // hand it a token outside its allowed set and cause a parser
            // error or silent mis-advance.
            let (sampled_token, token_val, token_lp) = {
                // Penalty rows use the incremental per-sequence sampler state
                // (lazily created); the no-penalty rows that reach this per-row
                // fallback take the original rebuild-free path unchanged.
                let (token_arr, adjusted_logits) = {
                    let seq = match self.active_batch.get_mut(seq_id) {
                        Some(s) => s,
                        None => continue,
                    };
                    if seq.sampling.needs_token_history() {
                        sample_token_optimized_with_state(
                            &logits_for_sampling,
                            &seq.sampling,
                            &seq.token_history,
                            &mut seq.sampler_state,
                        )
                    } else {
                        sample_token_optimized(
                            &logits_for_sampling,
                            &seq.sampling,
                            &seq.token_history,
                        )
                    }
                };
                // #822: force-evaluate the sampled token through the fallible
                // boundary now that the `active_batch` borrow has ended. On an
                // MLX throw, fail just this row and keep serving the rest of the
                // batch; the infallible `item_i32` readback below would otherwise
                // re-trigger the same throw and abort the process.
                if let Err(msg) = self.record_eval_outcome(
                    mlxcel_core::try_eval(&token_arr).map_err(|e| e.to_string()),
                ) {
                    Self::abort_sequence_with_error(
                        self.active_batch.get_mut(seq_id),
                        "inference backend",
                        &msg,
                    );
                    if self.eval_failures_exhausted() {
                        return;
                    }
                    continue;
                }
                let sampled = mlxcel_core::item_i32(&token_arr);
                let seq = match self.active_batch.get_mut(seq_id) {
                    Some(s) => s,
                    None => continue,
                };
                // apply the thinking-budget override first so that
                // when the override fires (sampled != final_id) we can skip
                // the log-softmax work entirely. The logprob metadata would
                // be dropped anyway because the emitted `</think>` differs
                // from the token the logits describe, so computing it first
                // is wasted GPU work on the decode hot path.
                let final_id = Self::apply_thinking_budget(&mut seq.thinking, sampled);
                let lp = if final_id == sampled {
                    compute_logprobs(&adjusted_logits, sampled, &seq.logprobs_config)
                } else {
                    // Override fired; token text and logprob metadata must
                    // stay consistent, so drop the logprob for this step.
                    None
                };
                (sampled, final_id, lp)
            };

            // advance the matcher state with the *pre-override*
            // sampled token (`sampled_token`), not the post-override
            // `token_val`. The matcher derived its mask from the unaltered
            // logits, so feeding it `final_id` after a thinking-budget
            // override would hand it a token outside its allowed set and
            // either cause a parser error or silently mis-advance. Mirrors
            // the pattern in `finish_prefill` which uses
            // `sampled_first_token`.
            //
            // If `consume_token` fails (matcher hit an error state),
            // transition the sequence to `Finished(Error)` and skip
            // emission so non-conforming output never reaches the client.
            let structured_stopped = if let Some(constraint) = constraint_clone {
                match Self::consume_structured_token(&constraint, sampled_token) {
                    Ok(stopped) => stopped,
                    Err(msg) => {
                        Self::abort_sequence_with_error(
                            self.active_batch.get_mut(seq_id),
                            "structured output",
                            &msg,
                        );
                        continue;
                    }
                }
            } else {
                false
            };

            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => continue,
            };

            if seq.merged_eos.contains(&token_val) {
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Stop))
                {
                    tracing::error!("State transition error: {err}");
                }
                continue;
            }

            seq.generated_tokens.push(token_val);

            // Incrementally update token_history
            if seq.sampling.needs_token_history() {
                seq.token_history.push(token_val);
            }

            // Stream through the request's stop matcher (issue #1466): text that
            // could still become a stop string is held back, and a completed
            // stop string ends the sequence with the match excluded.
            let stop_word = match seq.decode_state.on_token(token_val, &self.tokenizer) {
                Some(new_text) => seq.stream_decoded_text(new_text, token_lp),
                None => None,
            };

            if stop_word.is_some()
                && let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::StopSequence))
            {
                tracing::error!("State transition error: {err}");
            }

            if !seq.state.is_finished()
                && structured_stopped
                && let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Stop))
            {
                tracing::error!("State transition error: {err}");
            }

            if !seq.state.is_finished()
                && seq.generated_tokens.len() >= seq.max_tokens
                && let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }

            // b10621 context guard (#1472): with context shifting disabled, a
            // bounded sequence stops before its next token would overflow the
            // KV window, reported as `truncated: true` with `stop_type:
            // "limit"` rather than silently discarding old tokens.
            if !seq.state.is_finished()
                && Self::context_bound_stop_due(
                    seq,
                    self.max_kv_size,
                    self.context_retention.context_shift,
                )
            {
                seq.retention.context_exhausted = true;
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Length))
                {
                    tracing::error!("State transition error: {err}");
                }
            }

            // Loop / repetition guard (issue #432): end early when the raw
            // generated stream collapses into a short repeated pattern. Skip if
            // the length limit already finished this sequence; the detector is
            // a zero-overhead no-op when loop detection is disabled (default).
            if !seq.state.is_finished()
                && mlxcel_core::detect_repetition_loop(
                    &seq.generated_tokens,
                    &seq.sampling.loop_detection,
                )
            {
                match seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
                {
                    Ok(()) => tracing::info!(
                        generated = seq.generated_tokens.len(),
                        "loop detection: ending generation early (repetition loop)"
                    ),
                    Err(err) => tracing::error!("State transition error: {err}"),
                }
            }

            // Periodic cache clearing, backend-aware cadence (#627): disabled by
            // default on CUDA (clear churns the pool and defeats CUDA-graph
            // reuse, mlx#2358), 256 on Metal, MLXCEL_CACHE_CLEAR_INTERVAL overrides.
            if mlxcel_core::memory::should_clear_cache_at(
                seq.generated_tokens.len(),
                mlxcel_core::memory::cache_clear_interval(),
            ) {
                mlxcel_core::clear_memory_cache();
            }

            if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
                cache_set.current_offset += 1;
            }
        }
    }

    /// Decide whether the batched decode fast path applies to `seq_ids`.
    ///
    /// Returns `Some(params)` with the shared scalar sampling parameters when
    /// EVERY active row can be sampled by a single `[B, vocab] -> [B]` fused
    /// dispatch: all rows share the same scalar parameters, none needs a
    /// history-based penalty or token bias, and none needs a structured-output
    /// mask, a thinking-budget override, or a per-token logprobs payload. Any
    /// row that needs per-row treatment returns `None`, which routes the caller
    /// to the unchanged per-row fallback loop.
    ///
    /// The per-row obligations map onto the generic predicate
    /// [`mlxcel_core::sampling::row_supports_fused_batch`] as: structured-output
    /// mask -> `needs_logit_mask` (`seq.structured`); thinking-budget override
    /// -> `needs_token_override` (`seq.thinking`); per-token logprobs ->
    /// `needs_per_token_payload` (`seq.logprobs_config`).
    pub(super) fn batched_decode_fused_params(
        &self,
        seq_ids: &[SequenceId],
    ) -> Option<FusedSampleParams> {
        let mut shared: Option<FusedSampleParams> = None;
        for &seq_id in seq_ids {
            // A row that vanished from the batch forces the per-row fallback,
            // which carries its own missing-sequence guards.
            let seq = self.active_batch.get(seq_id)?;
            if !row_supports_fused_batch(
                &seq.sampling,
                seq.structured.is_some(),
                !seq.thinking.is_disabled(),
                seq.logprobs_config.enabled,
            ) {
                return None;
            }
            let params = FusedSampleParams::from_config(&seq.sampling);
            match shared {
                None => shared = Some(params),
                Some(first) if !first.matches(&params) => return None,
                Some(_) => {}
            }
        }
        shared
    }

    /// Bookkeeping for the batched fused fast path.
    ///
    /// Consumes the `[B]` token ids produced by
    /// [`mlxcel_core::sampling::batched_fused_sample`] and drives each
    /// sequence's EOS check, token history, streaming decode, length limit,
    /// periodic cache clear, and cache-offset advance. This mirrors the tail of
    /// the per-row loop in [`Self::execute_batched_decode`] minus the per-row
    /// sampling, structured-output, thinking-budget, and logprobs work that
    /// [`Self::batched_decode_fused_params`] already excluded. `tokens[i]` is
    /// the id sampled for `seq_ids[i]`.
    pub(super) fn apply_fused_decode_tokens(&mut self, seq_ids: &[SequenceId], tokens: &[i32]) {
        debug_assert_eq!(
            seq_ids.len(),
            tokens.len(),
            "apply_fused_decode_tokens: token count must match seq_ids"
        );
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let token_val = tokens[i];
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => continue,
            };

            if seq.merged_eos.contains(&token_val) {
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Stop))
                {
                    tracing::error!("State transition error: {err}");
                }
                continue;
            }

            seq.generated_tokens.push(token_val);

            // The gate guarantees no penalty config reaches the fast path, so
            // this is a no-op today; it is kept for exact parity with the
            // per-row loop in case the gate ever admits history-tracking
            // configs.
            if seq.sampling.needs_token_history() {
                seq.token_history.push(token_val);
            }

            // Same stop-string enforcement as the per-row loop (issue #1466).
            // The fast path excludes penalty-bearing configs, not stop strings,
            // so it must honor them or a request would silently change behavior
            // depending on which decode kernel the batch happened to take.
            let stop_word = match seq.decode_state.on_token(token_val, &self.tokenizer) {
                Some(new_text) => seq.stream_decoded_text(new_text, None),
                None => None,
            };

            if stop_word.is_some()
                && let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::StopSequence))
            {
                tracing::error!("State transition error: {err}");
            }

            if !seq.state.is_finished()
                && seq.generated_tokens.len() >= seq.max_tokens
                && let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }

            // b10621 context guard (#1472): with context shifting disabled, a
            // bounded sequence stops before its next token would overflow the
            // KV window, reported as `truncated: true` with `stop_type:
            // "limit"` rather than silently discarding old tokens.
            if !seq.state.is_finished()
                && Self::context_bound_stop_due(
                    seq,
                    self.max_kv_size,
                    self.context_retention.context_shift,
                )
            {
                seq.retention.context_exhausted = true;
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Length))
                {
                    tracing::error!("State transition error: {err}");
                }
            }

            // Loop / repetition guard (issue #432): end early when the raw
            // generated stream collapses into a short repeated pattern. Skip if
            // the length limit already finished this sequence; the detector is
            // a zero-overhead no-op when loop detection is disabled (default).
            if !seq.state.is_finished()
                && mlxcel_core::detect_repetition_loop(
                    &seq.generated_tokens,
                    &seq.sampling.loop_detection,
                )
            {
                match seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
                {
                    Ok(()) => tracing::info!(
                        generated = seq.generated_tokens.len(),
                        "loop detection: ending generation early (repetition loop)"
                    ),
                    Err(err) => tracing::error!("State transition error: {err}"),
                }
            }

            // Periodic cache clearing, backend-aware cadence (#627): disabled by
            // default on CUDA (clear churns the pool and defeats CUDA-graph
            // reuse, mlx#2358), 256 on Metal, MLXCEL_CACHE_CLEAR_INTERVAL overrides.
            if mlxcel_core::memory::should_clear_cache_at(
                seq.generated_tokens.len(),
                mlxcel_core::memory::cache_clear_interval(),
            ) {
                mlxcel_core::clear_memory_cache();
            }

            if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
                cache_set.current_offset += 1;
            }
        }
    }

    pub(super) fn execute_decode_step_sequential_remaining(
        &mut self,
        seq_ids: &[SequenceId],
        start_from: usize,
    ) {
        for &seq_id in &seq_ids[start_from..] {
            self.decode_single_step(seq_id);
        }
    }

    pub(super) fn decode_single_step(&mut self, seq_id: SequenceId) {
        let last_token = {
            let seq = match self.active_batch.get_mut(seq_id) {
                Some(s) => s,
                None => return,
            };
            *seq.generated_tokens.last().unwrap_or(&0)
        };

        // trim the oldest tokens from plain KVCache layers so the
        // live window stays within the configured --max-kv-size bound before
        // each decode forward pass. Sliding-window layers are managed by the
        // model and bypass this pool path; Turbo-quantized caches silently skip
        // the trim (KVCache::trim_front returns 0 for Turbo modes).
        let retention = self
            .active_batch
            .get(seq_id)
            .map(|seq| seq.retention)
            .unwrap_or_default();
        self.enforce_max_kv_size_for(seq_id, retention);

        let input = mlxcel_core::from_slice_i32(&[last_token], &[1, 1]);
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq_id) {
                Some(c) => c,
                None => {
                    tracing::error!("Cache not found for {seq_id} during decode");
                    return;
                }
            };
            self.model
                .forward_with_sequence_id(&input, Some(seq_id), caches, None)
        };
        self.sync_sequence_storage(seq_id);

        // apply structured-output mask to per-step logits when
        // the sequence has an attached constraint. Errors abort the
        // sequence cleanly rather than emitting non-conforming output.
        let constraint_clone = self
            .active_batch
            .get_mut(seq_id)
            .and_then(|s| s.structured.clone());
        let logits_for_sampling = if let Some(constraint) = constraint_clone.as_ref() {
            let shape = mlxcel_core::array_shape(&logits);
            let vocab = *shape.last().unwrap_or(&0) as usize;
            match Self::apply_structured_mask(constraint, mlxcel_core::copy(&logits), vocab) {
                Ok(masked) => masked,
                Err(msg) => {
                    Self::abort_sequence_with_error(
                        self.active_batch.get_mut(seq_id),
                        "structured output",
                        &msg,
                    );
                    return;
                }
            }
        } else {
            mlxcel_core::copy(&logits)
        };

        // Use cached token_history from SequenceInfo (incrementally maintained)
        // and cached merged_eos (computed once during prefill) to avoid
        // per-step allocation and reconstruction overhead.
        //
        // follow-up: we capture `sampled` separately from
        // `final_id`. The structured-output matcher (below) must be
        // advanced by the pre-override token because its mask was derived
        // from the unaltered logits; passing the post-override forced
        // `</think>` would feed it a token outside its allowed set.
        let (sampled_token, token_val, token_lp) = {
            // Penalty sequences use the incremental per-sequence sampler state
            // (lazily created); a no-penalty sequence takes the original
            // rebuild-free path unchanged.
            let (token_arr, adjusted_logits) = {
                let seq = self.active_batch.get_mut(seq_id).unwrap();
                if seq.sampling.needs_token_history() {
                    sample_token_optimized_with_state(
                        &logits_for_sampling,
                        &seq.sampling,
                        &seq.token_history,
                        &mut seq.sampler_state,
                    )
                } else {
                    sample_token_optimized(&logits_for_sampling, &seq.sampling, &seq.token_history)
                }
            };
            // #822: force-evaluate the sampled token through the fallible
            // boundary now that the `active_batch` borrow has ended. On an MLX
            // throw, fail just this request; the infallible `item_i32` readback
            // below would otherwise re-trigger the same throw and abort the
            // process.
            if let Err(msg) = self
                .record_eval_outcome(mlxcel_core::try_eval(&token_arr).map_err(|e| e.to_string()))
            {
                Self::abort_sequence_with_error(
                    self.active_batch.get_mut(seq_id),
                    "inference backend",
                    &msg,
                );
                self.eval_failures_exhausted();
                return;
            }
            let sampled = mlxcel_core::item_i32(&token_arr);
            let seq = self.active_batch.get_mut(seq_id).unwrap();
            // apply the thinking-budget override first so that
            // when the override fires the log-softmax work is skipped: the
            // logprob metadata for the sampled token would be dropped anyway
            // (token text and logprob `token_id` must stay consistent), so
            // computing it up-front wastes GPU time on every override step.
            let final_id = Self::apply_thinking_budget(&mut seq.thinking, sampled);
            let lp = if final_id == sampled {
                compute_logprobs(&adjusted_logits, sampled, &seq.logprobs_config)
            } else {
                None
            };
            (sampled, final_id, lp)
        };

        // advance the matcher state with the *pre-override*
        // sampled token. See the parallel comment in
        // `execute_batched_decode` for why this must not be `token_val`.
        let structured_stopped = if let Some(constraint) = constraint_clone {
            match Self::consume_structured_token(&constraint, sampled_token) {
                Ok(stopped) => stopped,
                Err(msg) => {
                    Self::abort_sequence_with_error(
                        self.active_batch.get_mut(seq_id),
                        "structured output",
                        &msg,
                    );
                    return;
                }
            }
        } else {
            false
        };

        let seq = match self.active_batch.get_mut(seq_id) {
            Some(s) => s,
            None => return,
        };

        if seq.merged_eos.contains(&token_val) {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
            {
                tracing::error!("State transition error: {err}");
            }
            return;
        }

        seq.generated_tokens.push(token_val);

        // Incrementally update token_history instead of rebuilding from scratch
        if seq.sampling.needs_token_history() {
            seq.token_history.push(token_val);
        }

        // Stop-string enforcement for the single-step decode path (issue #1466).
        let stop_word = match seq.decode_state.on_token(token_val, &self.tokenizer) {
            Some(new_text) => seq.stream_decoded_text(new_text, token_lp),
            None => None,
        };

        if stop_word.is_some()
            && let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::StopSequence))
        {
            tracing::error!("State transition error: {err}");
        }

        if !seq.state.is_finished()
            && structured_stopped
            && let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
        {
            tracing::error!("State transition error: {err}");
        }

        if !seq.state.is_finished()
            && seq.generated_tokens.len() >= seq.max_tokens
            && let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
        {
            tracing::error!("State transition error: {err}");
        }

        // b10621 context guard (#1472): see the batched-loop twin above.
        if !seq.state.is_finished()
            && Self::context_bound_stop_due(
                seq,
                self.max_kv_size,
                self.context_retention.context_shift,
            )
        {
            seq.retention.context_exhausted = true;
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }
        }

        // Loop / repetition guard (issue #432): end early when the raw
        // generated stream collapses into a short repeated pattern. Skip if the
        // length limit already finished this sequence; the detector is a
        // zero-overhead no-op when loop detection is disabled (default).
        if !seq.state.is_finished()
            && mlxcel_core::detect_repetition_loop(
                &seq.generated_tokens,
                &seq.sampling.loop_detection,
            )
        {
            match seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::RepetitionLoop))
            {
                Ok(()) => tracing::info!(
                    generated = seq.generated_tokens.len(),
                    "loop detection: ending generation early (repetition loop)"
                ),
                Err(err) => tracing::error!("State transition error: {err}"),
            }
        }

        // Periodic cache clearing, backend-aware cadence (#627): disabled by
        // default on CUDA (clear churns the pool and defeats CUDA-graph
        // reuse, mlx#2358), 256 on Metal, MLXCEL_CACHE_CLEAR_INTERVAL overrides.
        if mlxcel_core::memory::should_clear_cache_at(
            seq.generated_tokens.len(),
            mlxcel_core::memory::cache_clear_interval(),
        ) {
            mlxcel_core::clear_memory_cache();
        }

        if let Some(cache_set) = self.cache_pool.get_mut(seq_id) {
            cache_set.current_offset += 1;
        }
    }

    // ------------------------------------------------------------------
    // Completion and cleanup
    // ------------------------------------------------------------------

    pub(super) fn finalize_completed(&mut self) {
        // Any completion or cancellation changes batch membership and may donate
        // a sequence's KV to the prompt cache. Tear down a live lookahead first
        // so its speculative KV position is trimmed off before donation / slot
        // reuse and the surviving sequences rebuild their pipeline next tick
        // (#632 constraints 2, 3, 6). No-op on the steady no-finish path.
        if self.decode_lookahead.is_some()
            && self
                .active_batch
                .iter_sequences()
                .any(|s| s.state.is_finished() || s.cancelled.load(Ordering::Relaxed))
        {
            self.discard_lookahead();
        }

        // First, transition any cancelled sequences to Finished(Cancelled).
        // This must happen before the finished-ID scan so that newly cancelled
        // sequences are collected in the same pass.
        let cancelled_ids: Vec<SequenceId> = self
            .active_batch
            .iter_sequences()
            .filter(|s| !s.state.is_finished() && s.cancelled.load(Ordering::Relaxed))
            .map(|s| s.seq_id)
            .collect();

        for id in &cancelled_ids {
            if let Some(seq) = self.active_batch.get_mut(*id) {
                if let Err(err) = seq
                    .state
                    .transition_to(SequenceState::Finished(FinishReason::Cancelled))
                {
                    tracing::warn!("Failed to cancel sequence {id}: {err}");
                } else {
                    tracing::info!("Sequence {id} cancelled (client disconnected)");
                }
            }
        }

        // Cancel a chunked-prefill-in-progress sequence if client disconnected.
        if let Some(ref seq) = self.chunked_prefill_seq
            && seq.cancelled.load(Ordering::Relaxed)
        {
            let seq = self.chunked_prefill_seq.take().unwrap();
            tracing::info!(
                "Chunked-prefill sequence {} cancelled (client disconnected)",
                seq.seq_id
            );
            let _ = seq.response_tx.send(GenerateEvent::Error(
                "Request cancelled: client disconnected".to_string(),
            ));
            // Cancellation during prefill means the KV cache is only
            // partially populated; skip donate-back and just release. The
            // context map still needs cleanup so no dangling entries leak.
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            self.batch_observability.record_sequence_completed();
        }

        // Also cancel queued sequences whose client has already disconnected,
        // so they never enter the active batch.
        self.cancel_queued_disconnected();

        // Collect finished IDs by scanning active sequences. Uses iter_sequences()
        // to avoid allocating a full key snapshot when no sequences are finished.
        let finished_ids: Vec<SequenceId> = self
            .active_batch
            .iter_sequences()
            .filter(|s| s.state.is_finished())
            .map(|s| s.seq_id)
            .collect();

        let has_completed = !finished_ids.is_empty();
        for id in finished_ids {
            if let Some(mut seq) = self.active_batch.remove(id) {
                let tokens_generated = seq.generated_tokens.len();

                // Forward the incremental detokenizer's held tail as one final
                // token event before Done, so streaming clients are not missing
                // text the non-streaming result.text still carries (issue #633).
                // The tail passes through the stop matcher, which also releases
                // whatever the matcher itself was holding back (issue #1466).
                let tail = seq.decode_state.flush(&self.tokenizer);
                seq.close_text_stream(tail);
                let cached = seq.already_cached_tokens;
                let result = seq.take_generation_result(&self.tokenizer, cached);
                // Per-request TTFT / decode-rate telemetry (epic #623 #624).
                // Recorded once here, where the finished sequence's timings are
                // available, never on the per-token hot path.
                self.batch_observability.record_request_completion(
                    result.prompt_tokens,
                    result.cached_tokens,
                    result.prompt_eval_ms,
                    result.generation_only_ms,
                    result.completion_tokens,
                );
                tracing::info!(
                    prompt_tokens = seq.prompt_tokens.len(),
                    cached_tokens = cached,
                    generation_time_ms = result.generation_time_ms,
                    "prompt-cache: request completed: \
                     cached={}/{} prompt tokens, total {}ms",
                    cached,
                    seq.prompt_tokens.len(),
                    result.generation_time_ms,
                );
                let _ = seq.response_tx.send(GenerateEvent::Done(result));

                // donate the full KV cache back to
                // the prompt-cache store on *healthy* finishes (Stop /
                // StopSequence / Length / Cancelled) so the next turn of the
                // same conversation can adopt it. `Finished(Error)` paths bypass
                // this branch — their cache is assumed tainted.
                let healthy = matches!(
                    seq.state,
                    SequenceState::Finished(
                        FinishReason::Stop
                            | FinishReason::StopSequence
                            | FinishReason::Length
                            | FinishReason::RepetitionLoop
                            | FinishReason::Cancelled,
                    )
                );
                self.donate_finished_sequence_cache(
                    id,
                    &seq.prompt_tokens,
                    &seq.generated_tokens,
                    healthy,
                );
                // `donate_finished_sequence_cache` already removed the
                // context from `prompt_cache_seq_ctx` on donate; drop it
                // defensively on the non-donate paths so the map cannot
                // grow unbounded across long-lived workers.
                self.prompt_cache_seq_ctx.remove(&id);

                self.release_sequence_caches(id);
                self.batch_metrics
                    .record_sequence_completed(tokens_generated);
                self.batch_observability.record_sequence_completed();

                tracing::debug!("Sequence {id} completed ({tokens_generated} tokens)");
            }
        }

        if has_completed {
            self.publish_metrics();
        }
    }

    /// Remove queued sequences whose client has already disconnected.
    ///
    /// This prevents cancelled requests from ever entering the active batch,
    /// freeing the prefill queue slot immediately.
    pub(super) fn cancel_queued_disconnected(&mut self) {
        let drained: Vec<SequenceInfo> = self.prefill_queue.drain_cancelled();
        for seq in drained {
            tracing::info!(
                "Queued sequence {} cancelled before prefill (client disconnected)",
                seq.seq_id
            );
            let _ = seq.response_tx.send(GenerateEvent::Error(
                "Request cancelled: client disconnected".to_string(),
            ));
            // No prefill ran → no valid cache to donate. Clear the
            // context entry so it cannot linger.
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            self.batch_observability.record_sequence_completed();
        }
    }

    pub(super) fn abort_sequence(&mut self, seq: SequenceInfo, error: &str) {
        let _ = seq
            .response_tx
            .send(GenerateEvent::Error(error.to_string()));
        // Abort paths produce an error outcome (OOM / transition failure /
        // invalid cache); the KV cache is untrustworthy and must not be
        // donated back. Dropping the context entry prevents a future
        // finalize pass from trying.
        self.prompt_cache_seq_ctx.remove(&seq.seq_id);
        self.release_sequence_caches(seq.seq_id);
    }
}
