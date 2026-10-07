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

use super::*;

impl BatchScheduler {
    /// The scheduler's KV bound and context-shift setting, as the shared
    /// finish step's context-bound stop (#1472) reads them.
    pub(super) fn context_bound(&self) -> ContextBound {
        ContextBound {
            max_kv_size: self.max_kv_size,
            context_shift: self.context_retention.context_shift,
        }
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
        match self.select_eviction_victim() {
            Some(victim_id) => self.preempt_sequence(victim_id),
            None => false,
        }
    }

    /// Preempt `victim_id`: drop its KV, reset it for re-prefill under a fresh
    /// id, and re-queue it. Returns `true` when a sequence left the batch.
    /// Shared by slot preemption ([`Self::try_evict_for_preemption`]) and paged
    /// block reclaim, which choose their victims differently (issue #1982).
    pub(super) fn preempt_sequence(&mut self, victim_id: SequenceId) -> bool {
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
            let mrope_snapshot = self.engine.model().take_qwen_vl_mrope_entry(victim.seq_id);

            // same lifecycle invariant for Gemma 4 E2B/E4B
            // `per_layer_inputs`. The tensor is projected exactly once
            // by `prepare_request_vlm_embeddings` at enqueue time and
            // is consumed at prefill time; preemption-and-reallocate
            // would otherwise drop it and the re-prefill would observe
            // `per_layer_inputs == None` for an E2B/E4B request. Take
            // it out before `release_sequence_caches` drains the map.
            let pli_snapshot = self
                .engine
                .model()
                .take_gemma4_per_layer_inputs_entry(victim.seq_id);

            // Issue #85: same for Gemma 3n VLM `per_layer_inputs`.
            // Without this round trip the re-prefill would panic in
            // `Gemma3nVLModel::forward_with_embeddings_and_sequence_id`
            // (per_layer_inputs missing for this sequence).
            let pli3n_snapshot = self
                .engine
                .model()
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
                    self.engine
                        .model()
                        .install_qwen_vl_mrope_entry(new_id, mrope_snapshot);
                    // same for Gemma 4 `per_layer_inputs`.
                    // The tensor is reused unchanged across re-prefill
                    // because both depend only on the request's
                    // input_ids (no decode-time updates).
                    self.engine
                        .model()
                        .install_gemma4_per_layer_inputs_entry(new_id, pli_snapshot);
                    // Issue #85: same for Gemma 3n `per_layer_inputs`.
                    self.engine
                        .model()
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

        // Reserve the pool blocks this tick will mint before the forward runs
        // (#1982): a paged write that cannot acquire a block panics mid-forward.
        // Under budget pressure this evicts cold prompt-cache prefixes, then
        // preempts, and drops rows that still cannot grow.
        let reserved = self.reserve_decode_step_blocks(seq_ids);
        let seq_ids = reserved.as_deref().unwrap_or(seq_ids);
        if seq_ids.is_empty() {
            self.batch_observability.record_decode_step(0);
            return;
        }

        let _span = tracing::info_span!("decode_step", batch_size = seq_ids.len(),).entered();
        // Apply the batch's runtime-LoRA snapshot (#1439); admission
        // guarantees every member shares it, so the first member speaks for
        // the batch.
        let batch_lora = self
            .active_batch
            .iter_sequences()
            .next()
            .and_then(|seq| seq.lora_scales.clone());
        self.ensure_lora_applied(batch_lora.as_ref());
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
        if seq_ids.len() <= 1 || !self.engine.model().supports_batching() {
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

        // The raised command-buffer input budget (`DecodeCommandBufferBudget`)
        // is applied only around pipelined work, where step n+1 is encoded
        // while the GPU still runs step n. A synchronous step encodes and then
        // waits, so one large buffer there removes the CPU-encode / GPU-execute
        // overlap inside the step: on M1 Ultra, command-r7b sync decode went
        // from a steady 98-102 tok/s to 68-100 with the budget raised.
        match self.decode_lookahead.take() {
            Some(la) if la.ids == seq_ids && params.is_some() && self.lookahead_safe() => {
                let _decode_budget = mlxcel_core::DecodeCommandBufferBudget::enter();
                self.pipelined_steady_decode(la, seq_ids, &params.unwrap());
            }
            Some(la) => {
                // Stale id set or no longer eligible/safe: no step n+1 prime was
                // issued, so undo the one speculative KV position and fall back
                // to a clean sync step.
                let failed =
                    self.apply_lookahead_trim(&la.ids, lookahead_teardown_positions(false));
                drop(la);
                let live = Self::without_failed(seq_ids, &failed);
                if live.is_empty() {
                    return;
                }
                self.dispatch_sync_decode(&live);
                let _decode_budget = mlxcel_core::DecodeCommandBufferBudget::enter();
                self.maybe_prime_lookahead(&live);
            }
            None => {
                self.dispatch_sync_decode(seq_ids);
                let _decode_budget = mlxcel_core::DecodeCommandBufferBudget::enter();
                self.maybe_prime_lookahead(seq_ids);
            }
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
            // A trimmable KV tail (dense, or pool-backed paged), or a
            // model-owned family that rewinds its own state (#2159), keyed on
            // the model's natural backend (#1754): the one rule the
            // raw-completion client's pipeline uses too.
            if !self.engine.can_unwind_lookahead(seq_id) {
                return None;
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
    /// Every cache trims through [`KVCache::trim`], which rewinds the pool block
    /// table for a pool-backed sequence and the dense KV tail otherwise, moving
    /// `offset` with it in both cases. Dense-natural paged mirrors then re-mirror
    /// the shorter length into the paged bookkeeping state.
    ///
    /// `positions` is the number of speculative appends to unwind: `1` for a
    /// teardown before the step-n+1 prime forward has run (admission,
    /// preemption, stale id set, cancellation seen in `finalize_completed`),
    /// `2` for the steady-tick teardown that already issued the step-n+1 prime
    /// (both the step-n and step-n+1 appends).
    ///
    /// A model-owned family (natural backend) is rewound through
    /// [`LanguageModel::rewind_decode_appends`], which reaches the model's own
    /// per-sequence state (#2159). A sequence whose rewind fails is finished
    /// with an error on the spot, so it is neither decoded nor donated from a
    /// desynchronized state; its id is returned so a caller about to decode
    /// the same id set can drop it. Empty on success.
    pub(super) fn apply_lookahead_trim(
        &mut self,
        ids: &[SequenceId],
        positions: usize,
    ) -> Vec<SequenceId> {
        let mut failed = Vec::new();
        if positions == 0 {
            return failed;
        }
        let want = positions as i32;
        for &seq_id in ids {
            // The engine trims the pool caches (a short trim is logged there:
            // a speculative position not unwound desyncs the KV against
            // generated_tokens), rewinds a model-owned family's own state and
            // re-mirrors the shorter length into any paged bookkeeping.
            if let Err(err) = self.engine.unwind_appends(seq_id, want) {
                tracing::error!(
                    seq_id = %seq_id,
                    positions = want,
                    error = %err,
                    "lookahead teardown: model-owned rewind failed, failing the request"
                );
                self.fail_desynchronized_sequence(seq_id, &err.to_string());
                failed.push(seq_id);
            }
        }
        failed
    }

    /// Finish `seq_id` with an error because its state no longer matches its
    /// tokens (#2159). Unlike [`Self::abort_sequence_with_error`] this also
    /// overrides a finish already recorded this tick (`length`, `stop`), which
    /// would otherwise donate the desynchronized state to the prompt cache.
    fn fail_desynchronized_sequence(&mut self, seq_id: SequenceId, err: &str) {
        let Some(seq) = self.active_batch.get_mut(seq_id) else {
            return;
        };
        let _ = seq.response_tx.send(GenerateEvent::Error(format!(
            "decode lookahead teardown: {err}"
        )));
        seq.state = SequenceState::Finished(FinishReason::Error(err.to_string()));
    }

    /// `seq_ids` without the ids a failed teardown just finished.
    fn without_failed(seq_ids: &[SequenceId], failed: &[SequenceId]) -> Vec<SequenceId> {
        seq_ids
            .iter()
            .copied()
            .filter(|id| !failed.contains(id))
            .collect()
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
            // A failed model-owned rewind already finished its sequence with
            // an error; `finalize_completed` releases it.
            let _ = self.apply_lookahead_trim(&la.ids, lookahead_teardown_positions(false));
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
        // The prime appends a KV position; skip it when that would need a
        // paged block the budget cannot give (#1982).
        if !self.prime_fits_block_budget(seq_ids) {
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

    /// Submit one pipelined step for `seq_ids` on `input` (`[B, 1]`): the
    /// engine runs the speculative forward, fused-samples the next tokens
    /// on-device and schedules them with `async_eval`. The forward appends one
    /// speculative KV position per sequence (undone by
    /// [`Self::apply_lookahead_trim`]). Returns `None` if a sequence's caches
    /// vanished or the schedule threw; the caller decides whether to keep the
    /// step (store it in `decode_lookahead`) or unwind it.
    pub(super) fn prime_lookahead_with_input(
        &mut self,
        seq_ids: &[SequenceId],
        input: &mlxcel_core::MlxArray,
        params: &FusedSampleParams,
    ) -> Option<DecodeLookahead> {
        // Every row's token bias, in the per-row sampler's chain position.
        // Without this the gate could not admit a biased row at all, which is
        // what kept `ignore_eos` and `logit_bias` requests on the synchronous
        // path.
        let biases: Option<Vec<&TokenBiasMap>> = seq_ids
            .iter()
            .map(|&seq_id| {
                self.active_batch
                    .get(seq_id)
                    .map(|seq| &seq.sampling.token_bias)
            })
            .collect();
        // A row left the batch between the gate and here: nothing was
        // appended yet, so the caller just decodes synchronously.
        let biases = biases?;
        let batch = StepBatch { seq_ids, input };
        match self.engine.submit(&batch, params, &biases) {
            Ok(tokens) => {
                // The schedule itself succeeded; the collect half evaluates
                // the tokens through its own guard.
                self.note_eval_success();
                Some(DecodeLookahead {
                    ids: seq_ids.to_vec(),
                    tokens,
                })
            }
            Err(mlxcel_core::engine::EngineError::Eval(msg)) => {
                // #822: the forward already appended one speculative KV
                // position per sequence before the async schedule threw.
                // Record the throw and unwind that append before bailing;
                // otherwise the untrimmed position desyncs the KV against
                // `generated_tokens` and the fallback synchronous decode runs
                // on a corrupted cache. The caller then takes the synchronous
                // path, which re-evaluates through the same guard and fails
                // the affected request(s) cleanly, so the failure is handled
                // once, there.
                let _ = self.record_eval_outcome(Err(msg));
                let _ = self.apply_lookahead_trim(seq_ids, lookahead_teardown_positions(false));
                None
            }
            Err(err) => {
                // A row left the batch between the gate and here, or its
                // caches vanished: nothing was appended.
                tracing::debug!("lookahead prime skipped: {err}");
                None
            }
        }
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
        let next_input = mlxcel_core::engine::lookahead_feedback_input(&la.tokens);
        let next = self.prime_lookahead_with_input(seq_ids, &next_input, params);

        // Step 2: read step n's tokens to host (the sync point) and finish-check.
        let toks = mlxcel_core::engine::tokens_to_host(&la.tokens);
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
            let failed = self.apply_lookahead_trim(&la.ids, positions);
            drop(next);
            drop(la);
            let live = Self::without_failed(seq_ids, &failed);
            if live.is_empty() {
                return;
            }
            let seq_ids = live.as_slice();
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

    /// Batched decode: one engine step for all active sequences.
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
        if !self.shared_budget_has_decode_room(seq_ids.len()) {
            self.finish_all_for_shared_kv_budget();
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

        self.run_engine_step(seq_ids, &input);
    }

    /// One engine step over `seq_ids`: the forward, then every row's sampling
    /// and finish step inside the engine. The engine takes the fused
    /// `[B, vocab] -> [B]` path when every row shares a fused-compatible
    /// config and none needs a structured-output mask, a thinking-budget
    /// override or a per-token logprobs payload, and the exact per-row chain
    /// otherwise; a batch of one always runs the per-row chain. Each row's
    /// outcome is then applied to its sequence (#822: a failed row finishes
    /// alone).
    pub(super) fn run_engine_step(
        &mut self,
        seq_ids: &[SequenceId],
        input: &mlxcel_core::MlxArray,
    ) {
        let context = self.context_bound();
        let batch = StepBatch { seq_ids, input };
        let stepped = {
            let Some(seqs) = self.active_batch.get_rows_mut(seq_ids) else {
                tracing::warn!("a sequence left the active batch before its decode step");
                return;
            };
            let tokenizer = &self.tokenizer;
            let mut rows: Vec<_> = seqs
                .into_iter()
                .map(|seq| step_rows::step_row(seq, tokenizer, context, false))
                .collect();
            self.engine.step(&batch, &mut rows)
        };
        match stepped {
            Ok(out) => {
                self.finish_rows_state(&out.rows);
                self.apply_row_outcomes(&out.rows);
            }
            Err(err) => tracing::error!("{err} during decode"),
        }
    }

    /// Apply each finished row's cause to its sequence state (the engine's
    /// finish step reports the cause; the sequence's `FinishReason` and its
    /// side effects are the scheduler's).
    pub(super) fn finish_rows_state(&mut self, outcomes: &[RowOutcome]) {
        for outcome in outcomes {
            if let Some(cause) = outcome.finish
                && let Some(seq) = self.active_batch.get_mut(outcome.seq_id)
            {
                apply_finish_cause(seq, cause);
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
    /// This is the engine's own rule
    /// ([`mlxcel_core::engine::shared_fused_params`]) over each row's
    /// [`step_rows::fused_gate`], the same gate the engine's fused branch
    /// reads through the step row, so the lookahead pipeline and the
    /// synchronous step cannot disagree on which batches are fused. Token
    /// bias is folded into the logits at both dispatch points, so a biased
    /// row stays eligible (keeping it off the pipeline cost `ignore_eos` and
    /// `logit_bias` requests 13% decode). A row that vanished from the batch
    /// forces the per-row fallback, which carries its own missing-sequence
    /// guards.
    pub(super) fn batched_decode_fused_params(
        &self,
        seq_ids: &[SequenceId],
    ) -> Option<FusedSampleParams> {
        mlxcel_core::engine::shared_fused_params(
            seq_ids
                .iter()
                .map(|&seq_id| self.active_batch.get(seq_id).map(step_rows::fused_gate)),
        )
    }

    /// Collect half of a pipelined step: `tokens[i]` is the id the engine's
    /// fused draw produced for `seq_ids[i]`. Each row runs the engine's finish
    /// step ([`mlxcel_core::engine::Engine::finish_rows`]): EOS check, token
    /// history, streaming decode, length limit, periodic cache clear and the
    /// cache-offset advance. The pipeline's gate already excluded structured
    /// output, thinking budgets and logprobs, so no per-row sampling runs
    /// here.
    pub(super) fn apply_fused_decode_tokens(&mut self, seq_ids: &[SequenceId], tokens: &[i32]) {
        debug_assert_eq!(
            seq_ids.len(),
            tokens.len(),
            "apply_fused_decode_tokens: token count must match seq_ids"
        );
        let context = self.context_bound();
        let outcomes = {
            let Some(seqs) = self.active_batch.get_rows_mut(seq_ids) else {
                tracing::warn!("a sequence left the active batch before its pipelined step");
                return;
            };
            let tokenizer = &self.tokenizer;
            let mut rows: Vec<_> = seqs
                .into_iter()
                .map(|seq| step_rows::step_row(seq, tokenizer, context, false))
                .collect();
            self.engine.finish_rows(tokens, &mut rows)
        };
        self.finish_rows_state(&outcomes);
        // The gate keeps per-row failures off this path; a failed row here is
        // the engine refusing a token count that does not match the rows, and
        // it must finish with an error rather than stall.
        if outcomes.iter().any(|outcome| outcome.error.is_some()) {
            self.apply_row_outcomes(&outcomes);
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

    /// Decode one token for a single sequence: a batch of one through the
    /// same engine step the batched path runs.
    pub(super) fn decode_single_step(&mut self, seq_id: SequenceId) {
        if !self.shared_budget_has_decode_room(1) {
            self.finish_all_for_shared_kv_budget();
            return;
        }

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
        self.run_engine_step(std::slice::from_ref(&seq_id), &input);
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
                // Classic decode: no drafter ran, so no acceptance block
                // reaches the client (#1314).
                let result = seq.take_generation_result(&self.tokenizer, cached, None);
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
                // Only the generated tokens the model state actually holds
                // (#1754): every finish but a merged-EOS stop leaves the last
                // pushed token unforwarded.
                self.donate_finished_sequence_cache(
                    id,
                    &seq.prompt_tokens,
                    seq.generated_in_state(),
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
