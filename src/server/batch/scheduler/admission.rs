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
    pub(super) fn drain_incoming_requests(&mut self) {
        loop {
            match self.request_rx.try_recv() {
                Ok(req) => {
                    if self.handle_incoming(req) {
                        self.shutdown_requested = true;
                        return;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.shutdown_requested = true;
                    return;
                }
            }
        }
    }

    pub(super) fn handle_incoming(&mut self, req: ModelRequest) -> bool {
        match req {
            ModelRequest::Generate {
                prompt,
                prompt_token_ids,
                options,
                runtime,
                images,
                audio,
                videos,
                media: _,
                queue_reservation: _,
                response_tx,
                cancelled,
            } => {
                let use_worker_token_bias = runtime.is_none();
                self.enqueue_request(
                    prompt,
                    prompt_token_ids,
                    options,
                    images,
                    audio,
                    videos,
                    response_tx,
                    cancelled,
                    use_worker_token_bias,
                );
                false
            }
            ModelRequest::PromptCacheWarmup { tokens, ctx } => {
                self.enqueue_prompt_cache_warmup(tokens, ctx);
                false
            }
            ModelRequest::Shutdown => {
                tracing::info!("BatchScheduler received shutdown signal");
                true
            }
        }
    }

    pub(super) fn enqueue_request(
        &mut self,
        prompt: String,
        prompt_token_ids: Option<Vec<i32>>,
        options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<crate::server::media::ResolvedVideo>,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
        use_worker_token_bias: bool,
    ) {
        // Prefer the ids tokenized on the request-dispatch thread (issue #633);
        // fall back to scheduler-side tokenization when the dispatcher had no
        // pre-tokenizer. `tokenize_prompt_for_generation` is the shared
        // `add_special` convention so both paths are byte-identical.
        let mut prompt_tokens: Vec<i32> = match prompt_token_ids {
            Some(ids) => ids,
            None => match if audio.is_empty() {
                crate::server::model_provider::tokenize_prompt_for_generation(
                    &self.tokenizer,
                    &prompt,
                )
            } else {
                crate::server::model_provider::tokenize_prompt_for_generation_with_ordered_media(
                    &self.tokenizer,
                    &prompt,
                    true,
                )
            } {
                Ok(ids) => ids,
                Err(err) => {
                    let _ = response_tx
                        .send(GenerateEvent::Error(format!("Tokenization error: {err}")));
                    return;
                }
            },
        };

        // Empty-prompt guard (null/empty-cache safety):
        //
        // A zero-token prompt cannot be prefilled — the forward pass would
        // run with a `[1, 0]` input and the per-sequence KV cache would
        // remain in the `keys is None, offset == 0` state. Admitting such a
        // request into the batch could later crash the scheduler when the
        // cache is used alongside populated caches in `execute_batched_*`.
        // Mirrors the upstream `mlx-lm` `BatchKVCache.extend` null-guard
        // that refuses to pad/concatenate a cache with no tensors. VLM
        // requests may legitimately start with an empty token list (image
        // tokens are injected later by `prepare_request_vlm_embeddings`),
        // so this guard only applies to pure-text requests without images,
        // audio, or videos.
        if prompt_tokens.is_empty() && images.is_empty() && audio.is_empty() && videos.is_empty() {
            let _ = response_tx.send(GenerateEvent::Error(
                "Empty prompt: request has no input tokens to process".to_string(),
            ));
            return;
        }

        let mut sampling = merge_config_stop_tokens(options.sampling.clone(), &self.config_eos);

        // Axis B (B8): attach the scheduler-wide token bias to each sequence's
        // sampling config when no per-request override is present. Empty
        // cached bias = bit-exact baseline (the `is_empty()` short-circuit in
        // `sample_token_optimized` keeps hot-path cost at zero).
        //
        // Phase 1 limitation: one policy per batch. Per-request overrides
        // via `/v1/chat/completions` request body are deferred to B12.
        if use_worker_token_bias && !self.token_bias.is_empty() && sampling.token_bias.is_empty() {
            sampling.token_bias = self.token_bias.clone();
        }

        // issue #350: force-suppress the model's reserved multimodal
        // placeholder tokens (audio / image / video span markers) on every
        // sequence's output logits. Applied after the lang-bias merge and
        // unconditionally, so suppression always wins over any per-request
        // bias and a placeholder id can never become the sampled argmax. A
        // no-op (and zero alloc) for non-multimodal models whose suppressed
        // set is empty.
        if !self.model_output_suppressed.is_empty() {
            sampling
                .token_bias
                .suppress_tokens(&self.model_output_suppressed);
        }

        // XTC (Exclude Top Choices) special-token allowlist: the tokenizer's
        // newline id(s) plus every id in this request's merged end-of-sequence
        // set (built the same way EOS detection does during decode, see the
        // `merged_eos_token_ids` calls in `execute_batched_decode` and the
        // classic decode paths). Only computed when XTC is actually enabled
        // for this request — the overwhelming majority of requests leave
        // `xtc_probability == 0.0` and skip this entirely.
        if sampling.xtc_probability > 0.0 {
            let mut allowlist = self.xtc_newline_token_ids.clone();
            for id in merged_eos_token_ids(self.model.eos_token_ids(), &sampling.stop_token_ids) {
                if !allowlist.contains(&id) {
                    allowlist.push(id);
                }
            }
            sampling.xtc_special_token_ids = allowlist;
        }

        // b10621 --ignore-eos / ignore_eos (#1436): upstream implements it
        // as a -inf logit bias on every end-of-generation token, so the
        // model keeps generating until the token budget or a string stop.
        // Suppressing through the shared token-bias map reproduces that
        // exactly; the EOS stop check then never fires because the id can
        // never be sampled. Opt-in only, so the common path stays bit-exact
        // (and fused-batch eligible: a non-empty bias map already routes to
        // the per-row sampler).
        if options.ignore_eos {
            let eos = merged_eos_token_ids(self.model.eos_token_ids(), &sampling.stop_token_ids);
            sampling.token_bias.suppress_tokens(&eos);
        }

        let is_multimodal = !images.is_empty() || !audio.is_empty() || !videos.is_empty();

        // Experimental VLM prompt-prefix cache sharing (#124 step c). Off by
        // default; the operator opts in with `--enable-vlm-prefix-cache`. Video
        // payloads are excluded because video frame bytes are not folded into
        // the request's multimodal digest yet. Phi4MM is also excluded because
        // cache adoption cannot yet restore its per-sequence speech/vision
        // adapter mode alongside the detached KV payload.
        let vlm_sharing_ok = vlm_prefix_sharing_allowed(
            self.enable_vlm_prefix_cache,
            is_multimodal,
            !videos.is_empty(),
            matches!(self.model, LoadedModel::Phi4MMVLM(_)),
        );

        // For VLM sharing the image/audio placeholder tokens must be expanded
        // BEFORE probing the cache: `prepare_request_vlm_embeddings` rewrites
        // `prompt_tokens` into the post-injection stream the KV cache is built
        // over, and both the cache key (via the request's multimodal digest)
        // and the matched-prefix length are computed against that stream. No
        // sequence id exists yet, so a preparation error just aborts the
        // request with nothing to clean up. `Some(_)` marks "prepared early";
        // the inner value is the optional merged embeddings.
        let prepared_early = if vlm_sharing_ok {
            match prepare_request_vlm_embeddings(
                &self.model,
                &self.tokenizer,
                &prompt,
                &mut prompt_tokens,
                &images,
                &audio,
                &videos,
                Some(self.vision_caches.as_ref()),
                options.image_soft_tokens,
                cancelled.as_ref(),
                self.batch_observability.as_ref(),
            ) {
                Ok(emb) => Some(emb),
                Err(err) => {
                    let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
                    return;
                }
            }
        } else {
            None
        };

        // before allocating a fresh KV-cache slot,
        // probe the prompt-prefix cache for a reusable detached set. On a
        // hit, adopt under a brand-new SequenceId and record how many
        // leading tokens the prefill can skip. On a miss (which includes
        // feature-disabled, no ctx, and race paths), fall through to the
        // cold-allocation path below.
        //
        // Text-only requests use the cache whenever the route attached a
        // context. Multimodal requests opt in only under `vlm_sharing_ok`;
        // when they do, the adopt is restricted to a whole-entry match
        // (`require_whole_entry == is_multimodal`) so the prefilled suffix is
        // guaranteed to be the newly-appended text turn: every image/audio
        // token sits inside the matched prefix, and a different media payload
        // lands in a different digest bucket. Multimodal requests with sharing
        // off keep the legacy cold-prefill path (their pre-injection token
        // stream is not self-describing).
        let ctx_ref = if is_multimodal && !vlm_sharing_ok {
            None
        } else {
            options.prompt_cache_ctx.as_ref()
        };
        let (seq_id, prefill_start_offset, already_cached_tokens) = match ctx_ref
            .and_then(|ctx| self.try_adopt_cached_prefix(ctx, &prompt_tokens, is_multimodal))
        {
            Some((adopted_id, adopted_len)) => (adopted_id, adopted_len, adopted_len),
            None => {
                // Miss or feature disabled → regular allocate.
                // count misses only when the cache is actually
                // active (ctx_ref is Some) to avoid inflating the miss
                // counter for multimodal or cache-disabled requests.
                if ctx_ref.is_some() {
                    self.batch_metrics.record_prompt_cache_miss();
                }
                let seq_id = match self.allocate_sequence_state() {
                    Ok(id) => id,
                    Err(err) => {
                        tracing::warn!("Cache pool allocation failed: {err}");
                        let _ =
                            response_tx.send(GenerateEvent::Error(format!("Server busy: {err}")));
                        return;
                    }
                };
                (seq_id, 0, 0)
            }
        };

        // Resolve the per-sequence input embeddings. On the VLM-sharing path
        // the tokens were already expanded above; if a prefix was adopted
        // (`prefill_start_offset > 0`) the remaining suffix is the appended
        // text turn, so the full-prompt embeddings are dropped and the suffix
        // runs through the token path (the adopted KV already holds the image
        // rows, and the MRoPE / per-layer state bound below covers the suffix
        // positions). Otherwise the request prepares embeddings here exactly as
        // before.
        let vlm_embeddings = match prepared_early {
            Some(emb) => {
                if prefill_start_offset > 0 {
                    None
                } else {
                    emb
                }
            }
            None => match prepare_request_vlm_embeddings(
                &self.model,
                &self.tokenizer,
                &prompt,
                &mut prompt_tokens,
                &images,
                &audio,
                &videos,
                Some(self.vision_caches.as_ref()),
                options.image_soft_tokens,
                cancelled.as_ref(),
                self.batch_observability.as_ref(),
            ) {
                Ok(emb) => emb,
                Err(err) => {
                    // Clean up the context map so a donate-back won't fire for
                    // a sequence that never reached a healthy finish.
                    self.prompt_cache_seq_ctx.remove(&seq_id);
                    self.release_sequence_caches(seq_id);
                    let _ = response_tx.send(GenerateEvent::Error(err.to_string()));
                    return;
                }
            },
        };

        // mlx-vlm PR #1095: per-sequence MRoPE alignment.
        //
        // The Qwen VL families compute the MRoPE position-id tensor and
        // `rope_deltas` scalar inside `prepare_request_vlm_embeddings`
        // (it runs the vision encoder and writes the result to the text
        // model's fallback slot). Without binding that state to *this*
        // sequence id, the next text-only request's decode step would
        // pick up the previous VL row's delta and produce wrong
        // attention positions. Bind unconditionally for Qwen VL models;
        // the call is a no-op for everything else.
        self.model.bind_qwen_vl_mrope_state_to_sequence(seq_id);

        // per-sequence `per_layer_inputs` for Gemma 4
        // E2B/E4B. `prepare_request_vlm_embeddings` writes the
        // freshly projected tensor to the VL model's fallback slot;
        // this call drains the slot into a per-`SequenceId` map so a
        // burst of Gemma 4 VLM requests in a single drain tick cannot
        // have one row consume another row's tensor. No-op for
        // everything that is not a Gemma 4 VLM.
        self.model.bind_gemma4_per_layer_inputs_to_sequence(seq_id);

        // Same lifecycle invariant for Falcon-OCR: the prefill state
        // (temporal positions, spatial coordinates, rope delta) is
        // written to a fallback slot during embedding preparation and
        // must be bound to this sequence before another request in the
        // same drain tick overwrites it. No-op for everything else.
        self.model.bind_falcon_ocr_state_to_sequence(seq_id);

        // Issue #85: same lifecycle invariant for Gemma 3n VLM. The
        // legacy `Gemma3nVLModel.cached_per_layer_inputs` cell was a
        // single fallback slot with no per-sequence binding; under a
        // burst of Gemma 3n VLM requests the next prepare would
        // overwrite the slot before the first prefill consumed it (or
        // panic on `Option::unwrap` when the timing flipped). The
        // call below is a no-op for everything that is not a
        // Gemma 3n VLM.
        self.model.bind_gemma3n_per_layer_inputs_to_sequence(seq_id);

        let decode_state = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);

        // resolve the effective thinking-token budget for this
        // sequence from the per-request override + server default. The route
        // layer supplies `thinking_enter_block_on_start` as `true` when the
        // rendered prompt primes `<think>` (chat endpoints) and `false` for
        // raw text endpoints.
        let thinking = self.build_thinking_state(
            options.reasoning_budget,
            options.thinking_enter_block_on_start,
            options.reasoning_control.clone(),
        );

        // Record the per-request prompt-cache context so the donate-back
        // path can compose the insert key without reaching back into the
        // HTTP layer. Only stored when the feature is active and the
        // request actually carried a context — otherwise the map stays
        // empty and the donate-back short-circuits. Multimodal requests are
        // stored only when VLM sharing is enabled (#124 step c); otherwise
        // they keep opting out of the cache entirely.
        if self.prompt_cache_active()
            && (!is_multimodal || vlm_sharing_ok)
            && let Some(mut ctx) = options.prompt_cache_ctx.clone()
        {
            self.resolve_history_boundary(&mut ctx, &prompt_tokens, is_multimodal);
            self.prompt_cache_seq_ctx.insert(seq_id, ctx);
        }

        // Guard against a degenerate cache hit where the adopted prefix
        // covers the entire tokenized prompt. This can legitimately happen
        // when a client replays an identical prompt. Back off one token so
        // the prefill path still runs and the sampler sees fresh logits.
        let prefill_start_offset =
            if prefill_start_offset >= prompt_tokens.len() && !prompt_tokens.is_empty() {
                tracing::debug!(
                    seq_id = %seq_id,
                    "prompt-cache hit covered the entire prompt; re-running the \
                     last token through prefill to produce a sampling logit"
                );
                prompt_tokens.len() - 1
            } else {
                prefill_start_offset
            };

        let seq = SequenceInfo {
            seq_id,
            state: SequenceState::Queued,
            prompt_tokens,
            sampling,
            max_tokens: options.max_tokens,
            eos_token_ids: self.config_eos.clone(),
            priority: options.priority,
            logprobs_config: options.logprobs,
            vlm_embeddings,
            images,
            audio,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            decode_state,
            // Honor the request's string stop sequences on the MLX serving path
            // (issue #1466). Empty / absent leaves the matcher inactive, in
            // which case every decoded piece is emitted verbatim.
            stop_matcher: StopMatcher::new(options.stop_sequences.clone().unwrap_or_default()),
            prefill_offset: 0,
            prefill_start_offset,
            already_cached_tokens,
            response_tx,
            cancelled,
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
            token_history: Vec::new(),
            sampler_state: None,
            merged_eos: Vec::new(),
            thinking,
            // forward the structured-output constraint built by
            // the route layer so the per-step sampling path can consult it.
            structured: options.structured.clone(),
        };

        if let Err(rejected) = self.prefill_queue.enqueue(seq) {
            self.prompt_cache_seq_ctx.remove(&rejected.seq_id);
            self.release_sequence_caches(rejected.seq_id);
            let _ = rejected.response_tx.send(GenerateEvent::Error(
                "Server busy: prefill queue full".to_string(),
            ));
        }
    }

    // ------------------------------------------------------------------
    // Scheduling decision
    // ------------------------------------------------------------------

    /// Snapshot the scheduler state the tick policy reads.
    ///
    /// `should_preempt()` is evaluated eagerly here, where the pre-#908
    /// `decide_action` reached it only inside one branch. That is safe and
    /// cheap: it takes `&self` and mutates nothing, so hoisting it cannot
    /// change behaviour, and it returns on the first condition
    /// (`!self.enable_preemption`) unless `--enable-preemption` is on, which is
    /// off by default. Even enabled it is O(active batch), bounded by
    /// `--parallel`, which is why `decide_action_is_o1_regardless_of_queue_size`
    /// still holds.
    pub(super) fn tick_state(&self) -> TickState {
        TickState {
            speculative_pending: self.speculative_slice.is_some()
                || !self.speculative_slice_backlog.is_empty(),
            speculative_yielded: self.speculative_slice_yielded,
            chunked_prefill_in_progress: self.chunked_prefill_seq.is_some(),
            active_is_empty: self.active_batch.is_empty(),
            active_is_full: self.active_batch.is_full(),
            queue_is_empty: self.prefill_queue.is_empty(),
            should_preempt: self.should_preempt(),
            mixed_step_enabled: self.mixed_step_enabled,
            decode_ticks_since_prefill_grant: self.decode_ticks_since_prefill_grant,
            prefill_grant_interval: self.prefill_grant_interval,
        }
    }

    /// Determine the next action. Runs in O(1) time.
    ///
    /// The policy itself lives in [`decide_tick`], a pure function over
    /// [`TickState`]; this method only snapshots the state and attaches the
    /// active sequence ids that `Decode` and `MixedStep` carry. The split
    /// exists because the unit tests used to re-implement the policy locally
    /// and drifted from it, which is how the chunked-prefill starvation
    /// documented in ADR 0005 stayed hidden behind a test named
    /// `chunked_prefill_interleaving_pattern` (issue #908).
    ///
    /// Behaviour is unchanged from before issue #908 unless `MLXCEL_MIXED_STEP`
    /// is set or the issue #1011 fairness grant fires.
    /// `tick_policy_tests::default_policy_differs_from_pre_908_only_where_the_grant_fires`
    /// characterises both directions of that divergence over the complete
    /// 128-state boolean space, and
    /// `grant_disabled_is_identical_to_the_pre_908_policy` pins the
    /// `--prefill-grant-interval 0` escape hatch as byte-identical to the old
    /// arbitration.
    ///
    /// This is the ONLY place the #1011 fairness ledger is written, and it is
    /// written from the value the policy returned alongside the choice, so the
    /// counter cannot drift out of step with the arbitration it feeds. Taking
    /// `&mut self` for that write is what makes forgetting it impossible: there
    /// is no way to obtain a choice without also obtaining, and storing, its
    /// successor counter.
    pub(super) fn decide_action(&mut self) -> BatchSchedulerAction {
        tracing::debug!(
            active = self.active_batch.len(),
            queued = self.prefill_queue.len(),
            chunked_in_progress = self.chunked_prefill_seq.is_some(),
            speculative_slice = self.speculative_slice.is_some(),
            slice_backlog = self.speculative_slice_backlog.len(),
            prefill_grant_wait = self.decode_ticks_since_prefill_grant,
            "scheduler tick"
        );
        let decision = decide_tick(&self.tick_state());
        // A granted tick is one the parked prefill took from a live decode
        // batch. Count it before the action runs, from the arbitration itself,
        // so the counter answers "did the fairness policy engage" and not
        // "did a chunk happen to run"; the ordinary drained-batch continuation
        // must not move it or it stops being a dispatch proof.
        if decision.choice == TickChoice::Prefill
            && self.chunked_prefill_seq.is_some()
            && !self.active_batch.is_empty()
        {
            self.batch_observability.record_prefill_grant();
        }
        self.decode_ticks_since_prefill_grant = decision.decode_ticks_since_prefill_grant;
        match decision.choice {
            TickChoice::SpeculativeRound => BatchSchedulerAction::SpeculativeRound,
            TickChoice::Decode => BatchSchedulerAction::Decode(self.active_batch.sequence_ids()),
            TickChoice::MixedStep => {
                BatchSchedulerAction::MixedStep(self.active_batch.sequence_ids())
            }
            TickChoice::Prefill => BatchSchedulerAction::Prefill(SequenceId::from_raw(0)),
            TickChoice::Idle => BatchSchedulerAction::Idle,
        }
    }

    /// Check if preemption should occur: batch is full, preemption is
    /// enabled, and a higher-priority request is waiting.
    pub(super) fn should_preempt(&self) -> bool {
        if !self.enable_preemption || !self.active_batch.is_full() {
            return false;
        }
        // Only preempt if waiting request has higher priority than some
        // active sequence.
        let waiting_priority = match self.prefill_queue.peek_priority() {
            Some(p) => p,
            None => return false,
        };
        // Find the lowest-priority active sequence
        let min_active_priority = self
            .active_batch
            .iter_min_priority()
            .unwrap_or(RequestPriority::High);

        waiting_priority > min_active_priority
    }

    // ------------------------------------------------------------------
    // Paged KV block-budget admission (#122 b2)
    // ------------------------------------------------------------------

    /// Estimate the pool blocks a sequence's prefill will pin: one block per
    /// `block_size` prompt tokens, per layer. Returns 0 when there is no paged
    /// pool (the budget gate is then a no-op for this model).
    pub(super) fn estimate_prefill_blocks(&self, prompt_len: usize) -> usize {
        match self.cache_pool.paged_block_size() {
            Some(block_size) if block_size > 0 => prompt_len
                .div_ceil(block_size)
                .saturating_mul(self.model.num_layers()),
            _ => 0,
        }
    }

    /// Reclaim paged pool blocks until at least `need` are acquirable, or no
    /// further reclamation is possible. First evicts cold prompt-cache prefixes
    /// (LRU; releasing their pins frees real blocks), then preempts running
    /// sequences (which re-prefill on resume). Returns whether `need` blocks are
    /// now acquirable.
    pub(super) fn reclaim_paged_blocks(&mut self, need: usize) -> bool {
        let room = |pool: &CachePool| pool.free_paged_block_budget().is_none_or(|f| f >= need);
        // 1. Evict cold cross-request prefixes; releasing their pins frees blocks.
        if let Some(store) = self.prompt_cache.clone() {
            while !room(&self.cache_pool) {
                if store.evict_one_lru() == 0 {
                    break; // nothing left to evict
                }
                self.drain_store_paged_releases();
            }
        }
        if room(&self.cache_pool) {
            return true;
        }
        // 2. Preempt running sequences (drop their KV; they re-prefill on resume).
        while !room(&self.cache_pool) {
            if !self.try_evict_for_preemption() {
                break; // no preemptible victim left
            }
        }
        room(&self.cache_pool)
    }

    /// Paged block-budget admission gate. Returns `Some(seq)` to proceed with
    /// the prefill, or `None` when the sequence was deferred (re-queued for a
    /// later tick once decodes free blocks) or rejected (it cannot fit the whole
    /// budget). A no-op (`Some(seq)`) when no budget is configured.
    pub(super) fn admit_paged_prefill(&mut self, seq: SequenceInfo) -> Option<SequenceInfo> {
        // Opt-in: no budget configured ⇒ admit (default unbounded behaviour).
        let total = match self.cache_pool.paged_block_budget() {
            Some(t) => t,
            None => return Some(seq),
        };
        let need = self.estimate_prefill_blocks(seq.prompt_tokens.len());
        if need == 0 {
            return Some(seq); // model does not use the paged pool
        }
        // If it cannot fit the entire budget, reject — deferring forever would
        // wedge the queue behind a request that can never run.
        if need > total {
            self.abort_sequence(
                seq,
                &format!(
                    "prompt needs {need} KV blocks, exceeding the {total}-block KV cache budget"
                ),
            );
            return None;
        }
        // Acquirable blocks (budget − live). `None` means the pool is not yet
        // created (nothing allocated ⇒ the whole budget is free).
        let free = self.cache_pool.free_paged_block_budget().unwrap_or(total);
        if need <= free {
            return Some(seq);
        }
        if self.reclaim_paged_blocks(need) {
            return Some(seq);
        }
        // Still no room — defer to a later tick. Decodes in flight will free
        // blocks as their sequences finish; this request retries then.
        if let Err(rejected) = self.prefill_queue.enqueue(seq) {
            self.prompt_cache_seq_ctx.remove(&rejected.seq_id);
            self.release_sequence_caches(rejected.seq_id);
            let _ = rejected.response_tx.send(GenerateEvent::Error(
                "Server busy: prefill queue full".to_string(),
            ));
        }
        None
    }
}
