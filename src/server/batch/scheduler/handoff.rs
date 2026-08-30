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
    pub(super) fn release_sequence_caches(&mut self, seq_id: SequenceId) {
        self.model.release_sequence_state_by_id(seq_id);
        if let Some(caches) = self.cache_pool.get_caches_mut(seq_id) {
            self.model.release_sequence_state(caches);
        }
        self.cache_pool.release(seq_id);
    }

    /// #822: note a successful MLX eval at the decode/prefill boundary, clearing
    /// the consecutive-failure run. Cheap single-field write on the hot path.
    #[inline]
    pub(super) fn note_eval_success(&mut self) {
        self.consecutive_decode_eval_failures = 0;
    }

    /// #822: record the outcome of a fallible decode/prefill eval and translate
    /// a failure into a request-facing error string.
    ///
    /// `result` is the `Result` from `mlxcel_core::try_eval` /
    /// `mlxcel_core::try_async_eval` (an MLX C++ throw already caught at the cxx
    /// boundary and mapped to `String`). On `Ok` the consecutive-failure counter
    /// is reset. On `Err` the counter is bumped, the MLX message is logged, and
    /// a request-facing error is returned so the caller can abort the affected
    /// sequence(s) instead of letting the throw abort the whole process.
    ///
    /// The counter bookkeeping is deliberately split from the eval call itself:
    /// several eval sites run while a `cache_pool` / `active_batch` borrow is
    /// still live, so the raw `try_*` call is made inline there and its `Result`
    /// is handed to this method once the borrow has ended.
    pub(super) fn record_eval_outcome(&mut self, result: Result<(), String>) -> Result<(), String> {
        self.consecutive_decode_eval_failures =
            advance_eval_failure_count(self.consecutive_decode_eval_failures, result.is_ok());
        match result {
            Ok(()) => Ok(()),
            Err(mlx_msg) => {
                tracing::error!(
                    consecutive = self.consecutive_decode_eval_failures,
                    threshold = MAX_CONSECUTIVE_EVAL_FAILURES,
                    "MLX threw at the decode/prefill eval FFI boundary; failing the affected request(s) instead of aborting the process: {mlx_msg}"
                );
                Err(format!("inference backend error: {mlx_msg}"))
            }
        }
    }

    /// #822: after an eval failure has been recorded and the affected
    /// sequence(s) aborted, decide whether the backend is unrecoverable.
    ///
    /// Returns `false` (keep serving other sequences) while consecutive failures
    /// stay under [`MAX_CONSECUTIVE_EVAL_FAILURES`]. Once the threshold is
    /// reached it logs a fatal-level line, drains every remaining in-flight
    /// sequence with an error, requests a clean scheduler shutdown, and returns
    /// `true`. This is the guard that stops a persistently-failing backend (a
    /// corrupted allocator/device after an OOM, say) from spinning forever: the
    /// contract is graceful request-scoped failure OR clean shutdown, never
    /// pretend-recovery.
    pub(super) fn eval_failures_exhausted(&mut self) -> bool {
        if !eval_failures_reached_limit(self.consecutive_decode_eval_failures) {
            return false;
        }
        tracing::error!(
            consecutive = self.consecutive_decode_eval_failures,
            "FATAL: {MAX_CONSECUTIVE_EVAL_FAILURES} consecutive MLX eval failures at the decode boundary; treating the backend as unrecoverable, draining in-flight requests and shutting down the scheduler."
        );
        self.drain_all_inflight_with_error(
            "inference backend unavailable: too many consecutive MLX evaluation failures",
        );
        self.shutdown_requested = true;
        true
    }

    /// #822: fail every in-flight sequence (the active decode batch plus any
    /// chunked prefill in progress) with `err` so no request hangs when the
    /// scheduler shuts down on an unrecoverable backend. A sequence already in a
    /// terminal state this tick (e.g. the row whose eval just failed) is only
    /// reclaimed, not re-notified, so it never receives a duplicate error event.
    pub(super) fn drain_all_inflight_with_error(&mut self, err: &str) {
        for seq_id in self.active_batch.sequence_ids() {
            if let Some(seq) = self.active_batch.remove(seq_id) {
                if seq.state.is_finished() {
                    self.prompt_cache_seq_ctx.remove(&seq.seq_id);
                    self.release_sequence_caches(seq.seq_id);
                } else {
                    self.abort_sequence(seq, err);
                }
            }
        }
        if let Some(seq) = self.chunked_prefill_seq.take() {
            self.abort_sequence(seq, err);
        }
    }

    pub(super) fn begin_prefill(seq: &mut SequenceInfo) -> Result<(), String> {
        seq.state.transition_to(SequenceState::Prefilling)?;
        seq.prefill_start = Some(Instant::now());
        seed_rng_if_needed(&seq.sampling);
        Ok(())
    }

    // ── Disaggregated serving-role handoff hooks ─────────────────────────
    //
    // The in-crate seam that lets a serving-role worker move a finished
    // pool-backed sequence's KV across nodes. The mechanism (serialize /
    // anchored restore / one-token geometry probe) lives in
    // `crate::distributed::disaggregated::handoff_impl` and is exercised
    // byte-for-byte against real models by `tests/paged_handoff_parity.rs`.
    // A live caller (the decode / prefill role serve loop) lands in a later
    // step, so these stay `#[allow(dead_code)]` until then.

    /// Prefill role: serialize sequence `seq_id`'s pool-backed KV into a single
    /// wire frame for handoff to a decode node. `token_history` is the
    /// sequence's prompt token ids (so the decode node continues with the same
    /// context).
    #[allow(dead_code)]
    pub(crate) fn extract_sequence_handoff(
        &self,
        seq_id: SequenceId,
        token_history: Vec<i32>,
        generated_tokens: Vec<i32>,
    ) -> anyhow::Result<Vec<u8>> {
        crate::distributed::disaggregated::handoff_impl::extract_sequence_handoff(
            &self.cache_pool,
            seq_id,
            None,
            token_history,
            generated_tokens,
        )
    }

    /// Decode role: reconstruct a handed-off sequence from `bytes` onto a fresh
    /// pool-backed slot, anchored to this worker model's real block geometry,
    /// and return the new local sequence id. The geometry probe runs once on
    /// the first call and is cached in `paged_handoff_geometry`.
    /// Return this worker model's paged block geometry for handoff restores,
    /// probing it once on the first call and caching it in
    /// `paged_handoff_geometry`. `ExpectedBlockGeometry` is `Copy`, so the
    /// cached value is returned by value and leaves no borrow on `self`.
    pub(super) fn ensure_handoff_geometry(
        &mut self,
    ) -> anyhow::Result<crate::distributed::kv_cache_serde::ExpectedBlockGeometry> {
        if let Some(geometry) = self.paged_handoff_geometry {
            return Ok(geometry);
        }
        let probed = crate::distributed::disaggregated::handoff_impl::probe_block_geometry(
            &self.model,
            DEFAULT_PAGED_BLOCK_SIZE,
        )?;
        self.paged_handoff_geometry = Some(probed);
        Ok(probed)
    }

    #[allow(dead_code)]
    pub(crate) fn ingest_sequence_handoff(&mut self, bytes: &[u8]) -> anyhow::Result<SequenceId> {
        let geometry = self.ensure_handoff_geometry()?;
        crate::distributed::disaggregated::handoff_impl::ingest_sequence_handoff(
            &mut self.cache_pool,
            &self.model,
            bytes,
            &crate::distributed::kv_cache_serde::CacheIngestLimits::default(),
            &geometry,
            DEFAULT_PAGED_BLOCK_SIZE,
        )
    }

    /// Whether this node's model can participate in the disaggregated pool-block
    /// KV handoff (#125). The handoff extracts pool-backed Fp16 KV, which only the
    /// dense external KV-cache families (natural backend `DenseKvCache`: qwen3 /
    /// llama3) produce. Model-owned paged families (gemma3 / gemma4 / llama4 /
    /// qwen3_5 / qwen3_next, natural backend `ModelOwned`) keep their KV
    /// model-internal and are routed through the paged backend for shadow
    /// accounting only, so there is nothing pool-paged to extract: a handoff
    /// attempt reads unwritten pool tensors and used to crash the prefill serving
    /// loop (#708). The whole node serves one model, so this is a node-level fact
    /// the serving-role loop checks once and applies to every request.
    pub(crate) fn handoff_supported(&self) -> bool {
        self.model.sequence_state_layout().backend == SequenceStateBackend::DenseKvCache
    }

    /// Prefill role (#126 B2b): run a full prefill for `seq`, then extract its
    /// pool-backed KV as a handoff frame for a decode node and release the local
    /// caches.
    ///
    /// Returns `Ok(None)` when the request completed during prefill (an immediate
    /// EOS at the first token), in which case [`finish_prefill`] already finalized
    /// and released it and there is nothing to hand off.
    ///
    /// This is the "reuse then extract" factoring (option C): it drives the
    /// standard [`Self::execute_full_prefill`] path (or, for prompts longer than
    /// `--prefill-chunk-size`, the standard chunked-prefill path driven to
    /// completion, issue #197) verbatim, so first-token sampling, structured
    /// output, thinking budget, and logprobs are byte-for-byte identical to a
    /// single-node prefill, then lifts the finished sequence back out of the
    /// active batch before any local decode step runs. The hot [`Self::run`] loop
    /// is never touched. Speculative burst is bypassed (it would complete the
    /// request locally, defeating the handoff).
    ///
    /// [`finish_prefill`]: Self::finish_prefill
    #[allow(dead_code)]
    pub(crate) fn prefill_request_for_handoff(
        &mut self,
        mut seq: SequenceInfo,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        // The serving-role intake is strictly sequential, so a chunked prefill
        // left in progress would be a wiring bug; refuse rather than clobber it.
        if self.chunked_prefill_seq.is_some() {
            anyhow::bail!("prefill-role handoff: another chunked prefill is already in progress");
        }
        let seq_id = seq.seq_id;
        // The decode node restores the prompt context from `token_history`; the
        // first sampled token rides in `generated_tokens` so it seeds decode.
        let prompt_tokens = seq.prompt_tokens.clone();
        if let Err(err) = Self::begin_prefill(&mut seq) {
            self.abort_sequence(seq, &err);
            anyhow::bail!("prefill-role handoff: begin_prefill failed: {err}");
        }
        // Reuse the standard prefill machinery: it samples the first token and
        // transitions the sequence into the active batch (no speculative burst on
        // the handoff path). Long prompts take the same chunked path the run()
        // loop uses, driven to completion here so the full prompt's KV is in the
        // pool before extraction (issue #197); the final chunk samples the first
        // token via finish_prefill exactly like a single-node chunked prefill.
        // The handoff path is text-only, so the VLM-embeddings full-prefill
        // exemption in execute_prefill's dispatch cannot apply.
        // Handoff prefills carry no snapshot of their own; run them under the
        // server default, which is what the requesting node's own admission
        // would have snapshotted (#1439).
        self.ensure_lora_applied(None);
        if self.prefill_chunk_size > 0 && prompt_tokens.len() > self.prefill_chunk_size {
            self.start_chunked_prefill(seq);
            while self.chunked_prefill_seq.is_some() {
                self.continue_chunked_prefill();
            }
        } else {
            self.execute_full_prefill(seq);
        }
        // Lift the just-prefilled sequence back out before any local decode runs.
        // If it finished at prefill (immediate EOS) it is already finalized and
        // released, so there is nothing to hand off.
        let Some(seq) = self.active_batch.remove(seq_id) else {
            return Ok(None);
        };
        let generated_tokens = seq.generated_tokens.clone();
        let bytes = self.extract_sequence_handoff(seq_id, prompt_tokens, generated_tokens);
        // Release the local caches and per-sequence tracking on BOTH outcomes:
        // on success the KV now belongs to the decode node (no donate-back, the
        // prefix left this node); on an extract error the sequence was already
        // lifted out of the active batch, so skipping the release would leak
        // its pool slot.
        self.prompt_cache_seq_ctx.remove(&seq_id);
        self.release_sequence_caches(seq_id);
        Ok(Some(bytes?))
    }

    /// Prefill role (#126 B3a): build a queued text sequence from the raw request
    /// parts the serving-role prefill loop carries, run a full prefill, and
    /// extract the pool-backed KV as a handoff frame for a decode node.
    ///
    /// The disaggregated path is text-only over the pool-backed Fp16 families
    /// (qwen3 / llama3), so this builds the minimal text sequence (no VLM
    /// embeddings, prompt-prefix adoption, or structured output) and reuses
    /// [`Self::prefill_request_for_handoff`] for the prefill
    /// and extract. `response_tx` carries the first sampled token's text back to
    /// the caller, which is the prefill node's half of the streamed output (the
    /// decode node emits the continuation, mirroring the router's first-token +
    /// decode-token merge). `cancelled` is the client's cancellation flag.
    /// Returns `Ok(None)` when the request hit EOS at the first token, in which
    /// case there is nothing to hand off.
    ///
    /// The empty-prompt guard runs before a cache slot is allocated, so a
    /// rejected request leaks no pool state. Prompts longer than
    /// `--prefill-chunk-size` are prefilled in chunks by
    /// [`Self::prefill_request_for_handoff`] (issue #197).
    #[allow(dead_code)]
    pub(crate) fn prefill_text_request_for_handoff(
        &mut self,
        prompt_tokens: Vec<i32>,
        sampling: mlxcel_core::generate::SamplingConfig,
        max_tokens: usize,
        reasoning_budget: i32,
        thinking_enter_block_on_start: bool,
        response_tx: mpsc::Sender<GenerateEvent>,
        cancelled: Arc<AtomicBool>,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        // Reject model-owned paged families up front, before any allocation or
        // prefill work: the pool-block handoff only supports pool-backed dense
        // Fp16 families (#125). A model-owned family keeps its KV model-internal,
        // so extraction would read unwritten pool tensors (#708). The serving-role
        // loop turns this error into a per-request failure frame and keeps serving.
        if !self.handoff_supported() {
            anyhow::bail!(
                "prefill-role handoff: the disaggregated handoff does not support this model's \
                 {:?} sequence-state backend; only pool-backed dense Fp16 families (qwen3 / \
                 llama3) can be handed off. Model-owned paged families (gemma3 / gemma4 / llama4 \
                 / qwen3_5 / qwen3_next) keep their KV model-internal (issue #708).",
                self.model.sequence_state_layout().backend
            );
        }
        if prompt_tokens.is_empty() {
            anyhow::bail!("prefill-role handoff: empty prompt has no tokens to prefill");
        }
        // Admission cap for the network-facing intake: with chunked prefill
        // supported (issue #197) the old chunk-size bail no longer bounds the
        // accepted prompt, so an oversized PrefillRequestFrame could drive a
        // multi-minute synchronous chunk loop on the node's only prefill
        // worker. 1M tokens matches the CacheIngestLimits philosophy (far
        // above any realistic context; the pool budget is the real bound).
        const MAX_HANDOFF_PROMPT_TOKENS: usize = 1 << 20;
        if prompt_tokens.len() > MAX_HANDOFF_PROMPT_TOKENS {
            anyhow::bail!(
                "prefill-role handoff: prompt of {} tokens exceeds the admission cap \
                 ({MAX_HANDOFF_PROMPT_TOKENS})",
                prompt_tokens.len()
            );
        }
        let thinking = build_handoff_thinking_state(
            self.thinking_token_ids,
            reasoning_budget,
            max_tokens,
            thinking_enter_block_on_start,
            &[],
        )?;
        // Merge the model's configured stop tokens into the request sampling
        // exactly as the single-node intake (`enqueue_request`) does, so the
        // handoff prefill samples the first token identically.
        let sampling = merge_config_stop_tokens(sampling, &self.config_eos);
        let seq_id = self
            .allocate_sequence_state()
            .map_err(|e| anyhow::anyhow!("prefill-role handoff: allocate sequence: {e}"))?;
        let decode_state = StreamingDecodeState::new(&self.tokenizer, &prompt_tokens);
        let seq = SequenceInfo {
            retention: Default::default(),
            seq_id,
            state: SequenceState::Queued,
            prompt_tokens,
            sampling,
            max_tokens,
            eos_token_ids: self.config_eos.clone(),
            // Disaggregated handoffs carry no snapshot; the scheduler
            // resolves the server default at application time (#1439).
            lora_scales: None,
            priority: RequestPriority::default(),
            logprobs_config: mlxcel_core::sampling::LogprobsConfig::default(),
            vlm_embeddings: None,
            images: Vec::new(),
            audio: Vec::new(),
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            decode_state,
            // The prefill-role handoff carries no request options, so no string
            // stop sequences reach it; the matcher is an exact pass-through.
            stop_matcher: StopMatcher::default(),
            prefill_offset: 0,
            prefill_start_offset: 0,
            already_cached_tokens: 0,
            response_tx,
            cancelled,
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
            token_history: Vec::new(),
            sampler_state: None,
            merged_eos: Vec::new(),
            thinking,
            structured: None,
        };
        self.prefill_request_for_handoff(seq)
    }

    /// Decode role (#126 B2b): reconstruct a handed-off sequence onto a fresh pool
    /// slot and register it as a live decode sequence in the active batch, seeded
    /// with the prefill node's generated token(s) so the next decode step feeds
    /// the right token.
    ///
    /// `max_tokens`, `sampling`, and `response_tx` are coordination parameters
    /// supplied by the decode node's request layer (the router / stream bridge in
    /// a real deployment, the test harness in B2c): the KV handoff frame carries
    /// the cache, the prompt token history, and the generated tokens, while the
    /// per-request budget, sampling policy, and output stream stay with the node
    /// that holds the client connection. The deserialization happens once here and
    /// is reused for both the paged restore and the request context.
    ///
    /// Logprobs and structured-output continuation across the handoff are out
    /// of scope for this step. Thinking state is reconstructed from the wire
    /// budget and the tokens already generated by prefill.
    #[allow(dead_code)]
    pub(crate) fn ingest_handoff_as_active(
        &mut self,
        bytes: &[u8],
        max_tokens: usize,
        sampling: mlxcel_core::generate::SamplingConfig,
        reasoning_budget: i32,
        thinking_enter_block_on_start: bool,
        response_tx: mpsc::Sender<GenerateEvent>,
    ) -> anyhow::Result<SequenceId> {
        let geometry = self.ensure_handoff_geometry()?;
        let limits = crate::distributed::kv_cache_serde::CacheIngestLimits::default();
        // Deserialize once: the restore below consumes the KV blocks, and the
        // request context (prompt token history + the prefill node's generated
        // tokens) seeds the live decode sequence.
        let state = crate::distributed::kv_cache_serde::deserialize_cache_state_with_limits(
            bytes, &limits,
        )?;
        let prompt_tokens = state.token_history.clone();
        let generated_tokens = state.generated_tokens.clone();
        // Validate the wire budget and rebuild its phase before restoring any
        // cache blocks, so malformed per-request metadata cannot leak a
        // newly allocated decode sequence.
        let thinking = build_handoff_thinking_state(
            self.thinking_token_ids,
            reasoning_budget,
            max_tokens,
            thinking_enter_block_on_start,
            &generated_tokens,
        )?;
        let seq_id =
            crate::distributed::disaggregated::handoff_impl::ingest_sequence_handoff_state(
                &mut self.cache_pool,
                &self.model,
                &state,
                &limits,
                &geometry,
                DEFAULT_PAGED_BLOCK_SIZE,
            )?;

        let needs_history = sampling.needs_token_history();
        // Rebuild the penalty history exactly as a single-node run would have it
        // after prefill: prompt prefix (when penalties need it) plus whatever the
        // prefill node already generated.
        let mut token_history = initial_token_history(&prompt_tokens, needs_history);
        if needs_history {
            token_history.extend_from_slice(&generated_tokens);
        }
        let merged_eos = merged_eos_token_ids(self.model.eos_token_ids(), &sampling.stop_token_ids);
        // Seed the incremental detokenizer with everything already produced (the
        // prompt plus the handed-off tokens) so the decode node's text continues
        // from the correct boundary.
        let detok_seed: Vec<i32> = prompt_tokens
            .iter()
            .chain(generated_tokens.iter())
            .copied()
            .collect();
        let decode_state = StreamingDecodeState::new(&self.tokenizer, &detok_seed);
        let prefill_offset = prompt_tokens.len();

        let seq = SequenceInfo {
            retention: Default::default(),
            seq_id,
            state: SequenceState::Decoding,
            prompt_tokens,
            sampling,
            max_tokens,
            eos_token_ids: self.config_eos.clone(),
            // Disaggregated handoffs carry no snapshot; the scheduler
            // resolves the server default at application time (#1439).
            lora_scales: None,
            priority: RequestPriority::default(),
            logprobs_config: mlxcel_core::sampling::LogprobsConfig::default(),
            vlm_embeddings: None,
            images: Vec::new(),
            audio: Vec::new(),
            generated_tokens,
            generated_text: String::new(),
            decode_state,
            // The decode-role ingest is driven by the handoff frame, which
            // carries no string stop sequences (#126 B2c scope).
            stop_matcher: StopMatcher::default(),
            prefill_offset,
            prefill_start_offset: 0,
            already_cached_tokens: 0,
            response_tx,
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: Some(Instant::now()),
            token_history,
            sampler_state: None,
            merged_eos,
            thinking,
            structured: None,
        };

        if !self.lora_compatible_with_active(seq.lora_scales.as_ref()) {
            self.release_sequence_caches(seq_id);
            anyhow::bail!(
                "handoff decode admission failed: the active batch runs a different LoRA \
                 adapter configuration"
            );
        }
        if self.active_batch.add(seq).is_err() {
            // No room in the active batch: release the restored KV so the rejected
            // handoff does not leak a sequence or its pool blocks.
            self.release_sequence_caches(seq_id);
            anyhow::bail!("handoff decode admission failed: active batch is full");
        }
        Ok(seq_id)
    }

    /// Decode role (#126 B2b): drive every active sequence to completion, reusing
    /// the same per-tick `execute_decode_step` + `finalize_completed` that the hot
    /// [`Self::run`] loop calls, without touching `run()` itself. Returns once the
    /// active batch has drained (each sequence reached its EOS or token budget).
    #[allow(dead_code)]
    pub(crate) fn decode_handoff_until_idle(&mut self) {
        while self.decode_handoff_step() {}
    }

    /// One decode tick of the handoff drive loop (issue #199): run a single
    /// `execute_decode_step` + `finalize_completed` over the active batch and
    /// report whether any sequence remains. The networked decode role calls
    /// this per tick so it can drain and ship newly produced tokens
    /// incrementally instead of buffering the whole continuation.
    ///
    /// Returns `false` (without stepping) when the active batch is already
    /// empty, so `while decode_handoff_step() {}` is exactly
    /// [`Self::decode_handoff_until_idle`].
    #[allow(dead_code)]
    pub(crate) fn decode_handoff_step(&mut self) -> bool {
        if self.active_batch.is_empty() {
            return false;
        }
        let ids = self.active_batch.sequence_ids();
        self.execute_decode_step(&ids);
        self.finalize_completed();
        !self.active_batch.is_empty()
    }
}
