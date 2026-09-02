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
use mlxcel_core::prefill_span::PrefillSpan;

impl BatchScheduler {
    // ------------------------------------------------------------------
    // Prefill execution (chunked or full)
    // ------------------------------------------------------------------

    /// Write `snapshot` into the runtime-LoRA handles when it differs from
    /// what is applied (#1439). `None` (a sequence admitted without a
    /// snapshot, e.g. a disaggregated handoff) resolves to the server
    /// default at application time. A no-op without runtime-LoRA state.
    pub(crate) fn ensure_lora_applied(&mut self, snapshot: Option<&Arc<Vec<f32>>>) {
        let Some(set) = &self.lora_runtime else {
            return;
        };
        let desired: Arc<Vec<f32>> = match snapshot {
            Some(scales) => scales.clone(),
            None => Arc::new(set.server_scales()),
        };
        if self
            .lora_applied
            .as_ref()
            .is_some_and(|applied| **applied == *desired)
        {
            return;
        }
        tracing::debug!("runtime lora: applying snapshot {:?}", desired);
        set.apply_scales(&desired);
        self.lora_applied = Some(desired);
    }

    /// Whether `seq` may run concurrently with the active batch under the
    /// b10621 lora batching rule (`can_batch_with` requires equal adapter
    /// sets): true when the batch is empty, there is no runtime-LoRA state,
    /// or the snapshots match (#1439).
    pub(crate) fn lora_compatible_with_active(&self, seq_lora: Option<&Arc<Vec<f32>>>) -> bool {
        if self.lora_runtime.is_none() || self.active_batch.is_empty() {
            return true;
        }
        let active = self
            .active_batch
            .iter_sequences()
            .next()
            .and_then(|seq| seq.lora_scales.as_ref());
        batched_window_admits_lora(true, active, seq_lora)
    }

    /// Announce, for the duration of the returned guard, how many positions the
    /// sequence whose prefill is about to run will span.
    ///
    /// A model that picks its RoPE frequency table from the length of the whole
    /// prompt (Phi-3 / Phi-4 LongRoPE) cannot get that from one forward's own
    /// `(cache_offset, seq_len)`, because the scheduler reaches the model with
    /// only a piece of the prompt in three separate ways: a `--prefill-chunk-size`
    /// chunk, the history-boundary segment split off by
    /// `capture_history_boundary_snapshot`, and the suffix left after a
    /// prompt-cache hit. Each of those pieces would resolve the table on its own
    /// and a prompt that straddles the threshold would end up with keys built
    /// from two tables in one cache.
    ///
    /// Every scheduler entry point that runs prefill work for one sequence calls
    /// this first, so the announcement covers every forward that entry point can
    /// reach, and it is scoped to the call so it cannot outlive the tick and be
    /// read by another sequence's decode step. Decode paths deliberately do not
    /// announce: there the pass's own offset is the position. See
    /// [`mlxcel_core::prefill_span`].
    ///
    /// Used by: `execute_full_prefill`, `start_chunked_prefill`,
    /// `continue_chunked_prefill`, `capture_history_boundary_snapshot`
    pub(super) fn announce_prefill_span(&self, seq: &SequenceInfo) -> PrefillSpan {
        mlxcel_core::prefill_span::announce(seq.prompt_tokens.len() as i32)
    }

    /// Prefill a sequence. If `prefill_chunk_size > 0` and the prompt
    /// exceeds one chunk, the prefill is split across multiple ticks with
    /// decode interleaving.
    pub(super) fn execute_prefill(&mut self, _action_id: SequenceId) {
        // Resume a chunked prefill already in progress?
        if self.chunked_prefill_seq.is_some() {
            self.continue_chunked_prefill();
            return;
        }

        // Preemption: if batch is full and preemption is enabled, evict
        // a lower-priority sequence to make room.
        if self.active_batch.is_full() && self.enable_preemption && !self.try_evict_for_preemption()
        {
            // Cannot evict -- skip prefill this tick
            return;
        }

        // b10621's lora batching rule (#1439, upstream `can_batch_with`):
        // a request whose adapter-scale snapshot differs from the active
        // batch's must not join it. Run a decode step instead so the tick
        // makes progress and the batch drains toward admitting the head.
        if let Some(head_lora) = self.prefill_queue.peek_lora_scales()
            && !self.lora_compatible_with_active(head_lora.as_ref())
        {
            tracing::debug!(
                "runtime lora: deferring prefill of a mismatched-snapshot head ({:?})",
                head_lora
            );
            let ids = self.active_batch.sequence_ids();
            if !ids.is_empty() {
                self.execute_decode_step(&ids);
            }
            return;
        }

        let seq = match self.prefill_queue.dequeue() {
            Some(s) => s,
            None => return,
        };

        // #122 b2: paged KV block-budget admission gate. Opt-in — a no-op
        // unless a budget is configured (`free_paged_block_budget()` is `None`
        // otherwise). When the sequence would not fit, this evicts cold
        // prompt-cache prefixes, then preempts running sequences, and as a last
        // resort re-queues the sequence for a later tick (or rejects it if it
        // can never fit the whole budget). Returning `None` means the sequence
        // was deferred or rejected and this tick is done.
        let seq = match self.admit_paged_prefill(seq) {
            Some(s) => s,
            None => return,
        };

        // speculative-decoding burst path.
        //
        // Why: the existing speculative round loops
        // (`MtpGenerator::generate`, `DFlashGenerator::run`, plus their
        // batched B>1 peers) are self-contained drive loops that own
        // prefill + decode + finish in a single function call. Folding
        // them into the scheduler's tick-based decode_single_step would
        // require refactoring every generator into a per-tick step API —
        // a much larger and riskier change. Instead, we run the entire
        // speculative request lifecycle as one "burst" right at prefill
        // time, bypassing the standard prefill → finish_prefill →
        // active_batch → decode pipeline. The classic non-speculative
        // path is bit-exact preserved (the gate's per-sequence
        // preconditions also include: no multimodal payload / VLM
        // embeddings, no structured output, no adopted prompt-cache
        // prefix — see
        // `speculative_burst::should_burst_for_sequence`).
        //
        // adds the B>1 batched burst: when `max_batch_size >
        // 1` and the dequeued head sequence is speculative-eligible, the
        // scheduler collects an equal-prompt-length window of additional
        // eligible requests and drives them all through the batched
        // round-loop driver in one tick. A window of size 1 falls back
        // to the B=1 burst.
        // Apply this request's runtime-LoRA snapshot before any forward it
        // triggers, the speculative burst's self-contained lifecycle
        // included (#1439).
        self.ensure_lora_applied(seq.lora_scales.as_ref());
        let seq = match self.try_speculative_burst(seq) {
            // Burst (B=1 or batched) handled the request(s) end-to-end,
            // or the scheduler took ownership of the sequence as a
            // slice-slot grant waiter (issue #746).
            None => return,
            // Burst declined; route the returned head sequence through
            // the classic prefill path. Any sibling rows that were
            // collected into a declined window were re-routed by
            // `try_speculative_burst` already.
            Some(rejected_seq) => rejected_seq,
        };

        let mut seq = seq;
        if let Err(err) = Self::begin_prefill(&mut seq) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }

        let prompt_len = seq.prompt_tokens.len();

        // Decide: chunked vs full prefill.
        //
        // VLM image requests carry pre-merged input embeddings spanning the
        // full (unpadded) prompt length and are consumed whole by
        // `forward_with_embeddings` (which ignores `input_ids` length when
        // embeddings are present). Chunked prefill would (a) feed the entire
        // embedding sequence on chunk 0 while advancing `prefill_offset` by
        // only one chunk — corrupting the cache/offset bookkeeping — and
        // (b) re-introduce the NA-tile padding/embedding shape mismatch that
        // `execute_full_prefill` guards against. So embeddings-bearing
        // sequences always take the full-prefill path, mirroring the batched
        // dispatch which already forces VLM requests to `execute_full_prefill`.
        if self.prefill_chunk_size > 0
            && prompt_len > self.prefill_chunk_size
            && seq.vlm_embeddings.is_none()
        {
            // Start chunked prefill: process first chunk
            self.start_chunked_prefill(seq);
        } else {
            // Full-prompt prefill (original path)
            self.execute_full_prefill(seq);
        }
    }

    /// Attempt to handle the dequeued head sequence through the
    /// speculative-decoding burst path (B=1 / batched B>1).
    ///
    /// Returns:
    /// - `None` — a burst (B=1 or batched) handled the request(s)
    ///   end-to-end. The caller must `return` immediately; the
    ///   sequence(s) are already finalized and their caches released.
    /// - `Some(seq)` — the burst declined (or the head was not
    ///   speculative-eligible). The caller routes `seq` through the
    ///   classic prefill path. If a batched window had been collected
    ///   and then declined, the sibling rows were re-enqueued onto the
    ///   prefill queue here so they retry on the next tick.
    ///
    /// ## Batched-window collection
    ///
    /// When `max_batch_size > 1` and the head is speculative-eligible,
    /// this method drains an *equal-prompt-length* window of additional
    /// eligible requests from the front of the head's priority lane (via
    /// [`super::queue::PrefillQueue::drain_matching_window`]). The
    /// batched MTP target adapter requires a rectangular `[B, L]`
    /// prefill, and equal-length prompts make the batched prefill
    /// byte-identical to B separate B=1 prefills (acceptance item 1). The window also requires matching `max_tokens` and
    /// sampling config so the single per-window sampler / budget the
    /// batched round loop takes is correct for every row. B > 1 also
    /// applies [`crate::server::batch::speculative_burst::can_join_batched_burst_window`]
    /// so requests that need payloads unsupported by the batched round
    /// loops (currently logprobs) stay on the B = 1 burst path. A window
    /// that collapses to size 1 falls back to the B=1 burst.
    /// Whether to run the B=1 MTP burst for the next singleton request.
    ///
    /// Routes through the adaptive policy (issue #333) when one is attached: it
    /// forces MTP on while profiling, then returns its settled verdict. Without
    /// a policy (adaptive disabled) this is the pre-#333 static per-hardware
    /// gate ([`crate::server::batch::speculative_burst::mtp_b1_burst_enabled`]), which reads
    /// `MLXCEL_ENABLE_MTP_B1` and the hardware default. A pure read with no
    /// per-token cost.
    pub(super) fn mtp_b1_should_run(&self) -> bool {
        match &self.mtp_policy {
            Some(policy) => policy.should_attempt_b1(),
            None => crate::server::batch::speculative_burst::mtp_b1_burst_enabled(
                self.model.supports_batching(),
            ),
        }
    }

    /// Whether `seq`, arriving while the speculative slice slot is busy,
    /// can be parked as a slot-grant waiter (issue #746) instead of
    /// falling back to classic decode.
    ///
    /// Re-applies the request-independent gates
    /// [`Self::start_mtp_slice_b1`] applies, so a parked waiter's later
    /// promotion is expected to succeed: MTP dispatch with the tick slice
    /// enabled, a usable block size, a Gemma 4 family target, a
    /// non-degenerate adopted prefix, and a B=1 verdict from the adaptive
    /// policy (`seq` already passed `should_burst_for_sequence` at the
    /// caller). Anything else keeps the pre-#746 classic fallback. The one
    /// gate that can drift while the request waits, the adaptive policy's
    /// B=1 verdict, is re-checked explicitly at promotion time
    /// ([`Self::promote_next_speculative_grantee`]); a then-declined
    /// waiter routes to classic prefill
    /// ([`Self::route_declined_slice_waiter_to_classic`]).
    pub(super) fn can_wait_for_slice_grant(&self, seq: &SequenceInfo) -> bool {
        let block_size = match &self.speculative_dispatch {
            crate::server::SpeculativeDispatch::Mtp { block_size, .. } => *block_size as usize,
            _ => return false,
        };
        block_size >= 2
            && crate::server::batch::speculative_slice::mtp_tick_slice_enabled()
            && crate::server::batch::speculative_burst::mtp_capable_target(&self.model, block_size)
            && !seq.prompt_tokens.is_empty()
            && crate::server::batch::speculative_burst::mtp_prefill_suffix_start(
                seq.prefill_start_offset,
                seq.prompt_tokens.len(),
            )
            .is_some()
            && self.mtp_b1_should_run()
    }

    /// #715: whether the head-of-queue prompt is short enough to enter the
    /// batched-prefill path under the padded-token budget.
    ///
    /// A batched cohort pads every row to the window's longest prompt `L`, so
    /// the head can only ever join a `>= 2`-row batch when `2 * head_len` stays
    /// within the budget. When it cannot (or the queue is empty), batching
    /// would collapse to a single-row unchunked `[L, L]` forward, so the head
    /// is instead routed to the normal chunk-aware single-sequence path. `0`
    /// (uncapped) always admits.
    pub(super) fn batched_prefill_admits_head(&self) -> bool {
        // #1439: a head whose runtime-LoRA snapshot differs from the active
        // batch's cannot join it at all, batched or not. Refusing here routes
        // the tick to `execute_prefill`, which owns the defer-and-decode path
        // that drains the batch until the head can be admitted.
        if let Some(head_lora) = self.prefill_queue.peek_lora_scales()
            && !self.lora_compatible_with_active(head_lora.as_ref())
        {
            return false;
        }
        let budget = self.max_batch_prefill_tokens;
        if budget == 0 {
            return true;
        }
        match self.prefill_queue.peek_prompt_len() {
            Some(head_len) => head_len.saturating_mul(2) <= budget,
            None => false,
        }
    }

    /// Batched prefill: drain up to `max_batch_prefill` requests from the
    /// prefill queue and process eligible cold text rows in a single forward
    /// pass.
    ///
    /// The drained window can mix requests the padded batched path supports
    /// (cold text: zero KV-history offset, no custom embeddings) with requests
    /// it does not (adopted prompt-cache prefixes, VLM / custom embeddings).
    /// #332: instead of falling the whole window back to sequential prefill
    /// when it contains one incompatible request, the window is split into
    /// cohorts ([`plan_prefill_cohorts`]). Cold cohorts run batched; everything
    /// else takes the offset-aware single-sequence path. Cohorts are dispatched
    /// in window order, and the queue dequeues in priority order, so request
    /// priority / FIFO fairness is preserved across cohort boundaries.
    pub(super) fn execute_batched_prefill(&mut self) {
        let batch_size = self.max_batch_prefill.min(self.prefill_queue.len());

        // Collect up to `batch_size` requests from the queue. The queue
        // dequeues in priority order (high lane, then normal, then low; FIFO
        // within a lane), so `seqs` is already in priority order.
        //
        // #715: the drain is also bounded by the padded-token budget
        // (`max_batch_prefill_tokens`). Because the padded batched path pads
        // every row to the window's longest prompt `L`, the drained window
        // costs `rows * L` padded tokens and materializes an `O(rows * L^2)`
        // FP32 mask. Draining stops before a row that would push `rows * L`
        // past the budget; the remaining rows stay queued and are prefilled on
        // a later tick (short ones re-batch, long ones take the chunked
        // single-sequence path). The head row is always taken so the drain
        // makes forward progress; the dispatch-time guard
        // ([`Self::batched_prefill_admits_head`]) has already kept a head too
        // long to batch out of this path entirely.
        let budget = self.max_batch_prefill_tokens;
        // #1439: b10621's `can_batch_with` requires an equal adapter set, and
        // a padded batched forward runs every row under one set of shared
        // scale handles, so the window stops at the first row whose snapshot
        // differs from the head's. Those rows stay queued and prefill on a
        // later tick under their own snapshot; without this the second
        // request's adapters would silently serve the first one's output.
        let head_lora: Option<Arc<Vec<f32>>> = self.prefill_queue.peek_lora_scales().flatten();
        let lora_partitioned = self.lora_runtime.is_some();
        let mut seqs: Vec<SequenceInfo> = Vec::with_capacity(batch_size);
        let mut window_max_len = 0usize;
        while seqs.len() < batch_size {
            let Some(next_len) = self.prefill_queue.peek_prompt_len() else {
                break;
            };
            if !batched_window_admits(seqs.len(), window_max_len, next_len, budget) {
                break;
            }
            let next_lora = self.prefill_queue.peek_lora_scales().flatten();
            if !batched_window_admits_lora(lora_partitioned, head_lora.as_ref(), next_lora.as_ref())
            {
                break;
            }
            let Some(seq) = self.prefill_queue.dequeue() else {
                break;
            };
            window_max_len = window_max_len.max(seq.prompt_tokens.len());
            seqs.push(seq);
        }

        if seqs.is_empty() {
            return;
        }

        // Classify each row, then plan cohorts. A row is "cold" only when it
        // has no VLM / custom embeddings AND no adopted prompt-cache prefix.
        // That is exactly the precondition the padded batched path assumes (a
        // zero cache offset for every row), so the planner's guarantee that a
        // BatchedCold cohort holds only cold rows is what keeps cache offsets
        // correct: an adopted prefix can never be folded into a batch and have
        // its KV resumed at the wrong position.
        let can_batch = self.model.supports_batched_prefill();
        let can_pad = self.model.supports_padded_prefill();
        let rows: Vec<PrefillRow> = seqs
            .iter()
            .map(|s| PrefillRow {
                is_cold: s.vlm_embeddings.is_none() && s.prefill_start_offset == 0,
                prompt_len: s.prompt_tokens.len(),
            })
            .collect();
        let plan = plan_prefill_cohorts(&rows, can_batch, can_pad);

        // Move sequences into index-addressable slots so each cohort can take
        // ownership of exactly its members. Dispatching cohorts in plan order
        // reproduces window (priority) order across the cohort boundaries.
        let mut slots: Vec<Option<SequenceInfo>> = seqs.into_iter().map(Some).collect();
        // Every row in the window shares `head_lora` (the drain above kept it
        // that way), so one application covers every cohort's forwards,
        // batched and sequential alike (#1439).
        self.ensure_lora_applied(head_lora.as_ref());
        for cohort in plan {
            match cohort.kind {
                // Behavior note (#332): a cold row that previously fell back to
                // *sequential* prefill (because the collected window held an
                // incompatible sibling) now runs batched here. A padded batched
                // forward (B > 1) is not bitwise-identical to single-sequence
                // prefill on Metal, so such a row's greedy decode can differ
                // from its old sequential output by an early near-tie token flip
                // (the documented #203 / #325 / #326 jitter class). That is the
                // intended effect of cohort splitting, not a correctness
                // regression: the guarantee is that a cohort-split cold row
                // decodes identically to the same row in an all-cold batched
                // window of the same composition (pinned by
                // scheduler_cohort_parity_tests).
                PrefillCohortKind::BatchedCold => {
                    let remaining_capacity = self
                        .active_batch
                        .max_size()
                        .saturating_sub(self.active_batch.len());
                    if remaining_capacity == 0 {
                        self.requeue_prefill_window_front(&mut slots);
                        return;
                    }
                    let group: Vec<SequenceInfo> = cohort
                        .members
                        .iter()
                        .take(remaining_capacity)
                        .filter_map(|&i| slots[i].take())
                        .collect();
                    if cohort.members.len() > remaining_capacity {
                        self.requeue_prefill_window_front(&mut slots);
                    }
                    self.run_padded_batched_prefill(group);
                    if cohort.members.len() > remaining_capacity {
                        return;
                    }
                }
                PrefillCohortKind::Sequential => {
                    for &i in &cohort.members {
                        if self.active_batch.is_full() {
                            self.requeue_prefill_window_front(&mut slots);
                            return;
                        }
                        let Some(mut seq) = slots[i].take() else {
                            continue;
                        };
                        if let Err(err) = Self::begin_prefill(&mut seq) {
                            tracing::error!("Batched prefill state transition error: {err}");
                            self.abort_sequence(seq, &err);
                            continue;
                        }
                        self.execute_full_prefill(seq);
                    }
                }
            }
        }
    }

    /// Return an already-drained prefill window to the front of the queue.
    ///
    /// Batched prefill drains optimistically, then cohort planning can route
    /// VLM/adopted rows through the sequential prefill path. If earlier rows in
    /// the same window filled the decode batch (for example `--parallel 1` with
    /// two concurrent image requests), the untouched rows must wait for a later
    /// tick instead of being prefilled into a nonexistent reserved slot.
    pub(super) fn requeue_prefill_window_front(&mut self, slots: &mut [Option<SequenceInfo>]) {
        let mut remaining: Vec<SequenceInfo> = slots.iter_mut().filter_map(Option::take).collect();
        for seq in remaining.drain(..).rev() {
            if let Err(rejected) = self.prefill_queue.enqueue_front(seq) {
                self.prompt_cache_seq_ctx.remove(&rejected.seq_id);
                self.release_sequence_caches(rejected.seq_id);
                let _ = rejected.response_tx.send(GenerateEvent::Error(
                    "Server busy: prefill queue full".to_string(),
                ));
            }
        }
    }

    /// Run a single padded batched prefill over a cohort of cold text rows.
    ///
    /// Every sequence in `seqs` must be cold (zero KV-history offset, no custom
    /// embeddings); [`plan_prefill_cohorts`] guarantees this, so the pipeline
    /// below can assume a zero cache offset for every row. Sequences are padded
    /// to the longest prompt in the cohort (aligned to a 32-token tile on M5+),
    /// each with a per-sequence causal + padding mask, and run in one forward
    /// pass. On any error the affected sequences fall back to the
    /// single-sequence prefill path so no request is lost.
    pub(super) fn run_padded_batched_prefill(&mut self, mut seqs: Vec<SequenceInfo>) {
        // Defensive: the planner only emits BatchedCold cohorts of >= 2 rows,
        // but keep the empty / single cases correct if called directly.
        if seqs.is_empty() {
            return;
        }
        if seqs.len() == 1 {
            let mut seq = seqs.remove(0);
            if let Err(err) = Self::begin_prefill(&mut seq) {
                tracing::error!("Batched prefill state transition error: {err}");
                self.abort_sequence(seq, &err);
                return;
            }
            self.execute_full_prefill(seq);
            return;
        }

        // Transition all sequences to Prefilling up front so every fallback
        // below routes through `execute_full_prefill` from the correct state.
        for seq in &mut seqs {
            if let Err(err) = Self::begin_prefill(seq) {
                tracing::error!("Batched prefill state transition error: {err}");
            }
        }

        let b = seqs.len();
        let max_len = seqs.iter().map(|s| s.prompt_tokens.len()).max().unwrap();
        let can_pad_prefill = self.model.supports_padded_prefill();
        if !can_pad_prefill && seqs.iter().any(|s| s.prompt_tokens.len() != max_len) {
            // Should not happen for a planner-approved cohort (it only batches
            // equal-length rows on equal-length-only models), but stay safe.
            tracing::debug!(
                "batched prefill: cohort fell back to sequential (model requires equal prompt lengths)"
            );
            for seq in seqs {
                self.execute_full_prefill(seq);
            }
            return;
        }

        let padded_len = if can_pad_prefill && should_align_prefill() {
            align_to_na_tile(max_len)
        } else {
            max_len
        };

        tracing::debug!("batched prefill: {} requests, padded to {}", b, padded_len);

        // Build padded input: [B, padded_len]
        let mut flat_tokens: Vec<i32> = Vec::with_capacity(b * padded_len);
        for seq in &seqs {
            let tokens = &seq.prompt_tokens;
            flat_tokens.extend_from_slice(tokens);
            // Pad with 0 to padded_len
            flat_tokens.extend(std::iter::repeat_n(0, padded_len - tokens.len()));
        }
        let input = mlxcel_core::from_slice_i32(&flat_tokens, &[b as i32, padded_len as i32]);

        // Build per-sequence attention masks and collect cache pointers.
        // Each mask has shape [padded_len, padded_len]. Stacking on axis 0
        // produces [B, padded_len, padded_len], which model batched-prefill
        // paths slice per sequence into [padded_len, padded_len].
        let stacked_mask = if seqs.iter().any(|s| s.prompt_tokens.len() != padded_len) {
            let mut batch_masks: Vec<UniquePtr<mlxcel_core::MlxArray>> = Vec::with_capacity(b);
            for seq in &seqs {
                let actual = seq.prompt_tokens.len() as i32;
                let padded = padded_len as i32;
                let mask = create_padded_prefill_mask(actual, padded, 0);
                batch_masks.push(mask);
            }
            Some(mlxcel_core::stack_owned(&batch_masks, 0))
        } else {
            None
        };

        let batch_ids: Vec<SequenceId> = seqs.iter().map(|seq| seq.seq_id).collect();
        let mut batch_caches = match self.cache_pool.get_batch_caches_mut(&batch_ids) {
            Ok(caches) => caches,
            Err(err) => {
                tracing::warn!("batched prefill: {err}, falling back");
                // Re-queue all sequences for sequential processing.
                for seq in seqs {
                    self.execute_full_prefill(seq);
                }
                return;
            }
        };

        if batch_caches.len() != b {
            // Re-queue all sequences for sequential processing.
            for seq in seqs {
                self.execute_full_prefill(seq);
            }
            return;
        }

        // Prefill, and deliberately NOT covered by `announce_prefill_span`: this
        // pass starts at offset 0 and spans `padded_len`, the longest row, so a
        // whole-prompt RoPE table already resolves correctly from the pass
        // itself for that row. One announcement is a scalar and cannot say
        // anything different for the shorter rows, which share this cohort's
        // decision the same way they already share its padded mask geometry.
        // A cohort can only straddle a table threshold when chunking is off or
        // its chunk exceeds the threshold, since any longer prompt takes the
        // chunked path instead. See `mlxcel_core::prefill_span`.
        // Single batched forward pass: [B, padded_len] → [B, padded_len, vocab]
        let raw_logits = self.model.forward_batched_with_context_and_ids(
            &input,
            Some(&batch_ids),
            &mut batch_caches,
            stacked_mask.as_deref(),
            None,
        );

        // Release the cache_pool borrow before the guarded eval touches
        // `&mut self`; the per-sequence loop below re-borrows caches anyway.
        drop(batch_caches);

        // #822: force-evaluate the cohort's prefill graph through the fallible
        // boundary. This single eval covers the whole batch, so an MLX C++ throw
        // (graph-cache abort, allocation failure) fails every sequence in this
        // cohort with the error instead of aborting the process.
        if let Err(msg) =
            self.record_eval_outcome(mlxcel_core::try_eval(&raw_logits).map_err(|e| e.to_string()))
        {
            for seq in seqs {
                self.abort_sequence(seq, &msg);
            }
            self.eval_failures_exhausted();
            return;
        }
        mlxcel_core::clear_memory_cache();

        // Process per-sequence results.
        for (i, mut seq) in seqs.into_iter().enumerate() {
            let actual_len = seq.prompt_tokens.len();
            let padded = padded_len;

            // Extract logits at the last real token position: index [i, actual_len-1, :]
            let last_pos = actual_len as i32 - 1;
            let vocab = {
                let shape = mlxcel_core::array_shape(&raw_logits);
                shape[2]
            };
            let seq_logits = mlxcel_core::slice(
                &raw_logits,
                &[i as i32, last_pos, 0],
                &[i as i32 + 1, last_pos + 1, vocab],
            );

            // Trim padding positions from this sequence's KV cache so that the
            // decode phase starts with the correct cache offset.
            let excess = (padded - actual_len) as i32;
            if excess > 0
                && let Some(caches) = self.cache_pool.get_caches_mut(seq.seq_id)
            {
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }

            self.sync_sequence_storage(seq.seq_id);

            seq.prefill_offset = actual_len;
            self.batch_observability.record_prefill_start(actual_len);

            let eos_tokens =
                merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
            let needs_history = seq.sampling.needs_token_history();
            let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

            self.finish_prefill(seq, seq_logits, eos_tokens, token_history, needs_history);
        }
    }

    /// Full-prompt prefill: process the entire prompt in one pass.
    ///
    /// when `seq.prefill_start_offset > 0`, a
    /// prompt-cache hit has installed the first `prefill_start_offset` tokens
    /// of KV state on this sequence. Only the suffix tokens are fed to the
    /// model. The VLM-prefix path deliberately opts out of cache adoption at
    /// the enqueue site, so this branch never has to mix the two.
    pub(super) fn execute_full_prefill(&mut self, mut seq: SequenceInfo) {
        let _span = self.announce_prefill_span(&seq);
        // Split off the history-boundary segment first (issue #1143). On the
        // vast majority of requests this is an early-return; when it does run
        // it advances `prefill_start_offset`, so everything below sees the
        // remaining suffix exactly as it would see an adopted-prefix suffix.
        if let Err(msg) = self.capture_history_boundary_snapshot(&mut seq) {
            self.abort_sequence(seq, &msg);
            self.eval_failures_exhausted();
            return;
        }
        // `cached` reports the ADOPTED prefix only. `prefill_start_offset` may
        // also have been advanced past a freshly-forwarded history-boundary
        // segment (issue #1143), and reporting those as cached would misread as
        // reuse in a trace. `start` carries the real cursor.
        let _span = tracing::info_span!(
            "prefill",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
            cached = seq.already_cached_tokens,
            start = seq.prefill_start_offset,
        )
        .entered();
        // Only the suffix enters the prefill counters — the first
        // `prefill_start_offset` tokens were resolved from the adopted
        // detached cache with zero model work.
        let suffix_len = seq.prompt_tokens.len() - seq.prefill_start_offset;
        self.batch_observability.record_prefill_start(suffix_len);

        // Non-batching models use internal RefCell caches that are shared
        // across all sequences.  Reset them now (at prefill time) rather
        // than at enqueue time so that queued requests don't corrupt an
        // in-flight generation.
        if !self.model.supports_batching() {
            let _ = self.model.make_caches();
        }

        let eos_tokens =
            merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
        let needs_history = seq.sampling.needs_token_history();
        let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

        // Feed only the suffix tokens to the model when a cached prefix was
        // adopted. For cold prefills `start == 0` and this is identical to
        // the legacy behavior.
        let suffix_tokens: Vec<i32> = seq.prompt_tokens[seq.prefill_start_offset..].to_vec();

        // Run prefill (with or without VLM embeddings).
        // On M5+ hardware pad the prompt to a 32-token tile boundary for
        // optimal Neural Accelerator throughput.
        let actual_len = suffix_tokens.len();
        // VLM image requests inject pre-merged input embeddings at the real
        // (unpadded) sequence length and run through `forward_with_embeddings`
        // below. NA-tile alignment pads only the token-id vector and builds a
        // matching padded mask — it does NOT pad the injected embeddings. So
        // aligning here would hand the model a padded mask (e.g. 320x320) that
        // cannot broadcast against the unpadded embeddings (e.g. [1,H,293,293]),
        // aborting the process. Skip alignment when embeddings are present; the
        // text backbone then builds a causal mask sized to the embeddings,
        // matching the CLI generate path. Token-id (text-only) prefill — for
        // VLMs and plain text models alike — is unaffected.
        let (effective_tokens, pad_mask_opt) = if self.model.supports_padded_prefill()
            && should_align_prefill()
            && seq.vlm_embeddings.is_none()
        {
            let padded_len = align_to_na_tile(actual_len);
            if padded_len > actual_len {
                let mut padded = suffix_tokens.clone();
                padded.resize(padded_len, 0);
                // The padding mask anchors to the adopted cache offset so
                // the newly-prefilled positions see the correct KV-history
                // positions on M5+ hardware.
                let mask = create_padded_prefill_mask(
                    actual_len as i32,
                    padded_len as i32,
                    seq.prefill_start_offset as i32,
                );
                (padded, Some(mask))
            } else {
                (suffix_tokens.clone(), None)
            }
        } else {
            (suffix_tokens.clone(), None)
        };

        let eff_len = effective_tokens.len() as i32;
        let input = mlxcel_core::from_slice_i32(&effective_tokens, &[1, eff_len]);
        // #822: the VLM branch force-evaluates the prefill graph while `caches`
        // still borrows the cache pool, so capture the fallible eval outcome
        // here and act on it below once the borrow has ended.
        let mut prefill_eval: Option<Result<(), String>> = None;
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found for sequence during prefill");
                    return;
                }
            };

            let raw_logits = if let Some(ref embeddings) = seq.vlm_embeddings {
                // VLM path: apply provided mask or the tile-alignment mask.
                match prepared_embedding_refs(embeddings) {
                    Ok((input_embeds, caller_mask)) => {
                        // Caller-supplied mask takes precedence; tile-alignment mask
                        // is used only when the caller does not provide one.
                        let effective_mask =
                            caller_mask.or(pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()));
                        let logits = self
                            .model
                            .forward_last_logits_with_embeddings_and_sequence_id(
                                &input,
                                Some(input_embeds),
                                Some(seq.seq_id),
                                caches,
                                effective_mask,
                                actual_len.saturating_sub(1),
                            );
                        prefill_eval =
                            Some(mlxcel_core::try_eval(&logits).map_err(|e| e.to_string()));
                        self.model.after_prefill();
                        logits
                    }
                    Err(err) => {
                        self.abort_sequence(seq, &err.to_string());
                        return;
                    }
                }
            } else {
                self.model.forward_last_logits_with_sequence_id(
                    &input,
                    Some(seq.seq_id),
                    caches,
                    pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
                    actual_len.saturating_sub(1),
                )
            };

            // The sequence-aware last-logits hook already extracts the last
            // real row. Trim padding from KV caches so decode begins at the
            // correct cache offset.
            if pad_mask_opt.is_some() && effective_tokens.len() > actual_len {
                let padded_len = effective_tokens.len();
                // Trim padding positions from all KV caches.
                let excess = (padded_len - actual_len) as i32;
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }
            raw_logits
        };

        // #822: if the VLM prefill eval threw, fail just this request and, if the
        // backend has failed too many times in a row, shut the scheduler down.
        if let Some(outcome) = prefill_eval
            && let Err(msg) = self.record_eval_outcome(outcome)
        {
            self.abort_sequence(seq, &msg);
            self.eval_failures_exhausted();
            return;
        }

        self.sync_sequence_storage(seq.seq_id);

        // H2: enforce the `--max-kv-size` cap at the end of a
        // full prefill before the sequence transitions to decode. A long
        // prompt can overshoot the cap during a single forward pass; without
        // this trim the first decode step would start with a too-wide live
        // window. With no cap configured this is a cheap early-return.
        self.enforce_max_kv_size_for(seq.seq_id, seq.retention);

        mlxcel_core::clear_memory_cache();
        // `prefill_offset` is a cursor into `prompt_tokens`, so it must
        // include the adopted prefix even though those tokens bypassed the
        // forward pass.
        seq.prefill_offset = seq.prefill_start_offset + actual_len;

        self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
    }

    /// Begin a chunked prefill: process the first chunk and store the
    /// sequence for continuation on subsequent ticks.
    ///
    /// `seq.prefill_start_offset` skips over the
    /// leading tokens that the adopted prompt-cache entry already covers,
    /// so the first chunk starts *after* the cached prefix.
    pub(super) fn start_chunked_prefill(&mut self, mut seq: SequenceInfo) {
        let _span = self.announce_prefill_span(&seq);
        // Same history-boundary split as `execute_full_prefill` (issue #1143).
        // It advances `prefill_start_offset`, which is exactly the cursor the
        // first chunk starts from, so the chunk loop below needs no changes.
        if let Err(msg) = self.capture_history_boundary_snapshot(&mut seq) {
            self.abort_sequence(seq, &msg);
            self.eval_failures_exhausted();
            return;
        }
        let _span = tracing::info_span!(
            "chunked_prefill_start",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
            chunk_size = self.prefill_chunk_size,
            cached = seq.already_cached_tokens,
            start = seq.prefill_start_offset,
        )
        .entered();

        // Reset internal caches for non-batching models (same as execute_full_prefill).
        if !self.model.supports_batching() {
            let _ = self.model.make_caches();
        }

        let chunk_size = self.prefill_chunk_size;
        let chunk_range = match next_chunked_prefill_range(
            seq.prompt_tokens.len(),
            seq.prefill_start_offset,
            chunk_size,
        ) {
            Some(range) => range,
            None => {
                self.abort_sequence(seq, "Chunked prefill start had no suffix tokens to process");
                return;
            }
        };
        // Counter reflects only the work the model actually runs.
        let suffix_len = seq.prompt_tokens.len() - seq.prefill_start_offset;
        self.batch_observability.record_prefill_start(suffix_len);

        let start = chunk_range.start;
        let end = chunk_range.end;
        let chunk = &seq.prompt_tokens[start..end];

        // Align the first chunk to a 32-token tile boundary on M5+ hardware.
        let actual_chunk_len = chunk.len();
        let (eff_chunk, pad_mask_opt) =
            if self.model.supports_padded_prefill() && should_align_prefill() {
                let padded_len = align_to_na_tile(actual_chunk_len);
                if padded_len > actual_chunk_len {
                    let mut padded = chunk.to_vec();
                    padded.resize(padded_len, 0);
                    // Mask anchored to the KV offset the adopted prefix already
                    // installed (starts at zero for cold prefills).
                    let mask = create_padded_prefill_mask(
                        actual_chunk_len as i32,
                        padded_len as i32,
                        start as i32,
                    );
                    (padded, Some(mask))
                } else {
                    (chunk.to_vec(), None)
                }
            } else {
                (chunk.to_vec(), None)
            };

        let eff_len = eff_chunk.len() as i32;
        let input = mlxcel_core::from_slice_i32(&eff_chunk, &[1, eff_len]);
        // #822: this chunk's forward is force-evaluated while `caches` still
        // borrows the cache pool, so capture the fallible eval outcome and act
        // on it below once the borrow has ended. Deferred-init: every path that
        // reaches the check below assigns it exactly once; the others return.
        let prefill_eval: Option<Result<(), String>>;
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found for sequence during chunked prefill");
                    return;
                }
            };

            // VLM embeddings are applied only on the first chunk.
            let logits = if let Some(ref embeddings) = seq.vlm_embeddings {
                match prepared_embedding_refs(embeddings) {
                    Ok((input_embeds, caller_mask)) => {
                        let effective_mask =
                            caller_mask.or(pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()));
                        let logits = self
                            .model
                            .forward_last_logits_with_embeddings_and_sequence_id(
                                &input,
                                Some(input_embeds),
                                Some(seq.seq_id),
                                caches,
                                effective_mask,
                                actual_chunk_len.saturating_sub(1),
                            );
                        prefill_eval =
                            Some(mlxcel_core::try_eval(&logits).map_err(|e| e.to_string()));
                        self.model.after_prefill();
                        logits
                    }
                    Err(err) => {
                        self.abort_sequence(seq, &err.to_string());
                        return;
                    }
                }
            } else {
                let logits = self.model.forward_last_logits_with_sequence_id(
                    &input,
                    Some(seq.seq_id),
                    caches,
                    pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
                    actual_chunk_len.saturating_sub(1),
                );
                prefill_eval = Some(mlxcel_core::try_eval(&logits).map_err(|e| e.to_string()));
                logits
            };

            // Trim padding positions from KV caches when the chunk was padded.
            if pad_mask_opt.is_some() && eff_chunk.len() > actual_chunk_len {
                let excess = (eff_chunk.len() - actual_chunk_len) as i32;
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }
            logits
        };

        // #822: if this chunk's eval threw, fail just this request and, if the
        // backend has failed too many times in a row, shut the scheduler down.
        if let Some(outcome) = prefill_eval
            && let Err(msg) = self.record_eval_outcome(outcome)
        {
            self.abort_sequence(seq, &msg);
            self.eval_failures_exhausted();
            return;
        }

        self.sync_sequence_storage(seq.seq_id);

        // H2: enforce the `--max-kv-size` cap after each
        // prefill chunk so the live window cannot grow unbounded across
        // chunks of a long prompt. A 100k-token prompt with `--max-kv-size
        // 4096` would otherwise see the cap engage only after the entire
        // prefill completes — defeating the memory-bound the operator
        // configured. With no cap configured this is a cheap early-return.
        self.enforce_max_kv_size_for(seq.seq_id, seq.retention);

        mlxcel_core::clear_memory_cache();
        seq.prefill_offset = end;
        // One `prompt_progress` frame per evaluated chunk, b10621's per-batch
        // -iteration cadence (#1477).
        seq.report_prefill_progress(end);
        // Count the first chunk too. Before issue #908 only
        // `continue_chunked_prefill` recorded, so the counter reported
        // continuations and read as zero for a prompt that ran chunk 0 and was
        // then starved, which is precisely the state a reader needs to see.
        self.batch_observability.record_prefill_chunk();

        tracing::debug!(
            "Chunked prefill: seq {} chunk 0..{end}/{} tokens",
            seq.seq_id,
            seq.prompt_tokens.len()
        );

        // The chunked-vs-full decision in `prefill_sequence` keys off the
        // *full* prompt length, but the work we just ran covers only the
        // suffix `[prefill_start_offset..]`. When a prompt-cache hit adopts a
        // long prefix, that suffix can fit entirely in chunk 0 even though the
        // full prompt cleared the chunking threshold — so this first chunk has
        // already reached the end of the prompt and there is nothing to
        // continue. Finish the prefill now (mirroring the final-chunk handling
        // in `continue_chunked_prefill`). Storing the sequence for
        // continuation instead would feed an empty `[end..end]` chunk on the
        // next tick, producing a zero-length forward whose `[1, 0, vocab]`
        // logits crash in `slice_last_logits` (issue #179).
        if chunk_range.is_terminal {
            let eos_tokens =
                merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
            let needs_history = seq.sampling.needs_token_history();
            let token_history = initial_token_history(&seq.prompt_tokens, needs_history);
            self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
            return;
        }

        // Store the sequence for continuation
        self.chunked_prefill_seq = Some(seq);
    }

    /// Continue a chunked prefill that is already in progress.
    ///
    /// Returns `true` when a chunk forward actually ran. Every early return
    /// here (no parked sequence, an empty range, a missing cache, an exhausted
    /// eval) reports `false`, which is what lets the issue #908 mixed-step
    /// counter stay an honest dispatch proof instead of counting ticks on which
    /// no prefill work happened.
    ///
    /// The per-chunk `clear_memory_cache()` below is suppressed whenever a
    /// decode batch is live alongside this chunk. The decode path deliberately
    /// clears on a cadence instead (`cache_clear_interval()`, 256 tokens on
    /// Metal and off by default on CUDA, because a per-step clear churns the
    /// pool and defeats CUDA-graph reuse, ml-explore/mlx#2358). Before #908
    /// this function was only ever reachable with an empty active batch, so its
    /// per-chunk clear never touched a decode hot path.
    ///
    /// #908 introduced the first interleaved caller (`MixedStep`) and passed an
    /// explicit `mixed_tick` flag to suppress the clear. #1011 makes the
    /// DEFAULT policy interleaved too, via the fairness grant, so the condition
    /// is now read from the active batch rather than passed in: a caller that
    /// forgot the flag would silently put an allocator-pool clear on the decode
    /// hot path for the whole duration of a long prefill, inflating exactly the
    /// inter-token latency this issue has to measure. There is one source of
    /// truth for "is decode live" and it is the active batch.
    pub(super) fn continue_chunked_prefill(&mut self) -> bool {
        // Re-apply the parked sequence's runtime-LoRA snapshot (#1439): the
        // interleaved decode batch may have applied its own between chunks.
        let chunked_lora = self
            .chunked_prefill_seq
            .as_ref()
            .and_then(|seq| seq.lora_scales.clone());
        if self.chunked_prefill_seq.is_some() {
            self.ensure_lora_applied(chunked_lora.as_ref());
        }
        let mut seq = match self.chunked_prefill_seq.take() {
            Some(s) => s,
            None => return false,
        };
        let _span = self.announce_prefill_span(&seq);
        // Interleaved with decode (a #1011 grant or a #908 mixed step) rather
        // than running against a drained batch.
        let decode_batch_live = !self.active_batch.is_empty();

        let _span = tracing::info_span!(
            "chunked_prefill_continue",
            seq_id = %seq.seq_id,
            offset = seq.prefill_offset,
            total = seq.prompt_tokens.len(),
        )
        .entered();

        let chunk_size = self.prefill_chunk_size;
        let offset = seq.prefill_offset;
        let total = seq.prompt_tokens.len();
        let chunk_range = match next_chunked_prefill_range(total, offset, chunk_size) {
            Some(range) => range,
            None => {
                self.abort_sequence(
                    seq,
                    "Chunked prefill continuation had no remaining tokens to process",
                );
                return false;
            }
        };
        self.batch_observability.record_prefill_chunk();

        let end = chunk_range.end;
        let chunk = &seq.prompt_tokens[offset..end];

        // Align each continuation chunk to a 32-token tile boundary on M5+.
        let actual_chunk_len = chunk.len();
        // For non-batching models the scheduler's dummy caches always have
        // offset=0.  Use the prefill_offset (number of tokens already
        // processed) as the KV offset instead, which is accurate regardless
        // of whether the model uses internal or scheduler-managed caches.
        let kv_offset = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found during chunked prefill continuation");
                    return false;
                }
            };
            if self.model.supports_batching() {
                caches.first().map_or(0, |c| c.offset)
            } else {
                offset as i32
            }
        };
        let (eff_chunk, pad_mask_opt) =
            if self.model.supports_padded_prefill() && should_align_prefill() {
                let padded_len = align_to_na_tile(actual_chunk_len);
                if padded_len > actual_chunk_len {
                    let mut padded = chunk.to_vec();
                    padded.resize(padded_len, 0);
                    let mask = create_padded_prefill_mask(
                        actual_chunk_len as i32,
                        padded_len as i32,
                        kv_offset,
                    );
                    (padded, Some(mask))
                } else {
                    (chunk.to_vec(), None)
                }
            } else {
                (chunk.to_vec(), None)
            };

        let eff_len = eff_chunk.len() as i32;
        let input = mlxcel_core::from_slice_i32(&eff_chunk, &[1, eff_len]);
        let logits = {
            let caches = match self.cache_pool.get_caches_mut(seq.seq_id) {
                Some(c) => c,
                None => {
                    self.abort_sequence(seq, "Cache not found during chunked prefill continuation");
                    return false;
                }
            };

            let logits = self.model.forward_last_logits_with_sequence_id(
                &input,
                Some(seq.seq_id),
                caches,
                pad_mask_opt.as_ref().map(|m| m.as_ref().unwrap()),
                actual_chunk_len.saturating_sub(1),
            );

            // Trim padding positions from KV caches when the chunk was padded.
            if pad_mask_opt.is_some() && eff_chunk.len() > actual_chunk_len {
                let excess = (eff_chunk.len() - actual_chunk_len) as i32;
                for c in caches.iter_mut() {
                    c.trim(excess);
                }
            }
            logits
        };
        self.sync_sequence_storage(seq.seq_id);

        // H2: enforce the `--max-kv-size` cap after each
        // continuation chunk so a multi-chunk prefill stays bounded across
        // all chunks, not just at the very end. Cheap early-return when no
        // cap is configured.
        self.enforce_max_kv_size_for(seq.seq_id, seq.retention);

        seq.prefill_offset = end;
        // Per-chunk `prompt_progress`, as on the first chunk (#1477).
        seq.report_prefill_progress(end);

        tracing::debug!(
            "Chunked prefill: seq {} chunk {offset}..{end}/{total} tokens",
            seq.seq_id,
        );

        if !chunk_range.is_terminal {
            // More chunks remain -- store and yield back to the scheduler.
            // #822: evaluate this chunk through the fallible boundary so an MLX
            // throw fails just this request rather than aborting the process.
            if let Err(msg) =
                self.record_eval_outcome(mlxcel_core::try_eval(&logits).map_err(|e| e.to_string()))
            {
                self.abort_sequence(seq, &msg);
                self.eval_failures_exhausted();
                return false;
            }
            if !decode_batch_live {
                mlxcel_core::clear_memory_cache();
            }
            self.chunked_prefill_seq = Some(seq);
            return true;
        }

        // Final chunk -- complete the prefill and sample the first token
        if !decode_batch_live {
            mlxcel_core::clear_memory_cache();
        }

        let eos_tokens =
            merged_eos_token_ids(self.model.eos_token_ids(), &seq.sampling.stop_token_ids);
        let needs_history = seq.sampling.needs_token_history();
        let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

        self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
        true
    }

    /// Complete a prefill (full or chunked): sample the first token,
    /// handle EOS, and either finish immediately or move to the active
    /// decode batch.
    pub(super) fn finish_prefill(
        &mut self,
        mut seq: SequenceInfo,
        logits: UniquePtr<mlxcel_core::MlxArray>,
        eos_tokens: Vec<i32>,
        mut token_history: Vec<i32>,
        needs_history: bool,
    ) {
        // apply structured-output mask to the prefill logits
        // before sampling the first token so the very first emitted token
        // already conforms to the schema.
        let logits_for_sampling = if let Some(constraint) = seq.structured.clone() {
            // Read the vocab dimension from the prefill logits so the mask
            // matches the sampler's vocabulary exactly.
            let shape = mlxcel_core::array_shape(&logits);
            let vocab = *shape.last().unwrap_or(&0) as usize;
            match Self::apply_structured_mask(&constraint, mlxcel_core::copy(&logits), vocab) {
                Ok(masked) => masked,
                Err(msg) => {
                    let _ = seq
                        .response_tx
                        .send(GenerateEvent::Error(format!("structured output: {msg}")));
                    if let Err(err) = seq
                        .state
                        .transition_to(SequenceState::Finished(FinishReason::Error(msg)))
                    {
                        tracing::error!("State transition error: {err}");
                    }
                    self.prompt_cache_seq_ctx.remove(&seq.seq_id);
                    self.release_sequence_caches(seq.seq_id);
                    return;
                }
            }
        } else {
            mlxcel_core::copy(&logits)
        };
        // #347: reseed the global MLX RNG to THIS row's own seed at the exact
        // point it samples its first token. `begin_prefill` already seeded once,
        // but a batched cohort runs every row's `begin_prefill` up front before
        // any row reaches `finish_prefill`, so by the time row 0 samples here the
        // global RNG holds the LAST cohort row's seed ("last-seed-wins"). The
        // fused sampler draws from that process-global RNG with no per-call key
        // (`fused_sample` takes only the scalar params), so without this reseed a
        // seeded row's first token would depend on its siblings' seeds. Reseeding
        // here, rather than only in `begin_prefill`, guarantees the seed is live
        // at the exact sample point and makes each row's first token depend only
        // on its own seed. Greedy / `temperature == 0` / `top_k == 1` rows take
        // the argmax path and consume no RNG, so this is a no-op for them. The
        // batched fused DECODE path shares one global-RNG draw across the whole
        // `[B, vocab]` batch and is out of scope here (see issue #347).
        seed_rng_if_needed(&seq.sampling);
        let (first_token_arr, adjusted_logits, post_probs) = if seq.logprobs_config.enabled
            && seq.logprobs_config.source == LogprobSource::PostSampling
        {
            // b10621 post_sampling_probs (#1485): one chain pass, one XTC
            // gate, for both the draw and the report.
            let (token, adjusted, probs) = sample_token_with_state_and_distribution(
                &logits_for_sampling,
                &seq.sampling,
                &token_history,
                &mut seq.sampler_state,
            );
            (token, adjusted, Some(probs))
        } else if seq.sampling.needs_sampler_feedback_state() {
            // #1485: mirostat / adaptive-p carry per-sequence state that the
            // first sampled token must already update, so the stateful entry
            // point runs here too (penalty-only rows keep the stateless
            // rebuild path this call site always used).
            let (token, adjusted) = sample_token_optimized_with_state(
                &logits_for_sampling,
                &seq.sampling,
                &token_history,
                &mut seq.sampler_state,
            );
            (token, adjusted, None)
        } else {
            let (token, adjusted) =
                sample_token_optimized(&logits_for_sampling, &seq.sampling, &token_history);
            (token, adjusted, None)
        };
        // #822: force-evaluate the first sampled token through the fallible
        // boundary. On an MLX throw, fail just this request; the infallible
        // `item_i32` readback below would otherwise re-trigger the same throw
        // and abort the process.
        if let Err(msg) = self
            .record_eval_outcome(mlxcel_core::try_eval(&first_token_arr).map_err(|e| e.to_string()))
        {
            self.abort_sequence(seq, &msg);
            self.eval_failures_exhausted();
            return;
        }
        let sampled_first_token = mlxcel_core::item_i32(&first_token_arr);

        // advance the matcher state with the just-sampled token.
        // If consume_token errors, transition the sequence to Finished(Error)
        // and surface a clean SSE error event rather than leaking
        // non-conforming output.
        let structured_stopped = if let Some(constraint) = seq.structured.clone() {
            match Self::consume_structured_token(&constraint, sampled_first_token) {
                Ok(stopped) => stopped,
                Err(msg) => {
                    let _ = seq
                        .response_tx
                        .send(GenerateEvent::Error(format!("structured output: {msg}")));
                    if let Err(err) = seq
                        .state
                        .transition_to(SequenceState::Finished(FinishReason::Error(msg)))
                    {
                        tracing::error!("State transition error: {err}");
                    }
                    self.prompt_cache_seq_ctx.remove(&seq.seq_id);
                    self.release_sequence_caches(seq.seq_id);
                    return;
                }
            }
        } else {
            false
        };

        // thinking-budget override. Qwen3 chat templates prime
        // `<think>\n`, so the first prefill-completion token is already
        // inside the reasoning block when `enter_block_on_start == true`.
        let first_token = Self::apply_thinking_budget(&mut seq.thinking, sampled_first_token);

        // #1485: confirm the emitted first token with the sampler feedback
        // state (see the parallel comment in `execute_batched_decode`).
        if let Some(state) = seq.sampler_state.as_mut() {
            state.accept_token(first_token);
        }

        seq.mark_first_token();

        // if the budget fired and substituted the first token,
        // drop the logprob below (computed against the sampled token) so the
        // streamed metadata stays consistent with the emitted token text.
        let override_fired = first_token != sampled_first_token;

        // Check for immediate EOS
        if eos_tokens.contains(&first_token) {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Stop))
            {
                tracing::error!("State transition error: {err}");
            }
            let result = build_generation_result_with_cache(
                String::new(),
                seq.prompt_tokens.len(),
                0,
                seq.created_at.elapsed().as_millis() as u64,
                seq.prefill_start
                    .map(|t| (Instant::now() - t).as_millis() as u64)
                    .unwrap_or(0),
                seq.max_tokens,
                seq.already_cached_tokens,
            );
            tracing::info!(
                prompt_tokens = seq.prompt_tokens.len(),
                cached_tokens = seq.already_cached_tokens,
                saved_ms = 0,
                "prompt-cache: request completed (eos-at-prefill): \
                 cached={}/{} prompt tokens, saved ~0ms",
                seq.already_cached_tokens,
                seq.prompt_tokens.len(),
            );
            let _ = seq.response_tx.send(GenerateEvent::Done(result));
            // Prefill produced a valid KV cache (EOS on turn 1 is a healthy
            // stop). Donate it back so the next turn can reuse the prompt
            // prefix. `generated_tokens` is empty here by construction.
            self.donate_finished_sequence_cache(seq.seq_id, &seq.prompt_tokens, &[], true);
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            return;
        }

        // Optionally compute logprobs for the first token. When the override
        // fired, the sampled token differs from the emitted `first_token`;
        // suppress logprob emission in that case to keep token text and
        // logprob metadata consistent.
        let token_lp = if override_fired {
            None
        } else {
            match seq.logprobs_config.source {
                LogprobSource::PostSampling => post_probs.as_ref().map(|p| {
                    compute_post_sampling_probs(p, first_token, seq.logprobs_config.top_k)
                }),
                LogprobSource::RawModel if seq.logprobs_config.enabled => {
                    let raw_row = mlxcel_core::slice_last_logits(&logits);
                    compute_logprobs(&raw_row, first_token, &seq.logprobs_config)
                }
                _ => compute_logprobs(&adjusted_logits, first_token, &seq.logprobs_config),
            }
        };

        seq.generated_tokens.push(first_token);
        if needs_history {
            token_history.push(first_token);
        }

        // Store merged EOS and token history on the sequence so decode_single_step
        // can reuse them without per-step reconstruction.
        seq.merged_eos = eos_tokens;
        seq.token_history = token_history;

        // Stream the first token's text through the request's stop matcher
        // (issue #1466). A short stop string can complete on this very token, in
        // which case generation ends here and the matched text is never emitted.
        let prefill_stop_word = match seq.decode_state.on_token(first_token, &self.tokenizer) {
            Some(new_text) => seq.stream_decoded_text(new_text, Some(first_token), token_lp),
            None => None,
        };

        let mut prefill_finish_reason = if prefill_stop_word.is_some() {
            Some(FinishReason::StopSequence)
        } else if structured_stopped {
            Some(FinishReason::Stop)
        // A generation bound can already have fired on the very first token
        // (#1477); `t_max_predict_ms: 0` with a newline in that token is the
        // reachable case.
        } else if seq.bound_stopped() || seq.generated_tokens.len() >= seq.max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        // b10621 context guard (#1472): a prompt admitted just under the KV
        // bound can leave no room for a second token; stop here with
        // `truncated: true` rather than overflowing on the first decode step.
        if prefill_finish_reason.is_none()
            && Self::context_bound_stop_due(
                &seq,
                self.max_kv_size,
                self.context_retention.context_shift,
            )
        {
            seq.retention.context_exhausted = true;
            prefill_finish_reason = Some(FinishReason::Length);
        }
        if let Some(finish_reason) = prefill_finish_reason {
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(finish_reason))
            {
                tracing::error!("State transition error: {err}");
            }
            // Forward any tail the incremental detokenizer held back (a final
            // token carrying complete text plus a trailing incomplete UTF-8
            // byte) as one last token event before Done, so streaming clients
            // receive it (issue #633). It goes through the stop matcher too, so
            // a stop string that only becomes visible in the tail still fires
            // and the tail is dropped rather than leaked.
            let tail = seq.decode_state.flush(&self.tokenizer);
            seq.close_text_stream(tail);
            let cached = seq.already_cached_tokens;
            // Finished inside prefill, so no verify round ran (#1314).
            let result = seq.take_generation_result(&self.tokenizer, cached, None);
            tracing::info!(
                prompt_tokens = seq.prompt_tokens.len(),
                cached_tokens = cached,
                generation_time_ms = result.generation_time_ms,
                "prompt-cache: request completed during prefill: \
                 cached={}/{} prompt tokens, total {}ms",
                cached,
                seq.prompt_tokens.len(),
                result.generation_time_ms,
            );
            let _ = seq.response_tx.send(GenerateEvent::Done(result));
            self.donate_finished_sequence_cache(
                seq.seq_id,
                &seq.prompt_tokens,
                &seq.generated_tokens,
                true,
            );
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            return;
        }

        self.prepare_turbo4_delegated_for_sequence_decode(seq.seq_id);

        if let Err(err) = seq.state.transition_to(SequenceState::Decoding) {
            tracing::error!("State transition error: {err}");
            self.abort_sequence(seq, &err);
            return;
        }

        let prompt_len = seq.prompt_tokens.len() as i32;
        if let Some(cache_set) = self.cache_pool.get_mut(seq.seq_id) {
            cache_set.prompt_len = seq.prompt_tokens.len();
            cache_set.current_offset = prompt_len + 1;
        }

        // The slot this sequence was admitted into is reserved for it: admission
        // required `!active_batch.is_full()`, and while a chunked prefill is
        // parked the tick policy's chunked branch short-circuits above the
        // admission branch, so nothing else can take the slot in between. That
        // invariant used to be belt and braces because a chunked prefill could
        // only finish against an EMPTY batch; since #1011 it finishes against a
        // live one, so it is now the only thing standing between a completed
        // prefill and a full batch. `ActiveBatch::add` consumes the sequence and
        // cannot hand it back, so a failure there drops the request silently and
        // the client's stream just ends. Check first and fail it loudly instead.
        if self.active_batch.is_full() {
            self.abort_sequence(
                seq,
                "Internal scheduler error: the active batch filled while this prompt was \
                 being prefilled, so its reserved slot was lost",
            );
            return;
        }
        if let Err(err) = self.active_batch.add(seq) {
            tracing::error!("Failed to add sequence to active batch: {err}");
        }
    }
}
