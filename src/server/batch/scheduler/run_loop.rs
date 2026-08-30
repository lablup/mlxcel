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
    /// Apply thinking-budget enforcement to a freshly sampled
    /// token for a single sequence.
    ///
    /// Returns the final token id to commit to the sequence (either the
    /// sampled value, or the forced `</think>` id when the budget fires).
    /// Caller is responsible for using the returned id for the remainder of
    /// the decode step (EOS check, streaming emission, history update).
    ///
    /// The state advances with the final id so subsequent steps see the
    /// post-close phase.
    ///
    /// # Notes on bypass of sampling knobs
    ///
    /// When the budget fires the forced id bypasses the sampler's logits
    /// pipeline for that step. No retroactive re-penalization happens because
    /// - `token_history` is only appended once per step (caller uses the
    ///   returned id),
    /// - `merged_eos` checks use the returned id,
    /// - the next step samples fresh logits from the underlying model.
    pub(super) fn apply_thinking_budget(seq_thinking: &mut ThinkingState, sampled: i32) -> i32 {
        if seq_thinking.is_disabled() {
            return sampled;
        }
        let final_id = match seq_thinking.decide_override(sampled) {
            ThinkingDecision::NoOverride => sampled,
            // The `--reasoning-budget-message` run is forced token by token
            // ahead of the close tag, exactly as the close tag itself is
            // (#1470).
            ThinkingDecision::ForceClose(id) | ThinkingDecision::ForceMessage(id) => id,
        };
        seq_thinking.observe(final_id);
        final_id
    }

    /// apply the structured-output mask (if any) to logits before
    /// sampling.
    ///
    /// Returns either the masked logits or `Err(_)` describing why the
    /// matcher refused to advance. The scheduler propagates the error as
    /// `FinishReason::Error(...)` so the SSE stream terminates cleanly
    /// instead of emitting non-conforming output.
    pub(super) fn apply_structured_mask(
        constraint: &std::sync::Arc<
            std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>,
        >,
        logits: UniquePtr<mlxcel_core::MlxArray>,
        vocab_size_hint: usize,
    ) -> Result<UniquePtr<mlxcel_core::MlxArray>, String> {
        let mut guard = constraint
            .lock()
            .map_err(|e| format!("structured-output constraint poisoned: {e}"))?;
        let masked = crate::server::structured::apply_structured_mask_to_logits(
            &mut guard,
            &logits,
            vocab_size_hint,
        )
        .map_err(|e| e.to_string())?;
        Ok(masked)
    }

    /// advance the matcher state by the just-sampled token.
    ///
    /// Returns `Ok(true)` when the consumed token completes the structured
    /// output, `Ok(false)` when decoding must continue, and `Err(msg)` when
    /// `consume_token` fails or the matcher is in an error state. The caller
    /// transitions the sequence to `Finished(Stop)` on completion or
    /// `Finished(Error(msg))` on error.
    pub(super) fn consume_structured_token(
        constraint: &std::sync::Arc<
            std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>,
        >,
        token: i32,
    ) -> Result<bool, String> {
        let mut guard = constraint
            .lock()
            .map_err(|e| format!("structured-output constraint poisoned: {e}"))?;
        guard
            .consume_token_and_check_stopped(token)
            .map_err(|e| e.to_string())
    }

    /// send a clean SSE error event and transition the sequence
    /// to `Finished(Error(msg))`. Used by the structured-output path to
    /// abort cleanly when the matcher refuses to advance.
    pub(super) fn abort_sequence_with_error(
        seq: Option<&mut SequenceInfo>,
        prefix: &str,
        msg: &str,
    ) {
        if let Some(seq) = seq {
            let _ = seq
                .response_tx
                .send(GenerateEvent::Error(format!("{prefix}: {msg}")));
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Error(
                    msg.to_string(),
                )))
            {
                tracing::error!("State transition error: {err}");
            }
        }
    }

    /// Effective thinking-budget for a single sequence.
    ///
    /// Combines the server default with any per-request override attached to
    /// the request's `ServerGenerateOptions`. Returns a [`ThinkingState`]
    /// ready to be stored on `SequenceInfo`.
    ///
    /// `enter_block_on_start` is passed through to the [`ThinkingState`].
    /// Chat endpoints set `true` (the Qwen3 chat template primes `<think>\n`);
    /// raw text endpoints (`/v1/completions`, `/completion`) set `false` so
    /// the model must emit `<think>` before any in-block counting begins.
    /// The tokenized `--reasoning-budget-message` for this request (#1470).
    ///
    /// b10621 tokenizes the message once when it composes the forced run; the
    /// memo here is the same idea against a live setting that can change, and
    /// it keeps a configured message off the per-request encode path after the
    /// first request. Special-token parsing is off, matching the settings
    /// every other server-side tokenization uses.
    pub(super) fn forced_reasoning_message(
        &self,
        message: Option<&str>,
    ) -> Option<std::sync::Arc<Vec<i32>>> {
        let message = message?;
        if message.is_empty() {
            return None;
        }
        if let Ok(memo) = self.forced_reasoning_message.try_borrow()
            && let Some((cached, tokens)) = memo.as_ref()
            && cached == message
        {
            return Some(tokens.clone());
        }
        let ids: Vec<i32> = self
            .tokenizer
            .encode_with_special(message, false, false)
            .ok()?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        if ids.is_empty() {
            return None;
        }
        let tokens = std::sync::Arc::new(ids);
        if let Ok(mut memo) = self.forced_reasoning_message.try_borrow_mut() {
            *memo = Some((message.to_owned(), tokens.clone()));
        }
        Some(tokens)
    }

    pub(super) fn build_thinking_state(
        &self,
        override_: ReasoningBudgetOverride,
        enter_block_on_start: bool,
        reasoning_control: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        forced_message: Option<std::sync::Arc<Vec<i32>>>,
    ) -> ThinkingState {
        // No thinking tokens -> always disabled regardless of config.
        let Some(token_ids) = self.thinking_token_ids else {
            return ThinkingState::disabled();
        };
        let effective = match override_ {
            ReasoningBudgetOverride::InheritServerDefault => self.reasoning_budget,
            ReasoningBudgetOverride::Explicit(v) => v,
        };
        // b10621 `reasoning_control` (#1444): an armed control flag keeps the
        // tracker active even without a budget, so a live `reasoning_end`
        // request can close the block at the next sampled token.
        ThinkingState::new(Some(token_ids), effective, enter_block_on_start)
            .with_force_end(reasoning_control)
            .with_forced_message(forced_message)
    }

    /// Run the scheduler loop until shutdown or channel close.
    pub fn run(&mut self) {
        install_thread_local_default_stream(self.generation_stream.as_ref());

        loop {
            // 1. Non-blocking drain of all pending requests
            self.drain_incoming_requests();

            if self.shutdown_requested {
                break;
            }

            self.publish_metrics();

            // 2. Decide what to do this tick
            let action = self.decide_action();

            // 3. Execute
            match action {
                BatchSchedulerAction::Prefill(seq_id) => {
                    // Admission / chunked-prefill interleave / preemption all
                    // change batch membership; tear down any prebuilt lookahead
                    // (trimming its speculative KV) before the prefill runs so
                    // the next decode rebuilds against the new batch (#632
                    // invalidation).
                    self.discard_lookahead();
                    // Use batched prefill when max_batch_prefill > 1 and at
                    // least 2 requests are waiting, otherwise take the regular
                    // single-request path so there is zero overhead for the
                    // common case. #715: also require the head-of-queue prompt
                    // to be short enough to join a padded batch within the
                    // token budget; a head too long to batch takes the
                    // chunk-aware single-sequence path (which keeps the
                    // attention mask chunked to `[chunk, L]` instead of the
                    // unchunked `[L, L]` a single-row batched forward would
                    // build).
                    if self.max_batch_prefill > 1
                        && self.prefill_queue.len() >= 2
                        && self.chunked_prefill_seq.is_none()
                        && self.batched_prefill_admits_head()
                    {
                        self.execute_batched_prefill();
                    } else {
                        self.execute_prefill(seq_id);
                    }
                    // A classic action ran: the next tick goes back to the
                    // in-flight speculative slice, if any (issue #734).
                    self.speculative_slice_yielded = false;
                    self.publish_metrics();
                }
                BatchSchedulerAction::Decode(ids) => {
                    self.execute_decode_step(&ids);
                    // A classic action ran: the next tick goes back to the
                    // in-flight speculative slice, if any (issue #734).
                    self.speculative_slice_yielded = false;
                }
                BatchSchedulerAction::MixedStep(ids) => {
                    // Issue #908 prototype, reachable only under
                    // `MLXCEL_MIXED_STEP`. One tick advances both workloads:
                    // the decode batch by a token and the parked chunked
                    // prefill by a chunk. This is the *scheduling* half of a
                    // mixed step; the two forwards still run back to back
                    // rather than fused into one ragged forward. ADR 0005
                    // explains why that split is the useful experiment.
                    //
                    // Same #632 invalidation invariant as the Prefill arm.
                    // Like the SpeculativeRound arm's discard, this is a
                    // guaranteed no-op rather than load-bearing cleanup:
                    // `lookahead_safe()` returns false whenever a chunked
                    // prefill is parked, so no lookahead can exist on any tick
                    // that reaches here, and the decode below cannot build one.
                    // Kept so the invariant holds by construction if that
                    // precondition ever changes.
                    self.discard_lookahead();
                    // Decode runs first, against the id set `decide_action`
                    // captured. The terminal chunk calls `finish_prefill`,
                    // which admits the prompt into the active batch; decoding
                    // first keeps that id set valid for this tick and lets the
                    // freshly admitted sequence start decoding on the next one.
                    self.execute_decode_step(&ids);
                    // Count the tick only when a chunk really ran. The counter
                    // is this prototype's dispatch proof, so it must not move
                    // on a tick where the parked sequence was already drained
                    // or aborted.
                    if self.continue_chunked_prefill() {
                        self.batch_observability.record_mixed_step();
                    }
                    // A classic action ran: the next tick goes back to the
                    // in-flight speculative slice, if any (issue #734).
                    self.speculative_slice_yielded = false;
                    self.publish_metrics();
                }
                BatchSchedulerAction::SpeculativeRound => {
                    // Same invariant as the Prefill arm: any action that can
                    // mutate KV caches outside the decode fast path must tear
                    // down a prebuilt lookahead first (#632 invalidation).
                    // Under speculative dispatch the lookahead pipeline is
                    // globally disabled (`lookahead_params` returns `None`
                    // when `should_dispatch_speculative()`), so this is a
                    // guaranteed no-op kept for the invariant.
                    self.discard_lookahead();
                    self.execute_speculative_slice_round();
                    // Yield the next tick to the classic actions when they
                    // have work, so classic rows advance between rounds.
                    self.speculative_slice_yielded = true;
                    self.publish_metrics();
                }
                BatchSchedulerAction::Idle if self.can_run_prompt_cache_warmup() => {
                    // Strictly idle-time work (issue #1144). Reaching the Idle
                    // arm already means no decode batch and no queued prefill,
                    // so a warm-up here cannot delay a foreground request that
                    // exists; the guard below re-checks in case a request
                    // arrived between the decision and now.
                    self.run_next_prompt_cache_warmup();
                    self.publish_metrics();
                }
                BatchSchedulerAction::Idle => match self.request_rx.recv() {
                    Ok(req) => {
                        if self.handle_incoming(req) {
                            break;
                        }
                        self.publish_metrics();
                    }
                    Err(_) => {
                        tracing::info!("Request channel closed, scheduler exiting");
                        break;
                    }
                },
            }

            // 4. Clean up completed sequences
            self.finalize_completed();
        }
    }

    pub(super) fn publish_metrics(&self) {
        let active = self.active_batch.len();
        let queued = self.prefill_queue.len();
        let paged_stats = self.cache_pool.paged_stats();
        let paged_block_size = self.cache_pool.paged_block_size().unwrap_or(0);
        self.batch_metrics.set_active_count(active);
        self.batch_metrics.set_queue_depth(queued);
        self.batch_observability.update_gauges(
            active,
            queued,
            self.cache_pool.active_count(),
            self.cache_pool.memory_usage_bytes() as u64,
            paged_block_size,
            paged_stats,
            // #122 c: surface the configured block-budget cap (0 = unbounded)
            // so `/v1/cache/stats` and `/metrics` can report admission headroom.
            self.cache_pool.paged_block_budget().unwrap_or(0) as u64,
        );
        // Which attention kernel the decode loop actually ran, and how much
        // prefix the cascade decomposition hoisted (issues #899, #903). Cheap
        // relaxed loads of process-wide counters; publishing them here is what
        // makes the answer readable from `/health` without a profiler.
        self.batch_observability
            .update_paged_decode_gauges(mlxcel_core::cache::paged_batch_decode_stats());
    }
}
