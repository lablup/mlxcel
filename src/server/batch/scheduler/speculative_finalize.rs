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
    pub(super) fn try_speculative_burst(&mut self, seq: SequenceInfo) -> Option<SequenceInfo> {
        // Fast path: speculative dispatch off, or the head fails the
        // per-sequence gate (multimodal payload / VLM embeddings /
        // structured output / adopted cache prefix). History-dependent
        // penalties, logprobs, and thinking budgets are supported by
        // the B = 1 burst path.
        // Route straight to classic only for the hard per-request gates.
        if !crate::server::batch::speculative_burst::should_burst_for_sequence(
            &self.speculative_dispatch,
            &seq,
        ) {
            return Some(seq);
        }

        // At most ONE tick-cooperative slice is ACTIVE at a time (issue
        // #734): the drafter handle is held by the in-flight job, and the
        // slice state is a single scheduler slot. A further
        // speculative-eligible request arriving while the slot is busy
        // (active job, or grantees already queued behind it) no longer
        // always falls back to classic decode (issue #746): when the
        // request could itself be served by the tick-slice path and the
        // grant backlog has room, it is parked as a WAITER and the slot
        // rotates across grantees at round boundaries
        // (`MLXCEL_MTP_SLICE_GRANT_ROUNDS`), so one long stream cannot
        // monopolize speculative acceleration. Returning `None` keeps this
        // method's contract of "do not route to classic prefill", but note
        // the sequence is now OWNED by the scheduler in
        // `speculative_slice_backlog` rather than having been finalized
        // inline. Beyond the backlog cap, with rotation disabled (budget
        // 0), or for requests the slice path cannot serve (DFlash
        // dispatch, non-Gemma-4 target, tick slice disabled, degenerate
        // adopted prefix, adaptive-policy decline), the request falls back
        // to classic decode exactly as pre-#746. The B>1 batched-window
        // assembly below is untouched: it only runs when the slot is free.
        if self.speculative_slice.is_some() || !self.speculative_slice_backlog.is_empty() {
            if self.can_wait_for_slice_grant(&seq)
                && crate::server::batch::speculative_slice::slice_backlog_admits(
                    self.speculative_slice_backlog.len(),
                    crate::server::batch::speculative_slice::mtp_slice_grant_rounds(),
                )
            {
                tracing::debug!(
                    "speculative slice slot busy; parking seq {} as a slot-grant waiter \
                     (backlog {} -> {})",
                    seq.seq_id,
                    self.speculative_slice_backlog.len(),
                    self.speculative_slice_backlog.len() + 1,
                );
                self.speculative_slice_backlog.push_back(
                    crate::server::batch::speculative_slice::SliceBacklogEntry::waiter(Box::new(
                        seq,
                    )),
                );
                return None;
            }
            tracing::debug!(
                "speculative slice slot busy; seq {} falls back to classic decode",
                seq.seq_id,
            );
            return Some(seq);
        }

        // Try to assemble a B>1 batched window. The window head is
        // `seq`; siblings must (1) be speculative-eligible, (2) have the
        // same prompt length, (3) have the same `max_tokens`, and (4)
        // share the same sampling config — the batched round loop takes
        // one sampler and one per-row budget for the whole window. B>1
        // also excludes requests whose response payloads need per-row
        // data that batched round loops do not return yet (for example
        // logprobs), leaving them on the B=1 burst arm. The window cap
        // is the configured `max_batch_size`, surfaced via the active
        // batch's capacity (the scheduler constructs
        // `ActiveBatch::new(max_batch_size)`).
        let max_batch_size = self.active_batch.max_size();
        let head_can_join_batched =
            crate::server::batch::speculative_burst::can_join_batched_burst_window(&seq);
        // Ragged (variable-length-prompt) windows are gated behind a separate
        // opt-in subordinate to `MLXCEL_ENABLE_MTP_BATCH`. When off (the
        // default), the window collector keeps its original equal-prompt-length
        // constraint so the validated same-length batched burst is unchanged.
        // When on, the prompt-length equality is dropped and burst-eligible
        // rows of different lengths join one window; the batched MTP adapter
        // left-pads them to `max_prompt_len` and threads per-row positions /
        // valid lengths so greedy parity is preserved.
        let allow_ragged =
            crate::server::batch::speculative_burst::mtp_batched_ragged_window_enabled();
        let window: Vec<SequenceInfo> = if max_batch_size > 1 && head_can_join_batched {
            let head_prompt_len = seq.prompt_tokens.len();
            let head_max_tokens = seq.max_tokens;
            let head_sampling = seq.sampling.clone();
            let head_lane = seq.priority;
            // Reserve one slot for the head itself.
            let max_extra = max_batch_size.saturating_sub(1);
            let dispatch = &self.speculative_dispatch;
            let extra =
                self.prefill_queue
                    .drain_matching_window(head_lane, max_extra, |candidate| {
                        (allow_ragged || candidate.prompt_tokens.len() == head_prompt_len)
                        && candidate.max_tokens == head_max_tokens
                        && crate::server::batch::speculative_burst::sampling_config_eq(
                            &candidate.sampling,
                            &head_sampling,
                        )
                        && crate::server::batch::speculative_burst::should_burst_for_sequence(
                            dispatch, candidate,
                        )
                        && crate::server::batch::speculative_burst::can_join_batched_burst_window(
                            candidate,
                        )
                    });
            let mut window = Vec::with_capacity(extra.len() + 1);
            window.push(seq);
            window.extend(extra);
            window
        } else {
            vec![seq]
        };

        if window.len() == 1
            && matches!(
                self.speculative_dispatch,
                crate::server::SpeculativeDispatch::Mtp { .. }
            )
            && !self.mtp_b1_should_run()
        {
            // B=1 MTP decision (issue #333, adaptive): when an adaptive policy
            // is attached it profiles the first few B=1 bursts of this
            // (target, drafter, hardware) pairing and settles to a data-driven
            // verdict, overriding the static per-hardware gate where the
            // measured profile is clearly favorable or unfavorable. Without a
            // policy (MLXCEL_MTP_ADAPTIVE off) this falls back to the static
            // per-hardware default (issue #165, revised by #1217):
            // non-batchable 12B targets keep B=1 MTP on everywhere;
            // batch-capable 31B targets default it on from Apple GPU
            // generation 15, where a quantized projection at `M >= 2` runs as
            // one wide pass, and off on generation 13, which runs the verify
            // block as narrow per-position passes and measured a regression.
            // `MLXCEL_ENABLE_MTP_B1` overrides in both directions; on decline
            // the request falls back to classic decode.
            let seq = window.into_iter().next().expect("singleton window");
            tracing::info!(
                "MTP B=1 speculative burst declined for seq {} (adaptive policy \
                 verdict, per-hardware default, or MLXCEL_ENABLE_MTP_B1=0); \
                 falling back to classic decode",
                seq.seq_id,
            );
            return Some(seq);
        }

        if window.len() >= 2
            && matches!(
                self.speculative_dispatch,
                crate::server::SpeculativeDispatch::Mtp { .. }
            )
            && !crate::server::batch::speculative_burst::mtp_batched_burst_enabled()
        {
            // The B>1 batched MTP burst is off by default because it is not
            // consistently faster than classic batched decode on the 31B
            // (M5 Max: ~1.06x for a same-length window, ~0.78x once prompt
            // lengths differ and requests serialize into head-of-line-blocking
            // B=1 bursts). M5 Max runs observed greedy parity holding at
            // temperature 0. Set MLXCEL_ENABLE_MTP_BATCH=1 to force the path.
            let mut rejected_window = window;
            let head = rejected_window.remove(0);
            tracing::info!(
                "MTP batched speculative burst declined for seq {}: B>1 MTP is not \
                 consistently faster than classic batched decode on the 31B; falling \
                 back to classic decode. Set MLXCEL_ENABLE_MTP_BATCH=1 to force the \
                 experimental batched MTP path",
                head.seq_id,
            );
            for sibling in rejected_window {
                let sibling_id = sibling.seq_id;
                if let Err(boxed) = self.prefill_queue.enqueue(sibling) {
                    tracing::warn!(
                        "MTP batched speculative burst declined and prefill queue \
                         full; aborting sibling seq {sibling_id}"
                    );
                    self.prompt_cache_seq_ctx.remove(&sibling_id);
                    self.abort_sequence(
                        *boxed,
                        "MTP batched speculative burst declined and prefill queue full",
                    );
                }
            }
            return Some(head);
        }

        if window.len() >= 2 {
            // ---- Batched B>1 burst ----
            let ctx = crate::server::batch::speculative_burst::BurstContext {
                model: &self.model,
                tokenizer: &self.tokenizer,
                drafter_slot: &mut self.speculative_drafter_slot,
                dispatch: &self.speculative_dispatch,
                // Classic-step probes are a B=1 profiling concern (#736).
                profile_probe_rounds: 0,
            };
            match crate::server::batch::speculative_burst::try_run_burst_batched(ctx, window) {
                Ok(crate::server::batch::speculative_burst::BatchedBurstFinalized { rows }) => {
                    // Every row handled inline. For each row: donate the
                    // finished sequence's KV cache back to the
                    // prompt-cache store, release its cache slot, and
                    // record its per-sequence metric — the batched
                    // analogue of the B=1 cleanup below.
                    //
                    // `BatchedBurstRow` now carries the
                    // per-row prompt/committed token vectors and the
                    // healthy-finish flag, so the batched arm calls
                    // `donate_finished_sequence_cache` per row BEFORE
                    // the `remove`/`release` — symmetric with the B=1
                    // arm and the classic `finalize_completed` path.
                    // The donate helper chooses the model's supported
                    // cross-request reuse path: exact-prefix snapshots for
                    // opt-in model-owned families, otherwise detached KV for
                    // dense/paged families. Wiring it in removes the structural
                    // asymmetry between the two burst arms and future-proofs
                    // the batched path for any reusable family that later
                    // becomes batched-burst-eligible. Error /
                    // transition-failure rows carry an empty/`false`
                    // payload so the donate is a guaranteed no-op on
                    // those tainted-cache rows.
                    for crate::server::batch::speculative_burst::BatchedBurstRow {
                        seq_id,
                        tokens_generated,
                        prompt_tokens,
                        generated_tokens,
                        healthy_finish,
                    } in rows
                    {
                        self.donate_finished_sequence_cache(
                            seq_id,
                            &prompt_tokens,
                            &generated_tokens,
                            healthy_finish,
                        );
                        // Defensive non-donate cleanup: the donate path
                        // above already removed the `prompt_cache_seq_ctx`
                        // entry on the dense-KV path.
                        self.prompt_cache_seq_ctx.remove(&seq_id);
                        self.release_sequence_caches(seq_id);
                        self.batch_metrics
                            .record_sequence_completed(tokens_generated);
                        self.batch_observability.record_sequence_completed();
                    }
                    self.publish_metrics();
                    None
                }
                Err(mut rejected_window) => {
                    // The batched burst declined the whole window (e.g.
                    // unsupported model variant). The head goes back to
                    // the caller for the classic path; the sibling rows
                    // are re-enqueued so they retry next tick (they
                    // re-evaluate as their own potential window heads).
                    let head = rejected_window.remove(0);
                    for sibling in rejected_window {
                        let sibling_id = sibling.seq_id;
                        if let Err(boxed) = self.prefill_queue.enqueue(sibling) {
                            // Queue full: the sibling cannot be retried.
                            // Abort it with a clear error rather than
                            // dropping it silently. This is extremely
                            // unlikely — the sibling was just dequeued
                            // from this same queue.
                            tracing::warn!(
                                "speculative burst window declined and prefill queue \
                                 full; aborting sibling seq {sibling_id}"
                            );
                            self.prompt_cache_seq_ctx.remove(&sibling_id);
                            self.abort_sequence(
                                *boxed,
                                "speculative burst declined and prefill queue full",
                            );
                        }
                    }
                    Some(head)
                }
            }
        } else {
            // ---- B=1 arm (window collapsed to the head only) ----
            let seq = window.into_iter().next().expect("window has the head");

            // Tick-cooperative MTP slice (issue #734): a B=1 MTP request on
            // any `mtp_capable_target` family (Gemma 4 assistant, and since
            // issue #1165 Qwen 3.5 MTP) is served one speculative round per
            // scheduler tick instead of a run-to-completion burst, so
            // concurrent classic rows advance between rounds. The DFlash
            // arm keeps the legacy burst (its generator has no resumable
            // step API yet); `MLXCEL_MTP_TICK_SLICE=0` forces the MTP arm
            // back onto the legacy burst as an operator escape hatch.
            if matches!(
                self.speculative_dispatch,
                crate::server::SpeculativeDispatch::Mtp { .. }
            ) && crate::server::batch::speculative_slice::mtp_tick_slice_enabled()
            {
                return self.start_mtp_slice_b1(seq);
            }

            let ctx = crate::server::batch::speculative_burst::BurstContext {
                model: &self.model,
                tokenizer: &self.tokenizer,
                drafter_slot: &mut self.speculative_drafter_slot,
                dispatch: &self.speculative_dispatch,
                // While the adaptive policy is profiling this pairing, ask
                // the burst for a few classic-step probe rounds so the
                // measured-cost estimator has a classic step time (#736);
                // zero once the verdict is forced or settled.
                profile_probe_rounds: self
                    .mtp_policy
                    .as_ref()
                    .map(|p| p.profile_probe_rounds())
                    .unwrap_or(0),
            };
            match crate::server::batch::speculative_burst::try_run_burst_b1(ctx, seq) {
                Ok(finalized) => {
                    self.finish_speculative_b1(finalized);
                    None
                }
                Err(rejected_seq) => Some(rejected_seq),
            }
        }
    }

    /// Shared post-completion bookkeeping for a finalized B=1 speculative
    /// request, whether it ran as a legacy run-to-completion burst or as a
    /// tick-cooperative slice (issue #734): prompt-cache donate, cache-slot
    /// release, Prometheus metrics, adaptive-policy feed, and the HOL
    /// observability log.
    pub(super) fn finish_speculative_b1(
        &mut self,
        finalized: crate::server::batch::speculative_burst::BurstFinalized,
    ) {
        let crate::server::batch::speculative_burst::BurstFinalized {
            seq_id,
            tokens_generated,
            prompt_tokens,
            generated_tokens,
            healthy_finish,
            mtp_profile,
            burst_wall_ms,
            burst_active_ms,
            slices,
        } = finalized;
        // The request was handled end-to-end inline.
        //
        // donate the finished sequence's KV
        // cache back to the prompt-cache store BEFORE the
        // `remove`/`release` below; `donate_finished_sequence_cache`
        // both consumes the `prompt_cache_seq_ctx` entry and
        // needs the cache slot still attached. This mirrors
        // the classic path's `finalize_completed`, keeping
        // the burst and classic donate paths symmetric. The
        // donate helper snapshots opt-in model-owned families and
        // detaches dense/paged KV for the regular backends. Wiring
        // it in removes the structural asymmetry between the two
        // paths and future-proofs the burst for any reusable model
        // family that later becomes burst-eligible.
        self.donate_finished_sequence_cache(
            seq_id,
            &prompt_tokens,
            &generated_tokens,
            healthy_finish,
        );
        // Release the pre-allocated cache slot for symmetry
        // with `finalize_completed`, and mirror the classic
        // path's per-sequence metric recording so Prometheus
        // counters cover burst completions too. The `remove`
        // here is the defensive non-donate cleanup (the
        // donate path above already removed the
        // `prompt_cache_seq_ctx` entry on the dense-KV path).
        self.prompt_cache_seq_ctx.remove(&seq_id);
        self.release_sequence_caches(seq_id);
        self.batch_metrics
            .record_sequence_completed(tokens_generated);
        self.batch_observability.record_sequence_completed();
        // Feed the adaptive MTP policy (issue #333) the coarse
        // profile of this B=1 run. Only present for MTP runs that
        // executed a speculative round; a no-op once the policy has
        // settled, so there is no steady-state per-request cost.
        if let (Some(policy), Some(profile)) = (self.mtp_policy.as_mut(), mtp_profile) {
            let was_profiling = policy.is_profiling();
            policy.record_b1_sample(profile);
            // Republish for the supported read interface (issue #1257) only
            // while the state can still move. Once forced or settled the
            // published view is fixed, so the steady state pays nothing.
            if was_profiling {
                self.batch_observability.set_mtp_policy(policy.snapshot());
            }
        }
        // Observability (issue #638, re-scoped by issue #734):
        // `burst_wall_ms` is the maximum wall-clock any SINGLE scheduler
        // tick spent on this request: the head-of-line stall bound it
        // imposed on concurrent classic-decode rows. On the
        // tick-cooperative slice path that is about one speculative
        // round (plus the unavoidable prefill slice 0); on the legacy
        // run-to-completion arms (`slices == 1`) it is still the whole
        // burst. `burst_active_ms` is the cumulative worker occupancy
        // across all `slices`. `hol_waiters` shows how many rows share
        // the worker at finalize time.
        let (rounds, accepted) = mtp_profile
            .map(|p| (p.rounds, p.accepted_draft_tokens))
            .unwrap_or((0, 0));
        tracing::info!(
            seq_id = %seq_id,
            burst_wall_ms,
            burst_active_ms,
            slices,
            tokens_generated,
            rounds,
            accepted_draft_tokens = accepted,
            hol_waiters = self.active_batch.len() + self.prefill_queue.len(),
            "speculative B=1 burst finalized (burst_wall_ms is the max \
             single-tick HOL stall on concurrent rows)"
        );
        self.publish_metrics();
    }

    /// Promote the next backlog grantee into the empty slice slot (issue
    /// #746), priority lane first then FIFO
    /// ([`crate::server::batch::speculative_slice::next_grant_index`]).
    ///
    /// Returns `true` when a parked job was installed as the active slice
    /// and still needs its round run this tick; `false` when the tick's
    /// speculative action already ran inside the promotion (a waiter's
    /// slice 0, or a slice-0 inline finish / client-visible failure) or
    /// the backlog emptied. Cancelled waiters and grantees that decline
    /// at promotion time resolve with O(1) bookkeeping and the loop moves
    /// to the next grantee in the same tick.
    pub(super) fn promote_next_speculative_grantee(&mut self) -> bool {
        debug_assert!(self.speculative_slice.is_none());
        while let Some(entry) = self.pop_next_speculative_grantee() {
            match entry.kind {
                crate::server::batch::speculative_slice::SliceBacklogKind::Waiter(seq) => {
                    let seq = *seq;
                    if seq.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        // The client disconnected while the request waited
                        // for a grant: abort through the standard path
                        // (error event + cache release) and try the next
                        // grantee this tick.
                        tracing::debug!(
                            "slice waiter seq {} cancelled while queued; aborting",
                            seq.seq_id,
                        );
                        self.prompt_cache_seq_ctx.remove(&seq.seq_id);
                        self.abort_sequence(
                            seq,
                            "Request cancelled while waiting for the speculative slice slot",
                        );
                        continue;
                    }
                    // Re-check the B=1 verdict at promotion time: the
                    // adaptive policy (issue #333) may have settled to
                    // Decline while the request waited for its grant.
                    // `start_mtp_slice_b1` does not consult the policy
                    // (on the direct-admission path its caller checks it
                    // before the call), so without this re-check a
                    // declined pairing would still start speculative
                    // decode here.
                    if !self.mtp_b1_should_run() {
                        self.route_declined_slice_waiter_to_classic(seq);
                        continue;
                    }
                    match self.start_mtp_slice_b1(seq) {
                        // Slice 0 ran: the job now holds the slot (or
                        // parked straight back at a spent budget), or the
                        // request finished inline / failed with a
                        // client-visible error. The tick's speculative
                        // action is done either way.
                        None => return false,
                        // Declined by the start's own gates (target
                        // variant, degenerate adopted prefix): route to
                        // classic prefill, the same destination an
                        // admission-time decline takes.
                        Some(seq) => {
                            self.route_declined_slice_waiter_to_classic(seq);
                            continue;
                        }
                    }
                }
                crate::server::batch::speculative_slice::SliceBacklogKind::Parked(mut job) => {
                    // Re-acquire the worker drafter returned at park time.
                    // `ensure_loaded` is a no-op while the handle sits in
                    // the slot; it reloads from disk only if the park-time
                    // reset failed and dropped the handle. For the Gemma 4
                    // assistant drafter (trait-default no-op reset) this
                    // cannot happen; for the Qwen 3.5 MTP drafter, whose
                    // reset re-binds against the target and is therefore
                    // fallible in principle, this branch is the defensive
                    // path (not expected to fire against an already-bound
                    // live target, but handled for symmetry with the burst
                    // paths). A cancelled parked job is promoted normally:
                    // its next `step_session` observes the cancel at the
                    // round top and finishes with the tokens emitted so
                    // far, exactly like an active job's cancellation.
                    if let Err(e) = self.speculative_drafter_slot.ensure_loaded() {
                        self.fail_inflight_slice_job(
                            *job,
                            &format!("drafter reload failed at slice grant: {e}"),
                        );
                        continue;
                    }
                    let Some(mut drafter) = self.speculative_drafter_slot.take() else {
                        self.fail_inflight_slice_job(
                            *job,
                            "drafter slot empty after ensure_loaded at slice grant",
                        );
                        continue;
                    };
                    // Re-bind: idempotent for the handle returned at park
                    // time (recomputes the same target-derived state),
                    // required after a defensive from-disk reload. The
                    // per-round drafter state is rebuilt by
                    // `step_session`'s shared-KV re-arm either way.
                    if let Err(msg) = self.bind_drafter_to_target(&mut drafter) {
                        drop(drafter);
                        self.fail_inflight_slice_job(*job, &msg);
                        continue;
                    }
                    job.attach_drafter(drafter);
                    tracing::debug!(
                        seq_id = %job.seq.seq_id,
                        slices = job.slices,
                        "parked slice job granted the slot; resuming"
                    );
                    self.speculative_slice = Some(job);
                    // The promotion itself is bookkeeping; the round that
                    // runs this tick is the grant's first slice. Resolve
                    // the budget once per grant here, the second of the
                    // two grant-start sites.
                    self.speculative_slice_grant_slices = 0;
                    self.speculative_slice_grant_budget =
                        crate::server::batch::speculative_slice::mtp_slice_grant_rounds();
                    return true;
                }
            }
        }
        false
    }

    /// Route a slot-grant waiter that declined at promotion time back to
    /// the classic prefill path (issue #746): the same destination an
    /// admission-time burst decline takes. Used for the promotion-time
    /// B=1 verdict re-check and for `start_mtp_slice_b1`'s own declines.
    /// Aborts the sequence when the prefill queue is full (the sequence
    /// was originally dequeued from this same queue, so a full queue here
    /// is extremely unlikely).
    pub(super) fn route_declined_slice_waiter_to_classic(&mut self, seq: SequenceInfo) {
        let seq_id = seq.seq_id;
        tracing::debug!(
            "slice grant declined at promotion; seq {seq_id} re-queued for classic prefill"
        );
        if let Err(boxed) = self.prefill_queue.enqueue(seq) {
            tracing::warn!("slice grant declined and prefill queue full; aborting seq {seq_id}");
            self.prompt_cache_seq_ctx.remove(&seq_id);
            self.abort_sequence(
                *boxed,
                "speculative slice grant declined and prefill queue full",
            );
        }
    }

    /// Pop the next grantee per [`crate::server::batch::speculative_slice::next_grant_index`]:
    /// priority lane first with the skip-cap anti-starvation floor (issue
    /// #746). Every grant decision increments the skip counter of every
    /// NON-selected entry, so a lower-lane entry repeatedly passed over
    /// by higher-lane grants becomes overdue within
    /// `MTP_SLICE_GRANT_SKIP_CAP` decisions and must be granted next;
    /// this bounds the delay a sustained higher-lane stream can impose
    /// on any entry. (A cancelled or declined pop also counts as a
    /// decision, which only escalates the survivors sooner.)
    pub(super) fn pop_next_speculative_grantee(
        &mut self,
    ) -> Option<crate::server::batch::speculative_slice::SliceBacklogEntry> {
        let entries: Vec<_> = self
            .speculative_slice_backlog
            .iter()
            .map(|entry| (entry.priority(), entry.skipped_grants))
            .collect();
        let idx = crate::server::batch::speculative_slice::next_grant_index(&entries)?;
        for (i, entry) in self.speculative_slice_backlog.iter_mut().enumerate() {
            if i != idx {
                entry.skipped_grants += 1;
            }
        }
        self.speculative_slice_backlog.remove(idx)
    }

    /// Fail an in-flight slice job mid-session with a client-visible
    /// error: the mid-flight counterpart of
    /// [`Self::fail_speculative_slice_start`], used by the defensive arm
    /// of [`Self::execute_speculative_slice_round`] and by
    /// drafter-acquisition failures at parked-job promotion (issue #746).
    /// Any drafter still held by the job is dropped with it, so the next
    /// speculative request lazily reloads.
    pub(super) fn fail_inflight_slice_job(
        &mut self,
        job: crate::server::batch::speculative_slice::MtpSliceJob,
        msg: &str,
    ) {
        let seq_id = job.seq.seq_id;
        let _ = job
            .seq
            .response_tx
            .send(GenerateEvent::Error(format!("Speculative burst: {msg}")));
        drop(job);
        self.prompt_cache_seq_ctx.remove(&seq_id);
        self.release_sequence_caches(seq_id);
        self.batch_metrics.record_sequence_completed(0);
        self.batch_observability.record_sequence_completed();
        self.publish_metrics();
    }

    /// Fail a slice start with a client-visible error: the slice
    /// counterpart of the legacy burst's `BurstOutcome::Error` arm
    /// (`emit_error_and_finalize` + an errored `BurstFinalized`), with the
    /// identical `"Speculative burst: {msg}"` client message.
    pub(super) fn fail_speculative_slice_start(
        &mut self,
        seq: SequenceInfo,
        msg: &str,
        burst_start: Instant,
    ) {
        let seq_id = seq.seq_id;
        let _ = seq
            .response_tx
            .send(GenerateEvent::Error(format!("Speculative burst: {msg}")));
        drop(seq);
        let wall_ms = burst_start.elapsed().as_secs_f64() * 1000.0;
        self.finish_speculative_b1(crate::server::batch::speculative_burst::BurstFinalized {
            seq_id,
            tokens_generated: 0,
            prompt_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            healthy_finish: false,
            mtp_profile: None,
            burst_wall_ms: wall_ms,
            burst_active_ms: wall_ms,
            slices: 1,
        });
    }
}
