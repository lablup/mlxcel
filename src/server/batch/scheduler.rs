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

//! Real MTP dispatch methods kept at the legacy source path for source-inspection tests.
//!
//! `src/server/batch/speculative_burst_tests.rs` reads this file directly with
//! `include_str!("scheduler.rs")` to ensure every MTP-capable `LoadedModel`
//! variant is handled at every scheduler dispatch site. The directory module
//! compiles this exact file as `scheduler::mtp_dispatch`, so the scanned source
//! remains production code rather than a duplicated marker list.

use super::*;

impl BatchScheduler {
    /// Validate-and-bind a drafter against the loaded Gemma 4 target, the
    /// identical contracts as `run_mtp_burst` (bind is NOT called inside
    /// the generator; omitting it silently yields one seed-bonus token).
    /// Shared by the slice-0 start ([`Self::start_mtp_slice_b1`]) and the
    /// parked-job promotion (issue #746), for which the re-bind is
    /// idempotent on the handle returned at park time (it recomputes the
    /// same target-derived state) and required after a defensive
    /// from-disk reload.
    pub(super) fn bind_drafter_to_target(
        &self,
        drafter: &mut Box<dyn mlxcel_core::drafter::Drafter>,
    ) -> Result<(), String> {
        fn compat_and_bind(
            drafter: &mut Box<dyn mlxcel_core::drafter::Drafter>,
            target_lm: &dyn LanguageModel,
        ) -> Result<(), String> {
            drafter
                .validate_target_compat(target_lm)
                .map_err(|e| format!("MTP drafter incompatible with target: {e}"))?;
            drafter
                .bind(target_lm)
                .map_err(|e| format!("MTP drafter bind failed: {e}"))
        }
        match &self.model {
            LoadedModel::Gemma4(wrapper) => compat_and_bind(drafter, wrapper),
            LoadedModel::Gemma4VLM(vlm) => compat_and_bind(drafter, vlm),
            LoadedModel::Gemma4Unified(unified) => compat_and_bind(drafter, unified),
            LoadedModel::Qwen35(qwen) | LoadedModel::Qwen35Moe(qwen) => {
                compat_and_bind(drafter, qwen)
            }
            LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => {
                compat_and_bind(drafter, vlm)
            }
            LoadedModel::Inkling(inkling) => compat_and_bind(drafter, inkling),
            LoadedModel::InklingVLM(vlm) => compat_and_bind(drafter, &vlm.text),
            // Unreachable per the callers' variant gates; produce a clean
            // per-request error rather than panicking.
            _ => Err(
                "MTP slice: unsupported target after variant gate (should not happen)".to_string(),
            ),
        }
    }

    /// Start a tick-cooperative B=1 MTP slice for `seq` (issue #734).
    ///
    /// Applies the same gates as the legacy `run_mtp_burst` in the same
    /// order (variant gate before drafter IO, adopted-prefix suffix
    /// resolution, drafter lazy-load + take + compat-check + bind) and
    /// then runs slice 0 (prefill + seed + first bonus) inline in this
    /// tick. Returns `None` when the request was handled (the slice job
    /// is parked for later ticks, the request finished within slice 0, or
    /// it failed with a client-visible error) and `Some(seq)` when the
    /// slice path declines and the caller should route the request through
    /// classic decode, mirroring `try_run_burst_b1`'s contract.
    pub(super) fn start_mtp_slice_b1(&mut self, mut seq: SequenceInfo) -> Option<SequenceInfo> {
        let burst_start = Instant::now();
        if seq.prompt_tokens.is_empty() {
            // Defensive: mirrors try_run_burst_b1's empty-prompt decline
            // (the scheduler already rejects empty prompts at enqueue).
            return Some(seq);
        }
        let block_size = match &self.speculative_dispatch {
            crate::server::SpeculativeDispatch::Mtp { block_size, .. } => *block_size as usize,
            // Caller guarantees an Mtp dispatch; decline defensively.
            _ => return Some(seq),
        };
        if block_size < 2 {
            self.fail_speculative_slice_start(
                seq,
                &format!("MTP burst: block_size={block_size} < 2 produces no draft proposals"),
                burst_start,
            );
            return None;
        }

        // Variant gate BEFORE any drafter IO, same rationale and message
        // as `run_mtp_burst`: an unsupported pairing declines to classic
        // without surfacing a confusing drafter-load error.
        if !crate::server::batch::speculative_burst::mtp_capable_target(&self.model, block_size) {
            tracing::warn!(
                "MTP speculative dispatch declined: target is not \
                 Gemma 4 (text, VLM, or Unified) or Qwen 3.5 (text or VLM), \
                 or its verify block is not byte-identical to classic decode \
                 at block_size={block_size} on this hardware (see the \
                 exactness-probe log line above); falling back to classic decode",
            );
            return Some(seq);
        }

        // Adopted prompt-cache prefix (issue #518): resolve where the
        // suffix prefill starts; a degenerate offset (whole prompt cached)
        // declines to classic, which owns that edge. Checked BEFORE taking
        // the drafter so the decline leaves the slot untouched.
        let prefill_start_offset =
            match crate::server::batch::speculative_burst::mtp_prefill_suffix_start(
                seq.prefill_start_offset,
                seq.prompt_tokens.len(),
            ) {
                Some(offset) => offset,
                None => {
                    tracing::debug!(
                        "MTP speculative slice declined for seq {}: prefill_start_offset={} \
                     covers the whole prompt (len={}); falling back to classic decode",
                        seq.seq_id,
                        seq.prefill_start_offset,
                        seq.prompt_tokens.len(),
                    );
                    return Some(seq);
                }
            };

        // Drafter: lazy-load, take, compat-check, bind, the identical
        // contracts as `run_mtp_burst` (bind is NOT called inside the
        // generator; omitting it silently yields one seed-bonus token).
        if let Err(e) = self.speculative_drafter_slot.ensure_loaded() {
            self.fail_speculative_slice_start(seq, &e, burst_start);
            return None;
        }
        let Some(mut drafter) = self.speculative_drafter_slot.take() else {
            self.fail_speculative_slice_start(
                seq,
                "drafter slot empty after ensure_loaded",
                burst_start,
            );
            return None;
        };
        if let Err(msg) = self.bind_drafter_to_target(&mut drafter) {
            // Drop the failed drafter handle so the next request lazily
            // reloads from disk, same slot semantics as `run_mtp_burst`.
            drop(drafter);
            self.fail_speculative_slice_start(seq, &msg, burst_start);
            return None;
        }

        // Commit: the slice owns the request lifecycle from here. The
        // legacy burst transitions Queued -> Prefilling at success
        // finalize; the slice transitions up front because the request is
        // genuinely prefilling in this tick and stays in flight across
        // later ticks. `finalize_burst_stream` performs the
        // Prefilling -> Finished(reason) transition, as on the legacy path.
        seq.prefill_start = Some(burst_start);
        if let Err(err) = seq.state.transition_to(SequenceState::Prefilling) {
            self.speculative_drafter_slot.restore_unused(drafter);
            self.fail_speculative_slice_start(
                seq,
                &format!("State transition error: {err}"),
                burst_start,
            );
            return None;
        }

        // History-dependent-penalty context for the first-bonus sample,
        // identical to `run_mtp_burst` / the classic path's first token.
        let token_history =
            initial_token_history(&seq.prompt_tokens, seq.sampling.needs_token_history());
        // [#736] Classic-step probe rounds while the adaptive policy is
        // profiling; they run in the first slices exactly as they run in
        // the first rounds of a legacy burst.
        let probe_rounds = self
            .mtp_policy
            .as_ref()
            .map(|p| p.profile_probe_rounds())
            .unwrap_or(0);
        let model_eos = self.model.eos_token_ids();

        // Slice 0: prefill + seed + first bonus, streamed immediately.
        let started = match &self.model {
            LoadedModel::Gemma4(wrapper) => {
                let adapter = Gemma4MtpTargetAdapter::new_with_block_size(
                    wrapper,
                    Some(seq.seq_id),
                    block_size,
                )
                .with_prefill_start_offset(prefill_start_offset);
                Ok(
                    crate::server::batch::speculative_slice::begin_slice_session(
                        adapter,
                        drafter,
                        seq,
                        &self.tokenizer,
                        model_eos,
                        block_size,
                        probe_rounds,
                        prefill_start_offset,
                        &token_history,
                    ),
                )
            }
            LoadedModel::Gemma4VLM(vlm) => {
                let adapter = Gemma4VLMtpTargetAdapter::new_with_block_size(
                    vlm,
                    Some(seq.seq_id),
                    block_size,
                )
                .with_prefill_start_offset(prefill_start_offset);
                Ok(
                    crate::server::batch::speculative_slice::begin_slice_session(
                        adapter,
                        drafter,
                        seq,
                        &self.tokenizer,
                        model_eos,
                        block_size,
                        probe_rounds,
                        prefill_start_offset,
                        &token_history,
                    ),
                )
            }
            LoadedModel::Gemma4Unified(unified) => {
                let adapter = Gemma4UnifiedMtpTargetAdapter::new_with_block_size(
                    unified,
                    Some(seq.seq_id),
                    block_size,
                )
                .with_prefill_start_offset(prefill_start_offset);
                Ok(
                    crate::server::batch::speculative_slice::begin_slice_session(
                        adapter,
                        drafter,
                        seq,
                        &self.tokenizer,
                        model_eos,
                        block_size,
                        probe_rounds,
                        prefill_start_offset,
                        &token_history,
                    ),
                )
            }
            // Qwen 3.5 family (#1165): same slice-0 shape; the adapter is a
            // stateless per-tick view over the model's sequence slot, which
            // is exactly why the cache-ownership design routes through
            // `*_for_sequence` wrappers instead of an adapter-owned cache.
            LoadedModel::Qwen35(qwen) | LoadedModel::Qwen35Moe(qwen) => {
                let adapter = crate::models::qwen3_5_mtp_target::Qwen35MtpTargetAdapter::new(
                    qwen,
                    Some(seq.seq_id),
                )
                .with_prefill_start_offset(prefill_start_offset);
                Ok(
                    crate::server::batch::speculative_slice::begin_slice_session(
                        adapter,
                        drafter,
                        seq,
                        &self.tokenizer,
                        model_eos,
                        block_size,
                        probe_rounds,
                        prefill_start_offset,
                        &token_history,
                    ),
                )
            }
            LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => {
                let adapter = crate::models::qwen3_5_mtp_target::Qwen35VLMtpTargetAdapter::new(
                    vlm,
                    Some(seq.seq_id),
                )
                .with_prefill_start_offset(prefill_start_offset);
                Ok(
                    crate::server::batch::speculative_slice::begin_slice_session(
                        adapter,
                        drafter,
                        seq,
                        &self.tokenizer,
                        model_eos,
                        block_size,
                        probe_rounds,
                        prefill_start_offset,
                        &token_history,
                    ),
                )
            }
            LoadedModel::Inkling(inkling) => {
                let adapter = crate::models::inkling_mtp_target::InklingMtpTargetAdapter::new(
                    inkling,
                    Some(seq.seq_id),
                )
                .with_prefill_start_offset(prefill_start_offset);
                Ok(
                    crate::server::batch::speculative_slice::begin_slice_session(
                        adapter,
                        drafter,
                        seq,
                        &self.tokenizer,
                        model_eos,
                        block_size,
                        probe_rounds,
                        prefill_start_offset,
                        &token_history,
                    ),
                )
            }
            LoadedModel::InklingVLM(vlm) => {
                let adapter = crate::models::inkling_mtp_target::InklingVLMtpTargetAdapter::new(
                    vlm,
                    Some(seq.seq_id),
                )
                .with_prefill_start_offset(prefill_start_offset);
                Ok(
                    crate::server::batch::speculative_slice::begin_slice_session(
                        adapter,
                        drafter,
                        seq,
                        &self.tokenizer,
                        model_eos,
                        block_size,
                        probe_rounds,
                        prefill_start_offset,
                        &token_history,
                    ),
                )
            }
            // Defensive arm rather than `unreachable!()` so a future
            // LoadedModel variant admitted by the gate above surfaces as a
            // clean per-request error instead of a worker panic.
            _ => Err((seq, drafter)),
        };
        match started {
            Ok(job) => {
                if job.finished() {
                    // The request completed within slice 0 (EOS on the
                    // first bonus, max_tokens == 1, or a degenerate
                    // session). Finalize inline, behaviourally identical
                    // to a one-tick legacy burst.
                    self.finalize_speculative_slice(job);
                } else {
                    // A fresh slot grant begins with slice 0 already
                    // executed (issue #746). At
                    // MLXCEL_MTP_SLICE_GRANT_ROUNDS=1 with other grantees
                    // waiting, slice 0 already spends the whole grant, so
                    // the job parks right here instead of getting a free
                    // extra round; the park is pure bookkeeping, the
                    // slice-0 forward stays this tick's one model action.
                    // Reachable only from waiter promotion: a direct
                    // admission always sees an empty backlog because
                    // `try_speculative_burst` parks or declines new
                    // arrivals while the backlog is non-empty.
                    self.speculative_slice_grant_slices = 1;
                    // Resolve the budget once per grant; the per-round
                    // expiry check compares against this cached value.
                    self.speculative_slice_grant_budget =
                        crate::server::batch::speculative_slice::mtp_slice_grant_rounds();
                    if crate::server::batch::speculative_slice::slice_grant_expired(
                        self.speculative_slice_grant_slices,
                        self.speculative_slice_grant_budget,
                        !self.speculative_slice_backlog.is_empty(),
                    ) {
                        self.park_speculative_slice(Box::new(job));
                    } else {
                        self.speculative_slice = Some(Box::new(job));
                    }
                }
                None
            }
            Err((seq, drafter)) => {
                drop(drafter);
                self.fail_speculative_slice_start(
                    seq,
                    "MTP slice: unsupported target after variant gate (should not happen)",
                    burst_start,
                );
                None
            }
        }
    }

    /// Execute one tick-cooperative speculative action (issue #734): take
    /// the active slice job, reconstruct the borrowing target adapter for
    /// this tick, run exactly one generator round, stream its tokens, and
    /// either keep the job active, park it at an expired grant boundary
    /// (issue #746), or finalize the request. With the slot empty and the
    /// grant backlog non-empty (right after a rotation), the tick instead
    /// promotes the next grantee: a parked job's round runs below in this
    /// same tick (its promotion is cheap bookkeeping), while a waiter's
    /// slice 0 IS the tick's one action, preserving the #734 HOL bound of
    /// one model action per tick.
    pub(super) fn execute_speculative_slice_round(&mut self) {
        if self.speculative_slice.is_none() && !self.promote_next_speculative_grantee() {
            // The tick's speculative action already ran inside the
            // promotion (a waiter's slice 0), or the backlog emptied
            // (cancelled / declined entries resolved with O(1)
            // bookkeeping only).
            return;
        }
        let Some(mut job) = self.speculative_slice.take() else {
            // Defensive: decide_action only emits SpeculativeRound while
            // speculative work is pending, and the promotion above either
            // installed a job or reported the tick as consumed.
            return;
        };
        // Apply the slice owner's runtime-LoRA snapshot before its round's
        // forwards (#1439).
        let slice_lora = job.seq.lora_scales.clone();
        self.ensure_lora_applied(slice_lora.as_ref());
        let _span = tracing::info_span!(
            "speculative_slice_round",
            seq_id = %job.seq.seq_id,
            slice = job.slices,
        )
        .entered();
        let stepped = match &self.model {
            LoadedModel::Gemma4(wrapper) => {
                let adapter = Gemma4MtpTargetAdapter::new_with_block_size(
                    wrapper,
                    Some(job.seq.seq_id),
                    job.block_size,
                )
                .with_prefill_start_offset(job.prefill_start_offset);
                crate::server::batch::speculative_slice::step_slice_session(
                    adapter,
                    &mut job,
                    &self.tokenizer,
                );
                true
            }
            LoadedModel::Gemma4VLM(vlm) => {
                let adapter = Gemma4VLMtpTargetAdapter::new_with_block_size(
                    vlm,
                    Some(job.seq.seq_id),
                    job.block_size,
                )
                .with_prefill_start_offset(job.prefill_start_offset);
                crate::server::batch::speculative_slice::step_slice_session(
                    adapter,
                    &mut job,
                    &self.tokenizer,
                );
                true
            }
            LoadedModel::Gemma4Unified(unified) => {
                let adapter = Gemma4UnifiedMtpTargetAdapter::new_with_block_size(
                    unified,
                    Some(job.seq.seq_id),
                    job.block_size,
                )
                .with_prefill_start_offset(job.prefill_start_offset);
                crate::server::batch::speculative_slice::step_slice_session(
                    adapter,
                    &mut job,
                    &self.tokenizer,
                );
                true
            }
            LoadedModel::Qwen35(qwen) | LoadedModel::Qwen35Moe(qwen) => {
                let adapter = crate::models::qwen3_5_mtp_target::Qwen35MtpTargetAdapter::new(
                    qwen,
                    Some(job.seq.seq_id),
                )
                .with_prefill_start_offset(job.prefill_start_offset);
                crate::server::batch::speculative_slice::step_slice_session(
                    adapter,
                    &mut job,
                    &self.tokenizer,
                );
                true
            }
            LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => {
                let adapter = crate::models::qwen3_5_mtp_target::Qwen35VLMtpTargetAdapter::new(
                    vlm,
                    Some(job.seq.seq_id),
                )
                .with_prefill_start_offset(job.prefill_start_offset);
                crate::server::batch::speculative_slice::step_slice_session(
                    adapter,
                    &mut job,
                    &self.tokenizer,
                );
                true
            }
            LoadedModel::Inkling(inkling) => {
                let adapter = crate::models::inkling_mtp_target::InklingMtpTargetAdapter::new(
                    inkling,
                    Some(job.seq.seq_id),
                )
                .with_prefill_start_offset(job.prefill_start_offset);
                crate::server::batch::speculative_slice::step_slice_session(
                    adapter,
                    &mut job,
                    &self.tokenizer,
                );
                true
            }
            LoadedModel::InklingVLM(vlm) => {
                let adapter = crate::models::inkling_mtp_target::InklingVLMtpTargetAdapter::new(
                    vlm,
                    Some(job.seq.seq_id),
                )
                .with_prefill_start_offset(job.prefill_start_offset);
                crate::server::batch::speculative_slice::step_slice_session(
                    adapter,
                    &mut job,
                    &self.tokenizer,
                );
                true
            }
            _ => false,
        };
        if !stepped {
            // Defensive: the model cannot change mid-flight (slice 0 only
            // starts on an `mtp_capable_target` family: Gemma 4 assistant or
            // Qwen 3.5 MTP). Fail the request cleanly; the
            // drafter is dropped with the job so the next speculative
            // request lazily reloads it.
            self.fail_inflight_slice_job(
                *job,
                "MTP slice target variant changed mid-flight (should not happen)",
            );
            return;
        }
        if job.finished() {
            self.finalize_speculative_slice(*job);
        } else {
            self.speculative_slice_grant_slices += 1;
            // The uncontended per-round added cost is this counter
            // increment plus a comparison against the budget cached at
            // grant start (no per-round env read; see
            // `speculative_slice_grant_budget`).
            if crate::server::batch::speculative_slice::slice_grant_expired(
                self.speculative_slice_grant_slices,
                self.speculative_slice_grant_budget,
                !self.speculative_slice_backlog.is_empty(),
            ) {
                // Round boundary with the grant spent and other grantees
                // waiting (issue #746): park the job and leave the slot
                // empty. The next grantee's work runs on a LATER tick
                // (one action per tick preserves the #734 HOL bound).
                self.park_speculative_slice(job);
            } else {
                self.speculative_slice = Some(job);
            }
        }
    }

    /// Park the active slice job at an expired grant boundary (issue
    /// #746): release its drafter back through the worker slot (the same
    /// end-of-session plumbing `finalize_speculative_slice` uses, whose
    /// `Drafter::reset` is the trait default no-op for the MTP assistant
    /// drafter; see `MtpSliceJob::attach_drafter` for the correctness
    /// argument) and push the job onto the grant backlog ring.
    ///
    /// The Qwen 3.5 MTP drafter's reset is NOT a no-op: it clears the
    /// drafter-owned KV history. That is still correct at a park boundary —
    /// the shared worker handle may serve other grantees before this job is
    /// promoted, so per-session drafter state cannot survive rotation, and
    /// the resumed session's `set_shared_kv` re-anchors into the documented
    /// empty-cache mode (reduced draft context, identical output).
    pub(super) fn park_speculative_slice(
        &mut self,
        mut job: Box<crate::server::batch::speculative_slice::MtpSliceJob>,
    ) {
        if let Some(drafter) = job.take_drafter() {
            let target_lm: Option<&dyn LanguageModel> = match &self.model {
                LoadedModel::Gemma4(wrapper) => Some(wrapper),
                LoadedModel::Gemma4VLM(vlm) => Some(vlm),
                LoadedModel::Gemma4Unified(unified) => Some(unified),
                LoadedModel::Qwen35(qwen) | LoadedModel::Qwen35Moe(qwen) => Some(qwen),
                LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => Some(vlm),
                LoadedModel::Inkling(inkling) => Some(inkling),
                LoadedModel::InklingVLM(vlm) => Some(&vlm.text),
                _ => None,
            };
            match target_lm {
                Some(lm) => self.speculative_drafter_slot.return_drafter(drafter, lm),
                // Defensive: without a resolvable target the reset cannot
                // run; drop the handle so the next grantee lazily reloads.
                None => drop(drafter),
            }
        }
        tracing::debug!(
            seq_id = %job.seq.seq_id,
            slices = job.slices,
            backlog = self.speculative_slice_backlog.len(),
            "speculative slice grant expired; parking the job and rotating the slot"
        );
        self.speculative_slice_backlog
            .push_back(crate::server::batch::speculative_slice::SliceBacklogEntry::parked(job));
        self.speculative_slice_grant_slices = 0;
    }

    /// Finalize a finished slice job: return the drafter to the worker
    /// slot (WITH the end-of-request reset), emit the tail + `Done` event
    /// through the shared stream finalize, build the `BurstFinalized`
    /// payload with the per-slice HOL accounting, and run the shared B=1
    /// bookkeeping ([`Self::finish_speculative_b1`]).
    pub(super) fn finalize_speculative_slice(
        &mut self,
        mut job: crate::server::batch::speculative_slice::MtpSliceJob,
    ) {
        // Return the drafter for the next request or grantee. While a job
        // holds the slot, the BETWEEN-slice holds skip `return_drafter`
        // (there is nothing to hand off); the END-of-session return here
        // resets, exactly like the legacy burst, and the park boundary
        // (issue #746) routes through the same `return_drafter` plumbing.
        // Correctness does not depend on what `reset` does to drafter-side
        // state either way (the Gemma 4 assistant drafter's reset is the
        // trait default no-op; the Qwen 3.5 MTP drafter's reset destroys
        // its accumulated KV history instead): see
        // `MtpSliceJob::attach_drafter` for the full argument.
        if let Some(drafter) = job.take_drafter() {
            let target_lm: Option<&dyn LanguageModel> = match &self.model {
                LoadedModel::Gemma4(wrapper) => Some(wrapper),
                LoadedModel::Gemma4VLM(vlm) => Some(vlm),
                LoadedModel::Gemma4Unified(unified) => Some(unified),
                LoadedModel::Qwen35(qwen) | LoadedModel::Qwen35Moe(qwen) => Some(qwen),
                LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => Some(vlm),
                LoadedModel::Inkling(inkling) => Some(inkling),
                LoadedModel::InklingVLM(vlm) => Some(&vlm.text),
                _ => None,
            };
            match target_lm {
                Some(lm) => self.speculative_drafter_slot.return_drafter(drafter, lm),
                // Defensive: without a resolvable target the reset cannot
                // run; drop the handle so the next request lazily reloads.
                None => drop(drafter),
            }
        }

        let finish = job
            .finish
            .take()
            .expect("finalize_speculative_slice requires a finished job");
        // The generation stats mirror the legacy burst's, which discards
        // them at finalize as well (`finalize_burst_success` ignores its
        // timing arguments); the acceptance summary is the payload.
        let _ = finish.stats;
        let seq = job.seq;
        let prompt_len = seq.prompt_tokens.len();
        // Build the adaptive-policy profile from the acceptance summary
        // (issue #333), with the same zero-round filter and batch_size=1 as the
        // legacy `run_mtp_burst`.
        let profile = finish
            .summary
            .filter(|summary| summary.rounds > 0 || summary.probe_rounds > 0)
            .map(|summary| {
                crate::server::batch::mtp_policy::MtpBurstProfile::from_summary(
                    summary, 1, prompt_len,
                )
            });
        // Client-facing acceptance counters (issue #1314), from the same
        // `Copy` summary the policy profile above reads. The session spans
        // every slice of the request, so `finish_session` already returns the
        // run's totals and nothing has to be accumulated here. `drafter_kind`
        // resolves the kind from the dispatch that admitted the slice rather
        // than assuming one, and yields `None` for a dispatch that runs no
        // drafter, in which case there is nothing to report either.
        let speculative = finish
            .summary
            .zip(self.speculative_dispatch.drafter_kind())
            .and_then(|(summary, kind)| {
                crate::server::model_provider::SpeculativeStats::from_counts(
                    kind,
                    summary.rounds,
                    summary.proposed_tokens,
                    summary.accepted_draft_tokens,
                )
            });
        let seq_id = seq.seq_id;
        let outcome = crate::server::batch::speculative_burst::finalize_burst_stream(
            &self.tokenizer,
            seq,
            &job.stream,
            speculative,
        );
        self.finish_speculative_b1(crate::server::batch::speculative_burst::BurstFinalized {
            seq_id,
            tokens_generated: outcome.tokens_generated,
            prompt_tokens: outcome.prompt_tokens,
            generated_tokens: outcome.generated_tokens,
            healthy_finish: outcome.healthy_finish,
            mtp_profile: profile,
            // Per-slice HOL accounting (issue #734): the max single-tick
            // wall is the realized HOL bound; the total is the cumulative
            // worker occupancy across all slices.
            burst_wall_ms: job.max_slice_wall_ms,
            burst_active_ms: job.total_slice_wall_ms,
            slices: job.slices,
        });
    }
}
