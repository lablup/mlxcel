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

//! The scheduler's execution of a [`PrefillPlan`] (ADR 0007, issue #2170).
//!
//! The plan is a pure function of the sequence (prompt length, adopted
//! prompt-cache prefix, history boundary), the server's chunk and the model's
//! capabilities, so it is rebuilt from the sequence on every tick instead of
//! being parked next to it; `SequenceInfo::prefill_offset` is the cursor into
//! it. One tick runs one piece, so a long prompt keeps interleaving with
//! decode (ADR 0005), with one exception that keeps the pre-plan tick shape:
//! the history-boundary segment (issue #1143) is followed by the next piece in
//! the same tick, because the segment is the prompt cache's snapshot point and
//! not a chunk the tick policy scheduled.
//!
//! ## Not bit-exact across partitions, deliberately
//!
//! Two forwards do not reduce in the same order as one, so a prefill split
//! differently from another produces KV and logits that differ in the last
//! bit, and a greedy stream can flip at a near-tie token. The split at the
//! history boundary, `--prefill-chunk-size`, and a prompt-cache hit (which
//! forwards only the suffix after the adopted prefix) are all partitions. What
//! the plan guarantees is the converse: two prefills of the same prompt that
//! forward the same pieces from the same KV state are bitwise identical, so a
//! hit whose adopted prefix ends on a split point of the cold plan reproduces
//! the cold run exactly ([`PrefillPlan::reproduces`]). `docs/CONTINUOUS_BATCHING.md`
//! records the measured divergence of a hit that does not. An operator who
//! needs the unsplit shape of a snapshot family's cold prefill sets
//! `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1`.

use super::*;
use mlxcel_core::prefill_plan::{PrefillCaps, PrefillInput, PrefillPiece, PrefillPlan};

/// Why a prefill piece did not complete.
pub(super) enum PieceFailure {
    /// The request must be aborted with this message.
    Abort(String),
    /// The piece's graph evaluation threw (#822): abort the request and let
    /// the scheduler decide whether the backend has failed too often.
    EvalFailed(String),
}

impl BatchScheduler {
    /// The plan for `seq`'s prefill under the server's chunk (see
    /// [`mlxcel_core::prefill_plan`]).
    ///
    /// Used by: `execute_prefill`, `start_chunked_prefill`,
    /// `continue_chunked_prefill`, `chunked_prefill_reserved_blocks`, the
    /// prefill-role handoff.
    pub(super) fn prefill_plan_for(&self, seq: &SequenceInfo) -> PrefillPlan {
        self.prefill_plan_for_with_chunk(seq, self.prefill_chunk_size)
    }

    /// [`Self::prefill_plan_for`] with an explicit chunk; `0` keeps every
    /// segment one forward, which is the partition of a full prefill.
    pub(super) fn prefill_plan_for_with_chunk(
        &self,
        seq: &SequenceInfo,
        chunk: usize,
    ) -> PrefillPlan {
        let input = if seq.vlm_embeddings.is_some() {
            // The scheduler forwards pre-merged embeddings as they are; it
            // cannot extend them to a padded length (#1201).
            PrefillInput::Embeddings {
                executor_pads: false,
            }
        } else {
            PrefillInput::Tokens
        };
        let caps =
            PrefillCaps::for_model(self.engine.model(), should_align_prefill()).with_input(input);
        PrefillPlan::with_prefix(
            seq.prompt_tokens.len(),
            seq.prefill_start_offset,
            self.history_boundary_split(seq),
            chunk,
            caps,
        )
    }

    /// The padded length of a cold batched cohort whose longest row has
    /// `max_len` tokens: the plan's single piece for that row.
    pub(super) fn batched_prefill_padded_len(&self, max_len: usize) -> usize {
        let caps = PrefillCaps::for_model(self.engine.model(), should_align_prefill());
        PrefillPlan::new(max_len, 0, caps)
            .pieces()
            .first()
            .map_or(max_len, |piece| piece.padded_len)
    }

    /// Prefill `seq` the way its plan says: one forward when the plan is a
    /// single piece, else the first piece now and the rest across later
    /// ticks.
    ///
    /// Used by: `execute_prefill`, the prefill-role handoff.
    pub(super) fn run_planned_prefill(&mut self, seq: SequenceInfo) {
        if self.prefill_plan_for(&seq).is_single_pass() {
            self.execute_full_prefill(seq);
        } else {
            self.start_chunked_prefill(seq);
        }
    }

    /// Run one piece of `plan` for `seq`: build its (padded) input and mask,
    /// forward it, evaluate, trim the pad positions it wrote, sync the paged
    /// shadow and the `--max-kv-size` cap, and advance the cursor.
    ///
    /// `continuation` is a piece run on a later tick than the first, which
    /// must reserve its blocks before its forward (issue #2088): a MixedStep
    /// tick has just run a decode step, and nothing else restores the
    /// set-aside once the pool falls below it.
    pub(super) fn run_prefill_piece(
        &mut self,
        seq: &mut SequenceInfo,
        plan: &PrefillPlan,
        piece: &PrefillPiece,
        continuation: bool,
    ) -> Result<UniquePtr<mlxcel_core::MlxArray>, PieceFailure> {
        let _span = self.announce_prefill_span(seq);
        let tokens = &seq.prompt_tokens[piece.range.clone()];
        // The mask anchors to the positions already in the KV cache. For a
        // model-owned family the pool entry holds no `KVCache` to read; its
        // own offset equals the cursor, since every earlier padded piece was
        // rewound (#1755). For pooled caches the cache's own offset is read,
        // because a `--max-kv-size` trim between pieces can leave it below
        // the cursor.
        let kv_offset = if self.engine.model().supports_batching() {
            self.engine
                .pool_mut()
                .get_caches_mut(seq.seq_id)
                .ok_or_else(|| {
                    PieceFailure::Abort("Cache not found for sequence during prefill".into())
                })?
                .first()
                .map_or(piece.range.start as i32, |c| c.offset)
        } else {
            piece.range.start as i32
        };
        let (input, pad_mask) = mlxcel_core::engine::piece_input(plan, piece, tokens, kv_offset);
        if continuation && !self.reserve_prefill_chunk_blocks(seq.seq_id, piece.padded_len) {
            let total = self.engine.pool().paged_block_budget().unwrap_or_default();
            return Err(PieceFailure::Abort(format!(
                "KV cache budget exhausted: no free blocks in the {total}-block KV cache budget to continue the chunked prefill"
            )));
        }
        // The engine runs the piece: forward, then (for a non-terminal piece)
        // the forced eval that releases its transients before the next piece's
        // graph is built and fails just this request on an MLX throw (#822),
        // then the pad trim so the next piece and decode begin at the correct
        // cache offset. The terminal piece's logits are evaluated by the first
        // sample in `finish_prefill`, exactly as the single-forward prefill
        // always was.
        let outcome = self
            .engine
            .prefill(&PrefillStep {
                seq_id: seq.seq_id,
                input: &input,
                embeddings: None,
                mask: pad_mask.as_deref(),
                last_pos: piece.last_real_pos(),
                trim_excess: piece.trim_after().unwrap_or(0) as i32,
                eval: !plan.is_terminal(piece),
            })
            .map_err(|_| {
                PieceFailure::Abort("Cache not found for sequence during prefill".into())
            })?;
        self.record_eval_outcome(outcome.eval)
            .map_err(PieceFailure::EvalFailed)?;
        // A pad trim that could not rewind the model's own state leaves its
        // offset ahead of the token count; never decode from that (#1755).
        outcome.trim.map_err(PieceFailure::Abort)?;
        self.sync_sequence_storage(seq.seq_id);
        // H2: enforce the `--max-kv-size` cap after every piece so the live
        // window stays bounded across a long prompt instead of engaging only
        // once the whole prefill completes. A cheap early-return with no cap.
        self.enforce_max_kv_size_for(seq.seq_id, seq.retention);
        seq.prefill_offset = piece.range.end;
        Ok(outcome.logits)
    }

    /// Run the pieces of `plan` from `seq`'s cursor: every piece when `all`,
    /// else the next piece and, when that piece is the history-boundary
    /// segment, the one after it too. Returns the last piece's logits.
    ///
    /// After a boundary piece the model state is snapshotted into the prompt
    /// cache (issue #1143) before the next piece runs.
    pub(super) fn run_prefill_pieces(
        &mut self,
        seq: &mut SequenceInfo,
        plan: &PrefillPlan,
        all: bool,
        continuation: bool,
    ) -> Result<UniquePtr<mlxcel_core::MlxArray>, PieceFailure> {
        let mut logits = None;
        while let Some(piece) = plan.piece_starting_at(seq.prefill_offset).cloned() {
            logits = Some(self.run_prefill_piece(seq, plan, &piece, continuation)?);
            if plan.ends_at_boundary(&piece) {
                self.insert_history_boundary_snapshot(seq, piece.range.end);
                // Return the segment's intermediates to the allocator before
                // the next piece runs; without this they stay resident through
                // the rest of the prefill.
                mlxcel_core::clear_memory_cache();
                continue;
            }
            // A chunked prefill reports one `prompt_progress` frame per
            // evaluated chunk, b10621's per-batch-iteration cadence (#1477),
            // and counts each chunk: the counter is the dispatch proof for the
            // #908 / #1011 mixed-step work (ADR 0005). A plan that is not cut
            // into chunks (a prompt that fits one chunk, `--prefill-chunk-size
            // 0`, a model that cannot chunk, the segment-plus-suffix shape of a
            // history boundary) is one unchunked prefill, which neither counts
            // nor emits a progress frame, as before the plan existed. The
            // boundary segment is not a chunk the tick policy scheduled either.
            if plan.chunk().is_some() {
                self.batch_observability.record_prefill_chunk();
                seq.report_prefill_progress(piece.range.end);
            }
            if !all {
                break;
            }
        }
        logits.ok_or_else(|| {
            PieceFailure::Abort("Prefill had no suffix tokens to process".to_string())
        })
    }

    /// Abort `seq` for a piece failure and, after an eval throw, let the
    /// scheduler count it toward shutting the backend down.
    pub(super) fn fail_prefill_piece(&mut self, seq: SequenceInfo, failure: PieceFailure) {
        match failure {
            PieceFailure::Abort(msg) => self.abort_sequence(seq, &msg),
            PieceFailure::EvalFailed(msg) => {
                self.abort_sequence(seq, &msg);
                self.eval_failures_exhausted();
            }
        }
    }

    /// Begin a multi-piece prefill: run the first piece (and the piece after a
    /// history-boundary segment) now and park the sequence for continuation
    /// on later ticks.
    ///
    /// `seq.prefill_start_offset` is the adopted prompt-cache prefix, so the
    /// first piece starts after it.
    pub(super) fn start_chunked_prefill(&mut self, mut seq: SequenceInfo) {
        let plan = self.prefill_plan_for(&seq);
        let _span = tracing::info_span!(
            "chunked_prefill_start",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
            chunk_size = self.prefill_chunk_size,
            cached = seq.already_cached_tokens,
            start = seq.prefill_start_offset,
            boundary = plan.boundary(),
        )
        .entered();

        // Reset internal caches for non-batching models (same as execute_full_prefill).
        if !self.engine.model().supports_batching() {
            let _ = self.engine.model().make_caches();
        }
        // Counter reflects only the work the model actually runs.
        self.batch_observability
            .record_prefill_start(plan.forwarded_len());
        seq.prefill_offset = plan.adopted();

        let logits = match self.run_prefill_pieces(&mut seq, &plan, false, false) {
            Ok(logits) => logits,
            Err(failure) => {
                self.fail_prefill_piece(seq, failure);
                return;
            }
        };
        mlxcel_core::clear_memory_cache();

        tracing::debug!(
            "Chunked prefill: seq {} pieces {:?}..{}/{} tokens",
            seq.seq_id,
            plan.adopted(),
            seq.prefill_offset,
            seq.prompt_tokens.len()
        );

        // A prompt-cache hit can leave a suffix that fits in one piece even
        // though the full prompt cleared the chunking threshold, and the
        // boundary segment plus one piece can already cover the prompt. Finish
        // now rather than park a sequence with nothing to continue (#179).
        if seq.prefill_offset >= seq.prompt_tokens.len() {
            let eos_tokens = merged_eos_token_ids(
                self.engine.model().eos_token_ids(),
                &seq.sampling.stop_token_ids,
            );
            let needs_history = seq.sampling.needs_token_history();
            let token_history = initial_token_history(&seq.prompt_tokens, needs_history);
            self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
            return;
        }

        // Store the sequence for continuation
        self.chunked_prefill_seq = Some(seq);
    }

    /// Continue a chunked prefill that is already in progress: one more piece.
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
    /// pool and defeats CUDA-graph reuse, ml-explore/mlx#2358). #908 introduced
    /// the first interleaved caller (`MixedStep`) and #1011 made the default
    /// policy interleaved too, so the condition is read from the active batch:
    /// there is one source of truth for "is decode live" and it is the active
    /// batch.
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

        let plan = self.prefill_plan_for(&seq);
        if plan.piece_starting_at(seq.prefill_offset).is_none() {
            self.abort_sequence(
                seq,
                "Chunked prefill continuation had no remaining tokens to process",
            );
            return false;
        }
        let logits = match self.run_prefill_pieces(&mut seq, &plan, false, true) {
            Ok(logits) => logits,
            Err(failure) => {
                self.fail_prefill_piece(seq, failure);
                return false;
            }
        };

        tracing::debug!(
            "Chunked prefill: seq {} chunk ..{}/{} tokens",
            seq.seq_id,
            seq.prefill_offset,
            seq.prompt_tokens.len(),
        );

        if !decode_batch_live {
            mlxcel_core::clear_memory_cache();
        }

        if seq.prefill_offset < seq.prompt_tokens.len() {
            // More chunks remain -- store and yield back to the scheduler.
            self.chunked_prefill_seq = Some(seq);
            return true;
        }

        // Final chunk -- complete the prefill and sample the first token
        let eos_tokens = merged_eos_token_ids(
            self.engine.model().eos_token_ids(),
            &seq.sampling.stop_token_ids,
        );
        let needs_history = seq.sampling.needs_token_history();
        let token_history = initial_token_history(&seq.prompt_tokens, needs_history);

        self.finish_prefill(seq, logits, eos_tokens, token_history, needs_history);
        true
    }
}
