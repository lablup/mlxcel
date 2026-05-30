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

//! DFlash speculative-decoding round-loop driver (B > 1, continuous
//! batching).
//!
//! Rust port of upstream
//! `references/mlx-vlm/mlx_vlm/generate.py::_dflash_rounds_batch`
//! (`mlx-vlm` HEAD, lines 944-1098). sub-13.
//!
//! ## Scope
//!
//! **B >= 1, with continuous batching and per-row GDN-aware rollback.**
//! The B = 1 fast path lives in the sibling
//! [`crate::drafter::dflash::round_loop`] module (sub-12)
//! and must NOT regress as a result of this implementation. The two
//! drivers share the [`SpeculativeTarget`] trait and the
//! [`speculative_walk`] helper; everything else is duplicated to keep
//! both hot paths optimal and obvious.
//!
//! ## What batched DFlash adds vs. the B = 1 path
//!
//! 1. **Per-row accept counts.** [`speculative_walk_batched`] returns a
//!    `Vec<usize>` of accept counts (one per row) and a `Vec<Vec<i32>>`
//!    of per-row emitted tokens. Rows in different speculative-walk
//!    states share the same verify-forward pass but diverge on what
//!    they accept.
//! 2. **Per-row GDN-aware rollback.** Every linear-attention layer's
//!    GDN state must be restored from the per-row snapshot at the
//!    accepted position for that row. Sibling rows do NOT share rollback
//!    state. The target's
//!    [`SpeculativeTarget::rollback_partial_batched`] hook receives the
//!    full `&[i32]` accept slice and handles both KV (attention) and
//!    GDN (linear-attention) state per row.
//! 3. **Per-row early-EOS / max-tokens stop.** When a row hits EOS or
//!    saturates `max_new_tokens`, it is frozen in place: subsequent
//!    verify-forwards still include that row (the batch shape stays
//!    stable) but emitted tokens are dropped on the floor. Mirrors the
//!    upstream `_dflash_rounds_batch` "stay-in-batch" semantics with
//!    the simplification that we do not filter caches — see the module
//!    docstring on the [`crate::drafter::dflash::round_loop`] for why
//!    cache filtering is out of scope for this sub-issue.
//! 4. **Left-padding awareness.** Rows with different prefill lengths
//!    are left-padded so the verify pass receives a `[B, block_size]`
//!    tensor where each row's draft block sits at the rightmost
//!    positions. The caller is responsible for the initial bonus
//!    tokens and hidden tensor having the right left-padding; the
//!    round-loop only forwards what the target produces and propagates
//!    the captured-hidden shape it gets back.
//!
//! ## Greedy parity gate
//!
//! At `temperature = 0.0`, the round loop must be byte-identical to
//! running [`crate::drafter::dflash::round_loop::DFlashGenerator::run`]
//! `B` times sequentially on the same prompts. The
//! [`tests::greedy_parity_against_single_row_run`] test pins that
//! invariant for `B = 4`.

use crate::drafter::{Drafter, DrafterError};
use crate::ffi::{self, MlxArray};
use crate::generate::{GenerationStats, SamplingConfig};
use cxx::UniquePtr;
use std::time::{Duration, Instant};

use super::round_loop::SpeculativeTarget;

/// Per-row speculative-decoding walk (B >= 1).
///
/// Rust port of upstream
/// `references/mlx-vlm/mlx_vlm/generate.py::_speculative_walk_batch`.
///
/// Given `draft_tokens` of shape `[B][K-1]` (per-row drafter proposals)
/// and `target_tokens` of shape `[B][K]` (per-row target argmax over
/// the full verify block), plus per-row `budgets[r]` (remaining
/// `max_tokens - emitted[r]`):
///
/// - Accept the longest prefix of `draft_tokens[r]` that matches the
///   target's argmax (position-by-position).
/// - Take the target's argmax at the divergence position as the bonus.
/// - Truncate the resulting row to `budgets[r]`.
///
/// Returns `(accepted_per_row, new_tokens_per_row)` with the same
/// outer length `B`.
///
/// **Greedy parity at `temp = 0.0`**: the target's argmax determines
/// every token the loop emits for every row, identically to the B = 1
/// path. So the per-row walk is the natural lifting of [`super::round_loop::speculative_walk`]
/// to a batch of independent rows.
pub(crate) fn speculative_walk_batched(
    draft_tokens: &[Vec<i32>],
    target_tokens: &[Vec<i32>],
    budgets: &[usize],
) -> (Vec<usize>, Vec<Vec<i32>>) {
    let b = draft_tokens.len();
    debug_assert_eq!(
        target_tokens.len(),
        b,
        "speculative_walk_batched: draft and target row counts must match"
    );
    debug_assert_eq!(
        budgets.len(),
        b,
        "speculative_walk_batched: budgets row count must match"
    );

    let mut accepted_per_row: Vec<usize> = Vec::with_capacity(b);
    let mut new_tokens_per_row: Vec<Vec<i32>> = Vec::with_capacity(b);

    for r in 0..b {
        let d = &draft_tokens[r];
        let t = &target_tokens[r];
        let n = d.len().min(t.len().saturating_sub(0));

        // Position-by-position match up to the first divergence or the
        // end of the drafter's proposal list.
        let mut accepted = 0;
        while accepted < n {
            if accepted >= t.len() {
                break;
            }
            if d[accepted] != t[accepted] {
                break;
            }
            accepted += 1;
        }

        // Bonus token: target's choice at the divergence position (or
        // at the end of the block if the drafter matched everything).
        let bonus_idx = accepted.min(t.len().saturating_sub(1));
        let mut new_tokens: Vec<i32> = d[..accepted].to_vec();
        if bonus_idx < t.len() {
            new_tokens.push(t[bonus_idx]);
        }

        // Per-row budget truncation.
        let budget = budgets[r];
        if new_tokens.len() > budget {
            new_tokens.truncate(budget);
        }

        accepted_per_row.push(accepted);
        new_tokens_per_row.push(new_tokens);
    }

    (accepted_per_row, new_tokens_per_row)
}

/// Per-row argmax over a `[B, seq_len, vocab]` logits tensor.
///
/// Returns `out[b][s]` = argmax of `logits[b, s, :]` for each (batch,
/// position). Materializes to a nested `Vec<Vec<i32>>` so the per-row
/// walk can iterate without re-querying MLX for each cell.
///
/// Mirrors upstream `sampler(verify_out.logits)` with the greedy
/// `sampler = argmax(axis=-1)` — stochastic batched DFlash sampling is
/// a follow-up.
fn argmax_logits_per_row(logits: &MlxArray, batch_size: i32, seq_len: i32) -> Vec<Vec<i32>> {
    let shape = ffi::array_shape(logits);
    debug_assert_eq!(shape.len(), 3, "expected [B, seq_len, vocab] logits");
    debug_assert_eq!(shape[0], batch_size, "logits batch dim must match B");
    debug_assert_eq!(shape[1], seq_len, "logits seq dim must match block_size");

    // argmax_last_axis reduces over the trailing axis, producing [B, seq_len].
    let argmax = ffi::argmax_last_axis(logits);
    ffi::eval(&argmax);

    // Materialize per cell. We pay the O(B * seq_len) scalar-extraction
    // tax once per round; this dominates only at very small batch sizes
    // (which is exactly the regime this driver targets). A future
    // optimization could push the per-cell extraction into a single
    // contiguous copy, but the data must be evaluated anyway.
    let mut out: Vec<Vec<i32>> = Vec::with_capacity(batch_size as usize);
    for b in 0..batch_size {
        let mut row: Vec<i32> = Vec::with_capacity(seq_len as usize);
        for s in 0..seq_len {
            let cell = ffi::slice(&argmax, &[b, s], &[b + 1, s + 1]);
            let scalar = ffi::reshape(&cell, &[]);
            row.push(ffi::item_i32(&scalar));
        }
        out.push(row);
    }
    out
}

/// Round-loop run output for the batched driver.
///
/// Mirrors [`super::round_loop::DFlashRunOutput`] but with per-row
/// emitted tokens.
pub struct DFlashBatchedRunOutput {
    /// Per-row emitted tokens (does NOT include the per-row `first_bonus`
    /// tokens the caller passed into [`DFlashBatchedGenerator::run_batched`]
    /// — the caller is expected to have already streamed those out).
    pub tokens: Vec<Vec<i32>>,
    /// Per-round, per-row accept counts. `accept_lens[r][i]` is the
    /// accept count for row `r` on round `i`.
    pub accept_lens: Vec<Vec<u32>>,
    /// Wall-clock decode statistics. Prefill is the caller's
    /// responsibility (the loop runs after prefill).
    pub stats: GenerationStats,
}

/// B > 1 DFlash speculative-decoding round-loop driver.
///
/// Construction owns the boxed drafter and the runtime parameters; the
/// target is borrowed at call time via [`Self::run_batched`]. Shares
/// the [`SpeculativeTarget`] trait surface with the B = 1
/// [`super::round_loop::DFlashGenerator`] — the two are sibling drivers,
/// not a hierarchy.
pub struct DFlashBatchedGenerator {
    drafter: Box<dyn Drafter>,
    sampler: SamplingConfig,
    block_size: u32,
    /// Mask-token id reserved for future CLI override plumbing (see
    /// [`super::round_loop::DEFAULT_MASK_TOKEN_ID`]).
    #[allow(dead_code)]
    mask_token_id: i32,
}

impl DFlashBatchedGenerator {
    /// Construct a new batched generator with the given drafter, sampler,
    /// and block size.
    pub fn new(
        drafter: Box<dyn Drafter>,
        sampler: SamplingConfig,
        block_size: u32,
        mask_token_id: i32,
    ) -> Self {
        Self {
            drafter,
            sampler,
            block_size,
            mask_token_id,
        }
    }

    /// Convenience constructor that uses the default block size and
    /// mask-token id from [`super::round_loop`].
    pub fn with_drafter(drafter: Box<dyn Drafter>, sampler: SamplingConfig) -> Self {
        Self::new(
            drafter,
            sampler,
            super::round_loop::DEFAULT_BLOCK_SIZE,
            super::round_loop::DEFAULT_MASK_TOKEN_ID,
        )
    }

    /// Drafter handle for tests that need to inspect drafter state.
    pub fn drafter(&self) -> &dyn Drafter {
        self.drafter.as_ref()
    }

    /// Drafter mutable handle for tests / advanced callers.
    pub fn drafter_mut(&mut self) -> &mut dyn Drafter {
        self.drafter.as_mut()
    }

    /// Consume the generator and return the boxed drafter handle.
    ///
    /// Used by the server-side batched speculative burst path so a loaded drafter can be reused across multiple batched
    /// bursts on the same worker thread without re-loading from disk.
    /// Mirrors [`super::round_loop::DFlashGenerator::into_drafter`] for
    /// the B > 1 driver.
    pub fn into_drafter(self) -> Box<dyn Drafter> {
        self.drafter
    }

    /// Run the batched DFlash round loop.
    ///
    /// # Arguments
    ///
    /// - `target` — implements [`SpeculativeTarget`] over the target
    ///   model's cache type.
    /// - `target_lm` — `&dyn LanguageModel` for the
    ///   [`Drafter::bind`] capability smoke test (the drafter uses this
    ///   only at bind / reset time; the hot path goes through `target`).
    /// - `caches` — the target's per-layer cache slice, already filled
    ///   by the caller's batched prefill at shape `[B, ...]` per layer.
    ///   The round loop will advance and selectively rewind these
    ///   caches (per-row).
    /// - `first_bonus` — per-row first-bonus tokens sampled from the
    ///   target's last logits after prefill. Length `B`. The loop does
    ///   NOT emit these tokens (the caller already streamed them); they
    ///   are used only to seed the first per-row verify-input.
    /// - `first_hidden` — multi-layer hidden concatenation at the last
    ///   position of the prefill, shape `[B, 1, num_capture_layers *
    ///   hidden_size]`. Caller is responsible for building this; the
    ///   round-loop only consumes it.
    /// - `stop_tokens` — token ids that should stop a row. The loop
    ///   freezes-in-place any row that emits a matching id. Empty
    ///   slice = no EOS stop (the loop runs every row to
    ///   `max_new_tokens`).
    /// - `max_new_tokens` — per-row generation budget INCLUDING the
    ///   first-bonus token. Each row emits at most `max_new_tokens - 1`
    ///   further tokens.
    ///
    /// # Returns
    ///
    /// A [`DFlashBatchedRunOutput`] carrying per-row emitted tokens (excludes
    /// the per-row `first_bonus`) and per-round, per-row accept counts.
    ///
    /// # Errors
    ///
    /// Propagates [`DrafterError`] from the drafter's `draft_block_batched` /
    /// `reset` / `bind` calls.
    // Same shape as the B = 1 `run` plus per-row inputs; further
    // collapsing would obscure the per-row contract. The classic
    // sibling B = 1 path on `DFlashGenerator::run` carries 7 args; the
    // batched peer here adds `stop_tokens` as a slice to keep the EOS
    // contract identical between the two drivers.
    #[allow(clippy::too_many_arguments)]
    pub fn run_batched<T: SpeculativeTarget>(
        &mut self,
        target: &T,
        target_lm: &dyn crate::generate::LanguageModel,
        caches: &mut [T::Cache],
        first_bonus: &[i32],
        first_hidden: UniquePtr<MlxArray>,
        stop_tokens: &[i32],
        max_new_tokens: usize,
    ) -> Result<DFlashBatchedRunOutput, DrafterError> {
        // Bind + reset the drafter against the target.
        self.drafter.bind(target_lm)?;
        self.drafter.reset(target_lm)?;

        let batch_size = first_bonus.len();
        if batch_size == 0 {
            return Err(DrafterError::DraftFailed {
                reason: "DFlash batched round loop requires B >= 1; got first_bonus = []"
                    .to_string(),
            });
        }

        let mut tokens_out: Vec<Vec<i32>> = (0..batch_size).map(|_| Vec::new()).collect();
        let mut accept_lens_per_row: Vec<Vec<u32>> = (0..batch_size).map(|_| Vec::new()).collect();

        if max_new_tokens <= 1 {
            // First bonus is the caller's own; no further work.
            return Ok(DFlashBatchedRunOutput {
                tokens: tokens_out,
                accept_lens: accept_lens_per_row,
                stats: build_zero_stats(),
            });
        }

        let decode_start = Instant::now();

        let block_size_cfg = self.block_size as usize;
        let capture_layer_ids = self
            .drafter
            .dflash_target_layer_ids()
            .filter(|ids| !ids.is_empty())
            .map(<[usize]>::to_vec)
            .unwrap_or_else(|| target.capture_layer_ids().to_vec());

        // Per-row state. `b[r]` is the active bonus for row r. `emitted[r]`
        // counts every token the caller has seen for row r including the
        // first bonus. `finished[r]` flags rows that have hit EOS or
        // saturated max_new_tokens.
        let mut b: Vec<i32> = first_bonus.to_vec();
        let mut emitted: Vec<usize> = vec![1; batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];

        // Hidden tensor of shape [B, T, dim]; T grows / shrinks each
        // round per the captured-hidden slice contract.
        let mut hidden: UniquePtr<MlxArray> = first_hidden;
        let mut total_emitted: usize = emitted.iter().sum();

        loop {
            // Stop when every row is finished. We do NOT short-circuit on
            // "all rows reached max" — the loop's exit gate below catches
            // that via `bs <= 1`.
            if finished.iter().all(|&f| f) {
                break;
            }

            // Per-round draft-block length: bs = min(block_total, min(remaining_active)).
            // Mirrors upstream
            //   remaining = [max(1, max_tokens - emitted[orig] + 1) for ... in active]
            //   bs = min(block_total, min(remaining))
            //
            // **Active-only**: finished rows are frozen in place and
            // must NOT participate in the bs computation — they would
            // collapse `bs` to 1 as soon as any row finishes and stall
            // the entire batch (load-bearing for the per-row early-EOS
            // contract). Upstream's `_dflash_rounds_batch` achieves the
            // same effect by filtering `active_idx` out of `remaining`;
            // we keep finished rows in the batch shape but skip them
            // here.
            let remaining_min: usize = (0..batch_size)
                .filter(|&r| !finished[r])
                .map(|r| {
                    // max(1, max_new_tokens - emitted[r] + 1)
                    max_new_tokens
                        .saturating_sub(emitted[r])
                        .saturating_add(1)
                        .max(1)
                })
                .min()
                .unwrap_or(1);
            let bs = block_size_cfg.min(remaining_min);
            if bs <= 1 {
                break;
            }

            // ---- Draft ----
            // Build the per-row bonus slice. For finished rows, we still
            // need a token to seed the verify-forward; reuse the last
            // bonus (it will be discarded on emit).
            let bonus_arr: Vec<i32> = b.clone();
            let draft_tokens_per_row = self.drafter.draft_block_batched(
                &bonus_arr,
                Some(hidden.as_ref().expect("hidden must be Some")),
                bs,
                &self.sampler,
            )?;
            debug_assert_eq!(
                draft_tokens_per_row.len(),
                batch_size,
                "DFlash batched drafter must return B per-row proposals"
            );
            debug_assert!(
                draft_tokens_per_row.iter().all(|p| p.len() == bs - 1),
                "DFlash batched drafter rows must each have bs - 1 proposals"
            );

            // ---- Verify ----
            // Build the verify input as [B, bs] where row r is
            // [b[r], d_r[0], ..., d_r[bs-2]]. We flatten into a single
            // Vec<i32> for the FFI factory.
            let mut verify_buf: Vec<i32> = Vec::with_capacity(batch_size * bs);
            for r in 0..batch_size {
                verify_buf.push(b[r]);
                verify_buf.extend_from_slice(&draft_tokens_per_row[r]);
            }
            let verify_input = ffi::from_slice_i32(&verify_buf, &[batch_size as i32, bs as i32]);
            let verify_out = target.verify_forward_with_capture_layers(
                &verify_input,
                caches,
                &capture_layer_ids,
            );

            // ---- Argmax sample (greedy at temp=0) of the per-row logits ----
            let target_tokens_per_row = argmax_logits_per_row(
                target.verify_logits(&verify_out),
                batch_size as i32,
                bs as i32,
            );

            // ---- Walk (per-row) ----
            let budgets: Vec<usize> = (0..batch_size)
                .map(|r| max_new_tokens.saturating_sub(emitted[r]))
                .collect();
            let (accepted_per_row, new_tokens_per_row) =
                speculative_walk_batched(&draft_tokens_per_row, &target_tokens_per_row, &budgets);

            // Record per-row accept lens. Finished rows still get an
            // entry (consistent shape across rounds) but the value is
            // the walk's count; the round simply discards their emissions.
            for r in 0..batch_size {
                accept_lens_per_row[r].push(accepted_per_row[r] as u32);
            }

            // ---- Emit (per-row, frozen rows dropped) ----
            for r in 0..batch_size {
                if finished[r] {
                    continue;
                }
                let mut hit_eos = false;
                for tok in &new_tokens_per_row[r] {
                    tokens_out[r].push(*tok);
                    emitted[r] += 1;
                    if stop_tokens.contains(tok) {
                        hit_eos = true;
                        break;
                    }
                    if emitted[r] >= max_new_tokens {
                        break;
                    }
                }
                if hit_eos || emitted[r] >= max_new_tokens {
                    finished[r] = true;
                }
            }

            // ---- Update bonus + next-round hidden ----
            //
            // Upstream:
            //   for j in range(n_active):
            //       orig = active_idx[j]
            //       if new_tokens_list[j]:
            //           b[orig] = new_tokens_list[j][-1]
            //
            // We update the bonus for every row (frozen or not). Frozen
            // rows keep using their last bonus on subsequent verify
            // forwards; this is harmless because their emissions are
            // dropped on the floor.
            for r in 0..batch_size {
                if let Some(last) = new_tokens_per_row[r].last() {
                    b[r] = *last;
                }
            }

            if finished.iter().all(|&f| f) {
                break;
            }

            // Compute the next-round hidden slice. Mirrors upstream:
            //   if min_accepted < bs - 1:
            //       max_a = max(accepted_per_row)
            //       hidden = hidden_full[:, : max_a + 1, :]
            //   else:
            //       hidden = hidden_full
            let min_accepted = accepted_per_row.iter().copied().min().unwrap_or(0);
            let max_accepted = accepted_per_row.iter().copied().max().unwrap_or(0);

            let full_hidden = target.concat_hidden_for_drafter(&verify_out);
            hidden = if min_accepted + 1 < bs {
                let full_shape = ffi::array_shape(&full_hidden);
                debug_assert_eq!(
                    full_shape.len(),
                    3,
                    "concat_hidden_for_drafter must return a 3D [B, bs, dim] tensor"
                );
                ffi::slice(
                    &full_hidden,
                    &[0, 0, 0],
                    &[full_shape[0], max_accepted as i32 + 1, full_shape[2]],
                )
            } else {
                full_hidden
            };

            // ---- Rollback (per-row, only when at least one row accepted
            //                less than bs - 1) ----
            //
            // The verify forward advanced every row's caches by `bs`
            // tokens. We accepted `accepted[r]` drafter proposals + 1
            // bonus per row → keep `accepted[r] + 1` positions per row.
            // The shared trim amount is `bs - (max_accepted + 1)`; per-row
            // tail-zeroing handles rows whose accept counts are below
            // max_accepted.
            if min_accepted + 1 < bs {
                let accepted_i32: Vec<i32> = accepted_per_row.iter().map(|&a| a as i32).collect();
                target.rollback_partial_batched(caches, &verify_out, &accepted_i32, bs as i32);
            }

            // Periodic memory cache clear. Mirrors upstream
            //   if new_total // 256 > total_emitted // 256:
            //       mx.clear_cache()
            let new_total: usize = emitted.iter().sum();
            if new_total / 256 > total_emitted / 256 {
                ffi::clear_memory_cache();
            }
            total_emitted = new_total;
        }

        let stats = GenerationStats {
            prompt_tokens: 0,
            generated_tokens: tokens_out.iter().map(|v| v.len()).sum(),
            prefill_time_ms: 0.0,
            decode_time_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
            prefill_tok_per_sec: 0.0,
            decode_tok_per_sec: tokens_per_second(
                tokens_out.iter().map(|v| v.len()).sum(),
                decode_start.elapsed(),
            ),
        };

        Ok(DFlashBatchedRunOutput {
            tokens: tokens_out,
            accept_lens: accept_lens_per_row,
            stats,
        })
    }
}

/// Zero generation stats, used when [`DFlashBatchedGenerator::run_batched`]
/// short-circuits.
fn build_zero_stats() -> GenerationStats {
    GenerationStats {
        prompt_tokens: 0,
        generated_tokens: 0,
        prefill_time_ms: 0.0,
        decode_time_ms: 0.0,
        prefill_tok_per_sec: 0.0,
        decode_tok_per_sec: 0.0,
    }
}

/// Compute decode throughput in tokens/sec, guarding against
/// divide-by-zero.
fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        tokens as f64 / secs
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drafter::{DrafterKind, SharedKv};
    use crate::generate::LanguageModel;
    use crate::layers::KVCache;
    use crate::weights::WeightMap;
    use std::cell::Cell;
    use std::rc::Rc;

    // ============================================================
    // Per-row speculative walk tests
    // ============================================================

    /// All rows accept the full prefix when the drafter matches the
    /// target position-by-position.
    #[test]
    fn speculative_walk_batched_full_prefix_each_row() {
        let draft = vec![vec![10, 20, 30], vec![40, 50, 60]];
        let target = vec![vec![10, 20, 30, 99], vec![40, 50, 60, 88]];
        let budgets = vec![100, 100];
        let (accepted, new_tokens) = speculative_walk_batched(&draft, &target, &budgets);
        assert_eq!(accepted, vec![3, 3]);
        assert_eq!(new_tokens, vec![vec![10, 20, 30, 99], vec![40, 50, 60, 88]]);
    }

    /// Different rows accept different lengths in the same batch (the
    /// load-bearing per-row divergence test).
    #[test]
    fn speculative_walk_batched_per_row_divergence() {
        // Row 0: full match (accept 3, bonus 99).
        // Row 1: mismatch at position 1 (accept 1, bonus 51 — target's
        //        choice at divergence position).
        // Row 2: mismatch at position 0 (accept 0, bonus 70).
        let draft = vec![vec![10, 20, 30], vec![40, 99, 99], vec![99, 99, 99]];
        let target = vec![
            vec![10, 20, 30, 99],
            vec![40, 51, 52, 53],
            vec![70, 71, 72, 73],
        ];
        let budgets = vec![100, 100, 100];
        let (accepted, new_tokens) = speculative_walk_batched(&draft, &target, &budgets);
        assert_eq!(accepted, vec![3, 1, 0]);
        assert_eq!(
            new_tokens,
            vec![vec![10, 20, 30, 99], vec![40, 51], vec![70]]
        );
    }

    /// Per-row budgets truncate each row independently.
    #[test]
    fn speculative_walk_batched_per_row_budget_truncation() {
        let draft = vec![vec![10, 20, 30], vec![40, 50, 60]];
        let target = vec![vec![10, 20, 30, 99], vec![40, 50, 60, 88]];
        // Row 0 has budget 2 (truncate to 2 tokens), row 1 has budget 4
        // (no truncation, all 4 emit).
        let budgets = vec![2, 4];
        let (_, new_tokens) = speculative_walk_batched(&draft, &target, &budgets);
        assert_eq!(new_tokens, vec![vec![10, 20], vec![40, 50, 60, 88]]);
    }

    /// Zero-budget row emits no tokens (defensive case for finished
    /// rows that still pass through the walk).
    #[test]
    fn speculative_walk_batched_zero_budget_emits_empty() {
        let draft = vec![vec![10, 20, 30]];
        let target = vec![vec![10, 20, 30, 99]];
        let budgets = vec![0];
        let (accepted, new_tokens) = speculative_walk_batched(&draft, &target, &budgets);
        assert_eq!(accepted, vec![3]);
        assert_eq!(new_tokens, vec![Vec::<i32>::new()]);
    }

    // ============================================================
    // Synthetic target + drafter test fixtures (batched)
    //
    // Mirror the B = 1 fixtures in `round_loop.rs::tests` but with
    // per-row state plumbed through. All fixtures are deliberately
    // small (no real attention or matmul) so the tests run on every
    // developer workstation regardless of MLX availability beyond the
    // basic FFI surface mlxcel-core already links against.
    // ============================================================

    /// Synthetic target cache: a single i32 offset that counts tokens
    /// advanced through the cache. Just enough state to verify the
    /// per-row rollback hook trims by the expected amount.
    #[derive(Debug, Default)]
    struct SyntheticCache {
        offset: i32,
    }

    struct SyntheticVerifyOut {
        logits: UniquePtr<MlxArray>,
        captured_hidden: UniquePtr<MlxArray>,
        batch_size: i32,
        verify_len: i32,
    }

    type SyntheticArgmaxFn = dyn Fn(i32, i32, i32) -> i32;

    /// Synthetic target implementing [`SpeculativeTarget`] for B >= 1.
    ///
    /// `argmax_fn(row, position, prev_token) -> next_token`. Tests seed
    /// this so the accept pattern is deterministic.
    #[allow(clippy::type_complexity)]
    struct SyntheticTarget {
        argmax_fn: Box<SyntheticArgmaxFn>,
        concat_hidden_dim: i32,
        /// Recorded rollback events as `(per_row_accepted, block_size)`.
        /// Tests inspect to confirm per-row rollback was called with
        /// the right shape.
        rollback_events: Rc<Cell<Vec<(Vec<i32>, i32)>>>,
    }

    impl SyntheticTarget {
        fn new<F: Fn(i32, i32, i32) -> i32 + 'static>(
            concat_hidden_dim: i32,
            argmax_fn: F,
        ) -> Self {
            Self {
                argmax_fn: Box::new(argmax_fn),
                concat_hidden_dim,
                rollback_events: Rc::new(Cell::new(Vec::new())),
            }
        }

        fn rollback_events(&self) -> Vec<(Vec<i32>, i32)> {
            let v = self.rollback_events.take();
            self.rollback_events.set(v.clone());
            v
        }
    }

    impl SpeculativeTarget for SyntheticTarget {
        type Cache = SyntheticCache;
        type VerifyOut = SyntheticVerifyOut;

        fn capture_layer_ids(&self) -> &[usize] {
            &[]
        }

        fn verify_forward(
            &self,
            verify_input: &MlxArray,
            caches: &mut [Self::Cache],
        ) -> Self::VerifyOut {
            let shape = ffi::array_shape(verify_input);
            let b = shape[0];
            let k = shape[1];

            // Advance the synthetic caches by K positions (per-batch
            // tracking would require a wider cache struct; the SUT
            // tests pin advancement on the shared offset).
            for c in caches.iter_mut() {
                c.offset += k;
            }

            // Read back per-cell verify input tokens.
            let mut tokens: Vec<Vec<i32>> = Vec::with_capacity(b as usize);
            for bi in 0..b {
                let mut row: Vec<i32> = Vec::with_capacity(k as usize);
                for s in 0..k {
                    let cell = ffi::slice(verify_input, &[bi, s], &[bi + 1, s + 1]);
                    let scalar = ffi::reshape(&cell, &[]);
                    row.push(ffi::item_i32(&scalar));
                }
                tokens.push(row);
            }

            // Build per-row argmax (one-hot logits at the chosen channel).
            const VOCAB: usize = 1024;
            let mut buf = vec![-10.0f32; b as usize * k as usize * VOCAB];
            for bi in 0..b as usize {
                for s in 0..k as usize {
                    let id = (self.argmax_fn)(bi as i32, s as i32, tokens[bi][s]);
                    debug_assert!(
                        (0..VOCAB as i32).contains(&id),
                        "synthetic test token out of vocab: row={bi} pos={s} id={id} vocab={VOCAB}"
                    );
                    buf[(bi * k as usize + s) * VOCAB + id as usize] = 10.0;
                }
            }
            let logits = ffi::from_slice_f32(&buf, &[b, k, VOCAB as i32]);

            // Synthetic captured hidden: [B, K, concat_hidden_dim] zeros.
            let hidden = ffi::zeros(&[b, k, self.concat_hidden_dim], crate::dtype::FLOAT32);

            SyntheticVerifyOut {
                logits,
                captured_hidden: hidden,
                batch_size: b,
                verify_len: k,
            }
        }

        fn rollback_partial(
            &self,
            _caches: &mut [Self::Cache],
            _verify_out: &Self::VerifyOut,
            _accepted: i32,
            _block_size: i32,
        ) {
            panic!(
                "batched synthetic target should not call rollback_partial; \
                    use rollback_partial_batched"
            );
        }

        fn rollback_partial_batched(
            &self,
            caches: &mut [Self::Cache],
            verify_out: &Self::VerifyOut,
            accepted: &[i32],
            block_size: i32,
        ) {
            // Synthetic trim: trim every cache by `block_size - (max(accepted) + 1)`.
            let max_a = *accepted.iter().max().unwrap_or(&0);
            let trim = block_size - (max_a + 1);
            for c in caches.iter_mut() {
                c.offset -= trim;
            }
            // Sanity probe on the verify_out shape.
            debug_assert_eq!(verify_out.verify_len, block_size);
            let mut ev = self.rollback_events.take();
            ev.push((accepted.to_vec(), block_size));
            self.rollback_events.set(ev);
        }

        fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
            ffi::slice(
                &verify_out.captured_hidden,
                &[0, 0, 0],
                &[
                    verify_out.batch_size,
                    verify_out.verify_len,
                    self.concat_hidden_dim,
                ],
            )
        }

        fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
            verify_out.logits.as_ref().expect("logits must be Some")
        }
    }

    /// EmbedOnly LM shim used only for the drafter's `bind` smoke test.
    struct EmbedOnlyLm;

    impl LanguageModel for EmbedOnlyLm {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1, 1, 1], crate::dtype::FLOAT32)
        }
        fn make_caches(&self) -> Vec<KVCache> {
            Vec::new()
        }
        fn num_layers(&self) -> usize {
            0
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            Vec::new()
        }
        fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
            let shape = ffi::array_shape(input_ids);
            Some(ffi::zeros(&[shape[0], shape[1], 1], crate::dtype::FLOAT32))
        }
    }

    /// Synthetic batched drafter. Takes a propose-fn that returns the
    /// per-row proposal block given (bonus_per_row, block_size).
    type SyntheticProposeBatchedFn = dyn FnMut(&[i32], usize) -> Vec<Vec<i32>>;

    struct SyntheticBatchedDrafter {
        propose: Box<SyntheticProposeBatchedFn>,
    }

    impl SyntheticBatchedDrafter {
        fn new<F: FnMut(&[i32], usize) -> Vec<Vec<i32>> + 'static>(propose: F) -> Self {
            Self {
                propose: Box::new(propose),
            }
        }
    }

    impl Drafter for SyntheticBatchedDrafter {
        fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
            let dummy = ffi::from_slice_i32(&[0_i32], &[1, 1]);
            if target.embed_tokens(&dummy).is_none() {
                return Err(DrafterError::BindFailed {
                    reason: "embed_tokens missing on test target".to_string(),
                });
            }
            Ok(())
        }

        fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
            self.bind(target)
        }

        fn set_shared_kv(
            &mut self,
            _shared_kv: SharedKv<'_>,
            _kv_offset: usize,
            _position: usize,
            _left_padding: usize,
        ) -> Result<(), DrafterError> {
            Ok(())
        }

        fn draft_block(
            &mut self,
            _last_bonus: i32,
            _hidden: Option<&MlxArray>,
            _block_size: usize,
            _sampler: &SamplingConfig,
        ) -> Result<Vec<i32>, DrafterError> {
            // Batched-only synthetic drafter; B = 1 path is not exercised.
            Err(DrafterError::DraftFailed {
                reason: "synthetic batched drafter does not implement draft_block (B=1)"
                    .to_string(),
            })
        }

        fn draft_block_batched(
            &mut self,
            last_bonus: &[i32],
            _hidden: Option<&MlxArray>,
            block_size: usize,
            _sampler: &SamplingConfig,
        ) -> Result<Vec<Vec<i32>>, DrafterError> {
            let proposals = (self.propose)(last_bonus, block_size);
            debug_assert_eq!(proposals.len(), last_bonus.len());
            for row in &proposals {
                debug_assert_eq!(
                    row.len(),
                    block_size - 1,
                    "synthetic batched drafter row must produce block_size - 1 proposals"
                );
            }
            Ok(proposals)
        }

        fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
            Ok(())
        }

        fn kind(&self) -> DrafterKind {
            DrafterKind::Dflash
        }
    }

    fn run_synthetic_batched_round_loop(
        batch_size: usize,
        block_size: u32,
        max_new_tokens: usize,
        first_bonus: Vec<i32>,
        argmax_fn: impl Fn(i32, i32, i32) -> i32 + 'static,
        propose_fn: impl FnMut(&[i32], usize) -> Vec<Vec<i32>> + 'static,
        stop_tokens: Vec<i32>,
    ) -> (DFlashBatchedRunOutput, Vec<(Vec<i32>, i32)>) {
        let target = SyntheticTarget::new(5 * 8, argmax_fn);
        // 3 caches, one per "layer" (count is incidental for synthetic test).
        let mut caches: Vec<SyntheticCache> = (0..3).map(|_| SyntheticCache::default()).collect();

        let drafter = SyntheticBatchedDrafter::new(propose_fn);
        let lm = EmbedOnlyLm;
        let mut gen =
            DFlashBatchedGenerator::with_drafter(Box::new(drafter), SamplingConfig::greedy());
        gen.block_size = block_size;

        let first_hidden = ffi::zeros(&[batch_size as i32, 1, 5 * 8], crate::dtype::FLOAT32);

        let out = gen
            .run_batched(
                &target,
                &lm,
                &mut caches,
                &first_bonus,
                first_hidden,
                &stop_tokens,
                max_new_tokens,
            )
            .expect("synthetic batched round loop must not fail");
        let rollback_events = target.rollback_events();
        (out, rollback_events)
    }

    // ============================================================
    // Round-loop control-flow tests (B > 1)
    // ============================================================

    /// All rows fully accept every round: rollback NEVER fires.
    #[test]
    fn batched_full_accept_every_row_every_round_skips_rollback() {
        let argmax_fn = |_r: i32, _s: i32, prev: i32| prev + 1;
        let propose_fn = |bonus: &[i32], bs: usize| -> Vec<Vec<i32>> {
            bonus
                .iter()
                .map(|&b| (1..bs as i32).map(|s| b + s).collect())
                .collect()
        };

        let (out, rollback_events) = run_synthetic_batched_round_loop(
            4,
            /*block_size=*/ 8,
            /*max_new_tokens=*/ 24,
            /*first_bonus=*/ vec![100, 200, 300, 400],
            argmax_fn,
            propose_fn,
            /*stop_tokens=*/ vec![],
        );

        // Each row's per-round accept_len must be 7 (block_size - 1).
        for (r, lens) in out.accept_lens.iter().enumerate() {
            for (i, acc) in lens.iter().enumerate() {
                assert_eq!(*acc, 7, "row {r} round {i}: must accept all 7 proposals");
            }
        }
        assert!(
            rollback_events.is_empty(),
            "full-accept across all rows must never call rollback, got {rollback_events:?}"
        );
    }

    /// Per-row divergence within the same batch: one row accepts 0,
    /// one accepts K-1, the rest accept partials. Rollback fires every
    /// round and carries the per-row accept slice.
    #[test]
    fn batched_per_row_partial_accept_rollback_carries_per_row_slice() {
        let argmax_fn = |_r: i32, _s: i32, prev: i32| prev + 1;
        // 2-row batch:
        //   row 0: drafter always mismatches (accept 0 every round).
        //   row 1: drafter always matches (accept K-1 every round).
        let propose_fn = |bonus: &[i32], bs: usize| -> Vec<Vec<i32>> {
            let mut out = Vec::with_capacity(bonus.len());
            for (r, &b) in bonus.iter().enumerate() {
                if r == 0 {
                    out.push((1..bs as i32).map(|_| 999).collect());
                } else {
                    out.push((1..bs as i32).map(|s| b + s).collect());
                }
            }
            out
        };

        // Use a `max_new_tokens` large enough that row 1 (perfect
        // oracle, 8 emits/round) does not saturate during the rounds
        // we assert over. With 64 budget and 8 emits/round, row 1
        // finishes after round 7 (emitted=1 + 8*7 = 57; round 7 fills
        // budget). We only assert over the first 4 rounds — that
        // window is unambiguously full-block (bs == 8) for both rows.
        let (out, rollback_events) = run_synthetic_batched_round_loop(
            2,
            /*block_size=*/ 8,
            /*max_new_tokens=*/ 64,
            /*first_bonus=*/ vec![100, 200],
            argmax_fn,
            propose_fn,
            /*stop_tokens=*/ vec![],
        );

        // Within the first 4 rounds, both rows are still active and
        // every round runs at the full bs = 8. Row 0 always accepts 0;
        // row 1 always accepts 7 (block_size - 1).
        for i in 0..4 {
            assert_eq!(
                out.accept_lens[0][i], 0,
                "row 0 round {i}: must accept 0 (always-wrong drafter)"
            );
            assert_eq!(
                out.accept_lens[1][i], 7,
                "row 1 round {i}: must accept 7 (perfect-oracle drafter)"
            );
        }

        // Every observed rollback event in the first 4 rounds must
        // carry the per-row accept slice `[0, 7]` — the load-bearing
        // proof that rollback receives per-row data with B = 2.
        assert!(
            rollback_events.len() >= 4,
            "expected at least 4 rollback events, got {}",
            rollback_events.len()
        );
        for (i, (accepted, bs)) in rollback_events.iter().take(4).enumerate() {
            assert_eq!(
                *accepted,
                vec![0, 7],
                "round {i} rollback per-row accept slice"
            );
            assert_eq!(*bs, 8, "round {i} rollback block_size");
        }
    }

    /// Per-row early-EOS: one row hits EOS at round 1, another at round
    /// 3, others never. Rows that hit EOS freeze in place (no further
    /// emission); the round-loop keeps cranking for the rest until
    /// every row is finished.
    #[test]
    fn batched_per_row_early_eos_freezes_in_place() {
        let argmax_fn = |_r: i32, _s: i32, prev: i32| prev + 1;
        // 4-row batch: every drafter proposes the perfect chain.
        let propose_fn = |bonus: &[i32], bs: usize| -> Vec<Vec<i32>> {
            bonus
                .iter()
                .map(|&b| (1..bs as i32).map(|s| b + s).collect())
                .collect()
        };

        // first_bonus = [100, 200, 300, 400], block_size = 4.
        //   Round 0 emits per row: row 0 sees [101, 102, 103, 104];
        //                          row 1 sees [201, 202, 203, 204];
        //                          row 2 sees [301, 302, 303, 304];
        //                          row 3 sees [401, 402, 403, 404].
        // stop_tokens = [202]: row 1 stops after round 0.
        // stop_tokens = [304]: row 2 stops after round 0.
        // We need an EOS that hits row 0 LATER. Use first_bonus 100 + 6
        // rounds = 100 + 4*6 = 124. Pick stop = [109] (round 1).
        // Row 3 never matches a stop token.
        //
        // Compute expected emitted lengths:
        //   row 0 emits up to 109 (round 1, 9 tokens after first_bonus
        //                          → tokens 101..109 = 9 tokens).
        //   row 1 emits up to 202 (round 0, 2 tokens → 201, 202).
        //   row 2 emits up to 304 (round 0, 4 tokens → 301..304).
        //   row 3 emits to max_new_tokens - 1.
        let (out, _rollback_events) = run_synthetic_batched_round_loop(
            4,
            /*block_size=*/ 4,
            /*max_new_tokens=*/ 16,
            /*first_bonus=*/ vec![100, 200, 300, 400],
            argmax_fn,
            propose_fn,
            /*stop_tokens=*/ vec![109, 202, 304],
        );

        // Row 0 emits 101..109 = 9 tokens, last token = 109 (the EOS).
        assert_eq!(out.tokens[0].last(), Some(&109), "row 0 must stop on 109");
        // Row 1 emits 201, 202; stops on 202.
        assert_eq!(out.tokens[1], vec![201, 202]);
        // Row 2 emits 301..304; stops on 304.
        assert_eq!(out.tokens[2], vec![301, 302, 303, 304]);
        // Row 3 emits max_new_tokens - 1 = 15 tokens: 401..415.
        assert_eq!(out.tokens[3].len(), 15);
        assert_eq!(out.tokens[3][0], 401);
        assert_eq!(out.tokens[3][14], 415);
    }

    // ============================================================
    // Greedy-parity gate: batched output byte-identical to per-row B=1
    // ============================================================

    /// Greedy parity at temp=0: the batched run_batched output for B>1
    /// is byte-identical to running the B=1 round loop sequentially on
    /// each row's prompt, with the same target argmax law and the same
    /// drafter proposals. This is the load-bearing invariant for the
    /// batched DFlash path.
    ///
    /// We construct B=4 rows whose first_bonus tokens are distinct.
    /// The target argmax law is row-independent (`prev + 1`), so each
    /// row's reference sequence is deterministic.
    #[test]
    fn greedy_parity_against_single_row_run() {
        fn chain_next(prev: i32) -> i32 {
            (prev.rem_euclid(101) * 7 + 13).rem_euclid(200) + 30
        }
        let argmax_fn = |_r: i32, _s: i32, prev: i32| chain_next(prev);

        let first_bonus = vec![100, 150, 50, 200];
        let max_new_tokens = 33; // 1 first_bonus + 32 round-loop emissions

        // Build the per-row reference greedy sequence.
        let mut reference: Vec<Vec<i32>> = Vec::with_capacity(first_bonus.len());
        for &fb in &first_bonus {
            let mut seq = vec![fb];
            for _ in 1..max_new_tokens {
                let prev = *seq.last().unwrap();
                seq.push(chain_next(prev));
            }
            reference.push(seq);
        }

        // Drafter variant 1: always mismatches (accept 0 every round).
        let propose_always_wrong = |bonus: &[i32], bs: usize| -> Vec<Vec<i32>> {
            bonus
                .iter()
                .map(|_| (1..bs as i32).map(|_| 0).collect())
                .collect()
        };
        let (out, _) = run_synthetic_batched_round_loop(
            4,
            8,
            max_new_tokens,
            first_bonus.clone(),
            argmax_fn,
            propose_always_wrong,
            vec![],
        );

        // Compare each row's tokens (excluding the first_bonus that
        // is the caller's; the round loop emits the rest).
        for (r, row) in out.tokens.iter().enumerate() {
            let reference_tail = &reference[r][1..];
            assert_eq!(
                row.len(),
                reference_tail.len(),
                "row {r}: emitted {} tokens, reference has {}",
                row.len(),
                reference_tail.len()
            );
            for (i, (got, want)) in row.iter().zip(reference_tail.iter()).enumerate() {
                assert_eq!(
                    got, want,
                    "row {r} token {i}: got {got}, want {want} (greedy parity violation)"
                );
            }
        }

        // Drafter variant 2: oracle (always matches; full-accept every round).
        // At temp=0 every row's output MUST still be byte-identical to
        // the no-drafter baseline.
        let propose_oracle = |bonus: &[i32], bs: usize| -> Vec<Vec<i32>> {
            bonus
                .iter()
                .map(|&b| {
                    let mut row = Vec::with_capacity(bs - 1);
                    let mut prev = b;
                    for _ in 0..bs - 1 {
                        let next = chain_next(prev);
                        row.push(next);
                        prev = next;
                    }
                    row
                })
                .collect()
        };
        let (out2, _) = run_synthetic_batched_round_loop(
            4,
            8,
            max_new_tokens,
            first_bonus.clone(),
            argmax_fn,
            propose_oracle,
            vec![],
        );

        for (r, row) in out2.tokens.iter().enumerate() {
            let reference_tail = &reference[r][1..];
            for (i, (got, want)) in row.iter().zip(reference_tail.iter()).enumerate() {
                assert_eq!(
                    got, want,
                    "variant-2 row {r} token {i}: got {got}, want {want} (oracle drafter parity)"
                );
            }
        }

        // Cross-variant invariant: variant-1 and variant-2 must produce
        // identical token streams (greedy parity holds across drafters).
        assert_eq!(
            out.tokens, out2.tokens,
            "different drafters must produce byte-identical output at temp=0"
        );
    }

    /// Stress: two rows where one always full-accepts and the other
    /// always 0-accepts, exercising the per-row rollback divergence on
    /// every single round.
    #[test]
    fn batched_b2_extreme_rollback_divergence_every_round() {
        let argmax_fn = |_r: i32, _s: i32, prev: i32| prev + 1;
        let propose_fn = |bonus: &[i32], bs: usize| -> Vec<Vec<i32>> {
            let mut out = Vec::with_capacity(bonus.len());
            // Row 0: accept K-1 (perfect oracle).
            out.push((1..bs as i32).map(|s| bonus[0] + s).collect());
            // Row 1: accept 0 (always wrong).
            out.push((1..bs as i32).map(|_| 999).collect());
            out
        };
        // Use max_new_tokens large enough that row 0 (perfect oracle,
        // 8 emits/round) does not saturate during the assertion window.
        // 64 budget → row 0 finishes after round 7. We assert only over
        // the first 4 rounds where both rows are still active and
        // bs == 8 unambiguously.
        let (out, rollback_events) = run_synthetic_batched_round_loop(
            2,
            /*block_size=*/ 8,
            /*max_new_tokens=*/ 64,
            /*first_bonus=*/ vec![100, 200],
            argmax_fn,
            propose_fn,
            vec![],
        );

        // Per-round accept pattern in the all-active window: row 0 = 7,
        // row 1 = 0. This is the load-bearing per-row divergence stress
        // (one row always full-accepts; the other always rejects).
        for i in 0..4 {
            assert_eq!(
                out.accept_lens[0][i], 7,
                "row 0 round {i} accept (oracle drafter)"
            );
            assert_eq!(
                out.accept_lens[1][i], 0,
                "row 1 round {i} accept (always-wrong drafter)"
            );
        }

        // Every observed rollback event in the first 4 rounds must
        // carry `[7, 0]` (the inverse per-row accept slice from the
        // sibling test, exercising the symmetric rollback path).
        assert!(
            rollback_events.len() >= 4,
            "expected at least 4 rollback events, got {}",
            rollback_events.len()
        );
        for (i, (accepted, bs)) in rollback_events.iter().take(4).enumerate() {
            assert_eq!(*accepted, vec![7, 0], "round {i} per-row accept slice");
            assert_eq!(*bs, 8, "round {i} rollback block_size");
        }
    }
}
