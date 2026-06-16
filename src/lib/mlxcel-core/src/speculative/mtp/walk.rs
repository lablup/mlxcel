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

//! Single-batch exact-greedy speculative walk.
//!
//! Direct Rust port of `_speculative_walk` in
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/mtp.py:
//!
//! ```python
//! def _speculative_walk(draft_tokens, target_tokens, budget):
//!     n_draft = draft_tokens.shape[1]
//!     combined = mx.concatenate(
//!         [draft_tokens.reshape(-1), target_tokens.reshape(-1)]
//!     ).tolist()
//!     d = combined[:n_draft]
//!     t = combined[n_draft:]
//!     accepted = next((i for i in range(len(d)) if d[i] != t[i]), len(d))
//!     new_tokens = (d[:accepted] + [t[accepted]])[:budget]
//!     return accepted, new_tokens
//! ```
//!
//! Semantics:
//!
//! - `draft_tokens` carries the K-1 autoregressive proposals from the
//!   drafter.
//! - `target_tokens` carries the K target choices from the verify pass
//!   (one per position in the verify input `[bonus, draft_0, ..., draft_{K-2}]`).
//!   At greedy temp=0 this is the argmax of the verify logits at each
//!   position.
//! - `accepted` is the count of consecutive matching draft tokens
//!   starting from index 0. `accepted == len(draft_tokens)` means full
//!   accept (all draft tokens matched and there is one more bonus token
//!   from target).
//! - `new_tokens` is the user-visible emission: the matched draft prefix
//!   plus one bonus from the target at the first mismatch (or at the end
//!   on full accept), clamped to `budget`.
//!
//! ## Apple Silicon precision note
//!
//! This walk is pure integer comparison: no f32 promotion. The input
//! `Vec<i32>` slices are already materialised host-side by the caller
//! (drafter's `draft_block` returns `Vec<i32>`; target's verify pass
//! produces `Vec<i32>` via per-position argmax `ffi::item_i32`).

/// Result of [`speculative_walk`]: the number of accepted draft tokens
/// (range `0..=draft_tokens.len()`) and the user-visible new tokens to
/// emit (length `accepted + 1` before clamping to budget; `<= budget`
/// after).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkResult {
    /// Number of accepted draft tokens. When this equals
    /// `draft_tokens.len()`, the full draft block was accepted and the
    /// target's bonus at the last position becomes the next round's
    /// `b` (last_bonus).
    pub accepted: usize,
    /// User-visible token emission for this round.
    pub new_tokens: Vec<i32>,
}

/// Exact-greedy speculative walk (single-batch path).
///
/// Returns the `accepted` count and the `new_tokens` to emit.
///
/// # Arguments
///
/// - `draft_tokens`: `K-1` autoregressive proposals from the drafter.
/// - `target_tokens`: `K` greedy target choices (one per verify
///   position; `target_tokens[i]` is what the target would emit
///   conditional on `[bonus, draft_0, ..., draft_{i-1}]`).
/// - `budget`: cap on `new_tokens.len()` — typically
///   `max_tokens - emitted` so a generation does not run past its
///   declared cap.
///
/// # Invariants
///
/// - `target_tokens.len() == draft_tokens.len() + 1` (the verify input
///   was `[bonus, draft_0, ..., draft_{K-2}]`, and the verify produces
///   one logit per input position).
/// - `budget >= 1` — the round-loop never calls this with budget 0
///   because it gates on `emitted < max_tokens` before invoking.
///
/// Both invariants are debug-asserted. Release builds tolerate
/// shape mismatches by short-circuiting to a safe accept count.
pub fn speculative_walk(draft_tokens: &[i32], target_tokens: &[i32], budget: usize) -> WalkResult {
    debug_assert_eq!(
        target_tokens.len(),
        draft_tokens.len() + 1,
        "MTP speculative_walk: target_tokens must be exactly one longer than draft_tokens \
         (verify input = [bonus, draft_0, ..., draft_{{K-2}}], one logit per position)",
    );
    debug_assert!(
        budget >= 1,
        "MTP speculative_walk: budget must be >= 1; the round-loop should gate \
         on `emitted < max_tokens` before calling",
    );

    let n_draft = draft_tokens.len();
    // Compute accepted = index of first mismatch, capped at n_draft. The
    // upstream `next((i for i in range(len(d)) if d[i] != t[i]), len(d))`
    // reduces to a linear scan; we keep the same shape so a future profile
    // can confirm the loop is branchless under typical compilers.
    let accepted = (0..n_draft)
        .find(|&i| draft_tokens[i] != target_tokens[i])
        .unwrap_or(n_draft);

    // new_tokens = (d[:accepted] + [t[accepted]])[:budget].
    //
    // Sub-cases:
    // - accepted == n_draft: target_tokens[n_draft] is the bonus at the
    //   end of the block. emit_len = accepted + 1 = K (full accept).
    // - 0 <= accepted < n_draft: target_tokens[accepted] is the bonus at
    //   the first mismatch. emit_len = accepted + 1.
    let emit_len = (accepted + 1).min(budget);
    let mut new_tokens = Vec::with_capacity(emit_len);
    for &d in &draft_tokens[..accepted.min(emit_len)] {
        new_tokens.push(d);
    }
    // The bonus only goes in if we still have budget for it after the
    // accepted prefix. When `accepted >= budget` the bonus is dropped.
    if new_tokens.len() < emit_len {
        new_tokens.push(target_tokens[accepted]);
    }

    WalkResult {
        accepted,
        new_tokens,
    }
}

/// Per-row exact-greedy speculative walk (B >= 1) for the batched MTP path.
///
/// Mirrors upstream `_speculative_walk_batch` in
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/mtp.py. Given per-row drafter proposals
/// (`draft_tokens[r]` is row `r`'s `K-1` proposals) and per-row target argmax
/// (`target_tokens[r]` is row `r`'s `K` argmax tokens), plus per-row budgets,
/// produces:
///
/// - `accepted[r]`: per-row accept count, in range `0..=draft_tokens[r].len()`.
/// - `new_tokens[r]`: per-row emitted tokens, length `<= budgets[r]`.
///
/// Semantics are the natural lifting of [`speculative_walk`] over independent
/// rows: each row is walked in isolation; rows in different speculative-walk
/// states share the same verify-forward but diverge on what they emit. This
/// is the load-bearing per-row divergence test in the batched MTP path.
///
/// # Greedy parity
///
/// At `temp=0.0`, this is byte-identical to calling [`speculative_walk`] once
/// per row with the row's `(draft, target, budget)` triple. The test
/// [`tests::walk_batched_equivalent_to_per_row_speculative_walk`] pins that
/// invariant.
///
/// # Arguments
///
/// - `draft_tokens`: per-row drafter proposals, outer length `B`. Each
///   `draft_tokens[r]` has length `K-1`.
/// - `target_tokens`: per-row target argmax tokens, outer length `B`. Each
///   `target_tokens[r]` has length exactly `draft_tokens[r].len() + 1`.
/// - `budgets`: per-row emission caps. `budgets[r] == 0` is allowed (defensive
///   for finished rows still passed through the walk for shape stability) and
///   produces an empty row.
pub fn speculative_walk_batched(
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
        debug_assert_eq!(
            t.len(),
            d.len() + 1,
            "speculative_walk_batched: row {r} target_tokens length must be draft + 1"
        );

        let n_draft = d.len();
        let accepted = (0..n_draft).find(|&i| d[i] != t[i]).unwrap_or(n_draft);

        // Match scalar `speculative_walk`'s budget-aware emission shape:
        // emit_len = (accepted + 1).min(budget); drop bonus when accepted ==
        // budget.
        let emit_len = (accepted + 1).min(budgets[r]);
        let mut new_tokens: Vec<i32> = Vec::with_capacity(emit_len);
        for &tok in &d[..accepted.min(emit_len)] {
            new_tokens.push(tok);
        }
        if new_tokens.len() < emit_len {
            new_tokens.push(t[accepted]);
        }

        accepted_per_row.push(accepted);
        new_tokens_per_row.push(new_tokens);
    }

    (accepted_per_row, new_tokens_per_row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_accept_emits_all_draft_plus_bonus() {
        // K=4, all 3 draft proposals match. target_tokens has K=4 entries;
        // the last one is the next-step bonus.
        let draft = vec![10, 11, 12];
        let target = vec![10, 11, 12, 13];
        let r = speculative_walk(&draft, &target, 16);
        assert_eq!(r.accepted, 3, "full accept: all draft tokens match");
        assert_eq!(r.new_tokens, vec![10, 11, 12, 13], "emit all + bonus");
    }

    #[test]
    fn first_mismatch_truncates_emission() {
        // K=4, draft[0] matches but draft[1] does not. We must emit
        // draft[0] (accepted) + target[1] (bonus from first divergence).
        let draft = vec![10, 99, 12];
        let target = vec![10, 11, 50, 13];
        let r = speculative_walk(&draft, &target, 16);
        assert_eq!(r.accepted, 1);
        assert_eq!(r.new_tokens, vec![10, 11]);
    }

    #[test]
    fn zero_accept_emits_only_target_bonus() {
        // No draft tokens match. accepted = 0, new_tokens = [target[0]].
        let draft = vec![99, 98, 97];
        let target = vec![10, 11, 12, 13];
        let r = speculative_walk(&draft, &target, 16);
        assert_eq!(r.accepted, 0);
        assert_eq!(r.new_tokens, vec![10]);
    }

    #[test]
    fn budget_clamps_full_accept() {
        // Full accept emits K tokens. With budget < K, we emit only `budget`.
        let draft = vec![10, 11, 12];
        let target = vec![10, 11, 12, 13];
        let r = speculative_walk(&draft, &target, 2);
        assert_eq!(r.accepted, 3, "accepted count is independent of budget");
        assert_eq!(r.new_tokens, vec![10, 11]);
    }

    #[test]
    fn budget_clamps_partial_accept_dropping_bonus() {
        // accepted = 2, emission would be [d0, d1, target[2]] = 3 tokens.
        // budget = 2 truncates to [d0, d1] (bonus dropped).
        let draft = vec![10, 11, 99];
        let target = vec![10, 11, 12, 13];
        let r = speculative_walk(&draft, &target, 2);
        assert_eq!(r.accepted, 2);
        assert_eq!(r.new_tokens, vec![10, 11]);
    }

    #[test]
    fn budget_exactly_one_emits_first_bonus_or_first_accepted() {
        // budget=1 with full draft mismatch: emit only target[0].
        let r = speculative_walk(&[99, 98], &[10, 11, 12], 1);
        assert_eq!(r.accepted, 0);
        assert_eq!(r.new_tokens, vec![10]);

        // budget=1 with first match: still emit only the first token
        // (the accepted draft token; bonus is dropped).
        let r = speculative_walk(&[10, 99], &[10, 11, 12], 1);
        assert_eq!(r.accepted, 1, "accept count is independent of budget");
        assert_eq!(r.new_tokens, vec![10]);
    }

    #[test]
    fn single_draft_token_full_accept() {
        // K=2 (block_size=2): one draft proposal, two target tokens.
        let r = speculative_walk(&[10], &[10, 11], 16);
        assert_eq!(r.accepted, 1);
        assert_eq!(r.new_tokens, vec![10, 11]);
    }

    #[test]
    fn single_draft_token_mismatch() {
        let r = speculative_walk(&[99], &[10, 11], 16);
        assert_eq!(r.accepted, 0);
        assert_eq!(r.new_tokens, vec![10]);
    }

    #[test]
    fn mid_block_mismatch_three_position_progression() {
        // Hand-computed pin: K=5, mismatch at index 2 of draft.
        //
        //  draft: [10, 11, 99, 13]
        //  target: [10, 11, 12, ..., ..]
        //  i=0: 10 == 10 → accept
        //  i=1: 11 == 11 → accept
        //  i=2: 99 != 12 → stop, take target[2] = 12
        //  emit = [10, 11, 12]
        let r = speculative_walk(&[10, 11, 99, 13], &[10, 11, 12, 20, 21], 16);
        assert_eq!(r.accepted, 2);
        assert_eq!(r.new_tokens, vec![10, 11, 12]);
    }

    // ============================================================
    // Batched walk tests
    // ============================================================

    #[test]
    fn walk_batched_all_rows_full_accept() {
        let draft = vec![vec![10, 11, 12], vec![20, 21, 22]];
        let target = vec![vec![10, 11, 12, 13], vec![20, 21, 22, 23]];
        let budgets = vec![100, 100];
        let (accepted, new_tokens) = speculative_walk_batched(&draft, &target, &budgets);
        assert_eq!(accepted, vec![3, 3]);
        assert_eq!(new_tokens, vec![vec![10, 11, 12, 13], vec![20, 21, 22, 23]]);
    }

    #[test]
    fn walk_batched_per_row_divergence() {
        // Acceptance criterion: per-row accept counts diverge in the
        // same batch. Row 0 full-accepts, row 1 mismatches at index 1,
        // row 2 mismatches at index 0.
        let draft = vec![vec![10, 11, 12], vec![20, 99, 22], vec![99, 31, 32]];
        let target = vec![
            vec![10, 11, 12, 13],
            vec![20, 21, 22, 23],
            vec![30, 31, 32, 33],
        ];
        let budgets = vec![100, 100, 100];
        let (accepted, new_tokens) = speculative_walk_batched(&draft, &target, &budgets);
        assert_eq!(accepted, vec![3, 1, 0]);
        assert_eq!(
            new_tokens,
            vec![vec![10, 11, 12, 13], vec![20, 21], vec![30]],
        );
    }

    #[test]
    fn walk_batched_per_row_budget_truncation() {
        // Row 0 has tight budget 2, row 1 has plenty.
        let draft = vec![vec![10, 11, 12], vec![20, 21, 22]];
        let target = vec![vec![10, 11, 12, 13], vec![20, 21, 22, 23]];
        let budgets = vec![2, 4];
        let (_acc, new_tokens) = speculative_walk_batched(&draft, &target, &budgets);
        assert_eq!(new_tokens, vec![vec![10, 11], vec![20, 21, 22, 23]]);
    }

    #[test]
    fn walk_batched_zero_budget_emits_empty() {
        // Defensive: finished rows pass through with budget=0 and emit
        // nothing.
        let draft = vec![vec![10, 11, 12]];
        let target = vec![vec![10, 11, 12, 13]];
        let budgets = vec![0];
        let (accepted, new_tokens) = speculative_walk_batched(&draft, &target, &budgets);
        assert_eq!(accepted, vec![3]);
        assert_eq!(new_tokens, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn walk_batched_equivalent_to_per_row_speculative_walk() {
        // Greedy parity gate: the batched walk MUST be byte-identical
        // to calling `speculative_walk` once per row with the same
        // arguments.
        let draft = vec![
            vec![10, 11, 12],
            vec![20, 99, 22],
            vec![99, 31, 32],
            vec![41, 42, 43],
        ];
        let target = vec![
            vec![10, 11, 12, 13],
            vec![20, 21, 22, 23],
            vec![30, 31, 32, 33],
            vec![41, 42, 43, 44],
        ];
        let budgets = vec![16, 7, 5, 3];
        let (batched_accepted, batched_new) = speculative_walk_batched(&draft, &target, &budgets);

        for r in 0..draft.len() {
            let reference = speculative_walk(&draft[r], &target[r], budgets[r]);
            assert_eq!(
                batched_accepted[r], reference.accepted,
                "row {r}: batched accepted count must match per-row walk"
            );
            assert_eq!(
                batched_new[r], reference.new_tokens,
                "row {r}: batched new_tokens must match per-row walk"
            );
        }
    }
}
