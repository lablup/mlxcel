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

//! Cohort planning for batched prefill (#332).
//!
//! A collected prefill window can mix requests that the padded batched-prefill
//! path supports with requests that it does not. Historically one incompatible
//! request forced the *whole* window back to sequential prefill, wasting the
//! eligible concurrent work. This module turns that all-or-nothing decision
//! into a cohort split: the cold text requests that can share a single padded
//! forward pass are grouped into a batched cohort, while incompatible requests
//! (adopted prompt-cache prefixes, VLM / custom embeddings, length-incompatible
//! rows on equal-length-only models) are routed to the offset-aware
//! single-sequence path.
//!
//! The planner is a pure function over per-row classifications so its
//! correctness invariants can be pinned by unit tests without a real model:
//!
//! - **Order preserving.** Concatenating the members of the returned cohorts in
//!   order reproduces the input window order exactly. The window is drained in
//!   priority order (high lane, then normal, then low; FIFO within a lane), so
//!   preserving window order is exactly what keeps request priority / FIFO
//!   fairness intact across cohort boundaries.
//! - **Offset isolation.** A [`PrefillCohortKind::BatchedCold`] cohort contains
//!   only rows flagged `is_cold`, i.e. rows with a zero KV-history offset and no
//!   custom embeddings. The padded batched path assumes a zero cache offset for
//!   every row, so this guarantees it never receives an adopted-prefix row whose
//!   KV must resume at a non-zero offset.
//! - **Contiguity.** Cold rows batch only where they are *contiguous* in the
//!   window. A cold row separated from other cold rows by an incompatible row is
//!   prefilled sequentially rather than reordered into a distant batch, so the
//!   split never hoists a lower-priority request ahead of a higher-priority one.

/// How a contiguous group of prefill-window rows is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillCohortKind {
    /// Two or more cold text rows (zero KV-history offset, no custom
    /// embeddings) that share a single padded batched-prefill forward pass.
    BatchedCold,
    /// Rows prefilled one at a time on the offset-aware single-sequence path:
    /// adopted prompt-cache prefixes, VLM / custom-embedding rows, isolated
    /// cold rows, and (on equal-length-only models) length-incompatible cold
    /// rows.
    Sequential,
}

/// A planned cohort: its execution kind and the window indices of its member
/// rows, in priority (window) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PrefillCohort {
    pub kind: PrefillCohortKind,
    pub members: Vec<usize>,
}

/// Per-row inputs to [`plan_prefill_cohorts`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillRow {
    /// Eligible for the padded batched path: zero KV-history offset AND no
    /// custom / VLM embeddings. Adopted-prefix and embedding-bearing rows are
    /// not cold.
    pub is_cold: bool,
    /// Prompt length in tokens. Only consulted for equal-length-only models
    /// (`can_pad == false`), where a batched cohort must share one length.
    pub prompt_len: usize,
}

/// Partition a priority-ordered prefill window into execution cohorts.
///
/// `can_batch` is the model-level `supports_batched_prefill()` capability and
/// `can_pad` is `supports_padded_prefill()`. When `can_batch` is false the
/// model cannot batch any prefill, so the whole window is one sequential cohort.
///
/// See the module documentation for the order-preservation, offset-isolation,
/// and contiguity guarantees.
pub(crate) fn plan_prefill_cohorts(
    rows: &[PrefillRow],
    can_batch: bool,
    can_pad: bool,
) -> Vec<PrefillCohort> {
    let n = rows.len();
    if n == 0 {
        return Vec::new();
    }

    // Model cannot batch prefill at all: every row takes the single-sequence
    // path, in window order. This reproduces the pre-cohort fallback.
    if !can_batch {
        return vec![PrefillCohort {
            kind: PrefillCohortKind::Sequential,
            members: (0..n).collect(),
        }];
    }

    let mut cohorts: Vec<PrefillCohort> = Vec::new();
    // Incompatible (or isolated-cold) rows seen since the last flush, kept in
    // window order so the eventual Sequential cohort preserves priority.
    let mut pending_seq: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if !rows[i].is_cold {
            pending_seq.push(i);
            i += 1;
            continue;
        }

        // Maximal contiguous run of cold rows starting at `i`.
        let run_start = i;
        while i < n && rows[i].is_cold {
            i += 1;
        }
        let run: Vec<usize> = (run_start..i).collect();

        // A run batches only when it has at least two rows AND the model can
        // pad to a common length (or every row already shares one length, the
        // only case an equal-length-only model can batch). Otherwise every row
        // of the run falls through to the sequential path, in order.
        let run_batches = run.len() >= 2 && (can_pad || all_equal_len(rows, &run));
        if run_batches {
            flush_sequential(&mut cohorts, &mut pending_seq);
            cohorts.push(PrefillCohort {
                kind: PrefillCohortKind::BatchedCold,
                members: run,
            });
        } else {
            pending_seq.extend(run);
        }
    }
    flush_sequential(&mut cohorts, &mut pending_seq);
    cohorts
}

/// Whether every indexed row shares the same prompt length.
fn all_equal_len(rows: &[PrefillRow], idx: &[usize]) -> bool {
    match idx.first() {
        None => true,
        Some(&first) => {
            let len0 = rows[first].prompt_len;
            idx.iter().all(|&j| rows[j].prompt_len == len0)
        }
    }
}

/// Emit the accumulated sequential rows (if any) as one ordered cohort.
fn flush_sequential(cohorts: &mut Vec<PrefillCohort>, pending: &mut Vec<usize>) {
    if pending.is_empty() {
        return;
    }
    cohorts.push(PrefillCohort {
        kind: PrefillCohortKind::Sequential,
        members: std::mem::take(pending),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a window from `(is_cold, prompt_len)` tuples.
    fn rows(spec: &[(bool, usize)]) -> Vec<PrefillRow> {
        spec.iter()
            .map(|&(is_cold, prompt_len)| PrefillRow {
                is_cold,
                prompt_len,
            })
            .collect()
    }

    /// Flatten the cohort plan back into the dispatch order of window indices.
    fn dispatch_order(cohorts: &[PrefillCohort]) -> Vec<usize> {
        cohorts
            .iter()
            .flat_map(|c| c.members.iter().copied())
            .collect()
    }

    /// Every plan must dispatch each input row exactly once, in the original
    /// window (priority) order. This is the fairness invariant: no row is
    /// dropped, duplicated, or hoisted past a row that preceded it.
    fn assert_order_preserved(rows: &[PrefillRow], cohorts: &[PrefillCohort]) {
        let order = dispatch_order(cohorts);
        let expected: Vec<usize> = (0..rows.len()).collect();
        assert_eq!(
            order, expected,
            "cohort dispatch order must equal window order (priority preserved)"
        );
    }

    /// A BatchedCold cohort must never contain a non-cold row. This is the
    /// cache-offset-correctness guarantee: the padded batched path only ever
    /// sees zero-offset rows, so an adopted prefix can never have its KV
    /// resumed at the wrong position by being folded into a batch.
    fn assert_batched_cohorts_are_cold(rows: &[PrefillRow], cohorts: &[PrefillCohort]) {
        for cohort in cohorts {
            if cohort.kind == PrefillCohortKind::BatchedCold {
                assert!(
                    cohort.members.len() >= 2,
                    "a BatchedCold cohort must hold at least two rows"
                );
                for &idx in &cohort.members {
                    assert!(
                        rows[idx].is_cold,
                        "row {idx} in a BatchedCold cohort must be cold (zero offset, no embeddings)"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_window_yields_no_cohorts() {
        let plan = plan_prefill_cohorts(&[], true, true);
        assert!(plan.is_empty());
    }

    #[test]
    fn all_cold_window_forms_one_batched_cohort() {
        // No-regression: a window with no incompatible request batches exactly
        // as before (every row in one padded batched pass).
        let w = rows(&[(true, 10), (true, 20), (true, 30)]);
        let plan = plan_prefill_cohorts(&w, true, true);
        assert_eq!(
            plan,
            vec![PrefillCohort {
                kind: PrefillCohortKind::BatchedCold,
                members: vec![0, 1, 2],
            }]
        );
        assert_order_preserved(&w, &plan);
        assert_batched_cohorts_are_cold(&w, &plan);
    }

    #[test]
    fn all_incompatible_window_is_one_sequential_cohort() {
        // Every row adopted/VLM: each is handled on the single-sequence path,
        // in order. No batched cohort exists, but no row is lost.
        let w = rows(&[(false, 10), (false, 20), (false, 30)]);
        let plan = plan_prefill_cohorts(&w, true, true);
        assert_eq!(
            plan,
            vec![PrefillCohort {
                kind: PrefillCohortKind::Sequential,
                members: vec![0, 1, 2],
            }]
        );
        assert_order_preserved(&w, &plan);
        assert_batched_cohorts_are_cold(&w, &plan);
    }

    #[test]
    fn mixed_cold_and_adopted_splits_into_cohorts() {
        // The headline case: a window of cold rows followed by adopted-prefix
        // rows splits into a batched cold cohort and a sequential cohort. The
        // cold cohort runs batched even though the window contains incompatible
        // requests, and the batched cohort holds only cold rows.
        let w = rows(&[(true, 10), (true, 12), (false, 40), (false, 41)]);
        let plan = plan_prefill_cohorts(&w, true, true);
        assert_eq!(
            plan,
            vec![
                PrefillCohort {
                    kind: PrefillCohortKind::BatchedCold,
                    members: vec![0, 1],
                },
                PrefillCohort {
                    kind: PrefillCohortKind::Sequential,
                    members: vec![2, 3],
                },
            ]
        );
        assert_order_preserved(&w, &plan);
        assert_batched_cohorts_are_cold(&w, &plan);
    }

    #[test]
    fn adopted_prefix_before_cold_keeps_priority_order() {
        // A high-priority adopted row at the head must be dispatched before the
        // lower-priority cold rows that follow, so the sequential cohort comes
        // first in dispatch order.
        let w = rows(&[(false, 50), (true, 10), (true, 11)]);
        let plan = plan_prefill_cohorts(&w, true, true);
        assert_eq!(
            plan,
            vec![
                PrefillCohort {
                    kind: PrefillCohortKind::Sequential,
                    members: vec![0],
                },
                PrefillCohort {
                    kind: PrefillCohortKind::BatchedCold,
                    members: vec![1, 2],
                },
            ]
        );
        assert_order_preserved(&w, &plan);
        assert_batched_cohorts_are_cold(&w, &plan);
    }

    #[test]
    fn isolated_cold_row_between_incompatible_stays_sequential() {
        // A lone cold row surrounded by incompatible rows cannot batch without
        // reordering, so it joins the sequential cohort and order is preserved.
        let w = rows(&[(false, 50), (true, 10), (false, 60)]);
        let plan = plan_prefill_cohorts(&w, true, true);
        assert_eq!(
            plan,
            vec![PrefillCohort {
                kind: PrefillCohortKind::Sequential,
                members: vec![0, 1, 2],
            }]
        );
        assert_order_preserved(&w, &plan);
        assert_batched_cohorts_are_cold(&w, &plan);
    }

    #[test]
    fn two_separated_cold_runs_form_two_batches() {
        // Cold runs split by an incompatible row form independent batches; the
        // incompatible row is dispatched between them, preserving order.
        let w = rows(&[(true, 10), (true, 11), (false, 40), (true, 12), (true, 13)]);
        let plan = plan_prefill_cohorts(&w, true, true);
        assert_eq!(
            plan,
            vec![
                PrefillCohort {
                    kind: PrefillCohortKind::BatchedCold,
                    members: vec![0, 1],
                },
                PrefillCohort {
                    kind: PrefillCohortKind::Sequential,
                    members: vec![2],
                },
                PrefillCohort {
                    kind: PrefillCohortKind::BatchedCold,
                    members: vec![3, 4],
                },
            ]
        );
        assert_order_preserved(&w, &plan);
        assert_batched_cohorts_are_cold(&w, &plan);
    }

    #[test]
    fn single_cold_row_is_sequential_not_batched() {
        // A cold cohort of one has no batching benefit and takes the single
        // path (matching the pre-cohort single-request fast path).
        let w = rows(&[(true, 10)]);
        let plan = plan_prefill_cohorts(&w, true, true);
        assert_eq!(
            plan,
            vec![PrefillCohort {
                kind: PrefillCohortKind::Sequential,
                members: vec![0],
            }]
        );
        assert_order_preserved(&w, &plan);
    }

    #[test]
    fn model_without_batched_prefill_is_all_sequential() {
        // Even an all-cold window is fully sequential when the model does not
        // support batched prefill.
        let w = rows(&[(true, 10), (true, 20), (true, 30)]);
        let plan = plan_prefill_cohorts(&w, false, true);
        assert_eq!(
            plan,
            vec![PrefillCohort {
                kind: PrefillCohortKind::Sequential,
                members: vec![0, 1, 2],
            }]
        );
        assert_order_preserved(&w, &plan);
    }

    #[test]
    fn equal_length_only_model_batches_only_equal_lengths() {
        // can_pad == false (equal-length-only model). A run of equal lengths
        // batches; a run with mixed lengths falls back to sequential.
        let equal = rows(&[(true, 16), (true, 16), (true, 16)]);
        let plan_equal = plan_prefill_cohorts(&equal, true, false);
        assert_eq!(
            plan_equal,
            vec![PrefillCohort {
                kind: PrefillCohortKind::BatchedCold,
                members: vec![0, 1, 2],
            }]
        );
        assert_batched_cohorts_are_cold(&equal, &plan_equal);

        let mixed = rows(&[(true, 16), (true, 24)]);
        let plan_mixed = plan_prefill_cohorts(&mixed, true, false);
        assert_eq!(
            plan_mixed,
            vec![PrefillCohort {
                kind: PrefillCohortKind::Sequential,
                members: vec![0, 1],
            }]
        );
    }

    #[test]
    fn padded_model_batches_mixed_lengths() {
        // can_pad == true: mixed lengths still batch (the padded path pads to
        // the longest prompt). This is the qwen3 / llama3 path.
        let w = rows(&[(true, 16), (true, 24)]);
        let plan = plan_prefill_cohorts(&w, true, true);
        assert_eq!(
            plan,
            vec![PrefillCohort {
                kind: PrefillCohortKind::BatchedCold,
                members: vec![0, 1],
            }]
        );
        assert_batched_cohorts_are_cold(&w, &plan);
    }
}
