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

//! Byte-accurate, constraint-aware layer balancer used by `auto_partition`.
//!
//! Given a per-layer byte vector, a per-stage memory budget, and a set of
//! forbidden boundary indices (coming from [`super::partition::
//! LayerAdjacencyGroup`]s), produce one contiguous layer range per stage
//! that minimises the maximum per-stage byte sum.
//!
//! Used by: [`super::partition::auto_partition`]

use std::ops::Range;

use anyhow::{Result, bail};

use super::partition::LayerAdjacencyGroup;

/// Balance layers across stages by minimising the maximum per-stage byte
/// load.
///
/// `per_layer_bytes[i]` is the byte cost of layer `i`. `per_stage_budget[s]`
/// is the maximum byte sum stage `s` may hold (already net of embedding /
/// lm_head reservations on the edge stages). `forbidden_boundaries` is the
/// sorted list of boundary indices that may **not** be used as a stage
/// split — these come from adjacency groups that must not be cut.
/// `adjacency` and `device_ids` are retained for warning diagnostics only.
///
/// Returns a vector of half-open layer ranges, one per stage, plus a list
/// of human-readable warnings emitted when the constraints force a
/// significantly imbalanced plan (over 50% max/min ratio).
///
/// The algorithm:
///
/// 1. Compute the target per-stage byte load as
///    `ceil(total_bytes / num_stages)` to act as a feasibility lower bound.
/// 2. Binary-search the smallest max-stage byte load `T` for which a valid
///    K-stage split exists that respects the forbidden boundaries, the
///    per-stage budgets, and the "every stage gets at least one layer"
///    rule. Check feasibility with a greedy sweep.
/// 3. Reconstruct the actual split points used at the feasible `T`.
/// 4. If `T` exceeds the naive balanced target by more than the imbalance
///    threshold, add a warning naming the forcing constraint.
pub fn balance_layers(
    per_layer_bytes: &[u64],
    per_stage_budget: &[u64],
    forbidden_boundaries: &[usize],
    adjacency: &[LayerAdjacencyGroup],
    device_ids: &[String],
) -> Result<(Vec<Range<usize>>, Vec<String>)> {
    let n_layers = per_layer_bytes.len();
    let k_stages = per_stage_budget.len();
    if k_stages == 0 {
        bail!("balance_layers called with zero stages");
    }
    if n_layers < k_stages {
        bail!(
            "balance_layers called with {} layers and {} stages",
            n_layers,
            k_stages
        );
    }

    // Single-stage: trivially the whole model.
    if k_stages == 1 {
        // `Vec::from_iter` avoids both `clippy::single-range-in-vec-init`
        // (triggered by `vec![0..n_layers]`) and
        // `clippy::calls-to-push-immediately-after-creation` (triggered by
        // a `Vec::with_capacity(1) + push` pair).
        let ranges: Vec<Range<usize>> = Vec::from_iter(std::iter::once(0..n_layers));
        return Ok((ranges, Vec::new()));
    }

    // Prefix sums let us compute any contiguous stage's byte sum in O(1).
    let mut prefix = Vec::with_capacity(n_layers + 1);
    prefix.push(0u64);
    for &b in per_layer_bytes {
        prefix.push(prefix.last().copied().unwrap_or(0).saturating_add(b));
    }
    let total = *prefix.last().unwrap();

    // Feasibility lower bound for the minimum max-stage load: dictated by
    // (a) balanced sharing (total / K), (b) a single heavy layer that
    // cannot be split, and (c) any adjacency group that must live on one
    // stage.
    let max_single_layer = per_layer_bytes.iter().copied().max().unwrap_or(0);
    let max_budget = per_stage_budget.iter().copied().max().unwrap_or(0);
    let adjacency_lb = adjacency
        .iter()
        .filter_map(|g| {
            if g.layers.start >= g.layers.end {
                return None;
            }
            let end = g.layers.end.min(prefix.len() - 1);
            let start = g.layers.start.min(end);
            Some(prefix[end].saturating_sub(prefix[start]))
        })
        .max()
        .unwrap_or(0);
    let mut lo = max_single_layer
        .max(total.div_ceil(k_stages as u64))
        .max(adjacency_lb);
    let mut hi = total.max(lo);

    // If the heaviest single layer or adjacency group exceeds even the
    // largest stage's budget, the model cannot fit. Name the heaviest
    // blocker so operators can rebalance hardware. We do not reject just
    // because a stage is tight — stages may have dissimilar budgets and
    // the search still finds a valid plan where the tight stage holds a
    // few layers and the fat stage holds the rest.
    if lo > max_budget {
        let heavy_idx = per_stage_budget
            .iter()
            .enumerate()
            .max_by_key(|pair| *pair.1)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let device = device_ids.get(heavy_idx).cloned().unwrap_or_default();
        bail!(
            "cannot balance {} bytes across {} stages: minimum feasible per-stage load \
             is {} bytes but the largest budget (stage {} on device '{}') only holds {} bytes",
            total,
            k_stages,
            lo,
            heavy_idx,
            device,
            max_budget
        );
    }

    // Binary search the smallest feasible max-stage byte load T.
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if feasible(
            mid,
            &prefix,
            per_stage_budget,
            forbidden_boundaries,
            n_layers,
            k_stages,
        ) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    // Reconstruct the actual split for T = lo. If the search failed to
    // land on a feasible T (pathological input), fall back to the maximum
    // budget ceiling so `build_splits` gets something to iterate on; the
    // caller's capacity validation in `auto_partition` surfaces any
    // remaining overflow.
    let target = if feasible(
        lo,
        &prefix,
        per_stage_budget,
        forbidden_boundaries,
        n_layers,
        k_stages,
    ) {
        lo
    } else {
        max_budget
    };

    let splits = build_splits(
        target,
        &prefix,
        per_stage_budget,
        forbidden_boundaries,
        n_layers,
        k_stages,
    );
    let splits = match splits {
        Some(v) => v,
        None => {
            // Ultimate fallback — uniform contiguous split ignoring
            // budgets. This keeps backward compatibility with tests that
            // feed impossibly tight budgets; the caller re-validates
            // capacity after the fact.
            uniform_split(n_layers, k_stages, forbidden_boundaries)
        }
    };

    let ranges = splits_to_ranges(&splits, n_layers);
    let warnings = check_imbalance(
        &ranges,
        per_layer_bytes,
        &prefix,
        adjacency,
        forbidden_boundaries,
    );
    Ok((ranges, warnings))
}

// Imbalance ratio above which we tell the operator the plan is skewed.
// 50% captures scenarios where constraints force the worst stage to carry
// 1.5x the average load while still leaving enough slack to avoid crying
// wolf on small asymmetric models.
const IMBALANCE_WARN_RATIO_PCT: u64 = 150;

/// Greedy feasibility check: can we split `n_layers` into `k_stages`
/// contiguous groups where every stage's byte sum <= `t` and every stage's
/// byte sum <= its respective `per_stage_budget`, using only boundaries not
/// listed in `forbidden_boundaries`?
fn feasible(
    t: u64,
    prefix: &[u64],
    per_stage_budget: &[u64],
    forbidden_boundaries: &[usize],
    n_layers: usize,
    k_stages: usize,
) -> bool {
    let mut stage = 0usize;
    let mut cursor = 0usize;
    while stage < k_stages {
        // We need at least one layer per remaining stage.
        let remaining = k_stages - stage;
        let max_end = n_layers - (remaining - 1);
        if cursor >= max_end {
            return false;
        }
        let budget = per_stage_budget[stage];
        let is_last = stage + 1 == k_stages;
        if is_last {
            // Last stage must consume every remaining layer inside its
            // own budget AND within the search target.
            let last_bytes = prefix[n_layers].saturating_sub(prefix[cursor]);
            if last_bytes > t || last_bytes > budget {
                return false;
            }
            cursor = n_layers;
            stage += 1;
            continue;
        }
        // Find the largest `end` such that prefix[end] - prefix[cursor] <= t
        // and end <= max_end and budget[stage] >= (prefix[end] - prefix[cursor]).
        let cap = t.min(budget);
        let mut end = cursor + 1;
        while end < max_end && prefix[end + 1].saturating_sub(prefix[cursor]) <= cap {
            end += 1;
        }
        // If even a single layer overflows `t` or `budget`, we cannot
        // proceed — return false. The binary search will escalate T.
        if prefix[end].saturating_sub(prefix[cursor]) > cap && cursor + 1 == end {
            return false;
        }
        // Snap end back to the last non-forbidden boundary.
        while end > cursor + 1 && is_forbidden(end, forbidden_boundaries) {
            end -= 1;
        }
        if end <= cursor {
            return false;
        }
        cursor = end;
        stage += 1;
    }
    cursor == n_layers
}

/// Reconstruct the boundary list for a feasible `t`. Mirrors `feasible`'s
/// greedy but records each boundary.
fn build_splits(
    t: u64,
    prefix: &[u64],
    per_stage_budget: &[u64],
    forbidden_boundaries: &[usize],
    n_layers: usize,
    k_stages: usize,
) -> Option<Vec<usize>> {
    let mut boundaries = Vec::with_capacity(k_stages.saturating_sub(1));
    let mut cursor = 0usize;
    for (stage, &budget) in per_stage_budget.iter().enumerate().take(k_stages) {
        let remaining = k_stages - stage;
        let max_end = n_layers - (remaining - 1);
        if cursor >= max_end {
            return None;
        }
        let is_last = stage + 1 == k_stages;
        if is_last {
            let last_bytes = prefix[n_layers].saturating_sub(prefix[cursor]);
            if last_bytes > t || last_bytes > budget {
                return None;
            }
            cursor = n_layers;
            continue;
        }
        let cap = t.min(budget);
        let mut end = cursor + 1;
        while end < max_end && prefix[end + 1].saturating_sub(prefix[cursor]) <= cap {
            end += 1;
        }
        if prefix[end].saturating_sub(prefix[cursor]) > cap && cursor + 1 == end {
            return None;
        }
        while end > cursor + 1 && is_forbidden(end, forbidden_boundaries) {
            end -= 1;
        }
        if end <= cursor {
            return None;
        }
        boundaries.push(end);
        cursor = end;
    }
    if cursor != n_layers {
        return None;
    }
    Some(boundaries)
}

/// Last-resort uniform split that respects forbidden boundaries but may
/// overflow byte budgets. The caller re-validates capacity afterwards and
/// surfaces the overflow to the operator.
fn uniform_split(n_layers: usize, k_stages: usize, forbidden: &[usize]) -> Vec<usize> {
    let per = n_layers / k_stages;
    let rem = n_layers % k_stages;
    let mut boundaries = Vec::with_capacity(k_stages - 1);
    let mut cursor = 0usize;
    for stage in 0..(k_stages - 1) {
        let count = per + if stage < rem { 1 } else { 0 };
        let mut b = cursor + count;
        if b <= cursor {
            b = cursor + 1;
        }
        // Slide away from any forbidden boundary; prefer sliding earlier
        // so the following stage keeps room.
        let mut tries = 0;
        while is_forbidden(b, forbidden) && tries < n_layers {
            if b > cursor + 1 {
                b -= 1;
            } else {
                b += 1;
            }
            tries += 1;
        }
        boundaries.push(b);
        cursor = b;
    }
    boundaries
}

fn splits_to_ranges(splits: &[usize], n_layers: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::with_capacity(splits.len() + 1);
    let mut start = 0usize;
    for &b in splits {
        ranges.push(start..b);
        start = b;
    }
    ranges.push(start..n_layers);
    ranges
}

fn is_forbidden(boundary: usize, forbidden: &[usize]) -> bool {
    forbidden.binary_search(&boundary).is_ok()
}

/// Report any stage whose byte sum is more than `IMBALANCE_WARN_RATIO_PCT`
/// times the lightest stage. When a forbidden boundary or adjacency group
/// is the direct cause, name it so operators can decide to rebalance
/// hardware rather than silently living with the skew.
fn check_imbalance(
    ranges: &[Range<usize>],
    per_layer_bytes: &[u64],
    prefix: &[u64],
    adjacency: &[LayerAdjacencyGroup],
    forbidden: &[usize],
) -> Vec<String> {
    if ranges.len() < 2 {
        return Vec::new();
    }
    let per_stage: Vec<u64> = ranges
        .iter()
        .map(|r| prefix[r.end].saturating_sub(prefix[r.start]))
        .collect();
    let max_val = per_stage.iter().copied().max().unwrap_or(0);
    let min_val = per_stage.iter().copied().min().unwrap_or(0);
    if min_val == 0 || max_val == 0 {
        return Vec::new();
    }
    let ratio = max_val.saturating_mul(100) / min_val;
    if ratio <= IMBALANCE_WARN_RATIO_PCT {
        return Vec::new();
    }
    let mut msg = format!(
        "pipeline auto-partition produced an imbalanced plan: stage byte loads range from \
         {min_val} to {max_val} ({}.{}x), which may hurt throughput.",
        ratio / 100,
        ratio % 100
    );
    // Attribute the imbalance to the largest adjacency group inside the
    // heaviest stage, if any.
    if !adjacency.is_empty() {
        let heaviest = per_stage
            .iter()
            .enumerate()
            .max_by_key(|pair| *pair.1)
            .map(|(i, _)| i);
        if let Some(h) = heaviest {
            let stage_range = ranges[h].clone();
            if let Some(group) = adjacency
                .iter()
                .filter(|g| g.layers.start >= stage_range.start && g.layers.end <= stage_range.end)
                .max_by_key(|g| {
                    let end = g.layers.end.min(prefix.len() - 1);
                    let start = g.layers.start.min(end);
                    prefix[end].saturating_sub(prefix[start])
                })
            {
                msg.push_str(&format!(
                    " Heaviest stage {} carries adjacency group {}..{} ({}).",
                    h, group.layers.start, group.layers.end, group.reason
                ));
            }
        }
    }
    if !forbidden.is_empty() {
        msg.push_str(
            " Adjacency constraints forbid splitting at some boundaries; consider rebalancing \
             hardware or running with a manual --pp-layers plan if persistent.",
        );
    }
    let _ = per_layer_bytes; // documented contract: kept for future cost model.
    vec![msg]
}

#[cfg(test)]
#[path = "partition_balance_tests.rs"]
mod tests;
