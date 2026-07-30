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

//! Profiling harness for the kernel autotuner (issue #906).
//!
//! Times every candidate a [`TunableOp`] offers for the current shape bucket
//! and picks the min-latency one. Three properties matter more than raw speed
//! here, because a tuner that picks a different winner on every run is worse
//! than no tuner at all:
//!
//! 1. **Median of N, not mean of N.** GPU timings on a shared host are
//!    right-skewed: an occasional scheduling or thermal outlier inflates a
//!    mean while leaving a median untouched. `N >= 5` by default.
//! 2. **Real synchronization.** Each timed repetition submits the work
//!    ([`TunableOp::run`], which evals) and then blocks on the backend
//!    ([`TunableOp::sync`]). Without the sync the harness would time graph
//!    construction and rank candidates by CPU dispatch cost.
//! 3. **The default only loses by a margin.** A candidate must beat the op's
//!    own default by more than [`ProfileConfig::min_improvement`] to be
//!    selected. Within that band the default wins, so a tuning run on a noisy
//!    host converges back to today's behavior rather than to a coin flip.
//!
//! Candidates that fail to launch are dropped from the sweep, not fatal: a
//! launch configuration can be infeasible on a given device, and the right
//! answer is to choose among those that did run.

use std::time::Instant;

use super::tactic::{Tactic, TunableOp};

/// Default timed repetitions per candidate. The issue's determinism guard
/// requires median-of-N with `N >= 5`.
pub const DEFAULT_REPS: usize = 5;

/// Default untimed warmup repetitions per candidate. Each distinct tactic is
/// usually a distinct JIT specialization, so the first launch pays compilation;
/// two warmups keep that out of the measurement.
pub const DEFAULT_WARMUP: usize = 2;

/// Default relative margin a candidate must beat the default by. 2% is below
/// the run-to-run spread of every kernel benchmark recorded in
/// `docs/benchmark_results/`, so it filters noise without hiding real wins.
pub const DEFAULT_MIN_IMPROVEMENT: f64 = 0.02;

/// Knobs for one profiling sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileConfig {
    /// Untimed repetitions before measuring.
    pub warmup: usize,
    /// Timed repetitions; the median is the candidate's score.
    pub reps: usize,
    /// Relative improvement over the default required to switch away from it,
    /// e.g. `0.02` for 2%.
    pub min_improvement: f64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            warmup: DEFAULT_WARMUP,
            reps: DEFAULT_REPS,
            min_improvement: DEFAULT_MIN_IMPROVEMENT,
        }
    }
}

impl ProfileConfig {
    /// Clamp to at least one timed repetition and a non-negative margin, so a
    /// caller-supplied config can never produce an empty sample set or a
    /// nonsense threshold.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            warmup: self.warmup,
            reps: self.reps.max(1),
            min_improvement: if self.min_improvement.is_finite() && self.min_improvement >= 0.0 {
                self.min_improvement
            } else {
                DEFAULT_MIN_IMPROVEMENT
            },
        }
    }
}

/// One candidate's measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub tactic: Tactic,
    /// Median of `samples_us`, microseconds.
    pub median_us: f64,
    /// Per-repetition latencies, microseconds, in measurement order.
    pub samples_us: Vec<f64>,
}

/// Outcome of a sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileResult {
    /// The selected tactic.
    pub best: Tactic,
    /// Median latency of the selection.
    pub best_us: f64,
    /// Median latency of the op's default tactic, when it was among the
    /// candidates that ran.
    pub default_us: Option<f64>,
    /// Whether the selection differs from the op's default.
    pub changed: bool,
    /// Every candidate that ran, in candidate order.
    pub measurements: Vec<Measurement>,
    /// Timed repetitions behind each median.
    pub reps: usize,
}

impl ProfileResult {
    /// Relative speedup of the selection over the default, when both were
    /// measured. `> 1.0` means the tuned choice is faster.
    #[must_use]
    pub fn speedup_over_default(&self) -> Option<f64> {
        let default_us = self.default_us?;
        if self.best_us <= 0.0 {
            return None;
        }
        Some(default_us / self.best_us)
    }
}

/// Median of `samples`, or `None` when empty.
///
/// Sorts with [`f64::total_cmp`] so a NaN sample cannot produce an
/// inconsistent ordering, and averages the two central values for even counts.
#[must_use]
pub fn median(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    }
}

/// Time one candidate: `warmup` untimed repetitions, then `reps` timed ones.
///
/// Returns `None` when any repetition fails, which drops the candidate from
/// the sweep. Each repetition is bracketed individually (submit, then sync) so
/// the samples are per-invocation latencies rather than a single amortized
/// total; that is what makes the median meaningful.
fn measure(op: &dyn TunableOp, tactic: &Tactic, cfg: &ProfileConfig) -> Option<Measurement> {
    for _ in 0..cfg.warmup {
        if let Err(e) = op.run(tactic) {
            tracing::debug!(
                "autotune: dropping candidate {tactic} for {} during warmup: {e}",
                op.op_name()
            );
            return None;
        }
    }
    op.sync();

    let mut samples_us = Vec::with_capacity(cfg.reps);
    for _ in 0..cfg.reps {
        let start = Instant::now();
        if let Err(e) = op.run(tactic) {
            tracing::debug!(
                "autotune: dropping candidate {tactic} for {}: {e}",
                op.op_name()
            );
            return None;
        }
        op.sync();
        samples_us.push(start.elapsed().as_nanos() as f64 / 1000.0);
    }

    let median_us = median(&samples_us)?;
    Some(Measurement {
        tactic: tactic.clone(),
        median_us,
        samples_us,
    })
}

/// Profile every candidate the op offers for its current bucket and select the
/// min-latency tactic.
///
/// Returns `None` when the op offers no candidates, or when every candidate
/// failed to run. Callers treat `None` as "use the default".
#[must_use]
pub fn profile(op: &dyn TunableOp, cfg: ProfileConfig) -> Option<ProfileResult> {
    let cfg = cfg.sanitized();
    let bucket = op.bucket();
    let candidates = op.candidates(&bucket);
    if candidates.is_empty() {
        return None;
    }
    let default = op.default_tactic(&bucket);

    let measurements: Vec<Measurement> = candidates
        .iter()
        .filter_map(|tactic| measure(op, tactic, &cfg))
        .collect();
    if measurements.is_empty() {
        return None;
    }

    let default_us = measurements
        .iter()
        .find(|m| m.tactic == default)
        .map(|m| m.median_us);

    // Min-latency selection. Ties break toward the earlier candidate, and the
    // default (below) overrides anything inside the noise band, so the choice
    // is a deterministic function of the measured medians.
    let mut best = &measurements[0];
    for m in &measurements[1..] {
        if m.median_us < best.median_us {
            best = m;
        }
    }

    let (best_tactic, best_us) = match default_us {
        // The default ran: only switch when the win clears the noise margin.
        Some(default_us)
            if best.median_us >= default_us * (1.0 - cfg.min_improvement) || default_us <= 0.0 =>
        {
            (default.clone(), default_us)
        }
        _ => (best.tactic.clone(), best.median_us),
    };

    let changed = best_tactic != default;
    if changed {
        tracing::info!(
            "autotune: {} bucket {bucket} selected {best_tactic} at {best_us:.1}us (default {} at {}us, {} candidates, median-of-{})",
            op.op_name(),
            default,
            default_us.map_or_else(|| "n/a".to_string(), |v| format!("{v:.1}")),
            measurements.len(),
            cfg.reps,
        );
    } else {
        tracing::debug!(
            "autotune: {} bucket {bucket} kept default {default} at {best_us:.1}us ({} candidates, median-of-{})",
            op.op_name(),
            measurements.len(),
            cfg.reps,
        );
    }

    Some(ProfileResult {
        best: best_tactic,
        best_us,
        default_us,
        changed,
        measurements,
        reps: cfg.reps,
    })
}
