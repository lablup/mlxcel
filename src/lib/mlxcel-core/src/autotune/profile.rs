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
//! and picks the fastest one that can be *shown* to be faster. A tuner that
//! picks a different winner on every run is worse than no tuner at all, so the
//! measurement methodology, not the selection rule, is where the work is.
//!
//! ## Why a flat rep count was not enough
//!
//! The first cut of this harness used a fixed `warmup=2, reps=5` and a fixed 2%
//! switch threshold. On an idle M1 Ultra that made repeated `mlxcel tune` runs
//! disagree in a third of the cells: the *same* default tactic re-measured
//! 844.2us then 512.7us at batch 8 / ctx 1024, a 65% swing, while every ctx
//! 16384 cell re-measured within 0.6% to 3.7%. Two warmup iterations of a
//! ~500us launch is under a millisecond of work, which is nowhere near enough
//! to bring an Apple GPU to a steady clock, and a 2% threshold is meaningless
//! against a 20%-wide sample distribution.
//!
//! ## The four properties that make a sweep reproducible
//!
//! 1. **Warm up by wall clock, not by iteration count.** Warmup runs until
//!    [`ProfileConfig::warmup_budget_us`] has elapsed (with
//!    [`ProfileConfig::warmup`] as a floor), so a cheap launch gets thousands
//!    of iterations and an expensive one gets a handful. Both end up on a
//!    ramped clock; a fixed count only does that for expensive launches.
//! 2. **Scale repetitions by measured cost.** The warmup doubles as a cost
//!    probe, and each candidate then targets
//!    [`ProfileConfig::sample_budget_us`] of timed work, clamped to
//!    `[reps, max_reps]`. Cheap cells are the noisy ones *and* the cheap ones
//!    to sample harder, so this puts the effort exactly where the instability
//!    is. `reps` stays the documented floor (the issue's determinism guard
//!    wants median-of-N with `N >= 5`).
//! 3. **Interleave candidates instead of measuring them back to back.**
//!    Measuring candidate A to completion and then candidate B charges A for
//!    whatever thermal or clock state the machine was in at the start of the
//!    sweep, which is a systematic bias in candidate order, not noise. Samples
//!    are taken round-robin on a stride so every candidate's repetitions are
//!    spread evenly across the same wall-clock window and slow drift moves all
//!    of them together.
//! 4. **Require the win to clear the observed spread.** Each candidate carries
//!    a dispersion statistic over its own repetitions (scaled median absolute
//!    deviation, see [`Measurement::spread_us`]). A candidate must beat the
//!    default by more than [`ProfileConfig::min_improvement`] *and* by more
//!    than the two measurements' combined relative spread. Below that the
//!    default wins, which is both the honest answer (the difference was not
//!    measurable) and the safe one (the default is what ships today).
//!    Candidates that are indistinguishable from the leader collapse to the one
//!    nearest the default, so a coin flip between two equally good tactics
//!    resolves the same way on every run.
//!
//! Real synchronization underpins all of it: each repetition submits the work
//! ([`TunableOp::run`], which evals) and then blocks on the backend
//! ([`TunableOp::sync`]). Without the sync the harness would time graph
//! construction and rank candidates by CPU dispatch cost.
//!
//! Candidates that fail to launch are dropped from the sweep, not fatal: a
//! launch configuration can be infeasible on a given device, and the right
//! answer is to choose among those that did run.

use std::time::Instant;

use super::tactic::{Tactic, TunableOp};

/// Minimum timed repetitions per candidate. The issue's determinism guard
/// requires median-of-N with `N >= 5`; the adaptive budget only ever raises it.
pub const DEFAULT_REPS: usize = 5;

/// Ceiling on adaptively-grown repetitions. Bounds the sweep when a launch is
/// so cheap that the time budget would ask for tens of thousands of samples.
pub const DEFAULT_MAX_REPS: usize = 1000;

/// Target wall-clock time spent on the timed repetitions of one candidate.
///
/// Sized from the measured failure mode rather than picked round: at 100ms the
/// long-context cells of the paged-decode matrix got only 50-70 repetitions,
/// and their medians still moved 4-6% between runs, which was enough to walk a
/// cell across its own guard. 250ms roughly triples the sample count where the
/// signal actually lives while keeping a 12-cell x 6-candidate sweep well under
/// a minute.
pub const DEFAULT_SAMPLE_BUDGET_US: f64 = 250_000.0;

/// Minimum untimed warmup repetitions per candidate. Each distinct tactic is
/// usually a distinct JIT specialization, so the first launch pays compilation.
pub const DEFAULT_WARMUP: usize = 3;

/// Target wall-clock time spent warming one candidate. Long enough to bring the
/// GPU to a steady clock before anything is recorded, which a two-iteration
/// warmup of a sub-millisecond kernel does not do.
pub const DEFAULT_WARMUP_BUDGET_US: f64 = 40_000.0;

/// Default relative margin a candidate must beat the default by, on top of the
/// measured spread. Acts as a floor for cells whose samples are so tight that
/// the dispersion term rounds to nothing.
pub const DEFAULT_MIN_IMPROVEMENT: f64 = 0.02;

/// Consistency constant turning a median absolute deviation into a standard
/// deviation for normally distributed samples. Timing distributions are
/// right-skewed rather than normal, so this is a scale convention that makes
/// [`Measurement::spread_us`] readable as "roughly one sigma", not a
/// distributional claim.
pub const MAD_TO_SIGMA: f64 = 1.4826;

/// Knobs for one profiling sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileConfig {
    /// Minimum untimed repetitions before measuring.
    pub warmup: usize,
    /// Wall-clock budget for the warmup phase of one candidate, microseconds.
    /// Warmup keeps going past `warmup` until this elapses.
    pub warmup_budget_us: f64,
    /// Minimum timed repetitions; the median is the candidate's score.
    pub reps: usize,
    /// Ceiling on adaptively-grown timed repetitions.
    pub max_reps: usize,
    /// Wall-clock budget for the timed repetitions of one candidate,
    /// microseconds. The realized rep count is
    /// `clamp(sample_budget_us / per_iteration_us, reps, max_reps)`.
    pub sample_budget_us: f64,
    /// Relative improvement over the default required to switch away from it,
    /// e.g. `0.02` for 2%. The effective threshold is the larger of this and
    /// the two measurements' combined relative spread.
    pub min_improvement: f64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            warmup: DEFAULT_WARMUP,
            warmup_budget_us: DEFAULT_WARMUP_BUDGET_US,
            reps: DEFAULT_REPS,
            max_reps: DEFAULT_MAX_REPS,
            sample_budget_us: DEFAULT_SAMPLE_BUDGET_US,
            min_improvement: DEFAULT_MIN_IMPROVEMENT,
        }
    }
}

/// Replace a non-finite or negative budget with `fallback`.
fn sane_budget(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

impl ProfileConfig {
    /// Clamp to at least one timed repetition, a rep ceiling no lower than the
    /// floor, and finite non-negative budgets and margins, so a caller-supplied
    /// config can never produce an empty sample set or a nonsense threshold.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let reps = self.reps.max(1);
        Self {
            warmup: self.warmup,
            warmup_budget_us: sane_budget(self.warmup_budget_us, DEFAULT_WARMUP_BUDGET_US),
            reps,
            max_reps: self.max_reps.max(reps),
            sample_budget_us: sane_budget(self.sample_budget_us, DEFAULT_SAMPLE_BUDGET_US),
            min_improvement: sane_budget(self.min_improvement, DEFAULT_MIN_IMPROVEMENT),
        }
    }
}

/// One candidate's measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub tactic: Tactic,
    /// Median of `samples_us`, microseconds.
    pub median_us: f64,
    /// Scaled median absolute deviation of `samples_us`, microseconds.
    ///
    /// The dispersion statistic behind the flaky-tactic guard. A median
    /// absolute deviation rather than a standard deviation because the same
    /// scheduling outliers the median exists to reject would otherwise
    /// reappear in the spread and make every cell look untrustworthy.
    pub spread_us: f64,
    /// Per-repetition latencies, microseconds, in measurement order.
    pub samples_us: Vec<f64>,
}

impl Measurement {
    /// Build a measurement from raw samples, or `None` when there are none.
    #[must_use]
    pub fn from_samples(tactic: Tactic, samples_us: Vec<f64>) -> Option<Self> {
        let median_us = median(&samples_us)?;
        let spread_us = MAD_TO_SIGMA * median_absolute_deviation(&samples_us, median_us)?;
        Some(Self {
            tactic,
            median_us,
            spread_us,
            samples_us,
        })
    }

    /// Timed repetitions behind [`Self::median_us`].
    #[must_use]
    pub fn reps(&self) -> usize {
        self.samples_us.len()
    }

    /// [`Self::spread_us`] as a fraction of the median, e.g. `0.07` for a
    /// sample distribution 7% wide. `0.0` when the spread is unmeasurable,
    /// which makes the guard fall back to `min_improvement` alone.
    #[must_use]
    pub fn relative_spread(&self) -> f64 {
        if self.median_us > 0.0 && self.spread_us.is_finite() && self.spread_us > 0.0 {
            self.spread_us / self.median_us
        } else {
            0.0
        }
    }
}

/// Which measurement a sweep selected, and the threshold it applied.
///
/// Returned by [`select`], which is a pure function of the measurements so the
/// flaky-tactic guard is unit-testable without a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    /// Index into the measurement slice.
    pub index: usize,
    /// Index of the op's default tactic, when it was among the candidates that
    /// ran.
    pub default_index: Option<usize>,
}

/// Outcome of a sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileResult {
    /// The selected tactic.
    pub best: Tactic,
    /// Median latency of the selection.
    pub best_us: f64,
    /// Relative spread of the selection's samples.
    pub best_spread: f64,
    /// Median latency of the op's default tactic, when it was among the
    /// candidates that ran.
    pub default_us: Option<f64>,
    /// Relative spread of the default's samples, when the default ran.
    pub default_spread: Option<f64>,
    /// Relative improvement over the default the selection had to clear. The
    /// larger of `min_improvement` and the combined relative spread; a large
    /// value means the cell was measured on a noisy host and its row should be
    /// read with suspicion.
    pub required_improvement: f64,
    /// Whether the selection differs from the op's default.
    pub changed: bool,
    /// Every candidate that ran, in candidate order.
    pub measurements: Vec<Measurement>,
    /// Timed repetitions behind the selection's median.
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

/// Median absolute deviation of `samples` about `center`, or `None` when empty.
#[must_use]
pub fn median_absolute_deviation(samples: &[f64], center: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let deviations: Vec<f64> = samples.iter().map(|s| (s - center).abs()).collect();
    median(&deviations)
}

fn elapsed_us(start: Instant) -> f64 {
    start.elapsed().as_nanos() as f64 / 1000.0
}

/// Warm one candidate and probe its per-iteration cost.
///
/// Runs at least `cfg.warmup` repetitions, then keeps going until
/// `cfg.warmup_budget_us` has elapsed (bounded by `cfg.max_reps`). Returns the
/// estimated per-iteration cost in microseconds, taken as the median of the
/// most recent half of the warmup samples so that JIT compilation and the
/// low-clock ramp at the start do not inflate it. `Ok(None)` means no warmup
/// was configured and there is nothing to estimate from; `Err(())` means the
/// candidate is infeasible and must be dropped.
fn warm_up(op: &dyn TunableOp, tactic: &Tactic, cfg: &ProfileConfig) -> Result<Option<f64>, ()> {
    let mut samples_us: Vec<f64> = Vec::new();
    let phase_start = Instant::now();
    while samples_us.len() < cfg.warmup
        || (elapsed_us(phase_start) < cfg.warmup_budget_us && samples_us.len() < cfg.max_reps)
    {
        let start = Instant::now();
        if let Err(e) = op.run(tactic) {
            tracing::debug!(
                "autotune: dropping candidate {tactic} for {} during warmup: {e}",
                op.op_name()
            );
            return Err(());
        }
        op.sync();
        samples_us.push(elapsed_us(start));
    }
    if samples_us.is_empty() {
        return Ok(None);
    }
    let settled = &samples_us[samples_us.len() / 2..];
    Ok(median(settled))
}

/// Timed repetitions to spend on a candidate whose per-iteration cost is
/// `est_us`, clamped to `[cfg.reps, cfg.max_reps]`.
///
/// Without an estimate (no warmup configured) the floor is used, which keeps a
/// zero-budget config exactly as predictable as the pre-adaptive harness.
fn target_reps(est_us: Option<f64>, cfg: &ProfileConfig) -> usize {
    let Some(est_us) = est_us else {
        return cfg.reps;
    };
    if est_us.is_nan() || est_us <= 0.0 {
        // Too fast to time: sampling more is nearly free, so take the ceiling.
        return cfg.max_reps;
    }
    let wanted = cfg.sample_budget_us / est_us;
    if !wanted.is_finite() {
        return cfg.max_reps;
    }
    (wanted.ceil().max(0.0) as usize).clamp(cfg.reps, cfg.max_reps)
}

/// Whether candidate `slot` takes a sample in round `round`.
///
/// Spreads `target` samples evenly across `rounds` rounds so that every
/// candidate's repetitions cover the same wall-clock window. Without this a
/// candidate with a small target would finish early and a candidate with a
/// large one would spend its tail alone on an otherwise idle GPU, which is a
/// systematic difference between them rather than noise.
pub(super) fn samples_this_round(round: usize, target: usize, rounds: usize) -> bool {
    if rounds == 0 || target == 0 {
        return false;
    }
    (round + 1) * target / rounds > round * target / rounds
}

/// Select among completed measurements.
///
/// Two rules, both aimed at making repeated sweeps agree:
///
/// 1. **Collapse statistical ties.** Every candidate whose median is within the
///    combined relative spread of the fastest one is treated as equally good,
///    and the tie resolves to whichever of them sits nearest the default in
///    candidate order. Two tactics that genuinely cannot be told apart then
///    resolve the same way on every run instead of alternating.
/// 2. **Beat the default by more than the noise.** Switching away from the
///    default requires clearing both `cfg.min_improvement` and the default's
///    and the candidate's combined relative spread. Anything less is not a
///    measurable win, and the default is what ships today.
///
/// Returns the selection and the relative improvement threshold applied.
/// Panics only on an empty slice, which callers exclude.
#[must_use]
pub fn select(
    measurements: &[Measurement],
    default: &Tactic,
    cfg: &ProfileConfig,
) -> (Selection, f64) {
    assert!(!measurements.is_empty(), "select needs a measurement");
    let default_index = measurements.iter().position(|m| m.tactic == *default);

    let mut leader = 0usize;
    for (i, m) in measurements.iter().enumerate().skip(1) {
        if m.median_us < measurements[leader].median_us {
            leader = i;
        }
    }
    let lead_us = measurements[leader].median_us;
    let lead_spread = measurements[leader].relative_spread();

    // Deterministic ordering over the statistical tie: nearest the default in
    // candidate order first, then the earlier candidate.
    let rank = |i: usize| -> (usize, usize) {
        match default_index {
            Some(d) => (i.abs_diff(d), i),
            None => (0, i),
        }
    };
    let mut index = leader;
    for (i, m) in measurements.iter().enumerate() {
        if i == leader {
            continue;
        }
        let band_us = lead_us * (1.0 + lead_spread + m.relative_spread());
        if m.median_us <= band_us && rank(i) < rank(index) {
            index = i;
        }
    }

    // The candidate the medians favour once ties have collapsed. When that is
    // the default itself the leader stands in, so the reported threshold always
    // answers "how big a win did this cell demand" rather than degenerating to
    // the floor on the rows that were decided by noise.
    let challenger = if Some(index) == default_index {
        leader
    } else {
        index
    };
    let mut required = cfg.min_improvement;
    if let Some(d) = default_index {
        let default_m = &measurements[d];
        required = cfg
            .min_improvement
            .max(default_m.relative_spread() + measurements[challenger].relative_spread());
        if d != index {
            let picked = &measurements[index];
            let improvement = if default_m.median_us > 0.0 {
                (default_m.median_us - picked.median_us) / default_m.median_us
            } else {
                f64::NAN
            };
            // Phrased as a positive test that a NaN improvement fails, so an
            // unusable default median keeps the default rather than switching.
            let clears_the_guard = improvement.is_finite() && improvement > required;
            if !clears_the_guard {
                index = d;
            }
        }
    }

    (
        Selection {
            index,
            default_index,
        },
        required,
    )
}

/// Profile every candidate the op offers for its current bucket and select the
/// fastest tactic that is measurably faster than the default.
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

    // Phase 1: warm every candidate and size its sample budget. Infeasible
    // candidates drop out here, before they can perturb the timed phase.
    let mut targets: Vec<(usize, usize)> = Vec::with_capacity(candidates.len());
    for (i, tactic) in candidates.iter().enumerate() {
        if let Ok(est_us) = warm_up(op, tactic, &cfg) {
            targets.push((i, target_reps(est_us, &cfg)));
        }
    }
    if targets.is_empty() {
        return None;
    }

    // Phase 2: interleaved sampling. Every candidate's repetitions are spread
    // across the same window, so drift over the sweep is common-mode.
    let rounds = targets.iter().map(|(_, t)| *t).max().unwrap_or(0);
    let mut samples: Vec<Vec<f64>> = targets
        .iter()
        .map(|(_, t)| Vec::with_capacity(*t))
        .collect();
    let mut dropped = vec![false; targets.len()];
    for round in 0..rounds {
        for (slot, &(cand, target)) in targets.iter().enumerate() {
            if dropped[slot] || !samples_this_round(round, target, rounds) {
                continue;
            }
            let tactic = &candidates[cand];
            let start = Instant::now();
            if let Err(e) = op.run(tactic) {
                tracing::debug!(
                    "autotune: dropping candidate {tactic} for {}: {e}",
                    op.op_name()
                );
                dropped[slot] = true;
                continue;
            }
            op.sync();
            samples[slot].push(elapsed_us(start));
        }
    }

    // A candidate that failed part way through has an unfair repetition count,
    // so it is dropped outright rather than compared on fewer samples.
    let measurements: Vec<Measurement> = targets
        .iter()
        .zip(samples)
        .zip(&dropped)
        .filter(|(_, dropped)| !**dropped)
        .filter_map(|((&(cand, _), s), _)| Measurement::from_samples(candidates[cand].clone(), s))
        .collect();
    if measurements.is_empty() {
        return None;
    }

    let (selection, required_improvement) = select(&measurements, &default, &cfg);
    let picked = &measurements[selection.index];
    let best = picked.tactic.clone();
    let best_us = picked.median_us;
    let best_spread = picked.relative_spread();
    let reps = picked.reps();
    let default_m = selection.default_index.map(|d| &measurements[d]);
    let default_us = default_m.map(|m| m.median_us);
    let default_spread = default_m.map(Measurement::relative_spread);

    let changed = best != default;
    if changed {
        tracing::info!(
            "autotune: {} bucket {bucket} selected {best} at {best_us:.1}us +/-{:.1}% over default {default} at {}us (required {:.1}%, {} candidates, median-of-{reps})",
            op.op_name(),
            best_spread * 100.0,
            default_us.map_or_else(|| "n/a".to_string(), |v| format!("{v:.1}")),
            required_improvement * 100.0,
            measurements.len(),
        );
    } else {
        tracing::debug!(
            "autotune: {} bucket {bucket} kept default {default} at {best_us:.1}us +/-{:.1}% (nothing cleared {:.1}%, {} candidates, median-of-{reps})",
            op.op_name(),
            best_spread * 100.0,
            required_improvement * 100.0,
            measurements.len(),
        );
    }

    Some(ProfileResult {
        best,
        best_us,
        best_spread,
        default_us,
        default_spread,
        required_improvement,
        changed,
        measurements,
        reps,
    })
}
