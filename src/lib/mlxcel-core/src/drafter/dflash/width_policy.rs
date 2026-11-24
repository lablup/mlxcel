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

//! Acceptance-aware verify width for the B = 1 DFlash round loop (issue
//! #3410).
//!
//! A fixed verify width pays for rows the drafter rarely gets right. On
//! Inkling 2 Small an 8-row verify round costs about 120 ms against a 51 ms
//! decode step, because every extra row routes to its own experts; prose
//! emits 2.3 tokens per round there, so the drafter ran at 0.90x classic
//! decode while code (4.1 tokens per round) ran at 1.54x.
//!
//! [`ExpectedThroughputWidth`] picks each round's width `w` in `1..=ceiling`
//! that maximizes `E(w) / C(w)`:
//!
//! - `E(w)` is the expected number of tokens a `w`-row round emits:
//!   `1 + sum_{j < w} prod_{i <= j} p_i`, where `p_i` is the decayed
//!   estimate that draft `i` is accepted given drafts `1..i` were. A
//!   position with little evidence leans on the estimate of the position
//!   before it, so a width not tried lately is neither ruled out nor
//!   oversold.
//! - `C(w)` is the measured round time at `w` (an exponential average), or
//!   a linear interpolation between the nearest measured widths.
//!
//! Width 1 is a classic decode step: no draft, a one-row verify. The policy
//! falls back to it when no draft width pays for itself; the round loop then
//! runs [`ExpectedThroughputWidth::classic_run_len`] steps back to back,
//! pipelined like classic decode (`super::classic_steps`), and the policy
//! re-tries drafting after [`PROBE_AFTER_CLASSIC`] classic steps so a run
//! that turns repetitive (code after prose) gets the drafter back.
//!
//! It runs only for a target that declares
//! [`super::round_loop::SpeculativeTarget::exact_at_every_verify_width`]: a
//! block of any width up to the requested one, a single row included,
//! must reproduce that many decode steps bit for bit, because the
//! exactness probe ran at the requested width only.

/// Kill switch: `0` (or `false`/`off`/`no`) keeps the requested width every
/// round, as before issue #3410. Read once per run.
pub const ADAPTIVE_WIDTH_ENV: &str = "MLXCEL_DFLASH_ADAPTIVE_WIDTH";

/// Pure decision behind [`adaptive_width_enabled`]: unset or any other value
/// is on.
pub(crate) fn adaptive_width_enabled_from(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

/// Whether [`ADAPTIVE_WIDTH_ENV`] leaves the policy on.
pub(crate) fn adaptive_width_enabled() -> bool {
    adaptive_width_enabled_from(std::env::var(ADAPTIVE_WIDTH_ENV).ok().as_deref())
}

/// Per drafted round, every acceptance count is multiplied by this, so the
/// estimate follows the last ten or so rounds (prose and code differ by 2x
/// in acceptance inside one answer).
const DECAY: f64 = 0.9;
/// Prior acceptance of the first draft position.
const PRIOR_FIRST: f64 = 0.5;
/// Pseudo-count of the prior; deeper positions shrink toward the estimate of
/// the position before them.
const PRIOR_WEIGHT: f64 = 2.0;
/// Weight of a new round-time sample in a width's running cost.
const COST_ALPHA: f64 = 0.3;
/// Classic steps after which the policy drafts once more to refresh its
/// acceptance estimate.
pub(crate) const PROBE_AFTER_CLASSIC: usize = 8;

/// The width chooser. Owned by one `DFlashGenerator::run`.
#[derive(Debug, Clone)]
pub(crate) struct ExpectedThroughputWidth {
    ceiling: usize,
    /// Decayed rounds in which draft position `j + 1` was compared.
    tried: Vec<f64>,
    /// Decayed rounds in which draft position `j + 1` was accepted.
    hit: Vec<f64>,
    /// Running round time per width (`cost[w - 1]`), ms.
    cost: Vec<Option<f64>>,
    /// Rounds run at each width (`rounds_at[w - 1]`), for the run summary.
    rounds_at: Vec<usize>,
    classic_streak: usize,
}

impl ExpectedThroughputWidth {
    pub(crate) fn new(ceiling: usize) -> Self {
        let ceiling = ceiling.max(1);
        Self {
            ceiling,
            tried: vec![0.0; ceiling - 1],
            hit: vec![0.0; ceiling - 1],
            cost: vec![None; ceiling],
            rounds_at: vec![0; ceiling],
            classic_streak: 0,
        }
    }

    /// `p_j` for draft positions `1..ceiling`.
    pub(crate) fn acceptance(&self) -> Vec<f64> {
        let mut previous = PRIOR_FIRST;
        self.tried
            .iter()
            .zip(&self.hit)
            .map(|(&tried, &hit)| {
                let p = (hit + PRIOR_WEIGHT * previous) / (tried + PRIOR_WEIGHT);
                previous = p;
                p
            })
            .collect()
    }

    /// Expected tokens a `width`-row round emits.
    pub(crate) fn expected_emitted(&self, width: usize) -> f64 {
        let mut chain = 1.0;
        let mut total = 1.0;
        for p in self.acceptance().into_iter().take(width.saturating_sub(1)) {
            chain *= p;
            total += chain;
        }
        total
    }

    /// Measured or interpolated round time at `width`, ms.
    pub(crate) fn cost_estimate(&self, width: usize) -> Option<f64> {
        if width == 0 || width > self.ceiling {
            return None;
        }
        if let Some(cost) = self.cost[width - 1] {
            return Some(cost);
        }
        let measured = |w: usize| self.cost[w - 1].map(|c| (w, c));
        let below = (1..width).rev().find_map(measured);
        let above = (width + 1..=self.ceiling).find_map(measured);
        match (below, above) {
            (Some((lo, c_lo)), Some((hi, c_hi))) => {
                Some(c_lo + (c_hi - c_lo) * (width - lo) as f64 / (hi - lo) as f64)
            }
            (Some((_, c)), None) | (None, Some((_, c))) => Some(c),
            (None, None) => None,
        }
    }

    fn best_in(&self, widths: impl Iterator<Item = usize>) -> Option<usize> {
        widths
            .filter_map(|w| Some((w, self.expected_emitted(w) / self.cost_estimate(w)?)))
            .fold(None, |best: Option<(usize, f64)>, (w, rate)| match best {
                Some((_, best_rate)) if best_rate >= rate => best,
                _ => Some((w, rate)),
            })
            .map(|(w, _)| w)
    }

    /// The width of the next round, at most `remaining_budget` (which
    /// counts the bonus position, as the round loop passes it).
    pub(crate) fn decide(&self, remaining_budget: usize) -> usize {
        let cap = self.ceiling.min(remaining_budget);
        if cap <= 1 {
            return cap;
        }
        // Measure the requested width (past its warm-up round), then one
        // classic step, before comparing.
        if self.cost[self.ceiling - 1].is_none() {
            return cap;
        }
        if self.cost[0].is_none() {
            return 1;
        }
        if self.classic_streak >= PROBE_AFTER_CLASSIC {
            return self.best_in(2..=cap).unwrap_or(cap);
        }
        self.best_in(1..=cap).unwrap_or(cap)
    }

    /// `width:rounds@cost_ms` for every width run, then the acceptance
    /// estimates: the debug line the round loop logs at the end of a run.
    pub(crate) fn summary(&self) -> String {
        let widths: Vec<String> = (1..=self.ceiling)
            .filter(|&w| self.rounds_at[w - 1] > 0)
            .map(|w| {
                let cost = self.cost[w - 1].map_or("-".to_string(), |c| format!("{c:.1}"));
                format!("{w}:{}@{cost}", self.rounds_at[w - 1])
            })
            .collect();
        let accept: Vec<String> = self
            .acceptance()
            .iter()
            .map(|p| format!("{p:.2}"))
            .collect();
        format!(
            "widths [{}] acceptance [{}]",
            widths.join(" "),
            accept.join(" ")
        )
    }

    /// How many classic steps the round loop should run back to back once
    /// [`Self::decide`] picked width 1: up to the next re-try of the drafter.
    pub(crate) fn classic_run_len(&self) -> usize {
        PROBE_AFTER_CLASSIC
            .saturating_sub(self.classic_streak)
            .max(1)
    }

    /// Feed `steps` classic steps run back to back, at `step_ms` each when
    /// measured (the round loop excludes the first step of a run, which
    /// carries kernel warm-up and no overlap with the step before it).
    pub(crate) fn record_classic_steps(&mut self, steps: usize, step_ms: Option<f64>) {
        if steps == 0 {
            return;
        }
        self.rounds_at[0] += steps;
        self.classic_streak += steps;
        if let Some(ms) = step_ms.filter(|&ms| ms > 0.0) {
            self.add_cost_sample(1, ms);
        }
    }

    fn add_cost_sample(&mut self, width: usize, round_ms: f64) {
        let slot = &mut self.cost[width - 1];
        *slot = Some(match *slot {
            Some(old) => old + COST_ALPHA * (round_ms - old),
            None => round_ms,
        });
    }

    /// Feed one completed round: its width, the drafts it accepted and its
    /// wall time. The first round at each drafted width is not a cost
    /// sample: it compiles that width's kernels (the row-invariant matvec
    /// is specialized by row count), which measured 230 to 260 ms against
    /// 80 ms steady on Inkling 2 Small. A width-1 round is one classic step.
    pub(crate) fn record_round(&mut self, width: usize, accepted: usize, round_ms: f64) {
        if width == 0 || width > self.ceiling {
            return;
        }
        if width == 1 {
            self.record_classic_steps(1, Some(round_ms));
            return;
        }
        self.rounds_at[width - 1] += 1;
        self.classic_streak = 0;
        for value in self.tried.iter_mut().chain(self.hit.iter_mut()) {
            *value *= DECAY;
        }
        let compared = (accepted + 1).min(width - 1);
        for tried in &mut self.tried[..compared] {
            *tried += 1.0;
        }
        for hit in &mut self.hit[..accepted.min(width - 1)] {
            *hit += 1.0;
        }
        if self.rounds_at[width - 1] > 1 && round_ms > 0.0 {
            self.add_cost_sample(width, round_ms);
        }
    }
}

#[cfg(test)]
#[path = "width_policy_tests.rs"]
mod tests;
