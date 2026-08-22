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

//! Adaptive MTP block-size controllers.
//!
//! Two of them, one signal each:
//!
//! [`effective_mtp_block_size`] ports upstream
//! `mlx_vlm.speculative.mtp._effective_mtp_block_size`. Gemma 4 assistants
//! are configured for a 4-token verify block, but users may request a larger
//! `--draft-block-size`. The reference treats that larger value as a
//! ceiling: stay at the configured depth until the recent acceptance history
//! shows the configured prefix is usually fully accepted, then expand to the
//! requested ceiling. Issue #1207 measured the flaw in that proxy: on the
//! Gemma 4 12B pairing acceptance never clears the bar, the controller never
//! expands, and the pairing runs about 5% below its own measured optimum
//! (93.54 against 98.16 tok/s at requested width 5 on an M5 Max). The proxy
//! remains in use on the batched (B > 1) round loop, whose per-row averaged
//! acceptance is the only per-round signal that loop currently measures, and
//! behind `MLXCEL_MTP_BLOCK_CONTROLLER=proxy` as the escape hatch.
//!
//! [`BlockThroughputController`] replaces the proxy on the B = 1 round loop
//! with the direct signal: emitted tokens per millisecond of round time,
//! which is the quantity widening exists to improve and which the round
//! loop already pays to know. It alternates measurement windows between the
//! configured depth and the requested ceiling, adopts whichever measures
//! faster, and re-challenges the loser on a backoff schedule. The search
//! space is deliberately the same two widths the proxy chose between:
//! `--draft-block-size` stays a ceiling the user set, not a hint a walk
//! wanders away from.

/// Minimum number of completed MTP rounds before expanding above the
/// drafter's configured block size.
const MIN_HISTORY_FOR_EXPANSION: usize = 8;
/// Number of most-recent MTP rounds considered by the expansion gate.
const RECENT_HISTORY: usize = 32;
/// Required hit rate for accepting the entire configured draft prefix.
const CONFIGURED_PREFIX_HIT_RATE: f64 = 0.65;

/// Choose the next MTP verify block size.
///
/// `requested_block_total` is the user-facing ceiling. `configured_block_total`
/// comes from the drafter checkpoint. `accept_lens` stores accepted draft-token
/// counts per round (batched callers record the per-round row average, matching
/// upstream), and `remaining_budget` includes the prefix bonus position.
///
/// Used by: Gemma 4 MTP B=1 and B>1 round loops.
pub(crate) fn effective_mtp_block_size(
    requested_block_total: usize,
    configured_block_total: usize,
    accept_lens: &[f64],
    remaining_budget: usize,
) -> usize {
    let block_total = requested_block_total.min(remaining_budget);
    let configured_block_total = configured_block_total.min(block_total);
    if block_total <= configured_block_total || configured_block_total <= 1 {
        return block_total;
    }

    if accept_lens.len() < MIN_HISTORY_FOR_EXPANSION {
        return configured_block_total;
    }

    let recent_start = accept_lens.len().saturating_sub(RECENT_HISTORY);
    let recent = &accept_lens[recent_start..];
    let configured_draft_count = (configured_block_total - 1) as f64;
    let configured_prefix_hits = recent
        .iter()
        .filter(|&&accepted| accepted >= configured_draft_count)
        .count();
    let configured_prefix_hit_rate = configured_prefix_hits as f64 / recent.len() as f64;
    if configured_prefix_hit_rate < CONFIGURED_PREFIX_HIT_RATE {
        configured_block_total
    } else {
        block_total
    }
}

/// Rounds run at the configured depth before the first measurement window
/// opens. Mirrors [`MIN_HISTORY_FOR_EXPANSION`]: the first rounds of a
/// session carry prefill warm-up and kernel compilation, which would bias
/// whichever arm measured first.
const WARMUP_ROUNDS: usize = 8;
/// Rounds per measurement window. At the ~30 ms rounds of the measured
/// Gemma 4 pairing a window is about one second; acceptance variance makes
/// a single window a noisy estimate of a 5% difference, which the margin,
/// the re-challenge schedule, and the bounded two-arm search space are
/// sized to tolerate (a wrong adoption costs the measured 5%, is bounded by
/// the user's own ceiling, and is revisited).
const WINDOW_ROUNDS: usize = 32;
/// Relative lead the challenging arm must measure over the held arm's most
/// recent window to be adopted. Below this the tie goes to the held arm,
/// so measurement noise does not flap the width round-to-round.
const ADOPT_MARGIN: f64 = 0.02;
/// A challenge window aborts once it has this many rounds and trails the
/// held arm by more than [`EARLY_ABORT_DEFICIT`]. This is the Qwen 3.8
/// guard: at a requested width of 12 that pairing measures 5.80 against
/// 21.30 tok/s (issue #1207), and the deficit is visible within a few
/// rounds, so the collapsed arm is charged a handful of rounds rather
/// than a full window.
const EARLY_ABORT_MIN_ROUNDS: usize = 4;
const EARLY_ABORT_DEFICIT: f64 = 0.35;
/// Re-challenge schedule for the losing arm, in windows of the held arm:
/// starts at the base, multiplies by 4 per consecutive loss, and caps. A
/// consistently losing arm ends up probed for at most a few rounds per
/// couple of thousand, which prices the Qwen-style collapsed ceiling at
/// well under a percent of steady-state throughput.
const RECHALLENGE_BASE_WINDOWS: usize = 4;
const RECHALLENGE_GROWTH: usize = 4;
const RECHALLENGE_CAP_WINDOWS: usize = 64;

/// Which of the controller's two arms a round is charged to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Configured,
    Requested,
}

/// Two-arm throughput comparator for the B = 1 MTP verify width
/// (issue #1207).
///
/// Owned by the generator rather than the session, so a server process
/// keeps its evidence across requests; a CLI run is one session either
/// way. All state advances in [`Self::record_round`]; [`Self::decide`]
/// only reads the current arm and applies the remaining-budget cap.
///
/// Exactness scope: the widths this controller can pick are the two the
/// proxy controller already picked between, both bounded by the requested
/// width the exactness probe ran at, so it introduces no verify shape the
/// gate has not covered.
#[derive(Debug)]
pub(crate) struct BlockThroughputController {
    configured: usize,
    requested: usize,
    /// Arm the controller has settled on outside challenge windows.
    held: Arm,
    /// Arm the in-progress window is charged to. Differs from `held`
    /// exactly during a challenge.
    measuring: Arm,
    /// Most recent completed-window rate (emitted tokens per ms) per arm.
    rate_configured: Option<f64>,
    rate_requested: Option<f64>,
    window_rounds: usize,
    window_emitted: usize,
    window_ms: f64,
    warmup_rounds_left: usize,
    /// Held-arm windows to complete before the losing arm is re-tried.
    windows_until_challenge: usize,
    /// Current re-challenge delay, in windows. Grows per consecutive loss.
    challenge_backoff: usize,
}

impl BlockThroughputController {
    pub(crate) fn new(requested: usize, configured: usize) -> Self {
        Self {
            configured,
            requested,
            held: Arm::Configured,
            measuring: Arm::Configured,
            rate_configured: None,
            rate_requested: None,
            window_rounds: 0,
            window_emitted: 0,
            window_ms: 0.0,
            warmup_rounds_left: WARMUP_ROUNDS,
            // The first challenge follows the first completed configured
            // window, so the requested width is measured once per session
            // (or process) even when it never wins.
            windows_until_challenge: 1,
            challenge_backoff: RECHALLENGE_BASE_WINDOWS,
        }
    }

    /// Whether the two arms actually differ. When they do not (the request
    /// is at or below the configured depth), the controller is inert and
    /// [`Self::decide`] reproduces the proxy controller's early return.
    fn active(&self) -> bool {
        self.requested > self.configured && self.configured > 1
    }

    fn arm_width(&self, arm: Arm) -> usize {
        match arm {
            Arm::Configured => self.configured,
            Arm::Requested => self.requested,
        }
    }

    fn rate_of(&self, arm: Arm) -> Option<f64> {
        match arm {
            Arm::Configured => self.rate_configured,
            Arm::Requested => self.rate_requested,
        }
    }

    fn set_rate(&mut self, arm: Arm, rate: f64) {
        match arm {
            Arm::Configured => self.rate_configured = Some(rate),
            Arm::Requested => self.rate_requested = Some(rate),
        }
    }

    fn other(arm: Arm) -> Arm {
        match arm {
            Arm::Configured => Arm::Requested,
            Arm::Requested => Arm::Configured,
        }
    }

    /// The verify width the next round should use, bounded by the
    /// remaining emission budget (`remaining_budget` includes the prefix
    /// bonus position, exactly as [`effective_mtp_block_size`] takes it).
    pub(crate) fn decide(&self, remaining_budget: usize) -> usize {
        let ceiling = self.requested.min(remaining_budget);
        if !self.active() {
            return ceiling;
        }
        if self.warmup_rounds_left > 0 {
            return self.configured.min(ceiling);
        }
        self.arm_width(self.measuring).min(ceiling)
    }

    /// Feed one completed speculative round. `width` is the block the round
    /// actually verified: rounds the budget forced below the measuring
    /// arm's width are ignored rather than mis-charged.
    pub(crate) fn record_round(&mut self, width: usize, emitted: usize, round_ms: f64) {
        if !self.active() || round_ms <= 0.0 {
            return;
        }
        if self.warmup_rounds_left > 0 {
            self.warmup_rounds_left -= 1;
            return;
        }
        if width != self.arm_width(self.measuring) {
            return;
        }
        self.window_rounds += 1;
        self.window_emitted += emitted;
        self.window_ms += round_ms;

        let in_challenge = self.measuring != self.held;
        if in_challenge
            && self.window_rounds >= EARLY_ABORT_MIN_ROUNDS
            && let Some(held_rate) = self.rate_of(self.held)
        {
            let partial = self.window_emitted as f64 / self.window_ms;
            if partial < held_rate * (1.0 - EARLY_ABORT_DEFICIT) {
                // The challenger is collapsing; close its window now so a
                // Qwen-12-style arm costs rounds, not a full window.
                self.finish_window();
                return;
            }
        }
        if self.window_rounds >= WINDOW_ROUNDS {
            self.finish_window();
        }
    }

    fn finish_window(&mut self) {
        let rate = self.window_emitted as f64 / self.window_ms;
        let arm = self.measuring;
        self.set_rate(arm, rate);
        self.window_rounds = 0;
        self.window_emitted = 0;
        self.window_ms = 0.0;

        if arm != self.held {
            // A challenge window closed: adopt on a clear lead, otherwise
            // return to the held arm and back the loser off.
            let held_rate = self.rate_of(self.held);
            let adopted = match held_rate {
                Some(h) => rate > h * (1.0 + ADOPT_MARGIN),
                None => true,
            };
            if adopted {
                self.held = arm;
                self.challenge_backoff = RECHALLENGE_BASE_WINDOWS;
            } else {
                self.challenge_backoff =
                    (self.challenge_backoff * RECHALLENGE_GROWTH).min(RECHALLENGE_CAP_WINDOWS);
            }
            self.measuring = self.held;
            self.windows_until_challenge = self.challenge_backoff;
        } else {
            // A held-arm window closed: refresh its rate and count down to
            // the next challenge.
            self.windows_until_challenge = self.windows_until_challenge.saturating_sub(1);
            if self.windows_until_challenge == 0 {
                self.measuring = Self::other(self.held);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::effective_mtp_block_size;
    use super::{Arm, BlockThroughputController, WARMUP_ROUNDS, WINDOW_ROUNDS};

    /// Drive `rounds` rounds through the controller, asking `decide` first
    /// (as the round loop does) and answering with the arm's synthetic
    /// (emitted, ms) profile. Returns how many rounds ran at each width.
    fn drive(
        c: &mut BlockThroughputController,
        rounds: usize,
        budget: usize,
        profile: impl Fn(usize) -> (usize, f64),
    ) -> std::collections::HashMap<usize, usize> {
        let mut widths = std::collections::HashMap::new();
        for _ in 0..rounds {
            let w = c.decide(budget);
            *widths.entry(w).or_insert(0) += 1;
            let (emitted, ms) = profile(w);
            c.record_round(w, emitted, ms);
        }
        widths
    }

    /// The measured Gemma 4 12B pairing (issue #1207): configured 4 at
    /// 93.5 tok/s, requested 5 at 98.2. The controller must adopt 5.
    #[test]
    fn adopts_the_requested_width_when_it_measures_faster() {
        let mut c = BlockThroughputController::new(5, 4);
        let profile = |w: usize| match w {
            4 => (2705, 29160.0), // 2.705 emitted / 29.16 ms, scaled x1000
            5 => (3132, 32170.0),
            other => panic!("unexpected width {other}"),
        };
        drive(&mut c, WARMUP_ROUNDS + 3 * WINDOW_ROUNDS, 1 << 20, profile);
        assert_eq!(c.held, Arm::Requested);
        assert_eq!(c.decide(1 << 20), 5);
    }

    /// The measured Qwen 3.8 pairing: configured 3 at 21.30 tok/s,
    /// requested 12 at 5.80. The challenge must abort early and the
    /// controller must keep refusing to widen.
    #[test]
    fn refuses_a_requested_width_that_collapses() {
        let mut c = BlockThroughputController::new(12, 3);
        let profile = |w: usize| match w {
            3 => (2, 94.0),   // ~21.3 tok/s at ~2 emitted per round
            12 => (2, 345.0), // ~5.8 tok/s
            other => panic!("unexpected width {other}"),
        };
        let widths = drive(&mut c, WARMUP_ROUNDS + 20 * WINDOW_ROUNDS, 1 << 20, profile);
        assert_eq!(c.held, Arm::Configured);
        assert_eq!(c.decide(1 << 20), 3);
        // The collapsed arm was charged rounds, but only a few per
        // backoff cycle: bound it well below one full window per cycle.
        let wide_rounds = widths.get(&12).copied().unwrap_or(0);
        assert!(
            wide_rounds > 0 && wide_rounds < 3 * super::EARLY_ABORT_MIN_ROUNDS + 2,
            "collapsed arm charged {wide_rounds} rounds"
        );
    }

    /// A small lead inside the adoption margin does not flap the width.
    #[test]
    fn a_lead_inside_the_margin_stays_configured() {
        let mut c = BlockThroughputController::new(5, 4);
        let profile = |w: usize| match w {
            4 => (1000, 10000.0),
            5 => (1010, 10000.0), // +1.0%, inside the 2% margin
            other => panic!("unexpected width {other}"),
        };
        drive(&mut c, WARMUP_ROUNDS + 6 * WINDOW_ROUNDS, 1 << 20, profile);
        assert_eq!(c.held, Arm::Configured);
    }

    /// Budget caps every width, and capped rounds are not charged to the
    /// measuring arm's window.
    #[test]
    fn budget_caps_the_width_and_capped_rounds_are_ignored() {
        let mut c = BlockThroughputController::new(5, 4);
        assert_eq!(c.decide(3), 3);
        let before = c.window_rounds;
        c.record_round(3, 3, 30.0);
        assert_eq!(c.window_rounds, before);
    }

    /// Inert when the request is not above the configured depth: behaves
    /// like the proxy controller's early return.
    #[test]
    fn inert_at_or_below_the_configured_depth() {
        let c = BlockThroughputController::new(4, 4);
        assert_eq!(c.decide(1 << 20), 4);
        let c = BlockThroughputController::new(3, 4);
        assert_eq!(c.decide(1 << 20), 3);
    }

    #[test]
    fn stays_at_requested_when_not_above_configured() {
        assert_eq!(effective_mtp_block_size(4, 4, &[], 16), 4);
        assert_eq!(effective_mtp_block_size(3, 4, &[], 16), 3);
    }

    #[test]
    fn caps_by_remaining_budget() {
        assert_eq!(effective_mtp_block_size(8, 4, &[3.0; 16], 3), 3);
    }

    #[test]
    fn warms_up_at_configured_depth_before_history_threshold() {
        assert_eq!(effective_mtp_block_size(8, 4, &[3.0; 7], 16), 4);
    }

    #[test]
    fn stays_configured_when_recent_prefix_hit_rate_is_low() {
        let mut history = vec![3.0; 16];
        history.extend([0.0; 16]);
        assert_eq!(effective_mtp_block_size(8, 4, &history, 16), 4);
    }

    #[test]
    fn expands_to_requested_when_recent_prefix_hit_rate_is_high() {
        let mut history = vec![0.0; 8];
        history.extend([3.0; 24]);
        assert_eq!(effective_mtp_block_size(8, 4, &history, 16), 8);
    }
}
