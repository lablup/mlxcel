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

use super::*;

/// Inkling 2 Small's measured shape (issue #3410): a 51 ms decode step and
/// about 9 ms per extra verify row plus 13 ms of drafting.
fn small_cost(width: usize) -> f64 {
    if width == 1 {
        51.0
    } else {
        64.0 + 8.0 * (width - 1) as f64
    }
}

/// Drive the policy against a drafter whose draft `j` is accepted, given the
/// ones before it, exactly when `accepts(round, j)` holds.
fn drive(
    policy: &mut ExpectedThroughputWidth,
    rounds: usize,
    accepts: impl Fn(usize, usize) -> bool,
) -> Vec<usize> {
    (0..rounds)
        .map(|round| {
            let width = policy.decide(1000);
            let accepted = (1..width).take_while(|&j| accepts(round, j)).count();
            policy.record_round(width, accepted, small_cost(width));
            width
        })
        .collect()
}

#[test]
fn the_kill_switch_parses_like_the_other_inkling_switches() {
    assert!(adaptive_width_enabled_from(None));
    assert!(adaptive_width_enabled_from(Some("1")));
    for off in ["0", "false", "OFF", " no "] {
        assert!(!adaptive_width_enabled_from(Some(off)), "{off}");
    }
}

#[test]
fn it_measures_the_requested_width_then_a_classic_step() {
    let mut policy = ExpectedThroughputWidth::new(8);
    let widths = drive(&mut policy, 3, |_, _| false);
    assert_eq!(
        widths,
        [8, 8, 1],
        "warm-up round, cost sample, classic sample"
    );
}

#[test]
fn a_losing_drafter_falls_back_to_classic_decode_and_is_retried() {
    let mut policy = ExpectedThroughputWidth::new(8);
    // Nothing is ever accepted: every draft round costs more than a decode
    // step and emits the same one token.
    let widths = drive(&mut policy, 60, |_, _| false);
    let tail = &widths[10..];
    let classic = tail.iter().filter(|&&w| w == 1).count();
    assert!(
        classic * 10 >= tail.len() * 8,
        "mostly classic steps: {widths:?}"
    );
    // Every streak of classic steps ends in a draft round within the probe
    // period, so a drafter that starts winning is noticed.
    let mut streak = 0;
    for &width in tail {
        streak = if width == 1 { streak + 1 } else { 0 };
        assert!(streak <= PROBE_AFTER_CLASSIC, "{widths:?}");
    }
    assert!(tail.iter().any(|&w| w > 1), "the drafter is re-tried");
}

#[test]
fn a_strong_drafter_keeps_a_wide_block() {
    let mut policy = ExpectedThroughputWidth::new(8);
    let widths = drive(&mut policy, 40, |_, _| true);
    assert!(
        widths[10..].iter().all(|&w| w == 8),
        "full accepts keep the ceiling: {widths:?}"
    );
}

#[test]
fn a_middling_drafter_settles_between_classic_and_the_ceiling() {
    let mut policy = ExpectedThroughputWidth::new(8);
    // Prose-like: the first two drafts usually land, the rest rarely.
    let widths = drive(&mut policy, 80, |round, j| match j {
        1 => round % 5 != 0,
        2 => round % 3 != 0,
        _ => false,
    });
    let settled = &widths[40..];
    assert!(
        settled.iter().all(|&w| (2..8).contains(&w)),
        "a narrow block beats both classic and the full block: {widths:?}"
    );
    let mean = |w: usize| policy.expected_emitted(w) / small_cost(w);
    let chosen = policy.decide(1000);
    assert!(mean(chosen) > mean(1) && mean(chosen) > mean(8));
}

#[test]
fn the_budget_caps_the_width() {
    let mut policy = ExpectedThroughputWidth::new(8);
    drive(&mut policy, 20, |_, _| true);
    assert_eq!(policy.decide(3), 3);
    assert_eq!(policy.decide(1), 1);
}

#[test]
fn unmeasured_widths_interpolate_between_measured_neighbours() {
    let mut policy = ExpectedThroughputWidth::new(8);
    policy.record_round(8, 7, 999.0); // warm-up round: not a cost sample
    assert_eq!(policy.cost_estimate(8), None);
    policy.record_round(8, 7, 120.0);
    policy.record_round(1, 0, 50.0);
    assert_eq!(policy.cost_estimate(1), Some(50.0));
    assert_eq!(policy.cost_estimate(8), Some(120.0));
    assert!((policy.cost_estimate(4).unwrap() - 80.0).abs() < 1e-9);
    assert_eq!(policy.cost_estimate(0), None);
    assert_eq!(policy.cost_estimate(9), None);
}

#[test]
fn acceptance_estimates_follow_the_evidence() {
    let mut policy = ExpectedThroughputWidth::new(4);
    let prior = policy.acceptance();
    assert!(prior.iter().all(|&p| (p - PRIOR_FIRST).abs() < 1e-12));
    assert!((policy.expected_emitted(1) - 1.0).abs() < 1e-12);
    for _ in 0..30 {
        policy.record_round(4, 1, 10.0);
    }
    let p = policy.acceptance();
    assert!(p[0] > 0.9, "draft 1 always lands: {p:?}");
    // The prior keeps a little weight against about ten decayed rounds.
    assert!(p[1] < 0.2, "draft 2 never does: {p:?}");
    // Draft 3 was never compared, so it inherits draft 2's estimate.
    assert!((p[2] - p[1]).abs() < 1e-9, "{p:?}");
}

/// Issue #3410 on Small: the first round at a new width compiles its kernels
/// (250 ms against 80 steady). That sample must not stand for the width, or
/// the policy never comes back to it.
#[test]
fn the_first_round_at_each_width_is_not_a_cost_sample() {
    let mut policy = ExpectedThroughputWidth::new(8);
    policy.record_round(8, 7, 260.0);
    policy.record_round(8, 7, 120.0);
    policy.record_round(3, 2, 250.0);
    assert!(policy.summary().contains("3:1@-"), "{}", policy.summary());
    policy.record_round(3, 2, 80.0);
    assert!(
        policy.summary().contains("3:2@80.0"),
        "{}",
        policy.summary()
    );
    assert!(
        policy.summary().contains("8:2@120.0"),
        "{}",
        policy.summary()
    );
}

/// A run of pipelined classic steps counts every step toward the re-try
/// period and takes one steady per-step sample for width 1.
#[test]
fn a_classic_run_ends_at_the_next_draft_retry() {
    let mut policy = ExpectedThroughputWidth::new(8);
    assert_eq!(policy.classic_run_len(), PROBE_AFTER_CLASSIC);
    policy.record_classic_steps(3, Some(48.0));
    assert_eq!(policy.classic_run_len(), PROBE_AFTER_CLASSIC - 3);
    assert!(
        policy.summary().contains("1:3@48.0"),
        "{}",
        policy.summary()
    );
    policy.record_classic_steps(1, None);
    assert!(
        policy.summary().contains("1:4@48.0"),
        "{}",
        policy.summary()
    );
    policy.record_classic_steps(PROBE_AFTER_CLASSIC, None);
    assert_eq!(policy.classic_run_len(), 1, "never zero");
    policy.record_round(4, 0, 90.0);
    assert_eq!(policy.classic_run_len(), PROBE_AFTER_CLASSIC);
}

/// Lengths of the runs of classic steps in `widths`.
fn classic_streaks(widths: &[usize]) -> Vec<usize> {
    widths
        .split(|&w| w > 1)
        .map(<[usize]>::len)
        .filter(|&n| n > 0)
        .collect()
}

/// Issue #3425: with a backoff cap, each re-try that loses doubles the next
/// classic run up to the cap; without one the period stays fixed, as the
/// DFlash round loop runs it.
#[test]
fn a_probe_backoff_doubles_the_classic_run_after_each_lost_retry() {
    let mut fixed = ExpectedThroughputWidth::new(4);
    let streaks = classic_streaks(&drive(&mut fixed, 200, |_, _| false));
    assert!(
        streaks[1..streaks.len() - 1]
            .iter()
            .all(|&n| n == PROBE_AFTER_CLASSIC),
        "{streaks:?}"
    );

    let mut backoff = ExpectedThroughputWidth::new(4).with_probe_backoff(64);
    let streaks = classic_streaks(&drive(&mut backoff, 400, |_, _| false));
    // Every complete run is at least as long as the one before it (the
    // last run is cut off by the end of the drive).
    assert!(
        streaks[..streaks.len() - 1]
            .windows(2)
            .all(|pair| pair[1] >= pair[0]),
        "{streaks:?}"
    );
    assert!(streaks.contains(&64), "reaches the cap: {streaks:?}");
    assert!(streaks.iter().all(|&n| n <= 64), "{streaks:?}");
    assert_eq!(backoff.probe_period(), 64);
}

/// A re-try that wins resets the period, so a drafter that starts landing
/// (code after prose) is re-tried at the base period again.
#[test]
fn a_winning_retry_resets_the_probe_period() {
    let mut policy = ExpectedThroughputWidth::new(4).with_probe_backoff(64);
    drive(&mut policy, 300, |_, _| false);
    assert!(policy.probe_period() > PROBE_AFTER_CLASSIC);
    // Full accepts at 3 ms a round beat a 51 ms step only once the estimate
    // catches up, so drive with a cheap round instead of `small_cost`.
    for _ in 0..400 {
        let width = policy.decide(1000);
        let accepted = width - 1;
        let cost = if width == 1 { 51.0 } else { 20.0 };
        policy.record_round(width, accepted, cost);
    }
    assert_eq!(policy.probe_period(), PROBE_AFTER_CLASSIC);
    assert!(policy.decide(1000) > 1);
}
