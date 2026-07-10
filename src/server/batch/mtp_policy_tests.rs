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

//! Unit tests for the adaptive MTP policy (issue #333).
//!
//! These cover the pure decision logic, the profile accumulator, the verdict
//! state machine (including manual env overrides in both directions), and the
//! coarse-hint persistence (asserting no prompt / token data is ever
//! serialized). The temperature-0 MTP-vs-classic byte-identity gate is a
//! real-model check the orchestrator runs; the in-repo greedy-parity unit gate
//! lives in `mlxcel_core::speculative::mtp::tests`. The structural test
//! [`profile_sample_carries_no_token_level_data`] pins the contract that this
//! policy is decision-only and can never see or mutate emitted tokens.

use super::*;
use mlxcel_core::speculative::mtp::MtpAcceptanceSummary;

// ── helpers ─────────────────────────────────────────────────────────────────

fn key() -> PolicyKey {
    PolicyKey::new(
        "target-model".to_string(),
        "drafter-model".to_string(),
        "M5-16c".to_string(),
        4,
    )
}

/// Build a profile sample with explicit per-round means so the verdict is
/// deterministic. `rounds` rounds, each with `accepted_per_round` accepted
/// drafts out of `proposed_per_round` proposals, `draft_ms_per_round` drafter
/// time and `verify_ms_per_round` verify time.
fn sample(
    rounds: usize,
    proposed_per_round: usize,
    accepted_per_round: usize,
    draft_ms_per_round: f64,
    verify_ms_per_round: f64,
) -> MtpBurstProfile {
    MtpBurstProfile::from_summary(
        MtpAcceptanceSummary {
            rounds,
            proposed_tokens: proposed_per_round * rounds,
            accepted_draft_tokens: accepted_per_round * rounds,
            draft_ms: draft_ms_per_round * rounds as f64,
            verify_forward_ms: verify_ms_per_round * rounds as f64,
        },
        1,
        128,
    )
}

/// A clearly favorable sample: high acceptance, cheap drafter (optimistic
/// speedup ≈ 2.8, well above [`ENABLE_SPEEDUP_FLOOR`]).
fn favorable_sample() -> MtpBurstProfile {
    sample(10, 3, 2, 1.0, 4.0)
}

/// A clearly unfavorable sample: low acceptance, drafter as costly as verify
/// (optimistic speedup ≈ 0.75, below [`DECLINE_SPEEDUP_CEIL`]).
fn unfavorable_sample() -> MtpBurstProfile {
    // accepted_len = 5/10 = 0.5, drafter_ms == verify_ms → 1.5 / 2.0 = 0.75.
    let mut s = sample(10, 3, 0, 4.0, 4.0);
    s.accepted_draft_tokens = 5;
    s
}

/// An ambiguous sample: optimistic speedup ≈ 1.15, inside the dead-band.
fn ambiguous_sample() -> MtpBurstProfile {
    // accepted_len = 1.3, drafter_ms == verify_ms → 2.3 / 2.0 = 1.15.
    let mut s = sample(10, 3, 0, 4.0, 4.0);
    s.accepted_draft_tokens = 13;
    s
}

fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "mlxcel-mtp-policy-test-{}-{}-{n}",
        std::process::id(),
        tag
    ))
}

// ── pure decision core ────────────────────────────────────────────────────────

#[test]
fn estimate_speedup_matches_closed_form() {
    // (accepted_len + 1) / (1 + drafter_ms / verify_ms).
    let s = estimate_speedup(2.5, 4.0, 1.0).expect("finite");
    assert!((s - 2.8).abs() < 1e-9, "got {s}");
    let s = estimate_speedup(0.5, 4.0, 4.0).expect("finite");
    assert!((s - 0.75).abs() < 1e-9, "got {s}");
}

#[test]
fn estimate_speedup_scaled_de_rates_compute_bound() {
    // multiple == 1.0 reproduces the bandwidth-bound (Apple) estimate exactly.
    let apple = estimate_speedup_scaled(2.5, 4.0, 1.0, 1.0).expect("finite");
    assert!((apple - 2.8).abs() < 1e-9, "got {apple}");
    assert_eq!(estimate_speedup(2.5, 4.0, 1.0), Some(apple));
    // multiple == 2.0 (compute-bound, K=4 → sqrt(4)) halves the estimate.
    let cuda = estimate_speedup_scaled(2.5, 4.0, 1.0, 2.0).expect("finite");
    assert!((cuda - 1.4).abs() < 1e-9, "got {cuda}");
    // Degenerate multiple is rejected.
    assert_eq!(estimate_speedup_scaled(2.5, 4.0, 1.0, 0.0), None);
    assert_eq!(estimate_speedup_scaled(2.5, 4.0, 1.0, -1.0), None);
}

#[test]
fn verify_cost_multiple_is_backend_conditional() {
    // key() carries block_size K=4. Apple Silicon amortizes the K-wide verify
    // (multiple 1.0); compute-bound hardware de-rates by sqrt(K) = 2.0.
    let apple = adaptive_policy_hw(false, false, false);
    assert!((apple.verify_cost_multiple() - 1.0).abs() < 1e-9);
    let cuda = adaptive_policy_hw(false, false, true);
    assert!((cuda.verify_cost_multiple() - 2.0).abs() < 1e-9);
}

#[test]
fn compute_bound_derating_declines_a_borderline_pairing() {
    // A sample whose optimistic (Apple) estimate lands in the ENABLE zone
    // (accepted_len = 0.8, zero drafter cost → speedup 1.8) but whose
    // compute-bound scaled estimate (÷2.0) lands in the DECLINE zone (0.9).
    let mut borderline = sample(10, 3, 0, 0.0, 4.0);
    borderline.accepted_draft_tokens = 8; // accepted_len = 0.8
    let mut acc = ProfileAccumulator::default();
    acc.add(&borderline);
    let apple = acc.estimated_speedup_scaled(1.0).expect("finite");
    assert_eq!(
        classify_speedup(apple),
        SpeedupZone::Enable,
        "apple={apple}"
    );
    let cuda = acc.estimated_speedup_scaled(2.0).expect("finite");
    assert_eq!(
        classify_speedup(cuda),
        SpeedupZone::Decline,
        "compute-bound de-rating must decline the borderline pairing, cuda={cuda}"
    );
}

#[test]
fn compute_bound_derating_keeps_a_favorable_pairing_enabled() {
    // A high-acceptance pairing must still clear the ENABLE floor after the
    // compute-bound de-rate: accepted_len = 2.4, zero drafter cost gives an
    // optimistic speedup of 3.4, and the / 2.0 de-rate lands at 1.7, above
    // ENABLE_SPEEDUP_FLOOR (1.5). Pins the PR #733 claim that the de-rate
    // only declines marginal pairings, not genuinely favorable ones.
    let mut favorable = sample(10, 3, 0, 0.0, 4.0);
    favorable.accepted_draft_tokens = 24; // accepted_len = 2.4
    let mut acc = ProfileAccumulator::default();
    acc.add(&favorable);
    let cuda = acc.estimated_speedup_scaled(2.0).expect("finite");
    assert_eq!(
        classify_speedup(cuda),
        SpeedupZone::Enable,
        "favorable compute-bound pairing must stay enabled, cuda={cuda}"
    );
}

#[test]
fn estimate_speedup_zero_acceptance_is_below_one() {
    // No drafts accepted: speedup = 1 / (1 + drafter/verify) < 1 always.
    let s = estimate_speedup(0.0, 4.0, 1.0).expect("finite");
    assert!(s < 1.0, "zero-acceptance must never beat classic, got {s}");
}

#[test]
fn estimate_speedup_rejects_degenerate_inputs() {
    assert_eq!(estimate_speedup(2.0, 0.0, 1.0), None, "verify_ms == 0");
    assert_eq!(estimate_speedup(2.0, -1.0, 1.0), None, "verify_ms < 0");
    assert_eq!(estimate_speedup(2.0, f64::NAN, 1.0), None, "verify_ms NaN");
    assert_eq!(
        estimate_speedup(2.0, f64::INFINITY, 1.0),
        None,
        "verify_ms inf"
    );
    assert_eq!(estimate_speedup(-1.0, 4.0, 1.0), None, "accepted_len < 0");
    assert_eq!(estimate_speedup(2.0, 4.0, -1.0), None, "drafter_ms < 0");
}

#[test]
fn classify_speedup_zones() {
    assert_eq!(classify_speedup(0.75), SpeedupZone::Decline);
    assert_eq!(
        classify_speedup(DECLINE_SPEEDUP_CEIL),
        SpeedupZone::Decline,
        "boundary is inclusive on the decline side"
    );
    assert_eq!(classify_speedup(1.15), SpeedupZone::Ambiguous);
    assert_eq!(
        classify_speedup(ENABLE_SPEEDUP_FLOOR),
        SpeedupZone::Enable,
        "boundary is inclusive on the enable side"
    );
    assert_eq!(classify_speedup(2.8), SpeedupZone::Enable);
}

#[test]
fn resolve_verdict_overrides_and_defers() {
    // Confident zones override the static default in both directions.
    assert!(resolve_verdict(Some(SpeedupZone::Enable), false));
    assert!(!resolve_verdict(Some(SpeedupZone::Decline), true));
    // Ambiguous / missing estimate defers to the static default.
    assert!(resolve_verdict(Some(SpeedupZone::Ambiguous), true));
    assert!(!resolve_verdict(Some(SpeedupZone::Ambiguous), false));
    assert!(resolve_verdict(None, true));
    assert!(!resolve_verdict(None, false));
}

// ── accumulator ───────────────────────────────────────────────────────────────

#[test]
fn accumulator_aggregates_round_weighted() {
    let mut acc = ProfileAccumulator::default();
    acc.add(&sample(4, 3, 2, 1.0, 4.0));
    acc.add(&sample(6, 3, 2, 1.0, 4.0));
    assert_eq!(acc.samples(), 2);
    // Means are taken over total rounds (10), so equal per-round inputs give
    // back the same per-round means.
    assert!((acc.accepted_len() - 2.0).abs() < 1e-9);
    assert!((acc.drafter_ms() - 1.0).abs() < 1e-9);
    assert!((acc.verify_ms() - 4.0).abs() < 1e-9);
    assert!((acc.acceptance_rate() - (2.0 / 3.0)).abs() < 1e-9);
}

#[test]
fn accumulator_records_batch_and_prompt_shape() {
    // The profiler must record acceptance, latency, batch size, AND prompt
    // shape over the first few requests (acceptance criterion). Verify the
    // batch-size / prompt-shape dimensions are captured alongside the timing.
    let mut acc = ProfileAccumulator::default();
    acc.add(&MtpBurstProfile::from_summary(
        MtpAcceptanceSummary {
            rounds: 4,
            proposed_tokens: 12,
            accepted_draft_tokens: 8,
            draft_ms: 4.0,
            verify_forward_ms: 16.0,
        },
        1,
        64,
    ));
    acc.add(&MtpBurstProfile::from_summary(
        MtpAcceptanceSummary {
            rounds: 4,
            proposed_tokens: 12,
            accepted_draft_tokens: 8,
            draft_ms: 4.0,
            verify_forward_ms: 16.0,
        },
        1,
        192,
    ));
    assert_eq!(acc.max_batch_size(), 1, "B=1 singleton path");
    assert_eq!(acc.max_prompt_len(), 192);
    assert_eq!(acc.mean_prompt_len(), 128, "(64 + 192) / 2");
}

#[test]
fn accumulator_ignores_zero_round_samples() {
    let mut acc = ProfileAccumulator::default();
    acc.add(&sample(0, 0, 0, 0.0, 0.0));
    assert_eq!(acc.samples(), 0, "zero-round samples carry no signal");
    assert_eq!(acc.estimated_speedup(), None);
}

// ── policy state machine ──────────────────────────────────────────────────────

/// Drive a policy with `n` copies of `s`, returning it after.
fn drive(mut policy: MtpPolicy, s: MtpBurstProfile, n: usize) -> MtpPolicy {
    for _ in 0..n {
        policy.record_b1_sample(s);
    }
    policy
}

fn adaptive_policy(static_default_batching: bool, has_na: bool) -> MtpPolicy {
    // force = None → adaptive; no-dir store keeps these state-machine tests
    // off the filesystem (persistence is exercised separately). Bandwidth-bound
    // (Apple) hardware unless the test opts into the compute-bound path.
    adaptive_policy_hw(static_default_batching, has_na, false)
}

fn adaptive_policy_hw(
    static_default_batching: bool,
    has_na: bool,
    compute_bound: bool,
) -> MtpPolicy {
    MtpPolicy::from_parts(
        key(),
        static_default_batching,
        has_na,
        compute_bound,
        None,
        PolicyStore::with_dir(None),
    )
}

#[test]
fn favorable_profile_enables_overriding_static_decline() {
    // Batch-capable target + no neural accelerator → static default DECLINE.
    let policy = adaptive_policy(true, false);
    assert!(
        !policy.static_default(),
        "precondition: static default must decline here"
    );
    // While profiling, MTP is forced on to gather samples.
    assert!(policy.should_attempt_b1());
    assert!(!policy.is_settled());

    let policy = drive(policy, favorable_sample(), PROFILE_SAMPLE_TARGET);
    assert!(policy.is_settled());
    assert!(
        policy.should_attempt_b1(),
        "favorable profile must enable MTP, overriding the static decline"
    );
}

#[test]
fn unfavorable_profile_declines_overriding_static_enable() {
    // Non-batchable target → static default ENABLE.
    let policy = adaptive_policy(false, false);
    assert!(
        policy.static_default(),
        "precondition: static default must enable here"
    );
    let policy = drive(policy, unfavorable_sample(), PROFILE_SAMPLE_TARGET);
    assert!(policy.is_settled());
    assert!(
        !policy.should_attempt_b1(),
        "unfavorable profile must decline MTP, overriding the static enable"
    );
}

#[test]
fn ambiguous_profile_follows_static_default() {
    // Same ambiguous profile settles to whatever the static default says.
    let enabled = drive(
        adaptive_policy(false, false),
        ambiguous_sample(),
        PROFILE_SAMPLE_TARGET,
    );
    assert!(
        enabled.should_attempt_b1(),
        "ambiguous + static enable → enable"
    );
    let declined = drive(
        adaptive_policy(true, false),
        ambiguous_sample(),
        PROFILE_SAMPLE_TARGET,
    );
    assert!(
        !declined.should_attempt_b1(),
        "ambiguous + static decline → decline"
    );
}

#[test]
fn settles_only_after_enough_samples() {
    let mut policy = adaptive_policy(true, false);
    for i in 1..PROFILE_SAMPLE_TARGET {
        policy.record_b1_sample(favorable_sample());
        assert!(!policy.is_settled(), "must still profile after {i} samples");
        assert!(policy.should_attempt_b1(), "forced on while profiling");
    }
    policy.record_b1_sample(favorable_sample());
    assert!(policy.is_settled(), "settles on the target-th sample");
}

#[test]
fn zero_round_samples_do_not_advance_profiling() {
    let mut policy = adaptive_policy(true, false);
    for _ in 0..(PROFILE_SAMPLE_TARGET * 2) {
        policy.record_b1_sample(sample(0, 0, 0, 0.0, 0.0));
    }
    assert!(
        !policy.is_settled(),
        "zero-round samples must not count toward the profiling window"
    );
}

#[test]
fn env_force_on_pins_enable_and_never_profiles() {
    // force = Some(true): even with a target whose static default declines.
    let mut policy = MtpPolicy::from_parts(
        key(),
        true,
        false,
        false,
        Some(true),
        PolicyStore::with_dir(Some(unique_temp_dir("force-on"))),
    );
    assert!(policy.is_settled(), "a forced policy never profiles");
    assert!(policy.should_attempt_b1());
    // Recording an unfavorable sample must not flip the manual force.
    policy.record_b1_sample(unfavorable_sample());
    assert!(policy.should_attempt_b1(), "manual force-on is sticky");
}

#[test]
fn env_force_off_pins_decline_and_never_profiles() {
    let mut policy = MtpPolicy::from_parts(
        key(),
        false, // static default would enable
        true,
        false,
        Some(false),
        PolicyStore::with_dir(Some(unique_temp_dir("force-off"))),
    );
    assert!(policy.is_settled());
    assert!(!policy.should_attempt_b1());
    policy.record_b1_sample(favorable_sample());
    assert!(!policy.should_attempt_b1(), "manual force-off is sticky");
}

#[test]
fn parse_force_override_directions() {
    assert_eq!(parse_force_override(None), None);
    assert_eq!(parse_force_override(Some("1")), Some(true));
    assert_eq!(parse_force_override(Some("on")), Some(true));
    assert_eq!(parse_force_override(Some("0")), Some(false));
    assert_eq!(parse_force_override(Some("false")), Some(false));
    assert_eq!(parse_force_override(Some("off")), Some(false));
    assert_eq!(parse_force_override(Some("no")), Some(false));
}

#[test]
fn adaptive_enabled_defaults_on() {
    assert!(adaptive_enabled(None), "default is adaptive-on");
    assert!(adaptive_enabled(Some("1")));
    assert!(!adaptive_enabled(Some("0")));
    assert!(!adaptive_enabled(Some("off")));
}

// ── persistence ───────────────────────────────────────────────────────────────

#[test]
fn settling_persists_a_loadable_verdict() {
    let dir = unique_temp_dir("persist");
    let store = PolicyStore::with_dir(Some(dir.clone()));
    let policy = MtpPolicy::from_parts(key(), true, false, false, None, store.clone());
    let _ = drive(policy, favorable_sample(), PROFILE_SAMPLE_TARGET);

    // A fresh policy for the same pairing must load the persisted verdict and
    // skip profiling entirely (no force-on cost on restart).
    let reloaded = MtpPolicy::from_parts(key(), true, false, false, None, store);
    assert!(
        reloaded.is_settled(),
        "persisted verdict must be loaded at construction"
    );
    assert!(reloaded.should_attempt_b1(), "persisted ENABLE verdict");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn persisted_hint_holds_only_coarse_fields() {
    let dir = unique_temp_dir("coarse");
    let store = PolicyStore::with_dir(Some(dir.clone()));
    let policy = MtpPolicy::from_parts(key(), false, false, false, None, store.clone());
    // Settle on a favorable verdict and persist it.
    let _ = drive(policy, favorable_sample(), PROFILE_SAMPLE_TARGET);

    // Read the raw file back and assert its shape is exactly the coarse hint:
    // no prompt text, no token ids, nothing request-identifying.
    let file = std::fs::read_dir(&dir)
        .expect("hint dir exists")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "json"))
        .expect("a persisted .json hint");
    let raw = std::fs::read_to_string(&file).expect("read hint");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
    let obj = value.as_object().expect("hint is a json object");

    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "acceptance_rate",
            "block_size",
            "drafter",
            "hardware",
            "samples",
            "target",
            "verdict",
            "version",
        ],
        "persisted hint must carry only coarse, non-request-identifying fields"
    );
    // Defensive: no field name or value smells like request/prompt data.
    for forbidden in ["prompt", "token", "text", "input", "request", "content"] {
        assert!(
            !raw.to_lowercase().contains(forbidden),
            "hint leaked a '{forbidden}'-shaped field: {raw}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn persisted_acceptance_rate_is_coarse_rounded() {
    let hint = PolicyHint::new(&key(), Verdict::Enable, 0.666_666_7, 4);
    assert_eq!(hint.acceptance_rate, 0.67, "rounded to two decimals");
    assert_eq!(hint.samples, 4);
    assert_eq!(hint.verdict, Verdict::Enable);
}

#[test]
fn hint_roundtrips_through_serde() {
    let hint = PolicyHint::new(&key(), Verdict::Decline, 0.42, 4);
    let json = serde_json::to_string(&hint).expect("serialize");
    let back: PolicyHint = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.verdict, Verdict::Decline);
    assert_eq!(back.acceptance_rate, 0.42);
    assert_eq!(back.target, "target-model");
}

#[test]
fn store_with_no_dir_is_a_silent_noop() {
    // No resolvable cache root → persistence is skipped, but the policy still
    // works in-memory for the session.
    let store = PolicyStore::with_dir(None);
    assert!(store.load(&key()).is_none());
    let hint = PolicyHint::new(&key(), Verdict::Enable, 0.5, 4);
    assert!(store.save(&hint).is_ok(), "no-dir save must be a no-op Ok");
}

#[test]
fn loaded_hint_is_rejected_on_key_mismatch() {
    // Guards against a hash collision handing back another pairing's verdict.
    let dir = unique_temp_dir("mismatch");
    let store = PolicyStore::with_dir(Some(dir.clone()));
    let other = PolicyKey::new("a".into(), "b".into(), "c".into(), 4);
    store
        .save(&PolicyHint::new(&other, Verdict::Enable, 0.5, 4))
        .expect("save");
    // Loading a *different* key must not return the stored hint (different
    // hash → different file → None anyway; this pins the contract).
    assert!(store.load(&key()).is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn policy_key_hash_is_stable_and_distinct() {
    let a = PolicyKey::new("t".into(), "d".into(), "hw".into(), 4);
    let b = PolicyKey::new("t".into(), "d".into(), "hw".into(), 4);
    let c = PolicyKey::new("t".into(), "d".into(), "other-hw".into(), 4);
    assert_eq!(a.hash(), b.hash(), "same key → same hash");
    assert_ne!(a.hash(), c.hash(), "different hardware → different hash");
    assert_eq!(a.display(), "t|d|hw|K4");
    assert_eq!(a.hash().len(), 16, "16 hex chars");
}

// ── block_size key separation ─────────────────────────────────────────────────

/// A verdict profiled at block_size=K must not be reused when K changes.
/// Different block sizes produce different hashes (and therefore different hint
/// files), so a change in `--num-draft-tokens` / `MLXCEL_DRAFT_BLOCK_SIZE`
/// triggers a fresh profiling window rather than inheriting the old verdict.
#[test]
fn block_size_change_produces_different_key_and_re_profiles() {
    let k4 = PolicyKey::new("t".into(), "d".into(), "hw".into(), 4);
    let k8 = PolicyKey::new("t".into(), "d".into(), "hw".into(), 8);
    assert_ne!(
        k4.hash(),
        k8.hash(),
        "keys differing only in block_size must hash differently"
    );
    assert_ne!(k4.display(), k8.display());

    // A hint saved at K=4 must not load for K=8 (different file, different
    // key check), and vice versa.
    let dir = unique_temp_dir("block-size-sep");
    let store = PolicyStore::with_dir(Some(dir.clone()));

    let hint4 = PolicyHint::new(&k4, Verdict::Enable, 0.75, 4);
    store.save(&hint4).expect("save K=4 hint");

    // Loading with the same key (K=4) succeeds.
    assert!(
        store.load(&k4).is_some(),
        "same block_size must load the hint"
    );
    // Loading with a different block_size (K=8) returns None; re-profile.
    assert!(
        store.load(&k8).is_none(),
        "different block_size must not load the hint (triggers re-profile)"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── load degradation (LOW-3) ──────────────────────────────────────────────────

/// `PolicyStore::load` must return `None` (degrade to re-profile, no panic) on
/// truncated or garbage JSON. The `.ok()?` on `serde_json::from_str` covers
/// this; the test pins the contract so a future refactor cannot break it.
#[test]
fn load_returns_none_on_garbage_json() {
    let dir = unique_temp_dir("garbage-json");
    std::fs::create_dir_all(&dir).expect("create dir");
    let store = PolicyStore::with_dir(Some(dir.clone()));
    // Write garbage directly to the path where load would look.
    let k = key();
    let path = dir.join(format!("{}.json", k.hash()));
    std::fs::write(&path, b"this is not json {{{{").expect("write garbage");
    assert!(
        store.load(&k).is_none(),
        "garbage JSON must degrade to None, not panic"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `PolicyStore::load` must return `None` on a hint whose version field does
/// not match `HINT_VERSION`. The version check covers schema changes (e.g.
/// the v1 -> v2 addition of `block_size`); old hints are silently discarded
/// and the pairing re-profiles rather than crashing or handing back a stale
/// verdict.
#[test]
fn load_returns_none_on_version_mismatch() {
    let dir = unique_temp_dir("version-mismatch");
    std::fs::create_dir_all(&dir).expect("create dir");
    let store = PolicyStore::with_dir(Some(dir.clone()));
    let k = key();

    // Write a hint with a deliberately wrong version (0 instead of HINT_VERSION).
    let stale_json = serde_json::json!({
        "version": 0u32,
        "target": "target-model",
        "drafter": "drafter-model",
        "hardware": "M5-16c",
        "block_size": 4u32,
        "verdict": "enable",
        "acceptance_rate": 0.75,
        "samples": 4,
    });
    let path = dir.join(format!("{}.json", k.hash()));
    std::fs::write(
        &path,
        serde_json::to_string(&stale_json).unwrap().as_bytes(),
    )
    .expect("write stale hint");
    assert!(
        store.load(&k).is_none(),
        "version mismatch must degrade to None and trigger re-profile"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── exactness contract ────────────────────────────────────────────────────────

#[test]
fn profile_sample_carries_no_token_level_data() {
    // The policy is decision-only: it consumes coarse round counts and
    // latencies, never tokens. This compile-time/structural pin guarantees the
    // policy cannot observe or mutate emitted tokens, so MTP's temperature-0
    // byte-identity with classic decode is unaffected by any verdict. The
    // realized byte-identity is validated on real models by the orchestrator
    // and by the greedy-parity gate in mlxcel_core::speculative::mtp::tests.
    let summary = MtpAcceptanceSummary {
        rounds: 8,
        proposed_tokens: 24,
        accepted_draft_tokens: 18,
        draft_ms: 8.0,
        verify_forward_ms: 32.0,
    };
    let profile = MtpBurstProfile::from_summary(summary, 1, 256);
    // Only aggregate counts, latencies, batch size, and prompt length: the
    // exact fields the issue lists. No token ids, no prompt bytes.
    assert_eq!(profile.batch_size, 1);
    assert_eq!(profile.prompt_len, 256);
    assert_eq!(profile.rounds, 8);
    assert_eq!(profile.accepted_draft_tokens, 18);
    assert_eq!(profile.proposed_tokens, 24);
}
