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

//! Unit tests for the shape-bucketed kernel autotuner (issue #906).
//!
//! Every test here is CPU-only: the [`FakeOp`] double implements
//! [`TunableOp`] by sleeping for a per-tactic duration and overrides `sync` to
//! a no-op, so the harness, the cache, and the precedence chain are exercised
//! without a GPU. The GPU-backed consumers are validated by benchmark runs,
//! not here.

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Duration;

use super::bucket::{MAX_BUCKET_DIM, ShapeBucket, powers_of_two_up_to, round_up_pow2};
use super::ops::cuda_kernel_knobs::{
    QMV_MAX_ROWS_HARD, TILE_M_CAP_BLACKWELL, TILE_M_CAP_DEFAULT, TILE_M_MIN, default_tile_m,
    max_rows_candidates, tile_m_candidates,
};
use super::ops::paged_decode_splits::split_candidates;
use super::profile::{ProfileConfig, median, profile};
use super::store::{
    TACTIC_VERSION, TacticRecord, TacticStore, TuneKey, mlx_commit, mlxcel_version,
};
use super::tactic::{Tactic, TunableOp, TuneError};
use super::{Mode, Resolution, Source, clear_memo, parse_mode, resolve_with_mode};

// ── Test double ──────────────────────────────────────────────────────────────

/// A [`TunableOp`] whose "kernel" is a sleep of a per-tactic duration.
///
/// Sleeping is what makes the min-latency selection observable without a
/// backend. Durations are in the tens of microseconds so a full sweep stays
/// well under a millisecond.
struct FakeOp {
    op: String,
    bucket: ShapeBucket,
    /// `(param, micros)` pairs; the first entry is also the default.
    plan: Vec<(i64, u64)>,
    default_param: i64,
    env: Option<Tactic>,
    lazy: bool,
    /// Which tactics fail instead of running.
    failing: Vec<i64>,
    calls: RefCell<usize>,
}

impl FakeOp {
    fn new(op: &str, plan: Vec<(i64, u64)>) -> Self {
        let default_param = plan.first().map_or(0, |(p, _)| *p);
        Self {
            op: op.to_string(),
            bucket: ShapeBucket::from_dims(&[1, 32, 4096]),
            plan,
            default_param,
            env: None,
            lazy: true,
            failing: Vec::new(),
            calls: RefCell::new(0),
        }
    }

    fn with_default(mut self, param: i64) -> Self {
        self.default_param = param;
        self
    }

    fn with_env(mut self, tactic: Tactic) -> Self {
        self.env = Some(tactic);
        self
    }

    fn not_lazy(mut self) -> Self {
        self.lazy = false;
        self
    }

    fn failing(mut self, params: &[i64]) -> Self {
        self.failing = params.to_vec();
        self
    }

    fn calls(&self) -> usize {
        *self.calls.borrow()
    }
}

impl TunableOp for FakeOp {
    fn op_name(&self) -> &str {
        &self.op
    }
    fn runner_id(&self) -> String {
        "fake".to_string()
    }
    fn dtype_tag(&self) -> String {
        "f32".to_string()
    }
    fn bucket(&self) -> ShapeBucket {
        self.bucket.clone()
    }
    fn candidates(&self, _bucket: &ShapeBucket) -> Vec<Tactic> {
        self.plan
            .iter()
            .map(|(p, _)| Tactic::scalar("p", *p))
            .collect()
    }
    fn default_tactic(&self, _bucket: &ShapeBucket) -> Tactic {
        Tactic::scalar("p", self.default_param)
    }
    fn env_override(&self) -> Option<Tactic> {
        self.env.clone()
    }
    fn lazy_tunable(&self) -> bool {
        self.lazy
    }
    fn run(&self, tactic: &Tactic) -> Result<(), TuneError> {
        *self.calls.borrow_mut() += 1;
        let p = tactic.param(0).unwrap_or(-1);
        if self.failing.contains(&p) {
            return Err(TuneError::failed(tactic, "injected failure"));
        }
        let micros = self
            .plan
            .iter()
            .find(|(pp, _)| *pp == p)
            .map_or(0, |(_, us)| *us);
        std::thread::sleep(Duration::from_micros(micros));
        Ok(())
    }
    fn sync(&self) {
        // No backend in these tests.
    }
}

fn fast_cfg() -> ProfileConfig {
    ProfileConfig {
        warmup: 0,
        reps: 5,
        min_improvement: 0.02,
    }
}

fn tmp_store(dir: &tempfile::TempDir) -> TacticStore {
    TacticStore::with_dir(Some(dir.path().to_path_buf()))
}

fn key_for(op: &dyn TunableOp) -> TuneKey {
    TuneKey::new(
        op.op_name().to_string(),
        op.runner_id(),
        super::device_label(),
        op.bucket(),
        op.dtype_tag(),
    )
}

// ── Shape bucketing ──────────────────────────────────────────────────────────

#[test]
fn round_up_pow2_rounds_up_never_down() {
    assert_eq!(round_up_pow2(0), 1);
    assert_eq!(round_up_pow2(1), 1);
    assert_eq!(round_up_pow2(2), 2);
    assert_eq!(round_up_pow2(3), 4);
    assert_eq!(round_up_pow2(4097), 8192);
    assert_eq!(round_up_pow2(8192), 8192);
}

#[test]
fn round_up_pow2_saturates_instead_of_overflowing() {
    assert_eq!(round_up_pow2(usize::MAX), MAX_BUCKET_DIM);
    assert_eq!(round_up_pow2(MAX_BUCKET_DIM as usize), MAX_BUCKET_DIM);
}

#[test]
fn nearby_shapes_share_one_bucket() {
    // The whole point of bucketing: a decode sweep across a power-of-two span
    // must produce one cache entry, not thousands.
    let a = ShapeBucket::from_dims(&[1, 32, 4097]);
    let b = ShapeBucket::from_dims(&[1, 32, 8192]);
    assert_eq!(a, b);
    assert_eq!(a.to_string(), "1x32x8192");
}

#[test]
fn bucket_display_and_accessors() {
    let b = ShapeBucket::from_exact(&[1, 32, 8, 128, 4096]);
    assert_eq!(b.to_string(), "1x32x8x128x4096");
    assert_eq!(b.len(), 5);
    assert!(!b.is_empty());
    assert_eq!(b.dim(3), Some(128));
    assert_eq!(b.dim(9), None);
    assert!(!b.is_saturated());
    assert_eq!(ShapeBucket::from_exact(&[]).to_string(), "scalar");
}

#[test]
fn saturated_bucket_is_flagged() {
    let b = ShapeBucket::from_dims(&[usize::MAX, 4]);
    assert!(b.is_saturated());
}

#[test]
fn powers_of_two_up_to_is_inclusive_and_bounded() {
    assert_eq!(powers_of_two_up_to(0), Vec::<u32>::new());
    assert_eq!(powers_of_two_up_to(1), vec![1]);
    assert_eq!(powers_of_two_up_to(28), vec![1, 2, 4, 8, 16]);
    assert_eq!(powers_of_two_up_to(32), vec![1, 2, 4, 8, 16, 32]);
    // Must terminate rather than overflow the shift.
    assert!(!powers_of_two_up_to(u32::MAX).is_empty());
}

// ── Median ───────────────────────────────────────────────────────────────────

#[test]
fn median_odd_even_and_empty() {
    assert_eq!(median(&[]), None);
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
}

#[test]
fn median_is_robust_to_a_single_outlier() {
    // The reason the harness uses a median: one scheduling spike must not move
    // the score. The mean here would be 2004.0.
    let with_spike = [1.0, 1.0, 10000.0, 1.0, 1.0];
    assert_eq!(median(&with_spike), Some(1.0));
}

#[test]
fn median_handles_nan_without_panicking() {
    let m = median(&[1.0, f64::NAN, 2.0]);
    assert!(m.is_some());
}

// ── Profiling harness ────────────────────────────────────────────────────────

#[test]
fn profile_picks_the_min_latency_candidate() {
    // Default (2) is slow; 8 is 10x faster, well past the noise margin.
    let op = FakeOp::new("fake_min_latency", vec![(2, 800), (4, 400), (8, 80)]);
    let result = profile(&op, fast_cfg()).expect("sweep produced a result");
    assert_eq!(result.best, Tactic::scalar("p", 8));
    assert!(result.changed);
    assert_eq!(result.measurements.len(), 3);
    assert_eq!(result.reps, 5);
    assert!(result.default_us.is_some());
    assert!(result.speedup_over_default().unwrap_or(0.0) > 1.0);
}

#[test]
fn profile_keeps_the_default_inside_the_noise_band() {
    // Candidate 8 is genuinely ~2x faster, but the required margin here is 90%,
    // so the win does not clear it and the pre-tuner behavior must survive.
    // This is the regression guard: a win inside the host's noise floor
    // converges back to the default rather than to a coin flip. The margin is
    // exaggerated so that `thread::sleep` jitter cannot decide the assertion.
    let cfg = ProfileConfig {
        warmup: 0,
        reps: 5,
        min_improvement: 0.9,
    };
    let op = FakeOp::new("fake_noise_band", vec![(2, 400), (4, 300), (8, 200)]);
    let result = profile(&op, cfg).expect("sweep produced a result");
    assert_eq!(result.best, Tactic::scalar("p", 2));
    assert!(!result.changed);
}

#[test]
fn profile_drops_failing_candidates_without_aborting() {
    let op = FakeOp::new("fake_failing", vec![(2, 800), (4, 80), (8, 400)]).failing(&[4]);
    let result = profile(&op, fast_cfg()).expect("sweep survived the failure");
    // 4 was the fastest but is infeasible, so 8 wins.
    assert_eq!(result.best, Tactic::scalar("p", 8));
    assert_eq!(result.measurements.len(), 2);
}

#[test]
fn profile_returns_none_when_every_candidate_fails() {
    let op = FakeOp::new("fake_all_fail", vec![(2, 10), (4, 10)]).failing(&[2, 4]);
    assert!(profile(&op, fast_cfg()).is_none());
}

#[test]
fn profile_returns_none_without_candidates() {
    let op = FakeOp::new("fake_no_candidates", vec![]);
    assert!(profile(&op, fast_cfg()).is_none());
}

#[test]
fn profile_config_sanitizes_degenerate_values() {
    let cfg = ProfileConfig {
        warmup: 0,
        reps: 0,
        min_improvement: f64::NAN,
    }
    .sanitized();
    assert_eq!(cfg.reps, 1);
    assert!(cfg.min_improvement.is_finite());
}

#[test]
fn profile_runs_warmup_plus_reps_per_candidate() {
    let cfg = ProfileConfig {
        warmup: 2,
        reps: 5,
        min_improvement: 0.02,
    };
    let op = FakeOp::new("fake_call_count", vec![(1, 1), (2, 1)]);
    let _ = profile(&op, cfg).expect("sweep produced a result");
    assert_eq!(op.calls(), 2 * (2 + 5));
}

// ── Persistent cache ─────────────────────────────────────────────────────────

#[test]
fn store_round_trips_a_record() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_round_trip", vec![(1, 1)]);
    let key = key_for(&op);
    let record = TacticRecord::new(&key, Tactic::scalar("p", 8), 123.0, Some(200.0), 3, 5);
    store.save(&key, &record).expect("save");
    let loaded = store.load(&key).expect("load");
    assert_eq!(loaded, record);
    assert_eq!(loaded.tactic.param(0), Some(8));
}

#[test]
fn store_without_a_directory_is_a_silent_no_op() {
    let store = TacticStore::with_dir(None);
    let op = FakeOp::new("fake_no_dir", vec![(1, 1)]);
    let key = key_for(&op);
    let record = TacticRecord::new(&key, Tactic::scalar("p", 8), 1.0, None, 1, 5);
    assert!(store.save(&key, &record).is_ok());
    assert!(store.load(&key).is_none());
}

#[test]
fn store_rejects_a_stale_schema_version() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_version", vec![(1, 1)]);
    let key = key_for(&op);
    let mut record = TacticRecord::new(&key, Tactic::scalar("p", 8), 1.0, None, 1, 5);
    record.version = TACTIC_VERSION + 1;
    store.save(&key, &record).expect("save");
    assert!(store.load(&key).is_none());
}

#[test]
fn store_rejects_a_different_mlxcel_version() {
    // The metadata-mismatch invalidation gate: an mlxcel upgrade may have
    // changed the launcher, so the tactic must be re-measured, not reused.
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_mlxcel_version", vec![(1, 1)]);
    let key = key_for(&op);
    let mut record = TacticRecord::new(&key, Tactic::scalar("p", 8), 1.0, None, 1, 5);
    record.mlxcel_version = format!("{}-doctored", mlxcel_version());
    store.save(&key, &record).expect("save");
    assert!(store.load(&key).is_none());
}

#[test]
fn store_rejects_a_different_mlx_pin() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_mlx_pin", vec![(1, 1)]);
    let key = key_for(&op);
    let mut record = TacticRecord::new(&key, Tactic::scalar("p", 8), 1.0, None, 1, 5);
    record.mlx_commit = "0000000000000000000000000000000000000000".to_string();
    assert_ne!(record.mlx_commit, mlx_commit());
    store.save(&key, &record).expect("save");
    assert!(store.load(&key).is_none());
}

#[test]
fn store_rejects_a_key_field_mismatch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_key_mismatch", vec![(1, 1)]);
    let key = key_for(&op);
    let mut record = TacticRecord::new(&key, Tactic::scalar("p", 8), 1.0, None, 1, 5);
    // Simulate a hash collision: the file exists at this key's path but was
    // written for a different bucket.
    record.bucket = "999x999".to_string();
    store.save(&key, &record).expect("save");
    assert!(store.load(&key).is_none());
}

#[test]
fn store_ignores_a_corrupt_file_instead_of_failing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_corrupt", vec![(1, 1)]);
    let key = key_for(&op);
    let path: PathBuf = dir.path().join(format!("{}.json", key.hash()));
    std::fs::write(&path, b"{ this is not json").expect("write corrupt file");
    // Never panics, never propagates: a corrupt cache degrades to a miss.
    assert!(store.load(&key).is_none());
}

#[test]
fn store_ignores_a_truncated_but_valid_json_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_truncated", vec![(1, 1)]);
    let key = key_for(&op);
    let path = dir.path().join(format!("{}.json", key.hash()));
    std::fs::write(&path, br#"{"version": 1}"#).expect("write partial record");
    assert!(store.load(&key).is_none());
}

#[test]
fn keys_that_differ_in_any_field_hash_differently() {
    let bucket = ShapeBucket::from_dims(&[1, 2, 3]);
    let base = TuneKey::new("op", "metal", "M1-16c-128gb", bucket.clone(), "f32");
    let others = [
        TuneKey::new("op2", "metal", "M1-16c-128gb", bucket.clone(), "f32"),
        TuneKey::new("op", "cuda", "M1-16c-128gb", bucket.clone(), "f32"),
        TuneKey::new("op", "metal", "M5-16c-128gb", bucket.clone(), "f32"),
        TuneKey::new(
            "op",
            "metal",
            "M1-16c-128gb",
            ShapeBucket::from_dims(&[1, 2, 8]),
            "f32",
        ),
        TuneKey::new("op", "metal", "M1-16c-128gb", bucket, "f16"),
    ];
    for other in &others {
        assert_ne!(
            base.hash(),
            other.hash(),
            "{} vs {}",
            base.display(),
            other.display()
        );
    }
}

// ── Mode parsing ─────────────────────────────────────────────────────────────

#[test]
fn mode_defaults_to_off() {
    assert_eq!(parse_mode(None), Some(Mode::Off));
    assert_eq!(parse_mode(Some("0")), Some(Mode::Off));
    assert_eq!(parse_mode(Some("off")), Some(Mode::Off));
    assert_eq!(parse_mode(Some("false")), Some(Mode::Off));
    assert_eq!(parse_mode(Some("")), Some(Mode::Off));
    assert!(!Mode::Off.reads_cache());
    assert!(!Mode::Off.profiles());
}

#[test]
fn mode_parses_tune_and_cache_only() {
    assert_eq!(parse_mode(Some("1")), Some(Mode::Tune));
    assert_eq!(parse_mode(Some("ON")), Some(Mode::Tune));
    assert_eq!(parse_mode(Some("cache")), Some(Mode::CacheOnly));
    assert_eq!(parse_mode(Some(" read-only ")), Some(Mode::CacheOnly));
    assert!(Mode::CacheOnly.reads_cache());
    assert!(!Mode::CacheOnly.profiles());
    assert!(Mode::Tune.profiles());
}

#[test]
fn mode_rejects_an_unrecognized_value() {
    assert_eq!(parse_mode(Some("maybe")), None);
}

// ── Resolution precedence ────────────────────────────────────────────────────

fn resolve(op: &dyn TunableOp, store: &TacticStore, mode: Mode) -> Resolution {
    clear_memo();
    resolve_with_mode(op, store, fast_cfg(), mode)
}

#[test]
fn resolve_is_inert_when_the_mode_is_off() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_off", vec![(2, 800), (8, 80)]);
    let r = resolve(&op, &store, Mode::Off);
    assert_eq!(r.source, Source::Default);
    assert_eq!(r.tactic, Tactic::scalar("p", 2));
    // Nothing profiled and nothing written.
    assert_eq!(op.calls(), 0);
    assert!(store.load(&key_for(&op)).is_none());
}

#[test]
fn resolve_lets_an_explicit_env_override_win_over_a_cached_tactic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_env_wins", vec![(2, 800), (8, 80)]);
    let key = key_for(&op);
    let record = TacticRecord::new(&key, Tactic::scalar("p", 8), 80.0, Some(800.0), 2, 5);
    store.save(&key, &record).expect("save");

    let op = op.with_env(Tactic::scalar("p", 4));
    let r = resolve(&op, &store, Mode::Tune);
    assert_eq!(r.source, Source::EnvOverride);
    assert_eq!(r.tactic, Tactic::scalar("p", 4));
    assert_eq!(op.calls(), 0, "an env override must not profile");
}

#[test]
fn resolve_uses_a_cached_tactic_without_profiling() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_cache_hit", vec![(2, 800), (8, 80)]);
    let key = key_for(&op);
    let record = TacticRecord::new(&key, Tactic::scalar("p", 8), 80.0, Some(800.0), 2, 5);
    store.save(&key, &record).expect("save");

    let r = resolve(&op, &store, Mode::CacheOnly);
    assert_eq!(r.source, Source::Cache);
    assert_eq!(r.tactic, Tactic::scalar("p", 8));
    assert_eq!(op.calls(), 0);
}

#[test]
fn resolve_falls_back_to_default_on_a_cache_only_miss() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_cache_miss", vec![(2, 800), (8, 80)]);
    let r = resolve(&op, &store, Mode::CacheOnly);
    assert_eq!(r.source, Source::Default);
    assert_eq!(op.calls(), 0, "cache-only mode must never profile");
}

#[test]
fn resolve_profiles_and_persists_under_tune_mode() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_lazy_tune", vec![(2, 800), (8, 80)]);
    let r = resolve(&op, &store, Mode::Tune);
    assert_eq!(r.source, Source::Tuned);
    assert_eq!(r.tactic, Tactic::scalar("p", 8));
    assert!(op.calls() > 0);

    // The tuned choice is now persistent, so a fresh process would hit cache.
    let stored = store.load(&key_for(&op)).expect("record persisted");
    assert_eq!(stored.tactic, Tactic::scalar("p", 8));
    assert_eq!(stored.mlxcel_version, mlxcel_version());
    assert_eq!(stored.mlx_commit, mlx_commit());
}

#[test]
fn resolve_never_lazily_profiles_a_process_wide_op() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_not_lazy", vec![(2, 800), (8, 80)]).not_lazy();
    let r = resolve(&op, &store, Mode::Tune);
    assert_eq!(r.source, Source::Default);
    assert_eq!(op.calls(), 0);
}

#[test]
fn resolve_reports_out_of_bucket_when_there_are_no_candidates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_out_of_bucket", vec![]).with_default(7);
    let r = resolve(&op, &store, Mode::Tune);
    assert_eq!(r.source, Source::OutOfBucket);
    assert!(r.source.is_default());
    assert_eq!(r.tactic, Tactic::scalar("p", 7));
}

#[test]
fn resolve_discards_a_cached_tactic_that_is_no_longer_feasible() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_infeasible_cache", vec![(2, 800), (8, 80)]);
    let key = key_for(&op);
    // 64 was tuned under a wider budget that no longer exists.
    let record = TacticRecord::new(&key, Tactic::scalar("p", 64), 10.0, None, 2, 5);
    store.save(&key, &record).expect("save");
    let r = resolve(&op, &store, Mode::CacheOnly);
    assert_eq!(r.source, Source::Default);
    assert_eq!(r.tactic, Tactic::scalar("p", 2));
}

#[test]
fn resolve_retunes_over_an_infeasible_cached_tactic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_retune_infeasible", vec![(2, 800), (8, 80)]);
    let key = key_for(&op);
    let record = TacticRecord::new(&key, Tactic::scalar("p", 64), 10.0, None, 2, 5);
    store.save(&key, &record).expect("save");

    let r = resolve(&op, &store, Mode::Tune);
    assert_eq!(r.source, Source::Tuned);
    assert_eq!(r.tactic, Tactic::scalar("p", 8));
    // The unusable entry is replaced rather than left to be rejected forever.
    let stored = store.load(&key).expect("record rewritten");
    assert_eq!(stored.tactic, Tactic::scalar("p", 8));
}

#[test]
fn resolve_memoizes_so_a_hot_path_reads_the_cache_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = tmp_store(&dir);
    let op = FakeOp::new("fake_memo", vec![(2, 400), (8, 40)]);
    clear_memo();
    let first = resolve_with_mode(&op, &store, fast_cfg(), Mode::Tune);
    let calls_after_first = op.calls();
    let second = resolve_with_mode(&op, &store, fast_cfg(), Mode::Tune);
    assert_eq!(first, second);
    assert_eq!(
        op.calls(),
        calls_after_first,
        "memoized resolve must not re-profile"
    );
}

// ── Consumer candidate spaces ────────────────────────────────────────────────

#[test]
fn paged_decode_split_candidates_cover_the_budget_ceiling() {
    // head_dim 128 -> ceiling 32, already a power of two.
    assert_eq!(split_candidates(32), vec![1, 2, 4, 8, 16, 32]);
    // head_dim 256 -> ceiling 28, which is not a power of two, so it must be
    // appended: it is the pre-#906 default and has to be measured for the
    // "only switch on a margin" rule to mean anything.
    assert_eq!(split_candidates(28), vec![1, 2, 4, 8, 16, 28]);
    assert_eq!(split_candidates(1), vec![1]);
    assert_eq!(split_candidates(0), vec![1]);
}

#[test]
fn qmm_tile_default_mirrors_the_kernel_formula() {
    // make_cta_tiler: max(16, min(cap, next_power_of_2(m))).
    assert_eq!(default_tile_m(1, TILE_M_CAP_DEFAULT), TILE_M_MIN);
    assert_eq!(default_tile_m(100, TILE_M_CAP_DEFAULT), 64);
    assert_eq!(default_tile_m(8192, TILE_M_CAP_DEFAULT), 64);
    assert_eq!(default_tile_m(8192, TILE_M_CAP_BLACKWELL), 128);
    assert_eq!(default_tile_m(33, TILE_M_CAP_BLACKWELL), 64);
}

#[test]
fn qmm_tile_candidates_stop_at_the_arch_cap() {
    assert_eq!(tile_m_candidates(TILE_M_CAP_DEFAULT), vec![16, 32, 64]);
    assert_eq!(
        tile_m_candidates(TILE_M_CAP_BLACKWELL),
        vec![16, 32, 64, 128]
    );
    // A nonsense cap still yields the minimum tile rather than an empty set.
    assert_eq!(tile_m_candidates(0), vec![16]);
}

#[test]
fn qmv_multirow_candidates_span_the_documented_window() {
    let c = max_rows_candidates();
    assert_eq!(c.first().copied(), Some(1));
    assert_eq!(c.last().copied(), Some(QMV_MAX_ROWS_HARD));
    assert_eq!(c.len(), QMV_MAX_ROWS_HARD as usize);
}
