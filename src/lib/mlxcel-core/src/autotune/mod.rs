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

//! Shape-bucketed kernel autotuner (issue #906).
//!
//! ## What this replaces
//!
//! mlxcel's kernel configuration was manual and shape-blind. The Blackwell qmm
//! CTA tile was tuned once by hand and set by env
//! (`MLXCEL_QMM_TILE_{M,N,K}`); the multirow-qmv decode window was a
//! hardcoded 8-row ceiling; the Metal paged-decode kernel picks its
//! `NumSplits` launch shape from a threadgroup-memory budget alone, ignoring
//! context length and batch. Those are per-shape decisions, and a per-shape
//! decision made once by hand is wrong for most shapes.
//!
//! ## The durable part is the key, not the loop
//!
//! The profiling loop here is intentionally small. What matters is the cache
//! contract:
//!
//! - **Key** = `(op, kernel/runner identity, nearest shape bucket, extras such
//!   as dtype)`. Shapes round up to power-of-two buckets ([`ShapeBucket`]) so
//!   nearby shapes share one entry and the tuning matrix stays bounded.
//! - **Invalidation** = cached tactics carry the environment they were measured
//!   in (mlxcel version, pinned MLX commit, device label). A mismatch discards
//!   the entry instead of silently applying a stale config. See
//!   [`store`] for the field-by-field table.
//! - **Fallback** = an out-of-bucket or empty-candidate lookup warns and
//!   returns the op's own default. Nothing about the tuner can make a launch
//!   fail.
//!
//! ## Precedence
//!
//! Highest wins:
//!
//! 1. An explicitly-set environment variable for that knob
//!    ([`TunableOp::env_override`]). A user who set `MLXCEL_QMM_TILE_M=64`
//!    gets 64, tuned entry or not.
//! 2. A valid cached tactic for the bucket.
//! 3. A tactic profiled now (only under `MLXCEL_AUTOTUNE=1`).
//! 4. The op's default, which is exactly the pre-#906 behavior.
//!
//! ## Default off
//!
//! With `MLXCEL_AUTOTUNE` unset the autotuner is fully inert: [`resolve`]
//! returns the default without reading the cache, taking a lock, or touching
//! the filesystem, so cold-start latency and steady-state behavior are
//! unchanged. `MLXCEL_AUTOTUNE=cache` consumes tactics that an earlier
//! `mlxcel tune` wrote but never profiles at runtime; `MLXCEL_AUTOTUNE=1`
//! additionally profiles on the first use of a bucket.
//!
//! ## Extension point for paged decode v2 (issue #898, now occupied)
//!
//! Issue #898's paged-decode v2 plan chooses a `kv_chunk_size`, which is
//! exactly the shape-dependent knob this module exists for. That seam is now
//! filled by [`ops::paged_decode_v2_chunk`], which does precisely what this
//! note prescribed:
//!
//! 1. registers its op under [`OP_PAGED_DECODE_V2_KV_CHUNK`],
//! 2. implements [`TunableOp`] over its own candidate chunk sizes with
//!    `default_tactic` returning the plan's binary-search heuristic,
//! 3. calls [`resolve`] once per plan build and reads the chunk size out of
//!    [`Resolution::tactic`] via [`Tactic::param`].
//!
//! No code here changed for it: the key, the store, and the harness are
//! op-agnostic, and the feasible chunk sizes are enumerated by the v2 plan's
//! own accounting rather than being pre-built here.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub mod bucket;
pub mod ops;
pub mod profile;
pub mod store;
pub mod tactic;

pub use bucket::{MAX_BUCKET_DIM, ShapeBucket, powers_of_two_up_to, round_up_pow2};
pub use profile::{Measurement, ProfileConfig, ProfileResult, Selection, profile, select};
pub use store::{TACTIC_SUBDIR, TACTIC_VERSION, TacticRecord, TacticStore, TuneKey};
pub use tactic::{Tactic, TunableOp, TuneError};

/// Environment variable gating the autotuner. Default off.
pub const AUTOTUNE_ENV: &str = "MLXCEL_AUTOTUNE";

/// Reserved op name for issue #898's paged-decode v2 `kv_chunk_size` knob.
///
/// Declared here so the name is fixed before the consumer exists: a cache
/// entry's key embeds the op name, so choosing it late would strand entries.
/// See the module docs for how #898 plugs in.
pub const OP_PAGED_DECODE_V2_KV_CHUNK: &str = "paged_decode_v2_kv_chunk";

/// How much the autotuner is allowed to do this process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Fully inert. The default: no cache reads, no profiling, defaults
    /// untouched.
    #[default]
    Off,
    /// Consume cached tactics written by an earlier `mlxcel tune`, but never
    /// profile at runtime.
    CacheOnly,
    /// Consume cached tactics and profile the first use of an untuned bucket.
    Tune,
}

impl Mode {
    /// Whether the cache should be consulted at all.
    #[must_use]
    pub fn reads_cache(self) -> bool {
        !matches!(self, Mode::Off)
    }

    /// Whether runtime profiling is permitted.
    #[must_use]
    pub fn profiles(self) -> bool {
        matches!(self, Mode::Tune)
    }
}

/// Parse an `MLXCEL_AUTOTUNE` value.
///
/// Accepts the tree's usual on/off spellings plus `cache`/`read` for the
/// consume-only mode. An unrecognized value is treated as off (and the caller
/// warns), because silently enabling a profiling path on a typo is worse than
/// ignoring it.
#[must_use]
pub fn parse_mode(value: Option<&str>) -> Option<Mode> {
    let Some(raw) = value else {
        return Some(Mode::Off);
    };
    let v = raw.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "0" | "false" | "off" | "no" => Some(Mode::Off),
        "1" | "true" | "on" | "yes" => Some(Mode::Tune),
        "cache" | "read" | "readonly" | "read-only" => Some(Mode::CacheOnly),
        _ => None,
    }
}

/// The process-wide mode, read once from the environment.
#[must_use]
pub fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var(AUTOTUNE_ENV).ok();
        match parse_mode(raw.as_deref()) {
            Some(m) => m,
            None => {
                tracing::warn!(
                    "{AUTOTUNE_ENV}={:?} is not a recognized value (expected 1/on/true, cache, or 0/off); autotuner stays off",
                    raw.unwrap_or_default()
                );
                Mode::Off
            }
        }
    })
}

/// Coarse device label used as cache-key material.
///
/// Apple-silicon generation plus the performance-core proxy and unified-memory
/// size, matching the `hardware_label()` convention in the MTP policy store:
/// enough to distinguish an M1 Max from an M1 Ultra (different SLC, different
/// launch-shape optimum) without recording anything host-identifying.
/// Non-Apple hosts collapse to the `Unknown` generation, which is correct but
/// coarse; a CUDA device-name source is the natural refinement once a tuned
/// CUDA op is actually validated.
#[must_use]
pub fn device_label() -> String {
    static LABEL: OnceLock<String> = OnceLock::new();
    LABEL
        .get_or_init(|| {
            let hw = crate::hardware::get_hardware();
            format!(
                "{}-{}c-{}gb",
                hw.silicon_gen, hw.gpu_core_count, hw.unified_memory_gb
            )
        })
        .clone()
}

/// Where a resolved tactic came from. Recorded so logs and tests can assert
/// the precedence rules rather than inferring them from a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// An explicitly-set environment variable for this knob.
    EnvOverride,
    /// A valid entry in the persistent tactic cache.
    Cache,
    /// Profiled in this process just now.
    Tuned,
    /// The op's own default (autotuner off, cache miss without profiling, or
    /// a sweep in which nothing ran).
    Default,
    /// The shape fell outside the tuned matrix (saturated bucket or no
    /// candidates). The default is used and the miss is warned about once.
    OutOfBucket,
}

impl Source {
    /// Whether this source means the pre-#906 default was used verbatim.
    #[must_use]
    pub fn is_default(self) -> bool {
        matches!(self, Source::Default | Source::OutOfBucket)
    }
}

/// A resolved tactic plus where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    pub tactic: Tactic,
    pub source: Source,
}

impl Resolution {
    #[must_use]
    fn new(tactic: Tactic, source: Source) -> Self {
        Self { tactic, source }
    }

    /// Parameter `i` of the resolved tactic, or `None` when the tactic carries
    /// fewer parameters than the caller expected.
    #[must_use]
    pub fn param(&self, i: usize) -> Option<i64> {
        self.tactic.param(i)
    }
}

/// Process-local memo of resolved keys.
///
/// The decode path resolves the same key on every step, so a filesystem read
/// per step would be a real cost. Memoizing both hits and misses bounds the
/// autotuner to one cache read and at most one profiling sweep per key per
/// process.
type Memo = HashMap<TuneKey, Resolution>;

fn memo() -> &'static Mutex<Memo> {
    static MEMO: OnceLock<Mutex<Memo>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolve the tactic for one invocation of `op`, applying the full precedence
/// chain (env override, cache, profile, default).
///
/// Never fails. Every error path falls back to [`TunableOp::default_tactic`],
/// so an unusable cache, an unresolvable cache root, or a device where every
/// candidate is infeasible all reduce to today's behavior.
#[must_use]
pub fn resolve(op: &dyn TunableOp) -> Resolution {
    resolve_with(
        op,
        &TacticStore::from_cache_root(),
        ProfileConfig::default(),
    )
}

/// [`resolve`] against an explicit store and profile config. Separated so the
/// offline `mlxcel tune` path and the unit tests can drive the same logic
/// without depending on the ambient cache root.
#[must_use]
pub fn resolve_with(op: &dyn TunableOp, store: &TacticStore, cfg: ProfileConfig) -> Resolution {
    resolve_with_mode(op, store, cfg, mode())
}

/// [`resolve_with`] against an explicit mode.
///
/// [`mode`] latches the environment once per process, so tests that need to
/// exercise more than one mode drive this entry point instead of mutating the
/// environment (which would fix the mode for the whole test binary).
#[must_use]
pub fn resolve_with_mode(
    op: &dyn TunableOp,
    store: &TacticStore,
    cfg: ProfileConfig,
    mode: Mode,
) -> Resolution {
    let bucket = op.bucket();
    let default = op.default_tactic(&bucket);

    // 1. An explicit env override always wins, and short-circuits before any
    //    cache or lock work so setting the env var is also the cheapest path.
    if let Some(forced) = op.env_override() {
        return Resolution::new(forced, Source::EnvOverride);
    }

    if !mode.reads_cache() {
        return Resolution::new(default, Source::Default);
    }

    let key = TuneKey::new(
        op.op_name().to_string(),
        op.runner_id(),
        device_label(),
        bucket.clone(),
        op.dtype_tag(),
    );

    if let Ok(guard) = memo().lock()
        && let Some(hit) = guard.get(&key)
    {
        return hit.clone();
    }

    let resolution = resolve_uncached(op, store, cfg, mode, &key, &bucket, &default);
    if let Ok(mut guard) = memo().lock() {
        guard.insert(key, resolution.clone());
    }
    resolution
}

#[allow(clippy::too_many_arguments)]
fn resolve_uncached(
    op: &dyn TunableOp,
    store: &TacticStore,
    cfg: ProfileConfig,
    mode: Mode,
    key: &TuneKey,
    bucket: &ShapeBucket,
    default: &Tactic,
) -> Resolution {
    let candidates = op.candidates(bucket);
    if bucket.is_saturated() || candidates.is_empty() {
        tracing::warn!(
            "autotune: {} shape {bucket} is outside the tuned matrix ({}); using default {default}",
            op.op_name(),
            if candidates.is_empty() {
                "no candidates for this bucket"
            } else {
                "bucket dimension saturated"
            }
        );
        return Resolution::new(default.clone(), Source::OutOfBucket);
    }

    if let Some(record) = store.load(key) {
        // A cached tactic that is no longer in the candidate space (a device
        // budget shrank, an op narrowed its range) is discarded rather than
        // launched: the candidate list is the feasibility contract.
        if candidates.contains(&record.tactic) {
            tracing::debug!(
                "autotune: {} bucket {bucket} using cached {} ({:.1}us at tuning time)",
                op.op_name(),
                record.tactic,
                record.latency_us
            );
            return Resolution::new(record.tactic, Source::Cache);
        }
        tracing::warn!(
            "autotune: cached tactic {} for {} bucket {bucket} is no longer feasible; discarding it",
            record.tactic,
            op.op_name()
        );
        // Fall through: under `Mode::Tune` the sweep below re-tunes the bucket
        // and overwrites the unusable entry, and under `Mode::CacheOnly` it
        // resolves to the default just as a plain miss would.
    }

    if !mode.profiles() || !op.lazy_tunable() {
        return Resolution::new(default.clone(), Source::Default);
    }

    let Some(result) = profile(op, cfg) else {
        tracing::debug!(
            "autotune: {} bucket {bucket} produced no usable measurement; using default {default}",
            op.op_name()
        );
        return Resolution::new(default.clone(), Source::Default);
    };

    let record = TacticRecord::from_profile(key, &result);
    if let Err(e) = store.save(key, &record) {
        tracing::warn!(
            "autotune: could not persist tuned tactic for {}: {e}",
            key.display()
        );
    }
    Resolution::new(result.best, Source::Tuned)
}

/// Profile `op` and persist the winner, ignoring the process mode.
///
/// This is the offline `mlxcel tune` entry point: it always profiles, so it
/// works with `MLXCEL_AUTOTUNE` unset (the default). An explicit env override
/// still wins, because tuning a knob the user has pinned would write an entry
/// that can never take effect.
pub fn tune_and_store(
    op: &dyn TunableOp,
    store: &TacticStore,
    cfg: ProfileConfig,
) -> Option<(TuneKey, TacticRecord, ProfileResult)> {
    if let Some(forced) = op.env_override() {
        tracing::warn!(
            "autotune: skipping {} because an explicit env override ({forced}) is set; unset it to tune this op",
            op.op_name()
        );
        return None;
    }
    let bucket = op.bucket();
    let key = TuneKey::new(
        op.op_name().to_string(),
        op.runner_id(),
        device_label(),
        bucket.clone(),
        op.dtype_tag(),
    );
    let result = profile(op, cfg)?;
    let record = TacticRecord::from_profile(&key, &result);
    if let Err(e) = store.save(&key, &record) {
        tracing::warn!(
            "autotune: could not persist tuned tactic for {}: {e}",
            key.display()
        );
    }
    Some((key, record, result))
}

/// Drop the process-local memo. Used by tests that need a second `resolve`
/// call on the same key to actually re-read the store.
#[cfg(test)]
pub(crate) fn clear_memo() {
    if let Ok(mut guard) = memo().lock() {
        guard.clear();
    }
}

#[cfg(test)]
#[path = "autotune_tests.rs"]
mod autotune_tests;
