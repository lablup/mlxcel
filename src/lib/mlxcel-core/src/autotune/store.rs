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

//! Persistent tactic cache for the kernel autotuner (issue #906).
//!
//! Deliberately mirrors [`PolicyStore`] in `src/server/batch/mtp_policy.rs`
//! (the adaptive MTP hint store, issue #333): one JSON file per key hash under
//! the mlxcel cache root, atomic `write + rename` publication, and a load path
//! that validates a version field *and every key field* before accepting an
//! entry. Reusing that idiom rather than inventing a second cache shape means
//! the invalidation semantics are already understood in this tree.
//!
//! [`PolicyStore`]: https://github.com/lablup/mlxcel/blob/main/src/server/batch/mtp_policy.rs
//!
//! ## What invalidates an entry
//!
//! A stored tactic is only valid for the exact environment it was measured in,
//! so [`TacticStore::load`] rejects the entry (returning `None`, which makes
//! the caller re-tune or fall back to the default) when any of these differ:
//!
//! | Field | Why a mismatch invalidates |
//! |-------|----------------------------|
//! | `version` | The record schema changed; older bodies are not comparable. |
//! | `op` / `runner` / `device` / `bucket` / `dtype` | Hash collision, or a key the file was not written for. |
//! | `mlxcel_version` | mlxcel's own kernel launchers may have changed shape. |
//! | `mlx_commit` | The pinned MLX C++ commit changed, so the kernels themselves may differ. |
//!
//! An unreadable, truncated, or hand-corrupted file is never fatal: the load
//! warns once for that path and behaves exactly like a cache miss.
//!
//! ## What is persisted
//!
//! Only launch-shape configuration: the op identity, the bucketed shape, the
//! winning tactic's integer parameters, and the median latency and measured
//! spread behind the choice. No prompt data, no token ids, nothing
//! request-identifying.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::bucket::ShapeBucket;
use super::profile::ProfileResult;
use super::tactic::Tactic;

/// Subdirectory under the mlxcel cache root holding tuned tactics.
pub const TACTIC_SUBDIR: &str = "autotune";

/// Record schema version. Bump when the persisted body changes meaning; older
/// files are then ignored and the matrix re-tunes once.
///
/// v2 added the dispersion fields and, with them, a different selection rule
/// (the flaky-tactic guard). A v1 entry was chosen by a threshold that ignored
/// measurement spread, so it is not comparable and must not be reused.
pub const TACTIC_VERSION: u32 = 2;

/// mlxcel version the running binary was built at. Part of the invalidation
/// metadata: a different mlxcel may launch the tuned kernel differently.
#[must_use]
pub fn mlxcel_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Pinned MLX C++ commit the running binary was built against, exported by
/// `build.rs` as `MLXCEL_MLX_COMMIT`. Part of the invalidation metadata: a
/// different MLX pin can change the kernel a tactic was measured on.
#[must_use]
pub fn mlx_commit() -> &'static str {
    env!("MLXCEL_MLX_COMMIT")
}

/// Cache key: `(op, kernel/runner identity, device, shape bucket, dtype)`.
///
/// `runner` is the kernel/launcher identity, kept separate from `op` so two
/// implementations of the same logical op (for example the Metal and CUDA
/// paged-decode kernels) never share an entry. `dtype` is the "extras" slot
/// from the issue's key structure; ops that are dtype-invariant pass a fixed
/// tag rather than leaving the field out, so the key arity is constant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TuneKey {
    pub op: String,
    pub runner: String,
    pub device: String,
    pub bucket: ShapeBucket,
    pub dtype: String,
}

impl TuneKey {
    #[must_use]
    pub fn new(
        op: impl Into<String>,
        runner: impl Into<String>,
        device: impl Into<String>,
        bucket: ShapeBucket,
        dtype: impl Into<String>,
    ) -> Self {
        Self {
            op: op.into(),
            runner: runner.into(),
            device: device.into(),
            bucket,
            dtype: dtype.into(),
        }
    }

    /// Human-readable key (`op|runner|device|bucket|dtype`) for logs and the
    /// stored record body. Also the hash pre-image, so it must stay stable.
    #[must_use]
    pub fn display(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.op, self.runner, self.device, self.bucket, self.dtype
        )
    }

    /// Stable short hash used as the cache file stem.
    #[must_use]
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.display().as_bytes());
        let digest = hasher.finalize();
        // 16 hex chars (8 bytes) is ample for the handful of buckets a host
        // ever tunes, and keeps the filenames readable.
        digest[..8].iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// On-disk tuned tactic. Carries the readable key fields so a cache file is
/// self-describing (and so `load` can reject a hash collision), plus the
/// environment metadata that gates reuse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TacticRecord {
    pub version: u32,
    pub op: String,
    pub runner: String,
    pub device: String,
    /// [`ShapeBucket`] in its `Display` form, e.g. `"1x32x128x8192"`.
    pub bucket: String,
    pub dtype: String,
    /// mlxcel version at tuning time.
    pub mlxcel_version: String,
    /// Pinned MLX C++ commit at tuning time.
    pub mlx_commit: String,
    /// The winning tactic.
    pub tactic: Tactic,
    /// Median latency of the winning tactic, microseconds.
    pub latency_us: f64,
    /// Median latency of the op's default tactic in the same sweep, if the
    /// default was among the candidates. Lets a report state the win without
    /// re-running the sweep, and lets an integration test assert the tuned
    /// choice is no worse than the default.
    pub default_latency_us: Option<f64>,
    /// How many candidates were profiled to reach this choice.
    pub candidates: usize,
    /// Timed repetitions behind `latency_us` (median-of-N).
    pub reps: usize,
    /// Relative spread of the winning tactic's samples, e.g. `0.07` for a
    /// distribution 7% wide. Recorded so a human reading the cache (or the
    /// `mlxcel tune` table) can tell a tight measurement from a coin flip
    /// without re-running the sweep.
    #[serde(default)]
    pub spread: f64,
    /// Relative spread of the default's samples, when the default ran.
    #[serde(default)]
    pub default_spread: Option<f64>,
    /// Relative improvement the selection had to clear to be chosen: the larger
    /// of the sweep's `min_improvement` and the two measurements' combined
    /// spread. A large value means this row was measured on a noisy host.
    #[serde(default)]
    pub required_improvement: f64,
}

impl TacticRecord {
    /// Build a record for `key` from a winning tactic and its latency.
    ///
    /// Leaves the dispersion fields empty; [`Self::from_profile`] is the path
    /// production takes, and it fills them in from the sweep.
    #[must_use]
    pub fn new(
        key: &TuneKey,
        tactic: Tactic,
        latency_us: f64,
        default_latency_us: Option<f64>,
        candidates: usize,
        reps: usize,
    ) -> Self {
        Self {
            version: TACTIC_VERSION,
            op: key.op.clone(),
            runner: key.runner.clone(),
            device: key.device.clone(),
            bucket: key.bucket.to_string(),
            dtype: key.dtype.clone(),
            mlxcel_version: mlxcel_version().to_string(),
            mlx_commit: mlx_commit().to_string(),
            tactic,
            latency_us,
            default_latency_us,
            candidates,
            reps,
            spread: 0.0,
            default_spread: None,
            required_improvement: 0.0,
        }
    }

    /// Build a record for `key` from a completed profiling sweep, carrying the
    /// measured dispersion through so the entry records how trustworthy it is.
    #[must_use]
    pub fn from_profile(key: &TuneKey, result: &ProfileResult) -> Self {
        Self {
            spread: result.best_spread,
            default_spread: result.default_spread,
            required_improvement: result.required_improvement,
            ..Self::new(
                key,
                result.best.clone(),
                result.best_us,
                result.default_us,
                result.measurements.len(),
                result.reps,
            )
        }
    }

    /// Whether this record was written for `key` in the current environment.
    /// Every key field and both metadata fields must match; see the module
    /// docs for why each one invalidates.
    #[must_use]
    pub fn matches(&self, key: &TuneKey) -> bool {
        self.version == TACTIC_VERSION
            && self.op == key.op
            && self.runner == key.runner
            && self.device == key.device
            && self.bucket == key.bucket.to_string()
            && self.dtype == key.dtype
            && self.mlxcel_version == mlxcel_version()
            && self.mlx_commit == mlx_commit()
    }
}

/// Reads and writes per-key tactic records.
///
/// `dir == None` (no resolvable cache root) makes persistence a silent no-op:
/// tuning still works for the session, it just is not remembered.
#[derive(Debug, Clone)]
pub struct TacticStore {
    dir: Option<PathBuf>,
}

impl TacticStore {
    /// Resolve the store under the mlxcel cache root
    /// (`${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/autotune`).
    #[must_use]
    pub fn from_cache_root() -> Self {
        Self {
            dir: crate::cache_root().map(|root| root.join(TACTIC_SUBDIR)),
        }
    }

    /// Construct a store rooted at an explicit directory (test injection).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_dir(dir: Option<PathBuf>) -> Self {
        Self { dir }
    }

    /// The resolved directory, or `None` when persistence is disabled.
    #[must_use]
    pub fn dir(&self) -> Option<&PathBuf> {
        self.dir.as_ref()
    }

    fn record_file(&self, key: &TuneKey) -> Option<PathBuf> {
        self.dir
            .as_ref()
            .map(|dir| dir.join(format!("{}.json", key.hash())))
    }

    /// Load the stored record for `key`, or `None` when there is no usable one.
    ///
    /// Never fails and never panics. A missing file is a silent miss (the
    /// normal cold state). An unreadable or unparseable file warns and is
    /// treated as a miss, so a corrupted cache degrades to default behavior
    /// instead of taking the process down. A parseable record whose version,
    /// key fields, or environment metadata do not match is also a miss, which
    /// is what makes an mlxcel or MLX-pin bump re-tune rather than silently
    /// apply a stale config.
    #[must_use]
    pub fn load(&self, key: &TuneKey) -> Option<TacticRecord> {
        let path = self.record_file(key)?;
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => {
                tracing::warn!(
                    "autotune: ignoring unreadable tactic cache file {}: {e}",
                    path.display()
                );
                return None;
            }
        };
        let record: TacticRecord = match serde_json::from_str(&raw) {
            Ok(record) => record,
            Err(e) => {
                tracing::warn!(
                    "autotune: ignoring corrupt tactic cache file {} ({e}); will re-tune",
                    path.display()
                );
                return None;
            }
        };
        if !record.matches(key) {
            tracing::debug!(
                "autotune: discarding stale tactic cache entry {} for key {}",
                path.display(),
                key.display()
            );
            return None;
        }
        Some(record)
    }

    /// Persist a record for `key`. Best-effort: creates the directory, writes
    /// a temporary file, and renames it into place (atomic on the same
    /// volume). One file per key, so concurrent writers for different buckets
    /// never contend. Returns the IO error for the caller to log; never
    /// panics. On rename failure the temporary file is removed so no orphaned
    /// `.tmp.<pid>` files accumulate.
    pub fn save(&self, key: &TuneKey, record: &TacticRecord) -> std::io::Result<()> {
        let Some(dir) = self.dir.clone() else {
            return Ok(());
        };
        let Some(path) = self.record_file(key) else {
            return Ok(());
        };
        std::fs::create_dir_all(&dir)?;
        let body = serde_json::to_string_pretty(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = dir.join(format!("{}.json.tmp.{}", key.hash(), std::process::id()));
        std::fs::write(&tmp, body.as_bytes())?;
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }
}
