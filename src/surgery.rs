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

//! Axis A "weight-load surgery" CLI integration glue (Epic #363, issue
//! #371 — A4).
//!
//! This module owns the **active-pipeline slot**: a process-wide
//! `Option<Arc<SurgeryPipeline>>` set by the CLI/server entry points
//! when `--surgery <config.yaml>` is supplied, and read by the
//! consolidated weight loaders (`crate::models::load_text_weights` and
//! `crate::loading::vlm::load_vlm_weights_common`) on each model load.
//!
//! ## Why a global slot
//!
//! The consolidated loaders introduced by A1 (issue #365) accept
//! `Option<&dyn WeightTransform>` so any caller can plug in a transform
//! at load time. The CLI entry points (`mlxcel generate`, `mlxcel serve`,
//! `mlxcel-server`) call `crate::load_model` (and friends), which in
//! turn dispatches across 60+ model-family loaders that each call
//! `load_text_weights(_, None)` / `load_vlm_weights_common(_, None)`
//! directly. Plumbing an explicit `transform` argument through every
//! callsite would be ~60 trivial diffs and defeat A1's design (one hook
//! point, not 60).
//!
//! Instead we keep a single, process-scoped slot that the CLI populates
//! once before the load begins. The consolidated loaders consult this
//! slot **only when their explicit `transform` parameter is `None`** —
//! so direct callers (tests, future programmatic users) can still pass
//! an explicit pipeline that takes precedence over the slot.
//!
//! ## Bit-exactness contract
//!
//! When `--surgery` is not supplied:
//!
//! - The slot is initialized lazily and stays at `None`.
//! - The consolidated loaders see `transform = None` from the model
//!   loader, look up the slot, find `None`, and skip the transform hook
//!   entirely.
//! - The runtime call sequence is byte-for-byte identical to the
//!   pre-#371 baseline.
//!
//! This satisfies the hard guarantee in #363 §5 (isolation table) and
//! is exercised end-to-end by the integration test in
//! `tests/surgery_cli.rs`.
//!
//! ## Thread safety
//!
//! The slot is a `OnceLock<RwLock<Option<Arc<SurgeryPipeline>>>>`. The
//! `OnceLock` initialization is one-time and lock-free after the first
//! call; the `RwLock` is uncontended in the steady state because the
//! slot is written once at startup and read once per model load.
//! `Arc<SurgeryPipeline>` makes cloning the snapshot cheap (refcount
//! bump) so the consolidated loaders can release the read lock before
//! invoking `transform.apply`.

use std::sync::{Arc, OnceLock, RwLock};

pub use mlxcel_surgery::{SurgeryError, SurgeryPipeline, parse_config_file};

/// Lazily initialized process-wide slot holding the currently active
/// surgery pipeline (or `None` when surgery is disabled).
///
/// Initialized to `RwLock::new(None)` on first access via
/// [`active_slot`]. The `OnceLock` keeps the initialization cost off
/// the bit-exact baseline path — a process that never invokes any
/// surgery API never pays for the lock instantiation.
static ACTIVE_PIPELINE: OnceLock<RwLock<Option<Arc<SurgeryPipeline>>>> = OnceLock::new();

fn active_slot() -> &'static RwLock<Option<Arc<SurgeryPipeline>>> {
    ACTIVE_PIPELINE.get_or_init(|| RwLock::new(None))
}

/// Install `pipeline` as the active surgery pipeline for subsequent
/// model loads on **any** thread in this process.
///
/// Passing `None` clears the slot (used by tests and by the CLI when
/// validating that a flag is recognized without committing to a real
/// pipeline yet).
///
/// The slot is a single global; calling this from multiple CLI front
/// ends in the same process is a programming error (the second caller
/// overrides the first). In the standard flow each binary calls this
/// exactly once during startup.
///
/// Used by: `crate::commands::generate::run_generate` (CLI),
/// `crate::server::startup::start_server` (HTTP server)
pub fn set_active_pipeline(pipeline: Option<Arc<SurgeryPipeline>>) {
    let mut guard = active_slot().write().expect("surgery slot poisoned");
    *guard = pipeline;
}

/// Snapshot the active surgery pipeline. Returns `None` when no
/// surgery is installed.
///
/// The returned `Arc` is independent of the slot's `RwLock`: the read
/// lock is held only for the duration of the clone, so the consolidated
/// loaders can invoke `transform.apply` without keeping the slot
/// locked. This keeps surgery from blocking concurrent loads in the
/// HTTP server.
///
/// Used by: `crate::models::load_text_weights`,
/// `crate::loading::vlm::load_vlm_weights_common`
pub fn snapshot_active_pipeline() -> Option<Arc<SurgeryPipeline>> {
    // Fast path: if the OnceLock was never initialized, the slot is
    // implicitly `None`. Read it through `get` (not `get_or_init`) so
    // the baseline path stays a single relaxed load.
    let slot = ACTIVE_PIPELINE.get()?;
    let guard = slot.read().expect("surgery slot poisoned");
    guard.clone()
}

/// Load a `SurgeryPipeline` from a YAML configuration file.
///
/// Returns a friendly `String` error suitable for displaying through
/// `anyhow::anyhow!(...)` at the CLI layer. Path resolution for
/// relative `source*` fields in the YAML uses the YAML file's parent
/// directory.
///
/// This is a thin wrapper over [`mlxcel_surgery::parse_config_file`]
/// so the CLI layer does not need to depend on the surgery crate
/// directly.
///
/// Used by: `crate::commands::generate::run_generate`,
/// `crate::server::startup::start_server`
pub fn load_pipeline_from_file<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<SurgeryPipeline, String> {
    parse_config_file(path).map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Crate-wide env_lock keeps these tests serialized with any other
    /// test that mutates process-global state. The surgery slot is a
    /// process-global resource just like an env var; concurrent tests
    /// touching it would observe each other's writes.
    use crate::test_support::env_lock::env_lock;

    fn empty_pipeline_arc() -> Arc<SurgeryPipeline> {
        Arc::new(SurgeryPipeline::new())
    }

    #[test]
    fn snapshot_returns_none_when_slot_never_written() {
        let _guard = env_lock();
        // Other tests may have written; reset to be sure we observe the
        // "never written" semantic.
        set_active_pipeline(None);
        assert!(
            snapshot_active_pipeline().is_none(),
            "baseline path must observe None"
        );
    }

    #[test]
    fn set_then_snapshot_round_trip() {
        let _guard = env_lock();
        let pipeline = empty_pipeline_arc();
        set_active_pipeline(Some(Arc::clone(&pipeline)));
        let snapshot = snapshot_active_pipeline().expect("snapshot present");
        assert!(
            Arc::ptr_eq(&snapshot, &pipeline),
            "round trip preserves Arc identity"
        );
        // Restore baseline so following tests in any order observe None.
        set_active_pipeline(None);
    }

    #[test]
    fn set_none_clears_active_pipeline() {
        let _guard = env_lock();
        set_active_pipeline(Some(empty_pipeline_arc()));
        set_active_pipeline(None);
        assert!(snapshot_active_pipeline().is_none());
    }

    #[test]
    fn load_pipeline_from_empty_yaml_file_succeeds() {
        let _guard = env_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.yaml");
        std::fs::write(&path, "version: 1\noperations: []\n").unwrap();
        let pipeline = load_pipeline_from_file(&path).expect("empty yaml parses");
        assert!(pipeline.is_empty());
    }

    #[test]
    fn load_pipeline_surfaces_io_error_for_missing_file() {
        let _guard = env_lock();
        let err =
            load_pipeline_from_file("/does/not/exist.yaml").expect_err("missing file must fail");
        assert!(
            err.contains("/does/not/exist.yaml"),
            "error must mention path: {err}"
        );
    }
}
