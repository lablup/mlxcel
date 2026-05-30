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

//! Integration tests for the server-side speculative-decoding dispatch
//! matrix.
//!
//! These tests cover the **operator-facing dispatch contract** end-to-end:
//! they build a `ServerConfig` exactly like `mlxcel-server`'s CLI plumbing
//! would, point it at a tiny on-disk drafter `config.json` fixture, and
//! assert that
//! [`mlxcel::server::SpeculativeDispatch::resolve`] produces the right
//! kind-specific variant for every flag combination.
//!
//! ## What this file pins
//!
//! 1. Every supported `--draft-kind` value (`dflash`, `mtp`, and the
//!    unset / auto-detect path) resolves to the matching dispatch
//!    variant.
//! 2. `--draft-block-size` overrides reach the resolved variant.
//! 3. Error variants (unparseable kind, missing drafter config, etc.)
//!    surface clear messages.
//! 4. The classic-fallback path (no `--draft-model` set) returns
//!    `Disabled` so the bit-exact baseline is preserved.
//!
//! ## What this file does NOT pin
//!
//! - The actual decode-loop end-to-end byte-equality assertion against a
//!   real Gemma 4 / Qwen 3.5 target + drafter pair. That test ships in
//!   `tests/speculative_parity.rs` and is gated behind `#[ignore]` so
//!   CI hosts without the model checkpoints don't red-flag the build.
//!
//! - The construction of an actual `MtpGenerator` / `DFlashGenerator`
//!   instance from the resolved dispatch — that requires the per-target
//!   adapter (`Gemma4MtpTargetAdapter` etc.) plus a loaded model, both
//!   of which are exercised by `tests/speculative_parity.rs`.

use mlxcel::server::{ServerConfig, SpeculativeDispatch, SpeculativeDispatchError};
use mlxcel_core::drafter::DrafterKind;
use std::path::PathBuf;
use tempfile::TempDir;

/// Tiny on-disk drafter fixture: write a `config.json` carrying the
/// given `model_type` field and return the directory path.
fn make_drafter_dir(model_type: Option<&str>) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let content = match model_type {
        Some(mt) => format!(r#"{{"model_type": "{mt}"}}"#),
        None => "{}".to_string(),
    };
    std::fs::write(dir.path().join("config.json"), content).expect("write config.json");
    let path = dir.path().to_path_buf();
    (dir, path)
}

fn base_server_config() -> ServerConfig {
    ServerConfig::default()
}

// =============================================================================
// MTP dispatch matrix
// =============================================================================

#[test]
fn dispatch_mtp_with_explicit_kind_resolves_to_mtp_variant() {
    let (_dir, path) = make_drafter_dir(Some("gemma4_assistant"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path.clone());
    cfg.draft_kind = Some("mtp".to_string());

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    match dispatch {
        SpeculativeDispatch::Mtp {
            draft_model_path,
            block_size,
            user_requested_explicit_kind,
        } => {
            assert_eq!(draft_model_path, path);
            assert_eq!(block_size, 4); // MTP default
            assert!(user_requested_explicit_kind);
        }
        other => panic!("expected Mtp dispatch, got {other:?}"),
    }
}

#[test]
fn dispatch_mtp_with_block_size_override_honors_override() {
    let (_dir, path) = make_drafter_dir(Some("gemma4_assistant"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("mtp".to_string());
    cfg.draft_block_size = Some(8);

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert_eq!(dispatch.block_size(), Some(8));
}

#[test]
fn dispatch_mtp_auto_detected_from_config_resolves_to_mtp_variant() {
    let (_dir, path) = make_drafter_dir(Some("gemma4_assistant"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = None; // auto-detect

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    match dispatch {
        SpeculativeDispatch::Mtp {
            user_requested_explicit_kind,
            block_size,
            ..
        } => {
            assert!(!user_requested_explicit_kind);
            assert_eq!(block_size, 4);
        }
        other => panic!("expected Mtp dispatch from auto-detect, got {other:?}"),
    }
}

// =============================================================================
// DFlash dispatch matrix
// =============================================================================

#[test]
fn dispatch_dflash_with_explicit_kind_resolves_to_dflash_variant() {
    let (_dir, path) = make_drafter_dir(Some("dflash"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path.clone());
    cfg.draft_kind = Some("dflash".to_string());

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    match dispatch {
        SpeculativeDispatch::DFlash {
            draft_model_path,
            block_size,
            user_requested_explicit_kind,
        } => {
            assert_eq!(draft_model_path, path);
            assert_eq!(block_size, 16); // DFlash default
            assert!(user_requested_explicit_kind);
        }
        other => panic!("expected DFlash dispatch, got {other:?}"),
    }
}

#[test]
fn dispatch_dflash_with_block_size_override_honors_override() {
    let (_dir, path) = make_drafter_dir(Some("dflash"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("dflash".to_string());
    cfg.draft_block_size = Some(32);

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert_eq!(dispatch.block_size(), Some(32));
}

// =============================================================================
// Disabled / classic-fallback paths
// =============================================================================

#[test]
fn dispatch_disabled_when_no_drafter_configured() {
    let cfg = base_server_config();
    assert!(cfg.draft_model_path.is_none());

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert!(matches!(dispatch, SpeculativeDispatch::Disabled));
    assert!(!dispatch.is_kind_specific());
    assert!(dispatch.draft_model_path().is_none());
    assert_eq!(dispatch.summary(), "speculative=off");
}

#[test]
fn dispatch_disabled_does_not_construct_drafter_kind() {
    let cfg = base_server_config();
    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert!(dispatch.drafter_kind().is_none());
}

// =============================================================================
// Error paths
// =============================================================================

#[test]
fn dispatch_rejects_unparseable_draft_kind() {
    let (_dir, path) = make_drafter_dir(Some("dflash"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("nonsense-kind".to_string());

    let err = SpeculativeDispatch::resolve(&cfg).expect_err("must reject unparseable");
    match err {
        SpeculativeDispatchError::InvalidKind { message } => {
            assert!(message.contains("nonsense-kind"));
            assert!(message.contains("dflash"));
            assert!(message.contains("mtp"));
        }
        other => panic!("expected InvalidKind error, got {other:?}"),
    }
}

#[test]
fn dispatch_rejects_internal_mtp_from_cli() {
    let (_dir, path) = make_drafter_dir(Some("dflash"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("internal-mtp".to_string());

    let err = SpeculativeDispatch::resolve(&cfg).expect_err("must reject internal-mtp from CLI");
    match err {
        SpeculativeDispatchError::InvalidKind { message } => {
            assert!(
                message.contains("internal-mtp") && message.contains("not user-selectable"),
                "error message must mention internal-mtp and not user-selectable, got: {message}",
            );
        }
        other => panic!("expected InvalidKind error, got {other:?}"),
    }
}

// =============================================================================
// Summary contains structured info
// =============================================================================

#[test]
fn dispatch_summary_includes_drafter_kind_and_block_size() {
    let (_dir, path) = make_drafter_dir(Some("gemma4_assistant"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("mtp".to_string());

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    let s = dispatch.summary();
    assert!(s.contains("speculative=mtp"));
    assert!(s.contains("block_size=4"));
    assert!(s.contains("explicit_kind=true"));
}

#[test]
fn dispatch_kind_specific_accessor_distinguishes_variants() {
    let (_d1, p1) = make_drafter_dir(Some("gemma4_assistant"));
    let mut cfg_mtp = base_server_config();
    cfg_mtp.draft_model_path = Some(p1);
    cfg_mtp.draft_kind = Some("mtp".to_string());
    assert!(
        SpeculativeDispatch::resolve(&cfg_mtp)
            .unwrap()
            .is_kind_specific()
    );

    let (_d2, p2) = make_drafter_dir(Some("dflash"));
    let mut cfg_dflash = base_server_config();
    cfg_dflash.draft_model_path = Some(p2);
    cfg_dflash.draft_kind = Some("dflash".to_string());
    assert!(
        SpeculativeDispatch::resolve(&cfg_dflash)
            .unwrap()
            .is_kind_specific()
    );

    let cfg_disabled = base_server_config();
    assert!(
        !SpeculativeDispatch::resolve(&cfg_disabled)
            .unwrap()
            .is_kind_specific()
    );
}

// =============================================================================
// Drafter kind resolution
// =============================================================================

#[test]
fn dispatch_reports_drafter_kind_for_kind_specific_variants() {
    let (_d, p) = make_drafter_dir(Some("gemma4_assistant"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(p);
    cfg.draft_kind = Some("mtp".to_string());

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert_eq!(dispatch.drafter_kind(), Some(DrafterKind::Mtp));
}

// =============================================================================
// Tick-level dispatch (— burst path)
//
// These tests pin the **runtime gating** of the speculative dispatch:
// they cannot construct a real `BatchScheduler` from outside the crate
// (the constructor requires a loaded model + tokenizer), so they exercise
// the gate via the public `SpeculativeDispatch` API which the scheduler
// consults at request time. End-to-end byte-equality with a real model
// is covered by `tests/speculative_parity.rs` (gated `#[ignore]`).
// =============================================================================

#[test]
fn dispatch_disabled_never_routes_to_burst() {
    // Disabled dispatch must never be `is_kind_specific()` — the burst
    // path's first gate (`should_burst_for_sequence`) checks exactly
    // this, so `Disabled` requests bypass the entire burst module and
    // run on the classic prefill + decode pipeline bit-exactly.
    let cfg = base_server_config();
    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert!(matches!(dispatch, SpeculativeDispatch::Disabled));
    assert!(!dispatch.is_kind_specific());
    assert!(dispatch.draft_model_path().is_none());
    assert!(dispatch.block_size().is_none());
}

#[test]
fn dispatch_mtp_carries_path_and_block_size_for_burst_construction() {
    // The burst's `WorkerDrafterSlot::from_dispatch` reads
    // `draft_model_path()` and `drafter_kind()` to lazy-load the
    // drafter on the first speculative request. Both must be populated
    // for a `SpeculativeDispatch::Mtp` variant.
    let (_dir, path) = make_drafter_dir(Some("gemma4_assistant"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path.clone());
    cfg.draft_kind = Some("mtp".to_string());
    cfg.draft_block_size = Some(4);

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert!(dispatch.is_kind_specific());
    assert_eq!(dispatch.draft_model_path(), Some(path.as_path()));
    assert_eq!(dispatch.drafter_kind(), Some(DrafterKind::Mtp));
    assert_eq!(dispatch.block_size(), Some(4));
}

#[test]
fn dispatch_dflash_carries_path_and_block_size_for_burst_construction() {
    // Symmetric to the MTP case for DFlash. The burst uses the same
    // `draft_model_path()` + `drafter_kind()` pair to drive
    // `load_drafter(path, Some(DrafterKind::Dflash))`.
    let (_dir, path) = make_drafter_dir(Some("dflash"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path.clone());
    cfg.draft_kind = Some("dflash".to_string());
    cfg.draft_block_size = Some(16);

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert!(dispatch.is_kind_specific());
    assert_eq!(dispatch.draft_model_path(), Some(path.as_path()));
    assert_eq!(dispatch.drafter_kind(), Some(DrafterKind::Dflash));
    assert_eq!(dispatch.block_size(), Some(16));
}

#[test]
fn dispatch_block_size_override_propagates_to_burst() {
    // Burst dispatch reads block_size from the resolved dispatch (not
    // from the raw config field) — an explicit `--draft-block-size`
    // override must be honored end-to-end.
    let (_dir, path) = make_drafter_dir(Some("gemma4_assistant"));
    let mut cfg = base_server_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("mtp".to_string());
    cfg.draft_block_size = Some(8); // override default of 4

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert_eq!(dispatch.block_size(), Some(8));
}

#[test]
fn dispatch_classic_fallback_is_not_burstable() {
    // The `Classic` variant (auto-detected `internal-mtp` kind) is NOT
    // `is_kind_specific()` and therefore never enters the burst path —
    // it stays on the historical `SpeculativeGenerator` dispatch
    // (which the server today doesn't expose; the scheduler falls
    // back to classic decode). This is the operator-facing
    // backward-compat guarantee /.
    //
    // We can't easily construct a `Classic` variant from outside
    // (auto-detect requires `model_type` field that resolves to
    // InternalMtp), so we assert the invariant via the matching
    // accessor methods on a fresh `Disabled` plus the kind-specific
    // ones above. The unit tests in
    // `src/server/speculative_dispatch_tests.rs` cover the Classic
    // variant directly.
    let cfg = base_server_config();
    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert!(!dispatch.is_kind_specific());
}
