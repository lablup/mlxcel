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

//! Unit tests for [`SpeculativeDispatch::resolve`].
//!
//! These tests pin the dispatch matrix without touching the MLX FFI:
//! every test builds a minimal `ServerConfig` (defaults + selective
//! overrides) and a tiny on-disk drafter `config.json` fixture, then
//! asserts the resolved [`SpeculativeDispatch`] variant + its
//! kind/block_size fields.
//!
//! The mock drafter directory only needs `config.json::model_type`; the
//! drafter weight loading happens later inside the scheduler and is
//! covered by separate integration tests.

use super::*;
use crate::server::config::ServerConfig;
use std::path::PathBuf;
use tempfile::TempDir;

/// Build a minimal `ServerConfig` for testing. Uses `Default::default`
/// for everything except the speculative fields, which are the only
/// thing [`SpeculativeDispatch::resolve`] looks at.
fn base_config() -> ServerConfig {
    ServerConfig::default()
}

/// Write a tiny `config.json` with the given `model_type` into a fresh
/// temp dir. Returns the temp dir handle (so the caller can keep it
/// alive for the test scope) and the directory path.
fn write_drafter_config(model_type: Option<&str>) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let content = match model_type {
        Some(mt) => format!(r#"{{"model_type": "{mt}"}}"#),
        None => "{}".to_string(),
    };
    std::fs::write(dir.path().join("config.json"), content).expect("write config.json");
    let path = dir.path().to_path_buf();
    (dir, path)
}

#[test]
fn resolve_returns_disabled_when_no_drafter_configured() {
    let cfg = base_config();
    assert!(cfg.draft_model_path.is_none());

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    assert!(matches!(dispatch, SpeculativeDispatch::Disabled));
    assert!(dispatch.drafter_kind().is_none());
    assert!(dispatch.block_size().is_none());
    assert!(dispatch.draft_model_path().is_none());
    assert!(!dispatch.is_kind_specific());
    assert_eq!(dispatch.summary(), "speculative=off");
}

#[test]
fn resolve_mtp_explicit_kind() {
    // Drafter config sets `model_type=gemma4_assistant` which auto-detects
    // to MTP. Explicit `--draft-kind mtp` matches; the resolved dispatch
    // is `Mtp` with `user_requested_explicit_kind=true`.
    let (_dir, path) = write_drafter_config(Some("gemma4_assistant"));
    let mut cfg = base_config();
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
            assert!(user_requested_explicit_kind);
            // Default MTP block size is 4 (per default_block_size_for_kind).
            assert_eq!(block_size, 4);
        }
        other => panic!("expected Mtp, got {other:?}"),
    }
}

#[test]
fn resolve_mtp_explicit_kind_with_block_size_override() {
    let (_dir, path) = write_drafter_config(Some("gemma4_assistant"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("mtp".to_string());
    cfg.draft_block_size = Some(8);

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    match dispatch {
        SpeculativeDispatch::Mtp { block_size, .. } => assert_eq!(block_size, 8),
        other => panic!("expected Mtp, got {other:?}"),
    }
}

#[test]
fn resolve_mtp_auto_detected_from_drafter_config() {
    // No explicit --draft-kind; the drafter's `model_type=gemma4_assistant`
    // auto-detects to MTP. The dispatch is still `Mtp` (so the scheduler
    // takes the kind-specific path), but with
    // `user_requested_explicit_kind=false` for clearer error messages.
    let (_dir, path) = write_drafter_config(Some("gemma4_assistant"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = None;

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
        other => panic!("expected Mtp, got {other:?}"),
    }
}

#[test]
fn resolve_dflash_explicit_kind() {
    // Any model_type that isn't `gemma4_assistant` / `internal_mtp`
    // auto-detects to DFlash. Use the canonical Qwen3.5 DFlash signature.
    let (_dir, path) = write_drafter_config(Some("dflash"));
    let mut cfg = base_config();
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
            assert!(user_requested_explicit_kind);
            // Default DFlash block size is 16.
            assert_eq!(block_size, 16);
        }
        other => panic!("expected DFlash, got {other:?}"),
    }
}

#[test]
fn resolve_dflash_block_size_override() {
    let (_dir, path) = write_drafter_config(Some("dflash"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("dflash".to_string());
    cfg.draft_block_size = Some(32);

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    match dispatch {
        SpeculativeDispatch::DFlash { block_size, .. } => assert_eq!(block_size, 32),
        other => panic!("expected DFlash, got {other:?}"),
    }
}

#[test]
fn resolve_internal_mtp_kind_is_rejected_at_cli() {
    // The `internal-mtp` variant is auto-detected only — it cannot be
    // passed via `--draft-kind` from the CLI.
    let (_dir, path) = write_drafter_config(Some("dflash"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("internal-mtp".to_string());

    let err = SpeculativeDispatch::resolve(&cfg).expect_err("must reject internal-mtp");
    match err {
        SpeculativeDispatchError::InvalidKind { message } => {
            assert!(message.contains("internal-mtp"));
            assert!(message.contains("not user-selectable"));
        }
        other => panic!("expected InvalidKind, got {other:?}"),
    }
}

#[test]
fn resolve_unknown_kind_returns_invalid_kind_error() {
    let (_dir, path) = write_drafter_config(Some("dflash"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("bogus-kind".to_string());

    let err = SpeculativeDispatch::resolve(&cfg).expect_err("must reject unknown");
    match err {
        SpeculativeDispatchError::InvalidKind { message } => {
            assert!(message.contains("bogus-kind"));
            assert!(message.contains("accepted values"));
            assert!(message.contains("dflash"));
            assert!(message.contains("mtp"));
        }
        other => panic!("expected InvalidKind, got {other:?}"),
    }
}

#[test]
fn resolve_with_missing_drafter_config_falls_back_to_default_kind() {
    // Drafter directory has no `config.json`. The upstream
    // `resolve_drafter_kind` falls back to the default (DFlash) with a
    // log line rather than erroring — this preserves the historical
    // `--draft-model <path>` workflow for older drafter checkpoints
    // missing a `model_type` field. The dispatch reflects that with a
    // `DFlash` variant.
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().to_path_buf();

    let mut cfg = base_config();
    cfg.draft_model_path = Some(path.clone());
    // Drop the `_dir` guard at end of scope — we intentionally don't
    // hold it across `resolve` because the resolve only does a path
    // scan; the file is read inside.
    let _keep_dir_alive = dir;

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("must resolve via default fallback");
    match dispatch {
        SpeculativeDispatch::DFlash {
            draft_model_path,
            user_requested_explicit_kind,
            block_size,
        } => {
            assert_eq!(draft_model_path, path);
            assert!(
                !user_requested_explicit_kind,
                "fallback path must not flip the explicit-kind flag"
            );
            assert_eq!(block_size, 16);
        }
        other => panic!("expected DFlash (default fallback), got {other:?}"),
    }
}

#[test]
fn summary_contains_block_size_for_kind_specific_variants() {
    let (_dir, path) = write_drafter_config(Some("dflash"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("dflash".to_string());

    let dispatch = SpeculativeDispatch::resolve(&cfg).expect("resolve");
    let s = dispatch.summary();
    assert!(s.contains("speculative=dflash"));
    assert!(s.contains("block_size=16"));
    assert!(s.contains("drafter="));
}

#[test]
fn block_size_accessor_is_none_for_disabled_and_classic() {
    let dispatch = SpeculativeDispatch::Disabled;
    assert!(dispatch.block_size().is_none());

    let classic = SpeculativeDispatch::Classic {
        draft_model_path: PathBuf::from("/tmp/drafter"),
        num_draft_tokens: 3,
        auto_detected_kind: mlxcel_core::drafter::DrafterKind::InternalMtp,
    };
    assert!(classic.block_size().is_none());
    assert!(!classic.is_kind_specific());
    assert_eq!(
        classic.drafter_kind(),
        Some(mlxcel_core::drafter::DrafterKind::InternalMtp)
    );
}

#[test]
fn is_kind_specific_is_true_only_for_mtp_and_dflash() {
    let mtp = SpeculativeDispatch::Mtp {
        draft_model_path: PathBuf::from("/tmp/m"),
        block_size: 4,
        user_requested_explicit_kind: true,
    };
    let dflash = SpeculativeDispatch::DFlash {
        draft_model_path: PathBuf::from("/tmp/d"),
        block_size: 16,
        user_requested_explicit_kind: false,
    };
    assert!(mtp.is_kind_specific());
    assert!(dflash.is_kind_specific());
    assert!(!SpeculativeDispatch::Disabled.is_kind_specific());
}
