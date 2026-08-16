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

//! Unit tests for the `SpeculativeArgs` clap group and helpers.
//!
//! Env-var tests acquire [`crate::test_support::env_lock`] before any
//! `set_var` / `remove_var` call. The process environment is global, and
//! cargo's default test runner runs every `#[test]` in this binary across
//! a thread pool — see the module-level docs on `env_lock` for the full
//! rationale.

use super::{
    DEFAULT_DFLASH_BLOCK_SIZE, DEFAULT_MTP_BLOCK_SIZE, SpeculativeArgs,
    default_block_size_for_kind, env_fallback_draft_block_size, env_fallback_draft_kind,
    resolve_draft_block_size, user_selectable_kinds,
};
use crate::test_support::env_lock::env_lock;
use mlxcel_core::drafter::DrafterKind;

#[test]
fn parse_kind_accepts_dflash_and_mtp() {
    let args = SpeculativeArgs {
        draft_kind: Some("dflash".to_string()),
        draft_block_size: None,
    };
    assert_eq!(args.parse_kind().unwrap(), Some(DrafterKind::Dflash));

    let args = SpeculativeArgs {
        draft_kind: Some("mtp".to_string()),
        draft_block_size: None,
    };
    assert_eq!(args.parse_kind().unwrap(), Some(DrafterKind::Mtp));
}

#[test]
fn parse_kind_returns_none_when_unset() {
    let args = SpeculativeArgs::default();
    assert!(args.parse_kind().unwrap().is_none());
}

#[test]
fn parse_kind_rejects_internal_mtp_with_helpful_error() {
    // internal-mtp is intentionally NOT user-selectable on the CLI; it is
    // auto-detected from the target checkpoint. The error
    // message must explain this so an operator typing the flag gets a
    // hint instead of "unknown value".
    let args = SpeculativeArgs {
        draft_kind: Some("internal-mtp".to_string()),
        draft_block_size: None,
    };
    let err = args.parse_kind().expect_err("must reject internal-mtp");
    let msg = err.to_string();
    assert!(
        msg.contains("not user-selectable"),
        "error message should explain why: {msg}"
    );
}

#[test]
fn parse_kind_rejects_unknown_with_known_values_in_error() {
    let args = SpeculativeArgs {
        draft_kind: Some("bogus".to_string()),
        draft_block_size: None,
    };
    let err = args.parse_kind().expect_err("must reject bogus");
    let msg = err.to_string();
    assert!(msg.contains("dflash"), "should list dflash: {msg}");
    assert!(msg.contains("mtp"), "should list mtp: {msg}");
}

#[test]
fn user_selectable_kinds_excludes_internal_mtp() {
    let kinds = user_selectable_kinds();
    assert!(kinds.contains(&"dflash"));
    assert!(kinds.contains(&"mtp"));
    assert!(
        !kinds.contains(&"internal-mtp"),
        "internal-mtp should not be user-selectable"
    );
    assert_eq!(kinds.len(), 2);
}

#[test]
fn default_block_size_for_mtp_is_4() {
    assert_eq!(default_block_size_for_kind(DrafterKind::Mtp), 4);
    assert_eq!(DEFAULT_MTP_BLOCK_SIZE, 4);
}

#[test]
fn default_block_size_for_dflash_is_16() {
    // Matches https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_dflash/config.py#L31
    // and DEFAULT_BLOCK_SIZE in mlxcel_core::drafter::dflash::round_loop.
    assert_eq!(default_block_size_for_kind(DrafterKind::Dflash), 16);
    assert_eq!(DEFAULT_DFLASH_BLOCK_SIZE, 16);
}

#[test]
fn default_block_size_for_internal_mtp_shares_dflash_default() {
    // InternalMtp's CLI surface lands in for now we share the
    // DFlash 16-token default to keep the helper exhaustive.
    assert_eq!(
        default_block_size_for_kind(DrafterKind::InternalMtp),
        DEFAULT_DFLASH_BLOCK_SIZE
    );
}

#[test]
fn resolve_draft_block_size_uses_override_when_provided() {
    // An explicit override always wins, even against a drafter path whose
    // own config declares a different value.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3_5_mtp", "block_size": 3}"#,
    )
    .expect("write config.json");
    assert_eq!(
        resolve_draft_block_size(Some(8), DrafterKind::Mtp, dir.path()),
        8
    );
    assert_eq!(
        resolve_draft_block_size(Some(32), DrafterKind::Dflash, dir.path()),
        32
    );
}

#[test]
fn resolve_draft_block_size_falls_back_to_per_kind_default_without_a_drafter_hint() {
    // No config.json at all (nonexistent path). DFlash never peeks, and
    // an MTP drafter with no discoverable config falls back to the flat
    // Gemma-4-derived constant.
    let missing = std::path::Path::new("/nonexistent/mlxcel-drafter-test-path");
    assert_eq!(resolve_draft_block_size(None, DrafterKind::Mtp, missing), 4);
    assert_eq!(
        resolve_draft_block_size(None, DrafterKind::Dflash, missing),
        16
    );
}

#[test]
fn resolve_draft_block_size_honors_the_qwen35_mtp_drafters_own_configured_value() {
    // Issue #1165 hardening: a Qwen 3.5 MTP drafter's own `block_size`
    // (the published checkpoint declares 3) must win over the flat
    // Gemma-4-derived MTP constant (4) when the operator passes no
    // explicit `--draft-block-size` override.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3_5_mtp", "block_size": 3}"#,
    )
    .expect("write config.json");
    assert_eq!(
        resolve_draft_block_size(None, DrafterKind::Mtp, dir.path()),
        3
    );
}

#[test]
fn resolve_draft_block_size_ignores_block_size_from_a_non_qwen35_mtp_drafter() {
    // A Gemma 4 assistant drafter (or any non-`qwen3_5_mtp` model_type)
    // must not have a same-named `block_size` field misread as the hint;
    // the flat per-kind default still applies.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "gemma4_assistant", "block_size": 7}"#,
    )
    .expect("write config.json");
    assert_eq!(
        resolve_draft_block_size(None, DrafterKind::Mtp, dir.path()),
        4
    );
}

#[test]
fn env_fallback_draft_kind_keeps_cli_when_set() {
    // Acquire ENV_LOCK before any env mutation. Drop the guard at the
    // end of the test (after `remove_var`) so the next test only runs
    // against a clean env. SAFETY for set_var/remove_var: the lock
    // serializes all env-mutation tests in this crate's test binary.
    let _guard = env_lock();
    unsafe {
        std::env::set_var("MLXCEL_DRAFT_KIND", "dflash");
    }
    let mut value = Some("mtp".to_string());
    env_fallback_draft_kind(&mut value);
    assert_eq!(value, Some("mtp".to_string()), "CLI must win on conflict");
    unsafe {
        std::env::remove_var("MLXCEL_DRAFT_KIND");
    }
}

#[test]
fn env_fallback_draft_kind_fills_from_env_when_cli_unset() {
    let _guard = env_lock();
    unsafe {
        std::env::set_var("MLXCEL_DRAFT_KIND", "mtp");
    }
    let mut value = None;
    env_fallback_draft_kind(&mut value);
    assert_eq!(value, Some("mtp".to_string()));
    unsafe {
        std::env::remove_var("MLXCEL_DRAFT_KIND");
    }
}

#[test]
fn env_fallback_draft_block_size_keeps_cli_when_set() {
    let _guard = env_lock();
    unsafe {
        std::env::set_var("MLXCEL_DRAFT_BLOCK_SIZE", "8");
    }
    let mut value = Some(16);
    env_fallback_draft_block_size(&mut value);
    assert_eq!(value, Some(16), "CLI must win on conflict");
    unsafe {
        std::env::remove_var("MLXCEL_DRAFT_BLOCK_SIZE");
    }
}

#[test]
fn env_fallback_draft_block_size_fills_from_env_when_cli_unset() {
    let _guard = env_lock();
    unsafe {
        std::env::set_var("MLXCEL_DRAFT_BLOCK_SIZE", "8");
    }
    let mut value = None;
    env_fallback_draft_block_size(&mut value);
    assert_eq!(value, Some(8));
    unsafe {
        std::env::remove_var("MLXCEL_DRAFT_BLOCK_SIZE");
    }
}

#[test]
fn env_fallback_draft_block_size_ignores_unparseable_env() {
    let _guard = env_lock();
    unsafe {
        std::env::set_var("MLXCEL_DRAFT_BLOCK_SIZE", "not-a-number");
    }
    let mut value = None;
    env_fallback_draft_block_size(&mut value);
    assert_eq!(
        value, None,
        "unparseable env must leave value unset (warn-and-ignore)"
    );
    unsafe {
        std::env::remove_var("MLXCEL_DRAFT_BLOCK_SIZE");
    }
}
