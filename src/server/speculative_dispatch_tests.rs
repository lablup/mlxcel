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

/// A checkpoint directory that declares `model_type` and `vocab_size`, as a
/// full-model pair must on both sides.
fn write_vocab_config(vocab_size: usize) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    std::fs::write(
        dir.path().join("config.json"),
        format!(r#"{{"model_type": "qwen3", "vocab_size": {vocab_size}}}"#),
    )
    .expect("write config.json");
    let path = dir.path().to_path_buf();
    (dir, path)
}

/// A target checkpoint path with no `config.json`, so the #1797 block-width
/// policy classifies it as `TargetQuantization::Unknown` and every
/// expectation below stays the flat per-kind default. The policy's own arms
/// are covered in `src/cli/draft_block_policy_tests.rs`.
fn unmeasured_target() -> &'static std::path::Path {
    std::path::Path::new("/nonexistent/mlxcel-dispatch-test-target")
}

#[test]
fn resolve_returns_disabled_when_no_drafter_configured() {
    let cfg = base_config();
    assert!(cfg.draft_model_path.is_none());

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
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

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
    match dispatch {
        SpeculativeDispatch::Mtp {
            draft_model_path,
            block_size,
            user_requested_explicit_kind,
            ..
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

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
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

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
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

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
    match dispatch {
        SpeculativeDispatch::DFlash {
            draft_model_path,
            block_size,
            user_requested_explicit_kind,
            ..
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

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
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

    let err = SpeculativeDispatch::resolve(&cfg, unmeasured_target())
        .expect_err("must reject internal-mtp");
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

    let err =
        SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect_err("must reject unknown");
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
fn resolve_with_missing_drafter_config_reports_the_config_at_startup() {
    // Drafter directory has no `config.json`. The resolver swallows that
    // and falls back to DFlash, where the real cause would only surface as
    // a drafter load error on the first request; with no explicit kind the
    // dispatch reports the config problem at startup instead (#2754, #2757).
    // An explicit `--draft-kind dflash` keeps the historical lazy path.
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().to_path_buf();

    let mut cfg = base_config();
    cfg.draft_model_path = Some(path.clone());
    let _keep_dir_alive = dir;

    let err = SpeculativeDispatch::resolve(&cfg, unmeasured_target())
        .expect_err("an unreadable config is reported at startup");
    match err {
        SpeculativeDispatchError::DrafterConfig {
            path: reported,
            message,
        } => {
            assert_eq!(reported, path);
            assert!(message.contains("config.json"), "{message}");
        }
        other => panic!("expected DrafterConfig, got {other:?}"),
    }

    cfg.draft_kind = Some("dflash".to_string());
    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target())
        .expect("an explicit kind keeps the lazy load path");
    match dispatch {
        SpeculativeDispatch::DFlash {
            draft_model_path,
            user_requested_explicit_kind,
            block_size,
            ..
        } => {
            assert_eq!(draft_model_path, path);
            assert!(user_requested_explicit_kind);
            assert_eq!(block_size, 16);
        }
        other => panic!("expected DFlash, got {other:?}"),
    }
}

/// #2757: an ordinary model directory (`model_type` with no mapping, no
/// DFlash-family marker) resolves to the full-model arm, auto-detected or
/// under `--draft-kind full-model`, with the token-only tick's settings and
/// no kind-specific burst.
#[test]
fn resolve_full_model_auto_detected_and_explicit() {
    let (_dir, path) = write_vocab_config(151936);
    let (_target_dir, target) = write_vocab_config(151936);
    for (kind, explicit) in [(None, false), (Some("full-model".to_string()), true)] {
        let mut cfg = base_config();
        cfg.draft_model_path = Some(path.clone());
        cfg.draft_kind = kind;
        let dispatch = SpeculativeDispatch::resolve(&cfg, &target).expect("resolve");
        match &dispatch {
            SpeculativeDispatch::FullModel {
                draft_model_path,
                block_size,
                user_requested_explicit_kind,
                max_batch,
                ..
            } => {
                assert_eq!(draft_model_path, &path);
                assert_eq!(
                    *block_size,
                    crate::cli::speculative_args::DEFAULT_FULL_MODEL_BLOCK_SIZE
                );
                assert_eq!(*user_requested_explicit_kind, explicit);
                assert_eq!(*max_batch, cfg.prompt_lookup_max_batch);
            }
            other => panic!("expected FullModel, got {other:?}"),
        }
        assert!(!dispatch.is_kind_specific());
        assert_eq!(dispatch.drafter_kind(), Some(DrafterKind::FullModel));
        assert_eq!(dispatch.full_model_draft_path(), Some(path.as_path()));
        let rounds = dispatch.token_only_rounds().expect("token-only rounds");
        assert_eq!(
            rounds.max_draft,
            crate::cli::speculative_args::DEFAULT_FULL_MODEL_BLOCK_SIZE as usize
        );
        assert_eq!(rounds.max_batch, cfg.prompt_lookup_max_batch);
        assert!(dispatch.summary().starts_with("speculative=full-model"));
    }
}

/// `--draft-block-size` reaches the full-model arm, and a draft vocabulary
/// larger than the target's is refused at startup.
#[test]
fn resolve_full_model_block_size_and_vocab_guard() {
    let (_dir, path) = write_vocab_config(151936);
    let (_target_dir, smaller_target) = write_vocab_config(151936);
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path.clone());
    cfg.draft_block_size = Some(3);
    let dispatch = SpeculativeDispatch::resolve(&cfg, &smaller_target).expect("resolve");
    assert_eq!(dispatch.block_size(), Some(3));

    let draft = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        draft.path().join("config.json"),
        r#"{"model_type": "qwen3", "vocab_size": 152064}"#,
    )
    .unwrap();
    let target = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        target.path().join("config.json"),
        r#"{"model_type": "qwen3", "vocab_size": 151936}"#,
    )
    .unwrap();
    cfg.draft_model_path = Some(draft.path().to_path_buf());
    let err = SpeculativeDispatch::resolve(&cfg, target.path()).expect_err("vocab guard");
    match err {
        SpeculativeDispatchError::InvalidKind { message } => {
            assert!(message.contains("vocabulary"), "{message}");
        }
        other => panic!("expected InvalidKind, got {other:?}"),
    }
    // The other way round (a smaller draft vocabulary) passes.
    cfg.draft_model_path = Some(target.path().to_path_buf());
    SpeculativeDispatch::resolve(&cfg, draft.path()).expect("smaller draft vocabulary");
}

/// A pair with no declared vocabulary on either side cannot be checked, and
/// neither MLX's `take` nor the drafter's input clamp would catch an id past
/// the table, so it is refused at startup with the path that lacks it. Both a
/// config without the key and a target with no config at all count.
#[test]
fn resolve_full_model_refuses_an_undeclared_vocabulary() {
    let (_declared_dir, declared) = write_vocab_config(151936);
    let (_undeclared_dir, undeclared) = write_drafter_config(Some("qwen3"));
    let cases = [
        (&undeclared, undeclared.as_path(), "draft model's"),
        (&declared, undeclared.as_path(), "target's"),
        (&declared, unmeasured_target(), "target's"),
    ];
    for (draft, target, side) in cases {
        let mut cfg = base_config();
        cfg.draft_model_path = Some(draft.clone());
        cfg.draft_kind = Some("full-model".to_string());
        // The draft-side case names the draft path, the others the target's.
        let named = if side == "target's" {
            target
        } else {
            draft.as_path()
        };
        match SpeculativeDispatch::resolve(&cfg, target).expect_err("undeclared vocabulary") {
            SpeculativeDispatchError::InvalidKind { message } => {
                assert!(message.contains(side), "{message}");
                assert!(message.contains("vocab_size"), "{message}");
                assert!(message.contains("text_config"), "{message}");
                assert!(message.contains("cannot be checked"), "{message}");
                assert!(message.contains(&named.display().to_string()), "{message}");
            }
            other => panic!("expected InvalidKind, got {other:?}"),
        }
    }
    // Declared under `text_config` counts.
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3", "text_config": {"vocab_size": 151936}}"#,
    )
    .unwrap();
    let mut cfg = base_config();
    cfg.draft_model_path = Some(dir.path().to_path_buf());
    SpeculativeDispatch::resolve(&cfg, &declared).expect("vocabulary under text_config");
}

/// `--draft-kind full-model` on a DFlash drafter directory is refused at
/// startup; the same directory with no kind still resolves to DFlash.
#[test]
fn resolve_full_model_refuses_a_dflash_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3", "architectures": ["DFlashDraftModel"], "dflash_config": {"mask_token_id": 1}}"#,
    )
    .unwrap();
    let mut cfg = base_config();
    cfg.draft_model_path = Some(dir.path().to_path_buf());
    cfg.draft_kind = Some("full-model".to_string());
    match SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect_err("refused") {
        SpeculativeDispatchError::InvalidKind { message } => {
            assert!(message.contains("DFlash-family"), "{message}");
        }
        other => panic!("expected InvalidKind, got {other:?}"),
    }
    cfg.draft_kind = None;
    assert!(matches!(
        SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("dflash"),
        SpeculativeDispatch::DFlash { .. }
    ));
}

#[test]
fn summary_contains_block_size_for_kind_specific_variants() {
    let (_dir, path) = write_drafter_config(Some("dflash"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("dflash".to_string());

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
    let s = dispatch.summary();
    assert!(s.contains("speculative=dflash"));
    assert!(s.contains("block_size=16"));
    assert!(s.contains("drafter="));
}

#[test]
fn block_size_accessor_is_none_for_disabled() {
    assert!(SpeculativeDispatch::Disabled.block_size().is_none());
}

#[test]
fn is_kind_specific_is_true_only_for_mtp_and_dflash() {
    let mtp = SpeculativeDispatch::Mtp {
        draft_model_path: PathBuf::from("/tmp/m"),
        block_size: 4,
        block_size_source: crate::cli::draft_block_policy::BlockSizeSource::Override,
        user_requested_explicit_kind: true,
    };
    let dflash = SpeculativeDispatch::DFlash {
        draft_model_path: PathBuf::from("/tmp/d"),
        block_size: 16,
        block_size_source: crate::cli::draft_block_policy::BlockSizeSource::Override,
        user_requested_explicit_kind: false,
    };
    assert!(mtp.is_kind_specific());
    assert!(dflash.is_kind_specific());
    assert!(!SpeculativeDispatch::Disabled.is_kind_specific());
}

/// `--draft-kind prompt-lookup` with no drafter resolves before the
/// draft-model early return (#2255): `PromptLookup` with server defaults
/// and max batch, not a kind-specific burst, so the scheduler's
/// `should_dispatch_speculative()` (which is `is_kind_specific()`) stays
/// false and the burst path is never entered.
#[test]
fn resolve_prompt_lookup_without_a_drafter() {
    let mut cfg = base_config();
    cfg.draft_kind = Some("prompt-lookup".to_string());
    assert!(cfg.draft_model_path.is_none());

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
    let SpeculativeDispatch::PromptLookup { config, max_batch } = &dispatch else {
        panic!("expected PromptLookup, got {dispatch:?}");
    };
    let cli_defaults = PromptLookupConfig::default();
    assert_eq!(cli_defaults.min_tokens_per_round, None);
    assert_eq!(cli_defaults.shadow_confirm_tokens, 2);
    assert_eq!(cli_defaults.plain_rounds_before_pipeline, 2);
    let server_defaults = PromptLookupConfig {
        min_tokens_per_round: cfg!(feature = "cuda").then_some(4),
        shadow_confirm_tokens: if cfg!(feature = "cuda") { 7 } else { 2 },
        plain_rounds_before_pipeline: if cfg!(feature = "cuda") { 0 } else { 2 },
        ..cli_defaults
    };
    assert_eq!(*config, server_defaults);
    assert_eq!(*max_batch, DEFAULT_PROMPT_LOOKUP_MAX_BATCH);
    assert!(!dispatch.is_kind_specific());
    assert_eq!(
        dispatch.prompt_lookup_config(),
        Some((&server_defaults, DEFAULT_PROMPT_LOOKUP_MAX_BATCH))
    );
    assert_eq!(dispatch.drafter_kind(), Some(DrafterKind::PromptLookup));
    assert!(dispatch.draft_model_path().is_none());
    assert!(dispatch.block_size().is_none());
    let summary = dispatch.summary();
    assert!(
        summary.starts_with(&format!(
            "speculative=prompt-lookup (max_draft={}, policy=",
            PromptLookupConfig::default().max_draft
        )),
        "{summary}"
    );
}

/// A drafter checkpoint with `--draft-kind prompt-lookup` is refused.
#[test]
fn resolve_prompt_lookup_with_a_drafter_is_invalid() {
    let (_dir, path) = write_drafter_config(Some("qwen3"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("prompt-lookup".to_string());

    let err = SpeculativeDispatch::resolve(&cfg, unmeasured_target())
        .expect_err("prompt-lookup with a draft model must be refused");
    match err {
        SpeculativeDispatchError::InvalidKind { message } => {
            assert!(
                message.contains("prompt-lookup takes no draft model"),
                "{message}"
            );
        }
        other => panic!("expected InvalidKind, got {other:?}"),
    }
}

/// `--draft-block-size` sets `max_draft`; the other fields stay default;
/// `--prompt-lookup-max-batch` lands in the dispatch.
#[test]
fn resolve_prompt_lookup_block_size_and_max_batch() {
    let mut cfg = base_config();
    cfg.draft_kind = Some("prompt-lookup".to_string());
    cfg.draft_block_size = Some(4);
    cfg.prompt_lookup_max_batch = 3;

    let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
    let (config, max_batch) = dispatch.prompt_lookup_config().expect("prompt lookup");
    assert_eq!(config.max_draft, 4);
    assert_eq!(
        *config,
        PromptLookupConfig {
            max_draft: 4,
            min_tokens_per_round: cfg!(feature = "cuda")
                .then_some(PROMPT_LOOKUP_MIN_TOKENS_PER_ROUND),
            shadow_confirm_tokens: if cfg!(feature = "cuda") {
                PROMPT_LOOKUP_SHADOW_CONFIRM_TOKENS
            } else {
                2
            },
            plain_rounds_before_pipeline: if cfg!(feature = "cuda") { 0 } else { 2 },
            ..PromptLookupConfig::default()
        }
    );
    assert_eq!(max_batch, 3);
    assert!(dispatch.summary().contains("max_draft=4"));
    assert!(dispatch.summary().contains("max_batch=3"));
}

/// A block size the drafter cannot use and a zero max batch are refused at
/// startup rather than per request.
#[test]
fn resolve_prompt_lookup_rejects_out_of_range_settings() {
    let mut cfg = base_config();
    cfg.draft_kind = Some("prompt-lookup".to_string());
    cfg.draft_block_size = Some(0);
    assert!(matches!(
        SpeculativeDispatch::resolve(&cfg, unmeasured_target()),
        Err(SpeculativeDispatchError::InvalidKind { .. })
    ));
    cfg.draft_block_size = None;
    cfg.prompt_lookup_max_batch = 0;
    assert!(matches!(
        SpeculativeDispatch::resolve(&cfg, unmeasured_target()),
        Err(SpeculativeDispatchError::InvalidKind { .. })
    ));
}

/// The unrecognised-kind message lists prompt-lookup among the accepted
/// values.
#[test]
fn unknown_kind_message_lists_prompt_lookup() {
    let (_dir, path) = write_drafter_config(Some("qwen3"));
    let mut cfg = base_config();
    cfg.draft_model_path = Some(path);
    cfg.draft_kind = Some("ngram".to_string());
    let err = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect_err("unknown kind");
    assert!(err.to_string().contains("prompt-lookup"), "{err}");
}

#[cfg(feature = "cuda")]
#[test]
fn resolve_prompt_lookup_caps_break_even_floor_for_explicit_narrow_blocks() {
    for width in [1, 2] {
        let mut cfg = base_config();
        cfg.draft_kind = Some("prompt-lookup".to_string());
        cfg.draft_block_size = Some(width);
        let dispatch = SpeculativeDispatch::resolve(&cfg, unmeasured_target()).expect("resolve");
        let (config, _) = dispatch.prompt_lookup_config().expect("prompt lookup");
        assert_eq!(config.min_tokens_per_round, Some(width as usize + 1));
        assert_eq!(
            config.shadow_confirm_tokens, 7,
            "narrow widths still need strong resume evidence"
        );
    }
}
