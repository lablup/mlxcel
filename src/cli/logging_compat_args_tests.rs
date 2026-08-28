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

//! Unit tests for the b10621 logging, introspection, and preset group
//! (issue #1448).

use super::*;

fn args() -> LoggingCompatArgs {
    LoggingCompatArgs::default()
}

fn with_env<T>(pairs: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
    let _guard = crate::test_support::env_lock::env_lock();
    let saved: Vec<(String, Option<String>)> = pairs
        .iter()
        .map(|(k, _)| ((*k).to_owned(), std::env::var(k).ok()))
        .collect();
    unsafe {
        for (key, value) in pairs {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    let out = body();
    unsafe {
        for (key, value) in &saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
    out
}

/// The five variables `resolve_format` consults, all cleared, under the
/// process-wide environment lock.
///
/// `ensure_supported` validates the format values, so every test that calls
/// it now reads the environment. Without this the suite raced its own
/// `with_env` cases: a sibling test holding `LLAMA_ARG_LOG_PREFIX=on` made an
/// unrelated assertion about an empty argument set observe someone else's
/// environment.
fn with_clean_log_env<T>(body: impl FnOnce() -> T) -> T {
    with_env(
        &[
            ("LLAMA_ARG_LOG_PREFIX", None),
            ("LLAMA_ARG_NO_LOG_PREFIX", None),
            ("LLAMA_ARG_LOG_TIMESTAMPS", None),
            ("LLAMA_ARG_NO_LOG_TIMESTAMPS", None),
            ("LLAMA_ARG_LOG_VERBOSITY", None),
        ],
        body,
    )
}

// ── nothing supplied is inert ───────────────────────────────────────────

#[test]
fn an_empty_argument_set_is_supported() {
    with_clean_log_env(|| assert_eq!(args().ensure_supported(), Ok(())));
    assert_eq!(args().early_action(), None);
}

#[test]
fn the_default_format_matches_b10621s_server_defaults() {
    // Verified against the pinned macOS arm64 llama-server: it prints both a
    // level tag and an elapsed timestamp with no logging flags at all.
    let resolved = with_clean_log_env(|| args().resolve_format(false)).expect("defaults resolve");
    assert_eq!(resolved.colors, LogColors::Auto);
    assert!(resolved.prefix);
    assert!(resolved.timestamps);
    assert_eq!(resolved.cli_verbosity, None);
    assert_eq!(resolved.env_verbosity, None);
}

// ── --log-prompts-dir is refused, not silently accepted ─────────────────

#[test]
fn log_prompts_dir_is_refused_with_a_named_alternative() {
    let mut a = args();
    a.log_prompts_dir = Some(PathBuf::from("/tmp/prompts"));
    let rejection =
        with_clean_log_env(|| a.ensure_supported()).expect_err("--log-prompts-dir must be refused");
    assert_eq!(rejection.option, "--log-prompts-dir");
    let text = rejection.to_string();
    assert!(
        text.contains("/tmp/prompts"),
        "the diagnostic must name the directory that would have been used: {text}"
    );
    assert!(
        text.contains("--log-file"),
        "the diagnostic must name the supported debugging alternative: {text}"
    );
}

#[test]
fn an_absent_log_prompts_dir_is_not_refused() {
    let mut a = args();
    a.log_prompts_dir = None;
    with_clean_log_env(|| assert_eq!(a.ensure_supported(), Ok(())));
}

// ── presets are refused with a runnable download example ────────────────

#[test]
fn the_preset_table_covers_every_b10621_preset_flag() {
    // The twelve flags b10621 defines, read off `common/arg.cpp` at the
    // pinned commit. A new upstream preset must fail here rather than be
    // silently accepted and ignored by `requested_presets`.
    let expected = [
        "--embd-gemma-default",
        "--fim-qwen-1.5b-default",
        "--fim-qwen-3b-default",
        "--fim-qwen-7b-default",
        "--fim-qwen-7b-spec",
        "--fim-qwen-14b-spec",
        "--fim-qwen-30b-default",
        "--gpt-oss-20b-default",
        "--gpt-oss-120b-default",
        "--vision-gemma-4b-default",
        "--vision-gemma-12b-default",
        "--spec-default",
    ];
    let actual: Vec<&str> = PRESETS.iter().map(|p| p.flag).collect();
    assert_eq!(actual, expected);
}

/// Set the nth preset flag, in `PRESETS` order.
fn args_with_preset(index: usize) -> LoggingCompatArgs {
    let mut a = args();
    match index {
        0 => a.embd_gemma_default = true,
        1 => a.fim_qwen_1_5b_default = true,
        2 => a.fim_qwen_3b_default = true,
        3 => a.fim_qwen_7b_default = true,
        4 => a.fim_qwen_7b_spec = true,
        5 => a.fim_qwen_14b_spec = true,
        6 => a.fim_qwen_30b_default = true,
        7 => a.gpt_oss_20b_default = true,
        8 => a.gpt_oss_120b_default = true,
        9 => a.vision_gemma_4b_default = true,
        10 => a.vision_gemma_12b_default = true,
        11 => a.spec_default = true,
        other => panic!("no preset at index {other}"),
    }
    a
}

#[test]
fn every_preset_is_refused_and_names_its_own_flag() {
    for (index, preset) in PRESETS.iter().enumerate() {
        let rejection = with_clean_log_env(|| args_with_preset(index).ensure_supported())
            .expect_err(preset.flag);
        assert_eq!(rejection.option, preset.flag);
    }
}

#[test]
fn every_model_preset_refusal_carries_a_runnable_download_example() {
    for (index, preset) in PRESETS.iter().enumerate() {
        let Some(repo) = preset.mlxcel_repo else {
            continue;
        };
        let text = with_clean_log_env(|| args_with_preset(index).ensure_supported())
            .expect_err("refused")
            .to_string();
        assert!(
            text.contains(&format!("mlxcel download {repo}")),
            "{}: the refusal must show the exact download command: {text}",
            preset.flag
        );
        assert!(
            text.contains(&format!("mlxcel-server -m {repo}")),
            "{}: the refusal must show how to serve the checkpoint: {text}",
            preset.flag
        );
        assert!(
            text.contains(preset.upstream_repo),
            "{}: the refusal must name the GGUF repository it declined: {text}",
            preset.flag
        );
    }
}

#[test]
fn every_preset_names_an_mlx_community_checkpoint_or_none_at_all() {
    for preset in PRESETS {
        if let Some(repo) = preset.mlxcel_repo {
            assert!(
                repo.starts_with("mlx-community/"),
                "{}: {repo} is not an mlx-community checkpoint",
                preset.flag
            );
            assert!(
                !repo.contains("GGUF"),
                "{}: {repo} looks like a GGUF repository, which mlxcel cannot load",
                preset.flag
            );
        }
    }
}

#[test]
fn the_speculative_presets_name_a_draft_checkpoint_too() {
    for preset in PRESETS.iter().filter(|p| p.flag.ends_with("-spec")) {
        assert!(
            preset.mlxcel_draft_repo.is_some(),
            "{}: a speculative preset must map its draft model too",
            preset.flag
        );
    }
}

#[test]
fn spec_default_points_at_mlxcels_own_drafters() {
    let mut a = args();
    a.spec_default = true;
    let text = with_clean_log_env(|| a.ensure_supported())
        .expect_err("refused")
        .to_string();
    assert!(
        text.contains("--draft-kind mtp") && text.contains("--draft-kind dflash"),
        "--spec-default must name both mlxcel drafters: {text}"
    );
    assert!(
        !text.contains("mlxcel download"),
        "--spec-default configures no model, so it must not print a download line: {text}"
    );
}

#[test]
fn the_first_preset_in_help_order_is_the_one_reported() {
    let mut a = args();
    a.gpt_oss_20b_default = true;
    a.vision_gemma_4b_default = true;
    let rejection = with_clean_log_env(|| a.ensure_supported()).expect_err("refused");
    assert_eq!(rejection.option, "--gpt-oss-20b-default");
}

#[test]
fn log_prompts_dir_is_reported_before_a_preset() {
    let mut a = args();
    a.log_prompts_dir = Some(PathBuf::from("/tmp/prompts"));
    a.gpt_oss_20b_default = true;
    assert_eq!(
        with_clean_log_env(|| a.ensure_supported())
            .expect_err("refused")
            .option,
        "--log-prompts-dir"
    );
}

// ── early actions ───────────────────────────────────────────────────────

#[test]
fn cache_list_and_completion_bash_are_early_actions() {
    let mut a = args();
    a.cache_list = true;
    assert_eq!(a.early_action(), Some(EarlyAction::CacheList));

    let mut a = args();
    a.completion_bash = true;
    assert_eq!(a.early_action(), Some(EarlyAction::CompletionBash));
}

#[test]
fn cache_list_wins_when_both_introspection_flags_are_given() {
    // b10621's `--cache-list` handler calls `exit(0)` from inside the parser
    // while `--completion-bash` only sets a flag, so upstream's `--cache-list`
    // also wins whenever both appear.
    let mut a = args();
    a.cache_list = true;
    a.completion_bash = true;
    assert_eq!(a.early_action(), Some(EarlyAction::CacheList));
}

// ── colors ──────────────────────────────────────────────────────────────

#[test]
fn log_colors_accepts_exactly_b10621s_vocabulary() {
    for value in ["on", "enabled", "true", "1"] {
        assert_eq!(LogColors::parse(value), Ok(LogColors::On), "{value}");
    }
    for value in ["off", "disabled", "false", "0"] {
        assert_eq!(LogColors::parse(value), Ok(LogColors::Off), "{value}");
    }
    assert_eq!(LogColors::parse("auto"), Ok(LogColors::Auto));
    for value in ["yes", "ON", "AUTO", "", "maybe"] {
        assert!(
            LogColors::parse(value).is_err(),
            "{value:?} is outside b10621's case-sensitive vocabulary and must be refused"
        );
    }
}

#[test]
fn an_unknown_log_colors_value_fails_resolution_with_the_option_named() {
    let mut a = args();
    a.log_colors = "sometimes".to_owned();
    let rejection = with_clean_log_env(|| a.resolve_format(false)).expect_err("must be refused");
    assert_eq!(rejection.option, "--log-colors");
    assert!(rejection.detail.contains("sometimes"));
}

// ── bool pairs and their environment bindings ───────────────────────────

#[test]
fn the_command_line_beats_the_environment_for_both_bool_pairs() {
    let resolved = with_env(
        &[
            ("LLAMA_ARG_LOG_PREFIX", Some("on")),
            ("LLAMA_ARG_NO_LOG_PREFIX", None),
            ("LLAMA_ARG_LOG_TIMESTAMPS", Some("on")),
            ("LLAMA_ARG_NO_LOG_TIMESTAMPS", None),
            ("LLAMA_ARG_LOG_VERBOSITY", None),
        ],
        || {
            let mut a = args();
            a.no_log_prefix = true;
            a.no_log_timestamps = true;
            a.resolve_format(false)
        },
    )
    .expect("resolves");
    assert!(!resolved.prefix);
    assert!(!resolved.timestamps);
}

#[test]
fn the_bool_pair_environment_bindings_follow_b10621s_vocabulary() {
    for (value, expected) in [("off", false), ("0", false), ("on", true), ("1", true)] {
        let resolved = with_env(
            &[
                ("LLAMA_ARG_LOG_PREFIX", Some(value)),
                ("LLAMA_ARG_NO_LOG_PREFIX", None),
                ("LLAMA_ARG_LOG_TIMESTAMPS", None),
                ("LLAMA_ARG_NO_LOG_TIMESTAMPS", None),
                ("LLAMA_ARG_LOG_VERBOSITY", None),
            ],
            || args().resolve_format(false),
        )
        .expect("resolves");
        assert_eq!(resolved.prefix, expected, "LLAMA_ARG_LOG_PREFIX={value}");
    }
}

#[test]
fn the_llama_arg_no_alias_disables_the_pair() {
    let resolved = with_env(
        &[
            ("LLAMA_ARG_LOG_TIMESTAMPS", None),
            ("LLAMA_ARG_NO_LOG_TIMESTAMPS", Some("1")),
            ("LLAMA_ARG_LOG_PREFIX", None),
            ("LLAMA_ARG_NO_LOG_PREFIX", None),
            ("LLAMA_ARG_LOG_VERBOSITY", None),
        ],
        || args().resolve_format(false),
    )
    .expect("resolves");
    assert!(!resolved.timestamps);
}

#[test]
fn a_malformed_bool_pair_environment_value_fails_loudly() {
    let rejection = with_env(
        &[
            ("LLAMA_ARG_LOG_PREFIX", Some("sometimes")),
            ("LLAMA_ARG_NO_LOG_PREFIX", None),
            ("LLAMA_ARG_LOG_TIMESTAMPS", None),
            ("LLAMA_ARG_NO_LOG_TIMESTAMPS", None),
            ("LLAMA_ARG_LOG_VERBOSITY", None),
        ],
        || args().resolve_format(false),
    )
    .expect_err("a value outside b10621's vocabulary must not pick a side");
    assert_eq!(rejection.option, "--log-prefix");
    assert!(rejection.detail.contains("sometimes"));
}

// ── verbosity source attribution ────────────────────────────────────────

#[test]
fn a_command_line_verbosity_is_attributed_to_the_command_line() {
    let resolved = with_env(&[("LLAMA_ARG_LOG_VERBOSITY", None)], || {
        let mut a = args();
        a.verbosity = 5;
        a.resolve_format(true)
    })
    .expect("resolves");
    assert_eq!(resolved.cli_verbosity, Some(5));
    assert_eq!(resolved.env_verbosity, None);
}

#[test]
fn an_environment_verbosity_is_attributed_to_the_environment() {
    // clap folds `LLAMA_ARG_LOG_VERBOSITY` into the parsed value, so the only
    // way to tell it apart from a command-line value is the argv scan the
    // caller performs; `resolve_format(false)` is that answer.
    let resolved = with_env(&[("LLAMA_ARG_LOG_VERBOSITY", Some("4"))], || {
        let mut a = args();
        a.verbosity = 4;
        a.resolve_format(false)
    })
    .expect("resolves");
    assert_eq!(resolved.cli_verbosity, None);
    assert_eq!(resolved.env_verbosity, Some(4));
}

#[test]
fn the_compiled_default_is_attributed_to_neither() {
    let resolved = with_clean_log_env(|| args().resolve_format(false)).expect("resolves");
    assert_eq!(resolved.cli_verbosity, None);
    assert_eq!(resolved.env_verbosity, None);
}

// ── credential registration ─────────────────────────────────────────────

#[test]
fn registering_credentials_redacts_keys_tokens_and_key_files() {
    use crate::server::logging::redact;

    let dir = tempfile::tempdir().expect("tempdir");
    let key_file = dir.path().join("keys.txt");
    std::fs::write(&key_file, "canary-file-key-11111111\n").expect("write");

    with_env(
        &[
            ("HF_TOKEN", Some("canary-hf-token-22222222")),
            ("LLAMA_API_KEY", None),
            ("MLXCEL_API_KEY", None),
            ("LLAMA_ARG_API_KEY_FILE", None),
        ],
        || {
            register_credentials_for_redaction(
                &["canary-inline-key-33333333".to_owned()],
                std::slice::from_ref(&key_file),
                Some("canary-flag-token-44444444"),
            );
        },
    );

    for secret in [
        "canary-file-key-11111111",
        "canary-hf-token-22222222",
        "canary-inline-key-33333333",
        "canary-flag-token-44444444",
    ] {
        let line = format!("starting with secret={secret} rest");
        assert!(
            !redact(&line).contains(secret),
            "{secret} survived redaction in {line:?}"
        );
    }
}

#[test]
fn a_missing_api_key_file_does_not_abort_registration() {
    // A `--api-key-file` that cannot be read is diagnosed by the auth path,
    // not here; registration must not panic or short-circuit the rest.
    with_env(
        &[
            ("HF_TOKEN", None),
            ("LLAMA_API_KEY", None),
            ("MLXCEL_API_KEY", None),
            ("LLAMA_ARG_API_KEY_FILE", None),
        ],
        || {
            register_credentials_for_redaction(
                &[],
                &[PathBuf::from("/nonexistent/mlxcel/keys.txt")],
                None,
            );
        },
    );
}
