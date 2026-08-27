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

//! Unit tests for the b10621 model-source flag translation (issue #1434).

use super::*;
use std::path::Path;

fn args() -> LlamaModelSourceArgs {
    LlamaModelSourceArgs::default()
}

// ── --model ─────────────────────────────────────────────────────────────────

#[test]
fn model_alone_is_the_reference() {
    let resolved = resolve_llama_model_source(&LlamaModelSourceArgs {
        model: Some(PathBuf::from("models/Qwen3-4B-4bit")),
        ..args()
    })
    .expect("a plain -m resolves");
    assert_eq!(resolved.reference, PathBuf::from("models/Qwen3-4B-4bit"));
    assert_eq!(resolved.origin, "--model");
    assert_eq!(resolved.superseded_model, None);
}

#[test]
fn no_source_flag_names_all_three_ways_to_supply_one() {
    let err = resolve_llama_model_source(&args()).expect_err("a source is required");
    let text = format!("{err}");
    for expected in ["LLAMA_ARG_MODEL", "-m <PATH_OR_REPO_ID>", "--hf-repo"] {
        assert!(text.contains(expected), "{expected} missing from: {text}");
    }
}

// ── --hf-repo ───────────────────────────────────────────────────────────────

#[test]
fn hf_repo_becomes_the_reference() {
    let resolved = resolve_llama_model_source(&LlamaModelSourceArgs {
        hf_repo: Some("mlx-community/Qwen3-4B-4bit".to_owned()),
        ..args()
    })
    .expect("a plain --hf-repo resolves");
    assert_eq!(
        resolved.reference,
        PathBuf::from("mlx-community/Qwen3-4B-4bit")
    );
    assert_eq!(resolved.origin, "--hf-repo");
}

#[test]
fn hf_repo_supersedes_model_and_reports_it() {
    let resolved = resolve_llama_model_source(&LlamaModelSourceArgs {
        model: Some(PathBuf::from("models/local")),
        hf_repo: Some("mlx-community/Qwen3-4B-4bit".to_owned()),
        ..args()
    })
    .expect("both flags resolve");
    assert_eq!(
        resolved.reference,
        PathBuf::from("mlx-community/Qwen3-4B-4bit")
    );
    assert_eq!(
        resolved.superseded_model,
        Some(PathBuf::from("models/local"))
    );

    let notice = superseded_model_notice(&resolved.reference, &PathBuf::from("models/local"));
    assert!(notice.contains("mlx-community/Qwen3-4B-4bit"));
    assert!(notice.contains("models/local"));
}

#[test]
fn hf_repo_quant_suffix_is_rejected_with_the_mlx_naming_rule() {
    let err = parse_hf_repo("ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M")
        .expect_err("a :quant suffix must be rejected");
    let text = format!("{err}");
    assert!(text.contains("Q4_K_M"), "must quote the quant: {text}");
    assert!(
        text.contains("--hf-repo mlx-community/"),
        "must offer an MLX repository: {text}"
    );
}

#[test]
fn hf_repo_must_be_owner_slash_name() {
    for value in ["justname", "a/b/c", "/name", "owner/"] {
        let err = parse_hf_repo(value).expect_err("must be rejected");
        assert!(
            format!("{err}").contains("<owner>/<name>"),
            "{value} must report the expected form"
        );
    }
}

#[test]
fn hf_repo_is_trimmed() {
    assert_eq!(
        parse_hf_repo("  mlx-community/Qwen3-4B-4bit  ").expect("trimmed"),
        "mlx-community/Qwen3-4B-4bit"
    );
}

// ── rejected sources ────────────────────────────────────────────────────────

#[test]
fn docker_repo_is_rejected_before_any_other_source_is_considered() {
    let err = resolve_llama_model_source(&LlamaModelSourceArgs {
        model: Some(PathBuf::from("models/Qwen3-4B-4bit")),
        docker_repo: Some("ai/gemma3".to_owned()),
        ..args()
    })
    .expect_err("--docker-repo must be rejected even alongside a valid -m");
    let text = format!("{err}");
    assert!(text.contains("ai/gemma3"), "must quote the value: {text}");
    assert!(text.contains("GGUF"), "must name the format: {text}");
    assert!(
        text.contains("--hf-repo mlx-community/"),
        "must offer a replacement: {text}"
    );
}

#[test]
fn model_url_pointing_at_huggingface_offers_the_repo_id() {
    let err = resolve_llama_model_source(&LlamaModelSourceArgs {
        model_url: Some("https://huggingface.co/mlx-community/Qwen3-4B-4bit".to_owned()),
        ..args()
    })
    .expect_err("--model-url must be rejected");
    let text = format!("{err}");
    assert!(
        text.contains("--hf-repo mlx-community/Qwen3-4B-4bit"),
        "must translate the URL into the repo id: {text}"
    );
}

#[test]
fn model_url_elsewhere_is_rejected_generically() {
    let err = resolve_llama_model_source(&LlamaModelSourceArgs {
        model_url: Some("https://example.com/model.gguf".to_owned()),
        ..args()
    })
    .expect_err("--model-url must be rejected");
    let text = format!("{err}");
    assert!(text.contains("https://example.com/model.gguf"));
    assert!(text.contains("--hf-repo mlx-community/Qwen3-4B-4bit"));
}

#[test]
fn hf_file_is_rejected_and_explains_snapshot_loading() {
    let err = resolve_llama_model_source(&LlamaModelSourceArgs {
        hf_repo: Some("mlx-community/Qwen3-4B-4bit".to_owned()),
        hf_file: Some("model-Q4_K_M.gguf".to_owned()),
        ..args()
    })
    .expect_err("--hf-file must be rejected");
    let text = format!("{err}");
    assert!(text.contains("model-Q4_K_M.gguf"));
    assert!(
        text.contains("model.safetensors.index.json"),
        "must explain how MLX shards are named: {text}"
    );
}

#[test]
fn empty_environment_values_are_treated_as_absent() {
    // `LLAMA_ARG_DOCKER_REPO=` in an inherited environment must not make the
    // server refuse to start; b10621 tests the same fields with `.empty()`.
    let resolved = resolve_llama_model_source(&LlamaModelSourceArgs {
        model: Some(PathBuf::from("models/Qwen3-4B-4bit")),
        docker_repo: Some(String::new()),
        model_url: Some("   ".to_owned()),
        hf_file: Some(String::new()),
        hf_repo: Some(String::new()),
    })
    .expect("blank compatibility values are inert");
    assert_eq!(resolved.reference, PathBuf::from("models/Qwen3-4B-4bit"));
}

// ── --alias ─────────────────────────────────────────────────────────────────

#[test]
fn alias_list_is_split_stripped_and_deduplicated() {
    assert_eq!(
        parse_model_aliases("gpt-4o, gpt-4o-mini ,gpt-4o"),
        vec!["gpt-4o".to_owned(), "gpt-4o-mini".to_owned()]
    );
}

#[test]
fn the_alias_list_is_sorted_like_b10621s_set() {
    // b10621 inserts each alias into a `std::set<std::string>` and serves the
    // set's first element, which is the lexicographically smallest entry, not
    // the one typed first. `--alias zebra,apple` serves `apple` upstream, so
    // it must serve `apple` here too.
    assert_eq!(
        parse_model_aliases("zebra,apple,Mango"),
        vec!["Mango".to_owned(), "apple".to_owned(), "zebra".to_owned()],
        "aliases must come back in the same order b10621's std::set holds them"
    );
}

#[test]
fn a_single_alias_is_unchanged() {
    assert_eq!(parse_model_aliases("llama-remote"), vec!["llama-remote"]);
}

#[test]
fn an_alias_value_with_no_content_yields_no_aliases() {
    for value in ["", "   ", ",", " , , "] {
        assert!(
            parse_model_aliases(value).is_empty(),
            "{value:?} must yield no aliases"
        );
    }
}

#[test]
fn the_first_entry_is_the_served_id() {
    let aliases = parse_model_aliases("primary,secondary");
    assert_eq!(aliases.first().map(String::as_str), Some("primary"));
    // And that first entry is the smallest, not the one typed first.
    let reversed = parse_model_aliases("secondary,primary");
    assert_eq!(reversed.first().map(String::as_str), Some("primary"));
}

// ── diagnostics are readable ────────────────────────────────────────────────

#[test]
fn no_diagnostic_carries_a_run_of_collapsed_indentation() {
    // A multi-line string literal whose `\` continuations are lost keeps its
    // source indentation as literal spaces, and `cargo fmt` will not reflow a
    // string literal, so nothing else in the gate notices. Every assertion in
    // this file matches short fragments, which straddle none of the gaps.
    let mut messages: Vec<String> = Vec::new();
    for args in [
        LlamaModelSourceArgs {
            docker_repo: Some("ai/gemma3".to_owned()),
            ..args()
        },
        LlamaModelSourceArgs {
            model_url: Some("https://huggingface.co/owner/name".to_owned()),
            ..args()
        },
        LlamaModelSourceArgs {
            model_url: Some("https://example.com/model.gguf".to_owned()),
            ..args()
        },
        LlamaModelSourceArgs {
            hf_file: Some("model.gguf".to_owned()),
            ..args()
        },
        LlamaModelSourceArgs::default(),
    ] {
        messages.push(format!(
            "{}",
            resolve_llama_model_source(&args).expect_err("each case is an error")
        ));
    }
    messages.push(format!(
        "{}",
        parse_hf_repo("owner/name:Q4_K_M").expect_err("quant suffix")
    ));
    messages.push(format!(
        "{}",
        parse_hf_repo("not-a-repo").expect_err("bad shape")
    ));
    messages.push(superseded_model_notice(
        Path::new("owner/name"),
        Path::new("models/local"),
    ));

    for message in messages {
        for line in message.lines() {
            assert!(
                !line.trim().contains("   "),
                "diagnostic line carries collapsed indentation: {line:?}"
            );
        }
    }
}

// ── credentials are never echoed ────────────────────────────────────────────

#[test]
fn a_model_url_credential_is_redacted_from_the_diagnostic() {
    let err = resolve_llama_model_source(&LlamaModelSourceArgs {
        model_url: Some("https://alice:hunter2@example.com/model.gguf".to_owned()),
        ..args()
    })
    .expect_err("--model-url must be rejected");
    let text = format!("{err}");
    assert!(
        !text.contains("hunter2") && !text.contains("alice"),
        "the userinfo component must not reach the diagnostic: {text}"
    );
    assert!(
        text.contains("example.com/model.gguf"),
        "the rest of the URL must still be shown: {text}"
    );
}

#[test]
fn an_hf_repo_credential_is_redacted_from_the_diagnostic() {
    let err = parse_hf_repo("https://alice:hunter2@huggingface.co/owner/name")
        .expect_err("a URL is not a repository identifier");
    let text = format!("{err}");
    assert!(
        !text.contains("hunter2"),
        "the userinfo component must not reach the diagnostic: {text}"
    );
    assert!(
        text.contains("<owner>/<name>"),
        "a URL must get the shape diagnostic, not the quant one: {text}"
    );
    assert!(
        !text.contains("quantization"),
        "a colon outside the final segment is not a quant request: {text}"
    );
}

#[test]
fn a_windows_path_is_reported_as_a_bad_identifier_not_a_quant_request() {
    let err = parse_hf_repo("C:\\models\\foo").expect_err("not a repository identifier");
    let text = format!("{err}");
    assert!(
        text.contains("<owner>/<name>") && !text.contains("quantization"),
        "got: {text}"
    );
}

// ── LLAMA_ARG_OFFLINE (issue #1434) ─────────────────────────────────────────

/// Run `body` with `LLAMA_ARG_OFFLINE` set to `value` (or removed for `None`),
/// restoring the previous value afterwards. Serialized through the crate-wide
/// env lock: Rust 2024 makes `set_var` unsafe precisely because libc's env
/// block has no internal lock.
fn with_offline_env<T>(value: Option<&str>, body: impl FnOnce() -> T) -> T {
    let _guard = crate::test_support::env_lock::env_lock();
    let prev = std::env::var(LLAMA_ARG_OFFLINE).ok();
    unsafe {
        match value {
            Some(v) => std::env::set_var(LLAMA_ARG_OFFLINE, v),
            None => std::env::remove_var(LLAMA_ARG_OFFLINE),
        }
    }
    let out = body();
    unsafe {
        match prev {
            Some(v) => std::env::set_var(LLAMA_ARG_OFFLINE, v),
            None => std::env::remove_var(LLAMA_ARG_OFFLINE),
        }
    }
    out
}

fn offline_for(value: Option<&str>) -> bool {
    with_offline_env(value, || {
        let mut flag = false;
        env_fallback_offline(&mut flag);
        flag
    })
}

#[test]
fn the_offline_env_var_enables_exactly_b10621s_truthy_set() {
    // `common_arg_utils::is_truthy`: "on", "enabled", "true", "1".
    for value in ["on", "enabled", "true", "1"] {
        assert!(
            offline_for(Some(value)),
            "LLAMA_ARG_OFFLINE={value} must enable offline mode"
        );
    }
}

#[test]
fn every_other_offline_env_value_leaves_the_flag_alone() {
    // b10621 fires a value-less option from the environment only on the
    // truthy set, so an empty value, an explicit falsey value, and a spelling
    // outside the set are all no-ops. Notably `TRUE` and `yes` are NOT in the
    // set; clap's own boolish env parser would accept them.
    for value in [
        "", "0", "false", "off", "disabled", "no", "yes", "TRUE", "On",
    ] {
        assert!(
            !offline_for(Some(value)),
            "LLAMA_ARG_OFFLINE={value:?} must not enable offline mode"
        );
    }
    assert!(!offline_for(None), "an unset variable must not enable it");
}

#[test]
fn an_explicit_offline_flag_survives_a_falsey_environment() {
    // The CLI flag can only turn offline on, so the environment must never be
    // able to turn it back off.
    let resolved = with_offline_env(Some("0"), || {
        let mut flag = true;
        env_fallback_offline(&mut flag);
        flag
    });
    assert!(resolved, "--offline must win over LLAMA_ARG_OFFLINE=0");
}
