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

//! Unit tests for the HuggingFace snapshot downloader.

use super::*;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
// Env-var-sensitive tests must serialize through the crate-wide `ENV_LOCK`
// without it, two test threads can call `setenv`/`getenv`
// concurrently — undefined behavior under Rust 2024's unsafe env contract.
use crate::test_support::env_lock::env_lock;

fn args(repo_id: &str) -> DownloadArgs {
    DownloadArgs {
        repo_id: repo_id.to_string(),
        local_dir: None,
        models_dir: None,
        revision: None,
        token: None,
        force: false,
    }
}

#[test]
fn repo_basename_strips_owner() {
    assert_eq!(
        repo_basename("mlx-community/Qwen3-4B-4bit"),
        "Qwen3-4B-4bit"
    );
    assert_eq!(
        repo_basename("meta-llama/Llama-3.1-8B-Instruct"),
        "Llama-3.1-8B-Instruct"
    );
    assert_eq!(repo_basename("gpt2"), "gpt2");
}

#[test]
fn default_local_dir_is_global_store() {
    // Issue #93: with no --local-dir, the default is the location-independent
    // global store at `${MLXCEL_CACHE_DIR}/models/<owner>/<name>` (not the
    // legacy per-CWD `models/<basename>`).
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_CACHE_DIR").ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", "/tmp/mlxcel-resolve-test");
    }
    let opts = DownloadOptions::from_args(&args("mlx-community/Qwen3-4B-4bit"));
    let resolved = opts.resolve_local_dir();
    restore_env("MLXCEL_CACHE_DIR", prev);

    assert_eq!(
        resolved,
        PathBuf::from("/tmp/mlxcel-resolve-test")
            .join("models")
            .join("mlx-community")
            .join("Qwen3-4B-4bit")
    );
}

#[test]
fn bare_name_download_dest_lands_under_default_org() {
    // Issue #171: a bare, prefix-less name normalizes to `mlx-community/<name>`
    // BEFORE the store destination is derived, so the snapshot lands at
    // `<cache>/models/mlx-community/<name>` — not the slashless `models/<bare>`.
    // Composes the shared `normalize_repo_id` with `resolve_local_dir` to pin
    // the download destination the `download` verb now produces (network-free).
    let _guard = env_lock();
    let prev_cache = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_org = std::env::var("MLXCEL_DEFAULT_ORG").ok();
    let prev_models = std::env::var("MLXCEL_MODELS_DIR").ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", "/tmp/mlxcel-171-dest-test");
        std::env::remove_var("MLXCEL_DEFAULT_ORG");
        std::env::remove_var("MLXCEL_MODELS_DIR");
    }

    let repo_id = normalize_repo_id("gemma-4-26B-A4B-it-qat-4bit").unwrap();
    let opts = DownloadOptions::from_args(&args(&repo_id));
    let resolved = opts.resolve_local_dir();

    restore_env("MLXCEL_CACHE_DIR", prev_cache);
    restore_env("MLXCEL_DEFAULT_ORG", prev_org);
    restore_env("MLXCEL_MODELS_DIR", prev_models);

    assert_eq!(
        resolved,
        PathBuf::from("/tmp/mlxcel-171-dest-test")
            .join("models")
            .join("mlx-community")
            .join("gemma-4-26B-A4B-it-qat-4bit")
    );
}

#[test]
fn explicit_local_dir_is_respected() {
    // An explicit --local-dir is the opt-out: it is honored verbatim and does
    // not consult MLXCEL_CACHE_DIR / the global store.
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_CACHE_DIR").ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", "/tmp/should-be-ignored");
    }
    let mut a = args("mlx-community/Qwen3-4B-4bit");
    a.local_dir = Some(PathBuf::from("/tmp/custom-dir"));
    let opts = DownloadOptions::from_args(&a);
    let resolved = opts.resolve_local_dir();
    restore_env("MLXCEL_CACHE_DIR", prev);

    assert_eq!(resolved, PathBuf::from("/tmp/custom-dir"));
}

#[test]
fn models_dir_override_places_snapshot_without_models_subdir() {
    // Issue #107: with --models-dir set (and no --local-dir), the snapshot
    // lands directly at `<models_dir>/<owner>/<name>` — NO `models/` subdir —
    // and the override beats MLXCEL_CACHE_DIR.
    let _guard = env_lock();
    let prev = std::env::var("MLXCEL_CACHE_DIR").ok();
    let prev_models = std::env::var("MLXCEL_MODELS_DIR").ok();
    unsafe {
        std::env::set_var("MLXCEL_CACHE_DIR", "/tmp/should-be-ignored");
        std::env::remove_var("MLXCEL_MODELS_DIR");
    }
    let mut a = args("mlx-community/Qwen3-4B-4bit");
    a.models_dir = Some(PathBuf::from("/data/store"));
    let opts = DownloadOptions::from_args(&a);
    let resolved = opts.resolve_local_dir();
    restore_env("MLXCEL_CACHE_DIR", prev);
    restore_env("MLXCEL_MODELS_DIR", prev_models);

    assert_eq!(
        resolved,
        PathBuf::from("/data/store")
            .join("mlx-community")
            .join("Qwen3-4B-4bit")
    );
}

#[test]
fn local_dir_wins_over_models_dir_override() {
    // Issue #107: --local-dir is the verbatim opt-out and retains ultimate
    // precedence for the download destination, even when --models-dir is set.
    let _guard = env_lock();
    let prev_models = std::env::var("MLXCEL_MODELS_DIR").ok();
    unsafe {
        std::env::remove_var("MLXCEL_MODELS_DIR");
    }
    let mut a = args("owner/model");
    a.local_dir = Some(PathBuf::from("/tmp/verbatim"));
    a.models_dir = Some(PathBuf::from("/data/store"));
    let opts = DownloadOptions::from_args(&a);
    let resolved = opts.resolve_local_dir();
    restore_env("MLXCEL_MODELS_DIR", prev_models);

    assert_eq!(resolved, PathBuf::from("/tmp/verbatim"));
}

#[test]
fn from_args_carries_all_fields() {
    let a = DownloadArgs {
        repo_id: "owner/repo".to_string(),
        local_dir: Some(PathBuf::from("/tmp/x")),
        models_dir: Some(PathBuf::from("/tmp/store")),
        revision: Some("v1".to_string()),
        token: Some("hf_xxx".to_string()),
        force: true,
    };
    let opts = DownloadOptions::from_args(&a);
    assert_eq!(opts.repo_id, "owner/repo");
    assert_eq!(opts.local_dir, Some(PathBuf::from("/tmp/x")));
    assert_eq!(opts.models_dir, Some(PathBuf::from("/tmp/store")));
    assert_eq!(opts.revision.as_deref(), Some("v1"));
    assert_eq!(opts.token.as_deref(), Some("hf_xxx"));
    assert!(opts.force);
}

#[test]
fn allow_list_includes_safetensors_and_index() {
    assert!(is_wanted_file("model.safetensors"));
    assert!(is_wanted_file("model-00001-of-00002.safetensors"));
    assert!(is_wanted_file("model.safetensors.index.json"));
}

#[test]
fn allow_list_includes_configs_and_tokenizer_files() {
    assert!(is_wanted_file("config.json"));
    assert!(is_wanted_file("generation_config.json"));
    assert!(is_wanted_file("tokenizer_config.json"));
    assert!(is_wanted_file("tokenizer.json"));
    assert!(is_wanted_file("tokenizer.model"));
    assert!(is_wanted_file("special_tokens_map.json"));
    assert!(is_wanted_file("preprocessor_config.json"));
    assert!(is_wanted_file("processor_config.json"));
    assert!(is_wanted_file("chat_template.json"));
    assert!(is_wanted_file("chat_template.jinja"));
    assert!(is_wanted_file("merges.txt"));
    assert!(is_wanted_file("vocab.json"));
    assert!(is_wanted_file("vocab.txt"));
}

#[test]
fn allow_list_includes_tiktoken() {
    assert!(is_wanted_file("tokenizer.tiktoken"));
    assert!(is_wanted_file("o200k_base.tiktoken"));
}

#[test]
fn deny_list_excludes_pytorch_and_other_formats() {
    assert!(!is_wanted_file("pytorch_model.bin"));
    assert!(!is_wanted_file("pytorch_model-00001-of-00002.bin"));
    assert!(!is_wanted_file("pytorch_model.bin.index.json"));
    assert!(!is_wanted_file("model.gguf"));
    assert!(!is_wanted_file("model.pt"));
    assert!(!is_wanted_file("model.h5"));
    assert!(!is_wanted_file("model.msgpack"));
    assert!(!is_wanted_file("model.onnx"));
}

#[test]
fn deny_list_excludes_docs_and_metadata() {
    assert!(!is_wanted_file("README.md"));
    assert!(!is_wanted_file("readme.md"));
    assert!(!is_wanted_file("LICENSE"));
    assert!(!is_wanted_file("license.txt"));
    assert!(!is_wanted_file("NOTICE"));
    assert!(!is_wanted_file("USAGE.md"));
    assert!(!is_wanted_file(".gitattributes"));
    assert!(!is_wanted_file(".huggingface/cache.lock"));
}

#[test]
fn deny_list_excludes_media() {
    assert!(!is_wanted_file("preview.png"));
    assert!(!is_wanted_file("logo.jpg"));
    assert!(!is_wanted_file("demo.mp4"));
    assert!(!is_wanted_file("sample.wav"));
}

#[test]
fn deny_list_excludes_source_code() {
    assert!(!is_wanted_file("modeling_custom.py"));
    assert!(!is_wanted_file("notebook.ipynb"));
    assert!(!is_wanted_file("config.yaml"));
    assert!(!is_wanted_file("pyproject.toml"));
}

#[test]
fn allow_list_handles_subdirectories() {
    // VLM repos sometimes ship vision configs under a subfolder.
    assert!(is_wanted_file("vision/preprocessor_config.json"));
    assert!(is_wanted_file("vision/model.safetensors"));
    assert!(!is_wanted_file("docs/USAGE.md"));
}

// ---------------------------------------------------------------------------
// Path-traversal regression tests for security finding C1 on.
//
// A malicious HuggingFace repo (or MitM) can return a sibling whose
// `rfilename` is `/etc/cron.d/evil` or `../../etc/passwd.json`. Before the
// fix the basename-only allow-list happily accepted those values and
// `PathBuf::join` silently anchored absolute paths outside `--local-dir`.
// These tests pin the rejection at the filter layer; the canonicalized
// `starts_with` guard inside `download_repo` is the defense-in-depth backup.
// ---------------------------------------------------------------------------

#[test]
fn deny_list_rejects_absolute_paths() {
    assert!(!is_wanted_file("/etc/passwd"));
    assert!(!is_wanted_file("/tmp/evil.json"));
    assert!(!is_wanted_file("/var/lib/foo.safetensors"));
}

#[test]
fn deny_list_rejects_parent_traversal() {
    assert!(!is_wanted_file("../etc/passwd.json"));
    assert!(!is_wanted_file("subdir/../../etc/passwd.json"));
    assert!(!is_wanted_file("./model.safetensors"));
    assert!(!is_wanted_file("foo/../bar.json"));
}

#[test]
fn deny_list_rejects_backslash_separators() {
    assert!(!is_wanted_file("..\\..\\evil.json"));
    assert!(!is_wanted_file("subdir\\evil.safetensors"));
}

#[test]
fn deny_list_rejects_empty_or_dot_components() {
    assert!(!is_wanted_file(""));
    assert!(!is_wanted_file("./"));
    assert!(!is_wanted_file("foo//bar.json"));
}

#[test]
fn allow_list_still_accepts_normal_subdir_paths() {
    // sentinel: ensure we didn't break legitimate nested files
    assert!(is_wanted_file("config.json"));
    assert!(is_wanted_file("subdir/tokenizer.json"));
    assert!(is_wanted_file("model-00001-of-00002.safetensors"));
}

/// Live PoC: simulate a malicious `info.siblings` payload and assert that
/// every adversarial `rfilename` is rejected by `is_wanted_file` before any
/// IO can occur. This mirrors the exploit shape the security checker
/// empirically verified on.
#[test]
fn malicious_siblings_payload_is_filtered_out() {
    let malicious_siblings = vec![
        // Absolute paths that would bypass `local_dir.join(...)` entirely.
        "/etc/passwd",
        "/etc/cron.d/evil",
        "/tmp/evil.json",
        "/home/user/.ssh/authorized_keys",
        // Parent-traversal payloads.
        "../etc/passwd.json",
        "../../../home/user/.ssh/authorized_keys",
        "../../etc/passwd.json",
        "subdir/../../etc/passwd.json",
        // Windows-style separator smuggling.
        "..\\..\\evil.json",
    ];
    for rfilename in &malicious_siblings {
        assert!(
            !is_wanted_file(rfilename),
            "is_wanted_file({rfilename:?}) must return false, got true"
        );
    }

    // And confirm a benign payload still survives the same filter pass.
    let benign = vec![
        "config.json",
        "model.safetensors",
        "vision/preprocessor_config.json",
    ];
    for rfilename in &benign {
        assert!(
            is_wanted_file(rfilename),
            "is_wanted_file({rfilename:?}) must return true, got false"
        );
    }
}

#[test]
fn token_resolution_explicit_wins() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("HF_TOKEN", "from-env");
    }
    let resolved = resolve_token(Some("explicit"));
    assert_eq!(resolved.as_deref(), Some("explicit"));
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

#[test]
fn token_resolution_hf_token_env() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("HF_TOKEN", "tok-hf");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
    }
    let resolved = resolve_token(None);
    assert_eq!(resolved.as_deref(), Some("tok-hf"));
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

#[test]
fn token_resolution_falls_back_to_alt_env() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::remove_var("HF_TOKEN");
        std::env::set_var("HUGGING_FACE_HUB_TOKEN", "tok-alt");
    }
    let resolved = resolve_token(None);
    assert_eq!(resolved.as_deref(), Some("tok-alt"));
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

#[test]
fn token_resolution_anonymous_when_unset() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
    }
    assert_eq!(resolve_token(None), None);
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

#[test]
fn token_resolution_treats_empty_as_anonymous() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("HF_TOKEN", "");
        std::env::set_var("HUGGING_FACE_HUB_TOKEN", "  ");
    }
    assert_eq!(resolve_token(Some("  ")), None);
    assert_eq!(resolve_token(None), None);
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
}

fn restore_env(key: &str, prev: Option<String>) {
    unsafe {
        match prev {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

#[test]
fn snapshot_complete_requires_config_json() {
    let dir = tempfile::tempdir().unwrap();
    let wanted = vec!["config.json".to_string(), "model.safetensors".to_string()];
    // Missing both files.
    assert!(!snapshot_complete(dir.path(), &wanted));

    // config.json present but other file missing.
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    assert!(!snapshot_complete(dir.path(), &wanted));

    // All files present and non-empty.
    std::fs::write(dir.path().join("model.safetensors"), b"shard").unwrap();
    assert!(snapshot_complete(dir.path(), &wanted));
}

#[test]
fn snapshot_complete_rejects_zero_byte_files() {
    let dir = tempfile::tempdir().unwrap();
    let wanted = vec!["config.json".to_string(), "model.safetensors".to_string()];
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"").unwrap();
    assert!(!snapshot_complete(dir.path(), &wanted));
}

/// Verify that `stream_file` removes the partial tempfile on error.
///
/// A server that immediately drops the TCP connection after accept triggers a
/// reqwest stream error, which should cause `stream_file` to clean up the
/// `.mlxcel-partial.*` file it created (or attempted to create) before
/// propagating the error.
#[tokio::test]
async fn stream_file_cleans_up_tempfile_on_error() {
    use tokio::net::TcpListener;
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.safetensors");

    // Spawn a server that immediately drops connections — induces a stream error.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        if let Ok((stream, _)) = listener.accept().await {
            drop(stream);
        }
    });

    let client = reqwest::Client::new();
    let url = format!("http://{addr}/x");
    let mp = indicatif::MultiProgress::with_draw_target(indicatif::ProgressDrawTarget::hidden());
    let file_pb = mp.add(indicatif::ProgressBar::hidden());
    let agg_pb = mp.add(indicatif::ProgressBar::hidden());

    let result = stream_file(&client, &url, &dest, "model.safetensors", &file_pb, &agg_pb).await;
    assert!(
        result.is_err(),
        "expected stream_file to return Err on dropped connection"
    );

    // Crucially: no tempfile should remain after the error.
    let leftover: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(".mlxcel-partial.")
        })
        .collect();
    assert!(
        leftover.is_empty(),
        "expected no tempfile leftover, found {leftover:?}"
    );
}

// Integration test that hits the real Hugging Face Hub. Marked `#[ignore]`
// per the issue acceptance criteria so CI does not depend on network access.
#[test]
#[ignore]
fn live_download_smoke_test() {
    // Use a tiny repo so the test stays cheap when explicitly requested with
    // `cargo test -- --ignored`.
    let dir = tempfile::tempdir().unwrap();
    let opts = DownloadOptions {
        repo_id: "hf-internal-testing/tiny-random-gpt2".to_string(),
        local_dir: Some(dir.path().to_path_buf()),
        models_dir: None,
        revision: None,
        token: None,
        force: true,
    };
    download_repo(opts).expect("live download should succeed");
    assert!(dir.path().join("config.json").exists());
}

// ---------------------------------------------------------------------------
// Hardening tests
// ---------------------------------------------------------------------------

/// L1 — Adversarial filenames containing reserved URL characters must be
/// percent-encoded when composing the GET/HEAD URL so they cannot smuggle a
/// query string (`?`) or fragment (`#`) past the request.
#[test]
fn file_url_percent_encodes_reserved_characters_in_filename() {
    let endpoint = "https://huggingface.co";
    let url = file_url(
        endpoint,
        "mlx-community/Qwen3-4B-4bit",
        "main",
        "model.safetensors?foo=bar.safetensors",
    );
    // The `?` MUST be encoded so the HF backend treats the whole string as a
    // single path segment and not as a query string.
    assert!(
        url.contains("%3F"),
        "expected `?` to be percent-encoded as %3F in URL, got: {url}"
    );
    // And the literal `?` MUST NOT appear in the URL after the resolve/<rev>/
    // segment.
    assert!(
        !url.contains("safetensors?foo"),
        "raw `?` leaked into URL: {url}"
    );
    // Similarly for `#`.
    let url_hash = file_url(endpoint, "owner/repo", "main", "evil#fragment.json");
    assert!(url_hash.contains("%23"));
    assert!(!url_hash.contains("evil#fragment"));
}

#[test]
fn file_url_preserves_safe_subdir_paths() {
    // Legitimate nested filenames like `vision/preprocessor_config.json`
    // must still produce a normal URL — only segment-internal reserved chars
    // get encoded, not the inter-segment `/`.
    let url = file_url(
        "https://huggingface.co",
        "mlx-community/Qwen3-4B-4bit",
        "main",
        "vision/preprocessor_config.json",
    );
    assert_eq!(
        url,
        "https://huggingface.co/mlx-community/Qwen3-4B-4bit/resolve/main/vision/preprocessor_config.json"
    );
}

#[test]
fn file_url_encodes_repo_id_and_revision() {
    // Defensive: even though `repo_id` and `revision` are CLI-controlled,
    // encode them for consistency so a stray space in `--revision` does not
    // break the URL.
    let url = file_url(
        "https://huggingface.co",
        "owner/repo with space",
        "branch with space",
        "model.safetensors",
    );
    assert!(
        url.contains("%20"),
        "expected space to be percent-encoded as %20, got: {url}"
    );
}

/// L3 — Tokens containing non-ASCII or control characters must produce a
/// clean `anyhow::Error` that names the issue, not a panic.
#[test]
fn download_repo_rejects_non_ascii_token() {
    let _env_guard = env_lock();
    let prev_hf = std::env::var("HF_TOKEN").ok();
    let prev_alt = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();
    let prev_endpoint = std::env::var("HF_ENDPOINT").ok();
    let prev_optout = std::env::var("MLXCEL_ALLOW_INSECURE_ENDPOINT").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("HF_TOKEN", "hé");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
        std::env::remove_var("HF_ENDPOINT");
        std::env::remove_var("MLXCEL_ALLOW_INSECURE_ENDPOINT");
    }
    // Resolve goes through the same path `download_repo` uses; assert the
    // validator catches the bad char before it can panic in
    // `HeaderValue::from_str`.
    let tok = resolve_token(None).expect("HF_TOKEN should resolve");
    let err = validate_token(&tok).expect_err("non-ASCII token must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("invalid characters"),
        "expected error to mention 'invalid characters', got: {msg}"
    );
    restore_env("HF_TOKEN", prev_hf);
    restore_env("HUGGING_FACE_HUB_TOKEN", prev_alt);
    restore_env("HF_ENDPOINT", prev_endpoint);
    restore_env("MLXCEL_ALLOW_INSECURE_ENDPOINT", prev_optout);
}

#[test]
fn validate_token_rejects_control_chars() {
    // CR / LF embedded in a token would let an attacker smuggle header
    // injection if the validator did not reject control chars.
    let err =
        validate_token("hf_token\r\nX-Evil: yes").expect_err("control chars must be rejected");
    assert!(err.to_string().contains("invalid characters"));

    let err_tab = validate_token("hf_token\twith_tab").expect_err("tab is a control char");
    assert!(err_tab.to_string().contains("invalid characters"));

    // And confirm the happy path: a normal ASCII token passes.
    assert!(validate_token("hf_AbCdEf-_.123").is_ok());
}

/// M1 — Plaintext `HF_ENDPOINT` + token combination is refused unless the
/// operator explicitly opts out.
#[test]
fn require_secure_endpoint_refuses_plaintext_with_token() {
    // For the http+token case, `require_secure_endpoint_for_token` reads
    // MLXCEL_ALLOW_INSECURE_ENDPOINT (via `is_insecure_endpoint_opt_out`).
    // Serialize through the crate-wide ENV_LOCK so this read cannot observe the
    // value mid-window from the opt-out tests, which set that var under the lock.
    let _env_guard = env_lock();
    let err = require_secure_endpoint_for_token("http://mirror.internal/", Some("hf_xxx"))
        .expect_err("plaintext endpoint with token must error");
    let msg = err.to_string();
    assert!(
        msg.contains("HTTPS") || msg.contains("https"),
        "expected error to mention HTTPS, got: {msg}"
    );
    assert!(
        msg.contains("MLXCEL_ALLOW_INSECURE_ENDPOINT"),
        "expected error to mention the opt-out env var, got: {msg}"
    );
}

#[test]
fn require_secure_endpoint_allows_https_with_token() {
    require_secure_endpoint_for_token("https://huggingface.co", Some("hf_xxx"))
        .expect("HTTPS + token must succeed");
    // Case-insensitive scheme check.
    require_secure_endpoint_for_token("HTTPS://huggingface.co", Some("hf_xxx"))
        .expect("HTTPS (uppercase) + token must succeed");
}

#[test]
fn require_secure_endpoint_allows_plaintext_anonymous() {
    // No token = no exposure, plaintext is fine.
    require_secure_endpoint_for_token("http://mirror.internal/", None)
        .expect("anonymous + plaintext must succeed");
}

#[test]
fn require_secure_endpoint_honors_opt_out() {
    let _env_guard = env_lock();
    let prev = std::env::var("MLXCEL_ALLOW_INSECURE_ENDPOINT").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("MLXCEL_ALLOW_INSECURE_ENDPOINT", "1");
    }
    let result = require_secure_endpoint_for_token("http://mirror.internal/", Some("hf_xxx"));
    assert!(
        result.is_ok(),
        "opt-out env var must allow plaintext + token"
    );
    restore_env("MLXCEL_ALLOW_INSECURE_ENDPOINT", prev);
}

#[test]
fn require_secure_endpoint_treats_empty_opt_out_as_unset() {
    let _env_guard = env_lock();
    let prev = std::env::var("MLXCEL_ALLOW_INSECURE_ENDPOINT").ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var("MLXCEL_ALLOW_INSECURE_ENDPOINT", "  ");
    }
    let result = require_secure_endpoint_for_token("http://mirror.internal/", Some("hf_xxx"));
    assert!(
        result.is_err(),
        "whitespace-only opt-out must NOT allow plaintext + token"
    );
    restore_env("MLXCEL_ALLOW_INSECURE_ENDPOINT", prev);
}

/// A long-lived (100-year), self-signed test-only CA certificate. Generated
/// with `openssl req -x509 -newkey rsa:2048 -nodes -days 36500 -subj
/// "/CN=mlxcel-test-ca-1"`. Used only to exercise PEM parsing — no private
/// key is checked in, and nothing in these tests ever performs a TLS
/// handshake with it.
const TEST_CA_CERT_1: &str = "-----BEGIN CERTIFICATE-----
MIIDGTCCAgGgAwIBAgIUSsmj23NSs41Xai2FaftQRBoeu5MwDQYJKoZIhvcNAQEL
BQAwGzEZMBcGA1UEAwwQbWx4Y2VsLXRlc3QtY2EtMTAgFw0yNjA3MjQwNTA2NDha
GA8yMTI2MDYzMDA1MDY0OFowGzEZMBcGA1UEAwwQbWx4Y2VsLXRlc3QtY2EtMTCC
ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANC/6D66VKOploIqa3o9FRXL
fT55ETuw9pUCynK4GztDt+9Lc57feIu7ki8i2Roy8OLATn2N075Ln3+dYLBhCTtE
mJTLswEVZMgpYPLktiad/M9ks16YlpCg6tWYR88JFw7j6SyWraw2joAt0bR+3DTu
b+BYfRxBnWiA9yoUGiqX5LSKeK2rSB/t6A1by1Y63j34l+pW8K5maHp3Hl5adyM1
Eh8G9Waiz+WDJOKsofic5WasC8d1W+6he5HEHrWxqQKjgruJBAiKka5Ewyiv0IFP
llc7cjWafesnT15pClUH7fKKVyAn1WWWCpQkTLF3tme0uYJHbZysmWdsyhRhnEUC
AwEAAaNTMFEwHQYDVR0OBBYEFC6X73a0ymZD5gx2E5+EBwFc3pK6MB8GA1UdIwQY
MBaAFC6X73a0ymZD5gx2E5+EBwFc3pK6MA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZI
hvcNAQELBQADggEBAHQzGfPR39QYNg7dN/GnviwTYUiqHdE6xPTaX/sNeSWpIGPv
rVMHK5QrvAWHAigYrLPvW7P/CD/ltF+xKkREts1r0ZGuytzRyAyuQZ6NrAul+mg8
DHDXuMZB+uOHcJXB+EtKdgkWCB/xYrtQIkf/ES/+/JM3UDhDTg1ZYn0m5WhBwo3n
lnq2hW9ri3bXshuhUTsLTl/7FgwExbsWQt5gIgQJ2k3a2Pk2mkyCsebAIHBgoK/o
svCeox2DoJWa72nsN7CY/Y8y01ssrgt3H9goUXIq+L/NdLEA/gV98i226aAovvJr
qxK4mrWI36bI9ufQFtB9cVRnFicTP0me/6wxd8A=
-----END CERTIFICATE-----
";

/// A second, distinct self-signed test-only CA certificate (same generation
/// recipe as [`TEST_CA_CERT_1`] with `/CN=mlxcel-test-ca-2`), used to prove
/// that a bundle file containing more than one PEM block loads all of them.
const TEST_CA_CERT_2: &str = "-----BEGIN CERTIFICATE-----
MIIDGTCCAgGgAwIBAgIUX3xujdtXfMIvyHOIiIj8n2P9pJwwDQYJKoZIhvcNAQEL
BQAwGzEZMBcGA1UEAwwQbWx4Y2VsLXRlc3QtY2EtMjAgFw0yNjA3MjQwNTA2NDha
GA8yMTI2MDYzMDA1MDY0OFowGzEZMBcGA1UEAwwQbWx4Y2VsLXRlc3QtY2EtMjCC
ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMZ1xGjVqjbf5GXnPEAB2Scg
Uqur9WGP7YK41B0vST18SP2lstYNhUAiIbKUzDCkO3+YbMT7zI0xumqHkJcq+AvV
k9PJY2IeoIDjNNw2t0HlDLyMupXH3RTDuZJapRWUYGzBpH7r7JXKG86DklVqqpBN
4qCzfhVEEJ9WwQufmehJw6TmDAvE3k3K09I/SpbGEtscQdmBMT6EmTmr5CRTTrPH
3vB5j4PkiOc6O1RIpq6klr+db0nVjz328LLnFItqKWdDAKuWYH2Av5HFg8p1SKnJ
tgtO1ONA0I90BVbSSiH8Dio9R6uexoyHURhf+IR71fYUnnSOJhF/j3iBspJqL0UC
AwEAAaNTMFEwHQYDVR0OBBYEFOQVKDDfg6hO+prghwAgjdY2KIYhMB8GA1UdIwQY
MBaAFOQVKDDfg6hO+prghwAgjdY2KIYhMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZI
hvcNAQELBQADggEBAKgoAdxgPHr11Ckwrs0KqtTl0kf8Z4VPevIQrWnTU2g781bV
sZXcduAMbUM+F1oaoJinwo7zCCSlxRp3CZKulgq0GrfvYMapd7ca7I+MjZUhCyjm
MoXKu22e+/AWFD5Ql9l1klM4SBtMxBzUOYWp9BM1lOgIwJJ9wpsA4g+mimPMgaPJ
PnWh4xWx2xaxRAMcXo/5slrv8NvLGqbz+/V5643ZV3S9VdQItCo2FG9ShORfqjG+
LMuIM76ng7Siyp3OtQIcLvHxpbui5eG5bBzXqq5fT7PNIW55gajp/sV7iEA+JZ4C
fglnm3e/hZfNMax6H1w0/OLWUBaHU0AaJh5zNAs=
-----END CERTIFICATE-----
";

#[test]
fn extra_ca_certs_defaults_to_empty_when_unset() {
    let _env_guard = env_lock();
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::remove_var(EXTRA_CA_CERTS_ENV);
    }
    let certs = load_extra_ca_certificates().expect("unset env var must not error");
    assert!(
        certs.is_empty(),
        "expected no extra certs when var is unset"
    );
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

#[test]
fn extra_ca_certs_treats_whitespace_value_as_unset() {
    let _env_guard = env_lock();
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var(EXTRA_CA_CERTS_ENV, "   ");
    }
    let certs = load_extra_ca_certificates().expect("whitespace-only value must not error");
    assert!(
        certs.is_empty(),
        "whitespace-only value must be treated as unset"
    );
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

#[test]
fn extra_ca_certs_loads_single_cert_from_file() {
    let _env_guard = env_lock();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ca.pem");
    std::fs::write(&path, TEST_CA_CERT_1).unwrap();
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var(EXTRA_CA_CERTS_ENV, path.to_str().unwrap());
    }
    let certs = load_extra_ca_certificates().expect("valid single-cert PEM must load");
    assert_eq!(certs.len(), 1, "expected exactly one certificate");
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

#[test]
fn extra_ca_certs_loads_multiple_certs_from_bundle() {
    let _env_guard = env_lock();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bundle.pem");
    std::fs::write(&path, format!("{TEST_CA_CERT_1}{TEST_CA_CERT_2}")).unwrap();
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var(EXTRA_CA_CERTS_ENV, path.to_str().unwrap());
    }
    let certs = load_extra_ca_certificates().expect("concatenated PEM bundle must load");
    assert_eq!(certs.len(), 2, "expected both certificates from the bundle");
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

#[test]
fn extra_ca_certs_errors_on_missing_file() {
    let _env_guard = env_lock();
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("does-not-exist.pem");
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var(EXTRA_CA_CERTS_ENV, missing.to_str().unwrap());
    }
    let err = load_extra_ca_certificates().expect_err("missing file must error, not silently pass");
    assert!(
        err.to_string().contains(EXTRA_CA_CERTS_ENV),
        "error should name the env var so the operator can find the misconfiguration: {err}"
    );
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

#[test]
fn extra_ca_certs_rejects_oversized_bundle_before_parsing() {
    let _env_guard = env_lock();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oversized.pem");
    std::fs::write(&path, vec![b'x'; MAX_EXTRA_CA_CERTS_BYTES + 1]).unwrap();
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var(EXTRA_CA_CERTS_ENV, path.to_str().unwrap());
    }
    let err = load_extra_ca_certificates()
        .expect_err("an oversized CA bundle must be rejected before parsing");
    assert!(
        err.to_string().contains("safety limit"),
        "error should explain the bounded-input failure: {err}"
    );
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

#[cfg(unix)]
#[test]
fn extra_ca_certs_errors_on_unreadable_file() {
    use std::os::unix::fs::PermissionsExt;

    // SAFETY: reading our own euid, no env/global state touched.
    if unsafe { libc::geteuid() } == 0 {
        // root bypasses Unix permission bits entirely, so this test would
        // spuriously pass/fail depending on the CI runner's privilege level.
        return;
    }

    let _env_guard = env_lock();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unreadable.pem");
    std::fs::write(&path, TEST_CA_CERT_1).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var(EXTRA_CA_CERTS_ENV, path.to_str().unwrap());
    }
    let err =
        load_extra_ca_certificates().expect_err("unreadable file must error, not silently pass");
    assert!(
        err.to_string().contains(EXTRA_CA_CERTS_ENV),
        "error should name the env var: {err}"
    );
    // Restore permissions so the tempdir's own Drop cleanup can remove it.
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

#[cfg(unix)]
#[test]
fn extra_ca_certs_errors_on_non_utf8_value() {
    use std::os::unix::ffi::OsStringExt;

    let _env_guard = env_lock();
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    let invalid = std::ffi::OsString::from_vec(vec![0xFF, 0xFE]);
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var(EXTRA_CA_CERTS_ENV, &invalid);
    }
    let err = load_extra_ca_certificates()
        .expect_err("non-UTF8 value must error, not be silently treated as unset");
    assert!(
        err.to_string().contains(EXTRA_CA_CERTS_ENV),
        "error should name the env var: {err}"
    );
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

#[test]
fn extra_ca_certs_errors_on_malformed_pem() {
    let _env_guard = env_lock();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("garbage.pem");
    std::fs::write(&path, "this is not a certificate").unwrap();
    let prev = std::env::var(EXTRA_CA_CERTS_ENV).ok();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe {
        std::env::set_var(EXTRA_CA_CERTS_ENV, path.to_str().unwrap());
    }
    let result = load_extra_ca_certificates();
    assert!(result.is_err(), "malformed PEM content must error");
    restore_env(EXTRA_CA_CERTS_ENV, prev);
}

/// L2 — Tempfile creation must fail closed when a symlink (or any pre-existing
/// path) already exists at the predicted tempfile location. `create_new(true)`
/// provides `O_CREAT|O_EXCL`, and on Unix we add `O_NOFOLLOW`. Either alone
/// would fail on a pre-staged symlink, but the test confirms the combined
/// behavior.
#[cfg(unix)]
#[tokio::test]
async fn open_tempfile_no_symlink_refuses_pre_staged_symlink() {
    let dir = tempfile::tempdir().unwrap();
    let target_dir = tempfile::tempdir().unwrap();
    let target_file = target_dir.path().join("victim.txt");
    std::fs::write(&target_file, b"original contents").unwrap();

    let tmp_path = dir.path().join(".mlxcel-partial.99999.0");
    // Pre-stage a symlink at the path mlxcel would write to.
    std::os::unix::fs::symlink(&target_file, &tmp_path)
        .expect("test setup: symlink creation should succeed");

    let result = open_tempfile_no_symlink(&tmp_path).await;
    assert!(
        result.is_err(),
        "open_tempfile_no_symlink should refuse a pre-staged symlink, got Ok"
    );

    // Crucially, the victim file behind the symlink MUST still contain the
    // original bytes — i.e., the open did not follow through and truncate it.
    let bytes = std::fs::read(&target_file).expect("victim should still be readable");
    assert_eq!(
        bytes, b"original contents",
        "victim file behind symlink was clobbered — O_NOFOLLOW / O_EXCL failed"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn open_tempfile_no_symlink_refuses_pre_existing_regular_file() {
    // `create_new(true)` semantics: even a regular file at the target path
    // must cause EEXIST.
    let dir = tempfile::tempdir().unwrap();
    let tmp_path = dir.path().join(".mlxcel-partial.99999.0");
    std::fs::write(&tmp_path, b"squatted").unwrap();

    let result = open_tempfile_no_symlink(&tmp_path).await;
    assert!(
        result.is_err(),
        "open_tempfile_no_symlink should refuse a pre-existing regular file"
    );

    // And the squatter contents are preserved.
    let bytes = std::fs::read(&tmp_path).unwrap();
    assert_eq!(bytes, b"squatted");
}

#[cfg(unix)]
#[tokio::test]
async fn open_tempfile_no_symlink_succeeds_on_clean_path() {
    let dir = tempfile::tempdir().unwrap();
    let tmp_path = dir.path().join(".mlxcel-partial.99999.0");
    let result = open_tempfile_no_symlink(&tmp_path).await;
    assert!(
        result.is_ok(),
        "open_tempfile_no_symlink should succeed on a clean path, got: {:?}",
        result.err()
    );
    // The file should now exist on disk.
    assert!(tmp_path.exists());
}

/// L5 — Stale `.mlxcel-partial.*` orphans older than the threshold are
/// removed at the start of `download_repo`. Younger partials are left alone.
#[test]
fn cleanup_stale_partials_removes_aged_orphans() {
    let dir = tempfile::tempdir().unwrap();

    // Create a stale partial (modified > 1 hour ago).
    let stale = dir.path().join(".mlxcel-partial.42.0");
    std::fs::write(&stale, b"orphan").unwrap();
    let stale_file = std::fs::File::options()
        .write(true)
        .open(&stale)
        .expect("re-open stale partial for set_modified");
    // set_modified is stable since 1.75.
    let old = SystemTime::now() - Duration::from_secs(2 * 60 * 60);
    stale_file
        .set_modified(old)
        .expect("set_modified backdates the mtime");
    drop(stale_file);

    // Create a fresh partial (modified just now) — must NOT be removed.
    let fresh = dir.path().join(".mlxcel-partial.43.0");
    std::fs::write(&fresh, b"in-flight").unwrap();

    // And a non-partial file — must always be left alone.
    let unrelated = dir.path().join("config.json");
    std::fs::write(&unrelated, b"{}").unwrap();

    cleanup_stale_partials(dir.path());

    assert!(
        !stale.exists(),
        "stale .mlxcel-partial.* > 1h must be removed"
    );
    assert!(
        fresh.exists(),
        "fresh .mlxcel-partial.* (< 1h) must NOT be removed"
    );
    assert!(unrelated.exists(), "unrelated files must NOT be removed");
}

#[test]
fn cleanup_stale_partials_handles_missing_dir_gracefully() {
    // Pass a path that does not exist — must not panic, must not crash.
    let bogus = PathBuf::from("/nonexistent/path/that/should/not/exist/anywhere");
    cleanup_stale_partials(&bogus);
    // (no assertion needed; if it panicked or aborted, the test runner notices)
}

#[test]
fn cleanup_stale_partials_ignores_directories() {
    let dir = tempfile::tempdir().unwrap();
    // Create a *directory* whose name matches the partial prefix; must not be
    // removed even when stale.
    let pseudo = dir.path().join(".mlxcel-partial.bogus.0");
    std::fs::create_dir(&pseudo).unwrap();
    cleanup_stale_partials(dir.path());
    assert!(pseudo.exists(), "must not remove directories");
}

/// L6 — `encode_path_segments` is a private helper but is exercised end-to-end
/// by the `file_url_*` tests above. This test sanity-checks the helper in
/// isolation.
#[test]
fn encode_path_segments_handles_empty_input() {
    assert_eq!(encode_path_segments(""), "");
    assert_eq!(encode_path_segments("/"), "/");
    assert_eq!(encode_path_segments("a/b/c"), "a/b/c");
    assert_eq!(encode_path_segments("a b/c d"), "a%20b/c%20d");
}

/// Regression for issue #463: `download_repo` must not abort with "Cannot
/// start a runtime from within a runtime" when called on a thread that is
/// already driving a Tokio runtime (the `mlxcel serve` / `mlxcel-server`
/// auto-download startup path). Pins that the runtime-aware guard offloads
/// the blocking body and surfaces a normal `Err`. The endpoint points at a
/// closed local port so the offloaded download fails fast without touching
/// the network.
#[tokio::test(flavor = "multi_thread")]
async fn download_repo_inside_runtime_errors_instead_of_aborting() {
    let _guard = env_lock();
    let prev_endpoint = std::env::var("HF_ENDPOINT").ok();
    let prev_token = std::env::var("HF_TOKEN").ok();
    unsafe {
        std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:9");
        // A token from the host environment would trip the plaintext-endpoint
        // refusal before the connection attempt; clear it so the failure mode
        // under test is the (fast, local) connection refusal.
        std::env::remove_var("HF_TOKEN");
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let opts = DownloadOptions {
        repo_id: "mlx-community/issue-463-regression-not-a-real-repo".to_string(),
        local_dir: Some(dir.path().to_path_buf()),
        models_dir: None,
        revision: None,
        token: None,
        force: true,
    };
    let result = download_repo(opts);

    unsafe {
        match prev_endpoint {
            Some(v) => std::env::set_var("HF_ENDPOINT", v),
            None => std::env::remove_var("HF_ENDPOINT"),
        }
        if let Some(v) = prev_token {
            std::env::set_var("HF_TOKEN", v);
        }
    }

    assert!(
        result.is_err(),
        "a closed endpoint must surface a normal error, not a nested-runtime abort"
    );
}
