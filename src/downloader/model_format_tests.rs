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

//! Unit tests for the unsupported-model-reference classifier (issue #1434).

use std::path::Path;

use super::*;

// ── classification ──────────────────────────────────────────────────────────

#[test]
fn plain_gguf_paths_are_classified_as_gguf() {
    for value in [
        "model.gguf",
        "models/gemma-3-4b-it-Q4_K_M.gguf",
        "/abs/path/Model.GGUF",
        "C:\\models\\model.gguf",
        "./model.gguf",
    ] {
        assert_eq!(
            classify_model_reference(value),
            Some(UnsupportedModelReference::Gguf),
            "{value} must be classified as a GGUF file"
        );
    }
}

#[test]
fn split_shard_names_are_classified_as_split() {
    for value in [
        "model-00001-of-00003.gguf",
        "models/DeepSeek-V3-Q4_K_M-00002-of-00009.gguf",
    ] {
        assert_eq!(
            classify_model_reference(value),
            Some(UnsupportedModelReference::GgufSplit),
            "{value} must be classified as a split GGUF shard"
        );
    }
}

#[test]
fn huggingface_urls_are_classified_separately_from_other_urls() {
    for value in [
        "https://huggingface.co/mlx-community/Qwen3-4B-4bit",
        "https://huggingface.co/models/mlx-community/Qwen3-4B-4bit",
        "https://hf.co/mlx-community/Qwen3-4B-4bit/resolve/main/model.safetensors",
        "hf://mlx-community/Qwen3-4B-4bit",
    ] {
        assert_eq!(
            classify_model_reference(value),
            Some(UnsupportedModelReference::HuggingFaceUrl),
            "{value} must be classified as a HuggingFace URL"
        );
    }
    for value in [
        "https://example.com/model.gguf",
        "docker://ai/gemma3",
        "oci://registry.example.com/model:latest",
        "ftp://host/model",
    ] {
        assert_eq!(
            classify_model_reference(value),
            Some(UnsupportedModelReference::Url),
            "{value} must be classified as a generic URL"
        );
    }
}

#[test]
fn mlx_references_are_not_classified_as_unsupported() {
    for value in [
        "models/Qwen3-4B-4bit",
        "mlx-community/Qwen3-4B-4bit",
        "Qwen3-4B-4bit",
        "/abs/path/to/checkpoint",
        "./relative/dir",
        "C:\\models\\Qwen3-4B-4bit",
        // A directory whose name merely contains "gguf" is not a GGUF file.
        "models/Qwen3-4B-GGUF-mlx",
        "",
    ] {
        assert_eq!(
            classify_model_reference(value),
            None,
            "{value} must remain resolvable"
        );
    }
}

#[test]
fn a_bare_dot_gguf_name_is_not_treated_as_a_file() {
    // `.gguf` alone is a dotfile, not `<name>.gguf`.
    assert_eq!(classify_model_reference(".gguf"), None);
}

// ── repo id extraction ──────────────────────────────────────────────────────

#[test]
fn huggingface_repo_is_extracted_from_every_supported_url_shape() {
    for value in [
        "https://huggingface.co/mlx-community/Qwen3-4B-4bit",
        "https://huggingface.co/mlx-community/Qwen3-4B-4bit/",
        "https://huggingface.co/mlx-community/Qwen3-4B-4bit/tree/main",
        "https://www.huggingface.co/models/mlx-community/Qwen3-4B-4bit",
        "https://hf.co/mlx-community/Qwen3-4B-4bit?download=true",
        "hf://mlx-community/Qwen3-4B-4bit",
    ] {
        assert_eq!(
            huggingface_repo_from_url(value).as_deref(),
            Some("mlx-community/Qwen3-4B-4bit"),
            "{value} must yield the repo id"
        );
    }
}

#[test]
fn non_huggingface_urls_yield_no_repo_id() {
    for value in [
        "https://example.com/mlx-community/Qwen3-4B-4bit",
        "https://huggingface.co/only-owner",
        "docker://ai/gemma3",
        "not-a-url",
    ] {
        assert_eq!(
            huggingface_repo_from_url(value),
            None,
            "{value} must not yield a repo id"
        );
    }
}

// ── diagnostics ─────────────────────────────────────────────────────────────

#[test]
fn gguf_diagnostic_names_the_value_the_reason_and_a_replacement() {
    let err = ensure_mlx_model_reference(Path::new("models/gemma-3-4b-it-Q4_K_M.gguf"))
        .expect_err("a GGUF path must be rejected");
    let text = format!("{err}");
    assert!(
        text.contains("models/gemma-3-4b-it-Q4_K_M.gguf"),
        "diagnostic must quote the requested value: {text}"
    );
    assert!(
        text.contains("GGUF"),
        "diagnostic must name the format: {text}"
    );
    assert!(
        text.contains("-m mlx-community/gemma-3-4b-it-4bit"),
        "diagnostic must offer a concrete MLX replacement: {text}"
    );
}

#[test]
fn split_diagnostic_says_splits_are_not_reassembled() {
    let err =
        ensure_mlx_model_reference(Path::new("models/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf"))
            .expect_err("a split GGUF shard must be rejected");
    let text = format!("{err}");
    assert!(
        text.contains("split GGUF"),
        "diagnostic must name the split: {text}"
    );
    assert!(
        text.contains("-m mlx-community/DeepSeek-V3-4bit"),
        "diagnostic must offer a concrete MLX replacement: {text}"
    );
}

#[test]
fn huggingface_url_diagnostic_offers_the_exact_repo_id() {
    let err = ensure_mlx_model_reference(Path::new(
        "https://huggingface.co/mlx-community/Qwen3-4B-4bit/tree/main",
    ))
    .expect_err("a HuggingFace URL must be rejected");
    let text = format!("{err}");
    assert!(
        text.contains("-m mlx-community/Qwen3-4B-4bit"),
        "diagnostic must offer the repo id itself: {text}"
    );
}

#[test]
fn generic_url_diagnostic_mentions_registries() {
    let err = ensure_mlx_model_reference(Path::new("docker://ai/gemma3"))
        .expect_err("a docker URL must be rejected");
    let text = format!("{err}");
    assert!(
        text.contains("Docker") && text.contains("OCI"),
        "diagnostic must name the registry shapes it refuses: {text}"
    );
}

#[test]
fn supported_references_pass_the_gate() {
    for value in [
        "models/Qwen3-4B-4bit",
        "mlx-community/Qwen3-4B-4bit",
        "Qwen3-4B-4bit",
    ] {
        ensure_mlx_model_reference(Path::new(value))
            .unwrap_or_else(|e| panic!("{value} must pass the format gate: {e}"));
    }
}

// ── basename recovery ───────────────────────────────────────────────────────

#[test]
fn quant_tags_and_split_suffixes_are_stripped_from_the_suggested_name() {
    for (value, expected) in [
        ("Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B"),
        ("Qwen3-4B-IQ4_NL.gguf", "Qwen3-4B"),
        ("Qwen3-4B-f16.gguf", "Qwen3-4B"),
        ("Qwen3-4B-GGUF.gguf", "Qwen3-4B"),
        ("Qwen3-4B-Q8_0-00001-of-00002.gguf", "Qwen3-4B"),
        ("Qwen3-4B.gguf", "Qwen3-4B"),
    ] {
        let err = ensure_mlx_model_reference(Path::new(value)).expect_err("rejected");
        let text = format!("{err}");
        assert!(
            text.contains(&format!("-m mlx-community/{expected}-4bit")),
            "{value} should suggest {expected}: {text}"
        );
    }
}

// ── credential redaction (issue #1434) ──────────────────────────────────────

#[test]
fn a_urls_userinfo_is_redacted_before_it_reaches_a_diagnostic() {
    for (value, secret) in [
        ("https://alice:hunter2@example.com/model.gguf", "hunter2"),
        ("https://token@hf.co/owner/name", "token@"),
        ("ftp://user:pw@host/model", "pw"),
    ] {
        let redacted = redact_url_userinfo(value);
        assert!(
            !redacted.contains(secret),
            "{value} still leaks {secret}: {redacted}"
        );
        assert!(redacted.contains("***@"), "{value} -> {redacted}");
    }
}

#[test]
fn redaction_leaves_a_credential_free_value_untouched() {
    for value in [
        "https://huggingface.co/owner/name",
        "models/Qwen3-4B-4bit",
        "owner/name",
        "",
        // A `@` in the path is not userinfo.
        "https://example.com/a@b/model.gguf",
    ] {
        assert_eq!(
            redact_url_userinfo(value),
            value,
            "{value} must be unchanged"
        );
    }
}

#[test]
fn the_url_diagnostic_does_not_echo_a_credential() {
    let err = ensure_mlx_model_reference(Path::new("https://alice:hunter2@example.com/model.gguf"))
        .expect_err("a URL must be rejected");
    let text = format!("{err}");
    assert!(!text.contains("hunter2"), "credential leaked: {text}");
}

// ── the gate never touches a real directory ─────────────────────────────────

#[test]
fn a_directory_named_like_a_gguf_file_still_resolves() {
    // The gate is otherwise purely syntactic and runs before the resolver's
    // existing-path branch, so without the is_dir() guard an MLX checkpoint
    // directory someone named `mymodel.gguf` would stop loading.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("mymodel.gguf");
    std::fs::create_dir_all(&dir).unwrap();
    ensure_mlx_model_reference(&dir).expect("a real directory is never a GGUF artifact");
}

#[test]
fn a_real_gguf_file_is_still_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("mymodel.gguf");
    std::fs::write(&file, b"GGUF").unwrap();
    ensure_mlx_model_reference(&file).expect_err("a real GGUF file must be rejected");
}

// ── diagnostics are readable ────────────────────────────────────────────────

#[test]
fn no_format_diagnostic_carries_a_run_of_collapsed_indentation() {
    for value in [
        "model.gguf",
        "model-00001-of-00003.gguf",
        "https://huggingface.co/owner/name",
        "docker://ai/gemma3",
    ] {
        let err = ensure_mlx_model_reference(Path::new(value)).expect_err("rejected");
        for line in format!("{err}").lines() {
            assert!(
                !line.trim().contains("   "),
                "diagnostic line carries collapsed indentation: {line:?}"
            );
        }
    }
}
