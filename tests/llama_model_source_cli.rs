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

//! End-to-end behavior of the b10621 model-source flags on both server
//! binaries (issue #1434).
//!
//! The unit tests in `src/server/model_source_tests.rs` cover the decision
//! itself. This file covers the two things only the built binaries can answer:
//!
//! 1. An unsupported model source is refused *before* the model is loaded, on
//!    both `mlxcel serve` and `mlxcel-server`, with the diagnostic the issue
//!    asks for. Every case here would otherwise be a several-second load
//!    ending in an unrelated error, or worse, a successful start against a
//!    different checkpoint.
//! 2. A credential supplied through the environment never reaches `--help`.
//!    clap renders `[env: VAR=value]` with the *resolved* value by default, so
//!    `HF_TOKEN=... mlxcel-server --help` would print the token. Both
//!    credential-bearing flags set `hide_env_values`; this is the regression
//!    test for that.
//!
//! Every invocation here fails (or prints help) before any weight is read, so
//! the file needs no checkpoint and no network.

use std::path::Path;
use std::process::{Command, Output};

mod common;
use common::resolve_repo_binary;

/// Run one server binary and return its output, never inheriting a stray
/// `HF_TOKEN` / `LLAMA_ARG_*` from the developer's shell.
fn run(bin: &str, args: &[&str], env: &[(&str, &str)]) -> Output {
    let (path, resolution) = resolve_repo_binary(bin);
    let mut cmd = Command::new(&path);
    cmd.args(args);
    for key in [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "LLAMA_API_KEY",
        "LLAMA_ARG_MODEL",
        "LLAMA_ARG_HF_REPO",
        "LLAMA_ARG_HF_FILE",
        "LLAMA_ARG_MODEL_URL",
        "LLAMA_ARG_DOCKER_REPO",
        "LLAMA_ARG_OFFLINE",
        "LLAMA_ARG_ALIAS",
    ] {
        cmd.env_remove(key);
    }
    for (key, value) in env {
        cmd.env(key, value);
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin} from {path:?}: {e}\n{resolution}"))
}

/// Both server entry points, as (binary, leading args).
const ENTRY_POINTS: [(&str, &[&str]); 2] = [("mlxcel", &["serve"]), ("mlxcel-server", &[])];

/// Run `entry` with `extra` appended and return combined stdout+stderr,
/// asserting the process failed.
fn expect_startup_failure(entry: (&str, &[&str]), extra: &[&str]) -> String {
    let (bin, lead) = entry;
    let mut args: Vec<&str> = lead.to_vec();
    args.extend_from_slice(extra);
    let out = run(bin, &args, &[]);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "{bin} {args:?} unexpectedly succeeded; output: {text}"
    );
    text
}

// ── unsupported sources are refused before loading ──────────────────────────

#[test]
fn a_gguf_model_path_is_refused_on_both_binaries() {
    for entry in ENTRY_POINTS {
        let text = expect_startup_failure(entry, &["-m", "models/gemma-3-4b-it-Q4_K_M.gguf"]);
        assert!(
            text.contains("GGUF"),
            "{}: diagnostic must name the format: {text}",
            entry.0
        );
        assert!(
            text.contains("-m mlx-community/gemma-3-4b-it-4bit"),
            "{}: diagnostic must offer a concrete replacement: {text}",
            entry.0
        );
    }
}

#[test]
fn a_split_gguf_shard_is_refused_on_both_binaries() {
    for entry in ENTRY_POINTS {
        let text = expect_startup_failure(
            entry,
            &["-m", "models/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf"],
        );
        assert!(
            text.contains("split GGUF"),
            "{}: diagnostic must name the split: {text}",
            entry.0
        );
    }
}

#[test]
fn docker_repo_is_refused_on_both_binaries() {
    for entry in ENTRY_POINTS {
        let text = expect_startup_failure(entry, &["--docker-repo", "ai/gemma3"]);
        assert!(
            text.contains("ai/gemma3") && text.contains("--hf-repo mlx-community/"),
            "{}: diagnostic must quote the value and offer a replacement: {text}",
            entry.0
        );
    }
}

#[test]
fn model_url_is_refused_and_translates_a_huggingface_url() {
    for entry in ENTRY_POINTS {
        let text = expect_startup_failure(
            entry,
            &[
                "--model-url",
                "https://huggingface.co/mlx-community/Qwen3-4B-4bit",
            ],
        );
        assert!(
            text.contains("--hf-repo mlx-community/Qwen3-4B-4bit"),
            "{}: diagnostic must name the repo id to use: {text}",
            entry.0
        );
    }
}

#[test]
fn hf_file_is_refused_on_both_binaries() {
    for entry in ENTRY_POINTS {
        let text = expect_startup_failure(
            entry,
            &[
                "--hf-repo",
                "mlx-community/Qwen3-4B-4bit",
                "--hf-file",
                "model-Q4_K_M.gguf",
            ],
        );
        assert!(
            text.contains("model.safetensors.index.json"),
            "{}: diagnostic must explain MLX snapshot loading: {text}",
            entry.0
        );
    }
}

#[test]
fn an_hf_repo_quant_suffix_is_refused_on_both_binaries() {
    for entry in ENTRY_POINTS {
        let text =
            expect_startup_failure(entry, &["--hf-repo", "ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M"]);
        assert!(
            text.contains("Q4_K_M") && text.contains("--hf-repo mlx-community/"),
            "{}: diagnostic must quote the quant and offer a replacement: {text}",
            entry.0
        );
    }
}

#[test]
fn offline_mode_refuses_an_uncached_repo_without_downloading() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = tmp.path().join("store");
    let hf = tmp.path().join("hf");
    std::fs::create_dir_all(&store).expect("store");
    std::fs::create_dir_all(&hf).expect("hf");

    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.extend_from_slice(&[
            "-m",
            "mlx-community/definitely-not-cached-1434",
            "--offline",
        ]);
        let out = run(
            bin,
            &args,
            &[
                ("MLXCEL_CACHE_DIR", store.to_str().expect("utf-8")),
                ("HF_HUB_CACHE", hf.to_str().expect("utf-8")),
                ("HF_HOME", hf.to_str().expect("utf-8")),
            ],
        );
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!out.status.success(), "{bin}: expected failure: {text}");
        assert!(
            text.contains("offline mode is on"),
            "{bin}: must report the offline reason: {text}"
        );
        assert!(
            text.contains("mlxcel download"),
            "{bin}: must name the way to populate the cache: {text}"
        );
    }
}

#[test]
fn naming_no_model_source_at_all_lists_every_way_to_supply_one() {
    for entry in ENTRY_POINTS {
        let text = expect_startup_failure(entry, &[]);
        assert!(
            text.contains("--hf-repo"),
            "{}: must mention --hf-repo as an alternative to -m: {text}",
            entry.0
        );
    }
}

// ── credentials never reach --help ──────────────────────────────────────────

#[test]
fn help_never_renders_a_resolved_credential_from_the_environment() {
    const SECRET_TOKEN: &str = "hf_1434_secret_token_value";
    const SECRET_KEY: &str = "llama_1434_secret_api_key";

    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--help");
        let out = run(
            bin,
            &args,
            &[("HF_TOKEN", SECRET_TOKEN), ("LLAMA_API_KEY", SECRET_KEY)],
        );
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !text.contains(SECRET_TOKEN),
            "{bin} --help rendered the resolved HF_TOKEN value"
        );
        assert!(
            !text.contains(SECRET_KEY),
            "{bin} --help rendered the resolved LLAMA_API_KEY value"
        );
        // The flags themselves must still be discoverable.
        assert!(
            text.contains("--hf-token"),
            "{bin} --help must still document --hf-token"
        );
        assert!(
            text.contains("HF_TOKEN"),
            "{bin} --help must still name the HF_TOKEN binding"
        );
    }
}

// ── the compatibility-only flags stay out of the operator surface ───────────

#[test]
fn always_rejected_model_source_flags_are_hidden_from_help() {
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--help");
        let out = run(bin, &args, &[]);
        let help = String::from_utf8_lossy(&out.stdout);
        for hidden in ["--docker-repo", "--model-url", "--hf-file"] {
            assert!(
                !help.contains(hidden),
                "{bin} --help advertises {hidden}, which mlxcel always rejects; \
                 it must stay hidden so the help does not imply GGUF support"
            );
        }
        for visible in ["--hf-repo", "--offline"] {
            assert!(
                help.contains(visible),
                "{bin} --help must document {visible}"
            );
        }
    }
}

#[test]
fn help_never_presents_gguf_as_something_mlxcel_loads() {
    // mlxcel has no GGUF reader, so the operator-facing help must not imply
    // one. GGUF may still be *named*, but only where the surrounding text says
    // it is rejected or unsupported; that is what makes the boundary
    // discoverable instead of a surprise at startup.
    for (bin, lead) in ENTRY_POINTS {
        let mut args: Vec<&str> = lead.to_vec();
        args.push("--help");
        let out = run(bin, &args, &[]);
        let help = String::from_utf8_lossy(&out.stdout);
        for line in help.lines() {
            let lowered = line.to_ascii_lowercase();
            if !lowered.contains("gguf") {
                continue;
            }
            assert!(
                lowered.contains("rejected")
                    || lowered.contains("not supported")
                    || lowered.contains("cannot"),
                "{bin} --help names GGUF without saying it is unsupported, which \
                 implies a GGML backend mlxcel does not have:\n{line}"
            );
        }
    }
}

/// The binaries must exist where the harness looked; a silent skip here would
/// make every assertion above vacuous.
#[test]
fn both_server_binaries_are_resolvable() {
    for bin in ["mlxcel", "mlxcel-server"] {
        let (path, resolution) = resolve_repo_binary(bin);
        assert!(
            Path::new(&path).exists(),
            "{bin} not found at {path:?}\n{resolution}"
        );
    }
}
