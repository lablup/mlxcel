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

//! End-to-end integration tests for the `--surgery <config.yaml>` CLI
//! flag (Epic #363, issue #371 — A4).
//!
//! These tests exercise the full plumbing from clap argument parsing
//! through `crate::surgery::set_active_pipeline` into the consolidated
//! weight loaders (`load_text_weights` for text models,
//! `load_vlm_weights_common` for VLMs). They are gated behind the
//! `surgery` cargo feature and skip automatically when the small
//! reference model is not present on disk.
//!
//! Tested invariants:
//!
//! - With `--surgery <empty.yaml>` and an empty `operations: []`
//!   pipeline, model loading completes successfully and the generated
//!   tokens are byte-identical to the baseline (no `--surgery`) case.
//!   This is the contract from acceptance criterion (b) and (e).
//! - With a malformed YAML file the binary fails fast before any model
//!   weight is touched (acceptance criterion (a)).

#![cfg(feature = "surgery")]

use std::path::PathBuf;
use std::process::Command;

/// Reference model directory used for the bit-exactness check. Chosen
/// because it is small (~0.5 B parameters), quantized (no bf16
/// conversion to worry about), and is one of the recommended test
/// models in `AGENTS.md`.
const REFERENCE_MODEL: &str = "models/qwen2.5-0.5b-4bit";

/// Locate the freshly built `mlxcel` binary under `target/`.
///
/// `CARGO_BIN_EXE_<name>` is set by cargo for `[[bin]]` integration
/// tests automatically. The cargo-test harness builds the binary first
/// so the path is guaranteed to exist when this test runs.
fn mlxcel_binary() -> PathBuf {
    env!("CARGO_BIN_EXE_mlxcel").into()
}

/// Compose an absolute path inside the cargo workspace.
fn workspace_path(relative: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(relative);
    path
}

/// Skip the test when the reference model is not present on disk.
///
/// Returns the resolved absolute path when available, or `None` to
/// instruct the caller to short-circuit. Mirrors the skipping pattern
/// in `tests/model_loading.rs`.
fn locate_reference_model() -> Option<PathBuf> {
    let model = workspace_path(REFERENCE_MODEL);
    if model.exists() {
        return Some(model);
    }
    eprintln!(
        "Skipping surgery_cli test: reference model not found at {}\n  \
         Hint: download with `huggingface-cli download \
         mlx-community/Qwen2.5-0.5B-Instruct-4bit --local-dir {}`",
        model.display(),
        model.display()
    );
    None
}

/// Skip the end-to-end model-loading test when the test harness asks
/// us to (e.g. on a CPU-only Linux CI runner where each `mlxcel
/// generate` invocation takes minutes and would exceed reasonable
/// test timeouts).
///
/// Operators set `MLXCEL_SKIP_HEAVY_TESTS=1` in the test runner's
/// environment. The fail-fast tests still run because they do not
/// touch model weights.
fn skip_heavy_test_via_env() -> bool {
    matches!(
        std::env::var("MLXCEL_SKIP_HEAVY_TESTS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes")
    )
}

/// Helper: write an empty-but-valid surgery YAML config into a fresh
/// tempdir and return both the directory (so cleanup happens via
/// `drop`) and the YAML path.
fn write_empty_pipeline_yaml() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir for surgery yaml");
    let path = dir.path().join("surgery.yaml");
    std::fs::write(&path, "version: 1\noperations: []\n").expect("write yaml");
    (dir, path)
}

/// Helper: write a YAML file that is *not* a valid surgery config.
/// Used to confirm the CLI surfaces the parse error before touching
/// any model weight.
fn write_malformed_yaml() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir for malformed yaml");
    let path = dir.path().join("bad.yaml");
    // Schema version 99 is rejected by parse_config_file.
    std::fs::write(&path, "version: 99\noperations: []\n").expect("write yaml");
    (dir, path)
}

/// Run `mlxcel generate` with the given extra args and return stdout
/// captured as a UTF-8 string plus the exit status.
fn run_generate(model: &PathBuf, extra: &[&str]) -> (String, std::process::ExitStatus) {
    let mut cmd = Command::new(mlxcel_binary());
    cmd.arg("generate")
        .arg("-m")
        .arg(model)
        .arg("-p")
        .arg("Hello")
        .arg("-n")
        .arg("20")
        .arg("--temp")
        .arg("0.0");
    for &a in extra {
        cmd.arg(a);
    }
    let output = cmd.output().expect("spawn mlxcel");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        output.status,
    )
}

/// Acceptance criterion (b): `mlxcel generate --surgery <empty.yaml>`
/// against a small real model completes the load pipeline successfully.
///
/// Acceptance criterion (e): without `--surgery`, output is byte-identical
/// to the same invocation with `--surgery <empty.yaml>` (an empty
/// pipeline is a no-op). The check here is a *strong* equivalence —
/// the surgery-active path must produce the same generated tokens as
/// the baseline path when the pipeline is empty. This indirectly proves
/// that the active-pipeline slot integration does not perturb load
/// behaviour, and that the empty `transform.apply` short-circuits to
/// zero work.
#[test]
fn empty_surgery_pipeline_matches_baseline_output() {
    if skip_heavy_test_via_env() {
        eprintln!(
            "Skipping empty_surgery_pipeline_matches_baseline_output: \
             MLXCEL_SKIP_HEAVY_TESTS is set"
        );
        return;
    }
    let Some(model) = locate_reference_model() else {
        return;
    };

    let (_dir, yaml_path) = write_empty_pipeline_yaml();
    let yaml_str = yaml_path
        .to_str()
        .expect("yaml path is UTF-8 (tempdir contract)");

    // Greedy decoding (temp=0.0) makes the output deterministic across
    // runs, so we can compare stdout directly. Without `--seed` the
    // sampler is not invoked because temp == 0.0 short-circuits to
    // argmax inside `sample_token_optimized`.
    let (baseline_stdout, baseline_status) = run_generate(&model, &[]);
    assert!(
        baseline_status.success(),
        "baseline run must succeed, got: {baseline_status:?}\nstdout: {baseline_stdout}"
    );

    let (surgery_stdout, surgery_status) = run_generate(&model, &["--surgery", yaml_str]);
    assert!(
        surgery_status.success(),
        "surgery run must succeed, got: {surgery_status:?}\nstdout: {surgery_stdout}"
    );

    // Strip the leading "Loading..." / "Runtime device:" / timing lines
    // that vary across runs, then compare just the generated suffix.
    // The reproducible substring is the final "Hello" + generated text
    // emitted by `print_generation_result`.
    let baseline_generated = extract_generated_suffix(&baseline_stdout);
    let surgery_generated = extract_generated_suffix(&surgery_stdout);
    assert_eq!(
        baseline_generated, surgery_generated,
        "empty surgery pipeline must not change generated tokens\n\
         baseline:\n{baseline_stdout}\n\
         surgery:\n{surgery_stdout}"
    );
}

/// Helper: extract the bytes between the prompt echo (`Hello`) and the
/// trailing stats line (`[Generated ...`) — the substring that is
/// determined by the model + sampling config alone, not by per-run
/// timing.
fn extract_generated_suffix(stdout: &str) -> &str {
    // `print_generation_preamble` writes the prompt with no trailing
    // newline; `print_generation_result` appends the generated text
    // followed by a blank line and then the stats line.
    let start = stdout.find("Generating...\nHello").unwrap_or(0);
    let from_hello = &stdout[start..];
    let end = from_hello.find("\n[Generated").unwrap_or(from_hello.len());
    &from_hello[..end]
}

/// Acceptance criterion (a) for the YAML parser side: a malformed
/// config is surfaced as a clean error and the binary exits non-zero
/// before any model load begins. We assert non-zero exit and that the
/// error mentions `--surgery` (so the user knows which flag is at
/// fault) and the offending value (`99`).
#[test]
fn malformed_surgery_yaml_fails_fast() {
    // No need to skip on missing model — the YAML parse error fires
    // before the model directory is even read.
    let (_dir, yaml_path) = write_malformed_yaml();
    let yaml_str = yaml_path.to_str().expect("yaml path is UTF-8");

    let output = Command::new(mlxcel_binary())
        .arg("generate")
        .arg("-m")
        .arg("models/this-path-is-deliberately-bogus-for-the-test")
        .arg("-p")
        .arg("Hello")
        .arg("-n")
        .arg("1")
        .arg("--surgery")
        .arg(yaml_str)
        .output()
        .expect("spawn mlxcel");

    assert!(
        !output.status.success(),
        "malformed --surgery YAML must cause non-zero exit, got: {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", String::from_utf8_lossy(&output.stdout), stderr);
    assert!(
        combined.contains("surgery"),
        "error must mention 'surgery' so the user identifies the flag: {combined}"
    );
    assert!(
        combined.contains("99") || combined.contains("version"),
        "error must surface the schema-version mismatch: {combined}"
    );
}

/// Acceptance criterion (a): `--surgery /path/to/missing.yaml` is
/// rejected up front with a clear message — no model load is attempted.
#[test]
fn missing_surgery_yaml_fails_fast() {
    let output = Command::new(mlxcel_binary())
        .arg("generate")
        .arg("-m")
        .arg("models/this-path-is-deliberately-bogus-for-the-test")
        .arg("-p")
        .arg("Hello")
        .arg("--surgery")
        .arg("/path/does/not/exist/surgery.yaml")
        .output()
        .expect("spawn mlxcel");

    assert!(
        !output.status.success(),
        "missing --surgery file must cause non-zero exit"
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("surgery") && combined.contains("/path/does/not/exist/surgery.yaml"),
        "error must name the missing path: {combined}"
    );
}
