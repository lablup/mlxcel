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

//! Real-model regression tests for the auto-partitioner.
//!
//! These tests require local model weights and the compiled `mlxcel`
//! binary. They are `#[ignore]`-gated so the default CI run skips them;
//! run with `cargo test --test pipeline_auto_partition_real_models -- \
//! --ignored` against a checkout with `models/gemma-4-e2b-it-4bit` present.
//!
//! What the test verifies: the auto-partitioner now produces a valid
//! 2-stage plan for Gemma 4 without any manual `--pp-layers` flag. Before
//! operators had to specify layer ranges by hand because the
//! partitioner did not understand KV-shared layer adjacency.

mod common;

use std::process::Command;

use common::{extract_generated_body, repo_model_dir};

fn run_generate(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_mlxcel"))
        .args(args)
        .output()
        .expect("failed to execute mlxcel generate");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// Run `gemma-4-e2b-it-4bit` across 2 stages with zero manual partitioning
/// and assert the generated output is non-empty.
///
/// Without the work, this invocation fails because the naive
/// partitioner cuts the model between a KV-shared source layer and its
/// consumer, stranding the cache on the wrong stage. With the adjacency-
/// aware partitioner the plan is valid by default and generation
/// completes.
#[test]
#[ignore = "requires local gemma-4-e2b-it-4bit weights and the mlxcel binary"]
fn gemma4_auto_partition_two_stage_no_manual_layers() {
    let model_dir = repo_model_dir("gemma-4-e2b-it-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();

    // Baseline single-stage generation.
    let (ok_single, single_stdout, single_stderr) = run_generate(&[
        "generate",
        "-m",
        &model_arg,
        "-p",
        "Hello",
        "-n",
        "8",
        "--temp",
        "0",
        "--no-chat-template",
    ]);
    assert!(
        ok_single,
        "single-stage generation failed:\nstdout: {single_stdout}\nstderr: {single_stderr}"
    );
    let single_body = extract_generated_body(&single_stdout).expect("missing single-stage body");

    // Auto-partition 2-stage path. No `--pp-layers`. This is the
    // regression guard — before this sub-issue landed, the
    // same invocation required a manual layer specification.
    let (ok_pp, pp_stdout, pp_stderr) = run_generate(&[
        "generate",
        "-m",
        &model_arg,
        "-p",
        "Hello",
        "-n",
        "8",
        "--temp",
        "0",
        "--no-chat-template",
        "--pp-size",
        "2",
    ]);
    assert!(
        ok_pp,
        "auto-partition 2-stage generation failed (regression):\n\
         stdout: {pp_stdout}\nstderr: {pp_stderr}"
    );
    let pp_body = extract_generated_body(&pp_stdout).expect("missing 2-stage body");
    assert!(
        !pp_body.trim().is_empty(),
        "2-stage generation produced empty body"
    );

    // Greedy decoding with identical sampling seeds should match byte-for-
    // byte. If pipeline splitting perturbs the output, the KV-shared
    // adjacency constraint was honoured structurally but computation is
    // still wrong; fail loudly rather than silently accepting divergence.
    assert_eq!(
        pp_body, single_body,
        "auto-partition generation diverged from single-stage baseline"
    );
}

/// Guard that manual `--pp-layers` still works and produces the same
/// answer as the auto-partitioned path when the manual plan happens to
/// be valid. This protects operators who have legacy cluster configs
/// that hand-specify ranges.
#[test]
#[ignore = "requires local gemma-4-e2b-it-4bit weights and the mlxcel binary"]
fn gemma4_manual_partition_still_works() {
    let model_dir = repo_model_dir("gemma-4-e2b-it-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();

    // Gemma 4 2B has 26 hidden layers; split at 16. The exact split is
    // model-specific but guaranteed to not cut a KV-shared group for 2B
    // because shared layers are in a contiguous tail.
    let (ok_auto, auto_stdout, _) = run_generate(&[
        "generate",
        "-m",
        &model_arg,
        "-p",
        "Hello",
        "-n",
        "4",
        "--temp",
        "0",
        "--no-chat-template",
        "--pp-size",
        "2",
    ]);
    let (ok_manual, manual_stdout, _) = run_generate(&[
        "generate",
        "-m",
        &model_arg,
        "-p",
        "Hello",
        "-n",
        "4",
        "--temp",
        "0",
        "--no-chat-template",
        "--pp-layers",
        "0-15,16-25",
    ]);
    assert!(ok_auto, "auto partition generation failed");
    assert!(ok_manual, "manual partition generation failed");
    let auto_body = extract_generated_body(&auto_stdout).expect("missing auto body");
    let manual_body = extract_generated_body(&manual_stdout).expect("missing manual body");
    assert_eq!(
        auto_body, manual_body,
        "auto and manual partitions disagree on a valid manual split"
    );
}
