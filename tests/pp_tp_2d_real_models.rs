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

//! End-to-end 2D parallelism (PP × TP) parity test against a real model.
//!
//! The test uses the production `mlxcel` binary with `--pp-size 2 --tp-size 2`
//! (a 2×2 grid) and verifies that the greedy-decoded token sequence matches a
//! single-device reference on a fixed prompt. Marked `#[ignore]` because it
//! requires local model weights and enough aggregate memory to host four
//! shards concurrently.

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
        String::from_utf8(output.stdout).expect("stdout must be valid UTF-8"),
        String::from_utf8(output.stderr).expect("stderr must be valid UTF-8"),
    )
}

/// 2D parallelism parity test: a 2×2 (pp_size=2, tp_size=2) run should
/// produce the same greedy output as a single-device reference.
#[test]
#[ignore = "requires local model weights and a 2×2 PP×TP capable environment"]
fn pp_tp_2x2_llama_real_model_parity() {
    let model_dir = repo_model_dir("llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let model_arg = model_dir.to_string_lossy().to_string();

    let (ok_dense, dense_stdout, dense_stderr) = run_generate(&[
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
        ok_dense,
        "single-device reference run failed:\nstdout={dense_stdout}\nstderr={dense_stderr}"
    );

    let (ok_2d, pp_tp_stdout, pp_tp_stderr) = run_generate(&[
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
        "--tp-size",
        "2",
    ]);
    assert!(
        ok_2d,
        "2x2 PPxTP run failed:\nstdout={pp_tp_stdout}\nstderr={pp_tp_stderr}"
    );

    let dense_body = extract_generated_body(&dense_stdout).expect("missing dense generation body");
    let pp_tp_body = extract_generated_body(&pp_tp_stdout).expect("missing PP+TP generation body");
    assert_eq!(
        pp_tp_body, dense_body,
        "2x2 PPxTP output diverged from single-device reference"
    );
}

/// Sanity check that the CLI argument parser accepts the 2D flag combination
/// `--pp-size 2 --tp-size 2`.
///
/// The scope is deliberately the parser, not the validator. `run_generate`
/// resolves `-m` *before* it calls `validate_pipeline_parallel_args`, because
/// the validators read the resolved model directory. A subprocess invocation
/// therefore cannot reach the validator without a real model on disk, and a
/// nonexistent `-m` value that happens to be a valid bare repo segment is
/// expanded against `$MLXCEL_DEFAULT_ORG` and sent to HuggingFace, which would
/// put a network round trip in the non-ignored test surface. `--help` makes
/// clap exit as soon as parsing succeeds, which keeps this test hermetic: no
/// runtime is initialized and no request is made.
///
/// The validator itself has direct unit coverage in
/// `src/commands/generate_tests.rs`
/// (`validate_pipeline_parallel_args_accepts_2d_pp_tp`), so nothing is lost by
/// scoping this one to the parser.
#[test]
fn pp_tp_2d_flag_combination_is_accepted_by_the_parser() {
    let output = Command::new(env!("CARGO_BIN_EXE_mlxcel"))
        .args([
            "generate",
            "-m",
            "unread-because-help-exits-during-parsing",
            "-p",
            "x",
            "-n",
            "1",
            "--pp-size",
            "2",
            "--tp-size",
            "2",
            "--help",
        ])
        .output()
        .expect("failed to invoke mlxcel generate");

    // Positive assertion, so the test cannot pass by dying early. clap prints
    // help and exits 0 when every flag parsed; a flag the binary does not
    // accept exits 2 with a usage error instead. That is the case this test
    // exists to catch: it previously passed `--tensor-parallel-size`, a
    // spelling the binary has never accepted, and still passed because its
    // only assertions were negative.
    assert!(
        output.status.success(),
        "the 2D flag combination was rejected by the argument parser \
         (status={:?}):\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
