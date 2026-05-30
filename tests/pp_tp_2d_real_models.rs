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
//! The test uses the production `mlxcel` binary with
//! `--pp-size 2 --tensor-parallel-size 2` (a 2×2 grid) and verifies that the
//! greedy-decoded token sequence matches a single-device reference on a fixed
//! prompt. Marked `#[ignore]` because it requires local model weights and
//! enough aggregate memory to host four shards concurrently.

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
        "--tensor-parallel-size",
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

/// Sanity check that the CLI accepts the 2D combination (the validator no
/// longer rejects `--pp-size 2 --tensor-parallel-size 2`).
///
/// This test only asserts that the validator does not reject the combination
/// upfront; it does not run the model. It is kept in the non-ignored test
/// surface because it is cheap and directly verifies the validator change
/// required.
#[test]
fn pp_tp_2d_validator_accepts_combination() {
    // Unlike the parity test above, this one does not require any model
    // weights. We invoke `mlxcel generate --help` plus the 2D flags and
    // confirm that the binary's argument parser accepts the combination.
    // Runtime failures (e.g., model missing) are acceptable — only the
    // validator rejection at src/commands/generate.rs:141 is under test.
    let output = Command::new(env!("CARGO_BIN_EXE_mlxcel"))
        .args([
            "generate",
            "-m",
            "nonexistent-model-path-for-validator-only-check",
            "-p",
            "x",
            "-n",
            "1",
            "--pp-size",
            "2",
            "--tensor-parallel-size",
            "2",
        ])
        .output()
        .expect("failed to invoke mlxcel generate");

    // We expect a failure (model is missing), but NOT because of the old
    // "CLI pipeline parallelism does not support tensor parallelism yet"
    // rejection.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stderr.contains("pipeline parallelism does not support tensor parallelism")
            && !stdout.contains("pipeline parallelism does not support tensor parallelism"),
        "2D combination was rejected by the old validator guard:\nstdout={stdout}\nstderr={stderr}"
    );
}
