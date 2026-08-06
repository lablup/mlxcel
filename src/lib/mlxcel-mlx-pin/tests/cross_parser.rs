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

//! Cross-parser agreement between the Rust MLX-pin parser
//! (`mlxcel_mlx_pin::mlx_pin`) and the awk-based shell parser
//! (`scripts/ci/mlx_pinned_commit.sh`), for issue #1047.
//!
//! `mlx_pin.rs`'s own unit tests and `mlx_pinned_commit_test.sh`'s fixtures
//! each cover one parser in isolation; nothing ran both parsers over the same
//! input and compared the results. This file closes that gap by shelling out
//! to the real script for a battery of inputs and checking each against the
//! Rust parser's answer for the identical text.
//!
//! The property asserted is one-directional: whenever the shell parser
//! accepts an input and prints a commit, the Rust parser must accept the same
//! input and resolve to the identical commit. The reverse does not hold and
//! is not asserted here. The shell parser is intentionally the stricter of
//! the two (see the `mlx-pin` job in `.github/workflows/ci.yml` and the
//! comment above `GIT_TAG` in `src/lib/mlx-cpp/CMakeLists.txt`): it requires
//! `GIT_TAG` to be the first token on its own line, after the line naming the
//! MLX repository, with no closing parenthesis in between, while the Rust
//! parser scans the whole declaration body regardless of line order. An input
//! only the Rust parser accepts is exercised below (`git_tag_before_git_repository`)
//! to keep that asymmetry documented and deliberate rather than accidental.
//!
//! What this rules out is the dangerous case: two parsers that both succeed
//! but disagree on the value. `release.yml`'s cache purge decision and the
//! actual MLX fetch both ultimately trace back to whichever parser answered,
//! so a same-input value mismatch would be silent exactly where #1047 needs
//! it loudest. A shell rejection, by contrast, is a hard, visible CI failure
//! (see the `mlx-pin` job), not a silent divergence.

use std::path::{Path, PathBuf};
use std::process::Command;

use mlxcel_mlx_pin::mlx_pin;

const PIN: &str = "2c46b953db88965c4270cc7306eda6887a3247f2";
const OTHER_PIN: &str = "b7c3dd6d27f45b5365b08a840310187dc503f1db";

/// Absolute path to the shell parser under test, resolved from this crate's
/// manifest directory rather than the process working directory so the test
/// does not depend on where `cargo test` was invoked from.
fn shell_script_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/ci/mlx_pinned_commit.sh")
}

/// Run the shell parser against `body`. `Some(commit)` on a zero exit and a
/// printed commit, `None` on any rejection (matching what a CI step sees).
fn run_shell_parser(body: &str) -> Option<String> {
    let dir = tempfile::TempDir::new().expect("create fixture dir");
    let fixture = dir.path().join("CMakeLists.txt");
    std::fs::write(&fixture, body).expect("write fixture");

    let script = shell_script_path();
    let output = Command::new(&script)
        .env("MLX_PIN_CMAKE_FILE", &fixture)
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", script.display()));

    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn mlx_declaration(tag_line: &str) -> String {
    format!(
        "FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY \"https://github.com/ml-explore/mlx.git\"\n  {tag_line})\n"
    )
}

/// Named fixtures. Several mirror the shapes in
/// `scripts/ci/mlx_pinned_commit_test.sh` so the two suites are known to
/// agree on the same inputs rather than drifting apart independently.
fn fixtures() -> Vec<(&'static str, String)> {
    vec![
        ("canonical", mlx_declaration(&format!("GIT_TAG {PIN}"))),
        ("quoted", mlx_declaration(&format!("GIT_TAG \"{PIN}\""))),
        (
            "commented",
            format!(
                "FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY \"https://github.com/ml-explore/mlx.git\"\n  # single source of truth, issue #1047\n  GIT_TAG {PIN})\n"
            ),
        ),
        (
            "other_dependency_first",
            format!(
                "FetchContent_Declare(\n  fmt\n  GIT_REPOSITORY \"https://github.com/fmtlib/fmt.git\"\n  GIT_TAG {OTHER_PIN})\n{}",
                mlx_declaration(&format!("GIT_TAG {PIN}"))
            ),
        ),
        (
            "commented_out_declaration",
            format!(
                "# FetchContent_Declare(mlx GIT_REPOSITORY \"https://github.com/ml-explore/mlx.git\" GIT_TAG {OTHER_PIN})\n{}",
                mlx_declaration(&format!("GIT_TAG {PIN}"))
            ),
        ),
        (
            "missing_git_tag",
            "FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY \"https://github.com/ml-explore/mlx.git\")\n"
                .to_string(),
        ),
        ("branch_name", mlx_declaration("GIT_TAG main")),
        ("short_sha", mlx_declaration("GIT_TAG 2c46b95")),
        (
            "uppercase_sha",
            mlx_declaration(&format!("GIT_TAG {}", PIN.to_ascii_uppercase())),
        ),
        (
            "two_mlx_declarations",
            format!(
                "{}{}",
                mlx_declaration(&format!("GIT_TAG {PIN}")),
                mlx_declaration(&format!("GIT_TAG {OTHER_PIN}"))
            ),
        ),
        (
            "no_mlx_declaration",
            format!(
                "FetchContent_Declare(\n  fmt\n  GIT_REPOSITORY \"https://github.com/fmtlib/fmt.git\"\n  GIT_TAG {OTHER_PIN})\n"
            ),
        ),
        // The documented asymmetry: the shell parser requires GIT_TAG after
        // the GIT_REPOSITORY line, so it rejects this reordering (falls
        // through to the `None` arm below, no assertion). The Rust parser
        // scans the whole block and accepts it. This fixture exists so the
        // asymmetry is verified on every run rather than only described in a
        // comment; if a future edit makes the shell parser accept this too,
        // the `Some` arm's equality assertion below still holds.
        (
            "git_tag_before_git_repository",
            "FetchContent_Declare(\n  mlx\n  GIT_TAG 2c46b953db88965c4270cc7306eda6887a3247f2\n  GIT_REPOSITORY \"https://github.com/ml-explore/mlx.git\")\n"
                .to_string(),
        ),
    ]
}

#[test]
fn shell_acceptance_implies_rust_agreement_on_the_same_commit() {
    if Command::new("bash").arg("--version").output().is_err() {
        eprintln!("skipping: bash is not on PATH");
        return;
    }

    for (name, body) in fixtures() {
        let shell_result = run_shell_parser(&body);
        let rust_result = mlx_pin::parse_pinned_commit(&body, Path::new("<fixture>")).ok();

        if let Some(shell_commit) = shell_result {
            assert_eq!(
                rust_result.as_deref(),
                Some(shell_commit.as_str()),
                "fixture {name:?}: shell parser accepted {shell_commit:?} but the Rust parser did not resolve to the same commit (got {rust_result:?}); a same-input value mismatch is exactly the silent divergence issue #1047 exists to prevent"
            );
        }
        // A `None` shell result is not checked against the Rust result: the
        // shell parser accepting strictly less is documented and expected.
    }
}
