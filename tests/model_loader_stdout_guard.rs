// Copyright 2025-2026 Lablup Inc.
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

//! Model loaders may not write to stdout (issue #1794).
//!
//! `examples/logit_trace` writes its TSV to stdout, and other callers of
//! `mlxcel::load_model` parse their own stdout. A `println!` in a loader puts
//! progress lines into that stream, and `scripts/compare_logit_traces.py` then
//! fails on the first non-`#` line that is not a six-field data row.
//!
//! Loader progress goes through `tracing`. A notice that must show without
//! `RUST_LOG` uses `eprintln!`. Nothing under `src/models` or `src/loading`
//! is legitimately stdout, so the allow-list below starts empty. CLI command
//! output (`src/commands/`, `src/execution/quant_advisor.rs`) is out of scope.
//!
//! Files whose name ends in `tests.rs` are skipped, and so are lines whose
//! trimmed text starts with `//`. Inline `#[cfg(test)]` modules are not exempt
//! and use `eprintln!`.

use std::fs;
use std::path::{Path, PathBuf};

const SCANNED_DIRS: [&str; 2] = ["src/models", "src/loading"];
const PATTERNS: [&str; 3] = ["println!(", "print!(", "stdout()"];

/// `(repo-relative path, reason)`. An entry exempts a whole file and fails the
/// test once the file no longer matches, so it cannot go stale.
const ALLOW_LIST: [(&str, &str); 0] = [];

/// Recursively collects `*.rs` files with `lstat` semantics: symlinks are not
/// followed, matching `grep -r`.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read directory {}: {e}", dir.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|e| panic!("directory entry under {}: {e}", dir.display()));
        let file_type = entry
            .file_type()
            .unwrap_or_else(|e| panic!("failed to stat {}: {e}", entry.path().display()));
        let path = entry.path();
        if file_type.is_dir() {
            collect_rs_files(&path, out);
        } else if file_type.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// True when `line` holds one of [`PATTERNS`] not preceded by an ASCII
/// alphanumeric or `_`, so `eprintln!(` and `my_print!(` pass.
fn line_hits(line: &str) -> bool {
    if line.trim_start().starts_with("//") {
        return false;
    }
    PATTERNS.iter().any(|pat| {
        line.match_indices(pat).any(|(i, _)| {
            line[..i]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'))
        })
    })
}

/// Every `(line number, trimmed line)` in `source` that writes to stdout.
fn violations(source: &str) -> Vec<(usize, String)> {
    source
        .lines()
        .enumerate()
        .filter(|(_, l)| line_hits(l))
        .map(|(i, l)| (i + 1, l.trim().to_string()))
        .collect()
}

#[test]
fn model_loaders_do_not_write_to_stdout() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in SCANNED_DIRS {
        collect_rs_files(&root.join(dir), &mut files);
    }
    files.sort();
    assert!(files.len() > 50, "walker found only {} files", files.len());

    let mut failures = Vec::new();
    let mut allow_used = vec![false; ALLOW_LIST.len()];
    for file in &files {
        let rel = file
            .strip_prefix(root)
            .expect("file is under the repo root")
            .to_string_lossy()
            .replace('\\', "/");
        if rel.ends_with("tests.rs") {
            continue;
        }
        let source = fs::read_to_string(file).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        let hits = violations(&source);
        if hits.is_empty() {
            continue;
        }
        if let Some(i) = ALLOW_LIST.iter().position(|(p, _)| *p == rel) {
            allow_used[i] = true;
            continue;
        }
        failures.extend(
            hits.into_iter()
                .map(|(n, code)| format!("{rel}:{n}: {code}")),
        );
    }

    let stale: Vec<String> = ALLOW_LIST
        .iter()
        .zip(&allow_used)
        .filter(|(_, used)| !**used)
        .map(|((p, why), _)| format!("{p} ({why})"))
        .collect();
    assert!(
        stale.is_empty(),
        "stale ALLOW_LIST entries, remove them: {stale:?}"
    );
    assert!(
        failures.is_empty(),
        "model loaders must not write to stdout; it corrupts the output of callers such as \
         examples/logit_trace (issue #1794). Use `tracing::info!` or `tracing::warn!`, or \
         `eprintln!` for a notice that must show without RUST_LOG:\n{}",
        failures.join("\n")
    );
}

#[test]
fn matcher_distinguishes_stderr_comments_and_identifiers() {
    assert!(line_hits(r#"    println!("x");"#));
    assert!(line_hits("print!(\"x\")"));
    assert!(line_hits("let o = std::io::stdout();"));
    assert!(line_hits("(println!(\"x\"))"));
    assert!(!line_hits(r#"eprintln!("x");"#));
    assert!(!line_hits(r#"eprint!("x");"#));
    assert!(!line_hits(r#"    // println!("x");"#));
    assert!(!line_hits(r#"    /// println!("x");"#));
    assert!(!line_hits("my_print!(x)"));
}
