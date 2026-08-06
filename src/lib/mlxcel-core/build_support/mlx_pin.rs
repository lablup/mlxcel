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

//! Resolution and verification of the pinned MLX upstream commit (issue #1047).
//!
//! The pin has exactly one home: the `GIT_TAG` argument of the
//! `FetchContent_Declare(mlx ...)` block in `src/lib/mlx-cpp/CMakeLists.txt`.
//! That is the value CMake actually fetches, so everything else derives from it
//! rather than restating it. Before #1047 the same 40-character SHA was written
//! out in three places (the CMake file, a `MLX_EXPECTED_COMMIT` constant in
//! `build.rs`, and a workflow-scope `env` in `.github/workflows/release.yml`)
//! with nothing checking that they agreed, and the third had already fallen a
//! bump behind.
//!
//! Two files include this module:
//!
//! * `src/lib/mlxcel-core/build.rs` pulls it in with `#[path]` and calls it to
//!   resolve the pin once, before the `_deps/` purge decision, the
//!   `cargo:rustc-env=MLXCEL_MLX_COMMIT` export, the post-build HEAD
//!   verification and the `_deps/.mlx-build-commit` marker write, so all four
//!   consume the same resolved value.
//! * `src/lib/mlxcel-mlx-pin` includes the very same file as a library so the
//!   unit tests at the bottom run without compiling `mlxcel-core`, whose build
//!   script builds MLX C++ from source.
//!
//! The logic here is deliberately dependency-free (`std` only): a build script
//! that needs a crates.io dependency to read its own pin would be a worse
//! trade than the hand-rolled scanning below.

// `build.rs` includes this file as a private module and does not necessarily
// call every helper, which would make `dead_code` fire there (and, under
// `cargo clippy --workspace --all-targets -- -D warnings`, fail the gate) for
// code the `mlxcel-mlx-pin` unit tests do cover. The library view of this same
// file re-exports everything publicly, so nothing is hidden by this.
#![allow(dead_code)]

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Location of the CMake file that owns the pin, relative to `mlxcel-core`'s
/// `CARGO_MANIFEST_DIR`.
///
/// Resolved against the manifest directory rather than the process working
/// directory: a build script's CWD is the package root today, but that is a
/// Cargo implementation detail and not something a correctness check should
/// rest on.
pub const CMAKE_LISTS_RELATIVE_PATH: &str = "../mlx-cpp/CMakeLists.txt";

/// The same file spelled from the repository root, for human-facing messages.
/// Nothing resolves a path through this; it exists so a build failure points at
/// something the reader can open without first working out where the build
/// script was standing.
pub const CMAKE_LISTS_REPO_PATH: &str = "src/lib/mlx-cpp/CMakeLists.txt";

/// The MLX upstream repository, used to pick the right `FetchContent_Declare`
/// block out of the CMake file.
///
/// Matching on the repository rather than taking the file's only `GIT_TAG`
/// keeps a second declared dependency from silently supplying the pin. Today
/// there is exactly one `GIT_TAG` in the file, so a whole-file scan would work
/// and would break invisibly the day that stops being true, which is the same
/// class of defect #1047 is about.
pub const MLX_REPOSITORY_MARKER: &str = "ml-explore/mlx";

/// A `FetchContent_Declare` call, the CMake command the pin lives in.
const FETCH_CONTENT_DECLARE: &str = "FetchContent_Declare";

/// The `GIT_TAG` keyword whose argument is the pin.
const GIT_TAG_KEYWORD: &str = "GIT_TAG";

/// Length of a full git object name in hex characters.
const FULL_SHA_LEN: usize = 40;

/// Why the pin could not be resolved from the CMake file.
///
/// Every variant carries the path so the build failure names the file the
/// reader has to open, and enough of the offending value to act on.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PinError {
    /// The CMake file could not be read at all.
    Unreadable { path: PathBuf, message: String },
    /// No `FetchContent_Declare` block naming the MLX repository was found.
    NoMlxDeclaration { path: PathBuf },
    /// More than one `FetchContent_Declare` block names the MLX repository, so
    /// which one supplies the pin is not well defined.
    AmbiguousMlxDeclaration { path: PathBuf, count: usize },
    /// The MLX declaration carries no `GIT_TAG` argument.
    MissingGitTag { path: PathBuf },
    /// The MLX declaration carries more than one `GIT_TAG` argument.
    AmbiguousGitTag { path: PathBuf, count: usize },
    /// The `GIT_TAG` argument is not a full lowercase hex commit SHA.
    NotAFullCommitSha { path: PathBuf, value: String },
}

impl fmt::Display for PinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { path, message } => write!(
                f,
                "cannot read the pinned MLX commit: {} could not be opened ({message})",
                path.display()
            ),
            Self::NoMlxDeclaration { path } => write!(
                f,
                "cannot resolve the pinned MLX commit: {} has no FetchContent_Declare block whose GIT_REPOSITORY names {MLX_REPOSITORY_MARKER}. The pin is the GIT_TAG argument of that block and nothing else stores it.",
                path.display()
            ),
            Self::AmbiguousMlxDeclaration { path, count } => write!(
                f,
                "cannot resolve the pinned MLX commit: {} has {count} FetchContent_Declare blocks naming {MLX_REPOSITORY_MARKER}, so the pin is ambiguous. Leave exactly one.",
                path.display()
            ),
            Self::MissingGitTag { path } => write!(
                f,
                "cannot resolve the pinned MLX commit: the FetchContent_Declare block naming {MLX_REPOSITORY_MARKER} in {} has no {GIT_TAG_KEYWORD} argument.",
                path.display()
            ),
            Self::AmbiguousGitTag { path, count } => write!(
                f,
                "cannot resolve the pinned MLX commit: the FetchContent_Declare block naming {MLX_REPOSITORY_MARKER} in {} has {count} {GIT_TAG_KEYWORD} arguments. Leave exactly one.",
                path.display()
            ),
            Self::NotAFullCommitSha { path, value } => write!(
                f,
                "the pinned MLX commit in {} is {value:?}, which is not a {FULL_SHA_LEN}-character lowercase hex commit SHA. A branch or tag name cannot be used here: the build-cache marker and the fetched-HEAD check both compare against an exact commit, and a moving ref would make both meaningless. Pin a full commit SHA.",
                path.display()
            ),
        }
    }
}

impl std::error::Error for PinError {}

/// Whether a cached `_deps/` tree may be reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheState {
    /// The marker records the pin currently in effect; the cache is reusable.
    Valid,
    /// The marker is absent or records a different commit; `_deps/` must go.
    Stale { cached: Option<String> },
}

/// Result of comparing the fetched MLX checkout's HEAD against the pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeadCheck {
    /// HEAD is the pinned commit.
    Match,
    /// HEAD is readable and is a different commit. Always a hard error.
    Mismatch { found: String },
    /// HEAD could not be read, so nothing is proven either way.
    ///
    /// Vendored, exported or offline source trees legitimately arrive without
    /// `.git`, and `git` is not guaranteed to be on the build host's `PATH`.
    /// Neither is evidence of a wrong checkout, so both warn rather than fail.
    Unavailable { reason: String },
}

/// Read and validate the pinned MLX commit from `mlxcel-core`'s CMake file.
///
/// `manifest_dir` is `mlxcel-core`'s `CARGO_MANIFEST_DIR`.
pub fn read_pinned_commit(manifest_dir: &Path) -> Result<String, PinError> {
    let path = manifest_dir.join(CMAKE_LISTS_RELATIVE_PATH);
    let source = std::fs::read_to_string(&path).map_err(|err| PinError::Unreadable {
        path: path.clone(),
        message: err.to_string(),
    })?;
    parse_pinned_commit(&source, &path)
}

/// Parse and validate the pinned MLX commit out of CMake source text.
///
/// `path` is only used to build error messages.
pub fn parse_pinned_commit(source: &str, path: &Path) -> Result<String, PinError> {
    let uncommented = strip_cmake_comments(source);
    let mlx_blocks: Vec<String> = fetch_content_declare_blocks(&uncommented)
        .into_iter()
        .filter(|block| block.contains(MLX_REPOSITORY_MARKER))
        .collect();

    match mlx_blocks.len() {
        0 => {
            return Err(PinError::NoMlxDeclaration {
                path: path.to_path_buf(),
            });
        }
        1 => {}
        count => {
            return Err(PinError::AmbiguousMlxDeclaration {
                path: path.to_path_buf(),
                count,
            });
        }
    }

    let tags = git_tag_arguments(&mlx_blocks[0]);
    let tag = match tags.len() {
        0 => {
            return Err(PinError::MissingGitTag {
                path: path.to_path_buf(),
            });
        }
        1 => &tags[0],
        count => {
            return Err(PinError::AmbiguousGitTag {
                path: path.to_path_buf(),
                count,
            });
        }
    };

    if !is_full_commit_sha(tag) {
        return Err(PinError::NotAFullCommitSha {
            path: path.to_path_buf(),
            value: tag.clone(),
        });
    }

    Ok(tag.clone())
}

/// Decide whether a `_deps/` tree carrying `marker_contents` matches the pin.
///
/// `marker_contents` is `None` when `_deps/.mlx-build-commit` is missing or
/// unreadable, which is treated exactly like a wrong commit: nothing vouches
/// for the tree, so it does not get reused.
pub fn cache_state(marker_contents: Option<&str>, expected_commit: &str) -> CacheState {
    let cached = marker_contents.map(|raw| raw.trim().to_string());
    if cached.as_deref() == Some(expected_commit) {
        CacheState::Valid
    } else {
        CacheState::Stale { cached }
    }
}

/// Compare the HEAD of a fetched MLX checkout against the pin.
///
/// This is the half of the fix that `parse_pinned_commit` cannot cover: it
/// checks what actually landed on disk, so a `_deps/` tree restored from a CI
/// cache, seeded by hand, or left behind by a `FetchContent` that never re-ran
/// is caught rather than compiled.
pub fn check_fetched_head(mlx_src_dir: &Path, expected_commit: &str) -> HeadCheck {
    if !mlx_src_dir.exists() {
        return HeadCheck::Unavailable {
            reason: format!("{} does not exist", mlx_src_dir.display()),
        };
    }

    // The `.git` probe is load-bearing, not a fast path. `git -C <dir>` walks
    // up to the nearest enclosing repository, and `_deps/` sits inside the
    // mlxcel checkout, so without this a source tree with no git metadata
    // would answer with *mlxcel's* HEAD and be reported as a pin mismatch.
    if !mlx_src_dir.join(".git").exists() {
        return HeadCheck::Unavailable {
            reason: format!(
                "{} has no .git metadata (vendored or exported source tree)",
                mlx_src_dir.display()
            ),
        };
    }

    let output = match Command::new("git")
        .arg("-C")
        .arg(mlx_src_dir)
        .args(["rev-parse", "HEAD"])
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            return HeadCheck::Unavailable {
                reason: format!("could not run `git`: {err}"),
            };
        }
    };

    if !output.status.success() {
        return HeadCheck::Unavailable {
            reason: format!(
                "`git rev-parse HEAD` failed in {}: {}",
                mlx_src_dir.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        };
    }

    let found = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if found == expected_commit {
        HeadCheck::Match
    } else {
        HeadCheck::Mismatch { found }
    }
}

/// Build the failure message for a fetched checkout that disagrees with the pin.
pub fn head_mismatch_message(mlx_src_dir: &Path, expected_commit: &str, found: &str) -> String {
    format!(
        "the fetched MLX checkout does not match the pin: {} is at {found}, but {CMAKE_LISTS_REPO_PATH} pins {expected_commit}. \
         CMake reuses an already-populated mlx-src instead of re-running FetchContent, so this build would link the wrong MLX. \
         Delete the enclosing _deps/ directory to force a refetch.",
        mlx_src_dir.display(),
    )
}

/// Remove `#`-to-end-of-line CMake comments, preserving line structure.
///
/// Done before any scanning so a commented-out declaration or a `GIT_TAG`
/// mentioned in prose cannot be mistaken for the live pin. `#` inside a
/// double-quoted argument is left alone. CMake's bracket comments (`#[[ ]]`)
/// are not handled; they would leave extra text in place rather than remove
/// live text, so they cannot turn a correct parse into a wrong one, only into
/// a loud failure.
fn strip_cmake_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars();
    let mut in_string = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' if in_string => {
                out.push(c);
                if let Some(escaped) = chars.next() {
                    out.push(escaped);
                }
            }
            '"' => {
                in_string = !in_string;
                out.push(c);
            }
            '#' if !in_string => {
                for skipped in chars.by_ref() {
                    if skipped == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Return the argument text of every `FetchContent_Declare(...)` call, with the
/// outermost parentheses removed.
fn fetch_content_declare_blocks(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut blocks = Vec::new();
    let mut cursor = 0usize;

    while let Some(offset) = source[cursor..].find(FETCH_CONTENT_DECLARE) {
        let start = cursor + offset;
        let after_keyword = start + FETCH_CONTENT_DECLARE.len();
        cursor = after_keyword;

        // Reject a longer identifier that merely ends with the keyword.
        if start > 0 && is_identifier_byte(bytes[start - 1]) {
            continue;
        }
        // ... or one that merely starts with it.
        if bytes
            .get(after_keyword)
            .is_some_and(|b| is_identifier_byte(*b))
        {
            continue;
        }

        let mut index = after_keyword;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if bytes.get(index) != Some(&b'(') {
            continue;
        }

        if let Some((body, end)) = balanced_paren_body(source, index) {
            blocks.push(body);
            cursor = end;
        }
    }

    blocks
}

/// Read from the `(` at `open` to its matching `)`, returning the text between
/// them and the index just past the closing parenthesis.
///
/// Returns `None` for an unbalanced call, which then contributes no block and
/// surfaces as a "no MLX declaration" failure rather than a partial parse.
fn balanced_paren_body(source: &str, open: usize) -> Option<(String, usize)> {
    let bytes = source.as_bytes();
    debug_assert_eq!(bytes.get(open), Some(&b'('));
    let mut depth = 0usize;
    let mut in_string = false;
    let mut index = open;

    while index < bytes.len() {
        let byte = bytes[index];
        match byte {
            b'\\' if in_string => index += 1,
            b'"' => in_string = !in_string,
            b'(' if !in_string => depth += 1,
            b')' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some((source[open + 1..index].to_string(), index + 1));
                }
            }
            _ => {}
        }
        index += 1;
    }

    None
}

/// Collect every `GIT_TAG` argument inside one declaration body.
fn git_tag_arguments(block: &str) -> Vec<String> {
    let bytes = block.as_bytes();
    let mut values = Vec::new();
    let mut cursor = 0usize;

    while let Some(offset) = block[cursor..].find(GIT_TAG_KEYWORD) {
        let start = cursor + offset;
        let after_keyword = start + GIT_TAG_KEYWORD.len();
        cursor = after_keyword;

        if start > 0 && is_identifier_byte(bytes[start - 1]) {
            continue;
        }
        if bytes
            .get(after_keyword)
            .is_some_and(|b| is_identifier_byte(*b))
        {
            continue;
        }

        let mut index = after_keyword;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() {
            continue;
        }

        let quoted = bytes[index] == b'"';
        if quoted {
            index += 1;
        }
        let value_start = index;
        while index < bytes.len() {
            let byte = bytes[index];
            let terminates = if quoted {
                byte == b'"'
            } else {
                byte.is_ascii_whitespace() || byte == b'(' || byte == b')'
            };
            if terminates {
                break;
            }
            index += 1;
        }

        values.push(block[value_start..index].to_string());
        cursor = index;
    }

    values
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// A full git object name: exactly 40 lowercase hex characters.
///
/// Uppercase is rejected on purpose. `git rev-parse HEAD` prints lowercase and
/// the `_deps/.mlx-build-commit` marker is compared byte for byte, so an
/// uppercase pin would make every cache look stale forever.
fn is_full_commit_sha(value: &str) -> bool {
    value.len() == FULL_SHA_LEN
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const PIN: &str = "2c46b953db88965c4270cc7306eda6887a3247f2";
    const OTHER_PIN: &str = "b7c3dd6d27f45b5365b08a840310187dc503f1db";

    fn path() -> PathBuf {
        PathBuf::from("/repo/src/lib/mlx-cpp/CMakeLists.txt")
    }

    fn mlx_declaration(tag_line: &str) -> String {
        format!(
            "FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY \"https://github.com/ml-explore/mlx.git\"\n  {tag_line})\n"
        )
    }

    // ---------------------------------------------------------------- parsing

    /// The in-tree CMake file must parse. This deliberately asserts only that
    /// the result is a well-formed SHA, never a literal: asserting the value
    /// would put a second copy of the pin in the tree and recreate #1047 in the
    /// test suite.
    #[test]
    fn the_real_cmakelists_resolves_to_a_full_sha() {
        let core_manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../mlxcel-core");
        let commit = read_pinned_commit(&core_manifest).expect("in-tree CMakeLists must parse");
        assert!(
            is_full_commit_sha(&commit),
            "resolved pin {commit:?} is not a full lowercase hex SHA"
        );
    }

    #[test]
    fn parses_the_pin_from_a_well_formed_declaration() {
        let source = mlx_declaration(&format!("GIT_TAG {PIN}"));
        assert_eq!(parse_pinned_commit(&source, &path()).unwrap(), PIN);
    }

    #[test]
    fn parses_a_quoted_pin() {
        let source = mlx_declaration(&format!("GIT_TAG \"{PIN}\""));
        assert_eq!(parse_pinned_commit(&source, &path()).unwrap(), PIN);
    }

    #[test]
    fn parses_a_pin_preceded_by_a_comment_inside_the_block() {
        let source = mlx_declaration(&format!(
            "# single source of truth, see build_support/mlx_pin.rs\n  GIT_TAG {PIN}"
        ));
        assert_eq!(parse_pinned_commit(&source, &path()).unwrap(), PIN);
    }

    #[test]
    fn ignores_a_commented_out_declaration() {
        let source = format!(
            "# FetchContent_Declare(mlx GIT_REPOSITORY \"https://github.com/ml-explore/mlx.git\" GIT_TAG {OTHER_PIN})\n{}",
            mlx_declaration(&format!("GIT_TAG {PIN}"))
        );
        assert_eq!(parse_pinned_commit(&source, &path()).unwrap(), PIN);
    }

    #[test]
    fn missing_git_tag_is_an_error() {
        let source = "FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY \"https://github.com/ml-explore/mlx.git\")\n";
        assert_eq!(
            parse_pinned_commit(source, &path()),
            Err(PinError::MissingGitTag { path: path() })
        );
    }

    #[test]
    fn a_branch_name_is_rejected() {
        let source = mlx_declaration("GIT_TAG main");
        assert_eq!(
            parse_pinned_commit(&source, &path()),
            Err(PinError::NotAFullCommitSha {
                path: path(),
                value: "main".to_string(),
            })
        );
    }

    #[test]
    fn a_short_sha_is_rejected() {
        let source = mlx_declaration("GIT_TAG 2c46b95");
        assert!(matches!(
            parse_pinned_commit(&source, &path()),
            Err(PinError::NotAFullCommitSha { .. })
        ));
    }

    #[test]
    fn an_uppercase_sha_is_rejected() {
        let upper = PIN.to_ascii_uppercase();
        let source = mlx_declaration(&format!("GIT_TAG {upper}"));
        assert!(matches!(
            parse_pinned_commit(&source, &path()),
            Err(PinError::NotAFullCommitSha { .. })
        ));
    }

    #[test]
    fn two_mlx_declarations_are_ambiguous() {
        let source = format!(
            "{}{}",
            mlx_declaration(&format!("GIT_TAG {PIN}")),
            mlx_declaration(&format!("GIT_TAG {OTHER_PIN}"))
        );
        assert_eq!(
            parse_pinned_commit(&source, &path()),
            Err(PinError::AmbiguousMlxDeclaration {
                path: path(),
                count: 2,
            })
        );
    }

    #[test]
    fn two_git_tags_in_one_declaration_are_ambiguous() {
        let source = mlx_declaration(&format!("GIT_TAG {PIN}\n  GIT_TAG {OTHER_PIN}"));
        assert_eq!(
            parse_pinned_commit(&source, &path()),
            Err(PinError::AmbiguousGitTag {
                path: path(),
                count: 2,
            })
        );
    }

    /// The reason the parse is scoped to the MLX block rather than taking the
    /// file's only `GIT_TAG`: an unrelated dependency must not supply the pin.
    #[test]
    fn another_dependency_with_its_own_git_tag_is_not_picked_up() {
        let source = format!(
            "FetchContent_Declare(\n  fmt\n  GIT_REPOSITORY \"https://github.com/fmtlib/fmt.git\"\n  GIT_TAG {OTHER_PIN})\n{}",
            mlx_declaration(&format!("GIT_TAG {PIN}"))
        );
        assert_eq!(parse_pinned_commit(&source, &path()).unwrap(), PIN);
    }

    #[test]
    fn a_file_without_an_mlx_declaration_is_an_error() {
        let source = format!(
            "FetchContent_Declare(\n  fmt\n  GIT_REPOSITORY \"https://github.com/fmtlib/fmt.git\"\n  GIT_TAG {OTHER_PIN})\n"
        );
        assert_eq!(
            parse_pinned_commit(&source, &path()),
            Err(PinError::NoMlxDeclaration { path: path() })
        );
    }

    #[test]
    fn an_unreadable_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let err = read_pinned_commit(dir.path()).unwrap_err();
        assert!(matches!(err, PinError::Unreadable { .. }));
        assert!(
            err.to_string().contains("CMakeLists.txt"),
            "message should name the file: {err}"
        );
    }

    #[test]
    fn error_messages_name_the_file_and_what_was_expected() {
        let source = mlx_declaration("GIT_TAG main");
        let message = parse_pinned_commit(&source, &path())
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("/repo/src/lib/mlx-cpp/CMakeLists.txt"),
            "{message}"
        );
        assert!(message.contains("40-character lowercase hex"), "{message}");
    }

    // ------------------------------------------------------------ cache state

    #[test]
    fn a_marker_equal_to_the_pin_keeps_the_cache() {
        assert_eq!(cache_state(Some(PIN), PIN), CacheState::Valid);
    }

    #[test]
    fn marker_whitespace_is_ignored() {
        assert_eq!(
            cache_state(Some(&format!("{PIN}\n")), PIN),
            CacheState::Valid
        );
    }

    #[test]
    fn a_marker_from_another_pin_makes_the_cache_stale() {
        assert_eq!(
            cache_state(Some(OTHER_PIN), PIN),
            CacheState::Stale {
                cached: Some(OTHER_PIN.to_string()),
            }
        );
    }

    #[test]
    fn an_absent_marker_makes_the_cache_stale() {
        assert_eq!(cache_state(None, PIN), CacheState::Stale { cached: None });
    }

    // ---------------------------------------------------------- fetched HEAD

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success())
    }

    /// Create a real git repository with one empty commit and return its HEAD.
    fn init_repo_with_one_commit(dir: &Path) -> String {
        let run = |args: &[&str]| {
            let out = Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .output()
                .expect("git should run");
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        run(&["init", "-q"]);
        run(&[
            "-c",
            "user.name=mlxcel test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "pin fixture",
        ]);
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("git should run");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    #[test]
    fn head_matching_the_pin_passes() {
        if !git_available() {
            eprintln!("skipping: git is not on PATH");
            return;
        }
        let dir = TempDir::new().unwrap();
        let head = init_repo_with_one_commit(dir.path());
        assert_eq!(check_fetched_head(dir.path(), &head), HeadCheck::Match);
    }

    #[test]
    fn head_differing_from_the_pin_is_a_mismatch() {
        if !git_available() {
            eprintln!("skipping: git is not on PATH");
            return;
        }
        let dir = TempDir::new().unwrap();
        let head = init_repo_with_one_commit(dir.path());
        assert_eq!(
            check_fetched_head(dir.path(), PIN),
            HeadCheck::Mismatch { found: head }
        );
    }

    #[test]
    fn a_tree_without_git_metadata_is_unavailable_not_a_mismatch() {
        let dir = TempDir::new().unwrap();
        assert!(matches!(
            check_fetched_head(dir.path(), PIN),
            HeadCheck::Unavailable { .. }
        ));
    }

    #[test]
    fn a_missing_directory_is_unavailable() {
        let dir = TempDir::new().unwrap();
        assert!(matches!(
            check_fetched_head(&dir.path().join("mlx-src"), PIN),
            HeadCheck::Unavailable { .. }
        ));
    }

    /// `_deps/` lives inside the mlxcel checkout, so a source tree with no git
    /// metadata must not be answered with the enclosing repository's HEAD.
    #[test]
    fn the_check_does_not_walk_up_into_an_enclosing_repository() {
        if !git_available() {
            eprintln!("skipping: git is not on PATH");
            return;
        }
        let outer = TempDir::new().unwrap();
        let outer_head = init_repo_with_one_commit(outer.path());
        let inner = outer.path().join("build/_deps/mlx-src");
        std::fs::create_dir_all(&inner).unwrap();

        match check_fetched_head(&inner, PIN) {
            HeadCheck::Unavailable { .. } => {}
            other => panic!("expected Unavailable, got {other:?} (outer HEAD is {outer_head})"),
        }
    }

    #[test]
    fn the_mismatch_message_says_how_to_recover() {
        let message = head_mismatch_message(Path::new("/out/build/_deps/mlx-src"), PIN, OTHER_PIN);
        assert!(message.contains(PIN), "{message}");
        assert!(message.contains(OTHER_PIN), "{message}");
        assert!(message.contains(CMAKE_LISTS_REPO_PATH), "{message}");
        assert!(message.contains("_deps/"), "{message}");
    }
}
