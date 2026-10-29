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

//! Tests for [`super::StartupNotice`] and the source-scan guard that keeps
//! `tracing::` macros out of the code that runs before the server's tracing
//! subscriber exists (#2722).
//!
//! The guard reads the source files that hold the pre-subscriber startup
//! path, starts from its roots (every `env_fallback_*` helper,
//! `ServerStartupInput::into_startup_config`, and both binaries'
//! `build_startup_input`), follows calls to functions defined in those same
//! files, and fails on any `tracing` macro it reaches. Such a call is written
//! to nothing; it has to become a [`super::StartupNotice`] instead.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use super::{StartupNotice, StartupNoticeLevel};

#[test]
fn startup_notice_constructors_set_the_level() {
    let info = StartupNotice::info("flag wins");
    assert_eq!(info.level, StartupNoticeLevel::Info);
    assert_eq!(info.message, "flag wins");
    let warn = StartupNotice::warn(String::from("bad value"));
    assert_eq!(warn.level, StartupNoticeLevel::Warn);
    assert!(warn.contains("bad"));
    assert_eq!(warn.to_string(), "bad value");
}

// ── Source scan ──────────────────────────────────────────────────────────────

/// One source file of the scan: a path for messages and its text.
struct SourceFile {
    path: String,
    text: String,
}

/// A function definition found in a [`SourceFile`].
struct FnDef {
    name: String,
    file: String,
    line: usize,
    /// The body with comments and literal contents blanked.
    body: String,
}

/// A `tracing` macro reached from a root.
#[derive(Debug)]
struct Violation {
    file: String,
    line: usize,
    function: String,
    root: String,
}

/// Replace comments and the contents of string and char literals with spaces,
/// keeping every byte offset and newline, so the scan below never matches a
/// macro name or a brace inside a doc comment or a format string.
fn code_only(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = bytes.to_vec();
    let n = bytes.len();
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for b in out.iter_mut().take(to.min(n)).skip(from) {
            if *b != b'\n' {
                *b = b' ';
            }
        }
    };
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut i = 0;
    while i < n {
        let b = bytes[i];
        if b == b'/' && i + 1 < n && bytes[i + 1] == b'/' {
            let end = bytes[i..]
                .iter()
                .position(|&c| c == b'\n')
                .map_or(n, |p| i + p);
            blank(&mut out, i, end);
            i = end;
        } else if b == b'/' && i + 1 < n && bytes[i + 1] == b'*' {
            let mut depth = 0usize;
            let mut j = i;
            while j < n {
                if bytes[j] == b'/' && j + 1 < n && bytes[j + 1] == b'*' {
                    depth += 1;
                    j += 2;
                } else if bytes[j] == b'*' && j + 1 < n && bytes[j + 1] == b'/' {
                    depth -= 1;
                    j += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            blank(&mut out, i, j);
            i = j;
        } else if b == b'r'
            && (i == 0 || !is_ident(bytes[i - 1]) || bytes[i - 1] == b'b')
            && i + 1 < n
            && (bytes[i + 1] == b'"' || bytes[i + 1] == b'#')
        {
            // Raw string `r"..."` / `r#"..."#` (also `br#"..."#`).
            let mut j = i + 1;
            let mut hashes = 0;
            while j < n && bytes[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if j < n && bytes[j] == b'"' {
                let start = j + 1;
                let mut k = start;
                'raw: while k < n {
                    if bytes[k] == b'"' && bytes[k + 1..].iter().take(hashes).all(|&c| c == b'#') {
                        break 'raw;
                    }
                    k += 1;
                }
                blank(&mut out, start, k);
                i = k + 1 + hashes;
            } else {
                // `r#ident` raw identifier.
                i += 1;
            }
        } else if b == b'"' {
            let start = i + 1;
            let mut k = start;
            while k < n && bytes[k] != b'"' {
                if bytes[k] == b'\\' {
                    k += 1;
                }
                k += 1;
            }
            blank(&mut out, start, k);
            i = k + 1;
        } else if b == b'\'' {
            // A char literal (`'x'`, `'\n'`, `'\u{..}'`, `'{'`) or a lifetime.
            if i + 1 < n && bytes[i + 1] == b'\\' {
                // Skip the backslash and the escaped character, so `'\''`
                // ends at its closing quote.
                let mut k = i + 3;
                while k < n && bytes[k] != b'\'' {
                    k += 1;
                }
                blank(&mut out, i + 1, k);
                i = k + 1;
            } else {
                let ch_len = src[i + 1..].chars().next().map_or(0, char::len_utf8);
                if ch_len > 0 && i + 1 + ch_len < n && bytes[i + 1 + ch_len] == b'\'' {
                    blank(&mut out, i + 1, i + 1 + ch_len);
                    i += 2 + ch_len;
                } else {
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
    }
    String::from_utf8(out).expect("blanking keeps UTF-8 boundaries")
}

/// The index of the brace that closes the one at `open`.
fn matching_brace(code: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, &b) in code[open..].iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every identifier in `code` with its byte offset.
fn identifiers(code: &str) -> Vec<(usize, &str)> {
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push((start, &code[start..i]));
        } else {
            i += 1;
        }
    }
    out
}

/// The byte spans of `#[cfg(test)]` modules, whose functions never run at
/// startup.
fn test_module_spans(code: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut search = 0;
    while let Some(pos) = code[search..].find("#[cfg(test)]") {
        let attr = search + pos;
        let rest = &code[attr + "#[cfg(test)]".len()..];
        let trimmed = rest.trim_start();
        if trimmed.starts_with("mod ") {
            let after = attr + "#[cfg(test)]".len() + (rest.len() - trimmed.len());
            let brace = code[after..].find(['{', ';']).map(|p| after + p);
            if let Some(brace) = brace
                && code.as_bytes()[brace] == b'{'
                && let Some(close) = matching_brace(code.as_bytes(), brace)
            {
                spans.push((brace, close));
            }
        }
        search = attr + 1;
    }
    spans
}

/// Every function with a body in `file`, outside `#[cfg(test)]` modules.
fn function_defs(file: &SourceFile) -> Vec<FnDef> {
    let code = code_only(&file.text);
    let bytes = code.as_bytes();
    let skip = test_module_spans(&code);
    let ids = identifiers(&code);
    let mut defs = Vec::new();
    for window in ids.windows(2) {
        let [(fn_pos, keyword), (name_pos, name)] = window else {
            continue;
        };
        if *keyword != "fn" || skip.iter().any(|&(a, b)| (a..=b).contains(fn_pos)) {
            continue;
        }
        // The body opens at the first `{` outside the signature's parentheses
        // and brackets; a `;` there first means a declaration without one.
        let mut depth = 0i32;
        let mut open = None;
        for (offset, &b) in bytes[name_pos + name.len()..].iter().enumerate() {
            match b {
                b'(' | b'[' => depth += 1,
                b')' | b']' => depth -= 1,
                b';' if depth == 0 => break,
                b'{' if depth == 0 => {
                    open = Some(name_pos + name.len() + offset);
                    break;
                }
                _ => {}
            }
        }
        let Some(open) = open else { continue };
        let Some(close) = matching_brace(bytes, open) else {
            continue;
        };
        defs.push(FnDef {
            name: (*name).to_string(),
            file: file.path.clone(),
            line: code[..*fn_pos].matches('\n').count() + 1,
            body: code[open..=close].to_string(),
        });
    }
    defs
}

/// Names of the functions `body` calls (free, path, or method calls).
fn called_names(body: &str) -> BTreeSet<&str> {
    let bytes = body.as_bytes();
    identifiers(body)
        .into_iter()
        .filter(|&(pos, name)| {
            let after = pos + name.len();
            let next = bytes[after..].iter().find(|b| !b.is_ascii_whitespace());
            next == Some(&b'(')
        })
        .map(|(_, name)| name)
        .collect()
}

/// Line offsets (within `body`) of every `tracing`-style logging macro.
fn logging_macro_lines(body: &str) -> Vec<usize> {
    const MACROS: [&str; 6] = ["trace", "debug", "info", "warn", "error", "event"];
    let bytes = body.as_bytes();
    identifiers(body)
        .into_iter()
        .filter(|&(pos, name)| {
            let after = pos + name.len();
            MACROS.contains(&name) && bytes.get(after) == Some(&b'!')
        })
        .map(|(pos, _)| body[..pos].matches('\n').count())
        .collect()
}

/// Whether `name` is a root of the pre-subscriber startup path.
fn is_root(name: &str) -> bool {
    name.starts_with("env_fallback_")
        || name == "into_startup_config"
        || name == "build_startup_input"
}

/// Follow calls from the roots through the functions defined in `files` and
/// report every logging macro reached.
fn scan(files: &[SourceFile]) -> (Vec<Violation>, Vec<String>) {
    let defs: Vec<FnDef> = files.iter().flat_map(function_defs).collect();
    let mut by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, def) in defs.iter().enumerate() {
        by_name.entry(def.name.as_str()).or_default().push(index);
    }
    let mut roots = Vec::new();
    let mut origin: BTreeMap<usize, String> = BTreeMap::new();
    let mut queue = VecDeque::new();
    for (index, def) in defs.iter().enumerate() {
        if is_root(&def.name) {
            roots.push(format!("{}:{} {}", def.file, def.line, def.name));
            origin.insert(index, def.name.clone());
            queue.push_back(index);
        }
    }
    while let Some(index) = queue.pop_front() {
        let root = origin[&index].clone();
        for name in called_names(&defs[index].body) {
            for &callee in by_name.get(name).into_iter().flatten() {
                if let std::collections::btree_map::Entry::Vacant(slot) = origin.entry(callee) {
                    slot.insert(root.clone());
                    queue.push_back(callee);
                }
            }
        }
    }
    let mut violations = Vec::new();
    for (&index, root) in &origin {
        let def = &defs[index];
        for offset in logging_macro_lines(&def.body) {
            violations.push(Violation {
                file: def.file.clone(),
                line: def.line + offset,
                function: def.name.clone(),
                root: root.clone(),
            });
        }
    }
    (violations, roots)
}

/// The files that hold the pre-subscriber startup path.
fn startup_sources() -> Vec<SourceFile> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths: Vec<PathBuf> = [
        "src/server/cli_input.rs",
        "src/server/thinking_budget.rs",
        "src/server/chat_template_kwargs.rs",
        "src/server/model_source.rs",
        "src/bin/mlx_server.rs",
        "src/commands/serve.rs",
    ]
    .iter()
    .map(|p| root.join(p))
    .collect();
    let cli_dir = root.join("src/cli");
    let mut args_files: Vec<PathBuf> = std::fs::read_dir(&cli_dir)
        .expect("read src/cli")
        .map(|entry| entry.expect("src/cli entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_args.rs"))
        })
        .collect();
    args_files.sort();
    paths.extend(args_files);
    paths
        .into_iter()
        .map(|path| SourceFile {
            text: std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
            path: path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string(),
        })
        .collect()
}

/// #2722: nothing on the pre-subscriber startup path calls a `tracing` macro.
/// A line raised there must be pushed as a [`StartupNotice`] so
/// `start_server` logs it once the subscriber exists.
#[test]
fn startup_notices_guard_finds_no_tracing_before_the_subscriber() {
    let (violations, roots) = scan(&startup_sources());
    // The scan must keep reaching the code it guards; a rename that drops a
    // root would otherwise pass vacuously.
    for expected in [
        "src/server/cli_input.rs",
        "src/bin/mlx_server.rs",
        "src/commands/serve.rs",
        "src/cli/turbo_args.rs",
        "src/cli/speculative_args.rs",
    ] {
        assert!(
            roots.iter().any(|root| root.starts_with(expected)),
            "no startup root found in {expected}: {roots:#?}"
        );
    }
    for name in ["into_startup_config", "build_startup_input"] {
        assert!(
            roots.iter().any(|root| root.ends_with(&format!(" {name}"))),
            "{name} is not a scan root: {roots:#?}"
        );
    }
    assert_eq!(
        roots
            .iter()
            .filter(|root| root.ends_with(" build_startup_input"))
            .count(),
        2,
        "both binaries' build_startup_input must be scanned: {roots:#?}"
    );
    assert!(
        violations.is_empty(),
        "these run before the tracing subscriber exists, so the macro writes to \
         nothing; push a StartupNotice instead:\n{}",
        violations
            .iter()
            .map(|v| format!(
                "  {}:{} in {} (reached from {})",
                v.file, v.line, v.function, v.root
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// #2722 acceptance: a `tracing::warn!` inserted into a real env_fallback
/// helper fails the guard.
#[test]
fn startup_notices_guard_flags_a_tracing_call_in_an_env_fallback_helper() {
    let mut files = startup_sources();
    let cli_input = files
        .iter_mut()
        .find(|file| file.path == "src/server/cli_input.rs")
        .expect("cli_input.rs is scanned");
    let signature = "pub fn env_fallback_kv_bits(";
    let at = cli_input
        .text
        .find(signature)
        .expect("env_fallback_kv_bits exists");
    let open = at + cli_input.text[at..].find('{').expect("body");
    cli_input.text.insert_str(
        open + 1,
        "\n    tracing::warn!(\"lost before the subscriber\");",
    );

    let (violations, _) = scan(&files);
    assert!(
        violations
            .iter()
            .any(|v| v.function == "env_fallback_kv_bits" && v.file == "src/server/cli_input.rs"),
        "the inserted warning must be reported: {violations:?}"
    );
}

/// The scan follows helpers, and ignores comments, strings, unreached
/// functions, and test modules.
#[test]
fn startup_notices_guard_follows_helpers_and_ignores_comments_and_strings() {
    let source = r#"
pub fn env_fallback_demo(value: &mut bool) {
    // tracing::warn!("a comment is not a call");
    let _ = "info!(inside a string) {";
    let _brace = '{';
    helper(value);
}

fn helper(_value: &mut bool) {
    tracing::info!("reached through a call");
}

fn unrelated() {
    tracing::warn!("never called from a root");
}

#[cfg(test)]
mod tests {
    fn env_fallback_in_tests() {
        tracing::warn!("test code");
    }
}
"#;
    let files = [SourceFile {
        path: "demo.rs".to_string(),
        text: source.to_string(),
    }];
    let (violations, roots) = scan(&files);
    assert_eq!(roots, vec!["demo.rs:2 env_fallback_demo".to_string()]);
    assert_eq!(violations.len(), 1, "{violations:?}");
    let violation = &violations[0];
    assert_eq!(
        (
            violation.function.as_str(),
            violation.line,
            violation.root.as_str()
        ),
        ("helper", 10, "env_fallback_demo")
    );
}
