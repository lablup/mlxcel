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
//! The guard scans every `.rs` file under `src/`, starts from its roots
//! (every `env_fallback_*` helper, `ServerStartupInput::into_startup_config`,
//! and both binaries' `build_startup_input`), and follows calls with module
//! and `impl`-type awareness: `Type::f` reaches that type's methods,
//! `module::f` reaches that module's free functions, a bare `f` reaches free
//! functions in the same file or brought in by `use`, and `.f` reaches `impl`
//! methods. It fails on any `tracing` macro it reaches. Such a call is
//! written to nothing; it has to become a [`super::StartupNotice`] instead.

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
    /// The module path the file defines, e.g. `server::cli_input`.
    module: String,
    /// The `impl` type enclosing the function, when there is one.
    impl_type: Option<String>,
    line: usize,
    /// The body with comments and literal contents blanked.
    body: String,
}

/// How a call was written, which decides how its callee is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallKind {
    /// `f(...)`: a free function in this file, or one a `use` brought in.
    Bare,
    /// `Type::f(...)` or `module::f(...)`.
    Path,
    /// `receiver.f(...)`: an `impl` method, with an unknown receiver type.
    Method,
}

/// A call found in a function body.
#[derive(Debug)]
struct Call {
    kind: CallKind,
    /// The `::`-separated segments before the function name, if any.
    qualifiers: Vec<String>,
    name: String,
}

/// The names a file's `use` statements bring into scope.
#[derive(Debug, Default)]
struct Imports {
    /// Local name (the alias or the last path segment) to the full path.
    named: BTreeMap<String, Vec<String>>,
    /// Paths imported with a trailing `::*`.
    globs: Vec<Vec<String>>,
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

/// The module path a source file defines, derived from its `src/`-relative
/// path (`src/server/cli_input.rs` is `server::cli_input`).
fn module_path(file: &str) -> String {
    let rel = file.strip_prefix("src/").unwrap_or(file);
    let rel = rel.strip_suffix(".rs").unwrap_or(rel);
    let rel = rel.strip_suffix("/mod").unwrap_or(rel);
    if rel == "lib" || rel == "main" {
        return String::new();
    }
    rel.replace('/', "::")
}

/// The `impl` blocks in `code` as `(open brace, close brace, type name)`.
fn impl_spans(code: &str) -> Vec<(usize, usize, String)> {
    let bytes = code.as_bytes();
    let ids = identifiers(code);
    let mut spans = Vec::new();
    for &(pos, name) in &ids {
        if name != "impl" {
            continue;
        }
        if pos > 0 && (bytes[pos - 1].is_ascii_alphanumeric() || bytes[pos - 1] == b'_') {
            continue;
        }
        // The header ends at its body brace; a top-level `;` means no body.
        let mut depth = 0i32;
        let mut open = None;
        let mut end = pos + name.len();
        for (offset, &b) in bytes[pos + name.len()..].iter().enumerate() {
            let at = pos + name.len() + offset;
            match b {
                b'(' | b'[' | b'<' => depth += 1,
                b')' | b']' | b'>' => depth -= 1,
                b';' if depth == 0 => {
                    end = at;
                    break;
                }
                b'{' if depth == 0 => {
                    open = Some(at);
                    end = at;
                    break;
                }
                _ => {}
            }
        }
        let Some(open) = open else { continue };
        let Some(close) = matching_brace(bytes, open) else {
            continue;
        };
        if let Some(impl_type) = signature_type(&code[pos..end]) {
            spans.push((open, close, impl_type));
        }
    }
    spans
}

/// The type named last at the top level of an `impl` header: `Foo` for both
/// `impl Foo` and `impl Trait for Foo`, with generics skipped.
fn signature_type(signature: &str) -> Option<String> {
    let bytes = signature.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    let mut last = None;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => depth -= 1,
            b if b.is_ascii_alphabetic() || b == b'_' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                if depth == 0 {
                    let word = &signature[start..i];
                    if !matches!(
                        word,
                        "impl" | "for" | "where" | "unsafe" | "const" | "dyn" | "mut"
                    ) {
                        last = Some(word.to_string());
                    }
                }
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    last
}

/// Every function with a body in `file`, outside `#[cfg(test)]` modules,
/// tagged with its module path and enclosing `impl` type.
fn function_defs(file: &SourceFile) -> Vec<FnDef> {
    let code = code_only(&file.text);
    let bytes = code.as_bytes();
    let skip = test_module_spans(&code);
    let ids = identifiers(&code);
    let impls = impl_spans(&code);
    let module = module_path(&file.path);
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
        let impl_type = impls
            .iter()
            .filter(|(impl_open, impl_close, _)| *impl_open < *name_pos && *name_pos < *impl_close)
            .max_by_key(|(impl_open, _, _)| *impl_open)
            .map(|(_, _, impl_type)| impl_type.clone());
        defs.push(FnDef {
            name: (*name).to_string(),
            file: file.path.clone(),
            module: module.clone(),
            impl_type,
            line: code[..*fn_pos].matches('\n').count() + 1,
            body: code[open..=close].to_string(),
        });
    }
    defs
}

/// Every call in `body`, classified by how it is written.
fn calls(body: &str) -> Vec<Call> {
    let bytes = body.as_bytes();
    let mut out = Vec::new();
    for &(pos, name) in &identifiers(body) {
        let after = pos + name.len();
        let next = bytes[after..].iter().find(|b| !b.is_ascii_whitespace());
        if next != Some(&b'(') {
            continue;
        }
        let prev = body[..pos].bytes().rev().find(|b| !b.is_ascii_whitespace());
        if prev == Some(b'.') {
            out.push(Call {
                kind: CallKind::Method,
                qualifiers: Vec::new(),
                name: name.to_string(),
            });
            continue;
        }
        let qualifiers = path_qualifiers(body, pos);
        out.push(Call {
            kind: if qualifiers.is_empty() {
                CallKind::Bare
            } else {
                CallKind::Path
            },
            qualifiers,
            name: name.to_string(),
        });
    }
    out
}

/// The `::`-separated segments immediately before the identifier at `pos`.
fn path_qualifiers(body: &str, pos: usize) -> Vec<String> {
    let bytes = body.as_bytes();
    let mut end = pos;
    let mut out = Vec::new();
    loop {
        let mut k = end;
        while k > 0 && bytes[k - 1].is_ascii_whitespace() {
            k -= 1;
        }
        if k < 2 || bytes[k - 1] != b':' || bytes[k - 2] != b':' {
            break;
        }
        k -= 2;
        while k > 0 && bytes[k - 1].is_ascii_whitespace() {
            k -= 1;
        }
        let token_end = k;
        while k > 0 && (bytes[k - 1].is_ascii_alphanumeric() || bytes[k - 1] == b'_') {
            k -= 1;
        }
        if k == token_end {
            break;
        }
        out.push(body[k..token_end].to_string());
        end = k;
    }
    out.reverse();
    out
}

/// The names `file`'s `use` statements bring into scope.
fn imports(file: &SourceFile) -> Imports {
    let code = code_only(&file.text);
    let mut map = Imports::default();
    for window in identifiers(&code).windows(2) {
        let [(pos, keyword), _] = window else {
            continue;
        };
        if *keyword != "use" {
            continue;
        }
        let Some(len) = code[*pos..].find(';') else {
            continue;
        };
        let tree = code[pos + keyword.len()..pos + len].trim();
        parse_use_tree(&[], tree, &mut map);
    }
    map
}

/// Flatten one `use` tree, which may be a path, an `as` alias, a `{...}`
/// group, or a trailing `::*`.
fn parse_use_tree(prefix: &[String], tree: &str, map: &mut Imports) {
    let tree = tree.trim();
    if tree.is_empty() {
        return;
    }
    if tree == "*" {
        map.globs.push(prefix.to_vec());
        return;
    }
    if let Some(base) = tree.strip_suffix("::*") {
        let mut full = prefix.to_vec();
        full.extend(path_segments(base));
        map.globs.push(full);
        return;
    }
    if let Some(at) = tree.find('{') {
        let base = tree[..at].trim().trim_end_matches("::").trim();
        let mut next = prefix.to_vec();
        next.extend(path_segments(base));
        let inner = tree[at + 1..].trim_end_matches('}');
        for part in split_top_level(inner, ',') {
            parse_use_tree(&next, part.trim(), map);
        }
        return;
    }
    let (path, alias) = match tree.rsplit_once(" as ") {
        Some((path, alias)) => (path.trim(), Some(alias.trim())),
        None => (tree, None),
    };
    let segments = path_segments(path);
    let Some(name) = segments.last().cloned() else {
        return;
    };
    let mut full = prefix.to_vec();
    full.extend(segments);
    map.named.insert(alias.unwrap_or(&name).to_string(), full);
}

/// Split `text` on `separator` outside braces, brackets, parentheses, and
/// angle brackets.
fn split_top_level(text: &str, separator: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        match ch {
            '{' | '[' | '(' | '<' => depth += 1,
            '}' | ']' | ')' | '>' => depth -= 1,
            ch if ch == separator && depth == 0 => {
                out.push(&text[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    out.push(&text[start..]);
    out
}

/// The non-empty segments of a `::` path, with leading `::` dropped.
fn path_segments(path: &str) -> Vec<String> {
    path.split("::")
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(ToString::to_string)
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

/// The crate modules a path written in `current` can name.
///
/// A plain path is tried as a child and a sibling of `current`, as a
/// crate-absolute path, and with a leading external-crate segment dropped, so
/// `crate::server::hidden`, `super::hidden`, `server::hidden` and
/// `mlxcel::server::hidden` all reach the module `server::hidden` where the
/// call is written.
fn resolve_modules(qualifiers: &[String], current: &str) -> Vec<String> {
    let mut segments: Vec<&str> = qualifiers.iter().map(String::as_str).collect();
    let mut base: Vec<&str> = if current.is_empty() {
        Vec::new()
    } else {
        current.split("::").collect()
    };
    let mut relative = true;
    if segments.first() == Some(&"crate") {
        segments.remove(0);
        base.clear();
    } else if segments.first() == Some(&"self") {
        segments.remove(0);
    } else if segments.first() == Some(&"super") {
        while segments.first() == Some(&"super") {
            segments.remove(0);
            base.pop();
        }
    } else {
        relative = false;
    }
    let joined = |base: &[&str], rest: &[&str]| {
        base.iter()
            .chain(rest)
            .copied()
            .collect::<Vec<_>>()
            .join("::")
    };
    let mut candidates = BTreeSet::new();
    if relative {
        candidates.insert(joined(&base, &segments));
    } else {
        candidates.insert(joined(&base, &segments));
        let parent = base.split_last().map_or(&[][..], |(_, rest)| rest);
        candidates.insert(joined(parent, &segments));
        candidates.insert(segments.join("::"));
        for skip in 1..segments.len() {
            candidates.insert(segments[skip..].join("::"));
        }
    }
    candidates
        .into_iter()
        .filter(|module| !module.is_empty())
        .collect()
}

/// The definitions a call can reach given where it is written.
fn resolve_call(
    call: &Call,
    def: &FnDef,
    free_same_file: &BTreeMap<(String, String), Vec<usize>>,
    free_by_module: &BTreeMap<(String, String), Vec<usize>>,
    methods_by_type: &BTreeMap<(String, String), Vec<usize>>,
    methods_in_module: &BTreeMap<(String, String), Vec<usize>>,
    imports: &Imports,
) -> Vec<usize> {
    let mut found = Vec::new();
    match call.kind {
        CallKind::Method => {
            // The receiver's type cannot be inferred from source, so only an
            // `impl` method defined in the caller's own module is considered.
            // Matching by name across the crate would drag in unrelated
            // same-named methods; free functions are never callees of `.f(`.
            if let Some(indices) = methods_in_module.get(&(def.module.clone(), call.name.clone())) {
                found.extend(indices);
            }
        }
        CallKind::Bare => {
            if let Some(indices) = free_same_file.get(&(def.file.clone(), call.name.clone())) {
                found.extend(indices);
            }
            if let Some(target) = imports.named.get(&call.name)
                && let Some(real_name) = target.last()
            {
                let module_segments = &target[..target.len() - 1];
                for module in resolve_modules(module_segments, &def.module) {
                    if let Some(indices) = free_by_module.get(&(module, real_name.clone())) {
                        found.extend(indices);
                    }
                }
            }
            for glob in &imports.globs {
                for module in resolve_modules(glob, &def.module) {
                    if let Some(indices) = free_by_module.get(&(module, call.name.clone())) {
                        found.extend(indices);
                    }
                }
            }
        }
        CallKind::Path => {
            if let Some(qualifier) = call.qualifiers.last()
                && let Some(indices) = methods_by_type.get(&(qualifier.clone(), call.name.clone()))
            {
                found.extend(indices);
            }
            // A path that names a type resolves to its methods; otherwise it
            // names a module and reaches that module's free functions.
            if found.is_empty() {
                for module in resolve_modules(&call.qualifiers, &def.module) {
                    if let Some(indices) = free_by_module.get(&(module, call.name.clone())) {
                        found.extend(indices);
                    }
                }
            }
        }
    }
    found.sort_unstable();
    found.dedup();
    found
}

/// Follow calls from the roots through `files` and report every logging macro
/// reached.
fn scan(files: &[SourceFile]) -> (Vec<Violation>, Vec<String>) {
    let mut defs: Vec<FnDef> = Vec::new();
    let mut imports_by_file: BTreeMap<String, Imports> = BTreeMap::new();
    for file in files {
        imports_by_file.insert(file.path.clone(), imports(file));
        defs.extend(function_defs(file));
    }
    let mut free_same_file: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    let mut free_by_module: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    let mut methods_by_type: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    let mut methods_in_module: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (index, def) in defs.iter().enumerate() {
        if let Some(impl_type) = &def.impl_type {
            methods_by_type
                .entry((impl_type.clone(), def.name.clone()))
                .or_default()
                .push(index);
            methods_in_module
                .entry((def.module.clone(), def.name.clone()))
                .or_default()
                .push(index);
        } else {
            free_same_file
                .entry((def.file.clone(), def.name.clone()))
                .or_default()
                .push(index);
            free_by_module
                .entry((def.module.clone(), def.name.clone()))
                .or_default()
                .push(index);
        }
    }
    let no_imports = Imports::default();
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
        let def = &defs[index];
        let file_imports = imports_by_file.get(&def.file).unwrap_or(&no_imports);
        for call in calls(&def.body) {
            let callees = resolve_call(
                &call,
                def,
                &free_same_file,
                &free_by_module,
                &methods_by_type,
                &methods_in_module,
                file_imports,
            );
            for callee in callees {
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

/// Every Rust source under `src/`.
///
/// The guard used to read a fixed file list plus `src/cli/*_args.rs`, so a
/// helper in any other file hid its `tracing::` call. The pre-subscriber path
/// can reach any module, so every compiled source file is scanned. Files named
/// `*_tests.rs` are only compiled inside `#[cfg(test)]` and cannot run at
/// startup, so they are left out.
fn startup_sources() -> Vec<SourceFile> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths = Vec::new();
    collect_rust_sources(&root.join("src"), &mut paths);
    paths.sort();
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

/// Append every `.rs` file under `dir`, recursively and in sorted order.
fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("source directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rust_sources(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            let test_only = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_tests.rs") || name.ends_with("_test.rs"));
            if !test_only {
                out.push(path);
            }
        }
    }
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

/// A helper in a module outside the old fixed file list is still reached and
/// flagged when a root calls it.
#[test]
fn startup_notices_guard_reaches_a_helper_in_a_file_outside_the_old_list() {
    let files = [
        SourceFile {
            path: "src/server/root_module.rs".to_string(),
            text: r#"
pub fn env_fallback_demo() {
    super::hidden::warn_helper();
}
"#
            .to_string(),
        },
        SourceFile {
            path: "src/server/hidden.rs".to_string(),
            text: r#"
pub fn warn_helper() {
    tracing::warn!("hidden before the subscriber");
}
"#
            .to_string(),
        },
    ];
    let (violations, _) = scan(&files);
    assert!(
        violations
            .iter()
            .any(|v| v.function == "warn_helper" && v.file == "src/server/hidden.rs"),
        "a helper outside the old fixed list must be scanned: {violations:?}"
    );
}

/// An unrelated free function that shares a callee's name but lives in
/// another module is not pulled into the scan.
#[test]
fn startup_notices_guard_does_not_follow_unrelated_same_named_functions() {
    let files = [
        SourceFile {
            path: "src/server/root_module.rs".to_string(),
            text: r#"
pub fn env_fallback_demo() {
    helper();
}

fn helper() {}
"#
            .to_string(),
        },
        SourceFile {
            path: "src/server/other.rs".to_string(),
            text: r#"
fn helper() {
    tracing::warn!("unrelated function in another module");
}
"#
            .to_string(),
        },
    ];
    let (violations, _) = scan(&files);
    assert!(
        violations.is_empty(),
        "a bare call resolves in its own file, not by name across modules: {violations:?}"
    );
}
