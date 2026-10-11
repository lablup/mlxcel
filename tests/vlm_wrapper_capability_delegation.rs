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

//! A VLM wrapper may not silently re-enable a padded prefill its own text
//! backbone disabled.
//!
//! `LanguageModel::supports_padded_prefill` defaults to `true`. Every hybrid
//! and recurrent text model in this tree overrides it to `false`, and the
//! comments say why: tile-aligned padded prefill appends up to 31 pad
//! positions, and although the causal mask and `trim_caches_to_actual_len`
//! undo their effect on the KV caches, a Mamba / GatedDeltaNet / RWKV /
//! DeltaCache state that has already absorbed them cannot be rewound.
//!
//! A vision wrapper that forwards to such a backbone but does not forward this
//! predicate answers `true` by default, and the offline generator then pads.
//! Nothing fails: it compiles, it runs, and greedy output silently changes on
//! Neural Accelerator hardware whenever the prompt length is not a multiple of
//! 32. That is #1201, found only because a speculative path that never pads
//! disagreed with the classic path that did.
//!
//! This is a source-level check rather than a runtime one because constructing
//! every wrapper needs weights. The property is about which method each `impl`
//! block carries, which the source states directly.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const PREDICATES: [&str; 2] = [
    "supports_padded_prefill",
    "supports_maskless_padded_prefill",
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The body of the `impl LanguageModel for X` block in `src`, if any, plus the
/// type name `X`.
///
/// Brace-counted from the impl header rather than regex-matched, so a nested
/// block or a later `impl` for a different trait cannot leak in.
fn language_model_impl(src: &str) -> Option<(String, String)> {
    let header = src.find("impl LanguageModel for ")?;
    let after = &src[header + "impl LanguageModel for ".len()..];
    let name: String = after
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    let open = header + after.find('{')? + "impl LanguageModel for ".len();
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((name, src[open..open + i].to_string()));
                }
            }
            _ => {}
        }
    }
    None
}

/// Text-model types whose `LanguageModel` impl answers `false` for `predicate`.
fn backbones_answering_false(predicate: &str) -> BTreeSet<String> {
    let mut files = Vec::new();
    rust_sources(&repo_root().join("src/models"), &mut files);
    let needle = format!("fn {predicate}(&self) -> bool {{");
    let mut out = BTreeSet::new();
    for file in files {
        let Ok(src) = fs::read_to_string(&file) else {
            continue;
        };
        let Some((name, body)) = language_model_impl(&src) else {
            continue;
        };
        if let Some(at) = body.find(&needle) {
            let tail = &body[at + needle.len()..];
            let answer: String = tail
                .chars()
                .take_while(|c| *c != '}')
                .filter(|c| !c.is_whitespace())
                .collect();
            // Skip anything that is not a bare literal: a computed answer is
            // already delegating or deciding for itself.
            if answer == "false" {
                out.insert(name);
            }
        }
    }
    out
}

/// Types a wrapper holds as a field, which is what "wraps" has to mean here.
///
/// Matching a bare mention would fire on a doc comment or a use statement and
/// turn this test into noise the first time someone writes prose about a
/// backbone they do not embed.
fn field_types(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in src.lines() {
        let line = line.trim();
        if line.starts_with("//") {
            continue;
        }
        let Some((_, ty)) = line.split_once(':') else {
            continue;
        };
        for token in ty.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
            if !token.is_empty() {
                out.insert(token.to_string());
            }
        }
    }
    out
}

/// The brace-counted body of the first function whose header contains
/// `signature`.
fn function_body<'a>(src: &'a str, signature: &str) -> Option<&'a str> {
    let header = src.find(signature)?;
    let open = header + src[header..].find('{')?;
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[open + 1..open + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Locals that visibly derive from the padded-prefill model predicate.
///
/// The prefill planner centralizes the predicate in `PrefillCaps::can_pad`.
/// Recognize `caps.can_pad()` only after inspecting that helper's body; a
/// same-named helper without `supports_padded_prefill` provides no evidence.
fn padded_prefill_guard_aliases(src: &str) -> Vec<String> {
    let shared_caps_guarded = function_body(src, "fn can_pad(&self) -> bool")
        .is_some_and(|body| body.contains("supports_padded_prefill"));
    let mut aliases = Vec::new();
    for line in src.lines() {
        let Some(rest) = line.trim().strip_prefix("let ") else {
            continue;
        };
        if !line.contains("supports_padded_prefill")
            && !(shared_caps_guarded && line.contains("caps.can_pad()"))
        {
            continue;
        }
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            aliases.push(name);
        }
    }
    aliases
}

fn unguarded_tile_aligned_prefills(src: &str) -> Vec<(usize, String)> {
    const WINDOW: usize = 12;
    let lines: Vec<&str> = src.lines().collect();
    let aliases = padded_prefill_guard_aliases(src);
    let mut unguarded = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        // The definition, its doc examples and its own unit tests are not
        // prefill sites and have nothing to guard.
        if !line.contains("align_to_na_tile(")
            || trimmed.starts_with("//")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("assert")
        {
            continue;
        }
        let start = i.saturating_sub(WINDOW);
        if lines[start..=i].iter().any(|l| {
            l.contains("supports_padded_prefill") || aliases.iter().any(|a| l.contains(a.as_str()))
        }) {
            continue;
        }
        unguarded.push((i + 1, line.trim().to_string()));
    }
    unguarded
}

#[test]
fn a_vision_wrapper_delegates_padded_prefill_when_its_backbone_refuses_it() {
    let mut files = Vec::new();
    rust_sources(&repo_root().join("src/vision"), &mut files);
    files.sort();

    let mut violations = Vec::new();
    for predicate in PREDICATES {
        let refusing = backbones_answering_false(predicate);
        assert!(
            refusing.contains("Qwen35Model") || predicate.contains("maskless"),
            "expected at least the Qwen 3.5 backbone to refuse {predicate}; the \
             scanner probably stopped matching the source shape"
        );
        for file in &files {
            let Ok(src) = fs::read_to_string(file) else {
                continue;
            };
            let Some((wrapper, body)) = language_model_impl(&src) else {
                continue;
            };
            if body.contains(&format!("fn {predicate}(")) {
                continue;
            }
            let fields = field_types(&src);
            let wrapped: Vec<&String> = refusing.intersection(&fields).collect();
            if wrapped.is_empty() {
                continue;
            }
            violations.push(format!(
                "{}: `{wrapper}` holds {} but does not override `{predicate}`, so it \
                 answers the trait default `true` and re-enables a padded prefill its \
                 backbone disabled",
                file.strip_prefix(repo_root()).unwrap_or(file).display(),
                wrapped
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "vision wrappers must forward the padded-prefill predicates to their text \
         backbone (#1201). Add a delegating override to each of:\n  {}",
        violations.join("\n  ")
    );
}

/// Every tile-aligned padded prefill must be guarded on the model predicate.
///
/// The wrapper test above covers one way to lose the guard: a type that
/// answers the trait default instead of forwarding. This covers the other:
/// a call site that never asks. `src/server/batch/scheduler.rs` had three,
/// `execute_full_prefill` and both chunked-prefill paths, each gated on
/// hardware alone, so every hybrid model served on Neural Accelerator
/// hardware had its recurrent state corrupted by pad positions. The same
/// file's boundary-snapshot path documents the hazard in a comment and
/// declines to pad, which is what makes the omission a slip rather than a
/// disagreement.
///
/// The rule is deliberately shallow: the guard must appear within the dozen
/// lines before the call, either as the predicate itself or as a local bound
/// to it (`let can_pad_prefill = self.model.supports_padded_prefill()`, which
/// the batched path does read fourteen lines up). A guard further away than
/// that is one a reader cannot see either.
#[test]
fn every_tile_aligned_prefill_is_guarded_on_the_model_predicate() {
    let mut files = Vec::new();
    rust_sources(&repo_root().join("src"), &mut files);
    files.sort();

    let mut unguarded = Vec::new();
    for file in &files {
        let Ok(src) = fs::read_to_string(file) else {
            continue;
        };
        for (line_number, line) in unguarded_tile_aligned_prefills(&src) {
            unguarded.push(format!(
                "{}:{}: {}",
                file.strip_prefix(repo_root()).unwrap_or(file).display(),
                line_number,
                line
            ));
        }
    }

    assert!(
        unguarded.is_empty(),
        "a tile-aligned prefill must be gated on the model's own \
         `supports_padded_prefill()`, not on hardware alone: padding a hybrid \
         model corrupts its conv / SSM / GatedDeltaNet recurrent state, which \
         trimming the KV caches afterwards does not undo. Unguarded sites:\n  {}",
        unguarded.join("\n  ")
    );
}

#[test]
fn shared_can_pad_alias_requires_the_model_predicate() {
    let guarded = r#"
impl PrefillCaps {
    fn can_pad(&self) -> bool {
        self.align_prefill && self.supports_padded_prefill
    }
}

fn padded_len(caps: PrefillCaps, len: usize) -> usize {
    let can_pad = caps.can_pad();
    if can_pad { align_to_na_tile(len) } else { len }
}
"#;
    assert!(unguarded_tile_aligned_prefills(guarded).is_empty());

    let mutated = guarded.replace(
        "self.align_prefill && self.supports_padded_prefill",
        "self.align_prefill",
    );
    assert_eq!(unguarded_tile_aligned_prefills(&mutated).len(), 1);

    let direct = r#"
fn padded_len(model: &Model, len: usize) -> usize {
    if model.supports_padded_prefill() {
        align_to_na_tile(len)
    } else {
        len
    }
}
"#;
    assert!(unguarded_tile_aligned_prefills(direct).is_empty());
}
