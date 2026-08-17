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
