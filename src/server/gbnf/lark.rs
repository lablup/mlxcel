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

//! Lowering from b10621's GBNF element model to the `llguidance` Lark dialect.
//!
//! Two properties of the target decide the shape of the output.
//!
//! 1. `llguidance` lexes maximally. A regex terminal that can match more than
//!    one code point swallows what the next grammar element was meant to
//!    consume, so `start: /[ab]+/ "b"` accepts nothing at all. GBNF is
//!    code-point structured, so every terminal here is emitted as a
//!    **single-code-point** character class and every repetition stays at the
//!    grammar level, where the Earley parser resolves it.
//! 2. Rule names are emitted as `r<id>`, never the GBNF name. GBNF identifiers
//!    admit `-` and a leading digit and are case-sensitive; Lark reads an
//!    upper-case name as a terminal. Renaming sidesteps the whole question.
//!
//! Token terminals map exactly: GBNF `<[id]>` is Lark `<[id]>` and GBNF
//! `!<...>` is Lark's negated token range `<[^id]>`, so the one GBNF feature
//! that a purely code-point engine could not express needs no fallback here.
//!
//! Used by: server::gbnf

use super::GbnfError;
use super::parser::{Elem, GbnfGrammar};

/// Upper bound on the Lark text handed to `llguidance`.
///
/// GBNF repetition desugaring is expansive by design (`x{100}` copies the
/// element list a hundred times), and b10621's own guard is a rule-count
/// budget rather than a size budget, so a grammar can pass every upstream check
/// and still lower to something large.
const MAX_LARK_BYTES: usize = 4 * 1024 * 1024;

/// Highest Unicode scalar value; `\x{...}` above this is not a valid regex
/// class member.
const MAX_SCALAR: u32 = 0x10_FFFF;
const SURROGATE_LO: u32 = 0xD800;
const SURROGATE_HI: u32 = 0xDFFF;

/// A class that matches every code point, used for GBNF `.` and for a negated
/// class whose members are all unrepresentable.
const ANY_CLASS: &str = "/[\\s\\S]/";
/// A class that matches nothing, used for a positive class with no
/// representable members (GBNF `[]` never reaches here; an all-out-of-range
/// class does).
const NEVER_CLASS: &str = "/[^\\s\\S]/";

/// Clamp one inclusive GBNF code-point range onto the Unicode scalar space,
/// splitting around the surrogate block.
///
/// An inverted range (`lo > hi`, which GBNF writes as `[z-a]`) matches nothing
/// upstream, and a range entirely above U+10FFFF is unreachable through UTF-8
/// input, so both drop out. Dropping is correct for a negated class too: a
/// member that no input can equal cannot exclude anything.
fn scalar_ranges(lo: u32, hi: u32, out: &mut Vec<(u32, u32)>) {
    if lo > hi || lo > MAX_SCALAR {
        return;
    }
    let hi = hi.min(MAX_SCALAR);
    if hi < SURROGATE_LO || lo > SURROGATE_HI {
        out.push((lo, hi));
        return;
    }
    if lo < SURROGATE_LO {
        out.push((lo, SURROGATE_LO - 1));
    }
    if hi > SURROGATE_HI {
        out.push((SURROGATE_HI + 1, hi));
    }
}

fn render_class(negated: bool, raw: &[(u32, u32)]) -> String {
    let mut ranges: Vec<(u32, u32)> = Vec::with_capacity(raw.len());
    for &(lo, hi) in raw {
        scalar_ranges(lo, hi, &mut ranges);
    }
    if ranges.is_empty() {
        return if negated { ANY_CLASS } else { NEVER_CLASS }.to_string();
    }
    let mut body = String::new();
    for (lo, hi) in ranges {
        if lo == hi {
            body.push_str(&format!("\\x{{{lo:X}}}"));
        } else {
            body.push_str(&format!("\\x{{{lo:X}}}-\\x{{{hi:X}}}"));
        }
    }
    if negated {
        format!("/[^{body}]/")
    } else {
        format!("/[{body}]/")
    }
}

/// Emit one alternative, advancing `i` to the terminating `Alt` or `End`.
fn emit_alternative(rule: &[Elem], i: &mut usize, out: &mut String) {
    while *i < rule.len() && !matches!(rule[*i], Elem::Alt | Elem::End) {
        if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
        match rule[*i] {
            Elem::Char(c) | Elem::CharNot(c) => {
                let negated = matches!(rule[*i], Elem::CharNot(_));
                let mut parts: Vec<(u32, u32)> = Vec::new();
                *i += 1;
                let mut lo = c;
                let mut hi = c;
                if let Some(Elem::CharRngUpper(u)) = rule.get(*i).copied() {
                    hi = u;
                    *i += 1;
                }
                parts.push((lo, hi));
                while let Some(Elem::CharAlt(c2)) = rule.get(*i).copied() {
                    *i += 1;
                    lo = c2;
                    hi = c2;
                    if let Some(Elem::CharRngUpper(u)) = rule.get(*i).copied() {
                        hi = u;
                        *i += 1;
                    }
                    parts.push((lo, hi));
                }
                out.push_str(&render_class(negated, &parts));
            }
            Elem::CharAny => {
                out.push_str(ANY_CLASS);
                *i += 1;
            }
            Elem::Token(id) => {
                out.push_str(&format!("<[{id}]>"));
                *i += 1;
            }
            Elem::TokenNot(id) => {
                out.push_str(&format!("<[^{id}]>"));
                *i += 1;
            }
            Elem::RuleRef(v) => {
                out.push_str(&format!("r{v}"));
                *i += 1;
            }
            // A stray range/alt marker cannot start an item; skip it rather
            // than looping forever. The parser never emits one.
            Elem::CharRngUpper(_) | Elem::CharAlt(_) => {
                *i += 1;
            }
            Elem::Alt | Elem::End => break,
        }
    }
}

/// Rule ids reachable from the root, so an unreferenced rule does not reach
/// `llguidance`. b10621 validates every rule but only walks the reachable ones.
fn reachable(grammar: &GbnfGrammar) -> Vec<u32> {
    let mut seen = vec![false; grammar.rules.len()];
    let mut stack = vec![grammar.root];
    let mut order = Vec::new();
    while let Some(id) = stack.pop() {
        let idx = id as usize;
        if idx >= grammar.rules.len() || seen[idx] {
            continue;
        }
        seen[idx] = true;
        order.push(id);
        for elem in &grammar.rules[idx] {
            if let Elem::RuleRef(v) = *elem {
                stack.push(v);
            }
        }
    }
    order.sort_unstable();
    order
}

/// Lower a parsed GBNF grammar to Lark text.
pub fn to_lark(grammar: &GbnfGrammar) -> Result<String, GbnfError> {
    let mut out = String::new();
    out.push_str(&format!("start: r{}\n", grammar.root));

    for id in reachable(grammar) {
        let rule = &grammar.rules[id as usize];
        let mut line = format!("r{id}:");
        let mut i = 0usize;
        let mut first = true;
        loop {
            if !first {
                line.push_str(" |");
            }
            first = false;
            let mut alt = String::new();
            emit_alternative(rule, &mut i, &mut alt);
            if !alt.is_empty() {
                line.push(' ');
                line.push_str(&alt);
            }
            match rule.get(i) {
                Some(Elem::Alt) => {
                    i += 1;
                }
                _ => break,
            }
        }
        line.push('\n');
        out.push_str(&line);
        if out.len() > MAX_LARK_BYTES {
            return Err(GbnfError::Unsupported(format!(
                "grammar expands past mlxcel's {MAX_LARK_BYTES}-byte compiled limit; reduce the \
                 repetition counts or rule complexity"
            )));
        }
    }
    Ok(out)
}
