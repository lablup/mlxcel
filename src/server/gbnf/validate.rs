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

//! Post-parse GBNF validation, mirroring the checks b10621 runs between
//! `llama_grammar_parser::parse` and `llama_grammar_init_impl`: undefined
//! rules, the `root` symbol, rule-reference bounds and left recursion.
//!
//! Reference:
//! <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp>
//!
//! Used by: server::gbnf

use super::parser::{Elem, GbnfGrammar, MAX_GBNF_BYTES, Parser};
use super::{GbnfError, GbnfVocab};

/// `llama_grammar_detect_left_recursion`.
fn detect_left_recursion(
    rules: &[Vec<Elem>],
    rule_index: usize,
    visited: &mut [bool],
    in_progress: &mut [bool],
    may_be_empty: &mut [bool],
) -> bool {
    if in_progress[rule_index] {
        return true;
    }
    in_progress[rule_index] = true;

    let rule = &rules[rule_index];

    let mut at_rule_start = true;
    for elem in rule {
        if elem.is_end_of_sequence() {
            if at_rule_start {
                may_be_empty[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    let mut recurse_into_nonterminal = true;
    for elem in rule {
        match *elem {
            Elem::RuleRef(v) if recurse_into_nonterminal => {
                let target = v as usize;
                if detect_left_recursion(rules, target, visited, in_progress, may_be_empty) {
                    return true;
                }
                if !may_be_empty[target] {
                    recurse_into_nonterminal = false;
                }
            }
            e if e.is_end_of_sequence() => recurse_into_nonterminal = true,
            _ => recurse_into_nonterminal = false,
        }
    }

    in_progress[rule_index] = false;
    visited[rule_index] = true;
    false
}

/// Parse and validate GBNF source exactly as b10621 does before it builds a
/// grammar: syntax, undefined rules, the `root` symbol, rule-reference bounds
/// and left recursion, in that order.
pub fn parse_gbnf(src: &str, vocab: Option<&dyn GbnfVocab>) -> Result<GbnfGrammar, GbnfError> {
    if src.len() > MAX_GBNF_BYTES {
        return Err(GbnfError::Unsupported(format!(
            "grammar is {} bytes, which exceeds mlxcel's {MAX_GBNF_BYTES}-byte limit",
            src.len()
        )));
    }

    let bytes = src.as_bytes();
    let mut parser = Parser {
        src: bytes,
        names: Vec::new(),
        rules: Vec::new(),
        vocab,
    };

    let mut pos = parser.parse_space(0, true);
    while parser.at(pos) != 0 {
        pos = parser.parse_rule(pos)?;
    }

    // Post-parse validation, matching b10621's own two checks.
    for rule in &parser.rules {
        if rule.is_empty() {
            return Err(GbnfError::Parse("Undefined rule".to_string()));
        }
    }
    for rule in &parser.rules {
        for elem in rule {
            if let Elem::RuleRef(v) = *elem {
                let idx = v as usize;
                if idx >= parser.rules.len() || parser.rules[idx].is_empty() {
                    let name = parser
                        .names
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| v.to_string());
                    return Err(GbnfError::Parse(format!(
                        "Undefined rule identifier '{name}'"
                    )));
                }
            }
        }
    }

    let Some(root) = parser.names.iter().position(|n| n == "root") else {
        return Err(GbnfError::Parse(
            "grammar does not contain a 'root' symbol".to_string(),
        ));
    };
    if root >= parser.rules.len() || parser.rules[root].is_empty() {
        return Err(GbnfError::Parse(
            "grammar does not contain a 'root' symbol".to_string(),
        ));
    }

    let n_rules = parser.rules.len();
    for (i, rule) in parser.rules.iter().enumerate() {
        for elem in rule {
            if let Elem::RuleRef(v) = *elem
                && (v as usize >= n_rules || parser.rules[v as usize].is_empty())
            {
                return Err(GbnfError::Parse(format!(
                    "invalid grammar: rule {i} references undefined rule {v}"
                )));
            }
        }
    }

    let mut visited = vec![false; n_rules];
    let mut in_progress = vec![false; n_rules];
    let mut may_be_empty = vec![false; n_rules];
    for i in 0..n_rules {
        if visited[i] {
            continue;
        }
        if detect_left_recursion(
            &parser.rules,
            i,
            &mut visited,
            &mut in_progress,
            &mut may_be_empty,
        ) {
            return Err(GbnfError::Unsupported(format!(
                "unsupported grammar, left recursion detected for nonterminal at index {i}"
            )));
        }
    }

    Ok(GbnfGrammar {
        rules: parser.rules,
        root: root as u32,
    })
}
