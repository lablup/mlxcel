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

//! GBNF sequence parsing and repetition desugaring.
//!
//! Split out of [`super::parser`] for file size; this is
//! `llama_grammar_parser::parse_sequence` together with its
//! `handle_repetitions` lambda at b10621, kept in one file because the
//! multiplicative repetition budget (`n_prev_rules`) is threaded between them.
//!
//! Reference:
//! <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp>
//!
//! Used by: server::gbnf::parser

use super::GbnfError;
use super::parser::{Elem, MAX_REPETITION_THRESHOLD, Parser, is_digit_char, is_word_char};

impl Parser<'_> {
    /// `handle_repetitions`, with `max_times = None` standing in for upstream's
    /// `UINT64_MAX` no-maximum sentinel.
    #[allow(clippy::too_many_arguments)]
    fn handle_repetitions(
        &mut self,
        rule: &mut Vec<Elem>,
        rule_name: &str,
        last_sym_start: usize,
        n_prev_rules: &mut u64,
        min_times: u64,
        max_times: Option<u64>,
        pos: usize,
    ) -> Result<(), GbnfError> {
        let no_max = max_times.is_none();
        if last_sym_start == rule.len() {
            return Err(self.err("expecting preceding item to */+/?/{ at", pos));
        }

        let prev_rule: Vec<Elem> = rule[last_sym_start..].to_vec();

        let mut total_rules: u64 = 1;
        if let Some(max) = max_times {
            if max > 0 {
                total_rules = max;
            } else if min_times > 0 {
                total_rules = min_times;
            }
        } else if min_times > 0 {
            total_rules = min_times;
        }

        if n_prev_rules.saturating_mul(total_rules) >= MAX_REPETITION_THRESHOLD {
            return Err(GbnfError::Parse(
                "number of rules that are going to be repeated multiplied by the new repetition \
                 exceeds sane defaults, please reduce the number of repetitions or rule complexity"
                    .to_string(),
            ));
        }

        if min_times == 0 {
            rule.truncate(last_sym_start);
        } else {
            for _ in 1..min_times {
                rule.extend_from_slice(&prev_rule);
            }
        }

        let mut last_rec_rule_id: u32 = 0;
        let n_opt = match max_times {
            None => 1,
            Some(max) => max - min_times,
        };

        for i in 0..n_opt {
            let mut rec_rule = prev_rule.clone();
            let rec_rule_id = self.generate_symbol_id(rule_name);
            if i > 0 || no_max {
                rec_rule.push(Elem::RuleRef(if no_max {
                    rec_rule_id
                } else {
                    last_rec_rule_id
                }));
            }
            rec_rule.push(Elem::Alt);
            rec_rule.push(Elem::End);
            self.add_rule(rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if n_opt > 0 {
            rule.push(Elem::RuleRef(last_rec_rule_id));
        }
        *n_prev_rules = n_prev_rules.saturating_mul(total_rules).max(1);
        Ok(())
    }

    pub(super) fn parse_sequence(
        &mut self,
        src: usize,
        rule_name: &str,
        rule: &mut Vec<Elem>,
        is_nested: bool,
    ) -> Result<usize, GbnfError> {
        let mut last_sym_start = rule.len();
        let mut pos = src;
        let mut n_prev_rules: u64 = 1;

        while self.at(pos) != 0 {
            match self.at(pos) {
                b'"' => {
                    pos += 1;
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    while self.at(pos) != b'"' {
                        if self.at(pos) == 0 {
                            return Err(GbnfError::Parse("unexpected end of input".to_string()));
                        }
                        let (value, next) = self.parse_char(pos)?;
                        pos = next;
                        rule.push(Elem::Char(value));
                    }
                    pos = self.parse_space(pos + 1, is_nested);
                }
                b'[' => {
                    pos += 1;
                    let mut start_negated = false;
                    if self.at(pos) == b'^' {
                        pos += 1;
                        start_negated = true;
                    }
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    while self.at(pos) != b']' {
                        if self.at(pos) == 0 {
                            return Err(GbnfError::Parse("unexpected end of input".to_string()));
                        }
                        let (value, next) = self.parse_char(pos)?;
                        pos = next;
                        let elem = if last_sym_start < rule.len() {
                            Elem::CharAlt(value)
                        } else if start_negated {
                            Elem::CharNot(value)
                        } else {
                            Elem::Char(value)
                        };
                        rule.push(elem);
                        if self.at(pos) == b'-' && self.at(pos + 1) != b']' {
                            if self.at(pos + 1) == 0 {
                                return Err(GbnfError::Parse(
                                    "unexpected end of input".to_string(),
                                ));
                            }
                            let (end_value, next) = self.parse_char(pos + 1)?;
                            pos = next;
                            rule.push(Elem::CharRngUpper(end_value));
                        }
                    }
                    pos = self.parse_space(pos + 1, is_nested);
                }
                b'<' | b'!' => {
                    let negated = self.at(pos) == b'!';
                    if negated {
                        pos += 1;
                    }
                    let (token, token_end) = self.parse_token(pos)?;
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    rule.push(if negated {
                        Elem::TokenNot(token)
                    } else {
                        Elem::Token(token)
                    });
                    pos = self.parse_space(token_end, is_nested);
                }
                c if is_word_char(c) => {
                    let name_end = self.parse_name(pos)?;
                    let name = String::from_utf8_lossy(&self.src[pos..name_end]).into_owned();
                    let ref_rule_id = self.get_symbol_id(&name);
                    pos = self.parse_space(name_end, is_nested);
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    rule.push(Elem::RuleRef(ref_rule_id));
                }
                b'(' => {
                    pos = self.parse_space(pos + 1, true);
                    let n_rules_before = self.names.len();
                    let sub_rule_id = self.generate_symbol_id(rule_name);
                    pos = self.parse_alternates(pos, rule_name, sub_rule_id, true)?;
                    n_prev_rules = (self.names.len().saturating_sub(n_rules_before) as u64).max(1);
                    last_sym_start = rule.len();
                    rule.push(Elem::RuleRef(sub_rule_id));
                    if self.at(pos) != b')' {
                        return Err(self.err("expecting ')' at", pos));
                    }
                    pos = self.parse_space(pos + 1, is_nested);
                }
                b'.' => {
                    last_sym_start = rule.len();
                    n_prev_rules = 1;
                    rule.push(Elem::CharAny);
                    pos = self.parse_space(pos + 1, is_nested);
                }
                b'*' => {
                    pos = self.parse_space(pos + 1, is_nested);
                    self.handle_repetitions(
                        rule,
                        rule_name,
                        last_sym_start,
                        &mut n_prev_rules,
                        0,
                        None,
                        pos,
                    )?;
                }
                b'+' => {
                    pos = self.parse_space(pos + 1, is_nested);
                    self.handle_repetitions(
                        rule,
                        rule_name,
                        last_sym_start,
                        &mut n_prev_rules,
                        1,
                        None,
                        pos,
                    )?;
                }
                b'?' => {
                    pos = self.parse_space(pos + 1, is_nested);
                    self.handle_repetitions(
                        rule,
                        rule_name,
                        last_sym_start,
                        &mut n_prev_rules,
                        0,
                        Some(1),
                        pos,
                    )?;
                }
                b'{' => {
                    pos = self.parse_space(pos + 1, is_nested);
                    if !is_digit_char(self.at(pos)) {
                        return Err(self.err("expecting an int at", pos));
                    }
                    let int_end = self.parse_int(pos)?;
                    let min_times = parse_u64(&self.src[pos..int_end]);
                    pos = self.parse_space(int_end, is_nested);

                    let mut max_times: Option<u64> = None;
                    if self.at(pos) == b'}' {
                        max_times = Some(min_times);
                        pos = self.parse_space(pos + 1, is_nested);
                    } else if self.at(pos) == b',' {
                        pos = self.parse_space(pos + 1, is_nested);
                        if is_digit_char(self.at(pos)) {
                            let int_end = self.parse_int(pos)?;
                            max_times = Some(parse_u64(&self.src[pos..int_end]));
                            pos = self.parse_space(int_end, is_nested);
                        }
                        if self.at(pos) != b'}' {
                            return Err(self.err("expecting '}' at", pos));
                        }
                        pos = self.parse_space(pos + 1, is_nested);
                    } else {
                        return Err(self.err("expecting ',' at", pos));
                    }
                    if min_times > MAX_REPETITION_THRESHOLD {
                        return Err(GbnfError::Parse(
                            "number of repetitions exceeds sane defaults, please reduce the \
                             number of repetitions"
                                .to_string(),
                        ));
                    }
                    // b10621 SILENTLY widens an oversized maximum to unbounded.
                    if max_times.is_some_and(|m| m > MAX_REPETITION_THRESHOLD) {
                        max_times = None;
                    }
                    self.handle_repetitions(
                        rule,
                        rule_name,
                        last_sym_start,
                        &mut n_prev_rules,
                        min_times,
                        max_times,
                        pos,
                    )?;
                }
                _ => break,
            }
        }
        Ok(pos)
    }
}

/// `std::stoull` on a digit run that `parse_int` already validated. Saturates
/// rather than overflowing; every caller compares against
/// `MAX_REPETITION_THRESHOLD` immediately afterwards.
fn parse_u64(digits: &[u8]) -> u64 {
    let mut value: u64 = 0;
    for &d in digits {
        value = value.saturating_mul(10).saturating_add(u64::from(d - b'0'));
    }
    value
}
