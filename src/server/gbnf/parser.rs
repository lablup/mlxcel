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

//! GBNF parser, mirroring `llama_grammar_parser` at b10621.
//!
//! Reference:
//! <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp>
//!
//! Used by: server::gbnf

use super::{GbnfError, GbnfVocab};

/// b10621's `MAX_REPETITION_THRESHOLD`.
pub(super) const MAX_REPETITION_THRESHOLD: u64 = 2000;

/// Upper bound on the GBNF source mlxcel will parse.
///
/// b10621 has no bound; this one exists because the grammar text arrives over
/// HTTP. It is well above every grammar in llama.cpp's own `grammars/`
/// directory and above what `json_schema_to_grammar` emits for a schema that
/// passes [`crate::server::structured::MAX_SCHEMA_BYTES`].
pub const MAX_GBNF_BYTES: usize = 256 * 1024;

/// One element of a rule, mirroring `llama_gretype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Elem {
    End,
    Alt,
    RuleRef(u32),
    Char(u32),
    CharNot(u32),
    CharRngUpper(u32),
    CharAlt(u32),
    CharAny,
    Token(u32),
    TokenNot(u32),
}

impl Elem {
    /// `llama_grammar_is_end_of_sequence`.
    pub(super) fn is_end_of_sequence(self) -> bool {
        matches!(self, Self::End | Self::Alt)
    }
}

/// A parsed and validated GBNF grammar.
#[derive(Debug, Clone)]
pub struct GbnfGrammar {
    pub(super) rules: Vec<Vec<Elem>>,
    pub(super) root: u32,
}

pub(super) fn is_digit_char(c: u8) -> bool {
    c.is_ascii_digit()
}

/// b10621's `is_word_char`: ASCII letters, digits and `-`. Note that `_` is
/// **not** a legal GBNF identifier character.
pub(super) fn is_word_char(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'-' || is_digit_char(c)
}

/// Render a bounded, lossy view of the source from `pos` for an error message.
///
/// b10621 appends the entire remaining source to every diagnostic. Repeating
/// that would echo an arbitrarily large request body back to the client, so the
/// tail is capped; the message prefix, which is what callers match on, is
/// identical.
fn tail(src: &[u8], pos: usize) -> String {
    const MAX: usize = 48;
    let rest = &src[pos.min(src.len())..];
    let cut = rest.len().min(MAX);
    let shown = String::from_utf8_lossy(&rest[..cut]).into_owned();
    if rest.len() > cut {
        format!("{shown}...")
    } else {
        shown
    }
}

pub(super) struct Parser<'a> {
    pub(super) src: &'a [u8],
    /// Insertion-ordered symbol table; the index in this vector is the rule id.
    pub(super) names: Vec<String>,
    pub(super) rules: Vec<Vec<Elem>>,
    pub(super) vocab: Option<&'a dyn GbnfVocab>,
}

impl<'a> Parser<'a> {
    pub(super) fn at(&self, pos: usize) -> u8 {
        self.src.get(pos).copied().unwrap_or(0)
    }

    pub(super) fn err(&self, msg: &str, pos: usize) -> GbnfError {
        GbnfError::Parse(format!("{msg} {}", tail(self.src, pos)))
    }

    pub(super) fn get_symbol_id(&mut self, name: &str) -> u32 {
        if let Some(idx) = self.names.iter().position(|n| n == name) {
            return idx as u32;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        id
    }

    /// `generate_symbol_id`: a synthesized name that can never collide with a
    /// user-written one, because `_` is not a legal GBNF identifier character.
    pub(super) fn generate_symbol_id(&mut self, base: &str) -> u32 {
        let id = self.names.len() as u32;
        self.names.push(format!("{base}_{id}"));
        id
    }

    pub(super) fn add_rule(&mut self, id: u32, rule: Vec<Elem>) {
        let idx = id as usize;
        if self.rules.len() <= idx {
            self.rules.resize(idx + 1, Vec::new());
        }
        self.rules[idx] = rule;
    }

    pub(super) fn parse_space(&self, mut pos: usize, newline_ok: bool) -> usize {
        loop {
            let c = self.at(pos);
            let is_space =
                c == b' ' || c == b'\t' || c == b'#' || (newline_ok && (c == b'\r' || c == b'\n'));
            if !is_space {
                return pos;
            }
            if c == b'#' {
                while self.at(pos) != 0 && self.at(pos) != b'\r' && self.at(pos) != b'\n' {
                    pos += 1;
                }
            } else {
                pos += 1;
            }
        }
    }

    pub(super) fn parse_name(&self, src: usize) -> Result<usize, GbnfError> {
        let mut pos = src;
        while is_word_char(self.at(pos)) {
            pos += 1;
        }
        if pos == src {
            return Err(self.err("expecting name at", src));
        }
        Ok(pos)
    }

    pub(super) fn parse_int(&self, src: usize) -> Result<usize, GbnfError> {
        let mut pos = src;
        while is_digit_char(self.at(pos)) {
            pos += 1;
        }
        if pos == src {
            return Err(self.err("expecting integer at", src));
        }
        Ok(pos)
    }

    pub(super) fn parse_hex(&self, src: usize, size: usize) -> Result<(u32, usize), GbnfError> {
        let mut value: u32 = 0;
        let mut pos = src;
        let end = src + size;
        while pos < end && self.at(pos) != 0 {
            let c = self.at(pos);
            let digit = match c {
                b'0'..=b'9' => u32::from(c - b'0'),
                b'a'..=b'f' => u32::from(c - b'a') + 10,
                b'A'..=b'F' => u32::from(c - b'A') + 10,
                _ => break,
            };
            value = (value << 4) + digit;
            pos += 1;
        }
        if pos != end {
            return Err(self.err(&format!("expecting {size} hex chars at"), src));
        }
        Ok((value, pos))
    }

    /// `decode_utf8` from `llama-grammar.cpp`: a length-table decoder that does
    /// not validate continuation bytes.
    pub(super) fn decode_utf8(&self, src: usize) -> (u32, usize) {
        const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
        let first = self.at(src);
        let len = LOOKUP[(first >> 4) as usize];
        let mask: u8 = if len >= 8 {
            0
        } else {
            (1u16 << (8 - len)) as u8
        }
        .wrapping_sub(1);
        let mut value = u32::from(first & mask);
        let mut pos = src + 1;
        while pos < src + len && self.at(pos) != 0 {
            value = (value << 6) + u32::from(self.at(pos) & 0x3F);
            pos += 1;
        }
        (value, pos)
    }

    pub(super) fn parse_char(&self, src: usize) -> Result<(u32, usize), GbnfError> {
        if self.at(src) == b'\\' {
            return match self.at(src + 1) {
                b'x' => self.parse_hex(src + 2, 2),
                b'u' => self.parse_hex(src + 2, 4),
                b'U' => self.parse_hex(src + 2, 8),
                b't' => Ok((u32::from(b'\t'), src + 2)),
                b'r' => Ok((u32::from(b'\r'), src + 2)),
                b'n' => Ok((u32::from(b'\n'), src + 2)),
                c @ (b'\\' | b'"' | b'[' | b']' | b'-') => Ok((u32::from(c), src + 2)),
                _ => Err(self.err("unknown escape at", src)),
            };
        }
        if self.at(src) != 0 {
            return Ok(self.decode_utf8(src));
        }
        Err(GbnfError::Parse("unexpected end of input".to_string()))
    }

    /// `parse_token`: `<[id]>` or `<text>`. The caller has already consumed a
    /// leading `!` for the negated form.
    pub(super) fn parse_token(&self, src: usize) -> Result<(u32, usize), GbnfError> {
        let mut pos = src;
        if self.at(pos) != b'<' {
            return Err(self.err("expecting '<' at", pos));
        }
        pos += 1;

        if self.at(pos) == b'[' {
            pos += 1;
            let int_end = self.parse_int(pos)?;
            let digits = String::from_utf8_lossy(&self.src[pos..int_end]).into_owned();
            let token_id: u32 = digits
                .parse()
                .map_err(|_| self.err("expecting integer at", pos))?;
            pos = int_end;
            if self.at(pos) != b']' {
                return Err(self.err("expecting ']' at", pos));
            }
            pos += 1;
            if self.at(pos) != b'>' {
                return Err(self.err("expecting '>' at", pos));
            }
            pos += 1;
            return Ok((token_id, pos));
        }

        let Some(vocab) = self.vocab else {
            return Err(GbnfError::Token(format!(
                "no vocab to parse token at {}",
                tail(self.src, src)
            )));
        };

        while self.at(pos) != 0 && self.at(pos) != b'>' {
            pos += 1;
        }
        if self.at(pos) != b'>' {
            return Err(self.err("expecting '>' at", pos));
        }
        pos += 1;

        // b10621 tokenizes the whole terminal INCLUDING the angle brackets.
        let text = String::from_utf8_lossy(&self.src[src..pos]).into_owned();
        match vocab.tokenize_exact_one(&text) {
            Some(id) => Ok((id, pos)),
            None => Err(GbnfError::Token(format!("invalid token '{text}'"))),
        }
    }

    pub(super) fn parse_alternates(
        &mut self,
        src: usize,
        rule_name: &str,
        rule_id: u32,
        is_nested: bool,
    ) -> Result<usize, GbnfError> {
        let mut rule: Vec<Elem> = Vec::new();
        let mut pos = self.parse_sequence(src, rule_name, &mut rule, is_nested)?;
        while self.at(pos) == b'|' {
            rule.push(Elem::Alt);
            pos = self.parse_space(pos + 1, true);
            pos = self.parse_sequence(pos, rule_name, &mut rule, is_nested)?;
        }
        rule.push(Elem::End);
        self.add_rule(rule_id, rule);
        Ok(pos)
    }

    pub(super) fn parse_rule(&mut self, src: usize) -> Result<usize, GbnfError> {
        let name_end = self.parse_name(src)?;
        let mut pos = self.parse_space(name_end, false);
        let name = String::from_utf8_lossy(&self.src[src..name_end]).into_owned();
        let rule_id = self.get_symbol_id(&name);

        if !(self.at(pos) == b':' && self.at(pos + 1) == b':' && self.at(pos + 2) == b'=') {
            return Err(self.err("expecting ::= at", pos));
        }
        pos = self.parse_space(pos + 3, true);
        pos = self.parse_alternates(pos, &name, rule_id, false)?;

        if self.at(pos) == b'\r' {
            pos += if self.at(pos + 1) == b'\n' { 2 } else { 1 };
        } else if self.at(pos) == b'\n' {
            pos += 1;
        } else if self.at(pos) != 0 {
            return Err(self.err("expecting newline or end at", pos));
        }
        Ok(self.parse_space(pos, true))
    }
}
