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

//! GBNF (llama.cpp grammar) front end.
//!
//! b10621 constrains generation with GBNF text supplied through `--grammar`,
//! `--grammar-file` and the native `grammar` request field, and converts
//! `json_schema` to GBNF before handing it to the sampler. mlxcel already runs
//! per-step constrained decoding through `llguidance`
//! ([`crate::server::structured`]), so this module is the missing front end: it
//! parses GBNF with b10621's own grammar and lowers it to the Lark dialect that
//! `llguidance` compiles.
//!
//! The reference implementation is
//! <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp>, read
//! at the pinned b10621 commit `c1d0e7a004015f23bc0233470b747b596f29b264`. The
//! parser here mirrors its element model (`llama_gretype`) rather than building a
//! prettier AST, because the observable behaviour that has to match lives in the
//! desugaring: repetition rewriting, the multiplicative repetition budget, the
//! `{0}` erasure and the `x{0,5000}` silent widening all fall out of reproducing
//! that model directly.
//!
//! Used by: server::structured, server::routes::native_completion,
//! server::routes::chat, server::startup

mod lark;
mod parser;
mod sequence;
mod validate;

#[cfg(test)]
mod tests;

pub use lark::to_lark;
pub use parser::{GbnfGrammar, MAX_GBNF_BYTES};
pub use validate::parse_gbnf;

use std::fmt;

/// A GBNF surface that mlxcel refused, carrying b10621's own diagnostic text
/// where one exists.
///
/// b10621's `llama_grammar_parser::parse` never propagates its exception: it
/// prints to stderr, clears every rule and returns `false`, which the caller
/// reports as `failed to parse grammar`. mlxcel keeps the specific message
/// instead, because a server that silently degrades a constrained request to an
/// unconstrained one is exactly the failure mode the b10621 compatibility work
/// exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GbnfError {
    /// A syntax or validation error, worded as b10621 words it.
    Parse(String),
    /// A `<token>` terminal that needs a vocabulary the caller did not supply,
    /// or that does not tokenize to exactly one token.
    Token(String),
    /// The grammar parsed but mlxcel will not run it.
    Unsupported(String),
}

impl fmt::Display for GbnfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(m) | Self::Token(m) | Self::Unsupported(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for GbnfError {}

/// Vocabulary access needed to resolve `<token-text>` terminals.
///
/// b10621 tokenizes the whole terminal including its angle brackets with
/// `add_special = false, parse_special = true` and requires exactly one token
/// (`llama-grammar.cpp` `parse_token`). Implementors must reproduce that: return
/// `None` whenever the text does not tokenize to a single id.
pub trait GbnfVocab {
    /// Tokenize `text` (angle brackets included) and return the id when it is
    /// exactly one token.
    fn tokenize_exact_one(&self, text: &str) -> Option<u32>;
}

/// Compile GBNF source into the `llguidance` Lark dialect.
///
/// `vocab` may be `None`, in which case `<[id]>` terminals still resolve and
/// `<text>` terminals raise b10621's `no vocab to parse token at ...`.
pub fn compile_gbnf(src: &str, vocab: Option<&dyn GbnfVocab>) -> Result<String, GbnfError> {
    let grammar = parse_gbnf(src, vocab)?;
    to_lark(&grammar)
}
