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

//! `--grammar`, `--grammar-file`, `--json-schema` and `--json-schema-file`
//! resolution (#1485).
//!
//! b10621 gives all four handlers the same destination
//! (`params.sampling.grammar`), so passing two of them is not an error there:
//! the one that appears **last on the command line** wins. clap has no such
//! notion, and picking an arbitrary winner would silently constrain generation
//! with a grammar the operator did not intend, so the argv order is recovered
//! here with the same value-skipping walk
//! [`crate::cli::llama_short_flags`] uses.
//!
//! Reference:
//! <https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp>
//!
//! Used by: bin::mlx_server, main, server::cli_input

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use llguidance::api::{GrammarInit, ParserLimits, TopLevelGrammar};
use llguidance::earley::ValidationResult;

use crate::server::gbnf::{GbnfError, parse_gbnf};
use crate::server::grammar::GrammarSpec;

/// The winning grammar-source flag together with its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarSourceArg {
    /// `--grammar GRAMMAR`
    Gbnf(String),
    /// `--grammar-file FNAME`
    GbnfFile(PathBuf),
    /// `-j` / `--json-schema SCHEMA`
    Schema(String),
    /// `-jf` / `--json-schema-file FILE`
    SchemaFile(PathBuf),
}

/// The four flags in the fixed order used when argv order is unavailable.
const FLAG_SPELLINGS: &[(&str, usize)] = &[
    ("--grammar", 0),
    ("--grammar-file", 1),
    ("-j", 2),
    ("--json-schema", 2),
    ("-jf", 3),
    ("--json-schema-file", 3),
];

/// Index of the grammar flag a token names, if any. `--flag=value` counts.
fn flag_slot(token: &str) -> Option<usize> {
    let name = token.split_once('=').map_or(token, |(name, _)| name);
    FLAG_SPELLINGS
        .iter()
        .find(|(spelling, _)| *spelling == name)
        .map(|(_, slot)| *slot)
}

/// The slot of the last grammar flag on the command line.
///
/// `start` is the index of the first argument that could be an option: 1 for a
/// plain binary, 2 for `mlxcel serve`. Values of value-taking options are
/// skipped so a value that happens to spell a grammar flag is not mistaken for
/// one, and everything after `--` is ignored.
fn last_flag_slot(cmd: &mut clap::Command, argv: &[OsString], start: usize) -> Option<usize> {
    let value_taking = crate::cli::llama_short_flags::value_taking_tokens(cmd);
    let mut last = None;
    let mut i = start;
    while i < argv.len() {
        let Some(token) = argv[i].to_str() else {
            i += 1;
            continue;
        };
        if token == "--" {
            break;
        }
        if let Some(slot) = flag_slot(token) {
            last = Some(slot);
        }
        let consumes = !token.contains('=') && value_taking.contains(token);
        i += if consumes { 2 } else { 1 };
    }
    last
}

/// Pick the grammar source the operator actually asked for.
///
/// Returns `None` when no flag was supplied. When argv order cannot be read
/// (the flag came from somewhere other than the command line), the fixed order
/// `--grammar`, `--grammar-file`, `--json-schema`, `--json-schema-file` decides
/// and the later flag wins, which is what upstream does for the common case of
/// a wrapper script appending a flag.
pub fn resolve_grammar_source(
    cmd: &mut clap::Command,
    argv: &[OsString],
    start: usize,
    grammar: Option<String>,
    grammar_file: Option<PathBuf>,
    json_schema: Option<String>,
    json_schema_file: Option<PathBuf>,
) -> Option<GrammarSourceArg> {
    let mut slots: [Option<GrammarSourceArg>; 4] = [
        grammar.map(GrammarSourceArg::Gbnf),
        grammar_file.map(GrammarSourceArg::GbnfFile),
        json_schema.map(GrammarSourceArg::Schema),
        json_schema_file.map(GrammarSourceArg::SchemaFile),
    ];
    if slots.iter().all(Option::is_none) {
        return None;
    }
    if let Some(slot) = last_flag_slot(cmd, argv, start)
        && let Some(chosen) = slots[slot].take()
    {
        return Some(chosen);
    }
    slots.into_iter().flatten().next_back()
}

impl GrammarSourceArg {
    /// Read the flag's value, applying b10621's own split between what fails
    /// at argument-parse time and what fails at the first request.
    ///
    /// Upstream converts a schema **inside the argument handler**
    /// (`json_schema_to_grammar(json::parse(value))`), so a malformed or
    /// unconvertible `--json-schema` aborts the process before the server
    /// listens. It does not parse GBNF there at all: `--grammar` is stored
    /// verbatim and only parsed when a request builds its sampler, which
    /// answers `failed to parse grammar`. A missing `--grammar-file` still
    /// aborts, because `read_file` throws in the handler.
    ///
    /// mlxcel reproduces that split rather than being stricter: a bad GBNF
    /// grammar logs a loud startup error and then fails every constrained
    /// request with b10621's own diagnostic, instead of refusing to start a
    /// server that upstream would have started.
    pub fn load(&self) -> Result<GrammarSpec> {
        match self {
            Self::Gbnf(text) => {
                warn_on_unparseable_gbnf(text, "--grammar");
                Ok(GrammarSpec::from_gbnf(text.clone()))
            }
            Self::GbnfFile(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("failed to read grammar file {}", path.display()))?;
                warn_on_unparseable_gbnf(&text, "--grammar-file");
                Ok(GrammarSpec::from_gbnf(text))
            }
            Self::Schema(text) => Ok(GrammarSpec::from_schema(load_schema(
                text,
                "--json-schema",
            )?)),
            Self::SchemaFile(path) => {
                let text = std::fs::read_to_string(path).with_context(|| {
                    format!("failed to read JSON schema file {}", path.display())
                })?;
                Ok(GrammarSpec::from_schema(load_schema(
                    &text,
                    "--json-schema-file",
                )?))
            }
        }
    }
}

/// Parse and compile-check a schema, so an unusable one aborts startup exactly
/// as upstream's in-handler conversion does.
///
/// The compile check runs with no tokenizer, which `llguidance` supports: the
/// model is not loaded when the flags are resolved, and schema validity does
/// not depend on the vocabulary.
fn load_schema(text: &str, flag: &str) -> Result<serde_json::Value> {
    let value: serde_json::Value =
        serde_json::from_str(text).with_context(|| format!("{flag} value is not valid JSON"))?;
    let init = GrammarInit::Serialized(TopLevelGrammar::from_json_schema(value.clone()));
    match init.validate(None, ParserLimits::default()) {
        ValidationResult::Valid | ValidationResult::Warnings(_) => Ok(value),
        ValidationResult::Error(e) => {
            let first = e
                .lines()
                .next()
                .unwrap_or("schema is not usable")
                .to_string();
            bail!("{flag} schema cannot be compiled into a grammar: {first}")
        }
    }
}

/// Report an unparseable GBNF grammar at startup without refusing to start.
fn warn_on_unparseable_gbnf(text: &str, flag: &str) {
    match parse_gbnf(text, None) {
        Ok(_) => {}
        // `<token-text>` needs the vocabulary, which arrives with the model.
        Err(GbnfError::Token(_)) => {}
        Err(err) => tracing::error!(
            "{flag}: failed to parse grammar: {err}. b10621 also defers GBNF parsing to the first \
             request, so the server starts and every constrained request will be refused with \
             this diagnostic"
        ),
    }
}

#[cfg(test)]
#[path = "grammar_args_tests.rs"]
mod tests;
