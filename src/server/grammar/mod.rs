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

//! b10621 grammar request resolution.
//!
//! Turns the native `json_schema` / `grammar` / `grammar_lazy` /
//! `grammar_triggers` / `preserved_tokens` fields, and the `--grammar`,
//! `--grammar-file`, `--json-schema` and `--json-schema-file` flags, into a
//! [`GrammarSpec`] that [`crate::server::structured`] compiles into a
//! constrained-decoding matcher.
//!
//! Reference:
//! <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-schema.cpp>
//! and
//! <https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp>
//!
//! Used by: server::routes::native_completion, server::routes::infill,
//! server::startup, server::structured

mod lazy;

#[cfg(test)]
mod tests;

pub use lazy::{LazyGate, LazyOutcome};

use serde_json::Value;

use crate::tokenizer::MlxcelTokenizer;

/// b10621's `common_grammar_trigger_type` discriminants, in declaration order.
const TRIGGER_TYPE_TOKEN: i64 = 0;
const TRIGGER_TYPE_WORD: i64 = 1;
const TRIGGER_TYPE_PATTERN: i64 = 2;
const TRIGGER_TYPE_PATTERN_FULL: i64 = 3;

/// One resolved lazy-grammar trigger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarTrigger {
    /// A single token id. b10621 promotes a `WORD` trigger to this form when
    /// the word tokenizes to exactly one token.
    Token(u32),
    /// A literal string, matched as an escaped regex against the buffered
    /// output.
    Word(String),
    /// A regex, matched anywhere in the buffered output.
    Pattern(String),
    /// A regex anchored to the whole buffered output.
    PatternFull(String),
}

/// Where the grammar text came from. b10621 tracks the same distinction as
/// `common_grammar_type` and uses it for one behaviour: only a non-`USER`
/// grammar is prefilled with the generation prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarOrigin {
    /// Supplied by the caller as GBNF (`grammar`, `--grammar`).
    User,
    /// Derived from a JSON schema (`json_schema`, `--json-schema`).
    OutputFormat,
}

/// A fully resolved constrained-decoding request.
#[derive(Debug, Clone)]
pub struct GrammarSpec {
    /// GBNF text, when the caller supplied a grammar rather than a schema.
    pub gbnf: Option<String>,
    /// JSON-Schema document, when the caller supplied a schema.
    pub schema: Option<Value>,
    pub origin: GrammarOrigin,
    pub lazy: bool,
    pub triggers: Vec<GrammarTrigger>,
    pub preserved: Vec<u32>,
}

impl GrammarSpec {
    /// A GBNF-sourced spec with no lazy machinery, used by the server flags.
    pub fn from_gbnf(gbnf: String) -> Self {
        Self {
            gbnf: Some(gbnf),
            schema: None,
            origin: GrammarOrigin::User,
            lazy: false,
            triggers: Vec::new(),
            preserved: Vec::new(),
        }
    }

    /// A schema-sourced spec with no lazy machinery, used by the server flags.
    pub fn from_schema(schema: Value) -> Self {
        Self {
            gbnf: None,
            schema: Some(schema),
            origin: GrammarOrigin::OutputFormat,
            lazy: false,
            triggers: Vec::new(),
            preserved: Vec::new(),
        }
    }
}

/// A refusal carrying b10621's own wording where one exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarRequestError(pub String);

impl std::fmt::Display for GrammarRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Tokenize with `add_special = false, parse_special = true`, which is what
/// b10621 uses for every grammar-adjacent tokenization (`preserved_tokens`,
/// trigger words and `<token>` terminals).
fn tokenize_special(tokenizer: &MlxcelTokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode_with_special(text, false, true)
        .unwrap_or_default()
}

/// Resolve `preserved_tokens`.
///
/// b10621 keeps only the entries that tokenize to exactly one token and drops
/// the rest without a diagnostic.
fn resolve_preserved(
    tokenizer: &MlxcelTokenizer,
    value: &Value,
) -> Result<Vec<u32>, GrammarRequestError> {
    let Some(items) = value.as_array() else {
        return Err(GrammarRequestError(
            "preserved_tokens must be an array".to_string(),
        ));
    };
    let mut out = Vec::new();
    for item in items {
        let Some(text) = item.as_str() else {
            return Err(GrammarRequestError(
                "preserved_tokens entries must be strings".to_string(),
            ));
        };
        let ids = tokenize_special(tokenizer, text);
        if ids.len() == 1 && !out.contains(&ids[0]) {
            out.push(ids[0]);
        }
    }
    Ok(out)
}

/// Resolve `grammar_triggers`, applying b10621's word-to-token promotion.
fn resolve_triggers(
    tokenizer: &MlxcelTokenizer,
    value: &Value,
    preserved: &[u32],
) -> Result<Vec<GrammarTrigger>, GrammarRequestError> {
    let Some(items) = value.as_array() else {
        return Err(GrammarRequestError(
            "grammar_triggers must be an array".to_string(),
        ));
    };
    let mut out = Vec::new();
    for item in items {
        let obj = item.as_object().ok_or_else(|| {
            GrammarRequestError("grammar_triggers entries must be objects".to_string())
        })?;
        let kind = obj
            .get("type")
            .and_then(Value::as_i64)
            .ok_or_else(|| GrammarRequestError("grammar trigger requires a type".to_string()))?;
        let text = obj
            .get("value")
            .and_then(Value::as_str)
            .ok_or_else(|| GrammarRequestError("grammar trigger requires a value".to_string()))?
            .to_string();
        match kind {
            TRIGGER_TYPE_TOKEN => {
                let token = obj.get("token").and_then(Value::as_i64).ok_or_else(|| {
                    GrammarRequestError("grammar trigger requires a token".to_string())
                })?;
                let token = u32::try_from(token).map_err(|_| {
                    GrammarRequestError(format!("grammar trigger token {token} is out of range"))
                })?;
                out.push(GrammarTrigger::Token(token));
            }
            TRIGGER_TYPE_WORD => {
                let ids = tokenize_special(tokenizer, &text);
                if ids.len() == 1 {
                    if !preserved.contains(&ids[0]) {
                        return Err(GrammarRequestError(format!(
                            "Grammar trigger word should be marked as preserved token: {text}"
                        )));
                    }
                    out.push(GrammarTrigger::Token(ids[0]));
                } else {
                    out.push(GrammarTrigger::Word(text));
                }
            }
            TRIGGER_TYPE_PATTERN => out.push(GrammarTrigger::Pattern(text)),
            TRIGGER_TYPE_PATTERN_FULL => out.push(GrammarTrigger::PatternFull(text)),
            other => {
                return Err(GrammarRequestError(format!(
                    "unknown grammar trigger type: {other}"
                )));
            }
        }
    }
    Ok(out)
}

/// Resolve the native `/completion` and `/infill` grammar surfaces.
///
/// The precedence rule is b10621's, which contradicts its own field
/// description: the schema path is taken only when `json_schema` is present
/// **and `grammar` is absent**, so `{"grammar": "", "json_schema": {...}}`
/// leaves the request with no grammar at all. Presence, not emptiness, decides.
///
/// A non-string `grammar` is not an error upstream: `json_value` logs and
/// falls back to the empty default, which is the same as sending no grammar.
pub fn resolve_native_grammar(
    tokenizer: &MlxcelTokenizer,
    json_schema: Option<&Value>,
    grammar: Option<&Value>,
    grammar_lazy: Option<bool>,
    grammar_triggers: Option<&Value>,
    preserved_tokens: Option<&Value>,
    default: Option<&GrammarSpec>,
) -> Result<Option<GrammarSpec>, GrammarRequestError> {
    // b10621 registers `preserved_tokens` before `grammar_triggers`, and the
    // trigger handler reads the preserved set, so the order matters.
    let preserved = match preserved_tokens {
        Some(value) => resolve_preserved(tokenizer, value)?,
        None => Vec::new(),
    };

    let lazy = grammar_lazy.unwrap_or(false);

    // The "no triggers set for lazy grammar" check lives INSIDE upstream's
    // `grammar_triggers` handler, so it only fires when the key is present.
    let triggers = match grammar_triggers {
        Some(value) => {
            let triggers = resolve_triggers(tokenizer, value, &preserved)?;
            if lazy && triggers.is_empty() {
                return Err(GrammarRequestError(
                    "Error: no triggers set for lazy grammar!".to_string(),
                ));
            }
            triggers
        }
        None => Vec::new(),
    };

    let (gbnf, schema, origin) = if json_schema.is_some() && grammar.is_none() {
        let schema = json_schema
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        (None, Some(schema), GrammarOrigin::OutputFormat)
    } else {
        match grammar.and_then(Value::as_str) {
            Some(text) if !text.is_empty() => (Some(text.to_string()), None, GrammarOrigin::User),
            _ => {
                if grammar.is_some_and(|g| !g.is_string()) {
                    tracing::warn!(
                        "native request field 'grammar' is not a string; ignoring it as b10621 does"
                    );
                }
                // An empty or absent grammar does NOT clear a server default:
                // upstream only assigns `params.sampling.grammar` when the
                // string is non-empty, so the `--grammar` / `--json-schema`
                // value the server started with survives.
                let Some(default) = default else {
                    return Ok(None);
                };
                (default.gbnf.clone(), default.schema.clone(), default.origin)
            }
        }
    };

    Ok(Some(GrammarSpec {
        gbnf,
        schema,
        origin,
        lazy,
        triggers,
        preserved,
    }))
}
