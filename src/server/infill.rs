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

//! Fill-in-the-middle prompt assembly for `POST /infill` (issue #1442).
//!
//! Request validation and prompt ordering live here rather than in the route so
//! the ordering contract, which is the whole point of `--spm-infill`, has unit
//! tests that do not need an HTTP stack or a loaded model.
//!
//! # Ordering
//!
//! b10621 builds two blocks, `[FIM_PRE] prefix prompt` and `[FIM_SUF] suffix`,
//! emits the optional extra-context header before them, and closes with
//! `[FIM_MID]`. `--spm-infill` swaps which block comes first:
//!
//! | Mode | Order |
//! |---|---|
//! | default (PSM) | `extra` `FIM_PRE` prefix prompt `FIM_SUF` suffix `FIM_MID` |
//! | `--spm-infill` (SPM) | `extra` `FIM_SUF` suffix `FIM_PRE` prefix prompt `FIM_MID` |
//!
//! The flag exists because a model's FIM training data fixes one of the two,
//! and prompting a model in the order it was not trained on produces fluent
//! but wrong completions rather than an error, so this is not a preference.
//!
//! # What is assembled, and why it is a string
//!
//! Upstream assembles token ids. mlxcel's generation entry point takes a prompt
//! string and tokenizes it, and the FIM markers are added-vocabulary entries,
//! so writing their spellings into the string produces the same ids provided
//! special-token parsing is on, which is the default everywhere the prompt is
//! encoded. [`format_infill_prompt`] therefore emits spellings, and
//! `infill_prompt_tokenizes_to_the_expected_ids` in the tests holds the
//! equivalence rather than assuming it.
//!
//! Upstream reference: `format_infill` in
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/utils.hpp>

use crate::tokenizer::{FimTokens, FimTriple};

/// The separator upstream writes between extra-context chunks on a model whose
/// vocabulary has no dedicated file-separator token.
///
/// Upstream spells it as a byte array "to avoid confusing the AI", meaning to
/// keep the literal out of its own source text; the bytes are this string.
const CHUNK_SEPARATOR: &str = "\n\n--- snippet ---\n\n";

/// The repository name upstream hard-codes into the `FIM_REP` header. It is a
/// placeholder there too, with a `TODO` asking for it to become an input.
const REPO_PLACEHOLDER: &str = "myproject\n";

/// The filename upstream hard-codes into the trailing `FIM_SEP` marker, for the
/// same reason.
const CURRENT_FILE_PLACEHOLDER: &str = "filename\n";

/// Default filename for an extra-context chunk that did not name one.
const DEFAULT_CHUNK_FILENAME: &str = "tmp";

/// One `input_extra` entry: a snippet of surrounding repository context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfillChunk {
    pub text: String,
    pub filename: String,
}

/// The FIM-specific half of an `/infill` request, after validation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InfillInputs {
    pub input_prefix: String,
    pub input_suffix: String,
    /// The `/completion` `prompt` field, which `/infill` appends to the prefix
    /// rather than using on its own.
    pub prompt: String,
    pub input_extra: Vec<InfillChunk>,
}

/// Validate an `/infill` body's FIM fields.
///
/// Error strings reproduce b10621's own wording so a client matching on them
/// keeps working. `prompt` is optional and defaults to the empty string;
/// `input_prefix` and `input_suffix` are required; `input_extra` is optional
/// and, when present, must be an array of objects with a string `text` and an
/// optional string `filename`.
pub fn parse_infill_inputs(body: &serde_json::Value) -> Result<InfillInputs, String> {
    let prompt = match body.get("prompt") {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(_) => return Err("\"prompt\" must be a string".to_string()),
    };

    let input_prefix = match body.get("input_prefix") {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(_) => return Err("\"input_prefix\" must be a string".to_string()),
        None => return Err("\"input_prefix\" is required".to_string()),
    };

    let input_suffix = match body.get("input_suffix") {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(_) => return Err("\"input_suffix\" must be a string".to_string()),
        None => return Err("\"input_suffix\" is required".to_string()),
    };

    let mut input_extra = Vec::new();
    match body.get("input_extra") {
        None | Some(serde_json::Value::Null) => {}
        Some(serde_json::Value::Array(items)) => {
            for chunk in items {
                let text = chunk
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        "\"input_extra\" must be an array of {\"filename\": string, \"text\": \
                         string}"
                            .to_string()
                    })?;
                let filename = match chunk.get("filename") {
                    None | Some(serde_json::Value::Null) => DEFAULT_CHUNK_FILENAME.to_string(),
                    Some(serde_json::Value::String(name)) => name.clone(),
                    Some(_) => {
                        return Err(
                            "\"input_extra\" must be an array of {\"filename\": string, \
                                    \"text\": string}"
                                .to_string(),
                        );
                    }
                };
                input_extra.push(InfillChunk {
                    text: text.to_string(),
                    filename,
                });
            }
        }
        Some(_) => {
            return Err(
                "\"input_extra\" must be an array of {\"filename\": string, \"text\": string}"
                    .to_string(),
            );
        }
    }

    Ok(InfillInputs {
        input_prefix,
        input_suffix,
        prompt,
        input_extra,
    })
}

/// Refuse an `/infill` body whose user-supplied text contains one of the
/// model's own FIM marker spellings.
///
/// b10621 tokenizes `input_prefix`, `input_suffix` and each `input_extra` chunk
/// with special-token parsing **off**, so a marker written into the user's code
/// stays literal text there. mlxcel's generation entry point takes a prompt
/// string and re-tokenizes it with parsing on, and a string cannot express
/// "these characters are not that token". Carrying the text through anyway
/// would let a file the client did not write restructure the FIM prompt, and
/// the completion would come back fluent and wrong with nothing to notice.
///
/// Failing loudly is the lesser divergence: upstream serves the text, mlxcel
/// names the field and the marker and refuses. Recorded as a divergence on the
/// `POST /infill` manifest entry.
pub fn reject_marker_injection(tokens: &FimTokens, inputs: &InfillInputs) -> Result<(), String> {
    let markers: Vec<&str> = [
        tokens.pre.as_ref(),
        tokens.suf.as_ref(),
        tokens.mid.as_ref(),
        tokens.rep.as_ref(),
        tokens.sep.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(|token| token.spelling)
    .collect();

    let mut fields: Vec<(&str, &str)> = vec![
        ("input_prefix", inputs.input_prefix.as_str()),
        ("input_suffix", inputs.input_suffix.as_str()),
        ("prompt", inputs.prompt.as_str()),
    ];
    for chunk in &inputs.input_extra {
        // Both halves of a chunk reach the prompt: the filename follows the
        // file-separator marker and the text follows the filename, so a marker
        // in either one restructures the header exactly as one in the prefix
        // restructures the body.
        fields.push(("input_extra", chunk.filename.as_str()));
        fields.push(("input_extra", chunk.text.as_str()));
    }

    for (field, text) in fields {
        for marker in &markers {
            if text.contains(marker) {
                return Err(format!(
                    "\"{field}\" contains the fill-in-the-middle marker {marker}, which this \
                     server cannot carry as literal text: the infill prompt is assembled as a \
                     string and re-tokenized, so the marker would become that token and \
                     restructure the request. Remove it from the field, or send the text through \
                     POST /completion, which has no FIM structure to corrupt."
                ));
            }
        }
    }
    Ok(())
}

/// Assemble the FIM prompt. See the module docs for the ordering contract.
pub fn format_infill_prompt(
    tokens: &FimTokens,
    triple: &FimTriple,
    inputs: &InfillInputs,
    spm_infill: bool,
) -> String {
    let mut extra = String::new();
    if let Some(rep) = tokens.rep.as_ref() {
        extra.push_str(rep.spelling);
        extra.push_str(REPO_PLACEHOLDER);
    }
    for chunk in &inputs.input_extra {
        match tokens.sep.as_ref() {
            Some(sep) => {
                extra.push_str(sep.spelling);
                extra.push_str(&chunk.filename);
                extra.push('\n');
            }
            None => extra.push_str(CHUNK_SEPARATOR),
        }
        extra.push_str(&chunk.text);
    }
    if let Some(sep) = tokens.sep.as_ref() {
        extra.push_str(sep.spelling);
        extra.push_str(CURRENT_FILE_PLACEHOLDER);
    }

    let prefix_block = format!(
        "{}{}{}",
        triple.pre.spelling, inputs.input_prefix, inputs.prompt
    );
    let suffix_block = format!("{}{}", triple.suf.spelling, inputs.input_suffix);

    let (first, second) = if spm_infill {
        (suffix_block, prefix_block)
    } else {
        (prefix_block, suffix_block)
    };

    format!("{extra}{first}{second}{}", triple.mid.spelling)
}

#[cfg(test)]
#[path = "infill_tests.rs"]
mod infill_tests;
