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

//! Fill-in-the-middle (FIM) vocabulary discovery for `POST /infill` (#1442).
//!
//! `llama-server` refuses `/infill` outright on a model whose vocabulary has no
//! prefix, suffix and middle token, and names the missing ones in the error. It
//! finds them from GGUF metadata when the converter wrote it and otherwise by
//! scanning the vocabulary for a fixed list of spellings. mlxcel loads MLX
//! SafeTensors and has no GGUF metadata, so only the scan applies here, over
//! exactly the spellings upstream scans for.
//!
//! Upstream reference: the FIM-token block of
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/src/llama-vocab.cpp>

/// Prefix-token spellings, in upstream's order.
///
/// The list and its order were read out of the pinned `libllama` binary rather
/// than transcribed, because two of them are easy to get wrong from memory:
/// DeepSeek-Coder's markers use the FULLWIDTH vertical line U+FF5C, not ASCII
/// `|`, and CodeLlama's SentencePiece spellings carry the U+2581 lower-one-
/// eighth block. A mis-transcribed spelling is not a compile error; it is a
/// server that silently refuses `/infill` on a model that supports it.
const FIM_PRE: [&str; 8] = [
    "<|fim_prefix|>",                     // Qwen, Qwen2.5-Coder
    "<fim-prefix>",                       // StarCoder2
    "<fim_prefix>",                       // Granite, StarCoder
    "<\u{FF5C}fim\u{2581}begin\u{FF5C}>", // DeepSeek-Coder
    "<PRE>",                              // CodeLlama
    "\u{2581}<PRE>",                      // CodeLlama, SentencePiece spelling
    "<|code_prefix|>",
    "<|prefix|>",
];

/// Suffix-token spellings, in upstream's order.
const FIM_SUF: [&str; 8] = [
    "<|fim_suffix|>",
    "<fim-suffix>",
    "<fim_suffix>",
    "<\u{FF5C}fim\u{2581}hole\u{FF5C}>",
    "<SUF>",
    "\u{2581}<SUF>",
    "<|code_suffix|>",
    "<|suffix|>",
];

/// Middle-token spellings, in upstream's order.
const FIM_MID: [&str; 8] = [
    "<|fim_middle|>",
    "<fim-middle>",
    "<fim_middle>",
    "<\u{FF5C}fim\u{2581}end\u{FF5C}>",
    "<MID>",
    "\u{2581}<MID>",
    "<|code_middle|>",
    "<|middle|>",
];

/// Repository-name marker spellings. Optional: upstream emits the header only
/// when the vocabulary has one.
const FIM_REP: [&str; 5] = [
    "<|fim_repo|>",
    "<|repo_name|>",
    "<fim-repo>",
    "<REPO>",
    "<reponame>",
];

/// File-separator spellings. Optional, like [`FIM_REP`]. b10621 recognizes
/// exactly one.
///
/// b10621 also resolves a FIM PAD token (`<|fim_pad|>`, `<fim-pad>`,
/// `<fim_pad>`, `<PAD>`, `[PAD]`). It is deliberately absent here: nothing in
/// `format_infill` writes it, so discovering it would record a capability
/// `/infill` never uses.
const FIM_SEP: [&str; 1] = ["<|file_sep|>"];

/// One discovered FIM token: the spelling that matched and the id it holds.
///
/// The spelling is what the infill prompt is built from, because mlxcel's
/// generation entry point takes a prompt string and re-tokenizes it; the id is
/// carried so a caller can assert the round trip rather than assume it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FimToken {
    pub spelling: &'static str,
    pub id: u32,
}

/// The FIM tokens a loaded vocabulary declares.
///
/// `pre`, `suf` and `mid` are the three `/infill` requires; `rep` and `sep`
/// shape the optional extra-context header and are absent on most models.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FimTokens {
    pub pre: Option<FimToken>,
    pub suf: Option<FimToken>,
    pub mid: Option<FimToken>,
    pub rep: Option<FimToken>,
    pub sep: Option<FimToken>,
}

/// The three required tokens, resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FimTriple {
    pub pre: FimToken,
    pub suf: FimToken,
    pub mid: FimToken,
}

impl FimTokens {
    /// Scan a vocabulary lookup for every known spelling.
    pub fn discover(lookup: impl Fn(&str) -> Option<u32>) -> Self {
        let first = |spellings: &[&'static str]| {
            spellings
                .iter()
                .find_map(|spelling| lookup(spelling).map(|id| FimToken { spelling, id }))
        };
        Self {
            pre: first(&FIM_PRE),
            suf: first(&FIM_SUF),
            mid: first(&FIM_MID),
            rep: first(&FIM_REP),
            sep: first(&FIM_SEP),
        }
    }

    /// The three required tokens, or the upstream diagnostic naming the missing
    /// ones.
    ///
    /// The message reproduces b10621's own wording, including its trailing
    /// space after each clause, so a client that matches on the string keeps
    /// working: `Infill is not supported by this model: prefix token is
    /// missing. `.
    pub fn require_triple(&self) -> Result<FimTriple, String> {
        let mut missing = String::new();
        if self.pre.is_none() {
            missing.push_str("prefix token is missing. ");
        }
        if self.suf.is_none() {
            missing.push_str("suffix token is missing. ");
        }
        if self.mid.is_none() {
            missing.push_str("middle token is missing. ");
        }
        if !missing.is_empty() {
            return Err(format!("Infill is not supported by this model: {missing}"));
        }
        Ok(FimTriple {
            pre: self.pre.clone().expect("checked above"),
            suf: self.suf.clone().expect("checked above"),
            mid: self.mid.clone().expect("checked above"),
        })
    }

    /// Whether this vocabulary can serve `/infill` at all.
    pub fn supports_infill(&self) -> bool {
        self.pre.is_some() && self.suf.is_some() && self.mid.is_some()
    }
}

#[cfg(test)]
#[path = "fim_tests.rs"]
mod fim_tests;
