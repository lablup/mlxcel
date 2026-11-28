// Copyright 2025-2026 Lablup Inc.
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

//! Thinking-marker inference against mlx-lm#1974 (#3209 item a): the Mistral
//! `[THINK]` / `[/THINK]` pair from the vocabulary, and the chat template's
//! `<think>` text as the last resort.

use super::MlxcelTokenizer;
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE};

/// A tokenizer whose vocabulary is exactly `tokens`, registered as added
/// tokens (`special` controls how they are flagged).
fn with_added(tokens: &[&str], special: bool) -> MlxcelTokenizer {
    let mut hf = Tokenizer::new(BPE::default());
    let added: Vec<AddedToken> = tokens
        .iter()
        .map(|s| AddedToken::from(*s, special))
        .collect();
    hf.add_tokens(added).expect("added tokens register");
    MlxcelTokenizer::HuggingFace(hf)
}

#[test]
fn infers_mistral_think_pair_from_vocab() {
    // Magistral / Ministral reasoning: `[THINK]` and `[/THINK]` are added
    // tokens, so each marker is one id.
    let tok = with_added(&["[THINK]", "[/THINK]", "[TOOL_CALLS]"], true);
    let markers = tok.infer_thinking_markers();
    assert!(markers.has_thinking());
    assert_eq!(markers.think_start.as_deref(), Some("[THINK]"));
    assert_eq!(markers.think_end.as_deref(), Some("[/THINK]"));
    assert_eq!(markers.think_start_tokens.as_ref().map(Vec::len), Some(1));
    assert_eq!(markers.think_end_tokens.as_ref().map(Vec::len), Some(1));

    // The template form reports the vocabulary pair, not a template fallback.
    let with_template =
        tok.infer_thinking_markers_with_template(Some("{{ '<think>' }}{{ '[THINK]' }}"));
    assert_eq!(with_template, markers);

    // First hit wins: a vocabulary with both `<think>` and `[THINK]` keeps
    // `<think>`, as upstream's `THINK_TOKENS` order does.
    let both = with_added(&["<think>", "</think>", "[THINK]", "[/THINK]"], true);
    assert_eq!(
        both.infer_thinking_markers().think_start.as_deref(),
        Some("<think>")
    );
}

#[test]
fn falls_back_to_template_think_text() {
    // No think tokens in the vocabulary: `<think>` encodes to several pieces.
    let tok = with_added(&["<", "/", "think", ">", "hi"], false);
    assert!(!tok.infer_thinking_markers().has_thinking());
    assert!(
        !tok.infer_thinking_markers_with_template(None)
            .has_thinking()
    );
    // A template that never writes `<think>` gives no fallback either.
    assert!(
        !tok.infer_thinking_markers_with_template(Some("{{ messages[0].content }}"))
            .has_thinking()
    );

    let markers = tok.infer_thinking_markers_with_template(Some("{{ '<think>' }}"));
    assert!(markers.has_thinking());
    assert_eq!(markers.think_start.as_deref(), Some("<think>"));
    assert_eq!(markers.think_end.as_deref(), Some("</think>"));
    let start = markers.think_start_tokens.expect("start ids");
    let end = markers.think_end_tokens.expect("end ids");
    assert!(start.len() > 1, "multi-token open: {start:?}");
    assert!(end.len() > 1, "multi-token close: {end:?}");
    assert_eq!(start, tok.encode("<think>", false).unwrap());
    assert_eq!(end, tok.encode("</think>", false).unwrap());
}
