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

//! Prompt preparation shared by the decode benchmarks and the parity harness.
//!
//! Both decode paths must see the same token ids, so the prompt is rendered
//! and tokenized once here and the resulting ids are handed to the CLI
//! generator and to the server engine alike. Two prompt shapes exist:
//!
//! - a chat prompt, rendered through the checkpoint's chat template the way
//!   `mlxcel generate` renders a single user turn, plus the history-only
//!   render the server's prompt cache uses to find the history boundary
//!   (issue #1143);
//! - a synthesized prompt of an exact token length, built from a fixed corpus
//!   so `--prompt-tokens N` is byte-identical across runs and between
//!   `mlxcel-bench-decode` and `mlxcel-bench-engine` (epic #623 #624).

use std::path::Path;

use anyhow::Result;

use crate::server::chat_template::{ChatMessage, ChatTemplateProcessor};
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::tokenizer::MlxcelTokenizer;

/// Fixed corpus paragraph repeated to synthesize long prompts. Kept constant
/// so the `--prompt-tokens N` prompt is byte-identical across benchmark runs
/// and models (only the tokenizer differs). Neutral prose with punctuation and
/// varied vocabulary so the token stream resembles real text rather than a
/// single repeated token.
pub const LONG_PROMPT_CORPUS: &str = concat!(
    "The measurement of large language model inference performance depends on ",
    "both prefill and decode throughput. During prefill the entire prompt is ",
    "processed in a single forward pass, so its cost grows with the prompt ",
    "length and exercises the matrix-multiply kernels at large batch widths. ",
    "During decode each new token is generated one step at a time, which ",
    "stresses memory bandwidth and kernel launch overhead instead. A benchmark ",
    "that only uses short prompts cannot separate these two regimes, because a ",
    "few dozen prompt tokens are dominated by fixed launch costs. To study ",
    "prefill behaviour honestly we therefore need prompts that are hundreds or ",
    "thousands of tokens long, repeated deterministically so that every run ",
    "observes the same input and the numbers stay comparable over time.\n\n",
);

/// Build a deterministic prompt of exactly `target_len` tokens by repeating
/// [`LONG_PROMPT_CORPUS`], tokenizing once with the model's tokenizer, and
/// truncating. Returns fewer than `target_len` tokens only if `target_len` is
/// `0`.
pub fn synthesize_prompt_tokens(
    tokenizer: &MlxcelTokenizer,
    target_len: usize,
) -> Result<Vec<i32>> {
    if target_len == 0 {
        return Ok(Vec::new());
    }
    // Estimate tokens per corpus copy (without special tokens) to size the
    // repeated string, then over-provision so the final tokenization always
    // yields at least `target_len` tokens before truncation.
    let per_copy = tokenizer
        .encode(LONG_PROMPT_CORPUS, false)
        .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?
        .len()
        .max(1);
    let repeats = target_len / per_copy + 4;
    let corpus = LONG_PROMPT_CORPUS.repeat(repeats);
    let mut ids: Vec<i32> = tokenizer
        .encode(&corpus, true)
        .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?
        .into_iter()
        .map(|id| id as i32)
        .collect();
    ids.truncate(target_len);
    Ok(ids)
}

/// Clamp a requested synthesized prompt length to the model's context window,
/// leaving `reserve_for_generation` positions for the generated tokens.
#[must_use]
pub fn cap_prompt_len(
    target_len: usize,
    max_context: Option<usize>,
    reserve_for_generation: usize,
) -> usize {
    match max_context {
        Some(ctx) => target_len.min(ctx.saturating_sub(reserve_for_generation).max(1)),
        None => target_len,
    }
}

/// Tokenize a rendered prompt with the convention `mlxcel generate` and the
/// server share: a template that already renders a BOS token does not get a
/// second one from the tokenizer.
pub fn tokenize_rendered(tokenizer: &MlxcelTokenizer, prompt: &str) -> Result<Vec<i32>> {
    let add_special = !tokenizer.prompt_carries_bos(prompt);
    let ids = tokenizer
        .encode(prompt, add_special)
        .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?;
    Ok(ids.into_iter().map(|id| id as i32).collect())
}

/// A single-user-turn chat prompt rendered for both decode paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPrompt {
    /// Prompt token ids, identical for every path.
    pub tokens: Vec<i32>,
    /// The same conversation rendered with `add_generation_prompt = false`
    /// and tokenized with the same convention, i.e. the history prefix the
    /// server's prompt cache splits a cold prefill at. `None` when the model
    /// has no chat template or chat templating was disabled.
    pub history_tokens: Option<Vec<i32>>,
    /// Whether a chat template was applied.
    pub templated: bool,
}

/// Render `user_prompt` as one user turn through the checkpoint's chat
/// template (or verbatim when `raw` is set or no template exists), the way
/// `mlxcel generate` and `mlxcel-bench-decode` render a text prompt.
pub fn render_chat_prompt(
    model_path: &Path,
    tokenizer: &MlxcelTokenizer,
    user_prompt: &str,
    raw: bool,
) -> Result<ChatPrompt> {
    let processor = if raw {
        None
    } else {
        ChatTemplateProcessor::from_model_path(model_path)
            .ok()
            .flatten()
    };
    let Some(processor) = processor else {
        return Ok(ChatPrompt {
            tokens: tokenize_rendered(tokenizer, user_prompt)?,
            history_tokens: None,
            templated: false,
        });
    };
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    }];
    let rendered = processor
        .apply(&messages, None)
        .unwrap_or_else(|_| user_prompt.to_string());
    let history = processor
        .apply_history_with_kwargs(&messages, None, &ChatTemplateKwargs::new())
        .ok();
    let history_tokens = match history {
        Some(text) => Some(tokenize_rendered(tokenizer, &text)?),
        None => None,
    };
    Ok(ChatPrompt {
        tokens: tokenize_rendered(tokenizer, &rendered)?,
        history_tokens,
        templated: true,
    })
}

#[cfg(test)]
mod tests {
    use super::cap_prompt_len;

    #[test]
    fn cap_prompt_len_reserves_generation_room() {
        assert_eq!(cap_prompt_len(8192, Some(40960), 128), 8192);
        assert_eq!(cap_prompt_len(8192, Some(4096), 128), 3968);
        assert_eq!(cap_prompt_len(8192, None, 128), 8192);
        // A context smaller than the reservation still leaves one token.
        assert_eq!(cap_prompt_len(8192, Some(64), 128), 1);
    }
}
