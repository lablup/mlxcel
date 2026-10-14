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

//! The one chat-template front every path renders through (#2176).
//!
//! `mlxcel-server`, `mlxcel run`, `mlxcel generate`, the parity harness and
//! the decode benchmarks must render the same request to the same tokens
//! (ADR 0007's invariant). What they share is here: the checkpoint's
//! template with the `enable_thinking` default the server applies, so no
//! front can drift on that rule again (the harness and `mlxcel-bench-decode`
//! once skipped it, which made their chat arms diverge from `run -p` on Qwen3
//! at token 0).

use std::path::Path;

use super::chat_template::ChatTemplateProcessor;
use crate::tokenizer::MlxcelTokenizer;

/// Default `enable_thinking=true` when the tokenizer recognizes a think
/// marker pair, unless the template asks for thinking off.
///
/// Aligns with upstream `TokenizerWrapper.apply_chat_template`'s
/// `enable_thinking=self.has_thinking` (mlx-lm PR #1114): a template that
/// branches on `enable_thinking is defined and enable_thinking is false`
/// (Qwen3) would otherwise render an empty `<think>\n\n</think>` block, after
/// which a model not trained on it emits an immediate end-of-text. Per-request
/// kwargs and the server's `--chat-template-kwargs` still win on conflict.
///
/// Exception (issue #686): the Gemma 4 thinking-channel template's
/// thinking-off branch already renders a closed priming scaffold that matches
/// transformers' no-`enable_thinking` render; forcing thinking on yields a
/// bare `<|turn>model\n` that greedy-collapses to `<pad>`, so a template that
/// `wants_thinking_default_off` keeps its default.
///
/// Returns whether the default was flipped on.
pub fn apply_thinking_default(
    template: &mut ChatTemplateProcessor,
    tokenizer: &MlxcelTokenizer,
) -> bool {
    let markers = tokenizer.infer_thinking_markers();
    if markers.has_thinking() && !template.wants_thinking_default_off() {
        template.set_default_enable_thinking(true);
        return true;
    }
    false
}

/// The checkpoint's own chat template with [`apply_thinking_default`]
/// applied, or `None` when the checkpoint has no template (a raw-completion
/// checkpoint renders the prompt verbatim).
///
/// Used by: `mlxcel generate`, the parity harness, `mlxcel-bench-decode`.
/// The server's `load_chat_front` resolves a `--chat-template` override first
/// and then applies the same default.
#[must_use]
pub fn load_model_chat_template(
    model_path: &Path,
    tokenizer: &MlxcelTokenizer,
) -> Option<ChatTemplateProcessor> {
    let mut template = ChatTemplateProcessor::from_model_path(model_path)
        .ok()
        .flatten()?;
    apply_thinking_default(&mut template, tokenizer);
    Some(template)
}
