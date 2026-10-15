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

//! Counting a prompt through the worker's media expansion (issue #1788).
//!
//! An image, clip or audio part is a single placeholder in the rendered
//! prompt and becomes its soft tokens only on the worker, inside the same
//! per-family call that computes the media embeddings. The prompt-inspection
//! routes therefore cannot count a media body by encoding the rendered text:
//! that answered 57 for a two-clip body `/v1/chat/completions` reports as 2,129
//! prompt tokens. They send the prepared request here instead, and the worker
//! runs exactly the preparation a generation gets and stops before prefill.

use super::*;

impl GenerationResult {
    /// The result a count-only request answers with: the expanded prompt
    /// length and nothing generated.
    pub(crate) fn prompt_count(prompt_tokens: usize) -> Self {
        Self {
            text: String::new(),
            prompt_tokens,
            completion_tokens: 0,
            generation_time_ms: 0,
            prompt_eval_ms: 0,
            generation_only_ms: 0,
            finish_reason: "stop".to_string(),
            stop_kind: StopKind::Eos,
            logprobs: None,
            cached_tokens: 0,
            generated_token_ids: Vec::new(),
            structured_output: None,
            speculative: None,
        }
    }
}

impl ModelProvider {
    /// Count the tokens a prepared request would be prefilled with, after the
    /// worker expands its media into soft tokens.
    ///
    /// `options` carries the request's per-request media settings (the Gemma 4
    /// image soft-token budget and any natively rendered prompt ids); this
    /// marks it count-only. The batch scheduler then tokenizes the prompt and
    /// prepares the media exactly as it would for a generation, which also
    /// fills the vision feature cache a following generation of the same body
    /// reads, and answers before any prefill. A worker that does not honor the
    /// flag generates the single token `max_tokens = 1` allows and reports the
    /// same `prompt_tokens`. No prompt-cache context is attached, so the count
    /// neither adopts nor donates a cached prefix.
    ///
    /// Blocking: call it from a blocking-pool thread.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn count_prompt_tokens_with_media(
        &self,
        prompt: String,
        mut options: ServerGenerateOptions,
        images: Vec<Vec<u8>>,
        audio: Vec<Vec<u8>>,
        videos: Vec<ResolvedVideo>,
        media: MediaRequestMetadata,
        live: &crate::server::LiveSettings,
    ) -> Result<usize> {
        options.count_prompt_only = true;
        options.max_tokens = 1;
        options.prompt_cache_ctx = None;
        self.generate_with_media_and_videos_declared_live(
            prompt, options, images, audio, videos, media, live,
        )
        .map(|result| result.prompt_tokens)
    }
}
