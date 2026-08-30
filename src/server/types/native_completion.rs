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

//! Response types for the native `llama-server` completion routes (#1441).
//!
//! These are the shapes `POST /completion` and `POST /completions` answer
//! with. They are NOT the OpenAI shapes: `POST /v1/completions` keeps
//! [`crate::server::types::CompletionResponse`]. b10621 routes the two
//! spellings to two different handlers
//! (<https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server.cpp>),
//! and mlxcel used to answer the OpenAI shape on all three.
//!
//! The field set and its order come from a capture of the pinned b10621
//! binary, not from a reading of the source: `index`, `content`, `tokens`,
//! `id_slot`, `stop`, `model`, `tokens_predicted`, `tokens_evaluated`,
//! `generation_settings`, `prompt`, `has_new_line`, `truncated`, `stop_type`,
//! `stopping_word`, `tokens_cached`, `timings`.

use serde::Serialize;

/// Why generation stopped, in b10621's vocabulary.
///
/// Upstream emits this as a bare string on the `stop_type` field. `none`
/// appears on a non-final streaming frame, where generation has not stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StopType {
    /// Generation is still running (streaming frames only).
    None,
    /// The `n_predict` budget was reached.
    Limit,
    /// A stop string matched; `stopping_word` carries it.
    Word,
    /// The model emitted an end-of-sequence token.
    Eos,
}

/// Timing block. `cache_n` leads, matching b10621's field order.
#[derive(Debug, Clone, Serialize)]
pub struct NativeTimings {
    /// Prompt tokens served from the KV prefix cache rather than re-prefilled.
    pub cache_n: usize,
    /// Prompt tokens the model actually processed, which upstream reports as
    /// the whole prompt minus [`Self::cache_n`], not the whole prompt.
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
}

impl NativeTimings {
    /// Build the block the way b10621 builds it.
    ///
    /// Two conventions here were read off the pinned binary rather than
    /// guessed, because both are observable and neither is what a naive
    /// reading suggests:
    ///
    /// - `prompt_n` is the part of the prompt the model actually processed,
    ///   so it is `tokens_evaluated - cache_n`, not `tokens_evaluated`. A
    ///   request whose prefix came entirely from the KV cache reports
    ///   `prompt_n: 1` against `tokens_evaluated: 8` with `cache_n: 7`.
    /// - the prompt rates divide by `prompt_n`, but the generation rates
    ///   divide by `predicted_n - 1`, because `predicted_ms` is measured from
    ///   the first token rather than from the request. Measured on the pinned
    ///   binary: `predicted_n: 4`, `predicted_ms: 15.132` reports
    ///   `predicted_per_token_ms: 5.044` (15.132 / 3) and
    ///   `predicted_per_second: 198.255` (1e3 / 15.132 * 3).
    ///
    /// A zero divisor yields a zero rate rather than an infinity or a NaN,
    /// neither of which is representable in JSON.
    pub fn new(
        cache_n: usize,
        tokens_evaluated: usize,
        prompt_ms: f64,
        predicted_n: usize,
        predicted_ms: f64,
    ) -> Self {
        let per_token = |ms: f64, n: usize| if n > 0 { ms / n as f64 } else { 0.0 };
        let per_second = |ms: f64, n: usize| {
            if ms > 0.0 && n > 0 {
                n as f64 / (ms / 1000.0)
            } else {
                0.0
            }
        };
        let prompt_n = tokens_evaluated.saturating_sub(cache_n);
        // The generation rates are measured over the intervals BETWEEN emitted
        // tokens, so a single-token generation has no interval to report.
        let predicted_intervals = predicted_n.saturating_sub(1);
        Self {
            cache_n,
            prompt_n,
            prompt_ms,
            prompt_per_token_ms: per_token(prompt_ms, prompt_n),
            prompt_per_second: per_second(prompt_ms, prompt_n),
            predicted_n,
            predicted_ms,
            predicted_per_token_ms: per_token(predicted_ms, predicted_intervals),
            predicted_per_second: per_second(predicted_ms, predicted_intervals),
        }
    }
}

/// Apply b10621's `response_fields` projection to a finished native response
/// (issue #1441).
///
/// Upstream walks each requested path through the response object splitting on
/// `/`, and stores the value it lands on under the **full path string**, so
/// `generation_settings/n_predict` becomes a root key literally named
/// `"generation_settings/n_predict"` rather than `n_predict`. The schema's own
/// description says the field is "unnested", which the binary does not do; the
/// behavior here is the binary's, captured against it directly.
///
/// A path that does not resolve is omitted without an error, a path that walks
/// into a non-object is omitted, an empty `paths` list returns the whole
/// object, and the result preserves the order the request listed the paths in
/// (upstream builds it in an insertion-ordered JSON object, and mlxcel's
/// `serde_json` is built with `preserve_order`).
pub fn select_response_fields(value: serde_json::Value, paths: &[String]) -> serde_json::Value {
    if paths.is_empty() {
        return value;
    }
    let mut selected = serde_json::Map::new();
    for path in paths {
        let mut current = Some(&value);
        for key in path.split('/') {
            current = current
                .and_then(|node| node.as_object())
                .and_then(|m| m.get(key));
        }
        if let Some(found) = current {
            selected.insert(path.clone(), found.clone());
        }
    }
    serde_json::Value::Object(selected)
}

/// Prompt-processing progress, emitted on streaming frames when the request
/// asked for `return_progress`.
#[derive(Debug, Clone, Serialize)]
pub struct PromptProgress {
    pub total: usize,
    pub cache: usize,
    pub processed: usize,
    pub time_ms: u64,
}

/// The final (or only) native completion result.
///
/// Field order matches the capture. `tokens` is always present and is empty
/// unless the request set `return_tokens`, which is also what upstream does.
#[derive(Debug, Clone, Serialize)]
pub struct NativeCompletionResponse {
    /// Index of this completion. Always `0`: mlxcel serves one completion per
    /// request, and rejects `n_cmpl > 1` rather than silently ignoring it.
    pub index: usize,
    pub content: String,
    pub tokens: Vec<i32>,
    /// Scheduler slot that served the request, or `-1` when the request was
    /// not bound to a numbered slot.
    pub id_slot: i64,
    pub stop: bool,
    pub model: String,
    pub tokens_predicted: usize,
    pub tokens_evaluated: usize,
    /// The resolved sampling and generation settings actually used.
    pub generation_settings: serde_json::Value,
    pub prompt: String,
    /// Whether the emitted content contains a newline. b10621 reports it so a
    /// FIM client can tell whether the completion crossed a line boundary.
    pub has_new_line: bool,
    /// Whether the prompt was truncated to fit the context window.
    pub truncated: bool,
    pub stop_type: StopType,
    /// The stop string that matched, or the empty string.
    pub stopping_word: String,
    /// Prompt tokens present in the KV cache after this request.
    pub tokens_cached: usize,
    pub timings: NativeTimings,
    /// Per-token probability report (#1485), present only when the request
    /// set `n_probs` (alias `logprobs`) above zero: one entry per generated
    /// token, each carrying the token's own probability and its top-N
    /// alternatives, in b10621's `completion_probabilities` shape
    /// (`logprob`/`top_logprobs` keys pre-sampling, `prob`/`top_probs` under
    /// `post_sampling_probs`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_probabilities: Option<serde_json::Value>,
}

/// A non-final streaming frame.
///
/// Upstream emits a much smaller object per token and only attaches the full
/// metadata to the final frame. `timings` appears here only under
/// `timings_per_token`, and `prompt_progress` only under `return_progress`.
#[derive(Debug, Clone, Serialize)]
pub struct NativeCompletionChunk {
    pub index: usize,
    pub content: String,
    pub tokens: Vec<i32>,
    pub stop: bool,
    pub id_slot: i64,
    pub tokens_predicted: usize,
    pub tokens_evaluated: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<NativeTimings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_progress: Option<PromptProgress>,
    /// This frame's token probability report (#1485), present only under
    /// `n_probs`: a one-entry array in the same shape as the final object's
    /// `completion_probabilities`, which is how upstream streams it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_probabilities: Option<serde_json::Value>,
}

#[cfg(test)]
#[path = "native_completion_tests.rs"]
mod native_completion_tests;
