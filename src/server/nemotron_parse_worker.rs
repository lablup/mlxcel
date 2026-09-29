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

//! Single-stream (batch-1) serving loop for Nemotron-Parse (issue #1369).
//!
//! Nemotron-Parse is encoder-decoder like Florence-2: one C-RADIO encoder
//! pass per page, then a greedy decode that cross-attends to it through the
//! model-owned seq2seq cache. It cannot join the batched/paged scheduler, so
//! the model worker thread branches into [`run_nemotron_parse_worker_loop`]
//! after loading the checkpoint and serves one request at a time off the
//! same channel, exactly as `florence2_worker.rs` does.
//!
//! Request shape: one user message with one `image_url` part (the page) and
//! an optional text part (the task prompt that seeds the decoder). The
//! built-in chat template renders only the text parts, so `prompt` here is
//! that task prompt; an empty one is replaced by
//! [`DEFAULT_TASK_PROMPT`].
//!
//! Security boundaries: image bytes decode through
//! [`decode_request_images`] (payload, dimension, and decode-allocation
//! caps); the task prompt is untrusted, so it is bounded in bytes and in
//! decoder-seed tokens and must be free of control characters before it is
//! tokenized into the seed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use crate::models::NemotronParseVlmModel;
use crate::models::nemotron_parse::DEFAULT_TASK_PROMPT;
use crate::server::ServerGenerateOptions;
use crate::server::model_provider::model_worker::decode_request_images;
use crate::server::model_provider::{
    GenerateEvent, GenerationResult, ModelRequest, StopKind, TokenMeta,
};

pub(crate) const NEMOTRON_PARSE_MEDIA_UNSUPPORTED_MSG: &str =
    "Nemotron-Parse is a document-parsing model; audio and video inputs are not supported";

pub(crate) const NEMOTRON_PARSE_IMAGE_REQUIRED_MSG: &str = "Nemotron-Parse requires exactly one page image per request: attach one image_url content \
     part and optionally a text part with the task prompt";

pub(crate) const NEMOTRON_PARSE_CANCELLED_BEFORE_START_MSG: &str =
    "Nemotron-Parse request cancelled before generation started";

/// Upper bound, in bytes, on the caller's task prompt. The trained prompts
/// are a handful of control tokens (well under 200 bytes).
pub(crate) const MAX_TASK_PROMPT_BYTES: usize = 1024;

/// Upper bound on the decoder seed the task prompt tokenizes into. The seed
/// is prefilled in one call whose causal mask is built on the host, and it
/// eats into the decoder's position budget, so it is kept small.
pub(crate) const MAX_SEED_TOKENS: usize = 256;

/// Map the caller's text onto the task prompt: empty or whitespace-only
/// text selects the default, anything else is validated.
pub(crate) fn resolve_task_prompt(text: &str) -> Result<&str, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(DEFAULT_TASK_PROMPT);
    }
    if trimmed.len() > MAX_TASK_PROMPT_BYTES {
        return Err(format!(
            "Nemotron-Parse task prompt is {} bytes; the server accepts at most \
             {MAX_TASK_PROMPT_BYTES} bytes",
            trimmed.len()
        ));
    }
    if trimmed.chars().any(char::is_control) {
        return Err("Nemotron-Parse task prompt contains control characters".to_string());
    }
    Ok(trimmed)
}

pub(crate) fn reject_media(audio_present: bool, video_present: bool) -> Option<&'static str> {
    (audio_present || video_present).then_some(NEMOTRON_PARSE_MEDIA_UNSUPPORTED_MSG)
}

pub(crate) fn reject_image_count(image_count: usize) -> Option<String> {
    (image_count != 1)
        .then(|| format!("{NEMOTRON_PARSE_IMAGE_REQUIRED_MSG} (got {image_count} images)"))
}

/// `"stop"` when the decode ended on `</s>` (or was cancelled), `"length"`
/// when it ran out of budget or decoder positions.
pub(crate) fn nemotron_parse_finish_reason(
    hit_eos: bool,
    generated: usize,
    max: usize,
) -> &'static str {
    if !hit_eos && generated >= max {
        "length"
    } else {
        "stop"
    }
}

/// Serve Nemotron-Parse requests one at a time until shutdown.
pub(crate) fn run_nemotron_parse_worker_loop(
    model: &NemotronParseVlmModel,
    request_rx: mpsc::Receiver<ModelRequest>,
) {
    tracing::info!(
        "Nemotron-Parse seq2seq worker ready (single-stream, batch-1; one C-RADIO encoder pass \
         per page, greedy cross-attention decode)"
    );
    for request in request_rx {
        match request {
            ModelRequest::PromptCacheWarmup { .. } => continue,
            ModelRequest::Shutdown => {
                tracing::info!("Nemotron-Parse seq2seq worker received shutdown signal");
                break;
            }
            ModelRequest::Generate {
                prompt,
                prompt_token_ids: _,
                options,
                runtime: _,
                images,
                audio,
                videos,
                media: _,
                queue_reservation,
                response_tx,
                cancelled,
            } => {
                drop(queue_reservation);
                handle_request(
                    model,
                    &prompt,
                    &options,
                    &images,
                    !audio.is_empty(),
                    !videos.is_empty(),
                    &response_tx,
                    &cancelled,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_request(
    model: &NemotronParseVlmModel,
    prompt: &str,
    options: &ServerGenerateOptions,
    images: &[Vec<u8>],
    audio_present: bool,
    video_present: bool,
    response_tx: &mpsc::Sender<GenerateEvent>,
    cancelled: &std::sync::Arc<AtomicBool>,
) {
    let fail = |msg: String| {
        let _ = response_tx.send(GenerateEvent::Error(msg));
    };
    // The encoder pass is the expensive half; skip it for a request whose
    // client left while it waited in the serial queue.
    if cancelled.load(Ordering::Relaxed) {
        return fail(NEMOTRON_PARSE_CANCELLED_BEFORE_START_MSG.to_string());
    }
    if let Some(msg) = reject_media(audio_present, video_present) {
        return fail(msg.to_string());
    }
    if let Some(msg) = reject_image_count(images.len()) {
        return fail(msg);
    }
    let task_prompt = match resolve_task_prompt(prompt) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    match model.seed_ids(task_prompt) {
        Ok(seed) if seed.len() > MAX_SEED_TOKENS => {
            return fail(format!(
                "Nemotron-Parse task prompt tokenizes to {} tokens; the server accepts at most \
                 {MAX_SEED_TOKENS}",
                seed.len()
            ));
        }
        Ok(_) => {}
        Err(e) => return fail(format!("{e}")),
    }

    let mut decoded = match decode_request_images(images) {
        Ok(d) => d,
        Err(e) => return fail(format!("Image decode error: {e}")),
    };
    let Some(image) = decoded.pop() else {
        return fail("Image decode returned no image".to_string());
    };

    let max_new_tokens = options.max_tokens.max(1);
    let start = Instant::now();
    let run = match model.run(
        &image,
        task_prompt,
        max_new_tokens,
        options.sampling.repetition_penalty,
        Some(cancelled),
    ) {
        Ok(run) => run,
        Err(e) => {
            fail(format!("Nemotron-Parse generation error: {e}"));
            mlxcel_core::clear_memory_cache();
            return;
        }
    };
    let elapsed_ms = start.elapsed().as_millis() as u64;

    // The answer is sent as one delta: the decode is model-owned and not
    // streamed token by token, same as Florence-2.
    if !cancelled.load(Ordering::Relaxed) {
        let _ = response_tx.send(GenerateEvent::Token(run.text.clone(), TokenMeta::default()));
    }
    let finish_reason =
        nemotron_parse_finish_reason(run.hit_eos, run.generated_tokens, max_new_tokens);
    let _ = response_tx.send(GenerateEvent::Done(GenerationResult {
        text: run.text,
        prompt_tokens: run.prompt_tokens,
        completion_tokens: run.generated_tokens,
        generation_time_ms: elapsed_ms,
        prompt_eval_ms: 0,
        generation_only_ms: elapsed_ms,
        finish_reason: finish_reason.to_string(),
        stop_kind: if finish_reason == "length" {
            StopKind::Limit
        } else {
            StopKind::Eos
        },
        generated_token_ids: Vec::new(),
        logprobs: None,
        cached_tokens: 0,
        structured_output: None,
        speculative: None,
    }));
    mlxcel_core::clear_memory_cache();
}

#[cfg(test)]
#[path = "nemotron_parse_worker_tests.rs"]
mod tests;
