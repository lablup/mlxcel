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

//! Single-stream (batch-1) serving loop for Florence-2 (issue #1073).
//!
//! Florence-2 is a BART-style encoder-decoder (seq2seq) VLM: each request
//! runs one bidirectional encoder pass over the fused vision-plus-prompt
//! sequence and then decodes autoregressively with cross-attention against
//! the cached encoder output. That decode drives its own dual
//! [`crate::models::florence2::Florence2SeqCache`], not the decoder-only
//! `KVCache` list, so the model cannot join the batched/paged scheduler
//! (`supports_batching() == false`). Following the DiffusionGemma / LLaDA-2
//! precedent in `diffusion_worker.rs`, the model worker thread branches into
//! [`run_florence2_worker_loop`] after loading a Florence-2 checkpoint and
//! serves one request at a time off the same `mpsc` channel the batched
//! worker uses. Requests queue and are served serially: that IS the design
//! for this first landing (no in-flight concurrency > 1), because the
//! encoder pass has a different cost profile from the decode loop and no
//! batched admission policy for it has been designed or measured.
//!
//! Per-request state lifetime: the encoder output and the seq2seq decode
//! cache are created inside
//! [`Florence2VlmModel::run_task_with_cancel`] and dropped when it returns,
//! so nothing outlives the request that created it and no encoder state can
//! cross-contaminate a later request.
//!
//! Security boundaries carried forward from issue #855:
//!
//! - Image bytes decode through [`decode_request_images`], which applies the
//!   configured [`crate::server::media::ImageInputLimits`] (payload size,
//!   dimension caps, decode-allocation cap), so an oversized or
//!   decompression-bomb payload is rejected before any pixel work.
//! - The task prompt is untrusted. [`parse_task_prompt`] admits only a
//!   recognized task marker, and [`validate_task_input`] bounds and
//!   shape-checks the free-form input text the 7 input-taking task modes
//!   interpolate into the encoder prompt (length bound, control-character
//!   and angle-bracket rejection, strict `<loc_*>` quadruple form for the
//!   region tasks).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use crate::models::florence2::{
    Florence2Task, Florence2VlmModel, parse_task_prompt, render_task_result, structured_task_json,
};
use crate::server::ServerGenerateOptions;
use crate::server::model_provider::model_worker::decode_request_images;
use crate::server::model_provider::{GenerateEvent, GenerationResult, ModelRequest};

/// Error message sent for an audio/video Florence-2 request.
pub(crate) const FLORENCE2_MEDIA_UNSUPPORTED_MSG: &str =
    "Florence-2 is an image-task model; audio and video inputs are not supported";

/// Error message sent when the request does not carry exactly one image.
pub(crate) const FLORENCE2_IMAGE_REQUIRED_MSG: &str = "Florence-2 requires exactly one image per request: attach one image_url content part and \
     set the message text to a task prompt such as <CAPTION>, <OCR>, or <OD>";

/// Error message sent when the request's client already disconnected before
/// the worker reached it in the queue.
pub(crate) const FLORENCE2_CANCELLED_BEFORE_START_MSG: &str =
    "Florence-2 request cancelled before generation started";

/// Upper bound, in bytes, on the caller-supplied input text an input-taking
/// task may interpolate into the encoder prompt.
///
/// Generous enough for the longest legitimate input (a
/// `<CAPTION_TO_PHRASE_GROUNDING>` caption, e.g. a whole
/// `<MORE_DETAILED_CAPTION>` paragraph fed back for grounding) while
/// bounding request-boundary work. The encoder's own
/// `max_position_embeddings` check remains the final token-level bound, and
/// it can reject a shorter input than this one, because the fused encoder
/// sequence carries the image's projected tokens ahead of the prompt.
///
/// This is not a pre-tokenization bound on the server path: the dispatch
/// thread already pre-tokenizes the rendered prompt (issue #633) before the
/// request reaches this worker, which discards those ids and tokenizes
/// through the task processor instead. What the bound does refuse, before
/// any model work, is oversized text reaching `Florence2Task::expand` and
/// the encoder prompt.
pub(crate) const MAX_TASK_INPUT_BYTES: usize = 2048;

/// Validate the caller-supplied input text for a Florence-2 task at the
/// request boundary (issue #1073; security handoff from issue #855).
///
/// [`Florence2Task::expand`] already rejects a missing input on a task that
/// needs one and an input on a task that takes none; this function adds the
/// server-surface constraints on the input that IS accepted:
///
/// - at most [`MAX_TASK_INPUT_BYTES`] bytes;
/// - no control characters;
/// - for the four region tasks (`<REGION_TO_SEGMENTATION>`,
///   `<REGION_TO_CATEGORY>`, `<REGION_TO_DESCRIPTION>`, `<REGION_TO_OCR>`),
///   the input must be exactly four location tokens
///   `<loc_a><loc_b><loc_c><loc_d>` with each bin in `0..=999`;
/// - for the three free-text tasks (`<CAPTION_TO_PHRASE_GROUNDING>`,
///   `<REFERRING_EXPRESSION_SEGMENTATION>`, `<OPEN_VOCABULARY_DETECTION>`),
///   `<` and `>` are rejected so a caller cannot smuggle location or
///   sequence markers into the encoder prompt.
///
/// The CLI does not run this: its input comes from the operator's own flag,
/// not from the network.
pub(crate) fn validate_task_input(task: Florence2Task, input: Option<&str>) -> Result<(), String> {
    let Some(text) = input else {
        // Presence/absence itself is `Florence2Task::expand`'s call.
        return Ok(());
    };
    if !task.takes_input() {
        // Also expand's call, but reject here for a boundary-shaped message.
        return Err(format!(
            "Florence-2 task {} takes no input text",
            task.token()
        ));
    }
    if text.len() > MAX_TASK_INPUT_BYTES {
        return Err(format!(
            "Florence-2 task input for {} is {} bytes; the server accepts at most \
             {MAX_TASK_INPUT_BYTES} bytes",
            task.token(),
            text.len()
        ));
    }
    if text.chars().any(char::is_control) {
        return Err(format!(
            "Florence-2 task input for {} contains control characters",
            task.token()
        ));
    }

    use Florence2Task::*;
    match task {
        RegionToSegmentation | RegionToCategory | RegionToDescription | RegionToOcr => {
            if parse_region_bins(text).is_none() {
                return Err(format!(
                    "Florence-2 task {} takes a region as exactly four location tokens \
                     <loc_a><loc_b><loc_c><loc_d> with each value in 0..=999, \
                     e.g. <loc_52><loc_332><loc_932><loc_774>; got {text:?}",
                    task.token()
                ));
            }
        }
        CaptionToPhraseGrounding | ReferringExpressionSegmentation | OpenVocabularyDetection
            if text.contains('<') || text.contains('>') =>
        {
            return Err(format!(
                "Florence-2 task input for {} must be plain text without '<' or '>'",
                task.token()
            ));
        }
        // Input-less tasks were handled by the takes_input() rejection above.
        _ => {}
    }
    Ok(())
}

/// Parse a region input of exactly four `<loc_N>` tokens, `N` in `0..=999`.
///
/// Returns the four bin values, or `None` when the string deviates from that
/// form in any way (junk between tokens, too few/many tokens, out-of-range
/// or non-numeric bins, leading/trailing text).
pub(crate) fn parse_region_bins(input: &str) -> Option<[u16; 4]> {
    let mut bins = [0u16; 4];
    let mut rest = input;
    for slot in &mut bins {
        rest = rest.strip_prefix("<loc_")?;
        let end = rest.find('>')?;
        let digits = &rest[..end];
        if digits.is_empty() || digits.len() > 3 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let value: u16 = digits.parse().ok()?;
        if value > 999 {
            return None;
        }
        *slot = value;
        rest = &rest[end + 1..];
    }
    rest.is_empty().then_some(bins)
}

/// Whether a Florence-2 request must be rejected for carrying audio or video.
///
/// Mirrors [`crate::server::diffusion_worker::reject_audio_video`]: the guard
/// is a pure function over the two presence flags so the request-to-response
/// mapping is unit-testable without constructing a `ModelRequest` and a live
/// model.
pub(crate) fn reject_media(audio_present: bool, video_present: bool) -> Option<&'static str> {
    if audio_present || video_present {
        Some(FLORENCE2_MEDIA_UNSUPPORTED_MSG)
    } else {
        None
    }
}

/// Whether a Florence-2 request carries the exactly-one image the encoder
/// needs, returning the rejection message when it does not.
///
/// Florence-2 fuses one image's projected tokens in front of the task-prompt
/// embeddings and has no image placeholder token, so neither zero images nor
/// a second image has a meaning on this path.
pub(crate) fn reject_image_count(image_count: usize) -> Option<String> {
    (image_count != 1).then(|| format!("{FLORENCE2_IMAGE_REQUIRED_MSG} (got {image_count} images)"))
}

/// Map the greedy decode outcome onto the server's `finish_reason` string.
///
/// [`crate::models::florence2::Florence2VlmModel::run_task_with_cancel`]
/// pushes at most `max_new_tokens` ids and stops early on EOS, on the decoder
/// position bound, or on cancellation, so an exhausted budget is the only
/// case that reports `"length"`. A cancelled run reports `"stop"`, matching
/// the diffusion workers' aborted-maps-to-stop convention.
pub(crate) fn florence2_finish_reason(
    generated_tokens: usize,
    max_new_tokens: usize,
) -> &'static str {
    if generated_tokens >= max_new_tokens {
        "length"
    } else {
        "stop"
    }
}

/// Serve Florence-2 task generation one request at a time.
///
/// Drives the shared `mpsc` request channel: each `Generate` is parsed into
/// a task, validated, image-decoded under the configured admission limits,
/// run through the model-owned encoder-pass-plus-cross-attention-decode
/// pipeline, and answered as one `GenerateEvent::Token` (the rendered text)
/// followed by `GenerateEvent::Done`. A failing request emits
/// `GenerateEvent::Error` and the loop keeps serving; it returns only on
/// `ModelRequest::Shutdown` or when the channel closes.
pub(crate) fn run_florence2_worker_loop(
    model: &Florence2VlmModel,
    request_rx: mpsc::Receiver<ModelRequest>,
) {
    tracing::info!(
        "Florence-2 seq2seq worker ready (single-stream, batch-1; one encoder pass per request, \
         greedy cross-attention decode)"
    );

    for request in request_rx {
        match request {
            ModelRequest::Shutdown => {
                tracing::info!("Florence-2 seq2seq worker received shutdown signal");
                break;
            }
            ModelRequest::Generate {
                prompt,
                // The seq2seq path tokenizes through its own task processor;
                // the dispatch-thread pre-tokenized ids (issue #633) are not
                // used here.
                prompt_token_ids: _,
                options,
                images,
                audio,
                videos,
                media: _,
                queue_reservation,
                response_tx,
                cancelled,
            } => {
                drop(queue_reservation);
                handle_florence2_request(
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

/// Handle one Florence-2 task request end to end.
///
/// All failure paths send a single `GenerateEvent::Error` and return so one
/// bad request never tears down the worker.
#[allow(clippy::too_many_arguments)]
fn handle_florence2_request(
    model: &Florence2VlmModel,
    prompt: &str,
    options: &ServerGenerateOptions,
    images: &[Vec<u8>],
    audio_present: bool,
    video_present: bool,
    response_tx: &mpsc::Sender<GenerateEvent>,
    cancelled: &std::sync::Arc<AtomicBool>,
) {
    // Serving is serial, so a request can sit in the channel while its client
    // goes away. The per-step poll inside `generate_greedy_with_cancel` only
    // starts after the encoder pass, which is the expensive half (DaViT tower
    // plus the bidirectional BART encoder over the fused sequence), so an
    // already-abandoned request would still pay for it. Drop it here instead.
    if cancelled.load(Ordering::Relaxed) {
        let _ = response_tx.send(GenerateEvent::Error(
            FLORENCE2_CANCELLED_BEFORE_START_MSG.to_string(),
        ));
        return;
    }
    if let Some(msg) = reject_media(audio_present, video_present) {
        let _ = response_tx.send(GenerateEvent::Error(msg.to_string()));
        return;
    }
    if let Some(msg) = reject_image_count(images.len()) {
        let _ = response_tx.send(GenerateEvent::Error(msg));
        return;
    }

    // The Florence-2 built-in chat template renders the request messages'
    // text verbatim (no role prefixes, no generation prompt), so `prompt` is
    // the caller's task string exactly as the CLI would receive it via `-p`.
    let (task, input) = match parse_task_prompt(prompt) {
        Ok(parsed) => parsed,
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(format!(
                "Florence-2 task prompt: {err}"
            )));
            return;
        }
    };
    if let Err(err) = validate_task_input(task, input.as_deref()) {
        let _ = response_tx.send(GenerateEvent::Error(err));
        return;
    }

    // Decode the image bytes under the configured admission limits
    // (decompression-bomb defense): `preprocess_with_sizes` takes an
    // already-decoded image, so the bound has to hold at this boundary.
    let mut decoded = match decode_request_images(images) {
        Ok(decoded) => decoded,
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(format!("Image decode error: {err}")));
            return;
        }
    };
    let Some(image) = decoded.pop() else {
        let _ = response_tx.send(GenerateEvent::Error(
            "Image decode returned no image".to_string(),
        ));
        return;
    };

    let max_new_tokens = options.max_tokens.max(1);
    let start = Instant::now();
    let run = match model.run_task_with_cancel(
        task,
        input.as_deref(),
        &image,
        max_new_tokens,
        Some(cancelled),
    ) {
        Ok(run) => run,
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(format!(
                "Florence-2 generation error: {err}"
            )));
            // Release the transient encode/decode allocations even on failure.
            mlxcel_core::clear_memory_cache();
            return;
        }
    };
    let elapsed_ms = start.elapsed().as_millis() as u64;

    let rendered = render_task_result(&run.output.result, &run.output.raw_text);
    let structured = structured_task_json(task, &run.output.result);

    // One token event carrying the whole rendered answer: post-processing
    // needs the complete decode, so streaming clients receive the final text
    // as a single delta rather than a token-by-token stream of raw
    // `<loc_*>` markers that would not match the non-streaming content.
    if !cancelled.load(Ordering::Relaxed) {
        let _ = response_tx.send(GenerateEvent::Token(rendered.clone()));
    }

    let finish_reason = florence2_finish_reason(run.generated_tokens, max_new_tokens);
    let _ = response_tx.send(GenerateEvent::Done(GenerationResult {
        text: rendered,
        prompt_tokens: run.prompt_tokens,
        completion_tokens: run.generated_tokens,
        generation_time_ms: elapsed_ms,
        // The encoder pass and the decode loop run inside one model-owned
        // call; the split is not observable from here, so the whole wall
        // time is reported as generation time.
        prompt_eval_ms: 0,
        generation_only_ms: elapsed_ms,
        finish_reason: finish_reason.to_string(),
        logprobs: None,
        cached_tokens: 0,
        structured_output: Some(structured),
    }));

    // Release the transient encode/decode allocations before the next request.
    mlxcel_core::clear_memory_cache();
}

#[cfg(test)]
#[path = "florence2_worker_tests.rs"]
mod tests;
