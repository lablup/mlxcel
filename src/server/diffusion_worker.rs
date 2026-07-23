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

//! Single-stream (batch-1) serving loop for DiffusionGemma (issue #217,
//! phase 3).
//!
//! DiffusionGemma generates by model-owned block diffusion, so it cannot join
//! the batched/paged scheduler (`supports_batching() == false`). Instead the
//! model worker thread, after loading a DiffusionGemma checkpoint, branches
//! into [`run_diffusion_worker_loop`] which serves one request at a time off
//! the same `mpsc` channel the batched worker uses. Requests therefore queue
//! and are served serially: that IS the design (no in-flight concurrency > 1).
//!
//! The loop reuses the CLI generation path
//! ([`DiffusionGemmaModel::generate_diffusion_streaming`]) plus the shared
//! image-prompt helper ([`DiffusionGemmaModel::prepare_image_prompt`]) and the
//! server's incremental detokenizer ([`StreamingDecodeState`]) so streaming
//! SSE output, byte-fallback handling, and usage accounting match the
//! autoregressive serving path.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use crate::models::diffusion_gemma::{
    DiffusionFinishReason, DiffusionGenerateOptions, DiffusionSamplerKind,
};
use crate::models::llada2_moe::{Llada2FinishReason, Llada2GenerateOptions};
use crate::models::{DiffusionGemmaModel, Llada2MoeModel};
use crate::server::ServerGenerateOptions;
use crate::server::model_provider::model_worker::{StreamingDecodeState, decode_request_images};
use crate::server::model_provider::{GenerateEvent, GenerationResult, ModelRequest};
use crate::tokenizer::MlxcelTokenizer;

/// Serve-level diffusion knobs resolved once on the worker thread from the
/// `--diffusion-sampler` / `--diffusion-threshold` / `--max-denoising-steps`
/// flags. They set the per-request defaults for every diffusion generation
/// served by this worker; they only affect diffusion models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DiffusionServeDefaults {
    /// Per-step acceptance sampler (`--diffusion-sampler`).
    pub sampler: DiffusionSamplerKind,
    /// Confidence threshold for the confidence-threshold sampler
    /// (`--diffusion-threshold`).
    pub confidence_threshold: f32,
    /// Optional override for the checkpoint's `max_denoising_steps`
    /// (`--max-denoising-steps`).
    pub max_denoising_steps: Option<usize>,
}

impl Default for DiffusionServeDefaults {
    fn default() -> Self {
        Self {
            sampler: DiffusionSamplerKind::EntropyBound,
            confidence_threshold: 0.9,
            max_denoising_steps: None,
        }
    }
}

/// Parse the serve-level `--diffusion-sampler` flag value.
///
/// `entropy-bound` (the checkpoint default) and `confidence-threshold` are the
/// only supported samplers; anything else is an error the caller surfaces as a
/// warning before falling back to entropy-bound.
pub(crate) fn parse_diffusion_sampler(name: &str) -> Result<DiffusionSamplerKind, String> {
    match name {
        "entropy-bound" => Ok(DiffusionSamplerKind::EntropyBound),
        "confidence-threshold" => Ok(DiffusionSamplerKind::ConfidenceThreshold),
        other => Err(format!("unsupported diffusion sampler: {other:?}")),
    }
}

/// Error message sent for an audio/video diffusion request (unsupported in
/// server mode, phase 3 scope).
pub(crate) const AUDIO_VIDEO_UNSUPPORTED_MSG: &str =
    "DiffusionGemma server mode does not support audio/video input yet";

/// Decide whether a request must be rejected for carrying audio/video input.
///
/// Returns the rejection message when either modality is present (diffusion
/// server mode is text + image only), or `None` when the request is servable.
pub(crate) fn reject_audio_video(audio_present: bool, video_present: bool) -> Option<&'static str> {
    if audio_present || video_present {
        Some(AUDIO_VIDEO_UNSUPPORTED_MSG)
    } else {
        None
    }
}

/// Map a [`DiffusionFinishReason`] to the server's `finish_reason` string.
///
/// The server's OpenAI-compatible response surface only carries `"stop"` and
/// `"length"`; an aborted (client-cancelled) diffusion run maps to `"stop"`
/// because the client has already disconnected and there is no distinct
/// transport-level reason string for it.
pub(crate) fn diffusion_finish_reason_str(reason: DiffusionFinishReason) -> &'static str {
    match reason {
        DiffusionFinishReason::Length => "length",
        DiffusionFinishReason::Stop => "stop",
        DiffusionFinishReason::Aborted => "stop",
    }
}

/// Build the engine [`DiffusionGenerateOptions`] for one request from the
/// server request options and the serve-level diffusion defaults.
///
/// `config_eos` is the model's `generation_config.json` EOS set; together with
/// the request's `stop_token_ids` it forms the extra EOS union the engine adds
/// on top of the model's own embedded EOS ids. Canvas-length knobs
/// (`min/max canvas`, `full_canvas`) and the prefill chunk size keep their
/// engine defaults: the server surface does not expose per-request canvas
/// overrides.
pub(crate) fn diffusion_options_from_server(
    options: &ServerGenerateOptions,
    defaults: &DiffusionServeDefaults,
    config_eos: &[i32],
) -> DiffusionGenerateOptions {
    build_diffusion_options(
        options.max_tokens,
        options.sampling.temperature,
        &options.sampling.stop_token_ids,
        defaults,
        config_eos,
    )
}

/// Pure core of [`diffusion_options_from_server`] over primitive inputs, so the
/// mapping is unit-testable without constructing a full
/// [`ServerGenerateOptions`]. The extra-EOS field is the de-duplicated union of
/// the request stop tokens and the model's `generation_config` EOS set; the
/// engine unions the model's own embedded EOS ids on top of that.
pub(crate) fn build_diffusion_options(
    max_tokens: usize,
    temperature: f32,
    stop_token_ids: &[i32],
    defaults: &DiffusionServeDefaults,
    config_eos: &[i32],
) -> DiffusionGenerateOptions {
    let mut extra_eos: Vec<i32> = Vec::new();
    for &id in stop_token_ids.iter().chain(config_eos) {
        if !extra_eos.contains(&id) {
            extra_eos.push(id);
        }
    }
    DiffusionGenerateOptions {
        max_new_tokens: max_tokens,
        temperature,
        sampler: defaults.sampler,
        confidence_threshold: defaults.confidence_threshold,
        max_denoising_steps: defaults.max_denoising_steps,
        extra_eos_token_ids: extra_eos,
        ..DiffusionGenerateOptions::default()
    }
}

/// Serve DiffusionGemma block-diffusion generation one request at a time.
///
/// Drives the shared `mpsc` request channel: each `Generate` is tokenized,
/// optionally image-prefilled, denoised through
/// [`DiffusionGemmaModel::generate_diffusion_streaming`], and streamed back as
/// `GenerateEvent::Token` / `GenerateEvent::Done`. A single failing request
/// emits `GenerateEvent::Error` and the loop keeps serving; it returns only on
/// `ModelRequest::Shutdown` or when the channel closes.
pub(crate) fn run_diffusion_worker_loop(
    model: &DiffusionGemmaModel,
    tokenizer: &MlxcelTokenizer,
    model_path: &Path,
    request_rx: mpsc::Receiver<ModelRequest>,
    defaults: DiffusionServeDefaults,
    config_eos: &[i32],
) {
    tracing::info!(
        "DiffusionGemma block-diffusion worker ready (single-stream, batch-1; \
         sampler={:?}, max_denoising_steps={:?})",
        defaults.sampler,
        defaults.max_denoising_steps,
    );

    for request in request_rx {
        match request {
            ModelRequest::Shutdown => {
                tracing::info!("DiffusionGemma worker received shutdown signal");
                break;
            }
            ModelRequest::Generate {
                prompt,
                // Diffusion workers tokenize internally; the dispatch-thread
                // pre-tokenized ids (issue #633) are not used here.
                prompt_token_ids: _,
                options,
                images,
                audio,
                videos,
                media: _,
                response_tx,
                cancelled,
            } => {
                handle_diffusion_request(
                    model,
                    tokenizer,
                    model_path,
                    &defaults,
                    config_eos,
                    &prompt,
                    &options,
                    &images,
                    &audio,
                    &videos,
                    &response_tx,
                    &cancelled,
                );
            }
        }
    }
}

/// Handle one diffusion generation request end to end.
///
/// All failure paths send a single `GenerateEvent::Error` and return so one
/// bad request never tears down the worker.
#[allow(clippy::too_many_arguments)]
fn handle_diffusion_request(
    model: &DiffusionGemmaModel,
    tokenizer: &MlxcelTokenizer,
    model_path: &Path,
    defaults: &DiffusionServeDefaults,
    config_eos: &[i32],
    prompt: &str,
    options: &ServerGenerateOptions,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
    videos: &[crate::server::media::ResolvedVideo],
    response_tx: &mpsc::Sender<GenerateEvent>,
    cancelled: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    // Audio / video are not supported in diffusion server mode (phase 3 scope).
    if let Some(msg) = reject_audio_video(!audio.is_empty(), !videos.is_empty()) {
        let _ = response_tx.send(GenerateEvent::Error(msg.to_string()));
        return;
    }

    // Tokenize the already chat-templated prompt, mirroring the batched
    // scheduler's add-special heuristic.
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    let token_ids = match tokenizer.encode(prompt, add_special) {
        Ok(ids) => ids,
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(format!("Tokenization error: {err}")));
            return;
        }
    };
    let prompt_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();

    // Image input: decode the request bytes and run the shared phase-2 vision
    // path (preprocess + token expansion + prefill).
    let prepared = if images.is_empty() {
        None
    } else {
        if !model.supports_images() {
            let _ = response_tx.send(GenerateEvent::Error(
                "This DiffusionGemma checkpoint does not include a vision tower; \
                 send a text-only request"
                    .to_string(),
            ));
            return;
        }
        let decoded = match decode_request_images(images) {
            Ok(decoded) => decoded,
            Err(err) => {
                let _ =
                    response_tx.send(GenerateEvent::Error(format!("Image decode error: {err}")));
                return;
            }
        };
        match model.prepare_image_prompt(model_path, &decoded, &prompt_tokens) {
            Ok(prepared) => Some(prepared),
            Err(err) => {
                let _ = response_tx.send(GenerateEvent::Error(err));
                return;
            }
        }
    };

    let (engine_prompt, vision_prefill) = match &prepared {
        Some(prepared) => (prepared.expanded_ids.as_slice(), Some(&prepared.prefill)),
        None => (prompt_tokens.as_slice(), None),
    };

    if engine_prompt.is_empty() {
        let _ = response_tx.send(GenerateEvent::Error(
            "Empty prompt: request has no input tokens to process".to_string(),
        ));
        return;
    }

    let mut opts = diffusion_options_from_server(options, defaults, config_eos);
    // Per-step cooperative cancellation: a disconnected/cancelled client
    // aborts within one denoising step instead of finishing the block.
    opts.cancel = Some(cancelled.clone());
    if let Some(seed) = options.sampling.seed {
        mlxcel_core::random_seed(seed);
    }

    // Stream each committed token through the shared serving plumbing; the
    // generation call reports the diffusion engine's finish reason.
    serve_streaming_request(
        tokenizer,
        engine_prompt,
        opts.max_new_tokens,
        response_tx,
        cancelled,
        |on_token| {
            model
                .generate_diffusion_streaming(engine_prompt, &opts, vision_prefill, on_token)
                .map(|stats| diffusion_finish_reason_str(stats.finish_reason))
        },
    );
}

/// Shared serving plumbing for the block-diffusion worker loops: build the
/// incremental detokenizer, stream committed tokens (polling the cancel flag
/// and aborting on a closed receiver), drain any byte-fallback tail, and build
/// the usage/timing result.
///
/// The model-specific generation call is supplied as `generate`, which invokes
/// `on_token(id) -> keep_going` per committed token and returns the server
/// finish-reason string. This keeps one copy of the tokenize / stream / cancel
/// / usage plumbing shared by DiffusionGemma and LLaDA-2 MoE.
fn serve_streaming_request<G>(
    tokenizer: &MlxcelTokenizer,
    engine_prompt: &[i32],
    max_new_tokens: usize,
    response_tx: &mpsc::Sender<GenerateEvent>,
    cancelled: &Arc<AtomicBool>,
    generate: G,
) where
    G: FnOnce(&mut dyn FnMut(i32) -> bool) -> Result<&'static str, String>,
{
    let start = Instant::now();
    let mut decode_state = StreamingDecodeState::new(tokenizer, engine_prompt);

    // The on_token closure borrows `decode_state`; scope it so the borrow ends
    // before the usage result is built from the same state below.
    let result = {
        let mut on_token = |token_id: i32| -> bool {
            if let Some(text) = decode_state.on_token(token_id, tokenizer)
                && response_tx.send(GenerateEvent::Token(text)).is_err()
            {
                return false;
            }
            !cancelled.load(Ordering::Relaxed)
        };
        generate(&mut on_token)
    };

    match result {
        Ok(finish_reason) => {
            // Forward the incremental detokenizer's held tail as one final token
            // event before Done so streaming clients receive it (issue #633).
            if let Some(tail) = decode_state.flush(tokenizer) {
                let _ = response_tx.send(GenerateEvent::Token(tail));
            }
            let mut gen_result: GenerationResult = decode_state.finish_with_cache(
                start,
                engine_prompt.len(),
                max_new_tokens.max(1),
                0,
            );
            gen_result.finish_reason = finish_reason.to_string();
            let _ = response_tx.send(GenerateEvent::Done(gen_result));
        }
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(err));
        }
    }

    // Release the transient denoising allocations before the next request.
    mlxcel_core::clear_memory_cache();
}

// ---------------------------------------------------------------------------
// LLaDA-2 MoE serving (issue #546)
// ---------------------------------------------------------------------------

/// Error message sent for an image/audio/video LLaDA-2 request (the model is
/// text-only).
pub(crate) const LLADA2_MEDIA_UNSUPPORTED_MSG: &str =
    "LLaDA-2 MoE is text-only; send a text-only request";

/// Whether a LLaDA-2 request must be rejected for carrying any media input.
pub(crate) fn reject_llada2_media(
    image_present: bool,
    audio_present: bool,
    video_present: bool,
) -> Option<&'static str> {
    if image_present || audio_present || video_present {
        Some(LLADA2_MEDIA_UNSUPPORTED_MSG)
    } else {
        None
    }
}

/// Map a [`Llada2FinishReason`] to the server's `finish_reason` string. An
/// aborted (client-cancelled) run maps to `"stop"`, matching the diffusion
/// convention.
pub(crate) fn llada2_finish_reason_str(reason: Llada2FinishReason) -> &'static str {
    match reason {
        Llada2FinishReason::Length => "length",
        Llada2FinishReason::Stop => "stop",
        Llada2FinishReason::Aborted => "stop",
    }
}

/// Build the engine [`Llada2GenerateOptions`] for one request from the server
/// request options. `steps_override` comes from the serve-level
/// `--max-denoising-steps` flag; the sampler / threshold diffusion flags are
/// gemma-only and do not apply. `config_eos` is the model's
/// `generation_config.json` EOS set, unioned with the request stop ids.
pub(crate) fn llada2_options_from_server(
    options: &ServerGenerateOptions,
    steps_override: Option<usize>,
    config_eos: &[i32],
) -> Llada2GenerateOptions {
    build_llada2_options(
        options.max_tokens,
        options.sampling.temperature,
        options.sampling.top_k,
        options.sampling.top_p,
        &options.sampling.stop_token_ids,
        steps_override,
        config_eos,
    )
}

/// Pure core of [`llada2_options_from_server`] over primitive inputs, so the
/// mapping is unit-testable without a full [`ServerGenerateOptions`]. The
/// extra-EOS field is the de-duplicated union of the request stop tokens and
/// the model's `generation_config` EOS set; the engine unions the model's own
/// embedded EOS ids on top of that.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_llada2_options(
    max_tokens: usize,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    stop_token_ids: &[i32],
    steps_override: Option<usize>,
    config_eos: &[i32],
) -> Llada2GenerateOptions {
    let mut extra_eos: Vec<i32> = Vec::new();
    for &id in stop_token_ids.iter().chain(config_eos) {
        if !extra_eos.contains(&id) {
            extra_eos.push(id);
        }
    }
    let default = Llada2GenerateOptions::default();
    Llada2GenerateOptions {
        max_new_tokens: max_tokens,
        temperature,
        top_k,
        top_p,
        steps: steps_override.unwrap_or(default.steps),
        extra_eos_token_ids: extra_eos,
        ..default
    }
}

/// Serve LLaDA-2 MoE block-unmasking generation one request at a time.
///
/// Drives the shared `mpsc` request channel with the same tokenize / stream /
/// cancel / usage plumbing as the DiffusionGemma loop. `steps_override` sets
/// the per-block denoising step count for every request served by this worker.
pub(crate) fn run_llada2_worker_loop(
    model: &Llada2MoeModel,
    tokenizer: &MlxcelTokenizer,
    request_rx: mpsc::Receiver<ModelRequest>,
    steps_override: Option<usize>,
    config_eos: &[i32],
) {
    tracing::info!(
        "LLaDA-2 MoE block-unmasking worker ready (single-stream, batch-1; \
         steps_override={steps_override:?})"
    );

    for request in request_rx {
        match request {
            ModelRequest::Shutdown => {
                tracing::info!("LLaDA-2 MoE worker received shutdown signal");
                break;
            }
            ModelRequest::Generate {
                prompt,
                // Diffusion workers tokenize internally; the dispatch-thread
                // pre-tokenized ids (issue #633) are not used here.
                prompt_token_ids: _,
                options,
                images,
                audio,
                videos,
                media: _,
                response_tx,
                cancelled,
            } => {
                handle_llada2_request(
                    model,
                    tokenizer,
                    steps_override,
                    config_eos,
                    &prompt,
                    &options,
                    &images,
                    &audio,
                    &videos,
                    &response_tx,
                    &cancelled,
                );
            }
        }
    }
}

/// Handle one LLaDA-2 generation request end to end. All failure paths send a
/// single `GenerateEvent::Error` and return so one bad request never tears
/// down the worker.
#[allow(clippy::too_many_arguments)]
fn handle_llada2_request(
    model: &Llada2MoeModel,
    tokenizer: &MlxcelTokenizer,
    steps_override: Option<usize>,
    config_eos: &[i32],
    prompt: &str,
    options: &ServerGenerateOptions,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
    videos: &[crate::server::media::ResolvedVideo],
    response_tx: &mpsc::Sender<GenerateEvent>,
    cancelled: &Arc<AtomicBool>,
) {
    if let Some(msg) =
        reject_llada2_media(!images.is_empty(), !audio.is_empty(), !videos.is_empty())
    {
        let _ = response_tx.send(GenerateEvent::Error(msg.to_string()));
        return;
    }

    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    let token_ids = match tokenizer.encode(prompt, add_special) {
        Ok(ids) => ids,
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(format!("Tokenization error: {err}")));
            return;
        }
    };
    let prompt_tokens: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
    if prompt_tokens.is_empty() {
        let _ = response_tx.send(GenerateEvent::Error(
            "Empty prompt: request has no input tokens to process".to_string(),
        ));
        return;
    }

    let mut opts = llada2_options_from_server(options, steps_override, config_eos);
    opts.cancel = Some(cancelled.clone());
    if let Some(seed) = options.sampling.seed {
        mlxcel_core::random_seed(seed);
    }

    serve_streaming_request(
        tokenizer,
        &prompt_tokens,
        opts.max_new_tokens,
        response_tx,
        cancelled,
        |on_token| {
            model
                .generate_llada2_streaming(&prompt_tokens, &opts, on_token)
                .map(|stats| llada2_finish_reason_str(stats.finish_reason))
        },
    );
}

#[cfg(test)]
#[path = "diffusion_worker_tests.rs"]
mod tests;
