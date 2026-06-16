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
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Instant;

use crate::models::DiffusionGemmaModel;
use crate::models::diffusion_gemma::{
    DiffusionFinishReason, DiffusionGenerateOptions, DiffusionSamplerKind,
};
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
                options,
                images,
                audio,
                videos,
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

    let start = Instant::now();
    let mut decode_state = StreamingDecodeState::new(tokenizer, engine_prompt);

    // Stream each committed token, polling the cancellation flag in the engine
    // callback (returning false aborts the denoising loop). A closed receiver
    // (client gone) also aborts.
    let result =
        model.generate_diffusion_streaming(engine_prompt, &opts, vision_prefill, |token_id| {
            if let Some(text) = decode_state.on_token(token_id, tokenizer)
                && response_tx.send(GenerateEvent::Token(text)).is_err()
            {
                return false;
            }
            !cancelled.load(Ordering::Relaxed)
        });

    match result {
        Ok(stats) => {
            // Drain any byte-fallback tail held back by the streaming decoder,
            // then build the usage/timing result and override the finish
            // reason with the diffusion engine's verdict.
            decode_state.flush(tokenizer);
            let mut gen_result: GenerationResult = decode_state.finish_with_cache(
                start,
                engine_prompt.len(),
                opts.max_new_tokens.max(1),
                0,
            );
            gen_result.finish_reason = diffusion_finish_reason_str(stats.finish_reason).to_string();
            let _ = response_tx.send(GenerateEvent::Done(gen_result));
        }
        Err(err) => {
            let _ = response_tx.send(GenerateEvent::Error(err));
        }
    }

    // Release the transient denoising allocations before the next request.
    mlxcel_core::clear_memory_cache();
}

#[cfg(test)]
#[path = "diffusion_worker_tests.rs"]
mod tests;
